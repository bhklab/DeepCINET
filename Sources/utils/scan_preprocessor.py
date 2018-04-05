""""
This script is not to be called directly but to be imported
by other ones
"""

import shutil
import os
import os.path as path
from typing import Tuple, Dict

import pydicom
import scipy.misc
import numpy as np
import pandas as pd
import skimage.transform as skt
from joblib import Parallel, delayed

# Columns from CSV sheet containing the info that we need
COL_ID = 0
COL_AGE = 1
COL_SEX = 2
COL_EVENT = 35
COL_TIME = 36


class ScanNormalizer:
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0

    X_SIZE = 64
    Y_SIZE = 64
    Z_SIZE = 64

    def __init__(self, dirs, output_dir, censor_info, overwrite=False):
        self.dirs = []
        self.output_dir = output_dir
        self.output_cache_dir = path.join(output_dir, '.cache')  # Use a hidden folder for the cache
        self.medical_info_path = censor_info
        self.overwrite = overwrite

        self.dirs = [PseudoDir(d.name, d.path, d.is_dir()) for d in dirs]
        self.dir_names = [d.name for d in self.dirs]

    def process_data(self):
        # Create the place where numpy to dycom images will be saved
        if not os.path.exists(self.output_cache_dir):
            os.makedirs(self.output_cache_dir)

        # Can be used as a parallel version or as a single core version, just comment the necessary lines of code
        generator = (delayed(self.process_individual)(image, i + 1) for i, image in enumerate(self.dirs))
        Parallel(n_jobs=-1, backend='multiprocessing')(generator)

        # Use this part for debugging purposes
        # for i, image in enumerate(self.dirs):
        #     self.process_individual(image, i + 1)

        # Get censored data information
        df = pd.read_csv(self.medical_info_path)
        df = df.take([COL_ID, COL_AGE, COL_SEX, COL_EVENT, COL_TIME], axis=1)
        df.columns = ['id', 'age', 'sex', 'event', 'time']
        df = df[df['id'].isin(self.dir_names)]
        df['event'] = 1 - df['event']
        df.to_csv(path.join(self.output_dir, 'clinical_info.csv'))

        censored_count = df[df['event'] == 0].count()[0]
        uncensored_count = df.count()[0] - censored_count

        censored_pairs = censored_count*uncensored_count
        uncensored_pairs = scipy.misc.comb(uncensored_count, 2, exact=True)

        censored_pairs_augmented = censored_pairs*(4**3)*(4**3)
        uncensored_pairs_augmented = uncensored_pairs*(4**3)*(4**3)

        print("Total censored: {}".format(censored_count))
        print("Total uncensored: {}".format(uncensored_count))

        print("Total censored pairs: {:,}".format(censored_pairs))
        print("Total uncensored pairs: {:,}".format(uncensored_pairs))
        print("Total pairs {:,}".format(censored_pairs + uncensored_pairs))

        print("Total censored pairs augmented: {:,}".format(censored_pairs_augmented))
        print("Total uncensored pairs augmented: {:,}".format(uncensored_pairs_augmented))
        print("Total pairs augmented {:,}".format(censored_pairs_augmented + uncensored_pairs_augmented))

    def process_individual(self, image_dir: PseudoDir, count):

        save_dir = path.join(self.output_dir, image_dir.name)
        temp_dir = path.join(self.output_dir, image_dir.name + "_temp")

        # Check if the directory exists to avoid overwriting it
        if os.path.exists(save_dir):
            if self.overwrite:
                shutil.rmtree(save_dir)
            else:
                return

        # The directory should have the NAME and the NAME-MASS subdirectories which contain the files we need
        if not all(x in os.listdir(image_dir.path) for x in [image_dir.name, image_dir.name + "-MASS"]):
            raise FileNotFoundError("Dir {} does not have the necessary files".format(image_dir.name))

        print("Processing dataset {}, {} of {}".format(image_dir.name, count, len(self.dirs)))

        # Remove existing temporary directory from previous runs
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        # Get slice and normalize image
        sliced_norm = self.normalize_image(*self.get_raw_data(image_dir))

        # Rotate the image across the 3 axis for data augmentation
        rotations = self.get_rotations(sliced_norm)
        np.savez_compressed(path.join(temp_dir, image_dir.name + ".npz"), **rotations)

        # Rename after finishing to be able to stop in the middle
        os.rename(temp_dir, save_dir)

    def get_raw_data(self, image_dir: PseudoDir):
        """
        Return the raw data, stacking all the files and using a temporary cache
        :param image_dir:
        :return:
        """
        numpy_file = path.join(self.output_cache_dir, image_dir.name + ".npz")
        numpy_file_temp = path.join(self.output_cache_dir, image_dir.name + "_temp.npz")

        # Load .npz file instead of .dcm if we have already read it
        if os.path.exists(numpy_file):
            print("File {} found reading npz file".format(numpy_file))
            npz_file = np.load(numpy_file)
            main_stack = npz_file['main']
            mask_stack = npz_file['mask']
            npz_file.close()
        else:
            print("File {} not found reading dcm file".format(numpy_file))
            main_stack, mask_stack = self.compact_files(image_dir)
            print("Saving {} file".format(numpy_file_temp))
            np.savez_compressed(numpy_file_temp, main=main_stack, mask=mask_stack)
            os.rename(numpy_file_temp, numpy_file)  # Use a temp name to avoid problems when stopping the script
        return main_stack, mask_stack

    def normalize_image(self, main_stack: np.ndarray, mask_stack: np.ndarray) -> np.ndarray:
        """
        Slices the tumour using the mask and then
        normalizes the image (variance = 1 and mean = 0)
        :param main_stack: 3D vector containing the main image
        :param mask_stack: 3D vector containing the mask image where the tumour is marked
        :return: Slice of the tumour normalized
        """
        # Get sliced image
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_bounding_box(mask_stack)
        sliced = main_stack[x_min:x_max, y_min:y_max, z_min:z_max]

        # Normalize the sliced part
        sliced = sliced.clip(self.MIN_BOUND, self.MAX_BOUND)
        sliced_norm = (sliced - self.MIN_BOUND) / (self.MAX_BOUND - self.MIN_BOUND)

        # Apply mask
        sliced_norm *= mask_stack[x_min:x_max, y_min:y_max, z_min:z_max]

        print("Volume: {}".format(sliced_norm.shape))

        # Resize the normalized data
        sliced_norm = skt.resize(sliced_norm, (self.X_SIZE, self.Y_SIZE, self.Z_SIZE), mode='symmetric')
        return sliced_norm

    @staticmethod
    def get_rotations(sliced_norm: np.array) -> Dict[str, np.ndarray]:
        temp_dict = {}
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    name = "{:03}_{:03}_{:03}.npy".format(i * 90, j * 90, k * 90)
                    temp_dict[name] = sliced_norm.copy()
                    sliced_norm = np.rot90(sliced_norm, axes=(0, 1))
                sliced_norm = np.rot90(sliced_norm, axes=(2, 0))
            sliced_norm = np.rot90(sliced_norm, axes=(0, 1))
        return temp_dict

    @staticmethod
    def compact_files(image_dir: PseudoDir) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a numpy array containing the 3D image concatenating all the slices in the selected dir
        :param image_dir: Directory containing all the images
        :return: Tuple with the 3D image and the 3D mask as numpy arrays
        """
        main_path = path.join(image_dir.path, image_dir.name)
        mask_path = main_path + "-MASS"
        total_main = [pydicom.dcmread(x.path).pixel_array for x in os.scandir(main_path)]
        total_mask = [pydicom.dcmread(x.path).pixel_array for x in os.scandir(mask_path)]

        main_stack = np.stack(total_main, axis=2)
        mask_stack = np.stack(total_mask, axis=2)
        mask_stack[mask_stack > 1] = 1
        return main_stack, mask_stack

    @staticmethod
    def get_bounding_box(mask_stack: np.ndarray) -> Tuple:
        """
        Get the bounding box of all the area containing 1s
        :param mask_stack: 3D numpy array
        :return: Bounding box tuple with the minimum and maximum size in the 3 axis
        """
        # Code found in stack overflow
        x = np.any(mask_stack, axis=(1, 2))
        y = np.any(mask_stack, axis=(0, 2))
        z = np.any(mask_stack, axis=(0, 1))

        x_min, x_max = np.where(x)[0][[0, -1]]
        y_min, y_max = np.where(y)[0][[0, -1]]
        z_min, z_max = np.where(z)[0][[0, -1]]
        return x_min, x_max, y_min, y_max, z_min, z_max
