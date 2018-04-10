import os
import shutil
from typing import Tuple, Dict

import scipy
import numpy as np
import pandas as pd
import skimage.transform as skt
from joblib import Parallel, delayed

from .RawData import RawData
from .PseudoDir import PseudoDir

# Columns from CSV sheet containing the info that we need
COL_ID = 0
COL_AGE = 1
COL_SEX = 2
COL_EVENT = 35
COL_TIME = 36


class PreProcessedData:
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0

    X_SIZE = 64
    Y_SIZE = 64
    Z_SIZE = 64

    def __init__(self):
        self.data_path = os.getenv('DATA_PROCESSED')
        self.clinical_info_path = os.getenv('DATA_CLINICAL')
        self.overwrite = False
        self.raw_data = RawData()

    def store(self, overwrite=False) -> None:
        """
        Performs the pre process and stores all the data to disk
        """
        self.overwrite = overwrite
        self.raw_data.store_elements()

        # To pre-process on Mordor
        jobs = int(os.getenv("NSLOTS", -1))
        print("Jobs: {}".format(jobs))

        generator = (delayed(self._process_individual)(image_dir, main_stack, mask_stack, i + 1)
                     for i, (image_dir, main_stack, mask_stack) in enumerate(self.raw_data.elements()))
        Parallel(n_jobs=jobs, backend="multiprocessing")(generator)

        # For debugging purposes
        # for i, (image_dir, main_stack, mask_stack) in enumerate(self.raw_data.elements()):
        #     self._process_individual(image_dir, main_stack, mask_stack, i + 1)

        self._write_clinical_filtered()
        
    def _process_individual(self, image_dir: PseudoDir, main_stack: np.ndarray, mask_stack: np.ndarray,  count: int):
        save_dir = os.path.join(self.data_path, image_dir.name)
        temp_dir = os.path.join(self.data_path, image_dir.name + "_temp")

        # Check if the directory exists to avoid overwriting it
        if os.path.exists(save_dir):
            if self.overwrite:
                shutil.rmtree(save_dir)
            else:
                return

        print("Processing dataset {}, {} of {}".format(image_dir.name, count, self.raw_data.total_elements()))

        # Remove existing temporary directory from previous runs
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        # Get slice and normalize image
        sliced_norm = self._normalize_image(main_stack, mask_stack)

        # Rotate the image across the 3 axis for data augmentation
        rotations = self._get_rotations(sliced_norm)
        np.savez_compressed(os.path.join(temp_dir, image_dir.name + ".npz"), **rotations)

        # Rename after finishing to be able to stop in the middle
        os.rename(temp_dir, save_dir)

    def _normalize_image(self, main_stack: np.ndarray, mask_stack: np.ndarray) -> np.ndarray:
        """
        Slices the tumour using the mask and then
        normalizes the image (variance = 1 and mean = 0)
        :param main_stack: 3D vector containing the main image
        :param mask_stack: 3D vector containing the mask image where the tumour is marked
        :return: Slice of the tumour normalized
        """
        # Get sliced image
        x_min, x_max, y_min, y_max, z_min, z_max = self._get_bounding_box(mask_stack)
        sliced = main_stack[x_min:x_max, y_min:y_max, z_min:z_max]

        # Apply mask
        sliced = sliced*mask_stack[x_min:x_max, y_min:y_max, z_min:z_max]
        sliced = sliced.astype(float)

        # Normalize the sliced part (var = 1, mean = 0)
        sliced -= sliced.mean()
        sliced_norm = sliced/sliced.std()

        print("Volume: {}".format(sliced_norm.shape))

        # Resize the normalized data
        sliced_norm = skt.resize(sliced_norm, (self.X_SIZE, self.Y_SIZE, self.Z_SIZE), mode='symmetric')
        return sliced_norm

    def _write_clinical_filtered(self):
        if not os.path.exists(self.clinical_info_path):
            raise FileNotFoundError("The clinical info file has not been found at {}".format(self.clinical_info_path))

        df = pd.read_csv(self.clinical_info_path)
        df = df.take([COL_ID, COL_AGE, COL_SEX, COL_EVENT, COL_TIME], axis=1)
        df.columns = ['id', 'age', 'sex', 'event', 'time']
        df = df[df['id'].isin(self.raw_data.valid_dirs_path())]              # Remove elements that are not valid data
        df['event'] = 1 - df['event']                                   # The event column is inverted, fix it
        df.to_csv(os.getenv('DATA_CLINICAL_PROCESSED'))

        # Compute number of possible pairs
        censored_count = df[df['event'] == 0].count()[0]
        uncensored_count = df.count()[0] - censored_count

        censored_pairs = censored_count * uncensored_count
        uncensored_pairs = scipy.misc.comb(uncensored_count, 2, exact=True)

        censored_pairs_augmented = censored_pairs * (4 ** 3) * (4 ** 3)
        uncensored_pairs_augmented = uncensored_pairs * (4 ** 3) * (4 ** 3)

        print("Total censored: {}".format(censored_count))
        print("Total uncensored: {}".format(uncensored_count))

        print("Total censored pairs: {:,}".format(censored_pairs))
        print("Total uncensored pairs: {:,}".format(uncensored_pairs))
        print("Total pairs {:,}".format(censored_pairs + uncensored_pairs))

        print("Total censored pairs augmented: {:,}".format(censored_pairs_augmented))
        print("Total uncensored pairs augmented: {:,}".format(uncensored_pairs_augmented))
        print("Total pairs augmented {:,}".format(censored_pairs_augmented + uncensored_pairs_augmented))

    @staticmethod
    def _get_rotations(sliced_norm: np.array) -> Dict[str, np.ndarray]:
        temp_dict = {}
        for i in range(4):
            name = "{:03}".format(i * 90)
            temp_dict[name] = sliced_norm.copy()
            sliced_norm = np.rot90(sliced_norm, axes=(0, 1))

        # for i in range(4):
        #     for j in range(4):
        #         for k in range(4):
        #             name = "{:03}_{:03}_{:03}".format(i * 90, j * 90, k * 90)
        #             temp_dict[name] = sliced_norm.copy()
        #             sliced_norm = np.rot90(sliced_norm, axes=(0, 1))
        #         sliced_norm = np.rot90(sliced_norm, axes=(2, 0))
        #     sliced_norm = np.rot90(sliced_norm, axes=(0, 1))
        return temp_dict

    @staticmethod
    def _get_bounding_box(mask_stack: np.ndarray) -> Tuple:
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




