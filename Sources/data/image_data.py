import os
import shutil
from typing import List, Iterator, Tuple, Dict

import numpy as np
import pandas as pd
import pydicom as dcm
import scipy
from joblib import delayed, Parallel
from skimage import transform as skt

from data.data_structures import PseudoDir


class RawData:
    def __init__(self):
        super().__init__()
        self.data_path = os.getenv('DATA_RAW')
        self.cache_path = os.getenv('DATA_CACHE')
        self.elements_stored = False

        valid_dirs = filter(self._is_valid_dir, os.scandir(self.data_path))
        self._valid_dirs = [PseudoDir(x.name, x.path, x.is_dir()) for x in valid_dirs]
        self._valid_ids = [str(x.name) for x in self._valid_dirs]
        print("{} valid dirs have been found".format(len(self._valid_dirs)))

    def total_elements(self):
        return len(self._valid_dirs)

    def valid_ids(self):
        return self._valid_ids

    def elements(self, names: List[str]=None) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
        if not self.elements_stored:
            raise ValueError("To iterate over the elements first they have to be stored")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError("The data folder {} does not exist".format(self.data_path))

        if names is None:
            names = [d.name for d in self._valid_dirs]

        for name in names:
            main_stack, mask_stack = self._get_mask_main_stack(name)
            yield name, main_stack, mask_stack

    def store_elements(self):
        """
        Creates the npz files from the dcm files for faster readings
        :return:
        """

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        # Get the files that have to be created
        to_create = []
        for d in self._valid_dirs:
            name = os.path.join(self.cache_path, d.name + ".npz")
            if not os.path.exists(name):
                to_create.append(d)

        # To pre-process on Mordor
        jobs = int(os.getenv("NSLOTS", -1))
        print("Jobs: {}".format(jobs))

        generator = (delayed(self._generate_npz)(directory, i + 1, len(to_create))
                     for i, directory in enumerate(to_create))
        Parallel(n_jobs=jobs, backend='multiprocessing')(generator)

        self.elements_stored = True

    @staticmethod
    def _is_valid_dir(test_dir: os.DirEntry) -> bool:
        """
        Only some directories are valid, the ones that start with FHBO
        (which is our original Head and Neck dataset)
        and then only the only ones that have two sub dirs:
        * The scan directory
        * The mask directory
        :param test_dir: Directory to be tested
        :return: True or False depending on the folder conditions
        """

        # Only names starting with FHBO are valid
        # We are looking for directories
        name = str(test_dir.name)
        if not test_dir.is_dir() or not name.startswith("FHBO"):
            return False

        # Check if it has two sub dirs that start with the same name
        sub_dirs = list(filter(lambda x: x.is_dir() and str(x.name).startswith(name), os.scandir(test_dir.path)))
        return len(sub_dirs) >= 2

    def _generate_npz(self, image_dir: PseudoDir, count: int, total: int):
        numpy_file = os.path.join(self.cache_path, image_dir.name + ".npz")
        numpy_file_temp = os.path.join(self.cache_path, image_dir.name + "_temp.npz")

        print("Reading {} of {}".format(count, total))

        main_stack, mask_stack = self.compact_files(image_dir)
        print("Saving {} file".format(numpy_file_temp))
        np.savez_compressed(numpy_file_temp, main=main_stack, mask=mask_stack)
        os.rename(numpy_file_temp, numpy_file)  # Use a temp name to avoid problems when stopping the script

    def _get_mask_main_stack(self, image_name: str):
        """
        Return the raw data, stacking all the files and using a temporary cache
        :param image_name:
        :return:
        """
        numpy_file = os.path.join(self.cache_path, image_name + ".npz")

        # Load .npz file instead of .dcm if we have already read it
        if os.path.exists(numpy_file):
            print("File {} found reading npz file".format(numpy_file))
            npz_file = np.load(numpy_file)
            main_stack = npz_file['main']
            mask_stack = npz_file['mask']
            npz_file.close()
        else:
            raise FileNotFoundError("File {} does not exist, this should not happen".format(numpy_file))

        return main_stack, mask_stack

    @staticmethod
    def compact_files(image_dir: PseudoDir) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a numpy array containing the 3D image concatenating all the slices in the selected dir
        :param image_dir: Directory containing all the images
        :return: Tuple with the 3D image and the 3D mask as numpy arrays
        """
        # The directory should have the NAME and the NAME-MASS subdirectories which contain the files we need
        if not all(x in os.listdir(image_dir.path) for x in [image_dir.name, image_dir.name + "-MASS"]):
            raise FileNotFoundError("Dir {} does not have the necessary files".format(image_dir.name))

        main_path = os.path.join(image_dir.path, image_dir.name)
        mask_path = main_path + "-MASS"

        files_main = [PseudoDir(x.name, x.path, x.is_dir()) for x in os.scandir(main_path)
                      if str(x.name).startswith("IMG")]
        files_main = sorted(files_main, key=lambda x: x.name)

        files_mask = [PseudoDir(x.name, x.path, x.is_dir()) for x in os.scandir(mask_path)
                      if str(x.name).startswith("IMG")]
        files_mask = sorted(files_mask, key=lambda x: x.name)

        total_main = [dcm.dcmread(x.path).pixel_array for x in files_main]
        total_mask = [dcm.dcmread(x.path).pixel_array for x in files_mask]

        main_stack = np.stack(total_main, axis=2)
        mask_stack = np.stack(total_mask, axis=2)
        mask_stack[mask_stack > 1] = 1
        return main_stack, mask_stack


class PreProcessedData:
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0

    X_SIZE = 64
    Y_SIZE = 64
    Z_SIZE = 64

    # Columns from CSV sheet containing the info that we need
    COL_ID = 0
    COL_AGE = 1
    COL_SEX = 2
    COL_EVENT = 35
    COL_TIME = 36

    def __init__(self):
        self._data_path = os.getenv('DATA_PROCESSED')
        self._clinical_info_path = os.getenv('DATA_CLINICAL')
        self._overwrite = False
        self._raw_data = RawData()

    def store(self, overwrite=False) -> None:
        """
        Performs the pre process and stores all the data to disk
        """
        self._overwrite = overwrite
        self._raw_data.store_elements()
        to_create = []
        for idx in self._raw_data.valid_ids():
            save_dir = os.path.join(self._data_path, idx)

            # Overwrite the directory if necessary
            if os.path.exists(save_dir):
                if self._overwrite:
                    shutil.rmtree(save_dir)
                    to_create.append(idx)
            else:
                to_create.append(idx)

        # To pre-process on Mordor
        jobs = int(os.getenv("NSLOTS", -1))
        print("Jobs: {}".format(jobs))

        generator = (delayed(self._process_individual)(idx, main_stack, mask_stack, i + 1, len(to_create))
                     for i, (idx, main_stack, mask_stack) in enumerate(self._raw_data.elements(to_create)))
        Parallel(n_jobs=jobs, backend="multiprocessing")(generator)

        # For debugging purposes
        # for i, (image_dir, main_stack, mask_stack) in enumerate(self.raw_data.elements()):
        #     self._process_individual(image_dir, main_stack, mask_stack, i + 1)

        self._write_clinical_filtered()

    def _process_individual(self, image_name: str, main_stack: np.ndarray, mask_stack: np.ndarray,
                            count: int, total: int):
        save_dir = os.path.join(self._data_path, image_name)
        temp_dir = os.path.join(self._data_path, image_name + "_temp")

        print("Processing dataset {}, {} of {}".format(image_name, count, total))

        # Remove existing temporary directory from previous runs
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        # Get slice and normalize image
        sliced_norm = self._normalize_image(main_stack, mask_stack)

        # Rotate the image across the 3 axis for data augmentation
        rotations = self._get_rotations(sliced_norm)
        np.savez_compressed(os.path.join(temp_dir, image_name + ".npz"), **rotations)

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
        if not os.path.exists(self._clinical_info_path):
            raise FileNotFoundError("The clinical info file has not been found at {}".format(self._clinical_info_path))

        df = pd.read_csv(self._clinical_info_path)
        df = df.take([self.COL_ID, self.COL_AGE, self.COL_SEX, self.COL_EVENT, self.COL_TIME], axis=1)
        df.columns = ['id', 'age', 'sex', 'event', 'time']
        df = df[df['id'].isin(self._raw_data.valid_ids())]              # Remove elements that are not valid data
        df.to_csv(os.getenv('DATA_CLINICAL_PROCESSED'))

        # Compute number of possible pairs
        censored_count = df[df['event'] == 0].count()[0]
        uncensored_count = df.count()[0] - censored_count

        censored_pairs = censored_count * uncensored_count
        uncensored_pairs = scipy.misc.comb(uncensored_count, 2, exact=True)

        censored_pairs_augmented = censored_pairs * 4
        uncensored_pairs_augmented = uncensored_pairs * 4

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