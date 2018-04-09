"""
RawData class, useful to load the raw data, the images with all the slices in different
files
"""
import os
from typing import Tuple, Iterator

import pydicom as dcm
import numpy as np
from joblib import Parallel, delayed

from .PseudoDir import PseudoDir


class RawData:
    def __init__(self):
        super().__init__()
        self.data_path = os.getenv('DATA_RAW')
        self.cache_path = os.getenv('DATA_CACHE')
        self.elements_stored = False

        valid_dirs = filter(self._is_valid_dir, os.scandir(self.data_path))
        self._valid_dirs = [PseudoDir(x.name, x.path, x.is_dir()) for x in valid_dirs]
        self._valid_dirs_path = [str(x.name) for x in self._valid_dirs]
        print("{} valid dirs have been found".format(len(self._valid_dirs)))

    def total_elements(self):
        return len(self._valid_dirs)

    def valid_dirs_path(self):
        return self._valid_dirs_path

    def elements(self) -> Iterator[Tuple[PseudoDir, np.ndarray, np.ndarray]]:
        if not self.elements_stored:
            raise ValueError("To iterate over the elements first they have to be stored")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError("The data folder {} does not exist".format(self.data_path))

        for directory in self._valid_dirs:
            main_stack, mask_stack = self._get_mask_main_stack(directory)
            yield directory, main_stack, mask_stack

    def store_elements(self):
        """
        Creates the npz files from the dcm files for faster readings
        :return:
        """

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        jobs = os.getenv("NSLOTS", -1)
        print("Jobs: {}".format(jobs))

        generator = (delayed(self._generate_npz)(directory, i + 1) for i, directory in enumerate(self._valid_dirs))
        Parallel(n_jobs=jobs, backend='multiprocessing')(generator)

        self.elements_stored = True

    @staticmethod
    def _is_valid_dir(test_dir: os.DirEntry) -> bool:
        """
        Only some directories are valid, the ones that start with FHBO
        (which is our original Head and Neck dataset)
        and then only the only ones that have two sub dirs:
         - The scan directory
         - The mask directory
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

    def _generate_npz(self, image_dir: PseudoDir, count: int):
        numpy_file = os.path.join(self.cache_path, image_dir.name + ".npz")
        numpy_file_temp = os.path.join(self.cache_path, image_dir.name + "_temp.npz")

        print("Reading {} of {}".format(count, len(self._valid_dirs)))

        if os.path.exists(numpy_file):
            return

        main_stack, mask_stack = self.compact_files(image_dir)
        print("Saving {} file".format(numpy_file_temp))
        np.savez_compressed(numpy_file_temp, main=main_stack, mask=mask_stack)
        os.rename(numpy_file_temp, numpy_file)  # Use a temp name to avoid problems when stopping the script

    def _get_mask_main_stack(self, image_dir: PseudoDir):
        """
        Return the raw data, stacking all the files and using a temporary cache
        :param image_dir:
        :return:
        """
        numpy_file = os.path.join(self.cache_path, image_dir.name + ".npz")

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


