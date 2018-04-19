import os
import shutil
from typing import List, Iterator, Tuple, Dict

import numpy as np
import pandas as pd
import pydicom as dcm
import scipy
from joblib import delayed, Parallel
from skimage import transform as skt

import utils
from data.data_structures import PseudoDir
from settings import \
    DATA_PATH_CACHE, \
    DATA_PATH_CLINICAL, \
    DATA_PATH_CLINICAL_PROCESSED, \
    DATA_PATH_PROCESSED, \
    DATA_PATH_RAW, \
    IMAGE_ROTATIONS

logger = utils.get_logger('data.raw_data')


class RawData:
    """
    RAW data representation from the .dcm scans.
    """

    def __init__(self):
        super().__init__()
        self.data_path = DATA_PATH_RAW
        self.cache_path = DATA_PATH_CACHE
        self.elements_stored = False

        valid_dirs = filter(self._is_valid_dir, os.scandir(self.data_path))
        self._valid_dirs = [PseudoDir(x.name, x.path, x.is_dir()) for x in valid_dirs]
        self._valid_ids = [str(x.name) for x in self._valid_dirs]
        logger.info("{} valid dirs have been found".format(len(self._valid_dirs)))

    def total_elements(self) -> int:
        return len(self._valid_dirs)

    def valid_ids(self) -> List[str]:
        """
        Return the valid ids in all the different folders. The rules for valid directories are defined i_is_valid_dir`.

        :return: A list of ids that are valid
        """
        return self._valid_ids

    def elements(self, names: List[str] = None) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
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
        logger.debug("Jobs: {}".format(jobs))

        generator = (delayed(self._generate_npz)(directory, i + 1, len(to_create))
                     for i, directory in enumerate(to_create))
        Parallel(n_jobs=jobs, backend='multiprocessing')(generator)

        self.elements_stored = True

    @staticmethod
    def _is_valid_dir(test_dir: os.DirEntry) -> bool:
        """Returns :const:`True` if it's a valid directory

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

    def _generate_npz(self, image_dir: PseudoDir, count: int = 0, total: int = 0):
        """
        Having a directory read the mask and the main folders, generate a ``.npz`` compressed file and store it
        to disk. The file will contain two arrays that can be later loaded.

        >>> import numpy as np
        >>> npz_file = np.load('filename.npz')
        >>> main_array = npz_file['main']
        >>> mask_array = npz_file['mask']

        :param image_dir: Directory containing the ``<id>`` and ``<id>-MASS`` subdirectories
        :param count: To keep track of progress, current job number
        :param total: To keep track of progress, total number of jobs
        """
        numpy_file = os.path.join(self.cache_path, image_dir.name + ".npz")
        numpy_file_temp = os.path.join(self.cache_path, image_dir.name + "_temp.npz")

        logger.info(f"Reading {count} of {total}")

        main_stack, mask_stack = self._compact_files(image_dir)
        logger.debug("Saving {} file".format(numpy_file_temp))
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
            logger.debug(f"File {numpy_file} found reading npz file")
            npz_file = np.load(numpy_file)
            main_stack = npz_file['main']
            mask_stack = npz_file['mask']
            npz_file.close()
        else:
            raise FileNotFoundError(f"File {numpy_file} does not exist, this should not happen")

        return main_stack, mask_stack

    @staticmethod
    def _compact_files(image_dir: PseudoDir) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get both the mask and the raw image as a 3D :class:`numpy.ndarray` by collecting all the different
        ``.dcm`` files and compacting them in a single 3D array.

        :param image_dir: Directory containing the ``<id>`` and ``<id>-MASS`` directories
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
    """Minimum bound for dycom file pixels values"""

    MAX_BOUND = 400.0
    """Maximum bound for dycom file pixels values"""

    X_SIZE = 64
    """Sliced box resized size at X axis"""

    Y_SIZE = 64
    """Sliced box resized size at Y axis"""

    Z_SIZE = 64
    """Sliced box resized size at Z axis"""

    # Columns from CSV sheet containing the info that we need
    COL_ID = 0
    """Column with the Id at clinical CSV file"""

    COL_AGE = 1
    """Column with the age at clinical CSV file"""

    COL_SEX = 2
    """Column with the sex at clinical CSV file"""

    COL_EVENT = 35
    """Column with the event at clinical CSV file"""

    COL_TIME = 36
    """Column with the time at clinical CSV file"""

    def __init__(self):
        self._data_path = DATA_PATH_PROCESSED
        self._clinical_info_path = DATA_PATH_CLINICAL
        self._raw_data = RawData()

    def store(self, overwrite=False) -> None:
        """
        Performs the pre process and stores all the data to disk. The saved file name will be ``<id>/<id>.npz``.
        The save location is defined by the environment variable ``DATA_PROCESSED``. It also saves a CSV file
        with the clinical information in the location defined by ``DATA_CLINICAL_PROCESSED``.

        :param overwrite: If true overwrites the images that are being written, otherwise if an image is found
                          it skips the image and creates the next one instead
        :return:
        """
        self._raw_data.store_elements()
        to_create = []
        for idx in self._raw_data.valid_ids():
            save_dir = os.path.join(self._data_path, idx)

            # Overwrite the directory if necessary
            if os.path.exists(save_dir):
                if overwrite:
                    shutil.rmtree(save_dir)
                    to_create.append(idx)
            else:
                to_create.append(idx)

        # To pre-process on Mordor (computing cluster), this variable is defined with Sun Grid
        # Engine and is the number of threads that we are allowed to use
        jobs = int(os.getenv("NSLOTS", -1))
        logger.debug(f"Jobs: {jobs}")

        generator = (delayed(self._process_individual)(idx, main_stack, mask_stack, i + 1, len(to_create))
                     for i, (idx, main_stack, mask_stack) in enumerate(self._raw_data.elements(to_create)))
        Parallel(n_jobs=jobs, backend="multiprocessing")(generator)

        # For debugging purposes
        # for i, (image_dir, main_stack, mask_stack) in enumerate(self.raw_data.elements()):
        #     self._process_individual(image_dir, main_stack, mask_stack, i + 1)

        self._write_clinical_filtered()

    def _process_individual(self, image_id: str, main_stack: np.ndarray, mask_stack: np.ndarray,
                            count: int = 0, total: int = 0):
        """
        Processes a single image and stores it to disk

        :param image_id: ID for image to be processed
        :param main_stack: 3D Array containing the main image
        :param mask_stack: 3D Array containing the mask with 1 in the pixels that contain tumour and 0
                           for the pixels that do not contain tumour
        :param count: For debugging purposes and show the progress, number of the image being processed
        :param total: For debugging purposes and show the progress, total images being processed
        :return: No return, saves the image to disk
        """
        save_dir = os.path.join(self._data_path, image_id)
        temp_dir = os.path.join(self._data_path, image_id + "_temp")

        logger.info(f"Processing dataset {image_id}, {count} of {total}")

        # Remove existing temporary directory from previous runs
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        # Get slice and normalize image
        sliced_norm = self._normalize_image(main_stack, mask_stack)

        # Rotate the image across the 3 axis for data augmentation
        rotations = self._get_rotations(sliced_norm)
        np.savez_compressed(os.path.join(temp_dir, image_id + ".npz"), **rotations)

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
        sliced /= sliced.std()

        logger.debug("Volume: {}".format(sliced.shape))

        # Resize the normalized data
        sliced = skt.resize(sliced, (self.X_SIZE, self.Y_SIZE, self.Z_SIZE), mode='symmetric')
        return sliced

    def _write_clinical_filtered(self):
        """
        Write the clinical data in the file provided by the environment variable ``DATA_CLINICAL_PROCESSED``.
        The columns that are saved into the file are ``id``, ``age``, ``sex``, ``event`` and ``time``.

        It also prints the number of maximum pairs that can be achieved with the current data.
        """
        if not os.path.exists(self._clinical_info_path):
            raise FileNotFoundError("The clinical info file has not been found at {}".format(self._clinical_info_path))

        df = pd.read_csv(self._clinical_info_path)
        df = df.take([self.COL_ID, self.COL_AGE, self.COL_SEX, self.COL_EVENT, self.COL_TIME], axis=1)
        df.columns = ['id', 'age', 'sex', 'event', 'time']
        df = df[df['id'].isin(self._raw_data.valid_ids())]  # Remove elements that are not valid data
        df.to_csv(DATA_PATH_CLINICAL_PROCESSED)

        # Compute number of possible pairs
        censored_count = df[df['event'] == 0].count()[0]
        uncensored_count = df.count()[0] - censored_count

        censored_pairs = censored_count*uncensored_count
        uncensored_pairs = scipy.misc.comb(uncensored_count, 2, exact=True)

        censored_pairs_augmented = censored_pairs*4
        uncensored_pairs_augmented = uncensored_pairs*4

        logger.info(f"Total censored: {censored_count}")
        logger.info(f"Total uncensored: {uncensored_count}")

        logger.info(f"Total censored pairs: {censored_pairs:,}")
        logger.info(f"Total uncensored pairs: {uncensored_pairs:,}")
        logger.info(f"Total pairs {censored_pairs + uncensored_pairs:,}")

        logger.info(f"Total censored pairs augmented: {censored_pairs_augmented:,}")
        logger.info(f"Total uncensored pairs augmented: {uncensored_pairs_augmented:,}")
        logger.info(f"Total pairs augmented {censored_pairs_augmented + uncensored_pairs_augmented:,}")

    @staticmethod
    def _get_rotations(sliced_norm: np.array) -> Dict[str, np.ndarray]:
        """
        Creates different rotations for each 3D image, the number of rotations for each axis is defined
        with the environment variable ``IMAGE_ROTATIONS`` which is a list separated by ``,`` Note that the
        rotations mean multiples of 90 so if ``IMAGE_ROTATIONS=1,1,4`` this means that we will have the
        following rotations:

        +-----+-----+-----+-----+
        | Num |   X |   Y |   Z |
        +=====+=====+=====+=====+
        |  1  |   0 |   0 |   0 |
        +-----+-----+-----+-----+
        |  2  |   0 |   0 |  90 |
        +-----+-----+-----+-----+
        |  3  |   0 |   0 | 180 |
        +-----+-----+-----+-----+
        |  4  |   0 |   0 | 270 |
        +-----+-----+-----+-----+

        :param sliced_norm: 3D array containing the 3D image values
        :return: Dictionary where the key is the rotation id in the form ``<deg x>_<deg y>_<deg z>`` and the
                 value is the image rotated in the corresponding axis the amount of degrees specified

                 For example if the image is rotated 90 degrees in the x axis and 0 degrees in the other ones
                 the key will bee ``090_000_000``.
        """
        temp_dict = {}

        for i in range(IMAGE_ROTATIONS['x']):
            for j in range(IMAGE_ROTATIONS['y']):
                for k in range(IMAGE_ROTATIONS['z']):
                    name = "{:03}_{:03}_{:03}".format(i*90, j*90, k*90)
                    temp_dict[name] = sliced_norm.copy()
                    sliced_norm = np.rot90(sliced_norm, axes=(0, 1))
                sliced_norm = np.rot90(sliced_norm, axes=(2, 0))
            sliced_norm = np.rot90(sliced_norm, axes=(0, 1))
        return temp_dict

    @staticmethod
    def _get_bounding_box(mask_stack: np.ndarray) -> Tuple[int, int, int, int, int, int]:
        """
        Get the bounding box of all the area containing 1s for the provided 3D numpy array

        :param mask_stack: 3D numpy array
        :return: Bounding box tuple with the minimum and maximum size in the 3 axis.
                 It returns ``x_min``, ``x_max``, ``y_min``, ``y_max``, ``z_min``, ``z_max``.
        """
        # Code found in stack overflow
        x = np.any(mask_stack, axis=(1, 2))
        y = np.any(mask_stack, axis=(0, 2))
        z = np.any(mask_stack, axis=(0, 1))

        x_min, x_max = np.where(x)[0][[0, -1]]
        y_min, y_max = np.where(y)[0][[0, -1]]
        z_min, z_max = np.where(z)[0][[0, -1]]
        return x_min, x_max, y_min, y_max, z_min, z_max
