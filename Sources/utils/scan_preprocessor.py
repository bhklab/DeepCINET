import os
import math
import collections
from typing import Tuple

import pydicom
import numpy as np
import skimage.transform as skt
import shutil


BoundingBox = collections.namedtuple('BoundingBox', ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'])


class PseudoDir:
    def __init__(self, name, path, is_dir):
        self.name = name
        self.path = path
        self._is_dir = is_dir

    def is_dir(self):
        return self._is_dir


class ScanNormalizer:
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0

    X_SIZE = 64
    Y_SIZE = 64
    Z_SIZE = 64

    def __init__(self, dirs, output_dir, overwrite=False):
        self.dirs = []
        self.output_dir = output_dir
        self.count = 0
        self.overwrite = overwrite

        for d in dirs:
            self.dirs.append(PseudoDir(d.name, d.path, d.is_dir()))

    def process_images(self):
        for i, image in enumerate(self.dirs):
            self.process_single(image, i + 1)

    def process_single(self, image_dir: os.DirEntry, count):

        save_dir = os.path.join(self.output_dir, image_dir.name)
        temp_dir = os.path.join(self.output_dir, image_dir.name + "_temp")

        # Check if the directory exists to avoid overwriting it
        if os.path.exists(save_dir) and not self.overwrite:
            return

        if not all(x in os.listdir(image_dir.path) for x in [image_dir.name, image_dir.name + "-MASS"]):
            raise FileNotFoundError("Dir {} does not have the necessary files".format(image_dir.name))

        print("Processing dataset {}, {} of {}".format(image_dir.name, count, len(self.dirs)))

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        main_stack, mask_stack = self.compact_files(image_dir)

        # Get sliced image
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_bounding_box(mask_stack)
        sliced = main_stack[x_min:x_max, y_min:y_max, z_min:z_max]

        # Normalize the sliced part
        sliced = sliced.clip(self.MIN_BOUND, self.MAX_BOUND)
        sliced_norm = (sliced - self.MIN_BOUND)/(self.MAX_BOUND - self.MIN_BOUND)

        # Apply mask
        sliced_norm *= mask_stack[x_min:x_max, y_min:y_max, z_min:z_max]

        print("Volume: {}".format(sliced_norm.shape))

        # Resize the normalized data
        sliced_norm = skt.resize(sliced_norm, (self.X_SIZE, self.Y_SIZE, self.Z_SIZE), mode='symmetric')

        # Rotate the image across the 3 axis for data augmentation
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    save_name = os.path.join(temp_dir, "normalized_{:03}_{:03}_{:03}.npy".format(i*90, j*90, k*90))
                    np.save(save_name, sliced_norm)
                    sliced_norm = np.rot90(sliced_norm, axes=(0, 1))
                sliced_norm = np.rot90(sliced_norm, axes=(2, 0))
            sliced_norm = np.rot90(sliced_norm, axes=(0, 1))

        # Rename after finishing to be able to stop in the middle
        os.rename(temp_dir, save_dir)

        # For debugging purposes it saves all the images
        # for i in range(self.Z_SIZE):
        #     print("Saving {} of {}".format(i + 1, self.Z_SIZE))
        #     scipy.misc.imsave(os.path.join(save_dir, "deb_{}.png".format(i)), sliced_norm[:, :, i]*255)
        #

    @staticmethod
    def compact_files(image_dir: os.DirEntry) -> Tuple[np.array, np.array]:
        """
        Get a numpy array containing the 3D image concatenating all the slices in the selected dir
        :param image_dir: Directory containing all the images
        :return: Tuple with the 3D image and the 3D mask as numpy arrays
        """
        main_path = os.path.join(image_dir.path, image_dir.name)
        mask_path = main_path + "-MASS"
        total_main = [pydicom.dcmread(x.path).pixel_array for x in os.scandir(main_path)]
        total_mask = [pydicom.dcmread(x.path).pixel_array for x in os.scandir(mask_path)]

        main_stack = np.stack(total_main, axis=2)
        mask_stack = np.stack(total_mask, axis=2)
        mask_stack[mask_stack > 1] = 1
        return main_stack, mask_stack

    @staticmethod
    def get_bounding_box(mask_stack: np.array) -> BoundingBox:
        """
        Get the bounding box of all the area containing 1s
        :param mask_stack: 3D numpy array
        :return: Bounding box tuple with the minimum and maximum size in the 3 axis
        """
        x = np.any(mask_stack, axis=(1, 2))
        y = np.any(mask_stack, axis=(0, 2))
        z = np.any(mask_stack, axis=(0, 1))

        x_min, x_max = np.where(x)[0][[0, -1]]
        y_min, y_max = np.where(y)[0][[0, -1]]
        z_min, z_max = np.where(z)[0][[0, -1]]
        return BoundingBox(x_min, x_max, y_min, y_max, z_min, z_max)
