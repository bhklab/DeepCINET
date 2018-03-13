import os
import math
import pydicom
import numpy as np
import json
import scipy.misc as scm


class BoundingBox:
    def __init__(self, x_min=None, y_min=None, z_min=None, x_max=None, y_max=None, z_max=None):
        self.x_min = int(x_min)
        self.y_min = int(y_min)
        self.z_min = int(z_min)
        self.x_max = int(x_max)
        self.y_max = int(y_max)
        self.z_max = int(z_max)

    def set_extreme(self):
        self.x_min = self.y_min = self.z_min = math.inf
        self.x_max = self.y_max = self.z_max = -math.inf


class ScanNormalizer:
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0

    def __init__(self, dirs, output_dir):
        self.dirs = dirs
        self.output_dir = output_dir

    def process_images(self):
        list(map(self.process_single, self.dirs))

    def process_single(self, image_dir: os.DirEntry):
        # We need to get the two different dirs
        subs = os.scandir(image_dir.path)

        main_dir = None
        mask_dir = None
        for sub_dir in subs:
            if sub_dir.name == image_dir.name:
                main_dir = sub_dir
            elif sub_dir.name == image_dir.name + "-MASS":
                mask_dir = sub_dir

        if main_dir is None or mask_dir is None:
            raise FileNotFoundError("Dir {} does not have the necessary files".format(image_dir.name))

        print("Dataset {}".format(main_dir.name))

        main_files = os.scandir(main_dir)
        mask_files = os.scandir(mask_dir)

        total_mask = []
        total_main = []

        new_dir_name = os.path.join(self.output_dir, image_dir.name)
        os.makedirs(new_dir_name)

        for main, mask in zip(main_files, mask_files):
            # Safe check, make sure that both files have the same name
            if main.name != mask.name:
                raise FileNotFoundError("The two files do not have the same name")

            main_dcm = pydicom.dcmread(main.path)
            mask_dcm = pydicom.dcmread(mask.path)

            main_x = main_dcm.pixel_array
            mask_x = mask_dcm.pixel_array
            mask_x[mask_x > 1] = 1
            total_mask.append(mask_x)

            # Normalize the image
            main_x = main_x.clip(self.MIN_BOUND, self.MAX_BOUND)
            main_norm = (main_x - self.MIN_BOUND)/(self.MAX_BOUND - self.MIN_BOUND)

            main_norm *= mask_x
            total_main.append(main_norm)

        main_stack = np.stack(total_main, axis=2)
        mask_stack = np.stack(total_mask, axis=2)

        # TODO: Use scipy zoom to reshape image

        # Get bounding box
        x = np.any(mask_stack, axis=(1, 2))
        y = np.any(mask_stack, axis=(0, 2))
        z = np.any(mask_stack, axis=(0, 1))

        x_min, x_max = np.where(x)[0][[0, -1]]
        y_min, y_max = np.where(y)[0][[0, -1]]
        z_min, z_max = np.where(z)[0][[0, -1]]

        print(type(x_min))

        cut = main_stack[x_min:x_max, y_min:y_max, z_min:z_max]
        print(cut.shape)
        cut2 = scm.imresize(cut, (300, 300, 300))
        print(cut2.shape)

        np.save(os.path.join(new_dir_name, "main_masked.npy"), main_stack)
        np.save(os.path.join(new_dir_name, "mask.npy"), mask_stack)






