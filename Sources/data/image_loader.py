import random
import os
import nrrd
import torch
import numpy as np
from skimage.transform import rescale
from data.clinical_reader import ClinicalReader


class ImageLoader(ClinicalReader):
    """Class in charge of loading images

    Provides the method to load an image, and provides a method to augment the images
    """
    def __init__(self, hparams, idxs):
        ClinicalReader.__init__(self, hparams, idxs)
        self._random = random
        self._image_path = hparams.image_path
        self.use_images = hparams.use_images

    def load_image_from_index(self, idx, is_train):
        file_id = self.get_id_from_index(idx)
        file_path = os.path.join(self._image_path, "RADURE" + file_id + ".nrrd")
        image, headers = nrrd.read(file_path)
        image = np.transpose(image, (2, 0, 1))

        # Preprocess, consider making this offline
        image = self._preprocess(image)

        # Augmentation pipeline
        if is_train:
            self._augment(image)
        image = torch.tensor(image.copy(), dtype=torch.float32)
        image = image.view(1, image.size(0), image.size(1), image.size(2))
        return image

    @staticmethod
    def _preprocess(image):
        return rescale(image, 0.5, anti_aliasing=True, multichannel=False)

    @staticmethod
    def _augment(image):
        return image


