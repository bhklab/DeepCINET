import random
import os
import nrrd
import torch
import numpy as np
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
from skimage.transform import rescale
from data.clinical_reader import ClinicalReader

class ImageLoader(ClinicalReader):
    """Class in charge of loading images

    Provides the method to load an image, and provides a method to augment the images
    """
    _AIR_CONST = -3024
    def __init__(self, hparams, idxs):
        ClinicalReader.__init__(self, hparams, idxs)
        self._random = random
        self._image_path = hparams.image_path
        self.use_images = hparams.use_images
        self._use_volume_cache = hparams.use_volume_cache

        if self._use_volume_cache:
            pool = Pool(mp.cpu_count())
            ## https://stackoverflow.com/questions/54324215/does-pathos-multiprocessing-have-starmap
            f = lambda x : self._load_image(*x)
            self._volume_cache = pool.map(f, [(i, False) for i in range(len(self._id_list))])
            pool.close()
            pool.join()
            pool.terminate()
            pool.restart()
            # for i in range(len(self._id_list)):
                # self._volume_cache.append(self._load_image(i, False))
            print("DONE CACHING IMAGES")

    def load_image_from_index(self, idx, is_train):
        return self._volume_cache[idx]
        if self._use_volume_cache: # and self.get_event_from_index(idx):
            return self._volume_cache[idx]
        return self._load_image(idx, is_train)

    def _load_image(self, idx, is_train):
        file_id = self.get_id_from_index(idx)
        file_path = os.path.join(self._image_path, file_id + ".nrrd")
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

    def _preprocess(self, image):
        if image.shape[0] >= 256:
            image = image[:256]
        else:
            pad_size = ((256 - image.shape[0], 0), (0,0), (0,0))
            image = np.pad(image, pad_width=pad_size, mode='constant', constant_values=self._AIR_CONST)
        return rescale(image, 0.5, anti_aliasing=True, multichannel=False)

    @staticmethod
    def _augment(image):
        return image


