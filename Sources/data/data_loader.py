import torch
import torch.utils.data

import random
from random import randint

import os
import numpy as np

from pymrmre import mrmr
import pandas as pd

from data.clinical_loader import ClinicalLoader
from data.image_loader import ImageLoader
from data.radiomics_loader import RadiomicsLoader


class Dataset(torch.utils.data.Dataset):
    """Data set class which returns a pytorch data set object

        Returns a iterable data set object extending from the pytorch dataset
        object.
    """
    def __init__(self, idxs, hparams, is_train):
        self._clinical_loader = ClinicalLoader(hparams, idxs)
        self._image_loader = ImageLoader(hparams, idxs)
        self._radiomics_loader = RadiomicsLoader(hparams, idxs)

        self._use_images = hparams.use_images
        self._use_radiomics = hparams.use_radiomics
        self._use_clinical = hparams.use_clinical

        self._is_train = is_train
        self._sample_list = self._build_pairs(hparams.transitive_pairs)

    def __len__(self):
        return len(self._sample_list)

    def __getitem__(self, index):
        return self.train_item(index) if self._is_train else self.test_item(index)

    def train_item(self, pair_idx):
        row = self._sample_list[pair_idx]
        volume1, r_var1, c_var1 = self._load_item(row['idxA'])
        volume2, r_var2, c_var2 = self._load_item(row['idxB'])
        label = torch.tensor(row['label'], dtype=torch.float32)
        return {'volumeA': volume1,
                'scalarA': torch.cat((r_var1, c_var1), dim=1),
                'volumeB': volume2,
                'scalarB': torch.cat((r_var2, c_var2), dim=1),
                'labels': label}

    def test_item(self, idx):
        volume, r_var, c_var = self._load_item(idx)
        event = self._clinical_loader.get_event_from_index(idx)
        event_time = self._clinical_loader.get_survival_time_from_index(idx)
        return {'volume': volume,
                'scalar': torch.cat((r_var, c_var), dim=1),
                'event': event,
                'event_time': event_time}

    def _load_item(self, idx):
        """ Function to load the features and volumes of a patient

        if use_images, use_radiomics or use_clinical is not set, we expect the
        corresponding _load function to return an empty tensor
        :param idx: the index of the patient in our clinical csv
        :return: returns a tuple containing the volume, radiomic and clinical variables
        """
        volume = self._load_image(idx)
        r_var = self._load_pyradiomics(idx)
        c_var = self._load_clinical(idx)
        return volume, r_var, c_var

    def _load_pyradiomics(self, idx):
        return self._radiomics_loader.load_radiomics_from_index(idx) \
            if self._use_radiomics else torch.empty(0)

    def _load_clinical(self, idx):
        return torch.empty(0)

    def _load_image(self, idx):
        return self._image_loader.load_image_from_index(idx, self._is_train) \
                    if self._use_images else torch.empty(0)

    def _build_pairs(self, num_neighbours):
        return self._clinical_loader.get_concordant_pair_list(num_neighbours) \
            if self._is_train else self._clinical_loader.get_patient_list()

