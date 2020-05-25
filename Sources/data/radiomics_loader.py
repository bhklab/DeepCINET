from data.clinical_reader import ClinicalReader
import torch
import numpy as np
import pandas as pd


class RadiomicsLoader(ClinicalReader):
    """Class in charge of loading radiomics variables

    Provides functions to build all the concordant pairs as well as
    utilities to perform one hot encoding
    """
    def __init__(self, hparams, idxs):
        """
        We expect radiomics csv and clinical csv to have the same order
        :param hparams:
        :param idxs:
        """
        ClinicalReader.__init__(self, hparams, idxs)
        if hparams.use_radiomics:
            self._radiomics_csv = pd.read_csv(hparams.radiomics_path, index_col=0) \
                .iloc[idxs]
            self._radiomics_csv.drop(columns=['id'], inplace=True)

    def load_radiomics_from_index(self, idx):
        data = self._radiomics_csv.iloc[idx].to_numpy()
        return torch.tensor(np.nan_to_num(data), dtype=torch.float32)
