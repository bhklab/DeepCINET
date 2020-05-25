import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from data.clinical_reader import ClinicalReader


class KFoldGenerator(ClinicalReader):
    """Class which generates train test holds for training and tuning

    """
    def __init__(self, hparams):
        ClinicalReader.__init__(self, hparams)

    def k_cross_validation(self,
                           n_splits=5,
                           random_seed=520,
                           stratified=True):
        if stratified:
            return StratifiedKFold(n_splits=n_splits,
                                   shuffle=True,
                                   random_state=random_seed) \
                .split(X=list(range(len(self._clinical_csv))),
                       y=self._event_list)
        else:
            return KFold(n_splits=n_splits,
                         shuffle=True,
                         random_state=random_seed)\
                .split(X=list(range(len(self._clinical_csv))))
