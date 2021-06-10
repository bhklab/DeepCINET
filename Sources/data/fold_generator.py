import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from data.data_loader import Dataset


class KFoldGenerator(Dataset):
    """Class which generates train test holds for training and tuning

    """
    def __init__(self, hparams):
        Dataset.__init__(self, hparams, False)

    def k_cross_validation(self,
                           n_splits=5,
                           random_seed=520,
                           stratified=True):
        if stratified:
            return StratifiedKFold(n_splits=n_splits,
                                   shuffle=True,
                                   random_state=random_seed) \
                .split(X=self.gene_exprs,
                       y=self.drug_resps)
        else:
            return KFold(n_splits=n_splits,
                         shuffle=True,
                         random_state=random_seed)\
                .split(X=self.gene_exprs)
