import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import math

class PairProcessor:
    TIME_SPLIT = 1

    def __init__(self, clinical_path):
        self.clinical_info = pd.read_csv(clinical_path)

    def train_test_split(self,
                         val_ratio = 0.2,
                         test_ratio = 0.3,
                         split_model = TIME_SPLIT,
                         random_seed = 520):
        train_ids, val_ids, test_ids = self.split(val_ratio, test_ratio, split_model)
        return train_ids, val_ids, test_ids

    def k_cross_validation(self,
                          test_ratio = 0.3,
                          n_splits = 5,
                          random_seed = 520):
        pd_size = len(self.clinical_info.index)
        idx = list(range(pd_size))
        train_idx, test_idx = train_test_split(idx, test_size = test_ratio, random_state=random_seed)
        kfold = KFold(n_splits = 5, shuffle = True, random_state=random_seed).split(train_idx)
        return kfold, test_idx

    def split(self, val_ratio, test_ratio, split_model):
        pd_size = len(self.clinical_info.index)
        idx = list(range(pd_size))
        train_idx, test_idx = train_test_split(idx, test_size = test_ratio, random_state=random_seed)
        train_idx, val_idx = train_test_split(train_idx, test_size = val_ratio/(1-test_ratio), random_state=random_seed)
        return train_idx, val_idx, test_idx


