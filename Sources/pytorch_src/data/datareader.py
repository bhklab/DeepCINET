import pandas as pd
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

    def split(self, val_ratio, test_ratio, split_model):
        pd_size = len(self.clinical_info.index)
        train_idx = math.ceil(pd_size * (1-test_ratio-val_ratio))
        val_idx = math.ceil(pd_size * (1-test_ratio))
        return list(range(train_idx)), list(range(train_idx, val_idx)), list(range(val_idx, pd_size))


