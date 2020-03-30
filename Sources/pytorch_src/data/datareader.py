import pandas as pd
import math

class PairProcessor:
    TIME_SPLIT = 1

    def __init__(self, target_path):
        self.clinical_info = pd.read_csv(target_path, index_col = 0)

    def train_test_split(self, test_ratio = 0.25,
                         split_model = TIME_SPLIT,
                         random_seed = 520):
        train_ids, test_ids = self.split(test_ratio, split_model)
        return train_ids, test_ids

    def split(self, test_ratio, split_model):
        pd_size = len(self.clinical_info.index)
        train_idx = math.ceil(pd_size * (1-test_ratio))
        if split_model == PairProcessor.TIME_SPLIT:
            self.clinical_info.sort_values(by = 'time', axis = 0, ascending=False,
                                           inplace= True)
            self.clinical_info.reset_index()
            train_ids = self.clinical_info[:train_idx]
            test_ids = self.clinical_info[train_idx:]

        return train_ids[['id']].to_numpy().flatten(), test_ids[['id']].to_numpy().flatten()


