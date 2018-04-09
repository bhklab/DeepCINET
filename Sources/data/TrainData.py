import os
from typing import Tuple, List

import pandas as pd


class TrainData:
    """
    Divides the data in training and testing
    """

    def __init__(self):
        # To divide into test and validation sets we only need the clinical data
        self.clinical_path = os.getenv("DATA_CLINICAL")
        self.clinical_data = pd.read_csv(self.clinical_path)

        # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split

        self._train_x = []
        self._train_y = []
        self._test_x = []
        self._test_y = []

    def train(self) -> Tuple[List[Tuple[str, str]], List[bool]]:
        return self._train_x, self._train_y

    def test(self) -> Tuple[List[Tuple[str, str]], List[bool]]:
        return self._test_x, self._test_y


