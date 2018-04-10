import os
from typing import Tuple, Iterator
from itertools import combinations, product
from random import shuffle

import scipy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class TrainData:
    """
    Divides the data in training and testing
    """

    def __init__(self):
        # To divide into test and validation sets we only need the clinical data
        self.clinical_data = pd.read_csv(os.getenv("DATA_CLINICAL_PROCESSED"), index_col=0)

        # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        total_x = self.clinical_data['id'].values
        total_y = self.clinical_data['event'].values

        # NOTE: This part varies between executions unless a random state with a seed is passed
        train_id, test_id = train_test_split(total_x, test_size=.2, shuffle=True, stratify=total_y)

        self._train_data = self.clinical_data[self.clinical_data['id'].isin(train_id)]
        self._test_data = self.clinical_data[self.clinical_data['id'].isin(test_id)]

    def train_pairs(self) -> Iterator[Tuple[str, str, bool]]:
        return self._get_pairs(self._train_data)

    def test_pairs(self) -> Iterator[Tuple[str, str, bool]]:
        return self._get_pairs(self._test_data)

    def train_ids(self) -> np.ndarray:
        return self._train_data['id'].values

    def test_ids(self) -> np.ndarray:
        return self._test_data['id'].values

    def print_pairs(self, data_augmentation=True):
        test_pairs_cens, test_pairs_uncens = self._possible_pairs(self._test_data, data_augmentation)
        train_pairs_cens, train_pairs_uncens = self._possible_pairs(self._train_data, data_augmentation)

        values = [
            ["Test", test_pairs_cens, test_pairs_uncens],
            ["Train", train_pairs_cens, train_pairs_uncens]
        ]

        df = pd.DataFrame(values, columns=["Set", "Censored", "Uncensored"])
        df = df.append(df.sum(axis=0), ignore_index=True)
        df['Total'] = df.sum(axis=1)
        df.loc[df['Set'] == "TestTrain", 'Set'] = "Total"

        print(df)
        print(len(list(self.test_pairs()))*4)
        # print(len(list(self.train_pairs())) * 4)

    @staticmethod
    def _possible_pairs(df: pd.DataFrame, data_augmentation=True) -> Tuple[int, int]:
        count = df.groupby('event').count()['id']
        censored_count = count[0]
        uncensored_count = count[1]

        censored_pairs = censored_count*uncensored_count
        uncensored_pairs = scipy.misc.comb(uncensored_count, 2, exact=True)
        if data_augmentation:
            # For now we will only be using 4 rotations in one axis to avoid having too much data
            censored_pairs *= 4
            uncensored_pairs *= 4

        return censored_pairs, uncensored_pairs

    @staticmethod
    def _get_pairs(df: pd.DataFrame) -> Iterator[Tuple[str, str, bool]]:
        df = df.sort_values('time', ascending=True)
        print(df)

        pairs = []
        for idx, row in df.iterrows():
            if row['event'] == 0:
                values = df.loc[(df['time'] < row['time']) & (df['event'] == 1), 'id'].values
                print(row['time'], values)


        censored = df.loc[df['event'] == 0, 'id'].values
        uncensored = df.loc[df['event'] == 1, 'id'].values

        uncensored_pairs = list(combinations(uncensored, 2))
        censored_pairs = list(product(censored, uncensored))
        pairs = uncensored_pairs + censored_pairs
        shuffle(pairs)
        for p_1, p_2 in pairs:
            d_1 = df.loc[df['id'] == p_1]
            d_2 = df.loc[df['id'] == p_2]

            t_1 = d_1['time'].values[0]
            t_2 = d_2['time'].values[0]

            # Censor information event=1 -> uncensored, event=0 -> censored
            c_1 = bool(d_1['event'].values[0])
            c_2 = bool(d_2['event'].values[0])

            # Remove invalid pairs
            if not (c_1 or c_2):
                # Both elements are censored
                continue
            elif ((not c_1 and c_2) and t_1 < t_2) or ((not c_2 and c_1) and t_2 < t_1):
                # One element is censored and not the other one and the censored time is smaller than the uncensored
                continue

            # Patient 1, Patient 2, P1 lives less than P2?
            yield p_1, p_2, t_1 < t_2

        return pairs




