import os
import random
from typing import Tuple, Iterator, NamedTuple

import scipy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class PairComp(NamedTuple):
    """
    Class to compare pairs containing two ids and a comparison value saying if
    T(p1) < T(p2)
    """
    p1: str
    p2: str
    comp: bool


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

    def train_pairs(self) -> Iterator[PairComp]:
        return self._get_pairs(self._train_data)

    def test_pairs(self) -> Iterator[PairComp]:
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

        print("Maximum number of pairs")
        print(df)

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
    def _get_pairs(df: pd.DataFrame) -> Iterator[PairComp]:
        """
        Get all the possible pairs for a DataFrame containing the clinical data, keeping in mind the censored
        data
        :param df: DataFrame containing all the clinical data
        :return: Iterator over PairComp
        """
        df = df.sort_values('time', ascending=True)
        df1 = df[df['event'] == 1]

        pairs = []
        for _, row in df.iterrows():
            values = df1.loc[(df1['time'] < row['time']), 'id'].values
            elems = zip(values, [row['id']]*len(values), [True]*len(values))
            pairs += [PairComp(*x) for x in elems]

        # Since we have provided all the pairs sorted in the algorithm the output will be always
        # pair1 < pair2. We do not want the ML method to learn this but to understand the image features
        # That's why we swap random pairs
        random.shuffle(pairs)
        return map(TrainData._swap_random, pairs)

    @staticmethod
    def _swap_random(tup: PairComp) -> PairComp:
        if bool(random.getrandbits(1)):
            return PairComp(tup.p2, tup.p1, tup.comp)
        return tup






