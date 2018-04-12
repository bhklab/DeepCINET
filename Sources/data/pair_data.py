import os
import random
from itertools import takewhile, islice, repeat
from typing import Iterator, Tuple, Generator, Iterable, Set, Collection

import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split

from data.data_structures import PairComp, PairBatch


class SplitPairs:
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
        return map(SplitPairs._swap_random, pairs)

    @staticmethod
    def _swap_random(tup: PairComp) -> PairComp:
        if bool(random.getrandbits(1)):
            return PairComp(tup.p2, tup.p1, tup.comp)
        return tup


class BatchData:
    """
    This is a batch
    """
    def __init__(self):
        self._data_path = os.getenv("DATA_PROCESSED")

    def batches(self, pairs: Iterable[PairComp], batch_size: int = 64, group_by: str = 'ids') \
            -> Generator[PairBatch, None, None]:
        """
        Generates batches based on all the pairs and the batch size

        :param pairs:
        :param batch_size:
        :param group_by:
        :return:
        """
        if group_by == 'ids':
            return self._batch_by_ids(pairs, batch_size)
        else:
            return self._batch_by_pairs(pairs, batch_size)

    def _batch_by_ids(self, pairs: Iterable[PairComp], batch_size: int) -> Generator[PairBatch, None, None]:
        total_pairs = set(pairs)

        # Extract bath_size ids
        while len(total_pairs) > 0:
            ids = set()
            batch_pairs = set()

            # Create a batch of batch_size ids
            while len(ids) < batch_size and len(total_pairs) > 0:
                pair = total_pairs.pop()
                ids |= {pair.p1, pair.p2}
                batch_pairs.add(pair)

            # Get all the pairs that can be formed with those ids and then remove the batch pairs from the total pairs
            batch_pairs |= {x for x in total_pairs if x.p1 in ids and x.p2 in ids}
            total_pairs -= batch_pairs
            assert len(batch_pairs)*2 >= len(ids)

            yield self._create_pair_batch(batch_pairs, ids)

    def _batch_by_pairs(self, pairs: Iterable[PairComp], batch_size: int) -> Generator[PairBatch, None, None]:
        for i, values in enumerate(self._split(pairs, batch_size)):
            print(type(values))
            values = list(values)
            yield self._create_pair_batch(values, {idx for p in values for idx in (p.p1, p.p2)})

    def _create_pair_batch(self, pairs: Iterable[PairComp], ids: Set[str]) -> PairBatch:
        """
        Given all the ids and the pairs load the npz file for all the ids and create a PairBatch with the loaded
        npz files and the pairs

        :param pairs: Pairs to be added to the PairBatch
        :param ids: npz files' ids that will be added to the PairBatch
        :return: PairBatch containing the pairs and the requested npz files loaded
        """
        images = {idx: np.load(os.path.join(self._data_path, idx, idx + ".npz")).items() for idx in ids}
        return PairBatch(pairs=pairs, images=images)

    @staticmethod
    def _split(it: Iterable, n: int) -> Iterable[Iterable]:
        """
        Given an iterable create batches of size n

        :param it: Iterable
        :param n: Batch size
        :return: Batches of size n
        """
        it = iter(it)
        return takewhile(bool, (list(islice(it, n)) for _ in repeat(None)))
