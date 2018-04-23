import os
import random
from itertools import takewhile, islice, repeat
from typing import Iterator, Tuple, Generator, Iterable, Set, Collection

import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split

from data.data_structures import PairComp, PairBatch
from settings import DATA_PATH_CLINICAL_PROCESSED, TOTAL_ROTATIONS, DATA_PATH_PROCESSED, RANDOM_SEED


class SplitPairs:
    """
    Divides the data in training and testing
    """

    def __init__(self):
        # To divide into test and validation sets we only need the clinical data
        self.clinical_data = pd.read_csv(DATA_PATH_CLINICAL_PROCESSED, index_col=0)

        # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        total_x = self.clinical_data['id'].values
        total_y = self.clinical_data['event'].values

        # NOTE: This part varies between executions unless a random state with a seed is passed
        train_id, test_id = train_test_split(
            total_x,
            test_size=.2,
            shuffle=True,
            stratify=total_y,
            random_state=RANDOM_SEED
        )

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
            return PairComp(tup.p_b, tup.p_a, tup.comp)
        return tup


class BatchData:
    """
    This is a batch
    """

    @staticmethod
    def batches(pairs: Iterable[PairComp], batch_size: int = 64, group_by: str = 'ids') \
            -> Generator[PairBatch, None, None]:
        """
        Generates batches based on all the pairs and the batch size

        :param pairs:
        :param batch_size:
        :param group_by:
        :return:
        """
        if group_by == 'ids':
            return BatchData._batch_by_ids(pairs, batch_size)
        else:
            return BatchData._batch_by_pairs(pairs, batch_size)

    @staticmethod
    def _batch_by_ids(pairs: Iterable[PairComp], batch_size: int) -> Generator[PairBatch, None, None]:
        total_pairs = set(pairs)

        # Extract bath_size ids
        while len(total_pairs) > 0:
            ids = set()
            batch_pairs = set()

            # Create a batch of batch_size ids
            while len(ids) < batch_size and len(total_pairs) > 0:
                pair = total_pairs.pop()
                ids |= {pair.p_a, pair.p_b}
                batch_pairs.add(pair)

            # Get all the pairs that can be formed with those ids and then remove the batch pairs from the total pairs
            batch_pairs |= {x for x in total_pairs if x.p_a in ids and x.p_b in ids}
            total_pairs -= batch_pairs
            assert len(batch_pairs)*2 >= len(ids)

            yield BatchData._create_pair_batch(batch_pairs, ids)

    @staticmethod
    def _batch_by_pairs(pairs: Iterable[PairComp], batch_size: int) -> Generator[PairBatch, None, None]:
        for i, values in enumerate(BatchData._split(pairs, batch_size)):
            values = list(values)
            yield BatchData._create_pair_batch(values, {idx for p in values for idx in (p.p1, p.p2)})

    @staticmethod
    def _create_pair_batch(pairs: Collection[PairComp], ids: Set[str]) -> PairBatch:
        """
        Given all the ids and the pairs load the npz file for all the ids and create a PairBatch with the loaded
        npz files and the pairs

        :param pairs: Pairs to be added to the PairBatch
        :param ids: npz files' ids that will be added to the PairBatch
        :return: PairBatch containing the pairs and the requested npz files loaded
        """

        # Convert ids from string to int index. Since there can be multiple images with one pair this will mean that
        # We have to return more indices related to the same pair so that's why we are using the TOTAL_ROTATIONS
        # global variable to set the indices, the generated indices are in the range:
        # idx*TOTAL_ROTATIONS ... (idx + 1)*TOTAL_ROTATIONS
        ids_list = list(ids)
        ids_map = {idx: idx_num*TOTAL_ROTATIONS for idx_num, idx in enumerate(ids_list)}
        pairs_a = [idx for p in pairs for idx in range(ids_map[p.p_a], ids_map[p.p_a] + TOTAL_ROTATIONS)]
        pairs_b = [idx for p in pairs for idx in range(ids_map[p.p_b], ids_map[p.p_b] + TOTAL_ROTATIONS)]
        labels = [[float(l)] for p in pairs for l in [p.comp]*TOTAL_ROTATIONS]
        assert len(pairs_a) == len(pairs_b) == len(labels)

        images = []
        for idx in ids_list:
            loaded = np.load(os.path.join(DATA_PATH_PROCESSED, idx, idx + ".npz"))
            for item in loaded:
                images.append(loaded[item])
            loaded.close()

        images = np.array(images)
        images = images.reshape((-1, 64, 64, 64, 1))
        # images = {ids_map[idx]: np.array([0, 1, 2]) for idx in ids}

        return PairBatch(pairs_a=pairs_a, pairs_b=pairs_b, labels=labels, images=images, ids_map=ids_map)

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
