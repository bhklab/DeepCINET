import os
import random
import logging
from itertools import takewhile, islice, repeat
from typing import Iterator, Tuple, Generator, Iterable, Set, Collection, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from data.data_structures import PairComp, PairBatch
from settings import \
    DATA_PATH_CLINICAL_PROCESSED, \
    DATA_PATH_PROCESSED, \
    DATA_PATH_RADIOMIC_PROCESSED, \
    TOTAL_ROTATIONS, \
    RANDOM_SEED


class SplitPairs:
    """
    Divides the data in training and testing. It can be divided to use with CV or only with one train/test set
    """

    def __init__(self):
        # To divide into test and validation sets we only need the clinical data
        self.clinical_data = pd.read_csv(DATA_PATH_CLINICAL_PROCESSED, index_col=0)

        self.total_x = self.clinical_data['id'].values
        self.total_y = self.clinical_data['event'].values

    def folds(self, n_folds: int = 4, compare_train: bool = False) -> Iterator[Tuple[List[PairComp], List[PairComp]]]:
        """
        Creates different folds of data for use with CV

        :param n_folds: Number of folds to be created
        :param compare_train: When creating the test pairs, create this pairs with one member belonging to the
                              test set and the other one to the train set
        :return: Generator yielding a train/test pair
        """
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

        for train_ids, test_ids in skf.split(self.total_x, self.total_y):
            yield self._create_train_test(train_ids, test_ids, compare_train=compare_train)

    def train_test_split(self, test_size: float = .25, compare_train: bool = False) \
            -> Tuple[List[PairComp], List[PairComp]]:
        """
        Split data in train/test with the specified proportion

        :param test_size: ``float`` between ``0`` and ``1`` with the test set size
        :param compare_train: When creating the test pairs, create this pairs with one member belonging to the
                              test set and the other one to the train set
        :return: Tuple with the train set and the test set
        """
        rs = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)

        train_ids, test_ids = next(rs.split(self.total_x, self.total_y))
        return self._create_train_test(train_ids, test_ids, compare_train=compare_train)

    def _create_train_test(self, train_ids: List[int], test_ids: List[int], compare_train: bool = False) -> \
            Tuple[List[PairComp], List[PairComp]]:
        """
        Having the indices for the train and test sets, create the necessary List of PairComp
        for each set

        :param train_ids: Ids for the train set should be between ``0`` and ``len(self.total_x) - 1``
        :param test_ids: Ids for the test set should be between ``0`` and ``len(self.total_x) - 1``
        :param compare_train: When creating the test pairs, create this pairs with one member belonging to the
                              test set and the other one to the train set
        :return: List for the train set and list for the test set respectively
        """
        train_data = self.clinical_data.iloc[train_ids]
        test_data = self.clinical_data.iloc[test_ids]

        train_pairs = self._get_pairs(train_data)
        if not compare_train:
            test_pairs = self._get_pairs(test_data)
        else:
            test_pairs = self._get_compare_train(train_data, test_data)

        return list(train_pairs), list(test_pairs)

    @staticmethod
    def _get_pairs(df: pd.DataFrame) -> Iterator[PairComp]:
        """
        Get all the possible pairs for a DataFrame containing the clinical data, keeping in mind the censored
        data

        :param df: DataFrame containing all the clinical data
        :return: Iterator over PairComp with all the generated pairs
        """
        pairs = SplitPairs._get_inner_pairs(df, df)

        # Since we have provided all the pairs sorted in the algorithm the output will be always
        # pair1 < pair2. We do not want the ML method to learn this but to understand the image features
        # That's why we swap random pairs
        random.shuffle(pairs)
        # logger.debug(pairs)
        return map(SplitPairs._swap_random, pairs)

    @staticmethod
    def _get_compare_train(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Iterator[PairComp]:
        """
        Create the pairs by having one member belonging to the train dataset and the other to the test dataset

        :param train_df: DataFrame containing the clinical data for the train data set
        :param test_df: DataFrame containing the clinical data for the test data set
        :return: Iterator over PairComp with all the generated pairs
        """
        # Create the pairs where test_elem > train_elem
        pairs = SplitPairs._get_inner_pairs(test_df, train_df)

        # Create the pairs where train_elem > test_elem
        pairs += SplitPairs._get_inner_pairs(train_df, test_df)
        random.shuffle(pairs)

        return map(SplitPairs._swap_random, pairs)

    @staticmethod
    def _get_inner_pairs(df: pd.DataFrame, df_comp: pd.DataFrame) -> List[PairComp]:
        """
        Generate the pairs by iterating through a :class:`DataFrame` and for each element create pairs for all elements
        that have a survival time bigger than ``df1``.

        :param df: :class:`DataFrame` that will be iterated
        :param df_comp: :class:`DataFrame` that will be compared against and if its values are smaller than the compared
                    value a pair will be created
        :return: List with all the generated pairs
        """
        df_comp = df_comp[df_comp['event'] == 1]

        pairs = []
        for _, row in df.iterrows():
            values = df_comp.loc[(df_comp['time'] < row['time']), 'id'].values
            elements = zip(values, [row['id']]*len(values), [True]*len(values))
            pairs += [PairComp(*x) for x in elements]

        return pairs

    @staticmethod
    def _swap_random(tup: PairComp) -> PairComp:
        if bool(random.randint(0, 1)):
            return PairComp(p_a=tup.p_b, p_b=tup.p_a, comp=not tup.comp)
        return tup


def get_radiomic_features() -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.debug("Reading radiomic features")
    radiomic_df: pd.DataFrame = pd.read_csv(DATA_PATH_RADIOMIC_PROCESSED)
    radiomic_df = radiomic_df.sub(radiomic_df.mean(axis=1), axis=0)
    radiomic_df = radiomic_df.div(radiomic_df.std(axis=1), axis=0)
    logger.debug("Radiomic features processed")
    return radiomic_df


class BatchData:
    """
    Useful methods for working with batch data
    """
    radiomic_df = get_radiomic_features()

    @staticmethod
    def batches(pairs: Iterable[PairComp], batch_size: int = 64, group_by: str = 'ids', load_images: bool = True) \
            -> Generator[PairBatch, None, None]:
        """
        Generates batches based on all the pairs and the batch size

        :param pairs:
        :param batch_size:
        :param group_by:
        :param load_images:
        :return:
        """
        if group_by == 'ids':
            return BatchData._batch_by_ids(pairs, batch_size, load_images)
        else:
            return BatchData._batch_by_pairs(pairs, batch_size, load_images)

    @staticmethod
    def _batch_by_ids(pairs: Iterable[PairComp], batch_size: int, load_images: bool = True) \
            -> Generator[PairBatch, None, None]:
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

            yield BatchData._create_pair_batch(batch_pairs, ids, load_images)

    @staticmethod
    def _batch_by_pairs(pairs: Iterable[PairComp], batch_size: int, load_images=True) \
            -> Generator[PairBatch, None, None]:
        for i, values in enumerate(BatchData._split(pairs, batch_size)):
            values = list(values)
            yield BatchData._create_pair_batch(values, {idx for p in values for idx in (p.p1, p.p2)}, load_images)

    @staticmethod
    def _create_pair_batch(pairs: Collection[PairComp], ids: Set[str], load_images: bool = True) -> PairBatch:
        """
        Given all the ids and the pairs load the npz file for all the ids and create a PairBatch with the loaded
        npz files and the pairs

        :param pairs: Pairs to be added to the PairBatch
        :param ids: npz files' ids that will be added to the PairBatch
        :return: PairBatch containing the pairs and the requested npz files loaded
        """
        logger = logging.getLogger(__name__)

        # Convert ids from string to int index. Since there can be multiple images with one pair this will mean that
        # We have to return more indices related to the same pair so that's why we are using the TOTAL_ROTATIONS
        # global variable to set the indices, the generated indices are in the range:
        # idx*TOTAL_ROTATIONS ... (idx + 1)*TOTAL_ROTATIONS
        total_rotations = TOTAL_ROTATIONS if load_images else 1
        ids_list = list(ids)

        # Direct and inverse mapping
        ids_map = {idx: idx_num*total_rotations for idx_num, idx in enumerate(ids_list)}
        ids_inverse = {idx_num: idx for idx, i in ids_map.items() for idx_num in range(i, i + total_rotations)}

        pairs_a = [idx for p in pairs for idx in range(ids_map[p.p_a], ids_map[p.p_a] + total_rotations)]
        pairs_b = [idx for p in pairs for idx in range(ids_map[p.p_b], ids_map[p.p_b] + total_rotations)]
        labels = [float(l) for p in pairs for l in [p.comp]*total_rotations]
        assert len(pairs_a) == len(pairs_b) == len(labels)

        labels = np.array(labels).reshape((-1, 1))

        images = []
        features = []
        for idx in ids_list:
            file_path = os.path.join(DATA_PATH_PROCESSED, idx, idx + ".npz")

            # Check if the file exists, so the data has been preprocessed
            if not os.path.exists(file_path):
                logger.error(f"The file {file_path} could not be found. Have you pre-processed the data?")
                raise FileNotFoundError(f"The file {file_path} could not be found. Have you pre-processed the data?")

            column = BatchData.radiomic_df[idx].values
            features += [column]*total_rotations

            if load_images:
                loaded_npz = np.load(file_path)
                for item in loaded_npz:
                    images.append(loaded_npz[item])
                loaded_npz.close()
            else:
                images += [np.array([])]*total_rotations

        assert len(images) == len(features)

        images = np.array(images)
        images = images.reshape((-1, 64, 64, 64, 1))
        # images = {ids_map[idx]: np.array([0, 1, 2]) for idx in ids}

        features = np.array(features)

        return PairBatch(pairs_a=pairs_a,
                         pairs_b=pairs_b,
                         labels=labels,
                         images=images,
                         ids_map=ids_map,
                         ids_inverse=ids_inverse,
                         features=features)

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
