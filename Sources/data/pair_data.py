import os
import logging
from typing import Iterator, Tuple, Generator, Set, List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from data.data_structures import PairBatch
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
        self.mean = 0
        self.std = 1
        self.logger = logging.getLogger(__name__)

    def folds(self, n_folds: int = 4) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Creates different folds of data for use with CV

        :param n_folds: Number of folds to be created
        :return: Generator yielding a train/test pair
        """
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

        for train_ids, test_ids in skf.split(self.total_x, self.total_y):
            yield self._create_train_test(train_ids, test_ids)

    def train_test_split(self, test_size: float = .25) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data in train/test with the specified proportion

        :param test_size: ``float`` between ``0`` and ``1`` with the test set size
        :return: Tuple with the train set and the test set
        """
        rs = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)

        train_ids, test_ids = next(rs.split(self.total_x, self.total_y))
        return self._create_train_test(train_ids, test_ids)

    def _create_train_test(self, train_ids: List[int], test_ids: List[int]) -> \
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Having the indices for the train and test sets, create the necessary List of PairComp
        for each set

        :param train_ids: Ids for the train set should be between ``0`` and ``len(self.total_x) - 1``
        :param test_ids: Ids for the test set should be between ``0`` and ``len(self.total_x) - 1``
        :return: List for the train set and list for the test set respectively
        """
        train_data = self.clinical_data.iloc[train_ids]
        test_data = self.clinical_data.iloc[test_ids]

        self.logger.debug("Generating train pairs")
        train_pairs = self._get_pairs(train_data)

        self.logger.debug("Generating test pairs")
        test_pairs = self._get_pairs(test_data)

        self.logger.debug("Generating mixed pairs")
        test_mix_pairs = self._get_compare_train(train_data, test_data)

        train_pairs = self._normalize(train_pairs)
        test_pairs = self._normalize(test_pairs, train=False)
        test_mix_pairs = self._normalize(test_mix_pairs, train=False)

        return train_pairs, test_pairs, test_mix_pairs

    @staticmethod
    def _get_pairs(df: pd.DataFrame) -> pd.DataFrame:
        """
        Get all the possible pairs for a DataFrame containing the clinical data, keeping in mind the censored
        data

        :param df: DataFrame containing all the clinical data
        :return: Iterator over PairComp with all the generated pairs
        """
        pairs = SplitPairs._get_inner_pairs(df, df)
        return pairs.sample(frac=1).reset_index(drop=True)

    @staticmethod
    def _get_compare_train(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the pairs by having one member belonging to the train dataset and the other to the test dataset

        :param train_df: DataFrame containing the clinical data for the train data set
        :param test_df: DataFrame containing the clinical data for the test data set
        :return: Iterator over PairComp with all the generated pairs
        """
        # Create the pairs where test_elem > train_elem
        pairs_test = SplitPairs._get_inner_pairs(test_df, train_df)

        # Create the pairs where train_elem > test_elem
        pairs_train = SplitPairs._get_inner_pairs(train_df, test_df)

        pairs = pd.concat([pairs_test, pairs_train])
        return pairs.sample(frac=1).reset_index(drop=True)

    @staticmethod
    def _get_inner_pairs(df: pd.DataFrame, df_comp: pd.DataFrame) -> pd.DataFrame:
        """
        Generate pairs where the survival time of ``df`` is bigger than the survival time of ``df_comp``

        :param df: :class:`DataFrame` that will be iterated
        :param df_comp: :class:`DataFrame` that will be compared against and if its values are smaller than the compared
                    value a pair will be created
        :return: List with all the generated pairs
        """
        df_comp = df_comp[df_comp['event'] == 1]

        pairs = []
        for _, row in df.iterrows():
            temp_df = df_comp.loc[(df_comp['time'] < row['time'])]

            row_pairs = pd.DataFrame()
            row_pairs['pA'] = temp_df['id']
            row_pairs['pB'] = row['id']
            row_pairs['distance'] = row['time'] - temp_df['time']
            row_pairs['comp'] = True

            pairs.append(row_pairs)

        pairs = pd.concat(pairs)
        pairs = pairs.reset_index(drop=True)

        # Swap some pairs because the comparison value would be always true otherwise
        subset = pairs.sample(frac=.5)
        pairs: pd.DataFrame = pairs.drop(subset.index, axis=0)

        subset.loc[:, ['pA', 'pB']] = subset.loc[:, ['pB', 'pA']].values
        subset['distance'] *= -1
        subset['comp'] ^= True

        pairs = pairs.append(subset)

        return pairs

    def _normalize(self, pairs: pd.DataFrame, train: bool = True) -> pd.DataFrame:
        """
        Normalize the data and return the normalization values

        :param pairs: Pandas :class:`pandas.DataFrame` containing the information, the ``distance`` column will be
                      normalized
        :param train: If true the mean and the standard deviation will be computed with the passed data, otherwise
                      the previously computed variance will be used to normalize the data
        :return: Normalized :class:`pandas.DataFrame`
        """

        if train:
            self.mean = pairs['distance'].mean()
            self.std = pairs['distance'].std()

        pairs['distance'] -= self.mean
        pairs['distance'] /= self.std
        return pairs


class BatchData:
    """
    Useful methods for working with batch data
    """

    def __init__(self):
        self.radiomic_df: pd.DataFrame = pd.read_csv(DATA_PATH_RADIOMIC_PROCESSED)
        self.logger = logging.getLogger(__name__)
        self.norm_mean = 0.
        self.norm_std = 1.

    def batches(self, pairs: pd.DataFrame,
                batch_size: int = 64,
                group_by: str = 'ids',
                load_images: bool = True,
                train: bool = True) \
            -> Generator[PairBatch, None, None]:
        """
        Generates batches based on all the pairs and the batch size

        :param pairs: Pairs to create the batch from
        :param batch_size: Size of the batch that it's going to be created. The final size will depend on the
                           ``group_by`` parameter
        :param group_by: If ``ids`` then the batch size will imply that ``batch_size == len(different_ids)`` otherwise
                         if it's ``pairs`` the batch size will imply ``batch_size == len(selected_pairs)``.
        :param load_images: Whether to load the images or not when generating the batch. This can improve performance
                            if the images are not needed.
        :param train: If this batch is to be generated for training data or for validation data. If it's for training
                      then the normalization values will be computed and stored using this data, otherwise the
                      previously computed values will be used.
        :return: Generator with the different batches that should be sent to the Machine Learning model
        """

        total_ids = np.append(pairs["pA"].values, pairs["pB"].values)
        total_ids = list(set(total_ids))  # To avoid repetitions

        features: pd.DataFrame = self.radiomic_df[total_ids]

        if train:
            self.norm_mean = features.mean(axis=1)
            self.norm_std = features.std(axis=1)

        features = features.sub(self.norm_mean, axis=0)
        features = features.div(self.norm_std, axis=0)

        if group_by == 'ids':
            return self._batch_by_ids(pairs, features, batch_size, load_images)
        else:
            return self._batch_by_pairs(pairs, features, batch_size, load_images)

    def _batch_by_ids(self,
                      pairs: pd.DataFrame,
                      features: pd.DataFrame,
                      batch_size: int,
                      load_images: bool = True) -> Generator[PairBatch, None, None]:
        """
        Create batch of pairs by setting the number of different ids = ``batch_size``

        :param pairs: Pairs to create the batch from
        :param features: :class:`pandas.DataFrame` containing the patients' features
        :param batch_size: Size of the batch that it's going to be created. The final size will depend on the
                           ``group_by`` parameter
        :param load_images: Whether to load the images or not when generating the batch. This can improve performance
                            if the images are not needed.
        :return: Generator with the different batches that should be sent to the Machine Learning model
        """

        pairs_iter = iter(pairs.itertuples())
        row = next(pairs_iter)

        # Extract bath_size ids
        while row is not None:
            ids = set()

            # Create a batch of batch_size ids
            while len(ids) < batch_size and row is not None:
                ids |= {row.pA, row.pB}
                row = next(pairs_iter, None)

            # Get all the pairs that can be formed with those ids and then remove the batch pairs from the total pairs
            batch_pairs = pairs.loc[pairs['pA'].isin(ids) & pairs['pB'].isin(ids)]
            assert len(batch_pairs)*2 >= len(ids)

            yield self._create_pair_batch(batch_pairs, features, load_images)

    def _batch_by_pairs(self,
                        pairs: pd.DataFrame,
                        features: pd.DataFrame,
                        batch_size: int,
                        load_images=True) -> Generator[PairBatch, None, None]:
        """
        Create batch of pairs by setting the ``batch_size == len(selected_pairs)``

        :param pairs: Pairs to select the batch size from
        :param features: :class:`pandas.DataFrame` containing the patients' features
        :param batch_size: Size of the batch that it's going to be created. The final size will depend on the
                           ``group_by`` parameter
        :param load_images: Whether to load the images or not when generating the batch. This can improve performance
                            if the images are not needed.
        :return: Generator with the different batches that should be sent to the Machine Learning model
        """

        for i in range(0, len(pairs), batch_size):
            values = pairs.iloc[i:(i + batch_size)]
            yield self._create_pair_batch(values, features, load_images)

    def _create_pair_batch(self,
                           pairs: pd.DataFrame,
                           features: pd.DataFrame,
                           load_images: bool = True) -> PairBatch:
        """
        Given all the ids and the pairs load the npz file for all the ids and create a PairBatch with the loaded
        npz files and the pairs

        :param pairs: Pairs to be added to the PairBatch
        :param features: :class:`pandas.DataFrame` containing the patients' features
        :return: PairBatch containing the pairs and the requested npz files loaded
        """
        # Convert ids from string to int index. Since there can be multiple images with one pair this will mean that
        # We have to return more indices related to the same pair so that's why we are using the TOTAL_ROTATIONS
        # global variable to set the indices, the generated indices are in the range:
        # idx*TOTAL_ROTATIONS ... (idx + 1)*TOTAL_ROTATIONS
        total_rotations = TOTAL_ROTATIONS if load_images else 1

        ids_list = list(set(pd.concat([pairs["pA"], pairs["pB"]])))
        ids_map, patients = self._load_patients(ids_list, features, total_rotations, load_images)

        # Create pairs information
        pairs_a, pairs_b = [], []
        for row in pairs.itertuples():
            idx_a, idx_b = ids_map[row.pA], ids_map[row.pB]

            pairs_a += list(range(idx_a, idx_a + total_rotations))
            pairs_b += list(range(idx_b, idx_b + total_rotations))

        pairs: pd.DataFrame = pairs.copy()
        pairs["pA_id"] = pairs_a
        pairs["pB_id"] = pairs_b

        # Create labels column
        pairs["labels"] = pairs["comp"].values.astype(float)

        return PairBatch(pairs=pairs, patients=patients, ids_map=ids_map)

    def _load_patients(self,
                       ids_list: List[str],
                       features: pd.DataFrame,
                       total_rotations: int,
                       load_images: bool = True) -> Tuple[Dict[str, int], pd.DataFrame]:
        """
        Load the patients information for the batch of ids, such as the image CT scan and the radiomic features
        
        :param ids_list: List of string keys to load the information for 
        :param features: :class:`pandas.DataFrame` containing all the patients' features
        :param total_rotations: 
        :param load_images: 
        :return: 
        """
        # Direct and inverse mapping from string key to index
        ids_map = {}
        selected_features, images, final_ids = [], [], []
        for i, idx in enumerate(ids_list):
            ids_map[idx] = i
            final_ids += [idx]*total_rotations
            file_path = os.path.join(DATA_PATH_PROCESSED, idx, idx + ".npz")

            # Check if the file exists, so the data has been preprocessed
            if not os.path.exists(file_path):
                self.logger.error(f"The file {file_path} could not be found. Have you pre-processed the data?")
                raise FileNotFoundError(f"The file {file_path} could not be found. Have you pre-processed the data?")

            if load_images:
                loaded_npz = np.load(file_path)
                assert len(loaded_npz.files) == total_rotations
                for item in loaded_npz:
                    loaded_array = loaded_npz[item]
                    assert loaded_array.shape == (64, 64, 64)
                    images.append(loaded_array.reshape(64, 64, 64, 1))
                loaded_npz.close()
            else:
                images += [np.array([])]*total_rotations

            # Select radiomic features
            column = features[idx].values
            selected_features += [column]*total_rotations

        assert len(images) == len(selected_features) == len(final_ids)
        elements = pd.DataFrame({
            "ids": final_ids,
            "features": selected_features,
            "images": images
        })

        return ids_map, elements
