import os
import logging
import math
from typing import Iterator, Tuple, Generator, List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, LeaveOneOut, BaseCrossValidator

from data.data_structures import PairBatch
from tensorflow_src.config import TOTAL_ROTATIONS, \
    RANDOM_SEED


class SplitPairs:
    """
    Generates and divides the data into:
      - Training pairs: pairs where both elements belong to the training set
      - Testing pairs: paris where both elements belong to the test set
      - Mixed pairs: pairs where one element belongs to the train set and the other to the test set
    It can also be used to create the Cross Validation folds
    """

    def __init__(self, target_path, survival):
        # To divide into test and validation sets we only need the clinical data
        self.target_data = pd.read_csv(target_path, index_col=0)
        if ~survival:
            self.target_data['even'] = 1
            self.target_data['time']

        self.total_x = self.target_data['id'].values
        self.total_y = self.target_data['event'].values
        self.mean = 0
        self.std = 1
        self.logger = logging.getLogger(__name__)

    def folds(self, n_folds: int = 4, random_seed=RANDOM_SEED,
              random: bool = False, ) -> Iterator[Tuple[int, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]:
        """
        Creates different folds of data for use with CV
        :param n_folds: Number of folds to be created, if negative, the number of folds will be created using
                        Leave One Out
        :param random: Whether to create random pairs, use this to verify the model, **Never** to train a real model
                       It changes the labels randomly
        :return: Iterator with the fold number and its corresponding train and test sets
        """
        skf = self._get_folds_generator(n_folds, random_seed)
        n_folds = self.get_n_splits(n_folds)
        generator = skf.split(self.total_x, self.total_y)

        self.logger.info(f"Folds: {n_folds}")
        # Slurm configuration
        task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))
        task_count = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 0))
        task_count = int(os.getenv('TASKS_COUNT', task_count))
        if task_count > 0:
            tasks_list = self._tasks_distribution(n_folds, task_count)
            task_begin, task_end = tasks_list[task_id]

            enum_generator = zip(range(task_begin, task_end), list(generator)[task_begin:task_end])
        else:
            enum_generator = enumerate(generator)
        return enum_generator

    def get_n_splits(self, n_folds: int = 4, random_seed=RANDOM_SEED) -> int:
        return self._get_folds_generator(n_folds, random_seed).get_n_splits(self.total_y, self.total_y)

    def survival_categorizing(self, models, threshold, category: int = 5):
        """
         Designed for define the way of splitting data based on the survival distribution or based on the
         categorizing data by considering threshold. Setting classification to use in splitting data

         :type category: the number of categories
         :param models: int values can be  ``0``,``1`` or 2 in 0 is pure model and categorical data is event
         1 is the model based on the survival distribution
         2 is the model based on the threshould
         :param threshold: the threshold that is used in the second model for spliting data
         :param number of bins that is used for distribution
         """
        self.target_data['category'] = self.target_data['event']
        if models == 1:
            clinic_time = self.target_data['time'].copy()
            clinic_time.sort_values(inplace=True)
            clinic_time = clinic_time.reset_index(drop=True)
            block = int(clinic_time.size / category)
            for i in range(0, category):
                self.target_data.loc[self.target_data['time'] > clinic_time[
                    i * block], 'category'] = i  # +(self.clinical_data['event']) * category
            self.category = self.target_data['category'].values
        if models == 2:
            self.target_data.loc[self.target_data['time'] > threshold, 'category'] = 2 + self.target_data['event']
            self.target_data.loc[self.target_data['time'] <= threshold, 'category'] = 0 + self.target_data[
                'event']
            self.category = self.target_data['category'].values

    def train_test_split(self,
                         test_size: float = .25,
                         random: bool = False,
                         models: int = 0,
                         threshold: float = 2,
                         category: int = 4,
                         random_seed: int = RANDOM_SEED) -> Tuple[List[int], List[int]]:
        """
        Split data in train/test with the specified proportion

        :param random_seed: the random seed which need to use for categorizing
        :param category: the number of categories
        :param threshold: the threshold that is used in the second model for spliting data
        :param models: int values can be  ``0``,``1`` or 2 in 0 is pure model and categorical data is event
         1 is the model based on the survival distribution
        :param test_size: ``float`` between ``0`` and ``1`` with the test set size
        :param random: Whether to create random pairs, use this to verify the model, **Never** to train a real model
                       It changes the labels randomly
        :return: Tuple with the train set and the test set
        """
        self.survival_categorizing(models, threshold, category)
        rs = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)  # random_state=)

        train_ids, test_ids = next(rs.split(self.total_x, self.total_y))
        return train_ids, test_ids  # self._create_train_test(train_ids, test_ids, random)

    @staticmethod
    def _get_folds_generator(n_folds: int, random_seed=RANDOM_SEED) -> BaseCrossValidator:
        if n_folds < 0:
            return LeaveOneOut()
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    def create_train_test(self,
                          train_data: pd.DataFrame,
                          test_data: pd.DataFrame,
                          random: bool, distance: float = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Having the indices for the train and test sets, create the necessary List of PairComp
        for each set


        :param distance:
        :param train_data: dataFrame containing train data``
        :param test_data: dataFrame containging test data``
        :param random:
        :return: List for the train set and list for the test set respectively
        """

        self.logger.debug("Generating train pairs")
        train_pairs = self._get_pairs(train_data, random, distance=distance)

        self.logger.debug("Generating test pairs")
        test_pairs = self._get_pairs(test_data, random, distance=distance)

        self.logger.debug("Generating mixed pairs")
        test_mix_pairs = self._get_compare_train(train_data, test_data, random)

        train_pairs = self._normalize(train_pairs)
        test_pairs = self._normalize(test_pairs, train=False)
        test_mix_pairs = self._normalize(test_mix_pairs, train=False)

        return train_pairs, test_pairs, test_mix_pairs

    def _tasks_distribution(self, total_tasks: int, workers: int) -> List[Tuple[int, int]]:
        length = int(math.ceil(total_tasks / workers))
        limit = total_tasks - (length - 1) * workers

        self.logger.debug(f"Tasks: {total_tasks}, Workers: {workers}, Length: {length}, Limit: {limit}")

        task_list = []
        prev_end = 0
        for i in range(workers):
            task_begin = prev_end
            task_end = task_begin + length - (0 if i < limit else 1)
            task_end = prev_end = min(task_end, total_tasks)

            task_list.append((task_begin, task_end))

        return task_list

    @staticmethod
    def _get_pairs(df: pd.DataFrame, random: bool, distance: float = 0) -> pd.DataFrame:
        """
        Get all the possible pairs for a DataFrame containing the clinical data, keeping in mind the censored
        data

        :param df: DataFrame containing all the clinical data
        :return: Iterator over PairComp with all the generated pairs
        """
        pairs = SplitPairs._get_inner_pairs(df, df, random, distance=distance)

        if len(pairs) <= 0:
            return pairs
        return pairs.sample(frac=1).reset_index(drop=True)

    @staticmethod
    def _get_compare_train(train_df: pd.DataFrame,
                           test_df: pd.DataFrame,
                           random: bool, distance: float = 0) -> pd.DataFrame:
        """
        Create the pairs by having one member belonging to the train dataset and the other to the test dataset

        :param train_df: DataFrame containing the clinical data for the train data set
        :param test_df: DataFrame containing the clinical data for the test data set
        :return: Iterator over PairComp with all the generated pairs
        """
        # Create the pairs where test_elem > train_elem
        pairs = SplitPairs._get_inner_pairs(test_df, train_df, random, censoring=False, distance=distance)
        pairs = pairs.append(SplitPairs._get_inner_pairs(train_df, test_df, random, censoring=False, distance=distance),
                             ignore_index=True, sort=False)

        return pairs.sample(frac=1).reset_index(drop=True)

    @staticmethod
    def _get_inner_pairs(df: pd.DataFrame,
                         df_comp: pd.DataFrame,
                         random: bool,
                         censoring: bool = True, distance: float = 0) -> pd.DataFrame:
        """
        Generate pairs  where the survival time of ``df`` is bigger than the survival time of ``df_comp``

        :param df: :class:`DataFrame` that will be iterated
        :param df_comp: :class:`DataFrame` that will be compared against and if its values are smaller than the compared
                        value a pair will be created /todo write the description for this part
        :return: List with all the generated pairs. **Note**: The pairs are **not** in a random order
        """
        df_comp = df_comp[df_comp['event'] == 1]

        pairs = []
        for _, row in df.iterrows():
            # For mixed pairs only compare against the uncensored elements to avoid problems when predicting
            # Survival time
            # if not censoring or row['event'] == 1:
            #    temp_df = df_comp.loc[df_comp['id'] != row['id'], ['id', 'time']]
            # else: /todo take a look at this code again
            if (distance > 0):
                #temp_df = df_comp.loc[((row['time'] - df_comp['time']) > distance) & (row['dataSet'] == df_comp['dataSet']), ['id','time']]
                temp_df = df_comp.loc[((row['time'] -df_comp['time']) > distance),['id','time']]
            else:
                temp_df = df_comp.loc[(df_comp['time'] < row['time']), ['id', 'time']]
            row_pairs = pd.DataFrame()
            row_pairs['pA'] = temp_df['id']
            row_pairs['pB'] = row['id']
            row_pairs['distance'] = row['time'] - temp_df['time']
            row_pairs['comp'] = row['time'] > temp_df['time']

            pairs.append(row_pairs)

        pairs = pd.concat(pairs)
        pairs = pairs.reset_index(drop=True)

        rand_pairs = pairs.sample(frac=.5)
        rand_pairs[['pA', 'pB']] = rand_pairs[['pB', 'pA']]
        rand_pairs['distance'] *= -1
        rand_pairs['comp'] ^= True
        pairs.update(rand_pairs)

        if random:
            # Hack to test some values
            rand_bool = np.random.randint(2, size=len(pairs))
            pairs['comp'] = rand_bool.astype(bool)
            pairs['distance'] *= -1 * rand_bool

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

    def __init__(self, radiomic_df):
        self.radiomic_df = radiomic_df
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
                      load_images: bool = False) -> Generator[PairBatch, None, None]:
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

        pairs: pd.DataFrame = pairs.copy()

        while len(pairs) > 0:
            ids = set()
            for row in pairs.itertuples():
                # A set removes the duplicates when adding ids so we only end up with different ids
                ids |= {row.pA, row.pB}

                # Take ids until the number of different ids is greater or equal to the batch_size
                if len(ids) >= batch_size:
                    break

            batch_pairs = pairs.loc[pairs['pA'].isin(ids) & pairs['pB'].isin(ids)]
            pairs = pairs.drop(batch_pairs.index)

            assert len(batch_pairs) * 2 >= len(ids)

            yield self._create_pair_batch(batch_pairs, features, load_images)

    def _batch_by_pairs(self,
                        pairs: pd.DataFrame,
                        features: pd.DataFrame,
                        batch_size: int,
                        load_images=False) -> Generator[PairBatch, None, None]:
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
                           load_images: bool = False) -> PairBatch:
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
        pairs = pairs.iloc[np.repeat(np.arange(len(pairs)), total_rotations)]
        pairs["pA_id"] = pairs_a
        pairs["pB_id"] = pairs_b

        # Create labels column
        pairs["labels"] = pairs["comp"].values.astype(float)

        return PairBatch(pairs=pairs, patients=patients, ids_map=ids_map)

    def _load_patients(self,
                       ids_list: List[str],
                       features: pd.DataFrame,
                       total_rotations: int,
                       load_images: bool = False) -> Tuple[Dict[str, int], pd.DataFrame]:
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
            ids_map[idx] = i * total_rotations
            final_ids += [idx] * total_rotations
            file_path = os.path.join(DATA_PATH_PROCESSED, idx, idx + ".npz")

            # Check if the file exists, so the data has been preprocessed
            if load_images and not os.path.exists(file_path):
                self.logger.error(f"The file {file_path} could not be found. Have you pre-processed the data?")

            if load_images:
                loaded_npz = np.load(file_path)
                assert len(loaded_npz.files) == total_rotations
                for item in loaded_npz:
                    loaded_array = loaded_npz[item]
                    assert loaded_array.shape == (64, 64, 64)
                    images.append(loaded_array.reshape(64, 64, 64, 1))
                loaded_npz.close()
            else:
                images += [np.array([])] * total_rotations

            # Select radiomic features
            column = features[idx].values
            selected_features += [column] * total_rotations

        assert len(images) == len(selected_features) == len(final_ids)
        elements = pd.DataFrame({
            "ids": final_ids,
            "features": selected_features,
            "images": images
        })

        return ids_map, elements
