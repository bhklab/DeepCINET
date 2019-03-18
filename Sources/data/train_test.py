
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import tensorflow_src.settings as settings
import data
import utils
import pandas as pd
from typing import Iterator, Tuple, Generator, List, Dict
logger = utils.init_logger("train_test")



def get_sets_generator(dataset: data.pair_data.SplitPairs,
                       cv_folds: int,
                       test_size: int,
                       random_labels: bool,
                       model:int,
                       threshold:float,
                       random_seed: int = None
                       ) -> Iterator[Tuple[int, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]:
    """
    Get the generator that creates the train/test sets and the folds if Cross Validation is used

    :param cv_folds: Number of Cross Validation folds
    :param test_size: Number between ``0.0`` and ``1.0`` with a proportion of test size compared against the
                      whole set
    :param random_labels: Whether to randomize or not the labels. To be used ONLY when validating the model
    :param model: This parameter use for selecting the model for spliting data 0 censored, 1 target distribution, 2 based on threshold
    :param threshold: this is a threshold which is used in the model spliting
    :param random_seed that is used for model spliting
    :return: The sets generator then it can be used in a ``for`` loop to get the sets

                >>> folds = get_sets_generator(...)
                >>> for fold, (train_pairs, test_pairs, mixed_pairs) in folds:
                        # Insert your code
                        pass
    """


    # Decide whether to use CV or only a single test/train sets
    if 0 < cv_folds < 2:
        train_ids, test_ids = dataset.train_test_split(test_size, random=random_labels, models=model, threshold=threshold , random_seed=random_seed)
        enum_generator = [(0, (train_ids, test_ids))]
        logger.info("1 fold")
    else:
        dataset.survival_categorizing(model, threshold, category=5)  # todo get rid of hard code
        enum_generator = dataset.folds(cv_folds, random=random_labels)
    logger.debug("Folds created")

    return enum_generator

def get_sets_reader(cv_folds: int,
                    split_path,mrmr_size,data_type
                    ) -> Iterator[Tuple[int, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]:
    """
    Get the read the train/test sets which has been generated before ahead by train_test_generator.py ans is in the folders

    :param cv_folds: Number of Cross Validation folds (currently we only consider the number of folds which has been generated)
    :param test_size: Number between ``0.0`` and ``1.0`` with a proportion of test size compared against the
                      whole set (currently it is 0.25 but it can be change in yml file)
    :param random_labels: Whether to randomize or not the labels. To be used ONLY when validating the model
    :param model: This parameter use for selecting the model for spliting data 0 censored, 1 target distribution, 2 based on threshold
    :param threshold: this is a threshold which is used in the model spliting
    :param random_seed that is used for model spliting
    :return: The sets generator then it can be used in a ``for`` loop to get the sets

                >>> folds = get_sets_generator(...)
                >>> for fold, (train_pairs, test_pairs, mixed_pairs) in folds:
                        # Insert your code
                        pass
    """
    for i in range(0,cv_folds):
        train_df = pd.read_csv(os.path.join(split_path, "train_fold{i}.csv".format(i=i)), index_col=0)
        test_df = pd.read_csv(os.path.join(split_path, "test_fold{i}.csv".format(i=i)),index_col=0)
        train_ids = train_df
        test_ids = test_df
        if mrmr_size==0:
            if data_type == "radiomic":
                path = settings.DATA_PATH_RADIOMIC_PROCESSED
            elif data_type == "clinical":
                path = settings.DATA_PATH_CLINIC_PROCESSED
            elif data_type == "clinicalVolume":
                path = settings.DATA_PATH_VOLUME_CLINIC_PROCESSED
        else:
            path = os.path.join(split_path, "features_fold{i}".format(i=i), "radiomic{mrmr_size}.csv".format(mrmr_size=mrmr_size))
        features = pd.read_csv(path, index_col=0)
        yield (i,(train_ids,test_ids,features))
