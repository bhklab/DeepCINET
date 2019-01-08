import pandas as pd
import numpy as np
import data
import settings
import utils
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from typing import Dict, Tuple, Any, Iterator
import argparse
import os
import pathlib

logger = utils.init_logger("start")

from typing import Iterator, Tuple, Generator, List, Dict


def select_mrmr_features(dataframe_features: pd.DataFrame, mrmr_size: int, train_ids: List):
    """
      select the mrmr features

      :param dataframe_features: DataFrame of the features
      :param mrmr_size: The number of features which should be selected with mrmr
      :param train_ids: List of the train_ids that should be considered in mrmr
      :return: DataFrame that contain selected features
    """
    clinical_df = pd.read_csv(settings.DATA_PATH_CLINICAL_PROCESSED, index_col=0)
    clinicals = clinical_df.iloc[train_ids]  # clinical_df[train_ids.tolist()]
    # clinicals= pd.merge(clinical_df,pd.DataFrame(train_ids))
    mrmr_list = data.mrmr_selection(features=dataframe_features, clinical_info=clinicals, solution_count=1,
                                    feature_count=mrmr_size)
    logger.info(f"mrmr list contains{mrmr_list.values}")
    # print(dataframe_features)
    features = dataframe_features.loc[mrmr_list]

    return features


def get_sets_generator(dataset: data.pair_data.SplitPairs,
                       cv_folds: int,
                       test_size: int,
                       random_labels: bool,
                       model: int,
                       threshold: float,
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
        train_ids, test_ids = dataset.train_test_split(test_size, random=random_labels, models=model,
                                                       threshold=threshold, random_seed=random_seed)
        enum_generator = [(0, (train_ids, test_ids))]
        logger.info("1 fold")
    else:
        enum_generator = dataset.folds(cv_folds, random=random_labels)
    logger.debug("Folds created")

    return enum_generator


def del_low_var(df):
    low_variance = df.columns[df.var(0) < 10e-5]
    logger.info(f"The columns with low variance are removed{low_variance}")
    df.drop(low_variance, axis=1, inplace=True)
    logger.info(df.columns)


def main(args: Dict[str, Any]) -> None:
    number = args['number']

    values = {}
    for mrmr in (10, 20, 50, 100):
        c_index = []
        args['mrmr_size'] = mrmr
        for i in range(number):
            args['split-number'] = i
            c_index.append(cox(args))
        values["mrmr" + str(mrmr)] = c_index
    df_result = pd.DataFrame(values)
    df_result.to_csv(os.path.join(settings.DATA_PATH_PROCESSED, f"cox_results.csv"))


def get_sets_reader(cv_folds: int,
                    split_path,mrmr_size
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
        train_df = pd.read_csv(os.path.join(split_path, f"train_fold{i}.csv"))
        test_df = pd.read_csv(os.path.join(split_path, f"test_fold{i}.csv"))
        train_ids = train_df.iloc[:,1].values
        test_ids = test_df.iloc[:,1].values
        path = os.path.join(split_path, f"features_fold{i}", f"radiomic{mrmr_size}.csv")
        features = pd.read_csv(path, index_col=0)
        yield (i,(train_ids,test_ids,features))


def cox(args: Dict[str, Any]) -> float:
    """
    Main function
    :param args: Command Line Arguments
    """
    logger.info("Script to train a cox model")
    dataset = data.pair_data.SplitPairs()
    mrmr_size = args['mrmr_size']
    split_number = args['split-number']
    cv_folds = args['cv_folds']
    read_splits = args['read_splits']
    splitting_model = 1
    features_df = pd.read_csv(settings.DATA_PATH_RADIOMIC_PROCESSED, index_col=0)

    clinical_info = dataset.clinical_data
    # clinical_info.set_index('id', inplace=True)
    input_path = settings.DATA_PATH_INPUT_TEST_TRAIN
    logger.info("read Feature DataFrame")
    dataset = data.pair_data.SplitPairs()
    if (read_splits):
        cv_path = os.path.join(input_path, f"cv_{cv_folds}")
        random_path = os.path.join(cv_path, f"random_seed_{split_number}")
        split_path = os.path.join(random_path, f"splitting_models_{splitting_model}")
        uniq_random = os.path.join(split_path, f"T")
        enum_generator = get_sets_reader(cv_folds, split_path, mrmr_size)
        c_index = []
        for i, (train_ids, test_ids) in enum_generator:
            if mrmr_size > 0:
                features = select_mrmr_features(features_df.copy(), mrmr_size, train_ids)
            else:
                features = features_df.copy()
            logger.info(f"New fold {i}, {len(train_ids)} train pairs, {len(test_ids)} test pairs")
            summaries_dir = os.path.join(args['results_path'], f'fold_{i}')
            cph = CoxPHFitter()
            clinical = clinical_info.iloc[train_ids]
            radiomic_features = pd.merge(features.T, clinical[['id', 'event', 'time']], how='inner',
                                         left_index=True, right_on='id')
            radiomic_features.drop(['id'], axis='columns', inplace=True)
            radiomic_features = radiomic_features.dropna()
            del_low_var(radiomic_features)
            logger.info(radiomic_features.columns)

            # print(radiomic_features)
            cph.fit(radiomic_features, duration_col='time', event_col='event', show_progress=True, step_size=0.2)
            cph.print_summary()

            clinical = clinical_info.iloc[test_ids]
            radiomic_features_test = pd.merge(features.T, clinical[['id', 'event', 'time']], how='inner',
                                              left_index=True, right_on='id')
            radiomic_features_test.drop(['id'], axis='columns', inplace=True)
            c_index.append(concordance_index(radiomic_features_test['time'],
                                             -cph.predict_partial_hazard(radiomic_features_test).values,
                                             radiomic_features_test['event']))

        print(c_index)

    else:
        enum_generator = get_sets_generator(dataset,
                                        args['cv_folds'],
                                        args['test_size'],
                                        False,
                                        args['splitting_model'],
                                        args['threshold'],
                                        None
                                        )


    return np.mean(c_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fit the data with a Cox model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    # parser._action_groups.pop()

    required = parser.add_argument_group('required named arguments')
    optional = parser.add_argument_group('optional named arguments')

    # Optional arguments
    optional.add_argument(
        "-h", "--help",
        help="Show this help",
        action="help"
    )

    optional.add_argument(
        "--cv-folds",
        help="Number of cross validation folds. If 0 < folds < 2 CV won't be used and the test set size "
             "will be defined by --test-size. If folds < 0 Leave One Out Cross Validation will be used instead",
        default=1,
        type=int
    )
    optional.add_argument(
        "--test-size",
        help="Size of the test set as a float between 0 and 1",
        default=.25,
        type=float
    )

    optional.add_argument(
        "--results-path",
        help="Path where the results and the model should be saved",
        default=settings.SESSION_SAVE_PATH,
        type=str
    )
    optional.add_argument(
        "--splitting-model",
        help="The model of splitting data to train and test. "
             "0: split based on event censored data, "
             "1: split based on (same distribution of survival for train and test), "
             "2: split based on the threshold",
        default=0,
        type=int
    )
    optional.add_argument(
        "--threshold",
        help="The threshold for splitting ",
        default=3,
        type=float
    )
    optional.add_argument(
        "--bin-number",
        help="The number of bins for splitting based on the distribution",
        default=4,
        type=int
    )
    optional.add_argument(
        "--log-device",
        help="Log device placement when creating all the tensorflow tensors",
        action="store_true",
        default=False
    )

    optional.add_argument(
        "--mrmr-size",
        help="The number of features that should be selected with Mrmr- 0 means Mrmr shouldn't apply",
        default=0,
        type=int
    )

    optional.add_argument(
        "--number",
        help="The number of features that should be selected with Mrmr- 0 means Mrmr shouldn't apply",
        default=2,
        type=int
    )

    optional.add_argument(
        "--read-splits",
        help="Log device placement when creating all the tensorflow tensors",
        action="store_true",
        default=False
    )

    # See if we are running in a SLURM task array
    array_id = os.getenv('SLURM_ARRAY_TASK_ID', 0)

    arguments, unknown = parser.parse_known_args()
    arguments = vars(arguments)

    arguments['results_path'] = os.path.abspath(arguments['results_path'])
    results_path = pathlib.Path(arguments['results_path'])
    results_path.mkdir(parents=True, exist_ok=True)

    logger = utils.init_logger(f'train_{array_id}', str(results_path))

    logger.debug("Script starts")
    logger.debug(arguments)
    logger.info(f"Results path: {results_path}")

    if len(unknown) > 0:
        logger.warning(f"Warning: there are unknown arguments {unknown}")

    try:
        # For now the arguments are ignored
        main(arguments)
    except KeyboardInterrupt:
        logger.info("\r----------------------------------")
        logger.info("\rStopping due to keyboard interrupt")
        logger.info("\rTHANKS FOR THE RIDE 😀")
