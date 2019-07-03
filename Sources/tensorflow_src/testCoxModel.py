import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import data
import tensorflow_src.config as settings
import utils
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import os
import argparse
import pathlib
from typing import Dict, Tuple, Any, Iterator
from data.train_test import get_sets_generator, get_sets_reader

# !/usr/bin/env python3
"""
The train module is a script that trains cox model for clinical and radiomic data.

usage: 


"""
logger = utils.init_logger("start")


################
#     MAIN     #
################
def del_low_var(df):
    low_variance = df.columns[df.var(0) < 10e-5]
    logger.info(f"The columns with low variance are removed{low_variance}")
    df.drop(low_variance, axis=1, inplace=True)
    logger.info(df.columns)


from sklearn.feature_selection import VarianceThreshold


def variance_threshold_selector(data, threshold=0.01):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


def coxModel(cv_folds: int = 1,
             test_size: float = .20,
             data_type: str = 'radiomic',
             results_path: str = settings.SESSION_SAVE_PATH,
             regularization: float = 0.01,
             splitting_model: int = 0,
             threshold: float = 3,
             bin_number: int = 4,
             log_device=False,
             split=1,  # todo check if required to add split_seed and initial_seed to the argument
             split_seed=None,
             split_number=None,  # it is used for the time we are using the generated test and train sets
             initial_seed=None,
             mrmr_size=0,
             read_splits=False):
    """
    deepCient
    :param args: Command Line Arguments
    """
    print(cv_folds)
    results_path = pathlib.Path(os.path.join(results_path, "Cox"))
    results_path.mkdir(parents=True, exist_ok=True)

    logger = utils.init_logger(f'train_{0}', str(results_path))

    logger.debug("Script starts")
    logger.info(f"Results path: {results_path}")
    results_path.mkdir(parents=True, exist_ok=True)

    logger = utils.init_logger(f'train_{0}', str(results_path))

    logger.debug("Script starts")

    logger.info(f"Results path: {results_path}")

    logger.info("Script to train a Cox model")

    logger.info(f"Data type is {data_type}")
    # read features and clinical data frame the path is defined in the settings.py
    if data_type == "radiomic":
        features = pd.read_csv(settings.DATA_PATH_RADIOMIC_PROCESSED, index_col=0)
    elif data_type == "clinical":
        features = pd.read_csv(settings.DATA_PATH_CLINIC_PROCESSED, index_col=0)
    elif data_type == "clinicalVolume":
        features = pd.read_csv(settings.DATA_PATH_VOLUME_CLINIC_PROCESSED, index_col=0)

    logger.info(f"number of features is {len(features.index)}")
    clinical_df = pd.read_csv(settings.DATA_PATH_CLINICAL_PROCESSED, index_col=0)
    logger.info("read Feature DataFrame")

    # read the input path for the time that train and test are splitted before head by train_test_generator.py
    input_path = settings.DATA_PATH_INPUT_TEST_TRAIN
    number_feature = mrmr_size if mrmr_size > 0 else settings.NUMBER_FEATURES

    counts = {}
    for key in ['train', 'test', 'mixed']:
        counts[key] = {
            'c_index': []
        }
    dataset = data.pair_data.SplitPairs()
    if (read_splits):
        cv_path = os.path.join(input_path, f"cv_{cv_folds}")
        random_path = os.path.join(cv_path, f"random_seed_{split_number}")
        split_path = os.path.join(random_path, f"splitting_models_{splitting_model}")
        enum_generator = get_sets_reader(cv_folds, split_path, mrmr_size, data_type)
        logger.info(enum_generator)
        for i, (train_ids, test_ids, df_features) in enum_generator:

            test_ids['id'] = test_ids['id'].astype(str)
            train_ids['id'] = train_ids['id'].astype(str)
            train_data = dataset.clinical_data.merge(train_ids, left_on="id", right_on="id", how="inner")
            test_data = dataset.clinical_data.merge(test_ids, left_on="id", right_on="id", how="inner")
            logger.info(f"New fold {i}, {len(train_ids)} train pairs, {len(test_ids)} test pairs")
            cph = CoxPHFitter()
            features_train = pd.merge(df_features.T,
                                      train_data[['id', 'event', 'time']],
                                      how='inner',
                                      left_index=True,
                                      right_on='id')
            features_train.drop(['id'], axis='columns', inplace=True)
            features_train = features_train.dropna()
            del_low_var(features_train)
            cph.fit(features_train, duration_col='time', event_col='event', show_progress=True, step_size=0.02)

            print(concordance_index(features_train['time'], -cph.predict_partial_hazard(features_train), features_train['event']))

            features_test = pd.merge(df_features.T, test_data[['id', 'event', 'time']], how='inner',
                                     left_index=True, right_on='id')
            features_test.drop(['id'], axis='columns', inplace=True)

            features = pd.merge(df_features.T, clinical_df[['id', 'event', 'time']], how='inner',
                                left_index=True, right_on='id')

            features['predict'] = cph.predict_partial_hazard(features)
            print(concordance_index(features['time'], -cph.predict_partial_hazard(features), features['event']))
            train_pairs, test_pairs, mixed_pairs = dataset.create_train_test(train_data, test_data, random=False)

            predictions = {}
            for pairs, name in [(train_pairs, 'train'), (test_pairs, 'test'), (mixed_pairs, 'mixed')]:
                logger.info(f"Computing {name} c-index")
                result = pd.merge(features[['id', 'time', 'predict', 'event']], pairs, left_on='id',
                                  right_on='pA', how='inner')

                result = pd.merge(features[['id', 'time', 'predict', 'event']], result, left_on='id',
                                  right_on='pB', suffixes=('_b', '_a'), how='inner')

                result['predict_comp'] =  result['predict_a'] > result['predict_b']
                predictions[name] = result
                correct = (result['predict_comp'] == result['comp']).sum()
                total = (result['predict_comp'] == result['comp']).count()
                c_index = correct / total
                counts[name]['c_index'].append((i, c_index))

            results_save_path = os.path.join(results_path, f"split_{split:0>2}")
            results_save_path = os.path.join(results_save_path, f"fold_{i:0>2}")
            logger.info(f"Saving results at: {results_save_path}")
                # todo save
            logger.info(f"result{counts}")
            pathlib.Path(results_save_path).mkdir(parents=True, exist_ok=True)
            # pd.DataFrame(counts).to_csv(os.path.join(results_save_path, 'result.csv'))
            logger.info("\r ")
            logger.info(f"Saving results at: {results_save_path}")
            utils.save_cox_results(predictions, results_save_path)
            logger.info(f"result{counts}")
            logger.info("\r ")


    else:
        enum_generator = get_sets_generator(dataset,
                                            cv_folds,
                                            test_size,
                                            False,
                                            splitting_model,
                                            threshold,
                                            split_seed)
        for i, (train_idx, test_idx) in enum_generator:
            if mrmr_size > 0:
                df_features = data.select_mrmr_features(features, clinical_df.iloc[train_idx].copy(), mrmr_size).copy()
            else:
                df_features = features.copy()
            train_data = dataset.clinical_data.iloc[train_idx]
            test_data = dataset.clinical_data.iloc[test_idx]
            train_pairs, test_pairs, mixed_pairs = dataset.create_train_test(train_data, test_data,
                                                                             random=False)
            logger.info(f"New fold {i}, {len(train_idx)} train pairs, {len(test_idx)} test pairs")
            cph = CoxPHFitter(penalizer=0.5, alpha=0.99)
            # radiomic_features = pd.merge(df_features.T, dataset.clinical_data[['id', 'event', 'time']], how='inner',left_index=True, right_on='id')

            clinical_train = clinical_df.iloc[train_idx]
            radiomic_features_train = pd.merge(df_features.T, train_data[['id', 'event', 'time']], how='inner',
                                               left_index=True, right_on='id')
            radiomic_features_train.drop(['id'], axis='columns', inplace=True)
            radiomic_features_train = radiomic_features_train.dropna()
            radiomic_features_train = variance_threshold_selector(radiomic_features_train)
            radiomic_features_train.to_csv('test.csv')
            # logger.info(radiomic_features.columns)

            # print(radiomic_features)
            cph.fit(radiomic_features_train, duration_col='time', event_col='event', show_progress=True, step_size=0.11)
            # cph.print_summary()

            # clinical = clinical_df.iloc[train_idx]
            radiomic_features = pd.merge(df_features.T, clinical_df[['id', 'event', 'time']], how='inner',
                                         left_index=True, right_on='id')

            radiomic_features['predict'] = cph.predict_partial_hazard(radiomic_features)
            predictions = {}
            for pairs, name in [(train_pairs, 'train'), (test_pairs, 'test'), (mixed_pairs, 'mixed')]:
                logger.info(f"Computing {name} c-index")
                result = pd.merge(radiomic_features[['id', 'time', 'predict', 'event']], pairs, left_on='id',
                                  right_on='pA', how='inner')
                result = pd.merge(radiomic_features[['id', 'time', 'predict', 'event']], result, left_on='id',
                                  right_on='pB', suffixes=('_b', '_a'), how='inner')
                result['predict_comp'] = result['predict_b'] < result['predict_a']
                predictions[name] = result
                correct = (result['predict_comp'] == result['comp']).sum()
                total = (result['predict_comp'] == result['comp']).count()
                c_index = correct / total
                counts[name]['c_index'].append((i, c_index))
            results_save_path = os.path.join(results_path, f"split_{split:0>2}")
            results_save_path = os.path.join(results_save_path, f"fold_{i:0>2}")
            logger.info(f"Saving results at: {results_save_path}")
            # todo save
            logger.info(f"result{counts}")
            pathlib.Path(results_save_path).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(counts).to_csv(os.path.join(results_save_path, 'result.csv'))
            logger.info("\r ")
    return counts, predictions


def main(args: Dict[str, Any]) -> None:
    """
    Main function
    :param args: Command Line Arguments
    """
    logger.info("Script to train a siamese neural network model")
    data_type = args['data_type']
    mrmr_size = args['mrmr_size']
    regularization = args['regularization']
    log_device = args['log_device']
    cv_folds = args['cv_folds']
    test_size = args['test_size']
    splitting_model = args['splitting_model']
    threshold = args['threshold']
    results_path = args['results_path']
    read_splits = args['read_splits']

    coxModel(cv_folds=cv_folds,
             test_size=test_size,
             data_type=data_type,
             results_path=results_path,
             regularization=regularization,
             splitting_model=splitting_model,
             threshold=threshold,
             bin_number=4,
             log_device=log_device,
             split=1,
             split_seed=None,
             split_number=1,
             mrmr_size=mrmr_size,
             read_splits=read_splits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fit the data with a Tensorflow model",
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
        "--regularization",
        help="Regularization factor to apply",
        default=0.01,
        type=float
    )
    optional.add_argument(
        "--splitting-model",
        help="The model of splitting data to train and test. "
             "0: split based on event censored data, "
             "1: split based on (same distribution of survival for train and test), "
             "2: split based on the threshold",
        default=1,
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
        "--generation_model",
        help="The model of generationg and using train test",
        default=0,
        type=int
    )

    optional.add_argument(
        "--read-splits",
        help="The way that generate input for the model read split or read from the pre generated inputs",
        action="store_true",
        default=False
    )

    optional.add_argument(
        "--data-type",
        help="the type of data frame",
        type=str,
        default="radiomic"
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
