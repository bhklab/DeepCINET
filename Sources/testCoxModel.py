import pandas as pd
import data
import settings
import utils
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from typing import Dict, Tuple, Any, Iterator
import argparse
import os
import pathlib
import argparse
import os
import pathlib
from typing import Dict, Tuple, Any, Iterator

#!/usr/bin/env python3
"""
The train module is a script that trains cox model for clinical and radiomic data.

usage: 


"""
logger = utils.init_logger("start")


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

################
#     MAIN     #
################
def del_low_var(df):
    low_variance = df.columns[df.var(0) < 10e-5]
    logger.info(f"The columns with low variance are removed{low_variance}")
    df.drop(low_variance, axis=1, inplace=True)
    logger.info(df.columns)




def coxModel(cv_folds: int = 1,
              test_size: float = .25,
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

    results_path = pathlib.Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    logger = utils.init_logger(f'train_{0}', str(results_path))

    logger.debug("Script starts")
    logger.info(f"Results path: {results_path}")
    results_path.mkdir(parents=True, exist_ok=True)

    logger = utils.init_logger(f'train_{0}', str(results_path))

    logger.debug("Script starts")

    logger.info(f"Results path: {results_path}")

    logger.info("Script to train a siamese neural network model")

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
    for key in ['train', 'test']:
        counts[key] = {
            'c_index': []
        }
    dataset = data.pair_data.SplitPairs()
    if (read_splits):
        cv_path = os.path.join(input_path, f"cv_{cv_folds}")
        random_path = os.path.join(cv_path, f"random_seed_{split_number}")
        split_path = os.path.join(random_path, f"splitting_models_{splitting_model}")
        enum_generator = get_sets_reader(cv_folds, split_path, mrmr_size)
        for i, (train_ids, test_ids, df_features) in enum_generator:
            features = df_features.copy()
            logger.info(f"New fold {i}, {len(train_ids)} train pairs, {len(test_ids)} test pairs")
            cph = CoxPHFitter()
            clinical = clinical_df.iloc[train_ids]
            radiomic_features_train = pd.merge(features.T, clinical[['id', 'event', 'time']], how='inner',
                                         left_index=True, right_on='id')
            radiomic_features_train.drop(['id'], axis='columns', inplace=True)
            radiomic_features_train = radiomic_features_train.dropna()
            del_low_var(radiomic_features_train)
            cph.fit(radiomic_features_train, duration_col='time', event_col='event', show_progress=False, step_size=0.2)

            clinical = clinical_df.iloc[test_ids]
            radiomic_features_test = pd.merge(features.T, clinical[['id', 'event', 'time']], how='inner',
                                              left_index=True, right_on='id')
            radiomic_features_test.drop(['id'], axis='columns', inplace=True)

            test_cindex = concordance_index(radiomic_features_test['time'],
                                    -cph.predict_partial_hazard(radiomic_features_test).values,
                                    radiomic_features_test['event'])
            train_cindex = concordance_index(radiomic_features_test['time'],
                                    -cph.predict_partial_hazard(radiomic_features_test).values,
                                    radiomic_features_test['event'])
            # print(cph.predict_survival_function(radiomic_features))


            counts['train']['c_index'].append((i, train_cindex))
            counts['test']['c_index'].append((i, test_cindex))

            logger.info(f"{'test'} set c-index: {test_cindex}")

            # Save each fold in a different directory

            results_save_path = os.path.join(results_path, f"fold_{i:0>2}")
            results_save_path = os.path.join(results_save_path, f"split_{split:0>2}")
            logger.info(f"Saving results at: {results_save_path}")
            #utils.save_results(results_save_path)
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
        for i, (train_ids, test_ids) in enum_generator:
            if mrmr_size > 0:
                df_features = data.select_mrmr_features(features, clinical_df.copy(), mrmr_size, train_ids).copy()
            else:
                df_features = features.copy()
            logger.info(f"New fold {i}, {len(train_ids)} train pairs, {len(test_ids)} test pairs")
            cph = CoxPHFitter()
            clinical = clinical_df.iloc[train_ids]
            radiomic_features = pd.merge(df_features.T, clinical[['id', 'event', 'time']], how='inner',
                                         left_index=True, right_on='id')
            radiomic_features.drop(['id'], axis='columns', inplace=True)
            radiomic_features = radiomic_features.dropna()
            del_low_var(radiomic_features)
            logger.info(radiomic_features.columns)

            # print(radiomic_features)
            cph.fit(radiomic_features, duration_col='time', event_col='event', show_progress=False, step_size=0.2)
            cph.print_summary()

            clinical = clinical_df.iloc[test_ids]
            radiomic_features_test = pd.merge(df_features.T, clinical[['id', 'event', 'time']], how='inner',
                                              left_index=True, right_on='id')
            radiomic_features_test.drop(['id'], axis='columns', inplace=True)
            test_cindex = concordance_index(radiomic_features_test['time'],
                                            -cph.predict_partial_hazard(radiomic_features_test).values,
                                            radiomic_features_test['event'])
            train_cindex = concordance_index(radiomic_features_test['time'],
                                             -cph.predict_partial_hazard(radiomic_features_test).values,
                                             radiomic_features_test['event'])
            # print(cph.predict_survival_function(radiomic_features))
            counts['train']['c_index'].append((i, train_cindex))
            counts['test']['c_index'].append((i, test_cindex))

            logger.info(f"{'test'} set c-index: {test_cindex}")

            results_save_path = os.path.join(results_path, f"fold_{i:0>2}")
            results_save_path = os.path.join(results_save_path, f"split_{split:0>2}")
            logger.info(f"Saving results at: {results_save_path}")
            #todo save
            logger.info(f"result{counts}")
            logger.info("\r ")



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
    results_path=args['results_path']
    read_splits = args['read_splits']

    coxModel( cv_folds = cv_folds,
              test_size = test_size,
              data_type = data_type,
              results_path = results_path,
              regularization = regularization,
              splitting_model = splitting_model,
              threshold = threshold,
              bin_number = 4,
              log_device = log_device,
              split = 1,
              split_seed = None,
              split_number = 1,
              mrmr_size = mrmr_size,
              read_splits = read_splits)



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
        action = "store_true",
        default = False
    )

    optional.add_argument(
        "--data-type",
        help="the type of data frame",
        type= str,
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


























