import pandas as pd
import data
import settings
import utils
from lifelines import CoxPHFitter
from typing import Dict, Tuple, Any, Iterator
import argparse
import os
import pathlib

logger = utils.init_logger("start")

from typing import Iterator, Tuple, Generator, List, Dict


def select_mrmr_features(dataframe_features: pd.DataFrame , mrmr_size : int, train_ids: List):
    """
      select the mrmr features

      :param dataframe_features: DataFrame of the features
      :param mrmr_size: The number of features which should be selected with mrmr
      :param train_ids: List of the train_ids that should be considered in mrmr
      :return: DataFrame that contain selected features
    """
    clinical_df= pd.read_csv(settings.DATA_PATH_CLINICAL_PROCESSED, index_col=0)
    clinicals= clinical_df.iloc[train_ids] #clinical_df[train_ids.tolist()]
    #clinicals= pd.merge(clinical_df,pd.DataFrame(train_ids))
    mrmr_list= data.mrmr_selection(features=dataframe_features, clinical_info=clinicals, solution_count=1, feature_count=mrmr_size)
    logger.info(mrmr_list)
    #print(dataframe_features)
    features = dataframe_features.loc[mrmr_list]
    return features


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
        enum_generator = dataset.folds(cv_folds, random=random_labels)
    logger.debug("Folds created")

    return enum_generator




def main(args: Dict[str, Any]) -> None:
    """
    Main function
    :param args: Command Line Arguments
    """
    logger.info("Script to train a cox model")
    dataset = data.pair_data.SplitPairs()
    mrmr_size = args['mrmr_size']
    features_df  = pd.read_csv(settings.DATA_PATH_RADIOMIC_PROCESSED,index_col=0 )

    clinical_info = dataset.clinical_data
    #clinical_info.set_index('id', inplace=True)
    radiomic_features = features_df.T

    radiomic_features = pd.merge(radiomic_features, clinical_info[['id','event','time']] , how='inner', left_index=True, right_on='id')
    #radiomic_features['event'] = clinical_info['event']
    #radiomic_features['time'] = clinical_info['time']
    logger.info("read Feature DataFrame")

    enum_generator = get_sets_generator(dataset,
                                        args['cv_folds'],
                                        args['test_size'],
                                        args['random_labels'],
                                        args['splitting_model'],
                                        args['threshold'],
                                        None
                                        )
    for i, (train_ids, test_ids) in enum_generator:
        if mrmr_size > 0:
            logger.info(train_ids)
            features = select_mrmr_features(features_df.copy(), mrmr_size, train_ids)
        else:
            features = features_df.copy()
        logger.info(f"New fold {i}, {len(train_ids)} train pairs, {len(test_ids)} test pairs")
        summaries_dir = os.path.join(args['results_path'], f'fold_{i}')
        cph = CoxPHFitter()
        clinical = clinical_info.iloc[train_ids]
        radiomic_features = pd.merge(features.T, clinical[ ['id', 'event', 'time']], how='inner',
                                     left_index=True, right_on='id')
        radiomic_features.drop(['id'] ,axis = 'columns', inplace=True)
        print(radiomic_features)
        cph.fit(radiomic_features, duration_col='time', event_col='event', show_progress=True)

        clinical = clinical_info.iloc[test_ids]
        radiomic_features = pd.merge(features.T, clinical[['id', 'event', 'time']], how='inner',
                                     left_index=True, right_on='id')
        radiomic_features.drop(['id'], axis='columns', inplace=True)
        print(radiomic_features)
        print(cph.predict_survival_function(radiomic_features))


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
        "--gpu-level",
        help="Amount of GPU resources used when fitting the model. 0: no GPU usage, "
             "1: only second conv layers, 2: all conv layers, "
             "3: all layers and parameters are on the GPU",
        default=0,
        type=int
    )
    optional.add_argument(
        "--gpu-allow-growth",
        help="Allow Tensorflow to use dynamic allocations with GPU memory",
        default=False,
        action="store_true",
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
        "--save-model",
        help="Save the model to the location specified at the results path",
        action="store_true",
        default=False
    )
    optional.add_argument(
        "--random-labels",
        help="Whether to use or not random labels, use ONLY to validate a model",
        action="store_true",
        default=False
    )

    optional.add_argument(
        "--mrmr-size",
        help="The number of features that should be selected with Mrmr- 0 means Mrmr shouldn't apply",
        default=0,
        type=int
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