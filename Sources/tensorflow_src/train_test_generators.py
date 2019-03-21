import argparse
import os
import pathlib
import sys
# to import from other local packages in the model which are in diffrent folder
sys.path.append('../')
from typing import Dict, Tuple, Any, Iterator
import pandas as pd
import utils
from data import pair_data,mrmrpy
import yaml
import random
import shutil


def get_sets_generator(dataset: pair_data.SplitPairs,
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
        dataset.survival_categorizing(model, threshold, category = 5) # todo get rid of hard code
        enum_generator = dataset.folds(cv_folds, random=random_labels, random_seed=random_seed)
    logger.debug("Folds created")

    return enum_generator


def main(args: Dict[str, Any]) -> None:
    """
    Main function
    :param args: Command Line Arguments
    """
    logger.info("Script to split the data frame to test and train")

    with open("splitTestTrainConf.yml", 'r') as cfg_file:
        cfg = yaml.load(cfg_file)
    #Numbers of train and test spliting
    split_numbers = cfg['split_numbers']
    #The output path is refer to the folder that all the output are in
    output_path = cfg['output_path']
    logger.info("output path: {output_path}".format(output_path=output_path))

    #we randomly select a number to generate split based on the seed
    random.seed(1)
    random_states = random.sample(range(1, 100000), split_numbers)


    mrmr_sizes = cfg['mrmr']
    features= pd.read_csv(cfg['input_features'], index_col=0)
    clinical_info = pd.read_csv(cfg['input_clinical'], index_col=0)
    train_test_columns = ['cv_folds', 'spliting_models', 'random_seed','test_train_path','feature_path','mrmr_size' ]
    trains_tests_description = pd.DataFrame(columns=train_test_columns)

    # Always overwrite the previous weights
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    #results_path.mkdir(parents=True, exist_ok=True)
    dataset = pair_data.SplitPairs()
    dataset.clinical_data = clinical_info
    logger.info("read Feature DataFrame")
    for cv_folds in cfg['cv_folds']:
        cv_path=os.path.join("",f"cv_{cv_folds}")
        for n, random_seed in enumerate(random_states):
            random_path = os.path.join(cv_path,f"random_seed_{n}")
            for splitting_model in cfg['splitting_models']:
                logger.info('splitting_model')
                split_path = os.path.join(random_path, f"splitting_models_{splitting_model}")
                enum_generator = get_sets_generator(dataset,
                                                    cv_folds,
                                                    cfg['test_size'],
                                                    False,
                                                    splitting_model,
                                                    random_seed
                                                    )
                for i, (train_idx, test_idx) in enum_generator:
                    pathlib.Path(os.path.join(output_path, split_path)).mkdir(parents=True, exist_ok=True)
                    train_ids = clinical_info.iloc[train_idx]['id']
                    test_ids = clinical_info.iloc[test_idx]['id']
                    logger.info(f'test{i}:',test_ids)
                    logger.info(f'train{i}:',train_ids)
                    pd.DataFrame(train_ids).to_csv(os.path.join(output_path, split_path, f"train_fold{i}.csv"))
                    pd.DataFrame(test_ids).to_csv(os.path.join(output_path, split_path, f"test_fold{i}.csv"))
                    path=os.path.join(split_path,f"features_fold{i}")
                    for mrmr_size in mrmr_sizes:
                        logger.info(f'mrmr{mrmr_size}created\n')
                        features_path = os.path.join(output_path, path,f"radiomic{mrmr_size}.csv")
                        trains_tests_description.append(pd.DataFrame({'cv_folds': cv_folds,
                                                                                'spliting_models': splitting_model,
                                                                                'random_seed': random_seed,
                                                                                'test_train_path': path,
                                                                                'feature_path': features_path,
                                                                                'mrmr_size': mrmr_size
                                                                                },index=[0]))
                        pathlib.Path(os.path.join(output_path, path)).mkdir(parents=True, exist_ok=True)
                        if mrmr_size > 0:
                            df_features = mrmrpy.select_mrmr_features(features, clinical_info.iloc[train_idx] , mrmr_size).copy()
                            df_features.to_csv(features_path, index=False)
                        else:
                            df_features = features.copy()
                            df_features.to_csv(features_path, index=False)
    trains_tests_description.to_csv(os.path.join(output_path,'train_tests_description.csv'))

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
    arguments, unknown = parser.parse_known_args()
    arguments = vars(arguments)

    logger = utils.init_logger('train_test_generator')

    logger.debug("Script starts")
    logger.debug(arguments)

    if len(unknown) > 0:
        logger.warning(f"Warning: there are unknown arguments {unknown}")


    try:
        # For now the arguments are ignored
        main(arguments)
    except KeyboardInterrupt:
        logger.info("\r----------------------------------")
        logger.info("\rStopping due to keyboard interrupt")
        logger.info("\rTHANKS FOR THE RIDE 😀")





