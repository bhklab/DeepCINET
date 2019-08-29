import argparse
import os
import pathlib
import sys

# to import from other local packages in the model which are in diffrent folder
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.dirname(__file__))
from typing import Dict, Tuple, Any, Iterator
import pandas as pd
import utils
import data
import settings
from data import pair_data, mrmrpy, get_sets_generator
import random
import shutil
import config
import os

logger = utils.init_logger('train_test_generator')


def generator(mrmr_sizes,
              target_path: str,
              features: str,
              survival: bool,
              test_size=.20,
              split_numbers=50,
              spliting_model=[1],
              folds=[1, 5],
              threshold=0.5) -> None:
    logger.debug("Script to split the data frame to test and train")
    cwd = os.getcwd()  # Get the current working directory (cwd)
    files = os.listdir(cwd)  # Get all the files in that directory
    print("Files in '%s': %s" % (cwd, files))

    # Numbers of train and test spliting

    train_test_columns = ['cv_folds', 'spliting_models', 'random_seed', 'test_train_path', 'feature_path', 'mrmr_size']
    trains_tests_description = pd.DataFrame(columns=train_test_columns)

    # Always overwrite the previous weights
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # results_path.mkdir(parents=True, exist_ok=True)
    data_set = pair_data.SplitPairs(target_path=target_path, survival=survival)
    logger.info("read Feature DataFrame")
    for cv_folds in folds:
        cv_path = os.path.join("", f"cv_{cv_folds}")
        for n, random_seed in enumerate(random_states):
            random_path = os.path.join(cv_path, f"random_seed_{n}")
            for splitting_model in spliting_model:
                logger.info('splitting_model')
                split_path = os.path.join(random_path, f"splitting_models_{splitting_model}")
                data_set = data.pair_data.SplitPairs(target_path=target_path, survival=survival)
                enum_generator = get_sets_generator(dataset=data_set,
                                                    cv_folds=cv_folds,
                                                    test_size=test_size,
                                                    random_labels=False,
                                                    model=splitting_model,
                                                    threshold=threshold,
                                                    random_seed=random_seed
                                                    )

                for i, (train_idx, test_idx) in enum_generator:
                    pathlib.Path(os.path.join(output_path, split_path)).mkdir(parents=True, exist_ok=True)
                    train_ids = data_set.target_data.iloc[train_idx]['id']
                    test_ids = data_set.target_data.iloc[test_idx]['id']
                    pd.DataFrame(train_ids).to_csv(os.path.join(output_path, split_path, f"train_fold{i}.csv"))
                    pd.DataFrame(test_ids).to_csv(os.path.join(output_path, split_path, f"test_fold{i}.csv"))
                    path = os.path.join(split_path, f"features_fold{i}")
                    for mrmr_size in mrmr_sizes:
                        logger.info(f'mrmr{mrmr_size}created\n')
                        features_path = os.path.join(output_path, path, f"feature{mrmr_size}.csv")
                        trains_tests_description.append(pd.DataFrame({'cv_folds': cv_folds,
                                                                      'spliting_models': splitting_model,
                                                                      'random_seed': random_seed,
                                                                      'test_train_path': path,
                                                                      'feature_path': features_path,
                                                                      'mrmr_size': mrmr_size
                                                                      }, index=[0]))
                        pathlib.Path(os.path.join(output_path, path)).mkdir(parents=True, exist_ok=True)
                        if mrmr_size > 0:
                            df_features = mrmrpy.select_mrmr_features(features,
                                                                      data_set.target_data.iloc[train_idx],
                                                                      mrmr_size,survival=survival).copy()
                            df_features.to_csv(features_path, index=True)
                        else:
                            df_features = features.copy()
                            df_features.to_csv(features_path, index=False)
    trains_tests_description.to_csv(os.path.join(output_path, 'train_tests_description.csv'))


if __name__ == '__main__':

    logger.debug("Script starts")

    cfg = config.GENERATOR
    split_numbers = cfg['SPLIT_NUMBER']
    # The output path is refer to the folder that all the output are in
    output_path = cfg['OUTPUT_PATH']
    logger.info("output path: {output_path}".format(output_path=output_path))

    # we randomly select a number to generate split based on the seed
    random.seed(1)
    random_states = random.sample(range(1, 100000), split_numbers)

    mrmr_sizes = cfg['MRMR']
    target_path = cfg["TARGET_PATH"]
    print(target_path)
    features = pd.read_csv(cfg['INPUT_FEATURES'], index_col=0)
    survival = True #cfg['SURVIVAL']
    test_size = cfg['TEST_SIZE']
    spliting_model = cfg['SPLITING_MODEL']
    cv_fold = cfg['CV_FOLDS']

    try:
        # For now the arguments are ignored
        generator(mrmr_sizes=mrmr_sizes,
                  target_path=target_path,
                  features=features,
                  survival=survival,
                  test_size=test_size,
                  split_numbers=split_numbers,
                  spliting_model=spliting_model,
                  folds=cv_fold)
    except KeyboardInterrupt:
        logger.info("\r----------------------------------")
        logger.info("\rStopping due to keyboard interrupt")
        logger.info("\rTHANKS FOR THE RIDE 😀")



