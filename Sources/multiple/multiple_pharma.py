#!/usr/bin/env python3
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import config as settings

# import STprediction
from tensorflow_src import train_test_models
import random
import yaml
import utils
import tensorflow_src.config as config

import multiprocessing
results = pd.DataFrame()
mixed_c_index, train_c_index, test_c_index = [], [], []
running_times = 1
random_states = list(range(running_times * 2))
random.seed(1)
random.shuffle(random_states)
logger = utils.init_logger("multiple run")
results_path = config.SESSION_SAVE_PATH
for i in range(running_times):
    parameters = dict(model='ScalarOnlySiamese',
                      target_path=config.DATA_PATH_TARGET,
                      feature_path=config.DATA_PATH_FEATURE,
                      input_path=config.DATA_PATH_INPUT_TEST_TRAIN,
                      results_path=results_path,
                      num_epochs=12,
                      batch_size=125,
                      splitting_model=1,
                      learning_rate=0.0003,
                      dropout=.2,
                      threshold=0.4,
                      split=i, save_model=True,
                      regularization=5.,
                      split_seed=random_states[i],
                      initial_seed=None,
                      mrmr_size=0,
                      read_splits=True,
                      full_summary=True,
                      cv_folds=1,
                      split_number=i,
                      train_distance=0.15,
                      test_distance=0.2,
                      survival=False
                     )
    counts, predictions = train_test_models.deepCinet(**parameters)
    logger.info(f"Parameters: {parameters}")
    logger.info(f"test{[v[1] for v in counts['test']['c_index']]}")
    logger.info(f"test{len([v[1] for v in counts['test']['c_index']])}")
    logger.info(counts)
    counts['train']['c_index'] = sum([v[1] for v in counts['train']['c_index']]) / float(
        len(counts['train']['c_index']))
    counts['test']['c_index'] = sum([v[1] for v in counts['test']['c_index']]) / float(len(counts['test']['c_index']))
    counts['mixed']['c_index'] = sum([v[1] for v in counts['mixed']['c_index']]) / float(
        len(counts['mixed']['c_index']))

    # Store the mixed c-index values for boxplot
    mixed_c_index.append(counts['mixed']['c_index'])
    train_c_index.append(counts['train']['c_index'])
    test_c_index.append(counts['test']['c_index'])

    result = pd.DataFrame.from_dict(counts)
    result = result.drop(['correct', 'total'])
    result['random state'] = random_states[i]
    result['number'] = i
    results = results.append(result)
    results.to_csv(os.path.join(results_path, "result.csv"))
    pd.DataFrame.from_dict(parameters, orient='index').to_csv(os.path.join(results_path, "config.csv"))
results.to_csv(os.path.join(results_path, "result.csv"), index=False)
