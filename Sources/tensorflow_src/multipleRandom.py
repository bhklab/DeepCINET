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

import multiprocessing
results = pd.DataFrame()
mixed_c_index, train_c_index, test_c_index = [], [], []

for drug,iter,reg in [('Doxorubicin',30,10.),('Gemcitabine',30,4.)]:
    for random_size in [50,100,200,250,300,400]:
        epoch = int(iter * 200 / random_size)
        running_times = 10
        random_states = list(range(running_times * 2))
        #random.seed(1)
        #random.shuffle(random_states)
        results_path = f"{settings.SESSION_SAVE_PATH}/{drug}/random_{random_size}"
        target_path = f"{settings.DATA_PATH_TARGET}train_test_response_{drug}.csv"
        feature_path = f"{settings.DATA_PATH_FEATURE}train_test_expression_{drug}.csv"
        input_train_test = f"{settings.DATA_PATH_INPUT_TEST_TRAIN}_{drug}/rand{random_size}"
        logger = utils.init_logger("multiple run random")

        for i in range(running_times):
            parameters = {'model': 'ScalarOnlySiamese', 'target_path': target_path,
                          'feature_path': feature_path, 'input_path': input_train_test,
                          'results_path': results_path, 'num_epochs': epoch, 'batch_size': 200,
                          'splitting_model': 1, 'learning_rate': 0.0001, 'dropout': .3, 'threshold': 4, 'split': i,
                          'save_model': True, 'regularization': reg, 'split_seed': random_states[i], 'initial_seed': None,
                          'mrmr_size': 0, 'read_splits': True, 'full_summary': False, 'cv_folds': 1, 'split_number': i,
                          'test_distance': 0.2, 'train_distance': 0.2 , 'survival': False}
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
