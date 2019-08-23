#!/usr/bin/env python3
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import tensorflow_src.config as config
import utils
# import STprediction
from tensorflow_src import learning_models
import random
import yaml



import multiprocessing
results_all = pd.DataFrame()
mixed_c_index, train_c_index, test_c_index = [], [], []

for drug, iter, reg, learning  in [#('Doxorubicin',40,10.,0.0003),
                                   #('Gemcitabine',30,4.,0.00021),
                                   ('lapatinib',48,6.,0.0001),
                                   #('Bortezomib',24,3.,0.0003),
                                   ('Erlotinib',60,20.,0.0002),
                                  # ('Vorinostat',52,10.,0.0003)
                                    ]:
    for random_size in [50,100,200,250,300,400]:
        results = pd.DataFrame()
        epoch = int(iter * 40 / random_size)
        running_times = 20
        random_states = list(range(running_times * 2))
        #random.seed(1)
        #random.shuffle(random_states)
        results_path = f"{config.SESSION_SAVE_PATH}/ML/{drug}/random_{random_size}"
        target_path = f"{config.DATA_PATH_TARGET}train_test_response_{drug}.csv"
        feature_path = f"{config.DATA_PATH_FEATURE}train_test_expression_{drug}.csv"
        input_train_test = f"{config.DATA_PATH_INPUT_TEST_TRAIN}_{drug}/rand{random_size}"
        logger = utils.init_logger("multiple run Ml random")
        model_type = "ElasticNet"
        for i in range(running_times):
            feature_path = f"{input_train_test}/cv_1/random_seed_{i}/splitting_models_1/gene_expression.csv"
            parameters = { 'target_path': target_path,
                           'feature_path': feature_path,
                           'input_path': input_train_test,
                           'result_path': results_path,
                           'splitting_model': 1,
                           'threshold': 4,
                           'split': i,
                           'regularization': reg,
                           'split_seed': random_states[i],
                           'mrmr_size': 0,
                           'read_splits': True,
                           'cv_folds': 1,
                           'split_number': i,
                           'test_distance': 0.2,
                           'train_distance': 0.2 }
            counts, predictions = learning_models(**parameters)
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
            result['random state'] = random_states[i]
            result['number'] = i
            result['drug'] = drug
            result['random size'] = random_size
            results = results.append(result)
            results_all = results_all.append(result)
            results.to_csv(os.path.join(results_path, "result.csv"))
            pd.DataFrame.from_dict(parameters, orient='index').to_csv(os.path.join(results_path, "config.csv"))
        results.to_csv(os.path.join(results_path, "result.csv"), index=False)
    results_all.to_csv(os.path.join(f"{config.SESSION_SAVE_PATH}/ML/{drug}/", "Allresult.csv"), index=False)