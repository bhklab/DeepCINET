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

for drug in ['AZD7762', 'Erlotinib', 'AZD8055', 'Gefitinib', 'Bortezomib',
'Gemcitabine', 'Crizotinib', 'MK-2206', 'Dasatinib', 'Nilotinib', 'Docetaxel', 'PLX4720']:
    results_path = f"/Users/farnoosh/Documents/DATA/UHN-Project/Genomic/Processed/GeneratedData/CTRPv2_GDSC/Result_2/DeepCINET/{drug}"

    parameters = dict(model='ScalarOnlySiamese',
                      target_path=config.DATA_PATH_TARGET,
                      feature_path=config.DATA_PATH_FEATURE,
                      input_path=config.DATA_PATH_INPUT_TEST_TRAIN,
                      results_path=results_path,
                      num_epochs=8,
                      batch_size=125,
                      splitting_model=1,
                      learning_rate=0.0001,
                      dropout=.2,
                      threshold=0.4,
                      split=1, save_model=True,
                      regularization=3.,
                      split_seed=random_states[1],
                      initial_seed=None,
                      mrmr_size=0,
                      read_splits=True,
                      full_summary=True,
                      cv_folds=1,
                      split_number=1,
                      train_distance=0.15,
                      test_distance=0.2,
                      survival=False
                     )
    parameters = pd.read_csv(f"/Users/farnoosh/Documents/DATA/UHN-Project/Genomic/Processed/GeneratedData/CTRPv2_gCSI/Result/DeepCINET/{drug}/config.csv",
                             index_col=0)
    parameters = parameters.to_dict()['0']
    parameters["num_epochs"] = int(parameters["num_epochs"])
    parameters["batch_size"] = 125
    parameters["splitting_model"] = int(parameters["splitting_model"])
    parameters["learning_rate"] = float(parameters["learning_rate"])
    parameters["dropout"] = float(parameters["dropout"])
    parameters["threshold"] = float(parameters["threshold"])
    parameters["split"] = int(parameters["split"])
    parameters["regularization"]= float(parameters["regularization"])
    parameters["mrmr_size"] =  int(parameters["mrmr_size"])
    parameters["read_splits"] =  bool(parameters["read_splits"])
    parameters["full_summary"] =  bool(parameters["full_summary"]),
    parameters["cv_folds"]=  int(parameters["cv_folds"])
    parameters["split_number"] =  int(parameters["split_number"])
    parameters["train_distance"]= float(parameters["train_distance"])
    parameters["test_distance"]= float(parameters["test_distance"])
    parameters["survival"] = False
    parameters['target_path'] = parameters['target_path'].replace('gCSI','GDSC')
    parameters['feature_path'] = parameters['feature_path'].replace('gCSI', 'GDSC')
    parameters['input_path'] = parameters['input_path'].replace('gCSI', 'GDSC')
    parameters['results_path'] = f"/Users/farnoosh/Documents/DATA/UHN-Project/Genomic/Processed/GeneratedData/CTRPv2_GDSC/Result_2/DeepCINET/{drug}"
    parameters['initial_seed'] = None
    print(parameters['target_path'])
    print("=======================================")
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
    result['random state'] = random_states[1]
    result['number'] = 1
    results = results.append(result)
    results.to_csv(os.path.join(results_path, "result.csv"))
    pd.DataFrame.from_dict(parameters, orient='index').to_csv(os.path.join(results_path, "config.csv"))
results.to_csv(os.path.join(results_path, "result.csv"), index=False)
