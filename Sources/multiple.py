#!/usr/bin/env python3
import pandas as pd
#import STprediction
import train_test_models
import seaborn as sns
import random
import yaml

from joblib import Parallel, delayed
import multiprocessing

sns.set()

with open("modelConf.yml", 'r') as cfg_file:
    cfg = yaml.load(cfg_file)

results = pd.DataFrame()
mixed_c_index, train_c_index, test_c_index = [], [], []
running_times = 2
random_states = list(range(running_times * 2))
random.seed(1)
random.shuffle(random_states)


for i in range(running_times):
    counts,predictions = train_test_models.deepCinet('ScalarOnlySiamese', num_epochs =1, batch_size = 350, splitting_model = 1 , learning_rate = 0.0001 , split = i, save_model = True,regularization = 0.8 , split_seed = random_states[i], initial_seed = random_states[i]+5, mrmr_size = 400,read_splits=True,cv_folds=5, split_number = i )

    counts['train']['c_index'] = sum([v[1] for v in  counts['train']['c_index']]) / float(len( counts['train']['c_index']))
    counts['test']['c_index'] = sum([v[1] for v in  counts['test']['c_index']]) / float(len( counts['test']['c_index']))
    counts['mixed']['c_index'] = sum([v[1] for v in  counts['mixed']['c_index']]) / float(len( counts['mixed']['c_index']))

    # Store the mixed c-index values for boxplot
    mixed_c_index.append(counts['mixed']['c_index'])
    train_c_index.append(counts['train']['c_index'])
    test_c_index.append(counts['test']['c_index'])

    result = pd.DataFrame.from_dict(counts)
    #result = result.add_suffix("_" + str(i))
    result = result.drop(['correct', 'total'])
    result['random state'] = random_states[i]
    result['number'] = i
    print(result)
    results = results.append(result)
    if(i%3 == 0):
        results.to_csv(cfg['mixed_result_path'])
print(result.describe)

results.to_csv(cfg['mixed_result_path'], index = False)



