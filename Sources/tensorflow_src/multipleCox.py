#!/usr/bin/env python3
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import testCoxModel

#import STprediction
import seaborn as sns
import random
import yaml
import pathlib

import multiprocessing

sns.set()

with open("modelConf.yml", 'r') as cfg_file:
    cfg = yaml.load(cfg_file)

results = pd.DataFrame()
mixed_c_index, train_c_index, test_c_index = [], [], []
running_times = 50
random_states = list(range(running_times * 2))
random.seed(1)
random.shuffle(random_states)
for i in range(running_times):
    counts,predicts = testCoxModel.coxModel(data_type='clinicalVolume',
                                   mrmr_size=0,
                                   read_splits=True,
                                   splitting_model=1,
                                   split_seed=random_states[i],
                                   split=i,
                                   split_number=i,
                                   cv_folds=5)
    print(f"test{[v[1] for v in counts['test']['c_index']]}")
    print(f"test{len([v[1] for v in counts['test']['c_index']])}")
    print(counts)
    for name in ('train', 'test', 'mixed'):
        counts[name]['c_index'] = sum([v[1] for v in counts[name]['c_index']]) / float(len(counts[name]['c_index']))

    # Store the mixed c-index values for boxplot
    train_c_index.append(counts['train']['c_index'])
    test_c_index.append(counts['test']['c_index'])
    mixed_c_index.append(counts['mixed']['c_index'])

    result = pd.DataFrame.from_dict(counts)
    result['random state'] = random_states[i]
    result['number'] = i
    results = results.append(result)
    pathlib.Path(cfg['cox_result_path']).mkdir(parents=True, exist_ok=True)
    results.to_csv(os.path.join(cfg['cox_result_path'],'result.csv'))
results.to_csv(cfg['mixed_result_path'], index = False)



