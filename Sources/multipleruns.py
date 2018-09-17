#!/usr/bin/env python3
import numpy as np
import pandas as pd
#import STprediction
import train
import os
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import utils

from joblib import Parallel, delayed
import multiprocessing

sns.set()
def STPrediction(df : pd.DataFrame,
                 threshold : int = 2,
                 maximum : int = 100):
    # Variables for calculate BACC
    true_pos, true_neg, false_pos, false_neg = 0.0, 0.0, 0.0, 0.0
    total_num = len(df.groupby('pB'))
    correct_num = 0.0
    print(total_num)
    
    for name, group in df.groupby('pB'):
        label : str = 'incorrect'
        patient_ST = group['time_b'].iloc[0]
        high_risk_num = len(group[group['time_a'] < threshold])
        low_risk_num = len(group[group['time_a'] >= threshold])
        print(high_risk_num, low_risk_num)
        ones = len(group[(group['time_a'] >= threshold) & (group['predictions'] == True)])
        zeros = len(group[(group['time_a'] < threshold) & (group['predictions'] == False)])
        print(ones,zeros)
        if zeros != 0:
            ratio = (ones / low_risk_num) / (zeros / high_risk_num)
        else:
            ratio = maximum
        
        if ratio > 1 and patient_ST >= threshold:
            true_pos = true_pos + 1
            correct_num = correct_num + 1
            label = 'correct'
        elif ratio > 1 and patient_ST < threshold:
            false_pos = false_pos + 1
        elif ratio < 1 and patient_ST < threshold:
            true_neg = true_neg + 1
            correct_num = correct_num + 1
            label = 'correct'
        elif ratio < 1 and patient_ST >= threshold:
            false_neg = false_neg + 1
        
        print('The ratio of', name, 'is', ratio, 'and the real survival time is', patient_ST, '. The prediction result is', label)
    
    ACC = correct_num / total_num
    BACC = .5 * (true_pos / (true_pos + false_neg) + true_neg / (true_neg + false_pos))
    return (ACC, BACC)

with open("modelConf.yml", 'r') as cfg_file:
    cfg = yaml.load(cfg_file)

results = pd.DataFrame(columns=['c_index','correct','total', 'accuracy', 'balanced'])
mixed_c_index, train_c_index, test_c_index = [], [], []


for i in range(10):

    counts,predictions = train.deepCinet('VolumeOnlySiamese', num_epochs = 1, batch_size = 10, splitting_model = 1, threshold = 1, split = i, save_model = True, split_seed = i *10 , initial_seed = i*10)
    counts['train']['c_index'] = sum([v[1] for v in  counts['train']['c_index']]) / float(len( counts['train']['c_index']))
    counts['test']['c_index'] = sum([v[1] for v in  counts['test']['c_index']]) / float(len( counts['test']['c_index']))
    counts['mixed']['c_index'] = sum([v[1] for v in  counts['mixed']['c_index']]) / float(len( counts['mixed']['c_index']))

    # Store the mixed c-index values for boxplot
    mixed_c_index.append(counts['mixed']['c_index'])
    train_c_index.append(counts['train']['c_index'])
    test_c_index.append(counts['test']['c_index'])

    result = pd.DataFrame.from_dict(counts)
    result = result.add_suffix("_" + str(i))
    results = results.append(result.T, sort=True)
    dataFrames = utils.df_results(predictions)

    for name, df in dataFrames.items():
        accuracy, balance = STPrediction(df,threshold  = 2, maximum : int = 100)
        results.loc[name +"_"+ str(i), 'accuracy'] = accuracy
        results.loc[name +"_"+  str(i), 'balanced'] = balance

    if(i%2 == 0):
        results.to_csv(cfg['mixed_result_path'])




results.to_csv(cfg['mixed_result_path'])
#results_old = pd.read_csv('/Users/boli/Desktop/result_100times.csv')
#mixed_c_index_old = results[results['label'].str.contains('mixed')]['c_index']

#results_volonly.to_csv(cfg['mixed_result_volonly_path'])

###########################################
# Plot the boxplot of mixed CI of two models
'''
sns.distplot(mixed_c_index_old, label = 'DeepCinet')
sns.distplot(mixed_c_index, label = 'TumorOnly')
plt.legend()
plt.show()
'''

#accuracy_results = list()
#for i in range(1):
 #   print('---------------------------------------')
 #   print('No.', i, 'iteration starts')
#    print('python train.py '+cfg['Config1'])
#    os.system('python train.py '+cfg['Config1'])
 #   os.system('python STprediction.py')
 #   accuracy_results.append(STprediction.correct / STprediction.total)
 #   print('---------------------------------------')

#plt.hist(np.array(accuracy_results))
#plt.show()
