import tensorflow as tf
from test_tube import Experiment, HyperOptArgumentParser
import pandas as pd
import random
import yaml
from tensorflow_src import train_test_models
import os

"""
This script demonstrates how to do a hyperparameter search over 2 parameters in tensorflow
on 4 simultaneous GPUs. Each trial will also save its own experiment logs.   
A single trial gets allocated on a single GPU until all trials have completed.   
This means for 10 trials and 4 GPUs, we'll run 4 in parallel twice and the last 2 trials in parallel.   
"""


mixed_c_index, train_c_index, test_c_index = [], [], []


with open("modelConf.yml", 'r') as cfg_file:
    cfg = yaml.load(cfg_file)

#the number of times which should run a model
running_times = cfg['running_times']

#randomly feed random state.
random_states = list(range(running_times* 2))
random.seed(1)
random.shuffle(random_states)

# main training function (very simple)
def trainDeepCInet(hparams):
    # init exp and track all the parameters from the HyperOptArgumentParser
    exp = Experiment(
        name=hparams.test_tube_exp_name,
        save_dir=hparams.log_path,
        autosave=True,
    )
    results1 = pd.DataFrame()
    exp.argparse(hparams)
    for i in range(running_times):

        counts, predictions = train_test_models.deepCinet(model = hparams.model,
                                                          num_epochs= hparams.num_epochs,
                                                          batch_size= hparams.batch_size,
                                                          splitting_model=1,
                                                          split=random_states[i],
                                                          save_model=True,
                                                          split_seed=random_states[i],
                                                          regularization=hparams.regularization,
                                                          initial_seed=random_states[i],
                                                          learning_rate=hparams.learningRate,
                                                          mrmr_size = hparams.mrmr_size,
                                                          read_splits = True,
                                                          split_number=i,
                                                          cv_folds=5
                                                          )

        counts['train']['c_index'] = sum([v[1] for v in counts['train']['c_index']]) / float(
            len(counts['train']['c_index']))
        counts['test']['c_index'] = sum([v[1] for v in counts['test']['c_index']]) / float(
            len(counts['test']['c_index']))
        counts['mixed']['c_index'] = sum([v[1] for v in counts['mixed']['c_index']]) / float(
            len(counts['mixed']['c_index']))

        # Store the mixed c-index values for boxplot
        mixed_c_index.append(counts['mixed']['c_index'])
        train_c_index.append(counts['train']['c_index'])
        test_c_index.append(counts['test']['c_index'])

        log_content = {'train': counts['train']['c_index'],
                       'test': counts['test']['c_index'],
                       'mixed': counts['mixed']['c_index'],
                       'models':[hparams.model],
                       'num_epochs': [hparams.num_epochs],
                       'batch_size': [hparams.batch_size],
                       'regularization': [hparams.regularization],
                       'learning_rate': [hparams.learningRate],
                       'mrmr_size': [hparams.mrmr_size]
                       }
        exp.log(log_content)
        result = pd.DataFrame.from_dict(log_content)
        result['random state'] = random_states[i]
        result['number'] = i
        results1 = results1.append(result)
        #if (i % 3 == 0):
            #os.path.join(cfg['mixed_result_path'], f"epoch{hparams.num_epochs}",.csv"
            #results1.to_csv(pd.read_csv()))
    path = os.path.join(cfg['mixed_result_path'], f"epoch{hparams.num_epochs}", f"batch{hparams.batch_size}", f"regularization{hparams.regularization}",f"learningRate{hparams.learningRate}",f"mrmr{hparams.mrmr_size}")
    #print(results1)
    #results1.to_csv(os.path.join(path,"result.csv"), index=False)
    exp.save()


# set up our argparser and make the model tunable
# Use either random_search or grid_search for tuning
parser = HyperOptArgumentParser(strategy='random_search')
parser.add_argument('--test_tube_exp_name', default='DeepCINET_ScalarOnlySiamese7')
parser.add_argument('--log_path', default='/Users/farnoosh/Desktop/test')

parser.opt_list('--model', default='ScalarOnlySiamese', options=['ScalarOnlySiamese'], tunable=True)
parser.opt_list('--mrmr_size', default=40, options=[40, 200], tunable=True)
parser.opt_list('--num_epochs', default=100, options=[200,400], tunable=True)
parser.opt_list('--batch_size', default=250, options=[250], tunable=True)
parser.opt_list('--regularization', default=0.5, options=[0.5,0.8, 1.5], tunable=True)
parser.opt_list('--learningRate', default=0.0001, options=[0.0003, 0.0002, 0.0001], tunable=True)

hyperparams = parser.parse_args()


# optimize on 4 gpus at the same time
# each gpu will get 1 experiment with a set of hyperparams
hyperparams.optimize_parallel_cpu(trainDeepCInet, nb_trials=100, nb_workers=2)