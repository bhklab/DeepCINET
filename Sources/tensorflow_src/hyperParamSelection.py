import tensorflow as tf
from test_tube import Experiment, HyperOptArgumentParser
import pandas as pd
import random
import yaml
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

#import STprediction
from tensorflow_src import train_test_models


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
                       'models':hparams.model,
                       'num_epochs': hparams.num_epochs,
                       'batch_size': hparams.batch_size,
                       'regularization': hparams.regularization,
                       'learning_rate': hparams.learningRate,
                       'mrmr_size': hparams.mrmr_size
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
parser.add_argument('--test_tube_exp_name', default='DeepCINET_ScalarOnlySiamese9')
parser.add_argument('--log_path', default=cfg['log_path'])

parser.opt_list('--model', default='ClinicalOnlySiamese', options=cfg['hyper_param']['model'], tunable=True)
parser.opt_list('--mrmr_size', default=0, options= cfg['hyper_param']['mrmr_size'], tunable=True)
parser.opt_list('--num_epochs', default=100, options=cfg['hyper_param']['num_epochs'], tunable=True)
parser.opt_list('--batch_size', default=100, options=cfg['hyper_param']['batch_size'], tunable=True)
parser.opt_list('--regularization', default=0.5, options=cfg['hyper_param']['regularization'], tunable=True)
parser.opt_list('--learningRate', default=0.0003, options=cfg['hyper_param']['learningRate'], tunable=True)

hyperparams = parser.parse_args()


# optimize on 4 gpus at the same time
# each gpu will get 1 experiment with a set of hyperparams
hyperparams.optimize_parallel_cpu(trainDeepCInet, nb_trials=100, nb_workers=2)