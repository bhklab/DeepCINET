import tensorflow as tf
from test_tube import Experiment, HyperOptArgumentParser
import pandas as pd
import random
import config
import os
import sys
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# import STprediction
from tensorflow_src import train_test_models

mixed_c_index, train_c_index, test_c_index = [], [], []

cfg = config.HYPER_PARAM

# the number of times which should run a model
running_times = cfg['running_times']

# randomly feed random state.
random_states = list(range(running_times * 2))
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
        parameters = dict(model=hparams.model,
                          target_path=hparams.target_path,
                          feature_path=hparams.feature_path,
                          input_path=hparams.input_path,
                          results_path=hparams.result_path,
                          num_epochs=hparams.num_epochs,
                          batch_size=hparams.batch_size,
                          splitting_model=1,
                          split=random_states[i],
                          learning_rate=hparams.learningRate,
                          dropout=.2,
                          threshold=0.5,
                          save_model=True,
                          regularization=hparams.regularization,
                          split_seed=random_states[i],
                          initial_seed=random_states[i],
                          mrmr_size=hparams.mrmr_size,
                          read_splits=True,
                          full_summary=False,
                          cv_folds=1,
                          split_number=i,
                          train_distance=0.2,
                          test_distance=0.2,
                          survival=False
                          )
        counts, predictions = train_test_models.deepCinet(**parameters)

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
                       }
        log_content = {**log_content, **parameters}
        exp.log(log_content)
        result = pd.DataFrame(log_content, index=[i])
        result['random state'] = random_states[i]
        result['number'] = i
        results1 = results1.append(result)
        # if (i % 3 == 0):
        # os.path.join(cfg['mixed_result_path'], f"epoch{hparams.num_epochs}",.csv"
        # results1.to_csv(pd.read_csv()))
    path = os.path.join(cfg['mixed_result_path'], f"model{hparams.model}_"
                                                  f"epoch{hparams.num_epochs}_"
                                                  f"batch{hparams.batch_size}_"
                                                  f"regularization{hparams.regularization}_"
                                                  f"learningRate{hparams.learningRate}_"
                                                  f"mrmr{hparams.mrmr_size}")
    # print(results1)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    results1.to_csv(os.path.join(path, "result.csv"), index=False)
    exp.save()


def hyperParamSelection(target_path: str = config.DATA_PATH_TARGET,
                        feature_path: str = config.DATA_PATH_FEATURE,
                        input_path: str = config.DATA_PATH_INPUT_TEST_TRAIN,
                        results_path: str = config.SESSION_SAVE_PATH):
    # set up our argparser and make the model tunable
    # Use either random_search or grid_search for tuning
    parser = HyperOptArgumentParser(strategy='random_search')
    parser.add_argument('--test_tube_exp_name', default='DeepCINET_ScalarOnlySiamese7')
    parser.add_argument('--log_path', default=cfg['log_path'])
    parser.add_argument('--target_path', default=target_path)
    parser.add_argument('--feature_path', default=feature_path)
    parser.add_argument('--input_path', default=input_path)
    parser.opt_list('--result_path', default=results_path)
    parser.opt_list('--model', default='ScalarOnlySiamese', options=cfg['model'], tunable=True)
    parser.opt_list('--mrmr_size', default=0, options=cfg['mrmr_size'], tunable=True)
    parser.opt_list('--num_epochs', default=5, options=cfg['num_epochs'], tunable=True)
    parser.opt_list('--batch_size', default=40, options=cfg['batch_size'], tunable=True)
    parser.opt_list('--regularization', default=0.5, options=cfg['regularization'], tunable=True)
    parser.opt_list('--learningRate', default=0.0002, options=cfg['learningRate'], tunable=True)
    parser.opt_list('--drug')

    hyperparams = parser.parse_args()

    # optimize on 4 gpus at the same time
    # each gpu will get 1 experiment with a set of hyperparams
    hyperparams.optimize_parallel_cpu(trainDeepCInet, nb_trials=100, nb_workers=2)


hyperParamSelection()
