import os
import argparse

import numpy as np

import yaml

CUR_DIR = os.path.join(os.path.dirname(__file__), "./")

with open(os.path.join(CUR_DIR, "environment.yml")) as cfg_file:
    cfg = yaml.load(cfg_file, loader=yaml.FullLoader)

#DATA
DATA_ROOT = cfg['DATA']

CLINICAL_PATH = os.path.join(DATA_ROOT, cfg['DATA_PROCESSED']['CLINICAL_PATH'])
RADIOMICS_PATH = os.path.join(DATA_ROOT,cfg['DATA_PROCESSED']['RADIOMICS_PATH'])

#PATH
IMAGE_PATH = os.path.join(DATA_ROOT, cfg['PATH']['IMAGE'])

#HPARAMS
hparams = cfg['HPARAMS']
EPOCHS = int(hparams['EPOCHS'])
USE_IMAGES = bool(hparams['USE_IMAGES'])
USE_RADIOMICS = bool(hparams['USE_RADIOMICS'])

N_FC_LAYERS = len(hparams['FCLAYERS'])
FC_LAYES = hparams['FCLAYERS']

N_D_LAYERS = len(hparams['DLAYERS'])
D_LAYERS = hparams['DLAYERS']

N_CONV_LAYERS = len(hparams['CONVLAYERS'])
CONV_LAYERS = hparams['CONVLAYERS']

DROPOUT = hparams['DROPOUT']

TEST_RATIO = hparams['TEST_RATIO']
BATCH_SIZE = hparams['BATCH_SIZE']

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def add_bool_arg(parser, name, default=False, help_msg=''):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true', help=help_msg)
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name.replace('-','_'):default})

# Dataloader
data_arg = add_argument_group('Data')
data_arg.add_argument('--clinical-path', type=str, default=CLINICAL_PATH, help='Path to clinical variables')
data_arg.add_argument('--radiomics-path', type=str, default=RADIOMICS_PATH, help='Path to radiomics features')
data_arg.add_argument('--images-path', type=str, default=IMAGE_PATH, help='Path to patient CTs')

# Hparams
hparams_arg = add_argument_group('Hparams')
hparams_arg.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')

