import os
import argparse

import numpy as np

import yaml

CUR_DIR = os.path.join(os.path.dirname(__file__), "./")

with open(os.path.join(CUR_DIR, "environment.yml")) as cfg_file:
    cfg = yaml.load(cfg_file)

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

FC_LAYES = hparams['FCLAYERS']

D_LAYERS = hparams['DLAYERS']

CONV_LAYERS = hparams['CONVLAYERS']

DROPOUT = hparams['DROPOUT']

TEST_RATIO = hparams['TEST_RATIO']
BATCH_SIZE = hparams['BATCH_SIZE']
