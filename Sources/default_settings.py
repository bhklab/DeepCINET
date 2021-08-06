import os
import argparse

import numpy as np

import yaml

CUR_DIR = os.path.join(os.path.dirname(__file__), "./")

with open(os.path.join(CUR_DIR, "environment.yml")) as cfg_file:
    cfg = yaml.safe_load(cfg_file)

#DATA
DATA_ROOT = cfg['DATA']

GENE_PATH = os.path.join(DATA_ROOT, cfg['DATA_PROCESSED']['GENE_PATH'])
# RADIOMICS_PATH = os.path.join(DATA_ROOT,cfg['DATA_PROCESSED']['RADIOMICS_PATH'])

#PATH
# IMAGE_PATH = os.path.join(DATA_ROOT, cfg['PATH']['IMAGE'])

#HPARAMS
hparams = cfg['HPARAMS']
# EPOCHS = int(hparams['EPOCHS'])
# USE_IMAGES = bool(hparams['USE_IMAGES'])
# USE_RADIOMICS = bool(hparams['USE_RADIOMICS'])
# FC_LAYERS = hparams['FCLAYERS']
# DROPOUT = hparams['DROPOUT']

# USE_DISTANCE = bool(hparams['USE_DISTANCE'])
# D_LAYERS = hparams['DLAYERS']

# TEST_RATIO = hparams['TEST_RATIO']
# VAL_RATIO = hparams['VAL_RATIO']
BATCH_SIZE = hparams['BATCH_SIZE']
NUM_WORKERS = hparams['NUM_WORKERS']

USE_FOLDS = bool(hparams['USE_FOLDS'])
FOLDS = hparams['FOLDS']

# LR= hparams['LR']
MOMENTUM= hparams['MOMENTUM']
WEIGHT_DECAY= hparams['WEIGHT_DECAY']
SC_MILESTONES= hparams['SC_MILESTONES']
SC_GAMMA= hparams['SC_GAMMA']
