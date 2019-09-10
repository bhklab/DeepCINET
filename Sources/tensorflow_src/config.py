import os
import random
import argparse

import numpy as np

import dotenv
import yaml

APP_ROOT = os.path.join(os.path.dirname(__file__), "../../")

dotenv.load_dotenv(os.path.join(APP_ROOT, ".env"), verbose=True)

with open(os.path.join(APP_ROOT, "environment_Pharma.yml")) as cfg_file:
    cfg = yaml.load(cfg_file)

# todo config required variables

required_vars = [
    'ROOT_DIR',
    'DATA_RAW',
    'DATA_CLINICAL',
    'DATA_RADIOMIC',
    'DATA_CACHE',
    'DATA_PROCESSED',
    'DATA_CLINICAL_PROCESSED',
    'DATA_RADIOMIC_PROCESSED',
    'IMAGE_ROTATIONS'
]

# for v in cfg.keys():
#    print(v)

for var in required_vars:
    if var not in os.environ:
        print(f"ERROR: {var} is not defined in .env file")
        exit(1)

# Data variables
DATA_PATH = os.path.expandvars(cfg['DATA_RAW']['ROOT'])
DATA_PATH_RAW = os.path.expandvars(cfg['DATA_RAW']['ROOT'])
DATA_PATH_CLINICAL = os.path.expandvars(cfg['DATA_RAW']['CLINICAL'])
DATA_PATH_RADIOMIC = os.path.expandvars(cfg['DATA_RAW']['RADIOMIC'])

DATA_PATH_CACHE = os.path.abspath(os.getenv('DATA_CACHE'))

DATA_PATH_PROCESSED = os.path.expandvars(cfg['DATA_PROCESSED']['ROOT'])
DATA_PATH_TARGET = os.path.join(os.path.expandvars(DATA_PATH_PROCESSED), os.path.expandvars(cfg['DATA_PROCESSED']['TARGET']))
DATA_PATH_FEATURE = os.path.join(os.path.expandvars(DATA_PATH_PROCESSED), os.path.expandvars(cfg['DATA_PROCESSED']['FEATURE']))
DATA_PATH_INPUT_TEST_TRAIN = os.path.join(os.path.expandvars(DATA_PATH_PROCESSED), os.path.expandvars(cfg['DATA_PROCESSED']['INPUT_TEST_TRAIN']))

#DATA_PATH_CLINICAL_PROCESSED = os.path.join(DATA_PATH_PROCESSED, cfg['DATA_PROCESSED']['CLINICAL_INFO'])
#DATA_PATH_RADIOMIC_PROCESSED = os.path.join(DATA_PATH_PROCESSED, cfg['DATA_PROCESSED']['RADIOMIC'])
#DATA_PATH_CLINIC_PROCESSED = os.path.join(DATA_PATH_PROCESSED, cfg['DATA_PROCESSED']['CLINIC'])
DATA_PATH_VOLUME_CLINIC_PROCESSED = os.path.join(DATA_PATH_PROCESSED, cfg['DATA_PROCESSED']['VOLUME_CLINIC'])
#DATA_PATH_INPUT_TEST_TRAIN = os.path.join(DATA_PATH_PROCESSED, cfg['DATA_PROCESSED']['INPUT_TEST_TRAIN'])
DATA_PATH_IMAGE = os.path.join(cfg['DATA_PROCESSED']['IMAGE_PATH'])

LOG_DIR = os.path.join(os.path.expandvars(cfg['LOG']['DIR']))
# Log dir, fallback to current directory or we can specify it in the batch file or in the yaml file
# LOG_DIR = os.path.abspath(os.getenv('LOG_DIR', './'))
LOG_LEVEL_CONSOLE = int(cfg['LOG']['LEVEL_CONSOLE'])
LOG_LEVEL_FILE = int(cfg['LOG']['LEVEL_FILE'])

# Rotations for X, Y and Z axes
# At least there should be a rotation for each axis
rotations_list = [int(x) for x in os.getenv('IMAGE_ROTATIONS').split(',')]
IMAGE_ROTATIONS = cfg['ROTATION']

TOTAL_ROTATIONS = IMAGE_ROTATIONS['x'] * IMAGE_ROTATIONS['y'] * IMAGE_ROTATIONS['z']

args = argparse.Namespace()

DATA_BATCH_SIZE = int(cfg['DATA_BATCH_SIZE'])
assert DATA_BATCH_SIZE >= 2

SESSION_SAVE_PATH = os.path.expandvars(cfg['SESSION_SAVE_PATH'])
SUMMARIES_DIR = os.path.expandvars(cfg['SUMMARIES_DIR'])


RANDOM_SEED = int(os.getenv('RANDOM_SEED', 0))
if RANDOM_SEED < 0:
    RANDOM_SEED = random.randint(0, 1000)
else:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

CATEGORY = 5

GENERATOR = cfg['GENERATOR']
GENERATOR['ROOT'] = os.path.expandvars(GENERATOR['ROOT'])
GENERATOR['OUTPUT_PATH'] = os.path.join(GENERATOR['ROOT'], GENERATOR['OUTPUT_PATH'])
GENERATOR['TARGET_PATH'] = os.path.join(GENERATOR['ROOT'], GENERATOR['TARGET_PATH'])
GENERATOR['INPUT_FEATURES'] = os.path.join(GENERATOR['ROOT'], GENERATOR['INPUT_FEATURES'])

cfg['COX']['RESULT_PATH'] = os.path.expandvars(cfg['COX']['RESULT_PATH'])
COX = cfg['COX']

cfg['ML']['RESULT_PATH'] = os.path.expandvars(cfg['ML']['RESULT_PATH'])
ML = cfg['ML']

HYPER_PARAM = cfg['HYPER_PARAM']
HYPER_PARAM['log_path'] = os.path.expandvars(HYPER_PARAM['log_path'])

#: The total number of features that are provided by the CSV of radiomic features
NUMBER_FEATURES = 724
VOLUME_FEATURE_INDEX = 26
