import os
import random
import argparse

import numpy as np

import dotenv
import yaml

APP_ROOT = os.path.join(os.path.dirname(__file__), "../../")

dotenv.load_dotenv(os.path.join(APP_ROOT, ".env"), verbose=True)

with open(os.path.join(APP_ROOT,"environment.yml")) as cfg_file:
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

for v in cfg.keys():
    print(v)


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
DATA_PATH_CLINICAL_PROCESSED = os.path.join(DATA_PATH_PROCESSED, cfg['DATA_PROCESSED']['CLINICAL_INFO'])
DATA_PATH_RADIOMIC_PROCESSED = os.path.join(DATA_PATH_PROCESSED, cfg['DATA_PROCESSED']['RADIOMIC'])
DATA_PATH_CLINIC_PROCESSED = os.path.join(DATA_PATH_PROCESSED, cfg['DATA_PROCESSED']['CLINIC'])
DATA_PATH_VOLUME_CLINIC_PROCESSED = os.path.join(DATA_PATH_PROCESSED, cfg['DATA_PROCESSED']['VOLUME_CLINIC'])
DATA_PATH_INPUT_TEST_TRAIN = os.path.join(DATA_PATH_PROCESSED, cfg['DATA_PROCESSED']['VOLUME_CLINIC'])

# Log dir, fallback to current directory
LOG_DIR = os.path.abspath(os.getenv('LOG_DIR', './'))
LOG_LEVEL_CONSOLE = int(os.getenv('LOG_LEVEL_CONSOLE', 20))
LOG_LEVEL_FILE = int(os.getenv('LOG_LEVEL_FILE', 20))


# Rotations for X, Y and Z axes
# At least there should be a rotation for each axis
rotations_list = [int(x) for x in os.getenv('IMAGE_ROTATIONS').split(',')]
IMAGE_ROTATIONS = {
    'x': rotations_list[0],
    'y': rotations_list[1],
    'z': rotations_list[2]
}

TOTAL_ROTATIONS = IMAGE_ROTATIONS['x']*IMAGE_ROTATIONS['y']*IMAGE_ROTATIONS['z']

args = argparse.Namespace()

DATA_BATCH_SIZE = int(os.getenv('DATA_BATCH_SIZE', 2))
assert DATA_BATCH_SIZE >= 2

SESSION_SAVE_PATH = os.getenv('SESSION_SAVE_PATH', './Model')
SUMMARIES_DIR = os.getenv('SUMMARIES_DIR', './Summaries')

RANDOM_SEED = int(os.getenv('RANDOM_SEED', 0))
if RANDOM_SEED < 0:
    RANDOM_SEED = random.randint(0, 1000)
else:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

CATEGORY = 5


#: The total number of features that are provided by the CSV of radiomic features
NUMBER_FEATURES = 724
VOLUME_FEATURE_INDEX = 26
