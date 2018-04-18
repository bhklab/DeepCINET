import os

import dotenv

APP_ROOT = os.path.join(os.path.dirname(__file__), "..")

dotenv.load_dotenv(os.path.join(APP_ROOT, ".env"), verbose=True)

required_vars = [
    'DATA',
    'DATA_RAW',
    'DATA_CLINICAL',
    'DATA_CACHE',
    'DATA_PROCESSED',
    'DATA_CLINICAL_PROCESSED',
    'IMAGE_ROTATIONS'
]

for var in required_vars:
    if var not in os.environ:
        print(f"ERROR: {var} is not defined in .env file")
        exit(1)

# Data variables
DATA_PATH = os.path.abspath(os.getenv('DATA'))
DATA_RAW = os.path.abspath(os.getenv('DATA_RAW'))
DATA_CLINICAL = os.path.abspath(os.getenv('DATA_CLINICAL'))
DATA_CACHE = os.path.abspath(os.getenv('DATA_CACHE'))
DATA_PROCESSED = os.path.abspath(os.getenv('DATA_PROCESSED'))
DATA_CLINICAL_PROCESSED = os.path.abspath(os.getenv('DATA_CLINICAL_PROCESSED'))

# Log dir, fallback to current directory
LOG_DIR = os.path.abspath(os.getenv('LOG_DIR', './'))

# Rotations for X, Y and Z axes
# At least there should be a rotation for each axis
rotations_list = [int(x) for x in os.getenv('IMAGE_ROTATIONS').split(',')]
IMAGE_ROTATIONS = {
    'x': rotations_list[0],
    'y': rotations_list[1],
    'z': rotations_list[2]
}

TOTAL_ROTATIONS = IMAGE_ROTATIONS['x']*IMAGE_ROTATIONS['y']*IMAGE_ROTATIONS['z']


if __name__ == '__main__':
    print(IMAGE_ROTATIONS)
