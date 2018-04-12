# CNNSurv
Create a Survival Prediction model using a Convolutional Neural Network.

To do so a Siamese network will be used. Two individuals will be used as an input and the network
should predict which one will live longer.

## Installation

To install all the requirements python virtual environment is recomendend. Python 3 is used 
for this project. Once the project has been downloaded do:

```bash
cd CNNSurv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

The dataset files are in the `Data` folder (which has to be created). The easiest way is to 
create a symbolic link to where the data is located, if using Mordor it would be:

```bash
ln -s /mnt/work1/users/bhklab/Data/HNK Data
```

All the environment variables are defined in the .env file which should be located in the
project's root dir. An example can be this one:

```dotenv
# All the entries starting with DATA will have their absolute path expanded
ROOT_DIR= # your project's root dir here
DATA=${ROOT_DIR}/Data
DATA_RAW=${DATA}/HNK_raw
DATA_CLINICAL=${DATA}/HNK_raw/clinical_info.csv
DATA_CACHE=${DATA}/.cache
DATA_PROCESSED=${DATA}/HNK_processed
DATA_CLINICAL_PROCESSED=${DATA_PROCESSED}/clinical_info.csv
```

## Data directories

It's recommended to store the dataset under `<root_dir>/Data` or to create a symlink there.
Although all the scripts use the variables defined in the `.env` file.

The required environment variables are:

  - `DATA_RAW`: Path to directory containing the raw images in the `.dcm` file
  - `DATA_CLINICAL`: Path to `csv` file containing all the clinical data
  - `DATA_CACHE`: Path to location to be used as cache when pre-processing data
  - `DATA_PROCESSED`: Path to location where the pre-processed data will be stored
  - `DATA_CLINICAL_PROCESSED`: Path to the `csv` file containing the clinical data after
    pre-process step

The `Data` folder should be located in the project's root directory. The input (raw) data is
located in the `HNK_raw` directory. The pre-processed data is inside the `HNK_processed` dir.


### Input Directory structure

The `DATA_RAW` directory has some rules:
 - Each folder represents a patient
 - Each patient has two more folders
   - The patient CT scan
   - The CT scan mask where the tumour has been marked
 - Both folders should contain the same number of slices in `.dcm` files

Example:
```
${DATA_RAW}
├── FHBO613
│   ├── FHBO613
│   │   ├── IMG0001.dcm
│   │   ├── IMG0002.dcm
│   │   └── IMG0003.dcm
│   └── FHBO613-MASS
│       ├── IMG0001.dcm
│       ├── IMG0002.dcm
│       └── IMG0003.dcm
└── FHBO614
    ├── FHBO614
    │   ├── IMG0001.dcm
    │   ├── IMG0002.dcm
    │   └── IMG0003.dcm
    └── FHBO614-MASS
        ├── IMG0001.dcm
        ├── IMG0002.dcm
        └── IMG0003.dcm
``` 

### `${DATA_CACHE}` Directory structure

When pre-processing the `${DATA_RAW}` files the `${DATA_CACHE}` directory will be created. It contains all 
the images in the `.dcm` files but in a single file for each patient in a `.npz` file (numpy) that 
contains two arrays. Each array is a 3D array containing the whole 3D image.

To load an array just use the `np.load` function provided by `numpy`:

```python
import numpy as np

file = 'FHBO613.npz'
npz_file = np.load(file)
main_stack = npz_file['main']
mask_stack = npz_file['mask']
```

### `${DATA_PROCESSED}` directory
Contains the pre-processed data. Each folder contains a `.npz` file with 4 arrays of 64x64x64. The arrays
are the original image and all the possible rotations in the 3 axes. The name of each array is
`<rot_x>_<rot_y>_<rot_z>`

```
${DATA_PROCESSED}
├── FHBO613
│   └── FHBO613.npz
└── FHBO614
    └── FHBO614.npz
```

#### Clinical info
The clinical info `csv` file is located at `${DATA_CLINICAL_PROCESSED}` (defined by your `.env` file)
It contains the clinical info for the patients. The file can be loaded with `pandas` and it contains 
the fields: `id`, `age`, `time` and `event`.

## Data augmentation

For now the only data augmentation technique that we are using is rotating the image in 1 axis,
this means that we have `4` rotations for each image. All this rotations are saved into disk.
