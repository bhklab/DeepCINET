# CNNSurv
Create a Survival Prediction model using a Convolutional Neural Network.

To do so a Siamese network will be used. Two individuals will be used as an input and the network
should predict which one will live longer.

This project uses **python3**

More documentation can be found in the documentation page:
[https://jmigual.github.io/CNNSurv](https://jmigual.github.io/CNNSurv)

## Installation

To install all the requirements python virtual environment is recommended. Python 3 is used 
for this project. Once the project has been downloaded do:

```bash
cd CNNSurv
virtualenv venv --python=`which python3`
source venv/bin/activate
pip install -r requirements.txt
```

The dataset files are in the `Data` folder (which has to be created). The easiest way is to 
create a symbolic link to where the data is located, if using Mordor it would be:

```bash
cd CNNSurv # Go to project's root directory
ln -s /mnt/work1/users/bhklab/Data/HNK Data
```

All the environment variables are defined in the `.env` file which should be located in the
project's root dir. An example can be found under `Sources/env_default.txt` you can just
copy it to the root directory:

```bash
cp Sources/env_default.txt .env
```

## Running the scripts

There are two scripts that are prepared to be run:

- `Sources/preprocess.py` This script pre-processes all the data as long as all the `.env` variables are properly
defined
- `Sources/train.py` This script trains a model. All the options that can be used to train a model can be found
by passing the `--help` flag.

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
Contains the pre-processed data. Each folder contains a `.npz` file with 4 arrays of 64x64x64.
Each array has been pre-processed with the following process:

1. Process mask
    1. Smooth mask with Gaussian filter and set `1` for values `> 0`
    2. Get mask bounding box for all the `1` values in the mask
2. Slice main image with mask bounding box
3. Set main image values to `0` to `1` range
    1. Clip the minimum and maximum values to -1000 and 400 respectively
    2. Convert image to a `0` to `1` range
    3. Apply mask by setting to `0` main image values that are not inside the mask
4. Resize image to `64x64x64`
5. Normalize image by setting `mean = 0` and `std = 1`
6. Rotate the image and create the image augmentation arrays  

The arrays and the original image are stored then in the `npz` with the key corresponding
to the rotation by `<rot_x>_<rot_y>_<rot_z>`

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

For now the used data augmentation method is rotating the images among their axes. To define the rotations
there is the `IMAGE_ROTATIONS` variable defined in the `.env` file. Each value represents an axis (x, y, z)
so if `IMAGE_ROTATIONS=1,1,4` it means:

- 1 rotation on the x axis
- 1 rotation on the y axis
- 4 rotations on the z axis

## Documentation

The project's documentation can be found at 
[https://jmigual.github.io/CNNSurv](https://jmigual.github.io/CNNSurv)

To build the documentation install all the `requirements.txt` packages are needed plus graphviz and make. 
Once you have all the requirements just type: `make docs` to build it. The documentation will be on `docs_build/`.