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

## Data directories

The `Data` folder should be located in the project's root directory. The input (raw) data is
located in the `HNK_raw` directory. The pre-processed data is inside the `HNK_processed` dir.

The structure after pre-processing should be:

```
Data
├── .cache
├── HNK_raw
└── HNK_processed
```

### Input Directory structure

The `input` data directory has some rules:
 - Each folder represents a patient
 - Each patient has two more folders
   - The patient CT scan
   - The CT scan mask where the tumour has been marked
 - Both folders should contain the same number of slices in `.dcm` files

Example:
```
HNK_raw
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

### `.cache` Directory structure

When pre-processing the `HNK_raw` files the `.cache` directory will be created. It contains all the images
in the `.dcm` files but in a single file for each patient in a `.npz` file (numpy) that contains two arrays.
Each array is a 3D array containing the whole 3D image.

To load an array just use the `np.load` function provided by `numpy`:

```python
import numpy as np

file = 'FHBO613.npz'
npz_file = np.load(file)
main_stack = npz_file['main']
mask_stack = npz_file['mask']
```


## Data augmentation

For now the only data augmentation technique that we are using is rotating the image in the 3 different axis,
this means that we have `4x4x4=64` rotations for each image. All this rotations are saved into disk.
