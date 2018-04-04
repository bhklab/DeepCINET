# CNNSurv
Create a Survival Prediction model using a Convolutional Neural Network.

To do so a Siamese network will be used. Two individuals will be used as an input and the network
should predict which one will live longer.

## Installation

To install all the requirements python virtual environment is recomendend. Python 3 is used 
for this project. Once the project has been downloaded do:

```bash
cd CNNSurv
virtualenv virt
source virt/bin/activate
pip install -r requirements.txt
```

## Directory structure

When you call the preprocessor you can set the arguments for:

 - The `datasets` directory
 - The `input` data directory, then the path will be datasets + input
 - The `output` data directory

The `input` data directory has some rules:
 - Each folder represents a patient
 - Each patient has two more folders
   - The patient CT scan
   - The CT scan mask where the tumour has been marked
 - Both folders should contain the same number of slices in `.dcm` files

Example:
```
Dataset
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

