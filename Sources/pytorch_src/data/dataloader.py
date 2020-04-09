## dataloader
import torch
import torch.utils.data

import os
import numpy as np

import pandas as pd
class Dataset(torch.utils.data.Dataset):
    def __init__(self, idx_list, clinical_path, image_path, pyRadiomics_path, hparams):
        self.hparams = hparams
        self.image_path = image_path
        self.clinical_path = clinical_path
        self.clinical_csv = pd.read_csv(clinical_path, index_col=0)
        self.pyRadiomics_path = pyRadiomics_path
        if(self.hparams.use_radiomics):
            self.radiomics_csv = pd.read_csv(pyRadiomics_path)
        self.pairlist = []
        self.buildpairs(idx_list)

    def __len__(self):
        return len(self.pairlist)

    def __getitem__(self, index):
        row = self.pairlist[index]
        image1 = torch.empty(0)
        image2 = torch.empty(0)
        radiomics1 = torch.empty(0)
        radiomics2 = torch.empty(0)
        idA = row['pA']
        idB = row['pB']
        if(self.hparams.use_images):
            image1 = self.loadImage(idA)
            image2 = self.loadImage(idB)
        if(self.hparams.use_radiomics):
            radiomics1 = self.loadPyRadiomics(idA)
            radiomics2 = self.loadPyRadiomics(idB)
        label = torch.tensor(row['label'], dtype=torch.float32)
        return {'imageA': image1,
                'radiomicsA':radiomics1,
                'imageB': image2,
                'radiomicsB': radiomics2,
                'idA': idA,
                'idB': idB,
                'labels': label}

    def loadPyRadiomics(self, idx):
        data = self.radiomics_csv.iloc[idx]
        data = data.to_numpy()[2:]
        return torch.tensor(np.nan_to_num(data), dtype = torch.float32)

    def loadImage(self, idx):
        file_path = os.path.join(self.image_path, idx + ".npy")
        image = torch.tensor(np.load(file_path), dtype=torch.float32)
        image = image.view(1, image.size(0), image.size(1), image.size(2))
        return image

    def buildpairs(self, idx_list):
        for i in range(len(idx_list)):
            id1 = idx_list[i]
            patient1 = self.clinical_csv.iloc[id1]
            if(int(patient1['event']) == 1):
                for j in range(i+1, len(idx_list)):
                    id2 = idx_list[j]
                    # patient2 = self.clinical_csv.iloc[id2]
                    # if(id1 != id2):
                        # if(patient2['time'] > patient1['time']):
                            # print("Ids: {} - {}".format(id1, id2))
                            # print("Clinical Features:")
                            # print(patient1)
                            # print(self.clinical_csv.iloc[id2])
                    if((i + j)%2 == 1):
                        self.pairlist.append({
                            'pA': id1,
                            'pB': id2,
                            'label': 1}
                        )
                    else:
                        self.pairlist.append({
                            'pA': id2,
                            'pB': id1,
                            'label': 0}
                        )
