## dataloader
import torch
import torch.utils.data

import os
import numpy as np

import pandas as pd
class Dataset(torch.utils.data.Dataset):
    def __init__(self, id_list, clinical_path, image_path, pyRadiomics_path):
        self.image_path = image_path
        self.clinical_path = clinical_path
        self.clinical_csv = pd.read_csv(clinical_path, index_col=0)
        self.pyRadiomics_path = pyRadiomics_path
        if(pyRadiomics_path != None):
            self.radiomics_csv = pd.read_csv(pyRadiomics_path, index_col=0)
            self.standardize()
        self.pairlist = []
        self.buildpairs(id_list)

    def __len__(self):
        return len(self.pairlist)

    def __getitem__(self, index):
        row = self.pairlist[index]
        image1 = None
        image2 = None
        radiomics1 = None
        radiomics2 = None
        idA = row['pA']
        idB = row['pB']
        if(self.image_path != None):
            image1 = self.loadImage(idA)
            image2 = self.loadImage(idB)
        if(self.pyRadiomics_path != None):
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

    def loadPyRadiomics(self, id):
        data = self.radiomics_csv[(self.radiomics_csv['id'] == id)
                                  & (self.radiomics_csv['roi'] == 'GTV')].copy()
        data = data.drop(columns=['id', 'roi', 'exam']).iloc[0].to_numpy()
        return torch.tensor(np.nan_to_num(data), dtype = torch.float32)

    def loadImage(self, idx):
        file_path = os.path.join(self.image_path, idx + ".npy")
        image = torch.tensor(np.load(file_path), dtype=torch.float32)
        image = image.view(1, image.size(0), image.size(1), image.size(2))
        return image

    def standardize(self):
        for feature in self.radiomics_csv.columns:
            if(feature not in ['id', 'roi', 'exam']):
                self.radiomics_csv[feature] = (self.radiomics_csv[feature] - self.radiomics_csv[feature].mean())/(self.radiomics_csv[feature].std())

    def buildpairs(self, id_list):
        for i in range(len(id_list)):
            for j in range(len(id_list)):
                if(i == j):
                    continue
                id1 = id_list[i]
                id2 = id_list[j]
                patient1 = self.clinical_csv[self.clinical_csv['id'] == id1].iloc[0]
                patient2 = self.clinical_csv[self.clinical_csv['id'] == id2].iloc[0]
                label = self.comp(patient1,patient2)
                if(label != -1):
                    self.pairlist.append({
                        'pA': patient1['id'],
                        'pB': patient2['id'],
                        'label': label}
                    )

    def comp(self, patient1, patient2):
        label = -1
        if(patient1['event'] == 1):
            if(patient1['time'] < patient2['time']):
                label = 0
            elif(patient2['event'] == 1):
                label = 1
        else:
            if(patient2['event'] and patient2['time'] < patient1['time']):
                label = 1
        return label
