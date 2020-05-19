## dataloader
import torch
import torch.utils.data
import torchvision
import torch.nn.functional as F

import random
from random import randint

import os
import numpy as np

from pymrmre import mrmr
import pandas as pd

class Dataset(torch.utils.data.Dataset):
    def __init__(self, idx_list, hparams, is_train):
        self.random = random
        self.hparams = hparams
        self.idx_list = idx_list
        self.is_train = is_train
        self.image_path = hparams.image_path
        self.clinical_path = hparams.clinical_path
        self.clinical_csv = pd.read_csv(self.clinical_path, index_col=0).drop(columns=['Distant']).fillna(0)
        categorical_var = ['Sex', 'ECOG PS', 'Smoking Hx', 'Drinking hx', 'Subsite', 'T', 'N', 'Stage']
        self.clinical_csv = pd.get_dummies(self.clinical_csv, columns = categorical_var)

        self.radiomics_path = hparams.radiomics_path
        if(self.hparams.use_radiomics):
            self.radiomics_csv = pd.read_csv(self.radiomics_path, index_col = 0)
            self.radiomics_csv.drop(columns=['id'], inplace = True)
        self.pairlist = []
        self.buildpairs(idx_list)

    def __len__(self):
        return len(self.pairlist)

    def __getitem__(self, index):
        if(self.is_train):
            return self.train_item(index)
        else:
            return self.test_item(index)

    def mrmr(self):
        features = self.radiomics_csv.iloc[self.idx_list]
        targets = self.clinical_csv[['event', 'time']].iloc[self.idx_list]
        targets.event = targets.event.astype(int)
        solutions = mrmr.mrmr_ensemble_survival(
            features = features,
            targets = targets,
            solution_count = 1,
            solution_length = self.hparams.mrmr)
        return solutions.iloc[0][0]

    def apply_mrmr(self, solutions):
        self.radiomics_csv = self.radiomics_csv[solutions]

    def train_item(self, index):
        row = self.pairlist[index]
        image1 = torch.empty(0)
        image2 = torch.empty(0)
        p_var1 = torch.empty(0)
        p_var2 = torch.empty(0)
        idA = row['pA']
        idB = row['pB']
        if(self.hparams.use_images):
            image1 = self.loadImage(idA)
            image2 = self.loadImage(idB)
        if(self.hparams.use_radiomics):
            p_var1 = torch.cat((p_var1, self.loadPyRadiomics(idA)), dim = 0)
            p_var2 = torch.cat((p_var2, self.loadPyRadiomics(idB)), dim = 0)
        if(self.hparams.use_clinical):
            p_var1 = torch.cat((p_var1, self.loadClinical(idA)), dim = 0)
            p_var2 = torch.cat((p_var2, self.loadClinical(idB)), dim = 0)

        label = torch.tensor(row['label'], dtype=torch.float32)
        return {'imageA': image1,
                'radiomicsA':p_var1,
                'imageB': image2,
                'radiomicsB': p_var2,
                'idA': idA,
                'idB': idB,
                'labels': label}

    def test_item(self, index):
        row = self.pairlist[index]
        image1 = torch.empty(0)
        p_var1 = torch.empty(0)
        idA = row['pA']
        Tevent = self.clinical_csv['time'].iloc[idA]
        event = self.clinical_csv['event'].iloc[idA]
        if(self.hparams.use_images):
            image1 = self.loadImage(idA)
        if(self.hparams.use_radiomics):
            p_var1 = torch.cat((p_var1, self.loadPyRadiomics(idA)), dim = 0)
        if(self.hparams.use_clinical):
            p_var1 = torch.cat((p_var1, self.loadClinical(idA)), dim = 0)

        return {'imageA': image1,
                'radiomicsA':p_var1,
                'event': event,
                'Tevent': Tevent}

    def loadPyRadiomics(self, idx):
        data = self.radiomics_csv.iloc[idx].to_numpy()
        return torch.tensor(np.nan_to_num(data), dtype = torch.float32)

    def loadClinical(self, idx):
        data = self.clinical_csv.drop(columns=['id', 'time', 'event']).iloc[idx].to_numpy()
        return torch.tensor(np.nan_to_num(data), dtype = torch.float32)

    def loadImage(self, idx):
        fileId = str(self.clinical_csv['id'].iloc[idx])
        while(len(fileId) == 6):
            fileId = '0' + fileId
        file_path = os.path.join(self.image_path, fileId + ".npy")
        image = np.load(file_path)

        if(self.is_train):
            if(True):
                npad = ((0,0), (16,16), (16,16))
                image = np.pad(image, pad_width = npad, mode='constant', constant_values = 0)
            if(random.random() < 0.25): # flip on x
                image = np.flip(image, axis=1)
            if(random.random() < 0.25): # flip on y
                image = np.flip(image, axis=2)
            if(random.random() < 0.25): # rotate
                k = randint(0,3)
                image = np.rot90(image, k, (1,2))
            if(True):
                x0,y0 = randint(0,31),randint(0,31)
                image = image[:, x0:x0+64 , y0:y0+64]
        image = torch.tensor(image.copy(), dtype=torch.float32)
        image = image.view(1, image.size(0), image.size(1), image.size(2))
        # print(image.mean())
        # print(image.std())
        # image = (image - image.mean())/image.std()
        # print(torch.isnan(image).any(), flush=True)
        return image

    def buildpairs(self, idx_list):
        if(self.is_train):
            uncen_list = []
            for i in range(len(idx_list)):
                id1 = idx_list[i]
                patient1 = self.clinical_csv.iloc[id1]
                for j in range(len(uncen_list)):
                    id2 = uncen_list[j]
                    self.pairlist.append({
                        'pA': id1,
                        'pB': id2,
                        'label': 1}
                    )
                    if(self.hparams.transitive_pairs <= j and self.is_train):
                        break
                if(int(patient1['event']) == 1):
                    uncen_list.append(id1)
        else:
            for i in range(len(idx_list)):
                self.pairlist.append({
                    'pA' : idx_list[i]
                })


    ##UNUSED
    def get_scalar_feature_length(self):
        cnt = 0
        if(self.hparams.use_radiomics):
            cnt += len(self.radiomics_csv.columns)
        if(self.haparams.use_clinical):
            cnt += len(self.clinical_csv.columns) - 2
