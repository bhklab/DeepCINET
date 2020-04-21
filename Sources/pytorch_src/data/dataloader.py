## dataloader
import torch
import torch.utils.data

import os
import numpy as np

import pandas as pd
class Dataset(torch.utils.data.Dataset):
    def __init__(self, idx_list, hparams, is_train):
        self.hparams = hparams
        self.is_train = is_train
        self.image_path = hparams.image_path
        self.clinical_path = hparams.clinical_path
        self.clinical_csv = pd.read_csv(self.clinical_path, index_col=0)
        self.radiomics_path = hparams.radiomics_path
        if(self.hparams.use_radiomics):
            self.radiomics_csv = pd.read_csv(self.radiomics_path)
        self.pairlist = []
        self.buildpairs(idx_list)

    def __len__(self):
        return len(self.pairlist)

    def __getitem__(self, index):
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

    def loadPyRadiomics(self, idx):
        data = self.radiomics_csv.iloc[idx]
        data = data.to_numpy()[15] if self.hparams.volume_only else data.to_numpy()[2:]
        return torch.tensor(np.nan_to_num(data), dtype = torch.float32)

    def loadClinical(self, idx):
        clinical_var = [
            'Age',
            'Sex',
            'ECOG PS',
            'Smoking Hx',
            'Drinking hx',
            'T',
            'N',
            'Stage'
        ]
        data = self.clinical_csv[clinical_var].iloc[idx]
        return torch.tensor(np.nan_to_num(data), dtype = torch.float32)

    def loadImage(self, idx):
        fileId = str(self.clinical_csv['id'].iloc[idx])
        while(len(fileId) == 6):
            fileId = '0' + fileId
        file_path = os.path.join(self.image_path, fileId + ".npy")
        image = torch.tensor(np.load(file_path), dtype=torch.float32)
        image = image.view(1, image.size(0), image.size(1), image.size(2))
        return image

    def buildpairs(self, idx_list):
        uncen_list = []
        for i in range(len(idx_list)):
            id1 = idx_list[i]
            patient1 = self.clinical_csv.iloc[id1]
            for j in range(len(uncen_list)):
                id2 = uncen_list[j]
                self.pairlist.append({
                    'pA': id1,
                    'pB': id2,
                    'label': 0}
                )
                if(self.hparams.transitive_pairs <= j and self.is_train):
                    break
            if(int(patient1['event']) == 1):
                uncen_list.append(id1)
