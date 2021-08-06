import torch
import torch.utils.data

import random
from random import randint

import os
import numpy as np

import pandas as pd
class Dataset(torch.utils.data.Dataset):
    """Data set class which returns a pytorch data set object

        Returns a iterable data set object extending from the pytorch dataset
        object.
    """
    def __init__(self, hparams, is_train, delta=0, idxs=None):
        self.gene_exprs = pd.read_csv(hparams.gene_path)
        if idxs is not None:
            self.gene_exprs = self.gene_exprs.iloc[idxs]
        self.drug_resps = self.gene_exprs["target"].to_numpy()
        self.cell_lines = self.gene_exprs["cell_line"].to_numpy()
        self.gene_exprs = self.gene_exprs.drop(["target", "cell_line"], axis=1).to_numpy()
        self.gene_exprs = (self.gene_exprs - np.mean(self.gene_exprs, axis=0))/np.std(self.gene_exprs, axis=0)

        self._is_train = is_train
        self.delta = delta
        # might need to pass in delta in the future
        self._sample_list = self._build_pairs(self.delta)

    def __len__(self):
        return len(self._sample_list)

    def __getitem__(self, index):
        return self.train_item(index) if self._is_train else self.test_item(index)

    def train_item(self, pair_idx):
        row = self._sample_list[pair_idx]
        gene1 = self._load_item(row['idxA'])
        gene2 = self._load_item(row['idxB'])
        label = torch.tensor(row['label'], dtype=torch.float32)
        return {'geneA': gene1,
                'geneB': gene2,
                'labels': label}

    def test_item(self, idx):
        gene = self._load_item(idx)
        response = self._load_response(idx)
        cell_line = self._load_cell_line(idx)
        return {'gene': gene, 
                'response': response,
                'cell_line': cell_line}

    def _load_item(self, idx):
        """ Function to load the features of a cell line

        :param idx: the cell line index in our input csv
        :return: returns a gene expression variable
        """
        gene = self.gene_exprs[idx]
        gene = torch.tensor(gene.copy(), dtype=torch.float32)
        return gene
    
    def _load_response(self, idx):
        response = self.drug_resps[idx]
        response = torch.tensor(response.copy(), dtype=torch.float32)
        return response
    
    def _load_cell_line(self, idx):
        cell_lines_selected = self.cell_lines[idx]
        # cell_lines_selected = torch.tensor(cell_lines_selected.copy())
        return cell_lines_selected

    def _build_pairs(self, delta):
        ''' build pairs of indices and labels for training data
        '''
        if self._is_train:
            return self.get_concordant_pair_list(delta)
        else:
            return self.drug_resps

    def get_concordant_pair_list(self, delta):
        pairs = []
        size = self.gene_exprs.shape[0]
        for i in range(size-1):
            for j in range(i+1, size, 1):
                if (abs(self.drug_resps[i] - self.drug_resps[j]) > delta):
                    pairs.append({'idxA': i, 'idxB': j, 
                    'label': self.get_relationship_from_index(i, j)})
        return pairs

    def get_relationship_from_index(self, i, j):
        '''
        check if drug reponse at index i is greater than drug response at index j
        '''
        drug_i = self.drug_resps[i]
        drug_j = self.drug_resps[j]
        return int(drug_i > drug_j)
                

