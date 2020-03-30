import os
import argparse

import torch
import torch.nn as nn

import numpy as np

import pytorch_lightning as pl

from pytorch_src.models.conv_model import ConvolutionLayer
from pytorch_src.models.fc_model import FullyConnected
from pytorch_src.models.distance_model import DistanceLayer

import pytorch_src.default_settings as config

from pytorch_src.data.dataloader import Dataset
from pytorch_src.data.datareader import PairProcessor

class ImageSiamese(pl.LightningModule):
    def __init__(self, hparams):
        super(ImageSiamese, self).__init__()
        self.hparams = hparams
        print(hparams)

        pairProcessor = PairProcessor(hparams.clinical_path)
        train_ids, test_ids = pairProcessor.train_test_split(
            test_ratio = hparams.test_ratio,
            split_model = PairProcessor.TIME_SPLIT,
            random_seed = 520)
        self.train_set = Dataset(train_ids, hparams.clinical_path, hparams.image_path, hparams.radiomics_path)
        self.val_set = Dataset(test_ids, hparams.clinical_path, hparams.image_path, hparams.radiomics_path)

        print(len(self.val_set))
        self.use_images = hparams.use_images
        self.use_radiomics = hparams.use_radiomics

        self.criterion = nn.BCELoss()
        self.convolution = ConvolutionLayer()
        self.fc = FullyConnected()
        self.distance = DistanceLayer()
        self.log_model_parameters()

    def forward(self, iA, iB, rA, rB):
        if(self.use_images and self.use_radiomics):
            x = self.convolution(iA)
            y = self.convolution(iB)
            x.view(x.size(0), -1)
            y.view(y.size(0), -1)
            x = torch.cat((x, rA), dim=1)
            y = torch.cat((y, rB), dim=1)
        elif(self.use_radiomics):
            x = rA
            y = rB
        elif(self.use_images):
            x = self.convolution(iA)
            y = self.convolution(iB)
        x = self.fc(x)
        y = self.fc(y)
        z = torch.sub(x, y)

        z = self.distance(z)

        return torch.sigmoid(z)

    def training_step(self, batch, batch_idx):
        self.train()
        iA = batch['imageA']
        iB = batch['imageB']
        rA = batch['radiomicsA']
        rB = batch['radiomicsB']
        idA = batch['idA']
        idB = batch['idB']
        labels = batch['labels']

        output = self.forward(iA, iB, rA, rB)
        loss = self.criterion(output.view(-1), labels.view(-1))

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        self.eval()
        iA = batch['imageA']
        iB = batch['imageB']
        rA = batch['radiomicsA']
        rB = batch['radiomicsB']
        idA = batch['idA']
        idB = batch['idB']
        labels = batch['labels']

        output = self.forward(iA, iB, rA, rB)
        loss = self.criterion(output.view(-1), labels.view(-1))
        print("-------")
        print(output.view(-1).detach().cpu().numpy())
        print(labels.view(-1).detach().cpu().numpy())
        print("-------")
        np_output = output.view(-1).detach().cpu().numpy()
        output_class = np.where(np_output < 0.5, 0, 1)
        correct = np.sum(output_class == labels.view(-1).detach().cpu().numpy())
        total = len(np_output)
        return {'val_loss' : loss, 'correct' : correct, 'total': total}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        total = np.sum([x['total'] for x in outputs])
        correct = np.sum([x['correct'] for x in outputs])
        tensorboard_logs = {'val_loss': avg_loss,
                            'correct' : correct,
                            'total' : total,
                            'C-index' : correct/total}
        return {'loss' : avg_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)

    @pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set,
                                           batch_size = self.hparams.batch_size,
                                           shuffle = True,
                                           num_workers=self.hparams.num_workers)

    @pl.data_loader
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set,
                                           batch_size = self.hparams.batch_size,
                                           shuffle = True,
                                           num_workers=self.hparams.num_workers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        ## DATALOADER
        parser.add_argument("--num-workers", default=8, type=int)
        parser.add_argument("--batch-size", default=16, type=int)
        parser.add_argument("--test-ratio", default=0.15, type=float)

        ## NETWORK
        # parser.add_argument('--fc-layers', type=int, nargs='+', default=[1671, 480, 240, 128])
        # parser.add_argument('--d-layers', type=int, nargs='+', default=[128, 48, 16, 1])

        parser.add_argument('--dropout', type=float, default = 0.2)

        ## Training
        parser.add_argument("--epochs", default=10, type=int)
        # network params
        return parser


    def log_model_parameters(self):
        print("PARAMETERS**********************************************")
        print("Convolution layer parameters: %d" % (count_parameters(self.convolution)))
        print("FC layer parameters: %d" % (count_parameters(self.fc)))
        print("distance layer parameters: %d" % (count_parameters(self.distance)))
        print("*******************************************************")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
