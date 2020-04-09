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


class ImageSiamese(pl.LightningModule):
    def __init__(self, hparams):
        super(ImageSiamese, self).__init__()
        self.hparams = hparams

        self.use_images = hparams.use_images
        self.use_radiomics = hparams.use_radiomics

        self.criterion = nn.BCELoss()
        self.convolution = ConvolutionLayer() if hparams.use_images else nn.Identity()
        self.fc = FullyConnected(hparams)
        self.distance = DistanceLayer() if hparams.use_distance else nn.Identity()
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
        z = (x - y)

        z = self.distance(z)

        return torch.sigmoid(z)

    def on_epoch_start(self):
        print("")
        print("EPOCH START")

    def training_step(self, batch, batch_idx):
        iA = batch['imageA']
        iB = batch['imageB']
        rA = batch['radiomicsA']
        rB = batch['radiomicsB']
        idA = batch['idA']
        idB = batch['idB']
        labels = batch['labels']

        output = self.forward(iA, iB, rA, rB)
        loss = self.criterion(output.view(-1), labels.view(-1))

        if(batch_idx % 1000 == -1):
            np_output = output.view(-1).detach().cpu().numpy()
            print(np_output)

        np_output = output.view(-1).detach().cpu().numpy()
        output_class = np.where(np_output < 0.5, 0, 1)

        correct = np.sum(output_class == labels.view(-1).detach().cpu().numpy())
        total = len(np_output)
        tensorboard_logs = {'train_loss': loss.item(), 'correct' : correct, 'total': total}
        return {'loss': loss, 'custom_logs': tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = np.mean([x['custom_logs']['train_loss'] for x in outputs])
        correct = np.sum([x['custom_logs']['correct'] for x in outputs])
        total =   np.sum([x['custom_logs']['total'  ] for x in outputs])
        tensorboard_logs = {
            'avg_loss' : avg_loss,
            'train_CI' : correct/total}
        return {'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        iA = batch['imageA']
        iB = batch['imageB']
        rA = batch['radiomicsA']
        rB = batch['radiomicsB']
        idA = batch['idA']
        idB = batch['idB']
        labels = batch['labels']

        output = self.forward(iA, iB, rA, rB)
        loss = self.criterion(output.view(-1), labels.view(-1))
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
                            'val_CI' : correct/total}
        return {'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        iA = batch['imageA']
        iB = batch['imageB']
        rA = batch['radiomicsA']
        rB = batch['radiomicsB']
        idA = batch['idA']
        idB = batch['idB']
        labels = batch['labels']

        output = self.forward(iA, iB, rA, rB)
        loss = self.criterion(output.view(-1), labels.view(-1))
        np_output = output.view(-1).detach().cpu().numpy()
        output_class = np.where(np_output < 0.5, 0, 1)

        correct = np.sum(output_class == labels.view(-1).detach().cpu().numpy())
        total = len(np_output)
        return {'test_loss' : loss, 'correct' : correct, 'total': total}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        total = np.sum([x['total'] for x in outputs])
        correct = np.sum([x['correct'] for x in outputs])
        tensorboard_logs = {'test_loss': avg_loss,
                            'test_CI' : correct/total}
        return {'test_loss' : avg_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.hparams.learning_rate,
                                    momentum=self.hparams.momentum,
                                    weight_decay = self.hparams.weight_decay)

        scheduler = torch.optim.lr_scheduler .MultiStepLR(
            optimizer,
            milestones=self.hparams.sc_milestones,
            gamma=self.hparams.sc_gamma)

        return [optimizer], [scheduler]

    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        ## NETWORK
        parser.add_argument('--fc-layers', type=int, nargs='+',
                            default = config.FC_LAYERS)
        parser.add_argument('--dropout', type=float, nargs='+',
                            default = config.DROPOUT)

        ## OPTIMIZER
        parser.add_argument('--learning-rate', type=float, default=config.LR)
        parser.add_argument('--momentum', type=float, default=config.MOMENTUM)
        parser.add_argument('--weight-decay', type=float, default=config.WEIGHT_DECAY)
        parser.add_argument('--sc-milestones', type=int, nargs='+',
                            default = config.SC_MILESTONES)
        parser.add_argument('--sc-gamma', type=float, default=config.SC_GAMMA)

        parser.add_argument('--use-distance', action='store_true', default=config.USE_DISTANCE)
        ##TODO: implement Distance layer
        parser.add_argument('--d-layers', type=int, nargs='+', default=config.D_LAYERS)
        return parser

    def log_model_parameters(self):
        print("PARAMETERS**********************************************")
        print("Convolution layer parameters: %d" % (count_parameters(self.convolution)))
        print("FC layer parameters: %d" % (count_parameters(self.fc)))
        print("distance layer parameters: %d" % (count_parameters(self.distance)))
        print("*******************************************************")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
