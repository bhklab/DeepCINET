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
        self.use_clinical = hparams.use_clinical
        self.cvdata = []

        self.criterion = nn.BCELoss()
        self.convolution = ConvolutionLayer(hparams) if hparams.use_images else nn.Identity()
        self.fc = FullyConnected(hparams)
        self.distance = DistanceLayer(hparams) if hparams.use_distance else nn.Identity()
        self.log_model_parameters()

    def forward(self, iA, iB, rA, rB):
        if(self.use_images and (self.use_radiomics or self.use_clinical)):
            x = self.convolution(iA)
            y = self.convolution(iB)
            x.view(x.size(0), -1)
            y.view(y.size(0), -1)
            x = torch.cat((x, rA), dim=1)
            y = torch.cat((y, rB), dim=1)
        elif(self.use_radiomics or self.use_clinical):
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

        np_output = output.view(-1).detach()
        output_class = torch.where(np_output < 0.5,
                                   torch.tensor(0).type_as(np_output),
                                   torch.tensor(1).type_as(np_output))
        correct = torch.sum(output_class == labels).type_as(np_output)
        total = torch.tensor(np_output.size(0)).type_as(np_output)
        CI = correct/total

        tensorboard_logs = {'train_loss': loss, 'CI': CI}
        return {'loss': loss, 'custom_logs': tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['custom_logs']['train_loss'].mean() for x in outputs]).mean()
        CI = torch.stack([x['custom_logs']['CI'].mean() for x in outputs]).mean()

        #TODO: This does not work, as lightning does not update the
        # progress bar on training epoch end
        tensorboard_logs = {
            'avg_loss' : avg_loss,
            'train_CI' : CI}
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

        np_output = output.view(-1).detach()
        output_class = torch.where(np_output < 0.5,
                                   torch.tensor(0).type_as(np_output),
                                   torch.tensor(1).type_as(np_output))

        correct = torch.sum(output_class == labels).type_as(np_output)
        total = torch.tensor(np_output.size(0)).type_as(np_output)

        return {'val_loss' : loss, 'correct' : correct, 'total': total}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'].mean() for x in outputs]).mean()
        correct = torch.stack(([x['correct'].sum()  for x in outputs])).sum()
        total = torch.stack(([x['total'].sum() for x in outputs])).sum()

        tensorboard_logs = {'val_loss': avg_loss,
                            'val_CI' : correct/total}

        if(self.hparams.use_kfold):
            self.cvdata.append({
                'val_loss': avg_loss,
                'correct' : correct,
                'total': total
            })
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

        parser.add_argument('--use-distance', action='store_true', default=config.USE_DISTANCE)
        parser.add_argument('--d-layers', type=int, nargs='+', default=config.D_LAYERS)
        parser.add_argument('--d-dropout', type=float, nargs='+',
                            default = [])

        parser.add_argument('--use-images', action='store_true', default=config.USE_IMAGES)
        parser.add_argument('--conv-layers', type=int, nargs='+', default=[1,4,8,16])
        ## OPTIMIZER
        parser.add_argument('--learning-rate', type=float, default=config.LR)
        parser.add_argument('--momentum', type=float, default=config.MOMENTUM)
        parser.add_argument('--weight-decay', type=float, default=config.WEIGHT_DECAY)
        parser.add_argument('--sc-milestones', type=int, nargs='+',
                            default = config.SC_MILESTONES)
        parser.add_argument('--sc-gamma', type=float, default=config.SC_GAMMA)

        return parser

    def log_model_parameters(self):
        print("PARAMETERS**********************************************")
        print("Convolution layer parameters: %d" % (count_parameters(self.convolution)))
        print("FC layer parameters: %d" % (count_parameters(self.fc)))
        print("distance layer parameters: %d" % (count_parameters(self.distance)))
        print("*******************************************************")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
