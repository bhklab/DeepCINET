import os

import torch
import torch.nn as nn

import pytorch_lightning as pl

from pytorch_src.models.conv_model import ConvolutionLayer
from pytorch_src.models.fc_model import FullyConnected

from pytorch_src.data.dataloader import RadiomicsData as Dataset

class RadiomicsModel(pl.LightningModule):
    def __init__(self, train_pairs, val_pairs, batch_size, image_path):
        super(RadiomicsModel, self).__init__()
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.batch_size = batch_size
        self.image_path = image_path

        self.criterion = nn.BCELoss()

        self.layer = nn.Sequential(
            nn.Linear(1671, 640),
            nn.LeakyReLU(),
            nn.Dropout(0.7),

            nn.Linear(640, 160),
            nn.LeakyReLU(),
            nn.Dropout(0.7),

            nn.Linear(160, 80),
            nn.LeakyReLU(),
            nn.Dropout(0.7),

            nn.Linear(80, 48),
            nn.Sigmoid()
        )

        self.distance = nn.Sequential(
            nn.Linear(48, 24, bias=False),
            nn.Tanh(),
            nn.Linear(24, 8, bias=False),
            nn.Tanh(),
            nn.Linear(8, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, pA, pB):
        x = self.layer(pA)
        y = self.layer(pB)
        z = torch.sub(x, y)
        z = self.distance(z)
        return z

    def training_step(self, batch, batch_idx):
        pA = batch[0].clone().detach()
        pB = batch[1].clone().detach()
        labels = batch[2].clone().detach()

        output = self(pA, pB)
        loss = self.criterion(output.view(-1), labels.view(-1))

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        pA = batch[0].clone().detach()
        pB = batch[1].clone().detach()
        labels = batch[2].clone().detach()

        output = self.forward(pA, pB)
        loss = self.criterion(output.view(-1), labels.view(-1))

        return {'val_loss' : loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tqdm_dict = {'val_loss': avg_loss}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': avg_loss}
        return result

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.parameters()},
        ], lr=5e-3, momentum=0.9)

    def uses_images(self):
        return True

    @pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(Dataset(self.train_pairs, self.image_path),
                                           batch_size = 8,
                                           shuffle = True,
                                           num_workers=8)

    @pl.data_loader
    def val_dataloader(self):
        return torch.utils.data.DataLoader(Dataset(self.val_pairs, self.image_path),
                                           batch_size = 8,
                                           shuffle = True,
                                           num_workers=8)
