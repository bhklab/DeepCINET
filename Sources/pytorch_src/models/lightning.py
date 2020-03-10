import os

import torch
import torch.nn as nn

import pytorch_lightning as pl

from pytorch_src.models.conv_model import ConvolutionLayer
from pytorch_src.models.fc_model import FullyConnected
from pytorch_src.models.distance_model import DistanceLayer

from pytorch_src.data.dataloader import Dataset

class ImageSiamese(pl.LightningModule):
    def __init__(self, train_pairs, val_pairs, batch_size, image_path, radiomics_path, clinical_path, use_images, use_radiomics):
        super(ImageSiamese, self).__init__()
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.batch_size = batch_size
        self.image_path = image_path
        self.radiomics_path = radiomics_path
        self.clinical_path = clinical_path
        self.use_images = use_images
        self.use_radiomics = use_radiomics

        self.criterion = nn.BCELoss()
        self.convolution = ConvolutionLayer()
        self.fc = FullyConnected()
        self.distance = DistanceLayer()

    def forward(self, iA, iB, rA, rB):
        if(self.use_images and self.use_radiomics):
            x = self.convolution(pA)
            y = self.convolution(pB)
            x.view(x.size(0), -1)
            y.view(y.size(0), -1)
            x = torch.cat((x, rA), dim=1)
            y = torch.cat((y, rB), dim=1)
        elif(self.use_radiomics):
            x = rA
            y = rB
        elif(self.use_images):
            x = self.convolution(pA)
            y = self.convolution(pB)
            x.view(x.size(0), -1)
            y.view(y.size(0), -1)
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

        return {'val_loss' : loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tqdm_dict = {'val_loss': avg_loss}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': avg_loss}
        return result

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)

    @pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(Dataset(self.train_pairs,
                                                   self.clinical_path,
                                                   self.image_path,
                                                   self.radiomics_path),
                                           batch_size = 4,
                                           shuffle = True,
                                           num_workers=8)

    @pl.data_loader
    def val_dataloader(self):
        return torch.utils.data.DataLoader(Dataset(self.val_pairs,
                                                   self.clinical_path,
                                                   self.image_path,
                                                   self.radiomics_path),
                                           batch_size = 4,
                                           shuffle = True,
                                           num_workers=8)



    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # network params
        parser.add_argument('--in_features', default=28 * 28, type=int)
        parser.add_argument('--out_features', default=10, type=int)
        # use 500 for CPU, 50000 for GPU to see speed difference
        parser.add_argument('--hidden_dim', default=50000, type=int)
        parser.add_argument('--drop_prob', default=0.2, type=float)
        parser.add_argument('--learning_rate', default=0.001, type=float)

        # data
        parser.add_argument('--data_root', default=os.path.join(root_dir, 'mnist'), type=str)

        # training params (opt)
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--batch_size', default=64, type=int)
        return parser
