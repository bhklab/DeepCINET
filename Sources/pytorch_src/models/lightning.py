import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl

class ImageSiamese(pl.LightningModule):

    def __init__(self, pairs, batch_size):
        super(ImageSiamese, self).__init__()
        self.pairs = pairs
        self.batch_size = batch_size
        self.criterion = nn.BCELoss()
        self.convolution = ConvolutionLayer()
        self.fc = FullyConnected()

    def forward(self, x, pA, pB):
        x = self.fc(self.convolution(x))
        xA = torch.index_select(x, 0, pA)
        xB = torch.index_select(x, 0, pB)
        z = torch.sub(xA, xB)
        return torch.sigmoid(z)

    def training_step(self, batch, batch_idx):
        labels = torch.tensor(batch.pairs['labels'].values, device = device, dtype=torch.float32)
        pA = torch.tensor(batch.pairs['pA_id'].values, device = device)
        pB = torch.tensor(batch.pairs['pB_id'].values, device = device)
        tensors = torch.tensor(batch.patients['images'], device = device, dtype=torch.float32)
        tensors = tensors.permute(0,4,1,2,3)

        output = self.forward(tensors, pA, pB)
        loss = self.criterion(output.view(-1), labels)

        tensorboard_logs = {'train_loss': loss}
        if(batch_idx % 10 == 9):
            print(loss.item(), flush=True)
        return {'loss': loss, 'log': tensorboard_logs}

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     return {'val_loss': F.cross_entropy(y_hat, y)}

    # def validation_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     return {'test_loss': F.cross_entropy(y_hat, y)}

    # def test_end(self, outputs):
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'test_loss': avg_loss}
    #     return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.SGD(self.parameters(), lr=0.001)

    def uses_images(self):
        return True

    @pl.data_loader
    def train_dataloader(self):
        return batch_data.batches(self.pairs,
                                  batch_size=self.batch_size,
                                  load_images=self.uses_images(),
                                  train=True)

    # @pl.data_loader
    # def val_dataloader(self):
        # return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    # @pl.data_loader
    # def test_dataloader(self):
    #     return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=32)
