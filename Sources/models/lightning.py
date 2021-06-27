import argparse

import torch
import torch.nn as nn

import numpy as np

import pytorch_lightning as pl

from models.fc_model import FullyConnected

import default_settings as config
import sys

from lifelines.utils import concordance_index


class DeepCINET(pl.LightningModule):
    """ Base class for our DeepCINET implemented in pytorch lightning

    Provides methods to train and validate as well as configuring the optimizer
    scheduler.
    """

    def __init__(self, hparams):
        super(DeepCINET, self).__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)
        # for key in hparams.keys():
        #     self.hparams[key]=hparams[key]
        self.t_steps = 0
        self.cvdata = []
        self.criterion = nn.BCELoss()
        self.convolution = nn.Identity()
        self.fc = FullyConnected(hparams)
        self.log_model_parameters()

    def forward(self, geneA, geneB):
        tA = self.fc(geneA)
        tB = self.fc(geneB)
        z = (tA - tB)
        return torch.sigmoid(z)

    def computeEnergy(self, volume, scalar_features):
        """ Calculates the 'energy' of an input
        :param volume: 3D volume of the CT scan
            empty tensor if use_images is not set
        :param scalar_features: scalar features of our neural net
            empty tewnsor if use_clinical or use_radiomics is not set
        :return: returns a single scalar tA, the energy of our a sample
        """
        tA = self.convolution(volume).view(volume.size(0), -1)
        tA = torch.cat((tA, scalar_features), dim=1)
        tA = self.fc(tA)

        if self.hparams.use_exp:
            tA = torch.exp(tA).sum(dim=1)
        return tA

    def on_epoch_start(self):
        print("")
        print("EPOCH START")

    def training_step(self, batch, batch_idx):
        geneA = batch['geneA']
        geneB = batch['geneB']
        labels = batch['labels']

        output = self.forward(geneA, geneB)
        loss = self.criterion(output.view(-1), labels)
        # loggin number of steps
        self.t_steps += 1

        np_output = output.view(-1).detach()
        output_class = torch.where(np_output < 0.5,
                                   torch.tensor(0).type_as(np_output),
                                   torch.tensor(1).type_as(np_output))
        correct = torch.sum(output_class == labels).type_as(np_output)
        total = torch.tensor(np_output.size(0)).type_as(np_output)
        CI = correct / total

        tensorboard_logs = {'train_loss': loss, 'CI': CI}
        return {'loss': loss, 'custom_logs': tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['custom_logs']['train_loss'].mean() for x in outputs]).mean()
        CI = torch.stack([x['custom_logs']['CI'].mean() for x in outputs]).mean()

        # TODO: This does not work, as lightning does not update the
        # progress bar on training epoch end
        tensorboard_logs = {
            'avg_loss': avg_loss,
            'train_CI': CI}
        self.log_dict(tensorboard_logs, prog_bar = True)
        # return {'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        gene = batch['gene']
        drug_response = batch['response'].detach().cpu().numpy()
        # print("", file=sys.stderr)
        # print("val size", file=sys.stderr)
        # print(volume.size(0), file=sys.stderr)
        # print(torch.cuda.current_device(), file=sys.stderr)
        # print("", file=sys.stderr)
        output = self.fc(gene).view(-1).detach().cpu().numpy()
        # print(drug_response)
        # print(output.detach().cpu().numpy(), file=sys.stderr)

        # TODO: Pytorch currently doesn't reduce the output in validation when
        #       we use more than one GPU, becareful this might not be supported
        #       future versions
        return {'Drug_response': drug_response, 'Drug_response_pred': output}

    def validation_epoch_end(self, outputs):
        drug_response = np.concatenate([x['Drug_response'] for x in outputs])
        drug_response_pred = np.concatenate([x['Drug_response_pred'] for x in outputs])
        ## Have samples been averaged out??
        # print("", file=sys.stderr)
        # print("Total size", file=sys.stderr)
        # print(events.shape, file=sys.stderr)
        # print("", file=sys.stderr)
        # print(energies, file=sys.stderr)

        ci = concordance_index(drug_response, drug_response_pred)
        print("Validation CI: " + str(round(ci, 3)))
        # print(ci)
        tensorboard_logs = {'val_CI': ci}

        self.cvdata.append({
            'CI': ci,
            't_steps': self.t_steps
        })
        self.log('val_CI', ci)
        return {'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                    lr=self.hparams.learning_rate,
                                    #momentum=self.hparams.momentum,
                                    weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.hparams.sc_milestones,
            gamma=self.hparams.sc_gamma)

        return [optimizer], [scheduler]

    def log_model_parameters(self):
        print("PARAMETERS**********************************************")
        print("Convolution layer parameters: %d" % (count_parameters(self.convolution)))
        print("FC layer parameters: %d" % (count_parameters(self.fc)))
        print("********************************************************")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        ## NETWORK
        parser.add_argument('--fc-layers', type=int, nargs='+',
                            default=config.FC_LAYERS)
        parser.add_argument('--dropout', type=float, nargs='+',
                            default=config.DROPOUT)

        # parser.add_argument('--use-distance', action='store_true', default=config.USE_DISTANCE)
        parser.add_argument('--d-layers', type=int, nargs='+', default=config.D_LAYERS)
        parser.add_argument('--d-dropout', type=float, nargs='+',
                            default=[])

        # parser.add_argument('--use-images', action='store_true', default=config.USE_IMAGES)
        # parser.add_argument('--conv-layers', type=int, nargs='+', default=[1, 4, 8, 16])
        # parser.add_argument('--conv-model', type=str, default="Bottleneck")
        # parser.add_argument('--pool', type=int, nargs='+', default=[1, 1, 1, 1])
        ## OPTIMIZER
        parser.add_argument('--learning-rate', type=float, default=config.LR)
        parser.add_argument('--momentum', type=float, default=config.MOMENTUM)
        parser.add_argument('--weight-decay', type=float, default=config.WEIGHT_DECAY)
        parser.add_argument('--sc-milestones', type=int, nargs='+',
                            default=config.SC_MILESTONES)
        parser.add_argument('--sc-gamma', type=float, default=config.SC_GAMMA)
        # parser.add_argument('--use-exp', action='store_true', default=config.USE_IMAGES)
        return parser


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
