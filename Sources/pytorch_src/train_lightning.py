#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import pandas as pd

from pytorch_src.models.lightning import ImageSiamese
import numpy as np

import torch
import torch.nn as nn

import pytorch_src.default_settings as config

from pytorch_lightning import Trainer

################
#     MAIN     #
################
def deepCinet(args):
    hdict = vars(args)
    hdict['use_images'] = config.USE_IMAGES
    hdict['use_radiomics'] = config.USE_RADIOMICS
    hparams = argparse.Namespace(**hdict)
    siamese_model = ImageSiamese(hparams=hparams)
    trainer = Trainer(min_epochs = args.epochs, max_epochs=args.epochs, gpus=1, accumulate_grad_batches = 1)
    trainer.fit(siamese_model)

def main(args) -> None:
    """
    Main function
    :param args: Command Line Arguments
    """
    deepCinet(args)

arg_lists = []
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()

        # Dataloader
        data_arg = add_argument_group('Data')
        data_arg.add_argument('--clinical-path', type=str, default=config.CLINICAL_PATH, help='Path to clinical variables')
        data_arg.add_argument('--radiomics-path', type=str, default=config.RADIOMICS_PATH, help='Path to radiomics features')
        data_arg.add_argument('--image-path', type=str, default=config.IMAGE_PATH, help='Path to patient CTs')

        parser = ImageSiamese.add_model_specific_args(parser)
        args, unparsed = parser.parse_known_args()
        main(args)

    except KeyboardInterrupt:
        logger.info("\r----------------------------------")
        logger.info("\rStopping due to keyboard interrupt")
        logger.info("\rTHANKS FOR THE RIDE 😀")
