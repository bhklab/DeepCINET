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
from pytorch_src.data.datareader import PairProcessor
from pytorch_src.data.dataloader import Dataset

from pytorch_lightning import Trainer

################
#     MAIN     #
################
def Create_Dataloader(ids, hparams):
        dataset = Dataset(ids, hparams.clinical_path, hparams.image_path, hparams.radiomics_path, hparams)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size = hparams.batch_size,
                                           shuffle = True,
                                           num_workers= hparams.num_workers)

def deepCinet(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(520)
    torch.manual_seed(520)

    pairProcessor = PairProcessor(args.clinical_path)
    hdict = vars(args)
    hparams = argparse.Namespace(**hdict)
    print(hparams)

    if(hparams.use_kfold):
        folds, test_ids = pairProcessor.k_cross_validation(
            n_splits = hparams.folds,
            test_ratio = hparams.test_ratio,
            random_seed = 520)
        test_dl = Create_Dataloader(test_ids, hparams)
        for train_ids, val_ids in folds:
            train_dl = Create_Dataloader(train_ids, hparams)
            val_dl = Create_Dataloader(val_ids, hparams)

            siamese_model = ImageSiamese(hparams=hparams)
            trainer = Trainer(min_epochs = args.epochs,
                              max_epochs = args.epochs,
                              gpus=1,
                              accumulate_grad_batches = 1,
                              weights_summary ='full',
                              enable_benchmark = False)
            trainer.fit(siamese_model,
                        train_dataloader = train_dl,
                        val_dataloaders = val_dl,
                        test_dataloaders = test_dl)
            print("")
            print("END FOLD")
    else:
        train_ids, val_ids, test_ids = pairProcessor.train_test_split(
            val_ratio = hparams.val_ratio,
            test_ratio = hparams.test_ratio,
            split_model = PairProcessor.TIME_SPLIT,
            random_seed = 520)
        train_dl = Create_Dataloader(train_ids, hparams)
        val_dl = Create_Dataloader(val_ids, hparams)
        test_dl = Create_Dataloader(test_ids, hparams)
        siamese_model = ImageSiamese(hparams=hparams)
        trainer = Trainer(min_epochs = args.epochs,
                          max_epochs=args.epochs,
                          gpus=1,
                          accumulate_grad_batches = 1,
                          weights_summary='full')
        trainer.fit(siamese_model,
                    train_dataloader = train_dl,
                    val_dataloaders = val_dl,
                    test_dataloaders = test_dl)

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

        ## DATALOADER ######
        data_arg = add_argument_group('Data')
        data_arg.add_argument('--clinical-path', type=str, default=config.CLINICAL_PATH, help='Path to clinical variables')
        data_arg.add_argument('--radiomics-path', type=str, default=config.RADIOMICS_PATH, help='Path to radiomics features')
        data_arg.add_argument('--use-radiomics', action='store_true', default=config.USE_RADIOMICS)
        data_arg.add_argument('--image-path', type=str, default=config.IMAGE_PATH, help='Path to patient CTs')
        data_arg.add_argument('--use-images', action='store_true', default=config.USE_IMAGES)
        data_arg.add_argument("--num-workers", default=config.NUM_WORKERS, type=int)
        data_arg.add_argument("--batch-size", default=config.BATCH_SIZE, type=int)
        data_arg.add_argument("--test-ratio", default=config.TEST_RATIO, type=float)
        data_arg.add_argument("--val-ratio", default=config.VAL_RATIO, type=float)

        data_arg.add_argument("--folds", default=config.FOLDS, type=int)
        data_arg.add_argument('--use-kfold', action='store_true', default=config.USE_FOLDS)
        ####################

        ## TRAINING ########
        parser.add_argument("--epochs", default=config.EPOCHS, type=int)
        ####################



        parser = ImageSiamese.add_model_specific_args(parser)
        args, unparsed = parser.parse_known_args()
        main(args)

    except KeyboardInterrupt:
        logger.info("\r----------------------------------")
        logger.info("\rStopping due to keyboard interrupt")
        logger.info("\rTHANKS FOR THE RIDE 😀")
