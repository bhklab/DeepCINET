#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import pandas as pd

from pytorch_src.models.lightning import ImageSiamese
from pytorch_src.models.coxmodel import CoxModel
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
def Create_Dataloader(ids, hparams, is_train = False):
    dataset = Dataset(ids, hparams, is_train)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size = hparams.batch_size,
                                       shuffle = False,
                                       num_workers= hparams.num_workers)

def deepCinet(args):
    pairProcessor = PairProcessor(args.clinical_path)
    hdict = vars(args)
    hparams = argparse.Namespace(**hdict)
    print(hparams)

    if(hparams.use_kfold):
        folds, test_ids = pairProcessor.k_cross_validation(
            test_ratio = hparams.test_ratio,
            n_splits = hparams.folds,
            random_seed = hparams.seed)
        test_dl = Create_Dataloader(test_ids, hparams)

        cvdata = []
        for train_ids, val_ids in folds:
            train_dl = Create_Dataloader(train_ids, hparams, True)
            val_dl = Create_Dataloader(val_ids, hparams)

            siamese_model = ImageSiamese(hparams=hparams)
            trainer = Trainer(min_epochs = hparams.epochs,
                              max_epochs = hparams.epochs,
                              gpus=list(range(hparams.gpus)),
                              accumulate_grad_batches = 1,
                              distributed_backend='dp',
                              weights_summary ='full',
                              enable_benchmark = False,
			      auto_find_lr = hparams.auto_find_lr,
                              overfit_pct = hparams.overfit_pct)
            trainer.fit(siamese_model,
                        train_dataloader = train_dl,
                        val_dataloaders = val_dl)
            cvdata.append(siamese_model.cvdata)
        for i in range(hparams.epochs):
            avg_loss = torch.stack([x[i]['val_loss'] for x in cvdata]).mean().item()
            correct = torch.stack(([x[i]['correct'] for x in cvdata])).sum().item()
            total = torch.stack(([x[i]['total'] for x in cvdata])).sum().item()
            avg_ci = correct/total
            print("EPOCH %d -- Average loss: %.4f -- CI: %.4f"
				  % (i+1, avg_loss, avg_ci))
    else:
        train_ids, val_ids, test_ids = pairProcessor.train_test_split(
            val_ratio = hparams.val_ratio,
            test_ratio = hparams.test_ratio,
            random_seed = hparams.seed,
            split_model = PairProcessor.TIME_SPLIT)
        train_dl = Create_Dataloader(train_ids, hparams, True)
        val_dl = Create_Dataloader(val_ids, hparams)
        test_dl = Create_Dataloader(test_ids, hparams)
        siamese_model = ImageSiamese(hparams=hparams)
        trainer = Trainer(min_epochs = args.epochs,
                          max_epochs=args.epochs,
                          gpus=list(range(hparams.gpus)),
                          accumulate_grad_batches = 1,
                          distributed_backend='dp',
                          weights_summary='full',
                          enable_benchmark = False,
                          overfit_pct = hparams.overfit_pct)
        trainer.fit(siamese_model,
                    train_dataloader = train_dl,
                    val_dataloaders = val_dl)

def cox(hparams):
    pairProcessor = PairProcessor(args.clinical_path)
    folds, test_ids = pairProcessor.k_cross_validation(
        test_ratio = hparams.test_ratio,
        n_splits = hparams.folds,
        random_seed = hparams.seed)
    cvdata = []
    for train_ids, val_ids in folds:
        cox = CoxModel(hparams)
        data = cox.fit(train_ids, val_ids)
        cvdata.append(data)
        print(data)
        avg_ci = np.mean([x for x in cvdata])
        print("CI: %.4f" % (avg_ci))

def main(args) -> None:
    """
        Main function
        :param args: Command Line Arguments
        """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if(args.use_cox):
        cox(args)
    else:
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
        data_arg.add_argument('--volume-only', action='store_true', default=False)

        data_arg.add_argument('--image-path', type=str, default=config.IMAGE_PATH, help='Path to patient CTs')

        data_arg.add_argument("--num-workers", default=config.NUM_WORKERS, type=int)
        data_arg.add_argument("--batch-size", default=config.BATCH_SIZE, type=int)
        data_arg.add_argument("--test-ratio", default=config.TEST_RATIO, type=float)
        data_arg.add_argument("--val-ratio", default=config.VAL_RATIO, type=float)

        data_arg.add_argument("--folds", default=config.FOLDS, type=int)
        data_arg.add_argument('--use-kfold', action='store_true', default=config.USE_FOLDS)
        data_arg.add_argument('--transitive-pairs', default=-1, type=int)
        ####################

        data_arg.add_argument('--use-clinical', action='store_true', default=False)
        data_arg.add_argument('--use-cox', action='store_true', default=False)

        ## TRAINING ########
        parser.add_argument("--epochs", default=config.EPOCHS, type=int)
        data_arg.add_argument('--auto-find-lr', action='store_true', default=False)
        parser.add_argument("--gpus", default=config.EPOCHS, type=int)
        ####################

        ## DEBUG ###########
        parser.add_argument("--overfit-pct", default=0, type=float)
        ####################

        ## MISC ############
        data_arg.add_argument("--seed", default=520, type=int)
        ####################



        parser = ImageSiamese.add_model_specific_args(parser)
        args, unparsed = parser.parse_known_args()
        main(args)

    except KeyboardInterrupt:
        logger.info("\r----------------------------------")
        logger.info("\rStopping due to keyboard interrupt")
        logger.info("\rTHANKS FOR THE RIDE 😀")
