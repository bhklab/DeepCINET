#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import pandas as pd

from models.lightning import ImageSiamese
from models.coxmodel import CoxModel
import numpy as np

import torch
import torch.nn as nn

import default_settings as config
from data.datareader import PairProcessor
from data.dataloader import Dataset

from pytorch_lightning import Trainer

################
#     MAIN     #
################
def Create_Dataloader(ds, hparams):
    return torch.utils.data.DataLoader(ds,
                                       batch_size = hparams.batch_size,
                                       shuffle = True,
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
        cvdata = []
        mrmr_features = None
        for train_ids, val_ids in folds:
            train_ds = Dataset(train_ids, hparams, True)
            val_ds = Dataset(val_ids, hparams, False)
            if(hparams.mrmr > 0):
                if(mrmr_features == None):
                    mrmr_features = train_ds.mrmr()
                train_ds.apply_mrmr(mrmr_features)
                val_ds.apply_mrmr(mrmr_features)
            train_dl = Create_Dataloader(train_ds, hparams)
            val_dl = Create_Dataloader(val_ds, hparams)

            siamese_model = ImageSiamese(hparams=hparams)
            trainer = Trainer(min_epochs = hparams.min_epochs,
                              max_epochs = hparams.max_epochs,
                              min_steps = hparams.min_steps,
                              max_steps = hparams.max_steps,
                              gpus=list(range(hparams.gpus)),
                              accumulate_grad_batches = 1,
                              distributed_backend='dp',
                              weights_summary ='full',
                              enable_benchmark = False,
                              num_sanity_val_steps = 0,
			                  auto_find_lr = hparams.auto_find_lr,
                              check_val_every_n_epoch = hparams.check_val_every_n_epoch,
                              overfit_pct = hparams.overfit_pct)
            trainer.fit(siamese_model,
                        train_dataloader = train_dl,
                        val_dataloaders = val_dl)
            for i in range(len(siamese_model.cvdata)):
                if(len(cvdata) < i + 1):
                    cvdata.append([])
                cvdata[i].append(siamese_model.cvdata[i])
        paired_data = []
        print(cvdata)
        for i in range(len(cvdata)):
            steps = np.mean([x['t_steps'] for x in cvdata[i]])
            avg_ci = np.mean([x['CI'] for x in cvdata[i]])
            print("EPOCH %d -- Steps: %.4f -- CI: %.4f"
				  % (min((i+1)*hparams.check_val_every_n_epoch, hparams.max_epochs),
                     steps,
                     avg_ci))
            paired_data.append({'steps' : steps, 'CI' : avg_ci})
        for p in paired_data:
            print("%.5f, %.5f" % (p['steps'], p['CI']))
    else:
        train_ids, val_ids, test_ids = pairProcessor.train_test_split(
            val_ratio = hparams.val_ratio,
            test_ratio = hparams.test_ratio,
            random_seed = hparams.seed,
            split_model = PairProcessor.TIME_SPLIT)
        train_ds = Dataset(train_ids, hparams, True)
        val_ds = Dataset(val_ids, hparams, False)
        if(mrmr_features == None):
            mrmr_features = train_ds.mrmr()
        train_ds.apply_mrmr(mrmr_features)
        val_ds.apply_mrmr(mrmr_features)
        train_dl = Create_Dataloader(train_ds, hparams)
        val_dl = Create_Dataloader(val_ds, hparams)

        siamese_model = ImageSiamese(hparams=hparams)
        trainer = Trainer(min_epochs = args.min_epochs,
                          max_epochs=args.max_epochs,
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
    if(hparams.volume_only):
        cox = CoxModel(hparams)
        train_ids, val_ids, test_ids = pairProcessor.train_test_split(
            val_ratio = hparams.val_ratio,
            test_ratio = hparams.test_ratio,
            random_seed = hparams.seed,
            split_model = PairProcessor.TIME_SPLIT)
        cox.volume_only(train_ids + val_ids)
    else:
        folds, test_ids = pairProcessor.k_cross_validation(
            test_ratio = hparams.test_ratio,
            n_splits = hparams.folds,
            random_seed = hparams.seed)
        cvdata = []
        for train_ids, val_ids in folds:
            cox = CoxModel(hparams)
            data = cox.fit(train_ids, val_ids)
            cvdata.append(data)
            print(data, flush = True)
        avg_ci = np.mean([x for x in cvdata])
        print("CI: %.6f" % (avg_ci), flush = True)
        print(cvdata)

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
        data_arg.add_argument('--mrmr', default = -1, type=int)
        ####################

        data_arg.add_argument('--use-clinical', action='store_true', default=False)
        data_arg.add_argument('--use-cox', action='store_true', default=False)

        ## TRAINING ########
        parser.add_argument("--min-epochs", default=0, type=int)
        parser.add_argument("--min-steps", default=None, type=int)
        parser.add_argument("--max-epochs", default=1000, type=int)
        parser.add_argument("--max-steps", default=None, type=int)
        parser.add_argument("--check-val-every-n-epoch", default=None, type=int)
        parser.add_argument('--auto-find-lr', action='store_true', default=False)
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
