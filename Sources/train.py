#!/usr/bin/env python3
import argparse
import sys
import os

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback


import default_settings as config
from models.lightning import DeepCINET
from data.fold_generator import KFoldGenerator
from data.data_loader import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


def Create_Dataloader(ds, hparams, shuffle_ind):
    return torch.utils.data.DataLoader(ds,
                                       batch_size=hparams.batch_size,
                                       shuffle=shuffle_ind,
                                       num_workers=hparams.num_workers)


def deepCinet():
    hdict = vars(args)
    hparams = argparse.Namespace(**hdict)
    print(hparams)

    config = {
        "layers_size": [899, 16, 32, 64, 32, 32, 1],
        "dropout": [0, 0, 0, 0, 0, 0],
        "lr": 0.01,
        "batchnorm": False
    }

    pairProcessor = KFoldGenerator(hparams)
    folds = pairProcessor.k_cross_validation(
        n_splits=hparams.folds,
        random_seed=hparams.seed, stratified=False)
    cvdata = []
    for train_ids, val_ids in folds:
        train_dl = Create_Dataloader(
            Dataset(hparams, True, train_ids),
            hparams, shuffle_ind=True)
        val_dl = Create_Dataloader(
            Dataset(hparams, True, val_ids), # is_train = true here to get pairs
            hparams, shuffle_ind=True)

        siamese_model = DeepCINET(hparams=hparams, config=config)
        trainer = Trainer(min_epochs=hparams.min_epochs,
                          max_epochs=hparams.max_epochs,
                          min_steps=hparams.min_steps,
                          max_steps=hparams.max_steps,
                          gpus=1,
                          accumulate_grad_batches=hparams.accumulate_grad_batches,
                          distributed_backend='dp',
                          weights_summary='full',
                          # enable_benchmark=False,
                          num_sanity_val_steps=0,
                          # auto_find_lr=hparams.auto_find_lr,
                          # callbacks=[EarlyStopping(monitor='val_loss', patience=5, mode="min")],
                          check_val_every_n_epoch=hparams.check_val_every_n_epoch)
                          # overfit_pct=hparams.overfit_pct)
        trainer.fit(siamese_model,
                    train_dataloader=train_dl,
                    val_dataloaders=val_dl)

        for i in range(len(siamese_model.cvdata)):
            if len(cvdata) < i + 1:
                cvdata.append([])
            cvdata[i].append(siamese_model.cvdata[i])
    
    for i in range(len(cvdata)):
        avg_ci = float(np.mean([x['CI'] for x in cvdata[i]]))
        print("EPOCH %d -- CI: %.4f"
              % (min((i + 1) * hparams.check_val_every_n_epoch, hparams.max_epochs),
                 avg_ci))


def deepCinet_tune(config):
    hdict = vars(args)
    hparams = argparse.Namespace(**hdict)
    print(hparams)

    pairProcessor = KFoldGenerator(hparams)
    folds = pairProcessor.k_cross_validation(
        n_splits=hparams.folds,
        random_seed=hparams.seed, stratified=False)
    cvdata = []
    for train_ids, val_ids in folds:
        train_dl = Create_Dataloader(
            Dataset(hparams, True, train_ids),
            hparams, shuffle_ind=True)
        val_dl = Create_Dataloader(
            Dataset(hparams, True, val_ids), # is_train = true here to get pairs
            hparams, shuffle_ind=True)

        siamese_model = DeepCINET(hparams=hparams, config=config)
        trainer = Trainer(min_epochs=hparams.min_epochs,
                          max_epochs=hparams.max_epochs,
                          min_steps=hparams.min_steps,
                          max_steps=hparams.max_steps,
                          gpus=1,
                          accumulate_grad_batches=hparams.accumulate_grad_batches,
                          distributed_backend='dp',
                          weights_summary='full',
                          # enable_benchmark=False,
                          num_sanity_val_steps=0,
                          # auto_find_lr=hparams.auto_find_lr,
                          callbacks=[EarlyStopping(monitor='val_ci', patience=5, mode="max"),
                                     TuneReportCallback({
                                        "loss": "val_loss",
                                        "mean_accuracy": "val_ci"
                                     }, on="validation_end")],
                          check_val_every_n_epoch=hparams.check_val_every_n_epoch)
                          # overfit_pct=hparams.overfit_pct)
        trainer.fit(siamese_model,
                    train_dataloader=train_dl,
                    val_dataloaders=val_dl)

def tune_DeepCINET_asha(num_samples=10, num_epochs=10):
    config = {
        "layers_size": [899, 32, 32, 1],
        "dropout": [0, 0.2, 0.2, 0],
        "lr": tune.choice([0.01, 0.001]),
        "batchnorm": tune.choice([True, False])
    }
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["layers_size", "dropout", "lr", "batchnorm"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            deepCinet_tune),
        resources_per_trial={
            "cpu": 0,
            "gpu": 1
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_DeepCINET_asha")

    print("Best hyperparameters found were: ", analysis.best_config)


def main() -> None:
    """ Main function
    """
    # Making sure our results are deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    deepCinet()
    # tune_DeepCINET_asha()


arg_lists = []


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## DATALOADER ######
    data_arg = add_argument_group('Data')
    data_arg.add_argument('--gene-path', type=str,
                          default=config.GENE_PATH,
                          help='Path to clinical variables')
    # data_arg.add_argument('--use-clinical',
    #                       action='store_true',
    #                       default=config.USE_RADIOMICS)

    # data_arg.add_argument('--radiomics-path',
    #                       type=str,
    #                       default=config.RADIOMICS_PATH,
    #                       help='Path to radiomics features')
    # data_arg.add_argument('--use-radiomics',
    #                       action='store_true',
    #                       default=config.USE_RADIOMICS)

    # data_arg.add_argument('--image-path',
    #                       type=str,
    #                       default=config.IMAGE_PATH,
    #                       help='Path to patient CTs')

    data_arg.add_argument("--num-workers",
                          default=config.NUM_WORKERS,
                          type=int)
    data_arg.add_argument("--batch-size",
                          default=config.BATCH_SIZE,
                          type=int)
    data_arg.add_argument("--val-ratio",
                          default=config.VAL_RATIO,
                          type=float)

    data_arg.add_argument("--folds",
                          default=config.FOLDS,
                          type=int)
    # data_arg.add_argument('--transitive-pairs',
    #                       default=-1,
    #                       type=int)
    data_arg.add_argument('--use-volume-cache',
                          action='store_true')
    data_arg.add_argument('--accumulate-grad-batches',
                          default=1,
                          type=int)
    ####################

    ## TRAINING ########
    train_arg = add_argument_group('Train')
    train_arg.add_argument("--min-epochs", default=0, type=int)
    train_arg.add_argument("--min-steps", default=None, type=int)
    train_arg.add_argument("--max-epochs", default=100, type=int)
    train_arg.add_argument("--max-steps", default=None, type=int)
    train_arg.add_argument("--check-val-every-n-epoch", default=1, type=int)
    train_arg.add_argument('--auto-find-lr', action='store_true', default=False)
    train_arg.add_argument("--gpus", default=config.EPOCHS, type=int)
    ####################

    ## DEBUG ###########
    debug_arg = add_argument_group('Debug')
    debug_arg.add_argument("--overfit-pct", default=0, type=float)
    ####################

    ## MISC ############
    misc_arg = add_argument_group('Misc')
    misc_arg.add_argument("--seed", default=520, type=int)
    ####################

    parser = DeepCINET.add_model_specific_args(parser)
    args, unparsed = parser.parse_known_args()
    main()
