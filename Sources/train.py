#!/usr/bin/env python3
import argparse
import sys
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining, HyperBandForBOHB
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
        "hidden_one": 128,
        "hidden_two": 512,
        "hidden_three": 128,
        "hidden_four": 128,
        "dropout": 0.21081599189621927,
        "lr": 0.002224775178989976,
        "batchnorm": False
    }
    gene_data = Dataset(hparams, False)
    train_idx, val_idx = train_test_split(list(range(gene_data.__len__())), test_size=0.2)

    for delta in np.arange(0, 0.31, 0.01):
        train_dl = Create_Dataloader(
            Dataset(hparams, True, delta, train_idx),
            hparams, shuffle_ind=True)
        val_dl = Create_Dataloader(
            Dataset(hparams, True, delta, val_idx), # is_train = true here to get pairs
            hparams, shuffle_ind=True)

        # cvdata = []
        # best_loss = []
        filename_log = f'Gemcitabine-delta={delta:.2f}'
        checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./Saved_models/Gemcitabine/microarray/',
        filename=filename_log,
        save_top_k=1,
        mode='min'
        )

        siamese_model = DeepCINET(hparams=hparams, config=config)
        trainer = Trainer(min_epochs=hparams.min_epochs,
                            max_epochs=hparams.max_epochs,
                            min_steps=hparams.min_steps,
                            max_steps=hparams.max_steps,
                            gpus=2,
                            accumulate_grad_batches=hparams.accumulate_grad_batches,
                            distributed_backend='dp',
                            weights_summary='full',
                            # enable_benchmark=False,
                            num_sanity_val_steps=0,
                            # auto_find_lr=hparams.auto_find_lr,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode="min"),
                                    checkpoint_callback],
                            check_val_every_n_epoch=hparams.check_val_every_n_epoch)
                            # overfit_pct=hparams.overfit_pct)
        trainer.fit(siamese_model,
                    train_dataloader=train_dl,
                    val_dataloaders=val_dl)

    # for i in range(len(siamese_model.cvdata)):
    #     if len(cvdata) < i + 1:
    #         cvdata.append([])
    #     cvdata[i].append(siamese_model.cvdata[i])
    # best_loss.append(siamese_model.best_val_loss)

    # for i in range(len(cvdata)):
    #     avg_ci = float(np.mean([x['CI'] for x in cvdata[i]]))
    #     print("EPOCH %d -- CI: %.4f"
    #           % (min((i + 1) * hparams.check_val_every_n_epoch, hparams.max_epochs),
    #              avg_ci))
    # for i in range(len(best_loss)):
    #     print("CV %d -- best validation loss: %.4f" %(i, best_loss[i]))


def deepCinet_tune(config):
    hdict = vars(args)
    hparams = argparse.Namespace(**hdict)
    print(hparams)

    gene_data = Dataset(hparams, False)
    train_idx, val_idx = train_test_split(list(range(gene_data.__len__())), test_size=0.2)

    train_dl = Create_Dataloader(
        Dataset(hparams, True, train_idx),
        hparams, shuffle_ind=True)
    val_dl = Create_Dataloader(
        Dataset(hparams, True, val_idx), # is_train = true here to get pairs
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
                        callbacks=[#EarlyStopping(monitor='val_loss', patience=5, mode="min"),
                                    TuneReportCallback({
                                    "best_loss": "best_loss",
                                    "CI": "best_val_ci"
                                    }, on="validation_end")],
                        check_val_every_n_epoch=hparams.check_val_every_n_epoch)
                        # overfit_pct=hparams.overfit_pct)
    trainer.fit(siamese_model,
                train_dataloader=train_dl,
                val_dataloaders=val_dl)

# def layer2():
#     layers_2 = []
#     sizes = [32, 128, 512]
#     for i in sizes:
#         one_layer = [899, i]
#         for j in sizes:
#             two_layer = one_layer + [j, 1]
#             layers_2.append(two_layer)
#     return layers_2

# def layer3():
#     layers_3 = []
#     sizes = [32, 128, 512]
#     for i in sizes:
#         one_layer = [899, i]
#         for j in sizes:
#             two_layer = one_layer + [j]
#             for k in sizes:
#                 three_layer = two_layer + [k, 1]
#                 layers_3.append(three_layer)
#     return layers_3

# def layer4():
#     layers_4 = []
#     sizes = [32, 128, 512]
#     for i in sizes:
#         one_layer = [899, i]
#         for j in sizes:
#             two_layer = one_layer + [j]
#             for k in sizes:
#                 three_layer = two_layer + [k]
#                 for l in sizes:
#                     four_layer = three_layer + [l, 1]
#                     layers_4.append(four_layer)
#     return layers_4

# def layer_sizes():
#     return layer2() + layer3() + layer4()

# def tune_DeepCINET_asha(num_samples=1, num_epochs=10):

#     config = {
#         "layers_size": tune.grid_search(layer_sizes()),
#         "dropout": tune.grid_search([0, 0.2, 0.4]),
#         "lr": tune.grid_search([0.01, 0.001]),
#         "batchnorm": tune.grid_search([True, False])
#     }
#     scheduler = ASHAScheduler(
#         max_t=num_epochs,
#         grace_period=1,
#         reduction_factor=2)

#     reporter = CLIReporter(
#         parameter_columns=["layers_size", "dropout", "lr", "batchnorm"],
#         metric_columns=["loss", "mean_accuracy", "training_iteration"])

#     analysis = tune.run(
#         tune.with_parameters(
#             deepCinet_tune),
#         resources_per_trial={
#             "cpu": 0,
#             "gpu": 2
#         },
#         metric="loss",
#         mode="min",
#         config=config,
#         num_samples=num_samples,
#         scheduler=scheduler,
#         progress_reporter=reporter,
#         name="tune_DeepCINET_asha",
#         local_dir="../ray_results")

#     print("Best hyperparameters found were: ", analysis.best_config)

def tune_DeepCINET_bohb(num_samples=1000):
    import ConfigSpace as CS
    # import ConfigSpace.hyperparameters as CSH

    # config_space = CS.ConfigurationSpace()
    # hidden_one = CSH.CategoricalHyperparameter('hidden_one', choices=[32, 128, 512])
    # hidden_two = CSH.CategoricalHyperparameter('hidden_two', choices=[32, 128, 512])
    # hidden_three = CSH.CategoricalHyperparameter('hidden_three', choices=[0, 32, 128, 512])
    # hidden_four = CSH.CategoricalHyperparameter('hidden_four', choices=[0, 32, 128, 512], default_value=0)
    # dropout = CSH.CategoricalHyperparameter('dropout', choices=[0, 0.2, 0.4])
    # lr = CSH.CategoricalHyperparameter('lr', choices=[0.01, 0.001])
    # batchnorm = CSH.CategoricalHyperparameter('batchnorm', choices=[1, 0])
    # config_space.add_hyperparameters([hidden_one, hidden_two, hidden_three, hidden_four, dropout, lr, batchnorm])
    # cond = CS.NotEqualsCondition(hidden_four, hidden_three, 0)
    # config_space.add_condition(cond)
    config = {
        "hidden_one": tune.choice([32, 128, 512]),
        "hidden_two": tune.choice([32, 128, 512]),
        "hidden_three": tune.choice([0, 32, 128, 512]),
        "hidden_four": tune.choice([0, 32, 128, 512]),
        "dropout": tune.uniform(0, 0.4),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batchnorm": tune.choice([True, False])
    }

    algo = TuneBOHB(# space=config_space, 
                    max_concurrent=20)
    bohb = HyperBandForBOHB(time_attr="training_iteration")
    
    reporter = CLIReporter(
        parameter_columns=["hidden_one", "hidden_two", "hidden_three", "hidden_four", "dropout", "lr", "batchnorm"],
        metric_columns=["best_loss", "CI", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            deepCinet_tune),
        resources_per_trial={
            "cpu": 0,
            "gpu": 0.1
        },
        metric="best_loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=bohb,
        search_alg = algo,
        progress_reporter=reporter,
        name="tune_DeepCINET_bohb_dasatinib_best_loss",
        local_dir="../ray_results")

    print("Best hyperparameters found were: ", analysis.best_config)
    print ("Results were: ", analysis.best_result)


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
    # tune_DeepCINET_bohb()



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
    # data_arg.add_argument("--val-ratio",
    #                       default=config.VAL_RATIO,
    #                       type=float)

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
    train_arg.add_argument("--gpus", default=0, type=int)
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
