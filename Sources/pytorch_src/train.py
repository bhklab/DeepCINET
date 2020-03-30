#!/usr/bin/env python3
"""
The train_test_models module is a script that trains a deep learning model from medical imaging data, radiomic features and any other features.
There is a function,  deepCinet, as well .


usage: train_test_models.py --model MODEL
                [-h] [--cv-folds CV_FOLDS]
                [--test-size TEST_SIZE] [--gpu-level GPU_LEVEL]
                [--gpu-allow-growth] [-n NUM_EPOCHS] [--batch-size BATCH_SIZE]
                [--results-path RESULTS_PATH] [--learning-rate LEARNING_RATE]
                [--regularization REGULARIZATION] [--dropout {0.0 - 1.0}]
                [--log-device] [--use-distance] [--random-labels]
                [--full-summary][--mrmr-size][--full-summary]
                [--read_splits][--save-model][--bin-number]

Fit the data with a Tensorflow model

required named arguments:
  --model MODEL         Choose the model that you want to use for training.
                        The models that can be used are the following ones:

                          - :any:`SimpleImageSiamese`
                          - :any:`ImageScalarSiamese`
                          - :any:`ScalarOnlySiamese`
                          - :any:`ScalarOnlyDropoutSiamese`
                          - :any:`ImageSiamese`
                          - :any:`ResidualImageScalarSiamese`
                          - :any:`VolumeOnlySiamese`
                          - :any:`ClinicalOnlySiamese`
                          - :any:`ClinicalVolumeSiamese`
                          - :any:`ClinicalRadiomicSiamese`

optional named arguments:
  -h, --help            Show this help
  --cv-folds CV_FOLDS   Number of cross validation folds. If 0 < folds < 2 CV
                        won't be used and the test set size will be defined by
                        --test-size. If folds < 0 Leave One Out Cross
                        Validation will be used instead (default: 1)
  --test-size TEST_SIZE
                        Size of the test set as a float between 0 and 1
                        (default: 0.25)
  --gpu-level GPU_LEVEL
                        Amount of GPU resources used when fitting the model.
                        0: no GPU usage, 1: only second conv layers, 2: all
                        conv layers, 3: all layers and parameters are on the
                        GPU (default: 0)
  --gpu-allow-growth    Allow Tensorflow to use dynamic allocations with GPU
                        memory (default: False)
  -n NUM_EPOCHS, --num-epochs NUM_EPOCHS
                        Number of epochs to use when training. Times passed
                        through the entire dataset (default: 1)
  --batch-size BATCH_SIZE
                        Batch size for each train iteration (default: 20)
  --results-path RESULTS_PATH
                        Path where the results and the model should be saved
                        (default: ``${SESSION_SAVE_PATH}``)
  --learning-rate LEARNING_RATE
                        Optimizer (adam) learning rate (default: 0.001)
  --regularization REGULARIZATION
                        Regularization factor to apply (default: 0.01)
  --dropout
                        Dropout probability to use, the value must be between
                        0.0 and 1.0 (default: 0.2)
  --log-device          Log device placement when creating all the tensorflow
                        tensors (default: False)
  --use-distance        Whether to use distance or the boolean value when
                       pip  creating the siamese model (default: False)
  --random-labels       Whether to use or not random labels, use ONLY to
                        validate a model (default: False)
  --full-summary        Write a full summary for tensorboard, otherwise only
                        the scalar variables will be logged (default: False)
  --save-model          Save the model to the location specified at the results path

  --mrmr-size           Select the number of features that should be selected by mrmr
                        for feeding the model

  --read_splits         Select weather read from the file the spliting the model or generate during the training
  --bin-number          the number of bin that should be selected for train test spiliting it is used to have a
                        same distribution for training and testing


"""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import pathlib
import pandas as pd

from pytorch_src.models.lightning import ImageSiamese
import config as settings
import utils
import data
import numpy as np
from data.train_test import get_sets_generator, get_sets_reader

import torch
import torch.nn as nn

from pytorch_lightning import Trainer

from typing import Tuple, Dict, Any

logger = utils.init_logger("start")


def train_iterations(model: models.ImageSiamese,
                     epochs: int):
    """
    Execute the train iterations with all the epochs

    :param sess: Tensorflow session
    :param model: Model with a :func:`models.BasicModel.feed_dict` method to get the ``feed_dict`` for
                  ``sess.run(...)``
    :param batch_data: Class containing the information for the batch data, it's necessary because it contains
                       information regarding the mean and std of the radiomic features.
    :param pairs: List of pairs that can be trained. Usually this pairs can be obtained by calling
                  :func:`data.SplitPairs.folds` or :func:`data.SplitPairs.train_test_split`
    :param summary_writer: Summary writer provided by Tensorflow to show the training progress
    :param batch_size: Batch size for training Since usually images are used, the whole dataset does not fit in
                       memory so, setting the batch size, can avoid memory overflows.

                       The pairs will be generated by having a number of different ids among all pairs equal to
                       the batch size.
    :param epochs: Number of epochs, passes through the complete dataset, should be done when training
    """

    # Train iterations
    trainer = Trainer(min_epochs = 20, max_nb_epochs = 100, gpus=1)
    trainer.fit(model)
    # calculate C index


def select_model(model_key: str, number_feature: int, **kwargs):
    """
    Selects and constructs the model to be used based on the CLI options passed.

    :param model_key: String key to select the model
    :return: Instance of `models.BasicSiamese` with the proper subclass selected
    """
    return ImageSiameseL

################
#     MAIN     #
################
def deepCinet(model: str,
              survival: bool = False,
              cv_folds: int = 1,
              test_size: float = .2,
              gpu_level: int = 0,
              gpu_allow_growth=True,
              num_epochs: int = 1,
              batch_size: int = 20,
              target_path: str = settings.DATA_PATH_TARGET,
              feature_path: str = settings.DATA_PATH_FEATURE,
              results_path: str = settings.SESSION_SAVE_PATH,
              image_path: str = settings.DATA_PATH_IMAGE,
              learning_rate: int = 0.001,
              regularization: float = 0.01,
              splitting_model: int = 0,
              threshold: float = 3,
              bin_number: int = 4,
              dropout: float = 0.2,
              log_device=False,
              use_distance=False,
              random_labels=False,
              full_summary=True,
              save_model=True,
              split=1,  # todo check if required to add split_seed and initial_seed to the argument
              split_seed=None,
              split_number=None,  # it is used for the time we are using the generated test and train sets
              initial_seed=None,
              mrmr_size=0,
              read_splits=False,
              input_path=settings.DATA_PATH_INPUT_TEST_TRAIN,
              train_distance=0,
              test_distance=0):
    """
    deepCient
    :param args: Command Line Arguments
    """
    results_path = pathlib.Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    logger = utils.init_logger(f'train_{0}', str(results_path))
    logger.debug("Script starts")
    # logger.info(f"Results path: {results_path}")
    logger.info("Results path: {_results_path}".format(_results_path=results_path))
    results_path.mkdir(parents=True, exist_ok=True)

    logger = utils.init_logger(f'train_{0}', str(results_path))

    logger.debug("Script starts")

    logger.info(f"Results path: {results_path}")

    logger.info("Script to train a siamese neural network model")
    logger.info(f"Using batch size: {batch_size}")
    features = pd.DataFrame()
    # read features and clinical data frame the path is defined in the settings.py
    logger.info(f"feature path: {feature_path}")
    features = pd.read_csv(os.path.expandvars(feature_path), index_col=0)

    logger.info(f"number of features is {len(features.index)}")

    logger.info(f"target path: {target_path}")
    clinical_df = pd.read_csv(os.path.expandvars(target_path), index_col=0)
    logger.info("read Feature DataFrame")

    # read the input path for the time that train and test are splitted before head by train_test_generator.py
    number_feature = mrmr_size if mrmr_size > 0 else len(features.index)

    counts = {}
    for key in ['train', 'test', 'mixed']:
        counts[key] = {
            'total': 0,
            'correct': 0,
            'c_index': []
        }
    data_set = data.pair_data.SplitPairs(target_path=target_path, survival=survival)
    if read_splits:
        cv_path = os.path.join(input_path, f"cv_{cv_folds}")
        random_path = os.path.join(cv_path, f"random_seed_{split_number}")
        split_path = os.path.join(random_path, f"splitting_models_{splitting_model}")
        enum_generator = get_sets_reader(cv_folds, split_path, mrmr_size, feature_feature_path)

        for i, (train_ids, test_ids, df_features) in enum_generator:
            test_ids['id'] = test_ids['id'].astype(str)
            train_ids['id'] = train_ids['id'].astype(str)
            train_data = data_set.target_data.merge(train_ids, left_on="id", right_on="id", how="inner")
            test_data = data_set.target_data.merge(test_ids, left_on="id", right_on="id", how="inner")
            df_features = features.copy()
            train_pairs, test_pairs, mixed_pairs = data_set.create_train_test(train_data,
                                                                              test_data,
                                                                              random=random_labels,
                                                                              train_distance=train_distance,
                                                                              test_distance=test_distance)
            # Initialize all the variables
            logger.info(f"New fold {i}, {len(train_pairs)} train pairs, {len(test_pairs)} test pairs")
            summaries_dir = os.path.join(results_path, f"split_{split:0>2}")
            summaries_dir = os.path.join(summaries_dir, 'summaries', f'fold_{i:>2}')
            logger.info(f"Saving results at: {summaries_dir}")
            #Epoch iterations
            siamese_model = ImageSiameseL(train_pairs, batch_size, image_path)
            train_iterations(siamese_model,
                             num_epochs)

    else:
        enum_generator = get_sets_generator(data_set,
                                            cv_folds,
                                            test_size,
                                            random_labels,
                                            splitting_model,
                                            threshold,
                                            split_seed)
        for i, (train_idx, test_idx) in enum_generator:
            if mrmr_size > 0:
                df_features = data.select_mrmr_features(features.copy(), clinical_df.iloc[train_idx].copy(),
                                                        mrmr_size).copy()
            else:
                df_features = features.copy()
            train_data = data_set.target_data.iloc[train_idx]
            test_data = data_set.target_data.iloc[test_idx]
            train_pairs, test_pairs, mixed_pairs = data_set.create_train_test(train_data,
                                                                              test_data,
                                                                              random=random_labels,
                                                                              train_distance=train_distance,
                                                                              test_distance=test_distance)
            # Initialize all the variables
            logger.info(f"New fold {i}, {len(train_pairs)} train pairs, {len(test_pairs)} test pairs")
            summaries_dir = os.path.join(results_path, 'summaries', f'fold_{i}')
            summaries_dir = os.path.join(summaries_dir, f"split_{split:0>2}")
            logger.info(f"Saving results at: {summaries_dir}")

            # Epoch iterations
            siamese_model = RadiomicsModel(train_pairs, test_pairs, batch_size, image_path)
            train_iterations(siamese_model, num_epochs)

def main(args: Dict[str, Any]) -> None:
    """
    Main function
    :param args: Command Line Arguments
    """
    logger.info("Script to train a siamese neural network model")
    logger.info(f"Using batch size: {args['batch_size']}")
    mrmr_size = args['mrmr_size']
    random_labels = args['random_labels']
    model = args['model']
    gpu_level = args['gpu_level']
    regularization = args['regularization']
    dropout = args['dropout']
    learning_rate = args['learning_rate']
    use_distance = args['use_distance']
    full_summary = args['full_summary']
    log_device = args['log_device']
    gpu_allow_growth = args['gpu_allow_growth']
    cv_folds = args['cv_folds']
    test_size = args['test_size']
    random_labels = args['random_labels']
    splitting_model = args['splitting_model']
    threshold = args['threshold']
    batch_size = args['batch_size']
    num_epochs = args['num_epochs']
    results_path = args['results_path']
    save_model = args['save_model']
    read_splits = args['read_splits']
    input_path = args['input_path']
    train_distance = args['train_distance']
    test_distance = args['test_distance']
    feature_path = args['feature_path']
    target_path = args['target_path']
    image_path = args['image_path']
    survival = args['survival']
    number_feature = mrmr_size if mrmr_size > 0 else settings.NUMBER_FEATURES

    deepCinet(model=model,
              survival=survival,
              cv_folds=cv_folds,
              test_size=test_size,
              gpu_level=gpu_level,
              num_epochs=num_epochs,
              batch_size=batch_size,
              feature_path=feature_path,
              target_path=target_path,
              image_path=image_path,
              results_path=results_path,
              learning_rate=learning_rate,
              regularization=regularization,
              splitting_model=splitting_model,
              threshold=threshold,
              bin_number=4,
              dropout=dropout,
              log_device=log_device,
              use_distance=use_distance,
              random_labels=random_labels,
              full_summary=full_summary,
              split=1,
              split_seed=None,
              split_number=None,
              mrmr_size=mrmr_size,
              read_splits=read_splits,
              input_path=input_path,
              train_distance=train_distance,
              test_distance=test_distance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fit the data with a pytorch model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    # parser._action_groups.pop()

    required = parser.add_argument_group('required named arguments')
    optional = parser.add_argument_group('optional named arguments')

    # Required arguments
    required.add_argument(
        "--model",
        help="Choose the model that you want to use for training",
        type=str,
        required=True
    )

    # Optional arguments
    optional.add_argument(
        "-h", "--help",
        help="Show this help",
        action="help"
    )

    optional.add_argument(
        "--cv-folds",
        help="Number of cross validation folds. If 0 < folds < 2 CV won't be used and the test set size "
             "will be defined by --test-size. If folds < 0 Leave One Out Cross Validation will be used instead",
        default=1,
        type=int
    )
    optional.add_argument(
        "--test-size",
        help="Size of the test set as a float between 0 and 1",
        default=.25,
        type=float
    )
    optional.add_argument(
        "--gpu-level",
        help="Amount of GPU resources used when fitting the model. 0: no GPU usage, "
             "1: only second conv layers, 2: all conv layers, "
             "3: all layers and parameters are on the GPU",
        default=0,
        type=int
    )
    optional.add_argument(
        "--gpu-allow-growth",
        help="Allow Tensorflow to use dynamic allocations with GPU memory",
        default=False,
        action="store_true",
    )
    optional.add_argument(
        "-n", "--num-epochs",
        help="Number of epochs to use when training. Times passed through the entire dataset",
        metavar="NUM_EPOCHS",
        dest="num_epochs",
        default=1,
        type=int
    )
    optional.add_argument(
        "--batch-size",
        help="Batch size for each train iteration",
        default=20,
        type=int
    )
    optional.add_argument(
        "--results-path",
        help="Path where the results and the model should be saved",
        default=settings.SESSION_SAVE_PATH,
        type=str
    )
    optional.add_argument(
        "--learning-rate",
        help="Optimizer (adam) learning rate",
        default=0.001,
        type=float
    )
    optional.add_argument(
        "--regularization",
        help="Regularization factor to apply",
        default=0.01,
        type=float
    )
    optional.add_argument(
        "--splitting-model",
        help="The model of splitting data to train and test. "
             "0: split based on event censored data, "
             "1: split based on (same distribution of survival for train and test), "
             "2: split based on the threshold",
        default=0,
        type=int
    )
    optional.add_argument(
        "--feature-path",
        help="The type of the data ",
        default=settings.DATA_PATH_FEATURE,
        type=str
    )
    optional.add_argument(
        "--target-path",
        help=" ",
        default=settings.DATA_PATH_TARGET,
        type=str
    )
    optional.add_argument(
        "--image-path",
        help=" ",
        default=settings.DATA_PATH_IMAGE,
        type=str
    )

    optional.add_argument(
        "--threshold",
        help="The threshold for splitting ",
        default=3,
        type=float
    )
    optional.add_argument(
        "--bin-number",
        help="The number of bins for splitting based on the distribution",
        default=4,
        type=int
    )
    optional.add_argument(
        "--dropout",
        help="Dropout probability to use, the value must be between 0.0 and 1.0",
        default=0.2,
        type=float,
        choices=[utils.ArgRange(0., 1.)]
    )
    optional.add_argument(
        "--log-device",
        help="Log device placement when creating all the tensorflow tensors",
        action="store_true",
        default=False
    )
    optional.add_argument(
        "--use-distance",
        help="Whether to use distance or the boolean value when creating the siamese model",
        action="store_true",
        default=False
    )
    optional.add_argument(
        "--random-labels",
        help="Whether to use or not random labels, use ONLY to validate a model",
        action="store_true",
        default=False
    )
    optional.add_argument(
        "--full-summary",
        help="Write a full summary for tensorboard, otherwise only the scalar variables will be logged",
        action="store_true",
        default=False
    )
    optional.add_argument(
        "--save-model",
        help="Save the model to the location specified at the results path",
        action="store_true",
        default=False
    )

    optional.add_argument(
        "--mrmr-size",
        help="The number of features that should be selected with Mrmr- 0 means Mrmr shouldn't apply",
        default=0,
        type=int
    )

    optional.add_argument(
        "--read-splits",
        help="The way that generate input for the model read split or read from the pre generated inputs",
        action="store_true",
        default=False
    )
    optional.add_argument(
        "--input-path",
        help="when read from pre generated inputs, input-path showes the path to that location",
        type=str,
        required=False
    )

    optional.add_argument(
        "--train_distance",
        help="This is used to consider rCI when generating pairs only the pairs with dis > distance would be selected ",
        default=0,
        type=int
    )
    optional.add_argument(
        "--test_distance",
        help="This is used to consider rCI when generating pairs only the pairs with dis > distance would be selected ",
        default=0,
        type=int
    )

    optional.add_argument(
        "--survival",
        help="This is used to show that the data which you want to work with is survival data",
        action="store_true",
        default=False
    )

    # See if we are running in a SLURM task array
    array_id = os.getenv('SLURM_ARRAY_TASK_ID', 0)

    arguments, unknown = parser.parse_known_args()
    arguments = vars(arguments)

    arguments['results_path'] = os.path.abspath(arguments['results_path'])
    results_path = pathlib.Path(arguments['results_path'])
    results_path.mkdir(parents=True, exist_ok=True)

    logger = utils.init_logger(f'train_{array_id}', str(results_path))

    logger.debug("Script starts")
    logger.debug(arguments)
    logger.info(f"Results path: {results_path}")

    if len(unknown) > 0:
        logger.warning(f"Warning: there are unknown arguments {unknown}")

    if arguments['batch_size'] < 2:
        logger.error("Batch size is too small! It should be at least 2. Exiting")
        exit(1)

    try:
        # For now the arguments are ignored
        main(arguments)
    except KeyboardInterrupt:
        logger.info("\r----------------------------------")
        logger.info("\rStopping due to keyboard interrupt")
        logger.info("\rTHANKS FOR THE RIDE 😀")
