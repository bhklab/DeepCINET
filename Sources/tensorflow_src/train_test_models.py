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
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import pathlib
import tensorflow as tf
import pandas as pd

import data
from data.train_test import get_sets_generator, get_sets_reader
import tensorflow_src.models as models
import tensorflow_src.models.basics
import tensorflow_src.settings as settings
import utils

from tensorflow.core.protobuf import config_pb2
from typing import Iterator, Tuple, Dict, Any

logger = utils.init_logger("start")


def train_iterations(sess: tf.Session,
                     model: tensorflow_src.models.basics.BasicSiamese,
                     batch_data: data.BatchData,
                     pairs: pd.DataFrame,
                     summary_writer: tf.summary.FileWriter,
                     batch_size: int,
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
    final_iterations = 0
    sess.run(tf.global_variables_initializer(), options=config_pb2.RunOptions(
        report_tensor_allocations_upon_oom=True))
    for epoch in range(epochs):
        total_pairs = len(pairs) * (settings.TOTAL_ROTATIONS if model.uses_images() else 1)
        for i, batch in enumerate(batch_data.batches(pairs,
                                                     batch_size=batch_size,
                                                     load_images=model.uses_images(),
                                                     train=True)):

            total_pairs -= len(batch.pairs)

            # Execute graph operations but only write summaries once every 5 iterations
            if final_iterations % 5 == 0:
                _, c_index_result, loss, summary = sess.run(
                    [
                        model.minimizer,
                        model.c_index,
                        model.total_loss,
                        model.summary
                    ],
                    feed_dict=model.feed_dict(batch)
                )
                # logger.info(f"Epoch: {epoch:>3}, Batch: {i:>4}, size: {len(batch.pairs):>5}, remaining: "
                #             f"{total_pairs:>6}, "
                #             f"c-index: {c_index_result:>#5.3}, loss: {loss:>#5.3}")
                logger.info("Epoch: {_epoch:>3}, Batch: {_i:>4}, size: {_batch_pair:>5}, remaining: "
                            "{_total_pairs:>6}, "
                            "c-index: {_c_index_result:>#5.3}, loss: {_loss:>#5.3}".format
                            (_epoch=epoch, _i=i,
                             _batch_pair=len(batch.pairs),
                             _total_pairs=total_pairs,
                             _c_index_result=c_index_result,
                             _loss=loss))

                summary_writer.add_summary(summary, final_iterations)
            else:
                _, c_index_result, loss = sess.run(
                    [
                        model.minimizer,
                        model.c_index,
                        model.total_loss
                    ],
                    feed_dict=model.feed_dict(batch)
                )

            final_iterations += 1


def test_iterations(sess: tf.Session,
                    model: tensorflow_src.models.basics.BasicSiamese,
                    batch_data: data.BatchData,
                    pairs: pd.DataFrame,
                    batch_size: int) -> Tuple[int, int, pd.DataFrame]:
    """
    Iterations to test the data provided.

    :param sess: Tensorflow session
    :param model: Model with a :func:`models.BasicModel.feed_dict` method to get the ``feed_dict`` for
                  ``sess.run(...)``
    :param batch_data: Class containing the information for the batch data, it's necessary because it contains
                       information regarding the mean and std of the radiomic features.
    :param pairs: Lis of pairs that should be evaluated. Usually this pairs can be obtained by calling
                  :func:`data.SplitPairs.folds` or :func:`data.SplitPairs.train_test_split`
    :param batch_size: Batch size for testing. Since usually images are being used, the whole dataset does not fit
                       in memory so setting the batch_size can avoid memory overflows.

                       The pairs will be generated by having a number of different ids among all pairs, equal to
                       the batch size.
    :return:
    """
    # After we iterate over all the data inspect the test error
    total_pairs = len(pairs) * (settings.TOTAL_ROTATIONS if model.uses_images() else 1)
    correct_count = 0  # To store correct predictions
    pairs_count = 0
    result_data = []

    # Test iterations
    for i, batch in enumerate(batch_data.batches(pairs,
                                                 batch_size=batch_size,
                                                 load_images=model.uses_images(),
                                                 train=False)):
        # Execute test operations
        temp_sum, c_index_result, predictions, probabilities, gather_a, gather_b = sess.run(
            [
                model.good_predictions,
                model.c_index,
                model.y_estimate,
                model.y_prob,
                model.gathered_a,
                model.gathered_b
            ],
            feed_dict=model.feed_dict(batch, training=False)
        )

        correct_count += temp_sum
        total_pairs -= len(batch.pairs)
        pairs_count += len(batch.pairs)

        # Save results
        temp_results = batch.pairs.copy()
        temp_results['gather_a'] = gather_a
        temp_results['gather_b'] = gather_b
        temp_results['probabilities'] = probabilities
        temp_results['predictions'] = predictions

        result_data.append(temp_results)

        if i % 10 == 0 or total_pairs == 0:
            # logger.info(f"Batch: {i:>4}, size: {len(batch.pairs):>5}, remaining: {total_pairs:>5}, "
            #             f"c-index: {c_index_result:>#5.3}, final c-index:{correct_count/pairs_count:>#5.3}")
            logger.info("Batch: {_i:>4}, size: {_batch_pairs_len:>5}, remaining: {_total_pairs:>5}, "
                        "c-index: {_c_index_result:>#5.3}, final c-index:{_correct_count_pairs_count:>#5.3}".format
                        (_i=i,
                         _batch_pairs_len=len(batch.pairs),
                         _total_pairs=total_pairs,
                         _c_index_result=c_index_result,
                         _correct_count_pairs_count=correct_count / pairs_count))

    return correct_count, pairs_count, pd.concat(result_data)


def select_model(model_key: str, number_feature: int, **kwargs) -> tensorflow_src.models.basics.BasicSiamese:
    """
    Selects and constructs the model to be used based on the CLI options passed.

    :param model_key: String key to select the model
    :return: Instance of `models.BasicSiamese` with the proper subclass selected
    """
    if model_key == "SimpleImageSiamese":
        return models.SimpleImageSiamese(**kwargs)
    elif model_key == "ImageScalarSiamese":
        return models.ImageScalarSiamese(**kwargs)
    elif model_key == "ScalarOnlySiamese":
        return models.ScalarOnlySiamese(number_feature, **kwargs)
    elif model_key == "ScalarOnlyDropoutSiamese":
        return models.ScalarOnlyDropoutSiamese(**kwargs)
    elif model_key == "ImageSiamese":
        return models.ImageSiamese(**kwargs)
    elif model_key == "ResidualImageScalarSiamese":
        return models.ResidualImageScalarSiamese(**kwargs)
    elif model_key == "VolumeOnlySiamese":
        return models.VolumeOnlySiamese(**kwargs)
    elif model_key == "ClinicalOnlySiamese":
        return models.ClinicalOnlySiamese(number_feature, **kwargs)
    elif model_key == "ClinicalVolumeSiamese":
        return models.ClinicalVolumeSiamese(**kwargs)
    elif model_key == "ClinicalRadiomicSiamese":
        return models.ClinicalRadiomicSiamese(**kwargs)

    else:
        logger.error(f"Unknown option for model {model_key}")
        exit(1)


################
#     MAIN     #
################
def deepCinet(model: str,
              cv_folds: int = 1,
              test_size: float = .2,
              gpu_level: int = 0,
              gpu_allow_growth=False,
              num_epochs: int = 1,
              batch_size: int = 20,
              data_type: str = 'radiomic',
              results_path: str = settings.SESSION_SAVE_PATH,
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
              read_splits=False):
    """
    deepCient
    :param args: Command Line Arguments
    """
    tf.reset_default_graph()
    results_path = pathlib.Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    logger = utils.init_logger('train_{0}', str(results_path))

    logger.debug("Script starts")
    # logger.info(f"Results path: {results_path}")
    logger.info("Results path: {_results_path}".format(_results_path=results_path))
    results_path.mkdir(parents=True, exist_ok=True)

    logger = utils.init_logger(f'train_{0}', str(results_path))

    logger.debug("Script starts")

    logger.info("Script to train a siamese neural network model")
    # logger.info(f"Using batch size: {batch_size}")
    logger.info("Using batch size: {_batch_size}".format(_batch_size=batch_size))
    features = pd.DataFrame()
    # read features and clinical data frame the path is defined in the settings.py
    # logger.info(f"data type: {data_type}")
    logger.info("data type: {_data_type}".format(_data_type=data_type))
    if data_type == "radiomic":
        features = pd.read_csv(settings.DATA_PATH_RADIOMIC_PROCESSED, index_col=0)
    elif data_type == "clinical":
        features = pd.read_csv(settings.DATA_PATH_CLINIC_PROCESSED, index_col=0)
    elif data_type == "clinicalVolume":
        features = pd.read_csv(settings.DATA_PATH_VOLUME_CLINIC_PROCESSED, index_col=0)

    # logger.info(f"number of features is {len(features.index)}")
    logger.info("number of features is {_len_feature_index}".format(_len_feature_index=len(features.index)))
    clinical_df = pd.read_csv(settings.DATA_PATH_CLINICAL_PROCESSED, index_col=0)
    logger.info("read Feature DataFrame")

    # read the input path for the time that train and test are splitted before head by train_test_generator.py
    input_path = settings.DATA_PATH_INPUT_TEST_TRAIN
    # number of features set to the number of mrmr_size otherwise it should be set to the number of all features
    number_feature = mrmr_size if mrmr_size > 0 else len(features.index)
    siamese_model = select_model(model,
                                 number_feature=number_feature,
                                 gpu_level=gpu_level,
                                 regularization=regularization,
                                 dropout=dropout,
                                 learning_rate=learning_rate,
                                 use_distance=use_distance,
                                 full_summary=full_summary,
                                 seed=initial_seed)
    conf = tf.ConfigProto(log_device_placement=log_device)
    conf.gpu_options.allow_growth = gpu_allow_growth

    with tf.Session(config=conf) as sess:
        counts = {}
        for key in ['train', 'test', 'mixed']:
            counts[key] = {
                'total': 0,
                'correct': 0,
                'c_index': []
            }
        dataset = data.pair_data.SplitPairs()
        if (read_splits):
            # cv_path = os.path.join(input_path, f"cv_{cv_folds}")
            # random_path = os.path.join(cv_path, f"random_seed_{split_number}")
            # split_path = os.path.join(random_path, f"splitting_models_{splitting_model}")
            cv_path = os.path.join(input_path, "cv_{_cv_folds}".format(_cv_folds=cv_folds))
            random_path = os.path.join(cv_path, "random_seed_{_split_number}".format(_split_number=split_number))
            split_path = os.path.join(random_path,
                                      "splitting_models_{_splitting_model}".format(_splitting_model=splitting_model))
            enum_generator = get_sets_reader(cv_folds, split_path, mrmr_size, data_type)
            clinical_data = dataset.clinical_data.copy()
            for i, (train_ids, test_ids, df_features) in enum_generator:
                train_data = dataset.clinical_data.merge(train_ids, left_on="id", right_on="id", how="inner")
                test_data = dataset.clinical_data.merge(train_ids, left_on="id", right_on="id", how="inner")
                train_pairs, test_pairs, mixed_pairs = dataset.create_train_test(train_data, test_data,
                                                                                 random=random_labels)
                # Initialize all the variables
                # logger.info(f"New fold {i}, {len(train_pairs)} train pairs, {len(test_pairs)} test pairs")
                logger.info("New fold {i}, {_len_train_pairs} train pairs, {len(test_pairs)} test pairs".format(
                    i=i, _len_train_pairs=len(train_pairs), _len_test_pairs=len(test_pairs)))
                # summaries_dir = os.path.join(results_path, 'summaries', f'fold_{i}')
                summaries_dir = os.path.join(results_path, 'summaries', 'fold_{i}'.format(i=i))
                # summaries_dir = os.path.join(summaries_dir, f"split_{split:0>2}")
                summaries_dir = os.path.join(summaries_dir, "split_{split:0>2}".format(split=split))
                # logger.info(f"Saving results at: {summaries_dir}")
                logger.info("Saving results at: {summaries_dir}".format(summaries_dir=summaries_dir))
                train_summary = tf.summary.FileWriter(summaries_dir, sess.graph)
                batch_data = data.BatchData(df_features)
                train_summary.add_graph(sess.graph)
                # Epoch iterations
                train_iterations(sess,
                                 siamese_model,
                                 batch_data,
                                 train_pairs,
                                 train_summary,
                                 batch_size,
                                 num_epochs)

                predictions = {}
                for pairs, name in [(train_pairs, 'train'), (test_pairs, 'test'), (mixed_pairs, 'mixed')]:
                    if len(pairs) <= 0:
                        continue
                    # logger.info(f"Computing {name} c-index")
                    logger.info("Computing {name} c-index".format(name=name))
                    correct, total, results = \
                        test_iterations(sess,
                                        siamese_model,
                                        batch_data,
                                        pairs,
                                        batch_size)

                    correct = int(correct)

                    c_index = correct / total

                    counts[name]['total'] += total
                    counts[name]['correct'] += correct
                    counts[name]['c_index'].append((i, c_index))

                    predictions[name] = results

                    # logger.info(f"{name} set c-index: {c_index}, correct: {correct}, total: {total}, "
                    #             f"temp c-index: {counts[name]['correct']/counts[name]['total']}")
                    logger.info("{_name} set c-index: {_c_index}, correct: {_correct}, total: {total}, ".format(
                        _name=name,
                        _c_index=c_index,
                        _correct=correct,
                        _total=total))
                    logger.info("temp c-index: {temp_cindex}".format(
                        temp_cindex=counts[name]['correct'] / counts[name]['total']))

                # Save each fold in a different directory

                # results_save_path = os.path.join(results_path, f"fold_{i:0>2}")
                # results_save_path = os.path.join(results_save_path, f"split_{split:0>2}")
                results_save_path = os.path.join(results_path, "fold_{_i:0>2}".format(_i=i))
                results_save_path = os.path.join(results_save_path, "split_{_split:0>2}".format(_split=split))
                logger.info("Saving results at: {_results_save_path}".format(_results_save_path=results_save_path))
                utils.save_results(sess, predictions, results_save_path, save_model)
                logger.info("\r ")

        else:
            enum_generator = get_sets_generator(dataset,
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
                train_data = dataset.clinical_data.iloc[train_idx]
                test_data = dataset.clinical_data.iloc[test_idx]
                train_pairs, test_pairs, mixed_pairs = dataset.create_train_test(train_data, test_data,
                                                                                 random=random_labels)
                # Initialize all the variables
                logger.info("New fold {i}, {_len_train_pairs} train pairs, {_len_test_pairs} test pairs".format(
                    i=i, _len_train_pairs=len(train_pairs), _len_test_pairs=len(test_pairs)))

                # summaries_dir = os.path.join(results_path, 'summaries', f'fold_{i}')
                summaries_dir = os.path.join(results_path, 'summaries', 'fold_{i}'.format(i=i))
                # summaries_dir = os.path.join(summaries_dir, f"split_{split:0>2}")
                summaries_dir = os.path.join(summaries_dir, "split_{split:0>2}".format(split=split))
                # logger.info(f"Saving results at: {summaries_dir}")
                logger.info("Saving results at: {summaries_dir}".format(summaries_dir=summaries_dir))
                train_summary = tf.summary.FileWriter(summaries_dir, sess.graph)
                batch_data = data.BatchData(df_features)
                train_summary.add_graph(sess.graph)

                # Epoch iterations
                train_iterations(sess,
                                 siamese_model,
                                 batch_data,
                                 train_pairs,
                                 train_summary,
                                 batch_size,
                                 num_epochs)

                predictions = {}
                for pairs, name in [(train_pairs, 'train'), (test_pairs, 'test'), (mixed_pairs, 'mixed')]:
                    if len(pairs) <= 0:
                        continue
                    # logger.info(f"Computing {name} c-index")
                    logger.info("Computing {name} c-index".format(name=name))
                    correct, total, results = \
                        test_iterations(sess,
                                        siamese_model,
                                        batch_data,
                                        pairs,
                                        batch_size)

                    correct = int(correct)

                    c_index = correct / total

                    counts[name]['total'] += total
                    counts[name]['correct'] += correct
                    counts[name]['c_index'].append((i, c_index))

                    predictions[name] = results

                    # logger.info(f"{name} set c-index: {c_index}, correct: {correct}, total: {total}, "
                    # f"temp c-index: {counts[name]['correct']/counts[name]['total']}")
                    logger.info("{_name} set c-index: {_c_index}, correct: {_correct}, total: {_total}, ".format(
                        _name=name,
                        _c_index=c_index,
                        _correct=correct,
                        _total=total))
                    logger.info("temp c-index: {temp_cindex}".format(
                        temp_cindex=counts[name]['correct'] / counts[name]['total']))

                # Save each fold in a different directory

                # results_save_path = os.path.join(results_path, f"fold_{i:0>2}")
                # results_save_path = os.path.join(results_save_path, f"split_{split:0>2}")
                # logger.info(f"Saving results at: {results_save_path}")

                results_save_path = os.path.join(results_path, "fold_{_i:0>2}".format(_i=i))
                results_save_path = os.path.join(results_save_path, "split_{_split:0>2}".format(_split=split))
                logger.info("Saving results at: {_results_save_path}".format(_results_save_path=results_save_path))
                utils.save_results(sess, predictions, results_save_path, save_model)
                logger.info("\r ")

        for key in counts:
            if counts[key]['total'] <= 0:
                continue
            logger.info(f"Final {key} c-index: {counts[key]['correct']/counts[key]['total']}")
        return counts, predictions


def main(args: Dict[str, Any]) -> None:
    """
    Main function
    :param args: Command Line Arguments
    """
    logger.info("Script to train a siamese neural network model")
    logger.info("Using batch size: {batch_size}".format(batch_size=args['batch_size']))
    data_type = args['data_type']
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
    number_feature = mrmr_size if mrmr_size > 0 else settings.NUMBER_FEATURES

    deepCinet(model=model,
              cv_folds=cv_folds,
              test_size=test_size,
              gpu_level=gpu_level,
              num_epochs=num_epochs,
              batch_size=batch_size,
              data_type=data_type,
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
              read_splits=read_splits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fit the data with a Tensorflow model",
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
        "--data_type",
        help="The threshold for splitting ",
        default="radiomic",
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
        "--read_splits",
        help="The way that generate input for the model read split or read from the pre generated inputs",
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
