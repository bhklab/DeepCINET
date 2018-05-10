#!/usr/bin/env python3

import argparse
import os
from typing import Dict, Tuple, Any, Iterator

import tensorflow as tf
import pandas as pd

import data
import models
import models.basics
import settings
import utils


def train_iterations(sess: tf.Session,
                     model: models.basics.BasicSiamese,
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
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        total_pairs = len(pairs)*(settings.TOTAL_ROTATIONS if model.uses_images() else 1)
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
                summary_writer.add_summary(summary, final_iterations)
                logger.info(f"Epoch: {epoch:>3}, Batch: {i:>4}, size: {len(batch.pairs):>5}, remaining: "
                            f"{total_pairs:>6}, "
                            f"c-index: {c_index_result:>#5.3}, loss: {loss:>#5.3}")
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
                    model: models.basics.BasicSiamese,
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
    total_pairs = len(pairs)*(settings.TOTAL_ROTATIONS if model.uses_images() else 1)
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
        temp_results: pd.DataFrame = batch.pairs.copy()
        temp_results['gather_a'] = gather_a
        temp_results['gather_b'] = gather_b
        temp_results['probabilities'] = probabilities
        temp_results['predictions'] = predictions

        result_data.append(temp_results)

        if i % 10 == 0 or total_pairs == 0:
            logger.info(f"Batch: {i:>4}, size: {len(batch.pairs):>5}, remaining: {total_pairs:>5}, "
                        f"c-index: {c_index_result:>#5.3}, final c-index:{correct_count/pairs_count:>#5.3}")

    return correct_count, pairs_count, pd.concat(result_data)


def select_model(model_key: str, **kwargs) -> models.basics.BasicSiamese:
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
        return models.ScalarOnlySiamese(**kwargs)
    elif model_key == "VolumeOnlySiamese":
        return models.VolumeOnlySiamese(**kwargs)
    else:
        logger.error(f"Unknown option for model {model_key}")
        exit(1)


def get_sets_generator(cv_folds: int, test_size: int, bidirectional: bool) \
        -> Iterator[Tuple[int, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]:
    dataset = data.pair_data.SplitPairs()

    # Decide whether to use CV or only a single test/train sets
    if cv_folds < 2:
        generator = dataset.train_test_split(test_size, bidirectional=bidirectional)
        enum_generator = [(0, generator)]
    else:
        generator = dataset.folds(cv_folds, bidirectional=bidirectional)

        # Slurm configuration
        task_id = os.getenv('SLURM_ARRAY_TASK_ID', 0)
        if int(os.getenv('SLURM_ARRAY_TASK_COUNT', 0)) == cv_folds:
            task_id = int(task_id)
            logger.info(f"Task number: {task_id}")
            enum_generator = [(task_id, list(generator)[task_id])]
        else:
            enum_generator = enumerate(generator)

    logger.debug("Folds created")

    return enum_generator


################
#     MAIN     #
################
def main(args: Dict[str, Any]):
    logger.info("Script to train a siamese neural network model")
    logger.info(f"Using batch size: {args['batch_size']}")

    siamese_model = select_model(args['model'],
                                 gpu_level=args['gpu_level'],
                                 regularization=args['regularization'],
                                 dropout=args['dropout'],
                                 learning_rate=args['learning_rate'])

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = args['gpu_allow_growth']

    with tf.Session(config=conf) as sess:
        enum_generator = get_sets_generator(args['cv_folds'], args['test_size'], args['bidirectional'])

        counts = {
            'train': {
                'total': 0,
                'correct': 0,
            },
            'test': {
                'total': 0,
                'correct': 0
            },
            'mixed': {
                'total': 0,
                'correct': 0
            }
        }

        for i, (train_pairs, test_pairs, mixed_pairs) in enum_generator:
            # Initialize all the variables
            logger.info("\r ")
            logger.info(f"New fold {i}, {len(train_pairs)} train pairs, {len(test_pairs)} test pairs")

            summaries_dir = os.path.join(args['results_path'], 'summaries', f'fold_{i}')
            train_summary = tf.summary.FileWriter(summaries_dir, sess.graph)
            batch_data = data.BatchData()

            # Epoch iterations
            train_iterations(sess,
                             siamese_model,
                             batch_data,
                             train_pairs,
                             train_summary,
                             args['batch_size'],
                             args['num_epochs'])

            predictions = {}
            for pairs, name in [(train_pairs, 'train'), (test_pairs, 'test'), (mixed_pairs, 'mixed')]:
                logger.info(f"Computing {name} c-index")
                correct, total, results = \
                    test_iterations(sess,
                                    siamese_model,
                                    batch_data,
                                    pairs,
                                    args['batch_size'])

                counts[name]['total'] += total
                counts[name]['correct'] += correct

                predictions[name] = results

                logger.info(f"{name} set c-index: {correct/total}, correct: {correct}, total: {total}")

            # Save each fold in a different directory
            results_save_path = os.path.join(args['results_path'], f"fold_{i:0>2}")
            logger.info(f"Saving results at: {results_save_path}")
            utils.save_results(sess, predictions, results_save_path)

        for key in counts:
            logger.info(f"Final {key} c-index: {counts[key]['correct']/counts[key]['total']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fit the data with a Tensorflow model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--cv-folds",
        help="Number of cross validation folds. If < 2 CV won't be used and the test set size "
             "will be defined by --test-size",
        default=1,
        type=int
    )

    parser.add_argument(
        "--test-size",
        help="Size of the test set as a float between 0 and 1",
        default=.25,
        type=float
    )

    parser.add_argument(
        "--gpu-level",
        help="Amount of GPU resources used when fitting the model. 0: no GPU usage, "
             "1: only second conv layers, 2: all conv layers, "
             "3: all layers and parameters are on the GPU",
        default=0,
        type=int
    )

    parser.add_argument(
        "--gpu-allow-growth",
        help="Allow Tensorflow to use dynamic allocations with GPU memory",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "-n", "--num-epochs",
        help="Number of epochs to use when training. Times passed through the entire dataset",
        metavar="NUM_EPOCHS",
        dest="num_epochs",
        default=1,
        type=int
    )

    parser.add_argument(
        "--batch-size",
        help="Batch size for each train iteration",
        default=20,
        type=int
    )

    parser.add_argument(
        "--model",
        help="Choose the model that you want to use for training",
        default="SimpleImageSiamese",
        choices=['SimpleImageSiamese', 'ImageScalarSiamese', 'ScalarOnlySiamese', 'VolumeOnlySiamese'],
        type=str
    )

    parser.add_argument(
        "--results-path",
        help="Path where the results and the model should be saved",
        default=settings.SESSION_SAVE_PATH,
        type=str
    )

    parser.add_argument(
        "--learning-rate",
        help="Optimizer (adam) learning rate",
        default=0.001,
        type=float
    )

    parser.add_argument(
        "--regularization",
        help="Regularization factor to apply",
        default=0.01,
        type=float
    )

    parser.add_argument(
        "--dropout",
        help="Dropout probability to use",
        default=0.2,
        type=float,
        choices=[utils.ArgRange(0., 1.)]
    )

    parser.add_argument(
        "--bidirectional-pairs",
        help="When generating the pairs, for every two ids create the two possible pairs in the two possible "
             "directions",
        action="store_true",
        dest="bidirectional",
        default=True
    )

    parser.add_argument(
        "--no-bidirectional-pairs",
        help="Create pairs using only the comparisons in one direction",
        action="store_false",
        dest="bidirectional"
    )

    # See if we are running in a SLURM task array
    array_id = os.getenv('SLURM_ARRAY_TASK_ID', 0)

    arguments, unknown = parser.parse_known_args()
    arguments = vars(arguments)

    arguments['results_path'] = os.path.abspath(arguments['results_path'])

    if not os.path.exists(arguments['results_path']):
        os.makedirs(arguments['results_path'])

    logger = utils.init_logger(f'train_{array_id}', arguments['results_path'])

    logger.debug("Script starts")
    logger.debug(arguments)

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
