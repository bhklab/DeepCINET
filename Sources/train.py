#!/usr/bin/env python3

import argparse
import os
from typing import Dict, List, Tuple, Any, Iterator

import tensorflow as tf
import numpy as np
import pandas as pd

import data
import models
import settings
import utils

logger = utils.get_logger('train')


def train_iterations(sess: tf.Session,
                     model: models.BasicModel,
                     tensors: Dict[str, tf.Tensor],
                     pairs: List[data.PairComp],
                     summary_writer: tf.summary.FileWriter,
                     batch_size: int,
                     epochs: int):
    """
    Execute the train iterations with all the epochs

    :param sess: Tensorflow session
    :param model: Model with a :func:`models.BasicModel.feed_dict` method to get the ``feed_dict`` for
                  ``sess.run(...)``
    :param tensors: Tensors that can be run, provided as a dictionary with keys as strings
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

    for j in range(epochs):
        total_pairs = len(pairs)*(settings.TOTAL_ROTATIONS if model.uses_images() else 1)
        logger.info(f"Epoch: {j + 1} of {epochs}")
        for i, batch in enumerate(data.BatchData.batches(pairs,
                                                         batch_size=batch_size,
                                                         load_images=model.uses_images())):
            # Execute graph operations
            _, c_index_result, loss, summary = sess.run(
                [tensors['minimize'], tensors['c-index'], tensors['loss'], tensors['summary']],
                feed_dict=model.feed_dict(batch)
            )

            total_pairs -= len(batch.pairs_a)
            if i % 10 == 0 or total_pairs <= 0:
                logger.info(f"Batch: {i}, size: {len(batch.pairs_a)}, remaining: {total_pairs}, "
                            f"c-index: {c_index_result:.3}, loss: {loss:.3}")

            summary_writer.add_summary(summary, final_iterations + i)
            if total_pairs <= 0:
                final_iterations += i + 1


def test_iterations(sess: tf.Session,
                    model: models.BasicModel,
                    tensors: Dict[str, tf.Tensor],
                    pairs: List[data.PairComp],
                    batch_size: int) -> Tuple[int, int, pd.DataFrame]:
    """
    Iterations to test the data provided.

    :param sess: Tensorflow session
    :param model: Model with a :func:`models.BasicModel.feed_dict` method to get the ``feed_dict`` for
                  ``sess.run(...)``
    :param tensors: Tensors that can be run, provided as a dictionary with keys as strings
    :param pairs: Lis of pairs that should be evaluated. Usually this pairs can be obtained by calling
                  :func:`data.SplitPairs.folds` or :func:`data.SplitPairs.train_test_split`
    :param batch_size: Batch size for testing. Since usually images are being used, the whole dataset does not fit
                       in memory so setting the batch_size can avoid memory overflows.

                       The pairs will be generated by having a number of different ids among all pairs, equal to
                       the batch size.
    :return:
    """
    # After we iterate over all the data inspect the test error
    total_pairs = len(pairs)*settings.TOTAL_ROTATIONS
    correct_count = 0  # To store correct predictions
    pairs_count = 0

    result_data = {
        'pairs_a': [],
        'pairs_b': [],
        'labels': np.array([], dtype=bool),
        'predictions': np.array([], dtype=bool),
        'probabilities': np.array([]),
        'gather_a': np.array([]),
        'gather_b': np.array([]),
    }

    # Test iterations
    for i, batch in enumerate(data.BatchData.batches(pairs, batch_size=batch_size, load_images=model.uses_images())):
        # Execute test operations
        temp_sum, c_index_result, predictions, probabilities, gather_a, gather_b = sess.run(
            [
                tensors['true-predictions'],
                tensors['c-index'],
                tensors['predictions'],
                tensors['probabilities'],
                tensors['gather_a'],
                tensors['gather_b'],
            ],
            feed_dict=model.feed_dict(batch, training=False)
        )

        correct_count += temp_sum
        total_pairs -= len(batch.pairs_a)
        pairs_count += len(batch.pairs_a)

        # Save results
        result_data['pairs_a'] += [batch.ids_inverse[idx] for idx in batch.pairs_a]
        result_data['pairs_b'] += [batch.ids_inverse[idx] for idx in batch.pairs_b]
        result_data['labels'] = np.append(result_data['labels'], np.array(batch.labels).astype(bool))
        result_data['predictions'] = np.append(result_data['predictions'], np.array(predictions).astype(bool))
        result_data['probabilities'] = np.append(result_data['probabilities'], np.array(probabilities))
        result_data['gather_a'] = np.append(result_data['gather_a'], np.array(gather_a))
        result_data['gather_b'] = np.append(result_data['gather_b'], np.array(gather_b))

        if i % 10 == 0 or total_pairs == 0:
            logger.info(f"Batch: {i}, size: {len(batch.pairs_a)}, remaining: {total_pairs}, "
                        f"c-index: {c_index_result:.3}, final c-index:{correct_count/pairs_count:.3}")

    return correct_count, pairs_count, pd.DataFrame(result_data)


def select_model(model_key: str, gpu_level: int) -> models.BasicSiamese:
    """
    Selects and constructs the model to be used based on the CLI options passed.

    :param model_key: String key to select the model
    :param gpu_level: Amount of GPU to be used, required to create a model's instance
    :return: Instance of `models.BasicSiamese` with the proper subclass selected
    """
    if model_key == "SimpleImageSiamese":
        return models.SimpleImageSiamese(gpu_level)
    elif model_key == "ImageScalarSiamese":
        return models.ImageScalarSiamese(gpu_level)
    elif model_key == "ScalarOnlySiamese":
        return models.ScalarOnlySiamese(gpu_level)
    else:
        logger.error(f"Unknown option for model {model_key}")
        exit(1)


def get_tensors(siamese_model: models.BasicSiamese) -> Dict[str, tf.Tensor]:
    optimizer = tf.train.AdamOptimizer()
    tensors = {
        'loss': siamese_model.loss(),
        'classification_loss': siamese_model.classification_loss(),
        'regularization_loss': siamese_model.regularization_loss(),
        'c-index': siamese_model.c_index(),
        'true-predictions': siamese_model.good_predictions_count(),
        'predictions': siamese_model.y_estimate,
        'probabilities': siamese_model.y_prob,
        'gather_a': siamese_model.gathered_a,
        'gather_b': siamese_model.gathered_b
    }

    tensors['minimize'] = optimizer.minimize(tensors['loss'])

    # Create summaries
    with tf.name_scope("summaries"):
        tf.summary.scalar("loss", tensors['loss'])
        tf.summary.scalar("c-index", tensors['c-index'])
        tf.summary.scalar("classification_loss", tensors['classification_loss'])
        tf.summary.scalar("regularization_loss", tensors['regularization_loss'])

        for var in tf.trainable_variables():
            # We have to replace `:` with `_` to avoid a warning that ends doing this replacement
            tf.summary.histogram(str(var.name).replace(":", "_"), var)

    tensors['summary'] = tf.summary.merge_all()
    logger.debug("Tensors created")
    return tensors


def get_sets_generator(cv_folds: int, test_size: int, test_mode: str) \
        -> Iterator[Tuple[int, Tuple[List[data.PairComp], List[data.PairComp]]]]:
    dataset = data.pair_data.SplitPairs()

    # Decide whether to use CV or only a single test/train sets
    if cv_folds < 2:
        generator = dataset.train_test_split(test_size, compare_train=(test_mode == "compare_train"))
        enum_generator = (0, generator)
    else:
        generator = dataset.folds(cv_folds, compare_train=(test_mode == "compare_train"))

        # Slurm configuration
        task_id = os.getenv('SLURM_ARRAY_TASK_ID', 0)
        if int(os.getenv('SLURM_ARRAY_TASK_COUNT', 0)) == cv_folds:
            task_id = int(task_id)
            logger.info(f"Task number: {task_id}")
            enum_generator = [(task_id, list(generator)[task_id])]
        else:
            enum_generator = enumerate(generator)

    return enum_generator


################
#     MAIN     #
################
def main(args: Dict[str, Any]):
    logger.info("Script to train a siamese neural network model")
    logger.info(f"Using batch size: {args['batch_size']}")

    siamese_model = select_model(args['model'], args['gpu_level'])
    tensors = get_tensors(siamese_model)

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = args['gpu_allow_growth']

    with tf.Session(config=conf) as sess:
        enum_generator = get_sets_generator(args['cv_folds'], args['test_size'], args['test_mode'])

        counts = {
            'train': {
                'total': 0,
                'correct': 0,
            },
            'test': {
                'total': 0,
                'correct': 0
            }
        }

        for i, (train_pairs, test_pairs) in enum_generator:
            # Initialize all the variables
            sess.run(tf.global_variables_initializer())

            logger.info("\r ")
            logger.info(f"New fold {i}, {len(train_pairs)} train pairs, {len(test_pairs)} test pairs")

            summaries_dir = os.path.join(args['results_path'], 'summaries', f'fold_{i}')
            train_summary = tf.summary.FileWriter(summaries_dir, sess.graph)

            # Epoch iterations
            train_iterations(sess,
                             siamese_model,
                             tensors,
                             train_pairs,
                             train_summary,
                             args['batch_size'],
                             args['num_epochs'])

            # Get test error on the training set
            logger.info("Computing train c-index")
            train_correct, train_total, train_results = \
                test_iterations(sess, siamese_model, tensors, train_pairs, args['batch_size'])

            # Run the test iterations after all the epochs
            logger.info("Computing test c-index")
            test_correct, test_total, test_results = \
                test_iterations(sess, siamese_model, tensors, test_pairs, args['batch_size'])

            counts['train']['total'] += train_total
            counts['train']['correct'] += train_correct
            counts['test']['total'] += test_total
            counts['test']['correct'] += test_correct

            # Save each fold in a different directory
            results_save_path = os.path.join(args['results_path'], f"fold_{i:0>2}")
            logger.info(f"Saving results at: {results_save_path}")
            utils.save_results(sess, train_results, test_results, results_save_path)

            logger.info(f"Train set c-index: {train_correct/train_total}, correct: {train_correct}, total: "
                        f"{train_total}")
            logger.info(f"Test set c-index {test_correct/test_total}, correct: {test_correct}, total: "
                        f"{test_total}")

        logger.info(f"Final train c-index: {counts['train']['correct']/counts['train']['total']}")
        logger.info(f"Final test c-index: {counts['test']['correct']/counts['test']['total']}")


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
        choices=['SimpleImageSiamese', 'ImageScalarSiamese', 'ScalarOnlySiamese'],
        type=str
    )

    parser.add_argument(
        "--results-path",
        help="Path where the results and the model should be saved",
        default=settings.SESSION_SAVE_PATH,
        type=str
    )

    parser.add_argument(
        "--test-mode",
        help="When testing the results test one individual against the train set or against the other members of the "
             "test set",
        default="compare_test",
        choices=["compare_test", "compare_train"],
        type=str
    )

    # See if we are running in a SLURM task array
    array_id = os.getenv('SLURM_ARRAY_TASK_ID', 0)

    arguments, unknown = parser.parse_known_args()
    arguments = vars(arguments)

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
