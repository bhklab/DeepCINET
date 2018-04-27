import argparse
import os
from typing import Dict, List, Tuple

import tensorflow as tf
import numpy as np
import pandas as pd

import data
import models
import settings
import utils

logger = utils.init_logger('train')


def train_iterations(sess: tf.Session, model: models.BasicSiamese, tensors: Dict[str, tf.Tensor],
                     pairs: List[data.PairComp], summary_writer: tf.summary.FileWriter):

    total_pairs = len(pairs)*settings.TOTAL_ROTATIONS

    # Train iterations
    for i, batch in enumerate(data.BatchData.batches(pairs, batch_size=settings.args['batch_size'])):
        # Execute graph operations
        _, c_index_result, loss, summary = sess.run(
            [tensors['minimize'], tensors['c-index'], tensors['loss'], tensors['summary']],
            feed_dict=model.feed_dict(batch)
        )

        total_pairs -= len(batch.pairs_a)
        logger.info(f"Batch: {i}, size: {len(batch.pairs_a)}, remaining: {total_pairs}, "
                    f"c-index: {c_index_result:.3}, loss: {loss:.3}")

        summary_writer.add_summary(summary, i)


def test_iterations(sess: tf.Session, model: models.BasicSiamese, tensors: Dict[str, tf.Tensor],
                    pairs: List[data.PairComp]) -> Tuple[int, int, pd.DataFrame]:
    # After we iterate over all the data inspect the test error
    total_pairs = len(pairs)*settings.TOTAL_ROTATIONS
    correct_count = 0  # To store correct predictions
    pairs_count = 0

    result_data = {
        'pairs_a': [],
        'pairs_b': [],
        'labels': np.array([], dtype=bool),
        'predictions': np.array([], dtype=bool)
    }

    # Test iterations
    for i, batch in enumerate(data.BatchData.batches(pairs, batch_size=settings.args['batch_size'])):
        # Execute test operations
        temp_sum, c_index_result, predictions = sess.run(
            [tensors['true-predictions'], tensors['c-index'], tensors['predictions']],
            feed_dict=model.feed_dict(batch)
        )

        correct_count += temp_sum
        total_pairs -= len(batch.pairs_a)
        pairs_count += len(batch.pairs_a)

        # Save results
        result_data['pairs_a'] += [batch.ids_inverse[idx] for idx in batch.pairs_a]
        result_data['pairs_b'] += [batch.ids_inverse[idx] for idx in batch.pairs_b]
        result_data['labels'] = np.append(result_data['labels'], np.array(batch.labels).astype(bool))
        result_data['predictions'] = np.append(result_data['predictions'], np.array(predictions).astype(bool))

        logger.info(f"Batch: {i}, size: {len(batch.pairs_a)}, remaining: {total_pairs}, "
                    f"c-index: {c_index_result:.3}, accum c-index:{correct_count/pairs_count:.3}")

    return correct_count, len(pairs)*settings.TOTAL_ROTATIONS, pd.DataFrame(result_data)


def main():
    logger.info("Script to train a siamese neural network model")
    logger.info(f"Using batch size: {settings.args['batch_size']}")

    if settings.args['model'] == "SimpleSiamese":
        siamese_model = models.SimpleSiamese(settings.args['gpu_level'])
    elif settings.args['model'] == "ScalarSiamese":
        siamese_model = models.ScalarSiamese(settings.args['gpu_level'])
    else:
        logger.error(f"Unknown option for model {settings.args['model']}")
        siamese_model = None  # Make linter happy
        exit(1)

    optimizer = tf.train.AdamOptimizer()

    tensors = {
        'loss': siamese_model.loss(),
        'c-index': siamese_model.c_index(),
        'true-predictions': siamese_model.good_predictions_count(),
        'predictions': siamese_model.y_estimate
    }

    tensors['minimize'] = optimizer.minimize(tensors['loss'])

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True

    # Create summaries
    with tf.name_scope("summaries"):
        tf.summary.scalar("loss", tensors['loss'])
        tf.summary.scalar("c-index", tensors['c-index'])

        for var in tf.trainable_variables():
            tf.summary.histogram(str(var.name).replace(":", "_"), var)

    tensors['summary'] = tf.summary.merge_all()

    logger.debug("Tensors created")

    tf.set_random_seed(settings.RANDOM_SEED)
    with tf.Session(config=conf) as sess:
        train_summary = tf.summary.FileWriter(os.path.join(settings.SUMMARIES_DIR, 'train'), sess.graph)

        # TODO: Load the weights when it's required to show the predictions only

        dataset = data.pair_data.SplitPairs()

        # Decide whether to use CV or only a single test/train sets
        if settings.args['cv_folds'] < 2:
            generator = [dataset.train_test_split(settings.args['test_size'])]
        else:
            generator = dataset.folds(settings.args['cv_folds'])

        for train_pairs, test_pairs in generator:
            # Initialize all the variables
            sess.run(tf.global_variables_initializer())

            logger.info("")
            logger.info(f"New fold {len(train_pairs)} train pairs, {len(test_pairs)} test pairs")

            # Epoch iterations
            for j in range(settings.args['num_epochs']):
                logger.info(f"Epoch: {j + 1} of {settings.args['num_epochs']}")
                train_iterations(sess, siamese_model, tensors, train_pairs, train_summary)

            # Get test error on the training set
            logger.info("Computing train error")
            train_correct, train_total, train_results = test_iterations(sess, siamese_model, tensors, train_pairs)
            logger.info(f"Train set error: {train_correct/train_total}")

            # Run the test iterations after all the epochs
            logger.info("Computing test error")
            test_count, test_total, test_results = test_iterations(sess, siamese_model, tensors, test_pairs)
            logger.info(f"Test set error {test_count/test_total}")

            utils.save_model(sess, train_results, test_results)


if __name__ == '__main__':
    logger.debug("Script starts")
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
        "-n, --num-epochs",
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
        default="SimpleSiamese",
        choices=['SimpleSiamese', 'ScalarSiamese'],
        type=str
    )

    args = settings.add_args(parser)
    logger.debug(args)

    if args['batch_size'] < 2:
        logger.error("Batch size is too small! It should be at least 2. Exiting")
        exit(1)

    try:
        # For now the arguments are ignored
        main()
    except KeyboardInterrupt:
        logger.info("\n----------------------------------")
        logger.info("Stopping due to keyboard interrupt")
        logger.info("THANKS FOR THE RIDE 😀")
