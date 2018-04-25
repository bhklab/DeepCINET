import argparse
import os
from typing import Dict, List

import tensorflow as tf

import data
import models
import settings
import utils

logger = utils.init_logger('train')


def train_iterations(saver: tf.train.Saver, sess: tf.Session, model: models.BasicSiamese, tensors: Dict[str, tf.Tensor],
                     pairs: List[data.PairComp], summary_writer: tf.summary.FileWriter):

    total_pairs = len(pairs)*settings.TOTAL_ROTATIONS
    logger.info(f"We have {total_pairs} pairs")
    # Train iterations
    for i, batch in enumerate(data.BatchData.batches(pairs, batch_size=settings.args.batch_size)):
        # Execute graph operations
        _, c_index_result, loss, summary = sess.run(
            [tensors['minimize'], tensors['c-index'], tensors['loss'], tensors['summary']],
            feed_dict={
                model.x: batch.images,
                model.pairs_a: batch.pairs_a,
                model.pairs_b: batch.pairs_b,
                model.y: batch.labels
            })

        total_pairs -= len(batch.pairs_a)
        logger.info(f"Batch: {i}, size: {len(batch.pairs_a)}, remaining pairs: {total_pairs}, "
                    f"c-index: {c_index_result}, loss: {loss}")

        saver.save(sess, settings.SESSION_SAVE_PATH)
        summary_writer.add_summary(summary, i)


def test_iterations(sess: tf.Session, model: models.BasicSiamese, tensors: Dict[str, tf.Tensor],
                    pairs: List[data.PairComp]):
    # After we iterate over all the data inspect the test error
    total_pairs = len(pairs)*settings.TOTAL_ROTATIONS
    correct_count = 0  # To store correct predictions
    pairs_count = 0

    # Test iterations
    for i, batch in enumerate(data.BatchData.batches(pairs, batch_size=settings.args.batch_size)):
        # Execute test operations
        temp_sum, c_index_result = sess.run(
            [tensors['true-predictions'], tensors['c-index']],
            feed_dict={
                model.x: batch.images,
                model.pairs_a: batch.pairs_a,
                model.pairs_b: batch.pairs_b,
                model.y: batch.labels
            })

        correct_count += temp_sum
        total_pairs -= len(batch.pairs_a)
        pairs_count += len(batch.pairs_a)

        logger.info(f"Batch: {i}, size: {len(batch.pairs_a)}, remaining pairs: {total_pairs}, "
                    f"c-index: {c_index_result}, accum c-index:{correct_count/pairs_count}")
    logger.info(f"Final c-index: {correct_count/(len(pairs)*settings.TOTAL_ROTATIONS)}")


def main():
    logger.info(f"Using batch size: {settings.args.batch_size}")

    siamese_model = models.BasicSiamese(settings.args.gpu_level)
    optimizer = tf.train.AdamOptimizer()

    tensors = {
        'loss': siamese_model.loss(),
        'c-index': siamese_model.c_index(),
        'true-predictions': siamese_model.good_predictions_count(),
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

    saver = tf.train.Saver()

    with tf.Session(config=conf) as sess:
        train_summary = tf.summary.FileWriter(os.path.join(settings.SUMMARIES_DIR, 'train'), sess.graph)

        # Load the weights from the previous execution if we can
        if os.path.exists(settings.SESSION_SAVE_PATH) and not settings.args.overwrite_weights:
            saver.restore(sess, settings.SESSION_SAVE_PATH)
            logger.info("Previous weights found and loaded")
        else:
            sess.run(tf.global_variables_initializer())

        logger.info("Starting training")

        dataset = data.pair_data.SplitPairs()

        # Decide whether to use CV or only a single test/train sets
        if settings.args.cv_folds < 2:
            generator = [dataset.train_test_split(settings.args.test_size)]
        else:
            generator = dataset.folds(settings.args.cv_folds)

        for train_pairs, test_pairs in generator:
            for j in range(settings.args.num_epochs):
                logger.info(f"Epoch: {j + 1} of {settings.args.num_epochs}")

                train_iterations(saver, sess, siamese_model, tensors, train_pairs, train_summary)
                test_iterations(sess, siamese_model, tensors, test_pairs)


if __name__ == '__main__':
    logger.debug("Script starts")
    parser = argparse.ArgumentParser(
        description="Fit the data with a Tensorflow model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--overwrite-weights",
        help="Overwrite the weights that have been stored between each iteration",
        default=False,
        action="store_true"
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
        "--num-epochs",
        help="Number of epochs to use when training. Times passed through the entire dataset",
        default=1
    )

    parser.add_argument(
        "--batch-size",
        help="Batch size for each train iteration",
        default=20,
        type=int
    )

    args = settings.add_args(parser)

    if args.batch_size < 2:
        logger.error("Batch size is too small! It should be at least 2. Exiting")
        exit(1)

    try:
        # For now the arguments are ignored
        main()
    except KeyboardInterrupt:
        logger.info("\n----------------------------------")
        logger.info("Stopping due to keyboard interrupt")
        logger.info("THANKS FOR THE RIDE 😀")
