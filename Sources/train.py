import argparse
import os
from typing import Dict, List

import tensorflow as tf

import data
import models
import settings
import utils

logger = utils.init_logger('train')


def main(args):
    dataset = data.pair_data.SplitPairs()
    dataset.print_pairs()

    siamese_model = models.Siamese()
    optimizer = tf.train.AdamOptimizer()

    tensors = {
        'loss': siamese_model.loss(),
        'c-index': siamese_model.c_index(),
        'true-predictions': siamese_model.good_predictions_count(),
    }

    tensors['minimize'] = optimizer.minimize(tensors['loss'])

    train_pairs = list(dataset.train_pairs())
    test_pairs = list(dataset.test_pairs())

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True

    # Create summaries
    with tf.name_scope("summaries"):
        tf.summary.scalar("loss", tensors['loss'])
        tf.summary.scalar("c-index", tensors['c-index'])

    tensors['summary'] = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(settings.SUMMARIES_DIR)

    saver = tf.train.Saver()

    with tf.Session(config=conf) as sess:
        if os.path.exists(settings.SESSION_SAVE_PATH) and not args.overwrite_weights:
            saver.restore(sess, settings.SESSION_SAVE_PATH)
            logger.info("Previous weights found and loaded")
        else:
            sess.run(tf.global_variables_initializer())

        logger.info("Starting training")

        for i in range(settings.NUM_EPOCHS):
            logger.info(f"Epoch: {i + 1}")

            train_iterations(saver, sess, siamese_model, tensors, train_pairs, summary_writer)
            test_iterations(sess, siamese_model, tensors, test_pairs)


def train_iterations(saver: tf.train.Saver, sess: tf.Session, model, tensors: Dict[str, tf.Tensor],
                     pairs: List[data.PairComp], summary_writer: tf.summary.FileWriter):

    total_pairs = len(pairs)*settings.TOTAL_ROTATIONS
    logger.info(f"We have {total_pairs} pairs")
    # Train iterations
    for j, batch in enumerate(data.BatchData.batches(pairs, batch_size=settings.DATA_BATCH_SIZE)):
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
        logger.info(f"Batch: {j}, size: {len(batch.pairs_a)}, remaining pairs: {total_pairs}, "
                    f"c-index: {c_index_result}, loss: {loss}")

        logger.debug("Saving weights")
        saver.save(sess, settings.SESSION_SAVE_PATH)
        summary_writer.add_summary(summary)


def test_iterations(sess: tf.Session, model: models.Siamese, tensors: Dict[str, tf.Tensor], pairs: List[data.PairComp]):
    # After we iterate over all the data inspect the test error
    total_pairs = len(pairs)*settings.TOTAL_ROTATIONS
    correct_count = 0  # To store correct predictions

    # Test iterations
    for j, batch in enumerate(data.BatchData.batches(pairs, batch_size=settings.DATA_BATCH_SIZE)):
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
        logger.info(f"Batch: {j}, size: {len(batch.pairs_a)}, remaining pairs: {total_pairs}, "
                    f"c-index: {c_index_result}")
    logger.info(f"Final c-index: {correct_count/(len(pairs)*settings.TOTAL_ROTATIONS)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fit the data with a Tensorflow model")
    parser.add_argument(
        "--overwrite_weights",
        help="Overwrite the weights that have been stored between each iteration",
        default=False,
        action="store_true"
    )

    arguments = parser.parse_args()
    try:
        # For now the arguments are ignored
        main(arguments)
    except KeyboardInterrupt:
        logger.info("\n----------------------------------")
        logger.info("Stopping due to keyboard interrupt")
        logger.info("THANKS FOR THE RIDE 😀")
