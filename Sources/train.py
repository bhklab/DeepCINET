import argparse
import os
from typing import Dict, List

import tensorflow as tf

import data
import models
import settings
import utils

logger = utils.init_logger('train')


def main():
    args = settings.args
    siamese_model = models.Siamese(args.gpu_level)
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
            tf.summary.histogram(var.name, var)

    tensors['summary'] = tf.summary.merge_all()

    saver = tf.train.Saver()

    with tf.Session(config=conf) as sess:
        train_summary = tf.summary.FileWriter(os.path.join(settings.SUMMARIES_DIR, 'train'), sess.graph)

        # Load the weights from the previous execution if we can
        if os.path.exists(settings.SESSION_SAVE_PATH) and not args.overwrite_weights:
            saver.restore(sess, settings.SESSION_SAVE_PATH)
            logger.info("Previous weights found and loaded")
        else:
            sess.run(tf.global_variables_initializer())

        logger.debug(f"Batch size: {settings.DATA_BATCH_SIZE}")
        logger.debug(f"Num Epochs: {settings.NUM_EPOCHS}")
        logger.info("Starting training")

        dataset = data.pair_data.SplitPairs()
        for test_pairs, train_pairs in dataset.folds(args.cv_folds):
            for j in range(settings.NUM_EPOCHS):
                logger.info(f"Epoch: {j + 1}")

                train_iterations(saver, sess, siamese_model, tensors, train_pairs, train_summary)
                test_iterations(sess, siamese_model, tensors, test_pairs)


def train_iterations(saver: tf.train.Saver, sess: tf.Session, model: models.Siamese, tensors: Dict[str, tf.Tensor],
                     pairs: List[data.PairComp], summary_writer: tf.summary.FileWriter):

    total_pairs = len(pairs)*settings.TOTAL_ROTATIONS
    logger.info(f"We have {total_pairs} pairs")
    # Train iterations
    for i, batch in enumerate(data.BatchData.batches(pairs, batch_size=settings.DATA_BATCH_SIZE)):
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


def test_iterations(sess: tf.Session, model: models.Siamese, tensors: Dict[str, tf.Tensor],
                    pairs: List[data.PairComp]):
    # After we iterate over all the data inspect the test error
    total_pairs = len(pairs)*settings.TOTAL_ROTATIONS
    correct_count = 0  # To store correct predictions
    pairs_count = 0

    # Test iterations
    for i, batch in enumerate(data.BatchData.batches(pairs, batch_size=settings.DATA_BATCH_SIZE)):
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


if __name__ == '__main__':
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
        help="Number of cross validation folds",
        default=1,
        type=int
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
        help="Number of epochs to use when training. Times passed through the entire dataset"
    )

    settings.add_args(parser)
    try:
        # For now the arguments are ignored
        main()
    except KeyboardInterrupt:
        logger.info("\n----------------------------------")
        logger.info("Stopping due to keyboard interrupt")
        logger.info("THANKS FOR THE RIDE 😀")
