import argparse
import os

import tensorflow as tf

import data
import models
import settings
import utils

logger = utils.init_logger('train')


def main(args):
    dataset = data.pair_data.SplitPairs()
    dataset.print_pairs()
    batch_data = data.pair_data.BatchData()

    siamese_model = models.Siamese()

    tensor_loss = siamese_model.loss()
    tensor_c_index = siamese_model.c_index()
    tensor_true_sum = siamese_model.good_predictions_count()

    optimizer = tf.train.AdamOptimizer()
    tensor_minimize = optimizer.minimize(tensor_loss)

    train_pairs = list(dataset.train_pairs())
    test_pairs = list(dataset.test_pairs())

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True

    # Create summaries
    with tf.name_scope("summaries"):
        tf.summary.scalar("loss", tensor_loss)
        tf.summary.scalar("c-index", tensor_c_index)
        tf.summary.histogram("loss", tensor_loss)
        tf.summary.histogram("c-index", tensor_c_index)

    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(settings.SUMMARIES_DIR)

    saver = tf.train.Saver()

    with tf.Session(config=conf) as sess:
        if os.path.exists(settings.SESSION_SAVE_PATH) and not args.overwrite_weights:
            saver.restore(sess, settings.SESSION_SAVE_PATH)
        else:
            sess.run(tf.global_variables_initializer())

        logger.info("Starting training")

        for i in range(settings.NUM_EPOCHS):

            total_pairs = len(train_pairs)
            logger.info(f"Epoch: {i + 1}\tWe have {total_pairs} pairs")

            # Train iterations
            for j, batch in enumerate(batch_data.batches(train_pairs, batch_size=settings.DATA_BATCH_SIZE)):
                # Execute graph operations
                _, c_index_result, loss, summary = sess.run(
                    [tensor_minimize, tensor_c_index, tensor_loss, summary_op],
                    feed_dict={
                        siamese_model.x: batch.images,
                        siamese_model.pairs_a: batch.pairs_a,
                        siamese_model.pairs_b: batch.pairs_b,
                        siamese_model.y: batch.labels
                    })

                total_pairs -= len(batch.pairs_a)/settings.TOTAL_ROTATIONS
                logger.info(f"Batch: {j}, size: {len(batch.pairs_a)}, remaining pairs: {total_pairs}, "
                            f"c-index: {c_index_result}, loss: {loss}")

                logger.debug("Saving weights")
                saver.save(sess, settings.SESSION_SAVE_PATH)
                train_writer.add_summary(summary)

            # After we iterate over all the data inspect the test error
            total_pairs = len(test_pairs)
            correct_count = 0  # To store correct predictions

            # Test iterations
            for j, batch in enumerate(batch_data.batches(test_pairs, batch_size=settings.DATA_BATCH_SIZE)):
                # Execute test operations
                temp_sum, c_index_result = sess.run(
                    [tensor_true_sum, tensor_c_index],
                    feed_dict={
                        siamese_model.x: batch.images,
                        siamese_model.pairs_a: batch.pairs_a,
                        siamese_model.pairs_b: batch.pairs_b,
                        siamese_model.y: batch.labels
                    })

                correct_count += temp_sum
                total_pairs -= len(batch.pairs_a)/settings.TOTAL_ROTATIONS
                logger.info(f"Batch: {j}, size: {len(batch.pairs_a)}, remaining pairs: {total_pairs}, "
                            f"c-index: {c_index_result}")

            logger.info(f"Final c-index: {correct_count/len(test_pairs)}")


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
