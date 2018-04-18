import argparse

import tensorflow as tf

import data
import models
import settings


def main(args):
    dataset = data.pair_data.SplitPairs()
    dataset.print_pairs()
    batch_data = data.pair_data.BatchData()

    siamese_model = models.Siamese()

    optimizer = tf.train.AdamOptimizer()
    loss_tensor = siamese_model.loss()
    minimize_step = optimizer.minimize(loss_tensor)
    train_pairs = list(dataset.train_pairs())
    print(f"We have {len(train_pairs)} pairs")

    c_index = siamese_model.c_index()
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True

    with tf.Session(config=conf) as sess:
        sess.run(tf.global_variables_initializer())

        print("Starting training")

        for i, batch in enumerate(batch_data.batches(train_pairs, batch_size=7)):
            # print(batch.images)
            print(f"Training batch {i}, pairs: {len(batch.pairs_a)}")

            _, c_index_result, loss = sess.run([minimize_step, c_index, loss_tensor], feed_dict={
                siamese_model.x: batch.images,
                siamese_model.pairs_a: batch.pairs_a,
                siamese_model.pairs_b: batch.pairs_b,
                siamese_model.y: batch.labels
            })

            print(f"Batch: {i}, c-index: {c_index_result}, loss: {loss}")
            # break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fit the data with a Tensorflow model")
    parser.add_argument(
        "--datasets_dir",
        help="Directory where all the datasets can be found",
        default="$HOME/Documents/Datasets"
    )
    parser.add_argument(
        "--input_dir",
        help="Directory inside datasets_dir where the desired dataset is found",
        default="HNK_processed"
    )

    arguments = parser.parse_args()
    try:
        # For now the arguments are ignored
        main(arguments)
    except KeyboardInterrupt:
        print()
        print("----------------------------------")
        print("Stopping due to keyboard interrupt")
        print("THANKS FOR THE RIDE 😀")
