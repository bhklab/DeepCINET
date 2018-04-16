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
    train_pairs = list(dataset.train_pairs())[:1000]
    print(f"We have {len(train_pairs)} pairs")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i, batch in enumerate(batch_data.batches(train_pairs)):
            # print(batch.images)

            _, loss = sess.run([minimize_step, loss_tensor], feed_dict={
                siamese_model.x: batch.images,
                siamese_model.pairs_a: batch.pairs_a,
                siamese_model.pairs_b: batch.pairs_b,
                siamese_model.y: batch.labels
            })

            print(f"Batch: {i}, loss: {loss}")


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
