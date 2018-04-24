import tensorflow as tf

import settings
import utils


logger = utils.get_logger('train.siamese')


class Siamese:

    THRESHOLD = .5

    def __init__(self, gpu_level):
        self.gpu_level = gpu_level

        device = '/gpu:0' if self.gpu_level >= 3 else '/cpu:0'
        logger.debug(f"Using device: {device} for parameters")
        with tf.device(device):
            self.x = tf.placeholder(tf.float32, [None, 64, 64, 64, 1], name="X")
            self.y = tf.placeholder(tf.float32, [None], name="Y")
            self.y = tf.reshape(self.y, [-1, 1], name="Y_reshape")
            self.pairs_a = tf.placeholder(tf.int32, [None], name="pairs_a")
            self.pairs_b = tf.placeholder(tf.int32, [None], name="pairs_b")

            self.batch_size = tf.cast(tf.shape(self.y)[0], tf.float32, name="batch_size_cast")

        self.sister_out = self.sister(self.x)

        device = '/gpu:0' if self.gpu_level >= 3 else '/cpu:0'
        logger.debug(f"Using device: {device} for results")
        with tf.device(device):
            self.gathered_a = tf.gather(self.sister_out, self.pairs_a, name="gather_a")
            self.gathered_b = tf.gather(self.sister_out, self.pairs_b, name="gather_b")

            self.y_prob = tf.sigmoid(self.gathered_a - self.gathered_b, name="sigmoid")
            self.y_estimate = tf.greater_equal(self.y_prob, self.THRESHOLD)

    def sister(self, x: tf.Tensor) -> tf.Tensor:
        """
        Build one branch (sister) of the siamese network

        :param x: Initial input of shape ``[batch, 64, 64, 64, 1]``
        :return: Tensor of shape ``[batch, 1]``
        """
        # In: [batch, 64, 64, 64, 1]

        device = '/gpu:0' if self.gpu_level >= 2 else '/cpu:0'
        logger.debug(f"Using device: {device} for first conv layers")
        with tf.device(device):
            # Out: [batch, 31, 31, 31, 30]
            x = tf.layers.conv3d(
                x,
                filters=30,
                kernel_size=3,
                strides=2,
                activation=tf.nn.relu,
                name="conv1"
            )

            # Out: [batch, 29, 29, 29, 40]
            x = tf.layers.conv3d(
                x,
                filters=40,
                kernel_size=3,
                activation=tf.nn.relu,
                name="conv2"
            )

        device = '/gpu:0' if self.gpu_level >= 1 else '/cpu:0'
        logger.debug(f"Using device: {device} for second conv layers")
        with tf.device(device):
            # Out: [batch, 27, 27, 27, 40]
            x = tf.layers.conv3d(
                x,
                filters=40,
                kernel_size=3,
                activation=tf.nn.relu,
                name="conv3"
            )

            # Out: [batch, 25, 25, 25, 50]
            x = tf.layers.conv3d(
                x,
                filters=50,
                kernel_size=3,
                activation=tf.nn.relu,
                name="conv4"
            )

        device = '/gpu:0' if self.gpu_level >= 3 else '/cpu:0'
        logger.debug(f"Using device: {device} for FC layers")
        with tf.device(device):
            # Out: [batch, 25*25*25*50]
            x = tf.layers.flatten(
                x,
                name="flat"
            )

            # Out: [batch, 100]
            x = tf.layers.dense(
                x,
                100,
                activation=tf.nn.relu,
                name="fc1"
            )

            # Out: [batch, 50]
            x = tf.layers.dense(
                x,
                50,
                name="dense"
            )

            # Out: [batch, 1]
            x = tf.layers.dense(
                x,
                1,
                name="dense1"
            )

        return x

    def loss(self) -> tf.Tensor:
        return tf.losses.log_loss(self.y, self.y_prob)

    def good_predictions_count(self) -> tf.Tensor:
        """
        Return the count of elements that have been a good prediction

        :return: Tensorflow tensor with the count for good predictions. Then to get
                 the c-index we only have to divide by the batch size
        """
        # y ∈ {0, 1}   y_estimate ∈ {True, False}
        y_bool = tf.greater_equal(self.y, self.THRESHOLD)
        equals = tf.equal(y_bool, self.y_estimate)

        return tf.cast(tf.count_nonzero(equals), tf.float32)

    def c_index(self):
        return self.good_predictions_count()/self.batch_size


