import tensorflow as tf

import settings
import utils


logger = utils.get_logger('train.siamese')


class Siamese:

    def __init__(self):
        with tf.device('/cpu:0'):
            self.x = tf.placeholder(tf.float32, [None, 64, 64, 64, 1])
            self.y = tf.placeholder(tf.float32, [None])
            self.y = tf.reshape(self.y, [-1, 1])
            self.pairs_a = tf.placeholder(tf.int32, [None])
            self.pairs_b = tf.placeholder(tf.int32, [None])

            self.batch_size = tf.cast(tf.shape(self.y)[0], tf.float32)

        self.sister_out = self.sister(self.x)

        with tf.device('/cpu:0'):
            self.gathered_a = tf.gather(self.sister_out, self.pairs_a)
            self.gathered_b = tf.gather(self.sister_out, self.pairs_b)

            self.y_prob = tf.sigmoid(self.gathered_a - self.gathered_b)
            self.y_estimate = tf.round(self.y_prob)

    @staticmethod
    def sister(x: tf.Tensor) -> tf.Tensor:
        """
        Build one branch (sister) of the siamese network

        :param x: Initial input of shape ``[batch, 64, 64, 64, 1]``
        :return: Tensor of shape ``[batch, 1]``
        """
        # In: [batch, 64, 64, 64, 1]

        device = '/gpu:0' if settings.USE_GPU >= 2 else '/cpu:0'
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

        device = '/gpu:0' if settings.USE_GPU >= 1 else '/cpu:0'
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

        device = '/gpu:0' if settings.USE_GPU >= 3 else '/cpu:0'
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
            )

            # Out: [batch, 1]
            x = tf.layers.dense(
                x,
                1
            )

        return x

    def loss(self) -> tf.Tensor:
        return tf.losses.log_loss(self.y, self.y_prob)

    def good_predictions_count(self) -> tf.Tensor:
        """
        Return the count of elements that have been a good prediction

        :return:
        """
        # y ∈ {0, 1}   y_estimate ∈ {0, 1}
        # The bad predictions are the ones that are not equal so if we subtract one with
        # the other it should give us a result != 0, then counting the bad predictions
        # is only a fact of summing all the bad values

        equals = tf.equal(self.y, self.y_estimate)
        bad_predictions_count = tf.cast(tf.count_nonzero(equals), tf.float32)
        out = self.batch_size - bad_predictions_count

        # Conditions that should always be met, a bit ugly but it seems that Tensorflow
        # does not have any other method
        assert_op1 = tf.assert_less_equal(self.y_estimate, 1.)
        assert_op2 = tf.assert_greater_equal(self.y_estimate, 0.)
        with tf.control_dependencies([assert_op1, assert_op2]):
            out = tf.identity(out)
        return out

    def c_index(self):
        return self.good_predictions_count()/self.batch_size


