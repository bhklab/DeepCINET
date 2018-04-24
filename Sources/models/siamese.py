import tensorflow as tf

import utils


logger = utils.get_logger('train.siamese')


class BasicSiamese:
    """
    Class representing a basic siamese structure. It contains a few convolutional layers and then the
    contrastive loss.

    Convolutional Model
    -------------------

    It contains 4 convolutional layers and 3 FC layers

        - 3^3 kernel with 30 filters and stride = 2
        - 3^3 kernel with 40 filters and stride = 1
        - 3^3 kernel with 40 filters and stride = 1
        - 3^3 kernel with 50 filters and stride = 1
        - 100 units, activation ReLU
        - 50 units, activation ReLu
        - 1 unit, activation ReLu
    """

    THRESHOLD = .5
    """
    Threshold in when considering a float number between ``0`` and ``1`` a ``True`` value for classification
    """

    def __init__(self, gpu_level):
        """
        Construct a BasicSiamese model.

        :param gpu_level: Amount of GPU to be used with the model
        """
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
                activation=tf.nn.relu,
                name="fc2"
            )

            # Out: [batch, 1]
            x = tf.layers.dense(
                x,
                1,
                activation=tf.nn.relu,
                name="fc3"
            )

        return x

    def loss(self) -> tf.Tensor:
        r"""
        Loss function for the model. It uses the negative log loss function:

        .. math::
            \mathcal{L}(\boldsymbol{y}, \boldsymbol{\hat{y}}) = -\frac{1}{m}
            \sum_{i = 1}^{m} \left(y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i)\right)
            \quad m := \text{batch size}

        :return: Scalar tensor with the negative log loss function for the model computed.
        """
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


