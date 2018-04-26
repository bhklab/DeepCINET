import abc

import tensorflow as tf

import utils

logger = utils.get_logger('train.siamese')


class BasicSiamese:
    """
    Class representing a basic siamese structure. It contains a few convolutional layers and then the
    contrastive loss.

    The model has some tensors that need to be fed when using ``sess.run(...)``:

    :var BasicSiamese.x_image: Batch of input images, has shape ``[batch, 64, 64, 64, 1]``
    :var BasicSiamese.y: Batch of labels for all the pairs with shape ``[batch]``
    :var BasicSiamese.pairs_a: Indices to be selected as pairs A for the batch of input images, has shape ``[batch]``
    :var BasicSiamese.pairs_b: Indices to be selected as pairs B for the batch of input images, has shape ``[batch]``
    :vartype BasicSiamese.x_image: tf.Tensor
    :vartype BasicSiamese.y: tf.Tensor
    :vartype BasicSiamese.pairs_a: tf.Tensor
    :vartype BasicSiamese.pairs_b: tf.Tensor
    """

    #: Threshold in when considering a float number between ``0`` and ``1`` a :any:`True` value for classification
    THRESHOLD = .5

    def __init__(self, gpu_level: int = 0):
        """
        Construct a BasicSiamese model.

        :param gpu_level: Amount of GPU to be used with the model
        """
        self._gpu_level = gpu_level

        device = '/gpu:0' if self._gpu_level >= 3 else '/cpu:0'
        logger.debug(f"Using device: {device} for parameters")
        with tf.device(device):
            # The input size for the images is 64x64x64x1

            #: **Attribute**: Placeholder for the image input, it has shape ``[batch, 64, 64, 64, 1]``
            self.x_image = tf.placeholder(tf.float32, [None, 64, 64, 64, 1], name="X")

            #: **Attribute**: Placeholder for the labels, it has shape ``[batch]``
            self.y = tf.placeholder(tf.float32, [None], name="Y")
            self.y = tf.reshape(self.y, [-1, 1], name="Y_reshape")

            #: **Attribute**: Placeholder for the indices of the first pairs (A)
            self.pairs_a = tf.placeholder(tf.int32, [None], name="pairs_a")

            #: **Attribute**: Placeholder for the indices of the second pairs (B)
            self.pairs_b = tf.placeholder(tf.int32, [None], name="pairs_b")

        self._sister_out = self._sister(self.x_image)

        #: **Attribute**: Probability of :math:`\hat{y} = 1`
        self.y_prob = self._contrastive_loss(self._sister_out)

        #: **Attribute**: Estimation of :math:`\hat{y}` by using :any:`BasicSiamese.y_prob` and
        #: :any:`BasicSiamese.THRESHOLD`
        self.y_estimate = tf.greater_equal(self.y_prob, self.THRESHOLD)

    def _sister(self, x: tf.Tensor) -> tf.Tensor:
        """
        Build one branch (sister) of the siamese network

        :param x: Initial input of shape ``[batch, 64, 64, 64, 1]``
        :return: Tensor of shape ``[batch, 1]``
        """
        # In: [batch, 64, 64, 64, 1]
        # Out: [batch, 25, 25, 25, 50]
        x = self._conv_layers(x)

        # In: [batch, 25, 25, 25, 50]
        # Out: [batch, 1]
        x = self._fc_layers(x)

        return x

    @abc.abstractmethod
    def _conv_layers(self, x: tf.Tensor) -> tf.Tensor:
        """
        Implement this method to create the tensors for the Convolutional layers of the network

        :param x: Network's input images with shape ``[batch, 64, 64, 64, 1]``
        :return: Filtered image with the convolutions applied
        """

    @abc.abstractmethod
    def _fc_layers(self, x: tf.Tensor) -> tf.Tensor:
        """
        Implement this method to create the tensors for the Fully Connected layers of the network

        :param x: Image, usually previously filtered with the convolutional layers.
        :return: Tensor with shape ``[batch, 1]``
        """

    def _contrastive_loss(self, sister_out: tf.Tensor):
        r"""
        Implement the loss to compare the two sister networks. To get the pairs to be compared it uses the
        :any:`BasicSiamese.pairs_a` and :any:`BasicSiamese.pairs_b`. In this case the contrastive loss is as follows:

        .. math::
            G_W(\boldsymbol{X_A}) &:= \text{Outputs for inputs A} \\
            G_W(\boldsymbol{X_B}) &:= \text{Outputs for inputs B} \\
            \sigma(x) &:= \frac{1}{1 + \exp(-x)} \\
            \boldsymbol{\hat{y}} &= \sigma(G_W(\boldsymbol{X_A}) - G_W(\boldsymbol{X_B})) =
            \frac{1}{1 + \exp(G_W(\boldsymbol{X_B}) - G_W(\boldsymbol{X_A}))}

        :param sister_out: Sister's network output, then using the defined parameters  it selects the proper pairs to
                           be compared.
        :return: Tensor with the contrastive loss, comparing the two sister's output.
        """
        device = '/gpu:0' if self._gpu_level >= 3 else '/cpu:0'
        logger.debug(f"Using device: {device} for contrastive loss")
        with tf.device(device):
            gathered_a = tf.gather(sister_out, self.pairs_a, name="contrastive_gather_a")
            gathered_b = tf.gather(sister_out, self.pairs_b, name="contrastive_gather_b")

            sub = tf.subtract(gathered_a, gathered_b, name="contrastive_sub")
            return tf.sigmoid(sub, name="contrastive_sigmoid")

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

    def c_index(self) -> tf.Tensor:
        r"""
        Create the tensor for the c-index. It's obtained by counting the number of comparisons that are right
        and dividing them by the total amount of comparisons, it's as follows:

        .. math::
            \frac{\text{correct comparisons}}{\text{total comparisons}}

        :return: c-index tensor
        """
        batch_size = tf.cast(tf.shape(self.y)[0], tf.float32, name="batch_size_cast")
        return self.good_predictions_count()/batch_size


class SimpleSiamese(BasicSiamese):
    """
    Class representing the initial and simple siamese structure used for the first steps of the project. It
    inherits :any:`BasicSiamese` so it has the same tensors to be fed.

    **Convolutional Model**:

    It contains 4 convolutional layers and 3 FC layers

        - :math:`3^3` kernel with 30 filters and stride = 2
        - :math:`3^3` kernel with 40 filters and stride = 1
        - :math:`3^3` kernel with 40 filters and stride = 1
        - :math:`3^3` kernel with 50 filters and stride = 1
        - 100 units, activation ReLU
        - 50 units, activation ReLu
        - 1 unit, activation ReLu
    """

    def __init__(self, gpu_level: int = 0):
        """
        Construct a new SimpleSiamese class

        :param gpu_level: Amount of GPU to be used with the model

                            0. No GPU usage
                            1. Only second conv layers
                            2. All conv layers
                            3. All layers and parameters are on the GPU
        """
        super().__init__(gpu_level)

    def _conv_layers(self, x: tf.Tensor) -> tf.Tensor:
        """
        Implementation of abstract method :func:`~BasicSiamese._conv_layers`

        :param x: Network's input images with shape ``[batch, 64, 64, 64, 1]``
        :return: Filtered image with the convolutions applied
        """
        # In: [batch, 64, 64, 64, 1]

        device = '/gpu:0' if self._gpu_level >= 2 else '/cpu:0'
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

        device = '/gpu:0' if self._gpu_level >= 1 else '/cpu:0'
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
        return x

    def _fc_layers(self, x: tf.Tensor) -> tf.Tensor:
        """
        Implementation of abstract method ``BasicSiamese._fc_layers``

        :param x: Image, usually previously filtered with the convolutional layers.
        :return: Tensor with shape ``[batch, 1]``
        """
        device = '/gpu:0' if self._gpu_level >= 3 else '/cpu:0'
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

