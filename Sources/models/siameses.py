import abc
from typing import Dict

import tensorflow as tf

import utils
import data
import settings

logger = utils.get_logger('train.siamese')


class BasicModel:
    """
    Simple class to build a classification model.

    :var BasicModel.y: Batch of labels for all the pairs with shape ``[batch]``
    :var BasicModel.y_prob: Tensor with the probabilities of single class classification
    :var BasicModel.y_estimate: Tensor with the classification, derived from :any:`BasicModel.y_prob`
    :vartype BasicModel.y: tf.Tensor
    :vartype BasicModel.y_prob: tf.Tensor
    :vartype BasicModel.y_estimate: tf.Tensor
    """
    #: Threshold in when considering a float number between ``0`` and ``1`` a :any:`True` value for classification
    THRESHOLD = .5

    def __init__(self):
        """
        Construct a BasicModel. This model is a basic structure to create a classification model.
        """

        #: **Attribute**: Placeholder for the labels, it has shape ``[batch]``
        self.y = tf.placeholder(tf.float32, [None], name="Y")
        self._y = tf.reshape(self.y, [-1, 1], name="Y_reshape")

        #: **Attribute**: Probability of :math:`\hat{y} = 1`
        self.y_prob = self._model()

        #: **Attribute**: Estimation of :math:`\hat{y}` by using :any:`BasicSiamese.y_prob` and
        #: :any:`BasicSiamese.THRESHOLD`
        self.y_estimate = tf.greater_equal(self.y_prob, self.THRESHOLD)

    @abc.abstractmethod
    def _model(self) -> tf.Tensor:
        """
        Abstract method, the model should be build inside this method. Classes that Inherit :any:`BasicModel`
        should implement this method to create the model

        :return: Tensor with shape ``[batch]`` with the probability of single class classification.
        """

    def feed_dict(self, batch: data.PairBatch) -> Dict:
        """
        Get the ``feed_dict`` required by Tensorflow when calling ``sess.run(...)``. Classes that inherit
        :class:`BasicModel` should reimplement this function

        :param batch: Information about the batch, usually provided by :method:`BatchData.batches`
        :return: Dictionary that can be feed to the ``feed_dict`` parameter of ``sess.run(...)``.
        """
        return {
            self.y: batch.labels
        }

    def good_predictions_count(self) -> tf.Tensor:
        """
        Return the count of elements that have been a good prediction

        :return: Tensorflow tensor with the count for good predictions. Then to get
                 the c-index we only have to divide by the batch size
        """
        # y ∈ {0, 1}   y_estimate ∈ {True, False}
        y_bool = tf.greater_equal(self._y, self.THRESHOLD)
        equals = tf.equal(y_bool, self.y_estimate)

        return tf.cast(tf.count_nonzero(equals), tf.float32)

    def loss(self) -> tf.Tensor:
        r"""
        Loss function for the model. It uses the negative log loss function:

        .. math::
            \mathcal{L}(\boldsymbol{y}, \boldsymbol{\hat{y}}) = -\frac{1}{m}
            \sum_{i = 1}^{m} \left(y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i)\right)
            \quad m := \text{batch size}

        Also, the regularization term defined by the other layers is added

        :return: Scalar tensor with the negative log loss function for the model computed.
        """
        return tf.losses.log_loss(self._y, self.y_prob) + tf.losses.get_regularization_loss()

    def c_index(self) -> tf.Tensor:
        r"""
        Create the tensor for the c-index. It's obtained by counting the number of comparisons that are right
        and dividing them by the total amount of comparisons, it's as follows:

        .. math::
            \frac{\text{correct comparisons}}{\text{total comparisons}}

        :return: c-index tensor
        """
        batch_size = tf.cast(tf.shape(self._y)[0], tf.float32, name="batch_size_cast")
        return self.good_predictions_count()/batch_size


class BasicSiamese(BasicModel):
    """
    Class representing a basic siamese structure. It contains a few convolutional layers and then the
    contrastive loss.

    :var BasicSiamese._gpu_level: Amount of GPU that should be used when evaluating the model
    :var BasicSiamese.y: Batch of labels for all the pairs with shape ``[batch]``
    :var BasicSiamese.pairs_a: Indices to be selected as pairs A for the batch of input images, has shape ``[batch]``
    :var BasicSiamese.pairs_b: Indices to be selected as pairs B for the batch of input images, has shape ``[batch]``
    :vartype BasicSiamese.y: tf.Tensor
    :vartype BasicSiamese.pairs_a: tf.Tensor
    :vartype BasicSiamese.pairs_b: tf.Tensor
    """

    def __init__(self, gpu_level: int = 0):
        """
        Construct a BasicSiamese model.

        :param gpu_level: Amount of GPU to be used with the model
        """
        #: **Attribute**: Amount of GPU to be used with the model
        self._gpu_level = gpu_level

        #: **Attribute**: Placeholder for the indices of the first pairs (A)
        self.pairs_a = tf.placeholder(tf.int32, [None], name="pairs_a")

        #: **Attribute**: Placeholder for the indices of the second pairs (B)
        self.pairs_b = tf.placeholder(tf.int32, [None], name="pairs_b")

        super().__init__()

    def _model(self) -> tf.Tensor:
        """
        Implementation of :method:`BasicModel._model`

        :return: Tensor where a Siamese network has been applied to the input
        """
        sister_out = self._sister()
        return self._contrastive_loss(sister_out)

    @abc.abstractmethod
    def _sister(self) -> tf.Tensor:
        """
        Build one branch (sister) of the siamese network

        :return: Tensor of shape ``[batch, 1]``
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

    def feed_dict(self, batch: data.PairBatch) -> Dict:
        return {
            **super().feed_dict(batch),
            self.pairs_a: batch.pairs_a,
            self.pairs_b: batch.pairs_b,
        }


class BasicImageSiamese(BasicSiamese):
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

    def __init__(self, gpu_level: int = 0):
        """
        Construct a BasicSiamese model.

        :param gpu_level: Amount of GPU to be used with the model
        """
        #: **Attribute**: Placeholder for the image input, it has shape ``[batch, 64, 64, 64, 1]``
        self.x_image = tf.placeholder(tf.float32, [None, 64, 64, 64, 1], name="X")

        super().__init__(gpu_level=gpu_level)

    def _sister(self) -> tf.Tensor:
        """
        Build one branch (sister) of the siamese network

        :return: Tensor of shape ``[batch, 1]``
        """
        # In: [batch, 64, 64, 64, 1]
        # Out: [batch, 25, 25, 25, 50]
        x = self._conv_layers(self.x_image)

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

    def feed_dict(self, batch: data.PairBatch) -> Dict:
        """
        Method to create the ``feed_dict`` required by Tensorflow when passing the data. Classes that inherit
        :class:`BasicSiamese` must re-implement this method if they use different tensors than:

            - :any:`BasicSiamese.x_image`
            - :any:`BasicSiamese.y`
            - :any:`BasicSiamese.pairs_a`
            - :any:`BasicSiamese.pairs_b`

        :param batch: Data containing for the current batch, usually this would be generated by
                      :func:`~BatchData.batches`
        :return: Return the ``feed_dict`` as a dictionary
        """
        return {
            **super().feed_dict(batch),
            self.x_image: batch.images
        }


class SimpleImageSiamese(BasicImageSiamese):
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


class ImageScalarSiamese(BasicImageSiamese):
    """
    This class creates a Siamese model that uses both images and scalar features extracted using
    PyRadiomics. The features are not extracted by the model but they have to be provided in one of the placeholders

    ** Network structure **

        - :math:`3^3` kernel with 30 filters and stride = 2 with ReLu
        - :math:`3^3` kernel with 30 filters and stride = 2 with ReLu
        - :math:`3^3` kernel with 40 filters and stride = 1 with ReLu
        - :math:`3^3` kernel with 40 filters and stride = 1 with ReLu
        - :math:`3^3` kernel with 50 filters and stride = 1 with ReLu
        - :math:`3^3` kernel with 50 filters and stride = 1 with ReLu
        - Flattening layer
        - 8000 units, activation tanh
        - 100 units, activation tanh
        - 1 unit, activation ReLu
    """

    def __init__(self, gpu_level: int = 0, regularization_factor: int = 0.001):
        """
        Initialize a ScalarSiamese model. This model uses scalar features extracted with PyRadiomics and
        provided through a CSV file in the dataset, the model assumes that there are :any:`settings.NUMBER_FEATURES`
        for each input.

        :param gpu_level: Amount of GPU that should be used with the model
        :param regularization_factor: Regularization factor for the weights
        """
        self.x_scalar = tf.placeholder(tf.float32, [None, settings.NUMBER_FEATURES])
        self._reg_factor = regularization_factor

        super().__init__(gpu_level=gpu_level)

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
            x = self._conv3d(
                x,
                filters=30,
                kernel_size=3,
                strides=2,
                name="conv1"
            )

            # Out: [batch, 15, 15, 15, 30]
            x = self._conv3d(
                x,
                filters=30,
                kernel_size=3,
                strides=2,
                name="conv2"
            )

        device = '/gpu:0' if self._gpu_level >= 1 else '/cpu:0'
        logger.debug(f"Using device: {device} for second conv layers")
        with tf.device(device):
            # Out: [batch, 13, 13, 13, 40]
            x = self._conv3d(
                x,
                filters=40,
                kernel_size=3,
                name="conv3"
            )

            # Out: [batch, 11, 11, 11, 40]
            x = self._conv3d(
                x,
                filters=40,
                kernel_size=3,
                name="conv4"
            )

            # Out: [batch, 9, 9, 9, 50]
            x = self._conv3d(
                x=x,
                filters=50,
                kernel_size=3,
                name="conv5",
            )

        return x

    def _fc_layers(self, x: tf.Tensor) -> tf.Tensor:
        """
        Implementation of abstract method :func:`~BasicSiamese._fc_layers`

        :param x: Image, usually previously filtered with the convolutional layers.
        :return: Tensor with shape ``[batch, 1]``
        """

        # In this case we will be using the same idea seen in SimpleSiamese but we will be adding the scalar
        # features instead
        device = '/gpu:0' if self._gpu_level >= 3 else '/cpu:0'
        logger.debug(f"Using device: {device} for FC layers")
        with tf.device(device):
            # Out: [batch, 9*9*9*50]
            x = tf.layers.flatten(
                x,
                name="flat"
            )

            # This is where the magic happens
            # Out: [batch, 37 175]
            x = tf.concat([x, self.x_scalar], axis=1)

            # Out: [batch, 8000]
            x = self._dense(
                x=x,
                units=8000,
                name="fc1"
            )

            # Out: [batch, 100]
            x = self._dense(
                x=x,
                units=100,
                name="fc2"
            )

            # Out: [batch, 1]
            x = self._dense(
                x=x,
                units=1,
                activation=tf.nn.relu,
                name="fc3"
            )
        return x

    def _conv3d(self, x: tf.Tensor, filters: int, kernel_size: int, name: str, strides: int = 1) -> tf.Tensor:
        return tf.layers.conv3d(
            x,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self._reg_factor),
            name=name
        )

    def _dense(self, x: tf.Tensor, units: int, name: str, activation=tf.nn.tanh) -> tf.Tensor:
        return tf.layers.dense(
            x,
            units=units,
            activation=activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self._reg_factor),
            name=name
        )

    def feed_dict(self, batch: data.PairBatch) -> Dict:
        """
        Re-implementation of :func:`~BasicSiamese.feed_dict` to create a custom dict including the scalar values

        :param batch: Data containing for the current batch, usually this would be generated by
                      :func:`~BatchData.batches`
        :return: Return the ``feed_dict`` as a dictionary
        """
        return {
            **super().feed_dict(batch),
            self.x_scalar: batch.features
        }


class ScalarOnlySiamese(BasicModel):

    def __init__(self, gpu_level: int = 0, regularization_factor: int = 0.001):
        self._gpu_level = gpu_level

        self.x_scalar = tf.placeholder(tf.float32, [None, settings.NUMBER_FEATURES])
        self._reg_factor = regularization_factor

        super().__init__()

    def _model(self):
        pass
