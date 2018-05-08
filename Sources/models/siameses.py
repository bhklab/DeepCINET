from typing import Dict

import tensorflow as tf

import data
import settings
from .basics import BasicImageSiamese, BasicSiamese


class SimpleImageSiamese(BasicImageSiamese):
    r"""
    Simple siamese network implementation that uses images as input

    **Convolutional Model**:

    It contains 4 convolutional layers and 3 FC layers

        - :math:`3^3` kernel with 30 filters and stride = 2
        - :math:`3^3` kernel with 40 filters and stride = 1
        - :math:`3^3` kernel with 40 filters and stride = 1
        - :math:`3^3` kernel with 50 filters and stride = 1
        - 100 units, activation ReLU
        - 50 units, activation ReLu
        - 1 unit, activation ReLu

    **Attributes**:

    :var BasicModel.y: Batch of labels for all the pairs with shape ``[batch]``
    :var BasicModel.y_prob: Tensor with the probabilities of single class classification
    :var BasicModel.y_estimate: Tensor with the classification, derived from :any:`BasicModel.y_prob`
    :var BasicModel.classification_loss: Classification loss using the negative log loss function

                                         .. math::
                                             \mathcal{L}(\boldsymbol{y}, \boldsymbol{\hat{y}}) = -\frac{1}{m}
                                             \sum_{i = 1}^{m} \left(y_i \cdot \log(\hat{y}_i) +
                                             (1 - y_i) \cdot \log(1 - \hat{y}_i)\right)
                                             \quad m := \text{batch size}

    :var BasicModel.regularization_loss: L2 norm of the weights to add to the loss function to regularize
    :var BasicModel.total_loss: Total loss to be minimized with the optimizer
    :var BasicModel.good_predictions: Number of good predictions found on the current batch. Then to get the c-index
                                      it only needs to be divided by the batch size
    :var BasicModel.c_index: Concordance index for the current batch. It's obtained by diving the number of correct
                             comparisons between the total

                             .. math::
                                 \frac{\text{correct comparisons}}{\text{total comparisons}}

    :var BasicModel._regularization: Regularization factor
    :var BasicModel._dropout: Dropout probability
    :var BasicSiamese.pairs_a: Indices to be selected as pairs A for the batch of input images, has shape ``[batch]``
    :var BasicSiamese.pairs_b: Indices to be selected as pairs B for the batch of input images, has shape ``[batch]``
    :var BasicSiamese.gathered_a: Output results for pairs members' A. It has shape ``[pairs_batch, last_layer_units]``
    :var BasicSiamese.gathered_b: Output results for pairs members' B. It has shape ``[pairs_batch, last_layer_units]``
    :var BasicSiamese._gpu_level: Amount of GPU that should be used when evaluating the model
    :var BasicImageSiamese.x_image: Batch of input images, has shape ``[batch, 64, 64, 64, 1]``
    :vartype BasicModel.y: tf.Tensor
    :vartype BasicModel.y_prob: tf.Tensor
    :vartype BasicModel.y_estimate: tf.Tensor
    :vartype BasicModel.classification_loss: tf.Tensor
    :vartype BasicModel.regularization_loss: tf.Tensor
    :vartype BasicModel.total_loss: tf.Tensor
    :vartype BasicModel.good_predictions: tf.Tensor
    :vartype BasicModel.c_index: tf.Tensor
    :vartype BasicModel._regularization: float
    :vartype BasicModel._dropout: float
    :vartype BasicSiamese.pairs_a: tf.Tensor
    :vartype BasicSiamese.pairs_b: tf.Tensor
    :vartype BasicSiamese.gathered_a: tf.Tensor
    :vartype BasicSiamese.gathered_b: tf.Tensor
    :vartype BasicSiamese._gpu_level: int
    :vartype BasicImageSiamese.x_image: tf.Tensor
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
        self.logger.debug(f"Using device: {device} for first conv layers")
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
        self.logger.debug(f"Using device: {device} for second conv layers")
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
        self.logger.debug(f"Using device: {device} for FC layers")
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
    r"""
    Siamese model that uses both images and scalar values as input.

    This class creates a Siamese model that uses both images and scalar features extracted using
    `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_.
    The features are not extracted by the model but they have to be provided in one of the placeholders

    **Network structure**:

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

    **Attributes**:

    :var BasicModel.y: Batch of labels for all the pairs with shape ``[batch]``
    :var BasicModel.y_prob: Tensor with the probabilities of single class classification
    :var BasicModel.y_estimate: Tensor with the classification, derived from :any:`BasicModel.y_prob`
    :var BasicModel.classification_loss: Classification loss using the negative log loss function

                                         .. math::
                                             \mathcal{L}(\boldsymbol{y}, \boldsymbol{\hat{y}}) = -\frac{1}{m}
                                             \sum_{i = 1}^{m} \left(y_i \cdot \log(\hat{y}_i) +
                                             (1 - y_i) \cdot \log(1 - \hat{y}_i)\right)
                                             \quad m := \text{batch size}

    :var BasicModel.regularization_loss: L2 norm of the weights to add to the loss function to regularize
    :var BasicModel.total_loss: Total loss to be minimized with the optimizer
    :var BasicModel.good_predictions: Number of good predictions found on the current batch. Then to get the c-index
                                      it only needs to be divided by the batch size
    :var BasicModel.c_index: Concordance index for the current batch. It's obtained by diving the number of correct
                             comparisons between the total

                             .. math::
                                 \frac{\text{correct comparisons}}{\text{total comparisons}}

    :var BasicModel._regularization: Regularization factor
    :var BasicModel._dropout: Dropout probability
    :var BasicSiamese.pairs_a: Indices to be selected as pairs A for the batch of input images, has shape ``[batch]``
    :var BasicSiamese.pairs_b: Indices to be selected as pairs B for the batch of input images, has shape ``[batch]``
    :var BasicSiamese.gathered_a: Output results for pairs members' A. It has shape ``[pairs_batch, last_layer_units]``
    :var BasicSiamese.gathered_b: Output results for pairs members' B. It has shape ``[pairs_batch, last_layer_units]``
    :var BasicSiamese._gpu_level: Amount of GPU that should be used when evaluating the model
    :var BasicImageSiamese.x_image: Batch of input images, has shape ``[batch, 64, 64, 64, 1]``
    :var ImageScalarSiamese.x_scalar: Scalar features obtained with
                                      `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
    :vartype BasicModel.y: tf.Tensor
    :vartype BasicModel.y_prob: tf.Tensor
    :vartype BasicModel.y_estimate: tf.Tensor
    :vartype BasicModel.classification_loss: tf.Tensor
    :vartype BasicModel.regularization_loss: tf.Tensor
    :vartype BasicModel.total_loss: tf.Tensor
    :vartype BasicModel.good_predictions: tf.Tensor
    :vartype BasicModel.c_index: tf.Tensor
    :vartype BasicModel._regularization: float
    :vartype BasicModel._dropout: float
    :vartype BasicSiamese.pairs_a: tf.Tensor
    :vartype BasicSiamese.pairs_b: tf.Tensor
    :vartype BasicSiamese.gathered_a: tf.Tensor
    :vartype BasicSiamese.gathered_b: tf.Tensor
    :vartype BasicSiamese._gpu_level: int
    :vartype BasicImageSiamese.x_image: tf.Tensor
    :vartype ImageScalarSiamese.x_scalar: tf.Tensor
    """

    def __init__(self, gpu_level: int = 0, regularization: float = 0.001, dropout: float = 0.2):
        """
        Initialize a ScalarSiamese model. This model uses scalar features extracted with PyRadiomics and
        provided through a CSV file in the dataset, the model assumes that there are :any:`settings.NUMBER_FEATURES`
        for each input.

        :param gpu_level: Amount of GPU that should be used with the model
        :param regularization: Regularization factor for the weights
        :param dropout: Dropout probability
        """

        #: **Attribute**: Scalar features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
        self.x_scalar = tf.placeholder(tf.float32, [None, settings.NUMBER_FEATURES], name="radiomic_features")

        super().__init__(gpu_level=gpu_level, regularization=regularization, dropout=dropout)

    def _conv_layers(self, x: tf.Tensor) -> tf.Tensor:
        """
        Implementation of abstract method :func:`~BasicSiamese._conv_layers`

        :param x: Network's input images with shape ``[batch, 64, 64, 64, 1]``
        :return: Filtered image with the convolutions applied
        """
        # In: [batch, 64, 64, 64, 1]

        device = '/gpu:0' if self._gpu_level >= 2 else '/cpu:0'
        self.logger.debug(f"Using device: {device} for first conv layers")
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
        self.logger.debug(f"Using device: {device} for second conv layers")
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
        self.logger.debug(f"Using device: {device} for FC layers")
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

            x = self._dense(
                x=x,
                units=1000,
                name="fc3"
            )

            # x = tf.layers.dropout(
            #     x,
            #     rate=self._dropout,
            #     training=self.training
            # )

            # Out: [batch, 10]
            x = self._dense(
                x=x,
                units=100,
                activation=tf.nn.relu,
                name="fc4"
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
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self._regularization),
            name=name
        )

    def _dense(self, x: tf.Tensor, units: int, name: str, activation=tf.nn.relu) -> tf.Tensor:
        return tf.layers.dense(
            x,
            units=units,
            activation=activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self._regularization),
            name=name
        )

    def feed_dict(self, batch: data.PairBatch, training: bool = True) -> Dict:
        """
        Re-implementation of :func:`~BasicSiamese.feed_dict` to create a custom dict including the scalar values

        :param batch: Data containing for the current batch, usually this would be generated by
                      :func:`~BatchData.batches`
        :param training: Whether we are training or not. Useful for training layers like dropout where we do not
                         want to apply dropout if we are not training
        :return: Return the ``feed_dict`` as a dictionary
        """
        return {
            **super().feed_dict(batch, training=training),
            self.x_scalar: batch.features
        }


class ScalarOnlySiamese(BasicSiamese):
    r"""
    Model that uses only radiomic features as input to train

    Machine Learning model that only uses the radiomic features obtained with
    `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_

    :var BasicModel.y: Batch of labels for all the pairs with shape ``[batch]``
    :var BasicModel.y_prob: Tensor with the probabilities of single class classification
    :var BasicModel.y_estimate: Tensor with the classification, derived from :any:`BasicModel.y_prob`
    :var BasicModel.classification_loss: Classification loss using the negative log loss function

                                         .. math::
                                             \mathcal{L}(\boldsymbol{y}, \boldsymbol{\hat{y}}) = -\frac{1}{m}
                                             \sum_{i = 1}^{m} \left(y_i \cdot \log(\hat{y}_i) +
                                             (1 - y_i) \cdot \log(1 - \hat{y}_i)\right)
                                             \quad m := \text{batch size}

    :var BasicModel.regularization_loss: L2 norm of the weights to add to the loss function to regularize
    :var BasicModel.total_loss: Total loss to be minimized with the optimizer
    :var BasicModel.good_predictions: Number of good predictions found on the current batch. Then to get the c-index
                                      it only needs to be divided by the batch size
    :var BasicModel.c_index: Concordance index for the current batch. It's obtained by diving the number of correct
                             comparisons between the total

                             .. math::
                                 \frac{\text{correct comparisons}}{\text{total comparisons}}

    :var BasicModel._regularization: Regularization factor
    :var BasicModel._dropout: Dropout probability
    :var BasicSiamese.pairs_a: Indices to be selected as pairs A for the batch of input images, has shape ``[batch]``
    :var BasicSiamese.pairs_b: Indices to be selected as pairs B for the batch of input images, has shape ``[batch]``
    :var BasicSiamese.gathered_a: Output results for pairs members' A. It has shape ``[pairs_batch, last_layer_units]``
    :var BasicSiamese.gathered_b: Output results for pairs members' B. It has shape ``[pairs_batch, last_layer_units]``
    :var BasicSiamese._gpu_level: Amount of GPU that should be used when evaluating the model
    :var ScalarOnlySiamese.x_scalar: Radiomic features obtained with
                                     `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
    :vartype BasicModel.y: tf.Tensor
    :vartype BasicModel.y_prob: tf.Tensor
    :vartype BasicModel.y_estimate: tf.Tensor
    :vartype BasicModel.classification_loss: tf.Tensor
    :vartype BasicModel.regularization_loss: tf.Tensor
    :vartype BasicModel.total_loss: tf.Tensor
    :vartype BasicModel.good_predictions: tf.Tensor
    :vartype BasicModel.c_index: tf.Tensor
    :vartype BasicModel._regularization: float
    :vartype BasicModel._dropout: float
    :vartype BasicSiamese.pairs_a: tf.Tensor
    :vartype BasicSiamese.pairs_b: tf.Tensor
    :vartype BasicSiamese.gathered_a: tf.Tensor
    :vartype BasicSiamese.gathered_b: tf.Tensor
    :vartype BasicSiamese._gpu_level: int
    :vartype ScalarOnlySiamese.x_scalar: tf.Tensor
    """

    def __init__(self, gpu_level: int = 0, regularization: float = 0.01, dropout: float = 0.2):
        #: Radiomic features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
        self.x_scalar = tf.placeholder(tf.float32, [None, settings.NUMBER_FEATURES])

        super().__init__(gpu_level=gpu_level, regularization=regularization, dropout=dropout)

    def _sister(self):
        # Out: [batch, 500]
        x = self.x_scalar
        x = self._dense(
            x,
            500,
            "fc1"
        )

        # Out: [batch, 200]
        x = self._dense(
            x,
            200,
            "fc2"
        )

        x = tf.layers.dropout(
            x,
            rate=self._dropout,
            training=self.training
        )

        # Out: [batch, 50]
        x = self._dense(
            x,
            50,
            "fc3"
        )

        # Out: [batch, 1]
        x = self._dense(
            x,
            10,
            "fc4",
            activation=tf.nn.relu
        )

        return x

    def _dense(self, x: tf.Tensor, units: int, name: str, activation=tf.nn.tanh) -> tf.Tensor:
        return tf.layers.dense(
            x,
            units=units,
            activation=activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self._regularization),
            name=name
        )

    def feed_dict(self, batch: data.PairBatch, training: bool = True):
        return {
            **super().feed_dict(batch, training),
            self.x_scalar: batch.features,
        }

    def uses_images(self) -> bool:
        """
        Implementation of :func:`BasicModel.uses_images`. This model does not uses images to work.

        :return: :any:`False` since this model does not use images to work
        """
        return False


class VolumeOnlySiamese(BasicSiamese):
    r"""
    Model that only uses the volume radiomic feature

    The features are provided by the package from `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_

    It trains a model in the form :math:`y = w \cdot V + b`

    :var BasicModel.y: Batch of labels for all the pairs with shape ``[batch]``
    :var BasicModel.y_prob: Tensor with the probabilities of single class classification
    :var BasicModel.y_estimate: Tensor with the classification, derived from :any:`BasicModel.y_prob`
    :var BasicModel.classification_loss: Classification loss using the negative log loss function

                                         .. math::
                                             \mathcal{L}(\boldsymbol{y}, \boldsymbol{\hat{y}}) = -\frac{1}{m}
                                             \sum_{i = 1}^{m} \left(y_i \cdot \log(\hat{y}_i) +
                                             (1 - y_i) \cdot \log(1 - \hat{y}_i)\right)
                                             \quad m := \text{batch size}

    :var BasicModel.regularization_loss: L2 norm of the weights to add to the loss function to regularize
    :var BasicModel.total_loss: Total loss to be minimized with the optimizer
    :var BasicModel.good_predictions: Number of good predictions found on the current batch. Then to get the c-index
                                      it only needs to be divided by the batch size
    :var BasicModel.c_index: Concordance index for the current batch. It's obtained by diving the number of correct
                             comparisons between the total

                             .. math::
                                 \frac{\text{correct comparisons}}{\text{total comparisons}}

    :var BasicModel._regularization: Regularization factor
    :var BasicModel._dropout: Dropout probability
    :var BasicSiamese.pairs_a: Indices to be selected as pairs A for the batch of input images, has shape ``[batch]``
    :var BasicSiamese.pairs_b: Indices to be selected as pairs B for the batch of input images, has shape ``[batch]``
    :var BasicSiamese.gathered_a: Output results for pairs members' A. It has shape ``[pairs_batch, last_layer_units]``
    :var BasicSiamese.gathered_b: Output results for pairs members' B. It has shape ``[pairs_batch, last_layer_units]``
    :var BasicSiamese._gpu_level: Amount of GPU that should be used when evaluating the model
    :var ScalarOnlySiamese.x_volume: Radiomic volume feature obtained with
                                     `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
    :vartype BasicModel.y: tf.Tensor
    :vartype BasicModel.y_prob: tf.Tensor
    :vartype BasicModel.y_estimate: tf.Tensor
    :vartype BasicModel.classification_loss: tf.Tensor
    :vartype BasicModel.regularization_loss: tf.Tensor
    :vartype BasicModel.total_loss: tf.Tensor
    :vartype BasicModel.good_predictions: tf.Tensor
    :vartype BasicModel.c_index: tf.Tensor
    :vartype BasicModel._regularization: float
    :vartype BasicModel._dropout: float
    :vartype BasicSiamese.pairs_a: tf.Tensor
    :vartype BasicSiamese.pairs_b: tf.Tensor
    :vartype BasicSiamese.gathered_a: tf.Tensor
    :vartype BasicSiamese.gathered_b: tf.Tensor
    :vartype BasicSiamese._gpu_level: int
    :vartype VolumeOnlySiamese.x_volume: tf.Tensor
    """

    def __init__(self):
        #: **Attribute**: Radiomic volume feature obtained with
        #: `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
        self.x_volume = tf.placeholder(tf.float32, [None, 1])

        super().__init__()

    def _sister(self) -> tf.Tensor:
        """
        Super greedy predictor, more volume means less survival time so we only have to invert the volume size to
        create an inverse relation. This model does not have trainable variables

        :return: Greedy siamese applied
        """
        w = tf.Variable(-1., name="weight")
        b = tf.Variable(0., name="bias")
        return w*self.x_volume + b

    def feed_dict(self, batch: data.PairBatch, training: bool = True):

        volumes = batch.features[:, settings.VOLUME_FEATURE_INDEX].reshape((-1, 1))

        return {
            **super().feed_dict(batch, training),
            self.x_volume: volumes
        }

    def uses_images(self):
        return False
