"""
Definitions of siamese models. Each model uses a siamese network to convert the survival problem into a
classification problem.

  - The :class:`ImageSiamese` class creates a model that uses only the images as input and uses some blocks
  - The :class:`SimpleImageSiamese` class creates a basic model that only uses images as input
  - The :class:`ImageScalarSiamese` class creates a model that combines the image input with the scalar input
    of the radiomic features, extracted with PyRadiomics.
  - The :class:`ResidualImageScalarSiamese` class creates a model that combines the image input with the scalar input
    but it also uses a residual network to fit the images. It also uses multiple blocks, similar to the
    inception idea
  - The :class:`ScalarOnlySiamese` class creates a siamese model that only uses the radiomic features as an input.
  - The :class:`ScalarOnlyDropoutSiamese` class creates a siamese model that only uses the radiomic features as an
    input but adds multiple dropout layers to improve the results.
  - The :class:`VolumeOnlySiamese` class creates a siamese model that only uses the volume feature to fit the model.


.. inheritance-diagram:: models.siameses
   :parts: 1

"""

from typing import Dict, Any, Union, Tuple

import tensorflow as tf
import numpy as np
import data
import tensorflow_src.settings as settings
from .basics import BasicImageSiamese, BasicSiamese


class ImageSiamese(BasicImageSiamese):
    """
    Class representing the initial and simple siamese structure used for the first steps of the project. It
    inherits :any:`BasicSiamese` so it has the same tensors to be fed.

    **Convolutional Model**:

    It contains parallel in inception_block and 3 FC layers

    """

    def __init__(self, **kwargs):
        """
        Construct a new SimpleSiamese class

        :param gpu_level: Amount of GPU to be used with the model

                            0. No GPU usage
                            1. Only second conv layers
                            2. All conv layers
                            3. All layers and parameters are on the GPU
        """
        super().__init__(**kwargs)

    def _inception_block(self, x:tf.Tensor, stage: str, block:str):
        """

        :param x: Network's input images with shape ``[batch, 64, 64, 64, 1]``
        :param filters: list of integers, the number of filters in the CONV layers
        :param stage: integer, Used to name the layers, depending on their position in the network
        :param block: string, Used to name the layers, depending on their position in the network
        :return: Tensor of shape ``[n_X, n_Y, n_Z, n_C]``
        """
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        # Retrieve Filters
        #F1, F2, F3 = filters
        device = '/gpu:0' if self._gpu_level >= 1 else '/cpu:0'
        self.logger.debug(f"Using device: {device} for first conv layers")
        with tf.device(device):

            a1 = tf.layers.conv3d(

                x,

                filters=8,

                kernel_size=[1, 1, 1],

                strides=1,

                activation=tf.nn.relu,

                padding='SAME',

                name=conv_name_base + 'a1'

            )

            a1 = tf.layers.conv3d(

                a1,

                filters=8,

                kernel_size=[4, 4, 4],

                strides=1,

                activation=tf.nn.relu,

                padding='SAME',

                name=conv_name_base + 'a2'

            )
        device = '/gpu:0' if self._gpu_level >= 2 else '/cpu:0'
        self.logger.debug("Using device: {device} for first conv layers".format(device=device))
        with tf.device(device):

            b1 = tf.layers.conv3d(

                x,

                filters=8,

                kernel_size=[1, 1, 1],

                strides=1,

                activation=tf.nn.relu,

                padding='SAME',

                name=conv_name_base + 'b1'

            )

            b1 = tf.layers.conv3d(

                b1,

                filters=8,

                kernel_size=[2, 2, 2],

                strides=1,

                activation=tf.nn.relu,

                padding='SAME',

                name=conv_name_base + 'b2'

            )
        device = '/gpu:0' if self._gpu_level >= 3 else '/cpu:0'
        self.logger.debug(f"Using device: {device} for first conv layers")
        with tf.device(device):
            c1 = tf.nn.max_pool3d(

                x,

                ksize=[1, 4, 4, 4, 1],

                strides=[1,1,1,1,1],

                padding='SAME',


                name=conv_name_base + 'c1'

            )

            c1 = tf.layers.conv3d(

                c1,

                filters=8,

                kernel_size=[1, 1, 1],

                strides=1,

                activation=tf.nn.relu,

                padding='SAME',

                name=conv_name_base + 'c2'

            )

        d1 = tf.concat([a1, b1], 4)
        d1 = tf.concat([d1, c1], 4)

        d1 = tf.layers.conv3d(

            d1,

            filters=1,

            kernel_size=[1, 1, 1],

            strides=1,

            activation=tf.nn.relu,

            padding='SAME',

            name=conv_name_base + 'd'
        )
        d1 = tf.contrib.layers.batch_norm(d1,
                                     center=True, scale=True,
                                     scope='bn')
        tf.layers.BatchNormalization(d1)
        return d1

    def _conv_layers(self, x: tf.Tensor) -> tf.Tensor:
        """
        Implementation of abstract method :func:`~BasicSiamese._conv_layers`

        :param x: Network's input images with shape ``[batch, 64, 64, 64, 1]``
        :return: Filtered image with the convolutions applied
        """
        # In: [batch, 64, 64, 64, 1]
        x1 = self._inception_block(x,"1s", "b1")

        return x1


    def _fc_layers(self, x: tf.Tensor) -> tf.Tensor:
        """
        Implementation of abstract method ``BasicSiamese._fc_layers``

        :param x: Image, usually previously filtered with the convolutional layers.
        :return: Tensor with shape ``[batch, 1]``
        """
        device = '/gpu:0' if self._gpu_level >= 3 else '/cpu:0'
        self.logger.debug(f"Using device: {device} for FC layers")
        with tf.device(device):
            # Out: [batch, 64*64*64*1]
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

    def uses_images(self) -> bool:
        """
        Implementation of :func:`BasicModel.uses_images`.

        :return: :any:`True`, the model uses images as input to work
        """
        return True


class SimpleImageSiamese(BasicImageSiamese):
    r"""
    Simple siamese network implementation that uses images as input.

    Has the same parameters as :class:`BasicImageSiamese`

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

    Includes the same attributes as :class:`BasicImageSiamese`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _conv_layers(self, x: tf.Tensor) -> tf.Tensor:
        """
        Implementation of abstract method :func:`~BasicSiamese._conv_layers`

        :param x: Network's input images with shape ``[batch, 64, 64, 64, 1]``
        :return: Filtered image with the convolutions applied
        """
        # In: [batch, 256, 256, 256, 1]

        device = '/gpu:0' if self._gpu_level >= 2 else '/cpu:0'
        self.logger.debug(f"Using device: {device} for first conv layers")
        with tf.device(device):
            # Out: [batch, 127, 127, 127, 30]
            x = tf.layers.conv3d(
                x,
                filters=20,
                kernel_size=3,
                strides=2,
                activation=tf.nn.relu,
                name="conv1"
            )


            # Out: [batch, 63, 63, 63, 40]
            x = tf.layers.conv3d(
                x,
                filters=20,
                kernel_size=2,
                strides=2,
                activation=tf.nn.relu,
                name="conv2"
            )

            x=tf.layers.max_pooling3d(
                x,
                strides=1,
                pool_size=3,
                name = "pool2"
            )


        device = '/gpu:0' if self._gpu_level >= 1 else '/cpu:0'
        self.logger.debug(f"Using device: {device} for second conv layers")
        with tf.device(device):
            # Out: [batch, 19, 19, 19, 40]
            x = tf.layers.conv3d(
                x,
                filters=40,
                kernel_size=3,
                activation=tf.nn.relu,
                name="conv3"
            )

            # Out: [batch, 18, 18, 18, 50]
            x = tf.layers.conv3d(
                x,
                filters=50,
                kernel_size=2,
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
                activation=tf.nn.tanh,
                name="fc1"
            )

            # Out: [batch, 50]
            x = tf.layers.dense(
                x,
                50,
                activation=tf.nn.tanh,
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

    It uses scalar features extracted with PyRadiomics and provided through a Tensorflow ``placeholder``,
    the model assumes that there are :any:`settings.NUMBER_FEATURES` for each input.

    :param learning_rate:
    :param gpu_level: Amount of GPU that should be used with the model
    :param regularization: Regularization factor for the weights
    :param dropout: Dropout probability

    This class creates a Siamese model that uses both images and scalar features extracted using
    `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_.
    The features are not extracted by the model, so they have to be provided in one of the placeholders

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

    Includes the same attributes as :class:`BasicImageSiamese` and adds the following ones:

    :var ImageScalarSiamese.x_scalar: Scalar features obtained with
                                      `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
    :vartype ImageScalarSiamese.x_scalar: tf.Tensor
    """

    def __init__(self, **kwargs):
        #: **Attribute**: Scalar features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
        self.x_scalar = tf.placeholder(tf.float32, [None, settings.NUMBER_FEATURES], name="radiomic_features")

        super().__init__(**kwargs)

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

    def _conv3d(self, x: tf.Tensor,
                filters: int,
                kernel_size: Union[int, Tuple],
                name: str,
                strides: Union[int, Tuple] = 1,
                activation: Any = tf.nn.relu,
                padding="valid") -> tf.Tensor:
        return tf.layers.conv3d(
            name=name,
            inputs=x,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self._regularization),
            padding=padding
        )

    def _dense(self, x: tf.Tensor, units: int, name: str, activation=tf.nn.relu) -> tf.Tensor:
        return tf.layers.dense(
            x,
            units=units,
            activation=activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
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
            self.x_scalar: np.stack(batch.patients["features"].values)
        }


class ResidualImageScalarSiamese(ImageScalarSiamese):

    def __init__(self, **kwargs):
        self.residual_count_a = 0
        self.residual_count_b = 0

        super().__init__(**kwargs)

    def _conv_layers(self, x: tf.Tensor) -> tf.Tensor:
        device = '/gpu:0' if self._gpu_level >= 1 else '/cpu:0'
        with tf.device(device):
            x = self._stem_block(x)

            for i in range(2):
                x = self._res_block_a(x)

        device = '/gpu:1' if self._gpu_level >= 1 else '/cpu:0'
        with tf.device(device):
            x = self._reduction_a(x)

            for i in range(2):
                x = self._res_block_b(x)

            # Out: [batch, 7, 7, 7, 350]
            x = self._reduction_b(x)

            # Out: [batch, 2, 2, 2, 350]
            x = tf.layers.average_pooling3d(
                inputs=x,
                pool_size=6,
                strides=1
            )

        return x

    def _fc_layers(self, x: tf.Tensor) -> tf.Tensor:

        # Out: [batch, 2*2*2*350] = [batch, 2800]
        x = tf.layers.flatten(x, name="flat")

        # Out: [batch, 2800 + 725]
        x = tf.concat([x, self.x_scalar], axis=1)

        x = self._dense(
            x=x,
            units=800,
            name="fc_0",
        )

        x = self._dense(
            x=x,
            units=100,
            name="fc_1",
        )

        x = self._dense(
            x=x,
            units=10,
            name="fc_2",
            activation=tf.nn.relu
        )

        return x

    def _stem_block(self, x: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope("stem_reduce"):

            # Out: [batch, 31, 31, 31, 25]
            x_a = self._conv3d(
                x=x,
                name="a_0_conv_3x3x3",
                filters=25,
                kernel_size=3,
                strides=2,
            )

            # Out: [batch, 31, 31, 31, 1]
            x_b = tf.layers.max_pooling3d(
                inputs=x,
                name="b_0_pool_3x3x3",
                pool_size=3,
                strides=2,
            )

            # Out: [batch, 31, 31, 31, 25]
            x_b = self._conv3d(
                x=x_b,
                name="b_1_conv_1x1x1",
                filters=25,
                kernel_size=1,
                padding="same"
            )

            # Out: [batch, 31, 31, 31, 50]
            x_concat: tf.Tensor = tf.concat([x_a, x_b], axis=4)
            assert x_concat.get_shape()[-1] == 50

            self.logger.debug(x_concat.get_shape())

            return x_concat

    def _res_block_a(self, x: tf.Tensor, activation_fn=tf.nn.relu) -> tf.Tensor:
        """
        Residual block with size ``[batch, 31, 31, 31, 50]``
        :param x:
        :return:
        """

        with tf.variable_scope(f"block_31_{self.residual_count_a}"):
            self.residual_count_a += 1

            x_a = self._conv3d(
                x=x,
                name="a_0_conv_1x1x1",
                filters=32,
                kernel_size=1,
                padding="same"
            )

            x_b = self._conv3d(
                x=x,
                name="b_0_conv_1x1x1",
                filters=32,
                kernel_size=1,
                padding="same"
            )

            x_b = self._conv3d(
                x=x_b,
                name="b_1_conv_3x3x3",
                filters=32,
                kernel_size=3,
                padding="same"
            )

            x_c = self._conv3d(
                x=x,
                name="c_0_conv_1x1x1",
                filters=32,
                kernel_size=1,
                padding="same"
            )

            for i in range(1, 3):
                x_c = self._conv3d(
                    x=x_c,
                    name=f"c_{i}_conv_3x3x3",
                    filters=32,
                    kernel_size=3,
                    padding="same"
                )

            x_conv = tf.concat([x_a, x_b, x_c], axis=4)

            x_conv = self._conv3d(
                x=x_conv,
                name="conv_1x1",
                filters=x.get_shape()[-1],
                kernel_size=1,
                padding="same",
                activation=None
            )

            x += x_conv
            return activation_fn(x)

    def _reduction_a(self, x: tf.Tensor) -> tf.Tensor:
        """
        :param x: Tensor with shape ``[batch, 31, 31, 31, 50]``
        :return: Tensor with shape ``[batch, 15, 15, 15, 130]``
        """
        with tf.variable_scope("reduction_a"):
            x_a = tf.layers.max_pooling3d(
                inputs=x,
                name="a_0_pooling_3x3x3",
                pool_size=3,
                strides=2
            )

            x_b = self._conv3d(
                x=x,
                name="b_0_conv_3x3x3",
                filters=50,
                kernel_size=3,
                strides=2
            )

            x_c = self._conv3d(
                x=x,
                name="c_0_conv_1x1x1",
                filters=30,
                kernel_size=1,
                padding="same"
            )

            x_c = self._conv3d(
                x=x_c,
                name="c_1_conv_3x3x3",
                filters=30,
                kernel_size=3,
                padding="same"
            )

            x_c = self._conv3d(
                x=x_c,
                name="c_2_conv_3x3x3",
                filters=30,
                kernel_size=3,
                strides=2
            )

            return tf.concat([x_a, x_b, x_c], axis=4)

    def _res_block_b(self, x: tf.Tensor, activation_fn=tf.nn.relu) -> tf.Tensor:
        """
        :param x: Tensor with shape ``[batch, 15, 15, 15, 130]``
        :return: Tensor with shape ``[batch, 15, 15, 15, 130]``
        """

        with tf.variable_scope(f"block_15_{self.residual_count_b}"):
            self.residual_count_b += 1

            x_a = self._conv3d(
                x=x,
                name="a_0_conv_1x1x1",
                filters=50,
                kernel_size=1,
                padding="same"
            )

            x_b = self._conv3d(
                x=x,
                name="b_0_conv_1x1x1",
                filters=50,
                kernel_size=1,
                padding="same"
            )

            x_b = self._conv3d(
                x=x_b,
                name="b_1_conv_1x1x7",
                filters=50,
                kernel_size=(1, 1, 7),
                padding="same"
            )

            x_b = self._conv3d(
                x=x_b,
                name="b_1_conv_1x7x1",
                filters=50,
                kernel_size=(1, 7, 1),
                padding="same"
            )

            x_b = self._conv3d(
                x=x_b,
                name="b_1_conv_7x1x1",
                filters=50,
                kernel_size=(7, 1, 1),
                padding="same"
            )

            x_concat = tf.concat([x_a, x_b], axis=4)
            x_conv = self._conv3d(
                x=x_concat,
                name="conv_1x1",
                filters=x.get_shape()[-1],
                kernel_size=1,
                padding="same",
                activation=None
            )

            x += x_conv
            return activation_fn(x)

    def _reduction_b(self, x: tf.Tensor) -> tf.Tensor:
        """
        :param x: Tensor with shape ``[batch, 15, 15, 15, 130]``
        :return: Tensor with shape ``[batch, 7, 7, 7, 350]``
        """

        with tf.variable_scope("reduction_b"):
            x_a = self._conv3d(
                x=x,
                name="a_0_conv_3x3",
                filters=100,
                kernel_size=3,
                strides=2
            )

            x_b = self._conv3d(
                x=x,
                name="b_0_conv_1x1",
                filters=100,
                kernel_size=1,
                padding="same"
            )

            x_b = self._conv3d(
                x=x_b,
                name="b_1_conv_3x3",
                filters=100,
                kernel_size=3,
                strides=2
            )

            x_c = self._conv3d(
                x=x,
                name="c_0_conv_1x1",
                filters=100,
                kernel_size=1,
                padding="same"
            )

            x_c = self._conv3d(
                x=x_c,
                name="c_1_conv_3x3",
                filters=100,
                kernel_size=3,
                strides=2
            )

            x_d = self._conv3d(
                x=x,
                name="d_0_conv_1x1",
                filters=50,
                kernel_size=1,
                padding="same"
            )

            x_d = self._conv3d(
                x=x_d,
                name="d_1_conv_3x3",
                filters=50,
                kernel_size=3,
                padding="same"
            )

            x_d = self._conv3d(
                x=x_d,
                name="d_2_conv_3x3",
                filters=50,
                kernel_size=3,
                strides=2
            )

            # Out [batch, 7, 7, 7, 100 + 100 + 100 + 50]
            return tf.concat([x_a, x_b, x_c, x_d], axis=4)


class ScalarOnlySiamese(BasicSiamese):
    r"""
    Model that uses only radiomic features as input to train

    It has the same parameters as :class:`BasicSiamese`

    It only uses the radiomic features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_

    **Attributes**:

    Includes the same attributes as :class:`BasicSiamese` and adds the following ones:

    :var ScalarOnlySiamese.x_scalar: Radiomic features obtained with
                                     `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
    :vartype ScalarOnlySiamese.x_scalar: tf.Tensor
    """

    def __init__(self, number_features: int, **kwargs):
        #: Radiomic features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
        self.x_scalar = tf.placeholder(tf.float32, [None, number_features])

        super().__init__(**kwargs)

    def _sister(self):
        # Out: [batch, 500]
        x = self.x_scalar
        x = self._dense(
            x,
            500,
            "fc1"
        )
        x = tf.layers.dropout(
            x,
            rate=self._dropout,
            training=self.training
        )
        y = self._dense(
            x,
            20,
            "fcY"
        )

        x = self._dense(
            x,
            100,
            "fc2"
        )
        x = tf.layers.dropout(
            x,
            rate=self._dropout,
            training=self.training
        )


        x = self._dense(
            x,
            50,
            "fc3"
        )

        x = self._dense(
            x,
            20,
            "fc4"
        )

        # Out: [batch, 1]
        x = self._dense(
            #tf.concat([x , y],1),
            x,
            10,
            "fc5"
        )

        x = self._dense(
            x ,
            20,
            "fc6",
            activation=tf.nn.selu
        )
        return x

        x = self._dense(
            x ,
            1,
            "fc7",
            activation=tf.nn.relu
        )
        return x

    def _dense(self, x: tf.Tensor, units: int, name: str, activation=tf.nn.tanh) -> tf.Tensor:
        return tf.layers.dense(
            x,
            units=units,
            activation=activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self._regularization),
            name=name
        )

    def feed_dict(self, batch: data.PairBatch, training: bool = True):

        return {
            **super().feed_dict(batch, training),
            self.x_scalar: np.stack(batch.patients["features"]),
        }

    def uses_images(self) -> bool:
        """
        Implementation of :func:`BasicModel.uses_images`. This model does not uses images to work.

        :return: :any:`False` since this model does not use images to work
        """
        return False


class ScalarOnlySiamese1(BasicSiamese):
    def __init__(self, number_features: int, **kwargs):
        #: Radiomic features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
        self.x_scalar = tf.placeholder(tf.float32, [None, number_features])

        super().__init__(**kwargs)

    def _sister(self):
        # Out: [batch, 500]
        x = self.x_scalar
        x = self._dense(
            x,
            500,
            "fc1"
        )
        x = tf.layers.dropout(
            x,
            rate=self._dropout,
            training=self.training
        )
        y = self._dense(
            x,
            20,
            "fcY"
        )

        x = self._dense(
            x,
            100,
            "fc2"
        )
        x = tf.layers.dropout(
            x,
            rate=self._dropout,
            training=self.training
        )

        x = self._dense(
            x,
            50,
            "fc3"
        )
        x = tf.layers.dropout(
            x,
            rate=self._dropout,
            training=self.training
        )

        x = self._dense(
            x,
            20,
            "fc4"
        )
        x = tf.layers.dropout(
            x,
            rate=self._dropout,
            training=self.training
        )

        # Out: [batch, 1]
        x = self._dense(
            # tf.concat([x , y],1),
            x,
            10,
            "fc5"
        )
        x = tf.layers.dropout(
            x,
            rate=self._dropout,
            training=self.training
        )

        x = self._dense(
            x,
            1,
            "fc7",
            activation=tf.nn.relu
        )
        return x

    def _dense(self, x: tf.Tensor, units: int, name: str, activation=tf.nn.tanh) -> tf.Tensor:
        return tf.layers.dense(
            x,
            units=units,
            activation=activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self._regularization),
            name=name
        )

    def feed_dict(self, batch: data.PairBatch, training: bool = True):
        return {
            **super().feed_dict(batch, training),
            self.x_scalar: np.stack(batch.patients["features"]),
        }

    def uses_images(self) -> bool:
        """
        Implementation of :func:`BasicModel.uses_images`. This model does not uses images to work.

        :return: :any:`False` since this model does not use images to work
        """
        return False


class ClinicalOnlySiamese(BasicSiamese):
    r"""
    Model that uses only radiomic features as input to train

    It has the same parameters as :class:`BasicSiamese`

    It only uses the radiomic features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_

    **Attributes**:

    Includes the same attributes as :class:`BasicSiamese` and adds the following ones:

    :var ScalarOnlySiamese.x_scalar: Radiomic features obtained with
                                     `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
    :vartype ScalarOnlySiamese.x_scalar: tf.Tensor
    """

    def __init__(self, number_features: int, **kwargs):
        #: Radiomic features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
        self.x_scalar = tf.placeholder(tf.float32, [None, number_features])

        super().__init__(**kwargs)

    def _sister(self):
        # Out: [batch, 500]
        x = self.x_scalar
        x = self._dense(
            x,
            80,
            "fc1"
        )
        y = self._dense(
            x,
            5,
            "fcY"
        )

        x = self._dense(
            x,
            50,
            "fc2"
        )

  #      x = self._dense(
  #          x,
  #          100,
  #          "fc3"
  #      )

        x = self._dense(
            x,
            20,
            "fc4"
        )

        # Out: [batch, 1]
        x = self._dense(
            tf.concat([x , y],1),
            10,
            "fc5",
            activation=tf.nn.relu
        )

        x = self._dense(
            x,
            5,
            "fc6",
            activation=tf.nn.relu
        )
        return x

    def _dense(self, x: tf.Tensor, units: int, name: str, activation=tf.nn.tanh) -> tf.Tensor:
        return tf.layers.dense(
            x,
            units=units,
            activation=activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self._regularization),
            name=name
        )

    def feed_dict(self, batch: data.PairBatch, training: bool = True):

        return {
            **super().feed_dict(batch, training),
            self.x_scalar: np.stack(batch.patients["features"]),
        }

    def uses_images(self) -> bool:
        """
        Implementation of :func:`BasicModel.uses_images`. This model does not uses images to work.

        :return: :any:`False` since this model does not use images to work
        """
        return False


class ClinicalOnlySiamese2(BasicSiamese):
    r"""
    Model that uses only radiomic features as input to train

    It has the same parameters as :class:`BasicSiamese`

    It only uses the radiomic features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_

    **Attributes**:

    Includes the same attributes as :class:`BasicSiamese` and adds the following ones:

    :var ScalarOnlySiamese.x_scalar: Radiomic features obtained with
                                     `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
    :vartype ScalarOnlySiamese.x_scalar: tf.Tensor
    """

    def __init__(self, number_features: int, **kwargs):
        #: Radiomic features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
        self.x_scalar = tf.placeholder(tf.float32, [None, number_features])

        super().__init__(**kwargs)

    def _sister(self):
        # Out: [batch, 500]
        x = self.x_scalar
        x = self._dense(
            x,
            80,
            "fc1"
        )
        y = self._dense(
            x,
            5,
            "fcY"
        )

        x = self._dense(
            x,
            50,
            "fc2"
        )

  #      x = self._dense(
  #          x,
  #          100,
  #          "fc3"
  #      )

        x = self._dense(
            x,
            20,
            "fc4"
        )

        # Out: [batch, 1]
        x = self._dense(
            tf.concat([x , y],1),
            10,
            "fc5",
            activation=tf.nn.relu
        )

        x = self._dense(
            x,
            5,
            "fc6",
            activation=tf.nn.relu
        )
        return x

    def _dense(self, x: tf.Tensor, units: int, name: str, activation=tf.nn.tanh) -> tf.Tensor:
        return tf.layers.dense(
            x,
            units=units,
            activation=activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self._regularization),
            name=name
        )

    def feed_dict(self, batch: data.PairBatch, training: bool = True):

        return {
            **super().feed_dict(batch, training),
            self.x_scalar: np.stack(batch.patients["features"]),
        }

    def uses_images(self) -> bool:
        """
        Implementation of :func:`BasicModel.uses_images`. This model does not uses images to work.

        :return: :any:`False` since this model does not use images to work
        """
        return False

class ClinicalOnlySiamese3(BasicSiamese):
        r"""
        Model that uses only radiomic features as input to train

        It has the same parameters as :class:`BasicSiamese`

        It only uses the radiomic features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_

        **Attributes**:

        Includes the same attributes as :class:`BasicSiamese` and adds the following ones:

        :var ScalarOnlySiamese.x_scalar: Radiomic features obtained with
                                         `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
        :vartype ScalarOnlySiamese.x_scalar: tf.Tensor
        """

        def __init__(self, number_features: int, **kwargs):
            #: Radiomic features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
            self.x_scalar = tf.placeholder(tf.float32, [None, number_features])

            super().__init__(**kwargs)

        def _sister(self):
            # Out: [batch, 500]
            x = self.x_scalar
            x = self._dense(
                x,
                20,
                "fc1"
            )
            y = self._dense(
                x,
                5,
                "fcY"
            )

            x = self._dense(
                x,
                20,
                "fc2"
            )


            # Out: [batch, 1]
            x = self._dense(
                tf.concat([x, y], 1),
                10,
                "fc5"
            )

            x = self._dense(
                x,
                3,
                "fc6",
                activation=tf.nn.relu
            )
            return x

        def _dense(self, x: tf.Tensor, units: int, name: str, activation=tf.nn.tanh) -> tf.Tensor:
            return tf.layers.dense(
                x,
                units=units,
                activation=activation,
                kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self._regularization),
                name=name
            )

        def feed_dict(self, batch: data.PairBatch, training: bool = True):
            return {
                **super().feed_dict(batch, training),
                self.x_scalar: np.stack(batch.patients["features"]),
            }

        def uses_images(self) -> bool:
            """
            Implementation of :func:`BasicModel.uses_images`. This model does not uses images to work.

            :return: :any:`False` since this model does not use images to work
            """
            return False

class ScalarOnlyDropoutSiamese(ScalarOnlySiamese):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _sister(self):
        # Out: [batch, 500]
        x = self.x_scalar
        x = self._dense(
            x,
            200,
            "fc1"
        )

        x = tf.layers.dropout(
            x,
            rate=self._dropout,
            training=self.training
        )

        # Out: [batch, 200]
        x = self._dense(
            x,
            100,
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
            20,
            "fc3"
        )

        x = tf.layers.dropout(
            x,
            rate=self._dropout,
            training=self.training
        )

        # Out: [batch, 1]
        x = self._dense(
            x,
            10,
            "fc4",
            activation=tf.nn.relu
        )

        return x


class VolumeOnlySiamese(BasicSiamese):
    r"""
    Model that only uses the volume radiomic feature

    The features are provided by the package from `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_

    It trains a model in the form :math:`y = w \cdot V + b`

    **Attributes**:

    Includes the same attributes as :class:`BasicSiamese` and adds the following ones:

    :ivar VolumeOnlySiamese.x_volume: Placeholder for the volume feature
    :vartype VolumeOnlySiamese.x_volume: tf.Tensor
    """

    def __init__(self, **kwargs):
        #: **Attribute**: Radiomic volume feature obtained with
        #: `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
        self.x_volume = tf.placeholder(tf.float32, [None, 1])

        super().__init__(**kwargs)

    def _sister(self) -> tf.Tensor:
        """
        Super greedy predictor, more volume means less survival time so we only have to invert the volume size to
        create an inverse relation. This model does not have trainable variables

        :return: Greedy siamese applied
        """

        total = tf.Variable(0., name="bias")
        for i in range(1):
            w = tf.Variable(-1., name=f"weight_{i}")
            total = total + w*(self.x_volume**(i + 1))

        return total

    def feed_dict(self, batch: data.PairBatch, training: bool = True):

        features = np.stack(batch.patients["features"].values)
        volumes = features[:, settings.VOLUME_FEATURE_INDEX].reshape((-1, 1))

        return {
            **super().feed_dict(batch, training),
            self.x_volume: volumes
        }

    def uses_images(self):
        return False


class ClinicalVolumeSiamese(BasicSiamese):
    r"""
    Model that uses only radiomic features as input to train

    It has the same parameters as :class:`BasicSiamese`

    It only uses the radiomic features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_

    **Attributes**:

    Includes the same attributes as :class:`BasicSiamese` and adds the following ones:

    :var ScalarOnlySiamese.x_scalar: Radiomic features obtained with
                                     `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
    :vartype ScalarOnlySiamese.x_scalar: tf.Tensor
    """

    def __init__(self, number_features: int, **kwargs):
        #: Radiomic features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
        self.x_scalar = tf.placeholder(tf.float32, [None, number_features])

        super().__init__(**kwargs)

    def _sister(self):
        # Out: [batch, 500]
        x = self.x_scalar
        x = self._dense(
            x,
            80,
            "fc1"
        )
        y = self._dense(
            x,
            5,
            "fcY"
        )

        x = self._dense(
            x,
            50,
            "fc2"
        )

  #      x = self._dense(
  #          x,
  #          100,
  #          "fc3"
  #      )

        x = self._dense(
            x,
            20,
            "fc4"
        )

        # Out: [batch, 1]
        x = self._dense(
            tf.concat([x , y],1),
            10,
            "fc5",
            activation=tf.nn.relu
        )

        x = self._dense(
            x,
            5,
            "fc6",
            activation=tf.nn.relu
        )
        return x

    def _dense(self, x: tf.Tensor, units: int, name: str, activation=tf.nn.tanh) -> tf.Tensor:
        return tf.layers.dense(
            x,
            units=units,
            activation=activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self._regularization),
            name=name
        )

    def feed_dict(self, batch: data.PairBatch, training: bool = True):

        return {
            **super().feed_dict(batch, training),
            self.x_scalar: np.stack(batch.patients["features"]),
        }

    def uses_images(self) -> bool:
        """
        Implementation of :func:`BasicModel.uses_images`. This model does not uses images to work.

        :return: :any:`False` since this model does not use images to work
        """
        return False


class ClinicalVolumeSiamese2(BasicSiamese):
    r"""
    Model that uses only radiomic features as input to train

    It has the same parameters as :class:`BasicSiamese`

    It only uses the radiomic features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_

    **Attributes**:

    Includes the same attributes as :class:`BasicSiamese` and adds the following ones:

    :var ScalarOnlySiamese.x_scalar: Radiomic features obtained with
                                     `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
    :vartype ScalarOnlySiamese.x_scalar: tf.Tensor
    """

    def __init__(self, number_features: int, **kwargs):
        #: Radiomic features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
        self.x_scalar = tf.placeholder(tf.float32, [None, number_features])

        super().__init__(**kwargs)

    def _sister(self):
        # Out: [batch, 500]
        x = self.x_scalar
        x = self._dense(
            x,
            40,
            "fc1"
        )
        y = self._dense(
            x,
            5,
            "fcY"
        )

        x = self._dense(
            x,
            20,
            "fc2"
        )

  #      x = self._dense(
  #          x,
  #          100,
  #          "fc3"
  #      )

        x = self._dense(
            x,
            10,
            "fc4"
        )

        # Out: [batch, 1]
#        x = self._dense(
#            tf.concat([x , y],1),
#            10,
#            "fc5",
#            activation=tf.nn.relu
#        )

        x = self._dense(
            x,
            1,
            "fc6",
            activation=tf.nn.relu
        )
        return x

    def _dense(self, x: tf.Tensor, units: int, name: str, activation=tf.nn.tanh) -> tf.Tensor:
        return tf.layers.dense(
            x,
            units=units,
            activation=activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self._regularization),
            name=name
        )

    def feed_dict(self, batch: data.PairBatch, training: bool = True):

        return {
            **super().feed_dict(batch, training),
            self.x_scalar: np.stack(batch.patients["features"]),
        }

    def uses_images(self) -> bool:
        """
        Implementation of :func:`BasicModel.uses_images`. This model does not uses images to work.

        :return: :any:`False` since this model does not use images to work
        """
        return False

class ClinicalVolumeSiamese3(BasicSiamese):
        r"""
        Model that uses only radiomic features as input to train

        It has the same parameters as :class:`BasicSiamese`

        It only uses the radiomic features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_

        **Attributes**:

        Includes the same attributes as :class:`BasicSiamese` and adds the following ones:

        :var ScalarOnlySiamese.x_scalar: Radiomic features obtained with
                                         `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
        :vartype ScalarOnlySiamese.x_scalar: tf.Tensor
        """

        def __init__(self, number_features: int, **kwargs):
            #: Radiomic features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
            self.x_scalar = tf.placeholder(tf.float32, [None, number_features])

            super().__init__(**kwargs)

        def _sister(self):
            # Out: [batch, 500]
            x = self.x_scalar
            x = self._dense(
                x,
                80,
                "fc1"
            )
            y = self._dense(
                x,
                5,
                "fc1y"
            )

            x = self._dense(
                x,
                40,
                "fc2"
            )
            x = tf.layers.dropout(
                x,
                rate=self._dropout,
                training=self.training
            )
            x = self._dense(
                tf.concat([x, y], 1),
                20,
                "fc3"
            )

            x = self._dense(
                x,
                10,
                "fc6",
                activation=tf.nn.relu6
            )
            return x

        def _dense(self, x: tf.Tensor, units: int, name: str, activation=tf.nn.tanh) -> tf.Tensor:
            return tf.layers.dense(
                x,
                units=units,
                activation=activation,
                kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self._regularization),
                name=name
            )

        def feed_dict(self, batch: data.PairBatch, training: bool = True):
            return {
                **super().feed_dict(batch, training),
                self.x_scalar: np.stack(batch.patients["features"]),
            }

        def uses_images(self) -> bool:
            """
            Implementation of :func:`BasicModel.uses_images`. This model does not uses images to work.

            :return: :any:`False` since this model does not use images to work
            """
            return False






class ScalarOnlyInceptionSiamese(BasicSiamese):
    r"""
    Model that uses only radiomic features as input to train

    It has the same parameters as :class:`BasicSiamese`

    It only uses the radiomic features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_

    **Attributes**:

    Includes the same attributes as :class:`BasicSiamese` and adds the following ones:

    :var ScalarOnlySiamese.x_scalar: Radiomic features obtained with
                                     `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
    :vartype ScalarOnlySiamese.x_scalar: tf.Tensor
    """

    def __init__(self,number_features, **kwargs):
        #: Radiomic features obtained with `PyRadiomics <https://github.com/Radiomics/pyradiomics>`_
        self.x_scalar = tf.placeholder(tf.float32, [None, number_features])
        #self.y_scalar = tf.placeholder(tf.float32, [None, settings.NUMBER_FEATURES])

        super().__init__(**kwargs)

    def _sister(self):
        # Out: [batch, 500]
        x = self.x_scalar
        y = self.x_scalar
        z = y
        x = self._dense(
            x,
            50,
            "fc1"
        )
        x = tf.layers.dropout(
            x,
            rate=self._dropout,
            training=self.training
        )
        x = self._dense(
            x,
            50,
            "fc2"
        )
        x = tf.layers.dropout(
            x,
            rate=self._dropout,
            training=self.training
        )
        x = self._dense(
            x,
            20,
            "fc3"
        )
        x = tf.layers.dropout(
            x,
            rate=self._dropout,
            training=self.training
        )

        # Out: [batch, 200]
        x = self._dense(
            x,
            10,
            "fc4"
        )
        x = tf.layers.dropout(
            x,
            rate=self._dropout,
            training=self.training
        )
        # Out: [batch, 200]
        x = self._dense(
            x,
            10,
            "fc5"
        )
        # Out: [batch, 200]
        x = self._dense(
            x,
            5,
            "fc6"
        )

        x = tf.layers.dropout(
            x,
            rate=self._dropout,
            training=self.training
        )

        # Out: [batch, 50]
        x = self._dense(
            x,
            10,
            "f7"
        )

        z = self._dense(
            z,
            10,
            "fc11",
            # activation=tf.nn.relu
        )
        y = self._dense(
            y,
            10,
            "f21",
            #activation=tf.nn.relu
        )
        y = tf.layers.dropout(
            y,
            rate=self._dropout,
            training=self.training
        )
        y = self._dense(
            y+x,
            10,
            "fc22",
            activation=tf.nn.relu
        )
        x = self._dense(
            y + z,
            10,
            "fc31",
            activation=tf.nn.relu
        )
        return z

    def _myself(self):

        x = self.x_scalar
        x = self._dense(
            x,
            500,
            "fb1",
            # activation=tf.nn.relu
        )

        x = self._dense(
            x,
            100,
            "fb2",
            #activation=tf.nn.relu
        )
        x = self._dense(
            x,
            10,
            "fb3",
            # activation=tf.nn.relu
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
            self.x_scalar: np.stack(batch.patients["features"]),
        }

    def uses_images(self) -> bool:
        """
        Implementation of :func:`BasicModel.uses_images`. This model does not uses images to work.

        :return: :any:`False` since this model does not use images to work
        """
        return False

    def _model(self) -> tf.Tensor:
        """
        Implementation of :func:`BasicModel._model`

        :return: Tensor where a Siamese network has been applied to the input with shape ``[batch, 1]``
        """
        myself_out = self._myself()
        sister_out = self._sister()
        return self._contrastive_loss(sister_out,myself_out)

    def _contrastive_math(self):

        weight1 = tf.Variable(1., name="c_weight")
        weight2 = tf.Variable(10., name="sub_weight")

        # sub = tf.subtract(self.gathered_a, self.gathered_b, name="contrastive_sub")
        sub = tf.subtract(self.gathered_b, self.gathered_a, name="contrastive_sub")
        sub = tf.add(self.gathered_base, sub, name="contrastive_add")

        if self._use_distance:
            return weight1*tf.tanh(sub, name="contrastive_tanh")
        else:
            return tf.sigmoid(sub, name="contrastive_sigmoid")

    def _contrastive_loss(self, sister_out: tf.Tensor, myself_out: tf.Tensor):
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
        :return: Tensor with the contrastive loss, comparing the two sister's output with shape ``[batch, 1]``
        """
        device = '/gpu:0' if self._gpu_level >= 3 else '/cpu:0'
        self.logger.debug(f"Using device: {device} for contrastive loss")

        with tf.device(device):
            with tf.variable_scope("contrastive_loss"):
                self.gathered_base = tf.gather(myself_out, self.pairs_b, name="gather_base")
                self.gathered_a = tf.gather(sister_out, self.pairs_a, name="gather_a")
                self.gathered_b = tf.gather(sister_out, self.pairs_b, name="gather_b")

                self.gathered_base = tf.square(self.gathered_base, name="square_base")
                self.gathered_a = tf.square(self.gathered_a, name="square_a")
                self.gathered_b = tf.square(self.gathered_b, name="square_b")

                self.gathered_base = tf.reduce_sum(self.gathered_base, 1, keepdims=True, name="reduce_basic")
                self.gathered_a = tf.reduce_sum(self.gathered_a, 1, keepdims=True, name="reduce_b")
                self.gathered_b = tf.reduce_sum(self.gathered_b, 1, keepdims=True, name="reduce_a")

        return self._contrastive_math()


