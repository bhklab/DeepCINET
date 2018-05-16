import abc
import logging
from typing import Dict

import tensorflow as tf
import numpy as np

import data


class BasicModel:
    r"""
    Simple class to build a classification model.

    :param regularization: Regularization factor
    :param dropout: Dropout probability
    :param learning_rate: Learning rate for the gradient descent optimization algorithm

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
    """
    #: Threshold to cast a float number between ``0`` and ``1`` to a :any:`True` value for classification
    THRESHOLD = .5

    def __init__(self, regularization: float = .001, dropout: float = .2, learning_rate: float = 0.001, **ignored):
        self.logger = logging.getLogger(__name__)

        if len(ignored) > 0:
            self.logger.warning(f"There are unknown arguments {ignored}")

        #: **Attribute**: Placeholder for the labels, it has shape ``[batch]``
        self.y = tf.placeholder(tf.float32, [None, 1], name="Y")

        #: **Attribute**: Placeholder to tell the model if we are training (:any:`True`) or not (:any:`False`)
        self.training = tf.placeholder(tf.bool, shape=(), name="Training")

        #: **Attribute**: Regularization factor
        self._regularization = regularization

        #: **Attribute**: Dropout probability
        self._dropout = dropout

        #: **Attribute**: Probability of :math:`\hat{y} = 1`
        self.y_prob = self._model()  # This method is inherited and modified by its inheritors

        #: **Attribute**: Estimation of :math:`\hat{y}` by using :any:`BasicModel.y_prob` and
        #: :any:`BasicModel.THRESHOLD`
        self.y_estimate = tf.greater_equal(self.y_prob, self.THRESHOLD)

        with tf.variable_scope("loss"):
            #: **Attribute**: Classification loss using the negative log loss function
            self.classification_loss = tf.losses.log_loss(self.y, self.y_prob, scope="classification_loss")

            #: **Attribute**: L2 norm of the weights to add to the loss function to regularize
            self.regularization_loss = tf.losses.get_regularization_loss()

            #: **Attribute**: Total loss to be minimized with the optimizer
            self.total_loss = tf.add(self.classification_loss, self.regularization_loss, name="final_loss")

        with tf.variable_scope("c-index"):
            y_bool = tf.greater_equal(self.y, self.THRESHOLD)
            equals = tf.equal(y_bool, self.y_estimate)

            #: **Attribute**: Number of good predictions found on the current batch
            self.good_predictions = tf.cast(tf.count_nonzero(equals), tf.float32)

            with tf.variable_scope("batch_size"):
                batch_size = tf.cast(tf.shape(self.y, name="y_shape")[0], tf.float32, name="cast")

            #: **Attribute**: Concordance index for the current batch
            self.c_index = self.good_predictions/batch_size

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.minimizer = optimizer.minimize(self.total_loss)

        # Create summaries
        with tf.variable_scope("summaries"):
            tf.summary.scalar("loss", self.total_loss)
            tf.summary.scalar("c-index", self.c_index)
            tf.summary.scalar("classification_loss", self.classification_loss)
            tf.summary.scalar("regularization_loss", self.regularization_loss)

            for var in tf.trainable_variables():
                # We have to replace `:` with `_` to avoid a warning that ends doing this replacement
                tf.summary.histogram(str(var.name).replace(":", "_"), var)

        self.summary = tf.summary.merge_all()

    @abc.abstractmethod
    def _model(self) -> tf.Tensor:
        """
        Abstract method, the model should be build inside this method. Classes that Inherit :any:`BasicModel`
        should implement this method to create the model

        :return: Tensor with shape ``[batch, 1]`` with the probability of single class classification.
        """

    def feed_dict(self, batch: data.PairBatch, training: bool = True) -> Dict:
        """
        Get the ``feed_dict`` required by Tensorflow when calling ``sess.run(...)``. Classes that inherit
        :class:`BasicModel` should reimplement this function

        :param batch: Information about the batch, usually provided by :func:`BatchData.batches`
        :param training: Whether we are training or not. Useful for training layers like dropout where we do not
                         want to apply dropout if we are not training
        :return: Dictionary that can be feed to the ``feed_dict`` parameter of ``sess.run(...)``.
        """
        return {
            self.y: batch.pairs["labels"].values.reshape(-1, 1),
            self.training: training
        }

    @abc.abstractmethod
    def uses_images(self) -> bool:
        """
        Tells us if the model uses images. If it does not use images then loading images from disk can be avoided.
        This can have a huge performance boos since loading images from disk is a slow operation.

        :return: :any:`True` if the model needs images to work, otherwise returns :any:`False`
        """


class BasicSiamese(BasicModel):
    r"""
    Simple class to build a siamese model. Uses the contrastive loss for comparison

    :param gpu_level: Amount of GPU to be used with the model

                            0. No GPU usage
                            1. Only second conv layers
                            2. All conv layers
                            3. All layers and parameters are on the GPU
    :param regularization: Regularization factor
    :param dropout: Dropout probability
    :param learning_rate: Learning rate for the gradient descent optimization algorithm

    **Attributes**:

    Includes the same attributes as :class:`BasicModel` and adds the following ones:

    :var BasicSiamese.pairs_a: Indices to be selected as pairs A for the batch of input images, has shape ``[batch]``
    :var BasicSiamese.pairs_b: Indices to be selected as pairs B for the batch of input images, has shape ``[batch]``
    :var BasicSiamese.gathered_a: Output results for pairs members' A. It has shape ``[pairs_batch, last_layer_units]``
    :var BasicSiamese.gathered_b: Output results for pairs members' B. It has shape ``[pairs_batch, last_layer_units]``
    :var BasicSiamese._gpu_level: Amount of GPU that should be used when evaluating the model
    :vartype BasicSiamese.pairs_a: tf.Tensor
    :vartype BasicSiamese.pairs_b: tf.Tensor
    :vartype BasicSiamese.gathered_a: tf.Tensor
    :vartype BasicSiamese.gathered_b: tf.Tensor
    :vartype BasicSiamese._gpu_level: int
    """

    def __init__(self, gpu_level: int = 0, **kwargs):
        #: **Attribute**: Amount of GPU to be used with the model
        self._gpu_level = gpu_level

        #: **Attribute**: Placeholder for the indices of the first pairs (A)
        self.pairs_a = tf.placeholder(tf.int32, [None], name="pairs_a")

        #: **Attribute**: Placeholder for the indices of the second pairs (B)
        self.pairs_b = tf.placeholder(tf.int32, [None], name="pairs_b")

        #: **Attribute**: Output results for pairs members' A. It has shape ``[pairs_batch, last_layer_units]``
        self.gathered_a = None

        #: **Attribute**: Output results for pairs members' B. It has shape ``[pairs_batch, last_layer_units]``
        self.gathered_b = None

        super().__init__(**kwargs)

    def _model(self) -> tf.Tensor:
        """
        Implementation of :func:`BasicModel._model`

        :return: Tensor where a Siamese network has been applied to the input with shape ``[batch, 1]``
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
        :return: Tensor with the contrastive loss, comparing the two sister's output with shape ``[batch, 1]``
        """
        device = '/gpu:0' if self._gpu_level >= 3 else '/cpu:0'
        self.logger.debug(f"Using device: {device} for contrastive loss")
        with tf.device(device):
            with tf.variable_scope("contrastive_loss"):
                self.gathered_a = tf.gather(sister_out, self.pairs_a, name="gather_a")
                self.gathered_b = tf.gather(sister_out, self.pairs_b, name="gather_b")

                self.gathered_a = tf.square(self.gathered_a, name="square_a")
                self.gathered_b = tf.square(self.gathered_b, name="square_b")

                self.gathered_a = tf.reduce_sum(self.gathered_a, 1, keepdims=True, name="reduce_b")
                self.gathered_b = tf.reduce_sum(self.gathered_b, 1, keepdims=True, name="reduce_a")

                sub = tf.subtract(self.gathered_a, self.gathered_b, name="contrastive_sub")
                # sub = tf.subtract(self.gathered_b, self.gathered_a, name="contrastive_sub")
                return tf.sigmoid(10*sub, name="contrastive_sigmoid")

    def feed_dict(self, batch: data.PairBatch, training: bool = True) -> Dict:
        return {
            **super().feed_dict(batch, training),
            self.pairs_a: batch.pairs["pA_id"].values,
            self.pairs_b: batch.pairs["pB_id"].values,
        }


class BasicImageSiamese(BasicSiamese):
    r"""
    Basic class to build a siamese model that uses images as input.

    The construct arguments are the same as the :class:`BasicSiamese`.

    **Attributes**:

    Includes the same attributes as :class:`BasicSiamese` and adds the following ones:

    :var BasicImageSiamese.x_image: Batch of input images, has shape ``[batch, 64, 64, 64, 1]``
    :vartype BasicImageSiamese.x_image: tf.Tensor
    """

    def __init__(self, **kwargs):
        #: **Attribute**: Placeholder for the image input, it has shape ``[batch, 64, 64, 64, 1]``
        self.x_image = tf.placeholder(tf.float32, [None, 64, 64, 64, 1], name="X")

        super().__init__(**kwargs)

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

    def feed_dict(self, batch: data.PairBatch, training: bool = True) -> Dict:
        """
        Method to create the ``feed_dict`` required by Tensorflow when passing the data. Classes that inherit
        :class:`BasicSiamese` must re-implement this method if they use different tensors than:

            - :any:`BasicImageSiamese.x_image`
            - :any:`BasicModel.y`
            - :any:`BasicSiamese.pairs_a`
            - :any:`BasicSiamese.pairs_b`

        :param batch: Data containing for the current batch, usually this would be generated by
                      :func:`~BatchData.batches`
        :param training: Whether we are training or not. Useful for training layers like dropout where we do not
                         want to apply dropout if we are not training
        :return: Return the ``feed_dict`` as a dictionary
        """
        return {
            **super().feed_dict(batch, training=training),
            self.x_image: np.stack(batch.patients["images"].values)
        }

    def uses_images(self) -> bool:
        """
        Implementation of :func:`BasicModel.uses_images`.

        :return: :any:`True`, the model uses images as input to work
        """
        return True
