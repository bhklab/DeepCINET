import tensorflow as tf


class Siamese:

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 64, 64, 64, 1])
        self.y = tf.placeholder(tf.float32, [None, 1])

    def build_model(self):
        pass

    @staticmethod
    def sister(x: tf.Tensor) -> tf.Tensor:
        """
        Build one branch (sister) of the siamese network

        :param x: Initial input of shape ``[batch, 64, 64, 64, 1]``
        :return: Tensor of shape ``[batch, 1]``
        """

        # In: [batch, 64, 64, 64, 1]
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





