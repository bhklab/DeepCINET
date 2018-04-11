import tensorflow as tf


class Siamese:


    def __init__(self):
        pass

    def build_model(self):
        pass

    def build_sister(self, x: tf.Tensor):
        x = tf.layers.conv3d(
            x,
            filters=30,
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            activation=tf.nn.relu,
            name="conv1"
        )

        x = tf.layers.conv3d(
            x,
            filters=40,
            kernel_size=(3, 3, 3),
            activation=tf.nn.relu,
            name="conv2"
        )

        x = tf.layers.conv3d(
            x,
            filters=40,
            kernel_size=(3, 3, 3),
            activation=tf.nn.relu,
            name="conv3"
        )

        x = tf.layers.conv3d(
            x,
            filters=50,
            kernel_size=(3, 3, 3),
            activation=tf.nn.relu,
            name="conv4"
        )

        x = tf.contrib.layers.fully_connected(
            x,
            100,
        )






