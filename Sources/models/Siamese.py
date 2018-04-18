import tensorflow as tf


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

            # self.y_estimate = tf.sigmoid(self.gathered_a - self.gathered_b)
            self.y_estimate = tf.tanh(self.gathered_a - self.gathered_b)

    @staticmethod
    def sister(x: tf.Tensor) -> tf.Tensor:
        """
        Build one branch (sister) of the siamese network

        :param x: Initial input of shape ``[batch, 64, 64, 64, 1]``
        :return: Tensor of shape ``[batch, 1]``
        """

        # In: [batch, 64, 64, 64, 1]
        # Out: [batch, 31, 31, 31, 30]
        with tf.device('/cpu:0'):
            x = tf.layers.conv3d(
                x,
                filters=30,
                kernel_size=3,
                strides=2,
                activation=tf.nn.relu,
                name="conv1"
            )

        # with tf.device('/cpu:0'):
            # Out: [batch, 29, 29, 29, 40]
            x = tf.layers.conv3d(
                x,
                filters=40,
                kernel_size=3,
                activation=tf.nn.relu,
                name="conv2"
            )

        with tf.device('/gpu:0'):
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

        with tf.device('/cpu:0'):

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

    def loss(self):
        # return tf.losses.log_loss(self.y, self.y_estimate)
        loss = self.y_estimate*(2*(1 - self.y) - 1)
        loss = tf.reduce_sum(loss)/self.batch_size
        return loss
        # return (1 - self.c_index())**2

    def c_index(self):
        c_index = ((self.y_estimate + 1)/2)*self.y
        return tf.reduce_sum(c_index)/self.batch_size

