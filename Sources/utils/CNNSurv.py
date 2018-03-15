import tensorflow as tf


class CNNSurv:
    """
    Create a model for survival prediction. In this case the model will combine an image and features extracted from
    this image
    """
    def __init__(self, image_input_size, data_input_size):
        # Create placeholders
        self.x_image = tf.placeholder(tf.float32, shape=(None, *image_input_size))
        self.x_data = tf.placeholder(tf.float32, shape=(None, *data_input_size))

        # The y_data contains the Time and the Event -> 2 dimensions
        self.y_data = tf.placeholder(tf.float32, shape=(None, 2))
        self._create_computation_graph()

    def _create_computation_graph(self):
        # In this case the graph follows two different paths, the image path and the data path until they get merged
        # in the middle

        image_pass = tf.layers.conv3d(
            self.x_image,
            filters=64,
            kernel_size=(6,)*3,
            activation=tf.nn.relu
        )

        data_pass = tf.layers.dense(
            self.x_data,
            units=64,
            activation=tf.nn.relu
        )

        # Spatial filling, data_pass has shape [batch, units], we need to convert it to [batch, height, width, units]
        image_shape = tf.shape(image_pass)
        image_shape = image_shape[1:(len(image_shape) - 1)]  # support for 2D and 3D

        data_pass = tf.expand_dims(data_pass, -1)
        data_pass = tf.expand_dims(data_pass, -1)
        data_pass = tf.expand_dims(data_pass, -1)
        data_pass = tf.tile(data_pass, [1, 1, *image_shape])




