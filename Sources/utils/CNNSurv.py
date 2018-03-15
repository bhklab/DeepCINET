import tensorflow as tf


def spatial_filling(input_data: tf.Tensor, image_shape: tf.TensorShape, dims: int):
    """
    Fill input_data to have the same shape as image_shape (excluding the batch and channels).
    If input_data has shape [batchA, units] and image_shape is [batch, height, width, depth, channels]
    The output will be [batchA, height, width, depth, units]
    :param input_data:
    :param image_shape:
    :param dims:
    :return:
    """
    input_data = tf.expand_dims(input_data, -1)
    input_data = tf.expand_dims(input_data, -1)
    input_data = tf.expand_dims(input_data, -1)
    input_data = tf.tile(input_data, [1, 1, *image_shape[1:-1]])

    indices = [0, *range(2, dims + 2), 1]
    return tf.transpose(input_data, indices)


class CNNSurv:
    """
    Create a model for survival prediction. In this case the model will combine an image and features extracted from
    this image
    """
    def __init__(self, image_input_size, data_input_size):
        # Create placeholders
        self.x_image = tf.placeholder(tf.float32, shape=(None, *image_input_size))
        self.x_scalar = tf.placeholder(tf.float32, shape=(None, *data_input_size))
        self.dims = len(image_input_size) - 1  # Exclude the channels count

        # Y data contains two elements that are used in different ways
        self.y_E = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_T = tf.placeholder(tf.float32, shape=(None, 1))


        # Create graph
        self._create_computation_graph()

    def _create_computation_graph(self):
        # In this case the graph follows two different paths, the image path and the data path until they get merged
        # in the middle

        image_step = tf.layers.conv3d(
            self.x_image,
            filters=64,
            kernel_size=(6,)*3,
            strides=(2, 2, 2),
            activation=tf.nn.relu
        )

        image_step = tf.layers.max_pooling3d(
            image_step,
            pool_size=(3, 3, 3),
            strides=(2, 2, 2)
        )

        scalar_step = tf.layers.dense(
            self.x_scalar,
            units=64,
            activation=tf.nn.relu
        )

        # Spatial filling, data_pass has shape [batch, units], we need to convert it to [batch, height, width, units]
        scalar_step = spatial_filling(scalar_step, image_step.shape, self.dims)

        # Concatenate along the channels axis
        data_step = tf.concat([image_step, scalar_step], axis=-1)
        data_step = tf.contrib.layers.fully_connected(
            data_step,
            num_outputs=64,
            activation=tf.nn.relu
        )

        loss = self._negative_log_likelihood(data_step)

    def _negative_log_likelihood(self, risk: tf.Tensor)-> tf.Tensor:
        """
        Negative log likelihood function with batch of elements

        :param risk: Tensor of shape [batch_size, depth]
        :return:
        """
        # For now use the same implementation as DeepSurv
        hazard_ratio = tf.exp(risk)

        batch_size, _ = risk.shape
        # NOTE: This part has yet to be vectorized
        cost = tf.Variable(0., tf.float32)
        for i in range(batch_size):
            if self.y_E[i] <= 0:
                continue

















