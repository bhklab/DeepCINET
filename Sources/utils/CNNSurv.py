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



