import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    # Just some text example code
    im = tf.placeholder(tf.float32, shape=[None, 2, 2, 5])
    im_shape = im.get_shape().as_list()
    print(im_shape)

    a = tf.Variable([[1, 2, 3], [4, 5, 6]], tf.float32)
    b = tf.expand_dims(a, axis=-1)
    b = tf.expand_dims(b, axis=-1)
    c = tf.tile(b, [1, 1, *im_shape[1:-1]])
    c = tf.transpose(c, [0, 2, 3, 1])

    with tf.Session() as sess:
        sess.run([a.initializer])

        print(sess.run([a]))

        print(sess.run([tf.shape(b), tf.shape(c)]))

        print(sess.run([b, c, im], feed_dict={im: np.zeros((5, 2, 2, 5))}))

