import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    # Just some text example code
    im = tf.zeros([2, 3, 3, 1], tf.float32)
    im_shape = im.get_shape().as_list()
    print(im_shape)

    a = tf.Variable([[1, 2, 3], [4, 5, 6]], tf.float32)
    b = tf.expand_dims(tf.to_float(a), axis=-1)
    b = tf.expand_dims(b, axis=-1)
    c = tf.tile(b, [1, 1, *im_shape[1:-1]])
    c = tf.transpose(c, [0, 2, 3, 1])
    print(c.dtype)

    d = tf.concat([c, im], axis=-1)

    with tf.Session() as sess:
        sess.run([a.initializer])

        # print(sess.run([a, im]))
        print(sess.run([tf.shape(d), d]))

