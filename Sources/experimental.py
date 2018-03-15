import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    # Just some text example code
    im = tf.zeros([2, 3, 3, 1], tf.float32)
    im_shape = im.get_shape().as_list()
    print(im_shape)

    a = tf.Variable([[1, 2, 3], [4, 5, 6]], tf.float32)
    take = tf.Variable([0])
    b = tf.expand_dims(tf.to_float(a), axis=-1)
    b = tf.expand_dims(b, axis=-1)
    c = tf.tile(b, [1, 1, *im_shape[1:-1]])
    c = tf.transpose(c, [0, 2, 3, 1])
    zero = tf.Variable(0., tf.float32)
    print(c.dtype)

    d = tf.concat([c, im], axis=-1)
    e = tf.where(d > 5, d, tf.zeros(d.shape))

    init = tf.global_variables_initializer()

    # Sum
    for i in range(2):
        zero = zero + tf.reduce_sum(d[i])

    with tf.Session() as sess:
        sess.run(init)

        print(sess.run([zero, tf.reduce_sum(e), tf.gather(e, take)]))

