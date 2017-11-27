import tensorflow as tf
import math


def random_uniform_initializer(shape, name, val, trainable=True):
    out = tf.get_variable(name, shape=list(shape), dtype=tf.float32,
                          initializer=tf.random_uniform_initializer(minval=-val, maxval=val, dtype=tf.float32),
                          trainable=trainable)
    return out


def xavier_initializer(shape, name, trainable=True):
    val = math.sqrt(6. / sum(shape))
    return random_uniform_initializer(shape, name, val, trainable=trainable)


def random_normal_initializer(shape, name, mean=0., stddev=1, trainable=True):
    return tf.get_variable(name, shape=list(shape), dtype=tf.float32,
                           initializer=tf.random_normal(shape, mean=mean, stddev=stddev,
                                                                    dtype=tf.float32),
                           trainable=trainable)


def random_truncated_normal_initializer(shape, name, mean=0., stddev=1, trainable=True):
    return tf.get_variable(name, shape=list(shape), dtype=tf.float32,
                           initializer=tf.truncated_normal(shape, mean=mean, stddev=stddev,
                                                                       dtype=tf.float32), trainable=trainable)
