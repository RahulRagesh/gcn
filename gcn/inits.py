import tensorflow as tf
import numpy as np


def uniform(shape, scale=0.05, name=None, collections = None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name, collections=collections)

def normal(shape, mean = 0, std = 0.1, name=None, collections = None):
    """Normal init."""
    initial = tf.random.normal(shape, mean=mean, stddev=std, dtype=tf.dtypes.float32)
    return tf.Variable(initial, name=name, collections=collections)

def glorot(shape, name=None, collections = None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name, collections=collections)


def zeros(shape, name=None, collections = None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name, collections=collections)


def ones(shape, name=None, collections = None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name, collections=collections)

def from_tensor(tensor, shape, name=None, collections = None):
    return tf.Variable(tensor, name=name, dtype=tf.float32, collections=collections)