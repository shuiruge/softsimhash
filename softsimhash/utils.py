import numpy as np
import tensorflow as tf


def np_softplus(x, limit=30):
  x = np.array(x)
  return np.where(x > np.ones_like(x) * limit,
                  x, np.log(1.0 + np.exp(x)))


def l2_log_regularize(weight, name='l2_log_regularize'):
    """Notice that the (soft) simhash, as a linear algorithm, is sensitive to
    the relative value between weight-components, like `w[i] / w[j]`, rather
    than to the absolute magnitude, so we regularize the weight by its
    logarithmic value. We thus say it's regularized by a "L2-log-distance".

    Args:
      weight: Tensor with shape `[n_features]` and float-dtype. All components
        shall be positive.

    Returns:
      A float scalar.
    """
    with tf.name_scope(name):
      # `1e-8` is for numerical stability
      return tf.reduce_mean(tf.square(tf.log(weight + 1e-8)))
