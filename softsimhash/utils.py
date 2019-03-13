import numpy as np
import tensorflow as tf


def np_softplus(x, limit=30):
  x = np.array(x)
  return np.where(x > np.ones_like(x) * limit,
                  x, np.log(1.0 + np.exp(x)))


def std_log_regularize(weight, name='std_log_regularize'):
    """Notice that the (soft) simhash, as a linear algorithm, is sensitive to
    the relative value between weight-components, like `w[i] / w[j]`, so we
    regularize the weight by its logarithmic value.

    Args:
      weight: Tensor with shape `[n_features]` and float-dtype. All components
        shall be positive.

    Returns:
      A float scalar.
    """
    with tf.name_scope(name):
      # `1e-8` is for numerical stability
      _, std = tf.nn.moments(tf.log(weight + 1e-8), axes=[0])
      return std
