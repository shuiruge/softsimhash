import tensorflow as tf


def count(x, cond_fn, axis=None, name='count'):
  """Counts how many elements in x satisfying the condition `cond_fn(x)`.

  Args:
    x: Tensor.
    cond_fn: Callable takes `x` and returns a tensor with boolean dtype, and
      the the shape as `x`.
    axis: Integer or list of integers, along which the counting is taken.

  Returns:
    Scalar, with the same dtype as `x`.
  """
  with tf.name_scope(name):
    mask = tf.where(cond_fn(x), tf.ones_like(x), tf.zeros_like(x))
    return tf.reduce_sum(mask, axis=axis)


class SoftSimhash:
  """
  The standard discrete (hard) simhash is illustrated in the image:
    http://static.oschina.net/uploads/space/2013/0805/015507_uvv1_568818.jpg

  Symmetry-breaking:
    The standard (hard) simhash has re-scaling symmetry on weight. That is,
    replacing `w[i] -> c w[i]` for a constant `c` and any index `i` will not
    change the hard simhash value. However, this re-scaling does change the
    value of soft simhash. Thus, the re-scaling symmetry on weight does not
    hold for soft simhash.

  On negative weight:
    Weight of the simhash is generally positive, with magnigude reflecting
    the importance of the feature in the simhash process. However, if negative
    weight is allowed, then the training process can be significantly fasten,
    even though the final loss and the final accuracy are invariant. The
    negative weight can be explained as a combination of the the original
    weighting with an additional operation of the hash-code, i.e. flipping,
    wherein the absolute value of the weight reflects the importance, while
    the sign for whether flipping the hash code or not.

  Shortcuts for shape:
    * batch_size -> B
    * n_features -> F
    * codelen -> C

  Args:
    probs: Tensor with shape `[batch_size, codelen]`, as the `probs` of
      independent Bernoulli distributions.
  """

  def __init__(self, probs):
    self._probs = probs
    self._batch_size, self._codelen = self._probs.get_shape().as_list()

  @property
  def probs(self):
    return self._probs

  @property
  def codelen(self):
    return self._codelen

  @property
  def batch_size(self):
    return self._batch_size

  def hamming_distance(self, other):
    assert isinstance(other, SoftSimhash)
    with tf.name_scope('Hamming_distance'):
      sh = self.harden()
      other_sh = other.harden()

      def distinct_with_sh(other_sh):
        with tf.name_scope('distinct'):
          return tf.not_equal(sh, other_sh)

      return count(other_sh, distinct_with_sh, axis=1)  # [B]

  def harden(self, name='harden'):
    """Returns a tensor of the shape `[batch_size, codelen]`, with elements
    being either `0` or `1`, in float-dtype."""
    with tf.name_scope(name):
      with tf.name_scope('half'):
        half = tf.ones_like(self.probs) * 0.5
      return tf.where(tf.greater(self.probs, half),
                      tf.ones_like(self.probs),
                      tf.zeros_like(self.probs))

  @staticmethod
  def generate(weight,
               hash,
               name='softsimhash'):
    """
    Args:
      weight: Tensor with shape `[n_features]`.
      hash: Tensor with shape `[batch_size, n_features, codelen]`.
      name: String.

    Returns:
      A `SoftSimhash` instance.
    """
    # Check `n_features` consistancy
    n_features = weight.get_shape().as_list()[0]
    assert n_features == hash.get_shape().as_list()[1]

    with tf.name_scope(name):
      with tf.name_scope('signize_hash'):
        # The components of `signized_hash` are either `-1` or `1`
        hash = tf.cast(hash, weight.dtype)
        signized_hash = 2 * hash - 1  # [B, F, C]

      with tf.name_scope('probs'):
        with tf.name_scope('expand_dims'):
          weight = tf.expand_dims(weight, axis=0)  # [1, F]
          weight = tf.expand_dims(weight, axis=-1)  # [1, F, 1]
        weighted = tf.reduce_sum(signized_hash * weight,
                                 axis=1)  # [B, C]
        probs = tf.sigmoid(weighted)  # [B, C]

      return SoftSimhash(probs)
