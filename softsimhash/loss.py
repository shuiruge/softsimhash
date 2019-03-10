import tensorflow as tf
from softsimhash.softsimhash import SoftSimhash


class Loss:
  """
  For each pair of simhashes in batch, the total JS-distance is defined as
  the sum of the JS-distance between the Bernoulli probabilities of the pair
  over code-length, signed with `+` if the simhashes are for the same device
  and `-` if not. The loss is the mean over all pairs in the batch.

  Args:
    eps: Tiny positive float, for numerical stability.
  """

  def __init__(self, eps=1e-8):
    self.eps = eps

  def log(self, x, name='log'):
    with tf.name_scope(name):
      return tf.log(x + self.eps)

  def kl_divergence(self, p, q, name='KL_divergence'):
    """KL-divergence between Bernoulli distributions with probabilities
    `p` and `q`, element-wisely.

    Args:
      p: Tensor.
      q: Tensor with the same shape and dtype as `p`.
      name: String.

    Returns
      Tensor with the same shape and dtype as `p`.
    """
    with tf.name_scope(name):
      kl_div = 0
      kl_div += p * (self.log(p) - self.log(q))
      kl_div += (1 - p) * (self.log(1 - p) - self.log(1 - q))
      return kl_div

  def js_distance(self, p, q, name='JS_distance'):
    """JS-distance between Bernoulli distributions with probabilities
    `p` and `q`, element-wisely.

    Args:
      p: Tensor.
      q: Tensor with the same shape and dtype as `p`.
      name: String.

    Returns
      Tensor with the same shape and dtype as `p`.
    """
    with tf.name_scope(name):
      return (self.kl_divergence(p, q) + self.kl_divergence(q, p)) / 2

  def __call__(self,
               softsimhash0,
               softsimhash1,
               target,
               name='loss'):
    """
    Args:
      softsimhash0: Tensor with shape `[batch_size, n_features, codelen]`, with
        elements being either `0` or `1`.
      softsimhash1: Like `softsimhash0`.
      target: Tensor with shape `[batch_size]`, representing the probability
        that `softsimhash0` and `softsimhash1` represent the same device.
      name: String.

    Returns:
      Scalar.
    """
    assert isinstance(softsimhash0, SoftSimhash)
    assert isinstance(softsimhash1, SoftSimhash)

    with tf.name_scope(name):
      # Abbreviation
      ssh0, ssh1 = softsimhash0, softsimhash1

      with tf.name_scope('equal_prob'):
        # Compute the probability of being equal for each bit
        # If let X, Y ~ Bernoulli, then
        # P(x=y) = P(x=0) P(y=0) + P(x=1) P(y=1)
        # For instance,
        #   P(x=0) = 0.5 and P(y=0) = 0.5 give P(x=y) = 0.5;
        #   P(x=0) = 1.0 and P(y=0) = 1.0 give P(x=y) = 1.0; and
        #   P(x=0) = 1.0 and P(y=0) = 0.0 give P(x=y) = 0.0.
        # [B, C]
        equal_prob = (ssh0.probs * ssh1.probs
                      + (1 - ssh0.probs) * (1 - ssh1.probs))  # noqa:W503

      with tf.name_scope('target'):  # preprocess `target`
        # Unify dtype
        target = tf.cast(target, equal_prob.dtype)  # [B]
        # Expand dimension to `[B, C]`
        target = tf.expand_dims(target, axis=-1)  # [B, 1]
        target = target * tf.ones_like(equal_prob)  # [B, C]

      with tf.name_scope('total_js_distance'):
        js_distance = self.js_distance(equal_prob, target)  # [B, C]
        # Total JS-distance over `codelen`
        total_js_distance = tf.reduce_sum(js_distance, axis=1)  # [B]

      loss = tf.reduce_mean(total_js_distance)  # []
      return loss
