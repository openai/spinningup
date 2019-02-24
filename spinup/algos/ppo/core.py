import numpy as np
import tensorflow as tf
import scipy.signal

# the types of mathematical spaces for observations, actions, etc
# Discrete = a number representing a discrete value
# Box = a n-dimensional vector where values are bounded
# "example usage: self.action_space = spaces.Box(low=-10, high=10, shape=(1,))" -OpenAI
from gym.spaces import Box, Discrete

EPS = 1e-8 # TODO: figure out what this is

def combined_shape(length, shape=None):
  """
  combined_shape(length, shape=None)

  Returns the shape of a list of `length` tensors,
  each of which has the given shape.

  For example, combined_shape(4, (2, 3))
  yields (4, 2, 3), the shape of a 4x2x3
  tensor, which can be thought of as a list
  of 4 tensors with shape (2, 3).
  """
  if shape is None:
    return (length,)
  return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
  return tf.placeholder(dtype=tf.float32, shape=combined_shape(None, dim))

def placeholders(*args):
  return [placeholder(dim) for dim in args]

def placeholder_from_space(space):
  if isinstance(space, Box):
    return placeholder(space.shape)
  elif isinstance(shape, Discrete):
    # TODO: why don't we just return placeholder()
    # i think that does the same thing
    return tf.placeholder(dtype=tf.int32, shape=(None,))
  
  # we have only implemented dealing with obs/action spaces
  # which are an instance of Box or Discrete
  raise NotImplementedError

def placeholders_from_spaces(*args):
  return [placeholder_from_space(space) for space in args]

def get_vars(scope=''):
  """
  Get a list of all trainable variables in the given scope
  """
  return [x for x in tf.trainable_variables() if scope in x.name]

def count_vars(scope=''):
  """
  Get the number of variables in a given scope
  """
  v = get_vars(scope)
  return sum([np.prod(var.shape.as_list()) for var in v])

def discount_cumsum(x, discount):
  """
  "magic from rllab for computing discounted cumulative sums of vectors.

  input:
    vector x,
    [x0,
     x1,
     x2]
    
  output:
    [x0 + discount * x1 + discount^2 * x2,
     x1 + discount * x2,
     x2]
  " -OpenAI

  This rather opaque code uses the the following function:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
  The notes section on this page describe a bit what this computes.
  For now I'm planning to just accept that this works although I don't *fully*
  understand why to use this line of code; later it may be worth
  changing this to something a bit more transparent...
  Not sure if there is some reason to use this code over a more direct implementation.
  Maybe it's faster?

  TODO: come back to this and see if we should do this in a more opaque way.
  """
  return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
  """
  MLP = "multi-layer perceptron".
  This function constructs a dense neural network in tensorflow, and returns
  the tensorflow tensor representing the output.
  (The way tensorflow works is that you pass around this output tensor,
  and this tensor actually contains metadata that essentially contains the full
  neural network needed to compute that output from any given input.
  So when we return the output, we in effect return the entire neural network).

  The network has input tensor `x`,
  hidden layers with sizes described by `hidden_sizes` and activation
  functions given by `activation`,
  and an activation on the output layer given by `output_activation`.

  Note that the last value in `hidden_sizes` is the size of the output layer,
  so isn't really "hidden" in the usual sense.
  """
  for h in hidden_sizes[:-1]:
    x = tf.layers.dense(x, units=h, activation=activation)
  return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def gaussian_likelihood(x, mu, log_std):

  pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS))**2 + 2*log_std + np.log(2 * np.pi))
  
  # return the sum of the items in the pre_sum vector  
  return tf.reduce_sum(pre_sum, axis=1)

"""
Policies
"""

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
  # number of actions possible...they are numbered 0 through n
  act_dim = action_space.n

  # get a tensorflow neural network to give us a vector output
  # of pre-normalized log probabilities of each action
  logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)

  # then do a softmax to normalize the probabilities
  # so logp_ll is the log of the normalized probabilities of each action
  logp_all = tf.nn.log_softmax(logits)

  # squeeze removes all dimensions of size one
  # multinomial draws samples according to the multinomial distribution,
  # ie. according to the probabilities implied by the logits
  # https://www.tensorflow.org/api_docs/python/tf/random/multinomial
  # TODO: tf is deprecating multinomial;
  # we should change this to tf.random.categorical instead

  # we use these functions to create `pi`,
  # which will be a tensor containing the index
  # of the action we have selected (randomly, according to the
  # probabilities implied by the neural network)
  pi = tf.squeeze(tf.multinomial(logits, 1), axis=1)

  # TODO: come back and document this
  logp = tf.reducesum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)

  logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)

  return pi, logp, logp_pi

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
  # Get the length of the action vector
  # (here we are getting the last value in the action_space.shape, but
  # there should only be one value in the action_space vector, since
  # there the action space is a single-dimensional vector,
  # so we can just grab that)
  # (actually there may be multiple dimensions in the shape
  # if a is a collection of several actions, in which case the first
  # numbers in the shape will describe the dimensionality of the collection
  # of actions, and the final dimension will describe the shape of the actual action space)
  act_dim = a.shape.as_list()[-1]

  # create a densely connected neural network from inputs `x` to action space
  mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)

  log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))

  std = tf.exp(log_std)

  pi = mu + tf.random_normal(tf.shape(mu)) * std

  logp = gaussian_likelihood(a, mu, log_std)
  logp_pi = gaussian_likelyhood(pi, mu, log_std)

  return pi, logp, logp_pi



"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(64, 64), activation=tf.tanh
                     output_activation=None, policy=None, action_space=None):
    
  # "defalut policy builder depends on action space" -OpenAI
  if policy is None and isinstance(action_space, Box):
    policy = mpl_gaussian_policy
  elif policy is None and isinstance(action_space, Discrete):
    policy = mlp.categorical_policy
