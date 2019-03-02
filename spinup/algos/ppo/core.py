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
  elif isinstance(space, Discrete):
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
  """
  Compute the log of the pdf of a gaussian with the given `log_std`
  evaluated at (`x` - `mu`).
  
  In other words, this function returns the log of a measure
  of how likely it would be for a gaussian with mean `mu` and std
  exp(`log_std`) to have the value `x` randomly selected from it.
  This "measure of likelihood" is the probability density function
  (pdf) for such a gaussian.

  Note that `x`, `mu`, and `log_std`, are vectors, so the pdf
  of the overall distribution (over all slots in the vectors)
  is the product of the pdfs of each individual slot.
  """

  # this expression just calculates the log of the pdf of the gaussian for a single
  # vector index, as described in the function docstring.
  # note that since we are taking the *log* of the pdf, we add terms together
  # which are multiplied together in the pdf
  # also note that rather than dividing by the std_dev, like we do in the regular pdf,
  # we divide by (std_dev + EPS), where EPS (epsilon) is a tiny number we include
  # to ensure that we don't divide by zero if std_dev = 0.
  pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS))**2 + 2*log_std + np.log(2 * np.pi))
  
  # return the sum of the items in the pre_sum vector
  # since each item is the log of the pdf for a specific index,
  # when we sum these, we get the log of the product of each
  # individual pdf -- ie. the log of the pdf evaluated
  # at this vector as a whole
  return tf.reduce_sum(pre_sum, axis=1)

"""
Policies
"""

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
  """
  Constructs the tensorflow graph for a mlp-based policy over a discrete action space.

  Returns pointers to tensors in the constructed graph which are relevant outputs:
  
  `pi` - A tensor which is the index of the action to perform given input `x`
  `logp_pi` - The log of the probability that `pi` was chosen
  `logp` - The log of the probability that the policy would select action `a` given
    input `x` in the current state of the mlp.

  Note that this can also be used with `x` as a vector with several state vectors
  within, mapping to `pi`/`a` - a vector containing multiple actions to take, corresponding
  to the elements in the state vector.  In this case, the logp values are the log probabilities
  of the entire trajectory given by x and pi / x and a.

  For more commentary on why it makes sense to return `logp` and `logp_pi` in the same function,
  see comments for `mlp_gaussian_policy`.  (Short version: we aren't computing anything;
  we are just constructing the tensorflow graph.)
  """

  # number of actions possible...they are numbered 0 through n-1
  act_dim = action_space.n

  # get a tensorflow neural network to give us a vector output
  # of pre-normalized log probabilities of each action
  logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)

  # then do a softmax to normalize the probabilities
  # so logp_all is the log of the normalized probabilities of each action
  logp_all = tf.nn.log_softmax(logits)


  # now, create `pi`,
  # which will be a tensor containing the index
  # of the action we have selected (randomly, according to the
  # probabilities implied by the neural network)

  # the line that does this is dense, so here is some commentary:
  # squeeze removes all dimensions of size one, and
  # multinomial draws samples according to the multinomial distribution,
  # ie. according to the probabilities implied by the logits
  # https://www.tensorflow.org/api_docs/python/tf/random/multinomial
  # TODO: tf is deprecating multinomial;
  # we should probably change this to tf.random.categorical instead
  pi = tf.squeeze(tf.multinomial(logits, 1), axis=1)

  # calculate the log of the  probability of selecting the specific
  # actions (pi / a) given states x
  # to do this, use a one_hot on the action index to get a vector
  # with a one in that slot and 0s elsewhere,
  # then dot with logp_all (which we already constructed)
  # to get a the value of the probability of that specific action
  # reduce_sum will give us a tensor which is just a number with this value
  # (or the sum of the log probs of multiple actions, if we used this
  # function to calculate probabilities over a trajectory, ie.
  # x and a/pi both contain several elements, representing different
  # actions to take in different states.
  # in this case, by summing the log probs, we essentially
  # log the product of individual probabilities, ie. finding
  # the log prob of the entire trajectory)
  logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
  logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)

  return pi, logp, logp_pi

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
  """
  Constructs the tensorflow graph for a mlp-based gaussian policy over a continuous action space.

  In other words, this constructs a neural network (mlp) which maps from the state (x)
  to actions.  It then appends onto the tensorflow graph the addition of random
  gaussian noise, so that the policy is not deterministic.

  There are several outputs for this tensorflow graph constructed here:

  `pi` - a vector specifying the action
  this policy has chosen to perform given a specific x.  (This `pi` will be the output
  from the mlp + the random noise.)

  `logp_pi` - the log of probability (pdf) for this action.
    In other words, logp_pi gives a measure of how likely it was that, given the
    current state of the mlp and its parameters, the action pi would be selected
    after the noise was added in.  (Ie. logp_pi measures how far from the mean
    the random values were that were selected this time.)
    Since the action space is continuous, the raw probability of any specific
    action is zero, so instead of giving the log of that, this returns
    the log of the probability density function (pdf) describing
    the likelyhood of the policy (given its current parameters) selecting
    any given action.

  `logp` - this is similar to logp_pi, as it gives the log of the pdf describing
    the probabilities of selecting a specific action given the input x (in the current
    state of the mlp & its parameters). However, instead of giving the log of this pdf
    evaluated at the action `pi`, this instead evaluates it at the action in the tensor
    `a` which is passed into this function.
    
    (It might seem strange for this function ostentibly outputs both logp and logp_pi--
    why would the same function handle logp for a specific `pi` we compute given `x` AND
    for an action `a` we pass in?  Shouldn't these be handled by different functions?
    Well, keep in mind that this function `mlp_gaussian_policy` is merely constructing
    a tensorflow graph--it is not actually computing anything.  So yes, finding logp(a)
    and calculating `pi` given an `x`, and then finding logp_pi, are different actions
    that are done at different times--but both are based on outputs from the mlp we create
    here, so we construct the tensorflow graph to handle them in this same function.)
  """

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

  # set the standard deviation for the noise we calculate to be -1/2
  log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
  std = tf.exp(log_std)

  # calculate the action `pi` given the state `x` by passing `x` through
  # the mlp to get mu, then adding in some random guassian noise with the 
  # proper standard deviation
  pi = mu + tf.random_normal(tf.shape(mu)) * std

  # calculate the log of the pdf of the policy given x & the current
  # mlp parameters, evaluated either at a given action `a`
  # or at the `pi` we just computed
  logp = gaussian_likelihood(a, mu, log_std)
  logp_pi = gaussian_likelyhood(pi, mu, log_std)

  # return the important outputs we can now compute from
  # the tensorflow graph
  return pi, logp, logp_pi



"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(64, 64), activation=tf.tanh,
                     output_activation=None, policy=None, action_space=None):
  """
  Constructs the tensorflow graph for a policy and value function.

  Selects the type of policy to use depending on whether the
  action space is continuous or discrete.  Policies
  and value function are based on mlps with the given
  hidden_sizes and activation.

  Returns the following outputs from the tensorflow graph:

  `pi` - an action to perform according to this policy given input `x`
  `logp_pi` - the log of the probability (or pdf) of the policy
    choosing this action given `x`
  `logp` - the log probability (or pdf) of the policy choosing
    the action `a` given `x`.  (So we have a way in the tensorflow
    graph to find the probabilities of actions being selected
    which we never actually selected in the current state of the mlp.)
  `v` - the output from our value function estimator

  TODO: make this functionmore extensible.
  we might want different hidden_sizes and
  activation functions for the value function and the policy,
  but right now this function forces the same for both!
  """

  # "defalut policy builder depends on action space" -OpenAI
  # in other words, the policy works differently if we have
  # a discrete action space vs a continuous action space
  if policy is None and isinstance(action_space, Box):
    policy = mlp_gaussian_policy
  elif policy is None and isinstance(action_space, Discrete):
    policy = mlp_categorical_policy

  # construct the tensorflow graph for the chosen type of policy
  # and encapsulate it in the tf scope "pi"
  with tf.variable_scope('pi'):
    pi, logp, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, action_space)

  # also construct an mlp for the value function
  with tf.variable_scope('v'):
    v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)

  # return the important outputs of the tensorflow graph
  return pi, logp, logp_pi, v