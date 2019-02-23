import numpy as np
import tensorflow as tf
import scipy.signal

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

