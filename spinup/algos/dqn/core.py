import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape[0])
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError


def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]


def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(64, 64), activation=tf.nn.relu,
                     output_activation=tf.tanh, action_space=None):
    act_dim = action_space.n
    with tf.variable_scope('q'):
        # Q value for each action
        q = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    return q
