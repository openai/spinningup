import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete

EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32,shape=combined_shape(None, dim))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError

def shape_from_space(space, batch_dim=None):
    if isinstance(space, Box):
        return combined_shape(batch_dim, space.shape)
    elif isinstance(space, Discrete):
        return (batch_dim,)
    raise NotImplementedError

def type_from_space(space):
    if isinstance(space, Box):
        return np.float32
    elif isinstance(space, Discrete):
        return np.int32
    raise NotImplementedError

def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]

def mlp(x, hidden_sizes=(64,64,), activation=tf.tanh, output_activation=None, one_hot = None, discount_factor = None):
    # support for discrete inputs
    if one_hot != None:
        assert(len(x.shape) == 1)
        x = tf.one_hot(x, depth=one_hot)
    # support for scalar inputs
    elif len(x.shape) == 1:
        x = x[:,None]
    # send in the discount factor as one of the inputs for the value function
    if discount_factor != None:
        pad = tf.broadcast_to(discount_factor, shape=(x.shape[0], 1))
        x = tf.concat([x, pad], axis = -1)

    if x.dtype not in [tf.bfloat16, tf.float16, tf.float32, tf.float64, tf.complex64, tf.complex128]:
        x = tf.dtypes.cast(x, tf.float32)
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]

def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


"""
Policies
"""

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space, one_hot = None):
    act_dim = action_space.n
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None, one_hot)
    logp_all = tf.nn.log_softmax(logits)
    # pi = tf.math.argmax(logits, axis=1)
    pi = tf.squeeze(tf.multinomial(logits, 1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space, one_hot = None):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation, one_hot)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi

"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh,\
                     output_activation=None, policy=None, action_space=None, observation_space = None,\
                     discount_factor = None):

    if isinstance(observation_space, Discrete):
        one_hot = observation_space.n
    else:
        one_hot = None

    # default policy builder depends on action space
    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy

    with tf.variable_scope('pi'):
        pi, logp, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, action_space, one_hot)
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation=activation, output_activation=None, \
            one_hot=one_hot, discount_factor=discount_factor), axis=1)
    return pi, logp, logp_pi, v


def masked_suffix_sum(x, endmask, discount, axis=0, last_val = False):
    """
    x: time series
    endmask: indicator for end of sequences
    """
    x = tf.reverse(x, axis=[axis])
    endmask = tf.reverse(endmask, axis=[axis])
    p = list(range(len(x.get_shape())))
    p[axis] = 0
    p[0] = axis
    x = tf.transpose(x, perm=p)
    initializer = x[0] - x[0]
    endmask = tf.transpose(endmask, perm=p)
    def _scan_fn(cumsum, x_mask):
        x, mask = x_mask
        prev = cumsum * (1 - mask)
        if last_val:
            prev += x * mask
        return prev * discount + x
    ans = tf.scan(_scan_fn, (x, endmask), initializer = initializer)
    ans = tf.transpose(ans, perm=p)
    ans = tf.reverse(ans, axis=[axis])
    return ans

def exponential_avg(x, discount, batch_size, num_batches):
    exponent_array = np.zeros((batch_size, batch_size), dtype = np.float32)
    mask_array = np.zeros((batch_size, batch_size), dtype = np.float32)
    for i in range(batch_size):
        for j in range(batch_size):
            exponent_array[i,j] = i-j if i >= j else 0
            mask_array[i,j] = 1 if i >= j else 0
    batch_mat = tf.math.pow(discount, exponent_array) * mask_array
    ret = []
    for i in range(num_batches):
        data = x[i * batch_size: (i+1)*batch_size]
        data = tf.reshape(data, (1, batch_size))
        prod = tf.tensordot(data, batch_mat, ([1,0]))
        ret.append(tf.reshape(prod, (batch_size,)))
    return tf.concat(ret, axis = 0)


import os

def save_variables(save_path, variables=None, sess=None):
    import joblib
    sess = sess or tf.get_default_session()
    variables = variables or tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    ps = sess.run(variables)
    save_dict = {v.name: value for v, value in zip(variables, ps)}
    dirname = os.path.dirname(save_path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)
    joblib.dump(save_dict, save_path)

def load_variables(load_path, variables=None, sess=None):
    import joblib
    sess = sess or tf.get_default_session()
    variables = variables or tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    loaded_params = joblib.load(os.path.expanduser(load_path))
    restores = []
    for v in variables:
        if v.name in loaded_params:
            restores.append(v.assign(loaded_params[v.name]))

    sess.run(restores)

