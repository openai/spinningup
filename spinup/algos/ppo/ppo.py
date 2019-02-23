import numpy as np
import tensorflow as tf
import core

from spiningup.utils.mpi_tools import mpi_sctatistics_scalar

class PPOBuffer:
  """
  "A buffer for storing trajectories experienced by a PPO agent interacting
  with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
  for calculating the advantages of state-action pairs." -OpenAI
  """

  def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
    """
    `obs_dim` = shape of tensor representing an observation of the agent
    `act_dim` = shape of tensor representing an action of the agent
    `size` = maximum number of steps for the trajectory, ie. max needed size of buffer
    `gamma` = discount factor
    `lam` = lambda value for temporal difference value function updating
      - for information, see 
      https://amreis.github.io/ml/reinf-learn/2017/11/02/reinforcement-learning-eligibility-traces.html
    """
    # lists of observations and actions taken
    self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
    self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)

    # lists of advantages, rewards, returns (ie. discounted sum of rewards)
    # value functions, and ??? `logp` for each time step
    # TODO: figure out what `logp` is
    self.adv_buf = np.zeros(size, dtype=np.float32)
    self.rew_buf = np.zeros(size, dtype=np.float32)
    self.ret_buf = np.zeros(size, dtype=np.float32)
    self.val_buf = np.zeros(size, dtype=np.float32)
    self.logp_buf = np.zeros(size, dtype=np.float32)

    self.gamma, self.lam = gamma, lam

    # ptr is a pointer to the next time slot in the arrays/buffers which
    # is going to be filled in
    # path_start_idx is a pointer to the time slot at which
    # the most recent (or present) path began
    # (in case the buffer contains information from multiple
    # paths)
    # max_size is the maximum capacity for the array
    self.ptr, self.path_start_idx, self.max_size = 0, 0, size

  def store(self, obs, act, rew, val, logp):
    """
    "Append one timestep of agent-environment interaction to the buffer." -OpenAI
    """
    # make sure we have room in the buffer to add this time step
    assert self.ptr < self.max_size

    self.obs_buf[self.ptr] = obs
    self.act_buf[self.ptr] = act
    self.ret_buf[self.ptr] = rew
    self.val_buf[self.ptr] = val
    self.logp_buf[self.ptr] = logp

    self.ptr += 1
  
  def finish_path(self, last_val=0):
    """
    "Call this at the end of a trajectory, or when one gets cut off
    by an epoch ending.  This looks back in the buffer to where the
    trajectory started, and uses rewards and value etimates from
    the whole trajectory to compute advantage estimates with GAE-Lambda,
    as well as compute the rewards-to-go for each state, to use as
    the targets for the value function.

    The 'last_val' argument should be 0 if the trajectory ended
    because the agent reached a terminal state (died), and otherwise
    should be V(s_T), the value function estimated for the last state.
    This allows us to bootstrap the reward-to-go calculation to account
    for timesteps beyond the arbitrary episode horizon (or epoch cutoff)."
    -OpenAI
    """
    # a slice object which can be used to index into arrays
    # to get the relevant data for the current path
    path_slice = slice(self.path_start_idx, self.ptr)

    # rewards & vals for this path, including the value estimate
    # of the final state in the path [0 if it is a terminal state]
    rews = np.append(self.rew_buf[path_slice], last_val)
    vals = np.append(self.val_buf[path_slice], last_val)

    # "the next two lines implement GAE-Lambda" advantage calculation" -OpenAI

    # okay, I'm gonna put in some comments to explain what is going on here.

    # first we calculate 'deltas'.  This is an array such that
    # deltas[i] = (rews[i] + gamma * vals[i + 1]) - vals[i]
    # in other words, deltas[i] is the difference between our 
    # new estimate of the value of the state[i], and 
    # the previous value function's estimate of the value of state i
    # 
    # our new estimate of value[state[i]]
    # is based on the reward achieved at time step i, plus our old value
    # function's estimate of the value of state[i + 1]

    # so deltas[i] is an estimate of how much better the trajectory
    # was than the expected values for the current policy.
    # Note that this estimate is based ONLY ON ONE REWARD,
    # namely that received at time step i.
    deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

    # then we determine the "advantage" for each of these times
    # functionally, adv_buf[i] is similar to deltas[i] in that
    # it provides a measure of how much better this trajectory was
    # than that expected under the given policy.
    # however, this measure of advantage integrates information
    # from the entire trajectory, rather than just from a single reward
    
    # adv_buf[i] = delta_i + gamma * lambda * delta_(i + 1) + (gamma * lambda)^2 * delta_(i + 2) + ...
    # in other words, it is a sum of the delta value throughout the future of this trajectory,
    # there the term of delta[i + k] is discounted by (gamma * lambda)^k = lambda^k * gamma^k
    # Discounting by gamma^k is expected, since delta[i + k] occurs k time steps in the future.
    # we discount by lambda^k based on the approximation that the the further in the future
    # we go, the less influence the current state has on the reward recieved.
    # in other words, we have the lambda term so that we consider advantage observed very far
    # in the future to have less to do with present actions than advantage observed in the short term
    self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)[:-1]

    # TODO: consider integrating eligibility traces into this implementation
    # this may already be done in some form which I just haven't understood yet,
    # and it may be that the format of the algorithm doesn't allow for it to be
    # done in any useful way, but this is worth coming back to. -george 2019-02-16

    # fill in the return buffer with the discounted sum of the rewards
    # recieved during this path
    self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

    # advance the path_start_idx so it points to the beginning
    # of the next path which will be put into the buffer
    self.path_start_idx = self.ptr

  def get(self):
    """
    "Call this at the end of an epoch to get all of the data from
    the buffer, with advantages appropriately normalized (shifted to have
    mean zero and std one).  Also, resets some pointers in the buffer."
    -OpenAI

    Returns the list of vectors [obs_buf, act_buf, adv_buf, ret_buf, logp_buf].

    NOTE:
    I am concerned that something is lost by normalizing.  For example,
    it is possible that during this epoch, we had an advantage with a very high
    mean, because the actions taken during the trajectories happened to be quite
    a bit better.  If we normalize, we lose the information that this
    epoch represented a solid improvement; instead we just learn which
    particular actions tried out during this epoch gave the biggest
    improvement.  We don't just care how the actions in this trajectory
    compare to one another, though; we care about how they compare to all
    possible actions, including ones we didn't try.

    That said, since this sample was just sampled from the given trajectory,
    we expect the mean advantage to already be close to 0, since we expect
    our value function is already pretty accurate to the mean performance of
    trajectories under this policy.  Thus renormalizing to a mean of zero
    usually shouldn't represent a big change, and it may be that the mathematical
    convenience of this is worth it.  It may also be that there is a more fundamentally
    important reason to normalize which I'm not currently understanding.

    TODO: consider whether we should change this so it DOES NOT normalize advantages
    (or just don't normalize the mean, even if you do normalize the variance)
    """
    # "buffer has to be full before you can get" -OpenAI
    assert self.ptr == self.max_size
    self.ptr, self.path_start_idx = 0, 0

    # "the next two lines implement the advantage normalization trick" -OpenAI
    adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf) # TODO: import openAI mpi statistics stuff
    self.adv_buf = (self.adv_buf - adv_mean) / adv_std

    return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf]