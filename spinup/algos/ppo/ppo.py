import numpy as np
import tensorflow as tf
import gym
import time
import core

from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_sctatistics_scalar, mpi_fork, mpi_avg, proc_id, num_procs
from spinup.utils.logx import EpochLogger

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

"""

"Proximal Policy Optimization (by clipping), 

with early stopping based on approximate KL" -OpenAI

"""
def ppo(env_fn,
  # by default, use the neural network mlp we define in core
  actor_critic=core.mlp_actor_critic,
  ac_kwargs=dict(),
  seed=0,
  steps_per_epoch=4000,
  epochs=50,
  gamma=0.99,
  clip_ratio=0.2,
  pi_lr=3e-4,
  vf_lr=1e-3,
  train_pi_iters=80,
  train_v_iters=80,
  lam=0.97,
  max_ep_len=1000,
  target_kl=0.01,
  logger_kwargs=dict(),
  save_freq=10):
  """
  "Args:
  env_fn: A function which creates a copy of the environment.
  The environment must satisfy the OpenAI Gym API.

  actor_critic: A function with takes in placeholder symbols
  for state, ``x_ph``, and action ``a_ph``, and returns the main
  outputs from the agent's Tensorflow computation graph:

  ===========  ================  ======================================
  Symbol       Shape             Description
  ===========  ================  ======================================
  ``pi``       (batch, act_dim)  | Samples actions from policy given states.
  ``logp``     (batch,)          | Gives log probability according to
                                  | the policy, of taking actions ``a_ph``
                                  | in states ``x_ph``.
  ``logp_pi``  (batch,)          | Gives log probability, according to
                                  | the policy, of the action sampled by ``pi``.
  ``v``        (batch,)          | Gives the value estimate for states
                                  | in ``x_ph``.  (Critical: make sure
                                  | to flatten this!)
  ===========  ================  ======================================" -OpenAI
  Okay, quick interruption to OpenAI documentation here.
  actor_critic is the function which interfaces with tensorflow.  It takes in
  ``x_ph`` (x placeholder), ie. a representation of the current state, and
  ``a_ph``, a representation of the some actions.  (TODO: document
  *what* these actions are).
  actor_critic runs these inputs through the tensorflow graph and returns several
  pieces of information that are relevant to PPO; these are described above.

  Back to OpenAI:
  "
  ac_kwargs (dict): Any kwargs appropriate for actor_critic function
      you provided to PPO.

  seed (int): Seed for random number generators.

  setps_per_epoch (int): Number of steps of interaction (state-action pairs)
      for the agent and the environment in each epoch.

  epochs (int): Number of epochs of interaction (equivalent to
      number of policy updates) to perform.

  gamma (float): Discount factor. (Always between 0 and 1.)

  clip_ratio (float): Hyperparameter for clipping in the policy objective.
      Roughly: how far can the new policy go from the old policy while
      still profiting (improving the objective function)? The new policy
      can still go farther than the clip_ratio says, but it doesn't help
      on the objective anymore.  (Usually small, 0.1 to 0.3.)

  pi_lr (float): Learning rate for policy optimizer.

  vf_lr (float): Learning rate for value function optimizer.

  train_pi_iters (int): Maximum number of gradient descent steps to take
      on policy loss per epoch.  (Early stopping may cause optimizer
      to take fewer than this.)

  train_v_iters (int): Number of gradient descent steps to take on
      value funciton per epoch.

  lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
      close to 1).

  max_ep_len (int): Maximum length of trajectory / episode / rollout.

  target_kl (float): Roughly what KL divergence we think is appropriate
      between new and old policies after an update.  This will get used
      for early stopping.  (Usually small, 0.01 or 0.05.)

  logger_kwargs (dict): Keyword args for EpochLogger.

  save_freq (int): How often (in terms of gap between epochs) to save
      the current policy and value function." - OpenAI
  """
  logger = EpochLogger(**logger_kwargs)
  logger.save_config(locals())

  # modify the seed based on the process so if
  # we run this in multiple processes
  # simultaneously we don't do the
  # exact same thing
  seed += 10000 * proc_id()
  # set up our random stuff with this seed
  tf.set_random_seed(seed)
  np.random.seed(seed)

  # create the environment
  env = env_fn()
  obs_dim = env.observation_space.shape
  act_dim = env.action_space.shape

  # tell the policy (implemented in actor_critic function) what the action space is
  ac_kwargs['action_space'] = env.action_space

  # "Inputs to computation graph" -OpenAI
  # create tensorflow placeholders for observations (x_ph), actions (a_ph),
  # advantages (adv_ph), returns (ret_ph), log probabilities from the prev run (logp_old_ph)
  x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
  adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

  # "Main outputs from computation graph" -OpenAI
  # essentially here we fill in the tensorflow graph so we can compute
  # the pi, logp, logp_pi, and v tensors based on the
  # x_ph and a_ph we created above
  pi, logp, logp_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)

  # "Need all placeholders in *this* order later (to zip with data from buffer)" -OpenAI
  all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

  # "Every step, get: action, value, and logprob" -OpenAI
  # we later feed this list into tf.session.run()
  # to tell it to compute the value of pi, v, logp_pi
  # using the tensorflow graph we have created
  get_action_ops = [pi, v, logp_pi]

  # Experience buffer

  # number of steps per epoch per process
  local_steps_per_epoch = int(steps_per_epoch / num_procs())

  buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

  # Count the number of parameters we are gonna be training,
  # both for the policy and for the value function
  var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
  logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

  # PPO objectives
  # TODO

  # Info (useful to watch during learning)
  # TODO

  #Optimizers
  # TODO

  # initialize the tensorflow computation graph's parameters
  # with values
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # "Sync params across processes" -OpenAI
  sess.run(sync_all_params())

  # Setup model saving
  logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})

  def update():
    # TODO
    pass

  start_time = time.time()

  # initialize the variables we use while training
  # o = observation (env.reset() returns initial observation)
  # r = reward = (starts as 0)
  # d = done? (whether current episode in env is over)
  # ep_ret = episode return
  # ep_len = length of episode so far
  o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

  # "Main loop: collect experience in env and update/log each epoch"
  for epoch in range(epochs):
    for t in range(local_steps_per_epoch):
      
      # run the computation of the action, value function, and probability of the action
      # using the most recent observation in the x_ph slot
      a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1,-1)})

      # save and log
      buf.store(o, a, r, v_t, logp_t)
      logger.store(VVals=v_t)

      # take the action we computed and advance the environment
      o, r, d, _ = env.step(a[0])
      ep_ret += r
      ep_len += 1

      terminal = d or (ep_len == max_ep_len)
      if terminal or (t==local_steps_per_epoch - 1):
        # TODO
        pass

    # every save_freq epochs,
    # save the state of the environment
    # also save the current state of our value function model
    # and policy
    # these are automatically saved by the save_state function
    # since we have already called logger.setup_tf_saver
    if (epoch % save_freq == 0) or (epoch == epochs - 1):
      logger.save_state({'env': env}, None)

    # perform PPO update!
    update()
    
    # Log info about epoch
    # TODO