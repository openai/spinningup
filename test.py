from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo
import tensorflow as tf
import gym

# todo: define env_fn in terms of some gym environment

env_fn = lambda: gym.make('LunarLanderContinuous-v2')
ac_kwargs = dict(hidden_sizes=[64, 64], activation=tf.nn.relu)
logger_kwargs = dict(output_dir='data/test1', exp_name='test1')

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, max_ep_len=1000, epochs=500, logger_kwargs=logger_kwargs)