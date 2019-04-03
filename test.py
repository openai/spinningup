from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo
import tensorflow as tf
import gym

# todo: define env_fn in terms of some gym environment

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=25, max_ep_len=20, epochs=250, logger_kwargs=logger_kwargs)