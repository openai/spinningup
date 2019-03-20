from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo
import tensorflow as tf
from airover.rover_gym.gym_rover.envs.GoalPointEnv1 import GoalPointEnv1

env_fn = lambda: GoalPointEnv1()
ac_kwargs = dict(hidden_sizes=[64, 64], activation=tf.nn.relu)
logger_kwargs = dict(output_dir='data/script_test1', exp_name='script_test1')


ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=25, max_ep_len=20, epochs=250, logger_kwargs=logger_kwargs)