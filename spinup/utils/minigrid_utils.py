import gym_minigrid
from gym_minigrid.wrappers import SimpleObsWrapper
import gym

# Define constant
MINI_GRID_16 = 'MiniGrid-Deceptive-16x16-v0'
MINI_GRID_49 = 'MiniGrid-Deceptive-49x49-v0'
SEED = 1234


def make_simple_env(env_key, seed):
    env = SimpleObsWrapper(gym.make(env_key))
    env.seed(seed)
    return env