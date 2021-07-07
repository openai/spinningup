import gym_minigrid
from gym_minigrid.wrappers import SimpleObsWrapper
import gym

# Define constant
MINI_GRID_SIMPLE_16 = 'MiniGrid-Simple-Deceptive-16x16-v0'
MINI_GRID_MEDIUM_16 = 'MiniGrid-Medium-Deceptive-16x16-v0'
MINI_GRID_SIMPLE_49 = 'MiniGrid-Simple-Deceptive-49x49-v0'
SEED = 1234


def make_simple_env(env_key, seed, random_start=True):
    env = SimpleObsWrapper(gym.make(env_key, random_start=random_start))
    env.seed(seed)
    return env