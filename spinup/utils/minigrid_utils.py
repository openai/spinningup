import gym_minigrid
from gym_minigrid.wrappers import SimpleObsWrapper
import gym


def make_simple_env(env_key, seed, random_start=True):
    env = SimpleObsWrapper(gym.make(env_key, random_start=random_start))
    env.seed(seed)
    return env