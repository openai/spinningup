from minigrid_env_utils import make_simple_env
from spinningup.spinup.algos.pytorch.sac.sac_agent import DiscreteSacAgent
import multiprocessing as mp
import python.constants as constants
from python.runners.config import config
from gym_minigrid.envs.deceptive import DeceptiveEnv
from python.minigrid_env_utils import SimpleObsWrapper
from python.runners.env_reader import read_map

SEED = constants.Random.SEED

# ENV = config['simple_16']['env']
# SUBAGENTS = config['simple_16']['all_models']['value_iteration']
# AGENT_NAMES = config['simple_16']['all_model_names']

ENV = constants.EnvKeys.MINI_GRID_EMPTY_PATH
AGENT_NAMES1 = ['rg', 'fg1', 'fg2']
AGENT_NAMES2 = ['rg', 'fg1', 'fg2', 'fg3', 'fg4']
#
# ARGS = [(1, AGENT_NAMES1), (4, AGENT_NAMES1), (5, AGENT_NAMES2),
#         (6, AGENT_NAMES2), (7, AGENT_NAMES2), (8, AGENT_NAMES2)]

ARGS = [(9, AGENT_NAMES1)]


def run_subagent(num_env, agent_key):
    train_env, map_name = read_map(num_env, random_start=False, terminate_at_any_goal=False, goal_name=agent_key)
    test_env, map_name = read_map(num_env, random_start=False, terminate_at_any_goal=False, goal_name=agent_key)
    agent = DiscreteSacAgent(train_env.observation_space,
                             train_env.action_space,
                             agent_name=agent_key,
                             experiment_name=f'from_file_{map_name}{num_env}',
                             start_steps=40000,
                             max_ep_len=49**2,
                             steps_per_epoch=16000,
                             # batch_size=200,
                             # update_every=100,
                             # update_after=2000,
                             num_epochs=100,
                             seed=42,
                             alpha=0.2,
                             hidden_dimension=64,
                             critic_lr=1e-3,
                             pi_lr=1e-3)
    agent.train(train_env, test_env=test_env)


def run_subagents_parallel():
    for arg in ARGS:
        map_number = arg[0]
        agent_names = arg[1]
        pool = mp.Pool(len(agent_names))
        pool.starmap(run_subagent, [(map_number, name) for name in agent_names])


if __name__ == '__main__':
    run_subagents_parallel()