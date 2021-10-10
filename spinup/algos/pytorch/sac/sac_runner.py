from minigrid_env_utils import make_simple_env
from spinningup.spinup.algos.pytorch.sac.sac_agent import DiscreteSacAgent
import multiprocessing as mp
import python.path_manager as constants
from gym_minigrid.envs.deceptive import DeceptiveEnv
from python.minigrid_env_utils import SimpleObsWrapper
from python.runners.env_reader import read_map, read_grid_size
from copy import deepcopy
SEED = constants.Random.SEED

# ENV = config['simple_16']['env']
# SUBAGENTS = config['simple_16']['all_models']['value_iteration']
# AGENT_NAMES = config['simple_16']['all_model_names']

AGENT_NAMES1 = ['rg', 'fg1', 'fg2']
AGENT_NAMES2 = ['rg', 'fg1', 'fg2', 'fg3', 'fg4']

# ARGS = [(13, AGENT_NAMES2), (14, AGENT_NAMES2), (15, AGENT_NAMES2), (16, AGENT_NAMES2)]

ARGS = [(1, AGENT_NAMES1)]

def run_simple(agent_key = 'rg'):
    map = SimpleObsWrapper(DeceptiveEnv.load_from_file(
        fp=f'/Users/alanlewis/PycharmProjects/DeceptiveReinforcementLearning/maps/drl/empty.map',
        optcost=1,
        start_pos=(47, 47),
        real_goal=(1, 1, 'rg'),
        fake_goals=[(47, 1, 'fg1')],
        random_start=False,
        terminate_at_any_goal=False,
        goal_name=agent_key))

    train_env = deepcopy(map)
    test_env = deepcopy(map)
    train_env.seed(42)
    test_env.seed(42)

    agent = DiscreteSacAgent(train_env.observation_space,
                             train_env.action_space,
                             agent_name=agent_key,
                             experiment_name=f'ignore_from_file_simple',
                             start_steps=40000,
                             max_ep_len=49 ** 2,
                             steps_per_epoch=16000,
                             num_epochs=100,
                             policy_update_delay=1,
                             seed=42,
                             alpha=0.2,
                             polyak=0.995,
                             hidden_dimension=64,
                             critic_lr=1e-3,
                             pi_lr=1e-3)
    agent.train(train_env, test_env=test_env)


def run_subagent(num_env, agent_key):
    train_env, map_name = read_map(num_env, random_start=False, terminate_at_any_goal=False, goal_name=agent_key)
    test_env, map_name = read_map(num_env, random_start=False, terminate_at_any_goal=False, goal_name=agent_key)
    grid_size = read_grid_size(num_env)
    agent = DiscreteSacAgent(train_env.observation_space,
                             train_env.action_space,
                             agent_name=agent_key,
                             experiment_name=f'pretrained-sac-{map_name}{num_env}',
                             update_after=1000,
                             start_steps=80000,
                             max_ep_len=grid_size**2,
                             steps_per_epoch=20000,
                             batch_size=100,
                             num_epochs=100,
                             policy_update_delay=1,
                             seed=42,
                             alpha=0.2,
                             polyak=0.995,
                             hidden_dimension=64,
                             critic_lr=1e-3,
                             pi_lr=1e-3)
    agent.train(train_env, test_env=test_env)


def run_subagents_parallel():
    for arg in [(25, AGENT_NAMES1)]:
        map_number = arg[0]
        agent_names = arg[1]
        pool = mp.Pool(len(agent_names))
        pool.starmap(run_subagent, [(map_number, name) for name in agent_names])


if __name__ == '__main__':
    run_subagents_parallel()