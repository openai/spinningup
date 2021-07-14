from minigrid_env_utils import make_simple_env
from spinningup.spinup.algos.pytorch.sac.sac_agent import DiscreteSacAgent
import multiprocessing as mp
import python.constants as constants

SEED = constants.Random.SEED

config = {
    # optimal simple 16= (94.34168696318805, 94.34168696318805)
    'minigrid-simple-16': (constants.EnvKeys.MINI_GRID_SIMPLE_16, ['rg', 'fg1']),
    # optimal medium 16 = (95.37952065439602, 98.275697827198, 94.89411413599005, 97.23786413599005)
    'minigrid-medium-16': (constants.EnvKeys.MINI_GRID_MEDIUM_16, ['rg', 'fg1', 'fg2', 'fg3']),
    # optimal obstacle 16 = (94.89411413599005, 95.37952065439602)
    'minigrid-obstacle-16': (constants.EnvKeys.MINI_GRID_OBSTACLE_16, ['rg', 'fg1']),
    # 'minigrid-obstacle-16-2': (constants.EnvKeys.MINI_GRID_OBSTACLE_16, ['rg', 'rg', 'rg', 'rg', 'rg', 'rg'])
}

ENV = config['minigrid-obstacle-16'][0]
SUBAGENTS = config['minigrid-obstacle-16'][1]
ALPHA = [0.1, 0.2, 0.4, 0.6, 0.8, 1]


def run_subagent(agent_key, alpha=0.2, env_key=ENV):
    train_env = make_simple_env(env_key, SEED)
    test_env = make_simple_env(env_key, SEED, random_start=False)
    agent = DiscreteSacAgent(train_env.observation_space,
                             train_env.action_space,
                             agent_name=agent_key,
                             experiment_name=env_key,
                             start_steps=10000,
                             num_epochs=500,
                             seed=43,
                             alpha=alpha,
                             hidden_dimension=64,
                             critic_lr=1e-4,
                             pi_lr=1e-4)
    agent.train(train_env, test_env=test_env)

def run_subagents_parallel():
    pool = mp.Pool(len(SUBAGENTS))
    pool.starmap(run_subagent, zip(SUBAGENTS))


if __name__ == '__main__':
    run_subagents_parallel()
