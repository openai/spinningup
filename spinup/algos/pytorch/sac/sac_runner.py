from minigrid_env_utils import make_simple_env, MINI_GRID_MEDIUM_16, SEED
from spinningup.spinup.algos.pytorch.sac.sac_agent import DiscreteSacAgent
import multiprocessing as mp

SUBAGENTS = ['rg', 'fg1', 'fg2', 'fg3']


def run_subagent(agent_key, env_key=MINI_GRID_MEDIUM_16):
    train_env = make_simple_env(env_key, SEED)
    test_env = make_simple_env(env_key, SEED, random_start=False)
    agent = DiscreteSacAgent(train_env.observation_space,
                             train_env.action_space,
                             agent_name=agent_key,
                             experiment_name=env_key,
                             num_epochs=200)

    agent.train(train_env, test_env=test_env)


def run_subagents_parallel():
    pool = mp.Pool(len(SUBAGENTS))
    pool.map(run_subagent, SUBAGENTS)



if __name__ == '__main__':
    run_subagents_parallel()
