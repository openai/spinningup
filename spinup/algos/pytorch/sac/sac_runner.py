import multiprocessing as mp
import torch
from copy import deepcopy
from collections import defaultdict
from gym_minigrid.envs.deceptive import DeceptiveEnv
from python.minigrid_env_utils import SimpleObsWrapper
from python.runners.env_reader import read_map, read_grid_size, read_goals, read_start
from spinningup.spinup.algos.pytorch.sac.sac_agent import DiscreteSacAgent, ContinuousSacAgent, SacFactory
from python.path_manager import get_all_model_names
import python.path_manager as path_manager
from python.intention_recognition import IntentionRecognitionFactory
from vis_tools import append_results_to_json, convert_to_time_density

sacPathManager = path_manager.Sac()
valueIterationPathManager = path_manager.ValueIteration()


def run_simple(agent_key='rg'):
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


def train_subagent(map_num, agent_name, discrete=True):
    train_env, map_name = read_map(map_num, random_start=False, terminate_at_any_goal=False, goal_name=agent_name,
                                   discrete=discrete)
    test_env, map_name = read_map(map_num, random_start=False, terminate_at_any_goal=False, goal_name=agent_name,
                                  discrete=discrete)
    experiment_name = f'pretrained-sac-{map_name}{map_num}' if discrete else f'continuous-pretrained-sac-{map_name}{map_num}'
    agent = SacFactory.create(state_space=train_env.observation_space,
                              action_space=train_env.action_space,
                              subagent_name=agent_name,
                              experiment_name=experiment_name,
                              discount_rate=path_manager.HyperParams.DISCOUNT_FACTOR,
                              discrete=discrete,
                              num_epochs=200,
                              pi_lr=1e-3,
                              critic_lr=1e-3)
    agent.train(train_env=train_env, test_env=test_env)


def train_subagents_parallel():
    for map_number in range(1, 2):
        discrete = False
        agent_names = get_all_model_names(map_number)
        pool = mp.Pool(len(agent_names))
        pool.starmap(train_subagent, [(map_number, name, discrete) for name in agent_names])


def run_honest_agent(map_num, discrete=True):
    env, map_name = read_map(map_num, random_start=False, terminate_at_any_goal=False, goal_name='rg',
                             discrete=discrete)
    agent = torch.load(
        sacPathManager.get_path(agent_type='sac', map_name=f'{map_name}{map_num}', agent_name='rg', discrete=discrete)
    )
    start_state = read_start(map_num)
    goals, _ = read_goals(map_num)
    agent_names = path_manager.get_all_model_names(map_num, discrete)

    intention_recognition = IntentionRecognitionFactory.create(discrete=discrete,
                                                               state_space=env.observation_space,
                                                               action_space=env.action_space,
                                                               all_models=[valueIterationPathManager.get_path(
                                                                   map_name=f'{map_name}{map_num}',
                                                                   agent_name=name) for name in agent_names],
                                                               all_model_names=agent_names,
                                                               start_state=start_state,
                                                               goals=goals,
                                                               map_num=map_num)

    state_visitation_dict = defaultdict(int)
    state = env.reset()
    done = False
    path_cost = 0
    num_steps = 0
    max_steps = 2000

    while not done and num_steps < max_steps:
        env.render()
        state_visitation_dict[str(state)] += 1
        state = torch.as_tensor(state, dtype=torch.float32)
        action = agent.act(state, deterministic=True)
        path_cost += 1

        if intention_recognition is not None:
            _ = intention_recognition.predict_goal_probabilities(state, action)

        next_state, reward, done, info = env.step(action)
        state = next_state
        num_steps += 1
    env.close()
    state_visitation_dict[str(state)] += 1
    if intention_recognition is not None:
        rg_probs = intention_recognition.candidate_probabilities_dict['rg']
        rg_probs_fp = path_manager.ResultPaths.get_rg_probs_json_path(agent_type='honest', discrete=discrete)
        path_cost_fp = path_manager.ResultPaths.get_path_cost_json_path(agent_type='honest', discrete=discrete)
        rg_probs_vs_time_density = convert_to_time_density(rg_probs)
        append_results_to_json(rg_probs_fp, key=f'{map_name}{map_num}', results=rg_probs_vs_time_density)
        append_results_to_json(path_cost_fp, key=f'{map_name}{map_num}', results=float(path_cost))


if __name__ == '__main__':
    # run_subagents_parallel()
    # for i in range(21, 24):
        # train_subagent(map_num=i, agent_name='rg', discrete=False)
    run_honest_agent(map_num=23, discrete=False)
