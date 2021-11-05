# Generic imports
from copy import deepcopy
import itertools
import time
from abc import ABC, abstractmethod
from collections import defaultdict

# torch and numpy
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import torch.nn.functional as F

# gym stuff
import gym
import gym_minigrid
from spinup.utils.minigrid_utils import make_simple_env
from python.display_utils import VideoViewer

# local stuff
import spinup.algos.pytorch.sac.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.buffers import RandomisedSacBuffer
import python.path_manager as path_manager
from python.runners.env_reader import read_map

sacPathManager = path_manager.Sac()

SEED = path_manager.Random.SEED


class SacBaseAgent(ABC):
    def __init__(
            self,
            state_space,
            action_space,
            discount_rate=0.975,
            pi_lr=1e-3,
            critic_lr=1e-3,
            update_every=50,
            update_after=1000,
            max_ep_len=1000,
            seed=42,
            steps_per_epoch=4000,
            start_steps=10000,
            num_test_episodes=4,
            num_epochs=100,
            replay_size=int(1e6),
            save_freq=1,
            batch_size=100,
            polyak=0.995,
            policy_update_delay=2,
            alpha=0.02,
            experiment_name='sac-base-class',
            agent_name='rg'
    ) -> None:
        # set up logger
        logger_kwargs = setup_logger_kwargs(experiment_name + '-' + agent_name, seed)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        # seed the random stuff
        torch.manual_seed(seed)
        np.random.seed(seed)

        # agent name
        self.name = agent_name

        # store agent hyper-parameters
        self.discount_rate = discount_rate
        self.alpha = alpha  # This is the entropy regularisation coefficient
        self.pi_lr = pi_lr
        self.vf_lr = critic_lr
        self.polyak = polyak  # for updating the target model
        self.policy_update_delay = policy_update_delay  # delay policy and target critic updates to stabilise q-values

        # store episodes and epoch related details
        self.max_ep_len = max_ep_len
        self.num_epochs = num_epochs
        self.update_every = update_every
        self.update_after = update_after
        self.steps_per_epoch = steps_per_epoch
        self.start_steps = start_steps
        self.num_test_episodes = num_test_episodes
        self.total_steps_taken = 0

        # store details to track performance
        self.episode_reward = 0
        self.episode_length = 0
        self.epoch_number = 0
        # a count dict to track how much the agent has visited a state during training
        self.test_state_visitation_dict = defaultdict(int)
        self.train_state_visitation_dict = defaultdict(int)

        # store replay buffer store
        self.replay_size = replay_size
        self.save_freq = save_freq
        self.batch_size = batch_size

        # Store important environment details
        self.state_space = state_space
        self.action_space = action_space
        self.state_dim = state_space.shape

        # define stuff that should be over-ridden as None. I don't really like the abstract property way of doing it
        # right now
        self.action_dim = None
        self.actor_critic = None
        self.target_actor_critic = None
        self.replay_buffer = None
        self.pi_optimiser = None
        self.q_optimiser = None
        self.q_lr_schedule = None
        self.pi_lr_schedule = None

        # video viewer to look at agent performance
        self.video_viewer = VideoViewer()

    def update(self, data, time_step):
        # run a gradient descent step for q functions
        self.q_optimiser.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimiser.step()

        # record the losses
        self.logger.store(LossQ=loss_q.item(), **q_info)

        if time_step % self.policy_update_delay == 0:
            # freeze the Q-networks so that we don't waste computational effort computing the gradients for them during the
            # policy update
            for p in self.q_params:
                p.requires_grad = False

            # run a gradient descent step for the policy function
            self.pi_optimiser.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimiser.step()

            # unfreeze the Q-networks
            for p in self.q_params:
                p.requires_grad = True

            # Record things
            self.logger.store(LossPi=loss_pi.item(), **pi_info)

            # update the target networks using polyak averaging
            with torch.no_grad():
                for p, p_targ in zip(self.actor_critic.parameters(), self.target_actor_critic.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

    def select_action(self, state, deterministic=False):
        return self.get_action(state, deterministic)

    def add_experience(self, state, action, reward, next_state, done):
        # add to state visitation map
        self.train_state_visitation_dict[str(tuple(map(int, state)))] += 1
        # if the reward gets passed in as a dict, then extract the reward using the agent name
        if type(reward) == dict:
            reward = reward[self.name]

        self.episode_reward += reward
        self.episode_length += 1

        # if we complete without reward, then this is a result of early termination. To counter this, ignore the done
        # such that we bootstrap normally.
        # ignore the done signal if it comes from an artificial time horizon --> ie it comes from an artificial
        # ending rather than being a result of the agent's state
        if done and reward <= 0:
            done = False

        # if done and reward > 90:
        #     print("hello")

        self.replay_buffer.store(state, action, reward, next_state, done)

    def end_trajectory(self, test=False):
        if test:
            self.logger.store(TestEpRet=self.episode_reward, TestEpLen=self.episode_length)
        else:
            self.logger.store(EpRet=self.episode_reward, EpLen=self.episode_length)
        self.episode_reward = 0
        self.episode_length = 0

    def learn(self, time_step):
        # We only want to update the model after a certain number of experiences have been collected.
        # Then we want to update it periodically. This ensure that replay buffers are sufficiently uncorrelated
        if time_step >= self.update_after and time_step % self.update_every == 0:
            # We still want to keep a 1-1 ratio between number of actions and number of updates
            for j in range(self.update_every):
                data = self.replay_buffer.sample_batch(self.batch_size)
                self.update(data, j)

    def handle_end_of_epoch(self, time_step, test_env=None, test_env_key=None):
        # Handle the end of epoch: (1) Save the model. (2) test the agent. (3) log the results
        if (time_step + 1) % self.steps_per_epoch == 0:
            self.epoch_number = (time_step + 1) // self.steps_per_epoch

            # save the model at each save frequency or at the end
            if (self.epoch_number % self.save_freq == 0) or self.epoch_number == self.num_epochs:
                self.logger.save_state({'env': test_env}, None)
                self.logger.save_state_visitation_dict(self.test_state_visitation_dict,
                                                       'test_state_visitation_dict.json')
                self.logger.save_state_visitation_dict(self.train_state_visitation_dict,
                                                       'train_state_visitation_dict.json')

            self.test(test_env, test_env_key)

            self.log_stats(time_step)

            # adjust the learning rate at each epoch
            self.q_lr_schedule.step()
            self.pi_lr_schedule.step()

    def log_stats(self, time_step, epoch_number=None):
        epoch_number = self.epoch_number if epoch_number is None else epoch_number
        # Log info about epoch
        self.logger.log_tabular("AGENT", self.name)
        self.logger.log_tabular('Epoch', epoch_number)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('TestEpLen', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts', time_step)
        self.logger.log_tabular('Q1Vals', with_min_and_max=True)
        self.logger.log_tabular('Q2Vals', with_min_and_max=True)
        self.logger.log_tabular('LogPi', with_min_and_max=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.dump_tabular()

    def test(self, env, env_key):
        if env is None and env_key is not None:
            env = make_simple_env(env_key, SEED, random_start=False)  # test the agent with the actual start position
        else:
            env = deepcopy(env)
        for j in range(self.num_test_episodes):
            state = env.reset()
            self.test_state_visitation_dict[str(tuple(map(int, state)))] += 1
            done = False
            self.episode_length = 0
            while not (done or (self.episode_length == self.max_ep_len)):
                # Act deterministically because we are being tested
                action = self.get_action(state, deterministic=True)
                next_state, reward, done, _ = env.step(action)
                # add to the state visitation dict for the map
                self.test_state_visitation_dict[str(tuple(map(int, state)))] += 1
                if type(reward) == dict:
                    reward = reward[self.name]
                self.episode_reward += reward
                self.episode_length += 1
                state = next_state
            self.test_state_visitation_dict[str(tuple(map(int, state)))] += 1
            self.end_trajectory(test=True)

    def train(self, train_env, test_env=None, test_env_key=None):
        total_steps = self.steps_per_epoch * self.num_epochs
        start_time = time.time()
        state = train_env.reset()
        self.train_state_visitation_dict[str(tuple(map(int, state)))] += 1

        # collect experiences and update every epoch
        for t in range(total_steps):
            # train_env.render()

            # at the start, randomly sample actions from a uniform distribution for better exploration
            # only afterwards use the learned policy...
            if t > self.start_steps:
                action = self.get_action(state)
            else:
                action = train_env.action_space.sample()

            next_state, reward, done, _ = train_env.step(action)
            # done = False if self.episode_length == self.max_ep_len else done
            self.add_experience(state, action, reward, next_state, done)
            state = next_state

            # end of trajectory
            if done or self.episode_length == self.max_ep_len:
                self.end_trajectory()
                self.train_state_visitation_dict[str(tuple(map(int, state)))] += 1
                state = train_env.reset()

            # Update the model
            self.learn(time_step=t)

            # handle the end of each epoch
            self.handle_end_of_epoch(time_step=t, test_env=test_env, test_env_key=test_env_key)

    def interval_train(self, train_env, test_env=None, test_env_key=None, num_steps=100000):
        start_time = time.time()
        state = train_env.reset()
        self.train_state_visitation_dict[str(tuple(map(int, state)))] += 1

        # collect experiences and update every epoch
        for _ in range(num_steps):

            # at the start, randomly sample actions from a uniform distribution for better exploration
            # only afterwards use the learned policy...
            if self.total_steps_taken > self.start_steps:
                action = self.get_action(state)
            else:
                action = train_env.action_space.sample()

            next_state, reward, done, _ = train_env.step(action)
            self.add_experience(state, action, reward, next_state, done)
            state = next_state

            # end of trajectory
            if done or self.episode_length == self.max_ep_len:
                self.end_trajectory()
                self.train_state_visitation_dict[str(tuple(map(int, state)))] += 1
                state = train_env.reset()

            # Update the model
            self.learn(time_step=self.total_steps_taken)

            # handle the end of each epoch
            self.handle_end_of_epoch(time_step=self.total_steps_taken, test_env=test_env, test_env_key=test_env_key)
            self.total_steps_taken += 1
        end_time = time.time()
        train_time = end_time - start_time

        return train_time

    def save_state(self, epoch_number, train_env):
        if (epoch_number % self.save_freq == 0) or epoch_number == self.num_epochs:
            self.logger.save_state({'env': train_env}, None)
            self.logger.save_state_visitation_dict(self.test_state_visitation_dict,
                                                   'test_state_visitation_dict.json')
            self.logger.save_state_visitation_dict(self.train_state_visitation_dict,
                                                   'train_state_visitation_dict.json')

    def get_value_estimate(self, state, action):
        return self.actor_critic.get_value_estimate(state, action)

    def get_max_value_estimate(self, state):
        return self.actor_critic.get_max_value_estimate(state)

    @abstractmethod
    def compute_loss_q(self, data):
        raise NotImplementedError

    @abstractmethod
    def compute_loss_pi(self, data):
        raise NotImplementedError

    @abstractmethod
    def get_action(self, state, deterministic=False):
        raise NotImplementedError


class ContinuousSacAgent(SacBaseAgent):
    def __init__(self, state_space, action_space, hidden_dimension=64, num_hidden_layers=2, discount_rate=0.99,
                 pi_lr=1e-3, critic_lr=1e-3, update_every=50, update_after=1000, max_ep_len=1000, seed=42,
                 steps_per_epoch=4000, start_steps=10000, num_test_episodes=10, num_epochs=100, replay_size=int(1e6),
                 save_freq=1, batch_size=100, polyak=0.995, policy_update_delay=1, alpha=0.2,
                 experiment_name='continuous-sac-class-test', agent_name='rg') -> None:
        super().__init__(state_space=state_space, action_space=action_space, discount_rate=discount_rate, pi_lr=pi_lr,
                         critic_lr=critic_lr, update_every=update_every, update_after=update_after,
                         max_ep_len=max_ep_len, seed=seed, steps_per_epoch=steps_per_epoch, start_steps=start_steps,
                         num_test_episodes=num_test_episodes, num_epochs=num_epochs, replay_size=replay_size,
                         save_freq=save_freq, batch_size=batch_size, polyak=polyak,
                         policy_update_delay=policy_update_delay, alpha=alpha, experiment_name=experiment_name,
                         agent_name=agent_name)
        self.action_dim = action_space.shape
        # we need this to clamp the actions... Assume all actions have the same bound, otherwise we need to deal with each action bound separately
        self.action_limit = action_space.high

        # create the standard actor-critic network
        self.actor_critic = core.MLPActorCritic(self.state_space, self.action_space,
                                                **dict(hidden_sizes=[hidden_dimension] * num_hidden_layers))

        # create the target network (which is just a deep copy of the real network)
        self.target_actor_critic = deepcopy(self.actor_critic)

        # Freeze the target network with respect to the optimisers (we only want to update the target using polyak averaging)
        for p in self.target_actor_critic.parameters():
            p.requires_grad = False

        # store a list of params in the Q-networks
        self.q_params = itertools.chain(self.actor_critic.q1.parameters(), self.actor_critic.q2.parameters())

        # make a replay buffer
        self.replay_buffer = RandomisedSacBuffer(obs_dim=self.state_dim, act_dim=self.action_dim, size=self.replay_size)

        # count and log the number of variables in each model for informative purposes
        self.var_counts = tuple(
            core.count_vars(module) for module in [self.actor_critic.pi, self.actor_critic.q1, self.actor_critic.q2])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % self.var_counts)

        # make optimisers for the actor and the critic
        self.pi_optimiser = Adam(self.actor_critic.pi.parameters(), lr=self.pi_lr)
        self.q_optimiser = Adam(self.q_params, lr=self.vf_lr)
        self.pi_lr_schedule = ExponentialLR(optimizer=self.pi_optimiser, gamma=1)
        self.q_lr_schedule = ExponentialLR(optimizer=self.q_optimiser, gamma=1)

        # set up model saving
        self.logger.setup_pytorch_saver(self.actor_critic)

    def compute_loss_q(self, data):
        states, actions, rewards, next_states, dones = data['obs'], data['act'], data['rew'], data['next_obs'], data[
            'done']

        q1 = self.actor_critic.q1(states, actions)
        q2 = self.actor_critic.q2(states, actions)

        # do Bellman backup for Q functions
        with torch.no_grad():
            # use the target network to get the actions given the next state
            next_actions, next_logps = self.target_actor_critic.pi(next_states)

            # target q-values use the next states and next actions
            target_q1 = self.target_actor_critic.q1(next_states, next_actions)
            target_q2 = self.target_actor_critic.q2(next_states, next_actions)

            # use the minimum of the two as the actual target to avoid over-estimation problems
            target_q = torch.min(target_q1, target_q2)

            # determine backup (see SAC paper for more details on this derivation)
            backup = rewards + self.discount_rate * (1 - dones) * (target_q - self.alpha * next_logps)

        # compute losses using MSE
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # store useful logging info
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    def compute_loss_pi(self, data):
        states = data['obs']
        actions, logp = self.actor_critic.pi(states)

        q1 = self.actor_critic.q1(states, actions)
        q2 = self.actor_critic.q2(states, actions)
        q = torch.min(q1, q2)

        # calculate entropy regularised policy loss
        loss_pi = (self.alpha * logp - q).mean()

        pi_info = dict(LogPi=logp.detach().numpy())

        return loss_pi, pi_info

    def get_action(self, state, deterministic=False):
        return self.actor_critic.act(torch.as_tensor(state, dtype=torch.float32), deterministic=deterministic)


class DiscreteSacAgent(SacBaseAgent):

    def __init__(self, state_space, action_space, hidden_dimension=64, num_hidden_layers=2, discount_rate=0.975,
                 pi_lr=1e-3, critic_lr=1e-3, update_every=50, update_after=1000, max_ep_len=1000, seed=42,
                 steps_per_epoch=4000, start_steps=10000, num_test_episodes=10, num_epochs=100, replay_size=int(1e6),
                 save_freq=1, batch_size=100, polyak=0.995, policy_update_delay=2, alpha=0.2, experiment_name='ignore',
                 agent_name='rg') -> None:
        super().__init__(state_space, action_space, discount_rate, pi_lr, critic_lr, update_every, update_after,
                         max_ep_len, seed, steps_per_epoch, start_steps, num_test_episodes, num_epochs, replay_size,
                         save_freq, batch_size, polyak, policy_update_delay, alpha, experiment_name, agent_name)
        self.action_dim = action_space.n  # this is different to the continuous version

        # create the standard actor-critic network
        self.actor_critic = core.DiscreteMLPActorCritic(self.state_space, self.action_space,
                                                        **dict(hidden_sizes=[hidden_dimension] * num_hidden_layers))

        # create the target network (which is just a deep copy of the real network)
        self.target_actor_critic = deepcopy(self.actor_critic)

        # Freeze the target network with respect to the optimisers (we only want to update the target using polyak averaging)
        for p in self.target_actor_critic.parameters():
            p.requires_grad = False

        # store a list of params in the Q-networks
        self.q_params = itertools.chain(self.actor_critic.q1.parameters(), self.actor_critic.q2.parameters())

        # make a replay buffer (for the discrete case, there is only one action attribute)
        self.replay_buffer = RandomisedSacBuffer(obs_dim=self.state_dim, act_dim=1, size=self.replay_size)

        # count and log the number of variables in each model for informative purposes
        self.var_counts = tuple(
            core.count_vars(module) for module in [self.actor_critic.pi, self.actor_critic.q1, self.actor_critic.q2])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % self.var_counts)

        # make optimisers for the actor and the critic
        self.pi_optimiser = Adam(self.actor_critic.pi.parameters(), lr=self.pi_lr)
        self.q_optimiser = Adam(self.q_params, lr=self.vf_lr)
        self.pi_lr_schedule = ExponentialLR(optimizer=self.pi_optimiser, gamma=0.95)
        self.q_lr_schedule = ExponentialLR(optimizer=self.q_optimiser, gamma=0.95)

        # set up model saving
        self.logger.setup_pytorch_saver(self.actor_critic)

    def compute_loss_q(self, data):
        states, actions, rewards, next_states, dones = data['obs'], data['act'], data['rew'], data['next_obs'], data[
            'done']

        q1 = self.actor_critic.q1(states)
        q2 = self.actor_critic.q2(states)

        q1 = q1.gather(1, actions.long())
        q2 = q2.gather(1, actions.long())

        # do Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            next_actions, next_action_probs, next_log_action_probs = self.actor_critic.pi(next_states)

            # target q-values use the next states and next actions
            target_q1 = self.target_actor_critic.q1(next_states)
            # target_q1 = target_q1.gather(1, next_actions.long())
            target_q1 = (next_action_probs * target_q1).sum(dim=1, keepdim=True)

            target_q2 = self.target_actor_critic.q2(next_states)
            # target_q2 = target_q2.gather(1, next_actions.long())
            target_q2 = (next_action_probs * target_q2).sum(dim=1, keepdim=True)

            target_q = torch.min(target_q1, target_q2)
            log_pi_next_actions = (next_action_probs * next_log_action_probs).sum(dim=1, keepdim=True)

            rewards = rewards.unsqueeze(-1)
            dones = dones.unsqueeze(-1)

            assert rewards.shape == target_q.shape == dones.shape == log_pi_next_actions.shape, \
                "Rewards, dones and q values do not have the same dimension"

            # determine backup (see SAC paper for more details on this derivation)
            backup = rewards + self.discount_rate * (1 - dones) * (target_q)

        # compute losses using MSE
        # F.mse_loss(current_Q1, target_Q)
        loss_q1 = F.mse_loss(q1, backup)
        loss_q2 = F.mse_loss(q2, backup)
        loss_q = loss_q1 + loss_q2

        # store useful logging info
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    def compute_loss_pi(self, data):
        states = data['obs']
        actions, action_probs, log_action_probs = self.actor_critic.pi(states)

        with torch.no_grad():
            q1_pi = self.actor_critic.q1(states)
            q2_pi = self.actor_critic.q2(states)

        # calculate expectations of entropy
        entropies = -torch.sum(action_probs * log_action_probs, dim=1, keepdim=True)

        # calculate expectations of Q (the q-value for each action, weighted by the probability of it occurring).
        q1_pi = torch.sum(q1_pi * action_probs, dim=1, keepdim=True)
        q2_pi = torch.sum(q2_pi * action_probs, dim=1, keepdim=True)

        q_pi = torch.min(q1_pi, q2_pi)

        # calculate entropy regularised policy loss
        loss_pi = (-self.alpha * entropies - q_pi).mean()

        pi_info = dict(LogPi=entropies.detach().numpy())

        return loss_pi, pi_info

    def get_action(self, state, deterministic=False):
        action = self.actor_critic.act(torch.as_tensor(state, dtype=torch.float32), deterministic=deterministic)
        return int(action)


class SacFactory:
    @staticmethod
    def create(state_space, action_space, subagent_name, experiment_name, discount_rate, discrete=True, num_epochs=100,
               pi_lr=1e-3, critic_lr=1e-3):
        if discrete:
            return DiscreteSacAgent(state_space=state_space,
                                    action_space=action_space,
                                    agent_name=subagent_name,
                                    experiment_name=experiment_name,
                                    num_epochs=num_epochs,
                                    critic_lr=critic_lr,
                                    pi_lr=pi_lr,
                                    discount_rate=discount_rate,
                                    start_steps=40000,
                                    max_ep_len=49 ** 2,
                                    steps_per_epoch=10000,
                                    policy_update_delay=1,
                                    seed=42,
                                    alpha=0.2,
                                    polyak=0.995,
                                    hidden_dimension=64)
        else:
            return ContinuousSacAgent(state_space=state_space,
                                      action_space=action_space,
                                      agent_name=subagent_name,
                                      experiment_name=experiment_name,
                                      num_epochs=num_epochs,
                                      critic_lr=critic_lr,
                                      pi_lr=pi_lr,
                                      discount_rate=discount_rate,
                                      start_steps=10000,
                                      max_ep_len=49 ** 2,
                                      steps_per_epoch=4000,
                                      seed=42,
                                      alpha=0.1    ,
                                      polyak=0.995,
                                      hidden_dimension=64)


if __name__ == '__main__':
    env_number = 7
    # train_env, map_name = read_map(env_number, random_start=False, terminate_at_any_goal=True, discrete=False)
    # test_env = deepcopy(train_env)
    # continuous_agent = ContinuousSacAgent(state_space=test_env.observation_space,
    #                                       action_space=test_env.action_space,
    #                                       agent_name='rg',
    #                                       experiment_name=f'continuous-pretrained-sac-{map_name}{env_number}',
    #                                       start_steps=20000,
    #                                       max_ep_len=49 ** 2,
    #                                       steps_per_epoch=4000,
    #                                       num_epochs=100,
    #                                       # discount_rate=0.975,
    #                                       seed=42,
    #                                       alpha=0.2,
    #                                       polyak=0.995,
    #                                       hidden_dimension=64,
    #                                       critic_lr=1e-3,
    #                                       pi_lr=1e-3)
    # continuous_agent.train(train_env=train_env, test_env=test_env)

    env, map_name = read_map(env_number, random_start=False, terminate_at_any_goal=False, discrete=False,
                             goal_name='fg4')

    state = env.reset()
    dt1 = []
    dt2 = []
    agent = torch.load(
        sacPathManager.get_path(agent_type='sac', map_name=f'{map_name}{env_number}', agent_name='fg4', discrete=False)
    )

    accum_reward = 0
    for t in range(1000):
        env.render()

        start = time.time()
        state = torch.tensor(state, dtype=torch.float32)
        action = agent.act(state, deterministic=True)
        end = time.time()

        dt1.append(end - start)

        start = time.time()
        state, reward, done, info = env.step(action)
        accum_reward += reward['fg1']
        print("accum reward:", accum_reward)
        end = time.time()
        dt2.append(end - start)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

    print(f"accumulated reward: {accum_reward}")
    print(sum(dt1) / len(dt1), sum(dt2) / len(dt2))
