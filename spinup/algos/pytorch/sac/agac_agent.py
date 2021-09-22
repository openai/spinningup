# Generic imports
import itertools
from abc import ABC, abstractmethod
from copy import deepcopy
import time
from collections import defaultdict

# torch and numpy
import numpy as np
import torch
import torch.nn.functional as F

# gym stuff
from spinup.utils.minigrid_utils import make_simple_env
from spinup.utils.run_utils import setup_logger_kwargs
from torch.optim import Adam
import python.path_manager as constants

# local stuff
import spinup.algos.pytorch.sac.core as core
from intention_recognition import Adversary, OnlineSacAdversary, PretrainedSacAdversary
from python.display_utils import VideoViewer
from spinup.utils.buffers import RandomisedAGACBuffer
from spinup.utils.logx import EpochLogger
from torch.optim.lr_scheduler import ExponentialLR


class AGACBaseAgent(ABC):
    def __init__(
            self,
            state_space,
            action_space,
            discount_rate=0.99,
            pi_lr=1e-3,
            critic_lr=1e-3,
            update_every=50,
            update_after=1000,
            max_ep_len=1000,
            seed=42,
            steps_per_epoch=4000,
            start_steps=10000,
            num_test_episodes=10,
            num_epochs=100,
            replay_size=int(1e6),
            save_freq=1,
            batch_size=100,
            polyak=0.995,
            alpha=0.2,  # entropy coefficient
            beta=1.0,  # deceptiveness coefficient
            tau=1.0,  # temperature term to soften q-differences and therefore get a better spread of probabilities
            deception_type='entropy',
            experiment_name='agac-base-class',
            agent_name='rg'
    ) -> None:
        # set up logger
        self.experiment_name = experiment_name
        logger_kwargs = setup_logger_kwargs(experiment_name, seed)
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
        self.beta = beta  # This is the deceptiveness coefficient
        self.tau = tau  # temperature term to soften q-differences and therefore get a better spread of probabilities
        self.pi_lr = pi_lr
        self.vf_lr = critic_lr
        self.polyak = polyak  # for updating the target model

        # store episodes and epoch related details
        self.max_ep_len = max_ep_len
        self.num_epochs = num_epochs
        self.update_every = update_every
        self.update_after = update_after
        self.steps_per_epoch = steps_per_epoch
        self.start_steps = start_steps
        self.num_test_episodes = num_test_episodes

        # store details to track performance
        self.episode_reward = 0
        self.episode_length = 1
        self.epoch_number = 0
        self.accumulated_deceptiveness = 0
        # a count dict to track how much the agent has visited a state during training
        self.test_state_visitation_dict = defaultdict(int)
        self.train_state_visitation_dict = defaultdict(int)

        # a map to determine the level of deceptiveness for each state
        self.deceptiveness_dict = defaultdict(float)

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
        self.adversary = None
        self.replay_buffer = None
        self.pi_optimiser = None
        self.q_optimiser = None
        self.q_lr_schedule = None
        self.pi_lr_schedule = None

        # video viewer to look at agent performance
        self.video_viewer = VideoViewer()

        # not part of the actual agent
        self.bottom_right = False
        self.deception_type = deception_type

    def update(self, data):
        # do a gradient descent step for q function
        self.q_optimiser.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimiser.step()

        # record the losses
        self.logger.store(LossQ=loss_q.item(), **q_info)

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
        self.get_action(state, deterministic)

    def add_experience(self, state, action, reward, next_state, done):
        # if the reward gets passed in as a dict, then extract the reward using the agent name
        if type(reward) == dict:
            reward = reward[self.name]
        self.episode_reward += reward
        self.episode_length += 1

        # use the adversary model to generate probabilities and give the relevant deceptiveness measure
        self.adversary.update(state, action)
        if self.deception_type == 'entropy':
            deceptiveness = self.adversary.entropy_of_probabilities()
        else:
            rg_prob = self.adversary.probability_of_real_goal()
            deceptiveness = 1 - rg_prob
        if not done:
            self.deceptiveness_dict[str(next_state)] = deceptiveness

        self.accumulated_deceptiveness += deceptiveness

        self.replay_buffer.store(state, action, reward, next_state, done, deceptiveness)

    def end_trajectory(self, test=False):
        if test:
            self.logger.store(TestEpRet=self.episode_reward, TestEpLen=self.episode_length,
                              TestAccumulatedDeceptiveness=self.accumulated_deceptiveness)
        else:
            self.logger.store(EpRet=self.episode_reward, EpLen=self.episode_length,
                              EpAccumulatedDeceptiveness=self.accumulated_deceptiveness)
        self.episode_reward = 0
        self.episode_length = 1
        self.accumulated_deceptiveness = 0  # reset this for the next run

        # we also need to reset the adversary since the trajectory is over
        self.adversary.reset()

    def learn(self, time_step):
        # We only want to update the model after a certain number of experiences have been collected.
        # Then we want to update it periodically. This ensure that replay buffers are sufficiently uncorrelated
        if time_step >= self.update_after and time_step % self.update_every == 0:
            # We still want to keep a 1-1 ratio between number of actions and number of updates
            for j in range(self.update_every):
                data = self.replay_buffer.sample_batch(self.batch_size)
                self.update(data)

    def handle_end_of_epoch(self, time_step, train_env, test_env=None, test_env_key=None):
        # Handle the end of epoch: (1) Save the model. (2) test the agent. (3) log the results
        if (time_step + 1) % self.steps_per_epoch == 0:
            self.epoch_number = (time_step + 1) // self.steps_per_epoch

            # save the model at each save frequency or at the end
            if (self.epoch_number % self.save_freq == 0) or self.epoch_number == self.num_epochs:
                self.save_state(epoch_number=self.epoch_number, train_env=train_env)

            # end the previous trajectory to reset stuff like accumulated deceptiveness and adversary
            self.end_trajectory()

            self.test(test_env, test_env_key)

            self.log_stats(time_step)

            # adjust the learning rate at each epoch
            self.q_lr_schedule.step()
            self.pi_lr_schedule.step()
            self.episode_reward = 0
            self.episode_length = 1

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
        self.logger.log_tabular('LogPi', with_min_and_max=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('EpAccumulatedDeceptiveness', with_min_and_max=True)
        self.logger.log_tabular('TestAccumulatedDeceptiveness', with_min_and_max=True)
        self.logger.dump_tabular()

    def test(self, env, env_key):
        if env is None and env_key is not None:
            env = make_simple_env(env_key, constants.Random.SEED, random_start=False)
        else:
            env = deepcopy(env)
        for j in range(self.num_test_episodes):
            state = env.reset()
            self.test_state_visitation_dict[str(state)] += 1
            done = False
            while not done:
                # act deterministically because this is a test
                action = self.get_action(state, deterministic=True)

                # get your real goal probabilities since we have our state-action pair
                self.adversary.update(state=state, action=action)
                if self.deception_type == 'entropy':
                    deceptiveness = self.adversary.entropy_of_probabilities()
                else:
                    rg_prob = self.adversary.probability_of_real_goal()
                    deceptiveness = 1 - rg_prob

                # step through the environment
                next_state, reward, done, _ = env.step(action)

                self.test_state_visitation_dict[str(state)] += 1
                # extract the reward
                if type(reward) == dict:
                    reward = reward[self.name]

                # update the state
                self.episode_reward += reward
                self.episode_length += 1
                # track the deceptivness. We use 1 - rg_prob since low prob is good
                self.accumulated_deceptiveness += deceptiveness
                state = next_state

            self.test_state_visitation_dict[str(state)] += 1
            self.end_trajectory(test=True)  # this handles reseeting the adversary

    def train(self, train_env, test_env=None, test_env_key=None):
        total_steps = self.steps_per_epoch * self.num_epochs
        start_time = time.time()
        state = train_env.reset()

        # collect experiences and update every epoch
        for t in range(total_steps):
            if t > self.start_steps:
                action = self.get_action(state)
            else:
                action = train_env.action_space.sample()
            # action = self.get_action(state)
            next_state, reward, done, _ = train_env.step(action)
            self.train_state_visitation_dict[str(state)] += 1
            self.add_experience(state, action, reward, next_state, done)  # this also handles getting the adversary prob
            state = next_state

            if done or self.episode_length == self.max_ep_len:
                self.end_trajectory()  # this handles resetting the adversary
                self.train_state_visitation_dict[str(state)] += 1
                state = train_env.reset()

            # update the model
            self.learn(time_step=t)

            # handle the end of the epoch
            self.handle_end_of_epoch(time_step=t, train_env=train_env, test_env=test_env, test_env_key=test_env_key)

    def save_state(self, epoch_number, train_env):
        self.logger.save_state({'env': train_env}, None)
        self.logger.save_state_visitation_dict(self.test_state_visitation_dict,
                                               'test_state_visitation_dict.json')
        self.logger.save_state_visitation_dict(self.train_state_visitation_dict,
                                               'train_state_visitation_dict.json')
        self.logger.save_deceptiveness_dict(self.deceptiveness_dict,
                                            'deceptiveness_dict.json')

    @abstractmethod
    def compute_loss_q(self, data):
        raise NotImplementedError

    @abstractmethod
    def compute_loss_pi(self, data):
        raise NotImplementedError

    @abstractmethod
    def compute_loss_v(self, data):
        raise NotImplementedError

    @abstractmethod
    def get_action(self, state, deterministic=False):
        raise NotImplementedError

    @abstractmethod
    def get_value_estimate(self, state, action):
        raise NotImplementedError

    @abstractmethod
    def get_max_value_estimate(self, state):
        raise NotImplementedError


class DiscreteAGACAgent(AGACBaseAgent):
    def __init__(self, state_space, action_space, all_models, all_model_names, hidden_dimension=64, num_hidden_layers=2,
                 discount_rate=0.99, pi_lr=1e-3, critic_lr=1e-3, update_every=50, update_after=1000, max_ep_len=1000,
                 seed=42, steps_per_epoch=4000, start_steps=10000, num_test_episodes=10, num_epochs=100,
                 replay_size=int(1e6), save_freq=1.0, batch_size=100, polyak=0.995, alpha=0.2, beta=1.0,
                 lr_decay_rate=0.95,
                 tau=1.0, q_difference_queue_length=5000, deception_type='entropy',
                 experiment_name='discrete-agac-agent', agent_name='rg') -> None:
        super().__init__(state_space=state_space, action_space=action_space, discount_rate=discount_rate, pi_lr=pi_lr,
                         critic_lr=critic_lr, update_every=update_every, update_after=update_after,
                         max_ep_len=max_ep_len, seed=seed, steps_per_epoch=steps_per_epoch, start_steps=start_steps,
                         num_test_episodes=num_test_episodes, num_epochs=num_epochs, replay_size=replay_size,
                         save_freq=save_freq, batch_size=batch_size, polyak=polyak, alpha=alpha, beta=beta, tau=tau,
                         deception_type=deception_type, experiment_name=experiment_name, agent_name=agent_name)
        self.action_dim = action_space.n  # this is different to the continuous version
        self.lr_decay_rate = lr_decay_rate

        # define a max q difference queue length to controls the size of the trajectory to consider when determining
        # probabilities.
        self.q_difference_queue_length = q_difference_queue_length

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

        # create the intention recognition model to be used as the adversary. This is pretrained for now, but should
        # learn online in the future
        self.adversary = Adversary(state_space=state_space, action_space=action_space, all_models=all_models,
                                   all_model_names=all_model_names)

        # make a replay buffer (for the discrete case, there is only one action attribute)
        self.replay_buffer = RandomisedAGACBuffer(obs_dim=self.state_dim, act_dim=1, size=self.replay_size)

        # count and log the number of variables in each model for informative purposes
        self.var_counts = tuple(
            core.count_vars(module) for module in [self.actor_critic.pi, self.actor_critic.q1, self.actor_critic.q2])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % self.var_counts)

        # make optimisers for the actor and the critic
        self.pi_optimiser = Adam(self.actor_critic.pi.parameters(), lr=self.pi_lr)
        self.q_optimiser = Adam(self.q_params, lr=self.vf_lr)
        self.pi_lr_schedule = ExponentialLR(optimizer=self.pi_optimiser, gamma=self.lr_decay_rate)
        self.q_lr_schedule = ExponentialLR(optimizer=self.q_optimiser, gamma=self.lr_decay_rate)

        # set up model saving
        self.logger.setup_pytorch_saver(self.actor_critic)

    def compute_loss_q(self, data):
        states, actions, rewards, next_states, dones, deceptiveness = data['obs'], data['act'], data['rew'], data[
            'next_obs'], data['done'], data['deceptiveness']

        q1 = self.actor_critic.q1(states)
        q1 = q1.gather(1, actions.long())

        q2 = self.actor_critic.q2(states)
        q2 = q2.gather(1, actions.long())

        # do Bellman backup for Q functions
        with torch.no_grad():
            # use the target network to get the actions given the next state
            next_actions, next_action_probs, next_log_action_probs = self.target_actor_critic.pi(next_states)

            # target q-values use the next states and next actions
            target_q1 = self.target_actor_critic.q1(next_states)
            target_q1 = target_q1.gather(1, next_actions.long())

            target_q2 = self.target_actor_critic.q2(next_states)
            target_q2 = target_q2.gather(1, next_actions.long())

            target_q = torch.min(target_q1, target_q2)

            rewards = rewards.unsqueeze(-1)
            dones = dones.unsqueeze(-1)
            deceptiveness_scores = deceptiveness.unsqueeze(dim=-1)
            assert rewards.shape == target_q.shape == dones.shape == deceptiveness_scores.shape, \
                "Rewards, dones, deceptiveness-scores and q-values do not have the same dimension"

            backup = rewards + self.beta * deceptiveness_scores + self.discount_rate * (1 - dones) * target_q

        # compute losses using MSE
        loss_q1 = F.mse_loss(q1, backup)
        loss_q2 = F.mse_loss(q2, backup)
        loss_q = loss_q1 + loss_q2

        # store useful logging info
        q_info = dict(Q1Vals=q1.detach().numpy())

        return loss_q, q_info

    def compute_loss_pi(self, data):
        states, deceptiveness = data['obs'], data['deceptiveness']
        actions, action_probs, log_action_probs = self.actor_critic.pi(states)

        with torch.no_grad():
            q1 = self.actor_critic.q1(states)
            q2 = self.actor_critic.q2(states)

        # calculate expectations of entropy
        entropies = -torch.sum(action_probs * log_action_probs, dim=1, keepdim=True)

        # TODO: give a real goal probability for every action and then calculate a weighted real goal probability which
        #  is weighted by the action distribution... This is similar to the idea of the weighted q-values that you use.
        #  But for right now KISS
        # 1 - rg_prob to give a higher score for a more deceptive action
        deceptiveness_scores = deceptiveness.unsqueeze(dim=-1)

        # calculate expectations of Q (the q-value for each action, weighted by the probability of it occurring).
        q1 = torch.sum(q1 * action_probs, dim=1, keepdim=True)
        q2 = torch.sum(q2 * action_probs, dim=1, keepdim=True)
        q = torch.min(q1, q2)

        # Policy objective is to maximise (Q + beta * deceptiveness + alpha * entropy)
        loss_pi = (-q - self.alpha * entropies).mean()

        pi_info = dict(LogPi=entropies.detach().numpy(), Deceptiveness=deceptiveness_scores.detach().numpy())

        return loss_pi, pi_info

    def get_action(self, state, deterministic=False):
        action = self.actor_critic.act(torch.as_tensor(state, dtype=torch.float32), deterministic=deterministic)
        return int(action)

        # action = 'DONE'
        # if not self.bottom_right:
        #     action = self.go_to_bottom_right(state)
        # if not action == 'DONE':
        #     return action
        # return self.snake(state)

    def snake(self, state):
        x, y, _ = state
        if y % 2 == 0:
            if x == 1:
                return 2
            else:
                return 0
        else:
            if x == 47:
                return 2
            else:
                return 1

    def go_to_bottom_right(self, state):
        x, y, _ = state
        if x == 47 and y == 47:
            self.bottom_right = True
            return 'DONE'
        if x < 47 and y < 47:
            return 7
        if x < 47:
            return 1
        if y < 47:
            return 3

    def get_value_estimate(self, state, action):
        return self.actor_critic.get_value_estimate(state, action)

    def get_max_value_estimate(self, state):
        return self.actor_critic.get_max_value_estimate(state)

    def single_environment_run(self, env_key: str, video_viewer: VideoViewer = None):
        """
        This should be a single environment run to be used after training for analysis of the results
        :return:
        """
        env = make_simple_env(env_key, constants.Random.SEED, random_start=False)
        if video_viewer is not None:
            env = video_viewer.wrap_env(env=env, agent_name=self.name,
                                        folder=f'{constants.Saving.VIDEO_ROOT}/AGACAgent/{env_key}')
        state = env.reset()
        done = False

        while not done:
            state = torch.as_tensor(state, dtype=torch.float32)
            action = self.get_action(state)
            next_state, reward, done, info = env.step(action)
            # no need to update the agent
            state = next_state

    def compute_loss_v(self, data):
        pass


class DiscretePretrainedAGACAgent(DiscreteAGACAgent):

    def __init__(self,
                 state_space,
                 action_space,
                 all_models,
                 all_model_names,
                 hidden_dimension=64,
                 num_hidden_layers=2,
                 discount_rate=0.99,
                 pi_lr=1e-3,
                 critic_lr=1e-3,
                 update_every=50,
                 update_after=1000,
                 max_ep_len=1000,
                 seed=42,
                 steps_per_epoch=4000,
                 start_steps=10000,
                 num_test_episodes=10,
                 num_epochs=100,
                 replay_size=int(1e6),
                 save_freq=1,
                 batch_size=100,
                 polyak=0.995,
                 alpha=0.2,
                 beta=1,
                 lr_decay_rate=0.95,
                 tau=1.0,
                 q_difference_queue_length=5000,
                 deception_type='entropy',
                 experiment_name='discrete-agac-agent',
                 agent_name='rg') -> None:
        super().__init__(state_space=state_space,
                         action_space=action_space,
                         all_models=all_models,
                         all_model_names=all_model_names,
                         hidden_dimension=hidden_dimension,
                         num_hidden_layers=num_hidden_layers,
                         discount_rate=discount_rate,
                         pi_lr=pi_lr,
                         critic_lr=critic_lr,
                         update_every=update_every,
                         update_after=update_after,
                         max_ep_len=max_ep_len,
                         seed=seed,
                         steps_per_epoch=steps_per_epoch,
                         start_steps=start_steps,
                         num_test_episodes=num_test_episodes,
                         num_epochs=num_epochs,
                         replay_size=replay_size,
                         save_freq=save_freq,
                         batch_size=batch_size,
                         polyak=polyak,
                         alpha=alpha,
                         beta=beta,
                         lr_decay_rate=lr_decay_rate,
                         tau=tau,
                         q_difference_queue_length=q_difference_queue_length,
                         deception_type=deception_type,
                         experiment_name=experiment_name,
                         agent_name=agent_name)
        self.adversary = PretrainedSacAdversary(state_space=state_space,
                                                action_space=action_space,
                                                all_models=all_models,
                                                all_model_names=all_model_names,
                                                tau=tau,
                                                q_difference_queue_length=q_difference_queue_length)


class DiscreteOnlineAGACAgent(DiscreteAGACAgent):
    def __init__(self, 
                 state_space, 
                 action_space, 
                 all_models, 
                 all_model_names, 
                 hidden_dimension=64, 
                 num_hidden_layers=2,
                 discount_rate=0.99, 
                 pi_lr=1e-3, 
                 critic_lr=1e-3, 
                 update_every=50, 
                 update_after=1000, 
                 max_ep_len=1000,
                 seed=42, 
                 steps_per_epoch=4000, 
                 start_steps=10000, 
                 num_test_episodes=10, 
                 num_epochs=100,
                 replay_size=int(1e6), 
                 save_freq=1, 
                 batch_size=100, 
                 polyak=0.995, 
                 alpha=0.2, 
                 beta=1, 
                 lr_decay_rate=0.95,
                 tau=1.0, 
                 q_difference_queue_length=5000, 
                 deception_type='entropy', 
                 experiment_name='discrete-agac-agent', 
                 agent_name='rg') -> None:
        super().__init__(state_space=state_space, 
                         action_space=action_space, 
                         all_models=all_models, 
                         all_model_names=all_model_names, 
                         hidden_dimension=hidden_dimension, 
                         num_hidden_layers=num_hidden_layers,
                         discount_rate=discount_rate, 
                         pi_lr=pi_lr, 
                         critic_lr=critic_lr, 
                         update_every=update_every, 
                         update_after=update_after, 
                         max_ep_len=max_ep_len, 
                         seed=seed, 
                         steps_per_epoch=steps_per_epoch,
                         start_steps=start_steps, 
                         num_test_episodes=num_test_episodes, 
                         num_epochs=num_epochs, 
                         replay_size=replay_size, 
                         save_freq=save_freq, 
                         batch_size=batch_size, 
                         polyak=polyak, 
                         alpha=alpha,
                         beta=beta, 
                         lr_decay_rate=lr_decay_rate, 
                         tau=tau,
                         q_difference_queue_length=q_difference_queue_length,
                         deception_type=deception_type, 
                         experiment_name=experiment_name, 
                         agent_name=agent_name)

        self.adversary = OnlineSacAdversary(state_space=state_space,
                                            action_space=action_space,
                                            all_models=all_models,
                                            all_model_names=all_model_names,
                                            tau=tau)

    def learn(self, time_step):
        super().learn(time_step)
        self.adversary.learn(time_step=time_step)  # make sure that the adversary learns too

    def add_experience(self, state, action, reward, next_state, done):
        self.adversary.add_experience(state, action, next_state, reward,
                                      done)  # pass the experience on to the adversary
        super().add_experience(state, action, reward, next_state, done)

    def save_state(self, epoch_number, train_env):
        super().save_state(epoch_number=epoch_number, train_env=train_env)
        self.adversary.save_state(epoch_number=epoch_number,
                                  a_train_env=train_env)  # allow the adversary to save its own state
