# Generic imports
from copy import deepcopy
import itertools
import time
from abc import ABC, abstractmethod

# torch and numpy
import torch
from torch.optim import Adam
import numpy as np

# gym stuff
import gym
import gym_minigrid
from spinup.utils.minigrid_utils import MINI_GRID_SIMPLE_16, MINI_GRID_MEDIUM_16, MINI_GRID_SIMPLE_49, make_simple_env, \
    SEED
from python.display_utils import VideoViewer

# local stuff
import spinup.algos.pytorch.sac.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.buffers import RandomisedSacBuffer
from intention_recognition import IntentionRecognition


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
            alpha=0.2,
            experiment_name='agac-base-class',
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

        # video viewer to look at agent performance
        self.video_viewer = VideoViewer()

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

        # video viewer to look at agent performance
        self.video_viewer = VideoViewer()

    def update(self, data):
        # TODO: complete
        pass

    def select_action(self, state, deterministic=False):
        # TODO: complete
        pass

    def add_experience(self, state, action, reward, next_state, done):
        # TODO: complete
        pass

    def end_trajectory(self, test=False):
        # TODO: complete
        pass

    def learn(self, time_step):
        # TODO: complete
        pass

    def handle_end_of_epoch(self, time_step, test_env=None, test_env_key=None):
        # TODO: complete
        pass

    def log_stats(self, time_step, epoch_number=None):
        # TODO: complete
        pass

    def test(self, env, env_key):
        # TODO: complete
        pass

    def train(self, train_env, test_env=None, test_env_key=None):
        # TODO: complete
        pass

    @abstractmethod
    def compute_loss_q(self, data):
        raise NotImplementedError

    @abstractmethod
    def compute_loss_pi(self, data):
        raise NotImplementedError

    @abstractmethod
    def get_action(self, state, deterministic=False):
        raise NotImplementedError

    def save_state(self, epoch_number, train_env):
        # TODO: you should probably draw this out such that is only needs to know to save, not when to save
        if (epoch_number % self.save_freq == 0) or epoch_number == self.num_epochs:
            self.logger.save_state({'env': train_env}, None)


class DiscreteAGACAgent(AGACBaseAgent):
    def __init__(self, state_space, action_space, all_models, all_model_names, hidden_dimension=64, num_hidden_layers=2,
                 discount_rate=0.99, pi_lr=1e-3, critic_lr=1e-3, update_every=50, update_after=1000, max_ep_len=1000,
                 seed=42, steps_per_epoch=4000, start_steps=10000, num_test_episodes=10, num_epochs=100,
                 replay_size=int(1e6), save_freq=1, batch_size=100, polyak=0.995, alpha=0.2,
                 experiment_name='discrete-agac-agent', agent_name='rg') -> None:
        super().__init__(state_space, action_space, discount_rate, pi_lr, critic_lr, update_every, update_after,
                         max_ep_len, seed, steps_per_epoch, start_steps, num_test_episodes, num_epochs, replay_size,
                         save_freq, batch_size, polyak, alpha, experiment_name, agent_name)
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

        # create the intention recognition model to be used as the adversary. This is pretrained for now, but should
        # learn online in the future
        self.adversary = IntentionRecognition(state_space=state_space, action_space=action_space, all_models=all_models,
                                              all_model_names=all_model_names)

        # make a replay buffer (for the discrete case, there is only one action attribute)
        self.replay_buffer = RandomisedSacBuffer(obs_dim=self.state_dim, act_dim=1, size=self.replay_size)

        # count and log the number of variables in each model for informative purposes
        self.var_counts = tuple(
            core.count_vars(module) for module in [self.actor_critic.pi, self.actor_critic.q1, self.actor_critic.q2])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % self.var_counts)

        # make optimisers for the actor and the critic
        self.pi_optimiser = Adam(self.actor_critic.pi.parameters(), lr=self.pi_lr)
        self.q_optimiser = Adam(self.q_params, lr=self.vf_lr)

        # set up model saving
        self.logger.setup_pytorch_saver(self.actor_critic)

    def compute_loss_q(self, data):
        # TODO: complete
        pass

    def compute_loss_pi(self, data):
        # TODO: complete
        pass

    def get_action(self, state, deterministic=False):
        # TODO: complete
        pass

    def get_value_estimate(self, state, action):
        # TODO: complete
        pass

    def get_max_value_estimate(self, state):
        # TODO: complete
        pass
