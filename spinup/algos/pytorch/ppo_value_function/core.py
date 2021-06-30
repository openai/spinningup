import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
from gym_minigrid.minigrid import Actions, TRANSITION_TO_VEC

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        state_action = torch.hstack((obs, act.unsqueeze(-1)))
        q = self.q(state_action)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class MLPQActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        obs_dim = observation_space.shape[0]
        self.obs_dim = obs_dim
        self.act_dim = action_space.n

        # build policy
        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build q-learning based critic
        # FIXME: right now we just have one action value rather than a series of continuous values
        self.q1 = MLPQFunction(obs_dim, 1, hidden_sizes, activation)

    def step(self, obs, numpy=True):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            q1 = self.q1(obs, a)
        if numpy:
            return a.numpy(), q1, logp_a.numpy()
        else:
            return a, q1, logp_a.numpy()

    def act(self, obs, numpy=True):
        return self.step(obs, numpy)[0]

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        obs_dim = observation_space.shape[0]
        self.obs_dim = obs_dim

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.critic = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.critic(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

    def select_action(self, obs, test=False):
        return self.act(obs)

    def get_value_estimate(self, obs, action):
        # What we need to do in discrete action-space is:
        #   0.a) Since the obs is Tensor, we need to detach it from the graph and convert numpy
        #   1) Get the next state using the deterministic transition function
        #   1.a) convert state back to a Tensor to use in the critic MLP
        #   2) Estimate the value of the next state
        next_obs = self.get_transition(obs.detach().numpy(), action)
        next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32)
        value = self.critic(next_obs_tensor)
        return value

    def get_max_value_estimate(self, obs):
        # What we need to do in discrete action-space is:
        #   1) Select an optimal action given the state using the policy
        #   2) Get the next state using the deterministic transition function
        #   3) Estimate the value of the next state
        action = self.act(obs)
        value = self.get_value_estimate(obs, action)
        return value

    def get_transition(self, obs, action):
        """
        Because we are currently working in an environment which is deterministic, we can determine the next state
        using the action. Then we can estimate the value of the next state using the value function to estimate the
        Q-values. This is a hack and just a temporary fix for now.
        """
        transition_vec = TRANSITION_TO_VEC[int(action)]
        next_obs = np.zeros(3)
        next_obs[0] = self.clamp(obs[0] + transition_vec[0], 1, 15)
        next_obs[1] = self.clamp(obs[1] + transition_vec[1], 1, 15)
        next_obs[2] = obs[2]
        return next_obs

    @staticmethod
    def clamp(x, min_val, max_val):
        """
        Clamps a value between min_val and max_val
        """
        return max(min(x, max_val), min_val)
