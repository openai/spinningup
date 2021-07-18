import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


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


LOG_STD_MAX = 2
LOG_STD_MIN = -20


# ---------------------------------------- DISCRETE ACTOR CRITIC -----------------------------------#

class CategoricalActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs, deterministic=False, with_logprob=True):
        action_logits = self.logits_net(obs)
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = Categorical(probs=action_probs)
        actions = action_dist.sample().view(-1, 1)

        # avoid numerical instability
        z = (action_probs == 0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs

    def act(self, obs):
        action_logits = self.logits_net(obs)
        greed_actions = torch.argmax(action_logits, dim=1, keepdim=True)
        return greed_actions


class DiscreteMLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        q_vals = self.q(obs)
        return q_vals


class DiscreteMLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        # build policy and value functions
        self.pi = CategoricalActor(obs_dim, act_dim, hidden_sizes, activation)
        self.q1 = DiscreteMLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = DiscreteMLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _, _ = self.pi(obs, deterministic, False)
            return a.numpy()

    def select_action(self, obs, test=True):
        action = self.act(obs, test)
        return int(action)

    def get_value_estimate(self, obs, action):
        if torch.is_tensor(action):
            action = int(action.detach().numpy())

        with torch.no_grad():
            # gets the q values for all the actions
            q1_vals = self.q1(obs).detach().numpy()
            q2_vals = self.q2(obs).detach().numpy()

        # get the q_val for the specific actions
        q1_val = q1_vals[action]
        q2_val = q2_vals[action]

        # use the min q-vals as is done in the double q-learning technique for more accurate q-estimates
        return min(q1_val, q2_val)

    def get_max_value_estimate(self, obs):
        with torch.no_grad():
            # gets the q values for all the actions
            q1_vals = self.q1(obs).detach()
            q2_vals = self.q2(obs).detach()
            # use the min q-vals as is done in the double q-learning technique for more accurate q-estimates
            q_vals = torch.min(q1_vals, q2_vals)
            max_q_val = q_vals.max().numpy()

        return max_q_val


# ---------------------------------------- CONTINUOUS ACTOR CRITIC ----------------------------------- #

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()
