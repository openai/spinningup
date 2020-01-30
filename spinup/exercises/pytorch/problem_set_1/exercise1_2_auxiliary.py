import torch
import torch.nn as nn
import numpy as np

"""

Auxiliary code for Exercise 1.2. No part of the exercise requires you to 
look into or modify this file (and since it contains an mlp function, 
it has spoilers for the answer). Removed from the main file to avoid
cluttering it up.

In other words, nothing to see here, move along, these are not the
droids you're looking for, and all that...

"""

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class ExerciseActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh,
                 actor=None):
        super().__init__()
        obs_dim = observation_space.shape[0]
        self.pi = actor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi, _ = self.pi(obs)
            a = pi.sample()
            logp_a = pi.log_prob(a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]