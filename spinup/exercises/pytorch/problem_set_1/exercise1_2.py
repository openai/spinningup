import torch
import torch.nn as nn
import numpy as np
from spinup.exercises.pytorch.problem_set_1 import exercise1_1
from spinup.exercises.pytorch.problem_set_1 import exercise1_2_auxiliary

"""

Exercise 1.2: PPO Gaussian Policy

You will implement an MLP diagonal Gaussian policy for PPO by
writing an MLP-builder, and a few other key functions.

Log-likelihoods will be computed using your answer to Exercise 1.1,
so make sure to complete that exercise before beginning this one.

"""

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Build a multi-layer perceptron in PyTorch.

    Args:
        sizes: Tuple, list, or other iterable giving the number of units
            for each layer of the MLP. 

        activation: Activation function for all layers except last.

        output_activation: Activation function for last layer.

    Returns:
        A PyTorch module that can be called to give the output of the MLP.
        (Use an nn.Sequential module.)

    """
    #######################
    #                     #
    #   YOUR CODE HERE    #
    #                     #
    #######################
    pass

class DiagonalGaussianDistribution:

    def __init__(self, mu, log_std):
        self.mu = mu
        self.log_std = log_std

    def sample(self):
        """
        Returns:
            A PyTorch Tensor of samples from the diagonal Gaussian distribution with
            mean and log_std given by self.mu and self.log_std.
        """
        #######################
        #                     #
        #   YOUR CODE HERE    #
        #                     #
        #######################
        pass

    #================================(Given, ignore)==========================================#
    def log_prob(self, value):
        return exercise1_1.gaussian_likelihood(value, self.mu, self.log_std)

    def entropy(self):
        return 0.5 + 0.5 * np.log(2 * np.pi) + self.log_std.sum(axis=-1)
    #=========================================================================================#


class MLPGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        """
        Initialize an MLP Gaussian Actor by making a PyTorch module for computing the
        mean of the distribution given a batch of observations, and a log_std parameter.

        Make log_std a PyTorch Parameter with the same shape as the action vector, 
        independent of observations, initialized to [-0.5, -0.5, ..., -0.5].
        (Make sure it's trainable!)
        """
        #######################
        #                     #
        #   YOUR CODE HERE    #
        #                     #
        #######################
        # self.log_std = 
        # self.mu_net = 
        pass 

    #================================(Given, ignore)==========================================#
    def forward(self, obs, act=None):
        mu = self.mu_net(obs)
        pi = DiagonalGaussianDistribution(mu, self.log_std)
        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act)
        return pi, logp_a
    #=========================================================================================#



if __name__ == '__main__':
    """
    Run this file to verify your solution.
    """

    from spinup import ppo_pytorch as ppo
    from spinup.exercises.common import print_result
    from functools import partial
    import gym
    import os
    import pandas as pd
    import psutil
    import time

    logdir = "/tmp/experiments/%i"%int(time.time())

    ActorCritic = partial(exercise1_2_auxiliary.ExerciseActorCritic, actor=MLPGaussianActor)
    
    ppo(env_fn = lambda : gym.make('InvertedPendulum-v2'),
        actor_critic=ActorCritic,
        ac_kwargs=dict(hidden_sizes=(64,)),
        steps_per_epoch=4000, epochs=20, logger_kwargs=dict(output_dir=logdir))

    # Get scores from last five epochs to evaluate success.
    data = pd.read_table(os.path.join(logdir,'progress.txt'))
    last_scores = data['AverageEpRet'][-5:]

    # Your implementation is probably correct if the agent has a score >500,
    # or if it reaches the top possible score of 1000, in the last five epochs.
    correct = np.mean(last_scores) > 500 or np.max(last_scores)==1e3
    print_result(correct)