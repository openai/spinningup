import tensorflow as tf
import numpy as np
from spinup.exercises.tf1.problem_set_1 import exercise1_1

"""

Exercise 1.2: PPO Gaussian Policy

Implement an MLP diagonal Gaussian policy for PPO. 

Log-likelihoods will be computed using your answer to Exercise 1.1,
so make sure to complete that exercise before beginning this one.

"""

EPS = 1e-8

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    """
    Builds a multi-layer perceptron in Tensorflow.

    Args:
        x: Input tensor.

        hidden_sizes: Tuple, list, or other iterable giving the number of units
            for each hidden layer of the MLP.

        activation: Activation function for all layers except last.

        output_activation: Activation function for last layer.

    Returns:
        A TF symbol for the output of an MLP that takes x as an input.

    """
    #######################
    #                     #
    #   YOUR CODE HERE    #
    #                     #
    #######################
    pass

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    """
    Builds symbols to sample actions and compute log-probs of actions.

    Special instructions: Make log_std a tf variable with the same shape as
    the action vector, independent of x, initialized to [-0.5, -0.5, ..., -0.5].

    Args:
        x: Input tensor of states. Shape [batch, obs_dim].

        a: Input tensor of actions. Shape [batch, act_dim].

        hidden_sizes: Sizes of hidden layers for action network MLP.

        activation: Activation function for all layers except last.

        output_activation: Activation function for last layer (action layer).

        action_space: A gym.spaces object describing the action space of the
            environment this agent will interact with.

    Returns:
        pi: A symbol for sampling stochastic actions from a Gaussian 
            distribution.

        logp: A symbol for computing log-likelihoods of actions from a Gaussian 
            distribution.

        logp_pi: A symbol for computing log-likelihoods of actions in pi from a 
            Gaussian distribution.

    """
    #######################
    #                     #
    #   YOUR CODE HERE    #
    #                     #
    #######################
    # mu = 
    # log_std = 
    # pi = 

    logp = exercise1_1.gaussian_likelihood(a, mu, log_std)
    logp_pi = exercise1_1.gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi


if __name__ == '__main__':
    """
    Run this file to verify your solution.
    """

    from spinup import ppo_tf1 as ppo
    from spinup.exercises.common import print_result
    import gym
    import os
    import pandas as pd
    import psutil
    import time

    logdir = "/tmp/experiments/%i"%int(time.time())
    ppo(env_fn = lambda : gym.make('InvertedPendulum-v2'),
        ac_kwargs=dict(policy=mlp_gaussian_policy, hidden_sizes=(64,)),
        steps_per_epoch=4000, epochs=20, logger_kwargs=dict(output_dir=logdir))

    # Get scores from last five epochs to evaluate success.
    data = pd.read_table(os.path.join(logdir,'progress.txt'))
    last_scores = data['AverageEpRet'][-5:]

    # Your implementation is probably correct if the agent has a score >500,
    # or if it reaches the top possible score of 1000, in the last five epochs.
    correct = np.mean(last_scores) > 500 or np.max(last_scores)==1e3
    print_result(correct)