from spinup.algos.pytorch.ddpg.core import mlp, MLPActorCritic
from spinup.utils.run_utils import ExperimentGrid
from spinup import ddpg_pytorch as ddpg
import numpy as np
import torch
import torch.nn as nn

"""

Exercise 2.2: Silent Bug in DDPG (PyTorch Version)

In this exercise, you will run DDPG with a bugged actor critic. Your goal is
to determine whether or not there is any performance degredation, and if so,
figure out what's going wrong.

You do NOT need to write code for this exercise.

"""

"""
Bugged Actor-Critic
"""

class BuggedMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

class BuggedMLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1))

class BuggedMLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = BuggedMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = BuggedMLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--h', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--num_runs', '-n', type=int, default=3)
    parser.add_argument('--steps_per_epoch', '-s', type=int, default=5000)
    parser.add_argument('--total_steps', '-t', type=int, default=int(5e4))
    args = parser.parse_args()

    def ddpg_with_actor_critic(bugged, **kwargs):
        from spinup.exercises.pytorch.problem_set_2.exercise2_2 import BuggedMLPActorCritic
        actor_critic = BuggedMLPActorCritic if bugged else MLPActorCritic
        return ddpg(actor_critic=actor_critic, 
                    ac_kwargs=dict(hidden_sizes=[args.h]*args.l),
                    start_steps=5000,
                    max_ep_len=150,
                    batch_size=64,
                    polyak=0.95,
                    **kwargs)

    eg = ExperimentGrid(name='ex2-2_ddpg')
    eg.add('replay_size', int(args.total_steps))
    eg.add('env_name', args.env, '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', int(args.total_steps / args.steps_per_epoch))
    eg.add('steps_per_epoch', args.steps_per_epoch)
    eg.add('bugged', [False, True])
    eg.run(ddpg_with_actor_critic, datestamp=True)