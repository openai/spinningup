from spinup.algos.tf1.ddpg.core import mlp, mlp_actor_critic
from spinup.utils.run_utils import ExperimentGrid
from spinup import ddpg_tf1 as ddpg
import numpy as np
import tensorflow as tf

"""

Exercise 2.2: Silent Bug in DDPG

In this exercise, you will run DDPG with a bugged actor critic. Your goal is
to determine whether or not there is any performance degredation, and if so,
figure out what's going wrong.

You do NOT need to write code for this exercise.

"""

"""
Bugged Actor-Critic
"""
def bugged_mlp_actor_critic(x, a, hidden_sizes=(400,300), activation=tf.nn.relu, 
                            output_activation=tf.tanh, action_space=None):
    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high[0]
    with tf.variable_scope('pi'):
        pi = act_limit * mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    with tf.variable_scope('q'):
        q = mlp(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None)
    with tf.variable_scope('q', reuse=True):
        q_pi = mlp(tf.concat([x,pi], axis=-1), list(hidden_sizes)+[1], activation, None)
    return pi, q, q_pi


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
        actor_critic = bugged_mlp_actor_critic if bugged else mlp_actor_critic
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