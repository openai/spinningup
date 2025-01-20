import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

'''
Below is defined the network which is essentially the policy for the agent, it is a neural network which takes into 
account the observation dimensions.
'''
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

# The env-name is crucial and tells the code the pre-defined environment that will be used for testing the algorithm
def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    # This right here is demarcating the continuous space represented by the box and discrete ones represented by Discrete
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."
    # the observation space dimension is the input and the action space is the output. For language this can be a discrete vocabulary
    # this leads to the intuition of designing rewards based on tokens rather than the whole sentences which by itself is also an action space.
    #n_acts defines the action space of the environment left or right
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        #This is a function which returns the action by sampling from the logits by modeling it as a Categorical distribution
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer, this will be used for training the policy
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths
        ep_rews = []                # list for rewards accrued throughout ep
        # reset episode-specific variables
        obs, info = env.reset()     # first obs comes from starting distribution
        done = False                # signal from environment that episode is over
        
        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy, this thing is called an episode
        while True:
            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy()) # Copying the observations array of the tuple

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            #Every step of the action gives the rewards from the environment
            obs, reward, done, _, _ = env.step(act)
            # save action, reward
            batch_acts.append(act)
            ep_rews.append(reward)
            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                print("Episode Length", ep_len)
                # the weight for each logprob(a|s) is R(tau)
                batch_weights+= [ep_ret] * ep_len

                # reset episode-specific variables
                obs, info = env.reset()
                done, ep_rews = False, []
                # won't render again this epoch
                finished_rendering_this_epoch = True
                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break
        # take a single policy gradient update step
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)