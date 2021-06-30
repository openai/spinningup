# Generic imports
import numpy as np
import time
import gym_minigrid
import gym

# torch imports
import torch
from torch.optim import Adam

# local imports
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup.utils.run_utils import setup_logger_kwargs
import spinup.algos.pytorch.ppo_value_function.core as core
from spinup.algos.pytorch.ppo_value_function.core import MLPActorCritic, count_vars


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class PPOAgent:
    def __init__(
            self,
            state_space,
            action_space,
            hidden_dimension=64,
            num_hidden_layers=2,
            discount_rate=0.99,
            lam=0.97,
            clip_ratio=0.2,
            pi_lr=3e-4,
            vf_lr=1e-3,
            train_pi_iters=80,
            train_v_iters=80,
            max_ep_len=1000,
            target_kl=0.01,
            seed=42,
            steps_per_epoch=4000,
            num_epochs=100,
            num_cpus=4,
            save_freq=10,
            experiment_name='ppo-class-test'
    ) -> None:
        # set up MPI stuff
        mpi_fork(num_cpus)
        setup_pytorch_for_mpi()

        # set up logging stuff
        logger_kwargs = setup_logger_kwargs(experiment_name, seed)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        # Randomise seed
        seed += 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Store hyperparameters
        self.discount_rate = discount_rate
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        self.max_ep_len = max_ep_len
        self.num_epochs = num_epochs

        # Store important environment details
        self.state_space = state_space
        self.action_space = action_space
        self.state_size = state_space.shape[0]
        self.action_size = action_space.n
        self.action_dim = action_space.shape

        # create actor-critic agent using core
        self.actor_critic = MLPActorCritic(state_space, action_space,
                                           **dict(hidden_sizes=[hidden_dimension] * num_hidden_layers))

        # create optimisers for the policy and the value function
        self.policy_optimiser = Adam(self.actor_critic.pi.parameters(), lr=pi_lr)
        self.value_optimiser = Adam(self.actor_critic.v.parameters(), lr=vf_lr)

        # set up model saving now that we have the model params sorted
        self.logger.setup_pytorch_saver(self.actor_critic)
        self.save_freq = save_freq

        # sync params across MPI processes
        sync_params(self.actor_critic)

        # Count and log variables
        self.var_counts = tuple(count_vars(module) for module in [self.actor_critic.pi, self.actor_critic.v])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % self.var_counts)

        # set up replay buffer
        self.steps_per_epoch = steps_per_epoch
        self.local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.replay_buffer = PPOBuffer(self.state_size, self.action_dim, self.local_steps_per_epoch, discount_rate,
                                       lam)

    def compute_policy_loss(self, states, actions, advantages, logp_olds):
        pi, logp = self.actor_critic.pi(states, actions)

        # calculate the ratio according to the PPO paper...
        # The ratio refers to pi(a_t|s_t)/pi_old(a_t|s_t)
        # Since we take a log transformation it becomes pi(a_t|s_t) - pi_old(a_t|s_t)
        # I think exponential reverses effect of log -> ie exp(log(a)) = a
        ratio = torch.exp(logp - logp_olds)

        # clamp the ratio to ensure that the advantage ratio stays within an upper bound... This means that the new
        # Policy does not benefit from going beyond a given bound from the old policy.
        # Clipping acts as a regulariser to ensure that the new policy does not diverge greatly from the old policy
        clip_advantage = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
        policy_loss = -(torch.min(ratio * advantages, clip_advantage)).mean()

        # extra stuff which is kind of useful
        approx_kl = (logp_olds - logp).mean().item()
        entropy = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)  # checks to see if the ratio was clipped
        clip_frac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()  # the fraction of observations that were clipped
        pi_info = dict(approx_kl=approx_kl, entropy=entropy, clipped_frac=clip_frac)  # just package all the additional info into a dict

        return policy_loss, pi_info

    def compute_value_loss(self, states, returns):
        # Use MSE as the value loss. We can maybe change this to something beterr...
        loss = ((self.actor_critic.v(states) - returns) ** 2).mean()
        return loss

    def update(self):
        # get and separate out data
        data = self.replay_buffer.get()
        states, actions, advantages, logp_olds, returns = data['obs'], data['act'], data['adv'], data['logp'], data[
            'ret']

        # compute losses from the data
        policy_loss_old, policy_info_old = self.compute_policy_loss(states, actions, advantages, logp_olds)
        policy_loss_old = policy_loss_old.item()  # extract it from the tensor
        value_loss_old = self.compute_value_loss(states, returns)

        # train the policy using multiple gradient ascent steps
        for i in range(self.train_pi_iters):
            # set the gradient for all the tensors to zero, such that we don't accumulate gradients from previous passes
            self.policy_optimiser.zero_grad()

            # recompute policy losses using the newly updated network (updated in the previous iteration)
            policy_loss, policy_info = self.compute_policy_loss(states, actions, advantages, logp_olds)

            # extract the KL-divergence between current logp and orig logp
            approx_kl = mpi_avg(policy_info['approx_kl'])

            # do early stopping of updates if the KL-divergence goes too far
            if approx_kl > 1.5 * self.target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break

            # backward propagate the policy loss
            policy_loss.backward()

            # average the gradients across processes
            mpi_avg_grads(self.actor_critic.pi)

            # do an update step using the optimiser
            self.policy_optimiser.step()

        # store the iteration at which the policy stopped at
        self.logger.store(PolicyStopIter=i)

        # train value function using multiple gradient ascent steps
        for i in range(self.train_v_iters):
            # zero grad the optimiser
            self.value_optimiser.zero_grad()

            # recompute value function loss
            value_loss = self.compute_value_loss(states, returns)

            # back propogate the loss
            value_loss.backward()

            # average out the gradients across processes
            mpi_avg_grads(self.actor_critic.v)

            # do the optimisation step
            self.value_optimiser.step()

        # log the changes from the update
        approx_kl, entropy, clipped_fraction = policy_info['approx_kl'], policy_info['entropy'], policy_info['clipped_frac']
        self.logger.store(PolicyLoss=policy_loss_old,
                          ValueLoss=value_loss_old,
                          KL=approx_kl,
                          Entropy=entropy,
                          ClippedFraction=clipped_fraction,
                          DeltaPolicyLoss=(policy_loss.item()-policy_loss_old),
                          DeltaValueLoss=(value_loss.item()-value_loss_old))

    def collect_experiences(self, env):
        state, episode_return, episode_length = env.reset(), 0, 0
        for time_step in range(self.local_steps_per_epoch):
            # choose the action and estimate the value using the actor critic
            action, value, logp = self.actor_critic.step(torch.as_tensor(state, dtype=torch.float32))

            # use the action to step through the environment
            next_state, reward, done, info = env.step(action)
            episode_return += reward
            episode_length += 1

            # save the details of the experiences in the buffer
            self.replay_buffer.store(state, action, reward, value, logp)

            # log the value estimated by the critic
            self.logger.store(VVals=value)

            # change the current state
            state = next_state

            # early termination stuff
            timeout = episode_length == self.max_ep_len
            terminal = done or timeout
            epoch_ended = time_step == self.local_steps_per_epoch - 1

            # FIXME: I think we need to customise this for the actual task tbh
            # we are finished with the epoch
            if terminal or epoch_ended:

                # finished due to self-imposed limit, not due to task being completed
                if epoch_ended and not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' % episode_length, flush=True)

                # Due to self-imposed time-step limit and not a result of the task being completed
                if timeout or epoch_ended:

                    # Use the critic to estimate the value of the state which results from being cut-off
                    _, last_value, _ = self.actor_critic.step(torch.as_tensor(state, dtype=torch.float32))

                # The episode is finished due to the agent being done
                else:
                    last_value = 0

                self.replay_buffer.finish_path(last_value)

                # If the episode finished, save the epsiode return/episode length in the logger
                if terminal:
                    self.logger.store(EpRet=episode_return, EpLen=episode_length)
                state, episode_return, episode_length = env.reset(), 0, 0

    def train(self, env):
        for epoch in range(self.num_epochs):
            start_time = time.time()
            self.collect_experiences(env)

            # save the model
            if (epoch % self.save_freq == 0) or (epoch == self.num_epochs-1):
                self.logger.save_state({'env': env}, None)

            self.update()

            # Log info about epoch
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('VVals', with_min_and_max=True)
            self.logger.log_tabular('TotalEnvInteracts', (epoch + 1) * self.steps_per_epoch)
            self.logger.log_tabular('PolicyLoss', average_only=True)
            self.logger.log_tabular('ValueLoss', average_only=True)
            self.logger.log_tabular('DeltaPolicyLoss', average_only=True)
            self.logger.log_tabular('DeltaValueLoss', average_only=True)
            self.logger.log_tabular('Entropy', average_only=True)
            self.logger.log_tabular('KL', average_only=True)
            self.logger.log_tabular('ClippedFraction', average_only=True)
            self.logger.log_tabular('PolicyStopIter', average_only=True)
            self.logger.log_tabular('Time', time.time() - start_time)
            self.logger.dump_tabular()

MINI_GRID_16 = 'MiniGrid-Deceptive-16x16-v0'
MINI_GRID_49 = 'MiniGrid-Deceptive-49x49-v0'
SEED = 1234
from gym_minigrid.wrappers import SimpleObsWrapper


def make_simple_env(env_key, seed):
    env = SimpleObsWrapper(gym.make(env_key))
    env.seed(seed)
    return env

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=MINI_GRID_16)
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='ppo-value-function-fg1')
    args = parser.parse_args()

    env = make_simple_env(args.env, SEED)
    agent = PPOAgent(
        state_space=env.observation_space,
        action_space=env.action_space,
        hidden_dimension=args.hid,
        num_hidden_layers=args.l,
        discount_rate=args.gamma,
        seed=args.seed,
        steps_per_epoch=args.steps,
        num_epochs=args.epochs,
        num_cpus=args.cpu
    )

    agent.train(env)