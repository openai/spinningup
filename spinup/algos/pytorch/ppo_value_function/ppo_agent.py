# Generic imports
import time
import copy
import itertools

# Gym stuff
import gym
from gym_minigrid.wrappers import SimpleObsWrapper

# torch imports
from torch.optim import Adam

# local imports
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, num_procs
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.pytorch.ppo_value_function.core import MLPActorCritic, MLPQActorCritic, count_vars

from spinup.utils.buffers import *

# Define constant
MINI_GRID_16 = 'MiniGrid-Deceptive-16x16-v0'
MINI_GRID_49 = 'MiniGrid-Deceptive-49x49-v0'
SEED = 1234


# Util function to instantiate environment
def make_simple_env(env_key, seed):
    env = SimpleObsWrapper(gym.make(env_key))
    env.seed(seed)
    return env


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
            critic_lr=1e-3,
            train_pi_iters=80,
            train_critic_iters=80,
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
        self.vf_lr = critic_lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_critic_iters
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
        self.critic_optimiser = Adam(self.actor_critic.critic.parameters(), lr=critic_lr)

        # set up model saving now that we have the model params sorted
        self.logger.setup_pytorch_saver(self.actor_critic)
        self.save_freq = save_freq

        # sync params across MPI processes
        sync_params(self.actor_critic)

        # Count and log variables
        self.var_counts = tuple(count_vars(module) for module in [self.actor_critic.pi, self.actor_critic.critic])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % self.var_counts)

        # set up replay buffer
        self.steps_per_epoch = steps_per_epoch
        self.local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.replay_buffer = PPOBuffer(self.state_size, self.action_dim, self.local_steps_per_epoch, discount_rate,
                                       lam)

    def compute_policy_loss(self, data):
        states, actions, advantages, logp_olds = data['obs'], data['act'], data['adv'], data['logp']
        pi, logp = self.actor_critic.pi(states, actions)

        # calculate the ratio according to the PPO paper...
        # The ratio refers to pi(a_t|s_t)/pi_old(a_t|s_t)
        # Since we take a log transformation it becomes pi(a_t|s_t) - pi_old(a_t|s_t)
        # I think exponential reverses effect of log -> ie exp(log(a)) = a
        ratio = torch.exp(logp - logp_olds)

        # clamp the ratio to ensure that the advantage ratio stays within an upper bound... This means that the new
        # Policy does not benefit from going beyond a given bound from the old policy.
        # Clipping acts as a regulariser to ensure that the new policy does not diverge greatly from the old policy
        clip_advantage = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -(torch.min(ratio * advantages, clip_advantage)).mean()

        # extra stuff which is kind of useful
        approx_kl = (logp_olds - logp).mean().item()
        entropy = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(
            1 - self.clip_ratio)  # checks to see if the ratio was clipped
        clip_frac = torch.as_tensor(clipped,
                                    dtype=torch.float32).mean().item()  # the fraction of observations that were clipped
        pi_info = dict(approx_kl=approx_kl, entropy=entropy,
                       clipped_frac=clip_frac)  # just package all the additional info into a dict

        return policy_loss, pi_info

    def compute_critic_loss(self, data):
        states, returns = data['obs'], data['ret']
        # Use MSE as the value loss. We can maybe change this to something beterr...
        loss = ((self.actor_critic.critic(states) - returns) ** 2).mean()
        return loss, None

    def update(self):
        # get and separate out data
        data = self.replay_buffer.get()

        # compute losses from the data
        policy_loss_old, policy_info_old = self.compute_policy_loss(data)
        policy_loss_old = policy_loss_old.item()  # extract it from the tensor
        value_loss_old, value_info_old = self.compute_critic_loss(data)

        policy_loss, policy_info = self.update_policy(data)

        value_loss, value_info = self.update_critic(data)

        # log the changes from the update
        approx_kl, entropy, clipped_fraction = policy_info['approx_kl'], policy_info['entropy'], policy_info[
            'clipped_frac']
        self.logger.store(PolicyLoss=policy_loss_old,
                          ValueLoss=value_loss_old,
                          KL=approx_kl,
                          Entropy=entropy,
                          ClippedFraction=clipped_fraction,
                          DeltaPolicyLoss=(policy_loss.item() - policy_loss_old),
                          DeltaValueLoss=(value_loss.item() - value_loss_old))

    def update_policy(self, data):
        # train the policy using multiple gradient ascent steps
        for i in range(self.train_pi_iters):
            # set the gradient for all the tensors to zero, such that we don't accumulate gradients from previous passes
            self.policy_optimiser.zero_grad()

            # recompute policy losses using the newly updated network (updated in the previous iteration)
            policy_loss, policy_info = self.compute_policy_loss(data)

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

        return policy_loss, policy_info

    def update_critic(self, data):
        # train value function using multiple gradient ascent steps
        for i in range(self.train_v_iters):
            # zero grad the optimiser
            self.critic_optimiser.zero_grad()

            # recompute value function loss
            value_loss, value_info = self.compute_critic_loss(data)

            # back propogate the loss
            value_loss.backward()

            # average out the gradients across processes
            mpi_avg_grads(self.actor_critic.critic)

            # do the optimisation step
            self.critic_optimiser.step()

        return value_loss, value_info

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
            self.replay_buffer.store(state, action, next_state, reward, done, value, logp)

            # log the value estimated by the critic
            self.logger.store(VVals=value)

            # change the current state
            state = next_state

            # early termination stuff
            timeout = episode_length == self.max_ep_len
            terminal = done or timeout
            epoch_ended = time_step == self.local_steps_per_epoch - 1

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
            if (epoch % self.save_freq == 0) or (epoch == self.num_epochs - 1):
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


class PPOQAgent(PPOAgent):

    def __init__(self, state_space, action_space, hidden_dimension=64, num_hidden_layers=2, discount_rate=0.99,
                 polyak=0.995, lam=0.97, clip_ratio=0.2, pi_lr=3e-4, critic_lr=1e-3, train_pi_iters=80,
                 train_critic_iters=80, target_update_freq=2,
                 max_ep_len=1000, target_kl=0.01, seed=42, steps_per_epoch=4000, num_epochs=100, num_cpus=4,
                 save_freq=10, experiment_name='ppoq-class-test') -> None:
        super().__init__(state_space, action_space, hidden_dimension, num_hidden_layers, discount_rate, lam, clip_ratio,
                         pi_lr, critic_lr, train_pi_iters, train_critic_iters, max_ep_len, target_kl, seed,
                         steps_per_epoch,
                         num_epochs, num_cpus, save_freq, experiment_name)

        # interpolation factor for polyak averaging for target networks. Target networks are update toward main networks
        # accoring to:
        #   theta_targ <- rho * theta_targ + (1 - rho) theta_main
        # here rho is the polyak factor.
        # In other words, it is just a weighted average update where rho is the weight on the target network
        self.polyak = polyak

        # create actor-critic agent using core
        self.actor_critic = MLPQActorCritic(state_space, action_space,
                                            **dict(hidden_sizes=[hidden_dimension] * num_hidden_layers))

        # create a target network for Q-learning
        self.actor_critic_target = copy.deepcopy(self.actor_critic)
        self.target_update_freq = target_update_freq

        # freeze the target networks with respect to optimisers. We only want to update them via polyak averaging
        for p in self.actor_critic_target.parameters():
            p.requires_grad = False

        # save a list of both Q-network parameters for convenience
        self.q_params = itertools.chain(self.actor_critic.q1.parameters())

        # override the critic loss since now we need to account for both q functions
        self.critic_optimiser = Adam(self.q_params, lr=critic_lr)

        # count the number of variables in the network in total
        self.var_counts = tuple(
            core.count_vars(module) for module in [self.actor_critic.pi, self.actor_critic.q1])

        self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d\n' % self.var_counts)

    def compute_critic_loss(self, data):
        states, actions, rewards, next_states, dones = data['obs'], data['act'], data['rew'], data['next_obs'], data['done']

        # compute the Q-values for the Q-function
        q_values = self.actor_critic.q1(states, actions)

        # Do Bellman backup for Q-functions
        with torch.no_grad():
            # Use the target actor critic to generate actions and values for the next state. This avoids over
            # correlation between the actors choices now and the valuation of the actors next state choices
            next_actions = self.actor_critic_target.act(next_states, numpy=False)
            q_target_values = self.actor_critic_target.q1(next_states, next_actions)
            bellman_backup = rewards + self.discount_rate * (1 - dones) * q_target_values

        q_loss = ((q_values - bellman_backup)**2).mean()

        loss_info = dict(Q1Vals=q_values.detach().numpy())

        return q_loss, loss_info

    def update_critic(self, data):
        for i in range(self.train_v_iters):
            self.critic_optimiser.zero_grad()
            q_loss, loss_info = self.compute_critic_loss(data)
            q_loss.backward()
            self.critic_optimiser.step()

            if i % 2 == 0:
                # update target networks
                with torch.no_grad():
                    for p, p_targ in zip(self.actor_critic.parameters(), self.actor_critic_target.parameters()):
                        # do addition and multiplication in place for p_targs but not for ps
                        p_targ.data.mul_(self.polyak)
                        p_targ.data.add_((1 - self.polyak) * p.data)

        # Record things
        self.logger.store(LossQ=q_loss.item(), **loss_info)

        return q_loss, loss_info

    def collect_experiences(self, env):
        state, episode_return, episode_length = env.reset(), 0, 0
        for time_step in range(self.local_steps_per_epoch):
            # choose the action and estimate the value using the actor critic
            action, q_val, logp = self.actor_critic.step(torch.as_tensor(state, dtype=torch.float32))

            # use the action to step through the environment
            next_state, reward, done, info = env.step(action)
            episode_return += reward
            episode_length += 1

            # save the details of the experiences in the buffer
            self.replay_buffer.store(state, action, next_state, reward, done, q_val, logp)

            # log the value estimated by the critic
            self.logger.store(VVals=q_val)

            # change the current state
            state = next_state

            # early termination stuff
            timeout = episode_length == self.max_ep_len
            terminal = done or timeout
            epoch_ended = time_step == self.local_steps_per_epoch - 1

            # we are finished with the epoch
            if terminal or epoch_ended:

                # finished due to self-imposed limit, not due to task being completed
                if epoch_ended and not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' % episode_length, flush=True)

                # Due to self-imposed time-step limit and not a result of the task being completed
                if timeout or epoch_ended:

                    # Use the critic to estimate the value of the state which results from being cut-off
                    _, last_q_value, _ = self.actor_critic.step(torch.as_tensor(state, dtype=torch.float32))

                # The episode is finished due to the agent being done
                else:
                    last_q_value = 0

                self.replay_buffer.finish_path(last_q_value)

                # If the episode finished, save the epsiode return/episode length in the logger
                self.logger.store(EpRet=episode_return, EpLen=episode_length)
                state, episode_return, episode_length = env.reset(), 0, 0

    def train(self, env):
        for epoch in range(self.num_epochs):
            start_time = time.time()
            self.collect_experiences(env)

            # save the model
            if (epoch % self.save_freq == 0) or (epoch == self.num_epochs - 1):
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

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=MINI_GRID_49)
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='ppo-class-fg1-49')
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
        num_cpus=args.cpu,
        experiment_name=args.exp_name
    )

    agent.train(env)
