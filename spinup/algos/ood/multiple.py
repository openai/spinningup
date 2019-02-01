import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.sac import core as sac_core
from spinup.algos.ddpg import core as ddpg_core
from spinup.algos.ood.sac import SAC
from spinup.algos.ood.ddpg import DDPG


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


def run_multiple(algorithms, replay_buffer, batch_size=100, epochs=100, max_ep_len=1000, start_steps=10000,
                 steps_per_epoch=5000):
    start_time = time.time()
    total_steps = steps_per_epoch * epochs
    steps = [[a.env.reset(), None, 0, False, 0, 0] for a in algorithms]  # o, o2, r, d, ep_ret, ep_len

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        for i in range(len(algorithms)):
            algorithm = algorithms[i]
            o, o2, r, d, ep_ret, ep_len = steps[i]

            if t > start_steps:
                a = algorithm.get_action(o)
            else:
                a = algorithm.env.action_space.sample()

            # Step the env
            o2, r, d, _ = algorithm.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == max_ep_len else d

            # Store experiences to replay buffer
            replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # Update steps
            steps[i] = [o, o2, r, d, ep_ret, ep_len]

        done = any(step[3] for step in steps)
        reached_max_ep_len = any(step[5] == max_ep_len for step in steps)

        if done or reached_max_ep_len:
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            for _ in range(max(step[5] for step in steps)):
                batch = replay_buffer.sample_batch(batch_size)

                for algorithm in algorithms:
                    algorithm.update(batch)

            for i in range(len(algorithms)):
                algorithm = algorithms[i]
                o, o2, r, d, ep_ret, ep_len = steps[i]
                algorithm.logger.store(EpRet=ep_ret, EpLen=ep_len)

                o, r, d, ep_ret, ep_len = algorithm.env.reset(), 0, False, 0, 0
                steps[i] = [o, o2, r, d, ep_ret, ep_len]

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch
            for algorithm in algorithms:
                algorithm.wrap_up_epoch(epoch, t, start_time)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('algorithms', type=str)
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='multiple')
    args = parser.parse_args()

    session = tf.Session()
    env = gym.make(args.env)
    rb = ReplayBuffer(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        size=int(1e6)
    )

    from spinup.utils.run_utils import setup_logger_kwargs

    all_algorithms = []
    for i, algorithm in enumerate(args.algorithms.split(',')):
        algorithm_name = '%s-%d-%s' % (args.exp_name, i + 1, algorithm)
        logger_kwargs = setup_logger_kwargs(algorithm_name, args.seed)

        if algorithm == 'sac':
            all_algorithms.append(
                SAC(session, rb, lambda: gym.make(args.env), actor_critic=sac_core.mlp_actor_critic,
                    ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                    gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                    logger_kwargs=logger_kwargs, name=algorithm_name)
            )
        elif algorithm == 'sac_zero_alpha':
            all_algorithms.append(
                SAC(session, rb, lambda: gym.make(args.env), actor_critic=sac_core.mlp_actor_critic,
                    ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                    gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                    logger_kwargs=logger_kwargs, name=algorithm_name, alpha=0.0)
            )
        elif algorithm == 'ddpg':
            all_algorithms.append(
                DDPG(session, rb, lambda: gym.make(args.env), actor_critic=ddpg_core.mlp_actor_critic,
                     ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                     gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                     logger_kwargs=logger_kwargs, name=algorithm_name)
            )

    run_multiple(all_algorithms, rb)
