import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.dqn import core
from spinup.algos.dqn.core import get_vars
from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DQN agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


"""

Deep Q Network (DQN)

"""
def dqn(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=100, epochs=150, replay_size=int(1e6), gamma=0.99,
        epsilon_start=1, epsilon_step=1e-4, epsilon_end=0.1,
        q_lr=1e-3, batch_size=100, start_steps=5000,
        act_noise=0.1, max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    ## step is 1e-6
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols
            for state, ``x_ph``, and action, ``a_ph``, and returns the main
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``q``        (batch,)          | Gives the current estimate
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph = core.placeholders_from_spaces(env.observation_space,
                                                      env.action_space,
                                                      env.observation_space)
    r_ph, d_ph = core.placeholders(None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        q = actor_critic(x_ph, **ac_kwargs)
        q_2 = actor_critic(x2_ph, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/q', 'main'])
    print('\nNumber of parameters: \t q: %d, \t total: %d\n' % var_counts)

    # Bellman backup for Q function
    q_a = tf.reduce_sum(tf.one_hot(a_ph, depth=env.action_space.n)*q, axis=1)
    backup = r_ph + gamma*(1-d_ph)*tf.stop_gradient(tf.reduce_max(q_2, axis=1))

    # DQN losses
    q_loss = tf.reduce_mean((q_a-backup)**2)

    # Train ops for q
    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
    train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
                          outputs={'q': q,
                                   'pi': tf.argmax(q, axis=1)})

    def get_action(obs, eps):
        if np.random.random() < eps:
            a = env.action_space.sample()
        else:
            # t = time.time()
            action = tf.squeeze(tf.argmax(q, axis=1))
            a = sess.run(action, feed_dict={x_ph: obs.reshape(1, -1)})
            # print('Time to pick action: {}, obs size'.format(time.time() - t))
        return a

    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Anneal epsilon linearly from epsilon_start with epsilon_step
        With epsilon probabilty we choose a random action for
        better exploration.
        """
        if t % 500 == 0:
            print('t: {}'.format(t))
        epsilon = epsilon_start - (t * epsilon_step)
        if epsilon < epsilon_end:
            epsilon = epsilon_end
        a = get_action(o, epsilon)

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            """
            Perform all DQN updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if t > start_steps:
            batch = replay_buffer.sample_batch(batch_size)
            feed_dict = {
                x_ph: batch['obs1'],
                x2_ph: batch['obs2'],
                a_ph: batch['acts'],
                r_ph: batch['rews'],
                d_ph: batch['done']
            }

            # Q-learning update
            # t = time.time()
            outs = sess.run([q_loss, q, train_q_op], feed_dict)
            # print('Outs time: {}'.format(time.time() - t))
            logger.store(LossQ=outs[0], QVals=outs[1])


        # End of epoch wrap-up
        if t > start_steps and (t - start_steps) % steps_per_epoch == 0:
            epoch = (t - start_steps) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            try:
                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('EpRet', with_min_and_max=True)
                logger.log_tabular('TestEpRet', with_min_and_max=True)
                logger.log_tabular('EpLen', average_only=True)
                logger.log_tabular('TestEpLen', average_only=True)
                logger.log_tabular('TotalEnvInteracts', t)
                logger.log_tabular('QVals', with_min_and_max=True)
                logger.log_tabular('LossQ', average_only=True)
                logger.log_tabular('Epsilon', epsilon)
                logger.log_tabular('Time', time.time()-start_time)
                logger.dump_tabular()
            except Exception as e:
                import ipdb;ipdb.set_trace()
                print(e)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='dqn')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    dqn(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
