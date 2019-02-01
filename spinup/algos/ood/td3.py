import numpy as np
import tensorflow as tf
import time
from spinup.algos.td3 import core
from spinup.algos.td3.core import get_vars
from spinup.utils.logx import EpochLogger

"""

TD3 (Twin Delayed DDPG)

"""


class TD3:
    def __init__(self, sess, replay_buffer, env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
                 steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99, polyak=0.995, pi_lr=1e-3,
                 q_lr=1e-3, batch_size=100, start_steps=10000, act_noise=0.1, target_noise=0.2, noise_clip=0.5,
                 policy_delay=2, max_ep_len=1000, logger_kwargs=dict(), save_freq=1, name='td3'):
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
                ``pi``       (batch, act_dim)  | Deterministically computes actions
                                               | from policy given states.
                ``q1``       (batch,)          | Gives one estimate of Q* for
                                               | states in ``x_ph`` and actions in
                                               | ``a_ph``.
                ``q2``       (batch,)          | Gives another estimate of Q* for
                                               | states in ``x_ph`` and actions in
                                               | ``a_ph``.
                ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and
                                               | ``pi`` for states in ``x_ph``:
                                               | q1(x, pi(x)).
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the actor_critic
                function you provided to TD3.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs)
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs to run and train agent.

            replay_size (int): Maximum length of replay buffer.

            gamma (float): Discount factor. (Always between 0 and 1.)

            polyak (float): Interpolation factor in polyak averaging for target
                networks. Target networks are updated towards main networks
                according to:

                .. math:: \\theta_{\\text{targ}} \\leftarrow
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually
                close to 1.)

            pi_lr (float): Learning rate for policy.

            q_lr (float): Learning rate for Q-networks.

            batch_size (int): Minibatch size for SGD.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

            act_noise (float): Stddev for Gaussian exploration noise added to
                policy at training time. (At test time, no noise is added.)

            target_noise (float): Stddev for smoothing noise added to target
                policy.

            noise_clip (float): Limit for absolute value of target policy
                smoothing noise.

            policy_delay (int): Policy will only be updated once every
                policy_delay times for each update of the Q-networks.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

        """

        params = locals()
        params.pop('sess')
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(params)

        tf.set_random_seed(seed)
        np.random.seed(seed)

        env, test_env = env_fn(), env_fn()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = env.action_space.high[0]

        # Share information about action space with policy architecture
        ac_kwargs['action_space'] = env.action_space

        # Inputs to computation graph
        x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

        # Main outputs from computation graph
        with tf.variable_scope('%s/main' % name):
            pi, q1, q2, q1_pi = actor_critic(x_ph, a_ph, **ac_kwargs)

        # Target policy network
        with tf.variable_scope('%s/target' % name):
            pi_targ, _, _, _ = actor_critic(x2_ph, a_ph, **ac_kwargs)

        # Target Q networks
        with tf.variable_scope('%s/target' % name, reuse=True):
            # Target policy smoothing, by adding clipped noise to target actions
            epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
            epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = tf.clip_by_value(a2, -act_limit, act_limit)

            # Target Q-values, using action from target policy
            _, q1_targ, q2_targ, _ = actor_critic(x2_ph, a2, **ac_kwargs)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in
                           ['%s/%s' % (name, v) for v in ['main/pi', 'main/q1', 'main/q2', 'main']])
        print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n' % var_counts)

        # Bellman backup for Q functions, using Clipped Double-Q targets
        min_q_targ = tf.minimum(q1_targ, q2_targ)
        backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * min_q_targ)

        # TD3 losses
        pi_loss = -tf.reduce_mean(q1_pi)
        q1_loss = tf.reduce_mean((q1 - backup) ** 2)
        q2_loss = tf.reduce_mean((q2 - backup) ** 2)
        q_loss = q1_loss + q2_loss

        # Separate train ops for pi, q
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
        q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
        train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('%s/main/pi' % name))
        train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('%s/main/q' % name))

        # Polyak averaging for target variables
        target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                  for v_main, v_targ in zip(get_vars('%s/main' % name), get_vars('%s/target' % name))])

        # Initializing targets to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('%s/main' % name), get_vars('%s/target' % name))])

        sess.run(tf.global_variables_initializer())
        sess.run(target_init)

        # Setup model saving
        logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, outputs={'pi': pi, 'q1': q1, 'q2': q2})

        # parameters
        self.act_noise = act_noise
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_ep_len = max_ep_len
        self.policy_delay = policy_delay
        self.replay_buffer = replay_buffer
        self.save_freq = save_freq
        self.sess = sess
        self.start_steps = start_steps
        self.steps_per_epoch = steps_per_epoch

        # variables
        self.logger = logger
        self.env, self.test_env = env, test_env
        self.act_dim, self.act_limit = act_dim, act_limit
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = x_ph, a_ph, x2_ph, r_ph, d_ph
        self.pi, self.q1, self.q2, self.q1_pi = pi, q1, q2, q1_pi
        self.pi_loss, self.q_loss = pi_loss, q_loss
        self.train_pi_op, self.train_q_op = train_pi_op, train_q_op
        self.target_update = target_update

    def get_action(self, o, deterministic=False):
        a = self.sess.run(self.pi, feed_dict={self.x_ph: o.reshape(1, -1)})[0]
        a += 0 if deterministic else self.act_noise * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def test_agent(self, n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_len = self.test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = self.test_env.step(self.get_action(o, deterministic=True))
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def update(self, batch, step):
        feed_dict = {self.x_ph: batch['obs1'],
                     self.x2_ph: batch['obs2'],
                     self.a_ph: batch['acts'],
                     self.r_ph: batch['rews'],
                     self.d_ph: batch['done']
                     }
        q_step_ops = [self.q_loss, self.q1, self.q2, self.train_q_op]
        outs = self.sess.run(q_step_ops, feed_dict)
        self.logger.store(LossQ=outs[0], Q1Vals=outs[1], Q2Vals=outs[2])

        if step % self.policy_delay == 0:
            # Delayed policy update
            outs = self.sess.run([self.pi_loss, self.train_pi_op, self.target_update], feed_dict)
            self.logger.store(LossPi=outs[0])

    def wrap_up_epoch(self, epoch, t, start_time):
        # Save model
        if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
            self.logger.save_state({'env': self.env}, None)

        # Test the performance of the deterministic version of the agent.
        self.test_agent()

        # Log info about epoch
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('TestEpLen', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts', t)
        self.logger.log_tabular('Q1Vals', with_min_and_max=True)
        self.logger.log_tabular('Q2Vals', with_min_and_max=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('Time', time.time() - start_time)
        self.logger.dump_tabular()
