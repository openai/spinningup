import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.two import core
from spinup.algos.two.core import get_vars
from spinup.utils.logx import EpochLogger


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


"""

Soft Actor-Critic

(With slight variations that bring it closer to TD3)

"""


class SAC:
    def __init__(self, sess, env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
                 steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
                 polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
                 max_ep_len=1000, logger_kwargs=dict(), save_freq=1, name='sac'):
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
                ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                               | given states.
                ``pi``       (batch, act_dim)  | Samples actions from policy given
                                               | states.
                ``logp_pi``  (batch,)          | Gives log probability, according to
                                               | the policy, of the action sampled by
                                               | ``pi``. Critical: must be differentiable
                                               | with respect to policy parameters all
                                               | the way through action sampling.
                ``q1``       (batch,)          | Gives one estimate of Q* for
                                               | states in ``x_ph`` and actions in
                                               | ``a_ph``.
                ``q2``       (batch,)          | Gives another estimate of Q* for
                                               | states in ``x_ph`` and actions in
                                               | ``a_ph``.
                ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and
                                               | ``pi`` for states in ``x_ph``:
                                               | q1(x, pi(x)).
                ``q2_pi``    (batch,)          | Gives the composition of ``q2`` and
                                               | ``pi`` for states in ``x_ph``:
                                               | q2(x, pi(x)).
                ``v``        (batch,)          | Gives the value estimate for states
                                               | in ``x_ph``.
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the actor_critic
                function you provided to SAC.

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

            lr (float): Learning rate (used for both policy and value learning).

            alpha (float): Entropy regularization coefficient. (Equivalent to
                inverse of reward scale in the original SAC paper.)

            batch_size (int): Minibatch size for SGD.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

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
            mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)

        # Target value network
        with tf.variable_scope('%s/target' % name):
            _, _, _, _, _, _, _, v_targ = actor_critic(x2_ph, a_ph, **ac_kwargs)

        # Experience buffer
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in
                           ['%s/%s' % (name, v) for v in ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main']])
        print(('\nNumber of parameters: \t pi: %d, \t' +
               'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n') % var_counts)

        # Min Double-Q:
        min_q_pi = tf.minimum(q1_pi, q2_pi)

        # Targets for Q and V regression
        q_backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * v_targ)
        v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)

        # Soft actor-critic losses
        pi_loss = tf.reduce_mean(alpha * logp_pi - min_q_pi)
        q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
        v_loss = 0.5 * tf.reduce_mean((v_backup - v) ** 2)
        value_loss = q1_loss + q2_loss + v_loss

        # Policy train op
        # (has to be separate from value train op, because q1_pi appears in pi_loss)
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('%s/main/pi' % name))

        # Value train op
        # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
        value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        value_params = get_vars('%s/main/q' % name) + get_vars('%s/main/v' % name)
        with tf.control_dependencies([train_pi_op]):
            train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        with tf.control_dependencies([train_value_op]):
            target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                      for v_main, v_targ in
                                      zip(get_vars('%s/main' % name), get_vars('%s/target' % name))])

        # All ops to call during one training step
        step_ops = [pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi,
                    train_pi_op, train_value_op, target_update]

        # Initializing targets to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('%s/main' % name), get_vars('%s/target' % name))])

        # sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(target_init)

        # Setup model saving
        logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
                              outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2, 'v': v})

        # parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq
        self.start_steps = start_steps
        self.steps_per_epoch = steps_per_epoch

        # variables
        self.sess = sess
        self.logger = logger
        self.env, self.test_env = env, test_env
        self.replay_buffer = replay_buffer
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = x_ph, a_ph, x2_ph, r_ph, d_ph
        self.mu, self.pi, self.logp_pi = mu, pi, logp_pi
        self.q1, self.q2, self.q1_pi, self.q2_pi, v = q1, q2, q1_pi, q2_pi, v
        self.step_ops = step_ops

    def get_action(self, o, deterministic=False):
        act_op = self.mu if deterministic else self.pi
        return self.sess.run(act_op, feed_dict={self.x_ph: o.reshape(1, -1)})[0]

    def test_agent(self, n=10):
        # global sess, mu, pi, q1, q2, q1_pi, q2_pi
        for j in range(n):
            o, r, d, ep_ret, ep_len = self.test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = self.test_env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

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
        self.logger.log_tabular('VVals', with_min_and_max=True)
        self.logger.log_tabular('LogPi', with_min_and_max=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ1', average_only=True)
        self.logger.log_tabular('LossQ2', average_only=True)
        self.logger.log_tabular('LossV', average_only=True)
        self.logger.log_tabular('Time', time.time() - start_time)
        self.logger.dump_tabular()


def run_two(a1, a2, batch_size=100, epochs=100, max_ep_len=1000, start_steps=10000, steps_per_epoch=5000):
    start_time = time.time()
    total_steps = steps_per_epoch * epochs

    o_1, r_1, d_1, ep_ret_1, ep_len_1 = a1.env.reset(), 0, False, 0, 0
    o_2, r_2, d_2, ep_ret_2, ep_len_2 = a2.env.reset(), 0, False, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        if t > start_steps:
            a_1 = a1.get_action(o_1)
            a_2 = a2.get_action(o_2)
        else:
            a_1 = a1.env.action_space.sample()
            a_2 = a2.env.action_space.sample()

        # Step the env of a1
        o2_1, r_1, d_1, _ = a1.env.step(a_1)
        ep_ret_1 += r_1
        ep_len_1 += 1

        # Step the env of a2
        o2_2, r_2, d_2, _ = a2.env.step(a_2)
        ep_ret_2 += r_2
        ep_len_2 += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d_1 = False if ep_len_1 == max_ep_len else d_1

        # Store experiences to replay buffer
        a1.replay_buffer.store(o_1, a_1, r_1, o2_1, d_1)
        a1.replay_buffer.store(o_2, a_2, r_2, o2_2, d_2)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o_1 = o2_1
        o_2 = o2_2

        if d_1 or d_2 or (ep_len_1 == max_ep_len) or (ep_len_2 == max_ep_len):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            for j in range(ep_len_1):
                batch = a1.replay_buffer.sample_batch(batch_size)
                feed_dict_1 = {a1.x_ph: batch['obs1'],
                               a1.x2_ph: batch['obs2'],
                               a1.a_ph: batch['acts'],
                               a1.r_ph: batch['rews'],
                               a1.d_ph: batch['done'],
                               }
                feed_dict_2 = {a2.x_ph: batch['obs1'],
                               a2.x2_ph: batch['obs2'],
                               a2.a_ph: batch['acts'],
                               a2.r_ph: batch['rews'],
                               a2.d_ph: batch['done'],
                               }
                outs_1 = a1.sess.run(a1.step_ops, feed_dict_1)
                outs_2 = a2.sess.run(a2.step_ops, feed_dict_2)

                a1.logger.store(LossPi=outs_1[0], LossQ1=outs_1[1], LossQ2=outs_1[2],
                                LossV=outs_1[3], Q1Vals=outs_1[4], Q2Vals=outs_1[5],
                                VVals=outs_1[6], LogPi=outs_1[7])
                a2.logger.store(LossPi=outs_2[0], LossQ1=outs_2[1], LossQ2=outs_2[2],
                                LossV=outs_2[3], Q1Vals=outs_2[4], Q2Vals=outs_2[5],
                                VVals=outs_2[6], LogPi=outs_2[7])

            a1.logger.store(EpRet=ep_ret_1, EpLen=ep_len_1)
            a2.logger.store(EpRet=ep_ret_2, EpLen=ep_len_2)
            o_1, r_1, d_1, ep_ret_1, ep_len_1 = a1.env.reset(), 0, False, 0, 0
            o_2, r_2, d_2, ep_ret_2, ep_len_2 = a2.env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch
            a1.wrap_up_epoch(epoch, t, start_time)
            a2.wrap_up_epoch(epoch, t, start_time)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs_1 = setup_logger_kwargs(args.exp_name + '-a1', args.seed)
    logger_kwargs_2 = setup_logger_kwargs(args.exp_name + '-a2', args.seed)

    session = tf.Session()

    a1 = SAC(session, lambda: gym.make(args.env), actor_critic=core.mlp_actor_critic,
             ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
             gamma=args.gamma, seed=args.seed, epochs=args.epochs,
             logger_kwargs=logger_kwargs_1, name='sac')

    a2 = SAC(session, lambda: gym.make(args.env), actor_critic=core.mlp_actor_critic,
             ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
             gamma=args.gamma, seed=args.seed, epochs=args.epochs,
             logger_kwargs=logger_kwargs_2, alpha=0.0, name='ddpg')

    run_two(a1, a2)
