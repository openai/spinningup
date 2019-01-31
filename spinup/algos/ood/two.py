import tensorflow as tf
import gym
import time
from spinup.algos.sac import core
from spinup.algos.ood.sac import SAC


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

                a1.store_logger(outs_1)
                a2.store_logger(outs_2)

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
