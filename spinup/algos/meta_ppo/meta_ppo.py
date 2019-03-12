import spinup.algos.meta_ppo.core as core
import numpy as np
import tensorflow as tf
from spinup.utils.mpi_tf import MpiAdamOptimizer
import math
import sys
import gym

#env = gym.make('FrozenLake-v0')
env = gym.make('Swimmer-v2')
#env = gym.make('CartPole-v1')

def train(env = env, meta_learn = True, should_print = True, epochs = 5, max_ep_len = 250, episodes_per_epoch = 16, seed = 5):

    tf.reset_default_graph()

    n = 1000000
    lr = 1e-3
    meta_lr = 1e-1
    train_iters = 20

    v_loss_ratio=100
    display = epochs # number of update panels to show
    steps_per_epoch = max_ep_len * episodes_per_epoch
    clip_ratio = 0.2
    actor_critic=core.mlp_actor_critic
    ac_kwargs=dict()
    env.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)

    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['observation_space'] = env.observation_space


    '''
    input: dictionary of hyperparameters
    currently {'lam': tensor, 'gamma': tensor}

    return: placeholders, loss

    placeholders is a list of placeholders
    consisting of [obs_buf_ph, act_buf_ph, rew_buf_ph, msk_buf_ph, end_buf_ph, logp_old_buf_ph]

    loss is just a tensor which is the loss
    '''
    def compute_losses(hyperparams):

        obs_buf_ph = tf.placeholder(dtype=core.type_from_space(env.observation_space), \
            shape=core.shape_from_space(env.observation_space, steps_per_epoch))
        act_buf_ph = tf.placeholder(dtype=core.type_from_space(env.action_space), \
            shape=core.shape_from_space(env.action_space, steps_per_epoch))
        rew_buf_ph = tf.placeholder(dtype=tf.float32, shape=(steps_per_epoch,))
        msk_buf_ph = tf.placeholder(dtype=tf.float32, shape=(steps_per_epoch,))
        logp_old_buf_ph = tf.placeholder(dtype=tf.float32, shape=(steps_per_epoch,))
        old_v_ph = tf.placeholder(dtype=tf.float32, shape=(steps_per_epoch,))

        all_phs = (obs_buf_ph, act_buf_ph, rew_buf_ph, msk_buf_ph, logp_old_buf_ph, old_v_ph)

        gamma = 1. - 1. / tf.maximum(hyperparams['gamma'], 1.1)
        lam = 1. - 1 / tf.maximum(hyperparams['lam'], 1.1)

        ac_kwargs['discount_factor'] = gamma
        pi_buf, logp_buf, logp_pi_buf, v_buf = actor_critic(obs_buf_ph, act_buf_ph, **ac_kwargs)

        #training_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        #print ([v.name for v in training_vars])

        #with tf.variable_scope('test_scope'):
        #pi_buf, logp_buf, logp_pi_buf, v_buf = actor_critic(obs_buf_ph, act_buf_ph, **ac_kwargs)

        bootstrap_mask_array = np.zeros(steps_per_epoch, dtype=np.float32)
        bootstrap_mask_array[max_ep_len-1::max_ep_len] += 1.
        bootstrap_mask = tf.convert_to_tensor(bootstrap_mask_array)
        bootstrap_mask = bootstrap_mask * (1. - msk_buf_ph)

        rew_buf_adjusted = rew_buf_ph * (1 - bootstrap_mask) + old_v_ph * bootstrap_mask

        ret_buf = core.exponential_avg(rew_buf_adjusted, gamma, max_ep_len, episodes_per_epoch)
        # ret_buf = rew_buf_adjusted + gamma
        # ret_buf = tf.stop_gradient(ret_buf)

        v_loss = tf.sqrt(tf.reduce_mean((ret_buf - v_buf)**2 * (1. - msk_buf_ph)))

        all_episode_deltas = []
        for i in range(episodes_per_epoch):
            episode_rews = rew_buf_adjusted[i * max_ep_len: (i+1) * max_ep_len]
            episode_vals = old_v_ph[i * max_ep_len: (i+1) * max_ep_len]
            episode_deltas = episode_rews[:-1] + gamma * episode_vals[1:] - episode_vals[:-1]
            episode_deltas = tf.concat([episode_deltas, np.zeros(1, dtype = np.float32)], axis = 0)
            all_episode_deltas.append(episode_deltas)
        delta_buf = tf.concat(all_episode_deltas, axis = 0)
        adv_buf = core.exponential_avg(delta_buf, gamma * lam, max_ep_len, episodes_per_epoch)
        # adv_buf = delta_buf + gamma * lam
        mean, var = tf.nn.moments(adv_buf, axes=[0])
        raw_adv = adv_buf
        adv_buf = (adv_buf - mean) / tf.math.sqrt(var)
        # adv_buf = tf.stop_gradient(adv_buf)
        
        ratio = tf.exp(logp_buf - logp_old_buf_ph)
        min_adv = tf.where(adv_buf>0, (1+clip_ratio)*adv_buf, (1-clip_ratio)*adv_buf)
        pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_buf, min_adv))
        
        net_loss = pi_loss+v_loss*v_loss_ratio

        return all_phs, net_loss

    # print (grads)

    gamma_default = tf.constant(100., dtype=tf.float32)
    lam_default = tf.constant(20., dtype=tf.float32)


    hyperparams = {
        'gamma': tf.get_variable("gamma", dtype=tf.float32, initializer=gamma_default, trainable = True),\
        'lam': tf.get_variable("lam", dtype=tf.float32, initializer=lam_default, trainable = True)
    }

    hyper_values = list(hyperparams.values())

    metaparams = {'gamma': gamma_default, 'lam': lam_default}

    with tf.variable_scope('loss_scope'):
        traj0_phs, traj0_loss = compute_losses(hyperparams)

    params = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v not in hyper_values]

    params_optimizer = MpiAdamOptimizer(learning_rate=lr)
    metaparams_optimizer = MpiAdamOptimizer(learning_rate=meta_lr)

    param_grads = tf.gradients(traj0_loss, params)
    grad_dict = dict(zip(params, param_grads))

    def custom_get_var(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        return var - lr * grad_dict[var]

    with tf.variable_scope('loss_scope', reuse = True, custom_getter = custom_get_var):
        traj1_phs, traj1_loss = compute_losses(metaparams)

    grads_and_vars = params_optimizer.compute_gradients(traj0_loss, params)
    train_params = params_optimizer.apply_gradients(grads_and_vars)

    metaparam_grads = metaparams_optimizer.compute_gradients(traj1_loss, hyper_values)
    train_metaparams = metaparams_optimizer.apply_gradients(metaparam_grads)

    # This network is seperate, obs_ph is used one observation at a time to get pi, and logp_pi
    with tf.variable_scope('loss_scope', reuse = True):
        obs_ph = tf.placeholder(dtype=core.type_from_space(env.observation_space), shape=env.observation_space.shape)
        obs = tf.reshape(obs_ph, shape = core.shape_from_space(env.observation_space, 1))
        act_ph = tf.placeholder(dtype=core.type_from_space(env.action_space), shape=env.action_space.shape)
        act = tf.reshape(act_ph, shape = core.shape_from_space(env.action_space, 1))
        gamma = 1. - 1. / tf.maximum(hyperparams['gamma'], 1.1)
        ac_kwargs['discount_factor'] = gamma
        pi, _1, logp_pi, v = actor_critic(obs, act, **ac_kwargs)
        pi = pi[0]
        logp_pi = logp_pi[0]
        v = v[0]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # core.load_variables('/tmp/variables', sess = sess)

    '''
    print ([v.name for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
    '''
    # print (sess.run([tf.get_default_graph().get_tensor_by_name("pi/dense/kernel:0")]))

    def collect_observations():
        obs_buf = np.zeros(dtype=core.type_from_space(env.observation_space), \
        shape=core.shape_from_space(env.observation_space, steps_per_epoch))
        act_buf = np.zeros(dtype=core.type_from_space(env.action_space), \
            shape=core.shape_from_space(env.action_space, steps_per_epoch))
        rew_buf = np.zeros(dtype=np.float32, shape=(steps_per_epoch,))
        # did the agent just die? Replace all future rewards by zero.
        msk_buf = np.zeros(dtype=np.float32, shape=(steps_per_epoch,))

        logp_old_buf = np.zeros(dtype=np.float32, shape=(steps_per_epoch,))
        old_v_buf = np.zeros(dtype=np.float32, shape=(steps_per_epoch,))

        for i in range(episodes_per_epoch):
            done = False
            new_obs = env.reset()
            for j in range(max_ep_len):
                idx = i * max_ep_len + j
                if done:
                    rew_buf[idx] = 0.
                    msk_buf[idx] = 1.
                    continue

                obs = new_obs
                pi_, logp_pi_, v_ = sess.run([pi, logp_pi, v], feed_dict = {obs_ph:obs})
                a = pi_

                new_obs, rew, done, _ = env.step(a)
                '''
                if i < 100:
                    print(i, a, obs, done)
                '''
                #if epoch % 20 == 0 and i < 500:
                #    env.render()
                # print (a, obs, rew, done)
                obs_buf[idx] = obs
                act_buf[idx] = a
                rew_buf[idx] = rew
                msk_buf[idx] = 1. if done else 0.
                logp_old_buf[idx] = logp_pi_
                old_v_buf[idx] = v_
        return [obs_buf, act_buf, rew_buf, msk_buf, logp_old_buf, old_v_buf]

    ret = []

    for epoch in range(epochs):

        # Collect trajectories according to actor_critic
        
        trajectories_0 = collect_observations()
        trajectories_1 = collect_observations()

        bufs_dict = dict(zip(traj0_phs + traj1_phs, trajectories_0 + trajectories_1))

        # mean_, var_ = sess.run([mean, var], feed_dict=bufs_dict)
        # print (mean_, np.sqrt(var_))

        net_loss_1 = sess.run(traj0_loss, feed_dict=bufs_dict)
        
        '''
        if epoch % 1 == 0:
            raw_adv_, adv_buf_, delta_buf_, ret_buf_, rew_buf_adjusted_, val_buf_, logp_buf_ = sess.run([raw_adv, adv_buf, delta_buf, ret_buf, rew_buf_adjusted, v_buf, logp_buf], feed_dict=bufs_dict)
            for i in range(600):
                #print (raw_adv_[i], adv_buf_[i], ret_buf_[i], rew_buf_adjusted_[i], val_buf_[i], msk_buf[i], end_buf[i], act_buf[i], logp_buf_[i], obs_buf[i])
                print (delta_buf_[i], ret_buf_[i], val_buf_[i])
        '''
        
        


        #entropy, advantage = sess.run([logp_buf, adv_buf], feed_dict=bufs_dict)

        #entropy = [sum(entropy[i * 200: (i+1)*200]) / 200 for i in range(20)]
        #advantage = [sum(advantage[i * 200: (i+1)*200]) / 200 for i in range(20)]
        #print (np.around(entropy, decimals=1))
        #print (np.around(advantage, decimals=1))

        for _ in range(train_iters):
            sess.run(train_params, feed_dict=bufs_dict)
            if meta_learn:
                sess.run(train_metaparams, feed_dict=bufs_dict) # should this be done in a separate for loop?


        net_loss_2 = sess.run(traj0_loss, feed_dict=bufs_dict)

        gamma_, lam_ = sess.run([hyperparams['gamma'], hyperparams['lam']], feed_dict=bufs_dict)
        '''
        for j in range(1002):
            print (rew_buf[j])
        print()
        print (np.sum(rew_buf) / len(rew_buf))
        '''

        if epoch % int(epochs / display) == 0:
            num_trajectories = episodes_per_epoch
            ep_ret = np.sum(trajectories_0[2]) / num_trajectories
            ep_len = max_ep_len * (1. - np.sum(trajectories_0[3]) / steps_per_epoch)
            env_interacts = ep_len * episodes_per_epoch
            ret.append((env_interacts, ep_ret, gamma_, lam_))
            if should_print:
                print ('epoch: ', epoch)
                print ('ep_ret: ', ep_ret)
                print ('ep_len: ', ep_len)
                print ('loss_1: ', net_loss_1)
                print ('loss_2: ', net_loss_2)
                print ('hyperparams', (gamma_, lam_))
                print ('---------------------------------------------------------')
    return ret

# train(env)