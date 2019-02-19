import spinup.algos.vpg2.core as core
import numpy as np
import tensorflow as tf
from spinup.utils.mpi_tf import MpiAdamOptimizer
import math

import gym
#env = gym.make('FrozenLake-v0')
#env = gym.make('Swimmer-v2')
env = gym.make('CartPole-v1')
n = 1000000
learning_rate = 0.001
lr = 3e-4
train_iters=3
v_loss_ratio=100
epochs = 1000
display = epochs # number of update panels to show
steps_per_epoch = 4000
max_ep_len = 1000 # episodes end at step 1000 no matter what
actor_critic=core.mlp_actor_critic
ac_kwargs=dict()
seed = 20
tf.set_random_seed(seed)
np.random.seed(seed)

ac_kwargs['action_space'] = env.action_space

obs_buf = np.zeros(dtype=core.type_from_space(env.observation_space), \
    shape=core.shape_from_space(env.observation_space, steps_per_epoch))
act_buf = np.zeros(dtype=core.type_from_space(env.action_space), \
    shape=core.shape_from_space(env.action_space, steps_per_epoch))
rew_buf = np.zeros(dtype=np.float32, shape=(steps_per_epoch,))
msk_buf = np.zeros(dtype=np.float32, shape=(steps_per_epoch,))

obs_buf_ph = tf.placeholder(dtype=core.type_from_space(env.observation_space), \
    shape=core.shape_from_space(env.observation_space, None))
act_buf_ph = tf.placeholder(dtype=core.type_from_space(env.action_space), \
    shape=core.shape_from_space(env.action_space, None))
rew_buf_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
msk_buf_ph = tf.placeholder(dtype=tf.float32, shape=(None,))

gamma = tf.constant(0.99)
lam = tf.constant(0.95)

pi_buf, logp_buf, logp_pi_buf, v_buf = actor_critic(obs_buf_ph, act_buf_ph, **ac_kwargs)
ret_buf = core.masked_suffix_sum(rew_buf_ph, msk_buf_ph, gamma, axis=0, last_val=v_buf[-1])
ret_buf = tf.stop_gradient(ret_buf)

v_loss = tf.math.sqrt(tf.reduce_mean((ret_buf - v_buf)**2))

delta_buf = rew_buf_ph[:-1] + gamma * v_buf[1:] * (1 - msk_buf_ph[:-1]) - v_buf[:-1]
adv_buf = core.masked_suffix_sum(delta_buf, msk_buf_ph[:-1], gamma * lam, axis=0, last_val=0)
mean, var = tf.nn.moments(adv_buf, axes=[0])
adv_buf = (adv_buf - mean) / tf.math.sqrt(var)
adv_buf = tf.stop_gradient(adv_buf)

pi_loss = -tf.reduce_mean(logp_buf[:-1] * adv_buf)

train = MpiAdamOptimizer(learning_rate=lr).minimize(pi_loss+v_loss*v_loss_ratio)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(epochs):

    # Collect trajectories according to actor_critic
    done = True
    for i in range(steps_per_epoch):
        if done:
            obs = env.reset()
            j = 0
        if np.isscalar(obs):
            obs = np.array([obs])
        else:
            obs = obs.reshape((1,-1))
        a = sess.run(pi_buf, feed_dict = {obs_buf_ph:obs})[0]
        obs, rew, done, _ = env.step(a)
        j += 1
        done = done or j >= max_ep_len
        #if epoch % 20 == 0 and i < 500:
        #    env.render()
        # print (a, obs, rew, done)
        obs_buf[i] = obs
        act_buf[i] = a
        rew_buf[i] = rew
        msk_buf[i] = 1 if done else 0

    bufs_dict = {obs_buf_ph:obs_buf, act_buf_ph:act_buf, rew_buf_ph:rew_buf, msk_buf_ph:msk_buf}

    # mean_, var_ = sess.run([mean, var], feed_dict=bufs_dict)
    # print (mean_, np.sqrt(var_))

    pi_loss_1, v_loss_1 = sess.run([pi_loss, v_loss], feed_dict=bufs_dict)

    #entropy, advantage = sess.run([logp_buf, adv_buf], feed_dict=bufs_dict)

    #entropy = [sum(entropy[i * 200: (i+1)*200]) / 200 for i in range(20)]
    #advantage = [sum(advantage[i * 200: (i+1)*200]) / 200 for i in range(20)]
    #print (np.around(entropy, decimals=1))
    #print (np.around(advantage, decimals=1))

    for _ in range(train_iters):
        sess.run(train, feed_dict=bufs_dict)

    pi_loss_2, v_loss_2 = sess.run([pi_loss, v_loss], feed_dict=bufs_dict)


    if epoch % int(epochs / display) == 0:
        print ('epoch: ', epoch)
        print ('ep_ret: ', np.sum(rew_buf) / np.sum(msk_buf))
        print ('ep_len: ', steps_per_epoch / np.sum(msk_buf))
        print ('loss_1: ', (pi_loss_1, v_loss_1))
        print ('loss_2: ', (pi_loss_2, v_loss_2))
        print ('---------------------------------------------------------')