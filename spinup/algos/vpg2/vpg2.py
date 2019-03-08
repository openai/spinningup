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
lr = 3e-4
train_iters=5
v_loss_ratio=100
epochs = 1000
display = epochs # number of update panels to show
steps_per_epoch = 4000
train_iters = 80
max_ep_len = 500 # episodes end at step 1000 no matter what
clip_ratio = 0.2
actor_critic=core.mlp_actor_critic
ac_kwargs=dict()
seed = 10
env.seed(seed)
tf.set_random_seed(seed)
np.random.seed(seed)

ac_kwargs['action_space'] = env.action_space
ac_kwargs['observation_space'] = env.observation_space

obs_buf = np.zeros(dtype=core.type_from_space(env.observation_space), \
    shape=core.shape_from_space(env.observation_space, steps_per_epoch))
act_buf = np.zeros(dtype=core.type_from_space(env.action_space), \
    shape=core.shape_from_space(env.action_space, steps_per_epoch))
rew_buf = np.zeros(dtype=np.float32, shape=(steps_per_epoch,))
# did the agent just die? Replace all future rewards by zero.
msk_buf = np.zeros(dtype=np.float32, shape=(steps_per_epoch,))

# Did this trajectory get terminated by epoch or max_ep_len? Replace reward by value network output
end_buf = np.zeros(dtype=np.float32, shape=(steps_per_epoch,))
logp_old_buf = np.zeros(dtype=np.float32, shape=(steps_per_epoch,))

obs_buf_ph = tf.placeholder(dtype=core.type_from_space(env.observation_space), \
    shape=core.shape_from_space(env.observation_space, None))
act_buf_ph = tf.placeholder(dtype=core.type_from_space(env.action_space), \
    shape=core.shape_from_space(env.action_space, None))
rew_buf_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
msk_buf_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
end_buf_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
logp_old_buf_ph = tf.placeholder(dtype=tf.float32, shape=(None,))

gamma = tf.constant(0.99)
lam = tf.constant(0.97)

pi_buf, logp_buf, logp_pi_buf, v_buf = actor_critic(obs_buf_ph, act_buf_ph, **ac_kwargs)

rew_buf_adjusted = rew_buf_ph * (1 - end_buf_ph) + v_buf * end_buf_ph

ret_buf = core.masked_suffix_sum(rew_buf_adjusted, msk_buf_ph, gamma, axis=0)
ret_buf = tf.stop_gradient(ret_buf)

v_loss = tf.sqrt(tf.reduce_mean((ret_buf - v_buf)**2))

delta_buf = rew_buf_adjusted[:-1] + gamma * v_buf[1:] * (1 - msk_buf_ph[:-1]) - v_buf[:-1]
adv_buf = core.masked_suffix_sum(delta_buf, msk_buf_ph[:-1], gamma * lam, axis=0)
mean, var = tf.nn.moments(adv_buf, axes=[0])
raw_adv = adv_buf
adv_buf = (adv_buf - mean) / tf.math.sqrt(var)
adv_buf = tf.stop_gradient(adv_buf)

'''
clipped_logp = tf.minimum(logp_buf, logp_old_buf_ph + np.log(1+clip_ratio))
clipped_logp = tf.maximum(logp_buf, logp_old_buf_ph - np.log(1+clip_ratio))

# vpg objective
# pi_loss = -tf.reduce_mean(logp_buf[:-1] * adv_buf)
# ppo objective
pi_loss = -tf.reduce_mean(clipped_logp[:-1] * adv_buf)
'''

ratio = tf.exp(logp_buf - logp_old_buf_ph)
min_adv = tf.where(adv_buf>0, (1+clip_ratio)*adv_buf, (1-clip_ratio)*adv_buf)
pi_loss = -tf.reduce_mean(tf.minimum(ratio[:-1] * adv_buf, min_adv))

train = MpiAdamOptimizer(learning_rate=lr).minimize(pi_loss+v_loss*v_loss_ratio)

train_pi = MpiAdamOptimizer(learning_rate=3e-4).minimize(pi_loss)
train_v = MpiAdamOptimizer(learning_rate=1e-3).minimize(v_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# core.load_variables('/tmp/variables', sess = sess)

'''
print ([v.name for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
'''
# print (sess.run([tf.get_default_graph().get_tensor_by_name("pi/dense/kernel:0")]))

done = True

for epoch in range(epochs):

    # Collect trajectories according to actor_critic
    
    for i in range(steps_per_epoch):
        if done:
            new_obs = env.reset()
            j = 0

        if np.isscalar(new_obs):
            obs = np.array([new_obs])
        else:
            obs = new_obs.reshape((1,-1))
        pi_buf_, logp_pi_buf_  = sess.run([pi_buf, logp_pi_buf], feed_dict = {obs_buf_ph:obs})
        a = pi_buf_[0]

        new_obs, rew, done, _ = env.step(a)
        '''
        if i < 100:
            print(i, a, obs, done)
        '''
        j += 1
        should_end = j >= max_ep_len or ((i == steps_per_epoch-1) and (not done))
        done = done or should_end
        #if epoch % 20 == 0 and i < 500:
        #    env.render()
        # print (a, obs, rew, done)
        obs_buf[i] = obs
        act_buf[i] = a
        rew_buf[i] = rew
        msk_buf[i] = 1 if done else 0
        end_buf[i] = 1 if should_end else 0
        logp_old_buf[i] = logp_pi_buf_[0]

    bufs_dict = {obs_buf_ph:obs_buf, act_buf_ph:act_buf, rew_buf_ph:rew_buf, \
        msk_buf_ph:msk_buf, end_buf_ph:end_buf, logp_old_buf_ph:logp_old_buf}

    # mean_, var_ = sess.run([mean, var], feed_dict=bufs_dict)
    # print (mean_, np.sqrt(var_))

    pi_loss_1, v_loss_1 = sess.run([pi_loss, v_loss], feed_dict=bufs_dict)
    
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
        sess.run(train_pi, feed_dict=bufs_dict)
    for _ in range(train_iters):
        sess.run(train_v, feed_dict=bufs_dict)
        

    pi_loss_2, v_loss_2 = sess.run([pi_loss, v_loss], feed_dict=bufs_dict)


    if epoch % int(epochs / display) == 0:
        num_trajectories = np.sum(msk_buf)
        ep_ret = np.sum(rew_buf) / num_trajectories
        ep_len = steps_per_epoch / num_trajectories
        print ('epoch: ', epoch)
        print ('ep_ret: ', ep_ret)
        print ('ep_len: ', ep_len)
        print ('loss_1: ', (pi_loss_1, v_loss_1))
        print ('loss_2: ', (pi_loss_2, v_loss_2))
        print ('---------------------------------------------------------')