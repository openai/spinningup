#!/usr/bin/env python

import unittest
from functools import partial
import gym
import tensorflow as tf

from spinup import ppo_tf1 as ppo

tf.compat.v1.disable_eager_execution()

class TestPPO(tf.test.TestCase):
    def test_cartpole(self):
        ''' Test training a small agent in a simple environment '''
        env_fn = partial(gym.make, 'CartPole-v1')
        ac_kwargs = dict(hidden_sizes=(32,))
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)):
            ppo(env_fn, steps_per_epoch=100, epochs=10, ac_kwargs=ac_kwargs)

if __name__ == '__main__':
    unittest.main()
