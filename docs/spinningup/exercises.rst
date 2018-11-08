=========
Exercises
=========


.. contents:: Table of Contents
    :depth: 2

Problem Set 1: Basics of Implementation
---------------------------------------

.. admonition:: Exercise 1.1: Gaussian Log-Likelihood

    **Path to Exercise.** ``spinup/exercises/problem_set_1/exercise1_1.py``

    **Path to Solution.** ``spinup/exercises/problem_set_1_solutions/exercise1_1_soln.py``

    **Instructions.** Write a function which takes in Tensorflow symbols for the means and log stds of a batch of diagonal Gaussian distributions, along with a Tensorflow placeholder for (previously-generated) samples from those distributions, and returns a Tensorflow symbol for computing the log likelihoods of those samples.

    You may find it useful to review the formula given in `this section of the RL introduction`_.

    Implement your solution in ``exercise1_1.py``, and run that file to automatically check your work.

    **Evaluation Criteria.** Your solution will be checked by comparing outputs against a known-good implementation, using a batch of random inputs.

.. _`this section of the RL introduction`: ../spinningup/rl_intro.html#stochastic-policies


.. admonition:: Exercise 1.2: Policy for PPO

    **Path to Exercise.** ``spinup/exercises/problem_set_1/exercise1_2.py``

    **Path to Solution.** ``spinup/exercises/problem_set_1_solutions/exercise1_2_soln.py``

    **Instructions.** Implement an MLP diagonal Gaussian policy for PPO. 

    Implement your solution in ``exercise1_2.py``, and run that file to automatically check your work. 

    **Evaluation Criteria.** Your solution will be evaluated by running for 20 epochs in the InvertedPendulum-v2 Gym environment, and this should take in the ballpark of 3-5 minutes (depending on your machine, and other processes you are running in the background). The bar for success is reaching an average score of over 500 in the last 5 epochs, or getting to a score of 1000 (the maximum possible score) in the last 5 epochs.


.. admonition:: Exercise 1.3: Computation Graph for TD3

    **Path to Exercise.** ``spinup/exercises/problem_set_1/exercise1_3.py``

    **Path to Solution.** ``spinup/algos/td3/td3.py``

    **Instructions.** Implement the core computation graph for the TD3 algorithm.

    As starter code, you are given the entirety of the TD3 algorithm except for the computation graph. Find "YOUR CODE HERE" to begin. 

    You may find it useful to review the pseudocode in our `page on TD3`_.

    Implement your solution in ``exercise1_3.py``, and run that file to see the results of your work. There is no automatic checking for this exercise.

    **Evaluation Criteria.** Evaluate your code by running ``exercise1_3.py`` with HalfCheetah-v2, InvertedPendulum-v2, and one other Gym MuJoCo environment of your choosing (set via the ``--env`` flag). It is set up to use smaller neural networks (hidden sizes [128,128]) than typical for TD3, with a maximum episode length of 150, and to run for only 10 epochs. The goal is to see significant learning progress relatively quickly (in terms of wall clock time). Experiments will likely take on the order of ~10 minutes. 

    Use the ``--use_soln`` flag to run Spinning Up's TD3 instead of your implementation. Anecdotally, within 10 epochs, the score in HalfCheetah should go over 300, and the score in InvertedPendulum should max out at 150.

.. _`page on TD3`: ../algorithms/td3.html


Problem Set 2: Algorithm Failure Modes
--------------------------------------

.. admonition:: Exercise 2.1: Value Function Fitting in TRPO

    **Path to Exercise.** (Not applicable, there is no code for this one.)

    **Path to Solution.** `Solution available here. <../spinningup/exercise2_1_soln.html>`_

    Many factors can impact the performance of policy gradient algorithms, but few more drastically than the quality of the learned value function used for advantage estimation. 

    In this exercise, you will compare results between runs of TRPO where you put lots of effort into fitting the value function (``train_v_iters=80``), versus where you put very little effort into fitting the value function (``train_v_iters=0``). 

    **Instructions.** Run the following command:

    .. parsed-literal::

        python -m spinup.run trpo --env Hopper-v2 --train_v_iters[v] 0 80 --exp_name ex2-1 --epochs 250 --steps_per_epoch 4000 --seed 0 10 20 --dt

    and plot the results. (These experiments might take ~10 minutes each, and this command runs six of them.) What do you find?

.. admonition:: Exercise 2.2: Silent Bug in DDPG

    **Path to Exercise.** ``spinup/exercises/problem_set_2/exercise2_2.py``

    **Path to Solution.** `Solution available here. <../spinningup/exercise2_2_soln.html>`_

    The hardest part of writing RL code is dealing with bugs, because failures are frequently silent. The code will appear to run correctly, but the agent's performance will degrade relative to a bug-free implementation---sometimes to the extent that it never learns anything.

    In this exercise, you will observe a bug in vivo and compare results against correct code.

    **Instructions.** Run ``exercise2_2.py``, which will launch DDPG experiments with and without a bug. The non-bugged version runs the default Spinning Up implementation of DDPG, using a default method for creating the actor and critic networks. The bugged version runs the same DDPG code, except uses a bugged method for creating the networks.

    There will be six experiments in all (three random seeds for each case), and each should take in the ballpark of 10 minutes. When they're finished, plot the results. What is the difference in performance with and without the bug? 

    Without referencing the correct actor-critic code (which is to say---don't look in DDPG's ``core.py`` file), try to figure out what the bug is and explain how it breaks things.

    **Hint.** To figure out what's going wrong, think about how the DDPG code implements the DDPG computation graph. Specifically, look at this excerpt:

    .. code-block:: python

        # Bellman backup for Q function
        backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*q_pi_targ)

        # DDPG losses
        pi_loss = -tf.reduce_mean(q_pi)
        q_loss = tf.reduce_mean((q-backup)**2)

    How could a bug in the actor-critic code have an impact here?

    **Bonus.** Are there any choices of hyperparameters which would have hidden the effects of the bug? 


Challenges
----------

.. admonition:: Write Code from Scratch

    As we suggest in `the essay <../spinningup/spinningup.html#learn-by-doing>`_, try reimplementing various deep RL algorithms from scratch. 

.. admonition:: Requests for Research

    If you feel comfortable with writing deep learning and deep RL code, consider trying to make progress on any of OpenAI's standing requests for research:

    * `Requests for Research 1 <https://openai.com/requests-for-research/>`_
    * `Requests for Research 2 <https://blog.openai.com/requests-for-research-2/>`_