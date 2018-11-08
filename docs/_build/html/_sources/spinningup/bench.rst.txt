==========================================
Benchmarks for Spinning Up Implementations
==========================================

.. contents:: Table of Contents

We benchmarked the Spinning Up algorithm implementations in five environments from the MuJoCo_ Gym task suite: HalfCheetah, Hopper, Walker2d, Swimmer, and Ant.

.. _MuJoCo: https://gym.openai.com/envs/#mujoco

Performance in Each Environment
===============================

HalfCheetah
-----------

.. figure:: ../images/bench/bench_halfcheetah.svg
    :align: center

    3M timestep benchmark for HalfCheetah-2.


Hopper
------

.. figure:: ../images/bench/bench_hopper.svg
    :align: center

    3M timestep benchmark for Hopper-v2.

Walker
------

.. figure:: ../images/bench/bench_walker.svg
    :align: center

    3M timestep benchmark for Walker2d-v2.

Swimmer
-------
.. figure:: ../images/bench/bench_swim.svg
    :align: center

    3M timestep benchmark for Swimmer-v2.

Ant
---
.. figure:: ../images/bench/bench_ant.svg
    :align: center

    3M timestep benchmark for Ant-v2.

Experiment Details
==================

**Random seeds.** The on-policy algorithms (VPG, TPRO, PPO) were run for 3 random seeds each, and the off-policy algorithms (DDPG, TD3, SAC) were run for 10 random seeds each. Graphs show the average (solid line) and std dev (shaded) of performance over random seed over the course of training.

**Performance metric.** Performance for the on-policy algorithms is measured as the average trajectory return across the batch collected at each epoch. Performance for the off-policy algorithms is measured once every 10,000 steps by running the deterministic policy (or, in the case of SAC, the mean policy) without action noise for ten trajectories, and reporting the average return over those test trajectories.

**Network architectures.** The on-policy algorithms use networks of size (64, 32) with tanh units for both the policy and the value function. The off-policy algorithms use networks of size (400, 300) with relu units.

**Batch size.** The on-policy algorithms collected 4000 steps of agent-environment interaction per batch update. The off-policy algorithms used minibatches of size 100 at each gradient descent step.

All other hyperparameters are left at default settings for the Spinning Up implementations. See algorithm pages for details.