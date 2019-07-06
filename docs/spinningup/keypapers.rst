=====================
Key Papers in Deep RL
=====================

What follows is a list of papers in deep RL that are worth reading. This is *far* from comprehensive, but should provide a useful starting point for someone looking to do research in the field.

.. contents:: Table of Contents
    :depth: 2


1. Model-Free RL
================

a. Deep Q-Learning
------------------


.. [#] `Playing Atari with Deep Reinforcement Learning <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_, Mnih et al, 2013. **Algorithm: DQN.**

.. [#] `Deep Recurrent Q-Learning for Partially Observable MDPs <https://arxiv.org/abs/1507.06527>`_, Hausknecht and Stone, 2015. **Algorithm: Deep Recurrent Q-Learning.**

.. [#] `Dueling Network Architectures for Deep Reinforcement Learning <https://arxiv.org/abs/1511.06581>`_, Wang et al, 2015. **Algorithm: Dueling DQN.**

.. [#] `Deep Reinforcement Learning with Double Q-learning <https://arxiv.org/abs/1509.06461>`_, Hasselt et al 2015. **Algorithm: Double DQN.**

.. [#] `Prioritized Experience Replay <https://arxiv.org/abs/1511.05952>`_, Schaul et al, 2015. **Algorithm: Prioritized Experience Replay (PER).**

.. [#] `Rainbow: Combining Improvements in Deep Reinforcement Learning <https://arxiv.org/abs/1710.02298>`_, Hessel et al, 2017. **Algorithm: Rainbow DQN.**


b. Policy Gradients
-------------------


.. [#] `Asynchronous Methods for Deep Reinforcement Learning <https://arxiv.org/abs/1602.01783>`_, Mnih et al, 2016. **Algorithm: A3C.**

.. [#] `Trust Region Policy Optimization <https://arxiv.org/abs/1502.05477>`_, Schulman et al, 2015. **Algorithm: TRPO.**

.. [#] `High-Dimensional Continuous Control Using Generalized Advantage Estimation <https://arxiv.org/abs/1506.02438>`_, Schulman et al, 2015. **Algorithm: GAE.**

.. [#] `Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_, Schulman et al, 2017. **Algorithm: PPO-Clip, PPO-Penalty.**

.. [#] `Emergence of Locomotion Behaviours in Rich Environments <https://arxiv.org/abs/1707.02286>`_, Heess et al, 2017. **Algorithm: PPO-Penalty.**

.. [#] `Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation <https://arxiv.org/abs/1708.05144>`_, Wu et al, 2017. **Algorithm: ACKTR.**

.. [#] `Sample Efficient Actor-Critic with Experience Replay <https://arxiv.org/abs/1611.01224>`_, Wang et al, 2016. **Algorithm: ACER.**

.. [#] `Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor <https://arxiv.org/abs/1801.01290>`_, Haarnoja et al, 2018. **Algorithm: SAC.**

c. Deterministic Policy Gradients
---------------------------------


.. [#] `Deterministic Policy Gradient Algorithms <http://proceedings.mlr.press/v32/silver14.pdf>`_, Silver et al, 2014. **Algorithm: DPG.**

.. [#] `Continuous Control With Deep Reinforcement Learning <https://arxiv.org/abs/1509.02971>`_, Lillicrap et al, 2015. **Algorithm: DDPG.**

.. [#] `Addressing Function Approximation Error in Actor-Critic Methods <https://arxiv.org/abs/1802.09477>`_, Fujimoto et al, 2018. **Algorithm: TD3.**


d. Distributional RL
--------------------

.. [#] `A Distributional Perspective on Reinforcement Learning <https://arxiv.org/abs/1707.06887>`_, Bellemare et al, 2017. **Algorithm: C51.** 

.. [#] `Distributional Reinforcement Learning with Quantile Regression <https://arxiv.org/abs/1710.10044>`_, Dabney et al, 2017. **Algorithm: QR-DQN.**

.. [#] `Implicit Quantile Networks for Distributional Reinforcement Learning <https://arxiv.org/abs/1806.06923>`_, Dabney et al, 2018. **Algorithm: IQN.**

.. [#] `Dopamine: A Research Framework for Deep Reinforcement Learning <https://openreview.net/forum?id=ByG_3s09KX>`_, Anonymous, 2018. **Contribution:** Introduces Dopamine, a code repository containing implementations of DQN, C51, IQN, and Rainbow. `Code link. <https://github.com/google/dopamine>`_

e. Policy Gradients with Action-Dependent Baselines
---------------------------------------------------

.. [#] `Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic <https://arxiv.org/abs/1611.02247>`_, Gu et al, 2016. **Algorithm: Q-Prop.**

.. [#] `Action-depedent Control Variates for Policy Optimization via Stein's Identity <https://arxiv.org/abs/1710.11198>`_, Liu et al, 2017. **Algorithm: Stein Control Variates.**

.. [#] `The Mirage of Action-Dependent Baselines in Reinforcement Learning <https://arxiv.org/abs/1802.10031>`_, Tucker et al, 2018. **Contribution:** interestingly, critiques and reevaluates claims from earlier papers (including Q-Prop and stein control variates) and finds important methodological errors in them.


f. Path-Consistency Learning
----------------------------

.. [#] `Bridging the Gap Between Value and Policy Based Reinforcement Learning <https://arxiv.org/abs/1702.08892>`_, Nachum et al, 2017. **Algorithm: PCL.**

.. [#] `Trust-PCL: An Off-Policy Trust Region Method for Continuous Control <https://arxiv.org/abs/1707.01891>`_, Nachum et al, 2017. **Algorithm: Trust-PCL.**

g. Other Directions for Combining Policy-Learning and Q-Learning
----------------------------------------------------------------

.. [#] `Combining Policy Gradient and Q-learning <https://arxiv.org/abs/1611.01626>`_, O'Donoghue et al, 2016. **Algorithm: PGQL.**

.. [#] `The Reactor: A Fast and Sample-Efficient Actor-Critic Agent for Reinforcement Learning <https://arxiv.org/abs/1704.04651>`_, Gruslys et al, 2017. **Algorithm: Reactor.**

.. [#] `Interpolated Policy Gradient: Merging On-Policy and Off-Policy Gradient Estimation for Deep Reinforcement Learning <http://papers.nips.cc/paper/6974-interpolated-policy-gradient-merging-on-policy-and-off-policy-gradient-estimation-for-deep-reinforcement-learning>`_, Gu et al, 2017. **Algorithm: IPG.**

.. [#] `Equivalence Between Policy Gradients and Soft Q-Learning <https://arxiv.org/abs/1704.06440>`_, Schulman et al, 2017. **Contribution:** Reveals a theoretical link between these two families of RL algorithms.


h. Evolutionary Algorithms
--------------------------

.. [#] `Evolution Strategies as a Scalable Alternative to Reinforcement Learning <https://arxiv.org/abs/1703.03864>`_, Salimans et al, 2017. **Algorithm: ES.**



2. Exploration
==============

a. Intrinsic Motivation
-----------------------

.. [#] `VIME: Variational Information Maximizing Exploration <https://arxiv.org/abs/1605.09674>`_, Houthooft et al, 2016. **Algorithm: VIME.**

.. [#] `Unifying Count-Based Exploration and Intrinsic Motivation <https://arxiv.org/abs/1606.01868>`_, Bellemare et al, 2016. **Algorithm: CTS-based Pseudocounts.**

.. [#] `Count-Based Exploration with Neural Density Models <https://arxiv.org/abs/1703.01310>`_, Ostrovski et al, 2017. **Algorithm: PixelCNN-based Pseudocounts.**

.. [#] `#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning <https://arxiv.org/abs/1611.04717>`_, Tang et al, 2016. **Algorithm: Hash-based Counts.**

.. [#] `EX2: Exploration with Exemplar Models for Deep Reinforcement Learning <https://arxiv.org/abs/1703.01260>`_, Fu et al, 2017. **Algorithm: EX2.**

.. [#] `Curiosity-driven Exploration by Self-supervised Prediction <https://arxiv.org/abs/1705.05363>`_, Pathak et al, 2017. **Algorithm: Intrinsic Curiosity Module (ICM).**

.. [#] `Large-Scale Study of Curiosity-Driven Learning <https://arxiv.org/abs/1808.04355>`_, Burda et al, 2018. **Contribution:** Systematic analysis of how surprisal-based intrinsic motivation performs in a wide variety of environments.

.. [#] `Exploration by Random Network Distillation <https://arxiv.org/abs/1810.12894>`_, Burda et al, 2018. **Algorithm: RND.**


b. Unsupervised RL
------------------

.. [#] `Variational Intrinsic Control <https://arxiv.org/abs/1611.07507>`_, Gregor et al, 2016. **Algorithm: VIC.**

.. [#] `Diversity is All You Need: Learning Skills without a Reward Function <https://arxiv.org/abs/1802.06070>`_, Eysenbach et al, 2018. **Algorithm: DIAYN.**

.. [#] `Variational Option Discovery Algorithms <https://arxiv.org/abs/1807.10299>`_, Achiam et al, 2018. **Algorithm: VALOR.**


3. Transfer and Multitask RL
============================

.. [#] `Progressive Neural Networks <https://arxiv.org/abs/1606.04671>`_, Rusu et al, 2016. **Algorithm: Progressive Networks.**

.. [#] `Universal Value Function Approximators <http://proceedings.mlr.press/v37/schaul15.pdf>`_, Schaul et al, 2015. **Algorithm: UVFA.**

.. [#] `Reinforcement Learning with Unsupervised Auxiliary Tasks <https://arxiv.org/abs/1611.05397>`_, Jaderberg et al, 2016. **Algorithm: UNREAL.**

.. [#] `The Intentional Unintentional Agent: Learning to Solve Many Continuous Control Tasks Simultaneously <https://arxiv.org/abs/1707.03300>`_, Cabi et al, 2017. **Algorithm: IU Agent.**

.. [#] `PathNet: Evolution Channels Gradient Descent in Super Neural Networks <https://arxiv.org/abs/1701.08734>`_, Fernando et al, 2017. **Algorithm: PathNet.**

.. [#] `Mutual Alignment Transfer Learning <https://arxiv.org/abs/1707.07907>`_, Wulfmeier et al, 2017. **Algorithm: MATL.**

.. [#] `Learning an Embedding Space for Transferable Robot Skills <https://openreview.net/forum?id=rk07ZXZRb&noteId=rk07ZXZRb>`_, Hausman et al, 2018. 

.. [#] `Hindsight Experience Replay <https://arxiv.org/abs/1707.01495>`_, Andrychowicz et al, 2017. **Algorithm: Hindsight Experience Replay (HER).**

4. Hierarchy
============

.. [#] `Strategic Attentive Writer for Learning Macro-Actions <https://arxiv.org/abs/1606.04695>`_, Vezhnevets et al, 2016. **Algorithm: STRAW.**

.. [#] `FeUdal Networks for Hierarchical Reinforcement Learning <https://arxiv.org/abs/1703.01161>`_, Vezhnevets et al, 2017. **Algorithm: Feudal Networks**

.. [#] `Data-Efficient Hierarchical Reinforcement Learning <https://arxiv.org/abs/1805.08296>`_, Nachum et al, 2018. **Algorithm: HIRO.**

5. Memory
=========

.. [#] `Model-Free Episodic Control <https://arxiv.org/abs/1606.04460>`_, Blundell et al, 2016. **Algorithm: MFEC.**


.. [#] `Neural Episodic Control <https://arxiv.org/abs/1703.01988>`_, Pritzel et al, 2017. **Algorithm: NEC.**

.. [#] `Neural Map: Structured Memory for Deep Reinforcement Learning <https://arxiv.org/abs/1702.08360>`_, Parisotto and Salakhutdinov, 2017. **Algorithm: Neural Map.**

.. [#] `Unsupervised Predictive Memory in a Goal-Directed Agent <https://arxiv.org/abs/1803.10760>`_, Wayne et al, 2018. **Algorithm: MERLIN.**

.. [#] `Relational Recurrent Neural Networks <https://arxiv.org/abs/1806.01822>`_, Santoro et al, 2018. **Algorithm: RMC.**

6. Model-Based RL
=================

a. Model is Learned
-------------------

.. [#] `Imagination-Augmented Agents for Deep Reinforcement Learning <https://arxiv.org/abs/1707.06203>`_, Weber et al, 2017. **Algorithm: I2A.**

.. [#] `Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning <https://arxiv.org/abs/1708.02596>`_, Nagabandi et al, 2017. **Algorithm: MBMF.**

.. [#] `Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning <https://arxiv.org/abs/1803.00101>`_, Feinberg et al, 2018. **Algorithm: MVE.**

.. [#] `Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion <https://arxiv.org/abs/1807.01675>`_, Buckman et al, 2018. **Algorithm: STEVE.**

.. [#] `Model-Ensemble Trust-Region Policy Optimization <https://openreview.net/forum?id=SJJinbWRZ&noteId=SJJinbWRZ>`_, Kurutach et al, 2018. **Algorithm: ME-TRPO.**

.. [#] `Model-Based Reinforcement Learning via Meta-Policy Optimization <https://arxiv.org/abs/1809.05214>`_, Clavera et al, 2018. **Algorithm: MB-MPO.**

.. [#] `Recurrent World Models Facilitate Policy Evolution <https://arxiv.org/abs/1809.01999>`_, Ha and Schmidhuber, 2018. 

b. Model is Given
-----------------

.. [#] `Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm <https://arxiv.org/abs/1712.01815>`_, Silver et al, 2017. **Algorithm: AlphaZero.**

.. [#] `Thinking Fast and Slow with Deep Learning and Tree Search <https://arxiv.org/abs/1705.08439>`_, Anthony et al, 2017. **Algorithm: ExIt.**

7. Meta-RL
==========

.. [#] `RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning <https://arxiv.org/abs/1611.02779>`_, Duan et al, 2016. **Algorithm: RL^2.**

.. [#] `Learning to Reinforcement Learn <https://arxiv.org/abs/1611.05763>`_, Wang et al, 2016. 

.. [#] `Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks <https://arxiv.org/abs/1703.03400>`_, Finn et al, 2017. **Algorithm: MAML.**

.. [#] `A Simple Neural Attentive Meta-Learner <https://openreview.net/forum?id=B1DmUzWAW&noteId=B1DmUzWAW>`_, Mishra et al, 2018. **Algorithm: SNAIL.**

8. Scaling RL
=============

.. [#] `Accelerated Methods for Deep Reinforcement Learning <https://arxiv.org/abs/1803.02811>`_, Stooke and Abbeel, 2018. **Contribution:** Systematic analysis of parallelization in deep RL across algorithms. 

.. [#] `IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures <https://arxiv.org/abs/1802.01561>`_, Espeholt et al, 2018. **Algorithm: IMPALA.**

.. [#] `Distributed Prioritized Experience Replay <https://openreview.net/forum?id=H1Dy---0Z>`_, Horgan et al, 2018. **Algorithm: Ape-X.**

.. [#] `Recurrent Experience Replay in Distributed Reinforcement Learning <https://openreview.net/forum?id=r1lyTjAqYX>`_, Anonymous, 2018. **Algorithm: R2D2.**

.. [#] `RLlib: Abstractions for Distributed Reinforcement Learning <https://arxiv.org/abs/1712.09381>`_, Liang et al, 2017. **Contribution:** A scalable library of RL algorithm implementations. `Documentation link. <https://ray.readthedocs.io/en/latest/rllib.html>`_


9. RL in the Real World
=======================

.. [#] `Benchmarking Reinforcement Learning Algorithms on Real-World Robots <https://arxiv.org/abs/1809.07731>`_, Mahmood et al, 2018. 

.. [#] `Learning Dexterous In-Hand Manipulation <https://arxiv.org/abs/1808.00177>`_, OpenAI, 2018. 

.. [#] `QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation <https://arxiv.org/abs/1806.10293>`_, Kalashnikov et al, 2018. **Algorithm: QT-Opt.**

.. [#] `Horizon: Facebook's Open Source Applied Reinforcement Learning Platform <https://arxiv.org/abs/1811.00260>`_, Gauci et al, 2018. 


10. Safety
==========

.. [#] `Concrete Problems in AI Safety <https://arxiv.org/abs/1606.06565>`_, Amodei et al, 2016. **Contribution:** establishes a taxonomy of safety problems, serving as an important jumping-off point for future research. We need to solve these!

.. [#] `Deep Reinforcement Learning From Human Preferences <https://arxiv.org/abs/1706.03741>`_, Christiano et al, 2017. **Algorithm: LFP.**

.. [#] `Constrained Policy Optimization <https://arxiv.org/abs/1705.10528>`_, Achiam et al, 2017. **Algorithm: CPO.**

.. [#] `Safe Exploration in Continuous Action Spaces <https://arxiv.org/abs/1801.08757>`_, Dalal et al, 2018. **Algorithm: DDPG+Safety Layer.**

.. [#] `Trial without Error: Towards Safe Reinforcement Learning via Human Intervention <https://arxiv.org/abs/1707.05173>`_, Saunders et al, 2017. **Algorithm: HIRL.**

.. [#] `Leave No Trace: Learning to Reset for Safe and Autonomous Reinforcement Learning <https://arxiv.org/abs/1711.06782>`_, Eysenbach et al, 2017. **Algorithm: Leave No Trace.**


11. Imitation Learning and Inverse Reinforcement Learning
=========================================================

.. [#] `Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy <http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf>`_, Ziebart 2010. **Contributions:** Crisp formulation of maximum entropy IRL.

.. [#] `Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization <https://arxiv.org/abs/1603.00448>`_, Finn et al, 2016. **Algorithm: GCL.**

.. [#] `Generative Adversarial Imitation Learning <https://arxiv.org/abs/1606.03476>`_, Ho and Ermon, 2016. **Algorithm: GAIL.**

.. [#] `DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills <https://xbpeng.github.io/projects/DeepMimic/2018_TOG_DeepMimic.pdf>`_, Peng et al, 2018. **Algorithm: DeepMimic.**

.. [#] `Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow <https://arxiv.org/abs/1810.00821>`_, Peng et al, 2018. **Algorithm: VAIL.**

.. [#] `One-Shot High-Fidelity Imitation: Training Large-Scale Deep Nets with RL <https://arxiv.org/abs/1810.05017>`_, Le Paine et al, 2018. **Algorithm: MetaMimic.**


12. Reproducibility, Analysis, and Critique
===========================================

.. [#] `Benchmarking Deep Reinforcement Learning for Continuous Control <https://arxiv.org/abs/1604.06778>`_, Duan et al, 2016. **Contribution: rllab.**

.. [#] `Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control <https://arxiv.org/abs/1708.04133>`_, Islam et al, 2017.

.. [#] `Deep Reinforcement Learning that Matters <https://arxiv.org/abs/1709.06560>`_, Henderson et al, 2017. 

.. [#] `Where Did My Optimum Go?: An Empirical Analysis of Gradient Descent Optimization in Policy Gradient Methods <https://arxiv.org/abs/1810.02525>`_, Henderson et al, 2018. 

.. [#] `Are Deep Policy Gradient Algorithms Truly Policy Gradient Algorithms? <https://arxiv.org/abs/1811.02553>`_, Ilyas et al, 2018.

.. [#] `Simple Random Search Provides a Competitive Approach to Reinforcement Learning <https://arxiv.org/abs/1803.07055>`_, Mania et al, 2018.

.. [#] `Benchmarking Model-Based Reinforcement Learning <https://arxiv.org/abs/1907.02057>`_, Wang et al, 2019.

13. Bonus: Classic Papers in RL Theory or Review
================================================

.. [#] `Policy Gradient Methods for Reinforcement Learning with Function Approximation <https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf>`_, Sutton et al, 2000. **Contributions:** Established policy gradient theorem and showed convergence of policy gradient algorithm for arbitrary policy classes. 

.. [#] `An Analysis of Temporal-Difference Learning with Function Approximation <http://web.mit.edu/jnt/www/Papers/J063-97-bvr-td.pdf>`_, Tsitsiklis and Van Roy, 1997. **Contributions:** Variety of convergence results and counter-examples for value-learning methods in RL.

.. [#] `Reinforcement Learning of Motor Skills with Policy Gradients <http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Neural-Netw-2008-21-682_4867%5b0%5d.pdf>`_, Peters and Schaal, 2008. **Contributions:** Thorough review of policy gradient methods at the time, many of which are still serviceable descriptions of deep RL methods. 

.. [#] `Approximately Optimal Approximate Reinforcement Learning <https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf>`_, Kakade and Langford, 2002. **Contributions:** Early roots for monotonic improvement theory, later leading to theoretical justification for TRPO and other algorithms.

.. [#] `A Natural Policy Gradient <https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf>`_, Kakade, 2002. **Contributions:** Brought natural gradients into RL, later leading to TRPO, ACKTR, and several other methods in deep RL.

.. [#] `Algorithms for Reinforcement Learning <https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf>`_, Szepesvari, 2009. **Contributions:** Unbeatable reference on RL before deep RL, containing foundations and theoretical background.
