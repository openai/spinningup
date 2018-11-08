=================
Soft Actor-Critic
=================

.. contents:: Table of Contents

Background
==========

(Previously: `Background for TD3`_)

.. _`Background for TD3`: ../algorithms/td3.html#background

Soft Actor Critic (SAC) is an algorithm which optimizes a stochastic policy in an off-policy way, forming a bridge between stochastic policy optimization and DDPG-style approaches. It isn't a direct successor to TD3 (having been published roughly concurrently), but it incorporates the clipped double-Q trick, and due to the inherent stochasticity of the policy in SAC, it also winds up benefiting from something like target policy smoothing. 

A central feature of SAC is **entropy regularization.** The policy is trained to maximize a trade-off between expected return and `entropy`_, a measure of randomness in the policy. This has a close connection to the exploration-exploitation trade-off: increasing entropy results in more exploration, which can accelerate learning later on. It can also prevent the policy from prematurely converging to a bad local optimum. 

.. _`entropy`: https://en.wikipedia.org/wiki/Entropy_(information_theory)

Quick Facts
-----------

* SAC is an off-policy algorithm.
* The version of SAC implemented here can only be used for environments with continuous action spaces.
* An alternate version of SAC, which slightly changes the policy update rule, can be implemented to handle discrete action spaces.
* The Spinning Up implementation of SAC does not support parallelization.

Key Equations
-------------

To explain Soft Actor Critic, we first have to introduce the entropy-regularized reinforcement learning setting. In entropy-regularized RL, there are slightly-different equations for value functions. 

Entropy-Regularized Reinforcement Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Entropy is a quantity which, roughly speaking, says how random a random variable is. If a coin is weighted so that it almost always comes up heads, it has low entropy; if it's evenly weighted and has a half chance of either outcome, it has high entropy. 

Let :math:`x` be a random variable with probability mass or density function :math:`P`. The entropy :math:`H` of :math:`x` is computed from its distribution :math:`P` according to

.. math::

    H(P) = \underE{x \sim P}{-\log P(x)}.

In entropy-regularized reinforcement learning, the agent gets a bonus reward at each time step proportional to the entropy of the policy at that timestep. This changes `the RL problem`_ to:

.. math::

    \pi^* = \arg \max_{\pi} \underE{\tau \sim \pi}{ \sum_{t=0}^{\infty} \gamma^t \bigg( R(s_t, a_t, s_{t+1}) + \alpha H\left(\pi(\cdot|s_t)\right) \bigg)},

where :math:`\alpha > 0` is the trade-off coefficient. (Note: we're assuming an infinite-horizon discounted setting here, and we'll do the same for the rest of this page.) We can now define the slightly-different value functions in this setting. :math:`V^{\pi}` is changed to include the entropy bonuses from every timestep:

.. math::

    V^{\pi}(s) = \underE{\tau \sim \pi}{ \left. \sum_{t=0}^{\infty} \gamma^t \bigg( R(s_t, a_t, s_{t+1}) + \alpha H\left(\pi(\cdot|s_t)\right) \bigg) \right| s_0 = s}

:math:`Q^{\pi}` is changed to include the entropy bonuses from every timestep *except the first*:

.. math::

    Q^{\pi}(s,a) = \underE{\tau \sim \pi}{ \left. \sum_{t=0}^{\infty} \gamma^t  R(s_t, a_t, s_{t+1}) + \alpha \sum_{t=1}^{\infty} \gamma^t H\left(\pi(\cdot|s_t)\right)\right| s_0 = s, a_0 = a}

With these definitions, :math:`V^{\pi}` and :math:`Q^{\pi}` are connected by:

.. math::

    V^{\pi}(s) = \underE{a \sim \pi}{Q^{\pi}(s,a)} + \alpha H\left(\pi(\cdot|s)\right)

and the Bellman equation for :math:`Q^{\pi}` is

.. math::

    Q^{\pi}(s,a) &= \underE{s' \sim P \\ a' \sim \pi}{R(s,a,s') + \gamma\left(Q^{\pi}(s',a') + \alpha H\left(\pi(\cdot|s')\right) \right)} \\
    &= \underE{s' \sim P}{R(s,a,s') + \gamma V^{\pi}(s')}.

.. _`the RL problem`: ../spinningup/rl_intro.html#the-rl-problem

.. admonition:: You Should Know

    The way we've set up the value functions in the entropy-regularized setting is a little bit arbitrary, and actually we could have done it differently (eg make :math:`Q^{\pi}` include the entropy bonus at the first timestep). The choice of definition may vary slightly across papers on the subject.


Soft Actor-Critic
^^^^^^^^^^^^^^^^^

SAC concurrently learns a policy :math:`\pi_{\theta}`, two Q-functions :math:`Q_{\phi_1}, Q_{\phi_2}`, and a value function :math:`V_{\psi}`. 

**Learning Q.** The Q-functions are learned by MSBE minimization, using a **target value network** to form the Bellman backups. They both use the same target, like in TD3, and have loss functions:

.. math::

    L(\phi_i, {\mathcal D}) = \underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[
        \Bigg( Q_{\phi_i}(s,a) - \left(r + \gamma (1 - d) V_{\psi_{\text{targ}}}(s') \right) \Bigg)^2
        \right].

The target value network, like the target networks in DDPG and TD3, is obtained by polyak averaging the value network parameters over the course of training.

**Learning V.** The value function is learned by exploiting (a sample-based approximation of) the connection between :math:`Q^{\pi}` and :math:`V^{\pi}`. Before we go into the learning rule, let's first rewrite the connection equation by using the definition of entropy to obtain:

.. math::

    V^{\pi}(s) &= \underE{a \sim \pi}{Q^{\pi}(s,a)} + \alpha H\left(\pi(\cdot|s)\right) \\
    &= \underE{a \sim \pi}{Q^{\pi}(s,a) - \alpha \log \pi(a|s)}.

The RHS is an expectation over actions, so we can approximate it by sampling from the policy:

.. math::

    V^{\pi}(s) \approx Q^{\pi}(s,\tilde{a}) - \alpha \log \pi(\tilde{a}|s), \;\;\;\;\; \tilde{a} \sim \pi(\cdot|s).

SAC sets up a mean-squared-error loss for :math:`V_{\psi}` based on this approximation. But what Q-value do we use? SAC uses **clipped double-Q** like TD3 for learning the value function, and takes the minimum Q-value between the two approximators. So the SAC loss for value function parameters is:

.. math::

    L(\psi, {\mathcal D}) = \underE{s \sim \mathcal{D} \\ \tilde{a} \sim \pi_{\theta}}{\Bigg(V_{\psi}(s) - \left(\min_{i=1,2} Q_{\phi_i}(s,\tilde{a}) - \alpha \log \pi_{\theta}(\tilde{a}|s) \right)\Bigg)^2}.

Importantly, we do **not** use actions from the replay buffer here: these actions are sampled fresh from the current version of the policy. 

**Learning the Policy.** The policy should, in each state, act to maximize the expected future return plus expected future entropy. That is, it should maximize :math:`V^{\pi}(s)`, which we expand out (as before) into

.. math::
    
     \underE{a \sim \pi}{Q^{\pi}(s,a) - \alpha \log \pi(a|s)}.

The way we optimize the policy makes use of the **reparameterization trick**, in which a sample from :math:`\pi_{\theta}(\cdot|s)` is drawn by computing a deterministic function of state, policy parameters, and independent noise. To illustrate: following the authors of the SAC paper, we use a squashed Gaussian policy, which means that samples are obtained according to

.. math::

    \tilde{a}_{\theta}(s, \xi) = \tanh\left( \mu_{\theta}(s) + \sigma_{\theta}(s) \odot \xi \right), \;\;\;\;\; \xi \sim \mathcal{N}(0, I).

.. admonition:: You Should Know

    This policy has two key differences from the policies we use in the other policy optimization algorithms:

    **1. The squashing function.** The :math:`\tanh` in the SAC policy ensures that actions are bounded to a finite range. This is absent in the VPG, TRPO, and PPO policies. It also changes the distribution: before the :math:`\tanh` the SAC policy is a factored Gaussian like the other algorithms' policies, but after the :math:`\tanh` it is not. (You can still compute the log-probabilities of actions in closed form, though: see the paper appendix for details.)

    **2. The way standard deviations are parameterized.** In VPG, TRPO, and PPO, we represent the log std devs with state-independent parameter vectors. In SAC, we represent the log std devs as outputs from the neural network, meaning that they depend on state in a complex way. SAC with state-independent log std devs, in our experience, did not work. (Can you think of why? Or better yet: run an experiment to verify?)

The reparameterization trick allows us to rewrite the expectation over actions (which contains a pain point: the distribution depends on the policy parameters) into an expectation over noise (which removes the pain point: the distribution now has no dependence on parameters):

.. math::

    \underE{a \sim \pi_{\theta}}{Q^{\pi_{\theta}}(s,a) - \alpha \log \pi_{\theta}(a|s)} = \underE{\xi \sim \mathcal{N}}{Q^{\pi_{\theta}}(s,\tilde{a}_{\theta}(s,\xi)) - \alpha \log \pi_{\theta}(\tilde{a}_{\theta}(s,\xi)|s)}

To get the policy loss, the final step is that we need to substitute :math:`Q^{\pi_{\theta}}` with one of our function approximators. The same as in TD3, we use :math:`Q_{\phi_1}`. The policy is thus optimized according to

.. math::

    \max_{\theta} \underE{s \sim \mathcal{D} \\ \xi \sim \mathcal{N}}{Q_{\phi_1}(s,\tilde{a}_{\theta}(s,\xi)) - \alpha \log \pi_{\theta}(\tilde{a}_{\theta}(s,\xi)|s)},

which is almost the same as the DDPG and TD3 policy optimization, except for the stochasticity and entropy term.


Exploration vs. Exploitation
----------------------------

SAC trains a stochastic policy with entropy regularization, and explores in an on-policy way. The entropy regularization coefficient :math:`\alpha` explicitly controls the explore-exploit tradeoff, with higher :math:`\alpha` corresponding to more exploration, and lower :math:`\alpha` corresponding to more exploitation. The right coefficient (the one which leads to the stablest / highest-reward learning) may vary from environment to environment, and could require careful tuning.

At test time, to see how well the policy exploits what it has learned, we remove stochasticity and use the mean action instead of a sample from the distribution. This tends to improve performance over the original stochastic policy.

.. admonition:: You Should Know

    Our SAC implementation uses a trick to improve exploration at the start of training. For a fixed number of steps at the beginning (set with the ``start_steps`` keyword argument), the agent takes actions which are sampled from a uniform random distribution over valid actions. After that, it returns to normal SAC exploration.


Pseudocode
----------


.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{Soft Actor-Critic}
        \label{alg1}
    \begin{algorithmic}[1]
        \STATE Input: initial policy parameters $\theta$, Q-function parameters $\phi_1$, $\phi_2$, V-function parameters $\psi$, empty replay buffer $\mathcal{D}$
        \STATE Set target parameters equal to main parameters $\psi_{\text{targ}} \leftarrow \psi$
        \REPEAT
            \STATE Observe state $s$ and select action $a \sim \pi_{\theta}(\cdot|s)$
            \STATE Execute $a$ in the environment
            \STATE Observe next state $s'$, reward $r$, and done signal $d$ to indicate whether $s'$ is terminal
            \STATE Store $(s,a,r,s',d)$ in replay buffer $\mathcal{D}$
            \STATE If $s'$ is terminal, reset environment state.
            \IF{it's time to update}
                \FOR{$j$ in range(however many updates)}
                    \STATE Randomly sample a batch of transitions, $B = \{ (s,a,r,s',d) \}$ from $\mathcal{D}$
                    \STATE Compute targets for Q and V functions:
                    \begin{align*}
                        y_q (r,s',d) &= r + \gamma (1-d) V_{\psi_{\text{targ}}}(s') &&\\
                        y_v (s) &= \min_{i=1,2} Q_{\phi_i} (s, \tilde{a}) - \alpha \log \pi_{\theta}(\tilde{a}|s), && \tilde{a} \sim \pi_{\theta}(\cdot|s)
                    \end{align*}
                    \STATE Update Q-functions by one step of gradient descent using
                    \begin{align*}
                        & \nabla_{\phi_i} \frac{1}{|B|}\sum_{(s,a,r,s',d) \in B} \left( Q_{\phi,i}(s,a) - y_q(r,s',d) \right)^2 && \text{for } i=1,2
                    \end{align*}
                    \STATE Update V-function by one step of gradient descent using
                    \begin{equation*}
                        \nabla_{\psi} \frac{1}{|B|}\sum_{s \in B} \left( V_{\psi}(s) - y_v(s) \right)^2 
                    \end{equation*}
                    \STATE Update policy by one step of gradient ascent using
                    \begin{equation*}
                        \nabla_{\theta} \frac{1}{|B|}\sum_{s \in B} \Big( Q_{\phi,1}(s, \tilde{a}_{\theta}(s)) - \alpha \log \pi_{\theta} \left(\left. \tilde{a}_{\theta}(s) \right| s\right) \Big),
                    \end{equation*}
                    where $\tilde{a}_{\theta}(s)$ is a sample from $\pi_{\theta}(\cdot|s)$ which is differentiable wrt $\theta$ via the reparametrization trick.
                    \STATE Update target value network with
                    \begin{align*}
                        \psi_{\text{targ}} &\leftarrow \rho \psi_{\text{targ}} + (1-\rho) \psi 
                    \end{align*}
                \ENDFOR
            \ENDIF
        \UNTIL{convergence}
    \end{algorithmic}
    \end{algorithm}

Documentation
=============

.. autofunction:: spinup.sac

Saved Model Contents
--------------------

The computation graph saved by the logger includes:

========  ====================================================================
Key       Value
========  ====================================================================
``x``     Tensorflow placeholder for state input.
``a``     Tensorflow placeholder for action input.
``mu``    Deterministically computes mean action from the agent, given states in ``x``. 
``pi``    Samples an action from the agent, conditioned on states in ``x``.
``q1``    Gives one action-value estimate for states in ``x`` and actions in ``a``.
``q2``    Gives the other action-value estimate for states in ``x`` and actions in ``a``.
``v``     Gives the value estimate for states in ``x``. 
========  ====================================================================

This saved model can be accessed either by

* running the trained policy with the `test_policy.py`_ tool,
* or loading the whole saved graph into a program with `restore_tf_graph`_. 

Note: for SAC, the correct evaluation policy is given by ``mu`` and not by ``pi``. The policy ``pi`` may be thought of as the exploration policy, while ``mu`` is the exploitation policy.

.. _`test_policy.py`: ../user/saving_and_loading.html#loading-and-running-trained-policies
.. _`restore_tf_graph`: ../utils/logger.html#spinup.utils.logx.restore_tf_graph


References
==========

Relevant Papers
---------------

- `Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor`_, Haarnoja et al, 2018

.. _`Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor`: https://arxiv.org/abs/1801.01290


Other Public Implementations
----------------------------

- `SAC release repo`_

.. _`SAC release repo`: https://github.com/haarnoja/sac