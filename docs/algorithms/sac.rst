=================
Soft Actor-Critic
=================

.. contents:: Table of Contents

Background
==========

(Previously: `Background for TD3`_)

.. _`Background for TD3`: ../algorithms/td3.html#background

Soft Actor Critic (SAC) is an algorithm that optimizes a stochastic policy in an off-policy way, forming a bridge between stochastic policy optimization and DDPG-style approaches. It isn't a direct successor to TD3 (having been published roughly concurrently), but it incorporates the clipped double-Q trick, and due to the inherent stochasticity of the policy in SAC, it also winds up benefiting from something like target policy smoothing. 

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

SAC concurrently learns a policy :math:`\pi_{\theta}` and two Q-functions :math:`Q_{\phi_1}, Q_{\phi_2}`. There are two variants of SAC that are currently standard: one that uses a fixed entropy regularization coefficient :math:`\alpha`, and another that enforces an entropy constraint by varying :math:`\alpha` over the course of training. For simplicity, Spinning Up makes use of the version with a fixed entropy regularization coefficient, but the entropy-constrained variant is generally preferred by practitioners.

.. admonition:: You Should Know

    The SAC algorithm has changed a little bit over time. An older version of SAC also learns a value function :math:`V_{\psi}` in addition to the Q-functions; this page will focus on the modern version that omits the extra value function.



**Learning Q.** The Q-functions are learned in a similar way to TD3, but with a few key differences. 

First, what's similar?

1) Like in TD3, both Q-functions are learned with MSBE minimization, by regressing to a single shared target.

2) Like in TD3, the shared target is computed using target Q-networks, and the target Q-networks are obtained by polyak averaging the Q-network parameters over the course of training.

3) Like in TD3, the shared target makes use of the **clipped double-Q** trick.

What's different?

1) Unlike in TD3, the target also includes a term that comes from SAC's use of entropy regularization.

2) Unlike in TD3, the next-state actions used in the target come from the **current policy** instead of a target policy.

3) Unlike in TD3, there is no explicit target policy smoothing. TD3 trains a deterministic policy, and so it accomplishes smoothing by adding random noise to the next-state actions. SAC trains a stochastic policy, and so the noise from that stochasticity is sufficient to get a similar effect.

Before we give the final form of the Q-loss, let’s take a moment to discuss how the contribution from entropy regularization comes in. We'll start by taking our recursive Bellman equation for the entropy-regularized :math:`Q^{\pi}` from earlier, and rewriting it a little bit by using the definition of entropy:

.. math::

    Q^{\pi}(s,a) &= \underE{s' \sim P \\ a' \sim \pi}{R(s,a,s') + \gamma\left(Q^{\pi}(s',a') + \alpha H\left(\pi(\cdot|s')\right) \right)} \\
    &= \underE{s' \sim P \\ a' \sim \pi}{R(s,a,s') + \gamma\left(Q^{\pi}(s',a') - \alpha \log \pi(a'|s') \right)} 

The RHS is an expectation over next states (which come from the replay buffer) and next actions (which come from the current policy, and **not** the replay buffer). Since it's an expectation, we can approximate it with samples:

.. math::

    Q^{\pi}(s,a) &\approx r + \gamma\left(Q^{\pi}(s',\tilde{a}') - \alpha \log \pi(\tilde{a}'|s') \right), \;\;\;\;\;  \tilde{a}' \sim \pi(\cdot|s').

.. admonition:: You Should Know

    We switch next action notation to :math:`\tilde{a}'`, instead of :math:`a'`, to highlight that the next actions have to be sampled fresh from the policy (whereas by contrast, :math:`r` and :math:`s'` should come from the replay buffer).

SAC sets up the MSBE loss for each Q-function using this kind of sample approximation for the target. The only thing still undetermined here is which Q-function gets used to compute the sample backup: like TD3, SAC uses the clipped double-Q trick, and takes the minimum Q-value between the two Q approximators. 

Putting it all together, the loss functions for the Q-networks in SAC are:

.. math::

    L(\phi_i, {\mathcal D}) = \underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[
        \Bigg( Q_{\phi_i}(s,a) - y(r,s',d) \Bigg)^2
        \right],

where the target is given by

.. math::

    y(r, s', d) = r + \gamma (1 - d) \left( \min_{j=1,2} Q_{\phi_{\text{targ},j}}(s', \tilde{a}') - \alpha \log \pi_{\theta}(\tilde{a}'|s') \right), \;\;\;\;\; \tilde{a}' \sim \pi_{\theta}(\cdot|s').


**Learning the Policy.** The policy should, in each state, act to maximize the expected future return plus expected future entropy. That is, it should maximize :math:`V^{\pi}(s)`, which we expand out into

.. math::

    V^{\pi}(s) &= \underE{a \sim \pi}{Q^{\pi}(s,a)} + \alpha H\left(\pi(\cdot|s)\right) \\
    &= \underE{a \sim \pi}{Q^{\pi}(s,a) - \alpha \log \pi(a|s)}.

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

To get the policy loss, the final step is that we need to substitute :math:`Q^{\pi_{\theta}}` with one of our function approximators. Unlike in TD3, which uses :math:`Q_{\phi_1}` (just the first Q approximator), SAC uses :math:`\min_{j=1,2} Q_{\phi_j}` (the minimum of the two Q approximators). The policy is thus optimized according to

.. math::

    \max_{\theta} \underE{s \sim \mathcal{D} \\ \xi \sim \mathcal{N}}{\min_{j=1,2} Q_{\phi_j}(s,\tilde{a}_{\theta}(s,\xi)) - \alpha \log \pi_{\theta}(\tilde{a}_{\theta}(s,\xi)|s)},

which is almost the same as the DDPG and TD3 policy optimization, except for the min-double-Q trick, the stochasticity, and the entropy term.


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
        \STATE Input: initial policy parameters $\theta$, Q-function parameters $\phi_1$, $\phi_2$, empty replay buffer $\mathcal{D}$
        \STATE Set target parameters equal to main parameters $\phi_{\text{targ},1} \leftarrow \phi_1$, $\phi_{\text{targ},2} \leftarrow \phi_2$
        \REPEAT
            \STATE Observe state $s$ and select action $a \sim \pi_{\theta}(\cdot|s)$
            \STATE Execute $a$ in the environment
            \STATE Observe next state $s'$, reward $r$, and done signal $d$ to indicate whether $s'$ is terminal
            \STATE Store $(s,a,r,s',d)$ in replay buffer $\mathcal{D}$
            \STATE If $s'$ is terminal, reset environment state.
            \IF{it's time to update}
                \FOR{$j$ in range(however many updates)}
                    \STATE Randomly sample a batch of transitions, $B = \{ (s,a,r,s',d) \}$ from $\mathcal{D}$
                    \STATE Compute targets for the Q functions:
                    \begin{align*}
                        y (r,s',d) &= r + \gamma (1-d) \left(\min_{i=1,2} Q_{\phi_{\text{targ}, i}} (s', \tilde{a}') - \alpha \log \pi_{\theta}(\tilde{a}'|s')\right), && \tilde{a}' \sim \pi_{\theta}(\cdot|s')
                    \end{align*}
                    \STATE Update Q-functions by one step of gradient descent using
                    \begin{align*}
                        & \nabla_{\phi_i} \frac{1}{|B|}\sum_{(s,a,r,s',d) \in B} \left( Q_{\phi_i}(s,a) - y(r,s',d) \right)^2 && \text{for } i=1,2
                    \end{align*}
                    \STATE Update policy by one step of gradient ascent using
                    \begin{equation*}
                        \nabla_{\theta} \frac{1}{|B|}\sum_{s \in B} \Big(\min_{i=1,2} Q_{\phi_i}(s, \tilde{a}_{\theta}(s)) - \alpha \log \pi_{\theta} \left(\left. \tilde{a}_{\theta}(s) \right| s\right) \Big),
                    \end{equation*}
                    where $\tilde{a}_{\theta}(s)$ is a sample from $\pi_{\theta}(\cdot|s)$ which is differentiable wrt $\theta$ via the reparametrization trick.
                    \STATE Update target networks with
                    \begin{align*}
                        \phi_{\text{targ},i} &\leftarrow \rho \phi_{\text{targ}, i} + (1-\rho) \phi_i && \text{for } i=1,2
                    \end{align*}
                \ENDFOR
            \ENDIF
        \UNTIL{convergence}
    \end{algorithmic}
    \end{algorithm}


Documentation
=============

.. admonition:: You Should Know

    In what follows, we give documentation for the PyTorch and Tensorflow implementations of SAC in Spinning Up. They have nearly identical function calls and docstrings, except for details relating to model construction. However, we include both full docstrings for completeness.



Documentation: PyTorch Version
------------------------------

.. autofunction:: spinup.sac_pytorch

Saved Model Contents: PyTorch Version
-------------------------------------

The PyTorch saved model can be loaded with ``ac = torch.load('path/to/model.pt')``, yielding an actor-critic object (``ac``) that has the properties described in the docstring for ``sac_pytorch``. 

You can get actions from this model with

.. code-block:: python

    actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))


Documentation: Tensorflow Version
---------------------------------

.. autofunction:: spinup.sac_tf1

Saved Model Contents: Tensorflow Version
----------------------------------------

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
- `Soft Actor-Critic Algorithms and Applications`_, Haarnoja et al, 2018
- `Learning to Walk via Deep Reinforcement Learning`_, Haarnoja et al, 2018

.. _`Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor`: https://arxiv.org/abs/1801.01290
.. _`Soft Actor-Critic Algorithms and Applications`: https://arxiv.org/abs/1812.05905
.. _`Learning to Walk via Deep Reinforcement Learning`: https://arxiv.org/abs/1812.11103

Other Public Implementations
----------------------------

- `SAC release repo`_ (original "official" codebase)
- `Softlearning repo`_ (current "official" codebase)
- `Yarats and Kostrikov repo`_

.. _`SAC release repo`: https://github.com/haarnoja/sac
.. _`Softlearning repo`: https://github.com/rail-berkeley/softlearning
.. _`Yarats and Kostrikov repo`: https://github.com/denisyarats/pytorch_sac