=================
Twin Delayed DDPG
=================

.. contents:: Table of Contents

Background
==========

(Previously: `Background for DDPG`_)

.. _`Background for DDPG`: ../algorithms/ddpg.html#background

While DDPG can achieve great performance sometimes, it is frequently brittle with respect to hyperparameters and other kinds of tuning. A common failure mode for DDPG is that the learned Q-function begins to dramatically overestimate Q-values, which then leads to the policy breaking, because it exploits the errors in the Q-function. Twin Delayed DDPG (TD3) is an algorithm that addresses this issue by introducing three critical tricks:

**Trick One: Clipped Double-Q Learning.** TD3 learns *two* Q-functions instead of one (hence "twin"), and uses the smaller of the two Q-values to form the targets in the Bellman error loss functions.

**Trick Two: "Delayed" Policy Updates.** TD3 updates the policy (and target networks) less frequently than the Q-function. The paper recommends one policy update for every two Q-function updates.

**Trick Three: Target Policy Smoothing.** TD3 adds noise to the target action, to make it harder for the policy to exploit Q-function errors by smoothing out Q along changes in action.

Together, these three tricks result in substantially improved performance over baseline DDPG.

Quick Facts
-----------

* TD3 is an off-policy algorithm.
* TD3 can only be used for environments with continuous action spaces.
* The Spinning Up implementation of TD3 does not support parallelization.

Key Equations
-------------

TD3 concurrently learns two Q-functions, :math:`Q_{\phi_1}` and :math:`Q_{\phi_2}`, by mean square Bellman error minimization, in almost the same way that DDPG learns its single Q-function. To show exactly how TD3 does this and how it differs from normal DDPG, we'll work from the innermost part of the loss function outwards.

First: **target policy smoothing**. Actions used to form the Q-learning target are based on the target policy, :math:`\mu_{\theta_{\text{targ}}}`, but with clipped noise added on each dimension of the action. After adding the clipped noise, the target action is then clipped to lie in the valid action range (all valid actions, :math:`a`, satisfy :math:`a_{Low} \leq a \leq a_{High}`). The target actions are thus: 

.. math::

    a'(s') = \text{clip}\left(\mu_{\theta_{\text{targ}}}(s') + \text{clip}(\epsilon,-c,c), a_{Low}, a_{High}\right), \;\;\;\;\; \epsilon \sim \mathcal{N}(0, \sigma)

Target policy smoothing essentially serves as a regularizer for the algorithm. It addresses a particular failure mode that can happen in DDPG: if the Q-function approximator develops an incorrect sharp peak for some actions, the policy will quickly exploit that peak and then have brittle or incorrect behavior. This can be averted by smoothing out the Q-function over similar actions, which target policy smoothing is designed to do. 

Next: **clipped double-Q learning**. Both Q-functions use a single target, calculated using whichever of the two Q-functions gives a smaller target value:

.. math::

    y(r,s',d) = r + \gamma (1 - d) \min_{i=1,2} Q_{\phi_{i, \text{targ}}}(s', a'(s')),

and then both are learned by regressing to this target:

.. math::

    L(\phi_1, {\mathcal D}) = \underE{(s,a,r,s',d) \sim {\mathcal D}}{
        \Bigg( Q_{\phi_1}(s,a) - y(r,s',d) \Bigg)^2
        },

.. math::

    L(\phi_2, {\mathcal D}) = \underE{(s,a,r,s',d) \sim {\mathcal D}}{
        \Bigg( Q_{\phi_2}(s,a) - y(r,s',d) \Bigg)^2
        }.

Using the smaller Q-value for the target, and regressing towards that, helps fend off overestimation in the Q-function.

Lastly: the policy is learned just by maximizing :math:`Q_{\phi_1}`:

.. math::

    \max_{\theta} \underset{s \sim {\mathcal D}}{{\mathrm E}}\left[ Q_{\phi_1}(s, \mu_{\theta}(s)) \right],

which is pretty much unchanged from DDPG. However, in TD3, the policy is updated less frequently than the Q-functions are. This helps damp the volatility that normally arises in DDPG because of how a policy update changes the target.


Exploration vs. Exploitation
----------------------------

TD3 trains a deterministic policy in an off-policy way. Because the policy is deterministic, if the agent were to explore on-policy, in the beginning it would probably not try a wide enough variety of actions to find useful learning signals. To make TD3 policies explore better, we add noise to their actions at training time, typically uncorrelated mean-zero Gaussian noise. To facilitate getting higher-quality training data, you may reduce the scale of the noise over the course of training. (We do not do this in our implementation, and keep noise scale fixed throughout.)

At test time, to see how well the policy exploits what it has learned, we do not add noise to the actions.

.. admonition:: You Should Know

    Our TD3 implementation uses a trick to improve exploration at the start of training. For a fixed number of steps at the beginning (set with the ``start_steps`` keyword argument), the agent takes actions which are sampled from a uniform random distribution over valid actions. After that, it returns to normal TD3 exploration.


Pseudocode
----------


.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{Twin Delayed DDPG}
        \label{alg1}
    \begin{algorithmic}[1]
        \STATE Input: initial policy parameters $\theta$, Q-function parameters $\phi_1$, $\phi_2$, empty replay buffer $\mathcal{D}$
        \STATE Set target parameters equal to main parameters $\theta_{\text{targ}} \leftarrow \theta$, $\phi_{\text{targ},1} \leftarrow \phi_1$, $\phi_{\text{targ},2} \leftarrow \phi_2$
        \REPEAT
            \STATE Observe state $s$ and select action $a = \text{clip}(\mu_{\theta}(s) + \epsilon, a_{Low}, a_{High})$, where $\epsilon \sim \mathcal{N}$
            \STATE Execute $a$ in the environment
            \STATE Observe next state $s'$, reward $r$, and done signal $d$ to indicate whether $s'$ is terminal
            \STATE Store $(s,a,r,s',d)$ in replay buffer $\mathcal{D}$
            \STATE If $s'$ is terminal, reset environment state.
            \IF{it's time to update}
                \FOR{$j$ in range(however many updates)}
                    \STATE Randomly sample a batch of transitions, $B = \{ (s,a,r,s',d) \}$ from $\mathcal{D}$
                    \STATE Compute target actions
                    \begin{equation*}
                        a'(s') = \text{clip}\left(\mu_{\theta_{\text{targ}}}(s') + \text{clip}(\epsilon,-c,c), a_{Low}, a_{High}\right), \;\;\;\;\; \epsilon \sim \mathcal{N}(0, \sigma)
                    \end{equation*}
                    \STATE Compute targets
                    \begin{equation*}
                        y(r,s',d) = r + \gamma (1-d) \min_{i=1,2} Q_{\phi_{\text{targ},i}}(s', a'(s'))
                    \end{equation*}
                    \STATE Update Q-functions by one step of gradient descent using
                    \begin{align*}
                        & \nabla_{\phi_i} \frac{1}{|B|}\sum_{(s,a,r,s',d) \in B} \left( Q_{\phi_i}(s,a) - y(r,s',d) \right)^2 && \text{for } i=1,2
                    \end{align*}
                    \IF{ $j \mod$ \texttt{policy\_delay} $ = 0$}
                        \STATE Update policy by one step of gradient ascent using
                        \begin{equation*}
                            \nabla_{\theta} \frac{1}{|B|}\sum_{s \in B}Q_{\phi_1}(s, \mu_{\theta}(s))
                        \end{equation*}
                        \STATE Update target networks with
                        \begin{align*}
                            \phi_{\text{targ},i} &\leftarrow \rho \phi_{\text{targ}, i} + (1-\rho) \phi_i && \text{for } i=1,2\\
                            \theta_{\text{targ}} &\leftarrow \rho \theta_{\text{targ}} + (1-\rho) \theta
                        \end{align*}
                    \ENDIF
                \ENDFOR
            \ENDIF
        \UNTIL{convergence}
    \end{algorithmic}
    \end{algorithm}


Documentation
=============

.. admonition:: You Should Know

    In what follows, we give documentation for the PyTorch and Tensorflow implementations of TD3 in Spinning Up. They have nearly identical function calls and docstrings, except for details relating to model construction. However, we include both full docstrings for completeness.



Documentation: PyTorch Version
------------------------------

.. autofunction:: spinup.td3_pytorch

Saved Model Contents: PyTorch Version
-------------------------------------

The PyTorch saved model can be loaded with ``ac = torch.load('path/to/model.pt')``, yielding an actor-critic object (``ac``) that has the properties described in the docstring for ``td3_pytorch``. 

You can get actions from this model with

.. code-block:: python

    actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))


Documentation: Tensorflow Version
---------------------------------

.. autofunction:: spinup.td3_tf1

Saved Model Contents: Tensorflow Version
----------------------------------------

The computation graph saved by the logger includes:

========  ====================================================================
Key       Value
========  ====================================================================
``x``     Tensorflow placeholder for state input.
``a``     Tensorflow placeholder for action input.
``pi``    | Deterministically computes an action from the agent, conditioned 
          | on states in ``x``.
``q1``    Gives one action-value estimate for states in ``x`` and actions in ``a``.
``q2``    Gives the other action-value estimate for states in ``x`` and actions in ``a``.
========  ====================================================================

This saved model can be accessed either by

* running the trained policy with the `test_policy.py`_ tool,
* or loading the whole saved graph into a program with `restore_tf_graph`_. 

.. _`test_policy.py`: ../user/saving_and_loading.html#loading-and-running-trained-policies
.. _`restore_tf_graph`: ../utils/logger.html#spinup.utils.logx.restore_tf_graph

References
==========

Relevant Papers
---------------

- `Addressing Function Approximation Error in Actor-Critic Methods`_, Fujimoto et al, 2018

.. _`Addressing Function Approximation Error in Actor-Critic Methods`: https://arxiv.org/abs/1802.09477


Other Public Implementations
----------------------------

- `TD3 release repo`_

.. _`TD3 release repo`: https://github.com/sfujim/TD3