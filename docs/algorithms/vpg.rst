=======================
Vanilla Policy Gradient
=======================

.. contents:: Table of Contents


Background
==========

(Previously: `Introduction to RL, Part 3`_)

.. _`Introduction to RL, Part 3`: ../spinningup/rl_intro3.html

The key idea underlying policy gradients is to push up the probabilities of actions that lead to higher return, and push down the probabilities of actions that lead to lower return, until you arrive at the optimal policy.

Quick Facts
-----------

* VPG is an on-policy algorithm.
* VPG can be used for environments with either discrete or continuous action spaces.
* The Spinning Up implementation of VPG supports parallelization with MPI.

Key Equations
-------------

Let :math:`\pi_{\theta}` denote a policy with parameters :math:`\theta`, and :math:`J(\pi_{\theta})` denote the expected finite-horizon undiscounted return of the policy. The gradient of :math:`J(\pi_{\theta})` is

.. math:: 
    
    \nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{
        \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) A^{\pi_{\theta}}(s_t,a_t)
        },

where :math:`\tau` is a trajectory and :math:`A^{\pi_{\theta}}` is the advantage function for the current policy. 

The policy gradient algorithm works by updating policy parameters via stochastic gradient ascent on policy performance:

.. math::

    \theta_{k+1} = \theta_k + \alpha \nabla_{\theta} J(\pi_{\theta_k})

Policy gradient implementations typically compute advantage function estimates based on the infinite-horizon discounted return, despite otherwise using the finite-horizon undiscounted policy gradient formula. 

Exploration vs. Exploitation
----------------------------

VPG trains a stochastic policy in an on-policy way. This means that it explores by sampling actions according to the latest version of its stochastic policy. The amount of randomness in action selection depends on both initial conditions and the training procedure. Over the course of training, the policy typically becomes progressively less random, as the update rule encourages it to exploit rewards that it has already found. This may cause the policy to get trapped in local optima.


Pseudocode
----------

.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{Vanilla Policy Gradient Algorithm}
        \label{alg1}
    \begin{algorithmic}[1]
        \STATE Input: initial policy parameters $\theta_0$, initial value function parameters $\phi_0$
        \FOR{$k = 0,1,2,...$} 
        \STATE Collect set of trajectories ${\mathcal D}_k = \{\tau_i\}$ by running policy $\pi_k = \pi(\theta_k)$ in the environment.
        \STATE Compute rewards-to-go $\hat{R}_t$.
        \STATE Compute advantage estimates, $\hat{A}_t$ (using any method of advantage estimation) based on the current value function $V_{\phi_k}$.
        \STATE Estimate policy gradient as
            \begin{equation*}
            \hat{g}_k = \frac{1}{|{\mathcal D}_k|} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T \left. \nabla_{\theta} \log\pi_{\theta}(a_t|s_t)\right|_{\theta_k} \hat{A}_t.
            \end{equation*}
        \STATE Compute policy update, either using standard gradient ascent,
            \begin{equation*}
            \theta_{k+1} = \theta_k + \alpha_k \hat{g}_k,
            \end{equation*}
            or via another gradient ascent algorithm like Adam.
        \STATE Fit value function by regression on mean-squared error:
            \begin{equation*}
            \phi_{k+1} = \arg \min_{\phi} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T\left( V_{\phi} (s_t) - \hat{R}_t \right)^2,
            \end{equation*}
            typically via some gradient descent algorithm.
        \ENDFOR
    \end{algorithmic}
    \end{algorithm}


Documentation
=============

.. admonition:: You Should Know

    In what follows, we give documentation for the PyTorch and Tensorflow implementations of VPG in Spinning Up. They have nearly identical function calls and docstrings, except for details relating to model construction. However, we include both full docstrings for completeness.


Documentation: PyTorch Version
------------------------------

.. autofunction:: spinup.vpg_pytorch

Saved Model Contents: PyTorch Version
-------------------------------------

The PyTorch saved model can be loaded with ``ac = torch.load('path/to/model.pt')``, yielding an actor-critic object (``ac``) that has the properties described in the docstring for ``vpg_pytorch``. 

You can get actions from this model with

.. code-block:: python

    actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))


Documentation: Tensorflow Version
---------------------------------

.. autofunction:: spinup.vpg_tf1

Saved Model Contents: Tensorflow Version
----------------------------------------

The computation graph saved by the logger includes:

========  ====================================================================
Key       Value
========  ====================================================================
``x``     Tensorflow placeholder for state input.
``pi``    Samples an action from the agent, conditioned on states in ``x``.
``v``     Gives value estimate for states in ``x``. 
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

- `Policy Gradient Methods for Reinforcement Learning with Function Approximation`_, Sutton et al. 2000
- `Optimizing Expectations: From Deep Reinforcement Learning to Stochastic Computation Graphs`_, Schulman 2016(a)
- `Benchmarking Deep Reinforcement Learning for Continuous Control`_, Duan et al. 2016
- `High Dimensional Continuous Control Using Generalized Advantage Estimation`_, Schulman et al. 2016(b)

.. _`Policy Gradient Methods for Reinforcement Learning with Function Approximation`: https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf
.. _`Optimizing Expectations: From Deep Reinforcement Learning to Stochastic Computation Graphs`: http://joschu.net/docs/thesis.pdf
.. _`Benchmarking Deep Reinforcement Learning for Continuous Control`: https://arxiv.org/abs/1604.06778
.. _`High Dimensional Continuous Control Using Generalized Advantage Estimation`: https://arxiv.org/abs/1506.02438

Why These Papers?
-----------------

Sutton 2000 is included because it is a timeless classic of reinforcement learning theory, and contains references to the earlier work which led to modern policy gradients. Schulman 2016(a) is included because Chapter 2 contains a lucid introduction to the theory of policy gradient algorithms, including pseudocode. Duan 2016 is a clear, recent benchmark paper that shows how vanilla policy gradient in the deep RL setting (eg with neural network policies and Adam as the optimizer) compares with other deep RL algorithms. Schulman 2016(b) is included because our implementation of VPG makes use of Generalized Advantage Estimation for computing the policy gradient.


Other Public Implementations
----------------------------

- rllab_
- `rllib (Ray)`_

.. _rllab: https://github.com/rll/rllab/blob/master/rllab/algos/vpg.py
.. _`rllib (Ray)`: https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/pg
