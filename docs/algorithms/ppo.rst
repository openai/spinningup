============================
Proximal Policy Optimization
============================

.. contents:: Table of Contents


Background
==========


(Previously: `Background for TRPO`_)

.. _`Background for TRPO`: ../algorithms/trpo.html#background

PPO is motivated by the same question as TRPO: how can we take the biggest possible improvement step on a policy using the data we currently have, without stepping so far that we accidentally cause performance collapse? Where TRPO tries to solve this problem with a complex second-order method, PPO is a family of first-order methods that use a few other tricks to keep new policies close to old. PPO methods are significantly simpler to implement, and empirically seem to perform at least as well as TRPO.

There are two primary variants of PPO: PPO-Penalty and PPO-Clip. 

**PPO-Penalty** approximately solves a KL-constrained update like TRPO, but penalizes the KL-divergence in the objective function instead of making it a hard constraint, and automatically adjusts the penalty coefficient over the course of training so that it's scaled appropriately. 

**PPO-Clip** doesn't have a KL-divergence term in the objective and doesn't have a constraint at all. Instead relies on specialized clipping in the objective function to remove incentives for the new policy to get far from the old policy. 

Here, we'll focus only on PPO-Clip (the primary variant used at OpenAI).

Quick Facts
-----------

* PPO is an on-policy algorithm.
* PPO can be used for environments with either discrete or continuous action spaces.
* The Spinning Up implementation of PPO supports parallelization with MPI.

Key Equations
-------------

PPO-clip updates policies via

.. math::

    \theta_{k+1} = \arg \max_{\theta} \underset{s,a \sim \pi_{\theta_k}}{{\mathrm E}}\left[
        L(s,a,\theta_k, \theta)\right],

typically taking multiple steps of (usually minibatch) SGD to maximize the objective. Here :math:`L` is given by

.. math::

    L(s,a,\theta_k,\theta) = \min\left(
    \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), \;\;
    \text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, 1 - \epsilon, 1+\epsilon \right) A^{\pi_{\theta_k}}(s,a)
    \right),

in which :math:`\epsilon` is a (small) hyperparameter which roughly says how far away the new policy is allowed to go from the old.

This is a pretty complex expression, and it's hard to tell at first glance what it's doing, or how it helps keep the new policy close to the old policy. As it turns out, there's a considerably simplified version [1]_ of this objective which is a bit easier to grapple with (and is also the version we implement in our code):

.. math::

    L(s,a,\theta_k,\theta) = \min\left(
    \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), \;\;
    g(\epsilon, A^{\pi_{\theta_k}}(s,a))
    \right),

where

.. math::

    g(\epsilon, A) = \left\{ 
        \begin{array}{ll}
        (1 + \epsilon) A & A \geq 0 \\
        (1 - \epsilon) A & A < 0.
        \end{array}
        \right.

To figure out what intuition to take away from this, let's look at a single state-action pair :math:`(s,a)`, and think of cases. 

**Advantage is positive**: Suppose the advantage for that state-action pair is positive, in which case its contribution to the objective reduces to

.. math::
    
    L(s,a,\theta_k,\theta) = \min\left(
    \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, (1 + \epsilon)
    \right)  A^{\pi_{\theta_k}}(s,a).

Because the advantage is positive, the objective will increase if the action becomes more likely---that is, if :math:`\pi_{\theta}(a|s)` increases. But the min in this term puts a limit to how *much* the objective can increase. Once :math:`\pi_{\theta}(a|s) > (1+\epsilon) \pi_{\theta_k}(a|s)`, the min kicks in and this term hits a ceiling of :math:`(1+\epsilon) A^{\pi_{\theta_k}}(s,a)`. Thus: *the new policy does not benefit by going far away from the old policy*.

**Advantage is negative**: Suppose the advantage for that state-action pair is negative, in which case its contribution to the objective reduces to

.. math::
    
    L(s,a,\theta_k,\theta) = \max\left(
    \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, (1 - \epsilon)
    \right)  A^{\pi_{\theta_k}}(s,a).

Because the advantage is negative, the objective will increase if the action becomes less likely---that is, if :math:`\pi_{\theta}(a|s)` decreases. But the max in this term puts a limit to how *much* the objective can increase. Once :math:`\pi_{\theta}(a|s) < (1-\epsilon) \pi_{\theta_k}(a|s)`, the max kicks in and this term hits a ceiling of :math:`(1-\epsilon) A^{\pi_{\theta_k}}(s,a)`. Thus, again: *the new policy does not benefit by going far away from the old policy*.

What we have seen so far is that clipping serves as a regularizer by removing incentives for the policy to change dramatically, and the hyperparameter :math:`\epsilon` corresponds to how far away the new policy can go from the old while still profiting the objective.

.. admonition:: You Should Know

    While this kind of clipping goes a long way towards ensuring reasonable policy updates, it is still possible to end up with a new policy which is too far from the old policy, and there are a bunch of tricks used by different PPO implementations to stave this off. In our implementation here, we use a particularly simple method: early stopping. If the mean KL-divergence of the new policy from the old grows beyond a threshold, we stop taking gradient steps. 

    When you feel comfortable with the basic math and implementation details, it's worth checking out other implementations to see how they handle this issue!


.. [1] See `this note`_ for a derivation of the simplified form of the PPO-Clip objective.


.. _`this note`: https://drive.google.com/file/d/1PDzn9RPvaXjJFZkGeapMHbHGiWWW20Ey/view?usp=sharing


Exploration vs. Exploitation
----------------------------

PPO trains a stochastic policy in an on-policy way. This means that it explores by sampling actions according to the latest version of its stochastic policy. The amount of randomness in action selection depends on both initial conditions and the training procedure. Over the course of training, the policy typically becomes progressively less random, as the update rule encourages it to exploit rewards that it has already found. This may cause the policy to get trapped in local optima.


Pseudocode
----------

.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{PPO-Clip}
        \label{alg1}
    \begin{algorithmic}[1]
        \STATE Input: initial policy parameters $\theta_0$, initial value function parameters $\phi_0$
        \FOR{$k = 0,1,2,...$} 
        \STATE Collect set of trajectories ${\mathcal D}_k = \{\tau_i\}$ by running policy $\pi_k = \pi(\theta_k)$ in the environment.
        \STATE Compute rewards-to-go $\hat{R}_t$.
        \STATE Compute advantage estimates, $\hat{A}_t$ (using any method of advantage estimation) based on the current value function $V_{\phi_k}$.
        \STATE Update the policy by maximizing the PPO-Clip objective:
            \begin{equation*}
            \theta_{k+1} = \arg \max_{\theta} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T \min\left(
                \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_k}(a_t|s_t)}  A^{\pi_{\theta_k}}(s_t,a_t), \;\;
                g(\epsilon, A^{\pi_{\theta_k}}(s_t,a_t))
            \right),
            \end{equation*}
            typically via stochastic gradient ascent with Adam.
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

    In what follows, we give documentation for the PyTorch and Tensorflow implementations of PPO in Spinning Up. They have nearly identical function calls and docstrings, except for details relating to model construction. However, we include both full docstrings for completeness.


Documentation: PyTorch Version
------------------------------

.. autofunction:: spinup.ppo_pytorch

Saved Model Contents: PyTorch Version
-------------------------------------

The PyTorch saved model can be loaded with ``ac = torch.load('path/to/model.pt')``, yielding an actor-critic object (``ac``) that has the properties described in the docstring for ``ppo_pytorch``. 

You can get actions from this model with

.. code-block:: python

    actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))


Documentation: Tensorflow Version
---------------------------------

.. autofunction:: spinup.ppo_tf1

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

- `Proximal Policy Optimization Algorithms`_, Schulman et al. 2017
- `High Dimensional Continuous Control Using Generalized Advantage Estimation`_, Schulman et al. 2016
- `Emergence of Locomotion Behaviours in Rich Environments`_, Heess et al. 2017

.. _`Proximal Policy Optimization Algorithms`: https://arxiv.org/abs/1707.06347
.. _`High Dimensional Continuous Control Using Generalized Advantage Estimation`: https://arxiv.org/abs/1506.02438
.. _`Emergence of Locomotion Behaviours in Rich Environments`: https://arxiv.org/abs/1707.02286

Why These Papers?
-----------------

Schulman 2017 is included because it is the original paper describing PPO. Schulman 2016 is included because our implementation of PPO makes use of Generalized Advantage Estimation for computing the policy gradient. Heess 2017 is included because it presents a large-scale empirical analysis of behaviors learned by PPO agents in complex environments (although it uses PPO-penalty instead of PPO-clip). 



Other Public Implementations
----------------------------

- Baselines_
- ModularRL_ (Caution: this implements PPO-penalty instead of PPO-clip.)
- rllab_ (Caution: this implements PPO-penalty instead of PPO-clip.)
- `rllib (Ray)`_ 

.. _Baselines: https://github.com/openai/baselines/tree/master/baselines/ppo2
.. _ModularRL: https://github.com/joschu/modular_rl/blob/master/modular_rl/ppo.py
.. _rllab: https://github.com/rll/rllab/blob/master/rllab/algos/ppo.py
.. _`rllib (Ray)`: https://github.com/ray-project/ray/tree/master/python/ray/rllib/agents/ppo