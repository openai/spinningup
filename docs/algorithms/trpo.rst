================================
Trust Region Policy Optimization
================================

.. contents:: Table of Contents



Background
==========

(Previously: `Background for VPG`_)

.. _`Background for VPG`: ../algorithms/vpg.html#background

TRPO updates policies by taking the largest step possible to improve performance, while satisfying a special constraint on how close the new and old policies are allowed to be. The constraint is expressed in terms of `KL-Divergence`_, a measure of (something like, but not exactly) distance between probability distributions. 

This is different from normal policy gradient, which keeps new and old policies close in parameter space. But even seemingly small differences in parameter space can have very large differences in performance---so a single bad step can collapse the policy performance. This makes it dangerous to use large step sizes with vanilla policy gradients, thus hurting its sample efficiency. TRPO nicely avoids this kind of collapse, and tends to quickly and monotonically improve performance.

.. _`KL-Divergence`: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

Quick Facts
-----------

* TRPO is an on-policy algorithm.
* TRPO can be used for environments with either discrete or continuous action spaces.
* The Spinning Up implementation of TRPO supports parallelization with MPI.

Key Equations
-------------

Let :math:`\pi_{\theta}` denote a policy with parameters :math:`\theta`. The theoretical TRPO update is:

.. math:: 
    
    \theta_{k+1} = \arg \max_{\theta} \; & {\mathcal L}(\theta_k, \theta) \\
    \text{s.t.} \; & \bar{D}_{KL}(\theta || \theta_k) \leq \delta

where :math:`{\mathcal L}(\theta_k, \theta)` is the *surrogate advantage*, a measure of how policy :math:`\pi_{\theta}` performs relative to the old policy :math:`\pi_{\theta_k}` using data from the old policy:

.. math::

    {\mathcal L}(\theta_k, \theta) = \underE{s,a \sim \pi_{\theta_k}}{
        \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a)
        },

and :math:`\bar{D}_{KL}(\theta || \theta_k)` is an average KL-divergence between policies across states visited by the old policy:

.. math::

    \bar{D}_{KL}(\theta || \theta_k) = \underE{s \sim \pi_{\theta_k}}{
        D_{KL}\left(\pi_{\theta}(\cdot|s) || \pi_{\theta_k} (\cdot|s) \right)
    }.

.. admonition:: You Should Know

    The objective and constraint are both zero when :math:`\theta = \theta_k`. Furthermore, the gradient of the constraint with respect to :math:`\theta` is zero when :math:`\theta = \theta_k`. Proving these facts requires some subtle command of the relevant math---it's an exercise worth doing, whenever you feel ready!


The theoretical TRPO update isn't the easiest to work with, so TRPO makes some approximations to get an answer quickly. We Taylor expand the objective and constraint to leading order around :math:`\theta_k`:

.. math:: 

    {\mathcal L}(\theta_k, \theta) &\approx g^T (\theta - \theta_k) \\
    \bar{D}_{KL}(\theta || \theta_k) & \approx \frac{1}{2} (\theta - \theta_k)^T H (\theta - \theta_k)

resulting in an approximate optimization problem,

.. math:: 
    
    \theta_{k+1} = \arg \max_{\theta} \; & g^T (\theta - \theta_k) \\
    \text{s.t.} \; & \frac{1}{2} (\theta - \theta_k)^T H (\theta - \theta_k) \leq \delta.

.. admonition:: You Should Know

    By happy coincidence, the gradient :math:`g` of the surrogate advantage function with respect to :math:`\theta`, evaluated at :math:`\theta = \theta_k`, is exactly equal to the policy gradient, :math:`\nabla_{\theta} J(\pi_{\theta})`! Try proving this, if you feel comfortable diving into the math.

This approximate problem can be analytically solved by the methods of Lagrangian duality [1]_, yielding the solution:

.. math::

    \theta_{k+1} = \theta_k + \sqrt{\frac{2 \delta}{g^T H^{-1} g}} H^{-1} g.

If we were to stop here, and just use this final result, the algorithm would be exactly calculating the `Natural Policy Gradient`_. A problem is that, due to the approximation errors introduced by the Taylor expansion, this may not satisfy the KL constraint, or actually improve the surrogate advantage. TRPO adds a modification to this update rule: a backtracking line search,

.. math::

    \theta_{k+1} = \theta_k + \alpha^j \sqrt{\frac{2 \delta}{g^T H^{-1} g}} H^{-1} g,

where :math:`\alpha \in (0,1)` is the backtracking coefficient, and :math:`j` is the smallest nonnegative integer such that :math:`\pi_{\theta_{k+1}}` satisfies the KL constraint and produces a positive surrogate advantage. 

Lastly: computing and storing the matrix inverse, :math:`H^{-1}`, is painfully expensive when dealing with neural network policies with thousands or millions of parameters. TRPO sidesteps the issue by using the `conjugate gradient`_ algorithm to solve :math:`Hx = g` for :math:`x = H^{-1} g`, requiring only a function which can compute the matrix-vector product :math:`Hx` instead of computing and storing the whole matrix :math:`H` directly. This is not too hard to do: we set up a symbolic operation to calculate

.. math::

    Hx = \nabla_{\theta} \left( \left(\nabla_{\theta} \bar{D}_{KL}(\theta || \theta_k)\right)^T x \right),

which gives us the correct output without computing the whole matrix.

.. [1] See `Convex Optimization`_ by Boyd and Vandenberghe, especially chapters 2 through 5.

.. _`Convex Optimization`: http://stanford.edu/~boyd/cvxbook/
.. _`Natural Policy Gradient`: https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf
.. _`conjugate gradient`: https://en.wikipedia.org/wiki/Conjugate_gradient_method


Exploration vs. Exploitation
----------------------------

TRPO trains a stochastic policy in an on-policy way. This means that it explores by sampling actions according to the latest version of its stochastic policy. The amount of randomness in action selection depends on both initial conditions and the training procedure. Over the course of training, the policy typically becomes progressively less random, as the update rule encourages it to exploit rewards that it has already found. This may cause the policy to get trapped in local optima.


Pseudocode
----------

.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{Trust Region Policy Optimization}
        \label{alg1}
    \begin{algorithmic}[1]
        \STATE Input: initial policy parameters $\theta_0$, initial value function parameters $\phi_0$
        \STATE Hyperparameters: KL-divergence limit $\delta$, backtracking coefficient $\alpha$, maximum number of backtracking steps $K$
        \FOR{$k = 0,1,2,...$} 
        \STATE Collect set of trajectories ${\mathcal D}_k = \{\tau_i\}$ by running policy $\pi_k = \pi(\theta_k)$ in the environment.
        \STATE Compute rewards-to-go $\hat{R}_t$.
        \STATE Compute advantage estimates, $\hat{A}_t$ (using any method of advantage estimation) based on the current value function $V_{\phi_k}$.
        \STATE Estimate policy gradient as
            \begin{equation*}
            \hat{g}_k = \frac{1}{|{\mathcal D}_k|} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T \left. \nabla_{\theta} \log\pi_{\theta}(a_t|s_t)\right|_{\theta_k} \hat{A}_t.
            \end{equation*}
        \STATE Use the conjugate gradient algorithm to compute
            \begin{equation*}
            \hat{x}_k \approx \hat{H}_k^{-1} \hat{g}_k,
            \end{equation*}
            where $\hat{H}_k$ is the Hessian of the sample average KL-divergence.
        \STATE Update the policy by backtracking line search with
            \begin{equation*}
            \theta_{k+1} = \theta_k + \alpha^j \sqrt{ \frac{2\delta}{\hat{x}_k^T \hat{H}_k \hat{x}_k}} \hat{x}_k,
            \end{equation*}
            where $j \in \{0, 1, 2, ... K\}$ is the smallest value which improves the sample loss and satisfies the sample KL-divergence constraint.
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

    Spinning Up currently only has a Tensorflow implementation of TRPO. 

.. autofunction:: spinup.trpo_tf1


Saved Model Contents
--------------------

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

- `Trust Region Policy Optimization`_, Schulman et al. 2015
- `High Dimensional Continuous Control Using Generalized Advantage Estimation`_, Schulman et al. 2016
- `Approximately Optimal Approximate Reinforcement Learning`_, Kakade and Langford 2002

.. _`Trust Region Policy Optimization`: https://arxiv.org/abs/1502.05477
.. _`High Dimensional Continuous Control Using Generalized Advantage Estimation`: https://arxiv.org/abs/1506.02438
.. _`Approximately Optimal Approximate Reinforcement Learning`: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf

Why These Papers?
-----------------

Schulman 2015 is included because it is the original paper describing TRPO. Schulman 2016 is included because our implementation of TRPO makes use of Generalized Advantage Estimation for computing the policy gradient. Kakade and Langford 2002 is included because it contains theoretical results which motivate and deeply connect to the theoretical foundations of TRPO. 



Other Public Implementations
----------------------------

- Baselines_
- ModularRL_
- rllab_

.. _Baselines: https://github.com/openai/baselines/tree/master/baselines/trpo_mpi
.. _ModularRL: https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py
.. _rllab: https://github.com/rll/rllab/blob/master/rllab/algos/trpo.py