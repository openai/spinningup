==============
Extra Material
==============

Proof for Using Q-Function in Policy Gradient Formula
=====================================================

In this section, we will show that

.. math::

    \nabla_{\theta} J(\pi_{\theta}) &= \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \Big( \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \Big) Q^{\pi_{\theta}}(s_t, a_t)},

for the finite-horizon undiscounted return setting. (An analagous result holds in the infinite-horizon discounted case using basically the same proof.)


The proof of this claim depends on the `law of iterated expectations`_. First, let's rewrite the expression for the policy gradient, starting from the reward-to-go form (using the notation :math:`\hat{R}_t = \sum_{t'=t}^T R(s_t', a_t', s_{t'+1})` to help shorten things):

.. math::

    \nabla_{\theta} J(\pi_{\theta}) &= \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \hat{R}_t} \\
    &= \sum_{t=0}^{T} \underE{\tau \sim \pi_{\theta}}{\nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \hat{R}_t}

Define :math:`\tau_{:t} = (s_0, a_0, ..., s_t, a_t)` as the trajectory up to time :math:`t`, and :math:`\tau_{t:}` as the remainder of the trajectory after that. By the law of iterated expectations, we can break up the preceding expression into:

.. math::

    \nabla_{\theta} J(\pi_{\theta}) &= \sum_{t=0}^{T} \underE{\tau_{:t} \sim \pi_{\theta}}{ \underE{\tau_{t:} \sim \pi_{\theta}}{ \left. \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \hat{R}_t \right| \tau_{:t}}}

The grad-log-prob is constant with respect to the inner expectation (because it depends on :math:`s_t` and :math:`a_t`, which the inner expectation conditions on as fixed in :math:`\tau_{:t}`), so it can be pulled out, leaving:

.. math::

    \nabla_{\theta} J(\pi_{\theta}) &= \sum_{t=0}^{T} \underE{\tau_{:t} \sim \pi_{\theta}}{ \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \underE{\tau_{t:} \sim \pi_{\theta}}{ \left. \hat{R}_t \right| \tau_{:t}}}

In Markov Decision Processes, the future only depends on the most recent state and action. As a result, the inner expectation---which expects over the future, conditioned on the entirety of the past (everything up to time :math:`t`)---is equal to the same expectation if it only conditioned on the last timestep (just :math:`(s_t,a_t)`):

.. math::

    \underE{\tau_{t:} \sim \pi_{\theta}}{ \left. \hat{R}_t \right| \tau_{:t}} = \underE{\tau_{t:} \sim \pi_{\theta}}{ \left. \hat{R}_t \right| s_t, a_t},

which is the *definition* of :math:`Q^{\pi_{\theta}}(s_t, a_t)`: the expected return, starting from state :math:`s_t` and action :math:`a_t`, when acting on-policy for the rest of the trajectory. 

The result follows immediately.

.. _`law of iterated expectations`: https://en.wikipedia.org/wiki/Law_of_total_expectation
