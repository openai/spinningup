====================================
Part 3: Intro to Policy Optimization
====================================

.. contents:: Table of Contents
    :depth: 2


In this section, we'll discuss the mathematical foundations of policy optimization algorithms, and connect the material to sample code. We will cover three key results in the theory of **policy gradients**: 

* `the simplest equation`_ describing the gradient of policy performance with respect to policy parameters,
* a rule which allows us to `drop useless terms`_ from that expression,
* and a rule which allows us to `add useful terms`_ to that expression.

In the end, we'll tie those results together and describe the advantage-based expression for the policy gradient---the version we use in our `Vanilla Policy Gradient`_ implementation.

.. _`the simplest equation`: ../spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient
.. _`drop useless terms`: ../spinningup/rl_intro3.html#don-t-let-the-past-distract-you
.. _`add useful terms`: ../spinningup/rl_intro3.html#baselines-in-policy-gradients
.. _`Vanilla Policy Gradient`: ../algorithms/vpg.html

Deriving the Simplest Policy Gradient
=====================================

Here, we consider the case of a stochastic, parameterized policy, :math:`\pi_{\theta}`. We aim to maximize the expected return :math:`J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{R(\tau)}`. For the purposes of this derivation, we'll take :math:`R(\tau)` to give the `finite-horizon undiscounted return`_, but the derivation for the infinite-horizon discounted return setting is almost identical.

.. _`finite-horizon undiscounted return`: ../spinningup/rl_intro.html#reward-and-return

We would like to optimize the policy by gradient ascent, eg

.. math::

    \theta_{k+1} = \theta_k + \alpha \left. \nabla_{\theta} J(\pi_{\theta}) \right|_{\theta_k}.

The gradient of policy performance, :math:`\nabla_{\theta} J(\pi_{\theta})`, is called the **policy gradient**, and algorithms that optimize the policy this way are called **policy gradient algorithms.** (Examples include Vanilla Policy Gradient and TRPO. PPO is often referred to as a policy gradient algorithm, though this is slightly inaccurate.)

To actually use this algorithm, we need an expression for the policy gradient which we can numerically compute. This involves two steps: 1) deriving the analytical gradient of policy performance, which turns out to have the form of an expected value, and then 2) forming a sample estimate of that expected value, which can be computed with data from a finite number of agent-environment interaction steps. 

In this subsection, we'll find the simplest form of that expression. In later subsections, we'll show how to improve on the simplest form to get the version we actually use in standard policy gradient implementations.

We'll begin by laying out a few facts which are useful for deriving the analytical gradient.

**1. Probability of a Trajectory.** The probability of a trajectory :math:`\tau = (s_0, a_0, ..., s_{T+1})` given that actions come from :math:`\pi_{\theta}` is

.. math::

    P(\tau|\theta) = \rho_0 (s_0) \prod_{t=0}^{T} P(s_{t+1}|s_t, a_t) \pi_{\theta}(a_t |s_t).


**2. The Log-Derivative Trick.** The log-derivative trick is based on a simple rule from calculus: the derivative of :math:`\log x` with respect to :math:`x` is :math:`1/x`. When rearranged and combined with chain rule, we get:

.. math::

    \nabla_{\theta} P(\tau | \theta) = P(\tau | \theta) \nabla_{\theta} \log P(\tau | \theta).


**3. Log-Probability of a Trajectory.** The log-prob of a trajectory is just

.. math::

    \log P(\tau|\theta) = \log \rho_0 (s_0) + \sum_{t=0}^{T} \bigg( \log P(s_{t+1}|s_t, a_t)  + \log \pi_{\theta}(a_t |s_t)\bigg).


**4. Gradients of Environment Functions.** The environment has no dependence on :math:`\theta`, so gradients of :math:`\rho_0(s_0)`, :math:`P(s_{t+1}|s_t, a_t)`, and :math:`R(\tau)` are zero.

**5. Grad-Log-Prob of a Trajectory.** The gradient of the log-prob of a trajectory is thus

.. math::

    \nabla_{\theta} \log P(\tau | \theta) &= \cancel{\nabla_{\theta} \log \rho_0 (s_0)} + \sum_{t=0}^{T} \bigg( \cancel{\nabla_{\theta} \log P(s_{t+1}|s_t, a_t)}  + \nabla_{\theta} \log \pi_{\theta}(a_t |s_t)\bigg) \\
    &= \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t).


Putting it all together, we derive the following:

.. admonition:: Derivation for Basic Policy Gradient

    .. math::
        :nowrap:

        \begin{align*}
        \nabla_{\theta} J(\pi_{\theta}) &= \nabla_{\theta} \underE{\tau \sim \pi_{\theta}}{R(\tau)} & \\
        &= \nabla_{\theta} \int_{\tau} P(\tau|\theta) R(\tau) & \text{Expand expectation} \\
        &= \int_{\tau} \nabla_{\theta} P(\tau|\theta) R(\tau) & \text{Bring gradient under integral} \\
        &= \int_{\tau} P(\tau|\theta) \nabla_{\theta} \log P(\tau|\theta) R(\tau) & \text{Log-derivative trick} \\
        &= \underE{\tau \sim \pi_{\theta}}{\nabla_{\theta} \log P(\tau|\theta) R(\tau)} & \text{Return to expectation form} \\
        \therefore \nabla_{\theta} J(\pi_{\theta}) &= \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)} & \text{Expression for grad-log-prob}
        \end{align*}

This is an expectation, which means that we can estimate it with a sample mean. If we collect a set of trajectories :math:`\mathcal{D} = \{\tau_i\}_{i=1,...,N}` where each trajectory is obtained by letting the agent act in the environment using the policy :math:`\pi_{\theta}`, the policy gradient can be estimated with

.. math::

    \hat{g} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau),

where :math:`|\mathcal{D}|` is the number of trajectories in :math:`\mathcal{D}` (here, :math:`N`).

This last expression is the simplest version of the computable expression we desired. Assuming that we have represented our policy in a way which allows us to calculate :math:`\nabla_{\theta} \log \pi_{\theta}(a|s)`, and if we are able to run the policy in the environment to collect the trajectory dataset, we can compute the policy gradient and take an update step.

Implementing the Simplest Policy Gradient
=========================================

We give a short PyTorch implementation of this simple version of the policy gradient algorithm in ``spinup/examples/pytorch/pg_math/1_simple_pg.py``. (It can also be viewed `on github <https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py>`_.) It is only 128 lines long, so we highly recommend reading through it in depth. While we won't go through the entirety of the code here, we'll highlight and explain a few important pieces.


.. admonition:: You Should Know

    This section was previously written with a Tensorflow example. The old Tensorflow section can be found `here <../spinningup/extra_tf_pg_implementation.html#implementing-the-simplest-policy-gradient>`_. 

**1. Making the Policy Network.** 

.. code-block:: python
    :linenos:
    :lineno-start: 30

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

This block builds modules and functions for using a feedforward neural network categorical policy. (See the `Stochastic Policies`_ section in Part 1 for a refresher.) The output from the ``logits_net`` module can be used to construct log-probabilities and probabilities for actions, and the ``get_action`` function samples actions based on probabilities computed from the logits. (Note: this particular ``get_action`` function assumes that there will only be one ``obs`` provided, and therefore only one integer action output. That's why it uses ``.item()``, which is used to `get the contents of a Tensor with only one element`_.)

A lot of work in this example is getting done by the ``Categorical`` object on L36. This is a PyTorch ``Distribution`` object that wraps up some mathematical functions associated with probability distributions. In particular, it has a method for sampling from the distribution (which we use on L40) and a method for computing log probabilities of given samples (which we use later). Since PyTorch distributions are really useful for RL, check out `their documentation`_ to get a feel for how they work.

.. admonition:: You Should Know

    Friendly reminder! When we talk about a categorical distribution having "logits," what we mean is that the probabilities for each outcome are given by the Softmax function of the logits. That is, the probability for action :math:`j` under a categorical distribution with logits :math:`x_j` is:

    .. math::

        p_j = \frac{\exp(x_j)}{\sum_{i} \exp(x_i)}


.. _`Stochastic Policies`: ../spinningup/rl_intro.html#stochastic-policies
.. _`their documentation`: https://pytorch.org/docs/stable/distributions.html
.. _`get the contents of a Tensor with only one element`: https://pytorch.org/docs/stable/tensors.html#torch.Tensor.item

**2. Making the Loss Function.**

.. code-block:: python
    :linenos:
    :lineno-start: 42

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()


In this block, we build a "loss" function for the policy gradient algorithm. When the right data is plugged in, the gradient of this loss is equal to the policy gradient. The right data means a set of (state, action, weight) tuples collected while acting according to the current policy, where the weight for a state-action pair is the return from the episode to which it belongs. (Although as we will show in later subsections, there are other values you can plug in for the weight which also work correctly.)


.. admonition:: You Should Know
    
    Even though we describe this as a loss function, it is **not** a loss function in the typical sense from supervised learning. There are two main differences from standard loss functions.

    **1. The data distribution depends on the parameters.** A loss function is usually defined on a fixed data distribution which is independent of the parameters we aim to optimize. Not so here, where the data must be sampled on the most recent policy. 

    **2. It doesn't measure performance.** A loss function usually evaluates the performance metric that we care about. Here, we care about expected return, :math:`J(\pi_{\theta})`, but our "loss" function does not approximate this at all, even in expectation. This "loss" function is only useful to us because, when evaluated at the current parameters, with data generated by the current parameters, it has the negative gradient of performance. 

    But after that first step of gradient descent, there is no more connection to performance. This means that minimizing this "loss" function, for a given batch of data, has *no* guarantee whatsoever of improving expected return. You can send this loss to :math:`-\infty` and policy performance could crater; in fact, it usually will. Sometimes a deep RL researcher might describe this outcome as the policy "overfitting" to a batch of data. This is descriptive, but should not be taken literally because it does not refer to generalization error.

    We raise this point because it is common for ML practitioners to interpret a loss function as a useful signal during training---"if the loss goes down, all is well." In policy gradients, this intuition is wrong, and you should only care about average return. The loss function means nothing.




.. admonition:: You Should Know
    
    The approach used here to make the ``logp`` tensor--calling the ``log_prob`` method of a PyTorch ``Categorical`` object--may require some modification to work with other kinds of distribution objects. 

    For example, if you are using a `Normal distribution`_ (for a diagonal Gaussian policy), the output from calling ``policy.log_prob(act)`` will give you a Tensor containing separate log probabilities for each component of each vector-valued action. That is to say, you put in a Tensor of shape ``(batch, act_dim)``, and get out a Tensor of shape ``(batch, act_dim)``, when what you need for making an RL loss is a Tensor of shape ``(batch,)``. In that case, you would sum up the log probabilities of the action components to get the log probabilities of the actions. That is, you would compute: 

    .. code-block:: python

        logp = get_policy(obs).log_prob(act).sum(axis=-1)


.. _`Normal distribution`: https://pytorch.org/docs/stable/distributions.html#normal


**3. Running One Epoch of Training.**

.. code-block:: python
    :linenos:
    :lineno-start: 50

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

The ``train_one_epoch()`` function runs one "epoch" of policy gradient, which we define to be 

1) the experience collection step (L67-102), where the agent acts for some number of episodes in the environment using the most recent policy, followed by 

2) a single policy gradient update step (L104-111). 

The main loop of the algorithm just repeatedly calls ``train_one_epoch()``. 

.. admonition:: You Should Know

    If you aren't already familiar with optimization in PyTorch, observe the pattern for taking one gradient descent step as shown in lines 104-111. First, clear the gradient buffers. Then, compute the loss function. Then, compute a backward pass on the loss function; this accumulates fresh gradients into the gradient buffers. Finally, take a step with the optimizer. 


Expected Grad-Log-Prob Lemma
============================

In this subsection, we will derive an intermediate result which is extensively used throughout the theory of policy gradients. We will call it the Expected Grad-Log-Prob (EGLP) lemma. [1]_

**EGLP Lemma.** Suppose that :math:`P_{\theta}` is a parameterized probability distribution over a random variable, :math:`x`. Then: 

.. math::

    \underE{x \sim P_{\theta}}{\nabla_{\theta} \log P_{\theta}(x)} = 0.

.. admonition:: Proof

    Recall that all probability distributions are *normalized*:

    .. math::

        \int_x P_{\theta}(x) = 1.

    Take the gradient of both sides of the normalization condition:

    .. math::

        \nabla_{\theta} \int_x P_{\theta}(x) = \nabla_{\theta} 1 = 0.

    Use the log derivative trick to get:

    .. math::

        0 &= \nabla_{\theta} \int_x P_{\theta}(x) \\
        &= \int_x \nabla_{\theta} P_{\theta}(x) \\
        &= \int_x P_{\theta}(x) \nabla_{\theta} \log P_{\theta}(x) \\
        \therefore 0 &= \underE{x \sim P_{\theta}}{\nabla_{\theta} \log P_{\theta}(x)}.

.. [1] The author of this article is not aware of this lemma being given a standard name anywhere in the literature. But given how often it comes up, it seems pretty worthwhile to give it some kind of name for ease of reference.

Don't Let the Past Distract You
===============================

Examine our most recent expression for the policy gradient:

.. math::

    \nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)}.

Taking a step with this gradient pushes up the log-probabilities of each action in proportion to :math:`R(\tau)`, the sum of *all rewards ever obtained*. But this doesn't make much sense. 

Agents should really only reinforce actions on the basis of their *consequences*. Rewards obtained before taking an action have no bearing on how good that action was: only rewards that come *after*.

It turns out that this intuition shows up in the math, and we can show that the policy gradient can also be expressed by

.. math::

    \nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1})}.

In this form, actions are only reinforced based on rewards obtained after they are taken. 

We'll call this form the "reward-to-go policy gradient," because the sum of rewards after a point in a trajectory,

.. math::

    \hat{R}_t \doteq \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}),

is called the **reward-to-go** from that point, and this policy gradient expression depends on the reward-to-go from state-action pairs.

.. admonition:: You Should Know

    **But how is this better?** A key problem with policy gradients is how many sample trajectories are needed to get a low-variance sample estimate for them. The formula we started with included terms for reinforcing actions proportional to past rewards, all of which had zero mean, but nonzero variance: as a result, they would just add noise to sample estimates of the policy gradient. By removing them, we reduce the number of sample trajectories needed.

An (optional) proof of this claim can be found `here`_, and it ultimately depends on the EGLP lemma.

.. _`here`: ../spinningup/extra_pg_proof1.html

Implementing Reward-to-Go Policy Gradient
=========================================


We give a short PyTorch implementation of the reward-to-go policy gradient in ``spinup/examples/pytorch/pg_math/2_rtg_pg.py``. (It can also be viewed `on github <https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/2_rtg_pg.py>`_.) 

The only thing that has changed from ``1_simple_pg.py`` is that we now use different weights in the loss function. The code modification is very slight: we add a new function, and change two other lines. The new function is:

.. code-block:: python
    :linenos:
    :lineno-start: 17

    def reward_to_go(rews):
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs


And then we tweak the old L91-92 from:

.. code-block:: python
    :linenos:
    :lineno-start: 91

                    # the weight for each logprob(a|s) is R(tau)
                    batch_weights += [ep_ret] * ep_len

to:

.. code-block:: python
    :linenos:
    :lineno-start: 98

                    # the weight for each logprob(a_t|s_t) is reward-to-go from t
                    batch_weights += list(reward_to_go(ep_rews))


.. admonition:: You Should Know

    This section was previously written with a Tensorflow example. The old Tensorflow section can be found `here <../spinningup/extra_tf_pg_implementation.html#implementing-reward-to-go-policy-gradient>`_. 


Baselines in Policy Gradients
=============================

An immediate consequence of the EGLP lemma is that for any function :math:`b` which only depends on state,

.. math::

    \underE{a_t \sim \pi_{\theta}}{\nabla_{\theta} \log \pi_{\theta}(a_t|s_t) b(s_t)} = 0.

This allows us to add or subtract any number of terms like this from our expression for the policy gradient, without changing it in expectation:

.. math::

    \nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \left(\sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t)\right)}.

Any function :math:`b` used in this way is called a **baseline**. 

The most common choice of baseline is the `on-policy value function`_ :math:`V^{\pi}(s_t)`. Recall that this is the average return an agent gets if it starts in state :math:`s_t` and then acts according to policy :math:`\pi` for the rest of its life. 

Empirically, the choice :math:`b(s_t) = V^{\pi}(s_t)` has the desirable effect of reducing variance in the sample estimate for the policy gradient. This results in faster and more stable policy learning. It is also appealing from a conceptual angle: it encodes the intuition that if an agent gets what it expected, it should "feel" neutral about it.

.. admonition:: You Should Know

    In practice, :math:`V^{\pi}(s_t)` cannot be computed exactly, so it has to be approximated. This is usually done with a neural network, :math:`V_{\phi}(s_t)`, which is updated concurrently with the policy (so that the value network always approximates the value function of the most recent policy).

    The simplest method for learning :math:`V_{\phi}`, used in most implementations of policy optimization algorithms (including VPG, TRPO, PPO, and A2C), is to minimize a mean-squared-error objective:

    .. math:: \phi_k = \arg \min_{\phi} \underE{s_t, \hat{R}_t \sim \pi_k}{\left( V_{\phi}(s_t) - \hat{R}_t \right)^2},

    | 
    where :math:`\pi_k` is the policy at epoch :math:`k`. This is done with one or more steps of gradient descent, starting from the previous value parameters :math:`\phi_{k-1}`. 


Other Forms of the Policy Gradient
==================================

What we have seen so far is that the policy gradient has the general form

.. math::

    \nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \Phi_t},

where :math:`\Phi_t` could be any of

.. math:: \Phi_t &= R(\tau), 

or

.. math:: \Phi_t &= \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}), 

or 

.. math:: \Phi_t &= \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t).

All of these choices lead to the same expected value for the policy gradient, despite having different variances. It turns out that there are two more valid choices of weights :math:`\Phi_t` which are important to know.

**1. On-Policy Action-Value Function.** The choice

.. math:: \Phi_t = Q^{\pi_{\theta}}(s_t, a_t)

is also valid. See `this page`_ for an (optional) proof of this claim.

**2. The Advantage Function.** Recall that the `advantage of an action`_, defined by :math:`A^{\pi}(s_t,a_t) = Q^{\pi}(s_t,a_t) - V^{\pi}(s_t)`,  describes how much better or worse it is than other actions on average (relative to the current policy). This choice,

.. math:: \Phi_t = A^{\pi_{\theta}}(s_t, a_t)

is also valid. The proof is that it's equivalent to using :math:`\Phi_t = Q^{\pi_{\theta}}(s_t, a_t)` and then using a value function baseline, which we are always free to do.

.. admonition:: You Should Know

    The formulation of policy gradients with advantage functions is extremely common, and there are many different ways of estimating the advantage function used by different algorithms.

.. admonition:: You Should Know

    For a more detailed treatment of this topic, you should read the paper on `Generalized Advantage Estimation`_ (GAE), which goes into depth about different choices of :math:`\Phi_t` in the background sections.

    That paper then goes on to describe GAE, a method for approximating the advantage function in policy optimization algorithms which enjoys widespread use. For instance, Spinning Up's implementations of VPG, TRPO, and PPO make use of it. As a result, we strongly advise you to study it.


Recap
=====

In this chapter, we described the basic theory of policy gradient methods and connected some of the early results to code examples. The interested student should continue from here by studying how the later results (value function baselines and the advantage formulation of policy gradients) translate into Spinning Up's implementation of `Vanilla Policy Gradient`_.

.. _`on-policy value function`: ../spinningup/rl_intro.html#value-functions
.. _`advantage of an action`: ../spinningup/rl_intro.html#advantage-functions
.. _`this page`: ../spinningup/extra_pg_proof2.html
.. _`Generalized Advantage Estimation`: https://arxiv.org/abs/1506.02438
.. _`Vanilla Policy Gradient`: ../algorithms/vpg.html