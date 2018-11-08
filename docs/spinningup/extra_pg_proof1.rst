==============
Extra Material
==============

Proof for Don't Let the Past Distract You
=========================================

In this subsection, we will prove that actions should not be reinforced for rewards obtained in the past.

Expand out :math:`R(\tau)` in the expression for the `simplest policy gradient`_ to obtain:

.. math::

    \nabla_{\theta} J(\pi_{\theta}) &= \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)} \\
    &= \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \sum_{t'=0}^T R(s_{t'}, a_{t'}, s_{t'+1})} \\
    &= \sum_{t=0}^{T} \sum_{t'=0}^T  \underE{\tau \sim \pi_{\theta}}{\nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(s_{t'}, a_{t'}, s_{t'+1})},

and consider the term

.. math::

    \underE{\tau \sim \pi_{\theta}}{f(t,t')} = \underE{\tau \sim \pi_{\theta}}{\nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(s_{t'}, a_{t'}, s_{t'+1})}.

We will show that for the case of :math:`t' < t` (the reward comes before the action being reinforced), this term is zero. This is a complete proof of the original claim, because after dropping terms with :math:`t' < t` from the expression, we are left with the reward-to-go form of the policy gradient, as desired:

.. math::

    \nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1})}

**1. Using the Marginal Distribution.** To proceed, we have to break down the expectation in :math:`\underE{\tau \sim \pi_{\theta}}{f(t,t')}`. It's an expectation over trajectories, but the expression inside the expectation only deals with a few states and actions: :math:`s_t`, :math:`a_t`, :math:`s_{t'}`, :math:`a_{t'}`, and :math:`s_{t'+1}`. So in computing the expectation, we only need to worry about the `marginal distribution`_ over these random variables. 

We derive:

.. math:: 

    \underE{\tau \sim \pi_{\theta}}{f(t,t')} &= \int_{\tau} P(\tau|\pi_{\theta}) f(t,t') \\
    &= \int_{s_t, a_t, s_{t'}, a_{t'}, s_{t'+1}} P(s_t, a_t, s_{t'}, a_{t'}, s_{t'+1} | \pi_{\theta}) f(t,t') \\
    &= \underE{s_t, a_t, s_{t'}, a_{t'}, s_{t'+1} \sim \pi_{\theta}}{f(t,t')}.

**2. Probability Chain Rule.** Joint distributions can be calculated in terms of conditional and marginal probabilities via `chain rule of probability`_: :math:`P(A,B) = P(B|A) P(A)`. Here, we use this rule to compute

.. math::

    P(s_t, a_t, s_{t'}, a_{t'}, s_{t'+1} | \pi_{\theta}) = P(s_t, a_t | \pi_{\theta}, s_{t'}, a_{t'}, s_{t'+1}) P(s_{t'}, a_{t'}, s_{t'+1} | \pi_{\theta})


**3. Separating Expectations Over Multiple Random Variables.** If we have an expectation over two random variables :math:`A` and :math:`B`, we can split it into an inner and outer expectation, where the inner expectation treats the variable from the outer expectation as a constant. Our ability to make this split relies on probability chain rule. Mathematically:

.. math::

    \underE{A,B}{f(A,B)} &= \int_{A,B} P(A,B) f(A,B) \\
    &= \int_{A} \int_B P(B|A) P(A) f(A,B) \\
    &= \int_A P(A) \int_B P(B|A) f(A,B) \\
    &= \int_A P(A) \underE{B}{f(A,B) \Big| A} \\
    &= \underE{A}{\underE{B}{f(A,B) \Big| A} }

An expectation over :math:`s_t, a_t, s_{t'}, a_{t'}, s_{t'+1}` can thus be expressed by

.. math:: 

    \underE{\tau \sim \pi_{\theta}}{f(t,t')} &= \underE{s_t, a_t, s_{t'}, a_{t'}, s_{t'+1} \sim \pi_{\theta}}{f(t,t')} \\
    &= \underE{s_{t'}, a_{t'}, s_{t'+1} \sim \pi_{\theta}}{\underE{s_t, a_t \sim \pi_{\theta}}{f(t,t') \Big| s_{t'}, a_{t'}, s_{t'+1}}}

**4. Constants Can Be Pulled Outside of Expectations.** If a term inside an expectation is constant with respect to the variable being expected over, it can be pulled outside of the expectation. To give an example, consider again an expectation over two random variables :math:`A` and :math:`B`, where this time, :math:`f(A,B) = h(A) g(B)`. Then, using the result from before:

.. math::

    \underE{A,B}{f(A,B)} &= \underE{A}{\underE{B}{f(A,B) \Big| A}} \\
    &= \underE{A}{\underE{B}{h(A) g(B) \Big| A}}\\
    &= \underE{A}{h(A) \underE{B}{g(B) \Big| A}}.

The function in our expectation decomposes this way, allowing us to write:

.. math:: 

    \underE{\tau \sim \pi_{\theta}}{f(t,t')} &= \underE{s_{t'}, a_{t'}, s_{t'+1} \sim \pi_{\theta}}{\underE{s_t, a_t \sim \pi_{\theta}}{f(t,t') \Big| s_{t'}, a_{t'}, s_{t'+1}}} \\
    &= \underE{s_{t'}, a_{t'}, s_{t'+1} \sim \pi_{\theta}}{\underE{s_t, a_t \sim \pi_{\theta}}{\nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(s_{t'}, a_{t'}, s_{t'+1}) \Big| s_{t'}, a_{t'}, s_{t'+1}}} \\
    &= \underE{s_{t'}, a_{t'}, s_{t'+1} \sim \pi_{\theta}}{R(s_{t'}, a_{t'}, s_{t'+1})  \underE{s_t, a_t \sim \pi_{\theta}}{\nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \Big| s_{t'}, a_{t'}, s_{t'+1}}}.

**5. Applying the EGLP Lemma.** The last step in our proof relies on the `EGLP lemma`_. At this point, we will only worry about the innermost expectation, 

.. math::

    \underE{s_t, a_t \sim \pi_{\theta}}{\nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \Big| s_{t'}, a_{t'}, s_{t'+1}} = \int_{s_t, a_t} P(s_t, a_t | \pi_{\theta}, s_{t'}, a_{t'}, s_{t'+1}) \nabla_{\theta} \log \pi_{\theta}(a_t |s_t).

We now have to make a distinction between two cases: :math:`t' < t`, the case where the reward happened before the action, and :math:`t' \geq t`, where it didn't.

**Case One: Reward Before Action.** If :math:`t' < t`, then the conditional probabilities for actions at :math:`a_t` come from the policy:

.. math::

    P(s_t, a_t | \pi_{\theta}, s_{t'}, a_{t'}, s_{t'+1}) &= \pi_{\theta}(a_t | s_t) P(s_t | \pi_{\theta}, s_{t'}, a_{t'}, s_{t'+1}),

the innermost expectation can be broken down farther into

.. math::

    \underE{s_t, a_t \sim \pi_{\theta}}{\nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \Big| s_{t'}, a_{t'}, s_{t'+1}} &= \int_{s_t, a_t} P(s_t, a_t | \pi_{\theta}, s_{t'}, a_{t'}, s_{t'+1}) \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \\
    &= \int_{s_t} P(s_t | \pi_{\theta}, s_{t'}, a_{t'}, s_{t'+1}) \int_{a_t} \pi_{\theta}(a_t | s_t) \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \\
    &= \underE{s_t \sim \pi_{\theta}}{ \underE{a_t \sim \pi_{\theta}}{\nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \Big| s_t } \Big| s_{t'}, a_{t'}, s_{t'+1}}.

The EGLP lemma says that 

.. math::

    \underE{a_t \sim \pi_{\theta}}{\nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \Big| s_t } = 0,

allowing us to conclude that for :math:`t' < t`, :math:`\underE{\tau \sim \pi_{\theta}}{f(t,t')} = 0`. 

**Case Two: Reward After Action.** What about the :math:`t' \geq t` case, though? Why doesn't the same logic apply? In this case, the conditional probabilities for :math:`a_t` can't be broken down the same way, because you're conditioning **on the future.** Think about it like this: let's say that every day, in the morning, you make a choice between going for a jog and going to work early, and you have a 50-50 chance of each option. If you condition on a future where you went to work early, what are the odds that you went for a jog? Clearly, you didn't. But if you're conditioning on the past---before you made the decision---what are the odds that you will later go for a jog? Now it's back to 50-50. 

So in the case where :math:`t' \geq t`, the conditional distribution over actions :math:`a_t` is **not** :math:`\pi(a_t|s_t)`, and the EGLP lemma does not apply. 

.. _`simplest policy gradient`: ../spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient
.. _`marginal distribution`: https://en.wikipedia.org/wiki/Marginal_distribution
.. _`chain rule of probability`: https://en.wikipedia.org/wiki/Chain_rule_(probability)
.. _`EGLP lemma`: ../spinningup/rl_intro3.html#expected-grad-log-prob-lemma