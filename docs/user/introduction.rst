============
Introduction
============

.. contents:: Table of Contents

What This Is
============

Welcome to Spinning Up in Deep RL! This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: `reinforcement learning`_ (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with `deep learning`_. 

This module contains a variety of helpful resources, including:

- a short `introduction`_ to RL terminology, kinds of algorithms, and basic theory,
- an `essay`_ about how to grow into an RL research role,
- a `curated list`_ of important papers organized by topic,
- a well-documented `code repo`_ of short, standalone implementations of key algorithms,
- and a few `exercises`_ to serve as warm-ups.


.. _`reinforcement learning`: https://en.wikipedia.org/wiki/Reinforcement_learning
.. _`deep learning`: http://ufldl.stanford.edu/tutorial/

Why We Built This
=================

One of the single most common questions that we hear is 

    | If I want to contribute to AI safety, how do I get started?

At OpenAI, we believe that deep learning generally---and deep reinforcement learning specifically---will play central roles in the development of powerful AI technology. To ensure that AI is safe, we have to come up with safety strategies and algorithms that are compatible with this paradigm. As a result, we encourage everyone who asks this question to study these fields.

However, while there are many resources to help people quickly ramp up on deep learning, deep reinforcement learning is more challenging to break into. To begin with, a student of deep RL needs to have some background in math, coding, and regular deep learning. Beyond that, they need both a high-level view of the field---an awareness of what topics are studied in it, why they matter, and what's been done already---and careful instruction on how to connect algorithm theory to algorithm code. 

The high-level view is hard to come by because of how new the field is. There is not yet a standard deep RL textbook, so most of the knowledge is locked up in either papers or lecture series, which can take a long time to parse and digest. And learning to implement deep RL algorithms is typically painful, because either 

- the paper that publishes an algorithm omits or inadvertently obscures key design details,
- or widely-public implementations of an algorithm are hard to read, hiding how the code lines up with the algorithm.

While fantastic repos like garage_, Baselines_, and rllib_ make it easier for researchers who are already in the field to make progress, they build algorithms into frameworks in ways that involve many non-obvious choices and trade-offs, which makes them hard to learn from. Consequently, the field of deep RL has a pretty high barrier to entry---for new researchers as well as practitioners and hobbyists. 

So our package here is designed to serve as the missing middle step for people who are excited by deep RL, and would like to learn how to use it or make a contribution, but don't have a clear sense of what to study or how to transmute algorithms into code. We've tried to make this as helpful a launching point as possible.

That said, practitioners aren't the only people who can (or should) benefit from these materials. Solving AI safety will require people with a wide range of expertise and perspectives, and many relevant professions have no connection to engineering or computer science at all. Nonetheless, everyone involved will need to learn enough about the technology to make informed decisions, and several pieces of Spinning Up address that need. 



How This Serves Our Mission
===========================

OpenAI's mission_ is to ensure the safe development of AGI and the broad distribution of benefits from AI more generally. Teaching tools like Spinning Up help us make progress on both of these objectives. 

To begin with, we move closer to broad distribution of benefits any time we help people understand what AI is and how it works. This empowers people to think critically about the many issues we anticipate will arise as AI becomes more sophisticated and important in our lives.

Also, critically, `we need people to help <https://jobs.lever.co/openai>`_ us work on making sure that AGI is safe. This requires a skill set which is currently in short supply because of how new the field is. We know that many people are interested in helping us, but don't know how---here is what you should study! If you can become an expert on this material, you can make a difference on AI safety.



Code Design Philosophy
======================

The algorithm implementations in the Spinning Up repo are designed to be 

    - as simple as possible while still being reasonably good, 
    - and highly-consistent with each other to expose fundamental similarities between algorithms.

They are almost completely self-contained, with virtually no common code shared between them (except for logging, saving, loading, and `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_ utilities), so that an interested person can study each algorithm separately without having to dig through an endless chain of dependencies to see how something is done. The implementations are patterned so that they come as close to pseudocode as possible, to minimize the gap between theory and code. 

Importantly, they're all structured similarly, so if you clearly understand one, jumping into the next is painless. 

We tried to minimize the number of tricks used in each algorithm's implementation, and minimize the differences between otherwise-similar algorithms. To give some examples of removed tricks: we omit regularization_ terms present in the original Soft-Actor Critic code, as well as `observation normalization`_ from all algorithms. For an example of where we've removed differences between algorithms: our implementations of DDPG, TD3, and SAC all follow a convention of running gradient descent updates after fixed intervals of environment interaction. (By contrast, other public implementations of these algorithms usually take slightly different approaches from each other, making them a little bit harder to compare.)

All algorithms are "reasonably good" in the sense that they achieve roughly the intended performance, but don't necessarily match the best reported results in the literature on every task. Consequently, be careful if using any of these implementations for scientific benchmarking comparisons. Details on each implementation's specific performance level can be found on our `benchmarks`_ page.


Long-Term Support and Support History
=====================================

Spinning Up is currently in maintenance mode. If there are any breaking bugs, we'll repair them to ensure that Spinning Up can continue to help people study deep RL. 

Support history so far:

- **Nov 8, 2018:** Initial release!

- **Nov, 2018:** Release was followed by a three-week period of high-bandwidth support.

- **April, 2019:** Approximately six months after release, we conducted an internal review of Spinning Up based on feedback from the community. The review surfaced interest in a few key features:

    * **Implementations in Other Neural Network Libraries.** Several people expressed interest in seeing Spinning Up use alternatives to Tensorflow v1 for the RL implementations. A few members of the community even developed their own PyTorch versions of Spinning Up algorithms, such as Kashif Rasul's `Fired Up`_,  Kai Arulkumaran's `Spinning Up Basic`_, and Misha Laskin's `Torching Up`_. As a result, making this kind of "Rosetta Stone" for deep RL became a high priority for future work.

    * **Open Source RL Environments.** Many people expressed an interest in seeing Spinning Up use more open source RL environments (eg `PyBullet`_) for benchmarks, examples, and exercises.

    * **More Algorithms.** There was some interest in seeing other algorithms included in Spinning Up, especially Deep Q-Networks.

- **Jan, 2020:** The PyTorch update to Spinning Up was released! 

- **Future:** No major updates are currently planned for Spinning Up. In the event it makes sense for us to release an additional update, following what we found in the 6-month review, the next-highest priority features are to focus more on open source RL environments and adding algorithms.

Additionally, as discussed in the blog post, Spinning Up has been integrated into the curriculum for our Scholars_ and Fellows_ programs.


.. _`introduction`: ../spinningup/rl_intro.html
.. _`essay`: ../spinningup/spinningup.html
.. _`Spinning Up essay`: ../spinningup/spinningup.html
.. _`curated list`: ../spinningup/keypapers.html
.. _`code repo`: https://github.com/openai/spinningup
.. _`exercises`: ../spinningup/exercises.html
.. _`garage`: https://github.com/rlworkgroup/garage
.. _`Baselines`: https://github.com/openai/baselines
.. _`rllib`: https://github.com/ray-project/ray/tree/master/python/ray/rllib
.. _`mission`: https://blog.openai.com/openai-charter/
.. _`regularization`: https://github.com/haarnoja/sac/blob/108a4229be6f040360fcca983113df9c4ac23a6a/sac/distributions/normal.py#L69
.. _`observation normalization`: https://github.com/openai/baselines/blob/28aca637d0f13f4415cc5ebb778144154cff3110/baselines/run.py#L131
.. _`original TD3 code`: https://github.com/sfujim/TD3/blob/25dfc0a6562c54ae5575fad5b8f08bc9d5c4e26c/main.py#L89
.. _`benchmarks`: ../spinningup/bench.html
.. _`Fired Up`: https://github.com/kashif/firedup
.. _`Spinning Up Basic`: https://github.com/Kaixhin/spinning-up-basic
.. _`Torching Up`: https://github.com/MishaLaskin/torchingup
.. _`PyBullet`: https://pybullet.org/wordpress/
.. _`MuJoCo`: http://mujoco.org/
.. _Scholars : https://openai.com/blog/openai-scholars-spring-2020/
.. _Fellows : https://openai.com/blog/openai-fellows-fall-2018/