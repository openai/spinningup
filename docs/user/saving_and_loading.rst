==================
Experiment Outputs
==================

.. contents:: Table of Contents

In this section we'll cover

- what outputs come from Spinning Up algorithm implementations,
- what formats they're stored in and how they're organized,
- where they are stored and how you can change that,
- and how to load and run trained policies.

.. admonition:: You Should Know
    
    Spinning Up implementations currently have no way to resume training for partially-trained agents. If you consider this feature important, please let us know---or consider it a hacking project!

Algorithm Outputs
=================

Each algorithm is set up to save a training run's hyperparameter configuration, learning progress, trained agent and value functions, and a copy of the environment if possible (to make it easy to load up the agent and environment simultaneously). The output directory contains the following:

+-------------------------------------------------------------------------------------------------+
| **Output Directory Structure**                                                                  |
+---------------------------------+---------------------------------------------------------------+
|``simple_save/``                 | | A directory containing everything needed to restore the     |
|                                 | | trained agent and value functions. (`Details below.`_)      |
+---------------------------------+---------------------------------------------------------------+
|``config.json``                  | | A dict containing an as-complete-as-possible description    |
|                                 | | of the args and kwargs you used to launch the training      |
|                                 | | function. If you passed in something which can't be         |
|                                 | | serialized to JSON, it should get handled gracefully by the |
|                                 | | logger, and the config file will represent it with a string.|
|                                 | | Note: this is meant for record-keeping only. Launching an   |
|                                 | | experiment from a config file is not currently supported.   |
+---------------------------------+---------------------------------------------------------------+
|``progress.txt``                 | | A tab-separated value file containing records of the metrics|
|                                 | | recorded by the logger throughout training. eg, ``Epoch``,  |
|                                 | | ``AverageEpRet``, etc.                                      |
+---------------------------------+---------------------------------------------------------------+
|``vars.pkl``                     | | A pickle file containing anything about the algorithm state |
|                                 | | which should get stored. Currently, all algorithms only use |
|                                 | | this to save a copy of the environment.                     |
+---------------------------------+---------------------------------------------------------------+
|``events.out.tfevents.time.user``| | A tensorboard event file with algorithm dependent           |
|                                 | | performance and loss values. Visualize by running           |
|                                 | | ``tensorboard --logdir path/to/logs`` in the command line.  |
+---------------------------------+---------------------------------------------------------------+

.. admonition:: You Should Know

    Sometimes environment-saving fails because the environment can't be pickled, and ``vars.pkl`` is empty. This is known to be a problem for Gym Box2D environments in older versions of Gym, which can't be saved in this manner.

.. _`Details below.`:

The ``simple_save`` directory contains:

+----------------------------------------------------------------------------------+
| **Simple_Save Directory Structure**                                              |
+------------------+---------------------------------------------------------------+
|``variables/``    | | A directory containing outputs from the Tensorflow Saver.   |
|                  | | See documentation for `Tensorflow SavedModel`_.             |
+------------------+---------------------------------------------------------------+
|``model_info.pkl``| | A dict containing information (map from key to tensor name) |
|                  | | which helps us unpack the saved model after loading.        |
+------------------+---------------------------------------------------------------+
|``saved_model.pb``| | A protocol buffer, needed for a `Tensorflow SavedModel`_.   |
+------------------+---------------------------------------------------------------+

.. admonition:: You Should Know

    The only file in here that you should ever have to use "by hand" is the ``config.json`` file. Our agent testing utility will load things from the ``simple_save/`` directory and ``vars.pkl`` file, and our plotter interprets the contents of ``progress.txt``, and those are the correct tools for interfacing with these outputs. But there is no tooling for ``config.json``---it's just there so that if you forget what hyperparameters you ran an experiment with, you can double-check.


.. _`Tensorflow SavedModel`: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md


Save Directory Location
=======================

Experiment results will, by default, be saved in the same directory as the Spinning Up package, in a folder called ``data``:

.. parsed-literal::

    spinningup/
        **data/**
            ...
        docs/
            ...
        spinup/
            ...
        LICENSE
        setup.py

You can change the default results directory by modifying ``DEFAULT_DATA_DIR`` in ``spinup/user_config.py``. 


Loading and Running Trained Policies
====================================


If Environment Saves Successfully
---------------------------------

For cases where the environment is successfully saved alongside the agent, it's a cinch to watch the trained agent act in the environment using:


.. parsed-literal::

    python -m spinup.run test_policy path/to/output_directory


There are a few flags for options:


.. option:: -l L, --len=L, default=0

    *int*. Maximum length of test episode / trajectory / rollout. The default of 0 means no maximum episode length---episodes only end when the agent has reached a terminal state in the environment. (Note: setting L=0 will not prevent Gym envs wrapped by TimeLimit wrappers from ending when they reach their pre-set maximum episode length.)

.. option:: -n N, --episodes=N, default=100

    *int*. Number of test episodes to run the agent for.

.. option:: -nr, --norender

    Do not render the test episodes to the screen. In this case, ``test_policy`` will only print the episode returns and lengths. (Use case: the renderer slows down the testing process, and you just want to get a fast sense of how the agent is performing, so you don't particularly care to watch it.)

.. option:: -i I, --itr=I, default=-1

    *int*. This is an option for a special case which is not supported by algorithms in this package as-shipped, but which they are easily modified to do. Use case: Sometimes it's nice to watch trained agents from many different points in training (eg watch at iteration 50, 100, 150, etc.). The logger can do this---save snapshots of the agent from those different points, so they can be run and watched later. In this case, you use this flag to specify which iteration to run. But again: spinup algorithms by default only save snapshots of the most recent agent, overwriting the old snapshots. 

    The default value of this flag means "use the latest snapshot."

    To modify an algo so it does produce multiple snapshots, find the following lines (which are present in all of the algorithms):

    .. code-block:: python

        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

    and tweak them to

    .. code-block:: python

        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, epoch)

    Make sure to then also set ``save_freq`` to something reasonable (because if it defaults to 1, for instance, you'll flood your output directory with one ``simple_save`` folder for each snapshot---which adds up fast).


.. option:: -d, --deterministic

    Another special case, which is only used for SAC. The Spinning Up SAC implementation trains a stochastic policy, but is evaluated using the deterministic *mean* of the action distribution. ``test_policy`` will default to using the stochastic policy trained by SAC, but you should set the deterministic flag to watch the deterministic mean policy (the correct evaluation policy for SAC). This flag is not used for any other algorithms.



Environment Not Found Error
---------------------------

If the environment wasn't saved successfully, you can expect ``test_policy.py`` to crash with

.. parsed-literal::

    Traceback (most recent call last):
      File "spinup/utils/test_policy.py", line 88, in <module>
        run_policy(env, get_action, args.len, args.episodes, not(args.norender))
      File "spinup/utils/test_policy.py", line 50, in run_policy
        "page on Experiment Outputs for how to handle this situation."
    AssertionError: Environment not found!

     It looks like the environment wasn't saved, and we can't run the agent in it. :( 

     Check out the readthedocs page on Experiment Outputs for how to handle this situation.


In this case, watching your agent perform is slightly more of a pain but not impossible, as long as you can recreate your environment easily. Try the following in IPython:

>>> from spinup.utils.test_policy import load_policy, run_policy
>>> import your_env
>>> _, get_action = load_policy('/path/to/output_directory')
>>> env = your_env.make()
>>> run_policy(env, get_action)
Logging data to /tmp/experiments/1536150702/progress.txt
Episode 0    EpRet -163.830      EpLen 93
Episode 1    EpRet -346.164      EpLen 99
...


Using Trained Value Functions
-----------------------------

The ``test_policy.py`` tool doesn't help you look at trained value functions, and if you want to use those, you will have to do some digging by hand. Check the documentation for the `restore_tf_graph`_ function for details on how.

.. _`restore_tf_graph`: ../utils/logger.html#spinup.utils.logx.restore_tf_graph