===================
Running Experiments
===================


.. contents:: Table of Contents

One of the best ways to get a feel for deep RL is to run the algorithms and see how they perform on different tasks. The Spinning Up code library makes small-scale (local) experiments easy to do, and in this section, we'll discuss two ways to run them: either from the command line, or through function calls in scripts.


Launching from the Command Line
===============================


Spinning Up ships with ``spinup/run.py``, a convenient tool that lets you easily launch any algorithm (with any choices of hyperparameters) from the command line. It also serves as a thin wrapper over the utilities for watching trained policies and plotting, although we will not discuss that functionality on this page (for those details, see the pages on `experiment outputs`_ and `plotting`_).

The standard way to run a Spinning Up algorithm from the command line is

.. parsed-literal::

    python -m spinup.run [algo name] [experiment flags]

eg:

.. parsed-literal::

    python -m spinup.run ppo --env Walker2d-v2 --exp_name walker

.. _`experiment outputs`: ../user/saving_and_loading.html
.. _`plotting`: ../user/plotting.html

.. admonition:: You Should Know

    If you are using ZShell: ZShell interprets square brackets as special characters. Spinning Up uses square brackets in a few ways for command line arguments; make sure to escape them, or try the solution recommended `here <http://kinopyo.com/en/blog/escape-square-bracket-by-default-in-zsh>`_ if you want to escape them by default.

.. admonition:: Detailed Quickstart Guide

    .. parsed-literal::

        python -m spinup.run ppo --exp_name ppo_ant --env Ant-v2 --clip_ratio 0.1 0.2 
            --hid[h] [32,32] [64,32] --act tf.nn.tanh --seed 0 10 20 --dt
            --data_dir path/to/data

    runs PPO in the ``Ant-v2`` Gym environment, with various settings controlled by the flags.

    ``clip_ratio``, ``hid``, and ``act`` are flags to set some algorithm hyperparameters. You can provide multiple values for hyperparameters to run multiple experiments. Check the docs to see what hyperparameters you can set (click here for the `PPO documentation`_).

    ``hid`` and ``act`` are `special shortcut flags`_ for setting the hidden sizes and activation function for the neural networks trained by the algorithm.

    The ``seed`` flag sets the seed for the random number generator. RL algorithms have high variance, so try multiple seeds to get a feel for how performance varies.

    The ``dt`` flag ensures that the save directory names will have timestamps in them (otherwise they don't, unless you set ``FORCE_DATESTAMP=True`` in ``spinup/user_config.py``).

    The ``data_dir`` flag allows you to set the save folder for results. The default value is set by ``DEFAULT_DATA_DIR`` in ``spinup/user_config.py``, which will be a subfolder ``data`` in the ``spinningup`` folder (unless you change it).

    `Save directory names`_ are based on ``exp_name`` and any flags which have multiple values. Instead of the full flag, a shorthand will appear in the directory name. Shorthands can be provided by the user in square brackets after the flag, like ``--hid[h]``; otherwise, shorthands are substrings of the flag (``clip_ratio`` becomes ``cli``). To illustrate, the save directory for the run with ``clip_ratio=0.1``, ``hid=[32,32]``, and ``seed=10`` will be:

    .. parsed-literal::

        path/to/data/YY-MM-DD_ppo_ant_cli0-1_h32-32/YY-MM-DD_HH-MM-SS-ppo_ant_cli0-1_h32-32_seed10

.. _`PPO documentation`: ../algorithms/ppo.html#spinup.ppo
.. _`special shortcut flags`: ../user/running.html#shortcut-flags
.. _`Save directory names`: ../user/running.html#where-results-are-saved

Setting Hyperparameters from the Command Line
---------------------------------------------

Every hyperparameter in every algorithm can be controlled directly from the command line. If ``kwarg`` is a valid keyword arg for the function call of an algorithm, you can set values for it with the flag ``--kwarg``. To find out what keyword args are available, see either the docs page for an algorithm, or try

.. parsed-literal::

    python -m spinup.run [algo name] --help

to see a readout of the docstring.

.. admonition:: You Should Know

    Values pass through ``eval()`` before being used, so you can describe some functions and objects directly from the command line. For example:

    .. parsed-literal::

        python -m spinup.run ppo --env Walker2d-v2 --exp_name walker --act tf.nn.elu

    sets ``tf.nn.elu`` as the activation function.

.. admonition:: You Should Know

    There's some nice handling for kwargs that take dict values. Instead of having to provide

    .. parsed-literal::

        --key dict(v1=value_1, v2=value_2)

    you can give

    .. parsed-literal::

        --key:v1 value_1 --key:v2 value_2 

    to get the same result.

Launching Multiple Experiments at Once
--------------------------------------

You can launch multiple experiments, to be executed **in series**, by simply providing more than one value for a given argument. (An experiment for each possible combination of values will be launched.)

For example, to launch otherwise-equivalent runs with different random seeds (0, 10, and 20), do:

.. parsed-literal::

    python -m spinup.run ppo --env Walker2d-v2 --exp_name walker --seed 0 10 20

Experiments don't launch in parallel because they soak up enough resources that executing several at the same time wouldn't get a speedup.



Special Flags
-------------

A few flags receive special treatment.


Environment Flag
^^^^^^^^^^^^^^^^

.. option:: --env, --env_name

    *string*. The name of an environment in the OpenAI Gym. All Spinning Up algorithms are implemented as functions that accept ``env_fn`` as an argument, where ``env_fn`` must be a callable function that builds a copy of the RL environment. Since the most common use case is Gym environments, though, all of which are built through ``gym.make(env_name)``, we allow you to just specify ``env_name`` (or ``env`` for short) at the command line, which gets converted to a lambda-function that builds the correct gym environment.


Shortcut Flags
^^^^^^^^^^^^^^

Some algorithm arguments are relatively long, and we enabled shortcuts for them: 

.. option:: --hid, --ac_kwargs:hidden_sizes

    *list of ints*. Sets the sizes of the hidden layers in the neural networks (policies and value functions). 

.. option:: --act, --ac_kwargs:activation

    *tf op*. The activation function for the neural networks in the actor and critic.

These flags are valid for all current Spinning Up algorithms.

Config Flags
^^^^^^^^^^^^

These flags are not hyperparameters of any algorithm, but change the experimental configuration in some way.

.. option:: --cpu, --num_cpu

    *int*. If this flag is set, the experiment is launched with this many processes, one per cpu, connected by MPI. Some algorithms are amenable to this sort of parallelization but not all. An error will be raised if you try setting ``num_cpu`` > 1 for an incompatible algorithm. You can also set ``--num_cpu auto``, which will automatically use as many CPUs as are available on the machine.

.. option:: --exp_name

    *string*. The experiment name. This is used in naming the save directory for each experiment. The default is "cmd" + [algo name].

.. option:: --data_dir

    *path*. Set the base save directory for this experiment or set of experiments. If none is given, the ``DEFAULT_DATA_DIR`` in ``spinup/user_config.py`` will be used.

.. option:: --datestamp

    *bool*. Include date and time in the name for the save directory of the experiment.


Where Results are Saved
-----------------------

Results for a particular experiment (a single run of a configuration of hyperparameters) are stored in

::

    data_dir/[outer_prefix]exp_name[suffix]/[inner_prefix]exp_name[suffix]_s[seed]

where 

* ``data_dir`` is the value of the ``--data_dir`` flag (defaults to ``DEFAULT_DATA_DIR`` from ``spinup/user_config.py`` if ``--data_dir`` is not given), 
* the ``outer_prefix`` is a ``YY-MM-DD_`` timestamp if the ``--datestamp`` flag is raised, otherwise nothing,
* the ``inner_prefix`` is a ``YY-MM-DD_HH-MM-SS-`` timestamp if the ``--datestamp`` flag is raised, otherwise nothing,
* and ``suffix`` is a special string based on the experiment hyperparameters.

How is Suffix Determined?
^^^^^^^^^^^^^^^^^^^^^^^^^

Suffixes are only included if you run multiple experiments at once, and they only include references to hyperparameters that differ across experiments, except for random seed. The goal is to make sure that results for similar experiments (ones which share all params except seed) are grouped in the same folder.

Suffixes are constructed by combining *shorthands* for hyperparameters with their values, where a shorthand is either 1) constructed automatically from the hyperparameter name or 2) supplied by the user. The user can supply a shorthand by writing in square brackets after the kwarg flag. 

For example, consider:

.. parsed-literal::

    python -m spinup.run ddpg --env Hopper-v2 --hid[h] [300] [128,128] --act tf.nn.tanh tf.nn.relu

Here, the ``--hid`` flag is given a **user-supplied shorthand**, ``h``. The ``--act`` flag is not given a shorthand by the user, so one will be constructed for it automatically.

The suffixes produced in this case are:

.. parsed-literal::
    _h128-128_ac-actrelu
    _h128-128_ac-acttanh
    _h300_ac-actrelu
    _h300_ac-acttanh

Note that the ``h`` was given by the user. the ``ac-act`` shorthand was constructed from ``ac_kwargs:activation`` (the true name for the ``act`` flag).


Extra
-----

.. admonition:: You Don't Actually Need to Know This One

    Each individual algorithm is located in a file ``spinup/algos/ALGO_NAME/ALGO_NAME.py``, and these files can be run directly from the command line with a limited set of arguments (some of which differ from what's available to ``spinup/run.py``). The command line support in the individual algorithm files is essentially vestigial, however, and this is **not** a recommended way to perform experiments. 

    This documentation page will not describe those command line calls, and will *only* describe calls through ``spinup/run.py``. 

Launching from Scripts
======================

Each algorithm is implemented as a python function, which can be imported directly from the ``spinup`` package, eg

>>> from spinup import ppo

See the documentation page for each algorithm for a complete account of possible arguments. These methods can be used to set up specialized custom experiments, for example:

.. code-block:: python

    from spinup import ppo
    import tensorflow as tf
    import gym

    env_fn = lambda : gym.make('LunarLander-v2')

    ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)

    logger_kwargs = dict(output_dir='path/to/output_dir', exp_name='experiment_name')

    ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=250, logger_kwargs=logger_kwargs)


Using ExperimentGrid
--------------------

It's often useful in machine learning research to run the same algorithm with many possible hyperparameters. Spinning Up ships with a simple tool for facilitating this, called `ExperimentGrid`_. 


Consider the example in ``spinup/examples/bench_ppo_cartpole.py``:

.. code-block:: python
   :linenos:

    from spinup.utils.run_utils import ExperimentGrid
    from spinup import ppo
    import tensorflow as tf

    if __name__ == '__main__':
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--cpu', type=int, default=4)
        parser.add_argument('--num_runs', type=int, default=3)
        args = parser.parse_args()

        eg = ExperimentGrid(name='ppo-bench')
        eg.add('env_name', 'CartPole-v0', '', True)
        eg.add('seed', [10*i for i in range(args.num_runs)])
        eg.add('epochs', 10)
        eg.add('steps_per_epoch', 4000)
        eg.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
        eg.add('ac_kwargs:activation', [tf.tanh, tf.nn.relu], '')
        eg.run(ppo, num_cpu=args.cpu)

After making the ExperimentGrid object, parameters are added to it with

.. parsed-literal::

    eg.add(param_name, values, shorthand, in_name)

where ``in_name`` forces a parameter to appear in the experiment name, even if it has the same value across all experiments.

After all parameters have been added,

.. parsed-literal::

    eg.run(thunk, **run_kwargs)

runs all experiments in the grid (one experiment per valid configuration), by providing the configurations as kwargs to the function ``thunk``. ``ExperimentGrid.run`` uses a function named `call_experiment`_ to launch ``thunk``, and ``**run_kwargs`` specify behaviors for ``call_experiment``. See `the documentation page`_ for details.

Except for the absence of shortcut kwargs (you can't use ``hid`` for ``ac_kwargs:hidden_sizes`` in ``ExperimentGrid``), the basic behavior of ``ExperimentGrid`` is the same as running things from the command line. (In fact, ``spinup.run`` uses an ``ExperimentGrid`` under the hood.)

.. _`ExperimentGrid`: ../utils/run_utils.html#experimentgrid
.. _`the documentation page`: ../utils/run_utils.html#experimentgrid
.. _`call_experiment`: ../utils/run_utils.html#spinup.utils.run_utils.call_experiment