================
Plotting Results
================

Spinning Up ships with a simple plotting utility for interpreting results. Run it with:

.. parsed-literal::

    python -m spinup.run plot [path/to/output_directory ...] [--legend [LEGEND ...]] 
        [--xaxis XAXIS] [--value [VALUE ...]] [--count] [--smooth S]
        [--select [SEL ...]] [--exclude [EXC ...]]


**Positional Arguments:**

.. option:: logdir

    *strings*. As many log directories (or prefixes to log directories, which the plotter will autocomplete internally) as you'd like to plot from. Logdirs will be searched recursively for experiment outputs.

    .. admonition:: You Should Know

        The internal autocompleting is really handy! Suppose you have run several experiments, with the aim of comparing performance between different algorithms, resulting in a log directory structure of:

        .. parsed-literal::

            data/
                bench_algo1/
                    bench_algo1-seed0/
                    bench_algo1-seed10/
                bench_algo2/
                    bench_algo2-seed0/
                    bench_algo2-seed10/

        You can easily produce a graph comparing algo1 and algo2 with:

        .. parsed-literal::

            python spinup/utils/plot.py data/bench_algo

        relying on the autocomplete to find both ``data/bench_algo1`` and ``data/bench_algo2``.

**Optional Arguments:**

.. option:: -l, --legend=[LEGEND ...]

    *strings*. Optional way to specify legend for the plot. The plotter legend will automatically use the ``exp_name`` from the ``config.json`` file, unless you tell it otherwise through this flag. This only works if you provide a name for each directory that will get plotted. (Note: this may not be the same as the number of logdir args you provide! Recall that the plotter looks for autocompletes of the logdir args: there may be more than one match for a given logdir prefix, and you will need to provide a legend string for each one of those matches---unless you have removed some of them as candidates via selection or exclusion rules (below).)

.. option:: -x, --xaxis=XAXIS, default='TotalEnvInteracts'

    *string*. Pick what column from data is used for the x-axis.

.. option:: -y, --value=[VALUE ...], default='Performance'

    *strings*. Pick what columns from data to graph on the y-axis. Submitting multiple values will produce multiple graphs. Defaults to ``Performance``, which is not an actual output of any algorithm. Instead, ``Performance`` refers to either ``AverageEpRet``, the correct performance measure for the on-policy algorithms, or ``AverageTestEpRet``, the correct performance measure for the off-policy algorithms. The plotter will automatically figure out which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for each separate logdir.

.. option:: --count

    Optional flag. By default, the plotter shows y-values which are averaged across all results that share an ``exp_name``, which is typically a set of identical experiments that only vary in random seed. But if you'd like to see all of those curves separately, use the ``--count`` flag.

.. option:: -s, --smooth=S, default=1
    
    *int*. Smooth data by averaging it over a fixed window. This parameter says how wide the averaging window will be.

.. option:: --select=[SEL ...]

    *strings*. Optional selection rule: the plotter will only show curves from logdirs that contain all of these substrings.

.. option:: --exclude=[EXC ...]

    *strings*. Optional exclusion rule: plotter will only show curves from logdirs that do not contain these substrings.
