======
Logger
======

.. contents:: Table of Contents

Using a Logger
==============

Spinning Up ships with basic logging tools, implemented in the classes `Logger`_ and `EpochLogger`_. The Logger class contains most of the basic functionality for saving diagnostics, hyperparameter configurations, the state of a training run, and the trained model. The EpochLogger class adds a thin layer on top of that to make it easy to track the average, standard deviation, min, and max value of a diagnostic over each epoch and across MPI workers.

.. admonition:: You Should Know

    All Spinning Up algorithm implementations use an EpochLogger.

.. _`Logger`: ../utils/logger.html#spinup.utils.logx.Logger
.. _`EpochLogger`: ../utils/logger.html#spinup.utils.logx.EpochLogger

Examples
--------

First, let's look at a simple example of how an EpochLogger keeps track of a diagnostic value:

>>> from spinup.utils.logx import EpochLogger
>>> epoch_logger = EpochLogger()
>>> for i in range(10):
        epoch_logger.store(Test=i)
>>> epoch_logger.log_tabular('Test', with_min_and_max=True)
>>> epoch_logger.dump_tabular()
-------------------------------------
|     AverageTest |             4.5 |
|         StdTest |            2.87 |
|         MaxTest |               9 |
|         MinTest |               0 |
-------------------------------------

The ``store`` method is used to save all values of ``Test`` to the ``epoch_logger``'s internal state. Then, when ``log_tabular`` is called, it computes the average, standard deviation, min, and max of ``Test`` over all of the values in the internal state. The internal state is wiped clean after the call to ``log_tabular`` (to prevent leakage into the statistics at the next epoch). Finally, ``dump_tabular`` is called to write the diagnostics to file and to stdout.

Next, let's look at a full training procedure with the logger embedded, to highlight configuration and model saving as well as diagnostic logging:

.. code-block:: python
   :linenos:
   :emphasize-lines: 18, 19, 42, 43, 54, 58, 61, 62, 63, 64, 65, 66

    import numpy as np
    import tensorflow as tf
    import time
    from spinup.utils.logx import EpochLogger


    def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
        for h in hidden_sizes[:-1]:
            x = tf.layers.dense(x, units=h, activation=activation)
        return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


    # Simple script for training an MLP on MNIST.
    def train_mnist(steps_per_epoch=100, epochs=5, 
                    lr=1e-3, layers=2, hidden_size=64, 
                    logger_kwargs=dict(), save_freq=1):

        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())

        # Load and preprocess MNIST data
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28*28) / 255.0

        # Define inputs & main outputs from computation graph
        x_ph = tf.placeholder(tf.float32, shape=(None, 28*28))
        y_ph = tf.placeholder(tf.int32, shape=(None,))
        logits = mlp(x_ph, hidden_sizes=[hidden_size]*layers + [10], activation=tf.nn.relu)
        predict = tf.argmax(logits, axis=1, output_type=tf.int32)

        # Define loss function, accuracy, and training op
        y = tf.one_hot(y_ph, 10)
        loss = tf.losses.softmax_cross_entropy(y, logits)
        acc = tf.reduce_mean(tf.cast(tf.equal(y_ph, predict), tf.float32))
        train_op = tf.train.AdamOptimizer().minimize(loss)

        # Prepare session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Setup model saving
        logger.setup_tf_saver(sess, inputs={'x': x_ph}, 
                                    outputs={'logits': logits, 'predict': predict})

        start_time = time.time()

        # Run main training loop
        for epoch in range(epochs):
            for t in range(steps_per_epoch):
                idxs = np.random.randint(0, len(x_train), 32)
                feed_dict = {x_ph: x_train[idxs],
                             y_ph: y_train[idxs]}
                outs = sess.run([loss, acc, train_op], feed_dict=feed_dict)
                logger.store(Loss=outs[0], Acc=outs[1])

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state(state_dict=dict(), itr=None)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('Acc', with_min_and_max=True)
            logger.log_tabular('Loss', average_only=True)
            logger.log_tabular('TotalGradientSteps', (epoch+1)*steps_per_epoch)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

    if __name__ == '__main__':
        train_mnist()

In this example, observe that

* On line 19, `logger.save_config`_ is used to save the hyperparameter configuration to a JSON file.
* On lines 42 and 43, `logger.setup_tf_saver`_ is used to prepare the logger to save the key elements of the computation graph.
* On line 54, diagnostics are saved to the logger's internal state via `logger.store`_.
* On line 58, the computation graph is saved once per epoch via `logger.save_state`_.
* On lines 61-66, `logger.log_tabular`_ and `logger.dump_tabular`_ are used to write the epoch diagnostics to file. Note that the keys passed into `logger.log_tabular`_ are the same as the keys passed into `logger.store`_.

.. _`logger.save_config`: ../utils/logger.html#spinup.utils.logx.Logger.save_config
.. _`logger.setup_tf_saver`: ../utils/logger.html#spinup.utils.logx.Logger.setup_tf_saver
.. _`logger.store`: ../utils/logger.html#spinup.utils.logx.EpochLogger.store
.. _`logger.save_state`: ../utils/logger.html#spinup.utils.logx.Logger.save_state
.. _`logger.log_tabular`: ../utils/logger.html#spinup.utils.logx.EpochLogger.log_tabular
.. _`logger.dump_tabular`: ../utils/logger.html#spinup.utils.logx.Logger.dump_tabular


Logging and PyTorch
-------------------

The preceding example was given in Tensorflow. For PyTorch, everything is the same except for L42-43: instead of ``logger.setup_tf_saver``, you would use ``logger.setup_pytorch_saver``, and you would pass it `a PyTorch module`_ (the network you are training) as an argument.

The behavior of ``logger.save_state`` is the same as in the Tensorflow case: each time it is called, it'll save the latest version of the PyTorch module.

.. _`a PyTorch module`: https://pytorch.org/docs/stable/nn.html#torch.nn.Module

Logging and MPI
---------------

.. admonition:: You Should Know

    Several algorithms in RL are easily parallelized by using MPI to average gradients and/or other key quantities. The Spinning Up loggers are designed to be well-behaved when using MPI: things will only get written to stdout and to file from the process with rank 0. But information from other processes isn't lost if you use the EpochLogger: everything which is passed into EpochLogger via ``store``, regardless of which process it's stored in, gets used to compute average/std/min/max values for a diagnostic.


Logger Classes
==============


.. autoclass:: spinup.utils.logx.Logger
    :members:

    .. automethod:: spinup.utils.logx.Logger.__init__

.. autoclass:: spinup.utils.logx.EpochLogger
    :show-inheritance:
    :members:


Loading Saved Models (PyTorch Only)
===================================

To load an actor-critic model saved by a PyTorch Spinning Up implementation, run:

.. code-block:: python

    ac = torch.load('path/to/model.pt')

When you use this method to load an actor-critic model, you can minimally expect it to have an ``act`` method that allows you to sample actions from the policy, given observations:

.. code-block:: python

    actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))


Loading Saved Graphs (Tensorflow Only)
======================================

.. autofunction:: spinup.utils.logx.restore_tf_graph

When you use this method to restore a graph saved by a Tensorflow Spinning Up implementation, you can minimally expect it to include the following:

======  ===============================================
Key     Value
======  ===============================================
``x``   Tensorflow placeholder for state input.
``pi``  | Samples an action from the agent, conditioned
        | on states in ``x``.
======  ===============================================

The relevant value functions for an algorithm are also typically stored. For details of what else gets saved by a given algorithm, see its documentation page.