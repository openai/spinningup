=========
MPI Tools
=========

.. contents:: Table of Contents

Core MPI Utilities
==================

.. automodule:: spinup.utils.mpi_tools
    :members:


MPI + PyTorch Utilities
=======================

``spinup.utils.mpi_pytorch`` contains a few tools to make it easy to do data-parallel PyTorch optimization across MPI processes. The two main ingredients are syncing parameters and averaging gradients before they are used by the adaptive optimizer. Also there's a hacky fix for a problem where the PyTorch instance in each separate process tries to get too many threads, and they start to clobber each other.

The pattern for using these tools looks something like this: 

1) At the beginning of the training script, call ``setup_pytorch_for_mpi()``. (Avoids clobbering problem.)

2) After you've constructed a PyTorch module, call ``sync_params(module)``.

3) Then, during gradient descent, call ``mpi_avg_grads`` after the backward pass, like so:

.. code-block:: python

    optimizer.zero_grad()
    loss = compute_loss(module)
    loss.backward()
    mpi_avg_grads(module)   # averages gradient buffers across MPI processes!
    optimizer.step()


.. automodule:: spinup.utils.mpi_pytorch
    :members:

MPI + Tensorflow Utilities
==========================

The ``spinup.utils.mpi_tf`` contains a a few tools to make it easy to use the AdamOptimizer across many MPI processes. This is a bit hacky---if you're looking for something more sophisticated and general-purpose, consider `horovod`_.

.. _`horovod`: https://github.com/uber/horovod

.. automodule:: spinup.utils.mpi_tf
    :members:
