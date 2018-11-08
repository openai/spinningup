=========
MPI Tools
=========

.. contents:: Table of Contents

Core MPI Utilities
==================

.. automodule:: spinup.utils.mpi_tools
    :members:


MPI + Tensorflow Utilities
==========================

The ``spinup.utils.mpi_tf`` contains a a few tools to make it easy to use the AdamOptimizer across many MPI processes. This is a bit hacky---if you're looking for something more sophisticated and general-purpose, consider `horovod`_.

.. _`horovod`: https://github.com/uber/horovod

.. automodule:: spinup.utils.mpi_tf
    :members: