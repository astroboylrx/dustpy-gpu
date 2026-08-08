Overview
========

DustPy-GPU is a drop-in, GPU-enabled fork of the `upstream DustPy repository
<https://github.com/stammler/dustpy>`_. Existing programs continue to import
``dustpy`` and use the standard model API. GPU acceleration is selected when a
simulation is created::

   import dustpy

   sim = dustpy.Simulation(backend="cupy")

The default NumPy backend preserves upstream behavior. The CuPy backend moves
the main dust-evolution calculations to a CUDA GPU and returns CuPy arrays from
simulation fields.

Scope
-----

This site documents the fork-specific installation, backend selection, GPU
runtime behavior, adapted examples, and validation results. The complete model
description and upstream API remain documented by the `upstream DustPy documentation
<https://stammler.github.io/dustpy/>`_ and `Simframe
<https://simframe.readthedocs.io/>`_.

Citation
--------

Please cite `Stammler & Birnstiel (2022)
<https://doi.org/10.3847/1538-4357/ac7d58>`_ for DustPy and `Li & Chiang
(2026) <https://ui.adsabs.harvard.edu/abs/2026arXiv260614704L>`_ for
DustPy-GPU.
