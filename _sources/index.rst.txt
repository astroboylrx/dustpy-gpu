DustPy-GPU Documentation
========================

``DustPy-GPU`` is a GPU-enabled fork of `DustPy <https://github.com/stammler/dustpy>`_
for simulations of gas and dust evolution in protoplanetary disks. It preserves
the ``dustpy`` import name and upstream API while adding per-simulation NumPy and
CuPy backend selection.

This documentation covers the GPU fork, backend behavior, adapted examples,
numerical parity, and performance. Refer to the `upstream DustPy documentation
<https://stammler.github.io/dustpy/>`_ for the complete physics and standard API
reference.

.. toctree::
   :maxdepth: 2
   :caption: DustPy-GPU

   overview
   installation
   backends
   Backend Parity and Performance <backend_parity_and_benchmarks>
   troubleshooting
   api

.. toctree::
   :maxdepth: 3
   :caption: GPU Tutorials and Examples

   1. Basic Usage <1_basics>
   2. Simple Customization <2_simple_customization>
   3. Advanced Customization <3_advanced_customization>
   4. The Standard Model <4_standard_model>
   5. Dust Coagulation <5_dust_coagulation>
   6. Dust Evolution <6_dust_evolution>
   7. Gas Evolution <7_gas_evolution>
   Test: Gas Evolution <test_gas_evolution>
   Test: Analytical Coagulation Kernels <test_analytical_coagulation_kernels>
   Example: Ice Lines <example_ice_lines>
   Example: Planetary Gaps <example_planetary_gaps>
   Example: Planetesimal Formation <example_planetesimal_formation>

Indices and tables
==================

* :ref:`genindex`
