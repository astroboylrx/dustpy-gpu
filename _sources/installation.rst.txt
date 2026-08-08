Installation
============

DustPy-GPU keeps the upstream ``dustpy`` import name, so a fresh virtual
environment is recommended to avoid conflicts with an existing DustPy install.

PyPI
----

For NumPy-only use::

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install dustpy-gpu

For GPU use, first install the CuPy package matching the local CUDA major
version. For example, with CUDA 13::

   python -m pip install cupy-cuda13x
   python -m pip install dustpy-gpu

DustPy-GPU automatically installs ``simframe-gpu``. Building DustPy-GPU from
source also requires a Fortran compiler.

GitHub checkout
---------------

For an editable development install::

   git clone https://github.com/astroboylrx/dustpy-gpu.git
   python -m pip install -e ./dustpy-gpu

Confirm the selected backend after installation::

   import dustpy

   sim = dustpy.Simulation(backend="auto")
   print(sim.backend)
