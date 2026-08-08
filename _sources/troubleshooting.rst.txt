Troubleshooting
===============

CuPy cannot find a CUDA device
------------------------------

Install the CuPy distribution matching the machine's CUDA major version and
verify ``cupy.cuda.runtime.getDeviceCount()`` before selecting the CuPy backend.
Use ``backend="numpy"`` on systems without a CUDA GPU.

Matplotlib cannot plot a field
------------------------------

General plotting and analysis packages expect host arrays. Convert a CuPy field
with ``cupy.asnumpy(field)``. DustPy's own plotting helpers handle their
supported field inputs automatically.

The first GPU step is slow
--------------------------

CUDA context creation and just-in-time kernel compilation add one-time startup
cost. Measure performance after a warm-up interval and reuse the same Python
process for repeated runs.

Editable install fails after an environment change
--------------------------------------------------

Reinstall both forks into the active environment::

   python -m pip install --no-build-isolation -e ./simframe -e ./dustpy

Unexpected backend switching
----------------------------

Use an explicit backend instead of ``auto`` when reproducibility across
machines matters. Multiple simulation objects may be used sequentially, but do
not evolve them concurrently in separate threads.
