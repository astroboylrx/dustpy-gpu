Backends
========

Select a backend when constructing a simulation::

   import dustpy

   sim_cpu = dustpy.Simulation()                 # NumPy, the default
   sim_numpy = dustpy.Simulation(backend="numpy")
   sim_cupy = dustpy.Simulation(backend="cupy")
   sim_auto = dustpy.Simulation(backend="auto") # CuPy if available

``numpy``
   Preserves the upstream CPU behavior and uses the existing SciPy and Fortran
   implementations where applicable.

``cupy``
   Uses CuPy/CuPyX arrays, kernels, sparse matrices, and GPU solvers for the
   accelerated dust path. A CUDA-capable GPU and a compatible CuPy package are
   required.

``auto``
   Selects CuPy when it can be imported and a CUDA device is available;
   otherwise it selects NumPy. The default remains NumPy so hardware
   availability cannot silently change an existing simulation.

Multiple simulation objects
---------------------------

NumPy and CuPy simulation objects can coexist in one Python process when they
are used sequentially. DustPy-GPU activates and rebinds the requested backend
before ``initialize()``, ``run()``, or ``update()``. Concurrent execution of
different backends in the same process is not supported because some low-level
bindings remain process-global.

GPU runtime behavior
--------------------

Backend arrays
^^^^^^^^^^^^^^

Simulation fields contain NumPy arrays with ``backend="numpy"`` and CuPy arrays
with ``backend="cupy"``. Convert GPU results explicitly when passing them to
host-only analysis tools::

   import cupy as cp

   sigma_dust = cp.asnumpy(sim.dust.Sigma)

DustPy's plotting helpers perform the required conversion for their supported
inputs.

Sparse solver
^^^^^^^^^^^^^

The accelerated dust path uses CuPy sparse GMRES with Jacobi preconditioning.
CuPy's direct sparse-solver API currently executes its solve on the CPU, so it
does not provide the same accelerated path. DustPy-GPU retains a tiered solver
policy and CPU fallback for robustness.

Numerical reproducibility
^^^^^^^^^^^^^^^^^^^^^^^^^

CuPy results are not expected to be bitwise deterministic. Sparse iterative
solves and parallel scatter/accumulation can change floating-point operation
order between runs. These tiny differences can lead to nearby numerical
trajectories, but validation shows bounded, non-monotonic backend differences
rather than persistent error accumulation. See :doc:`backend_parity_and_benchmarks`.

Performance notes
^^^^^^^^^^^^^^^^^

The first GPU run in a process includes CUDA context setup and kernel
compilation. Benchmark continued evolution after a warm-up interval rather than
timing initialization. Avoid frequent ``cupy.asnumpy()``, ``.get()``, or GPU
scalar reads inside evolution loops because they synchronize the device.
