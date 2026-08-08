# DustPy-GPU

`dustpy-gpu` is a GPU-enabled fork of [DustPy](https://github.com/stammler/dustpy), a Python package for simulating gas and dust evolution in protoplanetary disks.

Designed as a drop-in replacement, it retains the upstream `dustpy` import name and original API, so migrating existing scripts requires minimal changes (mainly backend selection, see below).

## Installation

To avoid package conflicts, please install this drop-in replacement in a fresh virtual environment:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip

# For GPU runs: Install CuPy matching your CUDA version (e.g., CUDA 13)
python -m pip install cupy-cuda13x

# Install dustpy-gpu (automatically installs the required simframe-gpu fork)
# Alternatively: python -m pip install dustpy-gpu
python -m pip install git+https://github.com/astroboylrx/dustpy-gpu.git
```

## Backends

Select your backend when initializing a simulation:

```python
import dustpy

sim_cpu = dustpy.Simulation(backend="numpy") # Default (preserves upstream behavior)
sim_gpu = dustpy.Simulation(backend="cupy")
sim_auto = dustpy.Simulation(backend="auto") # Uses CuPy if available, else NumPy
```

When using the `cupy` backend, remember to convert CuPy arrays back to NumPy with `cupy.asnumpy()` for plotting or analysis. For those new to CuPy, we recommend checking out the [Basics of CuPy](https://docs.cupy.dev/en/stable/user_guide/basic.html).

Multiple backends can be used together in a single script if run sequentially, but concurrent execution is not supported due to process-global bindings.


## Documentation

The [DustPy-GPU documentation](https://astroboylrx.github.io/dustpy-gpu/) covers installation, backend behavior, GPU-adapted examples, numerical parity, and performance. Refer to the [upstream DustPy documentation](https://stammler.github.io/dustpy/) for the complete physics and standard API reference.

### Parity and benchmarks

Users should expect **numerical and scientific parity**, though not strictly identical arrays across long or repeated GPU runs.

* **NumPy Backend:** Reproduces upstream `DustPy` results bitwise.
* **CuPy Backend:** Agrees with NumPy to strict FP64 tolerances initially. Long-run differences remain quite small and non-accumulating. Official examples (ice lines, planetary gaps, planetesimal formation, etc.) are visually indistinguishable.
* **GPU Determinism:** CuPy runs are not bitwise deterministic. The use of an iterative solver (sparse GMRES) — necessitated because CuPy's direct sparse solver API currently falls back to the CPU — combined with parallel GPU accumulation, can produce tiny run-to-run differences.

For detailed comparisons and performance results, see the [backend parity and performance documentation](https://astroboylrx.github.io/dustpy-gpu/backend_parity_and_benchmarks.html).

## Citation

Please cite [Stammler & Birnstiel (2022)](https://doi.org/10.3847/1538-4357/ac7d58) for DustPy and [Li & Chiang (2026)](https://ui.adsabs.harvard.edu/abs/2026arXiv260614704L) for DustPy-GPU.
