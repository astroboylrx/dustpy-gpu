# DustPy-GPU

`dustpy-gpu` is a GPU-enabled fork of [DustPy](https://github.com/stammler/dustpy), a Python package for simulating gas and dust evolution in protoplanetary disks. It adds selectable NumPy and CuPy backends while fully preserving the original user-facing API.

> ⚠️ **Note on Installation:**
> Designed as a drop-in replacement, this fork intentionally uses the upstream `dustpy` import name so users can run existing scripts with minimal modifications (mainly backend selection, see below). To avoid package conflicts, please install `dustpy-gpu` in a new virtual environment.


## Installation

Create and activate a dedicated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For GPU runs, first install the CuPy package that matches your CUDA installation. For example, with CUDA 12:

```bash
python -m pip install cupy-cuda12x
```

Then install `dustpy-gpu` directly from GitHub:

```bash
python -m pip install git+https://github.com/astroboylrx/dustpy-gpu.git
```

This also installs the required [`simframe-gpu`](https://github.com/astroboylrx/simframe-gpu) fork.

## Backends

Select a backend when creating a simulation:

```python
import dustpy

sim_cpu = dustpy.Simulation(backend="numpy")
sim_gpu = dustpy.Simulation(backend="cupy")
sim_auto = dustpy.Simulation(backend="auto")
```

Calling `dustpy.Simulation()` without a backend inherits the currently active backend. In a fresh Python process, the active backend is NumPy, preserving the upstream DustPy behavior; it does not automatically select a GPU.

The `"auto"` backend selects CuPy when CuPy is installed and a CUDA device is available; otherwise, it falls back to NumPy.

Existing DustPy code can continue to use `import dustpy`. With the CuPy backend, backend-native fields and calculations use CuPy arrays; convert them explicitly with `cupy.asnumpy()` when NumPy arrays are required for analysis or plotting.

Multiple simulations may use different backends in one process when run sequentially. Concurrent simulation advancement is not supported because the active backend bindings are process-global.

## Documentation

The existing DustPy API and physical model are documented in the [upstream DustPy documentation](https://stammler.github.io/dustpy/). GPU-specific validation and benchmark examples will be included in this repository.
