"""Backend helpers for bridging NumPy-only boundaries."""

import os
import threading
import traceback
import inspect
from collections import Counter

import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    import cupyx.scipy.sparse.linalg as cp_splinalg
except Exception:  # pragma: no cover - optional dependency
    cp = None
    cp_sparse = None
    cp_splinalg = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

from simframe.backends.api import get_backend
from simframe.backends.api import xp


_AUDIT_ENABLED = os.getenv("DUSTPY_TRANSFER_AUDIT", "0") == "1"
_AUDIT_COUNTS = Counter()
_AUDIT_TOTAL = 0
_AUDIT_LOCK = threading.Lock()
_AUDIT_LOCAL = threading.local()
_GMRES_STATS = Counter()


_CUPY_GMRES_KWARGS = {"maxiter": 500}
_CUPY_GMRES_HAS_RTOL = False
_CUPY_GMRES_HAS_ATOL = False
_CUPY_GMRES_HAS_RESTART = False
if cp_splinalg is not None:
    try:
        _gmres_params = inspect.signature(cp_splinalg.gmres).parameters
    except Exception:
        _gmres_params = {}
    _CUPY_GMRES_HAS_RTOL = "rtol" in _gmres_params
    _CUPY_GMRES_HAS_ATOL = "atol" in _gmres_params
    _CUPY_GMRES_HAS_RESTART = "restart" in _gmres_params
    if "rtol" in _gmres_params:
        _CUPY_GMRES_KWARGS["rtol"] = 1e-10
    else:
        _CUPY_GMRES_KWARGS["tol"] = 1e-10
    if "atol" in _gmres_params:
        _CUPY_GMRES_KWARGS["atol"] = 1e-10


def _env_bool(name, default=False):
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _load_cupy_gmres_config():
    return {
        "tier1": {
            "rtol": float(os.getenv("DUSTPY_CUPY_GMRES_RTOL", "1e-14")),
            "atol": float(os.getenv("DUSTPY_CUPY_GMRES_ATOL", "1e-14")),
            "maxiter": int(os.getenv("DUSTPY_CUPY_GMRES_MAXITER", "1000")),
            "restart": int(os.getenv("DUSTPY_CUPY_GMRES_RESTART", "64")),
        },
        "tier2": {
            "rtol": float(os.getenv("DUSTPY_CUPY_GMRES_FALLBACK_RTOL", "1e-10")),
            "atol": float(os.getenv("DUSTPY_CUPY_GMRES_FALLBACK_ATOL", "1e-10")),
            "maxiter": int(os.getenv("DUSTPY_CUPY_GMRES_FALLBACK_MAXITER", "2000")),
            "restart": int(os.getenv("DUSTPY_CUPY_GMRES_FALLBACK_RESTART", "150")),
        },
        "cpu_fallback": _env_bool("DUSTPY_CUPY_GMRES_CPU_FALLBACK", default=False),
    }


_CUPY_GMRES_CONFIG = _load_cupy_gmres_config()
_BOUND_SOLVER_BACKEND = None
_SOLVE_SPARSE_IMPL = None
_BOUND_GAS_SOLVER_BACKEND = None
_BOUND_GAS_SOLVER_MODE = None
_SOLVE_GAS_SPARSE_IMPL = None


def _load_cupy_gas_solver_config():
    mode = os.getenv("DUSTPY_CUPY_GAS_SOLVER", "cpu_direct").strip().lower()
    aliases = {
        "cpu": "cpu_direct",
        "cpu_direct": "cpu_direct",
        "direct_cpu": "cpu_direct",
        "gpu": "gpu_gmres",
        "gpu_gmres": "gpu_gmres",
        "gmres": "gpu_gmres",
        "dense": "dense_gpu",
        "dense_gpu": "dense_gpu",
    }
    return {"mode": aliases.get(mode, "cpu_direct")}


_CUPY_GAS_SOLVER_CONFIG = _load_cupy_gas_solver_config()


def reload_cupy_gmres_config():
    """Reload CuPy GMRES settings from environment variables."""
    global _CUPY_GMRES_CONFIG, _CUPY_GAS_SOLVER_CONFIG
    _CUPY_GMRES_CONFIG = _load_cupy_gmres_config()
    _CUPY_GAS_SOLVER_CONFIG = _load_cupy_gas_solver_config()


def _cupy_gmres_kwargs(*, rtol, atol, maxiter, restart, precond, x0):
    kwargs = {
        "maxiter": int(maxiter),
        "M": precond,
        "x0": x0,
    }
    if _CUPY_GMRES_HAS_RTOL:
        kwargs["rtol"] = float(rtol)
    else:
        kwargs["tol"] = float(rtol)
    if _CUPY_GMRES_HAS_ATOL:
        kwargs["atol"] = float(atol)
    if _CUPY_GMRES_HAS_RESTART:
        kwargs["restart"] = int(restart)
    return kwargs


def set_transfer_audit(enabled):
    """Enable or disable transfer audit instrumentation."""
    global _AUDIT_ENABLED
    _AUDIT_ENABLED = bool(enabled)


def reset_transfer_audit():
    """Reset transfer audit counters."""
    global _AUDIT_TOTAL
    with _AUDIT_LOCK:
        _AUDIT_COUNTS.clear()
        _AUDIT_TOTAL = 0


def _audit_location():
    this_file = __file__
    stack = traceback.extract_stack(limit=40)
    for frame in reversed(stack[:-2]):
        if frame.filename != this_file:
            return f"{frame.filename}:{frame.lineno}:{frame.name}"
    return "unknown"


def _record_transfer(event, tag=None):
    global _AUDIT_TOTAL
    if not _AUDIT_ENABLED:
        return
    loc = _audit_location()
    evt = f"{event}:{tag}" if tag else event
    key = (evt, loc)
    with _AUDIT_LOCK:
        _AUDIT_COUNTS[key] += 1
        _AUDIT_TOTAL += 1


def get_transfer_audit_report(limit=20):
    """Return transfer audit counters."""
    with _AUDIT_LOCK:
        items = _AUDIT_COUNTS.most_common(limit)
        total = _AUDIT_TOTAL
    top = [
        {"event": event, "location": location, "count": count}
        for (event, location), count in items
    ]
    return {"enabled": _AUDIT_ENABLED, "total_transfers": total, "top": top}


def reset_gmres_stats():
    with _AUDIT_LOCK:
        _GMRES_STATS.clear()


def get_gmres_stats():
    with _AUDIT_LOCK:
        return dict(_GMRES_STATS)


def _contains_backend_array(value):
    if xp.is_array(value):
        return True
    if isinstance(value, tuple):
        return any(_contains_backend_array(v) for v in value)
    if isinstance(value, list):
        return any(_contains_backend_array(v) for v in value)
    if isinstance(value, dict):
        return any(_contains_backend_array(v) for v in value.values())
    return False


def to_numpy(value):
    """Convert an array-like backend value into a NumPy array."""
    if xp.is_array(value):
        _record_transfer("device_to_host:to_numpy")
    return xp.to_numpy(value)


def to_backend(value):
    """Convert a NumPy value to the active backend array type."""
    if get_backend() == "numpy":
        return value
    if isinstance(value, np.ndarray):
        ctx_tag = getattr(_AUDIT_LOCAL, "tag", None)
        _record_transfer("host_to_device:to_backend", tag=ctx_tag)
    return xp.asarray(value)


def _map_numpy(value):
    if isinstance(value, tuple):
        return tuple(_map_numpy(v) for v in value)
    if isinstance(value, list):
        return [_map_numpy(v) for v in value]
    if isinstance(value, dict):
        return {k: _map_numpy(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value
    if xp.is_array(value):
        _record_transfer("device_to_host:map_numpy")
        return to_numpy(value)
    return value


def _map_backend(value):
    if isinstance(value, tuple):
        return tuple(_map_backend(v) for v in value)
    if isinstance(value, list):
        return [_map_backend(v) for v in value]
    if isinstance(value, dict):
        return {k: _map_backend(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return to_backend(value)
    return value


def call_numpy(func, *args, to_backend_result=True, audit_tag=None, **kwargs):
    """Call a NumPy-only function with backend-safe argument conversion."""
    if _contains_backend_array(args) or _contains_backend_array(kwargs):
        _record_transfer("device_to_host:call_numpy", tag=audit_tag)
    prev_tag = getattr(_AUDIT_LOCAL, "tag", None)
    _AUDIT_LOCAL.tag = audit_tag
    try:
        np_args = tuple(_map_numpy(arg) for arg in args)
        np_kwargs = {key: _map_numpy(val) for key, val in kwargs.items()}
        ret = func(*np_args, **np_kwargs)
        if not to_backend_result:
            return ret
        return _map_backend(ret)
    finally:
        _AUDIT_LOCAL.tag = prev_tag


def _solve_sparse_linear_system_cupy(matrix, rhs):
    if cp is None or cp_sparse is None or cp_splinalg is None:
        raise RuntimeError("CuPy backend requested but cupy/cupyx is unavailable.")

    is_scipy_sparse = sp.issparse(matrix)
    is_cupy_sparse = isinstance(matrix, cp_sparse.spmatrix)

    if is_cupy_sparse:
        matrix_gpu = matrix if getattr(matrix, "format", None) == "csr" else matrix.tocsr()
    elif is_scipy_sparse:
        matrix_gpu = cp_sparse.csr_matrix(matrix)
    else:
        matrix_gpu = cp.asarray(matrix)
    rhs_raw = rhs._data if hasattr(rhs, "_data") else rhs
    rhs_gpu = cp.asarray(rhs_raw)

    if is_scipy_sparse or is_cupy_sparse:
        diag = matrix_gpu.diagonal()
        diag = cp.where(cp.abs(diag) > 1e-15, diag, cp.ones_like(diag))
        inv_diag = 1.0 / diag

        def matvec(x):
            return inv_diag * x

        precond = cp_splinalg.LinearOperator(matrix_gpu.shape, matvec=matvec)
        cfg = _CUPY_GMRES_CONFIG
        tier1 = cfg["tier1"]
        # Tier-1: strict tolerances (fast/default path).
        tier1_kwargs = _cupy_gmres_kwargs(
            rtol=tier1["rtol"],
            atol=tier1["atol"],
            maxiter=tier1["maxiter"],
            restart=tier1["restart"],
            precond=precond,
            x0=rhs_gpu,
        )
        sol, info = cp_splinalg.gmres(matrix_gpu, rhs_gpu, **tier1_kwargs)
        if info == 0:
            with _AUDIT_LOCK:
                _GMRES_STATS["tier1_success"] += 1
            return sol

        # Tier-2: relaxed tolerances + more iterations for hard timesteps.
        tier2 = cfg["tier2"]
        tier2_kwargs = _cupy_gmres_kwargs(
            rtol=tier2["rtol"],
            atol=tier2["atol"],
            maxiter=tier2["maxiter"],
            restart=tier2["restart"],
            precond=precond,
            x0=sol,
        )
        sol2, info2 = cp_splinalg.gmres(matrix_gpu, rhs_gpu, **tier2_kwargs)
        if info2 == 0:
            with _AUDIT_LOCK:
                _GMRES_STATS["tier2_success"] += 1
            return sol2

        # Optional last-resort CPU direct fallback (explicitly opt-in).
        if cfg["cpu_fallback"]:
            with _AUDIT_LOCK:
                _GMRES_STATS["cpu_fallback"] += 1
            matrix_cpu = sp.csr_matrix(cp.asnumpy(matrix_gpu))
            rhs_cpu = cp.asnumpy(rhs_gpu)
            matrix_lu = sp.linalg.splu(
                matrix_cpu.tocsc(),
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.0,
                options=dict(SymmetricMode=True),
            )
            return cp.asarray(matrix_lu.solve(rhs_cpu))

        with _AUDIT_LOCK:
            _GMRES_STATS["tier2_failed"] += 1
        raise RuntimeError(
            "CuPy GMRES failed to converge "
            f"(tier1_info={info}, tier2_info={info2})."
        )

    return cp.linalg.solve(matrix_gpu, rhs_gpu)


def _solve_sparse_linear_system_torch(matrix, rhs):
    if torch is None:
        raise RuntimeError("PyTorch backend requested but torch is unavailable.")

    rhs_t = xp.asarray(rhs)
    if rhs_t.dtype not in (torch.float32, torch.float64):
        rhs_t = rhs_t.to(dtype=torch.float64)
    device = rhs_t.device
    dtype = rhs_t.dtype

    can_try_torch_sparse = rhs_t.device.type != "cpu"
    if sp.issparse(matrix) and can_try_torch_sparse and hasattr(torch.sparse, "spsolve"):
        matrix_csr = matrix.tocsr()
        crow_indices = torch.from_numpy(
            matrix_csr.indptr.astype(np.int64, copy=False)
        ).to(device=device)
        col_indices = torch.from_numpy(
            matrix_csr.indices.astype(np.int64, copy=False)
        ).to(device=device)
        values = torch.from_numpy(matrix_csr.data).to(device=device, dtype=dtype)
        matrix_t = torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            values,
            size=matrix_csr.shape,
            device=device,
            dtype=dtype,
        )
        try:
            return torch.sparse.spsolve(matrix_t, rhs_t)
        except Exception:
            pass

    if sp.issparse(matrix):
        matrix_cpu = matrix if sp.isspmatrix_csc(matrix) else matrix.tocsc()
        matrix_lu = sp.linalg.splu(
            matrix_cpu,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.0,
            options=dict(SymmetricMode=True),
        )
        return xp.asarray(matrix_lu.solve(to_numpy(rhs_t)))

    dense = xp.asarray(matrix)
    if dense.layout != torch.strided:
        dense = dense.to_dense()
    return torch.linalg.solve(dense, rhs_t.unsqueeze(-1)).squeeze(-1)


def _solve_sparse_linear_system_numpy(matrix, rhs):
    matrix_cpu = matrix if sp.isspmatrix_csc(matrix) else matrix.tocsc()
    matrix_lu = sp.linalg.splu(
        matrix_cpu,
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.0,
        options=dict(SymmetricMode=True),
    )
    return matrix_lu.solve(np.asarray(to_numpy(rhs)))


def _solve_sparse_linear_system_cupy_cpu_direct(matrix, rhs):
    if cp is None or cp_sparse is None:
        raise RuntimeError("CuPy backend requested but cupy/cupyx is unavailable.")

    if isinstance(matrix, cp_sparse.spmatrix):
        matrix_cpu = matrix.get()
    elif sp.issparse(matrix):
        matrix_cpu = matrix
    else:
        matrix_cpu = sp.csr_matrix(np.asarray(to_numpy(matrix)))
    matrix_cpu = matrix_cpu if sp.isspmatrix_csc(matrix_cpu) else matrix_cpu.tocsc()
    rhs_cpu = np.asarray(to_numpy(rhs))

    matrix_lu = sp.linalg.splu(
        matrix_cpu,
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.0,
        options=dict(SymmetricMode=True),
    )
    sol_cpu = matrix_lu.solve(rhs_cpu)
    _record_transfer("host_to_device:gas_cpu_direct")
    return cp.asarray(sol_cpu)


def _solve_sparse_linear_system_cupy_dense(matrix, rhs):
    """Dense CuPy direct solve path (mainly for small gas systems / experiments)."""
    if cp is None or cp_sparse is None:
        raise RuntimeError("CuPy backend requested but cupy/cupyx is unavailable.")

    if isinstance(matrix, cp_sparse.spmatrix):
        matrix_gpu = matrix.toarray()
    elif sp.issparse(matrix):
        matrix_gpu = cp_sparse.csr_matrix(matrix).toarray()
    else:
        matrix_gpu = cp.asarray(matrix)
    rhs_raw = rhs._data if hasattr(rhs, "_data") else rhs
    rhs_gpu = cp.asarray(rhs_raw)
    return cp.linalg.solve(matrix_gpu, rhs_gpu)


def bind_sparse_solver(backend=None, force=False):
    """Bind sparse linear solver implementation for the selected backend."""
    global _BOUND_SOLVER_BACKEND, _SOLVE_SPARSE_IMPL
    backend = get_backend() if backend is None else backend
    if (not force) and backend == _BOUND_SOLVER_BACKEND and _SOLVE_SPARSE_IMPL is not None:
        return

    if backend == "cupy":
        _SOLVE_SPARSE_IMPL = _solve_sparse_linear_system_cupy
    elif backend == "torch":
        _SOLVE_SPARSE_IMPL = _solve_sparse_linear_system_torch
    else:
        _SOLVE_SPARSE_IMPL = _solve_sparse_linear_system_numpy
    _BOUND_SOLVER_BACKEND = backend


def solve_sparse_linear_system(matrix, rhs):
    """Solve a linear system with the backend-bound sparse solver path."""
    if _SOLVE_SPARSE_IMPL is None:
        bind_sparse_solver()
    return _SOLVE_SPARSE_IMPL(matrix, rhs)


def bind_gas_sparse_solver(backend=None, force=False):
    """Bind gas sparse solver implementation for selected backend."""
    global _BOUND_GAS_SOLVER_BACKEND, _BOUND_GAS_SOLVER_MODE, _SOLVE_GAS_SPARSE_IMPL
    backend = get_backend() if backend is None else backend
    mode = _CUPY_GAS_SOLVER_CONFIG["mode"] if backend == "cupy" else "default"
    if (
        (not force)
        and backend == _BOUND_GAS_SOLVER_BACKEND
        and mode == _BOUND_GAS_SOLVER_MODE
        and _SOLVE_GAS_SPARSE_IMPL is not None
    ):
        return

    if backend == "cupy" and mode == "cpu_direct":
        _SOLVE_GAS_SPARSE_IMPL = _solve_sparse_linear_system_cupy_cpu_direct
    elif backend == "cupy" and mode == "dense_gpu":
        _SOLVE_GAS_SPARSE_IMPL = _solve_sparse_linear_system_cupy_dense
    elif backend == "cupy":
        _SOLVE_GAS_SPARSE_IMPL = _solve_sparse_linear_system_cupy
    elif backend == "torch":
        _SOLVE_GAS_SPARSE_IMPL = _solve_sparse_linear_system_torch
    else:
        _SOLVE_GAS_SPARSE_IMPL = _solve_sparse_linear_system_numpy
    _BOUND_GAS_SOLVER_BACKEND = backend
    _BOUND_GAS_SOLVER_MODE = mode


def solve_gas_sparse_linear_system(matrix, rhs):
    """Solve gas linear system with gas-specific bound solver path."""
    if _SOLVE_GAS_SPARSE_IMPL is None:
        bind_gas_sparse_solver()
    return _SOLVE_GAS_SPARSE_IMPL(matrix, rhs)


# Default binding at import time (normally overwritten during run initialize).
bind_sparse_solver()
bind_gas_sparse_solver()
