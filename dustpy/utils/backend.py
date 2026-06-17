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
_GMRES_STATS_ENABLED = os.getenv("DUSTPY_GMRES_STATS", "0").strip().lower() in ("1", "true", "yes", "on")
_GMRES_PREV_SOL = None
_GMRES_PRECOND_CACHE = {}


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


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name, default, minimum=1):
    raw = os.getenv(name)
    if raw is None:
        return max(int(default), int(minimum))
    try:
        val = int(raw)
    except Exception:
        val = int(default)
    return max(val, int(minimum))


def _load_cupy_gmres_config():
    solver_optimized = _env_bool("DUSTPY_CUPY_SOLVER_OPTIMIZED", default=True)
    default_x0_mode = "zero"
    default_reuse_precond = not solver_optimized

    x0_mode = os.getenv("DUSTPY_CUPY_GMRES_X0", default_x0_mode).strip().lower()
    if x0_mode not in ("rhs", "zero", "prev"):
        x0_mode = default_x0_mode
    return {
        "x0_mode": x0_mode,
        "reuse_precond": _env_bool("DUSTPY_CUPY_GMRES_REUSE_PRECOND", default=default_reuse_precond),
        "tier1_rtol": _env_float("DUSTPY_CUPY_GMRES_TIER1_RTOL", 2e-15),
        "tier1_atol": _env_float("DUSTPY_CUPY_GMRES_TIER1_ATOL", 2e-15),
        "tier1_maxiter": _env_int("DUSTPY_CUPY_GMRES_TIER1_MAXITER", 1000, minimum=1),
        "tier1_restart": _env_int("DUSTPY_CUPY_GMRES_TIER1_RESTART", 64, minimum=1),
        "tier2_rtol": _env_float("DUSTPY_CUPY_GMRES_TIER2_RTOL", 1e-13),
        "tier2_atol": _env_float("DUSTPY_CUPY_GMRES_TIER2_ATOL", 1e-13),
        "tier2_maxiter": _env_int("DUSTPY_CUPY_GMRES_TIER2_MAXITER", 2000, minimum=1),
        "tier2_restart": _env_int("DUSTPY_CUPY_GMRES_TIER2_RESTART", 150, minimum=1),
        "cpu_fallback_max": _env_int("DUSTPY_CUPY_CPU_FALLBACK_MAX", 1000, minimum=0),
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
_RUNTIME_STATES = {}
_ACTIVE_RUNTIME_TOKEN = None

_RUNTIME_STATE_VARS = (
    "_GMRES_STATS",
    "_GMRES_STATS_ENABLED",
    "_GMRES_PREV_SOL",
    "_GMRES_PRECOND_CACHE",
    "_CUPY_GMRES_CONFIG",
    "_CUPY_GAS_SOLVER_CONFIG",
    "_BOUND_SOLVER_BACKEND",
    "_SOLVE_SPARSE_IMPL",
    "_BOUND_GAS_SOLVER_BACKEND",
    "_BOUND_GAS_SOLVER_MODE",
    "_SOLVE_GAS_SPARSE_IMPL",
)


def _fresh_runtime_state():
    return {
        "_GMRES_STATS": Counter(),
        "_GMRES_STATS_ENABLED": _env_bool("DUSTPY_GMRES_STATS", default=False),
        "_GMRES_PREV_SOL": None,
        "_GMRES_PRECOND_CACHE": {},
        "_CUPY_GMRES_CONFIG": _load_cupy_gmres_config(),
        "_CUPY_GAS_SOLVER_CONFIG": _load_cupy_gas_solver_config(),
        "_BOUND_SOLVER_BACKEND": None,
        "_SOLVE_SPARSE_IMPL": None,
        "_BOUND_GAS_SOLVER_BACKEND": None,
        "_BOUND_GAS_SOLVER_MODE": None,
        "_SOLVE_GAS_SPARSE_IMPL": None,
    }


def _capture_runtime_state():
    return {name: globals()[name] for name in _RUNTIME_STATE_VARS}


def _restore_runtime_state(state):
    for name, value in state.items():
        globals()[name] = value


def _switch_runtime_state(runtime_token):
    global _ACTIVE_RUNTIME_TOKEN
    if runtime_token is None or runtime_token == _ACTIVE_RUNTIME_TOKEN:
        return
    if _ACTIVE_RUNTIME_TOKEN is not None:
        _RUNTIME_STATES[_ACTIVE_RUNTIME_TOKEN] = _capture_runtime_state()
    state = _RUNTIME_STATES.get(runtime_token)
    if state is None:
        state = _fresh_runtime_state()
        _RUNTIME_STATES[runtime_token] = state
    _restore_runtime_state(state)
    _ACTIVE_RUNTIME_TOKEN = runtime_token


def reload_cupy_gmres_config():
    """Reload CuPy GMRES settings from environment variables."""
    global _CUPY_GMRES_CONFIG, _CUPY_GAS_SOLVER_CONFIG, _GMRES_STATS_ENABLED
    _CUPY_GMRES_CONFIG = _load_cupy_gmres_config()
    _CUPY_GAS_SOLVER_CONFIG = _load_cupy_gas_solver_config()
    _GMRES_STATS_ENABLED = _env_bool("DUSTPY_GMRES_STATS", default=False)


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
    global _GMRES_PREV_SOL
    with _AUDIT_LOCK:
        _GMRES_STATS.clear()
    _GMRES_PREV_SOL = None
    _GMRES_PRECOND_CACHE.clear()


def get_gmres_stats():
    with _AUDIT_LOCK:
        return dict(_GMRES_STATS)


def _record_gmres_stat(key, always=False):
    if (not always) and (not _GMRES_STATS_ENABLED):
        return
    with _AUDIT_LOCK:
        _GMRES_STATS[key] += 1


def _cupy_fallback_debug_enabled():
    raw = os.getenv("DUSTPY_CUPY_FALLBACK_DEBUG", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _cupy_fallback_dump_max():
    return _env_int("DUSTPY_CUPY_FALLBACK_DUMP_MAX", 0, minimum=0)


def _cupy_fallback_dump_dir():
    raw = os.getenv("DUSTPY_CUPY_FALLBACK_DUMP_DIR", "").strip()
    return raw


def _summarize_fallback_matrix(matrix_cpu, rhs_cpu):
    if sp.issparse(matrix_cpu):
        mat = matrix_cpu.tocsr()
        data = mat.data
        nnz = int(mat.nnz)
    else:
        mat = np.asarray(matrix_cpu)
        data = mat.ravel()
        nnz = int(np.count_nonzero(data))

    if data.size > 0:
        finite_data = np.isfinite(data)
        n_data_nonfinite = int(data.size - np.count_nonzero(finite_data))
        if np.any(finite_data):
            abs_data = np.abs(data[finite_data])
            data_abs_max = float(np.max(abs_data))
            nonzero = abs_data[abs_data > 0]
            data_abs_min_pos = float(np.min(nonzero)) if nonzero.size > 0 else 0.0
        else:
            data_abs_max = float("nan")
            data_abs_min_pos = float("nan")
    else:
        n_data_nonfinite = 0
        data_abs_max = 0.0
        data_abs_min_pos = 0.0

    rhs_arr = np.asarray(rhs_cpu).ravel()
    finite_rhs = np.isfinite(rhs_arr)
    n_rhs_nonfinite = int(rhs_arr.size - np.count_nonzero(finite_rhs))
    rhs_abs_max = float(np.max(np.abs(rhs_arr[finite_rhs]))) if np.any(finite_rhs) else float("nan")

    diag = np.asarray(mat.diagonal()).ravel() if sp.issparse(mat) else np.asarray(np.diag(mat)).ravel()
    finite_diag = np.isfinite(diag)
    n_diag_nonfinite = int(diag.size - np.count_nonzero(finite_diag))
    if np.any(finite_diag):
        abs_diag = np.abs(diag[finite_diag])
        diag_abs_max = float(np.max(abs_diag))
        nz = abs_diag[abs_diag > 0]
        diag_abs_min_pos = float(np.min(nz)) if nz.size > 0 else 0.0
    else:
        diag_abs_max = float("nan")
        diag_abs_min_pos = float("nan")

    return {
        "shape": tuple(int(v) for v in mat.shape),
        "nnz": nnz,
        "n_data_nonfinite": n_data_nonfinite,
        "n_rhs_nonfinite": n_rhs_nonfinite,
        "n_diag_nonfinite": n_diag_nonfinite,
        "data_abs_max": data_abs_max,
        "data_abs_min_pos": data_abs_min_pos,
        "rhs_abs_max": rhs_abs_max,
        "diag_abs_max": diag_abs_max,
        "diag_abs_min_pos": diag_abs_min_pos,
    }


def _dump_fallback_system(matrix_cpu, rhs_cpu, *, fallback_id, tier1_info, tier2_info, cause):
    dump_max = _cupy_fallback_dump_max()
    if dump_max <= 0 or fallback_id > dump_max:
        return
    dump_dir = _cupy_fallback_dump_dir()
    if not dump_dir:
        return
    os.makedirs(dump_dir, exist_ok=True)
    stem = os.path.join(dump_dir, f"fallback_{fallback_id:05d}")
    matrix_csr = matrix_cpu.tocsr() if sp.issparse(matrix_cpu) else sp.csr_matrix(np.asarray(matrix_cpu))
    sp.save_npz(stem + "_A.npz", matrix_csr)
    np.savez(
        stem + "_rhs_meta.npz",
        rhs=np.asarray(rhs_cpu),
        tier1_info=str(tier1_info),
        tier2_info=str(tier2_info),
        cause=str(cause),
    )


def _cupy_matrix_to_cpu_csr(matrix_gpu):
    if isinstance(matrix_gpu, cp_sparse.spmatrix):
        if hasattr(matrix_gpu, "get"):
            matrix_cpu = matrix_gpu.get()
        else:
            matrix_cpu = sp.csr_matrix(
                (
                    cp.asnumpy(matrix_gpu.data),
                    cp.asnumpy(matrix_gpu.indices),
                    cp.asnumpy(matrix_gpu.indptr),
                ),
                shape=matrix_gpu.shape,
            )
    else:
        matrix_cpu = sp.csr_matrix(cp.asnumpy(matrix_gpu))
    if not sp.issparse(matrix_cpu):
        matrix_cpu = sp.csr_matrix(matrix_cpu)
    elif not sp.isspmatrix_csr(matrix_cpu):
        matrix_cpu = matrix_cpu.tocsr()
    return matrix_cpu


def _equilibrate_sparse_system_cpu(matrix_cpu, rhs_cpu, *, passes=2):
    mat = matrix_cpu.tocsr(copy=True)
    rhs_eq = np.asarray(rhs_cpu, dtype=np.float64).copy()
    n = mat.shape[0]
    row_scale_total = np.ones(n, dtype=np.float64)
    col_scale_total = np.ones(n, dtype=np.float64)

    for _ in range(max(1, int(passes))):
        row_norm = np.asarray(np.abs(mat).max(axis=1).toarray()).ravel()
        row_norm[~np.isfinite(row_norm) | (row_norm <= 0.0)] = 1.0
        row_scale = 1.0 / row_norm
        mat = sp.diags(row_scale).dot(mat).tocsr()
        rhs_eq *= row_scale
        row_scale_total *= row_scale

        col_norm = np.asarray(np.abs(mat).max(axis=0).toarray()).ravel()
        col_norm[~np.isfinite(col_norm) | (col_norm <= 0.0)] = 1.0
        col_scale = 1.0 / col_norm
        mat = mat.dot(sp.diags(col_scale)).tocsr()
        col_scale_total *= col_scale

    return mat, rhs_eq, row_scale_total, col_scale_total


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
    global _GMRES_PREV_SOL
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

    def _gmres_true_residual_ok(sol, *, rtol, atol):
        b_norm = float(cp.linalg.norm(rhs_gpu))
        tol = max(float(atol), float(rtol) * b_norm)
        res = matrix_gpu.dot(sol) - rhs_gpu
        res_norm = float(cp.linalg.norm(res))
        return (res_norm <= tol), res_norm, tol

    def _solve_equilibrated_retry(*, tier1_info, tier2_info):
        _record_gmres_stat("tier3_equilibrate_attempt", always=True)
        matrix_cpu = _cupy_matrix_to_cpu_csr(matrix_gpu)
        rhs_cpu = cp.asnumpy(rhs_gpu)
        A_eq_cpu, rhs_eq_cpu, _, col_scale = _equilibrate_sparse_system_cpu(matrix_cpu, rhs_cpu, passes=2)
        A_eq_gpu = cp_sparse.csr_matrix(A_eq_cpu)
        rhs_eq_gpu = cp.asarray(rhs_eq_cpu)
        diag_eq = A_eq_gpu.diagonal()
        diag_eq = cp.where(cp.abs(diag_eq) > 1e-15, diag_eq, cp.ones_like(diag_eq))
        inv_diag_eq = 1.0 / diag_eq

        def eq_matvec(x):
            return inv_diag_eq * x

        precond_eq = cp_splinalg.LinearOperator(A_eq_gpu.shape, matvec=eq_matvec)
        tier3_kwargs = _cupy_gmres_kwargs(
            rtol=cfg["tier2_rtol"],
            atol=cfg["tier2_atol"],
            maxiter=cfg["tier2_maxiter"],
            restart=cfg["tier2_restart"],
            precond=precond_eq,
            x0=cp.zeros_like(rhs_eq_gpu),
        )
        tier3_info = None
        tier3_cause = "nonconverged"
        try:
            sol3_eq, tier3_info = cp_splinalg.gmres(A_eq_gpu, rhs_eq_gpu, **tier3_kwargs)
            if tier3_info == 0:
                sol3 = cp.asarray(col_scale) * sol3_eq
                ok, res_norm, tol = _gmres_true_residual_ok(
                    sol3,
                    rtol=cfg["tier2_rtol"],
                    atol=cfg["tier2_atol"],
                )
                if ok:
                    _record_gmres_stat("tier3_equilibrate_success", always=True)
                    return sol3
                tier3_info = f"postcheck:{res_norm:.6e}>{tol:.6e}"
                tier3_cause = "postcheck_failed"
                _record_gmres_stat("tier3_postcheck_failed", always=True)
        except Exception as exc:
            tier3_info = f"exc:{type(exc).__name__}"
            tier3_cause = f"exception:{type(exc).__name__}"
            _record_gmres_stat("tier3_exception", always=True)
        _record_gmres_stat("tier3_failed", always=True)
        return _solve_cpu_fallback(
            tier1_info=tier1_info,
            tier2_info=f"{tier2_info};tier3={tier3_info}",
            cause=f"tier3_equilibrate:{tier3_cause}",
        )

    def _solve_cpu_fallback(*, tier1_info, tier2_info, cause):
        cfg_local = _CUPY_GMRES_CONFIG
        _record_gmres_stat("cpu_fallback", always=True)
        n_fallback = int(_GMRES_STATS.get("cpu_fallback", 0))
        try:
            matrix_cpu = _cupy_matrix_to_cpu_csr(matrix_gpu)
            rhs_cpu = cp.asnumpy(rhs_gpu)

            if _cupy_fallback_debug_enabled():
                summary = _summarize_fallback_matrix(matrix_cpu, rhs_cpu)
                print(
                    "[CPU_FALLBACK] "
                    f"id={n_fallback}, cause={cause}, tier1_info={tier1_info}, tier2_info={tier2_info}, "
                    f"shape={summary['shape']}, nnz={summary['nnz']}, "
                    f"n_data_nonfinite={summary['n_data_nonfinite']}, "
                    f"n_rhs_nonfinite={summary['n_rhs_nonfinite']}, "
                    f"n_diag_nonfinite={summary['n_diag_nonfinite']}, "
                    f"data_abs_min_pos={summary['data_abs_min_pos']:.6e}, data_abs_max={summary['data_abs_max']:.6e}, "
                    f"diag_abs_min_pos={summary['diag_abs_min_pos']:.6e}, diag_abs_max={summary['diag_abs_max']:.6e}, "
                    f"rhs_abs_max={summary['rhs_abs_max']:.6e}",
                    flush=True,
                )
            _dump_fallback_system(
                matrix_cpu,
                rhs_cpu,
                fallback_id=n_fallback,
                tier1_info=tier1_info,
                tier2_info=tier2_info,
                cause=cause,
            )

            matrix_lu = sp.linalg.splu(
                matrix_cpu.tocsc(),
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.0,
                options=dict(SymmetricMode=True),
            )
            sol_cpu = matrix_lu.solve(rhs_cpu)
            sol_fallback = cp.asarray(sol_cpu)
            max_allowed = int(cfg_local.get("cpu_fallback_max", 0))
            if max_allowed > 0 and n_fallback > max_allowed:
                _record_gmres_stat("cpu_fallback_limit_exceeded", always=True)
                raise RuntimeError(
                    "CuPy GMRES fallback limit exceeded "
                    f"(cpu_fallback={n_fallback}, max={max_allowed}, "
                    f"tier1_info={tier1_info}, tier2_info={tier2_info}, cause={cause})."
                )
            return sol_fallback
        except Exception as exc:
            _record_gmres_stat("cpu_fallback_failed", always=True)
            raise RuntimeError(
                "CuPy GMRES failed to converge and CPU fallback failed "
                f"(tier1_info={tier1_info}, tier2_info={tier2_info}, cause={cause})."
            ) from exc

    if is_scipy_sparse or is_cupy_sparse:
        cfg = _CUPY_GMRES_CONFIG
        precond = None
        diag = matrix_gpu.diagonal()
        diag = cp.where(cp.abs(diag) > 1e-15, diag, cp.ones_like(diag))
        if cfg["reuse_precond"]:
            pkey = (int(diag.size), str(diag.dtype))
            cached = _GMRES_PRECOND_CACHE.get(pkey)
            if cached is None:
                inv_diag = 1.0 / diag

                def matvec(x, inv=inv_diag):
                    return inv * x

                precond = cp_splinalg.LinearOperator(matrix_gpu.shape, matvec=matvec)
                _GMRES_PRECOND_CACHE[pkey] = (inv_diag, precond)
            else:
                inv_diag, precond = cached
                inv_diag[...] = 1.0 / diag
        else:
            inv_diag = 1.0 / diag

            def matvec(x):
                return inv_diag * x

            precond = cp_splinalg.LinearOperator(matrix_gpu.shape, matvec=matvec)

        x0_mode = cfg["x0_mode"]
        if x0_mode == "zero":
            x0_tier1 = cp.zeros_like(rhs_gpu)
        elif x0_mode == "prev" and _GMRES_PREV_SOL is not None and _GMRES_PREV_SOL.shape == rhs_gpu.shape:
            x0_tier1 = _GMRES_PREV_SOL
        else:
            x0_tier1 = rhs_gpu

        # Tier-1: strict tolerances (fast/default path).
        tier1_kwargs = _cupy_gmres_kwargs(
            rtol=cfg["tier1_rtol"],
            atol=cfg["tier1_atol"],
            maxiter=cfg["tier1_maxiter"],
            restart=cfg["tier1_restart"],
            precond=precond,
            x0=x0_tier1,
        )
        tier1_info = None
        tier1_cause = "nonconverged"
        tier1_sol = None
        try:
            tier1_sol, tier1_info = cp_splinalg.gmres(matrix_gpu, rhs_gpu, **tier1_kwargs)
            if tier1_info == 0:
                ok, res_norm, tol = _gmres_true_residual_ok(
                    tier1_sol,
                    rtol=cfg["tier1_rtol"],
                    atol=cfg["tier1_atol"],
                )
                if not ok:
                    tier1_info = f"postcheck:{res_norm:.6e}>{tol:.6e}"
                    tier1_cause = "postcheck_failed"
                    _record_gmres_stat("tier1_postcheck_failed", always=True)
                else:
                    _GMRES_PREV_SOL = tier1_sol
                    _record_gmres_stat("tier1_success")
                    return tier1_sol
        except Exception as exc:
            tier1_info = f"exc:{type(exc).__name__}"
            tier1_cause = f"exception:{type(exc).__name__}"
            _record_gmres_stat("tier1_exception", always=True)

        # Tier-2: relaxed tolerances + more iterations for hard timesteps.
        x0_tier2 = tier1_sol if tier1_sol is not None else x0_tier1
        tier2_kwargs = _cupy_gmres_kwargs(
            rtol=cfg["tier2_rtol"],
            atol=cfg["tier2_atol"],
            maxiter=cfg["tier2_maxiter"],
            restart=cfg["tier2_restart"],
            precond=precond,
            x0=x0_tier2,
        )
        tier2_info = None
        tier2_cause = "nonconverged"
        try:
            sol2, tier2_info = cp_splinalg.gmres(matrix_gpu, rhs_gpu, **tier2_kwargs)
            if tier2_info == 0:
                ok, res_norm, tol = _gmres_true_residual_ok(
                    sol2,
                    rtol=cfg["tier2_rtol"],
                    atol=cfg["tier2_atol"],
                )
                if not ok:
                    tier2_info = f"postcheck:{res_norm:.6e}>{tol:.6e}"
                    tier2_cause = "postcheck_failed"
                    _record_gmres_stat("tier2_postcheck_failed", always=True)
                else:
                    _GMRES_PREV_SOL = sol2
                    _record_gmres_stat("tier2_success")
                    return sol2
        except Exception as exc:
            tier2_info = f"exc:{type(exc).__name__}"
            tier2_cause = f"exception:{type(exc).__name__}"
            _record_gmres_stat("tier2_exception", always=True)

        _record_gmres_stat("tier2_failed", always=True)
        sol_fallback = _solve_equilibrated_retry(
            tier1_info=tier1_info,
            tier2_info=tier2_info,
        )
        _GMRES_PREV_SOL = sol_fallback
        return sol_fallback

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


def bind_sparse_solver(backend=None, force=False, runtime_token=None):
    """Bind sparse linear solver implementation for the selected backend."""
    global _BOUND_SOLVER_BACKEND, _SOLVE_SPARSE_IMPL
    _switch_runtime_state(runtime_token)
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


def bind_gas_sparse_solver(backend=None, force=False, runtime_token=None):
    """Bind gas sparse solver implementation for selected backend."""
    global _BOUND_GAS_SOLVER_BACKEND, _BOUND_GAS_SOLVER_MODE, _SOLVE_GAS_SPARSE_IMPL
    _switch_runtime_state(runtime_token)
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
