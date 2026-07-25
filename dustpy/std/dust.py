'''Module containing standard functions for the dust.'''


import dustpy.constants as c
import os
from types import SimpleNamespace

# Re-export legacy private attributes so existing dumps and direct imports keep resolving through dustpy.std.dust.
import dustpy.std._dust_common as _dust_common_impl
import dustpy.std._dust_cupy as _dust_cupy_impl
import dustpy.std._dust_numpy as _dust_numpy_impl
from dustpy.std._dust_common import (
    _apply_gas_floor_freeze_to_flux,
    _apply_gas_floor_freeze_to_radial_field,
    _apply_inner_boundary_selective,
    _dust_transport_drain_interface_mask,
    _dust_transport_inactive_mask,
    _dust_transport_inactive_mask_from_arrays,
    _ensure_dust_floor_retry_state,
    _ensure_dust_floor_topup_state,
    _field_data,
    _gas_floor_freeze_interface_mask,
    _gas_floor_freeze_mask,
    _gas_floor_freeze_mask_from_arrays,
    _inner_boundary_advective_drain_mask,
    _inner_diode_block_mask,
    _interface_drain_mask_from_flux,
    _record_implicit_floor_retry,
    _record_implicit_floor_retry_exhausted,
    _refresh_dust_boundary_views,
    _transport_diffusion_freeze_interface_mask,
    _transport_velocity_freeze_interface_mask,
)
from dustpy.std._dust_cupy import (
    _D_cupy,
    _F_adv_cupy,
    _F_diff_cupy,
    _H_cupy,
    _S_coag_cupy,
    _S_hyd_cupy,
    _St_Epstein_StokesI_cupy,
    _a_cupy,
    _active_imax_per_radius,
    _apply_inner_zero_flux_dust_hyd_edge_cupy,
    _apply_zero_flux_dust_hyd_edges_cupy,
    _get_boundary_basis_cupy,
    _get_coag_pair_indices,
    _get_cupy_diag_positions_csr,
    _get_cupy_dust_solver_mode,
    _get_dust_hyd_boundary_pattern_cupy,
    _get_frag_p,
    _get_jcoag_chunk_size,
    _get_jcoag_pattern_cupy,
    _get_jcoag_precomp_cupy,
    _get_jcoag_work_buffer,
    _get_jfrag_map_cupy,
    _get_jstick_map_cupy,
    _get_mass_grid_q,
    _get_scoag_pair_map,
    _get_scoag_precomp,
    _interp_to_interfaces_cupy,
    _jacobian_coagulation_generator_cupy,
    _jacobian_hydrodynamic_generator_cupy,
    _kernel_cupy,
    _p_frag_cupy,
    _scatter_add_1d,
    _vrad_cupy,
    _vrel_azimuthal_drift_cupy,
    _vrel_brownian_motion_cupy,
    _vrel_radial_drift_cupy,
    _vrel_turbulent_motion_cupy,
    _vrel_vertical_settling_cupy,
)
from dustpy.std._dust_cupy_kernels import (
    _F_diff_cupy_elementwise,
    _get_collision_kernel_elementwise_kernel,
    _get_fdiff_elementwise_kernel,
    _get_pfrag_elementwise_kernel,
    _get_raw_scatter_kernel,
    _get_vrel_tot_elementwise_kernel,
    _get_vrel_turbulent_elementwise_kernel,
    _kernel_cupy_elementwise,
    _p_frag_cupy_elementwise,
    _vrel_tot_cupy_elementwise,
    _vrel_turbulent_motion_cupy_elementwise,
)
from dustpy.std._dust_numpy import (
    _D_fortran,
    _F_adv_fortran,
    _F_diff_fortran,
    _H_fortran,
    _S_coag_fortran,
    _S_hyd_fortran,
    _St_Epstein_StokesI_fortran,
    _a_fortran,
    _apply_inner_zero_flux_dust_hyd_edge_numpy,
    _apply_zero_flux_dust_hyd_edges_numpy,
    _coagulation_parameters_fortran,
    _coagulation_parameters_python,
    _dust_f_call,
    _get_jcoag_const_numpy,
    _get_jcoag_pattern,
    _interp_to_interfaces_numpy,
    _jacobian_hydrodynamic_generator_numpy,
    _kernel_fortran,
    _p_frag_fortran,
    _vrad_fortran,
    _vrel_azimuthal_drift_fortran,
    _vrel_brownian_motion_fortran,
    _vrel_radial_drift_fortran,
    _vrel_turbulent_motion_fortran,
    _vrel_vertical_settling_fortran,
)
from dustpy.std import dust_f
from dustpy.utils.backend import bind_sparse_solver
from dustpy.utils.backend import solve_sparse_linear_system
from dustpy.utils.backend import to_numpy
from dustpy.utils.boundary_modes import is_dust_inner_outflow_only_enabled, is_zero_flux_enabled
from simframe.backends.api import get_backend
from simframe.backends.api import select_backend
from simframe.backends.api import xp

import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None

try:
    import cupyx.scipy.sparse as cp_sparse
except Exception:  # pragma: no cover - optional dependency
    cp_sparse = None

from simframe.integration import Scheme


_BOUND_BACKEND = None
_K_A = None
_K_D = None
_K_H = None
_K_F_ADV = None
_K_F_DIFF = None
_K_S_COAG = None
_K_S_HYD = None
_K_KERNEL = None
_K_P_FRAG = None
_K_ST = None
_K_VRAD = None
_K_VREL_BROWN = None
_K_VREL_AZI = None
_K_VREL_RAD = None
_K_VREL_TURB = None
_K_VREL_VERT = None
_K_COAG_PARAMS = None
_K_IMPL_1_DIRECT = None
_K_IMPLICIT_FLOOR_RETRY = None
_K_JACOBIAN = None
_K_INTERP_TO_INTERFACES = None
_JCOAG_GEN_MODE = "baseline"
_S_COAG_MODE = "baseline"
_F_DIFF_MODE = "baseline"
_SCATTER_MODE = "addat"
_CUPY_DUST_SOLVER_MODE = "sparse"
_VREL_TURB_MODE = "baseline"
_VREL_TOT_MODE = "baseline"
_P_FRAG_MODE = "baseline"
_COLLISION_KERNEL_MODE = "baseline"
_CUPY_A_BUILD_MODE = "identity_sub"
_JCOAG_CHUNK_SIZE_OVERRIDE = None
_SANITIZER_CONFIG = {
    "enabled": False,
    "mode": "implicit",
    "tiny_max_ratio": 10000.0,
    "born_ratio": 1.0,
    "stop_thresh_mearth": 1.0e-2,
    "tinymax_report": False,
    "tinymax_report_cadence": 1,
}
_GAS_FLOOR_FREEZE_CONFIG = {
    "enabled": True,
    "ratio": 1.0e3,
}
_DUST_TRANSPORT_INACTIVE_FLOOR_CONFIG = {
    "enabled": False,
    "ratio": 1.0,
}
_IMPLICIT_FLOOR_RETRY_CONFIG = {
    "enabled": True,
    "threshold_mearth": 1.0e-6,
    "shrink": 0.5,
    "max_retries": 4,
}
_RUNTIME_STATES = {}
_ACTIVE_RUNTIME_TOKEN = None

_RUNTIME_STATE_VARS = (
    "_BOUND_BACKEND",
    "_K_A",
    "_K_D",
    "_K_H",
    "_K_F_ADV",
    "_K_F_DIFF",
    "_K_S_COAG",
    "_K_S_HYD",
    "_K_KERNEL",
    "_K_P_FRAG",
    "_K_ST",
    "_K_VRAD",
    "_K_VREL_BROWN",
    "_K_VREL_AZI",
    "_K_VREL_RAD",
    "_K_VREL_TURB",
    "_K_VREL_VERT",
    "_K_COAG_PARAMS",
    "_K_IMPL_1_DIRECT",
    "_K_IMPLICIT_FLOOR_RETRY",
    "_K_JACOBIAN",
    "_K_INTERP_TO_INTERFACES",
    "_JCOAG_GEN_MODE",
    "_S_COAG_MODE",
    "_F_DIFF_MODE",
    "_SCATTER_MODE",
    "_CUPY_DUST_SOLVER_MODE",
    "_VREL_TURB_MODE",
    "_VREL_TOT_MODE",
    "_P_FRAG_MODE",
    "_COLLISION_KERNEL_MODE",
    "_CUPY_A_BUILD_MODE",
    "_JCOAG_CHUNK_SIZE_OVERRIDE",
    "_SANITIZER_CONFIG",
    "_GAS_FLOOR_FREEZE_CONFIG",
    "_DUST_TRANSPORT_INACTIVE_FLOOR_CONFIG",
    "_IMPLICIT_FLOOR_RETRY_CONFIG",
)


def _fresh_runtime_state():
    return {
        "_BOUND_BACKEND": None,
        "_K_A": None,
        "_K_D": None,
        "_K_H": None,
        "_K_F_ADV": None,
        "_K_F_DIFF": None,
        "_K_S_COAG": None,
        "_K_S_HYD": None,
        "_K_KERNEL": None,
        "_K_P_FRAG": None,
        "_K_ST": None,
        "_K_VRAD": None,
        "_K_VREL_BROWN": None,
        "_K_VREL_AZI": None,
        "_K_VREL_RAD": None,
        "_K_VREL_TURB": None,
        "_K_VREL_VERT": None,
        "_K_COAG_PARAMS": None,
        "_K_IMPL_1_DIRECT": None,
        "_K_IMPLICIT_FLOOR_RETRY": None,
        "_K_JACOBIAN": None,
        "_K_INTERP_TO_INTERFACES": None,
        "_JCOAG_GEN_MODE": "baseline",
        "_S_COAG_MODE": "baseline",
        "_F_DIFF_MODE": "baseline",
        "_SCATTER_MODE": "addat",
        "_CUPY_DUST_SOLVER_MODE": "sparse",
        "_VREL_TURB_MODE": "baseline",
        "_VREL_TOT_MODE": "baseline",
        "_P_FRAG_MODE": "baseline",
        "_COLLISION_KERNEL_MODE": "baseline",
        "_CUPY_A_BUILD_MODE": "identity_sub",
        "_JCOAG_CHUNK_SIZE_OVERRIDE": None,
        "_SANITIZER_CONFIG": {
            "enabled": False,
            "mode": "implicit",
            "tiny_max_ratio": 10000.0,
            "born_ratio": 1.0,
            "stop_thresh_mearth": 1.0e-2,
            "tinymax_report": False,
            "tinymax_report_cadence": 1,
        },
        "_GAS_FLOOR_FREEZE_CONFIG": {
            "enabled": True,
            "ratio": 1.0e3,
        },
        "_DUST_TRANSPORT_INACTIVE_FLOOR_CONFIG": {
            "enabled": False,
            "ratio": 1.0,
        },
        "_IMPLICIT_FLOOR_RETRY_CONFIG": {
            "enabled": True,
            "threshold_mearth": 1.0e-6,
            "shrink": 0.5,
            "max_retries": 8,
        },
    }


def _capture_runtime_state():
    return {name: globals()[name] for name in _RUNTIME_STATE_VARS}


def _restore_runtime_state(state):
    for name, value in state.items():
        globals()[name] = value


def _switch_runtime_state(runtime_token):
    global _ACTIVE_RUNTIME_TOKEN
    _dust_common_impl.switch_runtime_state(runtime_token)
    _dust_cupy_impl.switch_runtime_state(runtime_token)
    _dust_numpy_impl.switch_runtime_state(runtime_token)
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


def _load_sanitizer_config():
    mode = os.getenv("DUSTPY_SANITIZER_MODE", "implicit").strip().lower()
    if mode not in ("implicit", "both"):
        mode = "implicit"
    return {
        "enabled": _env_bool("DUSTPY_SANITIZER_ENABLE", default=False),
        "mode": mode,
        "tiny_max_ratio": _env_float("DUSTPY_SANITIZER_TINY_MAX_RATIO", 10000.0),
        "born_ratio": _env_float("DUSTPY_SANITIZER_BORN_RATIO", 1.0),
        "stop_thresh_mearth": _env_float("DUSTPY_SANITIZER_STOP_THRESH_MEARTH", 1.0e-2),
        "tinymax_report": _env_bool("DUSTPY_SANITIZER_TINYMAX_REPORT", default=False),
        "tinymax_report_cadence": max(1, int(_env_float("DUSTPY_SANITIZER_TINYMAX_REPORT_CADENCE", 1.0))),
    }


def _load_gas_floor_freeze_config():
    ratio = _env_float("DUSTPY_GAS_FLOOR_FREEZE_RATIO", 0.0)
    if not np.isfinite(ratio):
        ratio = 0.0
    ratio = max(float(ratio), 0.0)
    return {
        "enabled": ratio > 0.0,
        "ratio": ratio,
    }


def _load_dust_transport_inactive_floor_config():
    ratio = _env_float("DUSTPY_DUST_TRANSPORT_INACTIVE_FLOOR_RATIO", 1.0)
    if not np.isfinite(ratio):
        ratio = 1.0
    ratio = max(float(ratio), 0.0)
    enabled = _env_bool("DUSTPY_DUST_TRANSPORT_INACTIVE_FLOOR_ENABLE", default=False) and ratio > 0.0
    return {
        "enabled": enabled,
        "ratio": ratio,
    }


def _load_implicit_floor_retry_config():
    shrink = _env_float("DUSTPY_IMPLICIT_FLOOR_RETRY_SHRINK", 0.5)
    if not np.isfinite(shrink):
        shrink = 0.5
    shrink = min(max(float(shrink), 1.0e-6), 0.99)
    threshold = _env_float("DUSTPY_IMPLICIT_FLOOR_RETRY_THRESH_MEARTH", 1.0e-6)
    if not np.isfinite(threshold):
        threshold = 1.0e-6
    threshold = max(float(threshold), 0.0)
    max_retries = _env_float("DUSTPY_IMPLICIT_FLOOR_RETRY_MAX_RETRIES", 4.0)
    if not np.isfinite(max_retries):
        max_retries = 4.0
    max_retries = max(0, int(max_retries))
    enabled = _env_bool("DUSTPY_IMPLICIT_FLOOR_RETRY_ENABLE", default=False) and threshold > 0.0
    return {
        "enabled": enabled,
        "threshold_mearth": threshold,
        "shrink": shrink,
        "max_retries": max_retries,
    }


def _ensure_dust_san_state(dust):
    state = getattr(dust, "san", None)
    if state is None:
        state = SimpleNamespace(
            enabled=bool(_SANITIZER_CONFIG.get("enabled", False)),
            mode=str(_SANITIZER_CONFIG.get("mode", "implicit")),
            stop_thresh_mearth=float(_SANITIZER_CONFIG.get("stop_thresh_mearth", 1.0e-2)),
            prev_ratio=None,
            Mdust0_mearth=None,
            dM_total_mearth=0.0,
            last_n_clipped=0,
            active_cycles=0,
            clip_cycles=0,
            sum_clipped_cells=0,
            max_cells_per_cycle=0,
            first_clip_cycle=None,
            first_clip_t_years=None,
            first_clip_cells=None,
            tinymax_block_cycles=0,
            tinymax_block_cells=0,
            tinymax_block_max_ratio=0.0,
            half_idx=None,
        )
        dust.san = state
    else:
        state.enabled = bool(_SANITIZER_CONFIG.get("enabled", False))
        state.mode = str(_SANITIZER_CONFIG.get("mode", "implicit"))
        state.stop_thresh_mearth = float(_SANITIZER_CONFIG.get("stop_thresh_mearth", 1.0e-2))
    return state


def _maybe_retry_implicit_floor_injection_numpy(x0, Y0, dx, sigma1):
    cfg = _IMPLICIT_FLOOR_RETRY_CONFIG
    if not cfg.get("enabled", True):
        return False
    sigma1_np = _field_data(sigma1)
    floor_np = _field_data(Y0._owner.dust.SigmaFloor)
    area_np = _field_data(Y0._owner.grid.A)
    delta_np = np.maximum(0.1 * floor_np - sigma1_np, 0.0)
    added_mearth = float(np.sum(delta_np * area_np[:, None] / c.M_earth))
    if added_mearth <= float(cfg.get("threshold_mearth", 0.0)):
        return False
    retry = _ensure_dust_floor_retry_state(Y0._owner.dust)
    if retry.step_count >= int(cfg.get("max_retries", 0)):
        _record_implicit_floor_retry_exhausted(Y0._owner, added_mearth)
        return False
    _record_implicit_floor_retry(Y0._owner, added_mearth)
    dx_host = float(dx)
    shrink = float(cfg.get("shrink", 0.5))
    x0.suggest(max(dx_host * shrink, 1.0e-300), reset=True)
    return True


def _maybe_retry_implicit_floor_injection_cupy(x0, Y0, dx, sigma1):
    cfg = _IMPLICIT_FLOOR_RETRY_CONFIG
    if not cfg.get("enabled", True):
        return False
    sigma1_cp = _field_data(sigma1)
    floor_cp = _field_data(Y0._owner.dust.SigmaFloor)
    area_cp = _field_data(Y0._owner.grid.A)
    delta_cp = cp.maximum(0.1 * floor_cp - sigma1_cp, 0.0)
    added_mearth = float(to_numpy(cp.sum(delta_cp * area_cp[:, None] / c.M_earth)))
    if added_mearth <= float(cfg.get("threshold_mearth", 0.0)):
        return False
    retry = _ensure_dust_floor_retry_state(Y0._owner.dust)
    if retry.step_count >= int(cfg.get("max_retries", 0)):
        _record_implicit_floor_retry_exhausted(Y0._owner, added_mearth)
        return False
    _record_implicit_floor_retry(Y0._owner, added_mearth)
    dx_host = float(dx)
    shrink = float(cfg.get("shrink", 0.5))
    x0.suggest(max(dx_host * shrink, 1.0e-300), reset=True)
    return True


def _maybe_retry_implicit_floor_injection(x0, Y0, dx, sigma1):
    return _K_IMPLICIT_FLOOR_RETRY(x0, Y0, dx, sigma1)


def _interp_to_interfaces(values, r, ri):
    return _K_INTERP_TO_INTERFACES(_field_data(values), _field_data(r), _field_data(ri))


def bind_backend_kernels(backend=None, force=False, runtime_token=None):
    """Bind hot dust kernels to backend-specific implementations once."""
    global _BOUND_BACKEND, _K_A, _K_D, _K_H, _K_F_ADV, _K_F_DIFF, _K_S_COAG, _K_S_HYD
    global _K_KERNEL, _K_P_FRAG, _K_ST, _K_VRAD, _K_VREL_BROWN, _K_VREL_AZI, _K_VREL_RAD, _K_VREL_TURB, _K_VREL_VERT
    global _K_COAG_PARAMS, _K_IMPL_1_DIRECT, _K_IMPLICIT_FLOOR_RETRY, _K_JACOBIAN, _K_INTERP_TO_INTERFACES
    global _JCOAG_GEN_MODE, _S_COAG_MODE, _F_DIFF_MODE, _SCATTER_MODE, _CUPY_DUST_SOLVER_MODE
    global _VREL_TURB_MODE, _VREL_TOT_MODE, _P_FRAG_MODE, _COLLISION_KERNEL_MODE
    global _CUPY_A_BUILD_MODE
    global _JCOAG_CHUNK_SIZE_OVERRIDE, _SANITIZER_CONFIG, _GAS_FLOOR_FREEZE_CONFIG, _DUST_TRANSPORT_INACTIVE_FLOOR_CONFIG, _IMPLICIT_FLOOR_RETRY_CONFIG

    _switch_runtime_state(runtime_token)
    backend = get_backend() if backend is None else backend
    bind_sparse_solver(backend=backend, force=force, runtime_token=runtime_token)
    cupy_kernel_optimized = _env_bool("DUSTPY_CUPY_KERNEL_OPTIMIZED", default=True) if backend == "cupy" else False
    cupy_solver_optimized = _env_bool("DUSTPY_CUPY_SOLVER_OPTIMIZED", default=True) if backend == "cupy" else False
    default_jcoag_gen_mode = "fused_spmm" if cupy_kernel_optimized else "baseline"
    default_scoag_mode = "fused_spmm" if cupy_kernel_optimized else "baseline"
    default_fdiff_mode = "elementwise" if cupy_kernel_optimized else "baseline"
    default_vrel_turb_mode = "elementwise" if cupy_kernel_optimized else "baseline"
    default_vrel_tot_mode = "elementwise" if cupy_kernel_optimized else "baseline"
    default_p_frag_mode = "elementwise" if cupy_kernel_optimized else "baseline"
    default_collision_kernel_mode = "elementwise" if cupy_kernel_optimized else "baseline"
    default_cupy_a_build_mode = "diag_inplace" if cupy_solver_optimized else "identity_sub"

    jcoag_gen_mode = os.getenv("DUSTPY_JCOAG_GEN_MODE", default_jcoag_gen_mode).strip().lower()
    if jcoag_gen_mode not in ("baseline", "fused_spmm"):
        jcoag_gen_mode = default_jcoag_gen_mode
    scoag_mode = os.getenv("DUSTPY_S_COAG_MODE", default_scoag_mode).strip().lower()
    if scoag_mode not in ("baseline", "fused_spmm"):
        scoag_mode = default_scoag_mode
    fdiff_mode = os.getenv("DUSTPY_F_DIFF_MODE", default_fdiff_mode).strip().lower()
    if fdiff_mode not in ("baseline", "elementwise"):
        fdiff_mode = default_fdiff_mode
    scatter_mode = os.getenv("DUSTPY_SCATTER_MODE", "addat").strip().lower()
    if scatter_mode not in ("addat", "scatter", "rawkernel"):
        scatter_mode = "addat"
    vrel_turb_mode = os.getenv("DUSTPY_VREL_TURB_MODE", default_vrel_turb_mode).strip().lower()
    if vrel_turb_mode not in ("baseline", "elementwise"):
        vrel_turb_mode = default_vrel_turb_mode
    vrel_tot_mode = os.getenv("DUSTPY_VREL_TOT_MODE", default_vrel_tot_mode).strip().lower()
    if vrel_tot_mode not in ("baseline", "elementwise"):
        vrel_tot_mode = default_vrel_tot_mode
    p_frag_mode = os.getenv("DUSTPY_P_FRAG_MODE", default_p_frag_mode).strip().lower()
    if p_frag_mode not in ("baseline", "elementwise"):
        p_frag_mode = default_p_frag_mode
    collision_kernel_mode = os.getenv("DUSTPY_COLLISION_KERNEL_MODE", default_collision_kernel_mode).strip().lower()
    if collision_kernel_mode not in ("baseline", "elementwise"):
        collision_kernel_mode = default_collision_kernel_mode
    cupy_a_build_mode = os.getenv("DUSTPY_CUPY_A_BUILD_MODE", default_cupy_a_build_mode).strip().lower()
    if cupy_a_build_mode not in ("identity_sub", "diag_inplace"):
        cupy_a_build_mode = default_cupy_a_build_mode
    jcoag_chunk_size_raw = os.getenv("DUSTPY_JCOAG_CHUNK_SIZE", "auto").strip().lower()
    jcoag_chunk_size_override = None
    if jcoag_chunk_size_raw and jcoag_chunk_size_raw != "auto":
        try:
            jcoag_chunk_size_override = max(1, int(jcoag_chunk_size_raw))
        except Exception:
            jcoag_chunk_size_override = None
    cupy_dust_solver_mode = _get_cupy_dust_solver_mode() if backend == "cupy" else "sparse"
    sanitizer_cfg = _load_sanitizer_config()
    gas_floor_freeze_cfg = _load_gas_floor_freeze_config()
    dust_transport_inactive_floor_cfg = _load_dust_transport_inactive_floor_config()
    implicit_floor_retry_cfg = _load_implicit_floor_retry_config()
    _dust_common_impl.configure_runtime(gas_floor_freeze_cfg, dust_transport_inactive_floor_cfg)
    _dust_cupy_impl.configure_runtime(scatter_mode, scoag_mode, jcoag_gen_mode, jcoag_chunk_size_override, fdiff_mode, vrel_turb_mode, p_frag_mode, collision_kernel_mode)
    if (
        (not force)
        and backend == _BOUND_BACKEND
        and _K_A is not None
        and jcoag_gen_mode == _JCOAG_GEN_MODE
        and scoag_mode == _S_COAG_MODE
        and fdiff_mode == _F_DIFF_MODE
        and scatter_mode == _SCATTER_MODE
        and cupy_dust_solver_mode == _CUPY_DUST_SOLVER_MODE
        and vrel_turb_mode == _VREL_TURB_MODE
        and vrel_tot_mode == _VREL_TOT_MODE
        and p_frag_mode == _P_FRAG_MODE
        and collision_kernel_mode == _COLLISION_KERNEL_MODE
        and cupy_a_build_mode == _CUPY_A_BUILD_MODE
        and jcoag_chunk_size_override == _JCOAG_CHUNK_SIZE_OVERRIDE
        and sanitizer_cfg == _SANITIZER_CONFIG
        and gas_floor_freeze_cfg == _GAS_FLOOR_FREEZE_CONFIG
        and dust_transport_inactive_floor_cfg == _DUST_TRANSPORT_INACTIVE_FLOOR_CONFIG
        and implicit_floor_retry_cfg == _IMPLICIT_FLOOR_RETRY_CONFIG
    ):
        return

    _K_A = select_backend({"cupy": _a_cupy}, backend=backend, default=_a_fortran)
    _K_D = select_backend({"cupy": _D_cupy}, backend=backend, default=_D_fortran)
    _K_H = select_backend({"cupy": _H_cupy}, backend=backend, default=_H_fortran)
    _K_F_ADV = select_backend({"cupy": _F_adv_cupy}, backend=backend, default=_F_adv_fortran)
    _K_F_DIFF = select_backend({"cupy": _F_diff_cupy}, backend=backend, default=_F_diff_fortran)
    _K_S_COAG = select_backend({"cupy": _S_coag_cupy}, backend=backend, default=_S_coag_fortran)
    _K_S_HYD = select_backend({"cupy": _S_hyd_cupy}, backend=backend, default=_S_hyd_fortran)
    _K_KERNEL = select_backend({"cupy": _kernel_cupy}, backend=backend, default=_kernel_fortran)
    _K_P_FRAG = select_backend({"cupy": _p_frag_cupy}, backend=backend, default=_p_frag_fortran)
    _K_ST = select_backend({"cupy": _St_Epstein_StokesI_cupy}, backend=backend, default=_St_Epstein_StokesI_fortran)
    _K_VRAD = select_backend({"cupy": _vrad_cupy}, backend=backend, default=_vrad_fortran)
    _K_VREL_BROWN = select_backend({"cupy": _vrel_brownian_motion_cupy}, backend=backend, default=_vrel_brownian_motion_fortran)
    _K_VREL_AZI = select_backend({"cupy": _vrel_azimuthal_drift_cupy}, backend=backend, default=_vrel_azimuthal_drift_fortran)
    _K_VREL_RAD = select_backend({"cupy": _vrel_radial_drift_cupy}, backend=backend, default=_vrel_radial_drift_fortran)
    _K_VREL_TURB = select_backend({"cupy": _vrel_turbulent_motion_cupy}, backend=backend, default=_vrel_turbulent_motion_fortran)
    _K_VREL_VERT = select_backend({"cupy": _vrel_vertical_settling_cupy}, backend=backend, default=_vrel_vertical_settling_fortran)
    _K_COAG_PARAMS = select_backend({"cupy": _coagulation_parameters_python}, backend=backend, default=_coagulation_parameters_fortran)
    if backend == "cupy" and cupy_dust_solver_mode == "dense_gpu":
        _K_IMPL_1_DIRECT = _f_impl_1_direct_cupy_dense
        _K_JACOBIAN = _jacobian_cupy_dense
    else:
        _K_IMPL_1_DIRECT = select_backend({"cupy": _f_impl_1_direct_cupy}, backend=backend, default=_f_impl_1_direct_numpy)
        _K_JACOBIAN = select_backend({"cupy": _jacobian_cupy}, backend=backend, default=_jacobian_numpy)
    _K_IMPLICIT_FLOOR_RETRY = select_backend({"cupy": _maybe_retry_implicit_floor_injection_cupy}, backend=backend, default=_maybe_retry_implicit_floor_injection_numpy)
    _K_INTERP_TO_INTERFACES = select_backend({"cupy": _interp_to_interfaces_cupy}, backend=backend, default=_interp_to_interfaces_numpy)

    _JCOAG_GEN_MODE = jcoag_gen_mode
    _S_COAG_MODE = scoag_mode
    _F_DIFF_MODE = fdiff_mode
    _SCATTER_MODE = scatter_mode
    _CUPY_DUST_SOLVER_MODE = cupy_dust_solver_mode
    _VREL_TURB_MODE = vrel_turb_mode
    _VREL_TOT_MODE = vrel_tot_mode
    _P_FRAG_MODE = p_frag_mode
    _COLLISION_KERNEL_MODE = collision_kernel_mode
    _CUPY_A_BUILD_MODE = cupy_a_build_mode
    _JCOAG_CHUNK_SIZE_OVERRIDE = jcoag_chunk_size_override
    _SANITIZER_CONFIG = sanitizer_cfg
    _GAS_FLOOR_FREEZE_CONFIG = gas_floor_freeze_cfg
    _DUST_TRANSPORT_INACTIVE_FLOOR_CONFIG = dust_transport_inactive_floor_cfg
    _IMPLICIT_FLOOR_RETRY_CONFIG = implicit_floor_retry_cfg

    _dust_cupy_impl.reset_runtime_caches()
    _dust_numpy_impl.reset_runtime_caches()

    _BOUND_BACKEND = backend
    if runtime_token is not None:
        _RUNTIME_STATES[runtime_token] = _capture_runtime_state()


# Initialize default bindings at import time.


def boundary(sim):
    """Function sets the boundary condition of dust surface density.
    Not implemented, yet.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    _refresh_dust_boundary_views(sim)
    if is_dust_inner_outflow_only_enabled(sim):
        _apply_inner_boundary_selective(sim, _inner_diode_block_mask(sim))
    else:
        sim.dust.boundary.inner.setboundary()
    sim.dust.boundary.outer.setboundary()


def enforce_floor_value(sim):
    """Function enforces floor value onto dust surface density.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    sigma = _field_data(sim.dust.Sigma)
    floor = _field_data(sim.dust.SigmaFloor)
    target = xp.where(sigma > floor, sigma, 0.1 * floor)
    delta = xp.maximum(target - sigma, 0.0)
    added_mearth = float(to_numpy(xp.sum(delta * _field_data(sim.grid.A)[:, None] / c.M_earth)))
    if added_mearth > 0.0:
        topup = _ensure_dust_floor_topup_state(sim.dust)
        topup.total_mearth += added_mearth
        topup.last_mearth = added_mearth
        topup.count += 1
    sim.dust.Sigma = target


def _sanitize_floorborn_islands(sim):
    """Optionally clip floor-born tiny islands after implicit/explicit floor enforcement."""
    cfg = _SANITIZER_CONFIG
    dust = sim.dust
    san = _ensure_dust_san_state(dust)
    san.last_n_clipped = 0

    if not cfg.get("enabled", False):
        return

    sigma = _field_data(dust.Sigma)
    sigma_floor = _field_data(dust.SigmaFloor)

    if sigma.shape != sigma_floor.shape:
        return

    ratio_now = xp.where(sigma_floor > 0.0, sigma / sigma_floor, 0.0)

    prev_ratio = san.prev_ratio
    if prev_ratio is None or prev_ratio.shape != ratio_now.shape:
        san.prev_ratio = ratio_now.copy()
        return

    t_years = float(sim.t) / c.year

    if san.Mdust0_mearth is None:
        area = _field_data(sim.grid.A)
        mdust0 = xp.sum(sigma * area[:, None]) / c.M_earth
        mdust0 = float(to_numpy(mdust0))
        san.Mdust0_mearth = max(mdust0, 1e-300)

    if not hasattr(dust, "dM_sanitizer"):
        dust.addfield("dM_sanitizer", xp.zeros_like(dust.Sigma), description="Cumulative sanitizer-removed dust mass [M_earth] per cell")
    dM_sanitizer = _field_data(dust.dM_sanitizer)

    _, nm = sigma.shape
    tiny_max = float(cfg["tiny_max_ratio"])
    born_ratio = float(cfg["born_ratio"])
    prev_floor_max = 0.11
    nrad = 1

    prev_floor = prev_ratio <= prev_floor_max
    prev_floor_i = prev_floor.astype(xp.int64, copy=False)
    suffix_ok = xp.flip(xp.cumprod(xp.flip(prev_floor_i, axis=1), axis=1), axis=1).astype(bool)

    half_idx = san.half_idx
    if half_idx is None or int(getattr(half_idx, "size", -1)) != int(nm):
        # Map each mass bin m_j to the closest bin to m_j / 2 on the actual grid.
        m_np = to_numpy(_field_data(sim.grid.m)).astype(np.float64, copy=False)
        m_half = 0.5 * m_np
        idx_hi = np.searchsorted(m_np, m_half, side="left")
        idx_hi = np.clip(idx_hi, 0, nm - 1)
        idx_lo = np.clip(idx_hi - 1, 0, nm - 1)
        use_lo = np.abs(m_np[idx_lo] - m_half) <= np.abs(m_np[idx_hi] - m_half)
        half_idx_np = np.where(use_lo, idx_lo, idx_hi).astype(np.int64, copy=False)
        half_idx = xp.asarray(half_idx_np, dtype=xp.int64)
        san.half_idx = half_idx
    row_ok = suffix_ok[:, half_idx]

    radial_ok = xp.ones_like(row_ok, dtype=bool)
    for dr in range(-nrad, nrad + 1):
        shifted = xp.zeros_like(row_ok, dtype=bool)
        if dr == 0:
            shifted[...] = row_ok
        elif dr > 0:
            shifted[:-dr, :] = row_ok[dr:, :]
        else:
            k = -dr
            shifted[k:, :] = row_ok[:-k, :]
        radial_ok &= shifted

    candidate = (
        (ratio_now > born_ratio)
        & (ratio_now <= tiny_max)
        & radial_ok
    )
    candidate[0, :] = False
    candidate[-1, :] = False

    tinymax_blocked = (
        (ratio_now > born_ratio)
        & (ratio_now > tiny_max)
        & radial_ok
    )
    tinymax_blocked[0, :] = False
    tinymax_blocked[-1, :] = False
    n_tinymax_blocked = int(to_numpy(tinymax_blocked.sum()))
    if n_tinymax_blocked > 0:
        san.tinymax_block_cycles += 1
        san.tinymax_block_cells += n_tinymax_blocked
        max_ratio_blocked = float(to_numpy(xp.max(xp.where(tinymax_blocked, ratio_now, 0.0))))
        if max_ratio_blocked > san.tinymax_block_max_ratio:
            san.tinymax_block_max_ratio = max_ratio_blocked
        if cfg.get("tinymax_report", False):
            cadence = max(1, int(cfg.get("tinymax_report_cadence", 1)))
            cycle_now = int(sim.RL_count_cycle)
            if cycle_now % cadence == 0:
                print(
                    "[SAN_TINYMAX_BLOCK] "
                    f"cycle={cycle_now}, t={t_years:.3f}yr, "
                    f"n_cells={n_tinymax_blocked}, "
                    f"tiny_max_ratio={tiny_max:.6e}, "
                    f"max_ratio={max_ratio_blocked:.6e}"
                )

    target = 0.1 * sigma_floor
    delta = xp.where(candidate, sigma - target, 0.0)
    to_clip = delta > 0.0
    n_clipped = int(to_numpy(to_clip.sum()))

    san.active_cycles += 1
    san.last_n_clipped = n_clipped
    if n_clipped == 0:
        san.prev_ratio = ratio_now.copy()
        return

    sigma[to_clip] = target[to_clip]
    area = _field_data(sim.grid.A)
    dM_inc = delta * to_clip * (area[:, None] / c.M_earth)
    dM_sanitizer[...] = dM_sanitizer + dM_inc
    dM_cycle = float(to_numpy(dM_inc.sum()))
    san.dM_total_mearth += dM_cycle
    san.clip_cycles += 1
    san.sum_clipped_cells += n_clipped
    if n_clipped > san.max_cells_per_cycle:
        san.max_cells_per_cycle = n_clipped
    if san.first_clip_cycle is None:
        san.first_clip_cycle = int(sim.RL_count_cycle)
        san.first_clip_t_years = t_years
        san.first_clip_cells = n_clipped

    ratio_after = xp.where(sigma_floor > 0.0, sigma / sigma_floor, 0.0)
    san.prev_ratio = ratio_after.copy()

    stop_thresh_mearth = float(cfg["stop_thresh_mearth"])
    if stop_thresh_mearth > 0.0:
        if san.dM_total_mearth > stop_thresh_mearth:
            print(
                "[SANITIZER_STOP] "
                f"cycle={int(sim.RL_count_cycle)}, t={t_years:.3f}yr, "
                f"dM_san={san.dM_total_mearth:.6e} Mearth, "
                f"stop_thresh_mearth={stop_thresh_mearth:.6e}"
            )
            if sim.writer is not None:
                datadir = sim.writer.datadir
                try:
                    sim.writer.write(sim, int(sim.RL_count_cycle), True, filename=os.path.join(str(datadir), "sanitizer_stop.hdf5"))
                except Exception:
                    pass
                try:
                    sim.writer.writedump(sim, filename=os.path.join(str(datadir), "frame_sanitizer_stop.dmp"))
                except Exception:
                    pass
            raise SystemExit("DustPy sanitizer stop threshold exceeded.")

def prepare(sim):
    """Function prepares implicit dust integration step.
    It stores the current value of the surface density in a hidden field.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    # Setting external sources at boundaries to zero
    sim.dust.S.ext[0] = 0.
    sim.dust.S.ext[-1] = 0.
    # Storing current surface density
    sim.dust._SigmaOld[...] = sim.dust.Sigma[...]
    _ensure_dust_floor_retry_state(sim.dust).step_count = 0


def finalize_explicit(sim):
    """Function finalizes integration step.

    Parameters
    ----------
    sim : Frame
        Parent integration frame"""
    # Closed-box mode must not re-impose value/gradient boundary forcing.
    if not is_zero_flux_enabled(sim):
        boundary(sim)
    enforce_floor_value(sim)
    if _SANITIZER_CONFIG.get("enabled", False) and _SANITIZER_CONFIG.get("mode", "implicit") == "both":
        _sanitize_floorborn_islands(sim)


def finalize_implicit(sim):
    """Function finalizes implicit integration step.

    Parameters
    ----------
    sim : Frame
        Parent integration frame"""
    # Closed-box mode must not re-impose value/gradient boundary forcing.
    if not is_zero_flux_enabled(sim):
        boundary(sim)
    enforce_floor_value(sim)
    if _SANITIZER_CONFIG.get("enabled", False):
        _sanitize_floorborn_islands(sim)
    sim.dust.v.rad.update()
    sim.dust.Fi.update()
    sim.dust.S.hyd.update()
    sim.dust.S.coag.update()
    set_implicit_boundaries(sim)


def set_implicit_boundaries(sim):
    """Function calculates the fluxes at the boundaries after the implicit integration step.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    # Total source terms
    sim.dust.S.tot[0] = (sim.dust.Sigma[0] -
                         sim.dust._SigmaOld[0])/(sim.t.prevstepsize+1.e-100)
    sim.dust.S.tot[-1] = (sim.dust.Sigma[-1] -
                          sim.dust._SigmaOld[-1])/(sim.t.prevstepsize+1.e-100)
    diode_block_mask = _inner_diode_block_mask(sim)
    if is_zero_flux_enabled(sim):
        sim.dust.Fi.adv[0] = 0.0
        sim.dust.Fi.adv[-1] = 0.0
        sim.dust.Fi.tot[0] = 0.0
        sim.dust.Fi.tot[-1] = 0.0
    else:
        # Hydrodynamic source terms
        sim.dust.S.hyd[0] = sim.dust.S.tot[0]
        sim.dust.S.hyd[-1] = sim.dust.S.tot[-1]
        # Fluxes
        sim.dust.Fi.adv[0] = (0.5*sim.dust.S.hyd[0]*(sim.grid.ri[1]**2 -
                                                     sim.grid.ri[0]**2) + sim.grid.ri[1]*sim.dust.Fi.adv[1])/sim.grid.ri[0]
        sim.dust.Fi.adv[-1] = (sim.dust.Fi.adv[-2]*sim.grid.ri[-2] - 0.5*sim.dust.S.hyd[-1]
                               * (sim.grid.ri[-1]**2-sim.grid.ri[-2]**2))/sim.grid.ri[-1]
        if is_dust_inner_outflow_only_enabled(sim):
            sim.dust.Fi.adv[0] = xp.minimum(sim.dust.Fi.adv[0], 0.0)
            if diode_block_mask is not None:
                sim.dust.Fi.adv[0][diode_block_mask] = 0.0
        sim.dust.Fi.tot[0] = sim.dust.Fi.adv[0]
        sim.dust.Fi.tot[-1] = sim.dust.Fi.adv[-1]
    _apply_gas_floor_freeze_to_flux(sim, sim.dust.Fi.adv)
    _apply_gas_floor_freeze_to_flux(sim, sim.dust.Fi.tot)


def dt_adaptive(sim):
    """Function returns the adaptive time step.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    dt : float
        Dust time step"""
    return sim.t.suggested


def dt(sim):
    """Function calculates the time step from the dust sources.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    dt : float
        Dust time step"""
    if xp.any(sim.dust.S.tot[1:-1, ...] < 0.):
        mask = (sim.dust.Sigma > sim.dust.SigmaFloor) & (sim.dust.S.tot < 0.)
        mask[0, :] = False
        mask[-1:, :] = False
        rate = sim.dust.Sigma[mask] / sim.dust.S.tot[mask]
        try:
            return xp.min(xp.abs(rate))
        except:
            return None


def a(sim):
    """Function calculates the particle size from the solid density and the filling factor.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    a : Field
        Particle sizes"""
    return _K_A(sim)


def D(sim):
    """Function calculates the dust diffusivity.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    D : Field
        Dust diffusivity

    Notes
    -----
    The diffusivity at the first and last two radial
    grid cells will be set to zero to avoid unwanted
    behavior at the boundaries."""
    return _K_D(sim)


def eps(sim):
    """Function returns the vertically integrated dust-to-gas ratio.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    eps : Field
        vertically integrated dust-to-gas ratio"""
    return xp.sum(sim.dust.Sigma, axis=-1) / sim.gas.Sigma


def F_adv(sim, Sigma=None):
    """Function calculates the advective flux at the cell interfaces. It is linearly interpolating
    the velocities onto the grid cel interfaces and is assuming
    vi(0, :) = vi(1, :) and vi(Nr, :) = vi(Nr-1, :).

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    Sigma : Field, optional, default : None
        Surface density to be used if not None

    Returns
    -------
    Fi : Field
        Advective mass fluxes through the grid cell interfaces"""
    Fi = _K_F_ADV(sim, Sigma=Sigma)
    if is_zero_flux_enabled(sim):
        Fi[0, :] = 0.0
        Fi[-1, :] = 0.0
    elif is_dust_inner_outflow_only_enabled(sim):
        # Inner diode BC: allow only outflow to the star, block inward supply.
        Fi[0, :] = xp.minimum(Fi[0, :], 0.0)
        diode_block_mask = _inner_diode_block_mask(sim)
        if diode_block_mask is not None:
            Fi[0, diode_block_mask] = 0.0
    return Fi


def F_diff(sim, Sigma=None):
    '''Function calculates the diffusive flux at the cell interfaces'''
    Fi = _K_F_DIFF(sim, Sigma=Sigma)
    _apply_gas_floor_freeze_to_flux(sim, Fi)
    return Fi


def F_tot(sim, Sigma=None):
    """Function calculates the total mass fluxes through grid cell interfaces.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    Sigma : Field, optional, default : None
        Surface density to be used if not None

    Returns
    -------
    Ftot : Field
        Total mass flux through interfaces"""
    Fi = xp.zeros_like(sim.dust.Fi.tot)
    if Sigma is None:
        Fdiff = sim.dust.Fi.diff
        Fadv = sim.dust.Fi.adv
    else:
        Fdiff = sim.dust.Fi.diff.updater.beat(sim, Sigma=Sigma)
        Fadv = sim.dust.Fi.adv.updater.beat(sim, Sigma=Sigma)
    if Fdiff is not None:
        Fi += _field_data(Fdiff)
    if Fadv is not None:
        Fi += _field_data(Fadv)
    if is_zero_flux_enabled(sim):
        Fi[0, :] = 0.0
        Fi[-1, :] = 0.0
    elif is_dust_inner_outflow_only_enabled(sim):
        # Inner diode BC: allow only outflow to the star, block inward supply.
        Fi[0, :] = xp.minimum(Fi[0, :], 0.0)
        diode_block_mask = _inner_diode_block_mask(sim)
        if diode_block_mask is not None:
            Fi[0, diode_block_mask] = 0.0
    _apply_gas_floor_freeze_to_flux(sim, Fi)
    return Fi


def H(sim):
    """Function calculates the dust scale height according Dubrulle et al. (1995).

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    H : Field
        Dust scale heights"""
    return _K_H(sim)


def _jacobian_numpy(sim, x, dx=None, *args, **kwargs):
    """Function calculates the Jacobian for implicit dust integration.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    x : IntVar
        Integration variable
    dx : float, optional, default : None
        stepsize
    args : additional positional arguments
    kwargs : additional keyworda arguments

    Returns
    -------
    jac : Sparse matrix
        Dust Jacobian

    Notes
    -----
    Function returns the Jacobian for ``Simulation.dust.Sigma.ravel()``
    instead of ``Simulation.dust.Sigma``. The Jacobian is stored as
    sparse matrix."""

    # Parameters for function call
    A = _field_data(sim.dust.coagulation.A)
    cstick = _field_data(sim.dust.coagulation.stick)
    eps = _field_data(sim.dust.coagulation.eps)
    ilf = _field_data(sim.dust.coagulation.lf_ind)
    irm = _field_data(sim.dust.coagulation.rm_ind)
    istick = _field_data(sim.dust.coagulation.stick_ind)
    m = _field_data(sim.grid.m)
    phi = _field_data(sim.dust.coagulation.phi)
    Rf = _field_data(sim.dust.kernel * sim.dust.p.frag)
    Rs = _field_data(sim.dust.kernel * sim.dust.p.stick)
    SigD = _field_data(sim.dust.Sigma)
    SigDfloor = _field_data(sim.dust.SigmaFloor)

    # Helper variables for convenience
    if dx is None:
        dt = x.stepsize
    else:
        dt = dx
    try:
        dt = float(dt)
    except Exception:
        dt = float(to_numpy(dt))
    r = _field_data(sim.grid.r)
    ri = _field_data(sim.grid.ri)
    r_np = r
    ri_np = ri
    Sigma_np = SigD
    area = _field_data(sim.grid.A)
    D = _field_data(sim.dust.D)
    SigmaGas = _field_data(sim.gas.Sigma)
    v = _field_data(sim.dust.v.rad)
    Nr = int(sim.grid.Nr)
    Nm = int(sim.grid.Nm)
    zero_flux = is_zero_flux_enabled(sim)
    inner_outflow_only = is_dust_inner_outflow_only_enabled(sim)
    diode_block_mask = _inner_diode_block_mask(sim) if inner_outflow_only else None
    inner_adv_drain_mask = None
    if inner_outflow_only:
        inner_adv_drain_mask = _inner_boundary_advective_drain_mask(sim, SigmaDust=sim.dust.Sigma, Fi_adv=sim.dust.Fi.adv)
    freeze_mask = np.asarray(to_numpy(_gas_floor_freeze_mask(sim)), dtype=bool)
    freeze_velocity_mask = np.asarray(to_numpy(_transport_velocity_freeze_interface_mask(sim, SigmaDust=sim.dust.Sigma, Fi_adv=sim.dust.Fi.adv)), dtype=bool)
    freeze_diffusion_mask = np.asarray(to_numpy(_transport_diffusion_freeze_interface_mask(sim, SigmaDust=sim.dust.Sigma)), dtype=bool)

    # Building coagulation Jacobian

    # Total problem size
    Ntot = int((Nr*Nm))
    dat, row, col = _dust_f_call(dust_f.jacobian_coagulation_generator, A, cstick, eps, ilf, irm, istick, m, phi, Rf, Rs, SigD, SigDfloor, to_backend_result=False)
    gen = (dat, (row, col))
    if np.any(freeze_mask[1:-1]):
        freeze_rows = freeze_mask[row // Nm]
        if np.any(freeze_rows):
            dat = np.array(dat, copy=True)
            dat[freeze_rows] = 0.0
            gen = (dat, (row, col))
    J_coag = sp.csc_matrix(gen, shape=(Ntot, Ntot))

    if np.any(freeze_velocity_mask) or np.any(freeze_diffusion_mask):
        A_h, B_h, C_h = _jacobian_hydrodynamic_generator_numpy(area, D, r, ri, SigmaGas, v, freeze_velocity_mask=freeze_velocity_mask, freeze_diffusion_mask=freeze_diffusion_mask)
    else:
        A_h, B_h, C_h = _dust_f_call(dust_f.jacobian_hydrodynamic_generator, area, D, r, ri, SigmaGas, v, to_backend_result=False)
    if zero_flux:
        A_h, B_h, C_h = _apply_zero_flux_dust_hyd_edges_numpy(A_h, B_h, C_h, area, D, r, ri, SigmaGas, v)
    elif diode_block_mask is not None and np.any(to_numpy(diode_block_mask)):
        A_h, B_h, C_h = _apply_inner_zero_flux_dust_hyd_edge_numpy(A_h, B_h, C_h, area, D, r, ri, SigmaGas, v, diode_block_mask, adv_drain_mask=inner_adv_drain_mask)
    J_hyd = sp.diags((A_h.ravel()[Nm:], B_h.ravel(), C_h.ravel()[:-Nm]), offsets=(-Nm, 0, Nm), shape=(Ntot, Ntot), format="csc")

    # Right-hand side defaults to the current state for all rows.
    sim.dust._rhs[:] = sim.dust.Sigma.ravel()

    # BOUNDARIES

    # Inner boundary

    # Initializing data and coordinate vectors for sparse matrix
    dat = np.zeros(int(3.*Nm))
    row0 = np.arange(int(Nm))
    col0 = np.arange(int(Nm))
    col1 = np.arange(int(Nm)) + Nm
    col2 = np.arange(int(Nm)) + 2.*Nm
    row = np.concatenate((row0, row0, row0))
    col = np.concatenate((col0, col1, col2))

    # Filling data vector depending on boundary condition
    inner_bc_mask = None
    if (not zero_flux) and sim.dust.boundary.inner is not None:
        if inner_outflow_only and diode_block_mask is not None:
            inner_bc_mask = ~np.asarray(to_numpy(diode_block_mask), dtype=bool)
            if not np.any(inner_bc_mask):
                inner_bc_mask = None
        else:
            inner_bc_mask = np.ones((Nm,), dtype=bool)
    if inner_bc_mask is not None:
        # Given value
        if sim.dust.boundary.inner.condition == "val":
            sim.dust._rhs[:Nm][inner_bc_mask] = sim.dust.boundary.inner.value[inner_bc_mask]
        # Constant value
        elif sim.dust.boundary.inner.condition == "const_val":
            dat[Nm:2*Nm][inner_bc_mask] = 1./dt
            sim.dust._rhs[:Nm][inner_bc_mask] = 0.
        # Given gradient
        elif sim.dust.boundary.inner.condition == "grad":
            K1 = - r_np[1]/r_np[0]
            dat[Nm:2*Nm][inner_bc_mask] = -K1/dt
            sim.dust._rhs[:Nm][inner_bc_mask] = (
                - ri_np[1]/r_np[0] * (r_np[1]-r_np[0]) * sim.dust.boundary.inner.value[inner_bc_mask]
            )
        # Constant gradient
        elif sim.dust.boundary.inner.condition == "const_grad":
            Di = ri_np[1]/ri_np[2] * (r_np[1]-r_np[0]) / (r_np[2]-r_np[0])
            K1 = - r_np[1]/r_np[0] * (1. + Di)
            K2 = r_np[2]/r_np[0] * Di
            dat[:Nm][inner_bc_mask] = 0.
            dat[Nm:2*Nm][inner_bc_mask] = -K1/dt
            dat[2*Nm:][inner_bc_mask] = -K2/dt
            sim.dust._rhs[:Nm][inner_bc_mask] = 0.
        # Given power law
        elif sim.dust.boundary.inner.condition == "pow":
            p = sim.dust.boundary.inner.value
            sim.dust._rhs[:Nm][inner_bc_mask] = sim.dust.Sigma[1][inner_bc_mask] * (r_np[0]/r_np[1])**p
        # Constant power law
        elif sim.dust.boundary.inner.condition == "const_pow":
            p = np.log(Sigma_np[2] /
                       Sigma_np[1]) / np.log(r_np[2]/r_np[1])
            K1 = - (r_np[0]/r_np[1])**p
            dat[Nm:2*Nm][inner_bc_mask] = -K1[inner_bc_mask]/dt
            sim.dust._rhs[:Nm][inner_bc_mask] = 0.

    # Creating sparce matrix for inner boundary
    gen = (dat, (row, col))
    J_in = sp.csc_matrix(gen, shape=(Ntot, Ntot))

    # Outer boundary

    # Initializing data and coordinate vectors for sparse matrix
    dat = np.zeros(int(3.*Nm))
    row0 = np.arange(int(Nm))
    col0 = np.arange(int(Nm))
    col1 = np.arange(int(Nm)) - Nm
    col2 = np.arange(int(Nm)) - 2.*Nm
    offset = (Nr-1)*Nm
    row = np.concatenate((row0, row0, row0)) + offset
    col = np.concatenate((col0, col1, col2)) + offset

    # Filling data vector depending on boundary condition
    if (not zero_flux) and sim.dust.boundary.outer is not None:
        # Given value
        if sim.dust.boundary.outer.condition == "val":
            sim.dust._rhs[-Nm:] = sim.dust.boundary.outer.value
        # Constant value
        elif sim.dust.boundary.outer.condition == "const_val":
            dat[-2*Nm:-Nm] = 1./dt
            sim.dust._rhs[-Nm:] = 0.
        # Given gradient
        elif sim.dust.boundary.outer.condition == "grad":
            KNrm2 = -r_np[-2]/r_np[-1]
            dat[-2*Nm:-Nm] = -KNrm2/dt
            sim.dust._rhs[-Nm:] = ri_np[-2]/r_np[-1] * \
                (r_np[-1]-r_np[-2])*sim.dust.boundary.outer.value
        # Constant gradient
        elif sim.dust.boundary.outer.condition == "const_grad":
            Do = ri_np[-2]/ri_np[-3] * (r_np[-1]-r_np[-2]) / (r_np[-2]-r_np[-3])
            KNrm2 = - r_np[-2]/r_np[-1] * (1. + Do)
            KNrm3 = r_np[-3]/r_np[-1] * Do
            dat[-2*Nm:-Nm] = -KNrm2/dt
            dat[-3*Nm:-2*Nm] = -KNrm3/dt
            sim.dust._rhs[-Nm:] = 0.
        # Given power law
        elif sim.dust.boundary.outer.condition == "pow":
            p = sim.dust.boundary.outer.value
            sim.dust._rhs[-Nm:] = sim.dust.Sigma[-2] * (r_np[-1]/r_np[-2])**p
        # Constant power law
        elif sim.dust.boundary.outer.condition == "const_pow":
            p = np.log(Sigma_np[-2] /
                       Sigma_np[-3]) / np.log(r_np[-2]/r_np[-3])
            KNrm2 = - (r_np[-1]/r_np[-2])**p
            dat[-2*Nm:-Nm] = -KNrm2/dt
            sim.dust._rhs[-Nm:] = 0.

    # Creating sparce matrix for outer boundary
    gen = (dat, (row, col))
    J_out = sp.csc_matrix(gen, shape=(Ntot, Ntot))

    return J_in + J_coag + J_hyd + J_out


def _jacobian_cupy(sim, x, dx=None, *args, **kwargs):
    # Parameters for function call
    A = _field_data(sim.dust.coagulation.A)
    cstick = _field_data(sim.dust.coagulation.stick)
    eps = _field_data(sim.dust.coagulation.eps)
    ilf = _field_data(sim.dust.coagulation.lf_ind)
    irm = _field_data(sim.dust.coagulation.rm_ind)
    istick = _field_data(sim.dust.coagulation.stick_ind)
    m = _field_data(sim.grid.m)
    phi = _field_data(sim.dust.coagulation.phi)
    Rf = _field_data(sim.dust.kernel * sim.dust.p.frag)
    Rs = _field_data(sim.dust.kernel * sim.dust.p.stick)
    SigD = _field_data(sim.dust.Sigma)
    SigDfloor = _field_data(sim.dust.SigmaFloor)

    if dx is None:
        dt = x.stepsize
    else:
        dt = dx
    try:
        dt = float(dt)
    except Exception:
        dt = float(to_numpy(dt))

    r = _field_data(sim.grid.r)
    ri = _field_data(sim.grid.ri)
    SigmaArr = SigD
    area = _field_data(sim.grid.A)
    D = _field_data(sim.dust.D)
    SigmaGas = _field_data(sim.gas.Sigma)
    v = _field_data(sim.dust.v.rad)
    Nr = int(sim.grid.Nr)
    Nm = int(sim.grid.Nm)
    Ntot = int((Nr * Nm))
    zero_flux = is_zero_flux_enabled(sim)
    inner_outflow_only = is_dust_inner_outflow_only_enabled(sim)
    diode_block_mask = _inner_diode_block_mask(sim) if inner_outflow_only else None
    inner_adv_drain_mask = None
    if inner_outflow_only:
        inner_adv_drain_mask = _inner_boundary_advective_drain_mask(sim, SigmaDust=sim.dust.Sigma, Fi_adv=sim.dust.Fi.adv)
    freeze_mask = _gas_floor_freeze_mask(sim)
    freeze_velocity_mask = _transport_velocity_freeze_interface_mask(sim, SigmaDust=sim.dust.Sigma, Fi_adv=sim.dust.Fi.adv)
    freeze_diffusion_mask = _transport_diffusion_freeze_interface_mask(sim, SigmaDust=sim.dust.Sigma)
    q = _get_mass_grid_q(m, Nm)
    _, _, _, _, _, jcoag_indices, jcoag_indptr, jcoag_perm = _get_jcoag_pattern_cupy(Nr, Nm, q)

    dat, _, _ = _jacobian_coagulation_generator_cupy(A, cstick, eps, ilf, irm, istick, m, phi, Rf, Rs, SigD, SigDfloor)
    if int(to_numpy(freeze_mask[1:-1].sum())) > 0:
        dat_mid = dat.reshape(Nr - 2, -1)
        dat_mid[freeze_mask[1:-1], :] = 0.0
    dat_csr = dat if jcoag_perm is None else dat[jcoag_perm]
    J_coag = cp_sparse.csr_matrix((dat_csr, jcoag_indices, jcoag_indptr), shape=(Ntot, Ntot))

    A_h, B_h, C_h = _jacobian_hydrodynamic_generator_cupy(area, D, r, ri, SigmaGas, v, freeze_velocity_mask=freeze_velocity_mask, freeze_diffusion_mask=freeze_diffusion_mask)
    if zero_flux:
        A_h, B_h, C_h = _apply_zero_flux_dust_hyd_edges_cupy(A_h, B_h, C_h, area, D, r, ri, SigmaGas, v)
    elif diode_block_mask is not None and int(to_numpy(diode_block_mask.sum())) > 0:
        A_h, B_h, C_h = _apply_inner_zero_flux_dust_hyd_edge_cupy(A_h, B_h, C_h, area, D, r, ri, SigmaGas, v, diode_block_mask, adv_drain_mask=inner_adv_drain_mask)
    n_hyd, jhb_map, jhb_indices, jhb_indptr = _get_dust_hyd_boundary_pattern_cupy(Nr, Nm)
    dat_hyd = cp.concatenate((A_h.ravel()[Nm:], B_h.ravel(), C_h.ravel()[:-Nm]))

    # Right-hand side defaults to the current state for all rows.
    sim.dust._rhs[:] = sim.dust.Sigma.ravel()
    c_in0 = cp.zeros((Nm,), dtype=dat_hyd.dtype)
    c_in1 = cp.zeros((Nm,), dtype=dat_hyd.dtype)
    c_in2 = cp.zeros((Nm,), dtype=dat_hyd.dtype)

    inner_bc_mask = None
    if (not zero_flux) and sim.dust.boundary.inner is not None:
        if inner_outflow_only and diode_block_mask is not None:
            inner_bc_mask = ~cp.asarray(diode_block_mask, dtype=bool)
            if int(to_numpy(inner_bc_mask.sum())) == 0:
                inner_bc_mask = None
        else:
            inner_bc_mask = cp.ones((Nm,), dtype=bool)
    if inner_bc_mask is not None:
        if sim.dust.boundary.inner.condition == "val":
            sim.dust._rhs[:Nm][inner_bc_mask] = sim.dust.boundary.inner.value[inner_bc_mask]
        elif sim.dust.boundary.inner.condition == "const_val":
            c_in1[inner_bc_mask] = 1. / dt
            sim.dust._rhs[:Nm][inner_bc_mask] = 0.
        elif sim.dust.boundary.inner.condition == "grad":
            K1 = - (r[1] / r[0])
            c_in1[inner_bc_mask] = -K1 / dt
            fac = - (ri[1] / r[0] * (r[1] - r[0]))
            sim.dust._rhs[:Nm][inner_bc_mask] = fac * sim.dust.boundary.inner.value[inner_bc_mask]
        elif sim.dust.boundary.inner.condition == "const_grad":
            Di = (ri[1] / ri[2] * (r[1] - r[0]) / (r[2] - r[0]))
            K1 = - (r[1] / r[0]) * (1. + Di)
            K2 = (r[2] / r[0]) * Di
            c_in0[inner_bc_mask] = 0.0
            c_in1[inner_bc_mask] = -K1 / dt
            c_in2[inner_bc_mask] = -K2 / dt
            sim.dust._rhs[:Nm][inner_bc_mask] = 0.
        elif sim.dust.boundary.inner.condition == "pow":
            p = sim.dust.boundary.inner.value
            ratio = (r[0] / r[1])
            sim.dust._rhs[:Nm][inner_bc_mask] = SigmaArr[1][inner_bc_mask] * ratio**p
        elif sim.dust.boundary.inner.condition == "const_pow":
            logr = cp.log(r[2] / r[1])
            p = cp.log(SigmaArr[2] / SigmaArr[1]) / logr
            K1 = - (r[0] / r[1])**p
            c_in1[inner_bc_mask] = -K1[inner_bc_mask] / dt
            sim.dust._rhs[:Nm][inner_bc_mask] = 0.

    c_out0 = 0.0
    c_out1 = 0.0
    c_out2 = 0.0

    if (not zero_flux) and sim.dust.boundary.outer is not None:
        if sim.dust.boundary.outer.condition == "val":
            sim.dust._rhs[-Nm:] = sim.dust.boundary.outer.value
        elif sim.dust.boundary.outer.condition == "const_val":
            c_out1 = 1. / dt
            sim.dust._rhs[-Nm:] = 0.
        elif sim.dust.boundary.outer.condition == "grad":
            KNrm2 = - (r[-2] / r[-1])
            c_out1 = -KNrm2 / dt
            fac = (ri[-2] / r[-1] * (r[-1] - r[-2]))
            sim.dust._rhs[-Nm:] = fac * sim.dust.boundary.outer.value
        elif sim.dust.boundary.outer.condition == "const_grad":
            Do = (ri[-2] / ri[-3] * (r[-1] - r[-2]) / (r[-2] - r[-3]))
            KNrm2 = - (r[-2] / r[-1]) * (1. + Do)
            KNrm3 = (r[-3] / r[-1]) * Do
            c_out1 = -KNrm2 / dt
            c_out0 = -KNrm3 / dt
            sim.dust._rhs[-Nm:] = 0.
        elif sim.dust.boundary.outer.condition == "pow":
            p = sim.dust.boundary.outer.value
            ratio = (r[-1] / r[-2])
            sim.dust._rhs[-Nm:] = SigmaArr[-2] * ratio**p
        elif sim.dust.boundary.outer.condition == "const_pow":
            logr = cp.log(r[-2] / r[-3])
            p = cp.log(SigmaArr[-2] / SigmaArr[-3]) / logr
            KNrm2 = - (r[-1] / r[-2])**p
            c_out1 = -KNrm2 / dt
            sim.dust._rhs[-Nm:] = 0.
    dat_bound = cp.empty((int(6 * Nm),), dtype=dat_hyd.dtype)
    dat_bound[:Nm] = c_in0
    dat_bound[Nm:2 * Nm] = c_in1
    dat_bound[2 * Nm:3 * Nm] = c_in2
    dat_bound[3 * Nm:4 * Nm] = c_out0
    dat_bound[4 * Nm:5 * Nm] = c_out1
    dat_bound[5 * Nm:6 * Nm] = c_out2

    dat_hb = cp.empty((n_hyd + int(6 * Nm),), dtype=dat_hyd.dtype)
    dat_hb[:n_hyd] = dat_hyd
    dat_hb[n_hyd:] = dat_bound
    dat_hb_csr = cp.zeros((int(jhb_indices.size),), dtype=dat_hyd.dtype)
    _scatter_add_1d(dat_hb_csr, jhb_map, dat_hb)
    J_hb = cp_sparse.csr_matrix((dat_hb_csr, jhb_indices, jhb_indptr), shape=(Ntot, Ntot))
    return J_coag + J_hb


def _jacobian_cupy_dense(sim, x, dx=None, *args, **kwargs):
    """CuPy Jacobian assembly into a dense matrix (no sparse conversion path)."""
    A = _field_data(sim.dust.coagulation.A)
    cstick = _field_data(sim.dust.coagulation.stick)
    eps = _field_data(sim.dust.coagulation.eps)
    ilf = _field_data(sim.dust.coagulation.lf_ind)
    irm = _field_data(sim.dust.coagulation.rm_ind)
    istick = _field_data(sim.dust.coagulation.stick_ind)
    m = _field_data(sim.grid.m)
    phi = _field_data(sim.dust.coagulation.phi)
    Rf = _field_data(sim.dust.kernel * sim.dust.p.frag)
    Rs = _field_data(sim.dust.kernel * sim.dust.p.stick)
    SigD = _field_data(sim.dust.Sigma)
    SigDfloor = _field_data(sim.dust.SigmaFloor)

    if dx is None:
        dt = x.stepsize
    else:
        dt = dx
    try:
        dt = float(dt)
    except Exception:
        dt = float(to_numpy(dt))

    r = _field_data(sim.grid.r)
    ri = _field_data(sim.grid.ri)
    SigmaArr = SigD
    area = _field_data(sim.grid.A)
    D = _field_data(sim.dust.D)
    SigmaGas = _field_data(sim.gas.Sigma)
    v = _field_data(sim.dust.v.rad)
    Nr = int(sim.grid.Nr)
    Nm = int(sim.grid.Nm)
    Ntot = int(Nr * Nm)
    zero_flux = is_zero_flux_enabled(sim)
    inner_outflow_only = is_dust_inner_outflow_only_enabled(sim)
    diode_block_mask = _inner_diode_block_mask(sim) if inner_outflow_only else None
    inner_adv_drain_mask = None
    if inner_outflow_only:
        inner_adv_drain_mask = _inner_boundary_advective_drain_mask(sim, SigmaDust=sim.dust.Sigma, Fi_adv=sim.dust.Fi.adv)
    freeze_mask = _gas_floor_freeze_mask(sim)
    freeze_velocity_mask = _transport_velocity_freeze_interface_mask(sim, SigmaDust=sim.dust.Sigma, Fi_adv=sim.dust.Fi.adv)
    freeze_diffusion_mask = _transport_diffusion_freeze_interface_mask(sim, SigmaDust=sim.dust.Sigma)

    q = _get_mass_grid_q(m, Nm)
    _, _, row, col, _, _, _, _ = _get_jcoag_pattern_cupy(Nr, Nm, q)
    dat_coag, _, _ = _jacobian_coagulation_generator_cupy(A, cstick, eps, ilf, irm, istick, m, phi, Rf, Rs, SigD, SigDfloor)
    if int(to_numpy(freeze_mask[1:-1].sum())) > 0:
        dat_coag_mid = dat_coag.reshape(Nr - 2, -1)
        dat_coag_mid[freeze_mask[1:-1], :] = 0.0
    dtype = dat_coag.dtype
    J = cp.zeros((Ntot, Ntot), dtype=dtype)
    J_flat = J.ravel()
    _scatter_add_1d(J_flat, row * Ntot + col, dat_coag)

    A_h, B_h, C_h = _jacobian_hydrodynamic_generator_cupy(area, D, r, ri, SigmaGas, v, freeze_velocity_mask=freeze_velocity_mask, freeze_diffusion_mask=freeze_diffusion_mask)
    if zero_flux:
        A_h, B_h, C_h = _apply_zero_flux_dust_hyd_edges_cupy(A_h, B_h, C_h, area, D, r, ri, SigmaGas, v)
    elif diode_block_mask is not None and int(to_numpy(diode_block_mask.sum())) > 0:
        A_h, B_h, C_h = _apply_inner_zero_flux_dust_hyd_edge_cupy(A_h, B_h, C_h, area, D, r, ri, SigmaGas, v, diode_block_mask, adv_drain_mask=inner_adv_drain_mask)

    idx = cp.arange(Ntot, dtype=cp.int64)
    Bflat = B_h.ravel()
    Aflat = A_h.ravel()
    Cflat = C_h.ravel()
    _scatter_add_1d(J_flat, idx * Ntot + idx, Bflat)
    _scatter_add_1d(J_flat, idx[Nm:] * Ntot + (idx[Nm:] - Nm), Aflat[Nm:])
    _scatter_add_1d(J_flat, idx[:-Nm] * Ntot + (idx[:-Nm] + Nm), Cflat[:-Nm])

    # Right-hand side defaults to the current state for all rows.
    sim.dust._rhs[:] = sim.dust.Sigma.ravel()
    c_in0 = cp.zeros((Nm,), dtype=dtype)
    c_in1 = cp.zeros((Nm,), dtype=dtype)
    c_in2 = cp.zeros((Nm,), dtype=dtype)

    inner_bc_mask = None
    if (not zero_flux) and sim.dust.boundary.inner is not None:
        if inner_outflow_only and diode_block_mask is not None:
            inner_bc_mask = ~cp.asarray(diode_block_mask, dtype=bool)
            if int(to_numpy(inner_bc_mask.sum())) == 0:
                inner_bc_mask = None
        else:
            inner_bc_mask = cp.ones((Nm,), dtype=bool)
    if inner_bc_mask is not None:
        if sim.dust.boundary.inner.condition == "val":
            sim.dust._rhs[:Nm][inner_bc_mask] = sim.dust.boundary.inner.value[inner_bc_mask]
        elif sim.dust.boundary.inner.condition == "const_val":
            c_in1[inner_bc_mask] = 1. / dt
            sim.dust._rhs[:Nm][inner_bc_mask] = 0.
        elif sim.dust.boundary.inner.condition == "grad":
            K1 = - (r[1] / r[0])
            c_in1[inner_bc_mask] = -K1 / dt
            fac = - (ri[1] / r[0] * (r[1] - r[0]))
            sim.dust._rhs[:Nm][inner_bc_mask] = fac * sim.dust.boundary.inner.value[inner_bc_mask]
        elif sim.dust.boundary.inner.condition == "const_grad":
            Di = (ri[1] / ri[2] * (r[1] - r[0]) / (r[2] - r[0]))
            K1 = - (r[1] / r[0]) * (1. + Di)
            K2 = (r[2] / r[0]) * Di
            c_in0[inner_bc_mask] = 0.0
            c_in1[inner_bc_mask] = -K1 / dt
            c_in2[inner_bc_mask] = -K2 / dt
            sim.dust._rhs[:Nm][inner_bc_mask] = 0.
        elif sim.dust.boundary.inner.condition == "pow":
            p = sim.dust.boundary.inner.value
            ratio = (r[0] / r[1])
            sim.dust._rhs[:Nm][inner_bc_mask] = SigmaArr[1][inner_bc_mask] * ratio**p
        elif sim.dust.boundary.inner.condition == "const_pow":
            logr = cp.log(r[2] / r[1])
            p = cp.log(SigmaArr[2] / SigmaArr[1]) / logr
            K1 = - (r[0] / r[1])**p
            c_in1[inner_bc_mask] = -K1[inner_bc_mask] / dt
            sim.dust._rhs[:Nm][inner_bc_mask] = 0.

    c_out0 = 0.0
    c_out1 = 0.0
    c_out2 = 0.0

    if (not zero_flux) and sim.dust.boundary.outer is not None:
        if sim.dust.boundary.outer.condition == "val":
            sim.dust._rhs[-Nm:] = sim.dust.boundary.outer.value
        elif sim.dust.boundary.outer.condition == "const_val":
            c_out1 = 1. / dt
            sim.dust._rhs[-Nm:] = 0.
        elif sim.dust.boundary.outer.condition == "grad":
            KNrm2 = - (r[-2] / r[-1])
            c_out1 = -KNrm2 / dt
            fac = (ri[-2] / r[-1] * (r[-1] - r[-2]))
            sim.dust._rhs[-Nm:] = fac * sim.dust.boundary.outer.value
        elif sim.dust.boundary.outer.condition == "const_grad":
            Do = (ri[-2] / ri[-3] * (r[-1] - r[-2]) / (r[-2] - r[-3]))
            KNrm2 = - (r[-2] / r[-1]) * (1. + Do)
            KNrm3 = (r[-3] / r[-1]) * Do
            c_out1 = -KNrm2 / dt
            c_out0 = -KNrm3 / dt
            sim.dust._rhs[-Nm:] = 0.
        elif sim.dust.boundary.outer.condition == "pow":
            p = sim.dust.boundary.outer.value
            ratio = (r[-1] / r[-2])
            sim.dust._rhs[-Nm:] = SigmaArr[-2] * ratio**p
        elif sim.dust.boundary.outer.condition == "const_pow":
            logr = cp.log(r[-2] / r[-3])
            p = cp.log(SigmaArr[-2] / SigmaArr[-3]) / logr
            KNrm2 = - (r[-1] / r[-2])**p
            c_out1 = -KNrm2 / dt
            sim.dust._rhs[-Nm:] = 0.

    rows = cp.arange(Nm, dtype=cp.int64)
    row_out = rows + int((Nr - 1) * Nm)
    _scatter_add_1d(J_flat, rows * Ntot + rows, cp.full((Nm,), c_in0, dtype=dtype))
    _scatter_add_1d(J_flat, row_out * Ntot + row_out, cp.full((Nm,), c_out0, dtype=dtype))
    if Nr >= 2:
        _scatter_add_1d(J_flat, rows * Ntot + (rows + Nm), cp.full((Nm,), c_in1, dtype=dtype))
        _scatter_add_1d(J_flat, row_out * Ntot + (row_out - Nm), cp.full((Nm,), c_out1, dtype=dtype))
    if Nr >= 3:
        _scatter_add_1d(J_flat, rows * Ntot + (rows + 2 * Nm), cp.full((Nm,), c_in2, dtype=dtype))
        _scatter_add_1d(J_flat, row_out * Ntot + (row_out - 2 * Nm), cp.full((Nm,), c_out2, dtype=dtype))

    return J


def jacobian(sim, x, dx=None, *args, **kwargs):
    return _K_JACOBIAN(sim, x, dx=dx, *args, **kwargs)


def kernel(sim):
    """Function calculates the vertically integrated collision kernel.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    K : Field
        Collision kernel"""
    return _K_KERNEL(sim)


def MRN_distribution(sim):
    """Function calculates the initial particle mass distribution. The parameters are taken from the
    ``Simulation.ini`` object.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    Sigma : Field
        Initial dust surface density

    Notes
    -----
    ``sim.ini.dust.aIniMax`` : maximum initial particle size
    ``sim.ini.dust.d2gRatio`` : initial dust-to-gas ratio
    ``sim.ini.dust.distExp`` : initial particle size distribution ``n(a) da ∝ a^{distExp} da``
    ``sim.ini.dust.allowDriftingParticles`` : if ``True`` the particle size distribution
    will be filled up to ``aIniMax``, if ``False`` the maximum particle size will be chosen
    such that there are no drifting particles initially. This prevents a particle wave traveling
    though the simulation, that is already drifting initially."""
    exp = sim.ini.dust.distExp
    a_arr = _field_data(sim.dust.a)
    sigma_g = _field_data(sim.gas.Sigma)
    # Set maximum particle size
    if sim.ini.dust.allowDriftingParticles:
        aIni = sim.ini.dust.aIniMax
    else:
        # Calculating pressure gradient
        P = _field_data(sim.gas.P)
        grid_ri = _field_data(sim.grid.ri)
        grid_r = _field_data(sim.grid.r)
        Pi = _interp_to_interfaces(P, grid_r, grid_ri)
        gamma = (Pi[1:] - Pi[:-1]) / (grid_ri[1:] - grid_ri[:-1])
        gamma = xp.abs(gamma)
        # Exponent of pressure gradient
        gamma *= grid_r / (P + 1e-300)
        # Maximum drift limited particle size with safety margin
        ad = 5.e-3 * 2. / c.pi * sim.ini.dust.d2gRatio * sigma_g \
            / (_field_data(sim.dust.fill[:, 0]) * _field_data(sim.dust.rhos[:, 0])) * (_field_data(sim.grid.OmegaK) * grid_r)**2. \
            / (_field_data(sim.gas.cs)**2. * (gamma + 1e-300))
        aIni = xp.minimum(sim.ini.dust.aIniMax, ad)[:, None]
    # Fill distribution
    ret = xp.where(a_arr <= aIni, a_arr**(exp+4), 0.)
    s = xp.sum(ret, axis=1)[..., None]
    s = xp.where(s > 0., s, 1.)
    # Normalize to mass
    ret = ret / s * sigma_g[..., None] * sim.ini.dust.d2gRatio
    return ret


def p_stick(sim):
    """Function calculates the sticking probability.
    The sticking probability is simply 1 minus the
    fragmentation probability.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    ps : Field
        Sticking probability"""
    p = 1. - sim.dust.p.frag
    p[0] = 0.
    p[-1] = 0.
    return p


def p_frag(sim):
    """Function calculates the fragmentation probability.
    It assumes a linear transition between sticking and
    fragmentation.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    pf : Field
        Fragmentation propability."""
    return _K_P_FRAG(sim)


def rho_midplane(sim):
    """Function calculates the midplane mass density.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    rho : Field
        Midplane mass density"""
    return sim.dust.Sigma / (xp.sqrt(2 * c.pi) * sim.dust.H)


def S_coag(sim, Sigma=None):
    """Function calculates the coagulation source terms.

    Parameters
    ----------
    sim : Frame
        Parent simulation Frame
    Sigma : Field, optional, default : None
        Surface density to be used if not None

    Returns
    -------
    Scoag : Field
        Coagulation source terms"""
    return _K_S_COAG(sim, Sigma=Sigma)


def S_hyd(sim, Sigma=None):
    """Function calculates the hydrodynamic source terms.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    Sigma : Field, optional, default : None
        Surface density to be used if not None

    Returns
    -------
    Shyd : Field
        Hydrodynamic source terms"""
    return _K_S_HYD(sim, Sigma=Sigma)


def S_tot(sim, Sigma=None):
    """Function calculates the total source terms.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    Sigma : Field, optional, default : None
        Surface density to be used if not None

    Returns
    -------
    Stot : Field
        Total source terms of surface density"""
    Sext = sim.dust.S.ext
    if Sigma is None:
        Sigma = sim.dust.Sigma
        Scoag = sim.dust.S.coag
        Shyd = sim.dust.S.hyd
    else:
        Scoag = sim.dust.S.coag.updater.beat(sim, Sigma=Sigma)
        if Scoag is None:
            Scoag = sim.dust.S.coag
        Shyd = sim.dust.S.hyd.updater.beat(sim, Sigma=Sigma)
        if Shyd is None:
            Shyd = sim.dust.S.hyd
    return Scoag + Shyd + Sext


def Sigma_deriv(sim, t, Sigma):
    """Function calculates the derivative of the dust surface density.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    t : IntVar
        Current time
    Sigma : Field
        Current Surface density

    Returns
    -------
    Sigma_dot: Field
        Derivative of Surface density"""
    return sim.dust.S.tot.updater.beat(sim, Sigma=Sigma)


def SigmaFloor(sim):
    """Function calculates the floor value for the dust distribution. Floor value means that there is less than
    one particle in an annulus.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    Sigma_floor : Field
        Floor value of surface density"""
    area = c.pi * (sim.grid.ri[1:]**2. - sim.grid.ri[:-1]**2.)
    return 1 / area[..., None] * sim.grid.m


def St_Epstein_StokesI(sim):
    """Function calculates the Stokes number using the Epstein and the Stokes I drag regimes.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    St : Field
        Stokes number"""
    return _K_ST(sim)


def coagulation_parameters(sim):
    """Function calculates the coagulation parameters needed for a simple
    sticking-erosion-fragmentation collision model. The sticking matrix
    is calculated with the method described in appendices A.1. and A.2. of
    Brauer et al. (2008).

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    (cstick, cstick_ind, A, eps, klf, krm, phi) : Tuple
        Coagulation parameters

    Notes
    -----
    The sticking matrix has technically a shape of ``(Nm, Nm, Nm)``.
    For example: ``cstick(k, j, i)`` describes the change of mass bin ``k``
    resulting from a sticking collision between particles ``j`` and ``k``.
    Since this matrix has at maximum four entries per particle collision,
    only the non-zero elemts are stored in ``cstick`` of shape ``(4, Nm, Nm)``.
    The positions of the non-zero elements along the first axis are stored
    in ``cstick_ind``. For details see Brauer et al. (2008)."""
    return _K_COAG_PARAMS(sim)


def vdriftmax(sim):
    """Function calculates the maximum drift velocity of the dust including back reaction of
    the gas onto the dust, if the back reaction coefficients are set.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    vdriftmax : Field
        Maximum drift velocity"""
    A = sim.dust.backreaction.A
    B = sim.dust.backreaction.B
    return 0.5 * B * sim.gas.v.visc - A * \
        sim.gas.eta * sim.grid.r * sim.grid.OmegaK


def vrad(sim):
    """Function calculated the radial velocity of the dust.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    vrad : Field
        Radial dust velocity"""
    return _K_VRAD(sim)


def vrel_azimuthal_drift(sim):
    """Function calculates the relative particle velocities due to azimuthal drift.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    vrel : Field
        Relative velocities"""
    return _K_VREL_AZI(sim)


def vrel_brownian_motion(sim):
    """Function calculates the relative particle velocities due to Brownian motion.
    The maximum value is set to the sound speed.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    vrel : Field
        Relative velocities"""
    return _K_VREL_BROWN(sim)


def vrel_radial_drift(sim):
    """Function calculates the relative particle velocities due to radial drift.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    vrel : Field
        Relative velocities"""
    return _K_VREL_RAD(sim)


def vrel_tot(sim):
    """Function calculates the total relative vparticle velocities by taking the root mean square
    of all individual sources.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    vrel : Field
        Relative velocities"""
    if _VREL_TOT_MODE == "elementwise" and getattr(xp, "name", "") == "cupy":
        vrel = _vrel_tot_cupy_elementwise(_field_data(sim.dust.v.rel.azi), _field_data(sim.dust.v.rel.brown), _field_data(sim.dust.v.rel.rad), _field_data(sim.dust.v.rel.turb), _field_data(sim.dust.v.rel.vert))
        if vrel is not None:
            return vrel

    return xp.sqrt(sim.dust.v.rel.azi**2 + sim.dust.v.rel.brown**2 + sim.dust.v.rel.rad**2 + sim.dust.v.rel.turb**2 + sim.dust.v.rel.vert**2)


def vrel_turbulent_motion(sim):
    """Function calculates the relative particle velocities due to turbulent motion.
    It uses the prescription of Ormel & Cuzzi (2007).

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    vrel : Field
        Relative velocities"""
    return _K_VREL_TURB(sim)


def vrel_vertical_settling(sim):
    """Function calculates the relative particle velocities due to vertical settling.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    vrel : Field
        Relative velocities"""
    return _K_VREL_VERT(sim)


def _f_impl_1_direct_numpy(x0, Y0, dx, jac=None, rhs=None, *args, **kwargs):
    """Implicit 1st-order integration scheme with direct matrix inversion

    Parameters
    ----------
    x0 : Intvar
        Integration variable at beginning of scheme
    Y0 : Field
        Variable to be integrated at the beginning of scheme
    dx : IntVar
        Stepsize of integration variable
    jac : Field, optional, defaul : None
        Current Jacobian. Will be calculated, if not set
    args : additional positional arguments
    kwargs : additional keyworda arguments

    Returns
    -------
    dY : Field
        Delta of variable to be integrated

    Butcher tableau
    ---------------
     1 | 1
    ---|---
       | 1
    """
    dx = float(to_numpy(dx))
    zero_flux = is_zero_flux_enabled(Y0._owner)

    if jac is None:
        jac = Y0.jacobian(x0, dx)
    if rhs is None:
        rhs = _field_data(Y0).ravel()

    Nm = Y0._owner.dust.Sigma.shape[1]

    # Add external source terms to right-hand side.
    if zero_flux:
        rhs[:] += dx * _field_data(Y0._owner.dust.S.ext).ravel()
    else:
        rhs[Nm:-Nm] += dx * _field_data(Y0._owner.dust.S.ext[1:-1, ...]).ravel()

    N = jac.shape[0]
    eye = sp.identity(N, format="csc")
    A = eye - dx * jac

    Y1_ravel = solve_sparse_linear_system(A, rhs)

    Y1 = Y1_ravel.reshape(Y0.shape)
    if _maybe_retry_implicit_floor_injection(x0, Y0, dx, Y1):
        return False

    return Y1 - Y0


def _f_impl_1_direct_cupy(x0, Y0, dx, jac=None, rhs=None, *args, **kwargs):
    """CuPy-only implicit 1st-order integration scheme with GPU sparse solve."""
    zero_flux = is_zero_flux_enabled(Y0._owner)
    if jac is None:
        jac = Y0.jacobian(x0, dx)
    if rhs is None:
        rhs = _field_data(Y0).ravel()

    Nm = Y0._owner.dust.Sigma.shape[1]
    if zero_flux:
        rhs[:] += dx * _field_data(Y0._owner.dust.S.ext).ravel()
    else:
        rhs[Nm:-Nm] += dx * _field_data(Y0._owner.dust.S.ext[1:-1, ...]).ravel()

    jac_gpu = jac.tocsr() if isinstance(jac, cp_sparse.spmatrix) else cp_sparse.csr_matrix(jac)
    if _CUPY_A_BUILD_MODE == "diag_inplace":
        A = jac_gpu.copy()
        A.data *= -dx
        diag_pos = _get_cupy_diag_positions_csr(A)
        if diag_pos is not None:
            A.data[diag_pos] += 1.0
        else:
            A = cp_sparse.identity(jac_gpu.shape[0], format="csr", dtype=jac_gpu.dtype) - dx * jac_gpu
    else:
        A = cp_sparse.identity(jac_gpu.shape[0], format="csr", dtype=jac_gpu.dtype) - dx * jac_gpu

    Y1_ravel = solve_sparse_linear_system(A, rhs)
    Y1 = Y1_ravel.reshape(Y0.shape)
    if _maybe_retry_implicit_floor_injection(x0, Y0, dx, Y1):
        return False
    return Y1 - Y0


def _f_impl_1_direct_cupy_dense(x0, Y0, dx, jac=None, rhs=None, *args, **kwargs):
    """CuPy implicit 1st-order scheme using dense GPU matrix solve."""
    zero_flux = is_zero_flux_enabled(Y0._owner)
    if jac is None:
        jac = Y0.jacobian(x0, dx)
    if rhs is None:
        rhs = _field_data(Y0).ravel()
    else:
        rhs = _field_data(rhs)

    Nm = Y0._owner.dust.Sigma.shape[1]
    if zero_flux:
        rhs[:] += dx * _field_data(Y0._owner.dust.S.ext).ravel()
    else:
        rhs[Nm:-Nm] += dx * _field_data(Y0._owner.dust.S.ext[1:-1, ...]).ravel()

    jac_dense = jac if isinstance(jac, cp.ndarray) else cp.asarray(jac)
    A = cp.eye(int(jac_dense.shape[0]), dtype=jac_dense.dtype) - dx * jac_dense
    Y1_ravel = cp.linalg.solve(A, rhs)
    Y1 = Y1_ravel.reshape(Y0.shape)
    if _maybe_retry_implicit_floor_injection(x0, Y0, dx, Y1):
        return False
    return Y1 - Y0


def _f_impl_1_direct(x0, Y0, dx, jac=None, rhs=None, *args, **kwargs):
    return _K_IMPL_1_DIRECT(x0, Y0, dx, jac=jac, rhs=rhs, *args, **kwargs)


class impl_1_direct(Scheme):
    """Modified class for implicit dust integration."""

    def __init__(self):
        super().__init__(_f_impl_1_direct, description="Implicit 1st-order direct solver")


# Initialize default bindings at import time after all functions are defined.
bind_backend_kernels()
