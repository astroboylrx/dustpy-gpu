"""Private CuPy implementations for dust evolution."""

import dustpy.constants as c
import math
import os

import numpy as np
import scipy.sparse as sp

from dustpy.std._dust_common import _apply_gas_floor_freeze_to_flux
from dustpy.std._dust_common import _apply_gas_floor_freeze_to_radial_field
from dustpy.std._dust_common import _field_data
from dustpy.std._dust_common import _gas_floor_freeze_mask
from dustpy.std._dust_cupy_kernels import _F_diff_cupy_elementwise
from dustpy.std._dust_cupy_kernels import _kernel_cupy_elementwise
from dustpy.std._dust_cupy_kernels import _p_frag_cupy_elementwise
from dustpy.std._dust_cupy_kernels import _get_raw_scatter_kernel
from dustpy.std._dust_cupy_kernels import _vrel_turbulent_motion_cupy_elementwise
from dustpy.std._dust_numpy import _get_jcoag_pattern
from dustpy.utils.boundary_modes import is_zero_flux_enabled
from dustpy.utils.backend import to_numpy
from simframe.backends.api import xp

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None

try:
    import cupyx.scipy.sparse as cp_sparse
except Exception:  # pragma: no cover - optional dependency
    cp_sparse = None

try:
    from cupyx.scipy.interpolate import interp1d as cp_interp1d
except Exception:  # pragma: no cover - optional dependency
    cp_interp1d = None

try:
    from cupyx import scatter_add as cp_scatter_add
except Exception:  # pragma: no cover - optional dependency
    cp_scatter_add = None


_COAG_PAIR_CACHE_KEY = None
_COAG_PAIR_CACHE_VALUE = None
_SCOAG_PAIRMAP_CACHE_KEY = None
_SCOAG_PAIRMAP_CACHE_VALUE = None
_SCOAG_PRECOMP_CACHE_KEY = None
_SCOAG_PRECOMP_CACHE_VALUE = None
_FRAG_P_CACHE_KEY = None
_FRAG_P_CACHE_VALUE = None
_MGRID_Q_CACHE_KEY = None
_MGRID_Q_CACHE_VALUE = None
_JCOAG_PATTERN_GPU_CACHE_KEY = None
_JCOAG_PATTERN_GPU_CACHE_VALUE = None
_JCOAG_PRECOMP_CACHE_KEY = None
_JCOAG_PRECOMP_CACHE_VALUE = None
_BOUNDARY_BASIS_CACHE_KEY = None
_BOUNDARY_BASIS_CACHE_VALUE = None
_JSTICK_MAP_CACHE_KEY = None
_JSTICK_MAP_CACHE_VALUE = None
_JFRAG_MAP_CACHE_KEY = None
_JFRAG_MAP_CACHE_VALUE = None
_JCOAG_WORK_CACHE_KEY = None
_JCOAG_WORK_CACHE_VALUE = None
_DUST_JHB_PATTERN_CACHE_KEY = None
_DUST_JHB_PATTERN_CACHE_VALUE = None
_COAG_CACHE_KEY = None
_COAG_CACHE_VALUE = None
_KERNEL_LOWER_MASK = None
_CUPY_DIAG_POS_CACHE_KEY = None
_CUPY_DIAG_POS_CACHE_VALUE = None
_RUNTIME_STATES = {}
_ACTIVE_RUNTIME_TOKEN = None

_SCATTER_MODE = "addat"
_S_COAG_MODE = "baseline"
_JCOAG_GEN_MODE = "baseline"
_JCOAG_CHUNK_SIZE_OVERRIDE = None
_F_DIFF_MODE = "baseline"
_VREL_TURB_MODE = "baseline"
_P_FRAG_MODE = "baseline"
_COLLISION_KERNEL_MODE = "baseline"

_RUNTIME_STATE_VARS = (
    "_COAG_PAIR_CACHE_KEY",
    "_COAG_PAIR_CACHE_VALUE",
    "_SCOAG_PAIRMAP_CACHE_KEY",
    "_SCOAG_PAIRMAP_CACHE_VALUE",
    "_SCOAG_PRECOMP_CACHE_KEY",
    "_SCOAG_PRECOMP_CACHE_VALUE",
    "_FRAG_P_CACHE_KEY",
    "_FRAG_P_CACHE_VALUE",
    "_MGRID_Q_CACHE_KEY",
    "_MGRID_Q_CACHE_VALUE",
    "_JCOAG_PATTERN_GPU_CACHE_KEY",
    "_JCOAG_PATTERN_GPU_CACHE_VALUE",
    "_JCOAG_PRECOMP_CACHE_KEY",
    "_JCOAG_PRECOMP_CACHE_VALUE",
    "_BOUNDARY_BASIS_CACHE_KEY",
    "_BOUNDARY_BASIS_CACHE_VALUE",
    "_JSTICK_MAP_CACHE_KEY",
    "_JSTICK_MAP_CACHE_VALUE",
    "_JFRAG_MAP_CACHE_KEY",
    "_JFRAG_MAP_CACHE_VALUE",
    "_JCOAG_WORK_CACHE_KEY",
    "_JCOAG_WORK_CACHE_VALUE",
    "_DUST_JHB_PATTERN_CACHE_KEY",
    "_DUST_JHB_PATTERN_CACHE_VALUE",
    "_COAG_CACHE_KEY",
    "_COAG_CACHE_VALUE",
    "_KERNEL_LOWER_MASK",
    "_CUPY_DIAG_POS_CACHE_KEY",
    "_CUPY_DIAG_POS_CACHE_VALUE",
)


def _fresh_runtime_state():
    return {name: None for name in _RUNTIME_STATE_VARS}


def _capture_runtime_state():
    return {name: globals()[name] for name in _RUNTIME_STATE_VARS}


def _restore_runtime_state(state):
    for name, value in state.items():
        globals()[name] = value


def switch_runtime_state(runtime_token):
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


def configure_runtime(scatter_mode, s_coag_mode, jcoag_gen_mode, jcoag_chunk_size_override, f_diff_mode, vrel_turb_mode, p_frag_mode, collision_kernel_mode):
    global _SCATTER_MODE, _S_COAG_MODE, _JCOAG_GEN_MODE, _JCOAG_CHUNK_SIZE_OVERRIDE
    global _F_DIFF_MODE, _VREL_TURB_MODE, _P_FRAG_MODE, _COLLISION_KERNEL_MODE
    _SCATTER_MODE = scatter_mode
    _S_COAG_MODE = s_coag_mode
    _JCOAG_GEN_MODE = jcoag_gen_mode
    _JCOAG_CHUNK_SIZE_OVERRIDE = jcoag_chunk_size_override
    _F_DIFF_MODE = f_diff_mode
    _VREL_TURB_MODE = vrel_turb_mode
    _P_FRAG_MODE = p_frag_mode
    _COLLISION_KERNEL_MODE = collision_kernel_mode


def reset_runtime_caches():
    for name in _RUNTIME_STATE_VARS:
        globals()[name] = None


def _scatter_add_1d(out, idx, vals):
    """1D scatter-add helper."""
    if int(idx.size) == 0:
        return
    if _SCATTER_MODE == "rawkernel":
        kernel = _get_raw_scatter_kernel(out.dtype)
        if (
            kernel is not None
            and getattr(out, "ndim", 0) == 1
            and getattr(idx, "ndim", 0) == 1
            and getattr(vals, "ndim", 0) == 1
        ):
            n = int(idx.size)
            threads = 256
            blocks = (n + threads - 1) // threads
            if blocks > 0:
                idx64 = idx if idx.dtype == cp.int64 else idx.astype(cp.int64, copy=False)
                vals_cast = vals if vals.dtype == out.dtype else vals.astype(out.dtype, copy=False)
                kernel((blocks,), (threads,), (out, idx64, vals_cast, np.int64(n)))
                return
    if _SCATTER_MODE == "scatter" and cp_scatter_add is not None:
        cp_scatter_add(out, idx, vals)
    else:
        cp.add.at(out, idx, vals)


def _get_jcoag_chunk_size(Nr_int, Nm):
    """Return chunk size for CuPy Jacobian radius batching."""
    if _JCOAG_CHUNK_SIZE_OVERRIDE is not None:
        return max(1, min(int(_JCOAG_CHUNK_SIZE_OVERRIDE), int(Nr_int)))

    # Auto heuristic tuned for production-like grids:
    # use one wide batch when grids are moderate, otherwise cap temporary size.
    if int(Nr_int) <= 256 and int(Nm) <= 128:
        return int(Nr_int)
    base = 128 if int(Nm) >= 96 else 64
    if int(Nr_int) > 384:
        base = 96 if int(Nm) >= 96 else 48
    return max(1, min(int(base), int(Nr_int)))


def _interp_to_interfaces_cupy(values, r, ri):
    """Interpolate CuPy arrays from cell centers to interfaces."""
    if cp_interp1d is None:
        if values.ndim == 1:
            out = cp.zeros((values.shape[0] + 1,), dtype=values.dtype)
            t = (ri[1:-1] - r[:-1]) / (r[1:] - r[:-1])
            out[1:-1] = values[:-1] + t * (values[1:] - values[:-1])
            m0 = (values[1] - values[0]) / (r[1] - r[0])
            m1 = (values[-1] - values[-2]) / (r[-1] - r[-2])
            out[0] = values[0] + m0 * (ri[0] - r[0])
            out[-1] = values[-1] + m1 * (ri[-1] - r[-1])
            return out
        out = cp.zeros((values.shape[0] + 1, values.shape[1]), dtype=values.dtype)
        t = ((ri[1:-1] - r[:-1]) / (r[1:] - r[:-1]))[:, None]
        out[1:-1, :] = values[:-1, :] + t * (values[1:, :] - values[:-1, :])
        m0 = (values[1, :] - values[0, :]) / (r[1] - r[0])
        m1 = (values[-1, :] - values[-2, :]) / (r[-1] - r[-2])
        out[0, :] = values[0, :] + m0 * (ri[0] - r[0])
        out[-1, :] = values[-1, :] + m1 * (ri[-1] - r[-1])
        return out
    f = cp_interp1d(r, values, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate")
    return f(ri)


def _get_coag_pair_indices(Nm):
    """Cache lower-triangle (j <= i) pair indices for coagulation scatters."""
    global _COAG_PAIR_CACHE_KEY, _COAG_PAIR_CACHE_VALUE
    if _COAG_PAIR_CACHE_KEY != Nm or _COAG_PAIR_CACHE_VALUE is None:
        j_np, i_np = np.tril_indices(Nm)
        _COAG_PAIR_CACHE_VALUE = (
            cp.asarray(j_np, dtype=cp.int64),
            cp.asarray(i_np, dtype=cp.int64),
        )
        _COAG_PAIR_CACHE_KEY = Nm
    return _COAG_PAIR_CACHE_VALUE


def _get_scoag_pair_map(Nm, p, imax):
    """Cache S_coag pair/erosion masks for each imax."""
    global _SCOAG_PAIRMAP_CACHE_KEY, _SCOAG_PAIRMAP_CACHE_VALUE
    key = (int(Nm), int(p))
    if _SCOAG_PAIRMAP_CACHE_KEY != key or _SCOAG_PAIRMAP_CACHE_VALUE is None:
        _SCOAG_PAIRMAP_CACHE_KEY = key
        _SCOAG_PAIRMAP_CACHE_VALUE = [None] * (int(Nm) + 1)

    imax = int(imax)
    cached = _SCOAG_PAIRMAP_CACHE_VALUE[imax]
    if cached is None:
        j_all, i_all = _get_coag_pair_indices(Nm)
        pair_mask = i_all < imax
        i_tri = i_all[pair_mask]
        j_tri = j_all[pair_mask]
        eros_mask = j_tri <= (i_tri - int(p) - 1)
        i_er = i_tri[eros_mask]
        j_er = j_tri[eros_mask]
        i_full = i_tri[~eros_mask]
        j_full = j_tri[~eros_mask]
        cached = (i_tri, j_tri, i_er, j_er, i_full, j_full)
        _SCOAG_PAIRMAP_CACHE_VALUE[imax] = cached
    return cached


def _get_frag_p(krm, Nm):
    """Cache fragmentation helper p derived from krm[:, -1] == -1."""
    global _FRAG_P_CACHE_KEY, _FRAG_P_CACHE_VALUE
    key = (int(Nm), id(krm))
    if _FRAG_P_CACHE_KEY == key and _FRAG_P_CACHE_VALUE is not None:
        return _FRAG_P_CACHE_VALUE

    mask_last = (krm[:, -1] == -1)
    if bool(cp.any(mask_last)):
        first_true = int(cp.argmax(mask_last).item())
        p = int(Nm - first_true - 1)
    else:
        p = int(Nm)

    _FRAG_P_CACHE_KEY = key
    _FRAG_P_CACHE_VALUE = p
    return p


def _get_mass_grid_q(m, Nm):
    """Cache Jacobian q derived from mass-grid spacing."""
    global _MGRID_Q_CACHE_KEY, _MGRID_Q_CACHE_VALUE
    key = (int(Nm), id(m))
    if _MGRID_Q_CACHE_KEY == key and _MGRID_Q_CACHE_VALUE is not None:
        return _MGRID_Q_CACHE_VALUE

    agrid = float((cp.log10(m[0] / m[-1]) / (1.0 - Nm)).item())
    q = int(math.ceil(math.log10(2.0) / agrid))
    _MGRID_Q_CACHE_KEY = key
    _MGRID_Q_CACHE_VALUE = q
    return q


def _get_scoag_precomp(cstick, cstick_ind, A, eps, klf, krm, phi, m, Nm):
    """Cache static pair/mask maps for _S_coag_cupy."""
    global _SCOAG_PRECOMP_CACHE_KEY, _SCOAG_PRECOMP_CACHE_VALUE
    key = (
        int(Nm),
        _S_COAG_MODE,
        id(cstick),
        id(cstick_ind),
        id(A),
        id(eps),
        id(klf),
        id(krm),
        id(phi),
        id(m),
    )
    if _SCOAG_PRECOMP_CACHE_KEY == key and _SCOAG_PRECOMP_CACHE_VALUE is not None:
        return _SCOAG_PRECOMP_CACHE_VALUE

    i_idx, j_idx = _get_coag_pair_indices(Nm)
    klf_p = klf[j_idx, i_idx]
    A_p = A[j_idx, i_idx]
    krm_p = krm[j_idx, i_idx]
    eps_p = eps[j_idx, i_idx]

    p_val = _get_frag_p(krm, Nm)
    erosion_p = j_idx <= (i_idx - p_val - 1)
    fullfrag_p = ~erosion_p
    ero_case1 = erosion_p & (krm_p == i_idx - 1)
    ero_case2 = erosion_p & ~ero_case1

    stick_maps = []
    for nz in range(4):
        k = cstick_ind[nz, j_idx, i_idx]
        valid = cp.where(k >= 0)[0]
        if int(valid.size) > 0:
            stick_maps.append(
                (
                    k[valid],
                    cstick[nz, j_idx, i_idx][valid],
                    valid,
                )
            )
        else:
            stick_maps.append((None, None, None))

    frag_valid = cp.where(klf_p >= 0)[0]
    if int(frag_valid.size) > 0:
        klf_v = klf_p[frag_valid]
        A_v = A_p[frag_valid]
    else:
        klf_v = None
        A_v = None

    c1 = cp.where(ero_case1)[0]
    c2 = cp.where(ero_case2)[0]
    ff = cp.where(fullfrag_p)[0]

    stick_sel_mat_gpu = None
    frag_a_sel_mat_gpu = None
    frag_sink_sel_mat_gpu = None
    if _S_COAG_MODE == "fused_spmm":
        i_idx_np = cp.asnumpy(i_idx).astype(np.int64, copy=False)
        j_idx_np = cp.asnumpy(j_idx).astype(np.int64, copy=False)
        n_pairs = int(i_idx_np.size)
        pair_idx = np.arange(n_pairs, dtype=np.int64)

        cstick_ind_np = cp.asnumpy(cstick_ind)
        cstick_np = cp.asnumpy(cstick)
        A_p_np = cp.asnumpy(A_p)
        klf_p_np = cp.asnumpy(klf_p)
        krm_p_np = cp.asnumpy(krm_p)
        eps_p_np = cp.asnumpy(eps_p)
        p_val_i = int(p_val)

        stick_rows = []
        stick_cols = []
        stick_vals = []
        for nz in range(4):
            k_np = cstick_ind_np[nz, j_idx_np, i_idx_np].astype(np.int64, copy=False)
            valid = k_np >= 0
            if np.any(valid):
                stick_rows.append(k_np[valid])
                stick_cols.append(pair_idx[valid])
                stick_vals.append(cstick_np[nz, j_idx_np[valid], i_idx_np[valid]])
        if len(stick_rows) > 0:
            stick_rows_np = np.concatenate(stick_rows)
            stick_cols_np = np.concatenate(stick_cols)
            stick_vals_np = np.concatenate(stick_vals)
            stick_mat_cpu = sp.coo_matrix((stick_vals_np, (stick_rows_np, stick_cols_np)), shape=(int(Nm), n_pairs)).tocsr()
            stick_sel_mat_gpu = cp_sparse.csr_matrix(stick_mat_cpu)

        frag_valid_np = np.where(klf_p_np >= 0)[0].astype(np.int64, copy=False)
        if int(frag_valid_np.size) > 0:
            klf_v_np = klf_p_np[frag_valid_np].astype(np.int64, copy=False)
            A_v_np = A_p_np[frag_valid_np]
            frag_a_mat_cpu = sp.coo_matrix((A_v_np, (klf_v_np, np.arange(int(frag_valid_np.size), dtype=np.int64))), shape=(int(Nm), int(frag_valid_np.size))).tocsr()
            frag_a_sel_mat_gpu = cp_sparse.csr_matrix(frag_a_mat_cpu)

        sink_rows = []
        sink_cols = []
        sink_vals = []
        for pidx in range(n_pairs):
            i_p = int(i_idx_np[pidx])
            j_p = int(j_idx_np[pidx])
            k_p = int(krm_p_np[pidx])
            e_p = float(eps_p_np[pidx])
            if j_p <= i_p - p_val_i - 1:
                if k_p == i_p - 1:
                    sink_rows.extend((k_p, k_p + 1, j_p))
                    sink_cols.extend((pidx, pidx, pidx))
                    sink_vals.extend((e_p, -e_p, -1.0))
                else:
                    sink_rows.extend((k_p, k_p + 1, i_p, j_p))
                    sink_cols.extend((pidx, pidx, pidx, pidx))
                    sink_vals.extend((e_p, 1.0 - e_p, -1.0, -1.0))
            else:
                sink_rows.extend((i_p, j_p))
                sink_cols.extend((pidx, pidx))
                sink_vals.extend((-1.0, -1.0))
        if len(sink_rows) > 0:
            sink_mat_cpu = sp.coo_matrix(
                (
                    np.asarray(sink_vals, dtype=np.float64),
                    (np.asarray(sink_rows, dtype=np.int64), np.asarray(sink_cols, dtype=np.int64)),
                ),
                shape=(int(Nm), n_pairs),
            ).tocsr()
            frag_sink_sel_mat_gpu = cp_sparse.csr_matrix(sink_mat_cpu)

    pre = {
        "i_idx": i_idx,
        "j_idx": j_idx,
        "krm_p": krm_p,
        "eps_p": eps_p,
        "stick_maps": stick_maps,
        "frag_valid": frag_valid,
        "klf_v": klf_v,
        "A_v": A_v,
        "c1": c1,
        "c2": c2,
        "ff": ff,
        "phi_lower": cp.tril(phi),
        "stick_sel_mat_gpu": stick_sel_mat_gpu,
        "frag_a_sel_mat_gpu": frag_a_sel_mat_gpu,
        "frag_sink_sel_mat_gpu": frag_sink_sel_mat_gpu,
    }
    _SCOAG_PRECOMP_CACHE_KEY = key
    _SCOAG_PRECOMP_CACHE_VALUE = pre
    return pre


def _active_imax_per_radius(Sigma, SigmaFloor):
    """Return per-radius active imax from Sigma > SigmaFloor as host int array."""
    active = Sigma > SigmaFloor
    any_active = cp.any(active, axis=1)
    last_from_end = cp.argmax(active[:, ::-1], axis=1)
    imax = cp.where(any_active, Sigma.shape[1] - last_from_end, 0).astype(cp.int64)
    return cp.asnumpy(imax)


def _get_jcoag_pattern_cupy(Nr, Nm, q):
    """Return cached CuPy sparse-pattern helpers for coagulation Jacobian packing."""
    global _JCOAG_PATTERN_GPU_CACHE_KEY, _JCOAG_PATTERN_GPU_CACHE_VALUE
    global _JSTICK_MAP_CACHE_KEY, _JSTICK_MAP_CACHE_VALUE
    key = (int(Nr), int(Nm), int(q))
    if _JCOAG_PATTERN_GPU_CACHE_KEY == key and _JCOAG_PATTERN_GPU_CACHE_VALUE is not None:
        return _JCOAG_PATTERN_GPU_CACHE_VALUE

    src_i, src_j, row, col, L = _get_jcoag_pattern(Nr, Nm, q)
    Ntot = int(Nr * Nm)
    # Build CSR structure once; in current ordering this is identity-permutation,
    # but keep a permutation fallback for safety.
    coo = sp.coo_matrix((np.arange(int(row.size), dtype=np.int64), (row, col)), shape=(Ntot, Ntot))
    csr = coo.tocsr()
    perm = csr.data.astype(np.int64, copy=False)
    ident = np.arange(int(row.size), dtype=np.int64)
    perm_gpu = None if np.array_equal(perm, ident) else cp.asarray(perm)
    _JCOAG_PATTERN_GPU_CACHE_KEY = key
    _JCOAG_PATTERN_GPU_CACHE_VALUE = (
        cp.asarray(src_i),
        cp.asarray(src_j),
        cp.asarray(row),
        cp.asarray(col),
        L,
        cp.asarray(csr.indices.astype(np.int64, copy=False)),
        cp.asarray(csr.indptr.astype(np.int64, copy=False)),
        perm_gpu,
    )
    return _JCOAG_PATTERN_GPU_CACHE_VALUE


def _get_jcoag_precomp_cupy(A, cStick, eps, iLF, iRM, iStick, m, phi):
    """Cache Jacobian scatter targets/constants to avoid per-step host transfers."""
    global _JCOAG_PRECOMP_CACHE_KEY, _JCOAG_PRECOMP_CACHE_VALUE
    Nm = int(m.shape[0])
    key = (int(Nm), id(A), id(cStick), id(eps), id(iLF), id(iRM), id(iStick), id(m), id(phi))
    if _JCOAG_PRECOMP_CACHE_KEY == key and _JCOAG_PRECOMP_CACHE_VALUE is not None:
        return _JCOAG_PRECOMP_CACHE_VALUE

    iStick_cpu = cp.asnumpy(iStick)
    cStick_cpu = cp.asnumpy(cStick)
    iLF_cpu = cp.asnumpy(iLF)
    iRM_cpu = cp.asnumpy(iRM)
    A_cpu = cp.asnumpy(A)
    eps_cpu = cp.asnumpy(eps)
    m_cpu = cp.asnumpy(m)
    phi_cpu = cp.asnumpy(phi)

    agrid = float(np.log10(m_cpu[0] / m_cpu[Nm - 1])) / (1.0 - Nm)
    q = int(np.ceil(np.log10(2.0) / agrid))
    src_i_cpu, src_j_cpu, _, _, _ = _get_jcoag_pattern(3, Nm, q)
    flat_sel = (src_i_cpu * Nm + src_j_cpu).astype(np.int64, copy=False)

    neg1_col = iRM_cpu[:, Nm - 1]
    neg1_positions = np.where(neg1_col == -1)[0]
    p_val = int(Nm - (neg1_positions[0] + 1)) if len(neg1_positions) > 0 else 0
    D_cpu = m_cpu[:, None] / m_cpu[None, :]

    tri = np.tri(Nm, dtype=bool)
    i_all, j_all = np.nonzero(tri)

    # Build sticking as a sparse linear operator over pair-rates:
    # (Nm*Nm, n_pairs) @ (n_pairs, span) -> (Nm*Nm, span).
    n_pairs = int(i_all.size)
    stick_rows_parts = []
    stick_cols_parts = []
    stick_vals_parts = []
    for l in range(4):
        k_all = iStick_cpu[l, j_all, i_all]
        valid = k_all >= 0
        if not np.any(valid):
            continue
        valid_idx = np.where(valid)[0].astype(np.int64, copy=False)
        k_v = k_all[valid]
        i_v = i_all[valid]
        coeff_v = D_cpu[k_v, i_v] * cStick_cpu[l, j_all[valid], i_all[valid]]
        stick_rows_parts.append((k_v * Nm + i_v).astype(np.int64, copy=False))
        stick_cols_parts.append(valid_idx)
        stick_vals_parts.append(coeff_v)
    if len(stick_rows_parts) > 0:
        stick_rows = np.concatenate(stick_rows_parts)
        stick_cols = np.concatenate(stick_cols_parts)
        stick_vals = np.concatenate(stick_vals_parts)
        stick_mat_cpu = sp.coo_matrix(
            (stick_vals, (stick_rows, stick_cols)),
            shape=(int(Nm * Nm), n_pairs),
        ).tocsr()
        stick_sel_mat_gpu = cp_sparse.csr_matrix(stick_mat_cpu[flat_sel, :])
        stick_idx_count = int(stick_rows.size)
    else:
        stick_sel_mat_gpu = None
        stick_mat_cpu = None
        stick_idx_count = 0

    frag_rows_parts = []
    frag_cols_parts = []
    frag_vals_parts = []

    # Precompute fragmentation redistribution once in packed (flat) form.
    klf_vals = iLF_cpu[j_all, i_all]
    frag_valid = klf_vals >= 0
    klf_v = klf_vals[frag_valid]
    A_v = A_cpu[j_all[frag_valid], i_all[frag_valid]]
    m_i_v = m_cpu[i_all[frag_valid]]
    i_frag = i_all[frag_valid]
    n_frag = int(np.sum(frag_valid))
    frag_dist_cpu = (A_v[:, None] * phi_cpu[klf_v, :]) / m_i_v[:, None]
    k_range = np.arange(Nm, dtype=np.int64)
    # Prebuild sparse redistribution operator (Nm*Nm, n_frag): each column is
    # one pair's mass redistribution over k for a fixed destination i.
    if n_frag > 0:
        frag_rows = (k_range[:, None] * Nm + i_frag[None, :]).ravel()
        frag_cols = np.broadcast_to(np.arange(n_frag, dtype=np.int64)[None, :], (Nm, n_frag)).ravel()
        frag_vals = frag_dist_cpu.T.ravel()
        frag_pair_idx = np.where(frag_valid)[0].astype(np.int64, copy=False)
        frag_cols_full = np.broadcast_to(frag_pair_idx[None, :], (Nm, n_frag)).ravel()
        frag_mat_cpu = sp.coo_matrix((frag_vals, (frag_rows, frag_cols)), shape=(int(Nm * Nm), n_frag)).tocsr()
        frag_sel_mat_gpu = cp_sparse.csr_matrix(frag_mat_cpu[flat_sel, :])
        frag_rows_parts.append(frag_rows.astype(np.int64, copy=False))
        frag_cols_parts.append(frag_cols_full.astype(np.int64, copy=False))
        frag_vals_parts.append(frag_vals)
        frag_idx_count = int(frag_rows.size)
    else:
        frag_sel_mat_gpu = None
        frag_idx_count = 0

    # Split pair space into erosion and full-fragmentation branches.
    is_erosion = j_all <= i_all - p_val - 1
    ero_idx = np.where(is_erosion)[0]
    ero_i = i_all[ero_idx]
    ero_j = j_all[ero_idx]
    ero_krm = iRM_cpu[ero_j, ero_i]
    ero_eps = eps_cpu[ero_j, ero_i]
    case1 = ero_krm == ero_i - 1
    case2 = ~case1

    # Build erosion/full-fragmentation branch operators once so runtime can use
    # sparse matmul instead of repeated scatter-index expansion.
    c1_i = ero_i[case1]
    c1_j = ero_j[case1]
    c1_eps = ero_eps[case1]
    c1_pair_idx = ero_idx[case1]
    if len(c1_i) > 0:
        c1_n = int(len(c1_i))
        c1_cols = np.arange(c1_n, dtype=np.int64)
        c1_rows = np.concatenate(
            (
                ((c1_i - 1) * Nm + c1_i).astype(np.int64, copy=False),
                (c1_i * Nm + c1_i).astype(np.int64, copy=False),
                (c1_j * Nm + c1_i).astype(np.int64, copy=False),
            )
        )
        c1_cols_rep = np.concatenate((c1_cols, c1_cols, c1_cols))
        c1_vals = np.concatenate(
            (
                D_cpu[c1_i - 1, c1_i] * c1_eps,
                -c1_eps,
                -D_cpu[c1_j, c1_i],
            )
        )
        c1_mat_cpu = sp.coo_matrix(
            (c1_vals, (c1_rows, c1_cols_rep)),
            shape=(int(Nm * Nm), c1_n),
        ).tocsr()
        c1_sel_mat_gpu = cp_sparse.csr_matrix(c1_mat_cpu[flat_sel, :])
        c1_pair_idx_gpu = cp.asarray(c1_pair_idx.astype(np.int64, copy=False))
        c1_cols_full = np.concatenate((c1_pair_idx, c1_pair_idx, c1_pair_idx)).astype(np.int64, copy=False)
        frag_rows_parts.append(c1_rows.astype(np.int64, copy=False))
        frag_cols_parts.append(c1_cols_full)
        frag_vals_parts.append(c1_vals)
        c1_idx_count = int(c1_rows.size)
    else:
        c1_sel_mat_gpu = None
        c1_pair_idx_gpu = None
        c1_idx_count = 0

    c2_i = ero_i[case2]
    c2_j = ero_j[case2]
    c2_eps = ero_eps[case2]
    c2_krm = ero_krm[case2]
    c2_pair_idx = ero_idx[case2]
    if len(c2_i) > 0:
        c2_n = int(len(c2_i))
        c2_cols = np.arange(c2_n, dtype=np.int64)
        c2_rows = np.concatenate(
            (
                (c2_krm * Nm + c2_i).astype(np.int64, copy=False),
                ((c2_krm + 1) * Nm + c2_i).astype(np.int64, copy=False),
                (c2_i * Nm + c2_i).astype(np.int64, copy=False),
                (c2_j * Nm + c2_i).astype(np.int64, copy=False),
            )
        )
        c2_cols_rep = np.concatenate((c2_cols, c2_cols, c2_cols, c2_cols))
        c2_vals = np.concatenate(
            (
                D_cpu[c2_krm, c2_i] * c2_eps,
                D_cpu[c2_krm + 1, c2_i] * (1.0 - c2_eps),
                -np.ones(c2_n, dtype=np.float64),
                -D_cpu[c2_j, c2_i],
            )
        )
        c2_mat_cpu = sp.coo_matrix(
            (c2_vals, (c2_rows, c2_cols_rep)),
            shape=(int(Nm * Nm), c2_n),
        ).tocsr()
        c2_sel_mat_gpu = cp_sparse.csr_matrix(c2_mat_cpu[flat_sel, :])
        c2_pair_idx_gpu = cp.asarray(c2_pair_idx.astype(np.int64, copy=False))
        c2_cols_full = np.concatenate((c2_pair_idx, c2_pair_idx, c2_pair_idx, c2_pair_idx)).astype(np.int64, copy=False)
        frag_rows_parts.append(c2_rows.astype(np.int64, copy=False))
        frag_cols_parts.append(c2_cols_full)
        frag_vals_parts.append(c2_vals)
        c2_idx_count = int(c2_rows.size)
    else:
        c2_sel_mat_gpu = None
        c2_pair_idx_gpu = None
        c2_idx_count = 0

    ff_idx = np.where(~is_erosion)[0]
    ff_i = i_all[ff_idx]
    ff_j = j_all[ff_idx]
    if len(ff_i) > 0:
        ff_n = int(len(ff_i))
        ff_cols = np.arange(ff_n, dtype=np.int64)
        ff_rows = np.concatenate(
            (
                (ff_i * Nm + ff_i).astype(np.int64, copy=False),
                (ff_j * Nm + ff_i).astype(np.int64, copy=False),
            )
        )
        ff_cols_rep = np.concatenate((ff_cols, ff_cols))
        ff_vals = np.concatenate((-np.ones(ff_n, dtype=np.float64), -D_cpu[ff_j, ff_i]))
        ff_mat_cpu = sp.coo_matrix(
            (ff_vals, (ff_rows, ff_cols_rep)),
            shape=(int(Nm * Nm), ff_n),
        ).tocsr()
        ff_sel_mat_gpu = cp_sparse.csr_matrix(ff_mat_cpu[flat_sel, :])
        ff_pair_idx_gpu = cp.asarray(ff_idx.astype(np.int64, copy=False))
        ff_cols_full = np.concatenate((ff_idx, ff_idx)).astype(np.int64, copy=False)
        frag_rows_parts.append(ff_rows.astype(np.int64, copy=False))
        frag_cols_parts.append(ff_cols_full)
        frag_vals_parts.append(ff_vals)
        ff_idx_count = int(ff_rows.size)
    else:
        ff_sel_mat_gpu = None
        ff_pair_idx_gpu = None
        ff_idx_count = 0

    if len(frag_rows_parts) > 0:
        frag_all_rows = np.concatenate(frag_rows_parts)
        frag_all_cols = np.concatenate(frag_cols_parts)
        frag_all_vals = np.concatenate(frag_vals_parts)
        frag_all_mat_cpu = sp.coo_matrix((frag_all_vals, (frag_all_rows, frag_all_cols)), shape=(int(Nm * Nm), n_pairs)).tocsr()
        frag_all_sel_mat_gpu = cp_sparse.csr_matrix(frag_all_mat_cpu[flat_sel, :])
    else:
        frag_all_sel_mat_gpu = None

    if stick_sel_mat_gpu is not None and frag_all_sel_mat_gpu is not None:
        jcoag_fused_sel_mat_gpu = cp_sparse.hstack([stick_sel_mat_gpu, frag_all_sel_mat_gpu], format="csr")
    else:
        jcoag_fused_sel_mat_gpu = None

    pre = {
        "q": q,
        "j_gpu": cp.asarray(j_all.astype(np.int64)),
        "i_gpu": cp.asarray(i_all.astype(np.int64)),
        "n_pairs": n_pairs,
        "stick_sel_mat_gpu": stick_sel_mat_gpu,
        "stick_idx_count": stick_idx_count,
        "frag_valid_gpu": cp.asarray(np.where(frag_valid)[0].astype(np.int64)),
        "frag_sel_mat_gpu": frag_sel_mat_gpu,
        "frag_idx_count": frag_idx_count,
        "n_frag": n_frag,
        "c1_sel_mat_gpu": c1_sel_mat_gpu,
        "c1_pair_idx_gpu": c1_pair_idx_gpu,
        "c1_idx_count": c1_idx_count,
        "c2_sel_mat_gpu": c2_sel_mat_gpu,
        "c2_pair_idx_gpu": c2_pair_idx_gpu,
        "c2_idx_count": c2_idx_count,
        "ff_sel_mat_gpu": ff_sel_mat_gpu,
        "ff_pair_idx_gpu": ff_pair_idx_gpu,
        "ff_idx_count": ff_idx_count,
        "frag_all_sel_mat_gpu": frag_all_sel_mat_gpu,
        "jcoag_fused_sel_mat_gpu": jcoag_fused_sel_mat_gpu,
    }
    _JCOAG_PRECOMP_CACHE_KEY = key
    _JCOAG_PRECOMP_CACHE_VALUE = pre
    return pre


def _get_boundary_basis_cupy(Nr, Nm, dtype):
    """Return cached sparse basis matrices for dust inner/outer boundaries."""
    global _BOUNDARY_BASIS_CACHE_KEY, _BOUNDARY_BASIS_CACHE_VALUE
    key = (int(Nr), int(Nm), str(dtype))
    if _BOUNDARY_BASIS_CACHE_KEY == key and _BOUNDARY_BASIS_CACHE_VALUE is not None:
        return _BOUNDARY_BASIS_CACHE_VALUE

    Ntot = int(Nr * Nm)
    row0 = cp.arange(int(Nm), dtype=cp.int64)
    offset = int((Nr - 1) * Nm)
    one = cp.ones((int(Nm),), dtype=dtype)

    Bin0 = cp_sparse.csr_matrix((one, (row0, row0)), shape=(Ntot, Ntot))
    Bin1 = cp_sparse.csr_matrix((one, (row0, row0 + int(Nm))), shape=(Ntot, Ntot))
    Bin2 = cp_sparse.csr_matrix((one, (row0, row0 + int(2 * Nm))), shape=(Ntot, Ntot))

    row_out = row0 + offset
    Bout0 = cp_sparse.csr_matrix((one, (row_out, row_out)), shape=(Ntot, Ntot))
    Bout1 = cp_sparse.csr_matrix((one, (row_out, row_out - int(Nm))), shape=(Ntot, Ntot))
    Bout2 = cp_sparse.csr_matrix((one, (row_out, row_out - int(2 * Nm))), shape=(Ntot, Ntot))

    _BOUNDARY_BASIS_CACHE_KEY = key
    _BOUNDARY_BASIS_CACHE_VALUE = (Bin0, Bin1, Bin2, Bout0, Bout1, Bout2)
    return _BOUNDARY_BASIS_CACHE_VALUE


def _get_dust_hyd_boundary_pattern_cupy(Nr, Nm):
    """Return cached duplicate-aware CSR helpers for (J_hyd + J_in + J_out)."""
    global _DUST_JHB_PATTERN_CACHE_KEY, _DUST_JHB_PATTERN_CACHE_VALUE
    key = (int(Nr), int(Nm))
    if _DUST_JHB_PATTERN_CACHE_KEY == key and _DUST_JHB_PATTERN_CACHE_VALUE is not None:
        return _DUST_JHB_PATTERN_CACHE_VALUE

    Nr_i = int(Nr)
    Nm_i = int(Nm)
    Ntot = int(Nr_i * Nm_i)

    # Hydrodynamic tridiagonal block pattern with +/- Nm offsets.
    idx = np.arange(Ntot, dtype=np.int64)
    row_hyd = np.hstack((idx[Nm_i:], idx, idx[:-Nm_i]))
    col_hyd = np.hstack((idx[:-Nm_i], idx, idx[Nm_i:]))

    # Inner/outer boundary placeholders (3*Nm each) aligned with dat_in/dat_out layout.
    row0 = np.arange(Nm_i, dtype=np.int64)
    row_out = row0 + (Nr_i - 1) * Nm_i
    row_b = np.hstack((row0, row0, row0, row_out, row_out, row_out))
    col_b = np.hstack(
        (
            row0,
            row0 + Nm_i,
            row0 + 2 * Nm_i,
            row_out,
            row_out - Nm_i,
            row_out - 2 * Nm_i,
        )
    )

    row_all = np.hstack((row_hyd, row_b))
    col_all = np.hstack((col_hyd, col_b))
    # Build CSR structure once and map each original COO entry to the unique
    # CSR data slot so runtime can reduce duplicates with a single scatter.
    coo_struct = sp.coo_matrix(
        (np.ones(int(row_all.size), dtype=np.float64), (row_all, col_all)),
        shape=(Ntot, Ntot),
    )
    csr_struct = coo_struct.tocsr()
    indptr = csr_struct.indptr.astype(np.int64, copy=False)
    indices = csr_struct.indices.astype(np.int64, copy=False)

    pair_to_pos = {}
    for r in range(Ntot):
        s = int(indptr[r])
        e = int(indptr[r + 1])
        cols = indices[s:e]
        for off, c in enumerate(cols):
            pair_to_pos[(r, int(c))] = s + off
    map_idx = np.fromiter(
        (pair_to_pos[(int(r), int(c))] for r, c in zip(row_all, col_all)),
        dtype=np.int64,
        count=int(row_all.size),
    )

    _DUST_JHB_PATTERN_CACHE_KEY = key
    _DUST_JHB_PATTERN_CACHE_VALUE = (
        int(row_hyd.size),
        cp.asarray(map_idx),
        cp.asarray(indices),
        cp.asarray(indptr),
    )
    return _DUST_JHB_PATTERN_CACHE_VALUE


def _get_jstick_map_cupy(iStick, cStick, Nm):
    """Cache static sticking scatter maps per i: (kv, jv, cv)."""
    global _JSTICK_MAP_CACHE_KEY, _JSTICK_MAP_CACHE_VALUE
    key = (int(Nm), id(iStick), id(cStick))
    if _JSTICK_MAP_CACHE_KEY == key and _JSTICK_MAP_CACHE_VALUE is not None:
        return _JSTICK_MAP_CACHE_VALUE

    maps = []
    for i in range(int(Nm)):
        j_idx = cp.arange(i + 1, dtype=cp.int64)
        k4 = iStick[:, j_idx, i]
        c4 = cStick[:, j_idx, i]
        valid = k4 >= 0
        kv = k4[valid]
        jv = cp.broadcast_to(j_idx[None, :], k4.shape)[valid]
        cv = c4[valid]
        maps.append((kv, jv, cv))

    _JSTICK_MAP_CACHE_KEY = key
    _JSTICK_MAP_CACHE_VALUE = maps
    return maps


def _get_jfrag_map_cupy(A, eps, iLF, iRM, Nm, p):
    """Cache static fragmentation maps per i."""
    global _JFRAG_MAP_CACHE_KEY, _JFRAG_MAP_CACHE_VALUE
    key = (int(Nm), int(p), id(A), id(eps), id(iLF), id(iRM))
    if _JFRAG_MAP_CACHE_KEY == key and _JFRAG_MAP_CACHE_VALUE is not None:
        return _JFRAG_MAP_CACHE_VALUE

    maps = []
    for i in range(int(Nm)):
        j_idx = cp.arange(i + 1, dtype=cp.int64)
        a_i = A[j_idx, i]
        lf_i = iLF[j_idx, i]

        eros_idx = j_idx <= (i - int(p) - 1)
        j_er = j_idx[eros_idx]
        k_er = iRM[j_er, i]
        eps_er = eps[j_er, i]
        eq_er = k_er == (i - 1)

        j_full = j_idx[~eros_idx]
        maps.append((a_i, lf_i, j_er, k_er, eps_er, eq_er, j_full))

    _JFRAG_MAP_CACHE_KEY = key
    _JFRAG_MAP_CACHE_VALUE = maps
    return maps


def _get_jcoag_work_buffer(Nr, Nm, dtype):
    """Cache reusable Jacobian work buffer to reduce allocator churn."""
    global _JCOAG_WORK_CACHE_KEY, _JCOAG_WORK_CACHE_VALUE
    key = (int(Nr), int(Nm), str(dtype))
    if _JCOAG_WORK_CACHE_KEY != key or _JCOAG_WORK_CACHE_VALUE is None:
        _JCOAG_WORK_CACHE_KEY = key
        _JCOAG_WORK_CACHE_VALUE = cp.zeros((int(Nr), int(Nm), int(Nm)), dtype=dtype)
    else:
        _JCOAG_WORK_CACHE_VALUE.fill(0.0)
    return _JCOAG_WORK_CACHE_VALUE


def _jacobian_coagulation_generator_cupy(A, cStick, eps, iLF, iRM, iStick, m, phi, Rf, Rs, Sigma, SigmaFloor):
    """Generate the coagulation Jacobian from CuPy arrays."""
    Nr = int(Sigma.shape[0])
    Nm = int(Sigma.shape[1])
    N = Sigma / m[None, :]

    pre = _get_jcoag_precomp_cupy(A, cStick, eps, iLF, iRM, iStick, m, phi)
    q = pre["q"]
    j_gpu = pre["j_gpu"]
    i_gpu = pre["i_gpu"]
    _, src_j, row, col, _, _, _, _ = _get_jcoag_pattern_cupy(Nr, Nm, q)

    Nr_int = Nr - 2
    L = int(src_j.size)
    dat_mid = cp.zeros((Nr_int, L), dtype=Sigma.dtype)
    n_mid = N[1:-1]
    Rs_mid = Rs[1:-1]
    Rf_mid = Rf[1:-1]

    # Chunk over radius to balance launch overhead and temporary memory.
    chunk_size = _get_jcoag_chunk_size(Nr_int, Nm)
    for start in range(0, Nr_int, chunk_size):
        stop = min(start + chunk_size, Nr_int)
        dat_chunk = dat_mid[start:stop]

        rates_s = n_mid[start:stop, j_gpu] * Rs_mid[start:stop, j_gpu, i_gpu]
        ratef = n_mid[start:stop, j_gpu] * Rf_mid[start:stop, j_gpu, i_gpu]

        if _JCOAG_GEN_MODE == "fused_spmm" and pre["jcoag_fused_sel_mat_gpu"] is not None:
            # Fused path: one SpMM over concatenated sticking and fragmentation rates.
            rates_fused = cp.concatenate((rates_s, ratef), axis=1)
            dat_chunk += pre["jcoag_fused_sel_mat_gpu"].dot(rates_fused.T).T
        else:
            if pre["stick_sel_mat_gpu"] is not None:
                # Sticking contribution from all pair-rates in one sparse matmul.
                dat_chunk += pre["stick_sel_mat_gpu"].dot(rates_s.T).T

            if pre["n_frag"] > 0:
                # Apply fragmentation via sparse operator to avoid building massive
                # per-step flat index tensors for (k, pair) redistribution.
                ratef_v = ratef[:, pre["frag_valid_gpu"]]
                dat_chunk += pre["frag_sel_mat_gpu"].dot(ratef_v.T).T

            if pre["c1_sel_mat_gpu"] is not None:
                ratef_c1 = ratef[:, pre["c1_pair_idx_gpu"]]
                dat_chunk += pre["c1_sel_mat_gpu"].dot(ratef_c1.T).T

            if pre["c2_sel_mat_gpu"] is not None:
                ratef_c2 = ratef[:, pre["c2_pair_idx_gpu"]]
                dat_chunk += pre["c2_sel_mat_gpu"].dot(ratef_c2.T).T

            if pre["ff_sel_mat_gpu"] is not None:
                ratef_ff = ratef[:, pre["ff_pair_idx_gpu"]]
                dat_chunk += pre["ff_sel_mat_gpu"].dot(ratef_ff.T).T

    # Apply active-mass masking directly on packed sparse data values.
    active = Sigma[1:-1] > SigmaFloor[1:-1]
    any_active = cp.any(active, axis=1)
    imax_arr = cp.where(any_active, Nm - cp.argmax(active[:, ::-1], axis=1), 0)
    dat_mid *= (src_j[None, :] < imax_arr[:, None])
    dat_gpu = dat_mid.reshape(-1)
    return dat_gpu, row, col


def _jacobian_hydrodynamic_generator_cupy(area, D, r, ri, SigmaGas, v, freeze_velocity_mask=None, freeze_diffusion_mask=None):
    """Generate the hydrodynamic Jacobian from CuPy arrays."""
    Nr = int(r.shape[0])
    Nm = int(D.shape[1])

    h = SigmaGas * r
    hi = _interp_to_interfaces(h, r, ri)
    vi = _interp_to_interfaces(v, r, ri)
    Di = _interp_to_interfaces(D, r, ri)

    if freeze_velocity_mask is not None:
        if int(to_numpy(freeze_velocity_mask.sum())) > 0:
            if freeze_velocity_mask.ndim == 1 and int(freeze_velocity_mask.shape[0]) == int(vi.shape[0]):
                vi[freeze_velocity_mask, :] = 0.0
            elif freeze_velocity_mask.shape == vi.shape:
                vi[freeze_velocity_mask] = 0.0
    if freeze_diffusion_mask is not None:
        if int(to_numpy(freeze_diffusion_mask.sum())) > 0:
            if freeze_diffusion_mask.ndim == 1 and int(freeze_diffusion_mask.shape[0]) == int(Di.shape[0]):
                Di[freeze_diffusion_mask, :] = 0.0
            elif freeze_diffusion_mask.shape == Di.shape:
                Di[freeze_diffusion_mask] = 0.0

    vim = xp.minimum(vi, 0.0)
    vip = xp.maximum(vi, 0.0)

    A = xp.zeros((Nr, Nm))
    B = xp.zeros((Nr, Nm))
    C = xp.zeros((Nr, Nm))

    Vinv = xp.zeros((Nr,))
    Vinv[:-1] = (2.0 * c.pi) / area[:-1]

    w = xp.ones((Nr,))
    w[:-1] = r[1:] - r[:-1]

    # Keep all indexing on backend arrays; avoid host NumPy index vectors.
    A[1:-1, :] += vip[1:-2, :] * r[:-2, None]
    B[1:-1, :] += -vip[2:-1, :] * r[1:-1, None] + vim[1:-2, :] * r[1:-1, None]
    C[1:-1, :] += -vim[2:-1, :] * r[2:, None]

    A[1:-1, :] += Di[1:-2, :] * hi[1:-2, None] / (w[:-2, None] * h[:-2, None]) * r[:-2, None]
    B[1:-1, :] += -Di[1:-2, :] * hi[1:-2, None] / (w[:-2, None] * h[1:-1, None]) * r[1:-1, None]
    B[1:-1, :] += -Di[2:-1, :] * hi[2:-1, None] / (w[1:-1, None] * h[1:-1, None]) * r[1:-1, None]
    C[1:-1, :] += Di[2:-1, :] * hi[2:-1, None] / (w[1:-1, None] * h[2:, None]) * r[2:, None]

    A = A * Vinv[:, None]
    B = B * Vinv[:, None]
    C = C * Vinv[:, None]

    return A, B, C


def _apply_zero_flux_dust_hyd_edges_cupy(A, B, C, area, D, r, ri, SigmaGas, v):
    """Inject conservative zero-flux boundary rows using CuPy arrays."""
    Nr = int(A.shape[0])
    if Nr < 2:
        return A, B, C

    h = SigmaGas * r
    hi = _interp_to_interfaces(h, r, ri)
    Di = _interp_to_interfaces(D, r, ri)
    vi = _interp_to_interfaces(v, r, ri)
    vip = xp.maximum(vi, 0.0)
    vim = xp.minimum(vi, 0.0)
    Vinv = (2.0 * c.pi) / area
    w_in = r[1] - r[0]
    w_out = r[-1] - r[-2]

    h0 = h[0] + 1.0e-300
    h1 = h[1] + 1.0e-300
    hm2 = h[-2] + 1.0e-300
    hm1 = h[-1] + 1.0e-300

    A[0, :] = 0.0
    B[0, :] = (
        -vip[1, :] * r[0]
        - Di[1, :] * hi[1] / (w_in * h0) * r[0]
    ) * Vinv[0]
    C[0, :] = (
        -vim[1, :] * r[1]
        + Di[1, :] * hi[1] / (w_in * h1) * r[1]
    ) * Vinv[0]

    A[-1, :] = (
        vip[-2, :] * r[-2]
        + Di[-2, :] * hi[-2] / (w_out * hm2) * r[-2]
    ) * Vinv[-1]
    B[-1, :] = (
        vim[-2, :] * r[-1]
        - Di[-2, :] * hi[-2] / (w_out * hm1) * r[-1]
    ) * Vinv[-1]
    C[-1, :] = 0.0
    return A, B, C


def _apply_inner_zero_flux_dust_hyd_edge_cupy(A, B, C, area, D, r, ri, SigmaGas, v, block_mask, adv_drain_mask=None, diff_drain_mask=None):
    """Inject a selective inner zero-flux row using CuPy arrays."""
    Nr = int(A.shape[0])
    if Nr < 2:
        return A, B, C

    h = SigmaGas * r
    hi = _interp_to_interfaces(h, r, ri)
    Di = _interp_to_interfaces(D, r, ri)
    vi = _interp_to_interfaces(v, r, ri)
    vip = xp.maximum(vi, 0.0)
    vim = xp.minimum(vi, 0.0)
    Vinv = (2.0 * c.pi) / area
    w_in = r[1] - r[0]

    h0 = h[0] + 1.0e-300
    h1 = h[1] + 1.0e-300

    if block_mask is None:
        return A, B, C
    if int(to_numpy(block_mask.sum())) == 0:
        return A, B, C

    vip1 = vip[1, :].copy()
    di1 = Di[1, :].copy()
    if adv_drain_mask is not None:
        if int(to_numpy(adv_drain_mask.sum())) > 0:
            vip1[adv_drain_mask] = 0.0
    if diff_drain_mask is not None:
        if int(to_numpy(diff_drain_mask.sum())) > 0:
            di1[diff_drain_mask] = 0.0

    b0 = (
        -vip1 * r[0]
        - di1 * hi[1] / (w_in * h0) * r[0]
    ) * Vinv[0]
    c0 = (
        -vim[1, :] * r[1]
        + di1 * hi[1] / (w_in * h1) * r[1]
    ) * Vinv[0]
    A0 = A[0, :]
    B0 = B[0, :]
    C0 = C[0, :]
    A0[block_mask] = 0.0
    B0[block_mask] = b0[block_mask]
    C0[block_mask] = c0[block_mask]
    return A, B, C

_interp_to_interfaces = _interp_to_interfaces_cupy


def _get_cupy_dust_solver_mode():
    """Return CuPy dust implicit solver mode."""
    raw = os.getenv("DUSTPY_CUPY_DUST_SOLVER", "sparse").strip().lower()
    aliases = {
        "sparse": "sparse",
        "sparse_gpu": "sparse",
        "gmres": "sparse",
        "dense": "dense_gpu",
        "dense_gpu": "dense_gpu",
    }
    return aliases.get(raw, "sparse")


def _get_cupy_diag_positions_csr(matrix):
    """Return cached CSR data positions for diagonal entries, or None if absent."""
    global _CUPY_DIAG_POS_CACHE_KEY, _CUPY_DIAG_POS_CACHE_VALUE
    if cp_sparse is None:
        return None
    if not isinstance(matrix, cp_sparse.spmatrix):
        return None
    if getattr(matrix, "format", None) != "csr":
        return None

    key = (int(matrix.shape[0]), int(matrix.shape[1]), int(matrix.nnz), str(matrix.dtype))
    if _CUPY_DIAG_POS_CACHE_KEY == key and _CUPY_DIAG_POS_CACHE_VALUE is not None:
        return _CUPY_DIAG_POS_CACHE_VALUE

    n = int(matrix.shape[0])
    indptr = np.asarray(cp.asnumpy(matrix.indptr), dtype=np.int64)
    indices = np.asarray(cp.asnumpy(matrix.indices), dtype=np.int64)
    pos = np.full((n,), -1, dtype=np.int64)

    ok = True
    for i in range(n):
        s = int(indptr[i])
        e = int(indptr[i + 1])
        row_cols = indices[s:e]
        j = int(np.searchsorted(row_cols, i))
        if j >= (e - s) or int(row_cols[j]) != i:
            ok = False
            break
        pos[i] = s + j

    if not ok:
        _CUPY_DIAG_POS_CACHE_KEY = key
        _CUPY_DIAG_POS_CACHE_VALUE = None
        return None

    out = cp.asarray(pos)
    _CUPY_DIAG_POS_CACHE_KEY = key
    _CUPY_DIAG_POS_CACHE_VALUE = out
    return out


def _a_cupy(sim):
    rho = sim.dust.fill * sim.dust.rhos
    m = xp.asarray(sim.grid.m)[None, :]
    return (3.0 * m / (4.0 * c.pi * rho)) ** (1.0 / 3.0)


def _D_cupy(sim):
    v2 = sim.dust.delta.rad * sim.gas.cs**2
    Diff = v2[:, None] / (sim.grid.OmegaK[:, None] * (1.0 + sim.dust.St**2))
    Diff[:2, ...] = 0.
    Diff[-2:, ...] = 0.
    _apply_gas_floor_freeze_to_radial_field(sim, Diff)
    return Diff


def _H_cupy(sim):
    Hp = sim.gas.Hp[:, None]
    H = Hp / xp.sqrt(1.0 + sim.dust.St / sim.dust.delta.vert[:, None])
    return xp.minimum(H, Hp)


def _F_adv_cupy(sim, Sigma=None):
    Sigma = Sigma if Sigma is not None else sim.dust.Sigma
    Sigma = _field_data(Sigma)
    v = _field_data(sim.dust.v.rad)
    r = _field_data(sim.grid.r)
    ri = _field_data(sim.grid.ri)

    Nr = int(sim.grid.Nr)
    Nm = int(sim.grid.Nm)

    vi = xp.zeros((Nr + 1, Nm))
    t = ((ri[1:-1] - r[:-1]) / (r[1:] - r[:-1]))[:, None]
    vi[1:-1, :] = v[:-1, :] + t * (v[1:, :] - v[:-1, :])
    vi[0, :] = vi[1, :]
    vi[-1, :] = vi[-2, :]

    Fi = xp.zeros((Nr + 1, Nm))
    Fi[1:-1, :] = Sigma[:-1, :] * xp.maximum(vi[1:-1, :], 0.0) + Sigma[1:, :] * xp.minimum(vi[1:-1, :], 0.0)
    Fi[0, :] = Sigma[0, :] * xp.minimum(vi[1, :], 0.0)
    Fi[-1, :] = Sigma[-1, :] * xp.maximum(vi[-2, :], 0.0)
    if is_zero_flux_enabled(sim):
        Fi[0, :] = 0.0
        Fi[-1, :] = 0.0
    _apply_gas_floor_freeze_to_flux(sim, Fi)
    return Fi


def _F_diff_cupy(sim, Sigma=None):
    if Sigma is None:
        Sigma = sim.dust.Sigma

    D = _field_data(sim.dust.D)
    SigmaD = _field_data(Sigma)
    SigmaG = _field_data(sim.gas.Sigma)
    St = _field_data(sim.dust.St)
    u = _field_data(xp.sqrt(sim.dust.delta.rad * sim.gas.cs**2))
    r = _field_data(sim.grid.r)
    ri = _field_data(sim.grid.ri)

    if _F_DIFF_MODE == "elementwise":
        Fi = _F_diff_cupy_elementwise(D, SigmaD, SigmaG, St, u, r, ri)
        if Fi is not None:
            _apply_gas_floor_freeze_to_flux(sim, Fi)
            return Fi

    SigGi = _interp_to_interfaces_cupy(SigmaG, r, ri)
    ui = _interp_to_interfaces_cupy(u, r, ri)
    Di = _interp_to_interfaces_cupy(D, r, ri)
    SigDi = _interp_to_interfaces_cupy(SigmaD, r, ri)
    Sti = _interp_to_interfaces_cupy(St, r, ri)

    eps = SigmaD / SigmaG[:, None]
    gradepsi = xp.zeros_like(SigDi)
    gradepsi[1:-1, :] = (eps[1:, :] - eps[:-1, :]) / (r[1:] - r[:-1])[:, None]

    Fi = xp.zeros_like(SigDi)

    w = ui[1:-1, None] * SigDi[1:-1, :] / (1.0 + Sti[1:-1, :]**2)
    Fi0 = -Di[1:-1, :] * SigGi[1:-1, None] * gradepsi[1:-1, :]

    mask = xp.abs(w) > 0.0
    P = xp.zeros_like(Fi0)
    P[mask] = xp.abs(Fi0[mask] / w[mask])
    lam = (1.0 + P) / (1.0 + P + P**2)

    Fi_int = lam * Fi0
    Fi_int = xp.where(mask, Fi_int, w)
    Fi[1:-1, :] = Fi_int

    Fi[0, :] = Fi[1, :]
    Fi[-1, :] = Fi[-2, :]

    # Preserve current dust.py behavior at boundaries.
    Fi[:1, :] = 0.
    Fi[-1:, :] = 0.
    _apply_gas_floor_freeze_to_flux(sim, Fi)
    return Fi


def _S_hyd_cupy(sim, Sigma=None):
    if Sigma is None:
        Sigma = sim.dust.Sigma
        Fi = sim.dust.Fi.tot
    else:
        Fi = sim.dust.Fi.tot.updater.beat(sim, Sigma=Sigma)
        if Fi is None:
            Fi = sim.dust.Fi.tot
    Fi = _field_data(Fi)
    ri = _field_data(sim.grid.ri)
    denom = (ri[1:]**2 - ri[:-1]**2)[:, None]
    return 2.0 * (Fi[:-1, :] * ri[:-1, None] - Fi[1:, :] * ri[1:, None]) / denom


def _S_coag_cupy(sim, Sigma=None):
    if Sigma is None:
        Sigma = sim.dust.Sigma

    global _COAG_CACHE_KEY, _COAG_CACHE_VALUE
    cache_key = (
        id(_field_data(sim.dust.coagulation.stick)),
        id(_field_data(sim.dust.coagulation.stick_ind)),
        id(_field_data(sim.dust.coagulation.A)),
        id(_field_data(sim.dust.coagulation.eps)),
        id(_field_data(sim.dust.coagulation.lf_ind)),
        id(_field_data(sim.dust.coagulation.rm_ind)),
        id(_field_data(sim.dust.coagulation.phi)),
        id(_field_data(sim.grid.m)),
    )
    if _COAG_CACHE_KEY != cache_key or _COAG_CACHE_VALUE is None:
        _COAG_CACHE_KEY = cache_key
        _COAG_CACHE_VALUE = (
            _field_data(sim.dust.coagulation.stick),
            _field_data(sim.dust.coagulation.stick_ind),
            _field_data(sim.dust.coagulation.A),
            _field_data(sim.dust.coagulation.eps),
            _field_data(sim.dust.coagulation.lf_ind),
            _field_data(sim.dust.coagulation.rm_ind),
            _field_data(sim.dust.coagulation.phi),
            _field_data(sim.grid.m),
        )

    cstick, cstick_ind, A, eps, klf, krm, phi, m = _COAG_CACHE_VALUE
    Kf = _field_data(sim.dust.kernel * sim.dust.p.frag)
    Ks = _field_data(sim.dust.kernel * sim.dust.p.stick)
    SigmaArr = _field_data(Sigma)
    SigmaFloor = _field_data(sim.dust.SigmaFloor)

    Nr, Nm = SigmaArr.shape
    Nr_int = Nr - 2
    if Nr_int <= 0:
        return cp.zeros_like(SigmaArr)

    pre = _get_scoag_precomp(cstick, cstick_ind, A, eps, klf, krm, phi, m, Nm)
    i_idx = pre["i_idx"]
    j_idx = pre["j_idx"]
    krm_p = pre["krm_p"]
    eps_p = pre["eps_p"]

    S_flat = cp.zeros(Nr_int * Nm, dtype=SigmaArr.dtype)
    As_flat = cp.zeros(Nr_int * Nm, dtype=SigmaArr.dtype)
    ir_offsets = cp.arange(Nr_int, dtype=cp.int64)[:, None] * Nm

    # Active-pair mask by interior radial cell.
    active = SigmaArr[1:-1] > SigmaFloor[1:-1]
    any_active = cp.any(active, axis=1)
    imax_arr = cp.where(any_active, Nm - cp.argmax(active[:, ::-1], axis=1), 0)
    p_active = i_idx[None, :] < imax_arr[:, None]

    n_all = SigmaArr[1:-1] / m[None, :]
    Rs_all = Ks[1:-1, j_idx, i_idx] * n_all[:, j_idx] * n_all[:, i_idx]
    Rf_all = Kf[1:-1, j_idx, i_idx] * n_all[:, j_idx] * n_all[:, i_idx]
    Rs_all = Rs_all * p_active
    Rf_all = Rf_all * p_active

    if _S_COAG_MODE == "fused_spmm":
        S_mid = cp.zeros((Nr_int, Nm), dtype=SigmaArr.dtype)
        stick_mat = pre.get("stick_sel_mat_gpu")
        frag_a_mat = pre.get("frag_a_sel_mat_gpu")
        sink_mat = pre.get("frag_sink_sel_mat_gpu")
        frag_valid = pre["frag_valid"]
        phi_lower = pre["phi_lower"]
        chunk_size = _get_jcoag_chunk_size(Nr_int, Nm)
        for start in range(0, Nr_int, chunk_size):
            stop = min(start + chunk_size, Nr_int)
            S_chunk = S_mid[start:stop]
            Rs_chunk = Rs_all[start:stop]
            Rf_chunk = Rf_all[start:stop]

            if stick_mat is not None:
                S_chunk += stick_mat.dot(Rs_chunk.T).T

            if frag_a_mat is not None and int(frag_valid.size) > 0:
                ratef_v = Rf_chunk[:, frag_valid]
                As_chunk = frag_a_mat.dot(ratef_v.T).T
                S_chunk += (As_chunk @ phi_lower) / m[None, :]

            if sink_mat is not None:
                S_chunk += sink_mat.dot(Rf_chunk.T).T

        S = cp.zeros_like(SigmaArr)
        S[1:-1] = S_mid * m[None, :]
        freeze_mask = _gas_floor_freeze_mask(sim)[1:-1]
        if int(to_numpy(freeze_mask.sum())) > 0:
            freeze_rows = cp.where(freeze_mask)[0] + 1
            S[freeze_rows, :] = 0.0
        return S

    # Sticking contribution.
    for k_v, c_v, valid in pre["stick_maps"]:
        if k_v is None:
            continue
        rates = Rs_all[:, valid]
        targets = ir_offsets + k_v[None, :]
        _scatter_add_1d(S_flat, targets.ravel(), (c_v[None, :] * rates).ravel())

    # Fragment distribution contribution.
    frag_valid = pre["frag_valid"]
    if int(frag_valid.size) > 0:
        klf_v = pre["klf_v"]
        A_v = pre["A_v"]
        rates_f = Rf_all[:, frag_valid]
        targets = ir_offsets + klf_v[None, :]
        _scatter_add_1d(As_flat, targets.ravel(), (A_v[None, :] * rates_f).ravel())

    As = As_flat.reshape(Nr_int, Nm)
    S_flat += ((As @ pre["phi_lower"]) / m[None, :]).ravel()

    # Erosion case 1: krm == i - 1.
    c1 = pre["c1"]
    if int(c1.size) > 0:
        krm_c1 = krm_p[c1]
        j_c1 = j_idx[c1]
        eps_c1 = eps_p[c1]
        rate_c1 = Rf_all[:, c1]
        _scatter_add_1d(S_flat, (ir_offsets + krm_c1[None, :]).ravel(), (eps_c1[None, :] * rate_c1).ravel())
        _scatter_add_1d(S_flat, (ir_offsets + krm_c1[None, :] + 1).ravel(), (-eps_c1[None, :] * rate_c1).ravel())
        _scatter_add_1d(S_flat, (ir_offsets + j_c1[None, :]).ravel(), (-rate_c1).ravel())

    # Erosion case 2: krm != i - 1.
    c2 = pre["c2"]
    if int(c2.size) > 0:
        i_c2 = i_idx[c2]
        j_c2 = j_idx[c2]
        krm_c2 = krm_p[c2]
        eps_c2 = eps_p[c2]
        rate_c2 = Rf_all[:, c2]
        _scatter_add_1d(S_flat, (ir_offsets + krm_c2[None, :]).ravel(), (eps_c2[None, :] * rate_c2).ravel())
        _scatter_add_1d(S_flat, (ir_offsets + krm_c2[None, :] + 1).ravel(), ((1.0 - eps_c2[None, :]) * rate_c2).ravel())
        _scatter_add_1d(S_flat, (ir_offsets + i_c2[None, :]).ravel(), (-rate_c2).ravel())
        _scatter_add_1d(S_flat, (ir_offsets + j_c2[None, :]).ravel(), (-rate_c2).ravel())

    # Full fragmentation.
    ff = pre["ff"]
    if int(ff.size) > 0:
        i_ff = i_idx[ff]
        j_ff = j_idx[ff]
        rate_ff = Rf_all[:, ff]
        _scatter_add_1d(S_flat, (ir_offsets + i_ff[None, :]).ravel(), (-rate_ff).ravel())
        _scatter_add_1d(S_flat, (ir_offsets + j_ff[None, :]).ravel(), (-rate_ff).ravel())

    S = cp.zeros_like(SigmaArr)
    S[1:-1] = S_flat.reshape(Nr_int, Nm) * m[None, :]
    freeze_mask = _gas_floor_freeze_mask(sim)[1:-1]
    if int(to_numpy(freeze_mask.sum())) > 0:
        freeze_rows = cp.where(freeze_mask)[0] + 1
        S[freeze_rows, :] = 0.0
    return S


def _kernel_cupy(sim):
    global _KERNEL_LOWER_MASK
    a = _field_data(sim.dust.a)
    H = _field_data(sim.dust.H)
    Sigma = _field_data(sim.dust.Sigma)
    SigmaFloor = _field_data(sim.dust.SigmaFloor)
    vrel = _field_data(sim.dust.v.rel.tot)
    Nm = int(sim.grid.Nm)

    if _COLLISION_KERNEL_MODE == "elementwise":
        K = _kernel_cupy_elementwise(a, H, Sigma, SigmaFloor, vrel)
        if K is not None:
            return K

    if _KERNEL_LOWER_MASK is None or int(_KERNEL_LOWER_MASK.shape[0]) != Nm:
        # Fortran kernels are stored/accessed as K(ir, j, i) with j <= i active.
        # In array form this is the upper triangle along mass axes (axis1 <= axis2).
        _KERNEL_LOWER_MASK = xp.asarray(np.triu(np.ones((Nm, Nm), dtype=bool), k=0))

    eye = xp.eye(Nm)
    lower_mask = _KERNEL_LOWER_MASK
    fac = (1.0 - 0.5 * eye)[None, :, :]

    area_term = c.pi * (a[:, :, None] + a[:, None, :])**2
    height_term = xp.sqrt(2.0 * c.pi * (H[:, :, None]**2 + H[:, None, :]**2))
    Kfull = fac * area_term * vrel / height_term

    active = (Sigma > SigmaFloor)
    mask = active[:, :, None] & active[:, None, :]
    mask = mask & lower_mask[None, :, :]

    K = xp.zeros_like(vrel)
    K[1:-1, :, :] = xp.where(mask[1:-1, :, :], Kfull[1:-1, :, :], 0.0)
    return K


def _St_Epstein_StokesI_cupy(sim):
    rho = sim.dust.rhos * sim.dust.fill
    a = sim.dust.a
    mfp = sim.gas.mfp[:, None]
    Sigma = sim.gas.Sigma[:, None]
    St_ep = 0.5 * c.pi * a * rho / Sigma
    St_st1 = (2.0 / 9.0) * c.pi * a**2 * rho / (mfp * Sigma)
    return xp.where(a < 2.25 * mfp, St_ep, St_st1)


def _vrad_cupy(sim):
    St = sim.dust.St
    vr = (sim.gas.v.rad[:, None] + 2.0 * sim.dust.v.driftmax[:, None] * St) / (St**2 + 1.0)
    _apply_gas_floor_freeze_to_radial_field(sim, vr)
    return vr


def _vrel_brownian_motion_cupy(sim):
    cs = _field_data(sim.gas.cs)[:, None, None]
    T = _field_data(sim.gas.T)[:, None, None]
    m = _field_data(sim.grid.m)
    mj = m[None, :, None]
    mi = m[None, None, :]
    fac = 8.0 * c.k_B / c.pi
    v = xp.sqrt(fac * T * (mj + mi) / (mj * mi))
    return xp.minimum(v, cs)


def _vrel_azimuthal_drift_cupy(sim):
    St2p1 = sim.dust.St**2 + 1.0
    num = St2p1[:, :, None] - St2p1[:, None, :]
    den = St2p1[:, :, None] * St2p1[:, None, :]
    return xp.abs(sim.dust.v.driftmax[:, None, None] * num / den)


def _vrel_radial_drift_cupy(sim):
    vr = sim.dust.v.rad
    return xp.abs(vr[:, :, None] - vr[:, None, :])


def _vrel_vertical_settling_cupy(sim):
    om_h_st = sim.grid.OmegaK[:, None] * sim.dust.H * xp.minimum(sim.dust.St, 0.5)
    return xp.abs(om_h_st[:, :, None] - om_h_st[:, None, :])


def _vrel_turbulent_motion_cupy(sim):
    alpha = _field_data(sim.dust.delta.turb)
    cs = _field_data(sim.gas.cs)
    mump = _field_data(sim.gas.mu)
    OmegaK = _field_data(sim.grid.OmegaK)
    SigmaGas = _field_data(sim.gas.Sigma)
    St = _field_data(sim.dust.St)

    if _VREL_TURB_MODE == "elementwise":
        vrel = _vrel_turbulent_motion_cupy_elementwise(alpha, cs, mump, OmegaK, SigmaGas, St)
        if vrel is not None:
            return vrel

    c0 = 1.6015125
    c1 = -0.63119577
    c2 = 0.32938936
    c3 = -0.29847604
    ya = 1.6
    yap1inv = 1.0 / (1.0 + ya)

    OmKinv = 1.0 / (OmegaK + 1e-300)
    Re = 0.5 * alpha * SigmaGas * c.sigma_H2 / (mump + 1e-300)
    ReInvSqrt = xp.sqrt(1.0 / (Re + 1e-300))
    vn = xp.sqrt(alpha) * cs
    vs = Re**(-0.25) * vn
    ts = OmKinv * ReInvSqrt
    vg2 = 1.5 * vn**2

    StL = xp.maximum(St[:, :, None], St[:, None, :])
    StS = xp.minimum(St[:, :, None], St[:, None, :])
    eps = StS / (StL + 1e-300)

    OmKinv3 = OmKinv[:, None, None]
    ReInvSqrt3 = ReInvSqrt[:, None, None]
    ts3 = ts[:, None, None]
    vg23 = vg2[:, None, None]
    vs3 = vs[:, None, None]

    tauL = StL * OmKinv3
    tauS = StS * OmKinv3

    ys = c0 + c1 * StL + c2 * StL**2 + c3 * StL**3
    h1 = (StL - StS) / (StL + StS + 1e-300) * (
        StL * yap1inv - StS**2 / (StS + ya * StL + 1e-300)
    )
    h2 = (
        2.0 * (ya * StL - ReInvSqrt3)
        + StL * yap1inv
        - StL**2 / (StL + ReInvSqrt3 + 1e-300)
        + StS**2 / (ya * StL + StS + 1e-300)
        - StS**2 / (StS + ReInvSqrt3 + 1e-300)
    )

    val1 = 1.5 * (vs3 / (ts3 + 1e-300) * (tauL - tauS))**2
    val2 = vg23 * (StL - StS) / (StL + StS + 1e-300) * (
        StL**2 / (StL + ReInvSqrt3 + 1e-300)
        - StS**2 / (StS + ReInvSqrt3 + 1e-300)
    )
    val3 = vg23 * (h1 + h2)
    val4 = vg23 * StL * (
        2.0 * ya
        - 1.0
        - eps
        + 2.0 / (1.0 + eps + 1e-300) * (yap1inv + eps**3 / (ya + eps + 1e-300))
    )
    val5 = vg23 * StL * (
        2.0 * ys
        - 1.0
        - eps
        + 2.0 / (1.0 + eps + 1e-300)
        * (1.0 / (1.0 + ys + 1e-300) + eps**3 / (ys + eps + 1e-300))
    )
    val6 = vg23 * (2.0 + StL + StS) / (1.0 + StL + StS + StL * StS + 1e-300)

    cond1 = tauL < 0.2 * ts3
    cond2 = tauL * ya < ts3
    cond3 = tauL < 5.0 * ts3
    cond4 = tauL < 0.2 * OmKinv3
    cond5 = tauL < OmKinv3

    v2 = xp.where(
        cond1,
        val1,
        xp.where(
            cond2,
            val2,
            xp.where(cond3, val3, xp.where(cond4, val4, xp.where(cond5, val5, val6))),
        ),
    )

    return xp.sqrt(xp.maximum(v2, 0.0))


def _p_frag_cupy(sim):
    vrel = _field_data(sim.dust.v.rel.tot)
    vfrag = _field_data(sim.dust.v.frag)

    if _P_FRAG_MODE == "elementwise":
        pf = _p_frag_cupy_elementwise(vrel, vfrag)
        if pf is not None:
            return pf

    vfrag3 = vfrag[:, None, None]
    mask = vrel != 0.0
    denom = xp.where(mask, vrel, 1.0)
    dum = (vfrag3 / denom) ** 2
    pf = xp.where(mask, (1.5 * dum + 1.0) * xp.exp(-1.5 * dum), 0.0)
    pf[0, :, :] = 0.0
    pf[-1, :, :] = 0.0
    return pf
