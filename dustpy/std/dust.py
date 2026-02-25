'''Module containing standard functions for the dust.'''


import dustpy.constants as c
import math
import os
from dustpy.std import dust_f
from dustpy.utils.backend import call_numpy
from dustpy.utils.backend import bind_sparse_solver
from dustpy.utils.backend import solve_sparse_linear_system
from dustpy.utils.backend import to_numpy
from dustpy.utils.boundary_modes import is_zero_flux_enabled
from simframe.backends.api import get_backend
from simframe.backends.api import select_backend
from simframe.backends.api import xp

import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    from cupyx.scipy.interpolate import interp1d as cp_interp1d
    try:
        from cupyx import scatter_add as cp_scatter_add
    except Exception:  # pragma: no cover - optional dependency
        cp_scatter_add = None
except Exception:  # pragma: no cover - optional dependency
    cp = None
    cp_sparse = None
    cp_interp1d = None
    cp_scatter_add = None

from simframe.integration import Scheme


def _dust_f_call(func, *args, to_backend_result=True, **kwargs):
    """Call NumPy/F2PY dust kernels with backend-safe conversions."""
    return call_numpy(
        func,
        *args,
        to_backend_result=to_backend_result,
        audit_tag=f"dust_f:{func.__name__}",
        **kwargs,
    )


def _field_data(value):
    """Unwrap simframe Field-like containers to backend array data."""
    return value._data if hasattr(value, "_data") else value


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
_K_JACOBIAN = None
_K_INTERP_TO_INTERFACES = None
_KERNEL_LOWER_MASK = None
_COAG_CACHE_KEY = None
_COAG_CACHE_VALUE = None
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
_JCOAG_CONST_CACHE_KEY = None
_JCOAG_CONST_CACHE_VALUE = None
_JCOAG_PATTERN_CACHE_KEY = None
_JCOAG_PATTERN_CACHE_VALUE = None
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
_CUPY_DIAG_POS_CACHE_KEY = None
_CUPY_DIAG_POS_CACHE_VALUE = None
_JCOAG_CHUNK_SIZE_OVERRIDE = None
_RUNTIME_STATES = {}
_ACTIVE_RUNTIME_TOKEN = None
_RAW_SCATTER_KERNEL_F32 = None
_RAW_SCATTER_KERNEL_F64 = None
_F_DIFF_EW_KERNEL_F32 = None
_F_DIFF_EW_KERNEL_F64 = None
_VREL_TURB_EW_KERNEL_F32 = None
_VREL_TURB_EW_KERNEL_F64 = None
_VREL_TOT_EW_KERNEL_F32 = None
_VREL_TOT_EW_KERNEL_F64 = None
_P_FRAG_EW_KERNEL_F32 = None
_P_FRAG_EW_KERNEL_F64 = None
_COLLISION_KERNEL_EW_F32 = None
_COLLISION_KERNEL_EW_F64 = None

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
    "_K_JACOBIAN",
    "_K_INTERP_TO_INTERFACES",
    "_KERNEL_LOWER_MASK",
    "_COAG_CACHE_KEY",
    "_COAG_CACHE_VALUE",
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
    "_JCOAG_CONST_CACHE_KEY",
    "_JCOAG_CONST_CACHE_VALUE",
    "_JCOAG_PATTERN_CACHE_KEY",
    "_JCOAG_PATTERN_CACHE_VALUE",
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
    "_CUPY_DIAG_POS_CACHE_KEY",
    "_CUPY_DIAG_POS_CACHE_VALUE",
    "_JCOAG_CHUNK_SIZE_OVERRIDE",
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
        "_K_JACOBIAN": None,
        "_K_INTERP_TO_INTERFACES": None,
        "_KERNEL_LOWER_MASK": None,
        "_COAG_CACHE_KEY": None,
        "_COAG_CACHE_VALUE": None,
        "_COAG_PAIR_CACHE_KEY": None,
        "_COAG_PAIR_CACHE_VALUE": None,
        "_SCOAG_PAIRMAP_CACHE_KEY": None,
        "_SCOAG_PAIRMAP_CACHE_VALUE": None,
        "_SCOAG_PRECOMP_CACHE_KEY": None,
        "_SCOAG_PRECOMP_CACHE_VALUE": None,
        "_FRAG_P_CACHE_KEY": None,
        "_FRAG_P_CACHE_VALUE": None,
        "_MGRID_Q_CACHE_KEY": None,
        "_MGRID_Q_CACHE_VALUE": None,
        "_JCOAG_CONST_CACHE_KEY": None,
        "_JCOAG_CONST_CACHE_VALUE": None,
        "_JCOAG_PATTERN_CACHE_KEY": None,
        "_JCOAG_PATTERN_CACHE_VALUE": None,
        "_JCOAG_PATTERN_GPU_CACHE_KEY": None,
        "_JCOAG_PATTERN_GPU_CACHE_VALUE": None,
        "_JCOAG_PRECOMP_CACHE_KEY": None,
        "_JCOAG_PRECOMP_CACHE_VALUE": None,
        "_BOUNDARY_BASIS_CACHE_KEY": None,
        "_BOUNDARY_BASIS_CACHE_VALUE": None,
        "_JSTICK_MAP_CACHE_KEY": None,
        "_JSTICK_MAP_CACHE_VALUE": None,
        "_JFRAG_MAP_CACHE_KEY": None,
        "_JFRAG_MAP_CACHE_VALUE": None,
        "_JCOAG_WORK_CACHE_KEY": None,
        "_JCOAG_WORK_CACHE_VALUE": None,
        "_DUST_JHB_PATTERN_CACHE_KEY": None,
        "_DUST_JHB_PATTERN_CACHE_VALUE": None,
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
        "_CUPY_DIAG_POS_CACHE_KEY": None,
        "_CUPY_DIAG_POS_CACHE_VALUE": None,
        "_JCOAG_CHUNK_SIZE_OVERRIDE": None,
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


def _get_raw_scatter_kernel(dtype):
    """Return cached raw scatter-add kernel for float32/float64 outputs."""
    global _RAW_SCATTER_KERNEL_F32, _RAW_SCATTER_KERNEL_F64

    if cp is None:
        return None

    if dtype == cp.float32:
        if _RAW_SCATTER_KERNEL_F32 is None:
            _RAW_SCATTER_KERNEL_F32 = cp.RawKernel(
                r"""
                extern "C" __global__
                void dustpy_scatter_add_f32(float* out, const long long* idx, const float* vals, long long n) {
                    long long tid = (long long)(blockDim.x * blockIdx.x + threadIdx.x);
                    if (tid < n) {
                        atomicAdd(out + idx[tid], vals[tid]);
                    }
                }
                """,
                "dustpy_scatter_add_f32",
            )
        return _RAW_SCATTER_KERNEL_F32

    if dtype == cp.float64:
        if _RAW_SCATTER_KERNEL_F64 is None:
            _RAW_SCATTER_KERNEL_F64 = cp.RawKernel(
                r"""
                extern "C" __global__
                void dustpy_scatter_add_f64(double* out, const long long* idx, const double* vals, long long n) {
                    long long tid = (long long)(blockDim.x * blockIdx.x + threadIdx.x);
                    if (tid < n) {
                        atomicAdd(out + idx[tid], vals[tid]);
                    }
                }
                """,
                "dustpy_scatter_add_f64",
            )
        return _RAW_SCATTER_KERNEL_F64

    return None


def _get_fdiff_elementwise_kernel(dtype):
    """Return cached elementwise kernel for diffusive flux interiors."""
    global _F_DIFF_EW_KERNEL_F32, _F_DIFF_EW_KERNEL_F64

    if cp is None:
        return None

    if dtype == cp.float32:
        if _F_DIFF_EW_KERNEL_F32 is None:
            _F_DIFF_EW_KERNEL_F32 = cp.ElementwiseKernel(
                "raw float32 D, raw float32 SigmaD, raw float32 SigmaG, raw float32 St, raw float32 u, raw float32 r, raw float32 ri, int64 nm",
                "float32 out",
                r"""
                const long long iface = i / nm + 1;
                const long long jm = i - (iface - 1) * nm;
                const long long l = iface - 1;
                const long long rr = iface;
                const long long idxL = l * nm + jm;
                const long long idxR = rr * nm + jm;

                const float eps0 = 1.0e-30f;
                const float dr = r[rr] - r[l];
                const float t = (ri[iface] - r[l]) / (dr + eps0);

                const float siggi = SigmaG[l] + t * (SigmaG[rr] - SigmaG[l]);
                const float ui = u[l] + t * (u[rr] - u[l]);
                const float di = D[idxL] + t * (D[idxR] - D[idxL]);
                const float sigdi = SigmaD[idxL] + t * (SigmaD[idxR] - SigmaD[idxL]);
                const float sti = St[idxL] + t * (St[idxR] - St[idxL]);

                const float epsL = SigmaD[idxL] / (SigmaG[l] + eps0);
                const float epsR = SigmaD[idxR] / (SigmaG[rr] + eps0);
                const float gradepsi = (epsR - epsL) / (dr + eps0);

                const float w = ui * sigdi / (1.0f + sti * sti);
                const float fi0 = -di * siggi * gradepsi;

                if (fabsf(w) > 0.0f) {
                    const float P = fabsf(fi0 / w);
                    const float lam = (1.0f + P) / (1.0f + P + P * P);
                    out = lam * fi0;
                } else {
                    out = w;
                }
                """,
                "dustpy_fdiff_elementwise_f32",
            )
        return _F_DIFF_EW_KERNEL_F32

    if dtype == cp.float64:
        if _F_DIFF_EW_KERNEL_F64 is None:
            _F_DIFF_EW_KERNEL_F64 = cp.ElementwiseKernel(
                "raw float64 D, raw float64 SigmaD, raw float64 SigmaG, raw float64 St, raw float64 u, raw float64 r, raw float64 ri, int64 nm",
                "float64 out",
                r"""
                const long long iface = i / nm + 1;
                const long long jm = i - (iface - 1) * nm;
                const long long l = iface - 1;
                const long long rr = iface;
                const long long idxL = l * nm + jm;
                const long long idxR = rr * nm + jm;

                const double eps0 = 1.0e-300;
                const double dr = r[rr] - r[l];
                const double t = (ri[iface] - r[l]) / (dr + eps0);

                const double siggi = SigmaG[l] + t * (SigmaG[rr] - SigmaG[l]);
                const double ui = u[l] + t * (u[rr] - u[l]);
                const double di = D[idxL] + t * (D[idxR] - D[idxL]);
                const double sigdi = SigmaD[idxL] + t * (SigmaD[idxR] - SigmaD[idxL]);
                const double sti = St[idxL] + t * (St[idxR] - St[idxL]);

                const double epsL = SigmaD[idxL] / (SigmaG[l] + eps0);
                const double epsR = SigmaD[idxR] / (SigmaG[rr] + eps0);
                const double gradepsi = (epsR - epsL) / (dr + eps0);

                const double w = ui * sigdi / (1.0 + sti * sti);
                const double fi0 = -di * siggi * gradepsi;

                if (fabs(w) > 0.0) {
                    const double P = fabs(fi0 / w);
                    const double lam = (1.0 + P) / (1.0 + P + P * P);
                    out = lam * fi0;
                } else {
                    out = w;
                }
                """,
                "dustpy_fdiff_elementwise_f64",
            )
        return _F_DIFF_EW_KERNEL_F64

    return None


def _get_vrel_turbulent_elementwise_kernel(dtype):
    """Return cached elementwise kernel for turbulent relative velocity."""
    global _VREL_TURB_EW_KERNEL_F32, _VREL_TURB_EW_KERNEL_F64

    if cp is None:
        return None

    if dtype == cp.float32:
        if _VREL_TURB_EW_KERNEL_F32 is None:
            _VREL_TURB_EW_KERNEL_F32 = cp.ElementwiseKernel(
                "raw float32 alpha, raw float32 cs, raw float32 mump, raw float32 omegak, raw float32 sigmag, raw float32 st, int64 nm, float32 sigma_h2",
                "float32 out",
                r"""
                const long long plane = nm * nm;
                const long long ir = i / plane;
                const long long rem = i - ir * plane;
                const long long jm = rem / nm;
                const long long im = rem - jm * nm;
                const long long base = ir * nm;

                const float eps0 = 1.0e-30f;
                float Stj = st[base + jm];
                float Sti = st[base + im];
                float StL = Stj >= Sti ? Stj : Sti;
                float StS = Stj >= Sti ? Sti : Stj;

                float OmKinv = 1.0f / (omegak[ir] + eps0);
                float Re = 0.5f * alpha[ir] * sigmag[ir] * sigma_h2 / (mump[ir] + eps0);
                float ReInvSqrt = sqrtf(1.0f / (Re + eps0));
                float vn = sqrtf(alpha[ir]) * cs[ir];
                float vs = powf(Re + eps0, -0.25f) * vn;
                float ts = OmKinv * ReInvSqrt;
                float vg2 = 1.5f * vn * vn;

                const float c0 = 1.6015125f;
                const float c1 = -0.63119577f;
                const float c2 = 0.32938936f;
                const float c3 = -0.29847604f;
                const float ya = 1.6f;
                const float yap1inv = 1.0f / (1.0f + ya);

                float eps = StS / (StL + eps0);
                float tauL = StL * OmKinv;
                float tauS = StS * OmKinv;
                float StL2 = StL * StL;
                float StL3 = StL2 * StL;
                float StS2 = StS * StS;
                float eps3 = eps * eps * eps;

                float ys = c0 + c1 * StL + c2 * StL2 + c3 * StL3;
                float h1 = (StL - StS) / (StL + StS + eps0) * (StL * yap1inv - StS2 / (StS + ya * StL + eps0));
                float h2 = 2.0f * (ya * StL - ReInvSqrt)
                    + StL * yap1inv
                    - StL2 / (StL + ReInvSqrt + eps0)
                    + StS2 / (ya * StL + StS + eps0)
                    - StS2 / (StS + ReInvSqrt + eps0);

                float t = vs / (ts + eps0) * (tauL - tauS);
                float val1 = 1.5f * t * t;
                float val2 = vg2 * (StL - StS) / (StL + StS + eps0)
                    * (StL2 / (StL + ReInvSqrt + eps0) - StS2 / (StS + ReInvSqrt + eps0));
                float val3 = vg2 * (h1 + h2);
                float val4 = vg2 * StL * (2.0f * ya - 1.0f - eps
                    + 2.0f / (1.0f + eps + eps0) * (yap1inv + eps3 / (ya + eps + eps0)));
                float val5 = vg2 * StL * (2.0f * ys - 1.0f - eps
                    + 2.0f / (1.0f + eps + eps0) * (1.0f / (1.0f + ys + eps0) + eps3 / (ys + eps + eps0)));
                float val6 = vg2 * (2.0f + StL + StS) / (1.0f + StL + StS + StL * StS + eps0);

                bool cond1 = tauL < 0.2f * ts;
                bool cond2 = tauL * ya < ts;
                bool cond3 = tauL < 5.0f * ts;
                bool cond4 = tauL < 0.2f * OmKinv;
                bool cond5 = tauL < OmKinv;

                float v2;
                if (cond1) {
                    v2 = val1;
                } else if (cond2) {
                    v2 = val2;
                } else if (cond3) {
                    v2 = val3;
                } else if (cond4) {
                    v2 = val4;
                } else if (cond5) {
                    v2 = val5;
                } else {
                    v2 = val6;
                }
                out = sqrtf(v2 > 0.0f ? v2 : 0.0f);
                """,
                "dustpy_vrel_turbulent_elementwise_f32",
            )
        return _VREL_TURB_EW_KERNEL_F32

    if dtype == cp.float64:
        if _VREL_TURB_EW_KERNEL_F64 is None:
            _VREL_TURB_EW_KERNEL_F64 = cp.ElementwiseKernel(
                "raw float64 alpha, raw float64 cs, raw float64 mump, raw float64 omegak, raw float64 sigmag, raw float64 st, int64 nm, float64 sigma_h2",
                "float64 out",
                r"""
                const long long plane = nm * nm;
                const long long ir = i / plane;
                const long long rem = i - ir * plane;
                const long long jm = rem / nm;
                const long long im = rem - jm * nm;
                const long long base = ir * nm;

                const double eps0 = 1.0e-300;
                double Stj = st[base + jm];
                double Sti = st[base + im];
                double StL = Stj >= Sti ? Stj : Sti;
                double StS = Stj >= Sti ? Sti : Stj;

                double OmKinv = 1.0 / (omegak[ir] + eps0);
                double Re = 0.5 * alpha[ir] * sigmag[ir] * sigma_h2 / (mump[ir] + eps0);
                double ReInvSqrt = sqrt(1.0 / (Re + eps0));
                double vn = sqrt(alpha[ir]) * cs[ir];
                double vs = pow(Re + eps0, -0.25) * vn;
                double ts = OmKinv * ReInvSqrt;
                double vg2 = 1.5 * vn * vn;

                const double c0 = 1.6015125;
                const double c1 = -0.63119577;
                const double c2 = 0.32938936;
                const double c3 = -0.29847604;
                const double ya = 1.6;
                const double yap1inv = 1.0 / (1.0 + ya);

                double eps = StS / (StL + eps0);
                double tauL = StL * OmKinv;
                double tauS = StS * OmKinv;
                double StL2 = StL * StL;
                double StL3 = StL2 * StL;
                double StS2 = StS * StS;
                double eps3 = eps * eps * eps;

                double ys = c0 + c1 * StL + c2 * StL2 + c3 * StL3;
                double h1 = (StL - StS) / (StL + StS + eps0) * (StL * yap1inv - StS2 / (StS + ya * StL + eps0));
                double h2 = 2.0 * (ya * StL - ReInvSqrt)
                    + StL * yap1inv
                    - StL2 / (StL + ReInvSqrt + eps0)
                    + StS2 / (ya * StL + StS + eps0)
                    - StS2 / (StS + ReInvSqrt + eps0);

                double t = vs / (ts + eps0) * (tauL - tauS);
                double val1 = 1.5 * t * t;
                double val2 = vg2 * (StL - StS) / (StL + StS + eps0)
                    * (StL2 / (StL + ReInvSqrt + eps0) - StS2 / (StS + ReInvSqrt + eps0));
                double val3 = vg2 * (h1 + h2);
                double val4 = vg2 * StL * (2.0 * ya - 1.0 - eps
                    + 2.0 / (1.0 + eps + eps0) * (yap1inv + eps3 / (ya + eps + eps0)));
                double val5 = vg2 * StL * (2.0 * ys - 1.0 - eps
                    + 2.0 / (1.0 + eps + eps0) * (1.0 / (1.0 + ys + eps0) + eps3 / (ys + eps + eps0)));
                double val6 = vg2 * (2.0 + StL + StS) / (1.0 + StL + StS + StL * StS + eps0);

                bool cond1 = tauL < 0.2 * ts;
                bool cond2 = tauL * ya < ts;
                bool cond3 = tauL < 5.0 * ts;
                bool cond4 = tauL < 0.2 * OmKinv;
                bool cond5 = tauL < OmKinv;

                double v2;
                if (cond1) {
                    v2 = val1;
                } else if (cond2) {
                    v2 = val2;
                } else if (cond3) {
                    v2 = val3;
                } else if (cond4) {
                    v2 = val4;
                } else if (cond5) {
                    v2 = val5;
                } else {
                    v2 = val6;
                }
                out = sqrt(v2 > 0.0 ? v2 : 0.0);
                """,
                "dustpy_vrel_turbulent_elementwise_f64",
            )
        return _VREL_TURB_EW_KERNEL_F64

    return None


def _get_vrel_tot_elementwise_kernel(dtype):
    """Return cached elementwise kernel for total relative velocity."""
    global _VREL_TOT_EW_KERNEL_F32, _VREL_TOT_EW_KERNEL_F64

    if cp is None:
        return None

    if dtype == cp.float32:
        if _VREL_TOT_EW_KERNEL_F32 is None:
            _VREL_TOT_EW_KERNEL_F32 = cp.ElementwiseKernel(
                "float32 az, float32 br, float32 rd, float32 tu, float32 vt",
                "float32 out",
                "out = sqrtf(az * az + br * br + rd * rd + tu * tu + vt * vt)",
                "dustpy_vrel_tot_elementwise_f32",
            )
        return _VREL_TOT_EW_KERNEL_F32

    if dtype == cp.float64:
        if _VREL_TOT_EW_KERNEL_F64 is None:
            _VREL_TOT_EW_KERNEL_F64 = cp.ElementwiseKernel(
                "float64 az, float64 br, float64 rd, float64 tu, float64 vt",
                "float64 out",
                "out = sqrt(az * az + br * br + rd * rd + tu * tu + vt * vt)",
                "dustpy_vrel_tot_elementwise_f64",
            )
        return _VREL_TOT_EW_KERNEL_F64

    return None


def _get_pfrag_elementwise_kernel(dtype):
    """Return cached elementwise kernel for fragmentation probability."""
    global _P_FRAG_EW_KERNEL_F32, _P_FRAG_EW_KERNEL_F64

    if cp is None:
        return None

    if dtype == cp.float32:
        if _P_FRAG_EW_KERNEL_F32 is None:
            _P_FRAG_EW_KERNEL_F32 = cp.ElementwiseKernel(
                "raw float32 vrel, raw float32 vfrag, int64 nm",
                "float32 out",
                r"""
                const long long nm2 = nm * nm;
                const long long ir_int = i / nm2;
                const float v = vrel[i];
                const float vf = vfrag[ir_int];
                if (v != 0.0f) {
                    const float dum = (vf / v) * (vf / v);
                    out = (1.5f * dum + 1.0f) * expf(-1.5f * dum);
                } else {
                    out = 0.0f;
                }
                """,
                "dustpy_pfrag_elementwise_f32",
            )
        return _P_FRAG_EW_KERNEL_F32

    if dtype == cp.float64:
        if _P_FRAG_EW_KERNEL_F64 is None:
            _P_FRAG_EW_KERNEL_F64 = cp.ElementwiseKernel(
                "raw float64 vrel, raw float64 vfrag, int64 nm",
                "float64 out",
                r"""
                const long long nm2 = nm * nm;
                const long long ir_int = i / nm2;
                const double v = vrel[i];
                const double vf = vfrag[ir_int];
                if (v != 0.0) {
                    const double dum = (vf / v) * (vf / v);
                    out = (1.5 * dum + 1.0) * exp(-1.5 * dum);
                } else {
                    out = 0.0;
                }
                """,
                "dustpy_pfrag_elementwise_f64",
            )
        return _P_FRAG_EW_KERNEL_F64

    return None


def _get_collision_kernel_elementwise_kernel(dtype):
    """Return cached elementwise kernel for collision-kernel assembly."""
    global _COLLISION_KERNEL_EW_F32, _COLLISION_KERNEL_EW_F64

    if cp is None:
        return None

    if dtype == cp.float32:
        if _COLLISION_KERNEL_EW_F32 is None:
            _COLLISION_KERNEL_EW_F32 = cp.ElementwiseKernel(
                "raw float32 a, raw float32 h, raw float32 sigma, raw float32 floor, raw float32 vrel, int64 nm",
                "float32 out",
                r"""
                const long long nm2 = nm * nm;
                const long long ir_int = i / nm2;
                const long long rem = i - ir_int * nm2;
                const long long jm = rem / nm;
                const long long im = rem - jm * nm;

                if (jm > im) {
                    out = 0.0f;
                    return;
                }

                const long long base = ir_int * nm;
                if (sigma[base + jm] <= floor[base + jm] || sigma[base + im] <= floor[base + im]) {
                    out = 0.0f;
                    return;
                }

                const float pi = 3.14159265358979323846f;
                const float sum_a = a[base + jm] + a[base + im];
                const float cross = pi * sum_a * sum_a;
                const float hj = h[base + jm];
                const float hi = h[base + im];
                const float sh = sqrtf(2.0f * pi * (hj * hj + hi * hi));
                const float diag_fac = (jm == im) ? 0.5f : 1.0f;
                out = diag_fac * cross * vrel[i] / sh;
                """,
                "dustpy_collision_kernel_elementwise_f32",
            )
        return _COLLISION_KERNEL_EW_F32

    if dtype == cp.float64:
        if _COLLISION_KERNEL_EW_F64 is None:
            _COLLISION_KERNEL_EW_F64 = cp.ElementwiseKernel(
                "raw float64 a, raw float64 h, raw float64 sigma, raw float64 floor, raw float64 vrel, int64 nm",
                "float64 out",
                r"""
                const long long nm2 = nm * nm;
                const long long ir_int = i / nm2;
                const long long rem = i - ir_int * nm2;
                const long long jm = rem / nm;
                const long long im = rem - jm * nm;

                if (jm > im) {
                    out = 0.0;
                    return;
                }

                const long long base = ir_int * nm;
                if (sigma[base + jm] <= floor[base + jm] || sigma[base + im] <= floor[base + im]) {
                    out = 0.0;
                    return;
                }

                const double pi = 3.14159265358979323846;
                const double sum_a = a[base + jm] + a[base + im];
                const double cross = pi * sum_a * sum_a;
                const double hj = h[base + jm];
                const double hi = h[base + im];
                const double sh = sqrt(2.0 * pi * (hj * hj + hi * hi));
                const double diag_fac = (jm == im) ? 0.5 : 1.0;
                out = diag_fac * cross * vrel[i] / sh;
                """,
                "dustpy_collision_kernel_elementwise_f64",
            )
        return _COLLISION_KERNEL_EW_F64

    return None


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


def _env_bool(name, default=False):
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


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
    if cp is None or cp_sparse is None:
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


def _a_fortran(sim):
    rho = sim.dust.fill * sim.dust.rhos
    return _dust_f_call(dust_f.a, sim.grid.m, rho)


def _a_cupy(sim):
    rho = sim.dust.fill * sim.dust.rhos
    m = xp.asarray(sim.grid.m)[None, :]
    return (3.0 * m / (4.0 * c.pi * rho)) ** (1.0 / 3.0)


def _D_fortran(sim):
    v2 = sim.dust.delta.rad * sim.gas.cs**2
    Diff = _dust_f_call(dust_f.d, v2, sim.grid.OmegaK, sim.dust.St)
    Diff[:2, ...] = 0.
    Diff[-2:, ...] = 0.
    return Diff


def _D_cupy(sim):
    v2 = sim.dust.delta.rad * sim.gas.cs**2
    Diff = v2[:, None] / (sim.grid.OmegaK[:, None] * (1.0 + sim.dust.St**2))
    Diff[:2, ...] = 0.
    Diff[-2:, ...] = 0.
    return Diff


def _H_fortran(sim):
    return _dust_f_call(dust_f.h_dubrulle1995, sim.gas.Hp, sim.dust.St, sim.dust.delta.vert)


def _H_cupy(sim):
    Hp = sim.gas.Hp[:, None]
    H = Hp / xp.sqrt(1.0 + sim.dust.St / sim.dust.delta.vert[:, None])
    return xp.minimum(H, Hp)


def _F_adv_fortran(sim, Sigma=None):
    Sigma = Sigma if Sigma is not None else sim.dust.Sigma
    return _dust_f_call(dust_f.fi_adv, Sigma, sim.dust.v.rad, sim.grid.r, sim.grid.ri)


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
    return Fi


def _interp_to_interfaces_numpy(values, r, ri):
    """NumPy/xp linear interpolation to interfaces with endpoint extrapolation."""
    values = _field_data(values)
    r = _field_data(r)
    ri = _field_data(ri)

    if values.ndim == 1:
        out = xp.zeros((values.shape[0] + 1,), dtype=values.dtype)
        t = (ri[1:-1] - r[:-1]) / (r[1:] - r[:-1])
        out[1:-1] = values[:-1] + t * (values[1:] - values[:-1])
        m0 = (values[1] - values[0]) / (r[1] - r[0])
        m1 = (values[-1] - values[-2]) / (r[-1] - r[-2])
        out[0] = values[0] + m0 * (ri[0] - r[0])
        out[-1] = values[-1] + m1 * (ri[-1] - r[-1])
        return out

    out = xp.zeros((values.shape[0] + 1, values.shape[1]), dtype=values.dtype)
    t = ((ri[1:-1] - r[:-1]) / (r[1:] - r[:-1]))[:, None]
    out[1:-1, :] = values[:-1, :] + t * (values[1:, :] - values[:-1, :])
    m0 = (values[1, :] - values[0, :]) / (r[1] - r[0])
    m1 = (values[-1, :] - values[-2, :]) / (r[-1] - r[-2])
    out[0, :] = values[0, :] + m0 * (ri[0] - r[0])
    out[-1, :] = values[-1, :] + m1 * (ri[-1] - r[-1])
    return out


def _interp_to_interfaces_cupy(values, r, ri):
    values = _field_data(values)
    r = _field_data(r)
    ri = _field_data(ri)
    if cp is None or cp_interp1d is None:
        return _interp_to_interfaces_numpy(values, r, ri)
    f = cp_interp1d(
        cp.asarray(r),
        cp.asarray(values),
        kind="linear",
        axis=0,
        bounds_error=False,
        fill_value="extrapolate",
    )
    return f(cp.asarray(ri))


def _interp_to_interfaces(values, r, ri):
    return _K_INTERP_TO_INTERFACES(values, r, ri)


def _F_diff_fortran(sim, Sigma=None):
    if Sigma is None:
        Sigma = sim.dust.Sigma
    Fi = _dust_f_call(
        dust_f.fi_diff,
        sim.dust.D,
        Sigma,
        sim.gas.Sigma,
        sim.dust.St,
        xp.sqrt(sim.dust.delta.rad * sim.gas.cs**2),
        sim.grid.r,
        sim.grid.ri,
    )
    Fi[:1, :] = 0.
    Fi[-1:, :] = 0.
    return Fi


def _F_diff_cupy_elementwise(D, SigmaD, SigmaG, St, u, r, ri):
    """Elementwise-kernel variant to reduce temporary interface arrays."""
    dtype = cp.result_type(D.dtype, SigmaD.dtype, SigmaG.dtype, St.dtype, u.dtype, r.dtype, ri.dtype)
    if dtype not in (cp.float32, cp.float64):
        return None

    Nr = int(SigmaD.shape[0])
    Nm = int(SigmaD.shape[1])
    if Nr < 2 or Nm < 1:
        return cp.zeros((Nr + 1, Nm), dtype=dtype)

    kernel = _get_fdiff_elementwise_kernel(dtype)
    if kernel is None:
        return None

    Dv = cp.asarray(D, dtype=dtype).reshape(-1)
    SDv = cp.asarray(SigmaD, dtype=dtype).reshape(-1)
    SGv = cp.asarray(SigmaG, dtype=dtype).reshape(-1)
    Stv = cp.asarray(St, dtype=dtype).reshape(-1)
    uv = cp.asarray(u, dtype=dtype).reshape(-1)
    rv = cp.asarray(r, dtype=dtype).reshape(-1)
    riv = cp.asarray(ri, dtype=dtype).reshape(-1)

    size = int((Nr - 1) * Nm)
    Fi = cp.zeros((Nr + 1, Nm), dtype=dtype)
    Fi[1:-1, :] = kernel(Dv, SDv, SGv, Stv, uv, rv, riv, np.int64(Nm), size=size).reshape(Nr - 1, Nm)
    Fi[0, :] = 0.0
    Fi[-1, :] = 0.0
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
            return Fi

    SigGi = _interp_to_interfaces(SigmaG, r, ri)
    ui = _interp_to_interfaces(u, r, ri)
    Di = _interp_to_interfaces(D, r, ri)
    SigDi = _interp_to_interfaces(SigmaD, r, ri)
    Sti = _interp_to_interfaces(St, r, ri)

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
    return Fi


def _S_hyd_fortran(sim, Sigma=None):
    if Sigma is None:
        Sigma = sim.dust.Sigma
        Fi = sim.dust.Fi.tot
    else:
        Fi = sim.dust.Fi.tot.updater.beat(sim, Sigma=Sigma)
        if Fi is None:
            Fi = sim.dust.Fi.tot
    return _dust_f_call(dust_f.s_hyd, Fi, sim.grid.ri)


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


def _S_coag_fortran(sim, Sigma=None):
    if Sigma is None:
        Sigma = sim.dust.Sigma
    return _dust_f_call(
        dust_f.s_coag,
        sim.dust.coagulation.stick,
        sim.dust.coagulation.stick_ind,
        sim.dust.coagulation.A,
        sim.dust.coagulation.eps,
        sim.dust.coagulation.lf_ind,
        sim.dust.coagulation.rm_ind,
        sim.dust.coagulation.phi,
        sim.dust.kernel * sim.dust.p.frag,
        sim.dust.kernel * sim.dust.p.stick,
        sim.grid.m,
        Sigma,
        sim.dust.SigmaFloor,
    )


def _get_coag_pair_indices(Nm):
    """Cache lower-triangle (j <= i) pair indices for coagulation scatters."""
    global _COAG_PAIR_CACHE_KEY, _COAG_PAIR_CACHE_VALUE
    if _COAG_PAIR_CACHE_KEY != Nm or _COAG_PAIR_CACHE_VALUE is None:
        j_np, i_np = np.tril_indices(Nm)
        if cp is None:
            _COAG_PAIR_CACHE_VALUE = (j_np, i_np)
        else:
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

    if cp is not None and isinstance(m, cp.ndarray):
        agrid = float((cp.log10(m[0] / m[-1]) / (1.0 - Nm)).item())
    else:
        agrid = math.log10(float(m[0]) / float(m[-1])) / (1.0 - Nm)
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
            stick_mat_cpu = sp.coo_matrix(
                (stick_vals_np, (stick_rows_np, stick_cols_np)),
                shape=(int(Nm), n_pairs),
            ).tocsr()
            stick_sel_mat_gpu = cp_sparse.csr_matrix(stick_mat_cpu)

        frag_valid_np = np.where(klf_p_np >= 0)[0].astype(np.int64, copy=False)
        if int(frag_valid_np.size) > 0:
            klf_v_np = klf_p_np[frag_valid_np].astype(np.int64, copy=False)
            A_v_np = A_p_np[frag_valid_np]
            frag_a_mat_cpu = sp.coo_matrix(
                (A_v_np, (klf_v_np, np.arange(int(frag_valid_np.size), dtype=np.int64))),
                shape=(int(Nm), int(frag_valid_np.size)),
            ).tocsr()
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
    if cp is not None and isinstance(active, cp.ndarray):
        any_active = cp.any(active, axis=1)
        last_from_end = cp.argmax(active[:, ::-1], axis=1)
        imax = cp.where(any_active, Sigma.shape[1] - last_from_end, 0).astype(cp.int64)
        return cp.asnumpy(imax)
    any_active = np.any(active, axis=1)
    last_from_end = np.argmax(active[:, ::-1], axis=1)
    return np.where(any_active, Sigma.shape[1] - last_from_end, 0).astype(np.int64)


def _S_coag_cupy(sim, Sigma=None):
    if cp is None:
        return _S_coag_fortran(sim, Sigma=Sigma)

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
            cp.asarray(_field_data(sim.dust.coagulation.stick)),
            cp.asarray(_field_data(sim.dust.coagulation.stick_ind)),
            cp.asarray(_field_data(sim.dust.coagulation.A)),
            cp.asarray(_field_data(sim.dust.coagulation.eps)),
            cp.asarray(_field_data(sim.dust.coagulation.lf_ind)),
            cp.asarray(_field_data(sim.dust.coagulation.rm_ind)),
            cp.asarray(_field_data(sim.dust.coagulation.phi)),
            cp.asarray(_field_data(sim.grid.m)),
        )

    cstick, cstick_ind, A, eps, klf, krm, phi, m = _COAG_CACHE_VALUE
    Kf = cp.asarray(_field_data(sim.dust.kernel * sim.dust.p.frag))
    Ks = cp.asarray(_field_data(sim.dust.kernel * sim.dust.p.stick))
    SigmaArr = cp.asarray(_field_data(Sigma))
    SigmaFloor = cp.asarray(_field_data(sim.dust.SigmaFloor))

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
    return S


def _get_jcoag_const_numpy(sim):
    """Return cached NumPy constants for jacobian_coagulation_generator."""
    global _JCOAG_CONST_CACHE_KEY, _JCOAG_CONST_CACHE_VALUE

    key = (
        id(_field_data(sim.dust.coagulation.A)),
        id(_field_data(sim.dust.coagulation.stick)),
        id(_field_data(sim.dust.coagulation.eps)),
        id(_field_data(sim.dust.coagulation.lf_ind)),
        id(_field_data(sim.dust.coagulation.rm_ind)),
        id(_field_data(sim.dust.coagulation.stick_ind)),
        id(_field_data(sim.grid.m)),
        id(_field_data(sim.dust.coagulation.phi)),
        id(_field_data(sim.dust.SigmaFloor)),
    )
    if _JCOAG_CONST_CACHE_KEY != key or _JCOAG_CONST_CACHE_VALUE is None:
        _JCOAG_CONST_CACHE_KEY = key
        _JCOAG_CONST_CACHE_VALUE = (
            to_numpy(_field_data(sim.dust.coagulation.A)),
            to_numpy(_field_data(sim.dust.coagulation.stick)),
            to_numpy(_field_data(sim.dust.coagulation.eps)),
            to_numpy(_field_data(sim.dust.coagulation.lf_ind)),
            to_numpy(_field_data(sim.dust.coagulation.rm_ind)),
            to_numpy(_field_data(sim.dust.coagulation.stick_ind)),
            to_numpy(_field_data(sim.grid.m)),
            to_numpy(_field_data(sim.dust.coagulation.phi)),
            to_numpy(_field_data(sim.dust.SigmaFloor)),
        )
    return _JCOAG_CONST_CACHE_VALUE


def _get_jcoag_pattern(Nr, Nm, q):
    """Return cached sparse-pattern helpers for coagulation Jacobian packing."""
    global _JCOAG_PATTERN_CACHE_KEY, _JCOAG_PATTERN_CACHE_VALUE
    key = (int(Nr), int(Nm), int(q))
    if _JCOAG_PATTERN_CACHE_KEY == key and _JCOAG_PATTERN_CACHE_VALUE is not None:
        return _JCOAG_PATTERN_CACHE_VALUE

    row_local = []
    col_local = []
    src_i = []
    src_j = []
    for i in range(Nm):
        s = max(i - q, 0)
        for j in range(s, Nm):
            row_local.append(i)
            col_local.append(j)
            src_i.append(i)
            src_j.append(j)

    row_local = np.asarray(row_local, dtype=np.int64)
    col_local = np.asarray(col_local, dtype=np.int64)
    src_i = np.asarray(src_i, dtype=np.int64)
    src_j = np.asarray(src_j, dtype=np.int64)
    L = int(src_i.shape[0])

    starts = (np.arange(1, Nr - 1, dtype=np.int64) * Nm)
    row = (starts[:, None] + row_local[None, :]).reshape(-1)
    col = (starts[:, None] + col_local[None, :]).reshape(-1)

    _JCOAG_PATTERN_CACHE_KEY = key
    _JCOAG_PATTERN_CACHE_VALUE = (src_i, src_j, row, col, L)
    return _JCOAG_PATTERN_CACHE_VALUE


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
        frag_mat_cpu = sp.coo_matrix(
            (frag_vals, (frag_rows, frag_cols)),
            shape=(int(Nm * Nm), n_frag),
        ).tocsr()
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
        ff_vals = np.concatenate(
            (
                -np.ones(ff_n, dtype=np.float64),
                -D_cpu[ff_j, ff_i],
            )
        )
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
        frag_all_mat_cpu = sp.coo_matrix(
            (frag_all_vals, (frag_all_rows, frag_all_cols)),
            shape=(int(Nm * Nm), n_pairs),
        ).tocsr()
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


def _jacobian_coagulation_generator_cupy(
    A, cStick, eps, iLF, iRM, iStick, m, phi, Rf, Rs, Sigma, SigmaFloor
):
    """CuPy-native equivalent of dust_f.jacobian_coagulation_generator."""
    A = cp.asarray(_field_data(A))
    cStick = cp.asarray(_field_data(cStick))
    eps = cp.asarray(_field_data(eps))
    iLF = cp.asarray(_field_data(iLF))
    iRM = cp.asarray(_field_data(iRM))
    iStick = cp.asarray(_field_data(iStick))
    m = cp.asarray(_field_data(m))
    phi = cp.asarray(_field_data(phi))
    Rf = cp.asarray(_field_data(Rf))
    Rs = cp.asarray(_field_data(Rs))
    Sigma = cp.asarray(_field_data(Sigma))
    SigmaFloor = cp.asarray(_field_data(SigmaFloor))

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


def _jacobian_hydrodynamic_generator_cupy(area, D, r, ri, SigmaGas, v):
    """CuPy-native equivalent of dust_f.jacobian_hydrodynamic_generator."""
    area = _field_data(area)
    D = _field_data(D)
    r = _field_data(r)
    ri = _field_data(ri)
    SigmaGas = _field_data(SigmaGas)
    v = _field_data(v)

    Nr = int(r.shape[0])
    Nm = int(D.shape[1])

    h = SigmaGas * r
    hi = _interp_to_interfaces(h, r, ri)
    vi = _interp_to_interfaces(v, r, ri)
    Di = _interp_to_interfaces(D, r, ri)

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


def _apply_zero_flux_dust_hyd_edges_numpy(A, B, C, area, D, r, ri, SigmaGas, v):
    """Inject conservative zero-flux boundary rows into dust hydrodynamic Jacobian."""
    Nr = int(A.shape[0])
    if Nr < 2:
        return A, B, C

    area = np.asarray(area)
    D = np.asarray(D)
    r = np.asarray(r)
    ri = np.asarray(ri)
    SigmaGas = np.asarray(SigmaGas)
    v = np.asarray(v)

    h = SigmaGas * r
    hi = np.asarray(_interp_to_interfaces_numpy(h, r, ri))
    Di = np.asarray(_interp_to_interfaces_numpy(D, r, ri))
    vi = np.asarray(_interp_to_interfaces_numpy(v, r, ri))
    vip = np.maximum(vi, 0.0)
    vim = np.minimum(vi, 0.0)
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


def _apply_zero_flux_dust_hyd_edges_cupy(A, B, C, area, D, r, ri, SigmaGas, v):
    """Inject conservative zero-flux boundary rows into dust hydrodynamic Jacobian."""
    Nr = int(A.shape[0])
    if Nr < 2:
        return A, B, C

    area = _field_data(area)
    D = _field_data(D)
    r = _field_data(r)
    ri = _field_data(ri)
    SigmaGas = _field_data(SigmaGas)
    v = _field_data(v)

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


def _kernel_fortran(sim):
    return _dust_f_call(
        dust_f.kernel,
        sim.dust.a,
        sim.dust.H,
        sim.dust.Sigma,
        sim.dust.SigmaFloor,
        sim.dust.v.rel.tot,
    )


def _kernel_cupy_elementwise(a, H, Sigma, SigmaFloor, vrel):
    """Elementwise-kernel variant for collision-kernel assembly."""
    dtype = cp.result_type(a.dtype, H.dtype, Sigma.dtype, SigmaFloor.dtype, vrel.dtype)
    if dtype not in (cp.float32, cp.float64):
        return None

    kernel = _get_collision_kernel_elementwise_kernel(dtype)
    if kernel is None:
        return None

    Nr = int(a.shape[0])
    Nm = int(a.shape[1])
    Nr_int = Nr - 2
    size = int(Nr_int * Nm * Nm)
    if size <= 0:
        return cp.zeros((Nr, Nm, Nm), dtype=dtype)

    a_int = cp.asarray(a[1:-1], dtype=dtype).reshape(-1)
    h_int = cp.asarray(H[1:-1], dtype=dtype).reshape(-1)
    sigma_int = cp.asarray(Sigma[1:-1], dtype=dtype).reshape(-1)
    floor_int = cp.asarray(SigmaFloor[1:-1], dtype=dtype).reshape(-1)
    vrel_int = cp.asarray(vrel[1:-1], dtype=dtype).reshape(-1)

    out_int = kernel(
        a_int,
        h_int,
        sigma_int,
        floor_int,
        vrel_int,
        np.int64(Nm),
        size=size,
    )

    K = cp.zeros((Nr, Nm, Nm), dtype=dtype)
    K[1:-1, :, :] = out_int.reshape(Nr_int, Nm, Nm)
    return K


def _kernel_cupy(sim):
    global _KERNEL_LOWER_MASK
    a = _field_data(sim.dust.a)
    H = _field_data(sim.dust.H)
    Sigma = _field_data(sim.dust.Sigma)
    SigmaFloor = _field_data(sim.dust.SigmaFloor)
    vrel = _field_data(sim.dust.v.rel.tot)
    Nm = int(sim.grid.Nm)

    if _COLLISION_KERNEL_MODE == "elementwise" and cp is not None:
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


def _St_Epstein_StokesI_fortran(sim):
    rho = sim.dust.rhos * sim.dust.fill
    return _dust_f_call(dust_f.st_epstein_stokes1, sim.dust.a, sim.gas.mfp, rho, sim.gas.Sigma)


def _St_Epstein_StokesI_cupy(sim):
    rho = sim.dust.rhos * sim.dust.fill
    a = sim.dust.a
    mfp = sim.gas.mfp[:, None]
    Sigma = sim.gas.Sigma[:, None]
    St_ep = 0.5 * c.pi * a * rho / Sigma
    St_st1 = (2.0 / 9.0) * c.pi * a**2 * rho / (mfp * Sigma)
    return xp.where(a < 2.25 * mfp, St_ep, St_st1)


def _vrad_fortran(sim):
    return _dust_f_call(dust_f.vrad, sim.dust.St, sim.dust.v.driftmax, sim.gas.v.rad)


def _vrad_cupy(sim):
    St = sim.dust.St
    return (sim.gas.v.rad[:, None] + 2.0 * sim.dust.v.driftmax[:, None] * St) / (St**2 + 1.0)


def _vrel_brownian_motion_fortran(sim):
    return _dust_f_call(dust_f.vrel_brownian_motion, sim.gas.cs, sim.grid.m, sim.gas.T)


def _vrel_brownian_motion_cupy(sim):
    cs = _field_data(sim.gas.cs)[:, None, None]
    T = _field_data(sim.gas.T)[:, None, None]
    m = _field_data(sim.grid.m)
    mj = m[None, :, None]
    mi = m[None, None, :]
    fac = 8.0 * c.k_B / c.pi
    v = xp.sqrt(fac * T * (mj + mi) / (mj * mi))
    return xp.minimum(v, cs)


def _vrel_azimuthal_drift_fortran(sim):
    return _dust_f_call(dust_f.vrel_azimuthal_drift, sim.dust.v.driftmax, sim.dust.St)


def _vrel_azimuthal_drift_cupy(sim):
    St2p1 = sim.dust.St**2 + 1.0
    num = St2p1[:, :, None] - St2p1[:, None, :]
    den = St2p1[:, :, None] * St2p1[:, None, :]
    return xp.abs(sim.dust.v.driftmax[:, None, None] * num / den)


def _vrel_radial_drift_fortran(sim):
    return _dust_f_call(dust_f.vrel_radial_drift, sim.dust.v.rad)


def _vrel_radial_drift_cupy(sim):
    vr = sim.dust.v.rad
    return xp.abs(vr[:, :, None] - vr[:, None, :])


def _vrel_vertical_settling_fortran(sim):
    return _dust_f_call(dust_f.vrel_vertical_settling, sim.dust.H, sim.grid.OmegaK, sim.dust.St)


def _vrel_vertical_settling_cupy(sim):
    om_h_st = sim.grid.OmegaK[:, None] * sim.dust.H * xp.minimum(sim.dust.St, 0.5)
    return xp.abs(om_h_st[:, :, None] - om_h_st[:, None, :])


def _vrel_turbulent_motion_fortran(sim):
    return _dust_f_call(
        dust_f.vrel_ormel_cuzzi_2007,
        sim.dust.delta.turb,
        sim.gas.cs,
        sim.gas.mu,
        sim.grid.OmegaK,
        sim.gas.Sigma,
        sim.dust.St,
    )


def _vrel_turbulent_motion_cupy_elementwise(alpha, cs, mump, OmegaK, SigmaGas, St):
    """Elementwise-kernel variant to reduce temporary 3D allocations."""
    dtype = cp.result_type(alpha.dtype, cs.dtype, mump.dtype, OmegaK.dtype, SigmaGas.dtype, St.dtype)
    if dtype not in (cp.float32, cp.float64):
        return None

    kernel = _get_vrel_turbulent_elementwise_kernel(dtype)
    if kernel is None:
        return None

    Nr = int(St.shape[0])
    Nm = int(St.shape[1])
    size = int(Nr * Nm * Nm)
    if size <= 0:
        return cp.zeros((Nr, Nm, Nm), dtype=dtype)

    alpha_arr = cp.asarray(alpha, dtype=dtype)
    cs_arr = cp.asarray(cs, dtype=dtype)
    mump_arr = cp.asarray(mump, dtype=dtype)
    omk_arr = cp.asarray(OmegaK, dtype=dtype)
    sig_arr = cp.asarray(SigmaGas, dtype=dtype)
    st_arr = cp.asarray(St, dtype=dtype).reshape(-1)

    out = kernel(
        alpha_arr,
        cs_arr,
        mump_arr,
        omk_arr,
        sig_arr,
        st_arr,
        np.int64(Nm),
        dtype.type(c.sigma_H2),
        size=size,
    )
    return out.reshape(Nr, Nm, Nm)


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


def _p_frag_fortran(sim):
    return _dust_f_call(dust_f.pfrag, sim.dust.v.rel.tot, sim.dust.v.frag)


def _p_frag_cupy_elementwise(vrel, vfrag):
    """Elementwise-kernel variant of fragmentation probability."""
    dtype = cp.result_type(vrel.dtype, vfrag.dtype)
    if dtype not in (cp.float32, cp.float64):
        return None

    kernel = _get_pfrag_elementwise_kernel(dtype)
    if kernel is None:
        return None

    Nr = int(vrel.shape[0])
    Nm = int(vrel.shape[1])
    Nr_int = Nr - 2
    size = int(Nr_int * Nm * Nm)
    if size <= 0:
        return cp.zeros_like(vrel, dtype=dtype)

    vrel_int = cp.asarray(vrel[1:-1], dtype=dtype).reshape(-1)
    vfrag_int = cp.asarray(vfrag[1:-1], dtype=dtype)
    out_int = kernel(vrel_int, vfrag_int, np.int64(Nm), size=size)

    pf = cp.zeros_like(vrel, dtype=dtype)
    pf[1:-1, :, :] = out_int.reshape(Nr_int, Nm, Nm)
    return pf


def _p_frag_cupy(sim):
    vrel = _field_data(sim.dust.v.rel.tot)
    vfrag = _field_data(sim.dust.v.frag)

    if _P_FRAG_MODE == "elementwise" and cp is not None:
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


def _coagulation_parameters_fortran(sim):
    return _dust_f_call(
        dust_f.coagulation_parameters,
        sim.ini.dust.erosionMassRatio,
        sim.ini.dust.excavatedMass,
        sim.ini.dust.fragmentDistribution,
        sim.grid.m,
    )


def _coagulation_parameters_python(sim):
    """Python equivalent of dust_f.coagulation_parameters for CuPy backend."""
    m_src = _field_data(sim.grid.m)
    if xp.is_array(m_src):
        m = np.asarray(xp.to_numpy(m_src), dtype=np.float64)
    else:
        m = np.asarray(m_src, dtype=np.float64)

    cratRatio = float(sim.ini.dust.erosionMassRatio)
    fExcav = float(sim.ini.dust.excavatedMass)
    fragSlope = float(sim.ini.dust.fragmentDistribution)
    Nm = int(m.shape[0])

    AFrag = np.zeros((Nm, Nm), dtype=np.float64)
    epsFrag = np.zeros((Nm, Nm), dtype=np.float64)
    klf = -np.ones((Nm, Nm), dtype=np.int64)
    krm = -np.ones((Nm, Nm), dtype=np.int64)
    cstick = np.zeros((4, Nm, Nm), dtype=np.float64)
    cstick_ind = -np.ones((4, Nm, Nm), dtype=np.int64)
    phiFrag = np.zeros((Nm, Nm), dtype=np.float64)

    cpod = np.zeros((Nm, Nm, Nm), dtype=np.float64)
    cpodmod = np.zeros((Nm, Nm, Nm), dtype=np.float64)
    D = -np.ones((Nm, Nm), dtype=np.float64)
    E = np.zeros((Nm, Nm), dtype=np.float64)

    a = np.log10(m[0] / m[-1]) / (1.0 - Nm)
    ce = int(-1.0 / a * np.log10(1.0 - 10.0**(-a))) + 1
    p = int(np.floor(np.log10(cratRatio) / a))

    mi_m_mim1 = m * (1.0 - 10.0**(-a))
    mip1_m_mi = m * (10.0**a - 1.0)
    mip1_m_mim1 = m * (10.0**a - 10.0**(-a))

    mtot = m[:, None] + m[None, :]
    valid = mtot < m[-1]
    jv, iv = np.where(valid)
    upper = np.searchsorted(m, mtot[jv, iv], side="right")
    lower = upper - 1
    epsv = (m[lower + 1] - mtot[jv, iv]) / mip1_m_mi[lower]
    cpod[lower, jv, iv] = epsv
    cpod[lower + 1, jv, iv] = 1.0 - epsv

    j_idx = np.arange(Nm)[:, None]
    i_idx = np.arange(Nm)[None, :]
    J = np.broadcast_to(m[:, None], (Nm, Nm))
    Mp = np.broadcast_to(mip1_m_mi[None, :], (Nm, Nm))
    Mm = np.broadcast_to(mi_m_mim1[None, :], (Nm, Nm))
    Mpm = np.broadcast_to(mip1_m_mim1[None, :], (Nm, Nm))

    maskD = j_idx <= (i_idx + 1 - ce)
    D[maskD] = -J[maskD] / Mp[maskD]

    maskE1 = j_idx <= (i_idx - ce)
    E[maskE1] = J[maskE1] / Mm[maskE1]
    tmpE = 1.0 - (J - Mm) / Mp
    tmpE *= (Mpm - J >= 0.0).astype(np.float64)
    E[~maskE1] = tmpE[~maskE1]

    for i in range(Nm - 1):
        jmax = min(Nm - 1, int(np.log10(10.0**(a * Nm) - 10.0**(a * (i + 1))) / a))
        for j in range(jmax):
            out = np.zeros((Nm,), dtype=np.float64)
            cpod_slice = cpod[:, j, i]
            if j == i:
                out += 0.5 * cpod_slice
            if j >= i + 1 and (j + 2) < Nm:
                out[j + 2:] += cpod_slice[j + 2:]
            out[j] += D[i, j]
            if j >= i + 1 and (j + 1) < Nm:
                out[j + 1] += E[i, j + 1]
            cpodmod[:, j, i] = out

    dum = cpodmod.copy()
    for i in range(Nm):
        for j in range(i + 1):
            cpodmod[:, i, j] = 0.0
            cpodmod[:, j, i] = dum[:, j, i] + dum[:, i, j]

    for i in range(Nm):
        for j in range(i + 1):
            nz = np.nonzero(cpodmod[:, j, i])[0]
            nnz = min(4, nz.shape[0])
            if nnz > 0:
                cstick_ind[:nnz, j, i] = nz[:nnz]
                cstick[:nnz, j, i] = cpodmod[nz[:nnz], j, i]

    for i in range(Nm):
        phiFrag[i, : i + 1] = m[: i + 1] ** (2.0 + fragSlope)
        norm = np.sum(phiFrag[i, : i + 1])
        if norm > 0.0:
            phiFrag[i, : i + 1] /= norm

    for i in range(Nm):
        upper = i - p
        if upper > 0:
            jarr = np.arange(0, upper)
            klf[jarr, i] = jarr
            mrm = m[i] - fExcav * m[jarr]
            AFrag[jarr, i] = (1.0 + fExcav) * m[jarr]

            kr = np.searchsorted(m, mrm, side="left") - 1
            eq = (mrm == m[i])
            if np.any(eq):
                kr[eq] = i - 1
                epsFrag[jarr[eq], i] = fExcav * m[jarr[eq]] / mi_m_mim1[i]
            ne = ~eq
            if np.any(ne):
                krn = kr[ne]
                epsFrag[jarr[ne], i] = (m[krn + 1] - mrm[ne]) / mip1_m_mi[krn]
            krm[jarr, i] = kr

        jmin = max(0, i - p)
        if jmin <= i:
            jarr = np.arange(jmin, i + 1)
            klf[jarr, i] = i
            AFrag[jarr, i] = m[i] + m[jarr]

    if get_backend() == "cupy" and cp is not None:
        return (
            cp.asarray(cstick),
            cp.asarray(cstick_ind),
            cp.asarray(AFrag),
            cp.asarray(epsFrag),
            cp.asarray(klf),
            cp.asarray(krm),
            cp.asarray(phiFrag),
        )
    return cstick, cstick_ind, AFrag, epsFrag, klf, krm, phiFrag


def bind_backend_kernels(backend=None, force=False, runtime_token=None):
    """Bind hot dust kernels to backend-specific implementations once."""
    global _BOUND_BACKEND, _K_A, _K_D, _K_H, _K_F_ADV, _K_F_DIFF, _K_S_COAG, _K_S_HYD
    global _K_KERNEL, _K_P_FRAG, _K_ST, _K_VRAD, _K_VREL_BROWN, _K_VREL_AZI, _K_VREL_RAD, _K_VREL_TURB, _K_VREL_VERT
    global _K_COAG_PARAMS, _K_IMPL_1_DIRECT, _K_JACOBIAN, _K_INTERP_TO_INTERFACES
    global _KERNEL_LOWER_MASK, _COAG_CACHE_KEY, _COAG_CACHE_VALUE
    global _COAG_PAIR_CACHE_KEY, _COAG_PAIR_CACHE_VALUE, _SCOAG_PAIRMAP_CACHE_KEY, _SCOAG_PAIRMAP_CACHE_VALUE
    global _FRAG_P_CACHE_KEY, _FRAG_P_CACHE_VALUE, _MGRID_Q_CACHE_KEY, _MGRID_Q_CACHE_VALUE
    global _JCOAG_CONST_CACHE_KEY, _JCOAG_CONST_CACHE_VALUE, _JCOAG_PATTERN_CACHE_KEY, _JCOAG_PATTERN_CACHE_VALUE
    global _JCOAG_PATTERN_GPU_CACHE_KEY, _JCOAG_PATTERN_GPU_CACHE_VALUE
    global _JCOAG_PRECOMP_CACHE_KEY, _JCOAG_PRECOMP_CACHE_VALUE
    global _BOUNDARY_BASIS_CACHE_KEY, _BOUNDARY_BASIS_CACHE_VALUE
    global _JSTICK_MAP_CACHE_KEY, _JSTICK_MAP_CACHE_VALUE
    global _JFRAG_MAP_CACHE_KEY, _JFRAG_MAP_CACHE_VALUE, _JCOAG_WORK_CACHE_KEY, _JCOAG_WORK_CACHE_VALUE
    global _DUST_JHB_PATTERN_CACHE_KEY, _DUST_JHB_PATTERN_CACHE_VALUE
    global _JCOAG_GEN_MODE, _S_COAG_MODE, _F_DIFF_MODE, _SCATTER_MODE, _CUPY_DUST_SOLVER_MODE
    global _VREL_TURB_MODE, _VREL_TOT_MODE, _P_FRAG_MODE, _COLLISION_KERNEL_MODE
    global _CUPY_A_BUILD_MODE, _CUPY_DIAG_POS_CACHE_KEY, _CUPY_DIAG_POS_CACHE_VALUE
    global _JCOAG_CHUNK_SIZE_OVERRIDE

    _switch_runtime_state(runtime_token)
    backend = get_backend() if backend is None else backend
    bind_sparse_solver(backend=backend, force=force)
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

    _KERNEL_LOWER_MASK = None
    _COAG_CACHE_KEY = None
    _COAG_CACHE_VALUE = None
    _COAG_PAIR_CACHE_KEY = None
    _COAG_PAIR_CACHE_VALUE = None
    _SCOAG_PAIRMAP_CACHE_KEY = None
    _SCOAG_PAIRMAP_CACHE_VALUE = None
    _FRAG_P_CACHE_KEY = None
    _FRAG_P_CACHE_VALUE = None
    _MGRID_Q_CACHE_KEY = None
    _MGRID_Q_CACHE_VALUE = None
    _JCOAG_CONST_CACHE_KEY = None
    _JCOAG_CONST_CACHE_VALUE = None
    _JCOAG_PATTERN_CACHE_KEY = None
    _JCOAG_PATTERN_CACHE_VALUE = None
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
    _CUPY_DIAG_POS_CACHE_KEY = None
    _CUPY_DIAG_POS_CACHE_VALUE = None

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
    sim.dust.boundary.inner.setboundary()
    sim.dust.boundary.outer.setboundary()


def enforce_floor_value(sim):
    """Function enforces floor value onto dust surface density.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    sim.dust.Sigma = xp.where(
        sim.dust.Sigma > sim.dust.SigmaFloor,
        sim.dust.Sigma,
        0.1*sim.dust.SigmaFloor)


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
        sim.dust.Fi.tot[0] = sim.dust.Fi.adv[0]
        sim.dust.Fi.tot[-1] = sim.dust.Fi.adv[-1]


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
    return Fi


def F_diff(sim, Sigma=None):
    '''Function calculates the diffusive flux at the cell interfaces'''
    return _K_F_DIFF(sim, Sigma=Sigma)


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
    A = sim.dust.coagulation.A
    cstick = sim.dust.coagulation.stick
    eps = sim.dust.coagulation.eps
    ilf = sim.dust.coagulation.lf_ind
    irm = sim.dust.coagulation.rm_ind
    istick = sim.dust.coagulation.stick_ind
    m = sim.grid.m
    phi = sim.dust.coagulation.phi
    Rf = sim.dust.kernel * sim.dust.p.frag
    Rs = sim.dust.kernel * sim.dust.p.stick
    SigD = sim.dust.Sigma
    SigDfloor = sim.dust.SigmaFloor

    # Helper variables for convenience
    if dx is None:
        dt = x.stepsize
    else:
        dt = dx
    try:
        dt = float(dt)
    except Exception:
        dt = float(to_numpy(dt))
    r = sim.grid.r
    ri = sim.grid.ri
    r_np = to_numpy(r)
    ri_np = to_numpy(ri)
    Sigma_np = to_numpy(sim.dust.Sigma)
    area = sim.grid.A
    Nr = int(sim.grid.Nr)
    Nm = int(sim.grid.Nm)
    zero_flux = is_zero_flux_enabled(sim)

    # Building coagulation Jacobian

    # Total problem size
    Ntot = int((Nr*Nm))
    dat, row, col = _dust_f_call(
        dust_f.jacobian_coagulation_generator,
        A, cstick, eps, ilf, irm, istick, m, phi, Rf, Rs, SigD, SigDfloor,
        to_backend_result=False,
    )
    gen = (dat, (row, col))
    J_coag = sp.csc_matrix(
        gen,
        shape=(Ntot, Ntot)
    )

    A_h, B_h, C_h = _dust_f_call(
        dust_f.jacobian_hydrodynamic_generator,
        area,
        sim.dust.D,
        r,
        ri,
        sim.gas.Sigma,
        sim.dust.v.rad,
        to_backend_result=False,
    )
    if zero_flux:
        A_h, B_h, C_h = _apply_zero_flux_dust_hyd_edges_numpy(
            A_h, B_h, C_h, area, sim.dust.D, r, ri, sim.gas.Sigma, sim.dust.v.rad
        )
    J_hyd = sp.diags(
        (A_h.ravel()[Nm:], B_h.ravel(), C_h.ravel()[:-Nm]),
        offsets=(-Nm, 0, Nm),
        shape=(Ntot, Ntot),
        format="csc"
    )

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
    if (not zero_flux) and sim.dust.boundary.inner is not None:
        # Given value
        if sim.dust.boundary.inner.condition == "val":
            sim.dust._rhs[:Nm] = sim.dust.boundary.inner.value
        # Constant value
        elif sim.dust.boundary.inner.condition == "const_val":
            dat[Nm:2*Nm] = 1./dt
            sim.dust._rhs[:Nm] = 0.
        # Given gradient
        elif sim.dust.boundary.inner.condition == "grad":
            K1 = - r_np[1]/r_np[0]
            dat[Nm:2*Nm] = -K1/dt
            sim.dust._rhs[:Nm] = - ri_np[1]/r_np[0] * \
                (r_np[1]-r_np[0])*sim.dust.boundary.inner.value
        # Constant gradient
        elif sim.dust.boundary.inner.condition == "const_grad":
            Di = ri_np[1]/ri_np[2] * (r_np[1]-r_np[0]) / (r_np[2]-r_np[0])
            K1 = - r_np[1]/r_np[0] * (1. + Di)
            K2 = r_np[2]/r_np[0] * Di
            dat[:Nm] = 0.
            dat[Nm:2*Nm] = -K1/dt
            dat[2*Nm:] = -K2/dt
            sim.dust._rhs[:Nm] = 0.
        # Given power law
        elif sim.dust.boundary.inner.condition == "pow":
            p = sim.dust.boundary.inner.value
            sim.dust._rhs[:Nm] = sim.dust.Sigma[1] * (r_np[0]/r_np[1])**p
        # Constant power law
        elif sim.dust.boundary.inner.condition == "const_pow":
            p = np.log(Sigma_np[2] /
                       Sigma_np[1]) / np.log(r_np[2]/r_np[1])
            K1 = - (r_np[0]/r_np[1])**p
            dat[Nm:2*Nm] = -K1/dt
            sim.dust._rhs[:Nm] = 0.

    # Creating sparce matrix for inner boundary
    gen = (dat, (row, col))
    J_in = sp.csc_matrix(
        gen,
        shape=(Ntot, Ntot)
    )

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
    J_out = sp.csc_matrix(
        gen,
        shape=(Ntot, Ntot)
    )

    return J_in + J_coag + J_hyd + J_out


def _jacobian_cupy(sim, x, dx=None, *args, **kwargs):
    # Parameters for function call
    A = sim.dust.coagulation.A
    cstick = sim.dust.coagulation.stick
    eps = sim.dust.coagulation.eps
    ilf = sim.dust.coagulation.lf_ind
    irm = sim.dust.coagulation.rm_ind
    istick = sim.dust.coagulation.stick_ind
    m = sim.grid.m
    phi = sim.dust.coagulation.phi
    Rf = sim.dust.kernel * sim.dust.p.frag
    Rs = sim.dust.kernel * sim.dust.p.stick
    SigD = sim.dust.Sigma
    SigDfloor = sim.dust.SigmaFloor

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
    SigmaArr = _field_data(sim.dust.Sigma)
    area = sim.grid.A
    Nr = int(sim.grid.Nr)
    Nm = int(sim.grid.Nm)
    Ntot = int((Nr * Nm))
    zero_flux = is_zero_flux_enabled(sim)
    q = _get_mass_grid_q(cp.asarray(_field_data(m)), Nm)
    _, _, _, _, _, jcoag_indices, jcoag_indptr, jcoag_perm = _get_jcoag_pattern_cupy(Nr, Nm, q)

    dat, _, _ = _jacobian_coagulation_generator_cupy(
        A, cstick, eps, ilf, irm, istick, m, phi, Rf, Rs, SigD, SigDfloor
    )
    dat_csr = dat if jcoag_perm is None else dat[jcoag_perm]
    J_coag = cp_sparse.csr_matrix((dat_csr, jcoag_indices, jcoag_indptr), shape=(Ntot, Ntot))

    A_h, B_h, C_h = _jacobian_hydrodynamic_generator_cupy(
        area, sim.dust.D, r, ri, sim.gas.Sigma, sim.dust.v.rad
    )
    if zero_flux:
        A_h, B_h, C_h = _apply_zero_flux_dust_hyd_edges_cupy(
            A_h, B_h, C_h, area, sim.dust.D, r, ri, sim.gas.Sigma, sim.dust.v.rad
        )
    n_hyd, jhb_map, jhb_indices, jhb_indptr = _get_dust_hyd_boundary_pattern_cupy(Nr, Nm)
    dat_hyd = cp.concatenate((A_h.ravel()[Nm:], B_h.ravel(), C_h.ravel()[:-Nm]))

    # Right-hand side defaults to the current state for all rows.
    sim.dust._rhs[:] = sim.dust.Sigma.ravel()
    c_in0 = 0.0
    c_in1 = 0.0
    c_in2 = 0.0

    if (not zero_flux) and sim.dust.boundary.inner is not None:
        if sim.dust.boundary.inner.condition == "val":
            sim.dust._rhs[:Nm] = sim.dust.boundary.inner.value
        elif sim.dust.boundary.inner.condition == "const_val":
            c_in1 = 1. / dt
            sim.dust._rhs[:Nm] = 0.
        elif sim.dust.boundary.inner.condition == "grad":
            K1 = - (r[1] / r[0])
            c_in1 = -K1 / dt
            fac = - (ri[1] / r[0] * (r[1] - r[0]))
            sim.dust._rhs[:Nm] = fac * sim.dust.boundary.inner.value
        elif sim.dust.boundary.inner.condition == "const_grad":
            Di = (ri[1] / ri[2] * (r[1] - r[0]) / (r[2] - r[0]))
            K1 = - (r[1] / r[0]) * (1. + Di)
            K2 = (r[2] / r[0]) * Di
            c_in0 = 0.0
            c_in1 = -K1 / dt
            c_in2 = -K2 / dt
            sim.dust._rhs[:Nm] = 0.
        elif sim.dust.boundary.inner.condition == "pow":
            p = sim.dust.boundary.inner.value
            ratio = (r[0] / r[1])
            sim.dust._rhs[:Nm] = SigmaArr[1] * ratio**p
        elif sim.dust.boundary.inner.condition == "const_pow":
            logr = cp.log(r[2] / r[1])
            p = cp.log(SigmaArr[2] / SigmaArr[1]) / logr
            K1 = - (r[0] / r[1])**p
            c_in1 = -K1 / dt
            sim.dust._rhs[:Nm] = 0.

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
    A = sim.dust.coagulation.A
    cstick = sim.dust.coagulation.stick
    eps = sim.dust.coagulation.eps
    ilf = sim.dust.coagulation.lf_ind
    irm = sim.dust.coagulation.rm_ind
    istick = sim.dust.coagulation.stick_ind
    m = sim.grid.m
    phi = sim.dust.coagulation.phi
    Rf = sim.dust.kernel * sim.dust.p.frag
    Rs = sim.dust.kernel * sim.dust.p.stick
    SigD = sim.dust.Sigma
    SigDfloor = sim.dust.SigmaFloor

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
    SigmaArr = _field_data(sim.dust.Sigma)
    area = sim.grid.A
    Nr = int(sim.grid.Nr)
    Nm = int(sim.grid.Nm)
    Ntot = int(Nr * Nm)
    zero_flux = is_zero_flux_enabled(sim)

    q = _get_mass_grid_q(cp.asarray(_field_data(m)), Nm)
    _, _, row, col, _, _, _, _ = _get_jcoag_pattern_cupy(Nr, Nm, q)
    dat_coag, _, _ = _jacobian_coagulation_generator_cupy(
        A, cstick, eps, ilf, irm, istick, m, phi, Rf, Rs, SigD, SigDfloor
    )
    dtype = dat_coag.dtype
    J = cp.zeros((Ntot, Ntot), dtype=dtype)
    J_flat = J.ravel()
    _scatter_add_1d(J_flat, row * Ntot + col, dat_coag)

    A_h, B_h, C_h = _jacobian_hydrodynamic_generator_cupy(
        area, sim.dust.D, r, ri, sim.gas.Sigma, sim.dust.v.rad
    )
    if zero_flux:
        A_h, B_h, C_h = _apply_zero_flux_dust_hyd_edges_cupy(
            A_h, B_h, C_h, area, sim.dust.D, r, ri, sim.gas.Sigma, sim.dust.v.rad
        )

    idx = cp.arange(Ntot, dtype=cp.int64)
    Bflat = B_h.ravel()
    Aflat = A_h.ravel()
    Cflat = C_h.ravel()
    _scatter_add_1d(J_flat, idx * Ntot + idx, Bflat)
    _scatter_add_1d(J_flat, idx[Nm:] * Ntot + (idx[Nm:] - Nm), Aflat[Nm:])
    _scatter_add_1d(J_flat, idx[:-Nm] * Ntot + (idx[:-Nm] + Nm), Cflat[:-Nm])

    # Right-hand side defaults to the current state for all rows.
    sim.dust._rhs[:] = sim.dust.Sigma.ravel()
    c_in0 = 0.0
    c_in1 = 0.0
    c_in2 = 0.0

    if (not zero_flux) and sim.dust.boundary.inner is not None:
        if sim.dust.boundary.inner.condition == "val":
            sim.dust._rhs[:Nm] = sim.dust.boundary.inner.value
        elif sim.dust.boundary.inner.condition == "const_val":
            c_in1 = 1. / dt
            sim.dust._rhs[:Nm] = 0.
        elif sim.dust.boundary.inner.condition == "grad":
            K1 = - (r[1] / r[0])
            c_in1 = -K1 / dt
            fac = - (ri[1] / r[0] * (r[1] - r[0]))
            sim.dust._rhs[:Nm] = fac * sim.dust.boundary.inner.value
        elif sim.dust.boundary.inner.condition == "const_grad":
            Di = (ri[1] / ri[2] * (r[1] - r[0]) / (r[2] - r[0]))
            K1 = - (r[1] / r[0]) * (1. + Di)
            K2 = (r[2] / r[0]) * Di
            c_in0 = 0.0
            c_in1 = -K1 / dt
            c_in2 = -K2 / dt
            sim.dust._rhs[:Nm] = 0.
        elif sim.dust.boundary.inner.condition == "pow":
            p = sim.dust.boundary.inner.value
            ratio = (r[0] / r[1])
            sim.dust._rhs[:Nm] = SigmaArr[1] * ratio**p
        elif sim.dust.boundary.inner.condition == "const_pow":
            logr = cp.log(r[2] / r[1])
            p = cp.log(SigmaArr[2] / SigmaArr[1]) / logr
            K1 = - (r[0] / r[1])**p
            c_in1 = -K1 / dt
            sim.dust._rhs[:Nm] = 0.

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


def _vrel_tot_cupy_elementwise(azi, brown, rad, turb, vert):
    """Elementwise-kernel variant for total relative velocity."""
    dtype = cp.result_type(azi.dtype, brown.dtype, rad.dtype, turb.dtype, vert.dtype)
    if dtype not in (cp.float32, cp.float64):
        return None

    kernel = _get_vrel_tot_elementwise_kernel(dtype)
    if kernel is None:
        return None

    return kernel(
        cp.asarray(azi, dtype=dtype),
        cp.asarray(brown, dtype=dtype),
        cp.asarray(rad, dtype=dtype),
        cp.asarray(turb, dtype=dtype),
        cp.asarray(vert, dtype=dtype),
    )


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
    if _VREL_TOT_MODE == "elementwise" and cp is not None and getattr(xp, "name", "") == "cupy":
        vrel = _vrel_tot_cupy_elementwise(
            _field_data(sim.dust.v.rel.azi),
            _field_data(sim.dust.v.rel.brown),
            _field_data(sim.dust.v.rel.rad),
            _field_data(sim.dust.v.rel.turb),
            _field_data(sim.dust.v.rel.vert),
        )
        if vrel is not None:
            return vrel

    return xp.sqrt(
        sim.dust.v.rel.azi**2
        + sim.dust.v.rel.brown**2
        + sim.dust.v.rel.rad**2
        + sim.dust.v.rel.turb**2
        + sim.dust.v.rel.vert**2
    )


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
        rhs = np.asarray(_field_data(Y0.ravel()))

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

    return Y1 - Y0


def _f_impl_1_direct_cupy(x0, Y0, dx, jac=None, rhs=None, *args, **kwargs):
    """CuPy-only implicit 1st-order integration scheme with GPU sparse solve."""
    zero_flux = is_zero_flux_enabled(Y0._owner)
    if jac is None:
        jac = Y0.jacobian(x0, dx)
    if rhs is None:
        rhs = cp.asarray(_field_data(Y0.ravel()))

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
    return Y1 - Y0


def _f_impl_1_direct_cupy_dense(x0, Y0, dx, jac=None, rhs=None, *args, **kwargs):
    """CuPy implicit 1st-order scheme using dense GPU matrix solve."""
    zero_flux = is_zero_flux_enabled(Y0._owner)
    if jac is None:
        jac = Y0.jacobian(x0, dx)
    if rhs is None:
        rhs = cp.asarray(_field_data(Y0.ravel()))
    else:
        rhs = cp.asarray(_field_data(rhs))

    Nm = Y0._owner.dust.Sigma.shape[1]
    if zero_flux:
        rhs[:] += dx * _field_data(Y0._owner.dust.S.ext).ravel()
    else:
        rhs[Nm:-Nm] += dx * _field_data(Y0._owner.dust.S.ext[1:-1, ...]).ravel()

    jac_dense = jac if isinstance(jac, cp.ndarray) else cp.asarray(jac)
    A = cp.eye(int(jac_dense.shape[0]), dtype=jac_dense.dtype) - dx * jac_dense
    Y1_ravel = cp.linalg.solve(A, rhs)
    Y1 = Y1_ravel.reshape(Y0.shape)
    return Y1 - Y0


def _f_impl_1_direct(x0, Y0, dx, jac=None, rhs=None, *args, **kwargs):
    return _K_IMPL_1_DIRECT(x0, Y0, dx, jac=jac, rhs=rhs, *args, **kwargs)


class impl_1_direct(Scheme):
    """Modified class for implicit dust integration."""

    def __init__(self):
        super().__init__(_f_impl_1_direct, description="Implicit 1st-order direct solver")


# Initialize default bindings at import time after all functions are defined.
bind_backend_kernels()
