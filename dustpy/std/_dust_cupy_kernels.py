"""CuPy kernel source templates and lazy kernel caches for dust evolution."""

import dustpy.constants as c
import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None


_FP_REPLACEMENTS = {
    "f32": {
        "@FP_API@": "float32",
        "@FP_C@": "float",
        "@FP_SUFFIX@": "f",
        "@FP_EPS@": "1.0e-30f",
        "@FP_TAG@": "f32",
    },
    "f64": {
        "@FP_API@": "float64",
        "@FP_C@": "double",
        "@FP_SUFFIX@": "",
        "@FP_EPS@": "1.0e-300",
        "@FP_TAG@": "f64",
    },
}

_RAW_SCATTER_TEMPLATE = (
    r"""
                extern "C" __global__
                void dustpy_scatter_add_@FP_TAG@(@FP_C@* out, const long long* idx, const @FP_C@* vals, long long n) {
                    long long tid = (long long)(blockDim.x * blockIdx.x + threadIdx.x);
                    if (tid < n) {
                        atomicAdd(out + idx[tid], vals[tid]);
                    }
                }
                """,
    "dustpy_scatter_add_@FP_TAG@",
)

_FDIFF_TEMPLATE = (
    "raw @FP_API@ D, raw @FP_API@ SigmaD, raw @FP_API@ SigmaG, raw @FP_API@ St, raw @FP_API@ u, raw @FP_API@ r, raw @FP_API@ ri, int64 nm",
    "@FP_API@ out",
    r"""
                const long long iface = i / nm + 1;
                const long long jm = i - (iface - 1) * nm;
                const long long l = iface - 1;
                const long long rr = iface;
                const long long idxL = l * nm + jm;
                const long long idxR = rr * nm + jm;

                const @FP_C@ eps0 = @FP_EPS@;
                const @FP_C@ dr = r[rr] - r[l];
                const @FP_C@ t = (ri[iface] - r[l]) / (dr + eps0);

                const @FP_C@ siggi = SigmaG[l] + t * (SigmaG[rr] - SigmaG[l]);
                const @FP_C@ ui = u[l] + t * (u[rr] - u[l]);
                const @FP_C@ di = D[idxL] + t * (D[idxR] - D[idxL]);
                const @FP_C@ sigdi = SigmaD[idxL] + t * (SigmaD[idxR] - SigmaD[idxL]);
                const @FP_C@ sti = St[idxL] + t * (St[idxR] - St[idxL]);

                const @FP_C@ epsL = SigmaD[idxL] / (SigmaG[l] + eps0);
                const @FP_C@ epsR = SigmaD[idxR] / (SigmaG[rr] + eps0);
                const @FP_C@ gradepsi = (epsR - epsL) / (dr + eps0);

                const @FP_C@ w = ui * sigdi / (1.0@FP_SUFFIX@ + sti * sti);
                const @FP_C@ fi0 = -di * siggi * gradepsi;

                if (fabs@FP_SUFFIX@(w) > 0.0@FP_SUFFIX@) {
                    const @FP_C@ P = fabs@FP_SUFFIX@(fi0 / w);
                    const @FP_C@ lam = (1.0@FP_SUFFIX@ + P) / (1.0@FP_SUFFIX@ + P + P * P);
                    out = lam * fi0;
                } else {
                    out = w;
                }
                """,
    "dustpy_fdiff_elementwise_@FP_TAG@",
)

_VREL_TURBULENT_TEMPLATE = (
    "raw @FP_API@ alpha, raw @FP_API@ cs, raw @FP_API@ mump, raw @FP_API@ omegak, raw @FP_API@ sigmag, raw @FP_API@ st, int64 nm, @FP_API@ sigma_h2",
    "@FP_API@ out",
    r"""
                const long long plane = nm * nm;
                const long long ir = i / plane;
                const long long rem = i - ir * plane;
                const long long jm = rem / nm;
                const long long im = rem - jm * nm;
                const long long base = ir * nm;

                const @FP_C@ eps0 = @FP_EPS@;
                @FP_C@ Stj = st[base + jm];
                @FP_C@ Sti = st[base + im];
                @FP_C@ StL = Stj >= Sti ? Stj : Sti;
                @FP_C@ StS = Stj >= Sti ? Sti : Stj;

                @FP_C@ OmKinv = 1.0@FP_SUFFIX@ / (omegak[ir] + eps0);
                @FP_C@ Re = 0.5@FP_SUFFIX@ * alpha[ir] * sigmag[ir] * sigma_h2 / (mump[ir] + eps0);
                @FP_C@ ReInvSqrt = sqrt@FP_SUFFIX@(1.0@FP_SUFFIX@ / (Re + eps0));
                @FP_C@ vn = sqrt@FP_SUFFIX@(alpha[ir]) * cs[ir];
                @FP_C@ vs = pow@FP_SUFFIX@(Re + eps0, -0.25@FP_SUFFIX@) * vn;
                @FP_C@ ts = OmKinv * ReInvSqrt;
                @FP_C@ vg2 = 1.5@FP_SUFFIX@ * vn * vn;

                const @FP_C@ c0 = 1.6015125@FP_SUFFIX@;
                const @FP_C@ c1 = -0.63119577@FP_SUFFIX@;
                const @FP_C@ c2 = 0.32938936@FP_SUFFIX@;
                const @FP_C@ c3 = -0.29847604@FP_SUFFIX@;
                const @FP_C@ ya = 1.6@FP_SUFFIX@;
                const @FP_C@ yap1inv = 1.0@FP_SUFFIX@ / (1.0@FP_SUFFIX@ + ya);

                @FP_C@ eps = StS / (StL + eps0);
                @FP_C@ tauL = StL * OmKinv;
                @FP_C@ tauS = StS * OmKinv;
                @FP_C@ StL2 = StL * StL;
                @FP_C@ StL3 = StL2 * StL;
                @FP_C@ StS2 = StS * StS;
                @FP_C@ eps3 = eps * eps * eps;

                @FP_C@ ys = c0 + c1 * StL + c2 * StL2 + c3 * StL3;
                @FP_C@ h1 = (StL - StS) / (StL + StS + eps0) * (StL * yap1inv - StS2 / (StS + ya * StL + eps0));
                @FP_C@ h2 = 2.0@FP_SUFFIX@ * (ya * StL - ReInvSqrt)
                    + StL * yap1inv
                    - StL2 / (StL + ReInvSqrt + eps0)
                    + StS2 / (ya * StL + StS + eps0)
                    - StS2 / (StS + ReInvSqrt + eps0);

                @FP_C@ t = vs / (ts + eps0) * (tauL - tauS);
                @FP_C@ val1 = 1.5@FP_SUFFIX@ * t * t;
                @FP_C@ val2 = vg2 * (StL - StS) / (StL + StS + eps0)
                    * (StL2 / (StL + ReInvSqrt + eps0) - StS2 / (StS + ReInvSqrt + eps0));
                @FP_C@ val3 = vg2 * (h1 + h2);
                @FP_C@ val4 = vg2 * StL * (2.0@FP_SUFFIX@ * ya - 1.0@FP_SUFFIX@ - eps
                    + 2.0@FP_SUFFIX@ / (1.0@FP_SUFFIX@ + eps + eps0) * (yap1inv + eps3 / (ya + eps + eps0)));
                @FP_C@ val5 = vg2 * StL * (2.0@FP_SUFFIX@ * ys - 1.0@FP_SUFFIX@ - eps
                    + 2.0@FP_SUFFIX@ / (1.0@FP_SUFFIX@ + eps + eps0) * (1.0@FP_SUFFIX@ / (1.0@FP_SUFFIX@ + ys + eps0) + eps3 / (ys + eps + eps0)));
                @FP_C@ val6 = vg2 * (2.0@FP_SUFFIX@ + StL + StS) / (1.0@FP_SUFFIX@ + StL + StS + StL * StS + eps0);

                bool cond1 = tauL < 0.2@FP_SUFFIX@ * ts;
                bool cond2 = tauL * ya < ts;
                bool cond3 = tauL < 5.0@FP_SUFFIX@ * ts;
                bool cond4 = tauL < 0.2@FP_SUFFIX@ * OmKinv;
                bool cond5 = tauL < OmKinv;

                @FP_C@ v2;
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
                out = sqrt@FP_SUFFIX@(v2 > 0.0@FP_SUFFIX@ ? v2 : 0.0@FP_SUFFIX@);
                """,
    "dustpy_vrel_turbulent_elementwise_@FP_TAG@",
)

_VREL_TOTAL_TEMPLATE = (
    "@FP_API@ az, @FP_API@ br, @FP_API@ rd, @FP_API@ tu, @FP_API@ vt",
    "@FP_API@ out",
    "out = sqrt@FP_SUFFIX@(az * az + br * br + rd * rd + tu * tu + vt * vt)",
    "dustpy_vrel_tot_elementwise_@FP_TAG@",
)

_PFRAG_TEMPLATE = (
    "raw @FP_API@ vrel, raw @FP_API@ vfrag, int64 nm",
    "@FP_API@ out",
    r"""
                const long long nm2 = nm * nm;
                const long long ir_int = i / nm2;
                const @FP_C@ v = vrel[i];
                const @FP_C@ vf = vfrag[ir_int];
                if (v != 0.0@FP_SUFFIX@) {
                    const @FP_C@ dum = (vf / v) * (vf / v);
                    out = (1.5@FP_SUFFIX@ * dum + 1.0@FP_SUFFIX@) * exp@FP_SUFFIX@(-1.5@FP_SUFFIX@ * dum);
                } else {
                    out = 0.0@FP_SUFFIX@;
                }
                """,
    "dustpy_pfrag_elementwise_@FP_TAG@",
)

_COLLISION_TEMPLATE = (
    "raw @FP_API@ a, raw @FP_API@ h, raw @FP_API@ sigma, raw @FP_API@ floor, raw @FP_API@ vrel, int64 nm",
    "@FP_API@ out",
    r"""
                const long long nm2 = nm * nm;
                const long long ir_int = i / nm2;
                const long long rem = i - ir_int * nm2;
                const long long jm = rem / nm;
                const long long im = rem - jm * nm;

                if (jm > im) {
                    out = 0.0@FP_SUFFIX@;
                    return;
                }

                const long long base = ir_int * nm;
                if (sigma[base + jm] <= floor[base + jm] || sigma[base + im] <= floor[base + im]) {
                    out = 0.0@FP_SUFFIX@;
                    return;
                }

                const @FP_C@ pi = 3.14159265358979323846@FP_SUFFIX@;
                const @FP_C@ sum_a = a[base + jm] + a[base + im];
                const @FP_C@ cross = pi * sum_a * sum_a;
                const @FP_C@ hj = h[base + jm];
                const @FP_C@ hi = h[base + im];
                const @FP_C@ sh = sqrt@FP_SUFFIX@(2.0@FP_SUFFIX@ * pi * (hj * hj + hi * hi));
                const @FP_C@ diag_fac = (jm == im) ? 0.5@FP_SUFFIX@ : 1.0@FP_SUFFIX@;
                out = diag_fac * cross * vrel[i] / sh;
                """,
    "dustpy_collision_kernel_elementwise_@FP_TAG@",
)

_ELEMENTWISE_TEMPLATES = {
    "fdiff": _FDIFF_TEMPLATE,
    "vrel_turbulent": _VREL_TURBULENT_TEMPLATE,
    "vrel_total": _VREL_TOTAL_TEMPLATE,
    "pfrag": _PFRAG_TEMPLATE,
    "collision": _COLLISION_TEMPLATE,
}

_RAW_KERNEL_CACHE = {}
_ELEMENTWISE_KERNEL_CACHE = {}


def _render_fp(template, precision):
    rendered = template
    for token, value in _FP_REPLACEMENTS[precision].items():
        rendered = rendered.replace(token, value)
    return rendered


def _render_raw_scatter_args(precision):
    return tuple(_render_fp(value, precision) for value in _RAW_SCATTER_TEMPLATE)


def _render_elementwise_args(kind, precision):
    return tuple(_render_fp(value, precision) for value in _ELEMENTWISE_TEMPLATES[kind])


def _precision_for_dtype(dtype):
    if dtype == cp.float32:
        return "f32"
    if dtype == cp.float64:
        return "f64"
    return None


def _get_raw_scatter_kernel(dtype):
    """Return cached raw scatter-add kernel for float32/float64 outputs."""
    precision = _precision_for_dtype(dtype)
    if precision is None:
        return None
    kernel = _RAW_KERNEL_CACHE.get(precision)
    if kernel is None:
        kernel = cp.RawKernel(*_render_raw_scatter_args(precision))
        _RAW_KERNEL_CACHE[precision] = kernel
    return kernel


def _get_elementwise_kernel(kind, dtype):
    precision = _precision_for_dtype(dtype)
    if precision is None:
        return None
    key = (kind, precision)
    kernel = _ELEMENTWISE_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = cp.ElementwiseKernel(*_render_elementwise_args(kind, precision))
        _ELEMENTWISE_KERNEL_CACHE[key] = kernel
    return kernel


def _get_fdiff_elementwise_kernel(dtype):
    """Return cached elementwise kernel for diffusive flux interiors."""
    return _get_elementwise_kernel("fdiff", dtype)


def _get_vrel_turbulent_elementwise_kernel(dtype):
    """Return cached elementwise kernel for turbulent relative velocity."""
    return _get_elementwise_kernel("vrel_turbulent", dtype)


def _get_vrel_tot_elementwise_kernel(dtype):
    """Return cached elementwise kernel for total relative velocity."""
    return _get_elementwise_kernel("vrel_total", dtype)


def _get_pfrag_elementwise_kernel(dtype):
    """Return cached elementwise kernel for fragmentation probability."""
    return _get_elementwise_kernel("pfrag", dtype)


def _get_collision_kernel_elementwise_kernel(dtype):
    """Return cached elementwise kernel for collision-kernel assembly."""
    return _get_elementwise_kernel("collision", dtype)


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

    out_int = kernel(a_int, h_int, sigma_int, floor_int, vrel_int, np.int64(Nm), size=size)

    K = cp.zeros((Nr, Nm, Nm), dtype=dtype)
    K[1:-1, :, :] = out_int.reshape(Nr_int, Nm, Nm)
    return K


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

    out = kernel(alpha_arr, cs_arr, mump_arr, omk_arr, sig_arr, st_arr, np.int64(Nm), dtype.type(c.sigma_H2), size=size)
    return out.reshape(Nr, Nm, Nm)


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


def _vrel_tot_cupy_elementwise(azi, brown, rad, turb, vert):
    """Elementwise-kernel variant for total relative velocity."""
    dtype = cp.result_type(azi.dtype, brown.dtype, rad.dtype, turb.dtype, vert.dtype)
    if dtype not in (cp.float32, cp.float64):
        return None

    kernel = _get_vrel_tot_elementwise_kernel(dtype)
    if kernel is None:
        return None

    return kernel(cp.asarray(azi, dtype=dtype), cp.asarray(brown, dtype=dtype), cp.asarray(rad, dtype=dtype), cp.asarray(turb, dtype=dtype), cp.asarray(vert, dtype=dtype))
