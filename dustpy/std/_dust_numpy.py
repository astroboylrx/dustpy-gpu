"""Private NumPy/Fortran implementations for dust evolution."""

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None

import dustpy.constants as c
from dustpy.std import dust_f
from dustpy.std._dust_common import (
    _apply_gas_floor_freeze_to_flux,
    _apply_gas_floor_freeze_to_radial_field,
    _field_data,
)
from dustpy.utils.backend import (
    call_numpy,
    to_numpy,
)
from simframe.backends.api import (
    get_backend,
    xp,
)


_JCOAG_CONST_CACHE_KEY = None
_JCOAG_CONST_CACHE_VALUE = None
_JCOAG_PATTERN_CACHE_KEY = None
_JCOAG_PATTERN_CACHE_VALUE = None
_RUNTIME_STATES = {}
_ACTIVE_RUNTIME_TOKEN = None

_RUNTIME_STATE_VARS = (
    "_JCOAG_CONST_CACHE_KEY",
    "_JCOAG_CONST_CACHE_VALUE",
    "_JCOAG_PATTERN_CACHE_KEY",
    "_JCOAG_PATTERN_CACHE_VALUE",
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


def reset_runtime_caches():
    for name in _RUNTIME_STATE_VARS:
        globals()[name] = None


def _interp_to_interfaces_numpy(values, r, ri):
    """Interpolate NumPy arrays from cell centers to interfaces."""
    if values.ndim == 1:
        out = np.zeros((values.shape[0] + 1,), dtype=values.dtype)
        t = (ri[1:-1] - r[:-1]) / (r[1:] - r[:-1])
        out[1:-1] = values[:-1] + t * (values[1:] - values[:-1])
        m0 = (values[1] - values[0]) / (r[1] - r[0])
        m1 = (values[-1] - values[-2]) / (r[-1] - r[-2])
        out[0] = values[0] + m0 * (ri[0] - r[0])
        out[-1] = values[-1] + m1 * (ri[-1] - r[-1])
        return out

    out = np.zeros((values.shape[0] + 1, values.shape[1]), dtype=values.dtype)
    t = ((ri[1:-1] - r[:-1]) / (r[1:] - r[:-1]))[:, None]
    out[1:-1, :] = values[:-1, :] + t * (values[1:, :] - values[:-1, :])
    m0 = (values[1, :] - values[0, :]) / (r[1] - r[0])
    m1 = (values[-1, :] - values[-2, :]) / (r[-1] - r[-2])
    out[0, :] = values[0, :] + m0 * (ri[0] - r[0])
    out[-1, :] = values[-1, :] + m1 * (ri[-1] - r[-1])
    return out


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


def _jacobian_hydrodynamic_generator_numpy(area, D, r, ri, SigmaGas, v, freeze_velocity_mask=None, freeze_diffusion_mask=None):
    """Generate the hydrodynamic Jacobian from NumPy arrays."""
    Nr = int(r.shape[0])
    Nm = int(D.shape[1])

    h = SigmaGas * r
    hi = _interp_to_interfaces_numpy(h, r, ri)
    vi = _interp_to_interfaces_numpy(v, r, ri)
    Di = _interp_to_interfaces_numpy(D, r, ri)

    if freeze_velocity_mask is not None:
        if freeze_velocity_mask.ndim == 1 and freeze_velocity_mask.size == vi.shape[0] and np.any(freeze_velocity_mask):
            vi[freeze_velocity_mask, :] = 0.0
        elif freeze_velocity_mask.shape == vi.shape and np.any(freeze_velocity_mask):
            vi[freeze_velocity_mask] = 0.0
    if freeze_diffusion_mask is not None:
        if freeze_diffusion_mask.ndim == 1 and freeze_diffusion_mask.size == Di.shape[0] and np.any(freeze_diffusion_mask):
            Di[freeze_diffusion_mask, :] = 0.0
        elif freeze_diffusion_mask.shape == Di.shape and np.any(freeze_diffusion_mask):
            Di[freeze_diffusion_mask] = 0.0

    vim = np.minimum(vi, 0.0)
    vip = np.maximum(vi, 0.0)

    A = np.zeros((Nr, Nm))
    B = np.zeros((Nr, Nm))
    C = np.zeros((Nr, Nm))

    Vinv = np.zeros((Nr,))
    Vinv[:-1] = (2.0 * c.pi) / area[:-1]

    w = np.ones((Nr,))
    w[:-1] = r[1:] - r[:-1]

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
    """Inject conservative zero-flux boundary rows using NumPy arrays."""
    Nr = int(A.shape[0])
    if Nr < 2:
        return A, B, C

    h = SigmaGas * r
    hi = _interp_to_interfaces_numpy(h, r, ri)
    Di = _interp_to_interfaces_numpy(D, r, ri)
    vi = _interp_to_interfaces_numpy(v, r, ri)
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


def _apply_inner_zero_flux_dust_hyd_edge_numpy(A, B, C, area, D, r, ri, SigmaGas, v, block_mask, adv_drain_mask=None, diff_drain_mask=None):
    """Inject a selective inner zero-flux row using NumPy arrays."""
    Nr = int(A.shape[0])
    if Nr < 2:
        return A, B, C

    h = SigmaGas * r
    hi = _interp_to_interfaces_numpy(h, r, ri)
    Di = _interp_to_interfaces_numpy(D, r, ri)
    vi = _interp_to_interfaces_numpy(v, r, ri)
    vip = np.maximum(vi, 0.0)
    vim = np.minimum(vi, 0.0)
    Vinv = (2.0 * c.pi) / area
    w_in = r[1] - r[0]

    h0 = h[0] + 1.0e-300
    h1 = h[1] + 1.0e-300

    if block_mask.ndim != 1 or block_mask.size != B.shape[1] or not np.any(block_mask):
        return A, B, C

    vip1 = vip[1, :].copy()
    di1 = Di[1, :].copy()
    if adv_drain_mask is not None:
        if adv_drain_mask.ndim == 1 and adv_drain_mask.size == B.shape[1]:
            vip1[adv_drain_mask] = 0.0
    if diff_drain_mask is not None:
        if diff_drain_mask.ndim == 1 and diff_drain_mask.size == B.shape[1]:
            di1[diff_drain_mask] = 0.0

    b0 = (
        -vip1 * r[0]
        - di1 * hi[1] / (w_in * h0) * r[0]
    ) * Vinv[0]
    c0 = (
        -vim[1, :] * r[1]
        + di1 * hi[1] / (w_in * h1) * r[1]
    ) * Vinv[0]
    A[0, block_mask] = 0.0
    B[0, block_mask] = b0[block_mask]
    C[0, block_mask] = c0[block_mask]
    return A, B, C


def _dust_f_call(func, *args, to_backend_result=True, **kwargs):
    """Call NumPy/F2PY dust kernels with backend-safe conversions."""
    args = tuple(_field_data(arg) for arg in args)
    kwargs = {key: _field_data(value) for key, value in kwargs.items()}
    return call_numpy(func, *args, to_backend_result=to_backend_result, audit_tag=f"dust_f:{func.__name__}", **kwargs)


def _a_fortran(sim):
    rho = sim.dust.fill * sim.dust.rhos
    return _dust_f_call(dust_f.a, sim.grid.m, rho)


def _D_fortran(sim):
    v2 = sim.dust.delta.rad * sim.gas.cs**2
    Diff = _dust_f_call(dust_f.d, v2, sim.grid.OmegaK, sim.dust.St)
    Diff[:2, ...] = 0.
    Diff[-2:, ...] = 0.
    _apply_gas_floor_freeze_to_radial_field(sim, Diff)
    return Diff


def _H_fortran(sim):
    return _dust_f_call(dust_f.h_dubrulle1995, sim.gas.Hp, sim.dust.St, sim.dust.delta.vert)


def _F_adv_fortran(sim, Sigma=None):
    Sigma = Sigma if Sigma is not None else sim.dust.Sigma
    Fi = _dust_f_call(dust_f.fi_adv, Sigma, sim.dust.v.rad, sim.grid.r, sim.grid.ri)
    _apply_gas_floor_freeze_to_flux(sim, Fi)
    return Fi


def _F_diff_fortran(sim, Sigma=None):
    if Sigma is None:
        Sigma = sim.dust.Sigma
    Fi = _dust_f_call(dust_f.fi_diff, sim.dust.D, Sigma, sim.gas.Sigma, sim.dust.St, xp.sqrt(sim.dust.delta.rad * sim.gas.cs**2), sim.grid.r, sim.grid.ri)
    Fi[:1, :] = 0.
    Fi[-1:, :] = 0.
    _apply_gas_floor_freeze_to_flux(sim, Fi)
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


def _S_coag_fortran(sim, Sigma=None):
    if Sigma is None:
        Sigma = sim.dust.Sigma
    S = _dust_f_call(dust_f.s_coag, sim.dust.coagulation.stick, sim.dust.coagulation.stick_ind, sim.dust.coagulation.A, sim.dust.coagulation.eps, sim.dust.coagulation.lf_ind, sim.dust.coagulation.rm_ind, sim.dust.coagulation.phi, sim.dust.kernel * sim.dust.p.frag, sim.dust.kernel * sim.dust.p.stick, sim.grid.m, Sigma, sim.dust.SigmaFloor)
    _apply_gas_floor_freeze_to_radial_field(sim, S)
    return S


def _kernel_fortran(sim):
    return _dust_f_call(dust_f.kernel, sim.dust.a, sim.dust.H, sim.dust.Sigma, sim.dust.SigmaFloor, sim.dust.v.rel.tot)


def _St_Epstein_StokesI_fortran(sim):
    rho = sim.dust.rhos * sim.dust.fill
    return _dust_f_call(dust_f.st_epstein_stokes1, sim.dust.a, sim.gas.mfp, rho, sim.gas.Sigma)


def _vrad_fortran(sim):
    vr = _dust_f_call(dust_f.vrad, sim.dust.St, sim.dust.v.driftmax, sim.gas.v.rad)
    _apply_gas_floor_freeze_to_radial_field(sim, vr)
    return vr


def _vrel_brownian_motion_fortran(sim):
    return _dust_f_call(dust_f.vrel_brownian_motion, sim.gas.cs, sim.grid.m, sim.gas.T)


def _vrel_azimuthal_drift_fortran(sim):
    return _dust_f_call(dust_f.vrel_azimuthal_drift, sim.dust.v.driftmax, sim.dust.St)


def _vrel_radial_drift_fortran(sim):
    return _dust_f_call(dust_f.vrel_radial_drift, sim.dust.v.rad)


def _vrel_vertical_settling_fortran(sim):
    return _dust_f_call(dust_f.vrel_vertical_settling, sim.dust.H, sim.grid.OmegaK, sim.dust.St)


def _vrel_turbulent_motion_fortran(sim):
    return _dust_f_call(dust_f.vrel_ormel_cuzzi_2007, sim.dust.delta.turb, sim.gas.cs, sim.gas.mu, sim.grid.OmegaK, sim.gas.Sigma, sim.dust.St)


def _p_frag_fortran(sim):
    return _dust_f_call(dust_f.pfrag, sim.dust.v.rel.tot, sim.dust.v.frag)


def _coagulation_parameters_fortran(sim):
    return _dust_f_call(dust_f.coagulation_parameters, sim.ini.dust.erosionMassRatio, sim.ini.dust.excavatedMass, sim.ini.dust.fragmentDistribution, sim.grid.m)


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

    if get_backend() == "cupy":
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
