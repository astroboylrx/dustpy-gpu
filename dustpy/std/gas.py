"""Module containing standard functions for the gas."""

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
try:
    from cupyx.scipy.interpolate import interp1d as cp_interp1d
except Exception:  # pragma: no cover - optional dependency
    cp_interp1d = None
from simframe.integration import Scheme
from simframe.backends.api import get_backend
from simframe.backends.api import select_backend
from simframe.backends.api import xp
from dustpy.std import gas_f
from dustpy.utils.backend import call_numpy
from dustpy.utils.backend import solve_sparse_linear_system
from dustpy.utils.backend import to_backend
from dustpy.utils.backend import to_numpy
import dustpy.constants as c


def _gas_f_call(func, *args, to_backend_result=True, **kwargs):
    """Call NumPy/F2PY gas kernels with backend-safe conversions."""
    return call_numpy(func, *args, to_backend_result=to_backend_result, **kwargs)


def _field_data(value):
    return value._data if hasattr(value, "_data") else value


def _interp_to_interfaces_1d_numpy(values, r, ri):
    values = _field_data(values)
    r = _field_data(r)
    ri = _field_data(ri)
    out = xp.zeros((values.shape[0] + 1,), dtype=values.dtype)
    t = (ri[1:-1] - r[:-1]) / (r[1:] - r[:-1])
    out[1:-1] = values[:-1] + t * (values[1:] - values[:-1])
    m0 = (values[1] - values[0]) / (r[1] - r[0])
    m1 = (values[-1] - values[-2]) / (r[-1] - r[-2])
    out[0] = values[0] + m0 * (ri[0] - r[0])
    out[-1] = values[-1] + m1 * (ri[-1] - r[-1])
    return out


def _interp_to_interfaces_1d_cupy(values, r, ri):
    values = _field_data(values)
    r = _field_data(r)
    ri = _field_data(ri)
    if cp is None or cp_interp1d is None:
        return _interp_to_interfaces_1d_numpy(values, r, ri)
    f = cp_interp1d(
        cp.asarray(r),
        cp.asarray(values),
        kind="linear",
        axis=0,
        bounds_error=False,
        fill_value="extrapolate",
    )
    return f(cp.asarray(ri))


def _interp_to_interfaces_1d(values, r, ri):
    return _K_INTERP_TO_INTERFACES_1D(values, r, ri)


def _jac_abc_cupy(area, nu, r, ri, v):
    area = _field_data(area)
    nu = _field_data(nu)
    r = _field_data(r)
    ri = _field_data(ri)
    v = _field_data(v)

    Nr = int(r.shape[0])
    vi = _interp_to_interfaces_1d(v, r, ri)
    vim = xp.minimum(vi, 0.0)
    vip = xp.maximum(vi, 0.0)

    g = nu / xp.sqrt(r)
    Di = 3.0 * xp.sqrt(ri)

    A = xp.zeros((Nr,), dtype=nu.dtype)
    B = xp.zeros((Nr,), dtype=nu.dtype)
    C = xp.zeros((Nr,), dtype=nu.dtype)

    Vinv = (2.0 * c.pi) / area
    w = xp.zeros((Nr,), dtype=r.dtype)
    w[:-1] = r[1:] - r[:-1]
    w[-1] = w[-2]

    ir = np.arange(1, Nr - 1)
    A[ir] += vip[ir] * r[ir - 1]
    B[ir] += -vip[ir + 1] * r[ir] + vim[ir] * r[ir]
    C[ir] += -vim[ir + 1] * r[ir + 1]

    A[ir] += Di[ir] * g[ir - 1] / w[ir - 1] * r[ir - 1]
    B[ir] += -Di[ir] * g[ir] / w[ir - 1] * r[ir]
    B[ir] += -Di[ir + 1] * g[ir] / w[ir] * r[ir]
    C[ir] += Di[ir + 1] * g[ir + 1] / w[ir] * r[ir + 1]

    A *= Vinv
    B *= Vinv
    C *= Vinv
    return A, B, C


def _modified_rhs_python(dt, rhs, S):
    ret = np.array(rhs, copy=True)
    S_np = np.asarray(to_numpy(S))
    ret[1:-1] += dt * S_np[1:-1]
    return ret


def _modified_rhs_cupy(dt, rhs, S):
    rhs[1:-1] += dt * _field_data(S)[1:-1]
    return rhs


def _modified_jacobian_python(dt, dat, ind, indptr):
    out = -dt * np.array(dat, copy=True)
    Nc = int(indptr.shape[0] - 1)
    for ic in range(Nc):
        start = int(indptr[ic])
        end = int(indptr[ic + 1])
        rows = ind[start:end]
        diag_mask = rows == ic
        if np.any(diag_mask):
            diag_idx = np.nonzero(diag_mask)[0] + start
            out[diag_idx] += 1.0
    return out


def _modified_jacobian_cupy(dt, dat, ind, indptr):
    out = -dt * dat.copy()
    indptr_np = cp.asnumpy(indptr)
    ind_np = cp.asnumpy(ind)
    Nc = int(indptr_np.shape[0] - 1)
    for ic in range(Nc):
        start = int(indptr_np[ic])
        end = int(indptr_np[ic + 1])
        cols = ind_np[start:end]
        diag_idx = np.where(cols == ic)[0]
        if diag_idx.size > 0:
            out[start + int(diag_idx[0])] += 1.0
    return out


_BOUND_BACKEND = None
_K_ENFORCE_FLOOR = None
_K_CS = None
_K_ETA = None
_K_FI = None
_K_HP = None
_K_N = None
_K_P = None
_K_RHO = None
_K_S_HYD = None
_K_TIMESTEP = None
_K_VRAD = None
_K_VVISC = None
_K_IMPLICIT_BOUNDARIES = None
_K_MFP = None
_K_NU = None
_K_S_TOT = None
_K_T_PASS = None
_K_IMPL_1_DIRECT = None
_K_INTERP_TO_INTERFACES_1D = None
_CSR_HOST_META_KEY = None
_CSR_HOST_META_VALUE = None


def _get_csr_host_meta(mat):
    """Cache host CSR pointer/index arrays for stable sparse structures."""
    global _CSR_HOST_META_KEY, _CSR_HOST_META_VALUE
    key = (int(mat.shape[0]), int(mat.shape[1]), int(mat.nnz))
    if _CSR_HOST_META_KEY != key or _CSR_HOST_META_VALUE is None:
        _CSR_HOST_META_KEY = key
        _CSR_HOST_META_VALUE = (cp.asnumpy(mat.indptr), cp.asnumpy(mat.indices))
    return _CSR_HOST_META_VALUE


def _enforce_floor_fortran(sim):
    return _gas_f_call(gas_f.enforce_floor, sim.gas.Sigma, sim.gas.SigmaFloor)


def _enforce_floor_cupy(sim):
    return xp.maximum(sim.gas.Sigma, sim.gas.SigmaFloor)


def _cs_isothermal_fortran(sim):
    return _gas_f_call(gas_f.cs_isothermal, sim.gas.mu, sim.gas.T)


def _cs_isothermal_cupy(sim):
    return xp.sqrt(c.k_B * sim.gas.T / sim.gas.mu)


def _eta_midplane_fortran(sim):
    return _gas_f_call(gas_f.eta_midplane, sim.gas.Hp, sim.gas.P, sim.grid.r, sim.grid.ri)


def _eta_midplane_cupy(sim):
    Hp = _field_data(sim.gas.Hp)
    P = _field_data(sim.gas.P)
    r = _field_data(sim.grid.r)
    ri = _field_data(sim.grid.ri)
    Pi = _interp_to_interfaces_1d(P, r, ri)
    grad = (Pi[1:] - Pi[:-1]) / (ri[1:] - ri[:-1])
    return -0.5 * Hp**2 / (r * P) * grad


def _fi_fortran(sim):
    return _gas_f_call(gas_f.fi, sim.gas.Sigma, sim.gas.v.rad, sim.grid.r, sim.grid.ri)


def _fi_cupy(sim):
    Sigma = _field_data(sim.gas.Sigma)
    v = _field_data(sim.gas.v.rad)
    r = _field_data(sim.grid.r)
    ri = _field_data(sim.grid.ri)
    Nr = int(sim.grid.Nr)

    vi = _interp_to_interfaces_1d(v, r, ri)
    vi[0] = v[0]
    vi[-1] = v[-1]

    vip = xp.maximum(vi, 0.0)
    vim = xp.minimum(vi, 0.0)
    vip[0] = 0.0
    vim[-1] = 0.0

    Fi = xp.zeros((Nr + 1,), dtype=Sigma.dtype)
    Fi[1:-1] = Sigma[:-1] * vip[1:-1] + Sigma[1:] * vim[1:-1]
    Fi[0] = Sigma[0] * vim[0]
    Fi[-1] = Sigma[-1] * vip[-1]
    return Fi


def _hp_fortran(sim):
    return _gas_f_call(gas_f.hp, sim.gas.cs, sim.grid.OmegaK)


def _hp_cupy(sim):
    return sim.gas.cs / sim.grid.OmegaK


def _n_midplane_fortran(sim):
    return _gas_f_call(gas_f.n_midplane, sim.gas.mu, sim.gas.rho)


def _n_midplane_cupy(sim):
    return sim.gas.rho / sim.gas.mu


def _p_midplane_fortran(sim):
    return _gas_f_call(gas_f.p_midplane, sim.gas.cs, sim.gas.rho)


def _p_midplane_cupy(sim):
    return sim.gas.rho * sim.gas.cs**2


def _rho_midplane_fortran(sim):
    return _gas_f_call(gas_f.rho_midplane, sim.gas.Hp, sim.gas.Sigma)


def _rho_midplane_cupy(sim):
    return sim.gas.Sigma / (xp.sqrt(2.0 * c.pi) * sim.gas.Hp)


def _s_hyd_fortran(sim):
    return _gas_f_call(gas_f.s_hyd, sim.gas.Fi, sim.grid.ri)


def _s_hyd_cupy(sim):
    Fi = _field_data(sim.gas.Fi)
    ri = _field_data(sim.grid.ri)
    return 2.0 * (Fi[:-1] * ri[:-1] - Fi[1:] * ri[1:]) / (ri[1:]**2 - ri[:-1]**2)


def _timestep_fortran(sim):
    return _gas_f_call(gas_f.timestep, sim.gas.S.tot, sim.gas.Sigma, sim.gas.SigmaFloor)


def _timestep_cupy(sim):
    S = _field_data(sim.gas.S.tot)
    Sigma = _field_data(sim.gas.Sigma)
    SigmaFloor = _field_data(sim.gas.SigmaFloor)
    mask = (S[1:-1] < 0.0) & (Sigma[1:-1] > SigmaFloor[1:-1])
    if xp.any(mask):
        dt = -Sigma[1:-1][mask] / S[1:-1][mask]
        return xp.min(dt)
    return 1.0e100


def _implicit_boundaries_fortran(sim):
    return _gas_f_call(
        gas_f.implicit_boundaries,
        sim.t.prevstepsize,
        sim.gas.Fi,
        sim.grid.ri,
        sim.gas.Sigma,
        sim.gas._SigmaOld,
    )


def _implicit_boundaries_cupy(sim):
    dt = sim.t.prevstepsize
    Fi = _field_data(sim.gas.Fi)
    ri = _field_data(sim.grid.ri)
    Sigma = _field_data(sim.gas.Sigma)
    SigmaOld = _field_data(sim.gas._SigmaOld)
    ret0 = (Sigma[0] - SigmaOld[0]) / (dt + 1.0e-100)
    ret1 = (Sigma[-1] - SigmaOld[-1]) / (dt + 1.0e-100)
    ret2 = (0.5 * ret0 * (ri[1]**2 - ri[0]**2) + ri[1] * Fi[1]) / ri[0]
    ret3 = (Fi[-2] * ri[-2] - 0.5 * ret1 * (ri[-1]**2 - ri[-2]**2)) / ri[-1]
    return ret0, ret1, ret2, ret3


def _vrad_fortran(sim):
    return _gas_f_call(
        gas_f.v_rad,
        sim.dust.backreaction.A,
        sim.dust.backreaction.B,
        sim.gas.eta,
        sim.grid.OmegaK,
        sim.grid.r,
        sim.gas.v.visc,
        sim.gas.torque.v,
    )


def _vrad_cupy(sim):
    vb = 2.0 * sim.gas.eta * sim.grid.r * sim.grid.OmegaK
    return sim.dust.backreaction.A * sim.gas.v.visc + sim.dust.backreaction.B * vb + sim.gas.torque.v


def _vvisc_fortran(sim):
    return _gas_f_call(gas_f.v_visc, sim.gas.Sigma, sim.gas.nu, sim.grid.r, sim.grid.ri)


def _vvisc_cupy(sim):
    Sigma = _field_data(sim.gas.Sigma)
    nu = _field_data(sim.gas.nu)
    r = _field_data(sim.grid.r)
    ri = _field_data(sim.grid.ri)
    Nr = int(sim.grid.Nr)

    arg = Sigma * nu * xp.sqrt(r)
    argi = _interp_to_interfaces_1d(arg, r, ri)
    grad = (argi[1:] - argi[:-1]) / (ri[1:] - ri[:-1])

    vvisc = -3.0 * grad / (Sigma * xp.sqrt(r))
    if Nr >= 3:
        vvisc[0] = (vvisc[2] - vvisc[1]) * (r[0] - r[1]) / (r[2] - r[1]) + vvisc[1]
        vvisc[-1] = (vvisc[-2] - vvisc[-3]) * (r[-1] - r[-3]) / (r[-2] - r[-3]) + vvisc[-3]
    return vvisc


def _mfp_fortran(sim):
    return _gas_f_call(gas_f.mfp_midplane, sim.gas.n)


def _mfp_cupy(sim):
    return 1.0 / (xp.sqrt(2.0) * c.sigma_H2 * sim.gas.n)


def _nu_fortran(sim):
    return _gas_f_call(gas_f.viscosity, sim.gas.alpha, sim.gas.cs, sim.gas.Hp)


def _nu_cupy(sim):
    return sim.gas.alpha * sim.gas.cs * sim.gas.Hp


def _s_tot_fortran(sim):
    return _gas_f_call(gas_f.s_tot, sim.gas.S.ext, sim.gas.S.hyd)


def _s_tot_cupy(sim):
    s = xp.asarray(_field_data(sim.gas.S.hyd)).copy()
    s[1:-1] = s[1:-1] + _field_data(sim.gas.S.ext)[1:-1]
    return s


def _t_passive_fortran(sim):
    return _gas_f_call(gas_f.t_pass, sim.star.L, sim.grid.r)


def _t_passive_cupy(sim):
    return (6.25e-3 * sim.star.L / (c.pi * sim.grid.r**2 * c.sigma_sb)) ** 0.25


def bind_backend_kernels(backend=None):
    global _BOUND_BACKEND, _K_ENFORCE_FLOOR, _K_CS, _K_ETA, _K_FI, _K_HP
    global _K_N, _K_P, _K_RHO, _K_S_HYD, _K_TIMESTEP, _K_VRAD, _K_VVISC
    global _K_IMPLICIT_BOUNDARIES, _K_MFP, _K_NU, _K_S_TOT, _K_T_PASS, _K_IMPL_1_DIRECT, _K_INTERP_TO_INTERFACES_1D
    global _CSR_HOST_META_KEY, _CSR_HOST_META_VALUE

    backend = get_backend() if backend is None else backend
    if backend == _BOUND_BACKEND and _K_FI is not None:
        return

    _K_ENFORCE_FLOOR = select_backend({"cupy": _enforce_floor_cupy}, backend=backend, default=_enforce_floor_fortran)
    _K_CS = select_backend({"cupy": _cs_isothermal_cupy}, backend=backend, default=_cs_isothermal_fortran)
    _K_ETA = select_backend({"cupy": _eta_midplane_cupy}, backend=backend, default=_eta_midplane_fortran)
    _K_FI = select_backend({"cupy": _fi_cupy}, backend=backend, default=_fi_fortran)
    _K_HP = select_backend({"cupy": _hp_cupy}, backend=backend, default=_hp_fortran)
    _K_N = select_backend({"cupy": _n_midplane_cupy}, backend=backend, default=_n_midplane_fortran)
    _K_P = select_backend({"cupy": _p_midplane_cupy}, backend=backend, default=_p_midplane_fortran)
    _K_RHO = select_backend({"cupy": _rho_midplane_cupy}, backend=backend, default=_rho_midplane_fortran)
    _K_S_HYD = select_backend({"cupy": _s_hyd_cupy}, backend=backend, default=_s_hyd_fortran)
    _K_TIMESTEP = select_backend({"cupy": _timestep_cupy}, backend=backend, default=_timestep_fortran)
    _K_IMPLICIT_BOUNDARIES = select_backend({"cupy": _implicit_boundaries_cupy}, backend=backend, default=_implicit_boundaries_fortran)
    _K_VRAD = select_backend({"cupy": _vrad_cupy}, backend=backend, default=_vrad_fortran)
    _K_VVISC = select_backend({"cupy": _vvisc_cupy}, backend=backend, default=_vvisc_fortran)
    _K_MFP = select_backend({"cupy": _mfp_cupy}, backend=backend, default=_mfp_fortran)
    _K_NU = select_backend({"cupy": _nu_cupy}, backend=backend, default=_nu_fortran)
    _K_S_TOT = select_backend({"cupy": _s_tot_cupy}, backend=backend, default=_s_tot_fortran)
    _K_T_PASS = select_backend({"cupy": _t_passive_cupy}, backend=backend, default=_t_passive_fortran)
    _K_IMPL_1_DIRECT = select_backend({"cupy": _f_impl_1_direct_cupy}, backend=backend, default=_f_impl_1_direct_numpy)
    _K_INTERP_TO_INTERFACES_1D = select_backend({"cupy": _interp_to_interfaces_1d_cupy}, backend=backend, default=_interp_to_interfaces_1d_numpy)

    _CSR_HOST_META_KEY = None
    _CSR_HOST_META_VALUE = None
    _BOUND_BACKEND = backend




def boundary(sim):
    """Function set the boundary conditions of the gas.
    Not implemented, yet.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    sim.gas.boundary.inner.setboundary()
    sim.gas.boundary.outer.setboundary()


def enforce_floor_value(sim):
    """Function enforces floor value to gas surface density.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    sim.gas.Sigma[:] = _K_ENFORCE_FLOOR(sim)


def prepare(sim):
    """Function prepares gas integration step.
    It stores the current value of the surface density in a hidden field.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    # Storing current surface density
    sim.gas._SigmaOld[:] = sim.gas.Sigma[:]


def finalize(sim):
    """Function finalizes gas integration step.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    boundary(sim)
    enforce_floor_value(sim)
    sim.gas.v.update()
    sim.gas.Fi.update()
    sim.gas.S.hyd.update()
    set_implicit_boundaries(sim)


def set_implicit_boundaries(sim):
    """Function calculates the fluxes at the boundaries after the implicit integration step.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    ret = _K_IMPLICIT_BOUNDARIES(sim)

    # Source terms
    sim.gas.S.tot[0] = ret[0]
    sim.gas.S.hyd[0] = ret[0]
    sim.gas.S.tot[-1] = ret[1]
    sim.gas.S.hyd[-1] = ret[1]

    # Fluxes through boundaries
    sim.gas.Fi[0] = ret[2]
    sim.gas.Fi[-1] = ret[3]


def dt(sim):
    """Function calculates the time step from the gas sources.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    dt : float
        Gas time step"""
    return _K_TIMESTEP(sim)


def cs_isothermal(sim):
    """Function calculates the isothermal sound speed of the gas.

    Paramters
    ---------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    cs : Field
        Sound speed"""
    return _K_CS(sim)


def eta_midplane(sim):
    """Function calculates the midplane pressure gradient parameter.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    eta : Field
        eta pressure gradient parameter"""
    return _K_ETA(sim)


def Fi(sim):
    """Function calculates the mass flux through the cell interfaces.
    The fluxes are calculated from the implicit integration outcome.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    Fi : Field
        Mass flux through grid cell interfaces"""
    return _K_FI(sim)


def Hp(sim):
    """Function calculates the pressure scale height.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    Hp : Field
        Pressure scale height"""
    return _K_HP(sim)


def jacobian(sim, x, *args, **kwargs):
    """Functions calculates the Jacobian for the gas.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    x : IntVar
        Integration variable
    args : additional positional arguments
    kwargs : additional keyworda arguments

    Returns
    -------
    jac : Field
        Jacobi matrix for gas evolution

    Notes
    -----
    The boundaries need information about the time step, which is only available at
    the integration stage. The boundary values are therefore meaningless and should not
    be used to calculate the source terms via matrix-vector multiplication.
    See the documentation for details."""

    # Parameters
    nu = sim.gas.nu * sim.dust.backreaction.A
    # Velocity contribution from dust back reaction
    v = sim.dust.backreaction.B * 2. * sim.gas.eta * sim.grid.r * sim.grid.OmegaK
    # Velocity contribution from torque
    v += _field_data(sim.gas.torque.v)

    # Helper variables for convenience
    r = sim.grid.r
    ri = sim.grid.ri
    area = sim.grid.A
    Nr = int(sim.grid.Nr)

    # Construct Jacobian
    if get_backend() == "cupy":
        A, B, C = _jac_abc_cupy(area, nu, r, ri, v)
    else:
        A, B, C = _gas_f_call(gas_f.jac_abc, area, nu, r, ri, v, to_backend_result=False)
    row_hyd = np.hstack(
        (np.arange(Nr-1)+1, np.arange(Nr), np.arange(Nr-1)))
    col_hyd = np.hstack(
        (np.arange(Nr-1), np.arange(Nr), np.arange(Nr-1)+1))
    if get_backend() == "cupy":
        dat_hyd = cp.concatenate((A.ravel()[1:], B.ravel(), C.ravel()[:-1]))
    else:
        dat_hyd = np.hstack((A.ravel()[1:], B.ravel(), C.ravel()[:-1]))

    # Right hand side
    sim.gas._rhs[:] = sim.gas.Sigma

    # Boundaries. This is only reserving space in the sparce matrix
    row_in = [0, 0, 0]
    col_in = [0, 1, 2]
    dat_in = [0., 0., 0.]
    row_out = [Nr-1, Nr-1, Nr-1]
    col_out = [Nr-3, Nr-2, Nr-1]
    dat_out = [0., 0., 0.]

    # Stitching together the generators
    row = np.hstack((row_hyd, row_in, row_out))
    col = np.hstack((col_hyd, col_in, col_out))
    if get_backend() == "cupy":
        dat = cp.concatenate((dat_hyd, cp.asarray(dat_in), cp.asarray(dat_out)))
        J = cp_sparse.coo_matrix(
            (dat, (cp.asarray(row), cp.asarray(col))),
            shape=(Nr, Nr),
        ).tocsr()
    else:
        dat = np.hstack((dat_hyd, dat_in, dat_out))
        gen = (dat, (row, col))
        J = sp.csc_matrix(
            gen,
            shape=(Nr, Nr)
        )

    return J


def lyndenbellpringle1974(r, rc, p, Mdisk):
    """Function calculates the surface density according the self similar solution of Lynden-Bell & Pringle (1974).

    Parameters
    ----------
    r : float or array of floats
        radial distance from star
    rc : float
        critical cutoff radius
    p : float
        power law exponent
    Mdisk : float
        disk mass

    Returns
    -------
    Sigma : float or array of floats
        Surface density profile"""
    return (2+p)*Mdisk / (2.*c.pi*rc**2) * (r/rc)**p * xp.exp(-(r/rc)**(2+p))


def mfp_midplane(sim):
    """Function calculates the midplane mean free path.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    mfp : Field
        Mean free path"""
    return _K_MFP(sim)


def n_midplane(sim):
    """Function calculates the midplane number density of the gas.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    n : Field
        Midplane number density"""
    return _K_N(sim)


def nu(sim):
    """Function calculates the kinematic viscocity of the gas.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    nu : Field
        Kinematic viscosity"""
    return _K_NU(sim)


def P_midplane(sim):
    """Function calculates the midplane gas pressure.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    P : Field
        Midplane pressure"""
    return _K_P(sim)


def rho_midplane(sim):
    """Function calculates the midplane mass density of the gas.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    rho : Field
        Midplane mass density"""
    return _K_RHO(sim)


def S_hyd(sim):
    """Function calculates the hydrodynamic source terms.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    S_hyd : Field
        Hydrodynamic source terms"""
    return _K_S_HYD(sim)


def S_tot(sim):
    """Function calculates the total source terms.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    S_tot : Field
        Total surface density source terms"""
    return _K_S_TOT(sim)


def T_passive(sim):
    """Function calculates the temperature profile of a passively irridiated disk with a constant irradiation
    angle of 0.05.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    T : Field
        Gas temperature"""
    return _K_T_PASS(sim)


def vrad(sim):
    """Function calculates the radial radial gas velocity.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    vrad : Field
        Radial gas velocity"""
    return _K_VRAD(sim)


def vtorque(sim):
    """Function calculates the velocity contribution from a torque profile

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    vtorque : array
        Velocity from torque"""
    return 2. * sim.gas.torque.Lambda / (sim.grid.OmegaK * sim.grid.r)


def vvisc(sim):
    """Function calculates the viscous radial gas velocity.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    vvisc : Field
        Viscous radial gas velocity"""
    return _K_VVISC(sim)


def _f_impl_1_direct_numpy(x0, Y0, dx, *args, **kwargs):
    """Implicit 1st-order Euler integration scheme with direct matrix inversion

    Parameters
    ----------
    x0 : Intvar
        Integration variable at beginning of scheme
    Y0 : Field
        Variable to be integrated at the beginning of scheme
    dx : IntVar
        Stepsize of integration variable
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
    # Getting keyword arguments. Default is standard gas.
    boundary = kwargs.get("boundary", Y0._owner.gas.boundary)
    Sext = kwargs.get("Sext", Y0._owner.gas.S.ext)
    dx = float(to_numpy(dx))

    jac = Y0.jacobian(x0, dx)
    Y0_np = np.asarray(to_numpy(Y0))
    rhs = Y0_np.copy()

    def _boundary_backend_array(boundary_obj, attr):
        arr = getattr(boundary_obj, attr)
        if xp.is_array(arr):
            return _field_data(arr)
        return cp.asarray(arr)

    # Setting boundary values in jac and rhs

    # Inner boundary
    if boundary.inner is not None:
        r_in = _boundary_backend_array(boundary.inner, "_r")
        ri_in = _boundary_backend_array(boundary.inner, "_ri")
        # Given value
        if boundary.inner.condition == "val":
            rhs[0] = boundary.inner.value
        # Constant value
        elif boundary.inner.condition == "const_val":
            jac[0, 1] = 1./dx
            rhs[0] = 0.
        # Given gradient
        elif boundary.inner.condition == "grad":
            K1 = - r_in[1]/r_in[0]
            jac[0, 1] = -K1/dx
            rhs[0] = - ri_in[1]/r_in[0] * \
                (r_in[1]-r_in[0]) * \
                boundary.inner.value
        # Constant gradient
        elif boundary.inner.condition == "const_grad":
            Di = ri_in[1]/ri_in[2] * (
                r_in[1]-r_in[0]) / (r_in[2]-r_in[0])
            K1 = - r_in[1]/r_in[0] * (1. + Di)
            K2 = r_in[2]/r_in[0] * Di
            jac[0, :3] = 0.
            jac[0, 1] = -K1/dx
            jac[0, 2] = -K2/dx
            rhs[0] = 0.
        # Given power law
        elif boundary.inner.condition == "pow":
            p = boundary.inner.value
            rhs[0] = Y0_np[1] * (r_in[0]/r_in[1])**p
        # Constant power law
        elif boundary.inner.condition == "const_pow":
            p = np.log(Y0_np[2] / Y0_np[1]) / \
                np.log(r_in[2]/r_in[1])
            K1 = - (r_in[0]/r_in[1])**p
            jac[0, 1] = -K1/dx
            rhs[0] = 0.

    # Outer boundary
    if boundary.outer is not None:
        r_out = _boundary_backend_array(boundary.outer, "_r")
        ri_out = _boundary_backend_array(boundary.outer, "_ri")
        # Given value
        if boundary.outer.condition == "val":
            rhs[-1] = boundary.outer.value
        # Constant value
        elif boundary.outer.condition == "const_val":
            jac[-1, -2] = (1./dx)
            rhs[-1] = 0.
        # Given gradient
        elif boundary.outer.condition == "grad":
            KNrm2 = - r_out[1]/r_out[0]
            jac[-1, -2] = -(KNrm2/dx)
            rhs[-1] = ri_out[1]/r_out[0] * \
                (r_out[0]-r_out[1]) * \
                boundary.outer.value
        # Constant gradient
        elif boundary.outer.condition == "const_grad":
            Do = ri_out[1]/ri_out[2] * (
                r_out[0]-r_out[1]) / (r_out[1]-r_out[2])
            KNrm2 = - r_out[1]/r_out[0] * (1. + Do)
            KNrm3 = r_out[2]/r_out[0] * Do
            jac[-1, -2] = -KNrm2/dx
            jac[-1, -3] = -KNrm3/dx
            rhs[-1] = 0.
        # Given power law
        elif boundary.outer.condition == "pow":
            p = boundary.outer.value
            rhs[-1] = Y0_np[-2] * (r_out[-0]/r_out[1])**p
        # Constant power law
        elif boundary.outer.condition == "const_pow":
            p = np.log(Y0_np[-2] / Y0_np[-3]) / \
                np.log(r_out[1]/r_out[2])
            KNrm2 = - (r_out[0]/r_out[1])**p
            jac[-1, -2] = -KNrm2/dx
            rhs[-1] = 0.

    # Add external source terms to right-hand side
    rhs[:] = _modified_rhs_python(dx, rhs, Sext)

    jac.data[:] = _modified_jacobian_python(dx, jac.data, jac.indices, jac.indptr)

    Y1 = solve_sparse_linear_system(jac, rhs)

    return to_backend(Y1) - Y0


def _f_impl_1_direct_cupy(x0, Y0, dx, *args, **kwargs):
    """CuPy-only implicit 1st-order Euler integration with GPU sparse solve."""
    boundary = kwargs.get("boundary", Y0._owner.gas.boundary)
    Sext = kwargs.get("Sext", Y0._owner.gas.S.ext)

    jac = Y0.jacobian(x0, dx)
    rhs = cp.asarray(_field_data(Y0)).copy()

    def _boundary_backend_array(boundary_obj, attr):
        arr = getattr(boundary_obj, attr)
        if xp.is_array(arr):
            return _field_data(arr)
        return cp.asarray(arr)

    jac_gpu = jac.tocsr() if isinstance(jac, cp_sparse.spmatrix) else cp_sparse.csr_matrix(jac)
    nrow = int(jac_gpu.shape[0])
    indptr_host, indices_host = _get_csr_host_meta(jac_gpu)

    def _set_csr_row(mat, row, values):
        start = int(indptr_host[row])
        end = int(indptr_host[row + 1])
        cols = indices_host[start:end]
        mat.data[start:end] = 0.0
        for col, val in values.items():
            pos = np.where(cols == col)[0]
            if pos.size > 0:
                mat.data[start + int(pos[0])] = val

    if boundary.inner is not None:
        r_in = _boundary_backend_array(boundary.inner, "_r")
        ri_in = _boundary_backend_array(boundary.inner, "_ri")
        if boundary.inner.condition == "val":
            rhs[0] = boundary.inner.value
        elif boundary.inner.condition == "const_val":
            _set_csr_row(jac_gpu, 0, {1: 1.0 / dx})
            rhs[0] = 0.
        elif boundary.inner.condition == "grad":
            K1 = - (r_in[1] / r_in[0])
            _set_csr_row(jac_gpu, 0, {1: -K1 / dx})
            rhs[0] = - (ri_in[1]/r_in[0] * (r_in[1]-r_in[0])) * boundary.inner.value
        elif boundary.inner.condition == "const_grad":
            Di = (ri_in[1]/ri_in[2] * (r_in[1]-r_in[0]) / (r_in[2]-r_in[0]))
            K1 = - (r_in[1]/r_in[0]) * (1. + Di)
            K2 = (r_in[2]/r_in[0]) * Di
            _set_csr_row(jac_gpu, 0, {1: -K1 / dx, 2: -K2 / dx})
            rhs[0] = 0.
        elif boundary.inner.condition == "pow":
            p = boundary.inner.value
            rhs[0] = rhs[1] * (r_in[0]/r_in[1])**p
        elif boundary.inner.condition == "const_pow":
            p = cp.log(rhs[2] / rhs[1]) / cp.log(r_in[2] / r_in[1])
            K1 = - (r_in[0] / r_in[1])**p
            _set_csr_row(jac_gpu, 0, {1: -K1 / dx})
            rhs[0] = 0.

    if boundary.outer is not None:
        r_out = _boundary_backend_array(boundary.outer, "_r")
        ri_out = _boundary_backend_array(boundary.outer, "_ri")
        if boundary.outer.condition == "val":
            rhs[-1] = boundary.outer.value
        elif boundary.outer.condition == "const_val":
            _set_csr_row(jac_gpu, int(jac_gpu.shape[0] - 1), {int(jac_gpu.shape[0] - 2): 1.0 / dx})
            rhs[-1] = 0.
        elif boundary.outer.condition == "grad":
            KNrm2 = - (r_out[1]/r_out[0])
            _set_csr_row(jac_gpu, int(jac_gpu.shape[0] - 1), {int(jac_gpu.shape[0] - 2): -(KNrm2 / dx)})
            rhs[-1] = (ri_out[1]/r_out[0] * (r_out[0]-r_out[1])) * boundary.outer.value
        elif boundary.outer.condition == "const_grad":
            Do = (ri_out[1]/ri_out[2] * (r_out[0]-r_out[1]) / (r_out[1]-r_out[2]))
            KNrm2 = - (r_out[1]/r_out[0]) * (1. + Do)
            KNrm3 = (r_out[2]/r_out[0]) * Do
            _set_csr_row(
                jac_gpu,
                int(jac_gpu.shape[0] - 1),
                {
                    int(jac_gpu.shape[0] - 2): -KNrm2 / dx,
                    int(jac_gpu.shape[0] - 3): -KNrm3 / dx,
                },
            )
            rhs[-1] = 0.
        elif boundary.outer.condition == "pow":
            p = boundary.outer.value
            rhs[-1] = rhs[-2] * (r_out[-0]/r_out[1])**p
        elif boundary.outer.condition == "const_pow":
            p = cp.log(rhs[-2] / rhs[-3]) / cp.log(r_out[1] / r_out[2])
            KNrm2 = - (r_out[0] / r_out[1])**p
            _set_csr_row(jac_gpu, int(jac_gpu.shape[0] - 1), {int(jac_gpu.shape[0] - 2): -KNrm2 / dx})
            rhs[-1] = 0.

    rhs = _modified_rhs_cupy(dx, rhs, Sext)
    jac_gpu = (-dx) * jac_gpu
    jac_gpu = jac_gpu + cp_sparse.identity(nrow, dtype=jac_gpu.dtype, format="csr")
    Y1 = solve_sparse_linear_system(jac_gpu, rhs)
    return Y1 - _field_data(Y0)


def _f_impl_1_direct(x0, Y0, dx, *args, **kwargs):
    return _K_IMPL_1_DIRECT(x0, Y0, dx, *args, **kwargs)


class impl_1_direct(Scheme):
    """Modified class for implicit gas integration."""

    def __init__(self):
        super().__init__(_f_impl_1_direct, description="Implicit 1st-order direct solver")


# Initialize default bindings at import time after all functions are defined.
bind_backend_kernels()
