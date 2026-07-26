from contextlib import nullcontext
from types import SimpleNamespace
import warnings

import numpy as np
from scipy.interpolate import interp1d

from dustpy.utils.backend import to_numpy
from simframe import Frame
from simframe.frame import field_data
from simframe.io.writers import hdf5writer


def read_data(data, filename="data", extension="hdf5", Na=50):
    """
    Function returns a SimpleNamespace with the most useful
    data that can be used for plotting or other purposes.
    This avoids reading the entirety of data files.

    Parameters
    ----------
    data : str | dustpy.Simulation
        Either a path to the data directory or a TriPoD
        simulation frame
    filename : str, optional, default: "data"
        Stem of the data files in the data directory
    extension : str, optional, default: "hdf5"
        File extension of the data file in the data directory

    Returns
    -------
    data : SimpleNamespace
        SimpleNamespace with the extracted data fields
    """

    # Loading data

    if isinstance(data, Frame):
        backend_context = getattr(data, "_backend_context", nullcontext())
        with backend_context:
            # Loading from Simulation object, converting to NumPy, and expanding dimension
            # Simulation
            t = to_numpy(field_data(data.t))[None, ...]
            # Dust
            deltaTurb = to_numpy(field_data(data.dust.delta.turb))[None, ...]
            eps = to_numpy(field_data(data.dust.eps))[None, ...]
            SigmaDust = to_numpy(field_data(data.dust.Sigma))[None, ...]
            St = to_numpy(field_data(data.dust.St))[None, ...]
            vFrag = to_numpy(field_data(data.dust.v.frag))[None, ...]
            # Gas
            cs = to_numpy(field_data(data.gas.cs))[None, ...]
            SigmaGas = to_numpy(field_data(data.gas.Sigma))[None, ...]
            # Grid
            OmegaK = to_numpy(field_data(data.grid.OmegaK))[None, ...]
            m = to_numpy(field_data(data.grid.m))[None, ...]
            r = to_numpy(field_data(data.grid.r))[None, ...]
            ri = to_numpy(field_data(data.grid.ri))[None, ...]
    else:
        # Loading from data directory
        wrtr = hdf5writer(datadir=data, filename=filename, extension=extension)
        # Simulation
        t = wrtr.read.sequence("t")
        # Dust
        deltaTurb = wrtr.read.sequence("dust.delta.turb")
        eps = wrtr.read.sequence("dust.eps")
        SigmaDust = wrtr.read.sequence("dust.Sigma")
        St = wrtr.read.sequence("dust.St")
        vFrag = wrtr.read.sequence("dust.v.frag")
        # Gas
        cs = wrtr.read.sequence("gas.cs")
        SigmaGas = wrtr.read.sequence("gas.Sigma")
        # Grid
        OmegaK = wrtr.read.sequence("grid.OmegaK")
        m = wrtr.read.sequence("grid.m")
        r = wrtr.read.sequence("grid.r")
        ri = wrtr.read.sequence("grid.ri")

    # Computations

    # Helper variables
    Nt, Nr, Nm = SigmaDust.shape
    vK = r*OmegaK

    # Masses
    Mdust = np.pi * ((ri[:, 1:]**2-ri[:, :-1]**2) * SigmaDust.sum(-1)).sum(-1)
    Mgas = np.pi * ((ri[:, 1:]**2-ri[:, :-1]**2) * SigmaGas).sum(-1)

    # Transformation of the density distribution
    a = np.array(np.mean(m[..., 1:] / m[..., :-1], axis=-1))
    dm = np.array(2. * (a - 1.) / (a + 1.))
    sigmaDust = SigmaDust[...] / dm[..., None, None]

    # Size limits
    # Fragmentation limited Stokes number
    b = vFrag**2 / (deltaTurb * cs**2)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            r'invalid value encountered in sqrt')
        StFr = 1 / (2 * b) * (3 - np.sqrt(9 - 4 * b**2))
    # Drift limited Stokes number
    p = SigmaGas * OmegaK * cs / np.sqrt(2.*np.pi)
    StDr = np.zeros_like(StFr)
    for i in range(int(Nt)):
        _f = interp1d(np.log10(r[i, ...]), np.log10(
            p[i, ...]), fill_value='extrapolate')
        pi = 10.**_f(np.log10(ri[i, ...]))
        gamma = np.abs(r[i, ...] / p[i, ...] *
                       np.diff(pi) / np.diff(ri[i, ...]))
        StDr[i, ...] = eps[i, ...] / gamma * (vK[i, ...] / cs[i, ...])**2

    # Assemble return object
    ret = {}
    # Simulation
    ret["t"] = t
    ret["Nt"] = Nt
    # Grid
    ret_grid = {}
    ret_grid["m"] = m
    ret_grid["Nm"] = Nm
    ret_grid["r"] = r
    ret_grid["ri"] = ri
    ret_grid["Nr"] = Nr
    ret_grid["OmegaK"] = OmegaK
    ret["grid"] = SimpleNamespace(**ret_grid)
    # Gas
    ret_gas = {}
    ret_gas["cs"] = cs
    ret_gas["M"] = Mgas
    ret_gas["Sigma"] = SigmaGas
    ret["gas"] = SimpleNamespace(**ret_gas)
    # Dust
    ret_dust = {}
    ret_dust["eps"] = eps
    ret_dust["delta"] = SimpleNamespace(**{"turb": deltaTurb})
    ret_dust["M"] = Mdust
    ret_dust["sigma"] = sigmaDust
    ret_dust["Sigma"] = SigmaDust
    ret_dust["St"] = St
    ret_dust["St_limits"] = SimpleNamespace(**{"drift": StDr, "frag": StFr})
    ret_dust["v"] = SimpleNamespace(**{"frag": vFrag})
    ret["dust"] = SimpleNamespace(**ret_dust)

    return SimpleNamespace(**ret)
