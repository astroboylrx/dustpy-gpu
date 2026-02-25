'''Module containing standard functions for the main simulation object.'''

import numpy as np

from dustpy import std
from simframe.backends.api import xp


def dt_adaptive(sim):
    """Function that returns the suggested adaptive timestep.
    By default DustPy uses adaptive integration schemes. The step
    size function is therefore simply returning the suggested
    step size.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    dt : float
        Time step"""
    return sim.t.suggested


def dt(sim):
    """Function returns the timestep depending on the source terms.

    Paramters
    ---------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    dt : float
        Time step"""

    dt_gas = std.gas.dt(sim)
    if dt_gas is None:
        dt_gas = 1.e100
    dt_dust = std.dust.dt(sim)
    if dt_dust is None:
        dt_dust = 1.e100

    # Compute once on backend, convert once to host scalar for control/debug paths.
    dt_host = float(xp.minimum(dt_gas, dt_dust))
    dt_step = sim.t.cfl * dt_host

    if sim.RL_count_cycle % sim.RL_ncycle_out == 0:
        median_dt = float(np.median(sim.RL_recent_dts))
        print(
            f"[RL_debug]: cycle={sim.RL_count_cycle:9d}, t={sim.t/31557600.0:12.3f}yr, "
            f"dt={dt_host/31557600.0:12.6f}yr, <dt>={sim.RL_recent_dts.mean()/31557600.0:12.6f}yr, "
            f"median_dt={median_dt/31557600.0:12.6f}yr"
        )  #, M_pl={sim.planetesimals.M/5.972e27:12.4f}M_e")
    sim.RL_recent_dts[sim.RL_count_cycle % 100] = dt_step
    sim.RL_count_cycle += 1
    return dt_step


def prepare_explicit_dust(sim):
    """This function is the preparation function that is called
    before every integration step.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    std.gas.prepare(sim)


def prepare_implicit_dust(sim):
    """This function is the preparation function that is called
    before every integration step.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    std.gas.prepare(sim)
    std.dust.prepare(sim)


def finalize_explicit_dust(sim):
    """This function is the finalization function that is called
    after every integration step. It is managing the boundary
    conditions and is enforcing floor values.

    Paramters
    ---------
    sim : Frame
        Parent simulation frame"""
    std.gas.finalize(sim)
    std.dust.finalize_explicit(sim)


def finalize_implicit_dust(sim):
    """This function is the finalization function that is called
    after every integration step. It is managing the boundary
    conditions and is enforcing floor values.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    std.gas.finalize(sim)
    std.dust.finalize_implicit(sim)
