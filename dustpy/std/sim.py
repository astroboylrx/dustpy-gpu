'''Module containing standard functions for the main simulation object.'''

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

    dt_gas = std.gas.dt(sim) or 1.e100
    dt_dust = std.dust.dt(sim) or 1.e100
    dt = xp.minimum(dt_gas, dt_dust)
    #print(f"[RL_debug]: dt={dt/31557600.0:.6f} yr")
    if sim.RL_count_cycle % sim.RL_ncycle_out == 0:
        median_dt = xp.median(sim.RL_recent_dts)
        print(f"[RL_debug]: cycle={sim.RL_count_cycle:9d}, t={sim.t/31557600.0:12.3f}yr, dt={dt/31557600.0:12.6f}yr, <dt>={sim.RL_recent_dts.mean()/31557600.0:12.6f}yr, median_dt={float(median_dt)/31557600.0:12.6f}yr")  #, M_pl={sim.planetesimals.M/5.972e27:12.4f}M_e")
    sim.RL_recent_dts[sim.RL_count_cycle % 100] = sim.t.cfl * dt
    sim.RL_count_cycle += 1
    return sim.t.cfl * dt


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
