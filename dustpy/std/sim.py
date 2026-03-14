'''Module containing standard functions for the main simulation object.'''

import os
import numpy as np
import time

from dustpy import std
from simframe.backends.api import xp


def _field_data(value):
    return value._data if hasattr(value, "_data") else value


def _to_numpy(value):
    if hasattr(value, "get"):
        return value.get()
    if hasattr(xp, "to_numpy"):
        return xp.to_numpy(value)
    return np.asarray(value)


def _env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    raw = raw.strip().lower()
    if raw in ("1", "true", "on", "yes"):
        return True
    if raw in ("0", "false", "off", "no"):
        return False
    return bool(default)


_DT_GAS_REFRESH_ENABLED = _env_bool("DUSTPY_DT_GAS_REFRESH_ENABLE", default=False)


def is_dt_gas_refresh_enabled():
    return _DT_GAS_REFRESH_ENABLED


def _ensure_rl_debug_state(sim):
    if not hasattr(sim, "RL_recent_dts"):
        sim.RL_recent_dts = np.zeros(100)
    if not hasattr(sim, "RL_recent_wts"):
        sim.RL_recent_wts = np.zeros(100)
    if not hasattr(sim, "RL_last_wall_s"):
        sim.RL_last_wall_s = None


def _rl_debug_line(sim, dt_step):
    _ensure_rl_debug_state(sim)
    median_dt = float(np.median(sim.RL_recent_dts))
    mean_dt = float(np.mean(sim.RL_recent_dts))
    wt_100 = float(np.sum(sim.RL_recent_wts))
    dust = getattr(sim, "dust", None)
    nfloor_retry = 0
    dM_san_mearth = 0.0
    if dust is not None and hasattr(dust, "floor_retry"):
        nfloor_retry = int(dust.floor_retry.count)
    if dust is not None and hasattr(dust, "san"):
        dM_san_mearth = float(dust.san.dM_total_mearth)
    return (
        f"[RL_debug]: cycle={sim.RL_count_cycle:9d}, t={sim.t/31557600.0:12.3f}yr, "
        f"dt={dt_step/31557600.0:.4e}/{mean_dt/31557600.0:.4e}/{median_dt/31557600.0:.4e}yr, "
        f"n_dt↘={nfloor_retry:6d}, dM_M⊕={dM_san_mearth:.4e}, wt={wt_100:.4e}"
    )


def _record_cycle_wall(sim):
    _ensure_rl_debug_state(sim)
    wall_now = time.perf_counter()
    wall_prev = getattr(sim, "RL_last_wall_s", None)
    if wall_prev is not None and sim.RL_count_cycle > 0:
        sim.RL_recent_wts[(sim.RL_count_cycle - 1) % 100] = wall_now - wall_prev
    sim.RL_last_wall_s = wall_now


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

    _record_cycle_wall(sim)

    # Gas dt relies on retrospective operator diagnostics. Refresh them from the
    # current coupled gas+dust state so the limiter does not read stale fields.
    if _DT_GAS_REFRESH_ENABLED:
        _refresh_coupled_gas_state(sim)

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
        print(_rl_debug_line(sim, dt_step))
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


def _refresh_coupled_gas_state(sim):
    """Refresh dust-coupled gas operator fields after dust state changes."""
    sim.dust.rho.update()
    sim.dust.eps.update()
    sim.dust.backreaction.update()
    sim.gas.v.update()
    sim.gas.Fi.update()
    sim.gas.S.hyd.update()
    sim.gas.S.tot.update()
    std.gas.set_implicit_boundaries(sim)


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
