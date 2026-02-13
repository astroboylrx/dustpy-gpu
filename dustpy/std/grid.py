'''Module containing standard functions for the grids.'''

import dustpy.constants as c
from simframe.backends.api import xp


def OmegaK(sim):
    """Function calculates the Keplerian frequency.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    OmegaK : Field
        Keplerian frequency"""
    return xp.sqrt(c.G * sim.star.M / sim.grid.r**3)
