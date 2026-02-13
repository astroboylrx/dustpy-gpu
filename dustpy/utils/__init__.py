'''Package containing utility classes and functions used in the simulation.'''

from dustpy.utils.boundary import Boundary
from dustpy.utils.backend import call_numpy
from dustpy.utils.backend import solve_sparse_linear_system
from dustpy.utils.backend import to_backend
from dustpy.utils.backend import to_numpy
from dustpy.utils.data import read_data
from dustpy.utils.simplenamespace import SimpleNamespace
from dustpy.utils.version import print_version_warning

__all__ = [
    "Boundary",
    "call_numpy",
    "read_data",
    "print_version_warning",
    "solve_sparse_linear_system",
    "SimpleNamespace",
    "to_backend",
    "to_numpy",
]
__version__ = None
