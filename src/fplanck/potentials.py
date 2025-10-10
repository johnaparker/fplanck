"""pre-defined convenience potential functions."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator

from fplanck.utility import value_to_vector


def harmonic_potential(center: npt.ArrayLike | float, k: npt.ArrayLike | float):
    """Return a harmonic potential.

    Args:
        center: center of harmonic potential (scalar or vector)
        k: spring constant of harmonic potential (scalar or vector)

    Returns:
        #TODO
    """
    center = np.atleast_1d(center)
    ndim = len(center)
    k = value_to_vector(k, ndim)

    def potential(*args):
        U = np.zeros_like(args[0])

        for i, arg in enumerate(args):
            U += 0.5 * k[i] * (arg - center[i]) ** 2

        return U

    return potential


def gaussian_potential(center: npt.ArrayLike | float, width: npt.ArrayLike | float, amplitude: float):
    """Return a Gaussian potential.

    Args:
        center: center of Gaussian (scalar or vector)
        width: width of Gaussian  (scalar or vector)
        amplitude: amplitude of Gaussian, (negative for repulsive)

    Returns:
        #TODO
    """
    center = np.atleast_1d(center)
    ndim = len(center)
    width = value_to_vector(width, ndim)

    def potential(*args):
        U = np.ones_like(args[0])

        for i, arg in enumerate(args):
            U *= np.exp(-np.square((arg - center[i]) / width[i]))

        return -amplitude * U

    return potential


def uniform_potential(func: Callable, U0: float):
    """Return a uniform potential.

    Args:
        func: a boolean function specifying region of uniform probability (default: everywhere)
        U0: value of the potential

    Returns:
        #TODO
    """

    def potential(*args):
        U = np.zeros_like(args[0])
        idx = func(*args)
        U[idx] = U0

        return U

    return potential


def potential_from_data(grid: npt.ArrayLike, data: npt.ArrayLike) -> npt.ArrayLike:
    """Create a potential from data on a grid.

    Args:
        grid: list of grid arrays along each dimension
        data: potential data

    Returns:
        #TODO
    """
    grid = np.asarray(grid)
    if grid.ndim == data.ndim == 1:
        grid = (grid,)

    f = RegularGridInterpolator(grid, data, bounds_error=False, fill_value=None)

    def potential(*args):
        return f(args)

    return potential
