"""Pre-defined convenience probability distribution functions."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from fplanck.utility import value_to_vector


def delta_function(r0: npt.ArrayLike) -> npt.ArrayLike:
    """A discrete equivalent of a dirac-delta function centered at r0.

    Args:
        r0: #TODO

    Returns:
        #TODO
    """
    r0 = np.atleast_1d(r0)

    def pdf(*args):
        values = np.zeros_like(args[0])

        diff = np.sum(np.array([(r0[i] - args[i]) ** 2 for i in range(len(args))]), axis=0)
        idx = np.unravel_index(np.argmin(diff), diff.shape)
        values[idx] = 1

        return values

    return pdf


def gaussian_pdf(center: npt.ArrayLike | float, width: npt.ArrayLike | float) -> npt.ArrayLike:
    """A Gaussian probability distribution function.

    Args:
        center: center of Gaussian (scalar or vector)
        width: width of Gaussian (scalar or vector)

    Returns:
        #TODO
    """
    center = np.atleast_1d(center)
    ndim = len(center)
    width = value_to_vector(width, ndim)

    def pdf(*args):
        values = np.ones_like(args[0])

        for i, arg in enumerate(args):
            values *= np.exp(-np.square((arg - center[i]) / width[i]))

        return values / np.sum(values)

    return pdf


def uniform_pdf(func: Callable[..., Any] | None = None) -> npt.ArrayLike:
    """A uniform probability distribution function.

    Args:
        func: a boolean function specifying region of uniform probability (default: everywhere)

    Returns:
        #TODO
    """

    def pdf(*args):
        if func is None:
            values = np.ones_like(args[0])
        else:
            values = np.zeros_like(args[0])
            idx = func(*args)
            values[idx] = 1

        return values / np.sum(values)

    return pdf
