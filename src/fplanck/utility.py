"""Utility functions."""

import enum
from collections.abc import Callable
from inspect import getfullargspec

import numpy as np
import numpy.typing as npt


def value_to_vector(value: npt.ArrayLike | float, ndim: int, dtype: type = float):
    """Convert a value to a n-dimensional vector.

    Args:
        value: scalar or array-like value to convert to vector
        ndim: number if dimensions in target vector
        dtype: type of target vector array

    Returns:
        numpy array
    """
    value = np.asarray(value, dtype=dtype)
    if value.ndim == 0:
        vec = np.asarray(np.repeat(value, ndim), dtype=dtype)
    else:
        vec = np.asarray(value)
        if vec.size != ndim:
            raise ValueError(f"input vector ({value}) does not have the correct dimensions (ndim = {ndim})")

    return vec


# TODO: this function seems questionable
def slice_idx(i: int, ndim: int, s0) -> tuple:
    """Return a boolean array for a ndim-1 slice along the i'th axis at value s0."""
    idx = [slice(None)] * ndim
    idx[i] = s0

    return tuple(idx)


# TODO: better name + what if funcs is empty
def combine[T](*funcs: Callable[..., T]) -> Callable[..., T]:
    """Combine a collection of functions into a single function (for probability, potential, and force functions)."""

    def combined_func(*args):
        values = funcs[0](*args)
        for func in funcs[1:]:
            values += func(*args)

        return values

    return combined_func


class Boundary(enum.Enum):
    """Enum for the types of boundary conditions."""

    REFLECTING = enum.auto()
    PERIODIC = enum.auto()


def vectorize_force(f: Callable[[npt.ArrayLike], npt.ArrayLike]) -> Callable[[npt.ArrayLike], npt.ArrayLike]:
    """Decorator to vectorize a force function."""
    ndim = len(getfullargspec(f).args)
    signature = ",".join(["()"] * ndim)
    signature += "->(N)"

    vec_f = np.vectorize(f, signature=signature)

    def new_func(*args):
        return np.rollaxis(vec_f(*args), axis=-1, start=0)

    return new_func
