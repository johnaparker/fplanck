import numpy as np
import enum

def value_to_vector(value, ndim, dtype=float):
    """convert a value to a vector in ndim"""

    if np.isscalar(value):
        vec = np.asarray(np.repeat(value, ndim), dtype=dtype)
    else:
        vec = np.asarray(vec, dtype=dtype)
        if vec.size != ndim:
            raise ValueError(f'input vector ({value}) does not have the correct dimensions (ndim = {ndim})')

    return vec

def slice_idx(i, ndim, s0):
    """return a boolean array for a ndim-1 slice along the i'th axis at value s0"""
    idx = [slice(None)]*ndim
    idx[i] = s0

    return tuple(idx)

class boundary(enum.Enum):
    reflecting = enum.auto()
    periodic   = enum.auto()
