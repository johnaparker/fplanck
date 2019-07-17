import numpy as np
import enum
from inspect import getfullargspec

def value_to_vector(value, ndim, dtype=float):
    """convert a value to a vector in ndim"""
    value = np.asarray(value, dtype=dtype)
    if value.ndim == 0:
        vec = np.asarray(np.repeat(value, ndim), dtype=dtype)
    else:
        vec = np.asarray(value)
        if vec.size != ndim:
            raise ValueError(f'input vector ({value}) does not have the correct dimensions (ndim = {ndim})')

    return vec

def slice_idx(i, ndim, s0):
    """return a boolean array for a ndim-1 slice along the i'th axis at value s0"""
    idx = [slice(None)]*ndim
    idx[i] = s0

    return tuple(idx)

def combine(*funcs):
    """combine a collection of functions into a single function (for probability, potential, and force functions)"""
    def combined_func(*args):
        values = funcs[0](*args)
        for func in funcs[1:]:
            values += func(*args)

        return values

    return combined_func

class boundary(enum.Enum):
    """enum for the types ofboundary conditions"""
    reflecting = enum.auto()
    periodic   = enum.auto()

def vectorize_force(f):
    """decorator to vectorize a force function"""
    ndim = len(getfullargspec(f).args)
    signature = ','.join(['()']*ndim)
    signature += '->(N)'

    vec_f = np.vectorize(f, signature=signature)
    def new_func(*args):
        return np.rollaxis(vec_f(*args), axis=-1, start=0)

    return new_func
