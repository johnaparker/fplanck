"""
pre-defined convenience probability distribution functions
"""
import numpy as np
from fplanck.utility import value_to_vector

def delta_function(r0):
    """a discrete equivalent of a dirac-delta function centered at r0"""
    r0 = np.atleast_1d(r0)

    def pdf(*args):
        values = np.zeros_like(args[0])

        diff = sum([(r0[i] - args[i])**2 for i in range(len(args))])
        idx = np.unravel_index(np.argmin(diff), diff.shape)
        values[idx] = 1

        return values
        
    return pdf

def gaussian_pdf(center, width):
    """A Gaussian probability distribution function

    Arguments:
        center    center of Gaussian (scalar or vector)
        width     width of Gaussian (scalar or vector)
    """

    center = np.atleast_1d(center)
    ndim = len(center)
    width = value_to_vector(width, ndim)

    def pdf(*args):
        values = np.ones_like(args[0])

        for i, arg in enumerate(args):
            values *= np.exp(-np.square((arg - center[i])/width[i]))

        return values/np.sum(values)

    return pdf

def uniform_pdf(func=None):
    """A uniform probability distribution function
    
    Arguments:
        func    a boolean function specifying region of uniform probability (default: everywhere)
    """

    def pdf(*args):
        if func is None:
            values = np.ones_like(args[0])
        else:
            values = np.zeros_like(args[0])
            idx = func(*args)
            values[idx] = 1

        return values/np.sum(values)

    return pdf
