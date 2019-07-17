import numpy as np

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
