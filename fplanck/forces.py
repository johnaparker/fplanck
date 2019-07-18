"""
pre-defined convenience force functions
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def force_from_data(grid, data):
    """create a force function from data on a grid
    
    Arguments:
        grid     list of grid arrays along each dimension
        data     force data (shape [ndim, ...])
    """
    grid = np.asarray(grid)
    if grid.ndim == data.ndim == 1:
        grid = (grid,)

    f = RegularGridInterpolator(grid, np.moveaxis(data, 0, -1), bounds_error=False, fill_value=None)
    def force(*args):
        return np.moveaxis(f(args), -1, 0)

    return force
