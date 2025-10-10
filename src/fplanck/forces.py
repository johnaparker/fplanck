"""Pre-defined convenience force functions."""

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator


def force_from_data(grid: npt.ArrayLike, data: npt.ArrayLike) -> npt.ArrayLike:
    """Create a force function from data on a grid.

    Args:
        grid: list of grid arrays along each dimension
        data: force data (shape [ndim, ...])

    Returns:
        force array
    """
    grid = np.asarray(grid)
    if grid.ndim == data.ndim == 1:
        grid = (grid,)

    f = RegularGridInterpolator(grid, np.moveaxis(data, 0, -1), bounds_error=False, fill_value=None)

    def force(*args):
        return np.moveaxis(f(args), -1, 0)

    return force
