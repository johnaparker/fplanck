import pytest
from fplanck import fokker_planck, boundary, k
import numpy as np

def test_uniform_periodic():
    """solution in a uniform force field with periodic boundary conditions"""
    F = lambda x: np.ones_like(x)
    sim = fokker_planck(temperature=1/k, drag=1, extent=1,
                resolution=.05, boundary=boundary.periodic, force=F)

    steady = sim.steady_state()
    dp = np.gradient(steady)

    assert np.allclose(dp, 0, atol=1e-15, rtol=0), 'PDF is uniform'
    assert np.allclose(np.sum(steady), 1, atol=1e-15, rtol=0), 'PDF adds to 1'

    current = sim.probability_current(steady)
    dp = np.gradient(steady)
    assert np.allclose(dp, 0, atol=1e-15, rtol=0), 'probability current is uniform'
