import pytest
from fplanck import fokker_planck, boundary, k
import numpy as np

def test_uniform_1d_steady_state():
    """the steady state solution should be uniform in 1D"""
    sim = fokker_planck(temperature=1/k, drag=1, extent=1,
                resolution=.1, boundary=boundary.periodic)

    steady = sim.steady_state()
    dp = np.gradient(steady)

    assert np.allclose(dp, 0, atol=1e-15, rtol=0), 'PDF is uniform'
    assert np.allclose(np.sum(steady), 1, atol=1e-15, rtol=0), 'PDF adds to 1'

def test_uniform_2d_steady_state():
    """the steady state solution should be uniform in 2D"""
    sim = fokker_planck(temperature=1/k, drag=1, extent=[1,1],
                resolution=.1, boundary=boundary.periodic)

    steady = sim.steady_state()
    dp = np.gradient(steady)

    assert np.allclose(dp, 0, atol=1e-15, rtol=0), 'PDF is uniform'
    assert np.allclose(np.sum(steady), 1, atol=1e-15, rtol=0), 'PDF adds to 1'
