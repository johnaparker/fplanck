import numpy as np

from fplanck.solver import FokkerPlanck
from fplanck.utility import Boundary


def testuniform_periodic():
    """Solution in a uniform force field with periodic boundary conditions."""
    sim = FokkerPlanck(
        temperature=1 / k,
        drag=1,
        extent=1,
        resolution=0.05,
        boundary=Boundary.PERIODIC,
        force=lambda x: np.ones_like(x),
    )

    steady = sim.steady_state()
    dp = np.gradient(steady)

    assert np.allclose(dp, 0, atol=1e-15, rtol=0), "PDF is uniform"
    assert np.allclose(np.sum(steady), 1, atol=1e-15, rtol=0), "PDF adds to 1"

    current = sim.probability_current(steady)
    dp = np.gradient(steady)
    assert np.allclose(dp, 0, atol=1e-15, rtol=0), "probability current is uniform"
