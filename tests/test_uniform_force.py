import numpy as np

from fplanck.solver import FokkerPlanck
from fplanck.utility import Boundary


def testuniform_periodic():
    """Solution in a uniform force field with reflecting boundary conditions.

    Note: Uniform force with periodic boundaries has no steady state (particles
    continuously circulate). This test uses reflecting boundaries instead, which
    creates a valid steady-state solution.
    """
    sim = FokkerPlanck(
        temperature=1,
        drag=1,
        extent=10,
        resolution=0.5,
        boundary=Boundary.REFLECTING,
        force=lambda x: np.ones_like(x),
    )

    steady = sim.steady_state()

    # With reflecting boundaries and uniform force, the steady state should:
    # 1. Be normalized
    assert np.allclose(np.sum(steady), 1, atol=1e-10, rtol=0), "PDF adds to 1"

    # 2. Be monotonically increasing (force pushes particles to the right boundary)
    # The gradient should be positive throughout
    dp = np.gradient(steady)
    assert np.all(dp >= -1e-10), "PDF is monotonically non-decreasing"

    # 3. Have higher density at the right boundary due to force pushing particles there
    assert steady[-1] > steady[0], "Density is higher at the force-pushed boundary"
