import pytest
from fplanck import fokker_planck, boundary, k, delta_function, potential_from_data, force_from_data
import numpy as np

K = 1
U = lambda x: 0.5*K*x**2
F = lambda x: -K*x

def test_harmonic_1d_steady_state(plot=False):
    """1D harmonic oscillator steady-state compared to analytic solution"""
    sim = fokker_planck(temperature=1/k, drag=1, extent=10,
                resolution=.1, boundary=boundary.reflecting, potential=U)

    steady = sim.steady_state()

    exact = np.exp(-U(sim.grid[0]))
    exact /= np.sum(exact)

    if not plot:
        assert np.allclose(exact, steady, atol=1e-15)
    else:
        plt.figure()
        plt.plot(steady, label='numerical')
        plt.plot(exact, 'o', label='exact')
        plt.legend()
        plt.title('1D harmonic oscillator: steady state')

def test_harmonic_1d_finite_time(plot=False):
    """1D harmonic oscillator at finite time compared to analytic solution"""
    sim = fokker_planck(temperature=1/k, drag=1, extent=10,
                resolution=.1, boundary=boundary.reflecting, potential=U)

    tf = 4e-1
    x0 = 3
    pdf = sim.propagate(delta_function(x0), tf)

    tau = 2/sim.diffusion[0]
    S = 1 - np.exp(-4*tf/tau)
    exact = np.exp(-U(sim.grid[0] - x0*np.exp(-2*tf/tau))/S)
    exact /= np.sum(exact)

    if not plot:
        assert np.allclose(exact, pdf, atol=2e-3)
    else:
        plt.figure()
        plt.plot(pdf, label='numerical')
        plt.plot(exact, 'o', label='exact')
        plt.legend()
        plt.title('1D harmonic oscillator: finite time')

def test_harmonic_1d_time_limit():
    """propagating by a large amount of time should yield the same solution as the steady-state"""
    sim = fokker_planck(temperature=1/k, drag=1, extent=10,
                resolution=.1, boundary=boundary.reflecting, potential=U)

    steady = sim.steady_state()

    tf = 400
    x0 = 3
    pdf = sim.propagate(delta_function(x0), tf, dense=True)

    assert np.allclose(pdf, steady, atol=1e-12, rtol=0)

def test_harmonic_1d_force_potential():
    """specifying the force should be identical to specifying the potential"""
    sim1 = fokker_planck(temperature=1/k, drag=1, extent=10,
                resolution=.1, boundary=boundary.reflecting, potential=U)
    sim2 = fokker_planck(temperature=1/k, drag=1, extent=10,
                resolution=.1, boundary=boundary.reflecting, force=F)

    steady_1 = sim1.steady_state()
    steady_2 = sim2.steady_state()

    assert np.allclose(steady_1, steady_2, atol=1e-15, rtol=0), 'steady state are equal'

    tf = 4e-1
    x0 = 3
    pdf_1 = sim1.propagate(delta_function(x0), tf, dense=True)
    pdf_2 = sim2.propagate(delta_function(x0), tf, dense=True)

    assert np.allclose(pdf_1, pdf_2, atol=1e-15, rtol=0), 'finite time are equal'

def test_potential_from_data():
    """the harmonic oscillator obtained from potential data vs. from functional"""
    sim1 = fokker_planck(temperature=1/k, drag=1, extent=10,
                resolution=.1, boundary=boundary.reflecting, potential=U)

    x = np.linspace(-5, 5, 100)
    Udata = U(x)
    newU = potential_from_data(x, Udata)

    sim2 = fokker_planck(temperature=1/k, drag=1, extent=10,
                resolution=.1, boundary=boundary.reflecting, potential=newU)

    assert np.allclose(sim1.master_matrix._deduped_data(), sim2.master_matrix._deduped_data(), atol=3e-3, rtol=0)

def test_force_from_data():
    """the harmonic oscillator obtained from force data vs. from functional"""
    sim1 = fokker_planck(temperature=1/k, drag=1, extent=10,
                resolution=.1, boundary=boundary.reflecting, force=F)

    x = np.linspace(-5, 5, 100)
    Fdata = F(x)
    newF = potential_from_data(x, Fdata)

    sim2 = fokker_planck(temperature=1/k, drag=1, extent=10,
                resolution=.1, boundary=boundary.reflecting, force=newF)

    assert np.allclose(sim1.master_matrix._deduped_data(), sim2.master_matrix._deduped_data(), atol=3e-3, rtol=0)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_harmonic_1d_steady_state(True)
    test_harmonic_1d_finite_time(True)
    plt.show()
