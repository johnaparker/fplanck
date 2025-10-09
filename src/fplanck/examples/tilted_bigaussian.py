import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from fplanck.functions import gaussian_pdf
from fplanck.potentials import gaussian_potential
from fplanck.solver import FokkerPlanck
from fplanck.utility import Boundary, combine

nm = 1e-9
viscosity = 8e-4
radius = 50 * nm
drag = 6 * np.pi * viscosity * radius

W = 60 * nm
A = 1.8e-20
U = combine(
    gaussian_potential(center=150 * nm, width=W, amplitude=A),
    gaussian_potential(center=-150 * nm, width=W, amplitude=A),
    lambda x: -2e-14 * x,
)

sim = FokkerPlanck(
    temperature=300,
    drag=drag,
    extent=600 * nm,
    resolution=5 * nm,
    boundary=Boundary.REFLECTING,
    potential=U,
)

### steady-state solution
steady = sim.steady_state()

### time-evolved solution
pdf = gaussian_pdf(-150 * nm, 30 * nm)
p0 = pdf(sim.grid[0])
Nsteps = 200
time, Pt = sim.propagate_interval(pdf, 0.1, Nsteps=Nsteps)

### animation
fig, ax = plt.subplots()

ax.plot(sim.grid[0] / nm, steady, color="k", ls="--", alpha=0.5, label="Steady-state")
ax.plot(sim.grid[0] / nm, p0, color="red", ls="--", alpha=0.3, label="Initial PDF")
(line,) = ax.plot(sim.grid[0] / nm, p0, lw=2, color="C3", label="Time-evolved PDF")


def update(i):
    line.set_ydata(Pt[i])
    return [line]


anim = FuncAnimation(fig, update, frames=range(Nsteps), interval=30)
ax.set(xlabel="x (nm)", ylabel="normalized PDF")
ax.margins(x=0)
ax.legend()

plt.show()
