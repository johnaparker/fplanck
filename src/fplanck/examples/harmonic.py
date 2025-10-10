import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from fplanck.functions import uniform_pdf
from fplanck.potentials import harmonic_potential
from fplanck.solver import FokkerPlanck
from fplanck.utility import Boundary

nm = 1e-9
viscosity = 8e-4
radius = 50 * nm
drag = 6 * np.pi * viscosity * radius

U = harmonic_potential((0, 0), 1e-6)
sim = FokkerPlanck(
    temperature=300,
    drag=drag,
    extent=600 * nm,
    resolution=2 * nm,
    boundary=Boundary.REFLECTING,
    potential=U,
)

# steady-state solution
steady = sim.steady_state()

# time-evolved solution
pdf = uniform_pdf(lambda x: (x > 100 * nm) & (x < 150 * nm))
p0 = pdf(sim.grid[0])
Nsteps = 200
time, Pt = sim.propagate_interval(pdf, 3e-3, Nsteps=Nsteps)

# animation
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
