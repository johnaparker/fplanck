import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from fplanck.solver import FokkerPlanck
from fplanck.functions import gaussian_pdf
from fplanck.utility import boundary

nm = 1e-9
viscosity = 8e-4
radius = 50 * nm
drag = 6 * np.pi * viscosity * radius

sim = FokkerPlanck(
    temperature=300,
    drag=drag,
    extent=200 * nm,
    resolution=5 * nm,
    boundary=boundary.periodic,
)

### steady-state solution
steady = sim.steady_state()

### time-evolved solution
w = 30 * nm
pdf = gaussian_pdf(0, w)
p0 = pdf(sim.grid[0])

Nsteps = 200
time, Pt = sim.propagate_interval(pdf, 1e-3, Nsteps=Nsteps)

### animation
fig, ax = plt.subplots()

ax.plot(sim.grid[0] / nm, steady, color="k", ls="--", alpha=0.5)
ax.plot(sim.grid[0] / nm, p0, color="red", ls="--", alpha=0.3)
(line,) = ax.plot(sim.grid[0] / nm, p0, lw=2, color="C3")


def update(i):
    line.set_ydata(Pt[i])
    return [line]


anim = FuncAnimation(fig, update, frames=range(Nsteps), interval=30)
ax.set(xlabel="x (nm)", ylabel="normalized PDF")
ax.margins(x=0)

plt.show()
