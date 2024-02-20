import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from fplanck import FokkerPlanck, boundary, gaussian_pdf

nm = 1e-9
viscosity = 8e-4
radius = 50 * nm
drag = 6 * np.pi * viscosity * radius

L = 20 * nm

U = lambda x: 5e-21 * np.cos(x / L)
sim = FokkerPlanck(
    temperature=300,
    drag=drag,
    extent=600 * nm,
    resolution=10 * nm,
    boundary=boundary.reflecting,
    potential=U,
)

### steady-state solution
steady = sim.steady_state()

### time-evolved solution
pdf = gaussian_pdf(-150 * nm, 30 * nm)
p0 = pdf(sim.grid[0])
Nsteps = 200
time, Pt = sim.propagate_interval(pdf, 10e-3, Nsteps=Nsteps)

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
