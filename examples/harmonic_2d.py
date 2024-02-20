import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from fplanck import FokkerPlanck, boundary, gaussian_pdf, harmonic_potential

nm = 1e-9
viscosity = 8e-4
radius = 50 * nm
drag = 6 * np.pi * viscosity * radius

U = harmonic_potential((0, 0), 1e-6)
sim = FokkerPlanck(
    temperature=300,
    drag=drag,
    extent=[600 * nm, 600 * nm],
    resolution=10 * nm,
    boundary=boundary.reflecting,
    potential=U,
)

### time-evolved solution
pdf = gaussian_pdf(center=(-150 * nm, -150 * nm), width=30 * nm)
p0 = pdf(*sim.grid)

Nsteps = 200
time, Pt = sim.propagate_interval(pdf, 2e-3, Nsteps=Nsteps)

### animation
fig, ax = plt.subplots(subplot_kw=dict(projection="3d"), constrained_layout=True)

surf = ax.plot_surface(*sim.grid / nm, p0, cmap="viridis")

ax.set_zlim([0, np.max(Pt) / 3])
ax.autoscale(False)


def update(i):
    global surf
    surf.remove()
    surf = ax.plot_surface(*sim.grid / nm, Pt[i], cmap="viridis")

    return [surf]


anim = FuncAnimation(fig, update, frames=range(Nsteps), interval=30)
ax.set(xlabel="x (nm)", ylabel="y (nm)", zlabel="normalized PDF")

plt.show()
