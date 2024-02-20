from my_pytools.my_matplotlib.animation import save_animation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from fplanck import FokkerPlanck, boundary, gaussian_pdf, harmonic_potential

mpl.rc("font", size=15)

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

# time-evolved solution
pdf = gaussian_pdf(center=(-150 * nm, -150 * nm), width=30 * nm)
p0 = pdf(*sim.grid)

Nsteps = 200
time, Pt = sim.propagate_interval(pdf, 2e-3, Nsteps=Nsteps)

# animation
fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

surf = ax.plot_surface(*sim.grid / nm, p0, cmap="viridis")

ax.set_zlim([0, np.max(Pt) / 3])
ax.autoscale(False)


def update(i):
    global surf
    surf.remove()
    surf = ax.plot_surface(*sim.grid / nm, Pt[i], cmap="viridis")

    return [surf]


anim = FuncAnimation(fig, update, frames=range(0, Nsteps, 6), interval=180)
# ax.set(xlabel='x (nm)', ylabel='y (nm)', zlabel='normalized PDF')
ax.set_xlabel("x", labelpad=10)
ax.set_ylabel("y", labelpad=10)
ax.set_zlabel("normalized PDF", labelpad=10)
plt.tight_layout()


save_animation(anim, "harmonic.gif", writer="imagemagick", dpi=50)

plt.show()
