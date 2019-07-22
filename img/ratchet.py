import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from fplanck import fokker_planck, boundary, gaussian_pdf
import matplotlib as mpl

mpl.rc('font', size=15)

nm = 1e-9
viscosity = 8e-4
radius = 50*nm
drag = 6*np.pi*viscosity*radius

L = 20*nm
F = lambda x: 5e-21*(np.sin(x/L) + 4)/L

sim = fokker_planck(temperature=300, drag=drag, extent=600*nm,
            resolution=10*nm, boundary=boundary.periodic, force=F)

### steady-state solution
steady = sim.steady_state()

### time-evolved solution
pdf = gaussian_pdf(-150*nm, 30*nm)
p0 = pdf(sim.grid[0])
Nsteps = 200
time, Pt = sim.propagate_interval(pdf, 2e-3, Nsteps=Nsteps)

### animation
fig, ax = plt.subplots(constrained_layout=True)

ax.plot(sim.grid[0]/nm, p0, color='red', ls='--', alpha=.3, lw=2, label='initial PDF')
ax.plot(sim.grid[0]/nm, steady, color='k', ls='--', alpha=.5, lw=2, label='steady-state')
line, = ax.plot(sim.grid[0]/nm, p0, lw=3, color='C3', label='solution')
ax.legend(loc=1)

def update(i):
    line.set_ydata(Pt[i])
    return [line]

anim = FuncAnimation(fig, update, frames=range(0,Nsteps,6), interval=180)
ax.set(xlabel='x', ylabel='normalized PDF')
ax.margins(x=0)

from my_pytools.my_matplotlib.animation import save_animation
save_animation(anim, 'ratchet.gif', writer='imagemagick', dpi=50)

plt.show()
