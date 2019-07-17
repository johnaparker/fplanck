import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from fplanck import fokker_planck, boundary, gaussian_pdf, combine
from functools import partial

nm = 1e-9
viscosity = 8e-4
radius = 50*nm
drag = 6*np.pi*viscosity*radius

def gaussian_potential(x, x0, w):
    return -1.8e-20*np.exp(-(x - x0)**2/w**2)

W = 60*nm
U = combine(partial(gaussian_potential, x0=150*nm, w=W),
            partial(gaussian_potential, x0=-150*nm, w=W),
            lambda x: -2e-14*x)

sim = fokker_planck(temperature=300, drag=drag, extent=600*nm,
            resolution=10*nm, boundary=boundary.reflecting, potential=U)

### steady-state solution
steady = sim.steady_state()

### time-evolved solution
pdf = gaussian_pdf(-150*nm, 30*nm)
p0 = pdf(sim.grid[0])
Nsteps = 200
time, Pt = sim.propagate_interval(pdf, 0.1, Nsteps=Nsteps)

### animation
fig, ax = plt.subplots()

ax.plot(sim.grid[0]/nm, steady, color='k', ls='--', alpha=.5)
ax.plot(sim.grid[0]/nm, p0, color='red', ls='--', alpha=.3)
line, = ax.plot(sim.grid[0]/nm, p0, lw=2, color='C3')

def update(i):
    line.set_ydata(Pt[i])
    return [line]

anim = FuncAnimation(fig, update, frames=range(Nsteps), interval=30)
ax.set(xlabel='x (nm)', ylabel='normalized PDF')
ax.margins(x=0)

plt.show()
