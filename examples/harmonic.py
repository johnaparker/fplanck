import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from fplanck import fokker_planck, boundary

nm = 1e-9
viscosity = 8e-4
radius = 50*nm
drag = 6*np.pi*viscosity*radius

K = 1e-6
U = lambda x: 0.5*K*x**2
sim = fokker_planck(temperature=300, drag=drag, extent=600*nm,
            resolution=10*nm, boundary=boundary.reflecting, potential=U)

steady = sim.steady_state()

w = 30*nm
x0 = -150*nm
# Pi = lambda x: np.exp(-(x - x0)**2/w**2)
def Pi(x):
    p0 = np.zeros_like(x)
    idx = (x > 100*nm) & (x < 150*nm)
    p0[idx] = 1
    return p0

p0 = Pi(sim.grid[0])
p0 /= np.sum(p0)

fig, ax = plt.subplots()

ax.plot(sim.grid[0]/nm, steady, color='k', ls='--', alpha=.5)
ax.plot(sim.grid[0]/nm, p0, color='red', ls='--', alpha=.3)
line, = ax.plot(sim.grid[0]/nm, p0, lw=2, color='C3')

Nsteps = 200
time, Pt = sim.propagate_interval(Pi, 3e-3, Nsteps=Nsteps)

def update(i):
    line.set_ydata(Pt[i])
    return [line]

anim = FuncAnimation(fig, update, frames=range(Nsteps), interval=30)
ax.set(xlabel='x (nm)', ylabel='normalized PDF')
ax.margins(x=0)

plt.show()
