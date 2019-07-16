import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from fplanck import fokker_planck

nm = 1e-9
viscosity = 8e-4
radius = 50*nm
drag = 6*np.pi*viscosity*radius

L = 20*nm
F = lambda x: 5e-21*(np.sin(x/L) + 4)/L

sim = fokker_planck(temperature=300, drag=drag, extent=600*nm,
            resolution=10*nm, boundary='periodic', force=F)

steady = sim.steady_state()

w = 30*nm
x0 = -150*nm
Pi = lambda x: np.exp(-(x - x0)**2/w**2)

p0 = Pi(sim.grid[0])
p0 /= np.sum(p0)

fig, ax = plt.subplots()

ax.plot(sim.grid[0]/nm, steady, color='k', ls='--', alpha=.5)
ax.plot(sim.grid[0]/nm, p0, color='red', ls='--', alpha=.3)
line, = ax.plot(sim.grid[0]/nm, p0, lw=2, color='C3')

def update(i):
    time = 3e-6*i
    Pt = sim.propagate(Pi, time)
    line.set_ydata(Pt)

    return [line]

anim = FuncAnimation(fig, update, frames=range(0, 1000, 3), interval=30)
ax.set(xlabel='x (nm)', ylabel='normalized PDF')
ax.margins(x=0)

plt.show()