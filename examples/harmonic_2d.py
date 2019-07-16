import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from fplanck import fokker_planck
from mpl_toolkits.mplot3d import Axes3D

nm = 1e-9
viscosity = 8e-4
radius = 50*nm
drag = 6*np.pi*viscosity*radius

K = 1e-6
U = lambda x, y: 0.5*K*(x**2 + y**2)
sim = fokker_planck(temperature=300, drag=drag, extent=[600*nm, 600*nm],
            resolution=10*nm, boundary='reflecting', potential=U)

steady = sim.steady_state()

w = 30*nm
x0 = -150*nm
y0 = -150*nm
Pi = lambda x, y: np.exp(-((x - x0)**2 + (y - y0)**2)/w**2)
p0 = Pi(*sim.grid)
p0 /= np.sum(p0)

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), constrained_layout=True)

surf = ax.plot_surface(*sim.grid/nm, p0, cmap='viridis')

def update(i):
    global surf

    time = 3e-6*i
    Pt = sim.propagate(Pi, time)
    surf.remove()
    surf = ax.plot_surface(*sim.grid/nm, Pt, cmap='viridis')

    return [surf]

anim = FuncAnimation(fig, update, frames=range(0, 1000, 9), interval=30)
ax.set(xlabel='x (nm)', ylabel='y (nm)', zlabel='normalized PDF')

plt.show()
