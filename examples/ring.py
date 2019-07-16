import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from fplanck import fokker_planck, boundary
from mpl_toolkits.mplot3d import Axes3D

nm = 1e-9
viscosity = 8e-4
radius = 50*nm
drag = 6*np.pi*viscosity*radius

K = 1e-6
F = lambda x, y: -K*np.array([x,y])

def F(x, y):
    rad = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    L = 200*nm

    Fphi = 1e-12*rad/L*np.exp(-rad/L)
    Frad = 1e-12*(1 - rad/L)*np.exp(-rad/L)

    Fx = -np.sin(phi)*Fphi + np.cos(phi)*Frad
    Fy = np.cos(phi)*Fphi + np.sin(phi)*Frad
    return np.array([Fx, Fy])

sim = fokker_planck(temperature=300, drag=drag, extent=[800*nm, 800*nm],
            resolution=10*nm, boundary=boundary.reflecting, force=F)

w = 30*nm
x0 = -150*nm
y0 = -150*nm
Pi = lambda x, y: np.exp(-((x - x0)**2 + (y - y0)**2)/w**2)
p0 = Pi(*sim.grid)
p0 /= np.sum(p0)

fig = plt.figure(figsize=plt.figaspect(1/2))
ax1 = fig.add_subplot(1,2,1, projection='3d')

surf = ax1.plot_surface(*sim.grid/nm, p0, cmap='viridis')

Nsteps = 200
time, Pt = sim.propagate_interval(Pi, 20e-3, Nsteps=Nsteps)

ax1.set_zlim([0,np.max(Pt)/5])
ax1.autoscale(False)

def update(i):
    global surf
    surf.remove()
    surf = ax1.plot_surface(*sim.grid/nm, Pt[i], cmap='viridis')

    return [surf]

anim = FuncAnimation(fig, update, frames=range(Nsteps), interval=30)
ax1.set(xlabel='x (nm)', ylabel='y (nm)', zlabel='normalized PDF')

steady = sim.steady_state()
current = sim.probability_current()

ax2 = fig.add_subplot(1,2,2)

skip = 5
idx = np.s_[::skip, ::skip]
ax2.pcolormesh(*sim.grid/nm, steady)
ax2.quiver(sim.grid[0][idx]/nm, sim.grid[1][idx]/nm, 
        current[0][idx], current[1][idx], pivot='mid')

xmax = 400
ax2.set_xlim([-xmax, xmax])
ax2.set_ylim([-xmax, xmax])

plt.tight_layout()

plt.show()
