import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from fplanck import fokker_planck, boundary, gaussian_pdf
from mpl_toolkits.mplot3d import Axes3D

nm = 1e-9
viscosity = 8e-4
radius = 50*nm
drag = 6*np.pi*viscosity*radius

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

### time-evolved solution
pdf = gaussian_pdf(center=(-150*nm, -150*nm), width=30*nm)
p0 = pdf(*sim.grid)

Nsteps = 200
time, Pt = sim.propagate_interval(pdf, 20e-3, Nsteps=Nsteps)

### animation
fig = plt.figure(figsize=plt.figaspect(1/2))
ax1 = fig.add_subplot(1,2,1, projection='3d')

surf = ax1.plot_surface(*sim.grid/nm, p0, cmap='viridis')

ax1.set_zlim([0,np.max(Pt)/5])
ax1.autoscale(False)

ax1.set(xlabel='x (nm)', ylabel='y (nm)', zlabel='normalized PDF')

ax2 = fig.add_subplot(1,2,2)

skip = 5
idx = np.s_[::skip, ::skip]
im = ax2.pcolormesh(*sim.grid/nm, p0, vmax=np.max(Pt)/5)
current = sim.probability_current(p0)
arrows = ax2.quiver(sim.grid[0][idx]/nm, sim.grid[1][idx]/nm, 
        current[0][idx], current[1][idx], pivot='mid')

xmax = 400
ax2.set_xlim([-xmax, xmax])
ax2.set_ylim([-xmax, xmax])

def update(i):
    global surf
    surf.remove()
    surf = ax1.plot_surface(*sim.grid/nm, Pt[i], cmap='viridis')

    data = Pt[i, :-1,:-1]
    im.set_array(np.ravel(data))
    im.set_clim(vmax=np.max(data))

    current = sim.probability_current(Pt[i])
    arrows.set_UVC(current[0][idx], current[1][idx])
    return [surf, im, arrows]

anim = FuncAnimation(fig, update, frames=range(Nsteps), interval=30)
plt.tight_layout()

plt.show()
