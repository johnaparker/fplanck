import numpy as np
import stoked
import matplotlib.pyplot as plt
from functools import partial
from scipy import constants

nm = 1e-9
temperature = 300
dt = 1e-6
Nparticles = 10000
initial = np.zeros([Nparticles,1], dtype=float)
drag = stoked.drag_sphere(50e-9, 8e-4)
k = 1e-6
kT = constants.k*temperature

def harmonic_force(t, rvec, orientation, k=1):
    return -k*rvec

bd = stoked.brownian_dynamics(temperature=temperature, dt=dt, position=initial, drag=drag, 
        force=partial(harmonic_force, k=k))
pos = bd.run(4000).position.squeeze()



fig, ax = plt.subplots()
hist, edges = np.histogram(pos[-1], bins=50, density=True)
ax.plot(edges[1:]/nm, hist)

x = np.linspace(edges[1], edges[-1], 200)
steady = np.sqrt(k/(2*np.pi*kT))*np.exp(-k*x**2/(2*kT))
ax.plot(x/nm, steady)


D = kT/drag.drag_T
tau = 2*kT/(k*D)
tf = bd.time
S = 1 - np.exp(-4*tf/tau)
P = np.sqrt(k/(2*np.pi*kT*S))*np.exp(-k*x**2/(2*kT*S))
ax.plot(x/nm, P)

plt.show()
