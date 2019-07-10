import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import constants
from scipy.linalg import expm
from scipy import sparse
from scipy.sparse.linalg import expm, eigs

nm = 1e-9

kT = constants.k*300
drag = 6*np.pi*8e-4*50e-9

xmax = 300*nm
dx = 10*nm
D = kT/drag  # diffusion
mu = 1/drag  # mobility


X = np.arange(-xmax, xmax + dx, dx)

K = 1e-6
U = 0.5*K*X**2
F = -K*X

L = 20*nm
U = 5e-21*np.cos(X/L)
F = 5e-21*np.sin(X/L)/L

F = 5e-21*(np.sin(X/L))/L
F = 5e-21*(np.sin(X/L) + 4)/L

### Conservative forces
# dU = np.roll(U, -1) - U
# Rt = D/dx**2*np.exp(-dU/(2*kT))

# dU = np.roll(U, 1) - U
# Lt = D/dx**2*np.exp(-dU/(2*kT))

### Non-conservative forces
dU = -(np.roll(F, -1) + F)/2*dx
Rt = D/dx**2*np.exp(-dU/(2*kT))

dU = (np.roll(F, 1) + F)/2*dx
Lt = D/dx**2*np.exp(-dU/(2*kT))

### Reflecting boundary condition
Rt[-1] = 0
Lt[0] = 0

### Periodic boundary conditions
dU = -F[-1]*dx
Rt[-1] = D/dx**2*np.exp(-dU/(2*kT))

dU = F[0]*dx
Lt[0] = D/dx**2*np.exp(-dU/(2*kT))

# N x N matrix, N = len(X)
UP = Lt[1:]
DIAG = -(Lt + Rt)
DOWN = Rt[:-1]

# Reflecting boundary conditions
R = sparse.diags((DOWN, DIAG, UP), offsets=(-1,0,1), format='csc')

# Periodic boundary conditions
L = len(X) - 1
R = sparse.diags((Lt[0], DOWN, DIAG, UP, Rt[-1]), offsets=(-L,-1,0,1,L), format='csc')

w = 30*nm 
Pi = np.exp(-(X - 100*nm)**2/w**2)
Pi = np.zeros_like(X)
idx = (X > 60*nm) & (X < 150*nm)
Pi[idx] = 1

Pi /= np.sum(Pi)

fig, ax = plt.subplots()
cmap = mpl.cm.viridis
ax.plot(X/nm, Pi, color=cmap(0))

for N in np.linspace(100, 2000, 5):
    Dt = 1e-6*N
    Pt = expm(R*Dt) @ Pi
    ax.plot(X/nm, Pt, color=cmap(N/2000))
    print(np.sum(Pt))

w, v = eigs(R, k=1, sigma=0, which='LM')
steady = v[:,0].real
steady /= np.sum(steady)
ax.plot(X/nm, steady, color='k', ls='--')
ax.set(xlabel='x (nm)', ylabel='normalized PDF')

fig, ax = plt.subplots()
J = -(D*np.gradient(steady, dx) + mu*F*steady)
ax.plot(X/nm, J)

fig, ax = plt.subplots()
from matplotlib.animation import FuncAnimation

ax.plot(X/nm, steady, color='k', ls='--', alpha=.5)
ax.plot(X/nm, Pi, color='red', ls='--', alpha=.3)
line, = ax.plot(X/nm, Pi, lw=2, color='C3')
def update(i):
    Dt = 3e-6*i
    Pt = expm(R*Dt) @ Pi
    line.set_ydata(Pt)

    return [line]

anim = FuncAnimation(fig, update, frames=range(0, 1000, 3), interval=30)
ax.set(xlabel='x (nm)', ylabel='normalized PDF')

plt.show()


# N = len(X)
# R = np.zeros([N, N], dtype=float)
# for i in range(N):
    # R[i,i] = -(Lt[i] + Rt[i])
    # if i != N -1:
        # R[i,i+1] = Lt[i+1]
    # if i != 0:
        # R[i,i-1] = Rt[i-1]

# w = 30*nm 
# Pi = np.exp(-X**2/w**2)
# Pi /= np.sum(Pi)

# fig, ax = plt.subplots()
# ax.plot(X/nm, Pi)

# for N in np.linspace(100, 2000, 5):
    # Dt = 1e-6*N
    # Pt = expm(R*Dt) @ Pi
    # ax.plot(X/nm, Pt)
    # print(np.sum(Pt))

# w, v = np.linalg.eig(R)
# i = np.argsort(w)[-1]
# steady = v[:,i]
# ax.plot(X/nm, steady/np.sum(steady), color='k')

# plt.show()
