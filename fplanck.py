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
D = kT/drag

X = np.arange(-xmax, xmax + dx, dx)

K = 1e-6
U = 0.5*K*X**2

dU = np.roll(U, -1) - U
Rt = D/dx**2*np.exp(-dU/(2*kT))
Rt[-1] = 0

dU = np.roll(U, 1) - U
Lt = D/dx**2*np.exp(-dU/(2*kT))
Lt[0] = 0

# N x N matrix, N = len(X)
UP = Lt[1:]
DIAG = -(Lt + Rt)
DOWN = Rt[:-1]
R = sparse.diags((DOWN, DIAG, UP), offsets=(-1,0,1), format='csc')

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
ax.plot(X/nm, steady/np.sum(steady), color='k', ls='--')
ax.set(xlabel='x (nm)', ylabel='normalized PDF')


fig, ax = plt.subplots()
from matplotlib.animation import FuncAnimation

ax.plot(X/nm, steady/np.sum(steady), color='k', ls='--')
line, = ax.plot(X/nm, Pi, lw=2, color='C3')
def update(i):
    Dt = 3e-6*i
    Pt = expm(R*Dt) @ Pi
    line.set_ydata(Pt)

    return [line]

anim = FuncAnimation(fig, update, frames=range(0, 1000, 5), interval=30)
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
