import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from fplanck.functions import delta_function
from fplanck.solver import FokkerPlanck
from fplanck.utility import Boundary

xc = -5


def drag(x):
    A = 1e-16
    return A * ((x - xc) ** 2)


sim = FokkerPlanck(temperature=1, drag=drag, extent=10, resolution=0.05, boundary=Boundary.REFLECTING)

### steady-state solution
steady = sim.steady_state()

### time-evolved solution
pdf = delta_function(xc)
p0 = pdf(sim.grid[0])

Nsteps = 200
time, Pt = sim.propagate_interval(pdf, 1e5, Nsteps=Nsteps)

### animation
fig, ax = plt.subplots()

ax.plot(sim.grid[0], steady, color="k", ls="--", alpha=0.5, label="Steady-state")
ax.plot(sim.grid[0], p0, color="red", ls="--", alpha=0.3, label="Initial PDF")
(line,) = ax.plot(sim.grid[0], p0, lw=2, color="C3", label="Time-evolved PDF")


def update(i):
    line.set_ydata(Pt[i])
    return [line]


anim = FuncAnimation(fig, update, frames=range(Nsteps), interval=30)
ax.set(xlabel="x", ylabel="normalized PDF")
ax.margins(x=0)
ax.set_ylim([0, 0.03])  # ty: ignore[invalid-argument-type]
ax.legend()

plt.show()
