import numpy as np
from scipy import constants
from scipy.linalg import expm
from scipy import sparse
from scipy.sparse.linalg import expm, eigs

class fokker_planck:
    def __init__(self, *, temperature, drag, extent, resolution,
            potential=None, force=None, boundary='reflecting'):
        """
        Solve the Fokker-Planck equation

        Arguments:
            temperature     temperature of the surrounding bath
            drag            drag coefficient
            extent          extent (size) of the grid
            resolution      spation resolution of the grid
            potential       external potential function
            force           external force function
            boundary        type of boundary condition (default: reflecting)
        """

        self.temperature = temperature
        self.drag = drag
        self.extent = extent
        self.resolution = resolution
        self.potential = potential
        self.force = force
        self.boundary = boundary

        self.diffusion = constants.k*temperature/drag
        self.mobility = 1/drag
        self.beta = 1/(constants.k*temperature)

        self.Ngrid = int(np.ceil(extent/resolution))
        self.grid = np.arange(self.Ngrid)*resolution
        self.grid -= np.average(self.grid)

        self.Rt = np.zeros_like(self.grid)
        self.Lt = np.zeros_like(self.grid)
        self.potential_values = np.zeros_like(self.grid)
        self.force_values = np.zeros_like(self.grid)

        if self.potential is not None:
            U = self.potential(self.grid)
            self.potential_values += U
            self.force_values += -np.gradient(U, resolution)

            dU = np.roll(U, -1) - U
            self.Rt += self.diffusion/self.resolution**2*np.exp(-self.beta*dU/2)

            dU = np.roll(U, 1) - U
            self.Lt += self.diffusion/self.resolution**2*np.exp(-self.beta*dU/2)

        if self.force is not None:
            F = self.force(self.grid)
            self.force_values += F

            dU = -(np.roll(F, -1) + F)/2*self.resolution
            self.Rt += self.diffusion/self.resolution**2*np.exp(-self.beta*dU/2)

            dU = (np.roll(F, 1) + F)/2*self.resolution
            self.Lt += self.diffusion/self.resolution**2*np.exp(-self.beta*dU/2)

        if self.force is None and self.potential is None:
            self.Rt[...] = self.diffusion/self.resolution**2
            self.Lt[...] = self.diffusion/self.resolution**2

        if boundary is 'reflecting':
            self.Rt[-1] = 0
            self.Lt[0] = 0
        elif boundary == 'periodic':
            dU = -self.force_values[-1]*resolution
            self.Rt[-1] = self.diffusion/self.resolution**2*np.exp(-self.beta*dU/2)

            dU = self.force_values[0]*resolution
            self.Lt[0] = self.diffusion/self.resolution**2*np.exp(-self.beta*dU/2)
        else:
            raise ValueError(f"'{boundary}' is not a valid a boundary condition")

        self._build_matrix()

    def _build_matrix(self):
        """build master equation matrix"""
        UP = self.Lt[1:]
        DIAG = -(self.Lt + self.Rt)
        DOWN = self.Rt[:-1]

        if self.boundary == 'reflecting':
            R = sparse.diags((DOWN, DIAG, UP), offsets=(-1,0,1), format='csc')
        elif self.boundary == 'periodic':
            L = self.Ngrid - 1
            R = sparse.diags((self.Lt[0], DOWN, DIAG, UP, self.Rt[-1]), offsets=(-L,-1,0,1,L), format='csc')

        self.master_matrix = R

    def steady_state(self):
        """Obtain the steady state solution"""
        vals, vecs = eigs(self.master_matrix, k=1, sigma=0, which='LM')
        steady = vecs[:,0].real
        steady /= np.sum(steady)

        return steady

    def propagate(self, initial, time, normalize=True):
        """Propagte an initial probability distribution in time

        Arguments:
            initial      initial probability density function
            time         amount of time to propagate
            normalize    if True, normalize the initial probability
        """
        p0 = initial(self.grid)
        if normalize:
            p0 /= np.sum(p0)

        return expm(self.master_matrix*time) @ p0

    def probability_current(self):
        """Obtain the probability current of the current probability state"""
        steady = self.steady_state()
        J = -(self.diffusion*np.gradient(steady, self.resolution) 
              + self.mobility*self.force_values*steady)

        return J
