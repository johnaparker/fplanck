import numpy as np
from scipy import constants
from scipy.linalg import expm
from scipy import sparse
from scipy.sparse.linalg import expm, eigs, expm_multiply


def value_to_vector(value, ndim, dtype=float):
    """convert a value to a vector in ndim"""

    if np.isscalar(value):
        vec = np.asarray(np.repeat(value, ndim), dtype=dtype)
    else:
        vec = np.asarray(vec, dtype=dtype)
        if vec.size != ndim:
            raise ValueError(f'input vector ({value}) does not have the correct dimensions (ndim = {ndim})')

    return vec

def slice_idx(i, ndim, s0):
    """return a boolean array for a ndim-1 slice along the i'th axis at value s0"""
    idx = [slice(None)]*ndim
    idx[i] = s0

    return tuple(idx)

class fokker_planck:
    def __init__(self, *, temperature, drag, extent, resolution,
            potential=None, force=None, boundary='reflecting'):
        """
        Solve the Fokker-Planck equation

        Arguments:
            temperature     temperature of the surrounding bath (scalar or vector)
            drag            drag coefficient (scalar or vector)
            extent          extent (size) of the grid (vector)
            resolution      spatial resolution of the grid (scalar or vector)
            potential       external potential function, U(ndim -> scalar)
            force           external force function, F(ndim -> ndim)
            boundary        type of boundary condition (default: reflecting)
        """

        self.extent = np.atleast_1d(extent)
        self.ndim = self.extent.size

        self.temperature = value_to_vector(temperature, self.ndim)
        self.drag        = value_to_vector(drag, self.ndim)
        self.resolution  = value_to_vector(resolution, self.ndim)

        self.potential = potential
        self.force = force
        self.boundary = boundary

        self.diffusion = constants.k*self.temperature/self.drag
        self.mobility = 1/self.drag
        self.beta = 1/(constants.k*self.temperature)

        self.Ngrid = np.ceil(self.extent/resolution).astype(int)
        axes = [np.arange(self.Ngrid[i])*self.resolution[i] for i in range(self.ndim)]
        for axis in axes:
            axis -= np.average(axis)
        self.grid = np.array(np.meshgrid(*axes, indexing='ij'))

        self.Rt = np.zeros_like(self.grid)
        self.Lt = np.zeros_like(self.grid)
        self.potential_values = np.zeros_like(self.grid[0])
        self.force_values = np.zeros_like(self.grid)

        if self.potential is not None:
            U = self.potential(*self.grid)
            self.potential_values += U
            self.force_values -= np.gradient(U, *self.resolution)

            for i in range(self.ndim):
                dU = np.roll(U, -1, axis=i) - U
                self.Rt[i] += self.diffusion[i]/self.resolution[i]**2*np.exp(-self.beta[i]*dU/2)

                dU = np.roll(U, 1, axis=i) - U
                self.Lt[i] += self.diffusion[i]/self.resolution[i]**2*np.exp(-self.beta[i]*dU/2)

        if self.force is not None:
            F = np.atleast_2d(self.force(*self.grid))
            self.force_values += F

            for i in range(self.ndim):
                dU = -(np.roll(F[i], -1, axis=i) + F[i])/2*self.resolution[i]
                self.Rt[i] += self.diffusion[i]/self.resolution[i]**2*np.exp(-self.beta[i]*dU/2)

                dU = (np.roll(F[i], 1, axis=i) + F[i])/2*self.resolution[i]
                self.Lt[i] += self.diffusion[i]/self.resolution[i]**2*np.exp(-self.beta[i]*dU/2)

        if self.force is None and self.potential is None:
            for i in range(self.ndim):
                self.Rt[i] = self.diffusion[i]/self.resolution[i]**2
                self.Lt[i] = self.diffusion[i]/self.resolution[i]**2

        if boundary == 'reflecting':
            for i in range(self.ndim):
                idx = slice_idx(i, self.ndim, -1)
                self.Rt[i][idx] = 0

                idx = slice_idx(i, self.ndim, 0)
                self.Lt[i][idx] = 0
        elif boundary == 'periodic':
            for i in range(self.ndim):
                idx = slice_idx(i, self.ndim, -1)
                dU = -self.force_values[i][idx]*self.resolution[i]
                self.Rt[i][idx] = self.diffusion[i]/self.resolution[i]**2*np.exp(-self.beta[i]*dU/2)

                idx = slice_idx(i, self.ndim, 0)
                dU = self.force_values[i][idx]*self.resolution[i]
                self.Lt[i][idx] = self.diffusion[i]/self.resolution[i]**2*np.exp(-self.beta[i]*dU/2)
        else:
            raise ValueError(f"'{boundary}' is not a valid a boundary condition")

        self._build_matrix()

    def _build_matrix(self):
        """build master equation matrix"""
        # UP = self.Lt[1:]
        # DIAG = -(self.Lt + self.Rt)
        # DOWN = self.Rt[:-1]

        # if self.boundary == 'reflecting':
            # R = sparse.diags((DOWN, DIAG, UP), offsets=(-1,0,1), format='csc')
        # elif self.boundary == 'periodic':
            # L = self.Ngrid - 1
            # R = sparse.diags((self.Lt[0], DOWN, DIAG, UP, self.Rt[-1]), offsets=(-L,-1,0,1,L), format='csc')

        R = np.zeros([*self.Ngrid, *self.Ngrid], dtype=float)
        N = np.product(self.Ngrid)

        for i in range(N):
            idx = np.unravel_index(i, self.Ngrid)
            R[idx][idx] = -(np.sum(self.Rt[:,idx]) + np.sum(self.Lt[:,idx]))

            for j in range(self.ndim):
                jdx = list(idx)
                jdx[j] = (jdx[j] + 1) % self.Ngrid[j]
                jdx = tuple(jdx)
                R[idx][jdx] =  self.Lt[j][jdx]

                jdx = list(idx)
                jdx[j] = (jdx[j] - 1) % self.Ngrid[j]
                jdx = tuple(jdx)
                R[idx][jdx] =  self.Rt[j][jdx]

        self.master_matrix = sparse.csc_matrix(R.reshape([N,N]))

    def steady_state(self):
        """Obtain the steady state solution"""
        vals, vecs = eigs(self.master_matrix, k=1, sigma=0, which='LM')
        steady = vecs[:,0].real.reshape(self.Ngrid)
        steady /= np.sum(steady)

        return steady

    def propagate(self, initial, time, normalize=True):
        """Propagte an initial probability distribution in time

        Arguments:
            initial      initial probability density function
            time         amount of time to propagate
            normalize    if True, normalize the initial probability
        """
        p0 = initial(*self.grid)
        if normalize:
            p0 /= np.sum(p0)

        # pf = expm(self.master_matrix*time) @ p0.flatten()
        pf = expm_multiply(self.master_matrix*time, p0.flatten())

        return pf.reshape(self.Ngrid)

    def probability_current(self):
        """Obtain the probability current of the current probability state"""
        steady = self.steady_state()
        J = np.zeros_like(self.force_values)
        for i in range(self.ndim):
            J[i] = -(self.diffusion[i]*np.gradient(steady, self.resolution[i], axis=i) 
                  + self.mobility[i]*self.force_values[i]*steady)

        return J
