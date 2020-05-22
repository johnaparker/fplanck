import numpy as np
from scipy import constants
from scipy.linalg import expm
from scipy import sparse
from scipy.sparse.linalg import expm, eigs, expm_multiply
import enum
import fplanck
from fplanck.utility import value_to_vector, slice_idx 

class fokker_planck:
    def __init__(self, *, temperature, drag, extent, resolution,
            potential=None, force=None, boundary=fplanck.boundary.reflecting):
        """
        Solve the Fokker-Planck equation

        Arguments:
            temperature     temperature of the surrounding bath (scalar or vector)
            drag            drag coefficient (scalar or vector or function)
            extent          extent (size) of the grid (vector)
            resolution      spatial resolution of the grid (scalar or vector)
            potential       external potential function, U(ndim -> scalar)
            force           external force function, F(ndim -> ndim)
            boundary        type of boundary condition (scalar or vector, default: reflecting)
        """

        self.extent = np.atleast_1d(extent)
        self.ndim = self.extent.size

        self.temperature = value_to_vector(temperature, self.ndim)
        self.resolution  = value_to_vector(resolution, self.ndim)

        self.potential = potential
        self.force = force
        self.boundary = value_to_vector(boundary, self.ndim, dtype=object)

        self.beta = 1/(constants.k*self.temperature)

        self.Ngrid = np.ceil(self.extent/resolution).astype(int)
        axes = [np.arange(self.Ngrid[i])*self.resolution[i] for i in range(self.ndim)]
        for axis in axes:
            axis -= np.average(axis)
        self.axes = axes
        self.grid = np.array(np.meshgrid(*axes, indexing='ij'))

        self.Rt = np.zeros_like(self.grid)
        self.Lt = np.zeros_like(self.grid)
        self.potential_values = np.zeros_like(self.grid[0])
        self.force_values = np.zeros_like(self.grid)

        self.drag = np.zeros_like(self.grid)
        self.diffusion = np.zeros_like(self.grid)
        if callable(drag):
            self.drag[...] = drag(*self.grid)
        elif np.isscalar(drag):
            self.drag[...] = drag
        elif isinstance(drag, Iterable) and len(drag) == self.ndim:
            for i in range(self.ndim):
                self.drag[i] = drag[i]
        else:
            raise ValueError(f'drag must be either a scalar, {self.ndim}-dim vector, or a function')

        self.mobility = 1/self.drag
        for i in range(self.ndim):
            self.diffusion[i] = constants.k*self.temperature[i]/self.drag[i]

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

        for i in range(self.ndim):
            if self.boundary[i] == fplanck.boundary.reflecting:
                    idx = slice_idx(i, self.ndim, -1)
                    self.Rt[i][idx] = 0

                    idx = slice_idx(i, self.ndim, 0)
                    self.Lt[i][idx] = 0
            elif self.boundary[i] == fplanck.boundary.periodic:
                    idx = slice_idx(i, self.ndim, -1)
                    dU = -self.force_values[i][idx]*self.resolution[i]
                    self.Rt[i][idx] = self.diffusion[i][idx]/self.resolution[i]**2*np.exp(-self.beta[i]*dU/2)

                    idx = slice_idx(i, self.ndim, 0)
                    dU = self.force_values[i][idx]*self.resolution[i]
                    self.Lt[i][idx] = self.diffusion[i][idx]/self.resolution[i]**2*np.exp(-self.beta[i]*dU/2)
            else:
                raise ValueError(f"'{self.boundary[i]}' is not a valid a boundary condition")

        self._build_matrix()

    def _build_matrix(self):
        """build master equation matrix"""
        N = np.product(self.Ngrid)

        size = N*(1 + 2*self.ndim)
        data = np.zeros(size, dtype=float)
        row  = np.zeros(size, dtype=int)
        col  = np.zeros(size, dtype=int)

        counter = 0
        for i in range(N):
            idx = np.unravel_index(i, self.Ngrid)
            data[counter] = -sum([self.Rt[n][idx] + self.Lt[n][idx]  for n in range(self.ndim)])
            row[counter] = i
            col[counter] = i
            counter += 1

            for n in range(self.ndim):
                jdx = list(idx)
                jdx[n] = (jdx[n] + 1) % self.Ngrid[n]
                jdx = tuple(jdx)
                j = np.ravel_multi_index(jdx, self.Ngrid)

                data[counter] = self.Lt[n][jdx]
                row[counter] = i
                col[counter] = j
                counter += 1

                jdx = list(idx)
                jdx[n] = (jdx[n] - 1) % self.Ngrid[n]
                jdx = tuple(jdx)
                j = np.ravel_multi_index(jdx, self.Ngrid)

                data[counter] = self.Rt[n][jdx]
                row[counter] = i
                col[counter] = j
                counter += 1

        self.master_matrix = sparse.csc_matrix((data, (row, col)), shape=(N,N))

    def steady_state(self):
        """Obtain the steady state solution"""
        vals, vecs = eigs(self.master_matrix, k=1, sigma=0, which='LM')
        steady = vecs[:,0].real.reshape(self.Ngrid)
        steady /= np.sum(steady)

        return steady

    def propagate(self, initial, time, normalize=True, dense=False):
        """Propagate an initial probability distribution in time

        Arguments:
            initial      initial probability density function
            time         amount of time to propagate
            normalize    if True, normalize the initial probability
            dense        if True, use dense method of expm (might be faster, at memory cost)
        """
        p0 = initial(*self.grid)
        if normalize:
            p0 /= np.sum(p0)

        if dense:
            pf = expm(self.master_matrix*time) @ p0.flatten()
        else:
            pf = expm_multiply(self.master_matrix*time, p0.flatten())

        return pf.reshape(self.Ngrid)

    def propagate_interval(self, initial, tf, Nsteps=None, dt=None, normalize=True):
        """Propagate an initial probability distribution over a time interval, return time and the probability distribution at each time-step

        Arguments:
            initial      initial probability density function
            tf           stop time (inclusive)
            Nsteps       number of time-steps (specifiy Nsteps or dt)
            dt           length of time-steps (specifiy Nsteps or dt)
            normalize    if True, normalize the initial probability
        """
        p0 = initial(*self.grid)
        if normalize:
            p0 /= np.sum(p0)

        if Nsteps is not None:
            dt = tf/Nsteps
        elif dt is not None:
            Nsteps = np.ceil(tf/dt).astype(int)
        else:
            raise ValueError('specifiy either Nsteps or Nsteps')

        time = np.linspace(0, tf, Nsteps)
        pf = expm_multiply(self.master_matrix, p0.flatten(), start=0, stop=tf, num=Nsteps, endpoint=True)
        return time, pf.reshape((pf.shape[0],) + tuple(self.Ngrid))

    def probability_current(self, pdf):
        """Obtain the probability current of the given probability distribution"""
        J = np.zeros_like(self.force_values)
        for i in range(self.ndim):
            J[i] = -(self.diffusion[i]*np.gradient(pdf, self.resolution[i], axis=i) 
                  - self.mobility[i]*self.force_values[i]*pdf)

        return J
