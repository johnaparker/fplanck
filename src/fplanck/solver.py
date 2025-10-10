"""FokkerPlanck solver class."""

from collections.abc import Callable, Iterable

import numpy as np
import numpy.typing as npt
from scipy import constants, sparse
from scipy.sparse.linalg import eigs, expm, expm_multiply

from fplanck.utility import Boundary, slice_idx, value_to_vector


class FokkerPlanck:
    """Class to manage solving the Fokker-Planck equation.

    Attrs:
        temperature: temperature the surrounding bath (scalar or vector)
        drag: drag coefficient (scalar or vector or function)
        extent: extent (size) of the grid (vector)
        resolution: spatial resolution of the grid (scalar or vector)
        potential: external potential function, U(ndim -> scalar)
        force: external force function, F(ndim -> ndim)
        boundary: type of boundary condition (scalar or vector, default: reflecting)
    """

    # Maximum value for exponent arguments to prevent overflow
    _MAX_EXPONENT = 100.0

    def __init__(
        self,
        *,
        temperature: npt.ArrayLike | float,
        drag: npt.ArrayLike | float,
        extent: npt.ArrayLike,
        resolution: npt.ArrayLike | float,
        potential: Callable[[npt.ArrayLike], float] | None = None,
        force: Callable[[npt.ArrayLike], float] | None = None,
        boundary: Boundary = Boundary.REFLECTING,
    ):
        self.extent = np.atleast_1d(extent)
        self.ndim = self.extent.size

        self.temperature = value_to_vector(temperature, self.ndim)
        self.resolution = value_to_vector(resolution, self.ndim)

        self.potential = potential
        self.force = force
        self.boundary = value_to_vector(boundary, self.ndim, dtype=object)

        self.beta = 1 / (constants.k * self.temperature)

        self.Ngrid = np.ceil(self.extent / resolution).astype(int)
        axes = [np.arange(self.Ngrid[i]) * self.resolution[i] for i in range(self.ndim)]
        for axis in axes:
            axis -= np.average(axis)
        self.axes = axes
        self.grid = np.array(np.meshgrid(*axes, indexing="ij"))

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
        elif isinstance(drag, Iterable) and len(drag) == self.ndim:  # ty: ignore[invalid-argument-type]
            for i in range(self.ndim):
                self.drag[i] = drag[i]
        else:
            raise ValueError(f"drag must be either a scalar, {self.ndim}-dim vector, or a function")

        self.mobility = 1 / self.drag
        for i in range(self.ndim):
            self.diffusion[i] = constants.k * self.temperature[i] / self.drag[i]

        if self.potential is not None:
            U = self.potential(*self.grid)
            self.potential_values += U
            self.force_values -= np.gradient(U, *self.resolution)

            for i in range(self.ndim):
                dU = np.roll(U, -1, axis=i) - U
                exponent = np.clip(-self.beta[i] * dU / 2, -self._MAX_EXPONENT, self._MAX_EXPONENT)
                self.Rt[i] += self.diffusion[i] / self.resolution[i] ** 2 * np.exp(exponent)

                dU = np.roll(U, 1, axis=i) - U
                exponent = np.clip(-self.beta[i] * dU / 2, -self._MAX_EXPONENT, self._MAX_EXPONENT)
                self.Lt[i] += self.diffusion[i] / self.resolution[i] ** 2 * np.exp(exponent)

        if self.force is not None:
            F = np.atleast_2d(self.force(*self.grid))
            self.force_values += F

            for i in range(self.ndim):
                dU = -(np.roll(F[i], -1, axis=i) + F[i]) / 2 * self.resolution[i]
                exponent = np.clip(-self.beta[i] * dU / 2, -self._MAX_EXPONENT, self._MAX_EXPONENT)
                self.Rt[i] += self.diffusion[i] / self.resolution[i] ** 2 * np.exp(exponent)

                dU = (np.roll(F[i], 1, axis=i) + F[i]) / 2 * self.resolution[i]
                exponent = np.clip(-self.beta[i] * dU / 2, -self._MAX_EXPONENT, self._MAX_EXPONENT)
                self.Lt[i] += self.diffusion[i] / self.resolution[i] ** 2 * np.exp(exponent)

        if self.force is None and self.potential is None:
            for i in range(self.ndim):
                self.Rt[i] = self.diffusion[i] / self.resolution[i] ** 2
                self.Lt[i] = self.diffusion[i] / self.resolution[i] ** 2

        for i in range(self.ndim):
            if self.boundary[i] == Boundary.REFLECTING:
                idx = slice_idx(i, self.ndim, -1)
                self.Rt[i][idx] = 0

                idx = slice_idx(i, self.ndim, 0)
                self.Lt[i][idx] = 0
            elif self.boundary[i] == Boundary.PERIODIC:
                idx = slice_idx(i, self.ndim, -1)
                dU = -self.force_values[i][idx] * self.resolution[i]
                exponent = np.clip(-self.beta[i] * dU / 2, -self._MAX_EXPONENT, self._MAX_EXPONENT)
                self.Rt[i][idx] = self.diffusion[i][idx] / self.resolution[i] ** 2 * np.exp(exponent)

                idx = slice_idx(i, self.ndim, 0)
                dU = self.force_values[i][idx] * self.resolution[i]
                exponent = np.clip(-self.beta[i] * dU / 2, -self._MAX_EXPONENT, self._MAX_EXPONENT)
                self.Lt[i][idx] = self.diffusion[i][idx] / self.resolution[i] ** 2 * np.exp(exponent)
            else:
                raise ValueError(f"'{self.boundary[i]}' is not a valid a boundary condition")

        self._build_matrix()

    def _build_matrix(self):
        """Build master equation matrix."""
        N = np.prod(self.Ngrid)

        size = N * (1 + 2 * self.ndim)
        data = np.zeros(size, dtype=float)
        row = np.zeros(size, dtype=int)
        col = np.zeros(size, dtype=int)

        counter = 0
        for i in range(N):
            idx = np.unravel_index(i, self.Ngrid)
            data[counter] = -sum([self.Rt[n][idx] + self.Lt[n][idx] for n in range(self.ndim)])
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

        self.master_matrix = sparse.csc_matrix((data, (row, col)), shape=(N, N))

    def steady_state(self) -> npt.ArrayLike:
        """Obtain the steady state solution.

        Returns:
            steady state vectors

        Raises:
            RuntimeError: If the master matrix is singular and no steady state exists.
                This can occur with configurations like uniform force + periodic boundaries.
        """
        try:
            # Try shift-invert mode (sigma=0) to find eigenvalues near zero
            vals, vecs = eigs(self.master_matrix, k=1, sigma=0, which="LM")
        except RuntimeError as e:
            if "singular" in str(e).lower():
                # Matrix is singular - try finding smallest magnitude eigenvalue without shift-invert
                try:
                    vals, vecs = eigs(self.master_matrix, k=2, which="SM")
                    # Use the eigenvector with the smallest eigenvalue
                    idx = np.argmin(np.abs(vals))
                    vecs = vecs[:, idx : idx + 1]
                except Exception:
                    # If that also fails, provide a helpful error message
                    raise RuntimeError(
                        "Unable to find steady state. This may occur with physically "
                        "invalid configurations (e.g., uniform force with periodic boundaries "
                        "has no equilibrium steady state)."
                    ) from e
            else:
                raise

        steady = vecs[:, 0].real.reshape(self.Ngrid)
        steady /= np.sum(steady)

        return steady

    def propagate(self, initial, time: float, normalize: bool = True, dense: bool = False) -> npt.ArrayLike:
        """Propagate an initial probability distribution in time.

        Args:
            initial: initial probability density function
            time: amount of time to propagate
            normalize: if True, normalize the initial probability
            dense: if True, use dense method of expm (might be faster, at memory cost)

        Returns:
            probability function at later time
        """
        p0 = initial(*self.grid)
        if normalize:
            p0 /= np.sum(p0)

        if dense:
            pf = expm(self.master_matrix * time) @ p0.flatten()
        else:
            pf = expm_multiply(self.master_matrix * time, p0.flatten())

        return pf.reshape(self.Ngrid)

    def propagate_interval(
        self,
        initial,
        tf: float,
        Nsteps: int | None = None,
        dt: float | None = None,
        normalize: bool = True,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Propagate an initial probability distribution over a time interval.

        Args:
            initial: initial probability density function
            tf: stop time (inclusive)
            Nsteps: number of time-steps (specify Nsteps or dt)
            dt: length of time-steps (specify Nsteps or dt)
            normalize: if True, normalize the initial probability

        Returns:
            (time array, probability distribution at each time step)
        """
        p0 = initial(*self.grid)
        if normalize:
            p0 /= np.sum(p0)

        if Nsteps is not None:
            dt = tf / Nsteps
        elif dt is not None:
            Nsteps = np.ceil(tf / dt).astype(int)
        else:
            raise ValueError("specify either Nsteps or Nsteps")

        time = np.linspace(0, tf, Nsteps)
        pf = expm_multiply(
            self.master_matrix,
            p0.flatten(),
            start=0,
            stop=tf,
            num=Nsteps,
        )
        return time, pf.reshape((pf.shape[0],) + tuple(self.Ngrid))

    def probability_current(self, pdf) -> npt.ArrayLike:
        """Obtain the probability current of the given probability distribution.

        Args:
            pdf: probability distribution function

        Returns:
            Probability currents
        """
        J = np.zeros_like(self.force_values)
        for i in range(self.ndim):
            J[i] = -(
                self.diffusion[i] * np.gradient(pdf, self.resolution[i], axis=i)
                - self.mobility[i] * self.force_values[i] * pdf
            )

        return J
