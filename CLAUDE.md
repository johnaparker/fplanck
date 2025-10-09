# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FPlanck is a Python library for numerically solving the Fokker-Planck partial differential equation (also known as the Smoluchowski equation) in N dimensions using a matrix numerical method. The implementation is based on the paper "Physically consistent numerical solver for time-dependent Fokker-Planck equations" by V. Holubec, K. Kroy, and S. Steffenoni.

## Development Setup

This project uses `uv` for package and dependency management:

- **Install dependencies**: `uv sync`
- **Run tests**: `uv run pytest`
- **Run a specific test**: `uv run pytest tests/test_harmonic_potential.py::test_harmonic_1d_steady_state`
- **Lint with ruff**: `uv run ruff check .`
- **Format with ruff**: `uv run ruff format .`
- **Auto-fix linting issues**: `uv run ruff check --fix .`

The project uses pre-commit hooks (ruff linter and formatter). When developing, ensure ruff formatting and linting passes before committing.

## Core Architecture

### Solver Structure (src/fplanck/solver.py)

The `FokkerPlanck` class is the central solver that:

1. **Builds a spatial grid** based on extent and resolution parameters
2. **Computes transition rates** (Rt and Lt arrays) for each dimension based on:
   - Conservative forces (from potential gradients)
   - Non-conservative forces (from force fields)
   - Diffusion coefficients
   - Boundary conditions
3. **Constructs a master equation matrix** as a sparse matrix representing the time evolution operator
4. **Provides solving methods**:
   - `steady_state()`: computes steady-state probability distribution using eigenvalue decomposition
   - `propagate()`: evolves an initial distribution forward in time using matrix exponential
   - `propagate_interval()`: propagates over a time interval returning distributions at each step
   - `probability_current()`: calculates probability currents from a given distribution

### Key Concepts

**Boundary Conditions** (src/fplanck/utility.py):
- `Boundary.REFLECTING`: particles bounce at edges (transition rates set to 0 at boundaries)
- `Boundary.PERIODIC`: particles wrap around (transition rates connect opposite edges)
- Boundaries can be mixed across different dimensions

**Forces and Potentials**:
- The solver accepts both `potential` (conservative field) and `force` (non-conservative field)
- Potentials are converted to forces via gradient calculation
- Both contribute to the transition rate matrices (Rt and Lt)
- Pre-defined convenience functions in `potentials.py` and `forces.py`

**Probability Functions** (src/fplanck/functions.py):
- Initial conditions are specified as callable functions of grid coordinates
- Common PDFs: `delta_function()`, `gaussian_pdf()`, `uniform_pdf()`
- The `combine()` utility (src/fplanck/utility.py) allows composing multiple functions

### Module Organization

- **src/fplanck/solver.py**: Main `FokkerPlanck` solver class
- **src/fplanck/utility.py**: Boundary enum, vector utilities, function composition
- **src/fplanck/potentials.py**: Pre-defined potential functions (harmonic, gaussian, uniform, from data)
- **src/fplanck/forces.py**: Pre-defined force functions (from data)
- **src/fplanck/functions.py**: Pre-defined probability distribution functions
- **examples/**: Demonstration scripts showing various physical scenarios
- **tests/**: Unit tests validating against analytical solutions

### Testing Strategy

Tests validate numerical solutions against known analytical solutions for:
- 1D harmonic oscillator (steady-state and time-evolved)
- Free space diffusion
- Driven periodic systems
- Uniform force fields

Tests use `np.allclose()` with appropriate tolerances to account for discretization errors. Time limits of propagation should converge to steady-state solutions.

## Important Implementation Details

- The solver uses sparse matrix representation (scipy.sparse.csc_matrix) for memory efficiency
- Transition rates (Rt/Lt) are computed using exponentials of potential differences to ensure physical consistency
- Grid is centered around origin by default (average of each axis is adjusted to zero)
- Resolution can be scalar (same for all dimensions) or vector (different per dimension)
- Temperature, drag, and boundary conditions can also be specified per-dimension
- The master matrix encodes the full time-evolution dynamics of the system
