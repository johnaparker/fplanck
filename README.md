[![Build Status](https://travis-ci.org/johnaparker/fplanck.svg?branch=master)](https://travis-ci.org/johnaparker/fplanck)
[![PyPi Version](https://img.shields.io/pypi/v/fplanck)](https://pypi.org/project/fplanck/)
[![Conda Version](https://img.shields.io/conda/v/japarker/fplanck)](https://anaconda.org/japarker/fplanck)

# FPlanck
FPlanck is a Python library for numerically solving the Fokker-Planck partial differential equation (also known as the Smoluchowski equation) in N dimensions using a matrix numerical method:

<p align="center">
  <img src="https://github.com/johnaparker/fplanck/blob/master/img/fokker_planck.svg">
</p>

The method is based on the paper *"Physically consistent numerical solver for time-dependent Fokker-Planck equations"* by V. Holubec, K. Kroy, and S. Steffenoni, available on [arXiv](https://arxiv.org/pdf/1804.01285.pdf) and published in [APS](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.99.032117).

## Features
+ Can specify an external potential (conservative) and force field (non-conservative) in N-dimensions
+ Solve for the steady-state probability distribution and probability currents
+ Propagate any initial probability distribution to the solution at any later time
+ Periodic and reflecting boundary conditions (can be mixed along different dimensions)

## Installation
FPlanck can be installed with pip
```shell
pip install fplanck
```
or conda
```shell
conda install -c japarker fplanck
```

## Examples
![](https://github.com/johnaparker/fplanck/blob/master/img/ratchet.gif)
![](https://github.com/johnaparker/fplanck/blob/master/img/harmonic.gif)

**On the left**: a single particle in a titled periodic potential with periodic boundary conditions.
The animation shows the time evolution of the probability distribution for the particle location.
The PDF is driven in the positive direction due to the tilted potential.

**On the right**: a single particle in a 2D harmonic potential.
The particle is initially away from the center of the harmonic well, and over time is restored to the center.

## Usage
See the examples folder for how to use FPlanck.

## License
FPlanck is licensed under the terms of the MIT license.


---

#### References
[1] Wikipedia contributors, "Fokker–Planck equation," Wikipedia, The Free Encyclopedia, https://en.wikipedia.org/w/index.php?title=Fokker%E2%80%93Planck_equation&oldid=906834519

[2] Holubec, V., Kroy, K. and Steffenoni, S., 2019. Physically consistent numerical solver for time-dependent Fokker-Planck equations. Physical Review E, 99(3), p.032117.
