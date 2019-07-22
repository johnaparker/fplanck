# FPlanck
FPlanck numerically solves the Fokker-Planck partial differential equation (also known as the Smoluchowski equation) in N dimensions using a matrix numerical method.
The method is based on the paper *"Physically consistent numerical solver for time-dependent Fokker-Planck equations"* by V. Holubec, K. Kroy, and S. Steffenoni, available on [arXiv](https://arxiv.org/pdf/1804.01285.pdf) and published in [APS](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.99.032117).

## Features
+ Can specify an external potential (conservative) and force field (non-conservative) in N-dimensions
+ Solve for the steady-state probability distribution and probability currents
+ Propagate any initial probability distribution to the solution at any later time
+ Periodic and reflecting boundary conditions (can be mixed along different dimensions)

## Usage
See the examples folder for how to use FPlanck.

## Installation
FPlanck can be installed with pip
```shell
pip install fplanck
```
## License
FPlanck is licensed under the terms of the MIT license.


---

#### References
[1] Wikipedia contributors, "Fokker–Planck equation," Wikipedia, The Free Encyclopedia, https://en.wikipedia.org/w/index.php?title=Fokker%E2%80%93Planck_equation&oldid=906834519

[2] Holubec, V., Kroy, K. and Steffenoni, S., 2019. Physically consistent numerical solver for time-dependent Fokker-Planck equations. Physical Review E, 99(3), p.032117.
