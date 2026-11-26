# Disper26 0.1.105

Initial public release of Disper26, a package for computing Rayleigh- and Love-wave dispersion in isotropic and vertically transversely isotropic (VTI) layered media.

## Features

- Compute phase and group velocities, eigenfunctions, and Rayleigh-wave ellipticity.
- Compute sensitivity kernels and phase-velocity Hessians for model analysis and inversion.
- Model depth-dependent material properties using grid interpolation and Romberg refinement.
- Use Disper26 from Python, C++, C, or Fortran. The repository includes documentation and runnable Python examples.

## Numerical fixes included in this release

- Fixed a numerical instability in Love-wave variational kernels and phase-velocity Hessians when the phase velocity approaches the horizontal shear-wave speed of a finite layer.
- Fixed an issue that could return zero sensitivity to half-space properties for weakly decaying Love waves.
