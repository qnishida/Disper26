# Romberg convergence of the GRID weak-form Hessian

[Japanese edition](../docs_jp/grid_wf_hessian_convergence.md)

This historical experiment used a two-node gradient model (reported depth range
0–20 km) for Love waves at 3 Hz. The recorded (Vs, Vp, ρ) values were
(200, 400, 1800) at the surface and (320, 640, 2100) in the half-space.
The public record omits the material units, whether the central-difference step
ε was absolute or relative, and the norm used for relative error. The tables
give the measurements as reported.

## Gradient

The reference used a central difference with ε=0.01 and a phase-velocity level
cap of 8. The current low-level `compute_dispersion_romberg` takes `max_steps`.

| Romberg level | Gradient relative error | Improvement factor |
|---:|---:|---:|
| 0 | 4.68e-02 | — |
| 1 | 3.12e-03 | 15.0x |
| 2 | 2.46e-04 | 12.7x |
| 3 | 5.76e-06 | 42.7x |
| 4 | 2.79e-07 | 20.6x |
| 5 | 2.06e-09 | 135.7x |
| 6 | 1.40e-10 | 14.7x |

## Hessian

The reference used central differences of level-7 weak-form gradients with ε=0.05.

| Romberg level | Hessian relative error | Improvement factor |
|---:|---:|---:|
| 0 | 5.50e-01 | — |
| 1 | 8.86e-02 | 6.2x |
| 2 | 1.88e-02 | 4.7x |
| 3 | 4.63e-05 | 406.7x |
| 4 | 1.84e-05 | 2.5x |
| 5 | 2.16e-07 | 85.4x |
| 6 | 5.85e-08 | 3.7x |

## Conditions and interpretation

A separate, coarser comparison (ε=0.5) reported a gradient error of about
1e-7 at level 3 and a Hessian error of about 3e-5. The level-3 gradient
error here was 5.76e-6 against a different reference. The two comparisons
do not isolate the contribution of finite-difference error. The largest Hessian
gains occurred from levels 2 to 3 and 4 to 5, with a factor of 2.5 from 3 to 4.
The recorded Hessian settings were `max_levels` of 5–6 and `tolerance` of
1e-6 or 1e-8. For a new model, assess convergence on that model. The
[practice example](../examples/README.md) compares level caps and estimated errors.
