# Romberg convergence of the GRID weak-form Hessian

[Japanese edition](../docs_jp/grid_wf_hessian_convergence.md)

This experiment compares extrapolated gradients and Hessians with central
finite differences (ε = 0.01–0.05). It uses a two-node gradient model from
0 to 20 km for Love waves at 3 Hz. Node 0 has Vs=200, Vp=400 and ρ=1800.
Node 1, the half-space, has Vs=320, Vp=640 and ρ=2100.

## Gradient

The reference uses a central difference with ε=0.01 and
`compute_dispersion_romberg` with `max_levels=8`.

| Romberg level | Gradient relative error | Improvement factor |
|---:|---:|---:|
| 0 | 4.68e-02 | — |
| 1 | 3.12e-03 | 15.0x |
| 2 | 2.46e-04 | 12.7x |
| 3 | 5.76e-06 | 42.7x |
| 4 | 2.79e-07 | 20.6x |
| 5 | 2.06e-09 | 135.7x |
| 6 | 1.40e-10 | 14.7x |


The relative error is about 2e-9 at the default cap of level 5 and 1e-10 at
level 6 for this model.

## Hessian

The reference is a finite difference of level-7 weak-form gradients with
ε=0.05.

| Romberg level | Hessian relative error | Improvement factor |
|---:|---:|---:|
| 0 | 5.50e-01 | — |
| 1 | 8.86e-02 | 6.2x |
| 2 | 1.88e-02 | 4.7x |
| 3 | 4.63e-05 | 406.7x |
| 4 | 1.84e-05 | 2.5x |
| 5 | 2.16e-07 | 85.4x |
| 6 | 5.85e-08 | 3.7x |


The relative error is about 2e-7 at level 5 and 6e-8 at level 6. The
improvement from level 3 to 4 is smaller than the gain at level 5.

## Comparison with the original checks

The historical scripts are held in the research archive. For checks in this
release, use the [practice examples](../examples/README.md) and the Python
regression suite. The original test used a coarser finite difference (ε=0.5).

| Quantity | Originally reported error | Reference comparison |
|---|---|---|
| Gradient | ~1e-7 | ~1e-7 in the original level-3 check, limited by the ε=0.5 reference |
| Hessian | ~3e-5 | ~2e-7 at level 5 with the finer reference |

The coarse finite difference limited the earlier accuracy assessment. With
the finer reference, level 5 gives about 2e-9 for the gradient and 2e-7 for
the Hessian.

## Parameter settings

| Parameter | Default | High-accuracy setting in this experiment | Interpretation |
|---|---|---|---|
| `max_levels` | 5 | 5–6 | Level 5 gives ~2e-9 gradient error. |
| `tolerance` | 1e-6 | 1e-8 | Threshold for adaptive stopping. |

The gains diminish by level 6, the practical cap in this comparison.

## Interpretation

As the degree of the Neville extrapolation polynomial increases, successive
even-power terms can cancel. This accounts for the uneven improvements,
particularly from level 3 to 4 and from level 4 to 5. Trapezoidal integration
preserves the even-power expansion starting at O(h²) in this analysis, with
Richardson factors of 4^j.
