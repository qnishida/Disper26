# Disper26 practice examples

These self-contained examples use small models to demonstrate dispersion,
sensitivity kernels and Hessians. Each script includes its inputs.
A [Japanese edition](../example_jp/README.md) is available.

## Setup

Follow the [installation guide](../docs/install.md) to set up a dedicated
Python environment. Run the commands below from the repository root using
that environment. The examples require NumPy and the tests require pytest.

```sh
python -m pip install '.[test]'
```

Rebuild the extension after changing its source. See the [development
instructions](../AGENTS.md) for build and regression commands.

## 1. Model construction and dispersion curves

```sh
python -m examples.basic_dispersion
```

[basic_dispersion.py](basic_dispersion.py) computes fundamental-mode
(`mode=0`) Rayleigh and Love phase velocities for an isotropic layer over a
half-space. The last depth marks the top of the half-space. Depth is in km,
velocity in km/s, density in g/cm³ and period in s. Change the layer thickness
or Vs and compare the short- and long-period ends of each curve.

Run `python -m examples.tracking_options` to set the starting period, initial
velocity range and frequency of mode checks. The default starts at the longest
period and returns results in input order. The [shared API guide](../docs/public_api.md)
describes the checks for each setting.

## 2. Gradient model and convergence diagnostics

```sh
python -m examples.gradient_convergence
```

[gradient_convergence.py](gradient_convergence.py) uses
`interpolation="stiffness"` to interpolate stiffness between depth nodes.
It varies the maximum Romberg level and prints phase velocity, the accepted
level, estimated relative error and convergence status. The small level cap
intentionally shows an unconverged result. The table reports the status, so
the example suppresses duplicate `ConvergenceWarning` messages. The error
estimate comes from extrapolation.

## 3. Sensitivity kernels and the Hessian

```sh
python -m examples.kernels_hessian
```

[kernels_hessian.py](kernels_hessian.py) computes the phase-velocity gradient
and Hessian with `basis="isotropic"`. It selects the upper layer's Vs from the
returned parameter axis, holds Vp and density fixed, and compares first- and
second-order Taylor predictions with a fresh forward calculation. The
improvement shown here applies to small perturbations on the same mode.
The Hessian diagnostics report `not_evaluated` when the method has no
independent error estimate.

## Check the examples

```sh
python -m pytest -q tests/python/test_examples.py
```

These tests run the same calculations and check the velocity ranges, requested
convergence and improvement in the second-order prediction. They also run in
the full Python regression suite. For historical data and figure reproduction,
see the [paper guide](../paper/README.md).

<a id="advanced"></a>

## Advanced usage

The [shared API guide](../docs/public_api.md) describes outputs, derivative
coordinates and supported operations. The [low-level API guide](../docs/lowlevel_api.md)
gives direct calls for individual methods. Inversion has a separate entry
point, `disper26_inv.ForwardEngine`, which also uses SciPy. Install its
dependencies with `python -m pip install '.[inversion]'`. See
[engine.py](../inversion/disper26_inv/engine.py) for the implementation and
[model.py](../inversion/disper26_inv/model.py) for parameterization.
