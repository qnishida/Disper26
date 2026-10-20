# Disper26

Disper26 computes Rayleigh and Love surface-wave dispersion in isotropic and
vertically transversely isotropic layered media. It provides C++, C, Fortran
and Python interfaces, automatic-differentiation kernels and phase-velocity
Hessians, and Romberg refinement for gradient models.

## Release status

The current software version is **0.1.104**, a local publication candidate.
The original Disper26 software is licensed under GPL-2.0-or-later. Verified
PREM provenance, its CC BY 4.0 terms, and dependency notices are recorded in
[NOTICE](NOTICE); see [License and attribution](#license-and-attribution).
No public release has been issued.

The reviewed initial history has been synchronized to GitHub. At manuscript
submission, the private repository will become public with its subsequent
commit history preserved, followed by a
versioned GitHub release archived by Zenodo. See the
[release preparation checklist](docs/release_preparation.md).

## Build and use

For a Git clone, initialize the pinned dependency with
`git submodule update --init`. Source archives use the dependency selection
described in [installation](docs/install.md). Requirements are C++17,
CMake 3.18+, Eigen 3.4.x and, for Python, NumPy and pybind11.
Use Python 3.8+ in a dedicated environment; do not install into Conda base.
The commands below use the Python interpreter from that environment. See
[installation](docs/install.md) for environment setup and native dependencies.

```sh
python -m pip install .
# Optional inverse modelling with SciPy:
python -m pip install '.[inversion]'
# Optional numerical and plotting dependencies for paper reproduction:
python -m pip install '.[paper]'
```

```python
import disper26 as d26
model = d26.create_model([0., 10.], kind="pancake",
                         vs=[3., 4.5], vp=[5.2, 8.], rho=[2.5, 3.3])
solution = d26.compute_dispersion(model, period=5.)
print(solution.phase_velocity)
```

Start with the [practice examples](examples/README.md): layered dispersion,
gradient-model convergence, and kernels/Hessian predictions. They use small,
self-contained inputs to introduce selected methods from the paper.
See also [installation](docs/install.md), [shared API](docs/public_api.md),
[low-level API](docs/lowlevel_api.md), and [historical paper reproduction](paper/README.md).
Documented low-level deprecated aliases remain callable. Method-specific
limitations and convergence diagnostics are part of the API contract.

## Verification

```sh
python setup.py build_ext --inplace
# Run just the practice examples, or the full Python regression suite:
python -m pytest -q tests/python/test_examples.py
python -m pytest -q tests/python
cmake -S . -B build -DDISPER26_BUILD_FORTRAN=ON
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

The public tests cover Disper26 and use bundled, self-contained inputs.
External-program comparisons and their reference output are maintained
separately. The tracking fixture contains model nodes only.

## Citation

If you use Disper26, please cite the software version you used. Machine-readable
citation metadata are provided in [CITATION.cff](CITATION.cff). Until the first
public release, identify the source commit used in your work. The publication
date, repository URL, and Zenodo DOI will be added when
confirmed; no software DOI has been assigned yet.

Please also cite the accompanying paper when it is available. Software and
paper citations identify different research contributions. Citation is
requested as scholarly practice and is not an additional license condition.

## License and attribution

Copyright (C) 2026 Kiwamu Nishida.

Except for third-party components with their own notices, the original
Disper26 software is distributed under the **GNU General Public License,
version 2 or, at your option, any later version** (`GPL-2.0-or-later`).
See [LICENSE](LICENSE) for the full terms. The software is provided without
warranty, including any implied warranty of merchantability or fitness for a
particular purpose.

Third-party components retain their original licenses. Bundled PREM tables and
adapted model inputs are redistributed under **CC BY 4.0**, based on
[EarthScope's data policy](https://resess.unavco.org/data/policies_forms/data-policy/data-license.html).
They come from the IRIS EMC anisotropic PREM table now distributed by
EarthScope. See [NOTICE](NOTICE) for source links, verified fingerprints,
model and data-provider credits, and descriptions of changes. The accompanying
manuscript is distributed separately and is not licensed by this software notice.
