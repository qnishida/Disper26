# Manuscript reproduction

For hands-on practice with selected methods from the paper, start with the
[examples](../examples/README.md). Their tests are part of the normal Python
regression suite; reproducing all manuscript figures is not required.
This directory supports optional historical reproduction from archived evidence.
It contains Disper26 results. External-program comparison tests, outputs, and
comparison plots are maintained separately and are outside the public package.

This directory contains fixed numerical evidence, plotting scripts, and
source-only snapshots needed to reproduce historical calculations. Snapshots
contain no old Git repository. A source identifier and SHA-256 manifest replace
the assumption that an old private commit can be checked out in this repository.
Historical commit identifiers retained inside scientific artifacts identify
their original provenance; they are not commits in this public history.

## Current manuscript figures

Run these commands from the public repository root in the paper environment
with the current public extension built. The long-period experiment uses the
archived `plan_l_v2` solver, with Rayleigh modulus exponent 0 and Love exponent 3.
Verify the snapshot before building it from inside its directory.

```sh
python -c "from paper.provenance import verify_reference; verify_reference('plan_l_v2', 'paper/reference_sources/plan_l_v2')"
(cd paper/reference_sources/plan_l_v2 && python setup.py build_ext --inplace)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/recompute_plan_l_eft.py all --output-dir paper/results/manuscript/eft
python paper/scripts/plot_results.py --plan-l-data paper/results/manuscript/eft
```

For the combined depth-sensitivity figure, both waves use modulus exponent 2.
The archived group-velocity kernels are bundled. Recompute the signed
ellipticity sensitivities with the current public solver and plot the six panels.

```sh
python paper/scripts/generate_ellipticity_figure.py --disper26-root . \
  --model paper/data/results/prem_like_kernel_model.json --output-dir paper/results/manuscript/ellipticity
python paper/scripts/plot_depth_sensitivities.py
```

To recompute the group-velocity kernels as well, build the archived extension
and its quadrature diagnostic. Set the Eigen include path for your installation.

```sh
snapshot=paper/reference_sources/kernel_figures
python -c "from paper.provenance import verify_reference; verify_reference('kernel_figures', 'paper/reference_sources/kernel_figures')"
(cd "$snapshot" && python setup.py build_ext --inplace)
mkdir -p paper/results/manuscript/kernels
g++ -std=c++17 -O2 -DNDEBUG -I"$snapshot/src" -I"$snapshot/third_party/autodiff" -I/usr/include/eigen3 \
  paper/scripts/continuous_group_kernel.cpp -o paper/results/manuscript/kernels/diagnostic
python paper/scripts/generate_kernel_figures.py --only kernels --disper26-root "$snapshot" \
  --diagnostic paper/results/manuscript/kernels/diagnostic --output-dir paper/results/manuscript/kernels
python paper/scripts/plot_depth_sensitivities.py --kernel-data paper/results/manuscript/kernels
```

The four figures are written as PDF and PNG under `paper/Fig/`. The runtime
figure uses timings from the machine running the benchmark.

## Fixed statistics and plots

From the repository root, with NumPy, SciPy and Matplotlib available:

```sh
python -m paper.reproduce_statistics
python paper/scripts/plot_results.py
python paper/scripts/plot_kernel_figures.py
```

Plots are written to paper/Fig and replay output to paper/results. These outputs
are not committed. This replays the previously opened holdout; it does not create
an independent validation set, refit the surrogate, or modify reference inputs.
The reconstruction check uses the existing absolute phase-relative tolerance 1e-12.

## Independent historical Hessian check

The plan_l_v2 snapshot contains the numerical code used for the frozen jets.
Build its Python extension with the same documented environment and verify the
reference manifest before running a diagnostic:

```sh
python -c "from paper.provenance import verify_reference; verify_reference('plan_l_v2', 'paper/reference_sources/plan_l_v2')"
(cd paper/reference_sources/plan_l_v2 && python setup.py build_ext --inplace)
python paper/scripts/verify_rayleigh_hessian.py \
  --disper26-root paper/reference_sources/plan_l_v2 \
  --plan-l-dir paper/data/plan_l_v2 --output-dir paper/results/hessian
```

The other snapshots preserve kernel figures, the earlier Hessian reference,
and a historical Disper26 baseline. Their source fingerprints and pinned
autodiff revisions are in reference_manifest.json. Do not replace snapshots
with the latest solver. Record latest-solver runs as separate new results.

The fixed Disper26 data are sufficient for the plotting commands above.
Original full scientific provenance is preserved in the research archive.
Public metadata
copies omit machine-specific paths and local environment names; numerical
arrays are unchanged. Export hashes verify these public copies; source hashes
identify the original historical evidence.

The original software uses GPL-2.0-or-later; see [LICENSE](../LICENSE) and
[NOTICE](../NOTICE) for its scope and third-party attribution. PREM-derived
model inputs are redistributed under CC BY 4.0 on the EarthScope policy basis
documented there, with their source, credits, and changes identified separately
from Disper26-generated numerical results. These are local release candidate
assets, not an issued software/data release.
