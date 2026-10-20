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
