# Public release candidate

Prepare a reviewed initial history before the first private GitHub sync.
Import only reviewed files or patches; never fetch, merge, cherry-pick, or
share Git objects from the separate private research project. From the first
GitHub sync onward, preserve commits as the eventual public history; do not
restart or squash that history when changing visibility for publication.
Review staged files and source/wheel archives for credentials, local paths,
unpublished research, reports, and unapproved binaries before every commit.

The software is intended to accompany the manuscript. After the initial history
review, development will be synchronized to a private GitHub repository. At
manuscript submission, make that repository public, retain its history, and
archive a versioned GitHub release with Zenodo. Public builds, documentation
and selected practice examples must work from this
repository alone. Reproducing all manuscript figures is not required. See
docs/publication.md and examples/README.md for scope and preparation.
External-program comparison tests, reference output, and comparison benchmarks
belong in the separate research test repository, outside this public tree.
Keep Disper26-only regression fixtures independent of those comparisons.

Preserve numerical formulas, units, signs, precision, defaults, reduction order,
seeds, tolerances and documented deprecated aliases. Follow docs/naming_convention.md.
Use a dedicated Conda environment with the dependencies in docs/install.md;
the environment name is user-selected. Never install globally or into base.

Inside that environment, build Python with python setup.py build_ext --inplace.
Run python -m pytest -q tests/python for regressions. For non-interactive
commands, use conda run -n ENV_NAME with the selected environment name.
For native checks use a fresh CMake build and ctest --output-on-failure.
Paper inputs and reference_sources are immutable; write recomputation to a
separate output directory. Frozen source manifests must be verified first.

Original Disper26 software uses GPL-2.0-or-later. Bundled PREM tables and adapted
model inputs use CC BY 4.0 on the policy basis recorded in NOTICE. Preserve
third-party licenses, data attribution, source fingerprints, and descriptions
of changes. Review the terms of new or replaced third-party inputs before
including them. Do not publish, push, issue a release or DOI without the owner's
authorization.
