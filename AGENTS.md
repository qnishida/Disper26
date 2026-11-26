# Public release candidate

The reviewed initial history has been prepared and synchronized to GitHub.
Import only reviewed files or patches; never fetch, merge, cherry-pick, or
share Git objects from the separate private research project. From the first
GitHub sync onward, preserve commits as the eventual public history; do not
restart or squash that history when changing visibility for publication.
Review staged files and source/wheel archives for credentials, local paths,
unpublished research, reports, and unapproved binaries before every commit.

At manuscript submission, publish this software repository with its history
and archive a versioned GitHub release with Zenodo. Its builds, documentation
and selected practice examples must work independently. See examples/README.md
for the examples included with the software.
External-program comparison tests, reference output, and comparison benchmarks
belong in the separate research test repository, outside this public tree.
Keep Disper26-only regression fixtures independent of those comparisons.

Use English for project documentation, code comments, docstrings and user-facing
messages, including test and diagnostic output. Keep Japanese documentation in
`docs_jp/` and Japanese example guides in `example_jp/` at the repository root.

Preserve numerical formulas, units, signs, precision, defaults, reduction order,
seeds, tolerances and documented deprecated aliases. Follow docs/naming_convention.md.
Use a dedicated Conda environment with the dependencies in docs/install.md;
the environment name is user-selected. Never install globally or into base.

Inside that environment, build Python with python setup.py build_ext --inplace.
Run python -m pytest -q tests/python for regressions. For non-interactive
commands, use conda run -n ENV_NAME with the selected environment name.
For native checks use a fresh CMake build and ctest --output-on-failure.
Treat the inputs and numerical results in `paper/data/` and the historical
solver snapshots in `paper/reference_sources/` as immutable. Before building or
using a snapshot, verify it against `paper/reference_manifest.json` as described
in [the reproduction guide](paper/README.md). Write new results to a separate
output directory.

Original Disper26 software uses GPL-2.0-or-later. Bundled PREM tables and adapted
model inputs use CC BY 4.0 on the policy basis recorded in NOTICE. Preserve
third-party licenses, data attribution, source fingerprints, and descriptions
of changes. Review the terms of new or replaced third-party inputs before
including them. Do not publish, push, issue a release or DOI without the owner's
authorization.
