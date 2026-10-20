# Publication preparation

The software is intended to accompany the manuscript. Before the first private
GitHub sync, prepare and review the initial history. Development then continues
in that private GitHub repository, with all subsequent commits retained when
it becomes public at manuscript submission. The released source, documentation,
and selected practice examples must work from this repository alone.
Manuscript source and private research directories are not runtime or test
dependencies.

The [practice examples](../examples/README.md) introduce selected methods used
in the paper. Reproducing every manuscript figure is not a release requirement.
The [paper directory](../paper/README.md) also preserves fixed evidence and
source snapshots for optional historical reproduction. Keep those inputs
immutable and distinguish historical replay from calculations with the current
solver.
External-program comparison tests, reference output, benchmark reports, and
comparison plots are maintained in the separate research test repository and
are excluded from the public tree and source distribution. Public regression
fixtures contain only the model inputs needed by Disper26. Preserve method
attribution in numerical source and technical documentation.

## Release candidate contract

The initial distribution targets source releases. The Python distribution,
CMake project, inversion version, and citation metadata use version 0.1.104.
Locally built wheels are installation intermediates, not portable release
artifacts; the existing native CPU optimization remains unchanged.

Native and Python builds use the same autodiff selection: explicit source
override, bundled/submodule source, then the pinned public upstream commit.
See [installation](install.md) for the commit and source-archive behavior.
The forward API needs NumPy; `.[inversion]` adds SciPy. The existing `paper`
extra continues to provide SciPy and Matplotlib.

The PREM source and attribution review was completed on 2026-10-20. The bundled
CSV matches EarthScope's IRIS EMC `PREM_ANISOTROPIC_IDV.csv`, with one additional
final LF newline; all 97 rows and nine numerical columns are unchanged.
[NOTICE](../NOTICE) records the source URL, source/copy fingerprints, model and
data-provider credits, adapted-input changes, and CC BY 4.0 terms based on the
published EarthScope facility data policy. PREM provenance is no longer an open
release item. External-program comparison assets are excluded as described above.

## Before publication

- Original Disper26 software uses GPL-2.0-or-later; see [LICENSE](../LICENSE)
  and [NOTICE](../NOTICE). Retain the reviewed dependency notices and the
  separate PREM CC BY 4.0 attribution, source links, and change descriptions
  in every source distribution. Review the terms of any new or replaced inputs.
- Check public documentation references, including literal paths in prose and
  docstrings. Historical research notes are identified as archival records;
  running the public examples must not depend on those records.
- Verify a fresh build, the full Python regression suite, native CTest and
  Fortran checks, and the selected examples from the source distribution.
  Keep runtime assertions enabled for test targets in Release builds.
- Install from outside the source checkout into a dedicated environment with
  NumPy and no SciPy; verify the forward API and all three practice examples.
  Then install `.[inversion]`, check the inversion import, and run `pip check`.
  A missing SciPy must produce an actionable installation message without
  masking unrelated dependency failures.
- Review files and the initial history before the first private GitHub sync.
  Transfer reviewed files without importing private research Git objects, and
  preserve the original working repository and its development history.
  After that sync, treat every commit as part of the eventual public history.
  Before publication, review the retained history and distribution archives
  again; a clean working tree alone does not establish publication suitability.
- Keep environment instructions independent of a developer's environment name,
  account, host, or absolute checkout path. Distinguish optional setup examples
  from requirements. Retain software versions and measurement conditions when
  they are needed to interpret historical numerical or timing results.
- Review the authors and other metadata in [CITATION.cff](../CITATION.cff).
  Keep version 0.1.104 consistent with the distribution and CMake project.
  Before the first release, add the publication date and public repository URL;
  these fields are omitted until confirmed. Add manuscript links when available,
  keeping the software itself as the citation target. Citation requests must not
  add restrictions to GPL.

## Source archive and history review

Build the source distribution in a temporary copy of the reviewed working
files, so packaging metadata and generated outputs do not enter the source
checkout. Check the archive after extraction, including the citation and
license files, dependency sources, optional-dependency metadata, and all frozen
source manifests. Inspect NPZ contents with `allow_pickle=False`, decoding
text and byte-array JSON fields as well as searching ordinary text files.

Check both the source distribution with bundled autodiff and a GitHub-style
source archive without submodule contents. The latter must fetch the same
pinned public commit. Both must build without the manuscript directory or
any private checkout. Keep generated results outside fixed evidence.

Prepare the initial history before the first private GitHub sync. Preserve the
original local development repository and make an explicit list of reviewed
files, including approved untracked files and required public dependency
sources. Copy those files into a separate empty directory without any `.git`
directories or submodule Git-pointer files. Review that exported tree before
initializing the history intended for GitHub. Do not import private research
Git objects. Confirm the public author name and email before the initial commit.

Once synchronized to the private GitHub repository, keep subsequent commits
as the future public history. Changing visibility at manuscript submission
must not restart or squash that history. Continue reviewing new commits and
release artifacts for publication suitability during private development.

## Validation record (2026-10-20)

Version 0.1.104 was checked on Linux x86-64 with GCC/GFortran 13.3.0 and
Eigen 3.4.0. Fresh native Release builds passed all 21 CTest checks, including
the C and Fortran interfaces and the three newly registered tests. Runtime
assertions were confirmed active in the test builds.
A GitHub-style source export without submodule contents fetched the pinned
autodiff commit and built the C library. An explicit dependency-source override
was also confirmed to take precedence over the bundled source.

With Python 3.8.18, NumPy 1.24.4, SciPy 1.10.1, and pytest 8.3.4, the full
Python suite now passes all 425 tests. The run completed with nine warnings
from convergence-limit cases and test functions returning values.

The earlier result of 424 passed and one failed, and the pre-preparation
baseline of 419 passed and the same failure, used a second-order finite-difference
reference at a log-parameter step of 1e-4. In
`test_grid_kernel_vs_fd.py::test_both_gradient_paths_match_finite_differences`,
Rayleigh branch 1 at 7.143 s, direction 9, phase velocities near 4.224 km/s
differed by only 3.3e-9 km/s. The resulting AD/reference relative discrepancy
was 7.9455e-7 against the existing 5e-7 tolerance.

The reference now uses fourth-order central differences, constructed as
`(4 D(h) - D(2h)) / 3`, where `D` is the second-order central difference and
`h=1e-3`. Before comparison with AD or weak-form gradients, the test checks
agreement with the fourth-order reference at `h=2e-3`. Across all four
wave/branch/period cases and five parameter directions, the maximum gradient
discrepancy is 4.94e-8 and the maximum reference step-size discrepancy is
9.80e-8. Both checks retain the existing 5e-7 tolerance. The previously failing
regression is resolved by improving its finite-difference reference; solver
formulas, defaults, floating-point precision, and numerical tolerances are
unchanged.

The targeted tracking, practice-example, and provenance checks passed all 58
tests. The Disper26 data exporter and three retained result figures also ran;
regenerated numerical arrays matched the frozen arrays exactly. Reusing an
output directory preserved pre-existing files and listed only current
Disper26 outputs in export provenance.

An isolated installation from the source distribution passed all three
practice examples with Python 3.12.13, NumPy 1.26.4, and no SciPy. Installing
the inversion extra selected NumPy 2.5.4 and SciPy 1.18.1; the installed
inversion import, packaged PREM loading, a forward solve, and `pip check`
passed. This was an installation smoke check, not a second full regression run.

The citation passed the official CFF 1.2 schema. Source-archive metadata,
license files, private-identifier checks, and decoded NPZ metadata were
reviewed. All 254 frozen-source manifest entries verified. Retained numerical
evidence and source snapshots are unchanged; publication-scope cleanup removes
external-comparison records and their export metadata. PREM source and terms
verification was completed separately and is recorded in NOTICE.

After the remaining external-comparison tables and plotting/audit scripts were
excluded, all 71 targeted provenance, practice-example, tracking, gravity, and
Q-correction tests passed. The extracted source archive contained 585 files,
verified all 254 frozen-source hashes, and replayed six retained Disper26 figures
as PDF/PNG pairs without external comparison files. The archive review also
checked decoded NPZ text fields and the absence of Git objects, compiled
binaries, developer environment names, and local paths. Numerical implementations
and tolerances were unchanged. These scope checks preceded the finite-difference
reference correction and successful full Python run described above. Native
CTest was not rerun for this test-reference and documentation change; native
sources are unchanged from the 21-check validation recorded above.

## GitHub and Zenodo

The intended sequence is:

1. Prepare and review the initial files and history locally.
2. Synchronize that history to a private GitHub repository and retain later
   development commits.
3. At manuscript submission, make the same repository public with its history
   preserved.
4. Confirm the release version and citation metadata, enable Zenodo integration,
   then create the corresponding Git tag and GitHub release.
5. Verify the archived release and record its assigned DOI in the software
   citation and manuscript.

Zenodo archives new releases after integration is enabled. See
[Enable a repository](https://help.zenodo.org/docs/github/enable-repository/).

Zenodo accepts `CITATION.cff` or `.zenodo.json`. If both exist, `.zenodo.json`
takes precedence for the archived metadata; keep the two consistent if both
are needed. See
[Describe software](https://help.zenodo.org/docs/github/describe-software/).

The GitHub integration cannot reserve a DOI before the release. Manual Zenodo
upload is a separate route that supports reservation, should the manuscript
need a DOI earlier. See
[DOI reservation and GitHub releases](https://support.zenodo.org/help/en-gb/24-github-integration/73-can-i-pre-reserved-a-doi-before-a-github-release).

The private GitHub repository is the future public software repository, separate
from the private research repository containing external-program comparisons.
Remote creation, pushes, visibility changes, releases, and DOI issuance await
the owner's instruction for those actions.
