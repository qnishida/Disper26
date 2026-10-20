# Release preparation

This is a temporary maintainer checklist for the software release, not
documentation of the accompanying manuscript. Keep only outstanding tasks;
remove completed items as preparation progresses. Delete this file and its
references before making the repository public. Earlier versions remain in
Git history, so do not record confidential information here.

## Before making the repository public

- [ ] Confirm the release version and update the distribution, CMake project,
  inversion version, and [CITATION.cff](../CITATION.cff) consistently if it changes.
- [ ] Confirm citation authors, the public repository URL, and manuscript links
  when available. Add the release date when the release is scheduled. Keep
  the software itself as the software citation target.
- [ ] Review changes since the last validation. For the final release candidate,
  verify a fresh native build with CTest and Fortran checks, the full Python
  regression suite, and the selected [practice examples](../examples/README.md).
  Keep runtime assertions enabled for native test targets in Release builds.
- [ ] Review the final source distribution in a temporary copy. Verify frozen
  source manifests, license and attribution files, citation metadata, and
  dependency sources. Inspect decoded NPZ metadata as well as ordinary text.
  Exclude credentials, developer environment names, local paths, unpublished
  research, external-program comparison assets, and unapproved binaries.
- [ ] Check both the source distribution with bundled autodiff and a GitHub-style
  archive without submodule contents. Verify independent builds and installation
  outside the checkout, the forward API and practice examples with NumPy and
  no SciPy, then the inversion extra and `pip check`. See [installation](install.md).
  Recheck third-party terms if inputs or dependencies have changed.
- [ ] Review the retained Git history for publication suitability. Preserve
  commits from private GitHub development when changing repository visibility;
  do not import Git objects from the separate private research repository.
- [ ] Update the README release status. Remove links to this checklist from
  README.md and AGENTS.md, delete this file, and commit those changes before
  making the repository public. Confirm that no broken references remain and
  that the final distribution excludes this file. Its earlier revisions remain
  in the retained Git history.

## Publication and release sequence

These actions require the owner's instruction. At present, the release
candidate targets source distribution; locally built wheels are not portable
release artifacts.

1. At manuscript submission, make the existing GitHub repository public,
   preserving its development history.
2. Confirm release metadata and enable the repository's Zenodo integration
   before creating the version tag and GitHub release. See
   [Zenodo setup](https://help.zenodo.org/docs/github/enable-repository/).
3. Verify the archived release and its metadata. Record the assigned DOI in
   the software citation and manuscript, and commit citation updates.

If the manuscript needs a DOI before the GitHub release, decide on the deposit
route before scheduling the release. See
[Zenodo DOI reservation](https://support.zenodo.org/help/en-gb/24-github-integration/73-can-i-pre-reserved-a-doi-before-a-github-release).
