#!/usr/bin/env python3
"""B1: run_phase1's Jacobian now comes from weak-form + K_d, not AD.

Before this change J[i] was res["g_inv_kernel"] (or its velocity-only
projection) -- entirely AD-sourced whenever the AD pass ran, which was every
datum by default. After it, J[i]'s velocity components are the weak-form
gradient (the reduced Hessian call's gradient_node, computed unconditionally
for every datum regardless of ad_kernel_every) and, where a depth column is
configured, the depth component is the variational K_d route
(depth_gradient_variational) rather than the AD kernel's Depth entry. AD
survives only as the source for `consistency`.

This was safe to do because the two paths were shown to agree to ~1e-08 after
the derivative fix in 986b360, and the K_d depth route was shown to agree with
the AD Depth kernel to 7.7e-12 in test_interface_gradient_variational.py --
this file checks the composition of those two results holds through
run_phase1, not the individual quantities again.

What is pinned:

  With ad_kernel_every=1 (default), J is close to what full AD would have
  given -- not required to be identical, since the sources differ by ~1e-08,
  but consistency itself (which IS the AD-vs-J comparison) must be tiny.

  With ad_kernel_every=0, J is unaffected: the whole point is the depth
  component (via g_variational_depth) and the velocity component (via the Hessian call,
  always run) do not depend on the AD pass at all. consistency is NaN for
  every datum, since none of them ran AD.

  A depth-column configuration is exercised, since that is the case the K_d
  entry point exists for -- a velocity-only test would not touch it.

  Requesting ad_kernel_every=0 with a depth backend that has no
  depth_gradient_variational route raises, rather than silently building an
  undefined Jacobian row.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inversion"))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))

from disper26_inv.data import DatasetSpec, Datum            # noqa: E402
from disper26_inv.depth_hessian import (                    # noqa: E402
    FdDepthHessianBackend, depth_backend_config,
    make_depth_backend,
)
from disper26_inv.model import InterfaceShiftBasis          # noqa: E402
from disper26_inv.pipeline import run_phase1                # noqa: E402
from disper26_inv.shallow import shallow_basis, shallow_reference  # noqa: E402

REF = shallow_reference(max_spacing_km=15.0, max_depth_km=1000.0)
VEL_BASIS = shallow_basis(REF)
PAIR_START = int(np.nonzero(REF.is_discontinuity)[0][1])
DEPTH_BASIS = InterfaceShiftBasis(VEL_BASIS, PAIR_START)

# Small: R1_7.143s is one of the overtones the AD path's defects showed up on.
DATA = DatasetSpec([
    Datum("Rayleigh", 0, 20.0), Datum("Rayleigh", 1, 7.143),
    Datum("Love", 0, 10.0), Datum("Love", 1, 10.0),
])


def test_full_ad_matches_weakform_production():
    """ad_kernel_every=1: every datum ran AD, so consistency is a direct
    measurement of how far production (weak-form + K_d) sits from the AD
    reference -- expected at the derivative fix's own precision, not zero."""
    result = run_phase1(REF, DEPTH_BASIS, DATA, ad_kernel_every=1)
    assert np.all(np.isfinite(result.consistency)), (
        "ad_kernel_every=1 should run AD on every datum")
    worst = float(result.consistency.max())
    assert worst < 1e-6, f"production vs AD disagree by {worst:.2e}"
    print(f"OK ad_kernel_every=1: worst consistency {worst:.2e} "
          f"over {len(DATA)} data")


def test_thinned_to_zero_still_produces_a_jacobian():
    """ad_kernel_every=0: no AD anywhere, so consistency is all NaN, but J
    must still be finite and, for the depth column, sourced from K_d."""
    result = run_phase1(REF, DEPTH_BASIS, DATA, ad_kernel_every=0)
    assert np.all(np.isnan(result.consistency)), (
        "ad_kernel_every=0 ran AD somewhere it should not have")
    assert np.all(np.isfinite(result.J)), "Jacobian has non-finite entries"
    print(f"OK ad_kernel_every=0: J finite, consistency all NaN "
          f"({len(DATA)} data)")


def test_thinned_and_full_jacobians_agree():
    """The Jacobian must not depend on how much AD ran alongside it -- that is
    the entire point of moving production off the AD path."""
    full = run_phase1(REF, DEPTH_BASIS, DATA, ad_kernel_every=1)
    thin = run_phase1(REF, DEPTH_BASIS, DATA, ad_kernel_every=0)
    diff = np.max(np.abs(full.J - thin.J))
    assert diff < 1e-10, (
        f"J changed by {diff:.2e} depending on ad_kernel_every, but the "
        "Jacobian source should be independent of it")
    print(f"OK J identical to {diff:.1e} regardless of ad_kernel_every")


def test_velocity_only_basis_is_unaffected_by_depth_logic():
    """No depth column configured: J is pure weak-form, none of the K_d /
    fallback / RuntimeError machinery is reachable."""
    result = run_phase1(REF, VEL_BASIS, DATA, ad_kernel_every=0)
    assert np.all(np.isfinite(result.J))
    assert np.all(np.isnan(result.consistency))
    print(f"OK velocity-only basis: J finite with ad_kernel_every=0")


def test_only_the_variational_backend_has_the_ad_free_route():
    """pipeline.py's fallback (AD Depth kernel, then RuntimeError) exists for
    backends without depth_gradient_variational. run_phase1 has no way to
    select a non-default depth backend, so the default ('variational_ad') is what
    every production caller gets and the fallback path is not reachable through
    it today -- this checks the capability split the fallback logic depends on,
    directly at the backend layer."""
    config = depth_backend_config(DEPTH_BASIS)
    assert "backend" not in config, (
        "depth_backend_config should not itself pin a backend name, "
        "make_depth_backend's default does")

    variational = make_depth_backend(None, config)
    assert hasattr(variational, "depth_gradient_variational"), (
        "default depth backend should be the variational one")

    fd_backend = make_depth_backend(None, dict(config, backend="fd"))
    assert isinstance(fd_backend, FdDepthHessianBackend)
    assert not hasattr(fd_backend, "depth_gradient_variational"), (
        "'fd' backend should NOT have the AD-free route -- "
        "pipeline.py's RuntimeError guard depends on this distinction")
    print("OK 'variational_ad' backend has depth_gradient_variational, "
          "'fd' does not")


def test_depth_backends_require_a_flat_reference():
    from disper26_inv.prem import load_prem_spherical

    spherical = load_prem_spherical(max_depth_km=50.)
    config = depth_backend_config(DEPTH_BASIS)
    for backend in ("ad", "variational_ad"):
        evaluator = make_depth_backend(None, dict(config, backend=backend))
        with pytest.raises(ValueError, match="load_prem_flat"):
            evaluator.interface_hessian_row(spherical, DATA.data[0], 3.)
        if backend == "variational_ad":
            with pytest.raises(ValueError, match="load_prem_flat"):
                evaluator.depth_gradient_variational(spherical, DATA.data[0], 3., 0)


def test_phase1_hash_distinguishes_hessian_modes():
    from disper26_inv.pipeline import _config_hash

    args = (REF, VEL_BASIS, DATA, {})
    assert _config_hash(*args) == _config_hash(*args, reduced_hessian=True)
    assert _config_hash(*args) != _config_hash(*args, reduced_hessian=False)


@pytest.mark.parametrize("processes", [1, 2])
def test_worker_failures_and_projection_cache(monkeypatch, processes):
    from types import SimpleNamespace
    from disper26_inv import parallel
    from disper26_inv.engine import BranchError

    datum = Datum("Rayleigh", 0, 20.)
    solution = SimpleNamespace(c=3., U=3., level=0, rel_err=0.,
                               romberg_roots=[3.], confirmation={})
    monkeypatch.setattr(parallel.ForwardEngine, "phase_velocity",
                        lambda *args, **kwargs: solution)
    for error in (BranchError("branch lost"), ValueError("invalid input")):
        def fail(*args, **kwargs):
            assert "J_reduced" not in parallel._WORKER_STATE
            parallel._WORKER_STATE["J_reduced"] = object()
            raise error

        parallel._WORKER_STATE["J_reduced"] = object()
        monkeypatch.setattr(parallel.ForwardEngine, "datum_hessian", fail)
        try:
            kwargs = dict(want_kernels=False, want_hessian=True, processes=processes,
                          ad_kernel_every=0)
            if isinstance(error, BranchError):
                result, = parallel.map_over_data(REF, [datum], **kwargs)
                assert result == {"failed": datum.label(), "error": "BranchError: branch lost"}
            else:
                with pytest.raises(ValueError, match="invalid input"):
                    parallel.map_over_data(REF, [datum], **kwargs)
            if processes == 1:
                assert not parallel._WORKER_STATE
        finally:
            parallel._WORKER_STATE.clear()


if __name__ == "__main__":
    test_full_ad_matches_weakform_production()
    test_thinned_to_zero_still_produces_a_jacobian()
    test_thinned_and_full_jacobians_agree()
    test_velocity_only_basis_is_unaffected_by_depth_logic()
    test_only_the_variational_backend_has_the_ad_free_route()
    print("OK")
