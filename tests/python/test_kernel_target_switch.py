#!/usr/bin/env python3
"""ForwardEngine's want_kernel_u / want_kernel_hv switches.

dU/dp and d(H/V)/dp are produced inside the same 528-direction AD pass as
dc/dp, once per Romberg level, and they cost 84% of it -- 6.833 -> 1.091
s/datum measured on the 88-node shallow grid. Nothing in disper26_inv reads
them: derivatives.py takes kernel.d_c only and Datum carries no U or
ellipticity observable. They are therefore off by default.

Two things have to hold for that default to be safe, and both are asserted
here:

  dc/dp must be bit-identical with the switches off. It is a separate output of
  the same AD pass, so this is a real question, not a formality -- if turning
  off the U kernel perturbed dc/dp by even an ulp the default would be a silent
  change to every gradient in the pipeline.

  The switches must reach the binding. A default that is quietly ignored would
  pass a d_c comparison while buying nothing, so d_U and d_HV are checked to be
  exactly zero when off and nonzero when on.

want_value_u / want_value_hv are deliberately left True (the phase_only=False
default). They govern the U and H/V *values* rather than their kernels, are
worth a further 0.1%, and turning them off would make solution.U NaN --
solution.U feeds _group_predict's warm-start hint in trace_dataset.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inversion"))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))

from disper26_inv.engine import Datum, ForwardEngine   # noqa: E402
from disper26_inv.shallow import shallow_reference     # noqa: E402

MODEL = shallow_reference(max_spacing_km=15.0, max_depth_km=1000.0)
DATA = [Datum("Rayleigh", 0, 12.0), Datum("Rayleigh", 0, 30.0),
        Datum("Love", 0, 20.0)]


def kernels_for(engine, datum):
    solution, kernels = engine.phase_velocity_with_kernels(MODEL, datum)
    return solution, kernels


def test_defaults_are_off():
    engine = ForwardEngine()
    assert engine.want_kernel_u is False
    assert engine.want_kernel_hv is False
    print("OK defaults: want_kernel_u=False, want_kernel_hv=False")


def test_dc_is_bit_identical_with_kernels_off():
    off = ForwardEngine()
    on = ForwardEngine(want_kernel_u=True, want_kernel_hv=True)
    for datum in DATA:
        sol_off, k_off = kernels_for(off, datum)
        sol_on, k_on = kernels_for(on, datum)
        assert sol_off.c == sol_on.c, (
            f"{datum.label()}: c moved, {sol_off.c!r} vs {sol_on.c!r}")
        dc_off = np.array([k.d_c for k in k_off])
        dc_on = np.array([k.d_c for k in k_on])
        assert dc_off.shape == dc_on.shape, "kernel list length changed"
        assert np.array_equal(dc_off, dc_on), (
            f"{datum.label()}: d_c changed, max diff "
            f"{np.max(np.abs(dc_off - dc_on)):.3e}")
        print(f"OK {datum.label()}: c and all {len(dc_off)} d_c entries "
              "bit-identical")


def test_switches_reach_the_binding():
    """Off must mean zero and on must mean nonzero, or the default is a no-op
    that still costs the 84%."""
    datum = DATA[0]

    _, k_off = kernels_for(ForwardEngine(), datum)
    assert all(k.d_U == 0.0 for k in k_off), "d_U nonzero with want_kernel_u=False"
    assert all(k.d_HV == 0.0 for k in k_off), "d_HV nonzero with want_kernel_hv=False"

    _, k_u = kernels_for(ForwardEngine(want_kernel_u=True), datum)
    assert any(k.d_U != 0.0 for k in k_u), "want_kernel_u=True produced no d_U"
    assert all(k.d_HV == 0.0 for k in k_u), "d_HV leaked in with only want_kernel_u"

    _, k_hv = kernels_for(ForwardEngine(want_kernel_hv=True), datum)
    assert any(k.d_HV != 0.0 for k in k_hv), "want_kernel_hv=True produced no d_HV"
    assert all(k.d_U == 0.0 for k in k_hv), "d_U leaked in with only want_kernel_hv"

    print("OK switches are independent and both reach the binding")


def test_group_velocity_value_survives():
    """Only the kernels are switched off; the U value still has to be there,
    because trace_dataset's warm-start hint is built from it."""
    for datum in DATA:
        solution, _ = kernels_for(ForwardEngine(), datum)
        assert np.isfinite(solution.U), f"{datum.label()}: U is {solution.U}"
        assert solution.U > 0.0, f"{datum.label()}: U = {solution.U}"
        print(f"OK {datum.label()}: U = {solution.U:.6f} still computed")


if __name__ == "__main__":
    test_defaults_are_off()
    test_dc_is_bit_identical_with_kernels_off()
    test_switches_reach_the_binding()
    test_group_velocity_value_survives()
    print("OK")
