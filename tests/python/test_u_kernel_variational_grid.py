#!/usr/bin/env python3
"""
Test the GRID-mode variational group-velocity kernel
(u_kernel_mode="variational" on compute_dispersion_romberg_with_kernels
and the velocity iso/vti variants) against the full-AD path.

The variational path computes dU/dp from U = I3/(c*I1) in one flat
vector-AD pass with c IFT-seeded by dc/dp; c/U/HV/level and the
d_c / d_HV kernel columns must be unchanged.

Kernels are Neville-extrapolated across Romberg levels by default
(kernel_extrapolation=True), matching how the reported c/U/HV values are
extrapolated. Under extrapolation both u_kernel_mode paths Newton-polish
to the same per-level roots, so variational and full_ad agree to ~1e-7
(finite-difference dF/dc noise), and the extrapolated kernel matches the
finite-difference of the extrapolated U/c to ~1e-8 (vs ~2e-4 / ~6e-4 for
the single-level kernel_extrapolation=False path).
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for build_lib in sorted((ROOT / "build").glob("lib.*")):
    sys.path.insert(0, str(build_lib))
import disper26py as dp
import numpy as np

D_U_RTOL = 1e-5


def model_4node():
    depths = [0.0, 10.0, 20.0, 30.0]
    params = [dp.create_isotropic_parameters(2.5 + 0.05 * z, 4.3 + 0.08 * z, 2.4 + 0.01 * z)
              for z in depths]
    return depths, params, [False] * 4


def model_discontinuity():
    depths = [0.0, 10.0, 10.0, 40.0]
    params = [
        dp.create_isotropic_parameters(2.5, 4.3, 2.4),
        dp.create_isotropic_parameters(2.7, 4.6, 2.5),
        dp.create_isotropic_parameters(3.5, 6.2, 2.8),
        dp.create_isotropic_parameters(4.0, 7.0, 3.0),
    ]
    return depths, params, [False, True, False, False]


def model_20node():
    N = 20
    depths = np.linspace(0, 100, N).tolist()
    params = []
    for i in range(N):
        v = 1.0 + 0.5 * (i / (N - 1))
        params.append(dp.create_isotropic_parameters(3.0 * v, 5.0 * v, 2.5 + 0.5 * (i / (N - 1))))
    return depths, params, [False] * N


def model_prem_like():
    """Crust/mantle-style model mirroring docs/Figure PREM-like usage."""
    depths = [0.0, 15.0, 24.4, 24.4, 71.0, 171.0, 220.0, 220.0, 400.0]
    vs = [3.2, 3.9, 3.9, 4.49, 4.45, 4.40, 4.40, 4.77, 4.77]
    vp = [5.8, 6.8, 6.8, 8.11, 8.08, 8.02, 8.02, 8.56, 8.56]
    rho = [2.6, 2.9, 2.9, 3.38, 3.38, 3.36, 3.36, 3.44, 3.44]
    params = [dp.create_isotropic_parameters(s, p, r) for s, p, r in zip(vs, vp, rho)]
    is_disc = [False, False, True, False, False, False, True, False, False]
    return depths, params, is_disc


def model_ocean():
    ocean = dp.create_fluid_parameters(1.5, 1.03)
    crust_top = dp.create_isotropic_parameters(3.2, 5.8, 2.7)
    crust_bottom = dp.create_isotropic_parameters(3.5, 6.2, 2.8)
    halfspace = dp.create_isotropic_parameters(4.5, 8.0, 3.3)
    depths = [0.0, 3.0, 3.0, 23.0, 23.0]
    params = [ocean, ocean, crust_top, crust_bottom, halfspace]
    return depths, params, [False, True, False, True, False]


def compare(tag, depths, params, disc, period, wave, c_guess, sens_mode="Isotropic",
            expect_identical=False):
    ref = dp.compute_dispersion_romberg_with_kernels(
        depths, params, disc, period, wave, c_guess, sens_mode, 5, 1e-10,
        u_kernel_mode="full_ad")
    var = dp.compute_dispersion_romberg_with_kernels(
        depths, params, disc, period, wave, c_guess, sens_mode, 5, 1e-10,
        u_kernel_mode="variational")

    for i, q in ((0, "c"), (1, "U"), (2, "HV"), (3, "level"), (4, "rel_err")):
        if isinstance(ref[i], float) and math.isnan(ref[i]):
            assert math.isnan(var[i]), f"{tag}: {q} NaN mismatch"
        else:
            assert var[i] == ref[i], f"{tag}: {q} changed ({var[i]} != {ref[i]})"

    assert len(var[5]) == len(ref[5]), f"{tag}: kernel count"
    max_rel = 0.0
    du_scale = max(abs(k.d_U) for k in ref[5])
    for k_var, k_ref in zip(var[5], ref[5]):
        assert k_var.d_c == k_ref.d_c, \
            f"{tag}: d_c changed for {k_ref.parameter_name}/{k_ref.layer_index}"
        assert k_var.d_HV == k_ref.d_HV, \
            f"{tag}: d_HV changed for {k_ref.parameter_name}/{k_ref.layer_index}"
        if expect_identical:
            assert k_var.d_U == k_ref.d_U, \
                f"{tag}: d_U not bit-identical (fluid fallback expected)"
        elif du_scale > 0:
            rel = abs(k_var.d_U - k_ref.d_U) / du_scale
            max_rel = max(max_rel, rel)
    if not expect_identical:
        assert max_rel < D_U_RTOL, f"{tag}: d_U max rel err {max_rel:.2e}"
    print(f"OK {tag}: n_kernels={len(var[5])} d_U max rel err "
          f"{'bit-identical' if expect_identical else f'{max_rel:.2e}'}")


def test_solid_models():
    for name, (depths, params, disc), c_guess in [
        ("4node", model_4node(), 3.0),
        ("disc", model_discontinuity(), 3.0),
        ("20node", model_20node(), 3.8),
        ("prem_like", model_prem_like(), 3.9),
    ]:
        for wave in ("Rayleigh", "Love"):
            for mode in ("Isotropic", "VTI"):
                compare(f"{name}/{wave}/{mode}", depths, params, disc,
                        20.0, wave, c_guess, mode)


def test_fluid_fallback():
    depths, params, disc = model_ocean()
    ocean = dp.create_fluid_parameters(1.5, 1.03)
    crust = dp.create_isotropic_parameters(3.2, 5.8, 2.7)
    halfspace = dp.create_isotropic_parameters(4.5, 8.0, 3.3)
    roots = dp.search_roots([ocean, crust], [3.0, 20.0], halfspace,
                            2.0 * math.pi / 20.0, 0.5, 5.0, 500, "Rayleigh")
    assert roots
    compare("ocean/Rayleigh (fallback)", depths, params, disc,
            20.0, "Rayleigh", roots[0], expect_identical=True)


def test_velocity_variants():
    depths, params, disc = model_4node()
    rhos = [2.4 + 0.01 * z for z in depths]
    vps = [4.3 + 0.08 * z for z in depths]
    vss = [2.5 + 0.05 * z for z in depths]
    etas = [1.0] * len(depths)

    for wave in ("Rayleigh", "Love"):
        ref = dp.compute_dispersion_romberg_velocity_iso_with_kernels(
            depths, rhos, vps, vss, disc, 20.0, wave, 3.0, 5, 1e-10,
            u_kernel_mode="full_ad")
        var = dp.compute_dispersion_romberg_velocity_iso_with_kernels(
            depths, rhos, vps, vss, disc, 20.0, wave, 3.0, 5, 1e-10,
            u_kernel_mode="variational")
        assert var[0] == ref[0] and var[3] == ref[3]
        du_scale = max(abs(k.d_U) for k in ref[5])
        max_rel = max(abs(kv.d_U - kr.d_U) / du_scale
                      for kv, kr in zip(var[5], ref[5]))
        assert max_rel < D_U_RTOL, f"velocity_iso/{wave}: {max_rel:.2e}"
        print(f"OK velocity_iso/{wave}: d_U max rel err {max_rel:.2e}")

    ref = dp.compute_dispersion_romberg_velocity_vti_with_kernels(
        depths, rhos, vps, vps, vss, vss, etas, disc, 20.0, "Rayleigh", 3.0, 5, 1e-10,
        u_kernel_mode="full_ad")
    var = dp.compute_dispersion_romberg_velocity_vti_with_kernels(
        depths, rhos, vps, vps, vss, vss, etas, disc, 20.0, "Rayleigh", 3.0, 5, 1e-10,
        u_kernel_mode="variational")
    assert var[0] == ref[0] and var[3] == ref[3]
    du_scale = max(abs(k.d_U) for k in ref[5])
    max_rel = max(abs(kv.d_U - kr.d_U) / du_scale for kv, kr in zip(var[5], ref[5]))
    assert max_rel < D_U_RTOL, f"velocity_vti: {max_rel:.2e}"
    print(f"OK velocity_vti/Rayleigh: d_U max rel err {max_rel:.2e}")


def test_kernel_extrapolation_accuracy():
    """kernel_extrapolation=True must match the finite difference of the
    Romberg-extrapolated U and c far better than the single-level
    (kernel_extrapolation=False) kernel does."""
    depths, params, disc = model_4node()
    period = 20.0

    # FD reference: derivative of the reported (extrapolated) U and c w.r.t. Vs
    # at node 1, by re-running the full Romberg on the perturbed model.
    def perturbed(dvs1):
        ps = []
        for i, z in enumerate((0.0, 10.0, 20.0, 30.0)):
            vs = 2.5 + 0.05 * z + (dvs1 if i == 1 else 0.0)
            ps.append(dp.create_isotropic_parameters(vs, 4.3 + 0.08 * z, 2.4 + 0.01 * z))
        return dp.compute_dispersion_romberg(depths, ps, disc, period, "Rayleigh", 3.0, 6, 1e-10)

    hstep = 1e-5
    rp, rm, r0 = perturbed(+hstep), perturbed(-hstep), perturbed(0.0)
    assert rp[3] == rm[3] == r0[3], "converged level must be stable for a clean FD reference"
    fd_dU = (rp[1] - rm[1]) / (2 * hstep)
    fd_dc = (rp[0] - rm[0]) / (2 * hstep)

    def kernel_dU_dc(extrap):
        r = dp.compute_dispersion_romberg_with_kernels(
            depths, params, disc, period, "Rayleigh", 3.0, "Isotropic", 6, 1e-10,
            compute_c=True, want_kernel_u=True, want_kernel_hv=False,
            u_kernel_mode="variational", kernel_extrapolation=extrap)
        k = [x for x in r[5] if x.parameter_name == "Vs" and x.layer_index == 1][0]
        return k.d_U, k.d_c

    dU_single, dc_single = kernel_dU_dc(False)
    dU_extrap, dc_extrap = kernel_dU_dc(True)

    err_dU_single = abs(dU_single - fd_dU) / abs(fd_dU)
    err_dU_extrap = abs(dU_extrap - fd_dU) / abs(fd_dU)
    err_dc_single = abs(dc_single - fd_dc) / abs(fd_dc)
    err_dc_extrap = abs(dc_extrap - fd_dc) / abs(fd_dc)

    print(f"dU/dVs1 err: single-level={err_dU_single:.2e}  extrapolated={err_dU_extrap:.2e}")
    print(f"dc/dVs1 err: single-level={err_dc_single:.2e}  extrapolated={err_dc_extrap:.2e}")

    # Extrapolated kernels match the extrapolated-value derivative to ~1e-7.
    assert err_dU_extrap < 1e-6, f"extrapolated dU/dp err {err_dU_extrap:.2e}"
    assert err_dc_extrap < 1e-6, f"extrapolated dc/dp err {err_dc_extrap:.2e}"
    # And they are at least 100x more accurate than the single-level kernel.
    assert err_dU_extrap < err_dU_single / 100.0, "extrapolation did not improve dU/dp"
    assert err_dc_extrap < err_dc_single / 100.0, "extrapolation did not improve dc/dp"


def test_extrapolation_false_matches_single_level():
    """kernel_extrapolation=False keeps the single converged-level kernel
    (its d_c/d_U differ from the extrapolated default; both are finite and
    the level/c/U are unchanged)."""
    depths, params, disc = model_4node()
    r_ex = dp.compute_dispersion_romberg_with_kernels(
        depths, params, disc, 20.0, "Rayleigh", 3.0, "Isotropic", 6, 1e-10,
        kernel_extrapolation=True)
    r_sl = dp.compute_dispersion_romberg_with_kernels(
        depths, params, disc, 20.0, "Rayleigh", 3.0, "Isotropic", 6, 1e-10,
        kernel_extrapolation=False)
    assert r_ex[0] == r_sl[0] and r_ex[3] == r_sl[3]  # c, level unchanged
    # kernels differ (extrapolation moves them), but both finite
    diff = max(abs(a.d_c - b.d_c) for a, b in zip(r_ex[5], r_sl[5]))
    assert diff > 0, "extrapolation should change the kernel values"
    assert all(math.isfinite(k.d_c) and math.isfinite(k.d_U) for k in r_sl[5])
    print(f"single-level vs extrapolated d_c max diff = {diff:.2e} (both finite)")


def test_invalid_mode_raises():
    depths, params, disc = model_4node()
    try:
        dp.compute_dispersion_romberg_with_kernels(
            depths, params, disc, 20.0, "Rayleigh", 3.0, "Isotropic", 5, 1e-10,
            u_kernel_mode="bogus")
    except ValueError:
        print("OK invalid u_kernel_mode raises")
    else:
        raise AssertionError("expected ValueError for bogus u_kernel_mode")


if __name__ == "__main__":
    test_solid_models()
    test_fluid_fallback()
    test_velocity_variants()
    test_kernel_extrapolation_accuracy()
    test_extrapolation_false_matches_single_level()
    test_invalid_mode_raises()
    print("All GRID variational U-kernel tests passed.")
