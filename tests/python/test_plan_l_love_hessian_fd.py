#!/usr/bin/env python3
"""Plan-L Love Hessian columns versus central FD of the independent AD gradient.

The production reduced Hessian differentiates the variational gradient.  This
test instead finite-differences ``ForwardEngine.phase_velocity_with_kernels``,
whose gradient comes from the propagation/full-AD kernel route.  It probes one
Vsv and one Vsh column, including their cross term, for Love root orders 0/1 at
the low, middle, and high ends of the primary frequency grid.

The 5e-3 column tolerance follows the existing GRID Hessian-vs-FD tolerance in
``test_depth_hessian_ad_vs_fd.py``: the comparison mixes a central difference
with independently extrapolated Romberg gradients, so machine precision is not
the relevant contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inversion"))

from disper26_inv.data import DatasetSpec  # noqa: E402
from disper26_inv.derivatives import (  # noqa: E402
    gradient_log,
    kernels_to_stiffness_gradient,
    project_gradient,
)
from disper26_inv.engine import Datum, ForwardEngine  # noqa: E402
from disper26_inv.long_period_taylor_reference import (  # noqa: E402
    build_plan_l_objects,
)
from disper26_inv.pipeline import run_phase1  # noqa: E402


FREQUENCY_HZ = (0.0075, 0.0200, 0.0300)
FD_STEPS = (2.0e-4, 1.0e-4)
COLUMN_RELATIVE_TOLERANCE = 5.0e-3
STEP_CONVERGENCE_TOLERANCE = 5.0e-3
CROSS_TERM_COLUMN_SCALED_TOLERANCE = 1.0e-2


def gradient_at(engine, basis, datum, delta_m, c_hint):
    model = basis.m_to_forward_model(delta_m)
    solution, kernels = engine.phase_velocity_with_kernels(
        model, datum, c_hint=float(c_hint)
    )
    stiffness_gradient = kernels_to_stiffness_gradient(kernels, model.n_nodes)
    gradient = project_gradient(
        gradient_log(model, stiffness_gradient), basis, model=model
    )
    return np.asarray(gradient, dtype=float), float(solution.c)


def main():
    reference, basis, _ = build_plan_l_objects()
    data = [
        Datum("Love", root_order, 1.0 / frequency)
        for root_order in (0, 1)
        for frequency in FREQUENCY_HZ
    ]
    phase1 = run_phase1(
        reference,
        basis,
        DatasetSpec(data),
        processes=1,
        engine_kwargs={
            "max_steps": 6,
            "tolerance": 1.0e-9,
            "hessian_max_levels": 5,
            "hessian_tolerance": 1.0e-6,
        },
        reduced_hessian=True,
        ad_kernel_every=1,
    )
    depth_index = int(np.flatnonzero(np.isclose(basis.inv_depths, 20.0))[0])
    vsv_index = depth_index
    vsh_index = basis.n_inv + depth_index
    column_indices = (vsv_index, vsh_index)
    engine = ForwardEngine(
        max_steps=6,
        tolerance=1.0e-9,
        hessian_max_levels=5,
        hessian_tolerance=1.0e-6,
    )
    failures = []
    for datum_index, datum in enumerate(data):
        hessian = np.asarray(phase1.H[datum_index], dtype=float)
        if not np.all(np.isfinite(hessian)):
            failures.append(f"{datum.label()}: non-finite production Hessian")
            continue
        fd_columns = {}
        for column_index in column_indices:
            columns_by_step = []
            for step in FD_STEPS:
                plus = np.zeros(basis.n_params)
                minus = np.zeros(basis.n_params)
                plus[column_index] = step
                minus[column_index] = -step
                gradient_plus, _ = gradient_at(
                    engine, basis, datum, plus, phase1.c0[datum_index]
                )
                gradient_minus, _ = gradient_at(
                    engine, basis, datum, minus, phase1.c0[datum_index]
                )
                columns_by_step.append((gradient_plus - gradient_minus) / (2.0 * step))
            coarse, fine = columns_by_step
            fd_columns[column_index] = fine
            scale = max(
                float(np.linalg.norm(hessian[:, column_index])),
                np.finfo(float).tiny,
            )
            relative = float(
                np.linalg.norm(fine - hessian[:, column_index]) / scale
            )
            convergence = float(np.linalg.norm(fine - coarse) / scale)
            if relative > COLUMN_RELATIVE_TOLERANCE:
                failures.append(
                    f"{datum.label()} column {column_index}: rel={relative:.3e}"
                )
            if convergence > STEP_CONVERGENCE_TOLERANCE:
                failures.append(
                    f"{datum.label()} column {column_index}: "
                    f"step convergence={convergence:.3e}"
                )
            print(
                f"{datum.label():16s} column={column_index:2d} "
                f"relative={relative:.3e} step={convergence:.3e}"
            )

        cross_scale = max(
            float(np.linalg.norm(hessian[:, vsh_index])),
            float(np.linalg.norm(hessian[:, vsv_index])),
            np.finfo(float).tiny,
        )
        cross_error = abs(
            fd_columns[vsh_index][vsv_index] - hessian[vsv_index, vsh_index]
        ) / cross_scale
        symmetry_error = abs(
            hessian[vsv_index, vsh_index] - hessian[vsh_index, vsv_index]
        ) / cross_scale
        if cross_error > CROSS_TERM_COLUMN_SCALED_TOLERANCE:
            failures.append(
                f"{datum.label()} Vsv-Vsh cross error={cross_error:.3e}"
            )
        if symmetry_error > 1.0e-12:
            failures.append(
                f"{datum.label()} Vsv-Vsh asymmetry={symmetry_error:.3e}"
            )
        print(
            f"{datum.label():16s} Vsv-Vsh cross={hessian[vsv_index, vsh_index]:+.6e} "
            f"scaled_error={cross_error:.3e}"
        )

    if failures:
        raise AssertionError("\n".join(failures))
    print("Plan-L Love Hessian finite-difference checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
