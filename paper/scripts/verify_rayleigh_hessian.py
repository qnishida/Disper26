#!/usr/bin/env python3
"""Compare frozen surrogate Hessians with FD of independent full-AD gradients.

Use the historical solver revision recorded in the artifact, not a current
build. No reference jets or holdout models are refitted or modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ARTIFACT_SHA256 = "4e92574a29a6a1398052b4a7744fc3c4131de4ae779906e348c05719fe555bb5"
FREQUENCIES = (0.0075, 0.0200, 0.0300)
STEPS = (2e-4, 1e-4)
SETTINGS = dict(max_steps=6, tolerance=1e-9, hessian_max_levels=5,
                hessian_tolerance=1e-6)
# Same contracts as tests/python/test_plan_l_love_hessian_fd.py.
COLUMN_TOLERANCE = 5e-3
STEP_TOLERANCE = 5e-3
CROSS_TOLERANCE = 1e-2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disper26-root", type=Path, required=True)
    parser.add_argument("--plan-l-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/hessian")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from paper.provenance import validate_output
    args.output_dir = validate_output(args.output_dir)
    return args


def main():
    args = parse_args()
    root = args.disper26_root.resolve()
    artifact = args.plan_l_dir / "long_period_mixed_response_v2.npz"
    if sha(artifact) != ARTIFACT_SHA256:
        raise ValueError("This diagnostic requires the manuscript's frozen v2 artifact")
    with np.load(artifact, allow_pickle=False) as source:
        jet = {key: source[key] for key in source.files}
    metadata = json.loads(jet["metadata_json_utf8"].tobytes().decode())
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from paper.provenance import verify_reference
    source_provenance = verify_reference('plan_l_v2', root, artifact)
    revision = source_provenance['source_tree_sha256']
    sys.path[:0] = [str(root), str(root / "inversion")]
    import disper26py as dp
    from disper26_inv.derivatives import (
        gradient_log, kernels_to_stiffness_gradient, project_gradient)
    from disper26_inv.engine import Datum, ForwardEngine
    from disper26_inv.long_period_taylor_manifest import (
        basis_fingerprint, prior_fingerprint, reference_fingerprint)
    from disper26_inv.long_period_taylor_reference import (
        REFERENCE_PHASE_TRACE_RELATIVE_TOLERANCE, build_plan_l_objects)

    if Path(dp.__file__).resolve().parent != root:
        raise ValueError("Python imported an extension outside the selected checkout")
    reference, basis, prior = build_plan_l_objects()
    fingerprints = dict(reference_fingerprint=reference_fingerprint(reference),
                        basis_fingerprint=basis_fingerprint(basis),
                        prior_fingerprint=prior_fingerprint(prior))
    for key, value in fingerprints.items():
        if value != metadata["provenance"][key]:
            raise ValueError("Frozen model/basis/prior mismatch: " + key)
    depth_index = int(np.flatnonzero(basis.inv_depths == 20.0)[0])
    columns = (depth_index, basis.n_inv + depth_index)
    cholesky = jet["prior_cholesky"]
    engine = ForwardEngine(**SETTINGS)
    confirmations = []

    def gradient_at(datum, delta_m, hint):
        model = basis.m_to_forward_model(delta_m)
        solution, kernels = engine.phase_velocity_with_kernels(
            model, datum, c_hint=float(hint))
        confirmation = solution.confirmation
        if (solution.branch != datum.branch or not confirmation
                or not confirmation["accepted"]):
            raise RuntimeError("Unconfirmed branch: " + datum.label())
        stiffness = kernels_to_stiffness_gradient(kernels, model.n_nodes)
        gradient = np.asarray(project_gradient(
            gradient_log(model, stiffness), basis, model=model), dtype=float)
        if not np.all(np.isfinite(gradient)) or not np.isfinite(solution.c):
            raise RuntimeError("Nonfinite independent gradient/root")
        confirmations.append(dict(datum=datum.label(), phase_velocity=solution.c,
                                  delta_m=delta_m.tolist(), confirmation=confirmation))
        return gradient, solution.c

    arrays = dict(fd_steps=np.array(STEPS), column_indices=np.array(columns),
                  prior_cholesky=cholesky, inversion_depths_km=basis.inv_depths)
    cases, failures = [], []
    for mode in (0, 1):
        prefix = "curve_%02d_" % mode
        if metadata["curves"][mode] != dict(
                response_kind="phase_velocity", root_order=mode, wave="rayleigh"):
            raise ValueError("Unexpected frozen response convention")
        for frequency in FREQUENCIES:
            index = int(np.flatnonzero(jet[prefix + "frequency_hz"] == frequency)[0])
            key = "rayleigh%d_%g" % (mode, frequency)
            datum = Datum("Rayleigh", mode, 1.0 / frequency)
            c0 = float(jet[prefix + "reference_phase_velocity"][index])
            hz = jet[prefix + "response_hessian_z"][index]
            # H_z = L.T H_m L; Rayleigh response is phase velocity itself.
            hessian = np.linalg.solve(cholesky.T,
                                      np.linalg.solve(cholesky.T, hz).T).T
            frozen_gradient = np.linalg.solve(
                cholesky.T, jet[prefix + "response_gradient_z"][index])
            if not np.all(np.isfinite(hessian)):
                raise ValueError("Nonfinite frozen Hessian")
            zero_gradient, zero_c = gradient_at(datum, np.zeros(basis.n_params), c0)
            phase_difference = abs(zero_c / c0 - 1)
            if phase_difference > REFERENCE_PHASE_TRACE_RELATIVE_TOLERANCE:
                failures.append(key + ": reference phase reproduction")
            case = dict(mode=mode, frequency_hz=frequency,
                        reference_phase_relative_difference=phase_difference,
                        reference_gradient_relative_difference=float(
                            np.linalg.norm(zero_gradient - frozen_gradient)
                            / np.linalg.norm(frozen_gradient)), columns=[])
            gradients = np.empty((2, 2, 2, basis.n_params))
            roots = np.empty((2, 2, 2))
            fd = np.empty((2, 2, basis.n_params))
            for j, column in enumerate(columns):
                for k, step in enumerate(STEPS):
                    for s, sign in enumerate((1, -1)):
                        delta_m = np.zeros(basis.n_params)
                        delta_m[column] = sign * step
                        gradients[j, k, s], roots[j, k, s] = gradient_at(
                            datum, delta_m, c0)
                    fd[j, k] = (gradients[j, k, 0] - gradients[j, k, 1]) / (2 * step)
                scale = float(np.linalg.norm(hessian[:, column]))
                if scale <= np.finfo(float).tiny:
                    raise ValueError("Selected Hessian column is unresolved")
                difference = float(np.linalg.norm(fd[j, 1] - hessian[:, column]) / scale)
                convergence = float(np.linalg.norm(fd[j, 1] - fd[j, 0]) / scale)
                if difference > COLUMN_TOLERANCE or convergence > STEP_TOLERANCE:
                    failures.append(key + ": column " + str(column))
                case["columns"].append(dict(index=column, norm=scale,
                                           relative_difference=difference,
                                           step_relative_difference=convergence))
            cross_scale = max(item["norm"] for item in case["columns"])
            cross = max(abs(fd[1, 1, columns[0]] - hessian[columns[0], columns[1]]),
                        abs(fd[0, 1, columns[1]] - hessian[columns[1], columns[0]])) / cross_scale
            case["cross_term_column_scaled_difference"] = float(cross)
            if cross > CROSS_TOLERANCE:
                failures.append(key + ": cross term")
            for name, value in dict(hessian_m=hessian, hessian_z=hz,
                                    gradients_m=gradients, roots_km_s=roots,
                                    fd_columns_m=fd, reference_gradient_m=zero_gradient,
                                    frozen_gradient_m=frozen_gradient).items():
                arrays[key + "_" + name] = value
            cases.append(case)
            print(key, "max column error", max(c["relative_difference"]
                                               for c in case["columns"]), flush=True)

    sources = [root / "setup.py", root / "tests/model_PREM_aniso.csv"]
    sources += sorted((root / "src").glob("*.cpp")) + sorted((root / "src").glob("*.hpp"))
    sources += sorted((root / "inversion/disper26_inv").glob("*.py"))
    sources += sorted((root / "third_party/autodiff/autodiff").rglob("*.hpp"))
    report = dict(
        role="Post-holdout independent verification of frozen Rayleigh phase Hessians",
        code_commit=revision, artifact_sha256=sha(artifact),
        extension_sha256=sha(dp.__file__), script_sha256=sha(__file__),
        source_files_sha256={str(p.relative_to(root)): sha(p) for p in sources},
        fingerprints=fingerprints, engine_settings=SETTINGS,
        parameter_directions="20-km Vsv and Vsh log parameters in the coupled 18-parameter basis",
        independent_route="central differences of propagation/full-AD gradients, not variational gradients",
        coordinate_conversion="H_m = L^{-T} H_z L^{-1}; Rayleigh response r=c",
        tolerances=dict(column=COLUMN_TOLERANCE, step_convergence=STEP_TOLERANCE,
                        cross_term_column_scaled=CROSS_TOLERANCE,
                        reference_phase=REFERENCE_PHASE_TRACE_RELATIVE_TOLERANCE),
        tolerance_source="Disper26 tests/python/test_plan_l_love_hessian_fd.py and reference-depth gate",
        cases=cases, branch_confirmations=confirmations, failures=failures, passed=not failures)
    report["summary"] = dict(
        maximum_column_relative_difference=max(c["relative_difference"] for a in cases for c in a["columns"]),
        maximum_step_relative_difference=max(c["step_relative_difference"] for a in cases for c in a["columns"]),
        maximum_cross_term_column_scaled_difference=max(a["cross_term_column_scaled_difference"] for a in cases),
        branch_confirmed_solutions=len(confirmations))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "rayleigh_hessian_verification.npz"
    np.savez_compressed(output, **arrays)
    report["data_sha256"] = sha(output)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))
    if failures:
        raise AssertionError("; ".join(failures))


if __name__ == "__main__":
    main()
