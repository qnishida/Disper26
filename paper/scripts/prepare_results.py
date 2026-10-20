#!/usr/bin/env python3
"""Export manuscript data from recorded Disper26 results; never run a solver.

See README.md for source revisions and commands. Existing opened holdout data
are only re-evaluated to reproduce the published statistics, not used to fit or
select a surrogate. No input archive is modified.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "results" / "imported_data"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(root, plan):
    sys.path.insert(0, str(PAPER.parent))
    from paper.provenance import verify_reference
    verify_reference("plan_l_v2", root, plan / "long_period_mixed_response_v2.npz")
    OUT.mkdir(parents=True, exist_ok=True)
    sources = []

    def record(path, role):
        sources.append({"role": role, "path": path.name, "sha256": sha(path)})

    artifact_path = plan / "long_period_mixed_response_v2.npz"
    report_path = plan / "holdout_report.json"
    manifest_path = plan / "holdout" / "split_manifest.json"
    report = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert report["evidence_role"] == "sealed_holdout"
    assert report["gate_result"]["passed"]
    assert sha(artifact_path) == report["artifact_sha256"]
    for p, role in [(artifact_path, "frozen v2 response jets"),
                    (report_path, "opened sealed-holdout validation"),
                    (manifest_path, "opened sealed-holdout manifest")]:
        record(p, role)
    arrays = {}
    summaries = []
    with np.load(artifact_path, allow_pickle=False) as jet:
        for wi, wave in enumerate(("rayleigh", "love")):
            split_path = plan / "holdout" / ("split_" + wave + ".npz")
            assert sha(split_path) == manifest["curves"][wave]["sha256"]
            record(split_path, wave + " exact holdout curves and perturbations")
            with np.load(split_path, allow_pickle=False) as split:
                amplitude = split["target_maximum_anomaly"]
                primary = amplitude <= 0.08 + 1e-14
                assert primary.sum() == 384
                assert np.all(split["trace_success"][:, :2])
                arrays[wave + "_amplitude"] = amplitude
                arrays[wave + "_achieved_amplitude"] = split["achieved_maximum_anomaly"]
                arrays[wave + "_direction_index"] = split["direction_index"]
                arrays[wave + "_sign"] = split["sign"]
                arrays["frequency_hz"] = split["frequency_hz"]
                for mode in (0, 1):
                    idx = 2 * wi + mode
                    prefix = "curve_%02d_" % idx
                    c0 = jet[prefix + "reference_phase_velocity"]
                    g = jet[prefix + "response_gradient_z"]
                    h = jet[prefix + "response_hessian_z"]
                    z = split["z"]
                    y1 = jet[prefix + "reference_response"] + z @ g.T
                    y2 = y1 + 0.5 * np.einsum("mp,fpq,mq->mf", z, h, z,
                                              optimize=True)
                    if wave == "love":
                        assert np.all(y1 > 0) and np.all(y2 > 0)
                        c1, c2 = c0 / np.sqrt(y1), c0 / np.sqrt(y2)
                    else:
                        c1, c2 = y1, y2
                    exact = split["phase_velocity"][:, mode, :]
                    assert np.all(np.isfinite(exact)) and np.all(exact > 0)
                    key = wave + str(mode)
                    arrays[key + "_exact"] = exact
                    arrays[key + "_order1"] = c1
                    arrays[key + "_order2"] = c2
                    expected = report["gate_result"]["curve_results"][idx]
                    assert (expected["wave"], expected["root_order"]) == (wave, mode)
                    for order, prediction in [(1, c1), (2, c2)]:
                        error = np.abs(prediction - exact) / exact
                        assert np.all(np.isfinite(error))
                        stats = expected["taylor%d_absolute_relative_error" % order]
                        # Same phase-relative absolute tolerance (1e-12) as
                        # Disper26's real-kernel equivalence check. This checks
                        # reconstruction, not a newly chosen accuracy gate.
                        got = np.r_[np.quantile(error[primary], [.5, .9, .99]),
                                    error[primary].max()]
                        ref = [stats[k] for k in ("p50", "p90", "p99", "maximum")]
                        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)
                    boundary = np.isclose(amplitude, .08, rtol=0, atol=1e-14)
                    err = 100 * np.abs(c2[boundary] - exact[boundary]) / exact[boundary]
                    summaries.append({"curve": key, "order2_8pct_p99_percent":
                                      float(np.quantile(err, .99)),
                                      "order2_8pct_max_percent": float(err.max())})
    np.savez_compressed(OUT / "plan_l_v2_holdout_curves.npz", **arrays)
    exported_files = ["plan_l_v2_holdout_curves.npz"]
    for filename in ("holdout_report.json", "benchmark_report.json"):
        path = plan / filename
        if filename != "holdout_report.json":
            record(path, "v2 timing: minimum of five repeats, batch size eight")
        (OUT / ("plan_l_v2_" + filename)).write_bytes(path.read_bytes())
        exported_files.append("plan_l_v2_" + filename)

    for filename in ("long_period_mixed_response_go_20260903.md",
                     "long_period_taylor_no_go_20260903.md"):
        path = PAPER / "data/results" / filename
        record(path, "recorded Plan L scope, costs, and stress-test results")
        (OUT / filename).write_bytes(path.read_bytes())
        exported_files.append(filename)
    result = {"plan_l_code_commit": report["code_commit"],
              "plan_l_artifact_sha256": report["artifact_sha256"],
              "plan_l_frequency_hz": arrays["frequency_hz"].tolist(),
              "plan_l_8pct_summary": summaries,
              "sources": sources,
              "exports": {name: sha(OUT / name) for name in sorted(exported_files)}}
    (OUT / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Exported 4 curves x 448 models x 10 frequencies; published quantiles reproduced.")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disper26-root", type=Path, required=True)
    parser.add_argument("--plan-l-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from paper.provenance import validate_output
    args.output_dir = validate_output(args.output_dir)
    OUT = args.output_dir
    if OUT.resolve() == (PAPER / "data/results").resolve():
        parser.error("Frozen inputs cannot be used as the output directory")
    export(args.disper26_root.resolve(), args.plan_l_dir.resolve())
