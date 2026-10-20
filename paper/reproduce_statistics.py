"""Replay previously opened Plan L holdout statistics without refitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from paper.provenance import PAPER, file_sha256, verify_reference


def reproduce(output):
    data = PAPER / 'data/plan_l_v2'
    artifact = data / 'long_period_mixed_response_v2.npz'
    source = PAPER / 'reference_sources/plan_l_v2'
    provenance = verify_reference('plan_l_v2', source, artifact)
    report = json.loads((data / 'holdout_report.json').read_text())
    manifest = json.loads((data / 'holdout/split_manifest.json').read_text())
    if report['evidence_role'] != 'sealed_holdout' or not report['gate_result']['passed']:
        raise ValueError('Unexpected frozen evidence role or gate')
    rows = []
    with np.load(artifact, allow_pickle=False) as jets:
        for wi, wave in enumerate(('rayleigh', 'love')):
            split = data / 'holdout' / ('split_' + wave + '.npz')
            if file_sha256(split) != manifest['curves'][wave]['sha256']:
                raise ValueError('Frozen split mismatch')
            with np.load(split, allow_pickle=False) as models:
                primary = models['target_maximum_anomaly'] <= 0.08 + 1e-14
                if primary.sum() != 384 or not np.all(models['trace_success'][:, :2]):
                    raise ValueError('Incomplete frozen holdout')
                z = models['z']
                for mode in (0, 1):
                    idx = 2 * wi + mode
                    prefix = 'curve_%02d_' % idx
                    c0 = jets[prefix + 'reference_phase_velocity']
                    y1 = jets[prefix + 'reference_response'] + z @ jets[prefix + 'response_gradient_z'].T
                    y2 = y1 + 0.5 * np.einsum('mp,fpq,mq->mf', z, jets[prefix + 'response_hessian_z'], z, optimize=True)
                    if wave == 'love':
                        if not np.all(y1 > 0) or not np.all(y2 > 0):
                            raise ValueError('Nonpositive slowness response')
                        predictions = (c0 / np.sqrt(y1), c0 / np.sqrt(y2))
                    else:
                        predictions = (y1, y2)
                    exact = models['phase_velocity'][:, mode, :]
                    for order, prediction in enumerate(predictions, 1):
                        error = np.abs(prediction - exact) / exact
                        if not np.all(np.isfinite(error)):
                            raise ValueError('Nonfinite error')
                        actual = np.r_[np.quantile(error[primary], [.5, .9, .99]), error[primary].max()]
                        expected = report['gate_result']['curve_results'][idx]['taylor%d_absolute_relative_error' % order]
                        reference = np.array([expected[k] for k in ('p50', 'p90', 'p99', 'maximum')])
                        # Existing manuscript reconstruction contract, not a new solver tolerance.
                        np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-12)
                        rows.append(dict(wave=wave, mode=mode, order=order,
                                         p50=float(actual[0]), p90=float(actual[1]),
                                         p99=float(actual[2]), maximum=float(actual[3]),
                                         maximum_reconstruction_difference=float(np.max(abs(actual-reference)))))
    result = dict(role='replay_of_previously_opened_holdout', provenance=provenance,
                  artifact_sha256=file_sha256(artifact), cases=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=PAPER / 'results/replayed_statistics.json')
    result = reproduce(ap.parse_args().output)
    print('Verified eight fixed holdout summaries; maximum reconstruction difference:',
          max(r['maximum_reconstruction_difference'] for r in result['cases']))
