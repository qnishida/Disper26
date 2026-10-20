"""Publication boundaries must reject stale source and altered frozen input."""
import json
from pathlib import Path
import pytest
from paper.provenance import PAPER, verify_reference


def test_reference_sources_are_complete():
    manifest = json.loads((PAPER / 'reference_manifest.json').read_text())
    for label, entry in manifest['references'].items():
        result = verify_reference(label, PAPER / entry['source_root'])
        assert result['source_tree_sha256'] == entry['source_tree_sha256']
        assert not list((PAPER / entry['source_root']).rglob('.git'))


def test_wrong_source_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='source mismatch'):
        verify_reference('plan_l_v2', tmp_path)


def test_changed_artifact_is_rejected(tmp_path):
    changed = tmp_path / 'changed.npz'
    changed.write_bytes(b'not the frozen artifact')
    with pytest.raises(ValueError, match='artifact mismatch'):
        verify_reference('plan_l_v2', PAPER / 'reference_sources/plan_l_v2', changed)


def test_frozen_holdout_statistics_replay(tmp_path):
    from paper.reproduce_statistics import reproduce
    report = reproduce(tmp_path / 'statistics.json')
    assert len(report['cases']) == 8
    assert report['role'] == 'replay_of_previously_opened_holdout'
    assert max(r['maximum_reconstruction_difference'] for r in report['cases']) <= 1e-12


def test_output_cannot_overlap_fixed_evidence():
    from paper.provenance import PAPER, validate_output
    import pytest
    for path in [PAPER, PAPER / 'data', PAPER / 'data/results/new', PAPER / 'reference_sources/plan_l_v2/build']:
        with pytest.raises(ValueError):
            validate_output(path)
    assert validate_output(PAPER / 'results/new') == (PAPER / 'results/new').resolve()


def test_hessian_diagnostic_defaults_to_separate_output(tmp_path):
    from paper.scripts.verify_rayleigh_hessian import parse_args
    args = parse_args(['--disper26-root', str(tmp_path), '--plan-l-dir', str(tmp_path)])
    assert args.output_dir == (PAPER / 'results/hessian').resolve()


@pytest.mark.parametrize('output', ['data/results', 'reference_sources/plan_l_v2'])
def test_hessian_diagnostic_rejects_fixed_output(tmp_path, output):
    from paper.scripts.verify_rayleigh_hessian import parse_args
    with pytest.raises(ValueError, match='immutable scientific inputs'):
        parse_args(['--disper26-root', str(tmp_path), '--plan-l-dir', str(tmp_path),
                    '--output-dir', str(PAPER / output)])
