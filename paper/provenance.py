"""Validate source-only reference snapshots without importing old Git history."""
import hashlib
import json
from pathlib import Path

PAPER = Path(__file__).resolve().parent


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_reference(reference_id, source_root, artifact=None):
    manifest = json.loads((PAPER / 'reference_manifest.json').read_text())
    entry = manifest['references'][reference_id]
    root = Path(source_root).resolve()
    files = entry['files']
    for rel, expected in files.items():
        p = root / rel
        if not p.is_file() or p.is_symlink() or file_sha256(p) != expected:
            raise ValueError('Reference source mismatch: ' + rel)
    # Extra numerical sources could shadow an approved include/import.
    for p in root.rglob('*'):
        rel = p.relative_to(root)
        if any(part in ('build', '__pycache__', '.git', '.pytest_cache') for part in rel.parts):
            continue
        if p.is_file() and p.suffix in ('.py', '.cpp', '.hpp', '.h', '.cmake', '.f90', '.csv') and str(rel) not in files:
            raise ValueError('Unlisted reference source: ' + str(rel))
    tree = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if tree != entry['source_tree_sha256']:
        raise ValueError('Reference manifest tree fingerprint mismatch')
    if artifact is not None and file_sha256(artifact) != entry['artifact_sha256']:
        raise ValueError('Frozen artifact mismatch')
    return {'reference_id': reference_id, 'source_tree_sha256': tree,
            'autodiff_commit': entry['autodiff_commit']}


def reference_for_root(source_root):
    root = Path(source_root).resolve()
    manifest = json.loads((PAPER / 'reference_manifest.json').read_text())
    for name, entry in manifest['references'].items():
        if root == (PAPER / entry['source_root']).resolve():
            return verify_reference(name, root)
    raise ValueError('Choose one of the source-only snapshots in reference_manifest.json')


def validate_output(path):
    """Keep new results outside immutable evidence and reference sources."""
    path = Path(path).resolve()
    for frozen in [PAPER / 'data', PAPER / 'reference_sources']:
        frozen = frozen.resolve()
        if path == frozen or frozen in path.parents or path in frozen.parents:
            raise ValueError('Output overlaps immutable scientific inputs: ' + str(path))
    return path
