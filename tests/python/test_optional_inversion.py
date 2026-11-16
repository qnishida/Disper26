"""The forward API works without SciPy; inverse users get an install hint."""
from pathlib import Path
import os
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[2]


def run_without_scipy(tmp_path, body, missing="scipy"):
    script = textwrap.dedent("""\
        import importlib.abc
        import sys

        class MissingScipy(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "scipy":
                    raise ModuleNotFoundError("Simulated unavailable dependency", name=MISSING)

        sys.meta_path.insert(0, MissingScipy())
        """)
    script = script.replace("MISSING", repr(missing)) + textwrap.dedent(body)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(ROOT / "inversion")))
    return subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env,
        text=True, capture_output=True,
    )


def test_forward_api_runs_without_scipy(tmp_path):
    result = run_without_scipy(tmp_path, """\
        import disper26 as d26
        model = d26.create_model([0.0, 10.0], kind="layer_cake",
                                 vs=[3.0, 4.5], vp=[5.2, 8.0], rho=[2.5, 3.3])
        solution = d26.compute_dispersion(model, period=5.0)
        assert 0.0 < solution.phase_velocity < 4.5
        assert "scipy" not in sys.modules
        """)
    assert result.returncode == 0, result.stderr


def test_inversion_without_scipy_explains_installation(tmp_path):
    result = run_without_scipy(tmp_path, """\
        try:
            import disper26_inv
        except ImportError as exc:
            assert ".[inversion]" in str(exc)
            assert isinstance(exc.__cause__, ModuleNotFoundError)
            assert exc.__cause__.name == "scipy"
        else:
            raise AssertionError("Inversion unexpectedly imported without SciPy")
        """)
    assert result.returncode == 0, result.stderr


def test_other_scipy_import_failures_are_preserved(tmp_path):
    result = run_without_scipy(tmp_path, """\
        try:
            import disper26_inv
        except ModuleNotFoundError as exc:
            assert exc.name == "scipy_internal_dependency"
        else:
            raise AssertionError("The original SciPy import failure was hidden")
        """, missing="scipy_internal_dependency")
    assert result.returncode == 0, result.stderr
