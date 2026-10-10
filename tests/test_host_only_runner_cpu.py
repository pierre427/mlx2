import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "scripts/run_host_only_pytest.py"
spec = importlib.util.spec_from_file_location("host_only_runner", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("module", ["mlx.core", "torch", "_paged_kv_native"])
def test_startup_guard_stops_framework_import_before_loading(tmp_path, module):
    (tmp_path / "sitecustomize.py").write_text(runner.GUARD)
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    process = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert process.returncode != 0
    assert "Framework import forbidden in host-only validation" in process.stderr


def test_guard_allows_stdlib_and_does_not_fake_framework_modules(tmp_path):
    (tmp_path / "sitecustomize.py").write_text(runner.GUARD)
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; assert 'mlx' not in sys.modules; print(json.dumps(1))",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert process.returncode == 0
    assert process.stdout.strip() == "1"


@pytest.mark.parametrize("startup", ["absent", "broken"])
def test_missing_guard_stops_before_pytest_import(tmp_path, startup):
    if startup == "broken":
        (tmp_path / "sitecustomize.py").write_text("raise RuntimeError('broken guard')")
    # A fake pytest reports if it was reached. Neither startup failure may get
    # as far as importing test machinery.
    (tmp_path / "pytest.py").write_text("raise AssertionError('pytest was reached')")
    process = subprocess.run(
        [sys.executable, "-c", runner.BOOTSTRAP],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert process.returncode != 0
    assert "Host-only startup guard is missing" in process.stderr
    assert "pytest was reached" not in process.stderr
