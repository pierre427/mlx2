"""Dry-run and ownership validation must never construct Metal work."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_parallel_hybrids_metal.py"


def module():
    spec = importlib.util.spec_from_file_location("hybrid_metal_cli", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_dry_run_imports_no_tensor_runtime_and_writes_no_receipt(tmp_path):
    out = tmp_path / "unexecuted.json"
    code = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('mlx', 'mlx2', 'numpy', 'torch'):
        raise AssertionError('dry-run imported a tensor runtime: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = [sys.argv[1], '--out', sys.argv[2], '--dry-run']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT), str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert len(report["cells_planned"]) == 48
    assert len(set(report["cells_planned"])) == 48
    assert not report["will_execute"] and not report["qualified"]
    assert not report["gpu_training"] and not out.exists()


def test_live_execution_requires_explicit_gpu_ownership(tmp_path):
    value = module()
    args = value.parser().parse_args(["--out", str(tmp_path / "receipt.json")])
    with pytest.raises(ValueError, match="i-own-the-gpu"):
        value.preflight(args)
    args.i_own_the_gpu = True
    assert value.preflight(args)["will_execute"]


@pytest.mark.parametrize("deadline", [0, 901])
def test_deadline_limits_precede_runtime_import(tmp_path, deadline):
    value = module()
    args = value.parser().parse_args(
        [
            "--out",
            str(tmp_path / "receipt.json"),
            "--dry-run",
            "--deadline-seconds",
            str(deadline),
        ]
    )
    with pytest.raises(ValueError, match="deadline"):
        value.preflight(args)


def test_source_identity_closes_script_src_and_fixture_sources():
    identity = module().source_identity()
    files = identity["files_sha256"]
    assert "scripts/validate_parallel_hybrids_metal.py" in files
    assert "src/mlx2/runtime/drafters/dpara.py" in files
    assert "tests/test_dpara_cpu.py" in files
    assert "tests/test_batched_mtp.py" in files
    assert all(len(digest) == 64 for digest in files.values())


def test_host_empty_tensor_never_exports_null_buffer():
    import numpy as np

    value = module().Validation.__new__(module().Validation)
    value.np = np

    class Empty:
        size = 0
        shape = (2, 0, 3)

        def astype(self, _dtype):
            raise AssertionError("empty tensor must not request native buffer")

    result = value.host(Empty())
    assert result.shape == (2, 0, 3) and result.dtype == np.float32
