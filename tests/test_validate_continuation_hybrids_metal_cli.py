"""Ownership CLI is host-only; numerical harness seam tested under explicit CPU."""

import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_continuation_hybrids_metal.py"


def module():
    spec = importlib.util.spec_from_file_location("continuation_hybrids_cli", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_dry_run_imports_no_tensor_libraries_or_creates_output(tmp_path):
    out = tmp_path / "not-executed.json"
    code = """
import builtins,runpy,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('mlx','mlx2','numpy','torch'):
        raise AssertionError('unexpected tensor import '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
sys.argv=[sys.argv[1],'--out',sys.argv[2],'--dry-run']
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT), str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert len(report["cells_planned"]) == len(set(report["cells_planned"])) == 18
    assert (
        report["unit"] == "complete_continuation_sequences"
        and report["max_sequences"] == 15
    )
    assert not report["will_execute"] and not report["qualified"] and not out.exists()
    assert not report["performance_claim"] and not report["gpu_training"]


def test_ownership_and_deadline_rejected_before_runtime_import(tmp_path):
    value = module()
    args = value.parser().parse_args(["--out", str(tmp_path / "report.json")])
    with pytest.raises(ValueError, match="i-own-the-gpu"):
        value.preflight(args)
    args.i_own_the_gpu = True
    for deadline in (0, 901):
        args.deadline_seconds = deadline
        with pytest.raises(ValueError, match="deadline"):
            value.preflight(args)


def test_source_identity_binds_executor_and_shared_fixture_helpers():
    identity = module().source_identity()
    files = identity["files_sha256"]
    for path in (
        "scripts/validate_continuation_hybrids_metal.py",
        "scripts/validate_parallel_hybrids_metal.py",
        "src/mlx2/runtime/continuation_verification.py",
        "src/mlx2/runtime/proposal_pool.py",
        "tests/test_nemotron_external_taps_cpu.py",
        "tests/test_qwen38_dflash2_cpu.py",
    ):
        assert path in files and len(files[path]) == 64


@pytest.mark.parametrize("family", ["mamba2", "gdn", "qsa"])
@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
def test_all_hybrid_harness_operations_use_real_cpu_substitute_and_not_gpu(
    family, kind, monkeypatch
):
    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.models import qwen4_exp

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    value = module()
    runner = value.Validation.__new__(value.Validation)
    runner.mx, runner.np = mx, np
    runner.report = {"explicit_cpu_substitute": True}
    runner.refs = {
        name: importlib.import_module(name) for name in value._helpers.REFERENCES
    }
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(qwen4_exp, "_QSA_POOLED_KEY_CACHE", True)
    monkeypatch.setattr(qwen4_exp, "_QSA_APC_SUMMARIES", True)

    def require_cpu():
        assert mx.default_device() == mx.cpu

    runner.require_gpu = require_cpu  # Explicit test seam, never a native Metal claim.
    try:
        for operation in ("pool15_prefix", "mixed_sampled", "late_failure_retry"):
            runner.current = {"comparisons": []}
            getattr(runner, operation)(family, kind)
            assert runner.current["comparisons"]
            assert all(row["passed"] for row in runner.current["comparisons"])
    finally:
        mx.set_default_device(previous)
