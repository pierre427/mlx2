"""CPU checks that failed Ripple evidence runs leave a failing receipt."""
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_ordinary_reference_writes_failing_receipt(tmp_path, monkeypatch):
    probe = _load("probe_ripple_serving")
    import mlx2.adapters.ripple as ripple

    lock = tmp_path / "lock"
    lock.mkdir()
    (lock / "owner.json").write_text(json.dumps({"cpg_task": "t", "cpg_generation": 1}))
    monkeypatch.setitem(sys.modules, "qualification_gpu_ownership", types.SimpleNamespace(
        require_qualification_lease=lambda **_kw: {"lease": "ok"}))
    monkeypatch.setattr(ripple, "inspect_artifact", lambda _path: {
        "storage_format": "affine", "identity": {"fingerprint": "f"}})
    monkeypatch.setattr(probe.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *_a, **_k: "stub")

    def failed_reference(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, "reference")

    monkeypatch.setattr(probe.subprocess, "run", failed_reference)
    output = tmp_path / "evidence"
    monkeypatch.setattr(sys, "argv", ["probe", "--model", str(tmp_path / "m"), "--output",
                                      str(output), "--cpg-lock", str(lock)])
    with pytest.raises(subprocess.CalledProcessError):
        probe.main()
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["passed"] is False
    assert receipt["phase"] == "ordinary_reference"
    assert "CalledProcessError" in receipt["error"]
    assert receipt["thermal_after"] == "stub"
