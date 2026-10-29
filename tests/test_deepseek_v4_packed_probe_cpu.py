import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

PATH = (
    Path(__file__).resolve().parents[1] / "scripts/research/probe_deepseek_v4_packed.py"
)
spec = importlib.util.spec_from_file_location("packed_probe", PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("kind", [8, 10, 16])
def test_independent_format_oracle(kind):
    pytest.importorskip("gguf")
    raw = probe.synthetic((3, 256), kind, seed=90)
    external = probe.independent_dequant(raw, kind, (3, 256))
    native = probe.decode_blocks(raw.tobytes(), kind).reshape(3, 256)
    assert probe.compare(native, external, atol=0, rtol=0)["passed"]


def test_numerical_failure_remains_json_serializable():
    result = probe.compare([np.nan], [1])
    assert not result["passed"]
    json.dumps(result, allow_nan=False)


def test_gpu_requires_owned_window_before_evidence_or_allocations(tmp_path):
    with pytest.raises(SystemExit):
        probe.main(["--output", str(tmp_path / "out"), "--run-gpu"])
    assert not (tmp_path / "out").exists()


def test_receipt_identity_and_expiry_fail_closed(tmp_path):
    lease = {
        "cpg_task": "task",
        "cpg_session": "session",
        "cpg_generation": 1,
        "cpg_worker": "worker",
        "expires": time.time() + 60,
        "pid": os.getpid(),
    }
    (tmp_path / "desktop-lease.json").write_text(json.dumps(lease))
    owner = {"session": "session", "lease_id": "paired", "pid": os.getpid()}
    paths = [tmp_path / "owner1.json", tmp_path / "owner2.json"]
    for path in paths:
        path.write_text(json.dumps(owner))
    assert probe.verify_ownership(tmp_path, paths)["paired_owner"] == owner
    lease["finished_at"] = time.time()
    (tmp_path / "desktop-lease.json").write_text(json.dumps(lease))
    with pytest.raises(RuntimeError, match="finished or expired"):
        probe.verify_ownership(tmp_path, paths)


def test_q2_fixture_exercises_nonzero_minimum_offset():
    raw = probe.synthetic((3, 256), 10, seed=90).reshape(-1, 84)
    assert np.all(
        np.frombuffer(np.ascontiguousarray(raw[:, 82:84]).tobytes(), dtype="<f2") > 0
    )
    assert np.any((raw[:, :16] >> 4) > 0)
    # Independent GGUF decoder must include negative offset-only coefficients.
    decoded = probe.independent_dequant(raw, 10, (3, 256))
    assert np.any(decoded < 0)
