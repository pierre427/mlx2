"""The new token native gate defaults to dry-run and uses fake bytes only in CPU mode."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/research/varlen_token_native_gate.py"
spec = importlib.util.spec_from_file_location("varlen_token_native_gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_dry_run_is_64_token_profile_and_no_native_claim():
    result = gate.dry_run()
    assert result["accepted_token_boundaries"] == [63, 64, 65]
    assert result["arena_bytes_total"] == 131072
    assert not result["gpu_executed"]
    assert "attention on opaque arena" in result["not_proven"]


def test_fixture_varies_by_token_head_and_plane():
    key_h0 = gate.pattern("key", 63, 2, 0, 256)
    key_h1 = gate.pattern("key", 63, 2, 1, 256)
    val_h0 = gate.pattern("value", 63, 2, 0, 256)
    assert key_h0.size == 512
    assert not (key_h0 == key_h1).all()
    assert not (key_h0 == val_h0).all()
    assert not (key_h0[:256] == key_h0[256:]).all()


def test_cpu_fake_exact_token_layout_and_publication():
    result = gate.cpu_fake()
    assert [(case["tokens"], case["pages"], case["regions"])
            for case in result["boundary_cases"]] == [
                (63, 1, 2), (64, 1, 2), (65, 2, 4)]
    assert result["exact_fake_bytes"] and result["generation_reuse"]
    assert not result["gpu_executed"] and not result["native_proof"]


def test_preflight_requires_matching_two_lock_receipts_and_pinned_bytes(tmp_path, monkeypatch):
    monkeypatch.delenv("GPUQ_LEASE", raising=False)
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    owner = {"session": "test-token", "lease_id": "test-token-token63-65",
             "label": "token63-65", "pid": os.getpid()}
    for lock in (left, right):
        (lock / "owner.json").write_text(json.dumps(owner))
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "native.cpp"
    source.write_bytes(b"reviewed source")
    binary = tmp_path / "native.so"
    binary.write_bytes(b"reviewed binary")
    expected_binary = gate.sha256(binary)
    expected_sources = {"native.cpp": gate.sha256(source)}
    result = gate.preflight(locks=(left, right), binary=binary,
                            source_root=source_root, session="test-token",
                            expected_binary=expected_binary,
                            expected_sources=expected_sources)
    assert result["gpuq_owner"] == owner
    assert result["binary_sha256"] == gate.sha256(binary)
    (right / "owner.json").write_text(json.dumps({**owner, "lease_id": "foreign"}))
    with pytest.raises(RuntimeError, match="owner mismatch"):
        gate.preflight(locks=(left, right), binary=binary,
                       source_root=source_root, session="test-token",
                       expected_binary=expected_binary,
                       expected_sources=expected_sources)
    (right / "owner.json").write_text(json.dumps(owner))
    source.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="source hashes"):
        gate.preflight(locks=(left, right), binary=binary,
                       source_root=source_root, session="test-token",
                       expected_binary=expected_binary,
                       expected_sources=expected_sources)
    source.write_bytes(b"reviewed source")
    binary.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="binary differs"):
        gate.preflight(locks=(left, right), binary=binary,
                       source_root=source_root, session="test-token",
                       expected_binary=expected_binary,
                       expected_sources=expected_sources)


def test_failed_gpu_preflight_writes_failure_receipt_without_mlx_import(tmp_path):
    receipt = tmp_path / "failure.json"
    env = {key: value for key, value in os.environ.items()
           if key not in ("GPUQ_SESSION", "GPUQ_LEASE")}
    # The default binary sits in /tmp and can be cleared by macOS; the receipt
    # must still hash whatever binary the gate was pointed at.
    binary = tmp_path / "_paged_kv_native.so"
    binary.write_bytes(b"stand-in build")
    env["MLX2_VARLEN_BINDING_SO"] = str(binary)
    command = [sys.executable, str(SCRIPT), "--execute-gpu", "--receipt", str(receipt)]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    failure = json.loads(receipt.read_text())
    assert failure["outcome"] == "failed" and not failure["gpu_executed"]
    assert failure["failure_stage"] == "gpuq-and-build-preflight"
    assert failure["binary_sha256"] and failure["source_sha256"]
    assert "mlx.core" not in result.stderr


def test_alarm_is_a_hard_failure():
    with pytest.raises(TimeoutError, match="300s hard cap"):
        gate._alarm(None, None)
