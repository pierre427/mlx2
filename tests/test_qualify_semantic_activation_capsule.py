"""Host-only tests for the semantic activation native gate harness."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/research/qualify_semantic_activation_capsule.py"
SPEC = importlib.util.spec_from_file_location("semantic_activation_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def _owner(path: Path, *, lease: str = "lease-1", cpg: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"lease_id": lease, "cpg_used": cpg, "pid": 4100}) + "\n"
    )


def test_import_and_default_path_are_host_only():
    assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)
    assert gate.DEFAULT_RECEIPT.parts[-3:] == (
        "artifacts",
        "research",
        "semantic-activation-capsule-m3.json",
    )
    assert gate.ARMS == (
        "ordinary",
        "zero_gate",
        "ordered",
        "reversed",
        "sign_negated",
        "wrong_capsule",
    )


def test_native_authorization_requires_exact_paired_cpg_lease(tmp_path, monkeypatch):
    first = tmp_path / "host/owner.json"
    second = tmp_path / "tmp/owner.json"
    _owner(first)
    _owner(second)
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(gate.os, "getppid", lambda: 4100)

    receipt = gate.require_native_authorization(
        execute_native=True,
        owns_gpu=True,
        lease_id="lease-1",
        hardware_model=lambda: gate.EXPECTED_HOST_MODEL,
        owner_paths=(first, second),
    )
    assert receipt["lease_id"] == "lease-1"

    _owner(second, lease="other")
    with pytest.raises(RuntimeError, match="do not match"):
        gate.require_native_authorization(
            execute_native=True,
            owns_gpu=True,
            lease_id="lease-1",
            hardware_model=lambda: gate.EXPECTED_HOST_MODEL,
            owner_paths=(first, second),
        )
    _owner(second, cpg=False)
    _owner(first, cpg=False)
    receipt = gate.require_native_authorization(
        execute_native=True,
        owns_gpu=True,
        lease_id=None,
        hardware_model=lambda: gate.EXPECTED_HOST_MODEL,
        owner_paths=(first, second),
    )
    assert receipt["cpg_used"] is False
    assert receipt["ownership_proof"] == "parent-wrapper-pid"


@pytest.mark.parametrize(
    ("execute", "owns", "lease", "message"),
    [
        (False, True, "lease-1", "--execute-native"),
        (True, False, "lease-1", "--i-own-the-gpu"),
    ],
)
def test_native_authorization_refuses_missing_explicit_authority(
    execute, owns, lease, message
):
    with pytest.raises(RuntimeError, match=message):
        gate.require_native_authorization(
            execute_native=execute,
            owns_gpu=owns,
            lease_id=lease,
            hardware_model=lambda: gate.EXPECTED_HOST_MODEL,
            owner_paths=(),
        )


def _fake_rows(*, break_zero=False, break_order=False, nonfinite=False):
    rows = []
    for prompt_index in range(len(gate.PROMPTS)):
        ordinary = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        values = {
            "ordinary": ordinary,
            "zero_gate": ordinary.copy(),
            "ordered": np.asarray([[3.0, 2.0, 1.0]], dtype=np.float32),
            "reversed": np.asarray([[2.0, 3.0, 1.0]], dtype=np.float32),
            "sign_negated": np.asarray([[-3.0, -2.0, -1.0]], dtype=np.float32),
            "wrong_capsule": np.asarray([[1.5, 0.5, 2.5]], dtype=np.float32),
        }
        if break_zero:
            values["zero_gate"][0, 0] += 1
        if break_order:
            values["reversed"] = values["ordered"].copy()
        if nonfinite:
            values["wrong_capsule"][0, 0] = np.nan
        for arm in gate.ARMS:
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "arm": arm,
                    "next_token": int(np.argmax(values[arm])),
                    "logits": values[arm],
                    "receipt": {
                        "route": "ordinary",
                        "forward_completed": True,
                        "evaluated_logits": True,
                        "observed_used": False,
                    },
                }
            )
    return rows


def test_evaluate_rows_requires_identity_finiteness_receipts_and_sensitivity():
    checks, public = gate.evaluate_rows(_fake_rows())
    assert all(value for key, value in checks.items() if key != "per_prompt")
    assert len(public) == len(gate.PROMPTS) * len(gate.ARMS)
    assert all("logits" not in row and len(row["logits_sha256"]) == 64 for row in public)

    assert not gate.evaluate_rows(_fake_rows(break_zero=True))[0][
        "zero_gate_exact_logits_identity"
    ]
    assert not gate.evaluate_rows(_fake_rows(break_order=True))[0][
        "ordered_vs_reversed_logits_sensitive"
    ]
    assert not gate.evaluate_rows(_fake_rows(nonfinite=True))[0]["all_logits_finite"]


def test_direct_forward_receipt_never_fabricates_serving_observed_use():
    prepared = {
        "receipt": {
            "schema": "mlx2-activation-capsule-injection-receipt-v1",
            "status": "prepared",
            "route": "ordinary",
            "selected": True,
            "engaged": False,
            "observed_used": False,
            "relative_gate": 0.2,
        }
    }
    receipt = gate._forward_receipt(prepared)

    assert receipt["selected"] is True
    assert receipt["forward_completed"] is True
    assert receipt["evaluated_logits"] is True
    assert receipt["engaged"] is False
    assert receipt["observed_used"] is False


def test_artifact_manifest_recomputes_weight_and_envelope_fingerprints(tmp_path):
    import hashlib

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    weights = b"host-only-test-weights"
    (artifact / "weights.npz").write_bytes(weights)
    manifest = {
        "schema": "mlx2-neural-concept-bridge-v1",
        "bindings": {
            "model": gate.MODEL_FINGERPRINT,
            "tokenizer": gate.MODEL_FINGERPRINT,
            "runtime": "runtime-fingerprint",
        },
        "hidden_dim": 4096,
        "weights_sha256": hashlib.sha256(weights).hexdigest(),
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["fingerprint"] = hashlib.sha256(encoded + weights).hexdigest()
    (artifact / "manifest.json").write_text(json.dumps(manifest))

    assert gate._artifact_manifest(artifact)["fingerprint"] == manifest["fingerprint"]
    (artifact / "weights.npz").write_bytes(weights + b"corrupt")
    with pytest.raises(ValueError, match="weight digest mismatch"):
        gate._artifact_manifest(artifact)


def test_real_arm_payloads_round_trip_through_strict_core_store(tmp_path):
    from mlx2.runtime.neural_concepts import (
        NeuralConceptArtifact,
        RecurrentConceptEncoder,
    )

    root = Path(
        os.environ.get(
            "MLX2_SEMANTIC_CAPSULE_ARTIFACT",
            ROOT / "qualification/artifacts/qwen35-9b-deep-concept-directory",
        )
    )
    if not (root / "manifest.json").is_file():
        pytest.skip("real neural concept artifact is not available")
    manifest = json.loads((root / "manifest.json").read_text())
    artifact = NeuralConceptArtifact.load(
        root,
        model_binding=manifest["bindings"]["model"],
        tokenizer_binding=manifest["bindings"]["tokenizer"],
        runtime_binding=manifest["bindings"]["runtime"],
    )
    payloads = gate.publish_arm_payloads(
        artifact,
        RecurrentConceptEncoder,
        tmp_path / "private-store",
        gate=0.2,
        producer_revision="a" * 40,
    )

    assert payloads["ordinary"] is None
    for name in gate.ARMS[1:]:
        payload = payloads[name]
        assert len(payload["capsule_digest"]) == 64
        assert payload["manifest"]["exact_state_schema"] == "mlx2-apcv2-exact-state"
        assert payload["manifest"]["provenance"]["source_digests"]
        assert payload["manifest"]["metadata"]["control_label"] == name
        assert payload["tensors"]["prefix"].flags.writeable is False
    assert payloads["zero_gate"]["gate"] == 0.0
    assert not np.array_equal(
        payloads["ordered"]["tensors"]["prefix"],
        payloads["reversed"]["tensors"]["prefix"],
    )
    assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)
