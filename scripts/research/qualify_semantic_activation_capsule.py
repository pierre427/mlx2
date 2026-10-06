#!/usr/bin/env python3
"""Native mechanism gate for Qwen3.5 semantic activation capsules.

This is a mechanism experiment, not a quality or performance qualification.
Manifest mode is host-only and never imports MLX.  Native mode is deliberately
hard to enter: it requires an explicit execution flag, the pinned M3 host, and
matching owner receipts for both shared GPU locks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_SNAPSHOT = "8b2b98c00a6b4d291155e4890773ca8f769aee53"
MODEL_FINGERPRINT = "07e2d06a4054c3fc5c015c9adbf5fae16a95f5be7d8cdcb2ed9884ab8113f657"
DEFAULT_MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots"
    / MODEL_SNAPSHOT
)
DEFAULT_ARTIFACT = ROOT / "qualification/artifacts/qwen35-9b-deep-concept-directory"
DEFAULT_RECEIPT = ROOT / "artifacts/research/semantic-activation-capsule-m3.json"
EXPECTED_HOST_MODEL = "Mac15,7"
LOCK_OWNER_PATHS = (
    Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
    Path("/tmp/gpu.lock/owner.json"),
)
SCHEMA = "mlx2.semantic-activation-capsule-mechanism-gate.v1"
PROMPTS = (
    "Continue the fixed marker Alder-17 with exactly one token.",
    "Continue the fixed marker Cobalt-29 with exactly one token.",
)
ARMS = (
    "ordinary",
    "zero_gate",
    "ordered",
    "reversed",
    "sign_negated",
    "wrong_capsule",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(root: Path = ROOT, explicit: str | None = None) -> str:
    if explicit is not None:
        if (
            len(explicit) != 40
            or any(character not in "0123456789abcdef" for character in explicit)
        ):
            raise ValueError("source revision must be a full lowercase Git SHA")
        return explicit
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _artifact_manifest(artifact: Path) -> dict[str, Any]:
    manifest_path = artifact / "manifest.json"
    weights_path = artifact / "weights.npz"
    if artifact.is_symlink() or not artifact.is_dir():
        raise ValueError("neural concept artifact must be a real directory")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "mlx2-neural-concept-bridge-v1":
        raise ValueError("unsupported neural concept artifact schema")
    if manifest.get("bindings", {}).get("model") != MODEL_FINGERPRINT:
        raise ValueError("neural concept artifact is not bound to the pinned model")
    if manifest.get("bindings", {}).get("tokenizer") != MODEL_FINGERPRINT:
        raise ValueError("neural concept artifact tokenizer binding mismatch")
    if manifest.get("weights_sha256") != sha256_file(weights_path):
        raise ValueError("neural concept artifact weight digest mismatch")
    payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    fingerprint = hashlib.sha256(encoded + weights_path.read_bytes()).hexdigest()
    if manifest.get("fingerprint") != fingerprint:
        raise ValueError("neural concept artifact fingerprint mismatch")
    if int(manifest.get("hidden_dim", 0)) != 4096:
        raise ValueError("neural concept artifact hidden geometry mismatch")
    return manifest


def _model_identity(model: Path) -> dict[str, Any]:
    """Reproduce the adapter's host-only, revision-bound artifact identity."""
    config = json.loads((model / "config.json").read_text())
    text = config.get("text_config", config)
    expected = {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "full_attention_interval": 4,
        "vocab_size": 248320,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    if config.get("model_type") != "qwen3_5" or text.get("num_experts", 0):
        raise ValueError("pinned model is not the dense qwen3_5 artifact")
    if any(text.get(key) != value for key, value in expected.items()):
        raise ValueError("pinned model topology mismatch")
    index = json.loads((model / "model.safetensors.index.json").read_text()).get(
        "weight_map"
    )
    if not isinstance(index, dict) or not index:
        raise ValueError("pinned model has no indexed weights")
    names = sorted(set(index.values()))
    digest = hashlib.sha256()
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        path = model / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    records = []
    for name in names:
        relative = Path(name) if isinstance(name, str) else Path("/")
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("weight shard paths must stay within the model")
        path = model / relative
        if not path.is_file():
            raise ValueError(f"missing model weight shard: {name}")
        stat = path.stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record)
        digest.update(json.dumps(record).encode())
    # The adapter's existing inspector identity intentionally includes local
    # ``mtime_ns`` values, so it is a host-local revision receipt rather than
    # a portable content digest.  The exact snapshot, topology, shard set and
    # all small-file bytes are independently pinned above; retain both IDs.
    content = hashlib.sha256()
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        path = model / name
        if path.is_file():
            content.update(name.encode())
            content.update(path.read_bytes())
    for name, size, _mtime_ns in records:
        content.update(json.dumps((name, size)).encode())
    return {
        "fingerprint": digest.hexdigest(),
        "content_manifest_sha256": content.hexdigest(),
        "files": records,
        "config": config,
    }


def build_manifest(
    model: Path,
    artifact: Path,
    output: Path,
    *,
    source_revision: str | None = None,
) -> dict[str, Any]:
    """Build a no-MLX execution plan and verify pinned local inputs."""
    model = model.expanduser().resolve()
    artifact = artifact.expanduser().resolve()
    if model.name != MODEL_SNAPSHOT:
        raise ValueError("model path is not the pinned Qwen3.5-9B-4bit snapshot")
    required = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    missing = [name for name in required if not (model / name).is_file()]
    if missing:
        raise ValueError(f"pinned model is incomplete: {missing}")
    identity = _model_identity(model)
    artifact_manifest = _artifact_manifest(artifact)
    return {
        "schema": SCHEMA,
        "mode": "manifest-only",
        "status": "planned-not-run",
        "claim_boundary": (
            "mechanism sensitivity only; no quality, performance, route "
            "qualification, selection, or production-use claim"
        ),
        "host_constraint": {"machine": "arm64", "hw_model": EXPECTED_HOST_MODEL},
        "model": {
            "path": str(model),
            "snapshot": MODEL_SNAPSHOT,
            "inspector_fingerprint": identity["fingerprint"],
            "content_manifest_sha256": identity["content_manifest_sha256"],
            "established_artifact_binding": MODEL_FINGERPRINT,
            "identity_note": (
                "inspector fingerprint is host-local because it includes shard "
                "mtime_ns; snapshot/topology/shards and content manifest are pinned"
            ),
            "weight_shards": len(identity["files"]),
            "config_sha256": sha256_file(model / "config.json"),
            "index_sha256": sha256_file(model / "model.safetensors.index.json"),
        },
        "artifact": {
            "path": str(artifact),
            "fingerprint": artifact_manifest["fingerprint"],
            "weights_sha256": artifact_manifest["weights_sha256"],
            "bindings": artifact_manifest["bindings"],
            "injection_layer": int(artifact_manifest["deep_injection_layer"]),
            "state_dim": int(artifact_manifest["state_dim"]),
            "hidden_dim": int(artifact_manifest["hidden_dim"]),
        },
        "execution": {
            "route": "ordinary",
            "batch_size": 1,
            "deterministic": True,
            "prompts": list(PROMPTS),
            "arms": list(ARMS),
            "gate": 0.2,
            "native_imports_deferred": True,
            "required_locks": [str(path) for path in LOCK_OWNER_PATHS],
            "receipt_path": str(output.expanduser().resolve()),
        },
        "criteria": {
            "zero_gate_exact_logits_identity": True,
            "zero_gate_exact_next_token_identity": True,
            "all_logits_finite": True,
            "all_arms_have_route_receipts": True,
            "ordered_vs_reversed_logits_sensitive": True,
            "ordered_vs_sign_negated_logits_sensitive": True,
        },
        "provenance": {
            "git_revision": git_head(explicit=source_revision),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "adapter_seam": "mlx2.adapters.qwen35_9b.Qwen359BAdapter.activation_capsule_prefill",
            "model_seam": "deep_concept_memory",
            "construction": "artifact-projected provenance-bound same-model target_hidden states",
        },
    }


def _hardware_model() -> str:
    return subprocess.run(
        ["sysctl", "-n", "hw.model"], check=True, capture_output=True, text=True
    ).stdout.strip()


def require_native_authorization(
    *,
    execute_native: bool,
    owns_gpu: bool,
    lease_id: str | None,
    hardware_model: Callable[[], str] = _hardware_model,
    owner_paths: Sequence[Path] = LOCK_OWNER_PATHS,
) -> dict[str, Any]:
    """Fail closed unless this process is inside one matching paired lease."""
    if not execute_native:
        raise RuntimeError("native execution requires --execute-native")
    if not owns_gpu:
        raise RuntimeError("native execution requires --i-own-the-gpu")
    if platform.machine() != "arm64" or hardware_model() != EXPECTED_HOST_MODEL:
        raise RuntimeError("native execution is restricted to the M3 Mac15,7 host")
    receipts = []
    for path in owner_paths:
        if not path.is_file():
            raise RuntimeError(f"missing GPU lock owner receipt: {path}")
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise TypeError(f"invalid GPU lock owner receipt: {path}")
        receipts.append(value)
    if receipts[0] != receipts[1]:
        raise RuntimeError("GPU lock owner receipts do not match")
    observed_lease = receipts[0].get("lease_id")
    if not isinstance(observed_lease, str) or not observed_lease:
        raise RuntimeError("GPU lock owner receipt has no lease_id")
    owner_pid = receipts[0].get("pid")
    inherited_lease = lease_id is not None and observed_lease == lease_id
    if owner_pid != os.getppid() and not inherited_lease:
        raise RuntimeError("GPU lock owner is not the parent wrapper or inherited lease")
    if lease_id is not None and observed_lease != lease_id:
        raise RuntimeError("GPU lock lease does not match --lease-id")
    return {
        "lease_id": observed_lease,
        "owner": receipts[0],
        "owner_paths": [str(path) for path in owner_paths],
        "hw_model": EXPECTED_HOST_MODEL,
        "ownership_proof": (
            "parent-wrapper-pid" if owner_pid == os.getppid() else "inherited-lease"
        ),
        "cpg_used": receipts[0].get("cpg_used"),
    }


def _semantic_graph(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    concepts: dict[str, dict[str, str]] = {}
    edges: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        subject = f"row-{index}-subject"
        object_ = f"row-{index}-object"
        concepts[subject] = {"label": str(row["subject"])}
        concepts[object_] = {"label": str(row["object"])}
        edges.append(
            {
                "subject": subject,
                "relation": str(row["relation"]),
                "object": object_,
                "authority": "committed",
            }
        )
    return {"concepts": concepts, "edges": edges}


def _prefixes(artifact: Any, encoder_type: Any) -> tuple[np.ndarray, np.ndarray]:
    world = json.loads((artifact.root / "micro_world.json").read_text())
    test_rows = world["splits"]["test"]
    if len(test_rows) < 4:
        raise ValueError("neural concept artifact needs four held-out rows")
    encoder = encoder_type(artifact)

    def project(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        encoded = encoder.encode_graph(_semantic_graph(rows))
        values = np.asarray([item.value_state for item in encoded], dtype=np.float32)
        hidden = values @ artifact.arrays["value_projection"]
        norms = np.maximum(np.linalg.norm(hidden, axis=-1, keepdims=True), 1e-6)
        return np.asarray(hidden / norms, dtype=np.float32)

    return project(test_rows[:2]), project(test_rows[2:4])


def publish_arm_payloads(
    artifact: Any,
    encoder_type: Any,
    store_root: Path,
    *,
    gate: float,
    producer_revision: str,
) -> dict[str, Any]:
    """Publish/load every arm through the canonical typed core envelope."""
    from mlx2.runtime.activation_capsules import (
        ActivationCapsuleBus,
        ActivationCapsuleSpec,
        ActivationExpectation,
        Authority,
        AuthorityScope,
        CapsuleProvenance,
        Geometry,
        ModelBindings,
        NormGateBounds,
        PayloadKind,
        PositionConvention,
        TapBinding,
        TensorPayloadStore,
    )
    from mlx2.runtime.semantic_capsules import CapsuleStore

    ordered, wrong = _prefixes(artifact, encoder_type)
    variants = {
        "zero_gate": (ordered, 0.0),
        "ordered": (ordered, gate),
        "reversed": (ordered[::-1].copy(), gate),
        "sign_negated": (-ordered, gate),
        "wrong_capsule": (wrong, gate),
    }
    result: dict[str, Any] = {"ordinary": None}
    capsule_store = CapsuleStore(store_root / "capsules")
    activation_store = ActivationCapsuleBus(
        capsule_store,
        TensorPayloadStore(store_root / "tensors", maximum_payload_bytes=1 << 20),
        maximum_total_bytes=4 << 20,
    )
    bindings = artifact.manifest["bindings"]
    layer = int(artifact.manifest["deep_injection_layer"])
    scope_id = "semantic-activation-mechanism-gate-20261006"
    now = time.time_ns()
    for name, (states, selected_gate) in variants.items():
        source_uri = f"artifact://{artifact.fingerprint}/test/{name}"
        source_digest = hashlib.sha256(
            artifact.fingerprint.encode() + name.encode() + states.tobytes()
        ).hexdigest()
        spec = ActivationCapsuleSpec(
            payload_kind=PayloadKind.CONTINUOUS_PREFIX,
            bindings=ModelBindings(
                source_model=bindings["model"],
                target_model=bindings["model"],
                tokenizer=bindings["tokenizer"],
                runtime=bindings["runtime"],
                projector_revision=artifact.fingerprint,
            ),
            tap=TapBinding(
                layer,
                "post_block_residual",
                layer,
                "qwen35-post-block-residual-v1",
            ),
            geometry=Geometry(
                hidden_width=int(states.shape[1]),
                sequence_length=int(states.shape[0]),
            ),
            dtype="<f4",
            normalization="target-unit-l2-v1",
            position=PositionConvention("none", "external-memory-order-v1"),
            provenance=CapsuleProvenance(
                "semantic-activation-mechanism-gate",
                producer_revision,
                (source_digest,),
                (source_uri,),
            ),
            authority=Authority(
                "mlx2-research",
                AuthorityScope.SESSION,
                scope_id,
                expires_at_unix_ns=now + 30 * 60 * 1_000_000_000,
            ),
            bounds=NormGateBounds(0.5, 0.05, 0.5),
            metadata={
                "representation_space": "target_hidden",
                "construction": "neural-concept-artifact-value-projection",
                "source_artifact_fingerprint": artifact.fingerprint,
                "control_label": name,
            },
        )
        # Explicitly exercise strict untrusted-manifest parsing before publish.
        parsed = ActivationCapsuleSpec.from_dict(spec.to_dict())
        identity = activation_store.publish(parsed, {"prefix": states})
        manifest, tensors = activation_store.load(
            identity.digest,
            ActivationExpectation(
                target_model=bindings["model"],
                tokenizer=bindings["tokenizer"],
                runtime=bindings["runtime"],
                projector_revision=artifact.fingerprint,
                target_layer=layer,
                target_injection="qwen35-post-block-residual-v1",
                tenant="mlx2-research",
                scope=AuthorityScope.SESSION,
                scope_id=scope_id,
                payload_kind=PayloadKind.CONTINUOUS_PREFIX,
            ),
            now_unix_ns=now,
        )
        result[name] = {
            "manifest": manifest,
            "tensors": tensors,
            "capsule_digest": identity.digest,
            "gate": selected_gate,
        }
    return result


def _forward_receipt(prepared: Mapping[str, Any] | None) -> dict[str, Any]:
    if prepared is None:
        return {
            "schema": "mlx2-activation-capsule-injection-receipt-v1",
            "status": "ordinary-observed",
            "route": "ordinary",
            "selected": False,
            "engaged": False,
            "observed_used": False,
            "forward_completed": True,
            "evaluated_logits": True,
        }
    receipt = dict(prepared["receipt"])
    receipt.update(
        status="forward-completed",
        selected=True,
        # Direct model calls do not traverse serving's request lifecycle, so
        # they cannot mint serving engagement or observed-used receipts.
        engaged=False,
        observed_used=False,
        forward_completed=True,
        evaluated_logits=True,
    )
    return receipt


def execute_native(
    model: Path,
    artifact_path: Path,
    *,
    gate: float,
    producer_revision: str,
) -> dict[str, Any]:
    """Run all arms through the real adapter/model seam. Imports MLX lazily."""
    import mlx.core as mx

    from mlx2.adapters.qwen35_9b import Qwen359BAdapter
    from mlx2.runtime.neural_concepts import (
        NeuralConceptArtifact,
        RecurrentConceptEncoder,
    )

    mx.random.seed(20261006)
    manifest = json.loads((artifact_path / "manifest.json").read_text())
    artifact = NeuralConceptArtifact.load(
        artifact_path,
        model_binding=manifest["bindings"]["model"],
        tokenizer_binding=manifest["bindings"]["tokenizer"],
        runtime_binding=manifest["bindings"]["runtime"],
    )
    adapter = Qwen359BAdapter(str(model))
    adapter.configure_activation_capsule_bridge(artifact)
    # Tiny ordinary preflight validates the exact call/shape/evaluation
    # contract before constructing or publishing any capsule arm.
    preflight_tokens = list(
        adapter.tokenizer.encode(
            "This is a deterministic cache preflight.", add_special_tokens=False
        )
    )
    if len(preflight_tokens) < 2:
        raise RuntimeError("adapter tokenizer preflight needs at least two tokens")
    preflight_cache = adapter.model.make_cache()
    prefill_logits = adapter.model(
        mx.array([preflight_tokens[:-1]]), cache=preflight_cache
    )[:, -1, :]
    decode_logits = adapter.model(
        mx.array([[preflight_tokens[-1]]]), cache=preflight_cache
    )[:, -1, :]
    mx.eval(prefill_logits, decode_logits)
    prefill_values = np.asarray(prefill_logits.astype(mx.float32))
    decode_values = np.asarray(decode_logits.astype(mx.float32))
    if (
        prefill_values.ndim != 2
        or prefill_values.shape[0] != 1
        or decode_values.shape != prefill_values.shape
    ):
        raise RuntimeError("adapter.model preflight returned invalid logits geometry")
    if not np.isfinite(prefill_values).all() or not np.isfinite(decode_values).all():
        raise RuntimeError("adapter.model preflight returned nonfinite logits")
    preflight = {
        "forward_completed": True,
        "evaluated_logits": True,
        "finite": True,
        "cache_contract": "prefill-then-single-token-decode",
        "prefill_logits_shape": list(prefill_values.shape),
        "decode_logits_shape": list(decode_values.shape),
        "next_token": int(mx.argmax(decode_logits, axis=-1).item()),
    }
    del preflight_cache
    mx.clear_cache()
    rows = []
    with tempfile.TemporaryDirectory(prefix="mlx2-semantic-activation-") as temporary:
        payloads = publish_arm_payloads(
            artifact,
            RecurrentConceptEncoder,
            Path(temporary),
            gate=gate,
            producer_revision=producer_revision,
        )
        for prompt_index, prompt in enumerate(PROMPTS):
            request = {
                "messages": [{"role": "user", "content": prompt}],
                "enable_thinking": False,
                "reasoning_effort": "none",
            }
            tokens = list(adapter.prompt_tokens(request))
            for arm in ARMS:
                payload = payloads[arm]
                prepared = None
                kwargs: dict[str, Any] = {}
                if payload is not None:
                    prepared = adapter.activation_capsule_prefill(
                        tokens,
                        payload,
                        prefill_step=max(2048, len(tokens)),
                        route="ordinary",
                        batch_size=1,
                    )
                    kwargs["deep_concept_memory"] = prepared["deep_concept_memory"]
                logits = adapter.model(mx.array([tokens]), **kwargs)[:, -1, :]
                mx.eval(logits)
                materialized = np.asarray(logits.astype(mx.float32))
                rows.append(
                    {
                        "prompt_index": prompt_index,
                        "arm": arm,
                        "next_token": int(mx.argmax(logits, axis=-1).item()),
                        "logits": materialized,
                        "receipt": _forward_receipt(prepared),
                    }
                )
                mx.clear_cache()
    return {"preflight": preflight, "rows": rows}


def evaluate_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate exact identity and sensitivity without interpreting quality."""
    grouped: dict[int, dict[str, Mapping[str, Any]]] = {}
    public_rows = []
    for row in rows:
        prompt_index = int(row["prompt_index"])
        arm = str(row["arm"])
        logits = np.asarray(row["logits"])
        if arm not in ARMS or arm in grouped.setdefault(prompt_index, {}):
            raise ValueError("duplicate or unknown mechanism arm")
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("native executor returned invalid logits geometry")
        grouped[prompt_index][arm] = row
        public_rows.append(
            {
                "prompt_index": prompt_index,
                "arm": arm,
                "next_token": int(row["next_token"]),
                "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
                "finite": bool(np.isfinite(logits).all()),
                "receipt": dict(row["receipt"]),
            }
        )
    if set(grouped) != set(range(len(PROMPTS))):
        raise ValueError("native executor did not return every deterministic prompt")
    per_prompt = []
    for prompt_index in range(len(PROMPTS)):
        arms = grouped[prompt_index]
        if set(arms) != set(ARMS):
            raise ValueError("native executor did not return every mechanism arm")
        ordinary = np.asarray(arms["ordinary"]["logits"])
        zero = np.asarray(arms["zero_gate"]["logits"])
        ordered = np.asarray(arms["ordered"]["logits"])
        reversed_ = np.asarray(arms["reversed"]["logits"])
        sign = np.asarray(arms["sign_negated"]["logits"])
        per_prompt.append(
            {
                "prompt_index": prompt_index,
                "zero_gate_exact_logits_identity": bool(np.array_equal(ordinary, zero)),
                "zero_gate_exact_next_token_identity": (
                    int(arms["ordinary"]["next_token"])
                    == int(arms["zero_gate"]["next_token"])
                ),
                "ordered_vs_reversed_logits_sensitive": bool(
                    not np.array_equal(ordered, reversed_)
                ),
                "ordered_vs_sign_negated_logits_sensitive": bool(
                    not np.array_equal(ordered, sign)
                ),
                "ordered_vs_reversed_max_abs_delta": float(
                    np.max(np.abs(ordered - reversed_))
                ),
                "ordered_vs_sign_negated_max_abs_delta": float(
                    np.max(np.abs(ordered - sign))
                ),
            }
        )
    checks = {
        "zero_gate_exact_logits_identity": all(
            item["zero_gate_exact_logits_identity"] for item in per_prompt
        ),
        "zero_gate_exact_next_token_identity": all(
            item["zero_gate_exact_next_token_identity"] for item in per_prompt
        ),
        "all_logits_finite": all(row["finite"] for row in public_rows),
        "all_arms_have_route_receipts": all(
            row["receipt"].get("route") == "ordinary"
            and row["receipt"].get("forward_completed") is True
            and row["receipt"].get("evaluated_logits") is True
            and row["receipt"].get("observed_used") is False
            for row in public_rows
        ),
        "ordered_vs_reversed_logits_sensitive": all(
            item["ordered_vs_reversed_logits_sensitive"] for item in per_prompt
        ),
        "ordered_vs_sign_negated_logits_sensitive": all(
            item["ordered_vs_sign_negated_logits_sensitive"] for item in per_prompt
        ),
        "per_prompt": per_prompt,
    }
    return checks, public_rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--gate", type=float, default=0.2)
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-native", action="store_true")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--lease-id")
    parser.add_argument(
        "--source-revision",
        help="full source Git SHA; required only when the execution mirror has no .git",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.gate) or not 0.05 <= args.gate <= 0.5:
        parser.error("--gate must be finite and in 0.05..0.5")
    if args.execute_native and (args.manifest or args.dry_run):
        parser.error("native execution cannot be combined with manifest/dry-run")
    plan = build_manifest(
        args.model,
        args.artifact,
        args.output,
        source_revision=args.source_revision,
    )
    if not args.execute_native:
        # Defaulting to a no-device plan makes accidental model execution
        # impossible even when neither spelling is supplied.
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2, sort_keys=True))
        return 0
    ownership = require_native_authorization(
        execute_native=args.execute_native,
        owns_gpu=args.i_own_the_gpu,
        lease_id=args.lease_id,
    )
    started = time.time()
    execution = execute_native(
        args.model.resolve(),
        args.artifact.resolve(),
        gate=args.gate,
        producer_revision=plan["provenance"]["git_revision"],
    )
    checks, public_rows = evaluate_rows(execution["rows"])
    passed = all(value for key, value in checks.items() if key != "per_prompt")
    receipt = {
        **plan,
        "mode": "native-mechanism-gate",
        "status": "passed-mechanism-gate" if passed else "failed-mechanism-gate",
        "qualification": "implemented-unqualified-default-off",
        "ownership": ownership,
        "preflight": execution["preflight"],
        "checks": checks,
        "rows": public_rows,
        "elapsed_seconds": time.time() - started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
