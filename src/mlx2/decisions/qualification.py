"""Revision-bound production qualification receipts for decision routes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

SCHEMA = "mlx2.decision-serving-qualification.v1"
PRODUCER = "mlx2-decision-qualification"
# Frozen after the qualification producer is finalized. A receipt from any
# other producer revision is data, not authority to advertise a qualified route.
APPROVED_PRODUCER_SHA256 = (
    "a6addcf209f28f35f31934df9bf8787a17d916f173dfd62f786945f2cd9513bb"
)
REQUIRED_CHECKS = frozenset(
    {
        "artifact_identity",
        "tokenizer_integrity",
        "http_contract",
        "family_semantic_oracles",
        "repeat_determinism",
        "restart_determinism",
        "context_ladder",
        "context_boundary",
        "fail_closed",
        "route_receipts",
        "prometheus_metrics",
        "stability",
        "source_derived_reference",
    }
)


def _read_object(payload: bytes) -> dict[str, Any]:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key {key!r} in qualification receipt")
            value[key] = item
        return value

    value = json.loads(payload, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise TypeError("qualification receipt must contain an object")
    return value


def source_identity() -> dict[str, Any]:
    package = Path(__file__).resolve().parents[1]
    paths = sorted(package.rglob("*.py"))
    digest = hashlib.sha256()
    files = []
    for path in paths:
        relative = str(path.relative_to(package))
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative.encode())
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        files.append(relative)
    return {"sha256": digest.hexdigest(), "files": files}


def runtime_identity() -> dict[str, Any]:
    import mlx.core as mx

    def version(name: str) -> str:
        return importlib.metadata.version(name)

    core_path = Path(mx.__file__).resolve()
    native_paths = [core_path]
    native_dir = core_path.parent / "lib"
    if native_dir.is_dir():
        native_paths.extend(
            path
            for path in sorted(native_dir.iterdir())
            if path.is_file() and path.suffix in {".dylib", ".metallib", ".so"}
        )
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
        "mlx": version("mlx"),
        "mlx_native_sha256": {
            str(path.relative_to(core_path.parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in native_paths
        },
        "transformers": version("transformers"),
        "tokenizers": version("tokenizers"),
    }


def artifact_identity(engine) -> dict[str, Any]:
    identity = engine.artifact["identity"]
    return {
        "family": engine.family,
        "variant": engine.variant,
        "fingerprint": identity["fingerprint"],
        "fingerprint_kind": identity["fingerprint_kind"],
        "revision": identity.get("revision"),
    }


def serving_settings(engine, *, max_connections: int, max_request_bytes: int) -> dict:
    from ..runtime.env_switches import serving_env_switches

    settings = {
        "served_model_name": engine.model_name,
        "capabilities": list(engine.capabilities),
        "max_connections": int(max_connections),
        "max_request_bytes": int(max_request_bytes),
        "route": "decision",
    }
    # Behaviour-changing MLX2_* switches the Qwen profile does not pin, exactly
    # as main serving records them (serving.py): MLX2_FUSED_SDPA_MIN_L selects
    # the d256 prefill SDPA kernel in the decision-prompt length regime.  Absent
    # switches are not recorded, so receipts taken without any stay valid.
    process_env = serving_env_switches()
    if process_env:
        settings["process_env"] = process_env
    return settings


def qualification_basis(engine, settings: dict) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "producer": {
            "name": PRODUCER,
            "sha256": APPROVED_PRODUCER_SHA256,
        },
        "source": source_identity(),
        "runtime": runtime_identity(),
        "artifact": artifact_identity(engine),
        "settings": settings,
    }


def load_qualification(path: str | Path, *, engine, settings: dict) -> dict[str, Any]:
    receipt_path = Path(path)
    payload = receipt_path.read_bytes()
    record = _read_object(payload)
    expected = qualification_basis(engine, settings)
    for key in ("schema", "producer", "source", "runtime", "artifact", "settings"):
        if record.get(key) != expected[key]:
            raise ValueError(f"decision qualification does not match {key}")
    checks = record.get("checks")
    if not isinstance(checks, dict):
        raise TypeError("decision qualification checks are missing")
    missing = sorted(REQUIRED_CHECKS - set(checks))
    if missing:
        raise ValueError("decision qualification checks are missing: " + ", ".join(missing))
    failed = sorted(
        name
        for name, value in checks.items()
        if not isinstance(value, dict) or value.get("passed") is not True
    )
    if record.get("passed") is not True or failed:
        raise ValueError(
            "decision qualification checks failed"
            + (": " + ", ".join(failed) if failed else "")
        )
    evidence = record.get("evidence")
    if not isinstance(evidence, dict):
        raise TypeError("decision qualification evidence identity is missing")
    evidence_name = evidence.get("path")
    if (
        not isinstance(evidence_name, str)
        or Path(evidence_name).is_absolute()
        or ".." in Path(evidence_name).parts
        or Path(evidence_name).name != evidence_name
    ):
        raise ValueError("decision qualification evidence path is invalid")
    evidence_payload = receipt_path.with_name(evidence_name).read_bytes()
    if hashlib.sha256(evidence_payload).hexdigest() != evidence.get("sha256"):
        raise ValueError("decision qualification evidence hash does not match")
    return {
        "qualification": "qualified",
        "qualified": True,
        "receipt_sha256": hashlib.sha256(payload).hexdigest(),
        "producer_sha256": APPROVED_PRODUCER_SHA256,
    }


def install_qualification(path: str | Path, *, engine, settings: dict) -> dict[str, Any]:
    state = load_qualification(path, engine=engine, settings=settings)
    engine._qualification = state
    return state
