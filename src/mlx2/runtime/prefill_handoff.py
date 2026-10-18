"""Revision-bound wire format for approximate cross-host prompt state."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

OPERATION = "prefill-handoff-approx-v1"
MANIFEST_SCHEMA = "mlx2.prefill-handoff.v1"
PLAN_SCHEMA = "mlx2.prefill-handoff-plan.v1"
RECORD_SCHEMA = "mlx2.approximate-qualification.v1"
_MAGIC = b"MLX2HO1\n"
_DESCRIPTOR = {
    "operation": OPERATION,
    "manifest_schema": MANIFEST_SCHEMA,
    "plan_schema": PLAN_SCHEMA,
    "record_schema": RECORD_SCHEMA,
    "wire": "magic-u64be-canonical-json-safetensors-v1",
    "payload_integrity": "sha256-entire-safetensors-payload-v1",
    "cache_metadata": "mlx2-runtime-models-cache-classes-only-v1",
    "plan_law": "plain-chunks-prefix-n-minus-one-v1",
}


def _canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


OPERATION_REVISION = _sha256(_canonical_json(_DESCRIPTOR))[:16]


class PrefillHandoffError(ValueError):
    """A fail-closed handoff refusal with a stable machine reason."""

    def __init__(self, reason: str, message: str | None = None):
        self.reason = str(reason)
        super().__init__(message or self.reason)


def runtime_digest(identity: dict) -> str:
    if not isinstance(identity, dict):
        raise PrefillHandoffError("malformed", "runtime identity must be an object")
    return _sha256(_canonical_json(identity))


def token_digest(token_ids: Sequence[int]) -> str:
    try:
        values = list(token_ids)
    except TypeError as error:
        raise PrefillHandoffError("malformed", "token ids must be a sequence") from error
    if any(type(token) is not int or token < 0 for token in values):
        raise PrefillHandoffError("malformed", "token ids must be nonnegative integers")
    return _sha256(_canonical_json(values))


def make_plan(prompt_tokens: int, chunk: int) -> dict:
    if type(prompt_tokens) is not int or prompt_tokens < 1:
        raise PrefillHandoffError("malformed", "prompt_tokens must be positive")
    if type(chunk) is not int or chunk < 1:
        raise PrefillHandoffError("malformed", "chunk must be positive")
    return {
        "schema": PLAN_SCHEMA,
        "prompt_tokens": prompt_tokens,
        "prefix_tokens": prompt_tokens - 1,
        "chunk": chunk,
    }


def _validate_plan(plan: dict) -> dict:
    if not isinstance(plan, dict) or set(plan) != {
        "schema", "prompt_tokens", "prefix_tokens", "chunk"
    }:
        raise PrefillHandoffError("malformed", "invalid handoff plan")
    if plan.get("schema") != PLAN_SCHEMA:
        raise PrefillHandoffError("plan_mismatch", "unknown handoff plan schema")
    expected = make_plan(plan.get("prompt_tokens"), plan.get("chunk"))
    if plan != expected:
        raise PrefillHandoffError("plan_mismatch", "inconsistent handoff plan")
    return expected


def plan_digest(plan: dict) -> str:
    return _sha256(_canonical_json(_validate_plan(plan)))


def _cache_arrays(cache):
    import mlx.core as mx

    from .models.cache import prompt_cache_payload

    arrays, metadata = prompt_cache_payload(cache)
    try:
        mx.eval(*arrays.values())
    except Exception as error:
        raise PrefillHandoffError("malformed", "cache arrays cannot be evaluated") from error
    return arrays, metadata


def _payload_bytes(arrays, metadata) -> bytes:
    import mlx.core as mx

    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as stream:
            path = stream.name
        mx.save_safetensors(path, arrays, metadata)
        return Path(path).read_bytes()
    finally:
        if path is not None:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def _array_bytes(value) -> bytes:
    import mlx.core as mx
    import numpy as np

    unsigned = getattr(mx, f"uint{value.itemsize * 8}")
    return np.asarray(value.reshape(-1).view(unsigned)).tobytes(order="C")


def prefill_prefix(model, cache, token_ids: Sequence[int], plan: dict) -> None:
    import mlx.core as mx

    from .models.cache import prompt_cache_payload

    plan = _validate_plan(plan)
    tokens = list(token_ids)
    if len(tokens) != plan["prompt_tokens"]:
        raise PrefillHandoffError("plan_mismatch", "token count differs from plan")
    for start in range(0, plan["prefix_tokens"], plan["chunk"]):
        stop = min(start + plan["chunk"], plan["prefix_tokens"])
        model(mx.array(tokens[start:stop])[None], cache=cache)
    arrays, _metadata = prompt_cache_payload(cache)
    mx.eval(*arrays.values())


def build_manifest(
    *,
    cache,
    token_ids,
    plan,
    artifact_fingerprint,
    tokenizer_digest,
    layout_fingerprint,
    producer: dict,
) -> dict:
    plan = _validate_plan(plan)
    tokens = list(token_ids)
    if len(tokens) != plan["prompt_tokens"]:
        raise PrefillHandoffError("plan_mismatch", "token count differs from plan")
    if not isinstance(producer, dict) or not isinstance(
        producer.get("runtime_digest"), str
    ):
        raise PrefillHandoffError("malformed", "producer runtime digest is required")
    arrays, metadata = _cache_arrays(cache)
    array_manifest = []
    for name, value in sorted(arrays.items()):
        array_manifest.append(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": _sha256(_array_bytes(value)),
            }
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "operation": OPERATION,
        "operation_revision": OPERATION_REVISION,
        "artifact_fingerprint": artifact_fingerprint,
        "tokenizer_digest": tokenizer_digest,
        "layout_fingerprint": layout_fingerprint,
        "layout": layout_fingerprint,
        "token_digest": token_digest(tokens[: plan["prefix_tokens"]]),
        "token_count": plan["prefix_tokens"],
        "plan": plan,
        "plan_digest": plan_digest(plan),
        "producer": dict(producer),
        "arrays": array_manifest,
    }
    manifest["payload_sha256"] = _sha256(_payload_bytes(arrays, metadata))
    return manifest


def serialize_handoff(cache, manifest: dict) -> bytes:
    if not isinstance(manifest, dict):
        raise PrefillHandoffError("malformed", "manifest must be an object")
    arrays, metadata = _cache_arrays(cache)
    declared = {entry.get("name"): entry for entry in manifest.get("arrays", ())}
    if set(declared) != set(arrays):
        raise PrefillHandoffError("array_digest_mismatch", "manifest arrays differ")
    for name, value in arrays.items():
        entry = declared[name]
        if (
            entry.get("dtype") != str(value.dtype)
            or entry.get("shape") != list(value.shape)
            or entry.get("sha256") != _sha256(_array_bytes(value))
        ):
            raise PrefillHandoffError("array_digest_mismatch", f"array {name} differs")
    payload = _payload_bytes(arrays, metadata)
    if manifest.get("payload_sha256") != _sha256(payload):
        raise PrefillHandoffError("payload_digest_mismatch")
    manifest_bytes = _canonical_json(manifest)
    return _MAGIC + struct.pack(">Q", len(manifest_bytes)) + manifest_bytes + payload


def _split_blob(blob: bytes) -> tuple[dict, bytes]:
    if not isinstance(blob, bytes) or len(blob) < len(_MAGIC) + 8:
        raise PrefillHandoffError("malformed", "handoff is truncated")
    if not blob.startswith(_MAGIC):
        raise PrefillHandoffError("malformed", "invalid handoff magic")
    length = struct.unpack(">Q", blob[len(_MAGIC) : len(_MAGIC) + 8])[0]
    start = len(_MAGIC) + 8
    end = start + length
    if length < 2 or end >= len(blob):
        raise PrefillHandoffError("malformed", "invalid manifest length")
    try:
        manifest = json.loads(blob[start:end])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrefillHandoffError("malformed", "invalid manifest JSON") from error
    if not isinstance(manifest, dict):
        raise PrefillHandoffError("malformed", "manifest must be an object")
    return manifest, blob[end:]


def _check_manifest(manifest: dict, expected: dict) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("operation") != OPERATION:
        raise PrefillHandoffError("malformed", "unknown manifest schema or operation")
    if manifest.get("operation_revision") != OPERATION_REVISION:
        raise PrefillHandoffError("operation_revision_mismatch")
    plan = _validate_plan(manifest.get("plan"))
    if manifest.get("plan_digest") != plan_digest(plan):
        raise PrefillHandoffError("plan_mismatch")
    if manifest.get("token_count") != plan["prefix_tokens"]:
        raise PrefillHandoffError("plan_mismatch")
    reasons = {
        "artifact_fingerprint": "artifact_mismatch",
        "tokenizer_digest": "tokenizer_mismatch",
        "layout_fingerprint": "layout_mismatch",
        "token_digest": "token_digest_mismatch",
        "plan_digest": "plan_mismatch",
    }
    if not isinstance(expected, dict):
        raise PrefillHandoffError("malformed", "expected identity must be an object")
    for field, reason in reasons.items():
        if field not in expected or manifest.get(field) != expected[field]:
            raise PrefillHandoffError(reason)


_SAFETENSORS_MLX_DTYPES = {
    "BF16": "mlx.core.bfloat16",
    "F16": "mlx.core.float16",
    "F32": "mlx.core.float32",
    "F64": "mlx.core.float64",
    "I8": "mlx.core.int8",
    "I16": "mlx.core.int16",
    "I32": "mlx.core.int32",
    "I64": "mlx.core.int64",
    "U8": "mlx.core.uint8",
    "U16": "mlx.core.uint16",
    "U32": "mlx.core.uint32",
    "U64": "mlx.core.uint64",
    "BOOL": "mlx.core.bool",
}
_SAFETENSORS_DTYPE_BYTES = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "I8": 1, "I16": 2, "I32": 4, "I64": 8,
    "U8": 1, "U16": 2, "U32": 4, "U64": 8, "BOOL": 1,
}


def _unflatten_metadata(metadata: dict[str, str]):
    """A small, dependency-free equivalent of mlx.utils.tree_unflatten."""
    if not metadata:
        raise PrefillHandoffError("malformed", "missing cache metadata")
    root = {}
    for flat_key, value in metadata.items():
        if (
            not isinstance(flat_key, str)
            or not flat_key
            or len(flat_key) > 512
            or not isinstance(value, str)
        ):
            raise PrefillHandoffError("malformed", "invalid cache metadata leaf")
        parts = flat_key.split(".")
        if len(parts) > 32 or any(not part or len(part) > 128 for part in parts):
            raise PrefillHandoffError("malformed", "invalid cache metadata path")
        node = root
        for part in parts[:-1]:
            existing = node.setdefault(part, {})
            if not isinstance(existing, dict):
                raise PrefillHandoffError("malformed", "conflicting cache metadata path")
            node = existing
        if parts[-1] in node:
            raise PrefillHandoffError("malformed", "duplicate cache metadata path")
        node[parts[-1]] = value

    def convert(node):
        if not isinstance(node, dict):
            return node
        if node and all(key.isdigit() and str(int(key)) == key for key in node):
            indices = sorted(int(key) for key in node)
            return [convert(node.get(str(index), {})) for index in range(indices[-1] + 1)]
        return {key: convert(value) for key, value in node.items()}

    return convert(root)


def _validate_cache_metadata(header_metadata, tensor_names: set[str]) -> None:
    if not isinstance(header_metadata, dict) or len(header_metadata) > 100_000:
        raise PrefillHandoffError("malformed", "invalid safetensors cache metadata")
    tree = _unflatten_metadata(header_metadata)
    if not isinstance(tree, list) or not 3 <= len(tree) <= 5:
        raise PrefillHandoffError("malformed", "invalid prompt-cache metadata tree")
    info, user_metadata, classes = tree[:3]
    if (
        not isinstance(info, list)
        or not isinstance(user_metadata, dict)
        or not isinstance(classes, list)
        or not classes
        or len(info) != len(classes)
        or len(classes) > 4096
    ):
        raise PrefillHandoffError("malformed", "invalid prompt-cache metadata shape")

    # Handoff v1 intentionally accepts only cache implementations defined by
    # this already-loaded module.  Do not call cache._resolve_cache_class here:
    # it may import a module named by an untrusted token.
    from .models import cache as cache_module

    for token in classes:
        if not isinstance(token, str):
            raise PrefillHandoffError("cache_class_disallowed")
        module_name, separator, class_name = token.rpartition(":")
        if separator and module_name != cache_module.__name__:
            raise PrefillHandoffError("cache_class_disallowed")
        name = class_name if separator else token
        cls = getattr(cache_module, name, None)
        if (
            not isinstance(cls, type)
            or cls.__module__ != cache_module.__name__
            or not issubclass(cls, cache_module._BaseCache)
        ):
            raise PrefillHandoffError("cache_class_disallowed")

    top_level = set()
    for name in tensor_names:
        first = name.split(".", 1)[0]
        if not first.isdigit():
            raise PrefillHandoffError("malformed", "invalid prompt-cache tensor tree")
        top_level.add(int(first))
    if top_level and max(top_level) >= len(classes):
        raise PrefillHandoffError("malformed", "prompt-cache tensor tree exceeds classes")
    if len(tree) > 3:
        try:
            empty = json.loads(tree[3])
        except (TypeError, json.JSONDecodeError) as error:
            raise PrefillHandoffError("malformed", "invalid empty-cache metadata") from error
        if not isinstance(empty, dict) or len(empty) > 100_000:
            raise PrefillHandoffError("malformed", "invalid empty-cache metadata")
        for key, placeholder in empty.items():
            if not isinstance(key, str) or key in tensor_names:
                raise PrefillHandoffError("malformed", "invalid empty-cache entry")
            if placeholder is None:
                continue
            if (
                not isinstance(placeholder, list)
                or len(placeholder) != 2
                or placeholder[0] not in _SAFETENSORS_MLX_DTYPES.values()
                or not isinstance(placeholder[1], list)
                or not placeholder[1]
                or 0 not in placeholder[1]
                or any(type(dim) is not int or dim < 0 for dim in placeholder[1])
            ):
                raise PrefillHandoffError("malformed", "invalid empty-cache placeholder")
    if len(tree) > 4:
        try:
            summary = json.loads(tree[4])
        except (TypeError, json.JSONDecodeError) as error:
            raise PrefillHandoffError("malformed", "invalid cache summary metadata") from error
        if not isinstance(summary, dict):
            raise PrefillHandoffError("malformed", "invalid cache summary metadata")


def _validate_payload(payload: bytes, entries: list[dict]) -> None:
    """Validate safetensors metadata and raw tensor bytes without MLX/NumPy."""
    if len(payload) < 8:
        raise PrefillHandoffError("malformed", "safetensors payload is truncated")
    header_length = struct.unpack("<Q", payload[:8])[0]
    header_end = 8 + header_length
    if header_length < 2 or header_end > len(payload):
        raise PrefillHandoffError("malformed", "invalid safetensors header length")
    try:
        header = json.loads(payload[8:header_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrefillHandoffError("malformed", "invalid safetensors header") from error
    if not isinstance(header, dict):
        raise PrefillHandoffError("malformed", "invalid safetensors header")
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    _validate_cache_metadata(header.get("__metadata__"), set(tensors))
    declared = {item.get("name"): item for item in entries}
    if set(tensors) != set(declared):
        raise PrefillHandoffError("array_digest_mismatch")
    data = memoryview(payload)[header_end:]
    spans = []
    for name, tensor in tensors.items():
        if not isinstance(tensor, dict):
            raise PrefillHandoffError("malformed", "invalid tensor metadata")
        dtype = tensor.get("dtype")
        shape = tensor.get("shape")
        offsets = tensor.get("data_offsets")
        if (
            dtype not in _SAFETENSORS_MLX_DTYPES
            or not isinstance(shape, list)
            or any(type(dim) is not int or dim < 0 for dim in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
        ):
            raise PrefillHandoffError("malformed", "invalid tensor metadata")
        start, end = offsets
        expected_bytes = math.prod(shape) * _SAFETENSORS_DTYPE_BYTES[dtype]
        if start < 0 or end < start or end > len(data) or end - start != expected_bytes:
            raise PrefillHandoffError("malformed", "invalid tensor byte range")
        spans.append((start, end))
        entry = declared[name]
        if (
            entry.get("shape") != shape
            or entry.get("dtype") != _SAFETENSORS_MLX_DTYPES[dtype]
            or entry.get("sha256") != _sha256(data[start:end])
        ):
            raise PrefillHandoffError("array_digest_mismatch")
    ordered = sorted(spans)
    if ordered and (ordered[0][0] != 0 or ordered[-1][1] != len(data)):
        raise PrefillHandoffError("malformed", "safetensors data is not contiguous")
    if any(left[1] != right[0] for left, right in pairwise(ordered)):
        raise PrefillHandoffError("malformed", "safetensors tensor ranges overlap or gap")


def validate_handoff(blob: bytes, *, expected: dict) -> dict:
    """Validate a handoff entirely as bytes, without creating MLX arrays."""
    manifest, payload = _split_blob(blob)
    _check_manifest(manifest, expected)
    entries = manifest.get("arrays")
    if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
        raise PrefillHandoffError("malformed", "invalid array manifest")
    declared = {item.get("name"): item for item in entries}
    if len(declared) != len(entries) or None in declared:
        raise PrefillHandoffError("malformed", "invalid array names")
    _validate_payload(payload, entries)
    if manifest.get("payload_sha256") != _sha256(payload):
        raise PrefillHandoffError("payload_digest_mismatch")
    return manifest


def materialize_handoff(blob: bytes, manifest: dict) -> list:
    """Build and eagerly evaluate cache arrays on the calling MLX thread."""
    import mlx.core as mx

    from .models.cache import load_prompt_cache, prompt_cache_payload

    embedded, payload = _split_blob(blob)
    if embedded != manifest:
        raise PrefillHandoffError("malformed", "validated manifest differs from blob")
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as stream:
            path = stream.name
            stream.write(payload)
        try:
            cache = load_prompt_cache(path)
            arrays, _metadata = prompt_cache_payload(cache)
            mx.eval(*arrays.values())
        except Exception as error:
            raise PrefillHandoffError("malformed", "cache reconstruction failed") from error
    finally:
        if path is not None:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
    return cache


def deserialize_handoff(blob: bytes, *, expected: dict) -> tuple[list, dict]:
    manifest = validate_handoff(blob, expected=expected)
    return materialize_handoff(blob, manifest), manifest


def load_records(directory: Path, *, allow_provisional: bool = False) -> list[dict]:
    records = []
    try:
        paths = sorted(Path(directory).glob("*.json"))
    except OSError:
        return records
    for path in paths:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not (
            isinstance(value, dict)
            and value.get("schema") == RECORD_SCHEMA
            and value.get("operation") == OPERATION
            and value.get("verdict") == "qualified"
        ):
            continue
        provisional = value.get("provisional") is True
        required = ("G1", "G2", "G3", "G4") if provisional else (
            "G1", "G2", "G3", "G4", "G5"
        )
        gates = value.get("gates")
        if (
            (provisional and not allow_provisional)
            or not isinstance(gates, dict)
            or any(not isinstance(gates.get(gate), dict)
                   or gates[gate].get("pass") is not True for gate in required)
        ):
            continue
        record = dict(value)
        record["record_state"] = "provisional" if provisional else "final"
        records.append(record)
    return records


def match_record(
    records,
    *,
    artifact_fingerprint,
    producer_runtime_digest,
    receiver_runtime_digest,
    receiver_execution_digest,
    plan: dict,
) -> dict | None:
    try:
        plan = _validate_plan(plan)
    except PrefillHandoffError:
        return None
    for record in records:
        if not isinstance(record, dict):
            continue
        producer = record.get("producer") or {}
        receiver = record.get("receiver") or {}
        if (
            record.get("schema") == RECORD_SCHEMA
            and record.get("verdict") == "qualified"
            and record.get("operation") == OPERATION
            and record.get("operation_revision") == OPERATION_REVISION
            and record.get("artifact_fingerprint") == artifact_fingerprint
            and producer.get("runtime_digest") == producer_runtime_digest
            and receiver.get("runtime_digest") == receiver_runtime_digest
            and receiver.get("execution_digest") == receiver_execution_digest
            and type(record.get("max_prompt_tokens")) is int
            and plan["prompt_tokens"] <= record["max_prompt_tokens"]
            and plan["chunk"] in record.get("plan_chunks", ())
        ):
            return record
    return None


def default_records_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "qualification" / "approximate" / "prefill-handoff"


def receipt(*, manifest, record, observed_used: bool, bytes_moved: int | None) -> dict:
    record_state = record.get("record_state") if record is not None else None
    value = {
        "schema": "mlx2.prefill-handoff.receipt.v1",
        "operation": OPERATION,
        "operation_revision": OPERATION_REVISION,
        "fidelity": "approximate",
        "qualified": record_state == "final",
        "record_id": record.get("record_id") if record is not None else None,
        "record_state": record_state,
        "producer": manifest.get("producer"),
        "plan_digest": manifest.get("plan_digest"),
        "token_count": manifest.get("token_count"),
        "bytes": bytes_moved,
        "observed_used": bool(observed_used),
    }
    if record is None:
        value["reason"] = "candidate_validation"
    return value


__all__ = [
    "MANIFEST_SCHEMA", "OPERATION", "OPERATION_REVISION", "PLAN_SCHEMA",
    "RECORD_SCHEMA", "PrefillHandoffError", "build_manifest",
    "default_records_dir", "deserialize_handoff", "load_records", "make_plan",
    "match_record", "materialize_handoff", "plan_digest", "prefill_prefix", "receipt",
    "runtime_digest", "serialize_handoff", "token_digest",
    "validate_handoff",
]
