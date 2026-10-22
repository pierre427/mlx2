"""Attach a separately shipped Qwen3.8 27B MTP head directory to a target.

Original mlx2 code.  Some Qwen3.8 27B MTP heads ship as their own artifact
(``model_type: qwen3_5_mtp`` with ``model.safetensors``, or a bare
``mtp.safetensors``) instead of inside the target directory.  This module
binds such a head to an exact target before any tensor loads:

- the target must not already ship a head (an external head would replace
  it; that is a different checkpoint pairing and is refused);
- a declared head config must match the target's text geometry key for key
  (hidden size, vocabulary, layer count, attention/MLP geometry, norms, RoPE)
  and declare exactly one MTP layer;
- the tensor schema must be exactly the single-layer head, each projection
  either float with the target-derived shape or affine-quantized with a bit
  width and group size derived exactly from the packed shapes (and equal to
  any declared quantization);
- the payload is hashed in full; an all-zero payload is refused as a
  degenerate artifact.

The head's content identity is folded into the adapter identity, so the
route receipts, the native-MTP draft revision and the APCv2 identity all
name the target+head pair.  Implemented, unqualified.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

SCHEMA = "mlx2.qwen38-external-mtp-head.v1"
POLICY_KEY = "mtp_head"
HEAD_FILES = ("model.safetensors", "mtp.safetensors")
HEAD_MODEL_TYPES = frozenset({"qwen3_5_mtp"})
_HEADER_LIMIT = 64 << 20
_FLOAT = frozenset({"BF16", "F16", "F32"})
_BITS = frozenset({2, 3, 4, 5, 6, 8})
_GROUPS = frozenset({32, 64, 128})
# Text-config keys a declared head must share with its target.
GEOMETRY_KEYS = (
    "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "vocab_size", "attn_output_gate",
    "partial_rotary_factor", "rms_norm_eps", "full_attention_interval",
    "tie_word_embeddings",
)


def _header(path: Path):
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"MTP head file is truncated: {path.name}")
        length = struct.unpack("<Q", raw)[0]
        if not 0 < length <= min(_HEADER_LIMIT, size - 8):
            raise ValueError(f"MTP head has an invalid safetensors header: {path.name}")
        raw_header = stream.read(length)
    header = json.loads(raw_header)
    if not isinstance(header, dict):
        # A refusal (ValueError) like every other artifact defect.
        raise ValueError(f"MTP head has an invalid safetensors header: {path.name}")  # noqa: TRY004
    return raw_header, header, 8 + length


def _head_file(path: Path) -> Path:
    found = [path / name for name in HEAD_FILES if (path / name).is_file()]
    if len(found) != 1 or (path / "model.safetensors.index.json").exists():
        raise ValueError(
            "an MTP head artifact holds exactly one of model.safetensors or "
            "mtp.safetensors (no sharded index)"
        )
    return found[0]


def _normalize(key: str) -> str:
    key = key.removeprefix("model.").removeprefix("language_model.")
    return key if key.startswith("mtp.") else "mtp." + key


def is_mtp_head_artifact(path) -> bool:
    """True for a directory that holds only a Qwen3.5-style MTP head."""
    try:
        path = Path(str(path)).expanduser()
        if not path.is_dir():
            return False
        config_file = path / "config.json"
        if config_file.is_file():
            config = json.loads(config_file.read_text())
            return isinstance(config, dict) and config.get("model_type") in HEAD_MODEL_TYPES
        _, header, _ = _header(_head_file(path))
    except (OSError, ValueError, TypeError):
        return False
    keys = {_normalize(key) for key in header if key != "__metadata__"}
    return bool(keys) and "mtp.fc.weight" in keys and all(
        key.startswith(("mtp.fc.", "mtp.pre_fc_norm_", "mtp.norm.", "mtp.layers.0."))
        for key in keys
    )


def expected_head(text: dict) -> tuple[dict, dict]:
    """(linear modules -> [out, in], vectors -> [n]) for the target geometry."""
    hidden = int(text["hidden_size"])
    inter = int(text["intermediate_size"])
    heads, kv, head_dim = (int(text["num_attention_heads"]),
                           int(text["num_key_value_heads"]), int(text["head_dim"]))
    gate = 2 if text.get("attn_output_gate", True) else 1
    linears = {
        "fc": [hidden, 2 * hidden],
        "layers.0.self_attn.q_proj": [heads * head_dim * gate, hidden],
        "layers.0.self_attn.k_proj": [kv * head_dim, hidden],
        "layers.0.self_attn.v_proj": [kv * head_dim, hidden],
        "layers.0.self_attn.o_proj": [hidden, heads * head_dim],
        "layers.0.mlp.gate_proj": [inter, hidden],
        "layers.0.mlp.up_proj": [inter, hidden],
        "layers.0.mlp.down_proj": [hidden, inter],
    }
    vectors = {
        "pre_fc_norm_hidden": [hidden], "pre_fc_norm_embedding": [hidden], "norm": [hidden],
        "layers.0.input_layernorm": [hidden], "layers.0.post_attention_layernorm": [hidden],
        "layers.0.self_attn.q_norm": [head_dim], "layers.0.self_attn.k_norm": [head_dim],
    }
    return linears, vectors


def _module_quantization(name, shape, entries):
    """None for a float projection, else its exact affine {bits, group_size}."""
    out, cols = shape
    weight = entries.get("weight")
    if weight is None:
        raise ValueError(f"MTP head is missing {name}.weight")
    extra = set(entries) - {"weight", "scales", "biases"}
    if extra:
        raise ValueError(f"MTP head has unexpected tensors on {name}: {sorted(extra)}")
    if weight["dtype"] in _FLOAT and set(entries) == {"weight"}:
        if weight["shape"] != [out, cols]:
            raise ValueError(f"MTP head {name} shape {weight['shape']} != {[out, cols]}")
        return None
    scales, biases = entries.get("scales"), entries.get("biases")
    if weight["dtype"] != "U32" or scales is None or biases is None:
        raise ValueError(f"MTP head {name} is neither float nor affine-quantized")
    packed, groups = weight["shape"][-1], scales["shape"][-1]
    if (weight["shape"][0] != out or scales["shape"] != biases["shape"]
            or scales["shape"][0] != out or scales["dtype"] not in _FLOAT
            or biases["dtype"] not in _FLOAT or (packed * 32) % cols or cols % groups):
        raise ValueError(f"MTP head {name} quantized shapes do not fit [{out}, {cols}]")
    bits, group_size = packed * 32 // cols, cols // groups
    if bits not in _BITS or group_size not in _GROUPS:
        raise ValueError(f"MTP head {name} has unsupported quantization {bits}-bit/g{group_size}")
    return {"bits": bits, "group_size": group_size, "mode": "affine"}


def _declared_quantization(config, name):
    quant = config.get("quantization") or config.get("quantization_config")
    if not isinstance(quant, dict):
        return None
    for key in (name, "mtp." + name, "language_model.mtp." + name):
        if isinstance(quant.get(key), dict):
            return {"bits": quant[key].get("bits"), "group_size": quant[key].get("group_size"),
                    "mode": quant[key].get("mode", "affine")}
        if quant.get(key) is False:
            return None
    return {"bits": quant.get("bits"), "group_size": quant.get("group_size"),
            "mode": quant.get("mode", "affine")}


def _hash_payload(path: Path, offset: int):
    """(sha256 of the whole file, any nonzero payload byte) in one pass."""
    digest = hashlib.sha256()
    nonzero = False
    with path.open("rb") as stream:
        digest.update(stream.read(offset))
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
            if not nonzero and chunk.count(0) != len(chunk):
                nonzero = True
    return digest.hexdigest(), nonzero


def inspect_mtp_head(head_path, target_artifact: dict) -> dict:
    """Bind an external head to an inspected target; refusal raises ValueError."""
    path = Path(str(head_path)).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"MTP head artifact is not a directory: {head_path}")
    if target_artifact.get("has_mtp"):
        raise ValueError(
            "target ships its own MTP head; an external head would replace it "
            "(different checkpoint pairing) and is refused"
        )
    target_config = target_artifact["config"]
    text = target_config.get("text_config", target_config)
    config, config_raw = None, None
    if (path / "config.json").is_file():
        config_raw = (path / "config.json").read_bytes()
        config = json.loads(config_raw)
        if not isinstance(config, dict) or config.get("model_type") not in HEAD_MODEL_TYPES:
            raise ValueError("MTP head config must declare model_type qwen3_5_mtp")
        head_text = config.get("text_config", config)
        if head_text.get("mtp_num_hidden_layers") != 1:
            raise ValueError("MTP head config must declare exactly one MTP layer")
        for key in GEOMETRY_KEYS:
            if key in head_text or key in text:
                ours = head_text.get(key, config.get(key))
                theirs = text.get(key, target_config.get(key))
                if ours != theirs:
                    raise ValueError(f"MTP head {key} {ours!r} does not match target {theirs!r}")
        theta = (head_text.get("rope_parameters") or {}).get("rope_theta", head_text.get("rope_theta"))
        target_theta = (text.get("rope_parameters") or {}).get("rope_theta", text.get("rope_theta"))
        if theta != target_theta:
            raise ValueError(f"MTP head rope_theta {theta!r} does not match target {target_theta!r}")
    file = _head_file(path)
    raw_header, header, offset = _header(file)
    tensors = {}
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        name = _normalize(key)
        if name in tensors:
            raise ValueError(f"MTP head tensor {name} appears twice")
        if not isinstance(entry, dict) or not isinstance(entry.get("shape"), list):
            raise ValueError(f"MTP head tensor {key} has an invalid header entry")  # noqa: TRY004
        tensors[name] = entry
    linears, vectors = expected_head(text)
    quantization, seen = {}, set()
    for name, shape in linears.items():
        entries = {
            suffix: tensors[f"mtp.{name}.{suffix}"]
            for suffix in ("weight", "scales", "biases")
            if f"mtp.{name}.{suffix}" in tensors
        }
        seen.update(f"mtp.{name}.{suffix}" for suffix in entries)
        derived = _module_quantization(name, shape, entries)
        if config is not None and derived is not None:
            # A float projection is unambiguous; a packed one must be the
            # quantization the head declares.
            declared = _declared_quantization(config, name)
            if declared != derived:
                raise ValueError(f"MTP head {name} quantization {derived} != declared {declared}")
        if derived is not None:
            quantization[name] = derived
    for name, shape in vectors.items():
        entry = tensors.get(f"mtp.{name}.weight")
        if entry is None or entry["shape"] != shape or entry["dtype"] not in _FLOAT:
            raise ValueError(f"MTP head {name}.weight is missing or not a float {shape} vector")
        seen.add(f"mtp.{name}.weight")
    if set(tensors) - seen:
        raise ValueError(f"MTP head has tensors outside the single-layer schema: {sorted(set(tensors) - seen)[:6]}")
    weight_sha256, nonzero = _hash_payload(file, offset)
    if not nonzero:
        raise ValueError(
            f"MTP head payload is all zeros (degenerate artifact): {file}"
        )
    stat = file.stat()
    digest = hashlib.sha256(SCHEMA.encode())
    digest.update(hashlib.sha256(config_raw or b"").digest())
    digest.update(hashlib.sha256(raw_header).digest())
    digest.update(weight_sha256.encode())
    return {
        "schema": SCHEMA,
        "path": str(path),
        "file": str(file),
        "prefix": "mtp.",
        "keys": sorted(tensors),
        "quantization": quantization,
        "declared_geometry": config is not None,
        "weight_sha256": weight_sha256,
        "fingerprint": digest.hexdigest(),
        "record": (file.name, stat.st_size, stat.st_mtime_ns),
    }


def attach_mtp_head(artifact: dict, head: dict) -> dict:
    """A copy of the target artifact with the external head bound into it."""
    config = dict(artifact["config"])
    if "text_config" in config:
        config["text_config"] = {**config["text_config"], "mtp_num_hidden_layers": 1}
    else:
        config["mtp_num_hidden_layers"] = 1
    identity = dict(artifact["identity"])
    target_fingerprint = identity["fingerprint"]
    identity.update(
        fingerprint=hashlib.sha256(json.dumps(
            [SCHEMA, target_fingerprint, head["fingerprint"]], separators=(",", ":")
        ).encode()).hexdigest(),
        target_fingerprint=target_fingerprint,
        mtp_head={
            "path": head["path"], "fingerprint": head["fingerprint"],
            "weight_sha256": head["weight_sha256"], "quantization": head["quantization"],
            "declared_geometry": head["declared_geometry"],
            "file": list(head["record"]),
        },
    )
    return {
        **artifact,
        "config": config,
        "has_mtp": True,
        "mtp_tensor_count": len(head["keys"]),
        "mtp_sidecar": {
            "file": head["file"], "prefix": head["prefix"],
            "quantization": head["quantization"], "external": True,
        },
        "identity": identity,
    }


def head_quantization_predicate(quantization: dict, prefix: str = "language_model.mtp."):
    """``nn.quantize`` decision for head modules: per-module entry or float."""

    def decide(name):
        if not name.startswith(prefix):
            return None
        return quantization.get(name[len(prefix):], False)

    return decide


__all__ = [
    "POLICY_KEY",
    "SCHEMA",
    "attach_mtp_head",
    "expected_head",
    "head_quantization_predicate",
    "inspect_mtp_head",
    "is_mtp_head_artifact",
]
