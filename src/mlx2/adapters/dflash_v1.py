"""Strict inspection, pairing and loading of a z-lab DFlash v1 head on a Qwen3 target.

A ``DFlashDraftModel`` artifact (z-lab ``modeling_dflash.py``, e.g.
``z-lab/Qwen3-4B-DFlash-b16``) is a bidirectional block drafter that shares the
target's embedding and LM head.  Header-only inspection and target pairing run
before any MLX payload load; ``load_drafter`` re-verifies the bytes, then binds
the target's embedding and head after a strict weight load.  The math is
``runtime/drafters/plain_dflash.PlainDFlashDraftModel``.
Provenance: provenance/qwen3-dflash-v1.json.

What is refused rather than guessed:

* any ``dflash_config`` key other than ``mask_token_id`` and
  ``target_layer_ids`` (a DSpine code bridge, a causal head, a sliding window
  or a logit transform would each change the forward);
* ``target_layer_ids`` that differ from the reference's own
  ``build_target_layer_ids(num_target_layers, num_hidden_layers)``: the
  reference computes its taps and never reads the configured list, so a
  disagreement means the config does not describe the trained forward;
* a tap on the target's last layer (mlx2 captures layers before the final norm,
  the reference's last hidden state is after it);
* a mask id that is an ordinary tokenizer token.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from .dflash2 import _decode_unique_json
from .draft_artifacts import inspect_files

ARCHITECTURE = "DFlashDraftModel"
FORMAT = "mlx2.dflash-v1-qwen3.v1"
# mlx2 standard decoder taps are post-layer, pre-final-norm features
# (``forward_with_taps``); the reference reads ``hidden_states[id + 1]``.
TAP_CONVENTION = "post_layer_pre_final_norm_eq_hf_hidden_states_id_plus_1"
_DFLASH_KEYS = frozenset({"mask_token_id", "target_layer_ids"})
_REQUIRED_KEYS = frozenset({
    "block_size", "dflash_config", "head_dim", "hidden_size", "intermediate_size",
    "layer_types", "num_attention_heads", "num_hidden_layers", "num_key_value_heads",
    "num_target_layers", "rms_norm_eps", "rope_theta", "vocab_size",
})
_ALLOWED_KEYS = frozenset({
    "architectures", "attention_bias", "attention_dropout", "auto_map", "block_size",
    "bos_token_id", "dflash_config", "dtype", "eos_token_id", "head_dim", "hidden_act",
    "hidden_size", "initializer_range", "intermediate_size", "layer_types",
    # z-lab configs carry these as top-level nulls; the nested values are used.
    "mask_token_id", "target_layer_ids",
    "max_position_embeddings", "max_window_layers", "model_type", "num_attention_heads",
    "num_hidden_layers", "num_key_value_heads", "num_target_layers", "rms_norm_eps",
    "rope_scaling", "rope_theta", "sliding_window", "tie_word_embeddings", "torch_dtype",
    "transformers_version", "use_cache", "use_sliding_window", "vocab_size",
})


def reference_target_layer_ids(num_target_layers, num_draft_layers):
    """z-lab ``build_target_layer_ids`` (modeling_dflash.py), reproduced exactly."""
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    span = end - start
    return [round(start + (i * span) / (num_draft_layers - 1))
            for i in range(num_draft_layers)]


def parse_config(config):
    """Strict ``DFlashConfig`` of a z-lab Qwen3-backbone DFlash v1 head."""
    from ..runtime.drafters.base_config import DFlashConfig

    if not isinstance(config, Mapping):
        raise TypeError("DFlash v1 config must be an object")
    unknown = set(config) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"Unsupported DFlash v1 config settings: {sorted(unknown)}")
    if config.get("architectures") != [ARCHITECTURE] or config.get("model_type") != "qwen3":
        raise ValueError("DFlash v1 expects a qwen3 DFlashDraftModel artifact")
    # DFlashConfig has defaults for every field; a missing geometry key would
    # silently take another family's value.
    missing = _REQUIRED_KEYS - set(config)
    if missing:
        raise ValueError(f"Missing DFlash v1 config settings: {sorted(missing)}")
    dflash = config.get("dflash_config")
    if not isinstance(dflash, Mapping) or set(dflash) != _DFLASH_KEYS:
        raise ValueError("DFlash v1 dflash_config must hold exactly mask_token_id and target_layer_ids")
    for key in ("mask_token_id", "target_layer_ids"):
        if key in config and config[key] is not None and config[key] != dflash[key]:
            raise ValueError(f"Conflicting DFlash v1 {key} metadata")
    if config.get("attention_bias", False) is not False:
        raise ValueError("DFlash v1 requires bias-free attention")
    if config.get("hidden_act", "silu") != "silu" or config.get("attention_dropout", 0.0) != 0.0:
        raise ValueError("DFlash v1 supports the silu, dropout-free backbone only")
    layers = config.get("num_hidden_layers")
    if (type(layers) is not int or layers < 1
            or config.get("layer_types") != ["full_attention"] * layers
            or config.get("sliding_window") is not None
            or config.get("use_sliding_window", False) is not False):
        raise ValueError("DFlash v1 requires full-attention draft layers")
    if config.get("rope_scaling") is not None:
        raise ValueError("DFlash v1 supports default full-head RoPE only")
    theta = config.get("rope_theta")
    if isinstance(theta, bool) or not isinstance(theta, (int, float)) or not theta > 0:
        raise ValueError("DFlash v1 requires an explicit positive rope_theta")
    if "dtype" in config and "torch_dtype" in config and config["dtype"] != config["torch_dtype"]:
        raise ValueError("Conflicting DFlash v1 dtype metadata")
    block = config.get("block_size")
    if type(block) is not int or block < 2:
        raise ValueError("DFlash v1 block_size must be an integer above one")
    flat = dict(config)
    flat["rope_theta"] = float(theta)
    args = DFlashConfig.from_dict(flat)
    target_layers = args.num_target_layers
    ids = list(args.target_layer_ids)
    if ids != reference_target_layer_ids(target_layers, layers):
        raise ValueError("DFlash v1 target_layer_ids differ from the reference's computed taps")
    if not ids or ids != sorted(set(ids)) or ids[0] < 0 or ids[-1] >= target_layers - 1:
        raise ValueError("DFlash v1 taps must be increasing layers below the target's last layer")
    if type(args.mask_token_id) is not int or not 0 <= args.mask_token_id < args.vocab_size:
        raise ValueError("DFlash v1 mask_token_id must be inside the vocabulary")
    return args


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_pairing(config, target_path, target_config):
    """Refuse a drafter whose ids cannot be the target's; return a receipt.

    The drafter ships no tokenizer (it uses the target's embedding and head).
    The mask id must be a reserved special token of the target tokenizer or an
    embedding row past every tokenizer id (never produced by tokenization).
    """
    target_path = Path(target_path)
    if config.get("vocab_size") != target_config.get("vocab_size"):
        raise ValueError("DFlash v1 vocabulary size differs from the target")
    for name in ("bos_token_id", "eos_token_id"):
        if name in config and config[name] != target_config.get(name):
            raise ValueError(f"DFlash v1 {name} differs from the target")
    tokenizer = json.loads((target_path / "tokenizer_config.json").read_text())
    added = tokenizer.get("added_tokens_decoder") or {}
    vocabulary = json.loads((target_path / "tokenizer.json").read_text())
    ordinary = vocabulary.get("model", {}).get("vocab") or {}
    last_id = max([int(i) for i in ordinary.values()] + [int(k) for k in added], default=-1)
    mask = int(config["dflash_config"]["mask_token_id"])
    entry = added.get(str(mask))
    if isinstance(entry, dict) and entry.get("special", False):
        kind = "reserved_special_token"
    elif mask > last_id:
        kind = "untokenizable_embedding_row"
    else:
        raise ValueError("DFlash v1 mask_token_id is an ordinary target token")
    return {
        "vocab_size": int(config["vocab_size"]),
        "bos_token_id": config.get("bos_token_id"), "eos_token_id": config.get("eos_token_id"),
        "mask_token_id": mask, "mask_token_kind": kind,
        "tokenizer_last_id": last_id,
        "drafter_tokenizer_files": "none (shares target embedding and head)",
    }


def _target_config(target):
    config = _decode_unique_json((Path(target) / "config.json").read_bytes(), "target config")
    return config.get("text_config", config)


def inspect_drafter(path, target):
    """Bind a DFlash v1 artifact's bytes and schema against a Qwen3 target."""
    from ..runtime.drafters.plain_dflash import expected_weight_shapes

    path = Path(path).expanduser().resolve()
    raw = (path / "config.json").read_bytes()
    config = _decode_unique_json(raw, "DFlash v1 config")
    args = parse_config(config)
    text = _target_config(target)
    if text.get("model_type") != "qwen3":
        raise ValueError("DFlash v1 target must be Qwen3")
    for name in ("hidden_size", "vocab_size"):
        if text.get(name) != getattr(args, name):
            raise ValueError(f"DFlash v1 target {name} mismatch")
    if text.get("num_hidden_layers") != args.num_target_layers:
        raise ValueError("DFlash v1 target layer count mismatch")
    pairing = tokenizer_pairing(config, target, text)
    dtype = config.get("dtype", config.get("torch_dtype"))
    if dtype != "bfloat16":
        raise ValueError("DFlash v1 binding accepts BF16 checkpoints only")
    record = inspect_files(path, raw, expected=expected_weight_shapes(args),
                           dtype=dtype, label="DFlash v1")
    config_sha = hashlib.sha256(raw).hexdigest()
    weights = [_sha256(path / name) for name, _, _ in record["files"]]
    # Content-stable (no mtimes): route, APCv2 and receipt identity.
    fingerprint = hashlib.sha256(json.dumps(
        [FORMAT, config_sha, record["header_sha256"], weights, TAP_CONVENTION],
        separators=(",", ":")).encode()).hexdigest()
    return {**record, "fingerprint": fingerprint, "config": config, "args": args,
            "config_sha256": config_sha, "weight_sha256": weights,
            "tap_convention": TAP_CONVENTION, "tokenizer_pairing": pairing}


def content_revision(record):
    """Payload-bound content pin, stable across identical downloads."""
    weights = record.get("weight_sha256")
    if not isinstance(weights, list) or not weights or any(
            not isinstance(value, str) or len(value) != 64 for value in weights):
        raise ValueError("DFlash v1 content revision requires full weight payload hashes")
    return hashlib.sha256(json.dumps(
        [FORMAT, record["config_sha256"], record["header_sha256"], weights, TAP_CONVENTION],
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_drafter(record, target_model, *, runtime_quantization=None,
                 draft_attention_windows=None):
    if runtime_quantization is not None:
        raise ValueError("DFlash v1 runtime quantization is not supported")
    if draft_attention_windows is not None:
        raise ValueError("DFlash v1 serves its trained full-attention block only")
    from ..runtime.drafters.plain_dflash import expected_weight_shapes

    path = Path(record["path"])
    raw = (path / "config.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["config_sha256"]:
        raise ValueError("DFlash v1 config changed since inspection")
    args = parse_config(_decode_unique_json(raw, "DFlash v1 config"))
    if args != record["args"]:
        raise ValueError("DFlash v1 load configuration differs from inspected source")
    current = inspect_files(path, raw, expected=expected_weight_shapes(args),
                            dtype=record["config"].get("dtype", record["config"].get("torch_dtype")),
                            label="DFlash v1")
    if current["files"] != record["files"] or current["header_sha256"] != record["header_sha256"]:
        raise ValueError("DFlash v1 artifact changed since inspection")
    if [_sha256(path / name) for name, _, _ in record["files"]] != record["weight_sha256"]:
        raise ValueError("DFlash v1 tensor bytes changed since inspection")
    import mlx.core as mx

    from ..runtime.drafters.plain_dflash import PlainDFlashDraftModel

    model = PlainDFlashDraftModel(args)
    weights = {}
    for name, _, _ in record["files"]:
        weights.update(mx.load(str(path / name)))
    # Target embedding/head are absent from the draft and bound AFTER the load.
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    weights.clear()
    return model if target_model is None else model.bind(target_model)


__all__ = ["ARCHITECTURE", "content_revision", "inspect_drafter", "load_drafter",
           "parse_config", "reference_target_layer_ids", "tokenizer_pairing"]
