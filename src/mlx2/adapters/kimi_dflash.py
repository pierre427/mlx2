"""Strict inspection, tokenizer pairing and loading of the Kimi Linear DFlash head.

Header-only inspection and target pairing run before any MLX payload load;
the target's embedding and LM head are bound only after a strict weight load.
Provenance: provenance/kimi-linear-speculative.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .dflash2 import _decode_unique_json
from .draft_artifacts import inspect_files

REPO = "Moonlight556/Kimi-Linear-48B-A3B-DFlash-240k"
REVISION = "9037a02a488d1c602e65e26d24d2e6013eb6d8c9"
# Content pins from the pinned revision (HF LFS sha256 for the weights).
CONFIG_SHA256 = "9aac8132bd06f5d3511f01b7b64117c7f8c23ca219ece85aecb607e56ba7ead3"
WEIGHTS_SHA256 = "4471e59faa2c8e0253b7c8f2e1dddbc391bfc5511b24ebd1b4b0aac38182e344"
# The drafter repository ships no tokenizer: it shares the target's
# embedding/head, so pairing is checked against the target's tokenizer
# files (the Kimi tiktoken vocabulary below and its special-token ids).
TARGET_TIKTOKEN_SHA256 = "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_pairing(config, target_path, target_config):
    """Refuse a drafter whose vocabulary ids cannot be the target's.

    Returns a receipt.  Checks: identical vocabulary size; the drafter's BOS,
    EOS and mask ids are the target's configured BOS/EOS and a reserved
    special token of the target tokenizer; the target's tiktoken vocabulary
    is the pinned Kimi vocabulary (the drafter has none of its own).
    """
    target_path = Path(target_path)
    if config.get("vocab_size") != target_config.get("vocab_size"):
        raise ValueError("Kimi DFlash vocabulary size differs from the target")
    for name in ("bos_token_id", "eos_token_id"):
        if config.get(name) != target_config.get(name):
            raise ValueError(f"Kimi DFlash {name} differs from the target")
    tokenizer = json.loads((target_path / "tokenizer_config.json").read_text())
    added = tokenizer.get("added_tokens_decoder") or {}
    mask = int(config["dflash_config"]["mask_token_id"])
    entry = added.get(str(mask))
    if not isinstance(entry, dict) or not entry.get("special", False):
        raise ValueError("Kimi DFlash mask_token_id is not a reserved target special token")
    tiktoken = _sha256(target_path / "tiktoken.model")
    if tiktoken != TARGET_TIKTOKEN_SHA256:
        raise ValueError("target tiktoken vocabulary differs from the pinned Kimi vocabulary")
    return {
        "vocab_size": int(config["vocab_size"]),
        "bos_token_id": config["bos_token_id"], "eos_token_id": config["eos_token_id"],
        "mask_token_id": mask, "mask_token": entry.get("content"),
        "target_tiktoken_sha256": tiktoken,
        "drafter_tokenizer_files": "none (shares target embedding and head)",
    }


def inspect_drafter(path, target_path, *, target_config=None, verify_content=True):
    from ..runtime.drafters.kimi_dflash import expected_weight_shapes, parse_config

    path = Path(path).expanduser().resolve()
    raw = (path / "config.json").read_bytes()
    config = _decode_unique_json(raw, "draft config.json")
    args = parse_config(config)
    if target_config is None:
        target_config = json.loads((Path(target_path) / "config.json").read_text())
    for name in ("hidden_size", "vocab_size"):
        if target_config[name] != getattr(args, name):
            raise ValueError(f"Kimi DFlash target {name} mismatch")
    if target_config["num_hidden_layers"] != args.num_target_layers:
        raise ValueError("Kimi DFlash target layer count mismatch")
    pairing = tokenizer_pairing(config, target_path, target_config)
    record = inspect_files(path, raw, expected=expected_weight_shapes(args),
                           dtype=config.get("dtype", config.get("torch_dtype")), label="Kimi DFlash")
    config_sha = hashlib.sha256(raw).hexdigest()
    weights = [_sha256(path / name) for name, _, _ in record["files"]] if verify_content else None
    # Pinned only with verified content: the pinned revision is one shard, so
    # a resharded or unhashed weight set is never reported as that revision.
    pinned = config_sha == CONFIG_SHA256 and weights == [WEIGHTS_SHA256]
    # A content-stable fingerprint (no mtimes) for route and APCv2 identity.
    fingerprint = hashlib.sha256(json.dumps(
        ["mlx2.kimi-dflash.v1", config_sha, record["header_sha256"], weights],
        separators=(",", ":")).encode()).hexdigest()
    return {**record, "fingerprint": fingerprint, "config": config, "args": args,
            "config_sha256": config_sha, "weight_sha256": weights,
            "pinned_revision": REVISION if pinned else None,
            "checkpoint": f"{REPO}@{REVISION}" if pinned else f"unpinned@{fingerprint}",
            "tokenizer_pairing": pairing}


def load_drafter(record, target_model):
    import mlx.core as mx

    from ..runtime.drafters.kimi_dflash import KimiDFlashDraftModel

    model = KimiDFlashDraftModel(record["args"])
    weights = {}
    for name, _, _ in record["files"]:
        weights.update(mx.load(str(Path(record["path"]) / name)))
    # Target embedding/head are absent from the draft and bound AFTER the load.
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    weights.clear()
    return model if target_model is None else model.bind(target_model)


__all__ = ["REPO", "REVISION", "inspect_drafter", "load_drafter", "tokenizer_pairing"]
