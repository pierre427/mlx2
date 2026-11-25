"""Strict binding of KRAFTON's official DFlash2 drafter to Raon-1-mini.

KRAFTON/Raon-1-mini-dflash2 (Apache-2.0) is mlx2's generic DFlash2 drafter:
five Qwen3-style sliding-window layers, grouped dynamic convolution and the
bilinear candidate selector, no embedding and no head of its own (both bound
from the target).  The one Raon-specific term is the tap width: each tapped
layer feeds its full 4-stream mHC residual (``target_streams_per_layer`` 4,
10240 per tap, 51200 per row).  This module pins the artifact, checks taps and
streams against the target and hands the generic loader the geometry.
Explicit ``draft_model`` only; unqualified.  See provenance/raon1-mini.json.
"""

import hashlib
import json
from pathlib import Path

from ..runtime.drafters.dflash2_config import DFlash2Config
from . import dflash2

SOURCE_REPOSITORY = "KRAFTON/Raon-1-mini-dflash2"
SOURCE_REVISION = "777533d82be8018b8e3c38cbb8a5a75d8e055a71"
CONFIG_SHA256 = "c18e57024746d4dc6e0521cb75ce98d6c64235e53037d05623102e89b25c8367"
WEIGHT_SHA256 = "923d9a3304a392db1327f0233922a0923d3d8e4a871b41b1620c186b8059e1bd"
TARGET_LAYER_IDS = (1, 6, 11, 16, 21)
STREAMS = 4
# A tap is the n-stream residual AFTER 0-based target layer i, before the
# stream mean and final norm (the card's eagle ids [2, 7, 12, 17, 22] are the
# same set in HF hidden_states numbering, index 0 the embeddings).
TAP_CONVENTION = "dflash-target-layer-ids-post-layer-full-mhc-streams-v1"
RECEIPT_SCHEMA = "mlx2.raon1-mini-dflash2-drafter.v1"


def _check_declared_contract(config, target_config):
    if not isinstance(config, dict) or config.get("architectures") != ["DFlash2DraftModel"]:
        raise ValueError("Raon DFlash2 route needs a DFlash2DraftModel artifact")
    nested = config.get("dflash_config") or {}
    if tuple(nested.get("target_layer_ids") or ()) != TARGET_LAYER_IDS:
        raise ValueError(f"Raon DFlash2 taps differ from the release {list(TARGET_LAYER_IDS)}")
    if nested.get("target_streams_per_layer") != STREAMS:
        raise ValueError("Raon DFlash2 must read all 4 mHC streams per tap")
    hidden = target_config["hidden_size"]
    if (nested.get("target_embedding_size") != hidden
            or config.get("target_hidden_size") != STREAMS * hidden
            or nested.get("target_feature_size") != len(TARGET_LAYER_IDS) * STREAMS * hidden):
        raise ValueError("Raon DFlash2 feature geometry differs from the 4-stream target")
    if config.get("num_target_layers") != target_config["num_hidden_layers"]:
        raise ValueError("Raon DFlash2 num_target_layers does not match the target")
    for name in ("hidden_size", "vocab_size"):
        if config.get(name) != target_config.get(name):
            raise ValueError(f"Raon DFlash2 {name} does not match the target")


def inspect_drafter(path, *, target_path, target_config, target_fingerprint):
    path = Path(path).expanduser().resolve()
    raw = (path / "config.json").read_bytes()
    config = dflash2._decode_unique_json(raw, "draft config.json")
    _check_declared_contract(config, target_config)
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("DFlash2 config differs from the pinned Raon-1-mini release")
    record = dflash2.inspect_drafter(path, target_path, target_text=target_config)
    if record["weight_sha256"] != [WEIGHT_SHA256]:
        raise ValueError("DFlash2 weights are not the pinned Raon-1-mini release")
    fingerprint = hashlib.sha256(json.dumps(
        [record["fingerprint"], TAP_CONVENTION, target_fingerprint], separators=(",", ":")).encode()).hexdigest()
    return {**record, "fingerprint": fingerprint, "artifact_fingerprint": record["fingerprint"],
            "content_revision": dflash2.content_revision(record),
            "checkpoint": f"Raon-1-mini-dflash2@sha256:{WEIGHT_SHA256[:16]}",
            "target_path": str(Path(target_path).expanduser().resolve()),
            "target_fingerprint": target_fingerprint, "tap_convention": TAP_CONVENTION,
            "source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION}


def load_drafter(record, target_model):
    if (record.get("config_sha256") != CONFIG_SHA256
            or record.get("tap_convention") != TAP_CONVENTION
            or DFlash2Config.from_dict(record["config"]) != record["args"]
            or tuple(record["args"].target_layer_ids) != TARGET_LAYER_IDS
            or record["args"].target_streams_per_layer != STREAMS):
        raise ValueError("Raon DFlash2 record changed since inspection")
    if (getattr(target_model, "tap_streams", None) != STREAMS
            or not callable(getattr(target_model, "forward_with_taps", None))
            or len(target_model.layers) != record["args"].num_target_layers):
        raise ValueError("Raon target does not expose the 4-stream DFlash2 tap contract")
    return dflash2.load_drafter(record, target_model)
