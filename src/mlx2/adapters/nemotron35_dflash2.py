"""Strict binding of the community DFlash2 draft to Nemotron 3.5 Super VL.

The draft (shadowpa0327, not an NVIDIA release) reuses mlx2's DFlash2 drafter
unchanged: five Qwen3-style sliding layers, grouped dynamic convolution and
the bilinear candidate selector, no embedding and no head of its own.  This
module only pins the artifact, checks its declared taps against the target's
88-layer hybrid backbone and hands the generic loader the target's text
geometry.  Opt-in (an explicit ``draft_model``), unqualified, never a default.
See provenance/nemotron35-super-vl-dflash2.json.
"""

import hashlib
import json
from pathlib import Path

from ..runtime.drafters.dflash2_config import DFlash2Config
from . import dflash2
from .nemotron35_draft_quantization import PROFILES, normalize_runtime_quantization

SOURCE_REPOSITORY = "shadowpa0327/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-DFlash2"
TRAINED_TARGET = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
# Both published revisions share this config.json byte for byte.
CONFIG_SHA256 = "348514a689e3a045ce4a6a50b272eb0a8034a5c4d5d95c91b5d36219a98d68d6"
# model.safetensors SHA-256 -> the Hugging Face revision it was fetched from.
REVISIONS = {
    "21ea2a44a0283656882599b45ca3ddb6c78ce5b523f710810ee1ed6655c9305d": {
        "revision": "6b781be2f2e657bde8bbf2719c5469c7359b1361",
        "label": "step-51000",
        "card_status": "the checkpoint the card's acceptance numbers were measured on",
    },
    "1c35cae00c8e15cd4f6a5c256a951a18f7a31aeaaeaa63528511da009e31eb32": {
        "revision": "1780120951f582930ebf805de25420af569a33e9",
        "label": "main-step-75000",
        "card_status": "latest; not served or evaluated by its author",
    },
}
TARGET_LAYER_IDS = (5, 25, 45, 65, 85)
# DFlash ``target_layer_ids`` name the hidden state AFTER decoder layer i
# (the reference reads hidden_states[i + 1], index 0 being the embedding):
# mlx2's ``forward_with_taps`` convention, so the ids pass through unshifted.
# Inferred from the DFlash reference and every other DFlash pair bound here
# (NVIDIA's LiLiCorr for this target included), not read from the SGLang
# Nemotron-H capture this draft was trained with; the GSM8K acceptance in
# the provenance entry is the evidence that it holds.
TAP_CONVENTION = "dflash-target-layer-ids-post-layer-pre-final-norm-v1"
RECEIPT_SCHEMA = "mlx2.nemotron35-super-vl-dflash2-drafter.v1"


def _check_declared_contract(config, text):
    """Architecture and taps against the target, before any hash or tensor."""
    if not isinstance(config, dict) or config.get("architectures") != ["DFlash2DraftModel"]:
        raise ValueError("Super VL DFlash2 route needs a DFlash2DraftModel artifact")
    nested = config.get("dflash_config")
    taps = nested.get("target_layer_ids") if isinstance(nested, dict) else None
    kinds = text.get("layers_block_type")
    if not isinstance(kinds, list) or config.get("num_target_layers") != len(kinds):
        raise ValueError(
            "DFlash2 draft num_target_layers does not match the Nemotron target backbone"
        )
    if (
        not isinstance(taps, list)
        or tuple(taps) != TARGET_LAYER_IDS
        or any(type(i) is not int or not 0 <= i < len(kinds) for i in taps)
    ):
        raise ValueError(
            "DFlash2 draft target_layer_ids differ from the Super VL release "
            f"{list(TARGET_LAYER_IDS)}"
        )
    for name in ("hidden_size", "vocab_size"):
        if type(config.get(name)) is not int or config[name] != text.get(name):
            raise ValueError(f"DFlash2 draft {name} does not match the Nemotron target")
    return [kinds[i] for i in taps]


def inspect_drafter(path, *, target_path, target_text, target_fingerprint, eos_token_ids=()):
    """Pin, schema-check and payload-hash the artifact; no MLX, no tensors."""
    path = Path(path).expanduser().resolve()
    raw = (path / "config.json").read_bytes()
    config = dflash2._decode_unique_json(raw, "draft config.json")
    tap_kinds = _check_declared_contract(config, target_text)
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("DFlash2 config differs from the pinned Super VL community release")
    if eos_token_ids and config.get("eos_token_id") not in tuple(eos_token_ids):
        raise ValueError("DFlash2 draft eos_token_id is not a target EOS id")
    record = dflash2.inspect_drafter(path, target_path, target_text=target_text)
    if len(record["weight_sha256"]) != 1 or record["weight_sha256"][0] not in REVISIONS:
        raise ValueError(
            "DFlash2 weights are not a pinned revision of the Super VL community release"
        )
    if config != record["config"]:
        raise ValueError("DFlash2 config changed while inspecting")
    revision = REVISIONS[record["weight_sha256"][0]]
    fingerprint = hashlib.sha256(
        json.dumps(
            [record["fingerprint"], TAP_CONVENTION, target_fingerprint],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        **record,
        "fingerprint": fingerprint,
        "artifact_fingerprint": record["fingerprint"],
        "content_revision": dflash2.content_revision(record),
        "target_path": str(Path(target_path).expanduser().resolve()),
        "target_fingerprint": target_fingerprint,
        "tap_convention": TAP_CONVENTION,
        "tap_layer_kinds": tap_kinds,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": revision["revision"],
        "source_revision_label": revision["label"],
    }


def drafter_receipt(record, *, target_is_trained_export, runtime_quantization=None):
    """What is bound, for adapter diagnostics; states stay distinct."""
    args = record["args"]
    return {
        "schema": RECEIPT_SCHEMA,
        "architecture": "DFlash2DraftModel",
        "implemented": True,
        "qualified": False,
        "selected": True,
        "authority": "operator_draft_model",
        # Engagement is counted per request, in the route receipt's
        # speculation block (kind external_dflash2, rounds, proposals,
        # acceptances) and the scheduler's external_rounds.
        "observed_used": "see route receipt speculation counters",
        "source_repository": record["source_repository"],
        "source_revision": record["source_revision"],
        "source_revision_label": record["source_revision_label"],
        "weight_sha256": record["weight_sha256"][0],
        "config_sha256": record["config_sha256"],
        "content_revision": record["content_revision"],
        "trained_target": TRAINED_TARGET,
        "target_is_trained_export": bool(target_is_trained_export),
        "target_fingerprint": record["target_fingerprint"],
        "target_layer_ids": list(args.target_layer_ids),
        "tap_layer_kinds": list(record["tap_layer_kinds"]),
        "tap_convention": record["tap_convention"],
        "trained_block_size": args.block_size,
        "mask_token_id": args.mask_token_id,
        "sliding_window": args.sliding_window,
        "shared_target_embedding_and_head": True,
        "speculative_rollback": "exact",
        "runtime_quantization": runtime_quantization,
        "scope": "text-only; images and tool calls unevaluated by the draft's author",
    }


def load_drafter(record, target_model, *, runtime_quantization=None):
    """Strict load through the generic DFlash2 loader, then target tap checks."""
    runtime_quantization = normalize_runtime_quantization(runtime_quantization)
    if (
        record.get("config_sha256") != CONFIG_SHA256
        or record.get("tap_convention") != TAP_CONVENTION
        or DFlash2Config.from_dict(record["config"]) != record["args"]
        or tuple(record["args"].target_layer_ids) != TARGET_LAYER_IDS
    ):
        raise ValueError("DFlash2 record changed since inspection")
    layers = getattr(target_model, "layers", None)
    if (
        not callable(getattr(target_model, "forward_with_taps", None))
        or not callable(getattr(target_model, "prefill_body", None))
        or layers is None
        or len(layers) != record["args"].num_target_layers
    ):
        raise ValueError("Nemotron target does not expose the DFlash2 tap contract")
    quantization = None
    if runtime_quantization is not None:
        profile = PROFILES[runtime_quantization]
        quantization = {"bits": profile["bits"], "group_size": profile["group_size"]}
    return dflash2.load_drafter(record, target_model, runtime_quantization=quantization)
