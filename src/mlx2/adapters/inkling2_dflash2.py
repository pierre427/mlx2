"""Strict binding of the community DFlash2 block drafter to Inkling 2 Mini.

``incoai/Inkling-2-Mini-DFlash2`` (Apache-2.0) is a generic five-layer DFlash2
drafter (sliding-window Qwen3-style layers, grouped dynamic convolution,
bilinear candidate selector; no embedding or head of its own).  Four terms
differ from the DFlash2 drafters mlx2 already serves, and each is a target or
config declaration rather than a model-name branch in the generic code:

* taps index Inkling 2's 48 layer VISITS in execution order (24 physical
  layers, middle block run three times); tap 48 is the final-normed hidden.
  This is the tap contract the Inkling 2 target already serves for DSpark
  (``Model.forward_with_taps``), declared here as ``speculative_tap_count``
  and ``speculative_final_norm_tap``;
* input tokens embed through the target's own embedding RMSNorm
  (``TextModel.embed``), declared as ``speculative_token_embedding``;
* the target head's padded rows (201024 stored, 200058 real) are cropped,
  declared as ``speculative_vocab_rows``;
* ``selector_softcap_mode: "after_correction"`` (a validated
  ``DFlash2Config`` field) caps the selector path score instead of the
  drafter logits.

Clean-room mlx2 code; see provenance/inkling2-dflash2.json.  Opt-in only
(``draft_model`` set to the artifact path), never a default; unqualified.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

from ..runtime.drafters.dflash2_config import DFlash2Config
from . import dflash2

SOURCE_REPOSITORY = "incoai/Inkling-2-Mini-DFlash2"
SOURCE_REVISION = "4f9088ea66c21f2b99ed657d1941c306b03e84d2"
CONFIG_SHA256 = "d094e435eb939e01fca6297f637c5af35e3da4b552f8e1ab0a181cd278a1cefa"
WEIGHT_SHA256 = "217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee"
TARGET_LAYER_IDS = (5, 23, 41, 48)
SELECTOR_SOFTCAP_MODE = "after_correction"
# Tap i < visits: the block state ENTERING visit i (Inkling 2's DSpark tap
# law, ``Model.tap_outputs``); tap == visits: the final-normed hidden.
TAP_CONVENTION = "inkling2-visit-taps-entering-visit-final-norm-v1"
EMBEDDING_LAW = "inkling2-embed-rmsnorm-v1"
# inkling2_dflash2_embed_norm: false (diagnostic A/B arm only).
RAW_EMBEDDING_LAW = "inkling2-raw-embed-table-diagnostic-v1"
RECEIPT_SCHEMA = "mlx2.inkling2-dflash2-drafter.v1"
ROUTE_TAG = "inkling2-dflash2-v1"
PROFILE = "inkling2-apcv2-dflash2"
# Execution-policy keys the DFlash2 route accepts.  Everything else the
# generic external policy knows (tree/lattice/continuation/progressive
# routes, device rounds) is refused: the post-correction selector cap is
# implemented on the chain paths only.
ALLOWED_POLICY_KEYS = frozenset({
    "draft_model", "num_draft", "adaptive_verification", "proposal_composition",
    # Reversible cost-per-commit handoff (2026-11-26, opt-in, unqualified):
    # the engine parks a session's speculation where it measurably loses to
    # plain decode at its width and re-probes later (width priors), the
    # measured form of width gating for multi-lane DFlash2.
    "cost_per_commit_handoff",
})
EMBED_NORM_OPTION = "inkling2_dflash2_embed_norm"
# Draft depth (3..num_draft) while more than one lane is active (opt-in, 2026-11-26): the
# generator's multilane_draft_cap. One lane keeps num_draft. Implemented, not
# qualified; the GPU sweep at 15bdbb21c found depth 7 losing to ordinary
# batched decode at 4 and 16 clients (each lane verifies 8 rows).
MULTILANE_DRAFT_CAP_OPTION = "inkling2_dflash2_multilane_draft_cap"
# How a lane hands out the tokens one verify accepted when the server has more
# than one lane (2026-11-26): "all" (the route's behaviour) emits them at once
# so the lanes stay in lockstep and verify together; "one" (the engine's
# default) emits one per poll, and a lane that accepted several sits out the
# next verifies while it drains, which split the 16-lane sweep at 98cbaa341
# into one-lane rounds (936 of 1,525 rounds, 9,889 lane skips). The committed
# tokens are the same. One lane is unaffected (the drain never splits a cohort).
READY_DRAIN_OPTION = "inkling2_dflash2_ready_drain"
READY_DRAIN_MULTILANE = "all"
# Fail-closed admission refusals of this route, by reason, process-wide.
# The adapter keeps its own per-instance count (diagnostics()["dflash2"]).
REFUSALS = Counter()


class DFlash2Refusal(ValueError):
    """A counted refusal of the Inkling 2 DFlash2 route (``reason``)."""

    def __init__(self, reason, message):
        super().__init__(f"{message} [dflash2_refusal={reason}]")
        self.reason = reason


def refuse(reason, message):
    """Count one refusal and return the error to raise."""
    REFUSALS[reason] += 1
    return DFlash2Refusal(reason, message)


def embedding_law(embed_norm):
    return EMBEDDING_LAW if embed_norm else RAW_EMBEDDING_LAW


def target_visit_count(text_config):
    """Layer visits of an Inkling 2 text config (no MLX import)."""
    layers = int(text_config["num_hidden_layers"])
    loop = text_config.get("loop_config")
    if not loop:
        return layers
    core = int(loop["core_end"]) - int(loop["core_start"])
    return layers + core * (int(loop["num_passes"]) - 1)


def is_dflash2_artifact(path):
    """Whether ``path`` names a directory holding a DFlash2 drafter config."""
    try:
        config = json.loads((Path(path).expanduser() / "config.json").read_text())
    except (OSError, ValueError):
        return False
    return isinstance(config, dict) and config.get("architectures") == ["DFlash2DraftModel"]


def _check_declared_contract(config, text):
    if not isinstance(config, dict) or config.get("architectures") != ["DFlash2DraftModel"]:
        raise refuse("not_dflash2", "Inkling 2 DFlash2 route needs a DFlash2DraftModel artifact")
    nested = config.get("dflash_config") or {}
    if tuple(nested.get("target_layer_ids") or ()) != TARGET_LAYER_IDS:
        raise refuse("tap_ids", f"Inkling 2 DFlash2 taps differ from the release {list(TARGET_LAYER_IDS)}")
    if nested.get("selector_softcap_mode") != SELECTOR_SOFTCAP_MODE:
        raise refuse("softcap_mode", "Inkling 2 DFlash2 selector_softcap_mode differs from the release")
    if nested.get("final_logit_softcapping") != text.get("final_logit_softcapping"):
        raise refuse("softcap_value", "Inkling 2 DFlash2 softcap differs from the target's")
    if config.get("num_target_layers") != target_visit_count(text):
        raise refuse("visit_count", "Inkling 2 DFlash2 num_target_layers differs from the target visit count")
    for name in ("hidden_size", "vocab_size"):
        if config.get(name) != text.get(name):
            raise refuse("geometry", f"Inkling 2 DFlash2 {name} does not match the target")
    rows = text.get("unpadded_vocab_size")
    mask = nested.get("mask_token_id")
    if type(rows) is not int or type(mask) is not int or not 0 <= mask < rows:
        raise refuse("mask_token", "Inkling 2 DFlash2 mask token lies outside the real vocabulary")


def inspect_drafter(path, *, target_path, text_config, target_fingerprint, embed_norm=True):
    """Fail-closed, pre-load inspection of the pinned release against the target.

    Every failure is a counted ``DFlash2Refusal`` (I/O and malformed
    metadata included).  ``embed_norm`` False (the diagnostic arm) is a
    different drafter law: it changes the record fingerprint, draft
    revision and embedding law.
    """
    try:
        return _inspect(path, target_path=target_path, text_config=text_config,
                        target_fingerprint=target_fingerprint, embed_norm=embed_norm)
    except DFlash2Refusal:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        raise refuse("inspection_error", f"Inkling 2 DFlash2 inspection failed: "
                     f"{type(error).__name__}: {error}") from error


def _inspect(path, *, target_path, text_config, target_fingerprint, embed_norm):
    if type(embed_norm) is not bool:
        raise refuse("embed_norm_type", "Inkling 2 DFlash2 embed_norm must be a boolean")
    law = embedding_law(embed_norm)
    path = Path(path).expanduser().resolve()
    raw = (path / "config.json").read_bytes()
    config = dflash2._decode_unique_json(raw, "draft config.json")
    _check_declared_contract(config, text_config)
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise refuse("config_pin", "DFlash2 config differs from the pinned Inkling-2-Mini-DFlash2 release")
    try:
        record = dflash2.inspect_drafter(
            path, target_path, target_text=text_config,
            target_tap_count=target_visit_count(text_config), target_final_norm_tap=True,
        )
    except ValueError as error:
        raise refuse("generic_admission", str(error)) from error
    if record["weight_sha256"] != [WEIGHT_SHA256]:
        raise refuse("weight_pin", "DFlash2 weights are not the pinned Inkling-2-Mini-DFlash2 release")
    fingerprint = hashlib.sha256(json.dumps(
        [record["fingerprint"], TAP_CONVENTION, law, target_fingerprint],
        separators=(",", ":")).encode()).hexdigest()
    return {**record, "fingerprint": fingerprint, "artifact_fingerprint": record["fingerprint"],
            "draft_revision": fingerprint,
            "content_revision": dflash2.content_revision(record),
            "checkpoint": f"Inkling-2-Mini-DFlash2@sha256:{WEIGHT_SHA256}",
            "target_path": str(Path(target_path).expanduser().resolve()),
            "target_fingerprint": target_fingerprint, "target_revision": target_fingerprint,
            "tap_convention": TAP_CONVENTION, "embedding_law": law, "embed_norm": embed_norm,
            "source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION}


def budget_draft(config):
    """The drafter's KV geometry in ``Inkling2CacheBudget``'s ``draft`` shape.

    Every DFlash2 layer is a sliding-window layer (``RotatingKVCache`` of
    ``sliding_window - 1`` rows); a full-attention layer is priced as global.
    The DSpark-shaped feature term (one tap row) slightly over-prices a
    drafter that keeps no feature row.
    """
    types = list(config.get("layer_types") or ["full_attention"] * config["num_hidden_layers"])
    return {
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "local_layer_ids": [i for i, kind in enumerate(types) if kind == "sliding_attention"],
        "num_key_value_heads": int(config["num_key_value_heads"]),
        "head_dim": int(config["head_dim"]),
        "sliding_window_size": int(config.get("sliding_window") or 0),
        "target_layer_ids": list(config["dflash_config"]["target_layer_ids"]),
        "hidden_size": int(config["hidden_size"]),
    }


def declare_target(target_model):
    """Attach the DFlash2 target declarations to an Inkling 2 ``Model``.

    Instance attributes, set only when this route is selected, so the
    ordinary and DSpark routes' model (and its hashed source) is untouched.
    """
    text = target_model.model
    if not callable(getattr(target_model, "forward_with_taps", None)) or not callable(
            getattr(text, "embed", None)):
        raise refuse("target_norm_or_taps", "Inkling 2 target does not expose visit taps and its embedding")
    visits = len(text.layer_indices)
    trained = target_model.config.loop_config.num_passes
    if text.num_passes != trained:
        raise refuse("loop_count", "Inkling 2 DFlash2 taps assume the trained loop count")
    rows = target_model.config.unpadded_vocab_size
    for name, value in (("speculative_tap_count", visits),
                        ("speculative_final_norm_tap", True),
                        ("speculative_token_embedding", text.embed),
                        ("speculative_token_embedding_law", EMBEDDING_LAW),
                        ("speculative_vocab_rows", int(rows))):
        object.__setattr__(target_model, name, value)
    return {"tap_count": visits, "final_norm_tap": True, "vocab_rows": int(rows),
            "embedding_law": EMBEDDING_LAW}


def load_drafter(record, target_model):
    """Load the inspected release; the embedding law comes from the record."""
    embed_norm = record.get("embed_norm")
    if (type(embed_norm) is not bool
            or record.get("embedding_law") != embedding_law(embed_norm)
            or record.get("config_sha256") != CONFIG_SHA256
            or record.get("tap_convention") != TAP_CONVENTION
            or DFlash2Config.from_dict(record["config"]) != record["args"]
            or tuple(record["args"].target_layer_ids) != TARGET_LAYER_IDS
            or record["args"].selector_softcap_mode != SELECTOR_SOFTCAP_MODE):
        raise refuse("record_changed", "Inkling 2 DFlash2 record changed since inspection")
    declared = declare_target(target_model)
    if declared["tap_count"] != record["args"].num_target_layers:
        raise refuse("visit_count", "Inkling 2 DFlash2 visit count differs from the loaded target")
    drafter = dflash2.load_drafter(record, target_model)
    if not embed_norm:
        # Diagnostic arm only (inkling2_dflash2_embed_norm: false); sticky
        # across a later bind/reset.
        drafter.use_raw_token_embedding()
    object.__setattr__(drafter, "inkling2_dflash2_receipt", {
        "schema": RECEIPT_SCHEMA, "checkpoint": record["checkpoint"],
        "source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION,
        "tap_convention": TAP_CONVENTION, "target_layer_ids": list(TARGET_LAYER_IDS),
        "selector_softcap_mode": SELECTOR_SOFTCAP_MODE,
        "embedding_law": record["embedding_law"], "draft_revision": record["draft_revision"],
        "vocab_rows": declared["vocab_rows"], "qualified": False,
    })
    return drafter
