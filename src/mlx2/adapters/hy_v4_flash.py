"""Tencent Hy4-flash (``hy_v4_flash``) ordinary text decode + opt-in DFly2.

Original mlx2 adapter; model math in ``runtime/models/hy_v4_flash.py`` (see
provenance/hy-v4-flash.json).  Unqualified.

Routes:

* ``ordinary`` (default): the reference path.  Engram tables are read row by
  row from their safetensors shards (``hy_v4_flash_engram``).  Images are
  opt-in by content: when the artifact carries the ``vit.*`` tower (and the
  policy does not set ``"vision": false``) an image part runs one isolated
  B=1 media prefill (``hy_v4_flash_media``; tower and 3D M-RoPE in
  ``runtime/models/hy_v4_flash_vision.py``), then ordinary decode.  Image
  input is implemented and unqualified; video parts and media on the
  external-draft routes fail closed.
* DFly2 (opt-in, ``{"draft_model": "embedded_dfly"}`` execution policy): the
  checkpoint's own five-layer drafter, KV Share + Index Share over target
  layers 25/29/33/37/41.  It is exposed as ``generate_dfly`` (exact greedy
  draft/verify, B = 1) and is **not** wired into the serving engine's
  external-draft batch generator, whose drafter protocol has no access to
  the target's KV.  Diagnostics report it implemented, unqualified, and
  selected only when the policy asks for it.

APCv2 (implemented, unqualified): the serving engine publishes exact
``HY4FlashLayerCache`` entries at prompt and generation boundaries and
restores them by copy.  A branch landing below a stored entry is a trim,
which ``HY4FlashLayerCache.can_trim`` admits only while the sliding window
and the Engram token tail still cover the next forward; otherwise APCv2
lands on a shallower entry or misses.  No cache projection is declared,
so interior/rolling checkpoints stay inert (docs/hy4-flash/apcv2.md).

* fp8 latent KV (opt-in, ``{"latent_kv": "fp8"}``; composes with either
  route): MLA cache rows stored as vLLM ``fp8_ds_mla``-style packed rows
  (E4M3 latent, pow2 tile scales, bf16 RoPE tail; 656 vs 1152 bytes per row
  at r 512 / p 64), dequantized before attention.  Approximate state,
  unqualified, off by default; the receipts say so.  The approximate rows
  are never published to APCv2: with fp8 selected the descriptor declares no
  prefix reuse and no approximate APC operation.  Notes:
  docs/hy4-flash/fp8-latent-kv.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..process_env import (
    PROCESS_NUMERICS,
    assert_process_profile_applied,
    clear_inherited_profile,
    require_process_numerics,
)
from ..sampling_defaults import GENERATION_CONFIG, SamplingDefaults, VendorSampling
from .ordinary_artifact import inspect_indexed_artifact
from .external_draft_policy import ExternalDraftAdapterMixin
from .ordinary_text import OrdinaryTextAdapter, _template_history

CACHE_LAYOUT = "hy-v4-flash-mla-latent-indexer-v1"
# DFly2 routes record indexer selections in the target cache (Index Share);
# their APCv2 entries carry that plane, so they live in their own namespace.
DFLY_CACHE_LAYOUT = CACHE_LAYOUT + "+dfly2-index-share"
# Distinct layout for the approximate fp8 latent rows (never the exact one).
FP8_CACHE_LAYOUT = "hy-v4-flash-mla-latent-fp8-ds-mla-indexer-v1"
LATENT_KV_FORMATS = ("native", "fp8")
# generation_config.json of tencent/Hy4-flash@d88c9dfd: do_sample, T 0.9,
# top_p 1, top_k -1 (off).
SAMPLING = VendorSampling.single(
    SamplingDefaults(temperature=0.9, top_p=1.0, source=GENERATION_CONFIG,
                     note="do_sample=true; top_k -1 (off)"),
    model="tencent/Hy4-flash",
)
DFLY_POLICY = "embedded_dfly"
# Written by scripts/convert_hy_v4_flash_mlx.py: trunk under Model.sanitize
# names, MLX affine quantization, Engram tables and DFly2 under checkpoint names.
SANITIZED_LAYOUT = "hy_v4_flash-sanitized-v1"

# Reference topology switches this port implements; anything else refuses.
_TEXT_EXPECTED = {
    "use_mla": True, "use_dsa": True, "use_compressed_indexer": True,
    "compressed_indexer_key_pooling": "gated", "compressed_indexer_skip_fine_select": True,
    "compressed_indexer_reuse_fine_indexer": False, "gated_mla": True,
    "gating_type": "elementwise", "enable_ihc": True, "sandwich_norm": True,
    "learnable_sink": True, "hidden_act": "silu", "n_group": 1, "topk_group": 1,
    "o_lora_op": "none", "gate_lora_op": "none", "dsa_skip_attn_post_norm": False,
    "full_nope": False, "moe_router_use_sigmoid": True, "moe_router_enable_expert_bias": True,
    "norm_topk_prob": True, "tie_word_embeddings": False, "num_nextn_predict_layers": 0,
    "qk_norm": True, "gate_groups": 1, "o_groups": 1, "engram_isolate_fffd_namespace": False,
}


# Packed Engram tables are a weight format, not cache state: the cache rows
# a model with ``affine4_g32`` tables writes are exact for that model, so
# they publish to APCv2 in their own namespace (like any quantized artifact).
# fp8 latent rows are approximate *cache state* and are never published.
def numerics_layout(layout: str, *, engram_table_format: str = "bf16",
                    engram_precision: str = "transformers") -> str:
    """The APCv2 namespace: every knob that changes numerics changes it.

    A packed Engram table or the non-default Engram precision profile gives
    different logits for the same tokens, so their entries must never mix
    with the default ones (fp8 latent rows and DFly2 already pick their own
    base layouts).
    """
    if engram_table_format != "bf16":
        layout = f"{layout}+engram-{engram_table_format}"
    if engram_precision != "transformers":
        layout = f"{layout}+engram-precision-{engram_precision}"
    return layout


def latent_kv_receipt(fmt: str, *, observed_used: bool = False, layout: str | None = None) -> dict:
    """Route-receipt block for the latent KV row format."""
    fp8 = fmt == "fp8"
    return {"format": fmt, "implemented": True, "selected": fp8, "approximate": fp8,
            "qualified": False if fp8 else None, "observed_used": bool(fp8 and observed_used),
            "state": "approximate; unqualified" if fp8 else "exact",
            "cache_layout": layout if layout is not None else (FP8_CACHE_LAYOUT if fp8 else CACHE_LAYOUT),
            "apc_v2_publication": ("none (approximate rows; no approximate operation declared)"
                                   if fp8 else "exact entries")}


def descriptor_for(*, dfly_selected: bool = False, engram_table_format: str = "bf16",
                   latent_kv: str = "native", vision: bool = False) -> ModelDescriptor:
    # Ordinary lanes batch (per-lane attention, joint projections/MoE/head).
    # APCv2: exact-prefix reuse at published boundaries; a branch landing
    # that would roll a bounded plane (sliding window, Engram tail) past its
    # retention is refused by ``preflight_trim``.
    # REASONING: opt-in per request (``no_think`` stays the default).
    # GRAMMAR: forced tool_choice decodes under the Hy4 call grammar; it also
    # admits the generic response_format/grammar processor.  All unqualified.
    capabilities = {Capability.TEXT, Capability.STREAMING, Capability.CONTINUOUS_BATCH,
                    Capability.TOOLS, Capability.REASONING, Capability.GRAMMAR}
    fp8 = latent_kv == "fp8"
    if not fp8:
        # Approximate fp8 rows are never published as exact prefixes.
        capabilities |= {Capability.PREFIX_REUSE, Capability.APC_V2}
    variant = "dfly2-greedy" if dfly_selected else "ordinary"
    if vision:
        capabilities.add(Capability.VISION)
    return ModelDescriptor(
        model_type="hy_v4_flash", family="hy-v4-flash",
        variant=variant + ("+fp8-latent-kv" if fp8 else ""),
        state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.SPARSE_INDEX,
                                StatePlane.RNG, StatePlane.TRANSCRIPT}),
        capabilities=frozenset(capabilities),
        cache_layout=(FP8_CACHE_LAYOUT if fp8 else DFLY_CACHE_LAYOUT if dfly_selected else CACHE_LAYOUT),
        metadata={"execution": "mlx2.adapters.hy_v4_flash.HYV4FlashAdapter",
                  "qualification": "pending",
                  "scope": "text+image (image implemented; unqualified)" if vision else "text-only",
                  "vision": ("image: ordinary route, isolated B=1 media prefill; video refused"
                             if vision else "not loaded"),
                  "dfly2": "selected; unqualified" if dfly_selected else "implemented; not selected",
                  "apc_v2": ("not declared (fp8 latent rows are approximate)" if fp8 else
                             "implemented; unqualified (exact published boundaries, "
                             "bounded-rollback landings refused)"),
                  "tools": "hy_v4 <tool_call:S> parser, auto/none; forced tool_choice "
                           "(required/named) under the Hy4 call grammar; implemented, unqualified",
                  "reasoning": "reasoning_effort no_think by default; high/max opt-in per request "
                               "(enable_thinking or reasoning_effort) to reasoning_content; "
                               "implemented, unqualified",
                  "grammar": "Hy4 forced-call grammar and generic structured output; "
                             "implemented, unqualified",
                  "engram_tables": "bf16 (default)" if engram_table_format == "bf16" else
                  f"{engram_table_format}; opt-in; unqualified"
                  + ("; approximate" if engram_table_format == "affine4_g32" else ""),
                  "latent_kv": ("fp8 (approximate state; selected; unqualified)" if fp8
                                else "native (exact); fp8 implemented, not selected")},
    )


def text_config(config: dict) -> dict:
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("Hy4-flash config has no text_config")
    return text


def inspect_artifact(model_path: str | Path) -> dict:
    """Fail-closed topology/tensor check with no MLX import."""
    artifact = inspect_indexed_artifact(model_path)
    config = artifact["config"]
    if config.get("model_type") != "hy_v4_flash":
        raise ValueError("artifact is not Hy4-flash")
    text = text_config(config)
    for key, value in _TEXT_EXPECTED.items():
        if text.get(key, value) != value:
            raise ValueError(f"Hy4-flash {key}={text.get(key)!r} is not implemented")
    layers = int(text["num_hidden_layers"])
    kinds = text.get("layer_types") or []
    if len(kinds) != layers or any(k not in ("deepseek_sparse_attention", "indexed_attention",
                                             "sliding_attention") for k in kinds):
        raise ValueError("Hy4-flash layer types are unsupported")
    if any(t != "full" for t in text.get("indexer_types") or ["full"]):
        raise ValueError("Hy4-flash fine indexer types are not implemented")
    for name, rope in (text.get("rope_parameters") or {}).items():
        if rope.get("rope_type", "default") != "default":
            raise ValueError(f"Hy4-flash {name} RoPE type is not implemented")
        section = rope.get("mrope_section")
        if section is not None and sum(section) * 2 != int(text["qk_rope_head_dim"]):
            raise ValueError("Hy4-flash M-RoPE sections do not cover the rotary width")
    if text.get("window_size") not in (None, [int(text["sliding_window"]) - 1, 0]):
        raise ValueError("Hy4-flash window_size disagrees with sliding_window")
    quant = config.get("quantization") or text.get("quantization")
    if config.get("quantization_config") is not None:
        raise ValueError("Hy4-flash FP8/compressed-tensors checkpoints are not implemented")
    if quant is not None and (not isinstance(quant, dict) or quant.get("mode", "affine") != "affine"):
        raise ValueError("Hy4-flash admits bf16 or MLX affine-quantized artifacts only")
    weights = artifact["weight_map"]
    sanitized = config.get("mlx2_layout") == SANITIZED_LAYOUT
    if config.get("mlx2_layout") not in (None, SANITIZED_LAYOUT):
        raise ValueError(f"unknown Hy4-flash mlx2 layout {config.get('mlx2_layout')!r}")
    required = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
                "model.hc_head.hc_head_fn"}
    mlp_kinds = text.get("mlp_layer_types") or ["dense"] + ["sparse"] * (layers - 1)
    for i in range(layers):
        p = f"model.layers.{i}."
        required |= {p + "self_attn.q_a_proj.weight", p + "self_attn.linear_gate.weight",
                     p + "self_attn.learnable_sink_param",
                     p + "hc_attn_layer.hc_pre.hc_fn", p + "hc_mlp_layer.hc_pre.hc_fn"}
        required |= ({p + "self_attn.embed_q.weight", p + "self_attn.unembed_out.weight"}
                     if sanitized else {p + "self_attn.kv_b_proj.weight"})
        if kinds[i] != "sliding_attention":
            required.add(p + "self_attn.compressed_indexer.query_projection.weight")
        if mlp_kinds[i] == "sparse":
            required |= {p + "mlp.gate.e_score_correction_bias", p + "mlp.combine_output_norm.weight"}
            required.add(p + ("mlp.experts.gate_proj.weight" if sanitized else "mlp.experts.gate_up_proj"))
    for lid in text.get("engram_layer_ids") or []:
        p = f"model.layers.{int(lid) - 1}.engram_layer."
        required |= {p + "lookup_table", p + "multi_head_embedding.embedding.weight", p + "value_proj.weight"}
    missing = sorted(required - weights.keys())
    if missing:
        raise ValueError(f"Hy4-flash tensors are incomplete: {missing[0]}")
    dfly_layers = int(config.get("dfly_num_hidden_layers") or 0)
    has_dfly = dfly_layers > 0 and all(
        f"model.dfly_layers.{i}.self_attn.kv_b_proj.weight" in weights for i in range(dfly_layers)
    ) and "model.dfly_layers.mtp_fusion.eh_proj.weight" in weights
    _fingerprint_engram_formats(artifact)
    # ``has_mtp``: the embedded DFly2 head serves AST's native-head drafter.
    vision = config.get("vision_config")
    has_vision = False
    if isinstance(vision, dict) and config.get("image_token_id") is not None:
        vit = {k for k in weights if k.startswith("vit.")}
        expected = vision_tensor_names(int(vision.get("num_hidden_layers") or 0))
        # Every tower tensor or none: a partial tower (e.g. a conversion
        # resumed across the converter change) refuses rather than loading
        # text-only or failing at the strict load.
        if vit and vit != expected:
            missing = sorted(expected - vit) or sorted(vit - expected)
            raise ValueError(f"Hy4-flash vision tensors are incomplete or unexpected: {missing[0]}")
        has_vision = bool(vit) and bool(expected)
    artifact.update(has_dfly=has_dfly, has_mtp=has_dfly, dfly_layers=dfly_layers,
                    quantized=quant is not None, has_vision=has_vision)
    return artifact


def _fingerprint_engram_formats(artifact: dict) -> None:
    """Check and fold the converter's extra Engram shards into the identity.

    ``config["mlx2_engram_formats"]`` lists shards outside the weight map
    (``fp8_e4m3_row`` / ``affine4_g32`` tables); rewriting one must change
    the artifact fingerprint, and no listed path may leave the artifact.
    """
    import hashlib

    from .artifact_paths import shard_within_artifact

    formats = artifact["config"].get("mlx2_engram_formats")
    if formats is None:
        return
    if not isinstance(formats, dict) or set(formats) - set(ENGRAM_TABLE_FORMATS):
        raise ValueError("Hy4-flash mlx2_engram_formats lists an unknown table format")
    root = Path(artifact["identity"]["path"])
    known = {name for name, *_ in artifact["identity"]["files"]}
    digest = hashlib.sha256(artifact["identity"]["fingerprint"].encode())
    records = []
    for fmt in sorted(formats):
        shards = formats[fmt]
        if not isinstance(shards, dict):
            raise ValueError(f"Hy4-flash mlx2_engram_formats[{fmt!r}] must map tensors to shards")
        for name in sorted(set(shards.values())):
            if (not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
                    or not shard_within_artifact(root, (root / name).resolve())
                    or not (root / name).is_file()):
                raise ValueError(f"missing or escaped Engram shard: {name}")
            if name in known:
                continue
            known.add(name)
            stat = (root / name).stat()
            record = (name, stat.st_size, stat.st_mtime_ns)
            records.append(record)
            digest.update(json.dumps(record).encode())
    if records:
        artifact["identity"] = dict(artifact["identity"], fingerprint=digest.hexdigest(),
                                    files=list(artifact["identity"]["files"]) + records)


def engram_format_record(fmt: str, precision: str = "transformers") -> dict:
    """Status label of the selected Engram table format (never qualified)."""
    exact = {"bf16": "source", "fp8_e4m3_row": "exact vs the bf16 table (transformers profile, "
             "CPU-verified; Metal parity unverified)", "affine4_g32": "approximate"}[fmt]
    approximate = fmt == "affine4_g32"
    if fmt == "fp8_e4m3_row" and precision == "vllm":
        # vLLM reads a bf16 table without the round trip; an FP8 table
        # carries it, so this pairing is not the vllm profile's bf16 result.
        exact, approximate = "differs from the vllm profile's bf16-table result (carries the round trip)", True
    return {"format": fmt, "default": fmt == "bf16", "implemented": True, "qualified": False,
            "approximate": approximate, "relation": exact}


def vision_tensor_names(layers: int) -> set:
    """Checkpoint names of the ``vit.*`` tower (``vision_config.num_hidden_layers``)."""
    if layers <= 0:
        return set()
    names = {"vit.embeddings.patch_embedding.weight", "vit.embeddings.patch_embedding.bias",
             "vit.embeddings.position_embedding.weight", "vit.perceive.image_newline",
             "vit.perceive.norm.weight", "vit.perceive.proj.0.weight", "vit.perceive.proj.0.bias",
             "vit.perceive.proj.2.weight", "vit.perceive.proj.2.bias"}
    for i in range(layers):
        p = f"vit.layers.{i}."
        for stem in ("input_layernorm", "post_attention_layernorm", "self_attn.q_proj",
                     "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"):
            names |= {p + stem + ".weight", p + stem + ".bias"}
    return names


def configure_environment() -> dict[str, str]:
    require_process_numerics("the Hy4-flash profile")
    profile = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               **PROCESS_NUMERICS, "MLX_LM_COMPILED_DECODE": "0"}
    clear_inherited_profile(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_"))
    os.environ.update(profile)
    assert_process_profile_applied("the Hy4-flash profile")
    return profile


def resident_bytes(root, weight_map: dict, skipped: set, keep_dfly: bool,
                   keep_vision: bool = False) -> int:
    """Bytes the loader will materialize: every tensor but tables/(vision)/(DFly2)."""
    from ..runtime.models.hy_v4_flash_engram import safetensors_header

    total = 0
    for shard in sorted(set(weight_map.values())):
        for name, entry in safetensors_header(Path(root) / shard).items():
            if not isinstance(entry, dict) or name == "__metadata__" or name in skipped:
                continue
            if not keep_vision and name.startswith("vit."):
                continue
            if not keep_dfly and name.startswith("model.dfly_layers."):
                continue
            begin, end = entry["data_offsets"]
            total += end - begin
    return total


MODEL_KEYS = frozenset({"engram_cache_rows", "engram_precision", "engram_table_format",
                        "latent_kv", "vision"})
# Engram table formats (hy_v4_flash_engram.TABLE_FORMATS); bf16 is the
# default, the others are opt-in, unqualified, and need converter shards.
ENGRAM_TABLE_FORMATS = ("bf16", "fp8_e4m3_row", "affine4_g32")
AST_KEYS = frozenset({"draft_model", "num_draft", "ast_topology", "ast_tiles", "ast_tile_rows",
                      "ast_multilane_tile_rows", "ast_adaptive_rows", "batch_size_route",
                      "pairwise_selection", "cost_per_commit_handoff", "ast_graft_prompt_lookup",
                      "ast_target_executor"})
# AST tree tiles admit one to four concurrent requests (lanes) per round.
AST_MAX_WIDTH = 4
AST_TARGET_EXECUTORS = ("packed", "serial")
PROMPT_LOOKUP_NAMES = frozenset({"prompt_lookup", "pld"})


def _parse_policy(execution_policy, model_path=None) -> tuple[dict, dict, str | None]:
    """``(model policy, external/AST policy, drafter kind)``.

    ``draft_model``: ``"embedded_dfly"``, ``"mtp"`` or the artifact's own path
    select the embedded DFly2 head (``server --ast`` passes the model path
    for its native-head drafter); ``"prompt_lookup"``/``"pld"`` the model-free
    prompt-lookup drafter.  Either runs on the external-draft engine with the
    chain-packed AST executor.
    """
    policy = dict(execution_policy or {})
    unknown = set(policy) - MODEL_KEYS - AST_KEYS
    if unknown:
        raise ValueError(f"Hy4-flash execution policy keys are not declared: {sorted(unknown)}")
    if policy.get("engram_precision", "transformers") not in ("transformers", "vllm"):
        raise ValueError("Hy4-flash engram_precision must be 'transformers' (default) or 'vllm'")
    if policy.get("engram_table_format", "bf16") not in ENGRAM_TABLE_FORMATS:
        raise ValueError(f"Hy4-flash engram_table_format must be one of {ENGRAM_TABLE_FORMATS} (default bf16)")
    if policy.get("latent_kv", "native") not in LATENT_KV_FORMATS:
        raise ValueError("Hy4-flash latent_kv must be 'native' (default) or 'fp8'")
    if type(policy.get("vision", True)) is not bool:
        raise ValueError("Hy4-flash vision must be a boolean (default true: load the tower if present)")
    model = {k: v for k, v in policy.items() if k in MODEL_KEYS}
    external = {k: v for k, v in policy.items() if k in AST_KEYS}
    if not external:
        return model, {}, None
    draft = external.get("draft_model")
    if not isinstance(draft, str):
        raise ValueError("Hy4-flash AST policy needs a draft_model")
    if draft in PROMPT_LOOKUP_NAMES:
        kind = "prompt_lookup"
    elif draft in (DFLY_POLICY, "mtp") or (
            model_path is not None
            and Path(draft).expanduser().resolve() == Path(model_path).expanduser().resolve()):
        kind = "dfly"
        num = external.setdefault("num_draft", 5)
        if type(num) is not int or not 1 <= num <= 15:
            raise ValueError("Hy4-flash num_draft must be an integer in 1..15")
        # DFly2 drafts one block; the chain is its trained shape.
        external.setdefault("ast_topology", "chain")
    else:
        raise ValueError("Hy4-flash draft_model must be embedded_dfly, mtp, the model path, or prompt_lookup")
    graft = external.get("ast_graft_prompt_lookup", False)
    if type(graft) is not bool:
        raise ValueError("Hy4-flash ast_graft_prompt_lookup must be a boolean")
    if graft and kind == "prompt_lookup":
        raise ValueError("ast_graft_prompt_lookup cannot graft prompt lookup onto itself")
    # ``packed`` (default) runs a wave's chains, across lanes, as one joint
    # forward; ``serial`` is the one-forward-per-chain reference.
    executor = external.pop("ast_target_executor", "packed")
    if executor not in AST_TARGET_EXECUTORS:
        raise ValueError("Hy4-flash ast_target_executor must be 'packed' (default) or 'serial'")
    external.setdefault("ast_tile_rows", 8)
    from .qwen38_27b import _native_mtp_tree_policy

    external = _native_mtp_tree_policy(external)
    external["ast_target_executor"] = executor
    return model, external, kind


def load_model(path, artifact, *, keep_dfly=False, engram_cache_rows=1 << 16, memory_budget=None,
               engram_precision="transformers", engram_table_format="bf16", latent_kv="native",
               vision=True):
    """``(model, drafter or None, engram readers)`` for an inspected artifact.

    Engram tables stay on disk; the DFly2 layers are read only with
    ``keep_dfly``; the ``vit.*`` tower is read and attached (checkpoint dtype, never
    quantized) when ``vision`` and the artifact carries it, else dropped
    before any byte is read.
    """
    import mlx.core as mx
    from mlx import nn

    from ..runtime.models.hy_v4_flash import HY4FlashDFly, Model, ModelArgs
    from ..runtime.models.hy_v4_flash_engram import open_engram_tables
    from ..runtime.ubc_evict import load_shards_evicting

    path = Path(path)
    config = artifact["config"]
    args = ModelArgs.from_hf_config(config)
    args.engram_precision = engram_precision
    if latent_kv not in LATENT_KV_FORMATS:
        raise ValueError("Hy4-flash latent_kv must be 'native' (default) or 'fp8'")
    args.latent_kv = latent_kv
    model = Model(args)
    tables = {f"model.layers.{i}.engram_layer.multi_head_embedding.embedding.weight"
              for i in args.engram_layers}

    keep_vision = bool(vision) and bool(artifact.get("has_vision"))

    def shard_filter(shard):
        return {k: v for k, v in shard.items()
                if k not in tables and (keep_vision or not k.startswith("vit."))
                and (keep_dfly or not k.startswith("model.dfly_layers."))}

    files = sorted(set(artifact["weight_map"].values()))
    resident = resident_bytes(path, artifact["weight_map"], tables, keep_dfly, keep_vision)
    if memory_budget is None:
        memory_budget = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") * 0.8)
    if resident > memory_budget:
        raise ValueError(
            f"Hy4-flash resident weights need {resident / 2**30:.1f} GiB (Engram on SSD), "
            "over 80% of this host's memory; convert the artifact to MLX affine "
            "quantization first (scripts/convert_hy_v4_flash_mlx.py)")
    shards = [path / name for name in files
              if any(k not in tables for k, v in artifact["weight_map"].items() if v == name)]
    raw = load_shards_evicting(shards, sanitize=shard_filter)
    dfly_raw = {k: raw.pop(k) for k in list(raw) if k.startswith("model.dfly_layers.")}
    vit_raw = {k: raw.pop(k) for k in list(raw) if k.startswith("vit.")}
    weights = model.sanitize(raw)
    raw.clear()
    quant = config.get("quantization")
    if quant is not None:
        def predicate(name, module):
            override = quant.get(name)
            if isinstance(override, dict):
                return override
            return hasattr(module, "to_quantized") and f"{name}.scales" in weights

        nn.quantize(model, group_size=quant["group_size"], bits=quant["bits"],
                    mode=quant.get("mode", "affine"), class_predicate=predicate)
    model.load_weights(list(weights.items()), strict=True)
    weights.clear()
    if keep_vision:
        from ..runtime.models.hy_v4_flash_vision import load_vision_tower

        # Attached after the trunk's quantization and strict load: the tower
        # stays in the checkpoint dtype.
        model.attach_vision(load_vision_tower(config, vit_raw))
    vit_raw.clear()
    readers = open_engram_tables(path, artifact["weight_map"], args.engram_layers,
                                 cache_rows=engram_cache_rows, table_format=engram_table_format,
                                 formats=config.get("mlx2_engram_formats"))
    try:
        model.bind_engram(readers)
        drafter = None
        if keep_dfly:
            intermediate, correction = HY4FlashDFly.geometry(dfly_raw)
            drafter = HY4FlashDFly(args, intermediate=intermediate, correction=correction)
            drafter.load_weights(list(drafter.sanitize(dfly_raw).items()), strict=True)
        dfly_raw.clear()
        model.eval()
        mx.eval(model.parameters())
        if drafter is not None:
            mx.eval(drafter.parameters())
        mx.clear_cache()
    except BaseException:
        for reader in readers.values():
            reader.close()
        raise
    return model, drafter, readers


class HYV4FlashAdapter(ExternalDraftAdapterMixin, OrdinaryTextAdapter):
    descriptor = descriptor_for()
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "hy-v4-flash-ordinary"
    # AST tree tiles: chain-packed exact verify (hy_v4_flash_ast); the
    # embedded DFly2 head fills the native-head drafter slot.
    ast_tree_tiles = True
    ast_tree_tile_dtype_gate = False
    prompt_lookup_default = False
    EXTERNAL_PROFILE = "hy-v4-flash-ast-b1-b4-v1"
    EXTERNAL_ROUTE_TAG = "hy4-flash-chain-packed-tree-v1"
    EXTERNAL_DEFAULT_NUM_DRAFT = 5
    # Media prefill and decode run on the ordinary route only; the
    # external-draft (AST/DFly2/prompt-lookup) routes refuse image requests.
    multimodal_routes = frozenset({"ordinary"})

    def __init__(self, model_path: str, *, execution_policy=None, weight_streaming=None):
        from .process_globals import guarded_construction

        try:
            guarded_construction(self, lambda: self._init_hy_v4_flash(
                model_path, execution_policy=execution_policy, weight_streaming=weight_streaming))
        except BaseException:
            self.close()
            raise

    def _init_hy_v4_flash(self, model_path, *, execution_policy=None, weight_streaming=None):
        if weight_streaming is not None:
            raise ValueError("Hy4-flash declares no weight streaming mode")
        artifact = inspect_artifact(model_path)
        policy, external, kind = _parse_policy(execution_policy, artifact["identity"]["path"])
        self.identity = artifact["identity"]
        # fp8 rows get their own layout name: APCv2 namespaces never mix
        # approximate state with the exact layout (and APCv2 refuses to
        # store non-exact-prefix caches in any case).
        self.layout = FP8_CACHE_LAYOUT if policy.get("latent_kv") == "fp8" else CACHE_LAYOUT
        self.engram_tables = {}
        self.dfly = None
        self.draft_model = None
        self.tree_executor = None
        self.external_policy = external
        self.drafter_kind = kind
        self.dfly_selected = kind == "dfly"
        self.num_draft = int(external.get("num_draft", 5))
        if self.dfly_selected and not artifact["has_dfly"]:
            raise ValueError("Hy4-flash artifact carries no DFly2 drafter")
        engram_format = policy.get("engram_table_format", "bf16")
        self.latent_kv = policy.get("latent_kv", "native")
        self.descriptor = descriptor_for(dfly_selected=self.dfly_selected,
                                         engram_table_format=engram_format, latent_kv=self.latent_kv)
        if self.latent_kv == "fp8":
            self.layout = FP8_CACHE_LAYOUT
        elif self.dfly_selected:
            self.layout = DFLY_CACHE_LAYOUT
        self.layout = numerics_layout(self.layout, engram_table_format=engram_format,
                                      engram_precision=policy.get("engram_precision", "transformers"))
        self.environment = configure_environment()
        path = Path(self.identity["path"])
        config = artifact["config"]
        from transformers import AutoTokenizer

        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper

        self.model, self.dfly, self.engram_tables = load_model(
            path, artifact, keep_dfly=self.dfly_selected,
            engram_cache_rows=int(policy.get("engram_cache_rows", 1 << 16)),
            engram_precision=policy.get("engram_precision", "transformers"),
            engram_table_format=policy.get("engram_table_format", "bf16"),
            latent_kv=self.latent_kv, vision=policy.get("vision", True))
        self.engram_table_format = policy.get("engram_table_format", "bf16")
        from ..runtime.models.hy_v4_flash_fp8kv import latent_fp8_stats

        self._fp8_baseline = latent_fp8_stats()["rows_packed"]
        args = self.model.args
        self.media_token_ids = {k: config.get(k) for k in (
            "image_token_id", "image_start_token_id", "image_end_token_id", "video_token_id")}
        self.descriptor = descriptor_for(dfly_selected=self.dfly_selected,
                                         engram_table_format=engram_format, latent_kv=self.latent_kv,
                                         vision=self.model.has_vision)
        self._sync_descriptor_layout()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer

        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        from ..runtime.tool_parsers.hy_v4 import detect_suffix

        template = path / "chat_template.jinja"
        self.tool_suffix = detect_suffix(
            tokenizer.get_vocab(), template.read_text() if template.is_file() else None)
        from .hy_v4_flash_output import FeatureCounters

        self.feature_stats = FeatureCounters()
        eos = config.get("text_config", {}).get("eos_token_id", 120025)
        self.eos_ids = [int(e) for e in (eos if isinstance(eos, list) else [eos])]
        self.tokenizer = TokenizerWrapper(tokenizer, detokenizer_class=BPEStreamingDetokenizer,
                                          eos_token_ids=self.eos_ids)
        self.max_context = int(args.max_position_embeddings)
        self.dfly_stats = {"requests": 0, "verify_rounds": 0, "verify_accept_hist": [0] * (self.num_draft + 1)}
        if kind is not None:
            self._bind_drafter(kind)
            if external.get("ast_graft_prompt_lookup", False):
                from .qwen38_27b import graft_prompt_lookup_drafter

                # Prompt-lookup copies join the DFly2 tile (copy path first
                # when a specific n-gram matched).
                graft_prompt_lookup_drafter(self, prompt_lookup=False)

    def _bind_drafter(self, kind):
        from .qwen38_27b import _prompt_lookup_record

        if kind == "dfly":
            from ..runtime.drafters.hy4_dfly import dfly_record, loader_for

            # DFly2 reads the indexer selection of each committed position.
            self.model.record_layers = frozenset(self.dfly.index_layers)
            record = dfly_record(self.num_draft, self.identity["fingerprint"])
            loader = loader_for(self.dfly)
        else:
            from ..runtime.drafters.prompt_lookup_draft import load_prompt_lookup_drafter

            record = _prompt_lookup_record(self.external_policy, self.identity["fingerprint"])
            loader = load_prompt_lookup_drafter
        self._bind_external_drafter(record, loader, self.descriptor)

    def create_external_batch(self, **kwargs):
        from .hy_v4_flash_ast import Hy4ChainTreeExecutor
        from .qwen38_27b import _tile_node_budgets

        # Tree tiles admit one to four requests per round (node budgets per
        # width from the tile knobs); a server lane count above four is
        # clamped rather than refused.
        lanes = kwargs.get("completion_batch_size")
        if lanes is not None and (type(lanes) is not int or lanes < 1):
            raise ValueError("completion_batch_size must be a positive integer")
        width = min(AST_MAX_WIDTH, lanes or 1)
        kwargs["completion_batch_size"] = width
        kwargs.setdefault("dynamic_singleton_tree", True)
        # The engine's bounded candidate width is 4; the lane capacity
        # (``completion_batch_size``) bounds each cohort.
        kwargs.setdefault("dynamic_tree_max_width", AST_MAX_WIDTH)
        if "target_tree_executor_provider" not in kwargs:
            kwargs["target_tree_executor_provider"] = Hy4ChainTreeExecutor.create(
                joint=self.external_policy.get("ast_target_executor", "packed") == "packed")
        self.tree_executor = kwargs["target_tree_executor_provider"]
        kwargs.setdefault("tree_node_budget_by_lanes",
                          _tile_node_budgets(self.external_policy, AST_MAX_WIDTH))
        for key in ("ast_topology", "ast_adaptive_rows"):
            if key in self.external_policy:
                kwargs.setdefault(key, self.external_policy[key])
        return super().create_external_batch(**kwargs)

    # Hy4's template takes ``reasoning_effort`` (no_think/high/max, default
    # high).  mlx2 renders no_think unless the request opts in
    # (``hy_v4_flash_output.reasoning_effort``): high mode can spend the whole
    # token budget reasoning (vllm#66454).
    reasoning_effort_semantics = "template_levels_no_think_high_max"

    def _template(self, request, *, tokenize):
        from ..output import callable_tools
        from .hy_v4_flash_output import reasoning_effort

        return self.tokenizer.apply_chat_template(
            _template_history(request["messages"]), add_generation_prompt=True, tokenize=tokenize,
            reasoning_effort=reasoning_effort(request), tools=callable_tools(request))

    @staticmethod
    def thinking_enabled(request):
        from .hy_v4_flash_output import thinking_enabled

        return thinking_enabled(request)

    def thinking_close_token_ids(self):
        """``</think:S>`` as one token, or None (structured output with
        reasoning on then stays refused)."""
        marker = f"</think{getattr(self, 'tool_suffix', '')}>"
        try:
            ids = list(self.tokenizer.encode(marker, add_special_tokens=False))
            if len(ids) != 1 or self.tokenizer.decode(ids) != marker:
                return None
        except Exception:  # noqa: BLE001 - undeclared marker, not a load failure
            return None
        return (int(ids[0]),)

    def structured_special_token_ids(self):
        """The call markers and ``</think:S>`` where the tokenizer keeps them
        as special tokens: the structured mask admits a special token only
        when the adapter declares it, and the forced grammar spells them."""
        from ..runtime.tool_parsers.hy_v4 import markers

        suffix = getattr(self, "tool_suffix", "")
        ids = []
        for marker in (*markers(suffix).values(), f"</think{suffix}>"):
            try:
                encoded = list(self.tokenizer.encode(marker, add_special_tokens=False))
                if len(encoded) == 1 and self.tokenizer.decode(encoded) == marker:
                    ids.append(int(encoded[0]))
            except Exception:  # noqa: BLE001 - an undeclared marker stays masked
                continue
        return tuple(ids)

    def prompt_tokens(self, request):
        if "_mlx2_prompt_tokens" in request:
            return list(request["_mlx2_prompt_tokens"])
        if "messages" in request:
            return self._template(request, tokenize=True)
        return self.tokenizer.encode(request["prompt"], add_special_tokens=False)

    def render_prompt(self, request):
        if "_mlx2_prompt_tokens" in request:
            raise ValueError("prepared Hy4-flash media prompt has no plain text rendering")
        if "messages" in request:
            return self._template(request, tokenize=False)
        return request["prompt"]

    # -- image input -------------------------------------------------------
    def prepare_multimodal_request(self, request, *, file_loader=None):
        """Decode, preprocess and frame a request's images (ordinary route).

        Text-only requests are returned unchanged.  Video parts, a missing
        tower and the external-draft routes fail closed.
        """
        from ..api_resources import CapabilityUnavailable
        from ..multimodal import media_fingerprint, resolve_media
        from .hy_v4_flash_media import (PROCESSOR_REVISION, expand_image_placeholders,
                                        image_token_count, preprocess_image)
        from .mlx_vlm import _source

        typed, media, pixels, grids = [], [], [], []
        for message in request.get("messages", ()):
            content = message.get("content")
            if not isinstance(content, list):
                typed.append(message)
                continue
            parts = []
            for part in content:
                kind = part.get("type")
                if kind == "text":
                    parts.append({"type": "text", "text": str(part.get("text", ""))})
                elif kind in ("image_url", "input_image"):
                    parts.append({"type": "image"})
                    media.append(_source(part, "image"))
                elif kind in ("input_video", "video_url", "video"):
                    raise CapabilityUnavailable("Hy4-flash video input is not implemented")
                else:
                    raise ValueError(f"Hy4-flash does not accept {kind!r} content parts")
            typed.append({**message, "content": parts})
        if not media:
            return request
        if not self.model.has_vision:
            raise CapabilityUnavailable("this Hy4-flash artifact has no loaded vision tower")
        if getattr(self, "draft_model", None) is not None:
            raise CapabilityUnavailable("Hy4-flash image input runs on the ordinary route only")
        tower = self.model.vision_tower.args
        min_pixels, max_pixels = self.image_pixel_bounds
        values = []
        for source in media:
            value = resolve_media(source, kind="image", file_loader=file_loader)
            values.append(value)
            patches, grid = preprocess_image(value.value, patch_size=tower.patch_size,
                                             merge_size=tower.spatial_merge_size,
                                             min_pixels=min_pixels, max_pixels=max_pixels)
            pixels.append(patches)
            grids.append(grid)
        prepared = {**request, "messages": typed}
        ids = list(self._template(prepared, tokenize=True))
        counts = [image_token_count(g, tower.spatial_merge_size) for g in grids]
        t = self.media_token_ids
        ids, media_end = expand_image_placeholders(
            ids, counts, image_token_id=t["image_token_id"],
            start_token_id=t["image_start_token_id"], end_token_id=t["image_end_token_id"])
        if media_end >= len(ids):
            raise ValueError("Hy4-flash image placeholder cannot be the final prompt token")
        import mlx.core as mx
        import numpy as np

        policy = {"processor": PROCESSOR_REVISION, "patch_size": tower.patch_size,
                  "merge_size": tower.spatial_merge_size, "min_pixels": min_pixels,
                  "max_pixels": max_pixels}
        return {
            **prepared,
            "_mlx2_prompt_tokens": ids,
            "_mlx2_prefill_inputs": {
                "pixel_values": mx.array(np.concatenate(pixels, axis=0)),
                "image_grid_thw": tuple(tuple(int(v) for v in g) for g in grids),
            },
            "_mlx2_media_token_end": media_end,
            "_mlx2_media_fingerprint": media_fingerprint(values, policy=policy),
        }

    @property
    def image_pixel_bounds(self) -> tuple[int, int]:
        """``(min_pixels, max_pixels)``: the artifact's preprocessor config, else
        the vision config's ``min/max_image_size`` squared (the reference
        processor's 512*512 / 2048*2048 defaults for tencent/Hy4-flash)."""
        tower = self.model.vision_tower.args
        bounds = (tower.min_image_size ** 2, tower.max_image_size ** 2)
        path = Path(self.identity["path"]) / "preprocessor_config.json"
        if path.is_file():
            data = json.loads(path.read_text())
            size = data.get("size") or {}
            bounds = (int(data.get("min_pixels", size.get("shortest_edge", bounds[0]))),
                      int(data.get("max_pixels", size.get("longest_edge", bounds[1]))))
        return bounds

    def validate_prefill_inputs(self, request, remaining_tokens, prefill_input):
        """Refuse a media payload whose grids disagree with the prompt's image tokens."""
        if prefill_input is None:
            return None
        from .hy_v4_flash_media import image_token_count

        if set(prefill_input) != {"pixel_values", "image_grid_thw"}:
            raise ValueError("Hy4-flash media prefill takes pixel_values and image_grid_thw only")
        merge = self.model.vision_tower.args.spatial_merge_size
        grids = prefill_input["image_grid_thw"]
        expected = sum(image_token_count(g, merge) for g in grids)
        image_id = self.media_token_ids["image_token_id"]
        if sum(1 for tok in remaining_tokens if int(tok) == image_id) != expected:
            raise ValueError("Hy4-flash image tokens disagree with the media payload")
        return prefill_input

    def output_parser(self, request):
        from .hy_v4_flash_output import output_parser

        return output_parser(request, suffix=self.tool_suffix,
                             stats=getattr(self, "feature_stats", None))

    def tool_constraint(self, request):
        from .hy_v4_flash_output import tool_constraint

        return tool_constraint(request, suffix=self.tool_suffix,
                               stats=getattr(self, "feature_stats", None))

    def execution_config(self, *, max_lanes, prefill_step):
        if self.draft_model is not None:
            # AST tiles: one to four lanes whatever the server's lane count.
            return self._external_execution_config(max_lanes=max(1, min(AST_MAX_WIDTH, max_lanes)),
                                                   prefill_step=prefill_step)
        return super().execution_config(max_lanes=max_lanes, prefill_step=prefill_step)

    def generate_dfly(self, prompt_tokens, *, max_tokens, prefill_step=512):
        """Exact greedy DFly2 decode (opt-in route); returns ``(tokens, receipt)``."""
        if self.dfly is None:
            raise ValueError("DFly2 is not selected; use execution policy "
                             f'{{"draft_model": "{DFLY_POLICY}"}}')

        from ..runtime.models.hy_v4_flash import dfly_greedy_generate

        tokens, receipt = dfly_greedy_generate(
            self.model, self.dfly, prompt_tokens, max_tokens=max_tokens,
            num_draft=self.num_draft, eos_ids=self.eos_ids, prefill_step=prefill_step)
        stats = self.dfly_stats
        stats["requests"] += 1
        stats["verify_rounds"] += receipt["verify_rounds"]
        for i, n in enumerate(receipt["verify_accept_hist"]):
            stats["verify_accept_hist"][i] += n
        receipt["checkpoint"] = f"{Path(self.identity['path']).name}@{self.identity['fingerprint']}"
        receipt["latent_kv_state"] = self._latent_kv_receipt()
        return tokens, receipt

    def _sync_descriptor_layout(self):
        """The descriptor names the same cache layout APCv2 uses (``self.layout``)."""
        from dataclasses import replace

        self.descriptor = replace(self.descriptor, cache_layout=self.layout)

    def _latent_kv_receipt(self):
        from ..runtime.models.hy_v4_flash_fp8kv import latent_fp8_stats

        fmt = getattr(self, "latent_kv", "native")
        # Process-wide counter: "observed" means fp8 rows were packed since
        # this adapter loaded (only fp8-selected adapters pack rows).
        packed = latent_fp8_stats()["rows_packed"] - getattr(self, "_fp8_baseline", 0)
        return latent_kv_receipt(fmt, observed_used=packed > 0, layout=getattr(self, "layout", None))

    def diagnostics(self):
        draft = getattr(self, "draft_model", None)
        grafted = getattr(draft, "base", None) is not None
        head = draft.base if grafted else draft
        stats = getattr(head, "stats", {}) if head is not None else {}
        engine_used = bool(stats.get("block_drafts")) and getattr(self, "drafter_kind", None) == "dfly"
        dfly = {
            "implemented": True,
            "qualified": False,
            "selected": bool(getattr(self, "dfly_selected", False)),
            "observed_used": bool(getattr(self, "dfly_stats", {}).get("requests")) or engine_used,
            "serving_engine": "external-draft engine, chain-packed AST executor (requires_target_caches)",
        }
        if getattr(self, "dfly_stats", None):
            dfly["stats"] = dict(self.dfly_stats)
        counters = getattr(self, "feature_stats", None)
        stats_ = counters.snapshot() if counters is not None else {}
        if engine_used:
            dfly["engine_stats"] = dict(stats)
        return {
            "route": "external_draft" if draft is not None else "ordinary",
            "drafter": getattr(self, "drafter_kind", None),
            "qualification": "pending",
            "checkpoint": f"{Path(self.identity['path']).name}@{self.identity['fingerprint']}"
            if getattr(self, "identity", None) else None,
            "continuous_batch": {"implemented": True, "qualified": False},
            "ast": {
                "implemented": True, "qualified": False, "selected": draft is not None,
                "max_width": AST_MAX_WIDTH,
                "target_executor": getattr(self, "external_policy", {}).get("ast_target_executor"),
                "executor": self.tree_executor.receipt()
                if getattr(self, "tree_executor", None) is not None else None,
                "graft_pld": {
                    "implemented": True, "qualified": False, "selected": grafted,
                    # Used only once prompt lookup actually found a copy path.
                    "observed_used": grafted and bool(draft.stats.get("copy_paths")),
                    **({"stats": dict(draft.stats)} if grafted else {}),
                },
            },
            "dfly2": dfly,
            "tool_choice_forced": {
                "implemented": True, "qualified": False,
                # The adapter builds the grammar; whether serving engaged it
                # is the request's tool_grammar receipt, not observed here.
                # Hook calls, not requests (precompile + admission, and the
                # auto-shape planner).
                "grammar_hook_calls": stats_.get("tool_grammar_hook_calls", 0),
                "observed_used": False,
                "note": "engagement is reported by the serving tool_grammar receipt",
            },
            "reasoning": {
                "implemented": True, "qualified": False, "default": "no_think",
                "selected": bool(stats_.get("reasoning_requests")),
                "observed_used": bool(stats_.get("reasoning_emitted")),
                "requests": stats_.get("reasoning_requests", 0),
            },
            "engram": {i: r.diagnostics() for i, r in getattr(self, "engram_tables", {}).items()},
            "engram_precision": getattr(getattr(getattr(self, "model", None), "args", None),
                                        "engram_precision", None),
            "engram_table_format": self._engram_format_record(),
            "latent_kv": self._latent_kv_receipt(),
            "approximate_state": (getattr(self, "latent_kv", "native") == "fp8"
                                  or bool((self._engram_format_record() or {}).get("approximate"))),
            "vision": {
                "implemented": True,
                "qualified": False,
                "loaded": bool(getattr(getattr(self, "model", None), "has_vision", False)),
                "modalities": ("image",),
                "video": "refused (not implemented)",
                "routes": tuple(sorted(type(self).multimodal_routes)),
                "observed_used": bool(getattr(getattr(self, "model", None), "vision_stats",
                                              {}).get("prefills")),
                "stats": dict(getattr(getattr(self, "model", None), "vision_stats", {})),
            },
        }

    def _engram_format_record(self):
        fmt = getattr(self, "engram_table_format", None)
        if fmt is None:
            return None
        precision = getattr(getattr(getattr(self, "model", None), "args", None), "engram_precision", "transformers")
        return {**engram_format_record(fmt, precision), "selected": True,
                "observed_used": any(r.stats.get("lookups") for r in getattr(self, "engram_tables", {}).values())}

    def close(self):
        for reader in getattr(self, "engram_tables", {}).values():
            reader.close()
        self.engram_tables = {}
        super().close()
