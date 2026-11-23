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

# v2 (2026-11-20): the Engram norms became torch.nn.RMSNorm(eps=None) (fp32
# epsilon, weight in fp32), so v1 entries were computed with other numerics.
CACHE_LAYOUT = "hy-v4-flash-mla-latent-indexer-v2"
# DFly2 routes record indexer selections in the target cache (Index Share);
# their APCv2 entries carry that plane, so they live in their own namespace.
DFLY_CACHE_LAYOUT = CACHE_LAYOUT + "+dfly2-index-share"
# Distinct layout for the approximate fp8 latent rows (never the exact one).
FP8_CACHE_LAYOUT = "hy-v4-flash-mla-latent-fp8-ds-mla-indexer-v2"
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
                    engram_precision: str = "transformers",
                    tensorfold_rows: str | None = None, moe_fused_gate_up: bool = False,
                    moe_nax: str = "off", moe_top_k: int | None = None) -> str:
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
    if tensorfold_rows is not None:
        # The TensorFold row law (``ast_tensorfold_rows``) computes the trunk
        # projections with other arithmetic than stock QMV, so its cache rows
        # are exact for that law only; each backend gets its own namespace.
        layout = f"{layout}+tensorfold-rows-{tensorfold_rows}-v1"
    # The opt-in MoE kernel routes are bit-identical on CPU (fused gate/up,
    # tests) and canary-proven per kernel instantiation (NAX), but are
    # unmeasured on the GPU: until a GPU A/B shows exact parity their APCv2
    # entries stay in their own namespace.
    if moe_fused_gate_up:
        layout = f"{layout}+moe-fused-gate-up"
    if moe_nax != "off":
        layout = f"{layout}+moe-nax-{moe_nax}"
    if moe_top_k is not None:
        # Fewer routed experts per token: other numerics, other cache rows.
        layout = f"{layout}+moe-top{int(moe_top_k)}"
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


# Label of the image-processor source when a vision artifact ships neither
# processor_config.json nor preprocessor_config.json.
IMAGE_PROCESSOR_FALLBACK = "class defaults (artifact has no image-processor config)"


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
                  "vision": ("image: ordinary route, isolated B=1 media prefill; plain decode on "
                             "external_draft (DFly2/AST); video refused"
                             if vision else "not loaded"),
                  "dfly2": "selected; unqualified" if dfly_selected else "implemented; not selected",
                  "apc_v2": ("not declared (fp8 latent rows are approximate)" if fp8 else
                             "implemented; unqualified (exact published boundaries, "
                             "bounded-rollback landings refused)"),
                  "tools": "hy_v4 <tool_call:S> parser, auto/none; forced tool_choice "
                           "(required/named) under the Hy4 call grammar; implemented, unqualified",
                  "reasoning": "reasoning_effort no_think by default; high opt-in per request "
                               "(enable_thinking or reasoning_effort) to reasoning_content; "
                               "implemented, unqualified",
                  "grammar": "Hy4 forced-call grammar and generic structured output; "
                             "implemented, unqualified",
                  "engram_tables": "bf16 (default)" if engram_table_format == "bf16" else
                  f"{engram_table_format}; not the bf16 default; unqualified"
                  + ("; approximate" if engram_table_format in ("affine4_g32", "iq4_nl") else ""),
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
    # mlx-lm writes its MLX quantization twice (``quantization`` and
    # ``quantization_config``); any other quantization_config is an
    # FP8/compressed-tensors checkpoint.
    if config.get("quantization_config") is not None and config.get("quantization_config") != quant:
        raise ValueError("Hy4-flash FP8/compressed-tensors checkpoints are not implemented")
    if quant is not None:
        check_quantization(quant)
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
            if sanitized:
                required.add(p + "mlp.experts.gate_proj.weight")
            elif p + "mlp.switch_mlp.gate_proj.weight" in weights:
                # mlx-lm expert naming (unfused, ``switch_mlp``).
                required |= {p + f"mlp.switch_mlp.{r}_proj.weight" for r in ("gate", "up", "down")}
            else:
                required.add(p + "mlp.experts.gate_up_proj")
    gguf = text.get("engram_gguf_tables")
    for lid in text.get("engram_layer_ids") or []:
        p = f"model.layers.{int(lid) - 1}.engram_layer."
        required |= {p + "lookup_table", p + "value_proj.weight"}
        if gguf is None:
            required.add(p + "multi_head_embedding.embedding.weight")
    missing = sorted(required - weights.keys())
    if missing:
        raise ValueError(f"Hy4-flash tensors are incomplete: {missing[0]}")
    dfly_layers = int(config.get("dfly_num_hidden_layers") or 0)
    has_dfly = dfly_layers > 0 and all(
        f"model.dfly_layers.{i}.self_attn.kv_b_proj.weight" in weights for i in range(dfly_layers)
    ) and "model.dfly_layers.mtp_fusion.eh_proj.weight" in weights
    _fingerprint_engram_formats(artifact)
    _fingerprint_engram_gguf(artifact)
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
                    quantized=quant is not None, has_vision=has_vision,
                    engram_table_formats=available_engram_formats(config))
    return artifact


# Per-module quantization modes the loader runs: MLX affine, and the
# third-party ``q2_0`` expert mode (affine 2-bit with biases = -scales,
# ``hy_v4_flash.Q2ZeroSwitchLinear``) on routed expert banks only.
Q2_ZERO = "q2_0"


def check_quantization(quant) -> None:
    """Fail closed on a quantization block the loader cannot run."""
    if not isinstance(quant, dict) or quant.get("mode", "affine") != "affine":
        raise ValueError("Hy4-flash admits bf16 or MLX affine-quantized artifacts only")
    for name, entry in quant.items():
        if not isinstance(entry, dict):
            continue
        mode = entry.get("mode", "affine")
        if mode == Q2_ZERO:
            if (not name.startswith("model.layers.") or entry.get("bits") != 2
                    or not (".mlp.switch_mlp." in name or ".mlp.experts." in name)):
                raise ValueError(f"Hy4-flash q2_0 is implemented for 2-bit routed experts only: {name}")
        elif mode != "affine":
            raise ValueError(f"Hy4-flash {name} quantization mode {mode!r} is not implemented")


def module_quantization(quant: dict) -> dict:
    """Per-module overrides under mlx2 module names (``switch_mlp`` -> ``experts``)."""
    from ..runtime.models.hy_v4_flash import switch_mlp_name

    return {switch_mlp_name(k): v for k, v in quant.items() if isinstance(v, dict)}


def resolve_engram_table_format(policy: dict, available) -> tuple[str, str]:
    """``(format, source)`` of the Engram table format a load uses.

    An explicit ``engram_table_format`` wins (``"policy"``).  Otherwise the
    default stays ``bf16`` (``"default"``) whenever the artifact carries it;
    an artifact that carries exactly one other format (a third-party pack
    with only IQ4_NL GGUF tables) uses that one (``"artifact_only_format"``)
    instead of failing to load with the default.
    """
    if "engram_table_format" in policy:
        return policy["engram_table_format"], "policy"
    available = list(available)
    if "bf16" not in available and len(available) == 1:
        return available[0], "artifact_only_format"
    return "bf16", "default"


def available_engram_formats(config: dict) -> list:
    """Engram table formats this artifact carries (``engram_table_format`` choices)."""
    text = config.get("text_config") or {}
    formats = config.get("mlx2_engram_formats")
    out = sorted(formats) if isinstance(formats, dict) else []
    if not out and text.get("engram_gguf_tables") is None:
        out = ["bf16"]
    if text.get("engram_gguf_tables") is not None:
        out.append("iq4_nl")
    return out


def _fingerprint_engram_gguf(artifact: dict) -> None:
    """Check the ``engram_gguf_tables`` bindings and fold their files into the identity."""
    import hashlib

    from ..runtime.models.hy_v4_flash_engram import gguf_bindings, gguf_table_layout
    from .artifact_paths import shard_within_artifact

    config = artifact["config"]
    text = text_config(config)
    gguf = text.get("engram_gguf_tables")
    if gguf is None:
        return
    if config.get("mlx2_engram_formats") is not None:
        raise ValueError("Hy4-flash artifact declares both mlx2_engram_formats and engram_gguf_tables")
    layer_ids = {int(lid) - 1: int(lid) for lid in text.get("engram_layer_ids") or []}
    bindings = gguf_bindings(gguf, layer_ids)
    root = Path(artifact["identity"]["path"])
    known = {name for name, *_ in artifact["identity"]["files"]}
    digest = hashlib.sha256(artifact["identity"]["fingerprint"].encode())
    records = []
    for index in sorted(bindings):
        name, tensor = bindings[index]
        if not shard_within_artifact(root, (root / name).resolve()) or not (root / name).is_file():
            raise ValueError(f"missing or escaped Engram GGUF file: {name}")
        gguf_table_layout(root / name, tensor)  # header-only: type, shape, extent
        digest.update(json.dumps([index, name, tensor]).encode())
        if name in known:
            continue
        known.add(name)
        stat = (root / name).stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record)
        digest.update(json.dumps(record).encode())
    artifact["identity"] = dict(artifact["identity"], fingerprint=digest.hexdigest(),
                                files=list(artifact["identity"]["files"]) + records)


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
    if not isinstance(formats, dict) or set(formats) - set(CONVERTER_ENGRAM_FORMATS):
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
             "CPU-verified; Metal parity unverified)", "affine4_g32": "approximate",
             "iq4_nl": "approximate (third-party IQ4_NL GGUF table; E4M3 round trip at runtime)"}[fmt]
    approximate = fmt in ("affine4_g32", "iq4_nl")
    if fmt == "iq4_nl" and precision == "vllm":
        # The vllm profile reads rows without the E4M3 round trip.
        exact = "approximate (third-party IQ4_NL GGUF table; vllm profile: no E4M3 round trip)"
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


def widened_bytes(root, weight_map: dict, quant, *, lm_head_fp32: bool) -> int:
    """Bytes the load adds over the on-disk planes: q2_0 scales widened to fp32
    plus their fp32 biases, and a quantized LM head's fp32 scales/biases."""
    from ..runtime.models.hy_v4_flash_engram import safetensors_header

    widen = {}  # tensor name -> bytes per element after the load
    for name, entry in (quant or {}).items():
        if isinstance(entry, dict) and entry.get("mode") == Q2_ZERO:
            widen[name + ".scales"] = 8  # fp32 scales + fp32 biases
    if lm_head_fp32 and "lm_head.scales" in weight_map:
        widen["lm_head.scales"] = 4
        widen["lm_head.biases"] = 4
    extra, headers = 0, {}
    for name, per in widen.items():
        shard = weight_map.get(name)
        if shard is None:
            continue
        header = headers.setdefault(shard, safetensors_header(Path(root) / shard))
        entry = header.get(name)
        if not isinstance(entry, dict):
            continue
        begin, end = entry["data_offsets"]
        width = {"F16": 2, "BF16": 2, "F32": 4}.get(entry["dtype"], 4)
        extra += max(0, (end - begin) // width * per - (end - begin))
    return extra


MODEL_KEYS = frozenset({"engram_cache_rows", "engram_hash", "engram_precision",
                        "engram_table_format", "latent_kv", "vision",
                        "moe_fused_gate_up", "moe_nax", "moe_top_k"})
# Where the Engram n-grams are hashed.  ``host``: NumPy after a host copy of
# the token ids.  ``device``: MLX ops on the generation stream for forwards of
# at most ROLLBACK tokens, read back once with the token ids after the
# pre-Engram layers are queued; fails closed at load when the tables' geometry
# is outside the exact device range.  ``auto`` (DEFAULT, Pierre 2026-11-21):
# ``device`` when the geometry is exactly hashable, else ``host`` with the
# reason in the receipt.  Bit-identical rows on every mode, so outputs and the
# APCv2 layout are unchanged.  Unqualified.
# Measured 2026-11-21 (Hy4-flash-MLX-4.6bit-mlx2@f2a8fa8d, 248541270), GSM8K
# first 30: ordinary decode 35.2 vs 34.0 tok/s (+3.4%, same 3155 tokens,
# 3194/3220 forwards hashed on device); DFly2 50.7 vs 51.2 (no gain: verify
# tokens are host-known).
ENGRAM_HASH_MODES = ("auto", "host", "device")
ENGRAM_HASH_DEFAULT = "auto"
# Opt-in MoE kernel routes (default off, unqualified; docs/hy4-flash/fused-moe.md).
# Measured 2026-11-21 (Hy4-flash-MLX-4.6bit-mlx2@f2a8fa8d, 248541270): both are
# bit-identical to the split route on Metal (prefill 512/2048, B1, 4 lanes,
# 4x6 verify); single-rep speed within noise (fused +1-3% prefill, 0.95x
# verify; NAX prefill +4-8%); DFly2 GSM8K with fused 50.1 vs 51.2 tok/s.  Kept
# off: no measured win.
# ``moe_fused_gate_up``: one [gate|up] gather per routed affine expert bank.
# ``moe_nax``: "gather" opens the NAX sorted gather for prefill forwards of
# the ordinary route (moe_nax_gather, M5 hosts; claims the process mode).
MOE_NAX_MODES = ("off", "gather")
# ``moe_top_k`` (opt-in, APPROXIMATE, unqualified; Pierre 2026-11-21): route
# each token to its best k routed experts instead of the checkpoint's
# num_experts_per_tok (16), same sigmoid+bias choice, the k weights
# renormalized as the checkpoint does (norm_topk_prob) and scaled.  Applies to
# the trunk's routers only (the DFly2 drafter has no routed experts); speculative
# routes stay exact against this reduced target.  Own APCv2 namespace.
# Engram table formats (hy_v4_flash_engram.TABLE_FORMATS); bf16 is the
# default, the others are opt-in, unqualified, and need converter shards.
# ``iq4_nl`` is read from GGUF files bound by text_config.engram_gguf_tables
# (the third-party Nace.AI MLX pack), never written by the converter.
ENGRAM_TABLE_FORMATS = ("bf16", "fp8_e4m3_row", "affine4_g32", "iq4_nl")
CONVERTER_ENGRAM_FORMATS = ("bf16", "fp8_e4m3_row", "affine4_g32")
AST_KEYS = frozenset({"draft_model", "num_draft", "ast_topology", "ast_tiles", "ast_tile_rows",
                      "ast_multilane_tile_rows", "ast_adaptive_rows", "batch_size_route",
                      "pairwise_selection", "cost_per_commit_handoff", "ast_graft_prompt_lookup",
                      "ast_target_executor", "ast_tensorfold_rows", "ast_tensorfold_rows_backend",
                      "draft_quantization"})
DFLY_DRAFT_QUANTIZATION_DEFAULT = {"bits": 8, "group_size": 64}
# ``draft_quantization`` (the shared drafter key, adapters/dflash2
# validate_runtime_quantization: {"bits": 4|8, "group_size": 32|64|128}):
# load-time MLX affine quantization of the embedded DFly2 head's nn.Linear
# projections (runtime/drafters/hy4_dfly.quantize_head; QUANTIZATION_KEPT
# lists what stays bf16).  DEFAULT {"bits": 8, "group_size": 64} on the DFly2
# route since 2026-11-21 (Pierre); ``null`` keeps bf16.  Implemented and
# unqualified; DFly2 route only (refused on ordinary and prompt lookup,
# which carry no drafter weights).  Committed tokens are the target's own
# verified choices, so only acceptance and draft time can move.  No APCv2
# namespace of its own: the only drafter state in the prefix cache is
# DFlyFeatureCache (the target's post-norm final hidden of committed
# tokens), computed by the target and identical under either drafter
# precision, so the layout and the drafter record fingerprint are unchanged
# and q8 serving reuses bf16 prefix entries.  Not route-neutral.
# Expected saving on Hy4-flash-MLX-4.6bit-mlx2 at {"bits": 8, "group_size":
# 64}: ~1.62 GB of bf16 Linear weights -> ~0.86 GB (8.5 bits/weight),
# about 0.76 GB resident.
# Opt-in TensorFold row law on the trunk's dense affine q8/q4-g64 projections
# (runtime/tensorfold_rows.py; docs/hy4-flash/ast-tensorfold-rows.md).  Off by
# default, implemented and unqualified.  ``reference`` is the CPU oracle.
# Kept off (2026-11-21, Hy4-flash-MLX-4.6bit-mlx2@f2a8fa8d, 6a449b12c): GSM8K
# first 30 with DFly2, 474 projections installed and engaged on every verify,
# decode 49.8 vs 53.1 tok/s (-6%), acceptance unchanged (3.81 vs 3.77): the
# stock QMV already reads the trunk about once per verify, and the routed
# experts (the verify's real cost) are not row-packable here.
TENSORFOLD_ROWS_BACKENDS = ("metal", "metal_shared2", "reference")
# Trunk subtrees the row law may touch; the LM head (fp32 path), embeddings,
# routed expert banks, the DFly2 head and the vision tower are not under them
# or are not plain ``nn.QuantizedLinear``.
TENSORFOLD_ROWS_SKIP_PARTS = ("lm_head", "embed_tokens", "experts", "switch_mlp")
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
    if policy.get("engram_hash", ENGRAM_HASH_DEFAULT) not in ENGRAM_HASH_MODES:
        raise ValueError(f"Hy4-flash engram_hash must be one of {ENGRAM_HASH_MODES} "
                         f"(default {ENGRAM_HASH_DEFAULT})")
    if policy.get("engram_table_format", "bf16") not in ENGRAM_TABLE_FORMATS:
        raise ValueError(f"Hy4-flash engram_table_format must be one of {ENGRAM_TABLE_FORMATS} (default bf16)")
    if policy.get("latent_kv", "native") not in LATENT_KV_FORMATS:
        raise ValueError("Hy4-flash latent_kv must be 'native' (default) or 'fp8'")
    if type(policy.get("vision", True)) is not bool:
        raise ValueError("Hy4-flash vision must be a boolean (default true: load the tower if present)")
    if type(policy.get("moe_fused_gate_up", False)) is not bool:
        raise ValueError("Hy4-flash moe_fused_gate_up must be a boolean (default false)")
    if policy.get("moe_nax", "off") not in MOE_NAX_MODES:
        raise ValueError("Hy4-flash moe_nax must be 'off' (default) or 'gather'")
    top_k = policy.get("moe_top_k")
    if top_k is not None and (type(top_k) is not int or top_k < 1):
        raise ValueError("Hy4-flash moe_top_k must be a positive integer (default: the checkpoint's)")
    model = {k: v for k, v in policy.items() if k in MODEL_KEYS}
    external = {k: v for k, v in policy.items() if k in AST_KEYS}
    if external and model.get("moe_nax", "off") != "off":
        # Hy4's verify forwards (chain-packed tiles, DFly2) are not marked as
        # verify forwards, so a multi-row verify would read as a prefill.
        raise ValueError("Hy4-flash moe_nax applies to the ordinary route only (not with a draft_model)")
    if "draft_quantization" in external and external.get("draft_model") is None:
        raise ValueError("Hy4-flash draft_quantization quantizes the DFly2 drafter; the "
                         "ordinary route has none (it needs draft_model embedded_dfly)")
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
    if "draft_quantization" in external and kind != "dfly":
        raise ValueError("Hy4-flash draft_quantization applies to the embedded DFly2 drafter only")
    from .dflash2 import validate_runtime_quantization

    # DEFAULT q8 for the embedded DFly2 head (Pierre 2026-11-21): GSM8K first 30
    # at 5259ade5f, 28/30 both, MAL 3.80 vs 3.78, 48.4 vs 47.9 tok/s, ~0.76 GB
    # less resident.  An explicit ``"draft_quantization": null`` keeps bf16.
    explicit = "draft_quantization" in external
    draft_quantization = validate_runtime_quantization(external.pop("draft_quantization", None))
    if not explicit and kind == "dfly":
        draft_quantization = dict(DFLY_DRAFT_QUANTIZATION_DEFAULT)
    executor = external.pop("ast_target_executor", "packed")
    if executor not in AST_TARGET_EXECUTORS:
        raise ValueError("Hy4-flash ast_target_executor must be 'packed' (default) or 'serial'")
    rows = external.pop("ast_tensorfold_rows", False)
    if type(rows) is not bool:
        raise ValueError("Hy4-flash ast_tensorfold_rows must be a boolean (default false)")
    backend = external.pop("ast_tensorfold_rows_backend", None)
    if backend is not None and not rows:
        raise ValueError("ast_tensorfold_rows_backend requires ast_tensorfold_rows")
    if rows and (backend or "metal") not in TENSORFOLD_ROWS_BACKENDS:
        raise ValueError(f"Hy4-flash ast_tensorfold_rows_backend must be one of {TENSORFOLD_ROWS_BACKENDS}")
    external.setdefault("ast_tile_rows", 8)
    from .qwen38_27b import _native_mtp_tree_policy

    external = _native_mtp_tree_policy(external)
    external["ast_target_executor"] = executor
    if draft_quantization is not None:
        external["draft_quantization"] = draft_quantization
    if rows:
        external["ast_tensorfold_rows"] = True
        external["ast_tensorfold_rows_backend"] = backend or "metal"
    return model, external, kind


def set_moe_top_k(model, top_k):
    """Set every trunk router's k (``moe_top_k``); returns the receipt block."""
    routers = [layer.mlp.gate for layer in model.model.layers
               if hasattr(getattr(layer, "mlp", None), "gate") and hasattr(layer.mlp.gate, "top_k")]
    native = routers[0].top_k if routers else None
    if top_k is None:
        return {"selected": False, "k": native, "native_k": native, "approximate": False}
    if not routers:
        raise ValueError("Hy4-flash moe_top_k: the model has no routed-expert layers")
    if top_k > native:
        raise ValueError(f"Hy4-flash moe_top_k {top_k} exceeds the checkpoint's {native}")
    for router in routers:
        router.top_k = int(top_k)
    return {"selected": True, "k": int(top_k), "native_k": native, "layers": len(routers),
            "approximate": top_k != native, "qualified": False}


def resolve_engram_hash(model) -> str:
    """Settle ``engram_hash: auto`` after ``bind_engram``: device when exact.

    The device hasher refuses (ValueError) a geometry it cannot reproduce
    bit for bit; ``auto`` then stays on the host and records why.  Explicit
    ``device`` already failed closed inside ``bind_engram``.
    """
    if getattr(model, "engram_hash_requested", None) != "auto" or not model.engram_hashers:
        return model.engram_hash
    try:
        model.engram_hash = "device"
        model._device_hasher = None
        model.device_hasher  # noqa: B018 - builds and validates the stacked hasher
    except ValueError as error:
        model.engram_hash = "host"
        model._device_hasher = None
        model.engram_hash_auto_reason = str(error)
    return model.engram_hash


def load_model(path, artifact, *, keep_dfly=False, engram_cache_rows=1 << 16, memory_budget=None,
               engram_precision="transformers", engram_table_format="bf16", latent_kv="native",
               vision=True, engram_hash=ENGRAM_HASH_DEFAULT, moe_fused_gate_up=False,
               dfly_quantization=None):
    """``(model, drafter or None, engram readers)`` for an inspected artifact.

    Engram tables stay on disk; the DFly2 layers are read only with
    ``keep_dfly``; the ``vit.*`` tower is read and attached (checkpoint dtype, never
    quantized) when ``vision`` and the artifact carries it, else dropped
    before any byte is read.  ``dfly_quantization`` (a validated
    ``draft_quantization`` value or None) quantizes the drafter's
    projections after its strict bf16 load (hy4_dfly.quantize_head).
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
    if engram_hash not in ENGRAM_HASH_MODES:
        raise ValueError(f"Hy4-flash engram_hash must be one of {ENGRAM_HASH_MODES}")
    # ``auto`` binds on the host, then switches to device hashing only if the
    # bound geometry is exactly hashable there (resolve_engram_hash).
    model.engram_hash = "host" if engram_hash == "auto" else engram_hash
    model.engram_hash_requested = engram_hash
    model.engram_hash_auto_reason = None
    tables = {f"model.layers.{i}.engram_layer.multi_head_embedding.embedding.weight"
              for i in args.engram_layers}

    keep_vision = bool(vision) and bool(artifact.get("has_vision"))

    def shard_filter(shard):
        return {k: v for k, v in shard.items()
                if k not in tables and (keep_vision or not k.startswith("vit."))
                and (keep_dfly or not k.startswith("model.dfly_layers."))}

    files = sorted(set(artifact["weight_map"].values()))
    resident = resident_bytes(path, artifact["weight_map"], tables, keep_dfly, keep_vision)
    resident += widened_bytes(path, artifact["weight_map"], config.get("quantization"),
                              lm_head_fp32=args.enable_lm_head_fp32)
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
        check_quantization(quant)
        overrides = module_quantization(quant)
        q2_zero = {n: o for n, o in sorted(overrides.items()) if o.get("mode") == Q2_ZERO}

        def predicate(name, module):
            override = overrides.get(name)
            if isinstance(override, dict):
                # q2_0 banks are installed below, not by nn.quantize.
                return False if override.get("mode") == Q2_ZERO else override
            return hasattr(module, "to_quantized") and f"{name}.scales" in weights

        nn.quantize(model, group_size=quant["group_size"], bits=quant["bits"],
                    mode=quant.get("mode", "affine"), class_predicate=predicate)
        install_q2_zero(model, weights, q2_zero)
    if moe_fused_gate_up:
        # After quantization: only resident affine banks of one format fuse
        # (q2_0 and mixed banks keep the split route, recorded); the planes
        # are concatenated in place, so the split planes are never resident.
        model.fuse_routed_gate_up(weights)
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
                                 formats=config.get("mlx2_engram_formats"),
                                 gguf_tables=text_config(config).get("engram_gguf_tables"))
    try:
        model.bind_engram(readers)
        resolve_engram_hash(model)
        drafter = None
        if keep_dfly:
            intermediate, correction = HY4FlashDFly.geometry(dfly_raw)
            drafter = HY4FlashDFly(args, intermediate=intermediate, correction=correction)
            drafter.load_weights(list(drafter.sanitize(dfly_raw).items()), strict=True)
            from ..runtime.drafters.hy4_dfly import quantize_head

            quantize_head(drafter, dfly_quantization)
        elif dfly_quantization is not None:
            raise ValueError("Hy4-flash dfly_quantization needs the DFly2 drafter (keep_dfly)")
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


def install_q2_zero(model, weights: dict, banks: dict) -> None:
    """Swap each ``q2_0`` expert bank (module name -> override) for
    ``Q2ZeroSwitchLinear`` and widen its planes in ``weights``."""
    from ..runtime.models.hy_v4_flash import Q2ZeroSwitchLinear, q2_0_planes

    for stem, entry in banks.items():
        group_size = entry.get("group_size", 64)
        parent_name, _, leaf = stem.rpartition(".")
        if not stem.startswith("model.layers.") or ".mlp.experts." not in stem + ".":
            raise ValueError(f"q2_0 is implemented for routed expert banks only: {stem}")
        parent = model
        for part in parent_name.split("."):
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        old = getattr(parent, leaf)
        if not hasattr(old, "num_experts") or "scales" in old:
            raise ValueError(f"{stem} is not an unquantized expert bank")
        group = int(group_size)
        w, scales = weights.get(stem + ".weight"), weights.get(stem + ".scales")
        if (w is None or scales is None or w.shape[-1] * 16 != old.input_dims
                or scales.shape[-1] * group != old.input_dims):
            raise ValueError(f"{stem}: q2_0 planes disagree with the bank geometry")
        bank = Q2ZeroSwitchLinear(old.input_dims, old.output_dims, old.num_experts, group_size=group)
        q2_0_planes(weights, stem)
        setattr(parent, leaf, bank)


class HYV4FlashAdapter(ExternalDraftAdapterMixin, OrdinaryTextAdapter):
    descriptor = descriptor_for()
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "hy-v4-flash-ordinary"
    # AST tree tiles: chain-packed exact verify (hy_v4_flash_ast); the
    # embedded DFly2 head fills the native-head drafter slot.
    ast_tree_tiles = True
    ast_tree_tile_dtype_gate = False
    # Execution-policy keys accepted unchanged on both routes the best-route
    # entry can select, ordinary and AST/DFly2 (best_route_defaults): a policy
    # of only these keeps the DFly2 default.  Each is a MODEL_KEYS value that
    # _parse_policy and load_model consume identically with or without a
    # draft_model.  Left out: ``moe_nax`` (refused with a draft_model),
    # ``latent_kv`` (fp8 changes the cache layout and drops prefix reuse, a
    # shape the default was not measured on) and every AST key.
    route_neutral_policy_keys = frozenset({
        "engram_table_format", "engram_precision", "engram_cache_rows", "engram_hash",
        "vision", "moe_fused_gate_up",
    })
    prompt_lookup_default = False
    EXTERNAL_PROFILE = "hy-v4-flash-ast-b1-b4-v1"
    EXTERNAL_ROUTE_TAG = "hy4-flash-chain-packed-tree-v1"
    EXTERNAL_DEFAULT_NUM_DRAFT = 5
    # Media prefill and decode run natively on the ordinary route; on the
    # external-draft route an image request decodes plain (below).
    multimodal_routes = frozenset({"ordinary"})
    # Image requests on the external-draft route (DFly2 / AST / prompt
    # lookup drafters) decode plain: an isolated media prefill, then the
    # ordinary one-token step, in that engine's permanent ordinary lane
    # (Pierre 2026-11-20: DFly2 by default and images on the same server).
    multimodal_plain_decode_routes = frozenset({"external_draft"})
    # Residual writers for the experimental residual-ablation tooling
    # (weight_space/residual_ablation.py; docs/research/heretic-hy4-flash-20261121).
    # Each branch is written into all hc_mult streams as ``post_j * x`` (iHC,
    # no stream-mixing matrix), so a direction removed from ``x`` leaves every
    # stream's increment.  Both branches end in an RMSNorm: attention
    # ``post_proj_norm(o_proj(.))``, MoE ``combine_output_norm(up(routed) +
    # shared down(.))``, layer-0 MLP ``post_fc2_norm(down(.))``; the edit
    # therefore projects ``normalize(gamma * v)`` from those linears.  The
    # routed bank reaches the residual only through ``routed_expert_up_proj``
    # (dense), so a MoE layer without it fails closed.  Captures average the
    # four streams.  Engram (layers 1/11/21/31) and the DFly2 head are not
    # edited.  Prompts render through the adapter (the template's own
    # default reasons; mlx2 serves no_think).
    residual_ablation = {
        "writers": {"attention": ("self_attn",), "mlp": ("mlp",)},
        "linears": {"attention": ("self_attn.o_proj",),
                    "mlp": ("mlp.down_proj", "mlp.routed_expert_up_proj",
                            "mlp.shared_experts.down_proj")},
        "post_norms": {"self_attn.o_proj": "self_attn.post_proj_norm",
                       "mlp.down_proj": "mlp.post_fc2_norm",
                       "mlp.routed_expert_up_proj": "mlp.combine_output_norm",
                       "mlp.shared_experts.down_proj": "mlp.combine_output_norm"},
        "hook_point": "linears",
        "capture": "stream_mean",
        "requires": {"mlp.experts": ("mlp.routed_expert_up_proj",)},
        "prompt_render": "adapter",
        # Engram adds per-stream scaled value_proj output before layers
        # 1/11/21/31; a rank-1 edit of the declared linears cannot remove it.
        "unedited_writers": ("embed", "engram"),
    }

    @property
    def residual_overlay_routes(self):
        """Serving routes whose target forwards bind residual-ablation overlay rows.

        ``external_draft`` (DFly2 / AST / prompt lookup): the engine binds each
        target forward's lanes and ``hy_v4_flash_ast`` binds every chain wave,
        so verify rows carry their own request's writer edit and committed
        tokens equal ordinary decode under the same overlay.  Not with the
        TensorFold row law (``ast_tensorfold_rows``), which replaces trunk
        projections the overlay wraps.
        """
        routes = {"ordinary"}
        if not (getattr(self, "external_policy", None) or {}).get("ast_tensorfold_rows"):
            routes.add("external_draft")
        return frozenset(routes)

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
        engram_format, self.engram_table_format_source = resolve_engram_table_format(
            policy, artifact["engram_table_formats"])
        if engram_format not in artifact["engram_table_formats"]:
            raise ValueError(f"Hy4-flash artifact carries no {engram_format} Engram tables (it has "
                             f"{artifact['engram_table_formats']}); select one with engram_table_format")
        self.latent_kv = policy.get("latent_kv", "native")
        self.descriptor = descriptor_for(dfly_selected=self.dfly_selected,
                                         engram_table_format=engram_format, latent_kv=self.latent_kv)
        if self.latent_kv == "fp8":
            self.layout = FP8_CACHE_LAYOUT
        elif self.dfly_selected:
            self.layout = DFLY_CACHE_LAYOUT
        self.tensorfold_rows = None
        rows_backend = (external.get("ast_tensorfold_rows_backend")
                        if external.get("ast_tensorfold_rows") else None)
        self.moe_fused_gate_up = bool(policy.get("moe_fused_gate_up", False))
        self.moe_nax = policy.get("moe_nax", "off")
        if self.moe_nax != "off" and not artifact["quantized"]:
            raise ValueError("Hy4-flash moe_nax applies only to an affine-quantized artifact")
        if self.moe_fused_gate_up and not artifact["quantized"]:
            raise ValueError("Hy4-flash moe_fused_gate_up fuses quantized expert banks only")
        if self.moe_fused_gate_up:
            from ..runtime.models import switch_grouped_experts

            if switch_grouped_experts.MODE != "off":
                raise ValueError("Hy4-flash moe_fused_gate_up needs MLX2_MOE_GROUPED_EXPERTS off "
                                 "(the grouped kernels read split gate/up tables)")
        self.layout = numerics_layout(self.layout, engram_table_format=engram_format,
                                      engram_precision=policy.get("engram_precision", "transformers"),
                                      tensorfold_rows=rows_backend,
                                      moe_fused_gate_up=self.moe_fused_gate_up, moe_nax=self.moe_nax,
                                      moe_top_k=policy.get("moe_top_k"))
        self.environment = configure_environment()
        from .process_globals import claim_moe, claim_stock_moe

        # The sorted expert gathers read two process globals (NAX mode, rhs
        # pad policy); bind them while this adapter lives.  The stock claim
        # pins the module defaults (NAX off, floor pad).
        if self.moe_nax == "gather":
            claim_moe(self, "Hy4-flash ordinary (moe_nax gather)", nax_mode="gather")
        else:
            claim_stock_moe(self, "the Hy4-flash adapter")
        path = Path(self.identity["path"])
        config = artifact["config"]
        from transformers import AutoTokenizer

        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper

        self.model, self.dfly, self.engram_tables = load_model(
            path, artifact, keep_dfly=self.dfly_selected,
            engram_cache_rows=int(policy.get("engram_cache_rows", 1 << 16)),
            engram_precision=policy.get("engram_precision", "transformers"),
            engram_table_format=engram_format,
            latent_kv=self.latent_kv, vision=policy.get("vision", True),
            engram_hash=policy.get("engram_hash", ENGRAM_HASH_DEFAULT),
            moe_fused_gate_up=self.moe_fused_gate_up,
            dfly_quantization=external.get("draft_quantization"))
        self.model.moe_nax_prefill = self.moe_nax != "off"
        self.moe_top_k = set_moe_top_k(self.model, policy.get("moe_top_k"))
        self.engram_table_format = engram_format
        if rows_backend is not None:
            from ..runtime.tensorfold_rows import install as install_rows

            # Fails closed (ValueError) when no trunk projection is admitted.
            self.tensorfold_rows = install_rows(
                {f"model.layers.{i}": layer for i, layer in enumerate(self.model.model.layers)},
                skip_parts=TENSORFOLD_ROWS_SKIP_PARTS, backend=rows_backend)
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
        self.dfly_stats = {"requests": 0, "verify_rounds": 0, "verify_accept_hist": [0] * (self.num_draft + 1),
                           "verify_depth_hist": [0] * (self.num_draft + 1)}
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
            rows = getattr(self, "tensorfold_rows", None)
            kwargs["target_tree_executor_provider"] = Hy4ChainTreeExecutor.create(
                joint=self.external_policy.get("ast_target_executor", "packed") == "packed",
                call_scope=None if rows is None else (lambda: rows.scope("ast_verify")))
        self.tree_executor = kwargs["target_tree_executor_provider"]
        kwargs.setdefault("tree_node_budget_by_lanes",
                          _tile_node_budgets(self.external_policy, AST_MAX_WIDTH))
        for key in ("ast_topology", "ast_adaptive_rows"):
            if key in self.external_policy:
                kwargs.setdefault(key, self.external_policy[key])
        return super().create_external_batch(**kwargs)

    # Hy4's template takes ``reasoning_effort`` (no_think/high, default high;
    # ``max`` was removed upstream in tencent/Hy4-flash@b2498eba).  mlx2 renders no_think unless the request opts in
    # (``hy_v4_flash_output.reasoning_effort``): high mode can spend the whole
    # token budget reasoning (vllm#66454).
    reasoning_effort_semantics = "template_levels_no_think_high"

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
        tower = self.model.vision_tower.args
        proc = self.image_processor_settings
        min_pixels, max_pixels = proc["min_pixels"], proc["max_pixels"]
        values = []
        for source in media:
            value = resolve_media(source, kind="image", file_loader=file_loader)
            values.append(value)
            patches, grid = preprocess_image(value.value, patch_size=tower.patch_size,
                                             merge_size=tower.spatial_merge_size,
                                             min_pixels=min_pixels, max_pixels=max_pixels,
                                             image_mean=proc["image_mean"],
                                             image_std=proc["image_std"])
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
                  "max_pixels": max_pixels, "image_mean": list(proc["image_mean"]),
                  "image_std": list(proc["image_std"])}
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
    def image_processor_settings(self) -> dict:
        """The artifact's image-processor settings, read as transformers does.

        ``processor_config.json``'s nested ``"image_processor"`` dict wins
        (tencent/Hy4-flash ships only this file: mean/std 0.5, ``min_pixels``
        4096, ``max_pixels`` 2048*2048), else ``preprocessor_config.json``;
        unset keys keep the processor class defaults (CLIP mean/std, the
        vision config's ``min/max_image_size`` squared).
        """
        from .hy_v4_flash_media import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

        tower = self.model.vision_tower.args
        out = {"min_pixels": tower.min_image_size ** 2, "max_pixels": tower.max_image_size ** 2,
               "image_mean": tuple(OPENAI_CLIP_MEAN), "image_std": tuple(OPENAI_CLIP_STD)}
        data, _ = self._image_processor_config()
        if isinstance(data, dict):
            size = data.get("size") or {}
            out["min_pixels"] = int(data.get("min_pixels", size.get("shortest_edge", out["min_pixels"])))
            out["max_pixels"] = int(data.get("max_pixels", size.get("longest_edge", out["max_pixels"])))
            for key in ("image_mean", "image_std"):
                if data.get(key) is not None:
                    value = tuple(float(v) for v in data[key])
                    if len(value) != 3:
                        raise ValueError(f"Hy4-flash image processor {key} must have 3 channels")
                    out[key] = value
            if data.get("do_normalize") is False or data.get("do_rescale") is False:
                raise ValueError("Hy4-flash image processor without rescale/normalize is not implemented")
            factor = data.get("rescale_factor")
            if factor is not None and abs(float(factor) - 1 / 255) > 1e-9:
                raise ValueError(f"Hy4-flash image processor rescale_factor {factor} is not implemented "
                                 "(only 1/255)")
            # ``resample`` is not read: the reference processor resizes with
            # BICUBIC whatever the config says (hy_v4_flash_media).
        return out

    def _image_processor_config(self):
        """``(settings dict or None, source label)`` of the artifact's image processor."""
        root = Path(self.identity["path"])
        nested = root / "processor_config.json"
        if nested.is_file():
            data = json.loads(nested.read_text()).get("image_processor")
            if data is not None:
                return data, "processor_config.json"
        flat = root / "preprocessor_config.json"
        if flat.is_file():
            return json.loads(flat.read_text()), "preprocessor_config.json"
        return None, IMAGE_PROCESSOR_FALLBACK

    def _image_processor_receipt(self):
        # A vision artifact without a processor config still loads, but its
        # images are normalized with the class defaults (CLIP mean/std), not
        # the checkpoint's; status must say so (a converted or assembled
        # artifact that lost the file otherwise looks healthy).
        if not getattr(getattr(self, "model", None), "has_vision", False) or not getattr(self, "identity", None):
            return None
        cached = getattr(self, "_image_processor_receipt_cache", None)
        if cached is not None:           # status snapshots call this about once a second: read the file once
            return dict(cached)
        try:
            _, source = self._image_processor_config()
        except (OSError, ValueError, AttributeError) as error:
            return {"source": f"unreadable ({type(error).__name__})", "fallback": False}
        self._image_processor_receipt_cache = {"source": source, "fallback": source == IMAGE_PROCESSOR_FALLBACK}
        return dict(self._image_processor_receipt_cache)

    def _checkpoint_label(self):
        # Binding an external drafter replaces identity["fingerprint"] with a
        # route digest and keeps the artifact's under "target_fingerprint";
        # a checkpoint label names the artifact.
        fingerprint = self.identity.get("target_fingerprint", self.identity["fingerprint"])
        return f"{Path(self.identity['path']).name}@{fingerprint}"

    @property
    def image_pixel_bounds(self) -> tuple[int, int]:
        """``(min_pixels, max_pixels)`` of ``image_processor_settings``."""
        proc = self.image_processor_settings
        return proc["min_pixels"], proc["max_pixels"]

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
        for i, n in enumerate(receipt.get("verify_depth_hist") or ()):
            stats["verify_depth_hist"][i] += n
        receipt["checkpoint"] = self._checkpoint_label()
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
        dfly["quantization"] = self._dfly_quantization_receipt(dfly["observed_used"])
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
            "checkpoint": self._checkpoint_label()
            if getattr(self, "identity", None) else None,
            "continuous_batch": {"implemented": True, "qualified": False},
            "ast": {
                "implemented": True, "qualified": False, "selected": draft is not None,
                "max_width": AST_MAX_WIDTH,
                "target_executor": getattr(self, "external_policy", {}).get("ast_target_executor"),
                "executor": self.tree_executor.receipt()
                if getattr(self, "tree_executor", None) is not None else None,
                "tensorfold_rows": self.tensorfold_rows.receipt()
                if getattr(self, "tensorfold_rows", None) is not None else {
                    "implemented": True, "qualified": False, "selected": False,
                    "observed_used": False},
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
            "moe_fused_gate_up": self._moe_fused_receipt(),
            "moe_nax_gather": self._moe_nax_receipt(),
            "moe_top_k": getattr(self, "moe_top_k", None),
            "engram": {i: r.diagnostics() for i, r in getattr(self, "engram_tables", {}).items()},
            "engram_precision": getattr(getattr(getattr(self, "model", None), "args", None),
                                        "engram_precision", None),
            "engram_table_format": self._engram_format_record(),
            "engram_hash": self._engram_hash_receipt(),
            "latent_kv": self._latent_kv_receipt(),
            "approximate_state": (getattr(self, "latent_kv", "native") == "fp8"
                                  or bool((self._engram_format_record() or {}).get("approximate"))),
            "vision": {
                "implemented": True,
                "qualified": False,
                "loaded": bool(getattr(getattr(self, "model", None), "has_vision", False)),
                "modalities": ("image",),
                "image_processor": self._image_processor_receipt(),
                "video": "refused (not implemented)",
                "routes": tuple(sorted(type(self).multimodal_routes)),
                "plain_decode_routes": tuple(sorted(type(self).multimodal_plain_decode_routes)),
                "observed_used": bool(getattr(getattr(self, "model", None), "vision_stats",
                                              {}).get("prefills")),
                "stats": dict(getattr(getattr(self, "model", None), "vision_stats", {})),
            },
        }

    def _dfly_quantization_receipt(self, drafter_used):
        """``draft_quantization`` of the DFly2 head (bf16 default, opt-in affine)."""
        loaded = getattr(getattr(self, "dfly", None), "quantization_receipt", None) or {}
        selected = bool(loaded.get("selected"))
        out = {"implemented": True, "qualified": False, "selected": selected,
               # The quantized head drafted at least one round.
               "observed_used": selected and bool(drafter_used)}
        for key in ("bits", "group_size", "mode", "bytes_before", "bytes_after"):
            out[key] = loaded.get(key)
        if selected:
            out["module_count"] = loaded["module_count"]
            out["skipped_group_width"] = list(loaded["skipped_group_width"])
            out["kept_checkpoint_dtype"] = list(loaded["kept_checkpoint_dtype"])
        return out

    def _engram_hash_receipt(self):
        model = getattr(self, "model", None)
        mode = getattr(model, "engram_hash", "host")
        stats = dict(getattr(model, "engram_hash_stats", {}))
        return {"mode": mode, "requested": getattr(model, "engram_hash_requested", mode),
                "auto_fallback_reason": getattr(model, "engram_hash_auto_reason", None),
                "modes": ENGRAM_HASH_MODES, "default": ENGRAM_HASH_DEFAULT, "implemented": True,
                "exact": True, "qualified": False, "selected": mode == "device",
                "observed_used": bool(stats.get("device_forwards")), "stats": stats}
    def _moe_fused_receipt(self):
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "moe_gate_up_status"):
            return None
        return model.moe_gate_up_status()

    def _moe_nax_receipt(self):
        from ..runtime.models import moe_nax_gather

        mode = getattr(self, "moe_nax", "off")
        status = moe_nax_gather.status() if mode != "off" else None
        engaged = sum((status or {}).get("calls", {}).values())
        return {
            "implemented": True, "qualified": False, "selected": mode != "off", "mode": mode,
            "scope": "prefill forwards of the ordinary route (trunk forward_scope)",
            # Process-wide counters: another live adapter on the same mode adds to them.
            "observed_used": engaged > 0,
            **({"status": status} if status is not None else {}),
        }

    def _engram_format_record(self):
        fmt = getattr(self, "engram_table_format", None)
        if fmt is None:
            return None
        precision = getattr(getattr(getattr(self, "model", None), "args", None), "engram_precision", "transformers")
        return {**engram_format_record(fmt, precision), "selected": True,
                "selected_by": getattr(self, "engram_table_format_source", "policy"),
                "observed_used": any(r.stats.get("lookups") for r in getattr(self, "engram_tables", {}).values())}

    def close(self):
        for reader in getattr(self, "engram_tables", {}).values():
            reader.close()
        self.engram_tables = {}
        super().close()  # releases a moe_nax process claim
