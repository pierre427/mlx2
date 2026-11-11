"""Owned Nemotron 3.5 Super VL adapter and source-bound mixed affine loader.

Artifact inspection is CPU-only. All serving routes remain unqualified until
tested against the completed converted artifact. See provenance/.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..conversion.nemotron35_plan import make_plan
from ..conversion.nvfp4 import read_header, sha256
from ..sampling_defaults import MODEL_CARD, SamplingDefaults, VendorSampling
from .external_draft_policy import ExternalDraftAdapterMixin
from .nemotron3_super import Nemotron3SuperAdapter, configure_environment
from .nemotron35_super_config import (
    NVFP4_EXPORT_CONFIG_SHA256,
    NVFP4_EXPORT_INDEX_SHA256,
    NVFP4_EXPORT_REPO,
    NVFP4_EXPORT_REVISION,
    NVFP4_EXPORT_SOURCE,
    NVFP4_EXPORT_SOURCE_REVISION,
    NVFP4_QUANTIZATION,
    SOURCE_CONFIG_SHA256,
    SOURCE_INDEX_SHA256,
    SOURCE_REVISION,
    language_config,
    nvfp4_export_schema,
    tensor_schema,
    vision_config,
)

CACHE_LAYOUT = "nemotron35-super-vl-hybrid-fp32-router-v1"
EXECUTION_LIMITS = {
    "active_sessions": "1_default_or_2_or_4_exact_progressive_external",
    "draft_depth": "head_or_runtime_bound",
    "verification_rows": "request_private_per_session",
}
DESCRIPTOR = ModelDescriptor(
    model_type="nemotron_h_omni",
    family="nemotron35-super-vl-120b-a12b",
    variant="mixed-affine",
    cache_layout=CACHE_LAYOUT,
    state_planes=frozenset(
        {
            StatePlane.ATTENTION_KV,
            StatePlane.RECURRENT,
            StatePlane.RNG,
            StatePlane.TRANSCRIPT,
            StatePlane.DRAFT,
        }
    ),
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.VISION,
            Capability.STREAMING,
            Capability.TOOLS,
            Capability.REASONING,
            Capability.CONTINUOUS_BATCH,
            Capability.PREFIX_REUSE,
            Capability.APC_V2,
            Capability.LAYERED_CACHE,
            Capability.GRAMMAR,
            # Row-exact B1 verify (nemotron_h.Model.pld_verify_forward).
            Capability.PROMPT_LOOKUP,
            Capability.MTP,
        }
    ),
    metadata={
        "execution": "mlx2.adapters.nemotron35_super.Nemotron35SuperAdapter",
        "qualification": "unqualified",
        "ordinary_reference": "implemented",
        "vision": "image-prefill-b1-candidate",
        "video": "prefill-b1-candidate-no-evs",
        "native_mtp": "candidate",
        "external_draft": "requires-bound-head",
        "execution_limits": EXECUTION_LIMITS,
    },
)
SAMPLING = VendorSampling(
    {
        "text": SamplingDefaults(temperature=1.0, top_p=0.95, source=MODEL_CARD),
        "vision": SamplingDefaults(temperature=0.6, top_p=0.95, source=MODEL_CARD),
    },
    general="text",
    model="NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
)


# Exact parent-indexed tree verify on the external route (opt-in, default
# off): B1 trees of up to ``ast_tile_rows`` rows (anchor included), B2+ chain.
NEMOTRON_TREE_ROUTE = "tree15_b1_chain_b2plus_v1"
TREE_POLICY_KEYS = frozenset(
    {"batch_size_route", "ast_target_executor", "ast_tile_rows", "ast_adaptive_rows"}
)
# Chain-only controls the tree route does not compose with.
_TREE_REFUSED_KEYS = frozenset({
    "proposal_composition", "continuation_pool", "continuation_strategy",
    "progressive_verification_tile", "adaptive_verification",
    "cost_per_commit_handoff",
    # The tree's drafter lattice captures no feedback payloads, and the
    # physical AST target is a chain-route executor.
    "lilicorr_feedback", "ast_physical_target",
})


SUPER_TREE_GUARDED_HANDOFF = {
    "enabled": True, "reversible": True, "ordinary_samples": 2,
    "speculative_samples": 2, "loss_ratio": 0.8, "loss_decisions": 2,
    "window": 8, "reprobe_tokens": 64, "reprobe_max_tokens": 512,
    "deferred_context_cap": 256, "settle_rounds": 1, "pipelined_park": True,
}


def _super_tree_cost_handoff(default="guarded"):
    """``(value, mode, defaulted)``: the Super tree route's cost-per-commit
    handoff (see create_external_batch); ``default`` is the class's mode when
    the switch is unset."""
    raw = os.environ.get("MLX2_SUPER_TREE_COST_HANDOFF")
    mode = (default if raw is None else raw).strip().lower()
    if mode == "off":
        return False, mode, raw is None
    if mode == "guarded":
        return dict(SUPER_TREE_GUARDED_HANDOFF), mode, raw is None
    raise ValueError("MLX2_SUPER_TREE_COST_HANDOFF must be off or guarded")


def _kapitsa_prefill_report():
    """Opt-in prefill kernels (MLX2_NEMOTRON_SSD_MMA / MLX2_NEMOTRON_ATTN_STAGED):
    switch, engaged and refused counts with reasons; implemented, unqualified,
    selected only when switched on."""
    from ..runtime.models import nemotron_attn_staged, nemotron_ssd_mma

    report = {}
    for name, module, engaged in (("ssd_mma", nemotron_ssd_mma, "mma"),
                                  ("attn_staged", nemotron_attn_staged, "staged")):
        status = module.status()
        report[name] = {**status, "implemented": True, "qualified": False,
                        "selected": bool(status["enabled"]),
                        "observed_used": status["counts"].get(engaged, 0) > 0}
    return report


def validate_super_tree_policy(policy) -> bool:
    """Tree-route controls; ``True`` when the exact tree route is selected."""
    route = policy.get("batch_size_route")
    if route is None:
        if set(policy) & (TREE_POLICY_KEYS - {"batch_size_route"}):
            raise ValueError(
                f"Super VL tree controls require batch_size_route {NEMOTRON_TREE_ROUTE}"
            )
        return False
    if route == "chain":
        # The operator's opt-out of the default tree (resolved before here
        # by ``Nemotron35SuperAdapter._default_tree_route``).
        if set(policy) & (TREE_POLICY_KEYS - {"batch_size_route"}):
            raise ValueError("Super VL chain route takes no tree controls")
        return False
    if route != NEMOTRON_TREE_ROUTE:
        raise ValueError(f"Super VL external batch_size_route must be {NEMOTRON_TREE_ROUTE}")
    if policy.get("ast_target_executor", "packed") not in ("packed", "serial"):
        raise ValueError("Super VL tree target executor must be packed or serial")
    rows = policy.get("ast_tile_rows", 16)
    if type(rows) is not int or not 4 <= rows <= 16:
        raise ValueError("Super VL tree tile rows must be an integer from 4 to 16")
    if policy.get("ast_adaptive_rows", None) not in (None, True, False):
        raise ValueError("Super VL ast_adaptive_rows must be boolean")
    refused = sorted(_TREE_REFUSED_KEYS & set(policy))
    if refused:
        raise ValueError(
            "Super VL tree verify does not compose with " + ", ".join(refused)
        )
    return True


def _split_super_execution_policy(execution_policy):
    if execution_policy is not None and not isinstance(execution_policy, dict):
        raise ValueError("Super VL execution policy must be an object")
    policy = dict(execution_policy or {})
    expert_sort = policy.pop("expert_gather_sort", "auto")
    mamba_conv = policy.pop("mamba_conv", "auto")
    mamba_norm = policy.pop("mamba_norm", "auto")
    mamba_ssm = policy.pop("mamba_ssm", "auto")
    mamba_projection = policy.pop("mamba_projection", "auto")
    mamba_segment = policy.pop("mamba_segment", "auto")
    relu2_down = policy.pop("moe_relu2_down", "auto")
    if not isinstance(expert_sort, str) or expert_sort not in {
        "auto",
        "unsorted_decode",
    }:
        raise ValueError("Super VL expert_gather_sort must be auto or unsorted_decode")
    if not isinstance(mamba_conv, str) or mamba_conv not in {
        "auto",
        "fused_b1",
        "fused_b1_compact",
        "fused_short",
    }:
        raise ValueError(
            "Super VL mamba_conv must be auto, fused_b1, fused_b1_compact, "
            "or fused_short"
        )
    if not isinstance(mamba_norm, str) or mamba_norm not in {
        "auto",
        "fused_b1",
        "fused_short",
    }:
        raise ValueError("Super VL mamba_norm must be auto, fused_b1, or fused_short")
    if not isinstance(mamba_ssm, str) or mamba_ssm not in {
        "auto",
        "fused_short",
        "fused_short_gate",
    }:
        raise ValueError(
            "Super VL mamba_ssm must be auto, fused_short, or fused_short_gate"
        )
    if not isinstance(relu2_down, str) or relu2_down not in {"auto", "fused_b1"}:
        raise ValueError("Super VL moe_relu2_down must be auto or fused_b1")
    if not isinstance(mamba_segment, str) or mamba_segment not in {
        "auto",
        "layerwise_short_exact",
        "layerwise_short_exact_routed_batch",
    }:
        raise ValueError(
            "Super VL mamba_segment must be auto, layerwise_short_exact, or "
            "layerwise_short_exact_routed_batch"
        )
    if not isinstance(mamba_projection, str) or mamba_projection not in {
        "auto",
        "compact_padded",
        "compact_row_exact",
        "adaptive_row_exact",
        "short_row_exact",
        "native_qmm_residual_b1",
    }:
        raise ValueError(
            "Super VL mamba_projection must be auto, compact_padded, or "
            "compact_row_exact, adaptive_row_exact, short_row_exact, or "
            "native_qmm_residual_b1"
        )
    return (
        expert_sort,
        mamba_conv,
        mamba_norm,
        mamba_ssm,
        mamba_projection,
        mamba_segment,
        relu2_down,
        policy,
    )


MOE_NAX_GATHER_MODES = ("off", "gather")


def _split_moe_nax_gather(execution_policy):
    """``(policy without "moe_nax_gather", mode)``; the mode defaults to off.

    "gather" runs prefill expert gathers through the segmented NAX sorted
    kernel (``runtime/models/moe_nax_gather.py``), bit-identical to the stock
    sorted kernel; "fused" needs a fused gate/up module Nemotron's relu2
    experts do not have.
    """
    if not isinstance(execution_policy, dict) or "moe_nax_gather" not in execution_policy:
        return execution_policy, "off"
    policy = dict(execution_policy)
    mode = policy.pop("moe_nax_gather")
    if mode not in MOE_NAX_GATHER_MODES:
        raise ValueError("Super VL moe_nax_gather must be off or gather")
    return policy, mode


def _moe_nax_gather_report(mode):
    from ..runtime.models import moe_nax_gather as _moe_nax

    # The qualifier reads engaged calls and fallbacks from the status keys
    # (qualification.moe_nax_gather_engagement), so they sit at top level.
    return {
        **(_moe_nax.status() if mode != "off" else {"mode": "off"}),
        "schema": "mlx2.nemotron35-super-moe-nax-gather.v1",
        "implemented": True,
        "qualified": False,
        "selected": mode != "off",
        "policy": mode,
    }


NEMOTRON_INT8_PREFILL_SCOPES = ("mlp", "all")


def nemotron_int8_prefill_select(scope):
    """Trunk projections that may take the int8 (W8A8) prefill route.

    ``mlp``: the 8-bit shared experts (``mixer.shared_experts.*``).  ``all``
    adds the Mamba ``in_proj``/``out_proj`` and attention ``q/k/v/o``.  The
    routed experts (sorted gathers), the bf16 router and latent projections,
    the LM head, the MTP head and any media tower stay stock.  Approximate:
    the Mamba projections feed 40 recurrent layers, so this needs a
    long-context KL check before it is ever selected by default.
    """

    def select(path, module):
        segments = path.split(".")
        if segments[:2] != ["backbone", "layers"] or len(segments) < 5:
            return False
        if segments[3] != "mixer" or "switch_mlp" in segments:
            return False
        if "shared_experts" in segments:
            return segments[-1] in {"up_proj", "down_proj"}
        if scope != "all":
            return False
        return segments[-1] in {"in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"}

    return select


def _super_expert_gathers(model):
    layers = list(model.layers)
    mtp = getattr(model, "mtp", None)
    if mtp is not None:
        layers.extend(mtp.layers)
    return [
        layer.mixer.switch_mlp
        for layer in layers
        if getattr(layer, "block_type", None) == "E"
    ]


def _super_mamba_mixers(model):
    layers = list(model.layers)
    mtp = getattr(model, "mtp", None)
    if mtp is not None:
        layers.extend(mtp.layers)
    return [
        layer.mixer for layer in layers if getattr(layer, "block_type", None) == "M"
    ]


def _super_moe_mixers(model):
    layers = list(model.layers)
    mtp = getattr(model, "mtp", None)
    if mtp is not None:
        layers.extend(mtp.layers)
    return [
        layer.mixer for layer in layers if getattr(layer, "block_type", None) == "E"
    ]


def native_draft_depth(value):
    """Preserve the runtime's bounded depth range without imposing a three-row cap."""
    from .mtp_depth_cap import MAX_SELF_MTP_DEPTH_CAP, validate_self_mtp_num_draft

    return validate_self_mtp_num_draft(
        value, {"MLX2_MTP_DEPTH_CAP": str(MAX_SELF_MTP_DEPTH_CAP)}
    )


# The unconverted vendor release, served unquantized (bf16 everywhere).  No
# mlx2 conversion ran, so there is no recipe; the source's own pins bind it.
SOURCE_CONVERSION = {
    "recipe": "bf16-source-unconverted",
    "expert_bits": None,
    "activation_aware": False,
}
SOURCE_DESCRIPTOR = replace(DESCRIPTOR, variant="bf16-source")


def inspect_source_artifact(path, config):
    """The pinned bf16 vendor release, without an mlx2 conversion.

    The config and index bytes must be the pinned source revision's, and every
    shard header must carry exactly the published per-expert geometry
    (``tensor_schema(fused=False)``).  Nothing is quantized at load.
    """
    if sha256(path / "config.json") != SOURCE_CONFIG_SHA256:
        raise ValueError("Super VL source config differs from the pinned revision")
    if sha256(path / "model.safetensors.index.json") != SOURCE_INDEX_SHA256:
        raise ValueError("Super VL source index differs from the pinned revision")
    schema = tensor_schema(config, fused=False)
    mapping = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    if set(mapping) != set(schema):
        raise ValueError("Super VL source index has missing or unexpected tensors")
    digest, records = hashlib.sha256(), []
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        item = path / name
        if item.is_symlink() or not item.is_file():
            raise ValueError(f"Missing or escaped Super VL metadata: {name}")
        digest.update(name.encode())
        digest.update(item.read_bytes())
    for name in sorted(set(mapping.values())):
        if Path(name).name != name or "\\" in name or not name.endswith(".safetensors"):
            raise ValueError("Super VL shard path escapes artifact")
        item = path / name
        if item.is_symlink() or not item.is_file():
            raise ValueError("Super VL source shard is absent or escaped")
        header, _ = read_header(item)
        tensors = {key: entry for key, entry in header.items() if key != "__metadata__"}
        if set(tensors) != {key for key, shard in mapping.items() if shard == name}:
            raise ValueError("Super VL source shard/index differs")
        for key, entry in tensors.items():
            if {field: entry[field] for field in ("shape", "dtype")} != schema[key]:
                raise ValueError(f"Super VL source tensor geometry differs: {key}")
        stat = item.stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record)
        digest.update(json.dumps(record).encode())
        digest.update(json.dumps(header, sort_keys=True).encode())
    return {
        "config": config,
        "weight_map": mapping,
        "conversion": dict(SOURCE_CONVERSION),
        "has_mtp": True,
        "quantization": {},
        "source_bf16": True,
        "identity": {
            "path": str(path),
            "fingerprint": digest.hexdigest(),
            "files": records,
        },
    }


NVFP4_CONVERSION = {
    "recipe": "nvfp4-native-mlx-vlm-export",
    "expert_bits": None,
    "precision": "nvfp4",
    "activation_aware": False,
    "repo": NVFP4_EXPORT_REPO,
    "revision": NVFP4_EXPORT_REVISION,
    "source_repo": NVFP4_EXPORT_SOURCE,
    "source_revision": NVFP4_EXPORT_SOURCE_REVISION,
}
NVFP4_DESCRIPTOR = replace(DESCRIPTOR, variant="nvfp4-export")
NVFP4_LAYOUT_SUFFIX = ":nvfp4-weight-scale2-v1"


def _is_nvfp4_export(path):
    """Whether ``path`` claims to be the MLX-VLM NVFP4 export (then pinned).

    Consulted only without an mlx2 conversion receipt; a malformed or
    unrelated ``conversion_info.json`` is not this export.
    """
    info = path / "conversion_info.json"
    if (path / "mlx2-conversion.json").exists() or info.is_symlink() or not info.is_file():
        return False
    try:
        data = json.loads(info.read_text())
        quantization = data.get("quantization")
        return (
            data.get("source_model") == NVFP4_EXPORT_SOURCE
            and isinstance(quantization, dict)
            and quantization.get("mode") == "nvfp4"
        )
    except (ValueError, AttributeError, OSError):
        return False


def inspect_nvfp4_export(path, config):
    """The pinned MLX-VLM NVFP4 export (see ``nvfp4_export_schema``).

    Config and index bytes must be the pinned export revision's, its
    conversion record must name the pinned vendor NVFP4 source, and every
    shard header must match the derived schema exactly.  Experts load as
    ``weight_scale_2`` NVFP4 modules; everything else is BF16 as published.
    """
    if sha256(path / "config.json") != NVFP4_EXPORT_CONFIG_SHA256:
        raise ValueError("Super VL NVFP4 export config differs from the pinned revision")
    if sha256(path / "model.safetensors.index.json") != NVFP4_EXPORT_INDEX_SHA256:
        raise ValueError("Super VL NVFP4 export index differs from the pinned revision")
    info = json.loads((path / "conversion_info.json").read_text())
    if (
        info.get("source_revision") != NVFP4_EXPORT_SOURCE_REVISION
        or info.get("quantization") != NVFP4_QUANTIZATION
        or info.get("mtp_weight_file") != "mtp.safetensors"
    ):
        raise ValueError("Super VL NVFP4 export is not bound to the pinned NVFP4 source")
    schema, modules = nvfp4_export_schema(config)
    index = json.loads((path / "model.safetensors.index.json").read_text())
    mapping = index["weight_map"]
    if set(mapping) != set(schema):
        raise ValueError("Super VL NVFP4 export index has missing or unexpected tensors")
    digest, records = hashlib.sha256(), []
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "conversion_info.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        item = path / name
        if item.is_symlink() or not item.is_file():
            raise ValueError(f"Missing or escaped Super VL metadata: {name}")
        digest.update(name.encode())
        digest.update(item.read_bytes())
    for name in sorted(set(mapping.values())):
        if Path(name).name != name or "\\" in name or not name.endswith(".safetensors"):
            raise ValueError("Super VL shard path escapes artifact")
        item = path / name
        if item.is_symlink() or not item.is_file():
            raise ValueError("Super VL NVFP4 export shard is absent or escaped")
        header, _ = read_header(item)
        tensors = {key: entry for key, entry in header.items() if key != "__metadata__"}
        if set(tensors) != {key for key, shard in mapping.items() if shard == name}:
            raise ValueError("Super VL NVFP4 export shard/index differs")
        for key, entry in tensors.items():
            if {field: entry[field] for field in ("shape", "dtype")} != schema[key]:
                raise ValueError(f"Super VL NVFP4 export tensor geometry differs: {key}")
        stat = item.stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record)
        digest.update(json.dumps(record).encode())
        digest.update(json.dumps(header, sort_keys=True).encode())
    generation = json.loads((path / "generation_config.json").read_text())
    return {
        "config": config,
        "weight_map": mapping,
        "conversion": {
            **NVFP4_CONVERSION,
            "tensor_bytes": int(index["metadata"]["total_size"]),
        },
        "has_mtp": True,
        "quantization": {},
        "nvfp4_modules": modules,
        "eos_token_ids": tuple(generation["eos_token_id"]),
        "identity": {
            "path": str(path),
            "fingerprint": digest.hexdigest(),
            "files": records,
        },
    }


def inspect_artifact(model_path):
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    language_config(config)
    vision_config(config)
    if _is_nvfp4_export(path):
        return inspect_nvfp4_export(path, config)
    if not (path / "mlx2-conversion.json").exists():
        return inspect_source_artifact(path, config)
    receipt = json.loads((path / "mlx2-conversion.json").read_text())
    if (
        receipt.get("schema") != "mlx2.nemotron35-super-vl-conversion.v1"
        or receipt.get("source_revision") != SOURCE_REVISION
        or receipt.get("source_config_sha256") != SOURCE_CONFIG_SHA256
        or receipt.get("output_config_sha256") != sha256(path / "config.json")
        or receipt.get("output_index_sha256")
        != sha256(path / "model.safetensors.index.json")
    ):
        raise ValueError("Super VL requires a source-bound completed mixed conversion")
    plan = make_plan(config, expert_bits=receipt.get("expert_bits"))
    expected_quant = {
        "bits": plan["expert_bits"],
        "group_size": 64,
        "mode": "affine",
        **plan["modules"],
    }
    if (
        config.get("quantization") != expected_quant
        or receipt.get("recipe") != plan["recipe"]
        or receipt.get("tensor_bytes") != plan["tensor_bytes"]
    ):
        raise ValueError("Super VL mixed precision policy differs from its recipe")
    mapping = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    if set(mapping) != set(plan["schema"]):
        raise ValueError("Super VL index has missing or unexpected tensors")
    digest, records = hashlib.sha256(), []
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "mlx2-conversion.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        item = path / name
        if item.is_symlink() or not item.is_file():
            raise ValueError(f"Missing or escaped Super VL metadata: {name}")
        digest.update(name.encode())
        digest.update(item.read_bytes())
    output_records = {item["file"]: item for item in receipt["output_shards"]}
    if set(output_records) != set(mapping.values()):
        raise ValueError("Super VL conversion receipt shard closure differs")
    for name in sorted(output_records):
        if Path(name).name != name or "\\" in name or not name.endswith(".safetensors"):
            raise ValueError("Super VL shard path escapes artifact")
        item = path / name
        if (
            item.is_symlink()
            or not item.is_file()
            or item.stat().st_size != output_records[name]["bytes"]
        ):
            raise ValueError("Super VL shard is absent, escaped or truncated")
        header, _ = read_header(item)
        tensors = {key: entry for key, entry in header.items() if key != "__metadata__"}
        if set(tensors) != {key for key, shard in mapping.items() if shard == name}:
            raise ValueError("Super VL shard/index differs")
        for key, entry in tensors.items():
            if {field: entry[field] for field in ("shape", "dtype")} != plan["schema"][
                key
            ]:
                raise ValueError(f"Super VL tensor geometry differs: {key}")
        stat = item.stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record)
        digest.update(json.dumps(record).encode())
        digest.update(json.dumps(header, sort_keys=True).encode())
    return {
        "config": config,
        "weight_map": mapping,
        "conversion": receipt,
        "has_mtp": True,
        "quantization": plan["modules"],
        "identity": {
            "path": str(path),
            "fingerprint": digest.hexdigest(),
            "files": records,
        },
    }


class Nemotron35SuperAdapter(ExternalDraftAdapterMixin, Nemotron3SuperAdapter):
    descriptor = DESCRIPTOR
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)
    # _validate_session_width serves B1 (B2/B4 only under exact progressive
    # external verification), so a bare start must not inherit the server's
    # 16-lane default and refuse to load.  An explicit --max-lanes still wins.
    default_max_lanes = 1
    # Locked in by Pierre on 2026-10-18 after the feature sweep: native MTP
    # depth 2 is the default text route.  Greedy output was identical to
    # ordinary on every probe (feature sweep 33235e9f1, mtp-d2 unit), row-exact
    # canaries 0 failed, decode about 1.3-2x ordinary at 4K-32K (exploratory
    # single-repetition ladders on snapshot 6f843160).  Images are served on
    # the ordinary route only (``multimodal_routes``), so a multimodal server
    # passes --ordinary.  Not a qualification: the route stays unqualified.
    default_route = "native_mtp"
    # Its prompt-lookup route selects the exact short-segment kernels
    # (``default_route_execution_policy["prompt_lookup"]``), so a subclass
    # whose default route is ordinary (Bolt) defaults to latched PLD.
    prompt_lookup_default = True
    # The external route (LiLiCorr chain or tree) prefills images through
    # ``Model.media_prefill_body``: the ordinary image prefill with the
    # drafter's target taps; decode is the route's own exact verify.
    multimodal_routes = frozenset({"ordinary", "external_draft"})
    default_mtp_ordinary_handoff_max_width = None
    # The exact kernels of that route: the fused short Mamba conv and gated
    # SSM, short row-exact projections, the layerwise short-exact routed-batch
    # segment, and the bit-identical NAX sorted MoE gather for prefill.  An
    # operator policy that names any of these keys wins.
    adapter_owned_default_policy_keys = frozenset({
        "mamba_conv", "mamba_projection", "mamba_segment", "mamba_ssm",
        "moe_nax_gather", "num_draft",
    })
    default_route_execution_policy = {
        "native_mtp": {
            "mamba_conv": "fused_short",
            "mamba_projection": "short_row_exact",
            "mamba_segment": "layerwise_short_exact_routed_batch",
            "mamba_ssm": "fused_short_gate",
            "moe_nax_gather": "gather",
            "num_draft": 2,
        },
        # An explicit --prompt-lookup verifies through the same exact
        # short-segment kernels (nemotron_h.Model.pld_verify_forward).
        # pld-families-20261021 at c7aa1e237 (Super-VL affine4-protected8):
        # greedy output identical to --ordinary on code, copy, 1.5K-token copy
        # and Spomin 4K prompts; decode 1.72x (copy) and 2.36x (long copy) of
        # ordinary, 1.00x code, 0.99x Spomin.  Tokenwise verify without them:
        # 0.84-1.00x.  Not a qualification.
        "prompt_lookup": {
            "mamba_conv": "fused_short",
            "mamba_projection": "short_row_exact",
            "mamba_segment": "layerwise_short_exact_routed_batch",
            "mamba_ssm": "fused_short_gate",
        },
        # An explicit --external-draft (LiLiCorr / DSpark) verifies through
        # the same exact short-segment kernels.  Default since 2026-11-10
        # (Pierre) after qualification/runs/super-vl-nvfp4-20261110:
        # LiLiCorr on ...-NVFP4@33a6f018 GSM8K 18.9 -> 28.5 tok/s with the
        # keys alone (unit1), 44.5 with the exact FP8/NVFP4 verify rows
        # (unit4, ordinary 38.3), greedy-equal to ordinary, GSM8K predictions
        # identical.  Without them every verify row is a full tokenwise
        # target forward.  The drafter binding stays operator-owned; an
        # operator policy naming any key wins.  Not a qualification.
        "external_draft": {
            "mamba_conv": "fused_short",
            "mamba_projection": "short_row_exact",
            "mamba_segment": "layerwise_short_exact_routed_batch",
            "mamba_ssm": "fused_short_gate",
        },
    }
    think_close_separator = ""
    # Prefill chunk default (Pierre, 2026-11-10) after
    # qualification/runs/nemotron-tree-20261110/{prefill1,prefill2,prefillstep1}
    # on ...-NVFP4@33a6f018: 4096-row chunks vs 2048 prefilled +8% in-process
    # at 8K/16K and +2.5% at 32K with bit-identical final logits, and served
    # (default route) 1.5-2.5% faster TTFT at 8K/16K/32K with an identical
    # 96-token greedy completion.  An explicit --prefill-step wins.  Class-
    # owned: Bolt keeps the family 2048.
    default_prefill_step = 4096
    EXTERNAL_DEFAULT_NUM_DRAFT = 7
    # LiLiCorr chain depth default (Pierre, 2026-11-10) after
    # qualification/runs/nemotron-tree-20261110/gsm3 on ...-NVFP4@33a6f018 at
    # 65d3e2a3f: GSM8K first 30, B1 greedy, num_draft 5 56.7 tok/s vs the
    # previous full block (7) 46.9 / 47.2 (gsm2 / gsm1), predictions and token
    # counts 30/30 identical.  GSM8K only; not a qualification.  An explicit
    # num_draft wins.  Class-owned: Bolt sets None and keeps its own law.
    EXTERNAL_LILICORR_NUM_DRAFT = 5
    # LiLiCorr B1 tree default (Pierre, 2026-11-10) after
    # qualification/runs/nemotron-tree-20261110/stream2 on ...-NVFP4@33a6f018
    # at 6943dc3f3: GSM8K first 30, B1 greedy, the exact tree route with
    # 8-row tiles (no plain-decode cost controller) 66.9 tok/s vs the
    # chain-5 default bracket 63.0 / 62.2, acceptance length 5.78 vs 4.80,
    # predictions identical; every node bit-equal to one-token decode
    # (check1).  B2+ stays the chain.  Applied only when the drafter is
    # LiLiCorr and the operator names no route and no chain-only control;
    # ``batch_size_route: "chain"`` opts out.  GSM8K only; not a
    # qualification.  Class-owned: Bolt sets None.
    EXTERNAL_LILICORR_TREE_ROWS = 8
    EXACT_ATTENTION_ROWS = True
    # Tree-route cost handoff when MLX2_SUPER_TREE_COST_HANDOFF is unset
    # (Pierre, 2026-11-11; evidence at create_external_batch).  Bolt: off.
    TREE_COST_HANDOFF_DEFAULT = "guarded"
    EXTERNAL_ROUTE_TAG = "super-vl-external-v1"
    EXTERNAL_PROFILE = "nemotron35-super-vl-apcv2-external"
    # int8 (W8A8) prefill: declared, never selected without an explicit
    # --int8-prefill or a matching qualification record (int8_prefill auto).
    int8_prefill_supported = staticmethod(lambda: NEMOTRON_INT8_PREFILL_SCOPES)
    int8_prefill_select = staticmethod(nemotron_int8_prefill_select)
    EXTERNAL_POLICY_EXTENSIONS = frozenset({"draft_weight_quantization"}) | TREE_POLICY_KEYS
    incremental_tokenizer_renderer_revision = "nemotron35-super-vl-renderer-v1"

    # Streams the routed experts (backbone and MTP head) of the unconverted
    # bf16 release (232 GiB, unquantized gather_mm tables) and of a converted
    # artifact (quantized tables).  Declared on this class only: Bolt and
    # other derivatives do not inherit it.
    weight_streaming_modes = frozenset({"moe_experts"})

    def __init__(self, model_path, *, execution_policy=None, weight_streaming=None):
        from .process_globals import guarded_construction

        try:
            guarded_construction(
                self,
                lambda: self._init_super(
                    model_path, execution_policy, weight_streaming=weight_streaming
                ),
            )
        except BaseException:
            self.close()
            raise

    def _init_super(self, model_path, execution_policy, weight_streaming=None):
        from ..runtime.streamed_load import require_declared

        self._stream_request = require_declared(type(self), weight_streaming)
        self.weight_stream = None
        nax_explicit = (
            isinstance(execution_policy, dict) and "moe_nax_gather" in execution_policy
        )
        execution_policy, self.moe_nax_gather = _split_moe_nax_gather(execution_policy)
        (
            expert_sort,
            mamba_conv,
            mamba_norm,
            mamba_ssm,
            mamba_projection,
            mamba_segment,
            relu2_down,
            policy,
        ) = _split_super_execution_policy(execution_policy)
        external = bool(policy.get("draft_model"))
        if external:
            self._parse_external_policy(policy, family="Nemotron 3.5 Super VL")
        elif policy and set(policy) - {"num_draft"}:
            raise ValueError("Super VL native policy supports num_draft only")
        else:
            self._parse_external_policy(None, family="Nemotron 3.5 Super VL")
        # Exact tree verify: defaulted for LiLiCorr, validated before any
        # weight is loaded.
        self.tree_route_authority = None
        if external:
            self._default_tree_route()
        self.tree_route = (
            validate_super_tree_policy(self.external_policy) if external else False
        )
        if self.tree_route:
            # A switch typo fails before any weight is loaded.
            _super_tree_cost_handoff(type(self).TREE_COST_HANDOFF_DEFAULT)
        if self.external_policy.get("batch_size_route") == "chain":
            del self.external_policy["batch_size_route"]
            self.tree_route_authority = "operator_chain"
        elif self.tree_route and self.tree_route_authority is None:
            self.tree_route_authority = "operator"
        from .nemotron35_draft_quantization import normalize_runtime_quantization

        draft_weight_quantization = normalize_runtime_quantization(
            self.external_policy.get("draft_weight_quantization")
        )
        self.expert_gather_sort = expert_sort
        self.mamba_conv = mamba_conv
        self.mamba_norm = mamba_norm
        self.mamba_ssm = mamba_ssm
        self.mamba_projection = mamba_projection
        self.mamba_segment = mamba_segment
        self.moe_relu2_down = relu2_down
        if self._stream_request is not None:
            # Both read the resident expert table (the fused down kernel) or
            # run per-row canary gathers over it (the routed-batch segment):
            # no evidence with streamed experts, so refuse before loading.
            refused = [
                name
                for name, selected in (
                    ("moe_relu2_down", relu2_down != "auto"),
                    (
                        "mamba_segment=layerwise_short_exact_routed_batch",
                        mamba_segment == "layerwise_short_exact_routed_batch",
                    ),
                )
                if selected
            ]
            if refused:
                raise ValueError(
                    "Super VL expert streaming has no evidence with "
                    + ", ".join(refused)
                    + "; refusing before any weight is loaded"
                )
        self._num_draft = native_draft_depth(
            3 if external else policy.get("num_draft", 3)
        )
        artifact = type(self).artifact_inspector(model_path)
        if artifact.get("source_bf16"):
            self.descriptor = SOURCE_DESCRIPTOR
        self.identity, self.layout = artifact["identity"], type(self).cache_layout
        self.eos_token_ids = tuple(artifact.get("eos_token_ids", type(self).eos_token_ids))
        if artifact.get("nvfp4_modules"):
            if self._stream_request is not None:
                # Expert streaming reads QuantizedSwitchLinear tables and
                # applies no per-expert weight_scale_2.
                raise ValueError(
                    "Super VL NVFP4 export has no expert-streaming path; "
                    "refusing before any weight is loaded"
                )
            self.descriptor = NVFP4_DESCRIPTOR
            self.layout += NVFP4_LAYOUT_SUFFIX
            # Checkpoint defaults for this export (Pierre, 2026-11-10, after
            # the A/B in qualification/runs/super-vl-nvfp4-20261110/profile2,
            # ...-NVFP4@33a6f018, mlx2 e4d842b74 + this change, M5 Max, no
            # thermal hold; 64 greedy decode tokens identical in every arm):
            # * MoE NAX gather on every route (prefill only; the NVFP4 kernel
            #   is canary-proven bit-identical to stock): prefill +7-8% over
            #   the fold alone.  An explicit moe_nax_gather key wins.
            # * vendor FP8 dense projections: decode 27.7 -> 35.3 tok/s,
            #   prefill -8% against fold+NAX.  MLX2_NVFP4_DENSE=bf16 restores
            #   the exported BF16.
            # * expert-scale fold (below): prefill +7-11%.
            #   MLX2_NVFP4_EXPERT_SCALE=apply restores the applied scales.
            # Not qualified.
            if not nax_explicit:
                self.moe_nax_gather = "gather"
            self.nvfp4_nax_defaulted = not nax_explicit
            refused = [
                name
                for name, selected in (
                    # Both need QuantizedSwitchLinear / QuantizedLinear modules
                    # and would refuse every call on this export.
                    ("moe_relu2_down", relu2_down != "auto"),
                    ("mamba_projection=native_qmm_residual_b1",
                     mamba_projection == "native_qmm_residual_b1"),
                )
                if selected
            ]
            if refused:
                raise ValueError(
                    "Super VL NVFP4 export has no " + ", ".join(refused)
                    + " path; refusing before any weight is loaded"
                )
            self.nvfp4_dense = os.environ.get("MLX2_NVFP4_DENSE", "fp8")
            if self.nvfp4_dense not in {"bf16", "fp8"}:
                raise ValueError("MLX2_NVFP4_DENSE must be bf16 or fp8")
            if self.nvfp4_dense == "fp8":
                self.layout += ":vendor-fp8-dense-w8a16-v1"
        self.source_config, self.conversion = artifact["config"], artifact["conversion"]
        draft_record = None
        draft_loader = None
        if external:
            draft_record, draft_loader = self._external_drafter(
                model_path, artifact, draft_weight_quantization
            )
            self.external_policy.setdefault(
                "num_draft", self._external_default_depth(draft_record)
            )
            self._check_num_draft(draft_record)
        self._load_super_model(artifact, draft_record, draft_loader)
        if self.tree_route:
            # The tree runs the chain's exact strip recipe; refuse an opt-in
            # tree route the loaded policies cannot serve, before serving.
            refusal = self.model.tree_policy_refusal(
                int(self.external_policy.get("ast_tile_rows", 16))
            )
            self._resolve_tree_refusal(refusal)

    def _resolve_tree_refusal(self, refusal):
        """A load-time tree refusal: the adapter's own default yields to the
        chain (as served before the default existed, the reason in the tree
        receipt); an operator-selected tree refuses to load."""
        if refusal is None:
            return
        if self.tree_route_authority != "adapter_default_lilicorr_tree8":
            raise ValueError(f"Super VL tree verify refused at load: {refusal}")
        for key in ("batch_size_route", "ast_tile_rows"):
            self.external_policy.pop(key, None)
        self.tree_route = False
        self.tree_route_authority = f"adapter_default_refused:{refusal}"

    def _default_tree_route(self):
        """Select the B1 tree route for a LiLiCorr drafter when the operator
        names no route and no control the tree does not compose with."""
        rows = type(self).EXTERNAL_LILICORR_TREE_ROWS
        policy = self.external_policy
        if (
            rows is None
            or "batch_size_route" in policy
            or set(policy) & (_TREE_REFUSED_KEYS | TREE_POLICY_KEYS)
        ):
            return
        try:
            config = json.loads(
                (Path(policy["draft_model"]) / "config.json").read_text()
            )
        except (OSError, ValueError):
            return  # the drafter inspection reports it
        if config.get("architectures") != ["LiLiCorrDraftModel"]:
            return
        policy["batch_size_route"] = NEMOTRON_TREE_ROUTE
        policy["ast_tile_rows"] = rows
        self.tree_route_authority = "adapter_default_lilicorr_tree8"

    @classmethod
    def _external_default_depth(cls, draft_record):
        """Chain depth when the operator names none: the head's full block,
        capped for LiLiCorr by ``EXTERNAL_LILICORR_NUM_DRAFT``."""
        depth = draft_record["args"].block_size - 1
        if cls.EXTERNAL_LILICORR_NUM_DRAFT is not None and hasattr(
            draft_record["args"], "lilicorr_candidate_topk"
        ):
            depth = min(depth, cls.EXTERNAL_LILICORR_NUM_DRAFT)
        return depth

    def _external_drafter(self, model_path, artifact, draft_weight_quantization):
        """Inspect this target's external head; returns (record, loader)."""
        draft_config = json.loads(
            (Path(self.external_policy["draft_model"]) / "config.json").read_text()
        )
        if draft_config.get("architectures") == ["DSparkDraftModel"]:
            from .nemotron35_dspark import inspect_drafter, load_drafter

            draft_record = inspect_drafter(
                self.external_policy["draft_model"], model_path
            )

            def draft_loader(record, target_model):
                return load_drafter(
                    record,
                    target_model,
                    runtime_quantization=draft_weight_quantization,
                )
        else:
            from .lilicorr_nemotron import inspect_drafter, load_drafter

            draft_record = inspect_drafter(
                self.external_policy["draft_model"], target_artifact=artifact
            )
            contract = self.nemotron_lilicorr_target_contract(draft_record)

            def draft_loader(record, target_model):
                return load_drafter(
                    record,
                    target_model,
                    target_contract=contract,
                    runtime_quantization=draft_weight_quantization,
                )
        return draft_record, draft_loader

    # Composition hooks; a declared derivative (Nemotron 3.5 Bolt) overrides
    # these instead of branching on model names in shared code.
    cache_layout = CACHE_LAYOUT
    eos_token_ids = (11,)

    @staticmethod
    def _runtime_language_config(config):
        return language_config(config)

    @staticmethod
    def _runtime_model_class():
        from ..runtime.models.nemotron35_super import Model

        return Model

    def _declare_model_contracts(self, model, text):
        """Adapter-owned model declarations made after construction (none)."""

    def _check_tokenizer(self, tokenizer):
        if tokenizer.convert_tokens_to_ids("<image>") != 18:
            raise ValueError("Super VL tokenizer image token differs")

    def _load_super_model(self, artifact, draft_record, draft_loader):
        from .process_globals import claim_moe, claim_stock_moe

        self.environment = configure_environment()
        self.environment["MLX_LM_MTP_BOUNDARY_COW"] = "1"
        os.environ["MLX_LM_MTP_BOUNDARY_COW"] = "1"
        if self.moe_nax_gather == "off":
            claim_stock_moe(self, "the Nemotron 3.5 Super VL adapter")
        else:
            claim_moe(self, "the Nemotron 3.5 Super VL adapter", self.moe_nax_gather)
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer

        from ..runtime.models.nemotron_h import ModelArgs
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        path = Path(self.identity["path"])
        Model = self._runtime_model_class()
        text = self._runtime_language_config(self.source_config)
        self.model = Model(ModelArgs.from_dict(text))
        self._declare_model_contracts(self.model, text)
        shard_names = sorted(set(artifact["weight_map"].values()))
        modules = {
            name.removeprefix("language_model."): quant
            for name, quant in artifact["quantization"].items()
        }

        def quantize(_weights):
            # The bf16 source declares no modules: nothing is quantized.
            nn.quantize(
                self.model,
                class_predicate=lambda name, module: (
                    dict(modules[name]) if name in modules else False
                ),
            )
            if artifact.get("nvfp4_modules"):
                from ..runtime.nvfp4_linear import install_weight_scale2

                self.nvfp4_modules = install_weight_scale2(
                    self.model, artifact["nvfp4_modules"]
                )
                if self.nvfp4_dense == "fp8":
                    from ..runtime.nvfp4_linear import install_vendor_fp8
                    from .nemotron35_super_config import vendor_fp8_scales

                    # Exact vendor E4M3 codes recovered from the BF16 export
                    # (checked bit for bit per tensor), on mxfp8 kernels.
                    self.vendor_fp8_modules = install_vendor_fp8(
                        self.model, _weights, vendor_fp8_scales()
                    )

        stream_request = getattr(self, "_stream_request", None)
        if stream_request is None:
            weights = self.model.sanitize(
                load_shards_evicting(path / name for name in shard_names)
            )
            quantize(weights)
            self.model.load_weights(list(weights.items()), strict=True)
        else:
            from ..runtime.streamed_load import force_stock_expert_arithmetic, load_streamed

            def installed(manager):
                manager.forced_stock = force_stock_expert_arithmetic(self.model)

            loaded = load_streamed(
                self.model,
                path,
                shard_names,
                request=stream_request,
                sanitize=self.model.sanitize,
                quantize=quantize,
                records=self.identity["files"],
                weight_map=artifact["weight_map"],
                top_k=int(text["num_experts_per_tok"]),
                on_installed=installed,
            )
            weights = loaded.weights
            self.weight_stream = loaded.manager
        self.model.eval()
        self._super_expert_gathers = _super_expert_gathers(self.model)
        self.nvfp4_expert_scale = None
        if artifact.get("nvfp4_modules"):
            # Default "fold" for this export (see the checkpoint defaults
            # above): both per-expert weight_scale_2 move into the routing
            # weights (ReLU2 is degree-2 homogeneous), removing two fp32
            # passes over every routed row.  A distinct numerics profile, so
            # it is in the layout.
            mode = os.environ.get("MLX2_NVFP4_EXPERT_SCALE", "fold")
            if mode not in {"apply", "fold"}:
                raise ValueError("MLX2_NVFP4_EXPERT_SCALE must be apply or fold")
            if mode == "fold":
                from ..runtime.nvfp4_linear import fold_relu2_expert_scales

                fold_relu2_expert_scales(
                    self._super_expert_gathers,
                    self.model.backbone.embeddings.weight.dtype,
                )
                self.layout += ":nvfp4-expert-scale-folded-v1"
            self.nvfp4_expert_scale = mode
        self._super_mamba_mixers = _super_mamba_mixers(self.model)
        self._super_moe_mixers = _super_moe_mixers(self.model)
        if self.expert_gather_sort != "auto":
            for switch in self._super_expert_gathers:
                switch.set_sort_policy(self.expert_gather_sort)
            self.layout += ":expert-gather-unsorted-decode-v1"
        if self.mamba_conv != "auto":
            for mixer in self._super_mamba_mixers:
                mixer.set_conv_policy(self.mamba_conv)
            self.layout += (
                ":mamba-conv4-silu-short-checkpoints-v1"
                if self.mamba_conv == "fused_short"
                else (
                    ":mamba-conv4-silu-b1-compact-state-v2"
                    if self.mamba_conv == "fused_b1_compact"
                    else ":mamba-conv4-silu-b1-v1"
                )
            )
        if self.mamba_norm != "auto":
            for mixer in self._super_mamba_mixers:
                mixer.norm.set_policy(self.mamba_norm)
            self.layout += (
                ":mamba-gate-group-rms-short-v1"
                if self.mamba_norm == "fused_short"
                else ":mamba-gate-group-rms-b1-v1"
            )
        if self.mamba_ssm != "auto":
            for mixer in self._super_mamba_mixers:
                mixer.set_ssm_policy(self.mamba_ssm)
            self.layout += (
                ":mamba-ssm-short-dt-gate-v1"
                if self.mamba_ssm == "fused_short_gate"
                else ":mamba-ssm-short-dt-v1"
            )
        if self.mamba_projection != "auto":
            for mixer in self._super_mamba_mixers:
                mixer.set_projection_policy(self.mamba_projection)
            projection_layout = {
                "compact_padded": ":mamba-varlen-q8-projection-v1",
                "compact_row_exact": ":mamba-varlen-row-exact-q8-projection-v1",
                "adaptive_row_exact": ":mamba-varlen-adaptive-row-exact-q8-projection-v1",
                "short_row_exact": ":mamba-short-row-exact-q8-projection-v1",
                "native_qmm_residual_b1": ":mamba-native-qmm-residual-b1-v1",
            }
            self.layout += projection_layout[self.mamba_projection]
        # Exact verify attention blocks with strip projections (Pierre,
        # 2026-11-10) after qualification/runs/nemotron-tree-20261110/stream2
        # on ...-NVFP4@33a6f018 at 6943dc3f3: GSM8K first 30, B1 greedy,
        # 64.9 tok/s vs the default bracket 63.0 / 62.2, rounds, acceptance
        # and predictions identical; Metal test bit-equal to one-token decode.
        # MLX2_NEMOTRON_EXACT_ATTENTION_ROWS=off restores the per-row block.
        # Class-owned: Bolt keeps the per-row block.
        self.model.__dict__["_mlx2_exact_attention_rows"] = type(self).EXACT_ATTENTION_ROWS
        if self.mamba_segment != "auto":
            self.model.set_short_segment_policy(self.mamba_segment)
            self.layout += (
                ":mamba-layerwise-short-exact-routed-batch-v1"
                if self.mamba_segment == "layerwise_short_exact_routed_batch"
                else ":mamba-layerwise-short-exact-v1"
            )
        if self.moe_relu2_down != "auto":
            for mixer in self._super_moe_mixers:
                mixer.set_relu2_down_policy(self.moe_relu2_down)
            self.layout += ":relu2-down-fp32-top22-b1-v1"
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        self._check_tokenizer(tokenizer)
        self.tokenizer = TokenizerWrapper(
            tokenizer,
            detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=list(self.eos_token_ids),
        )
        self.max_context = self.model.args.max_position_embeddings
        if draft_record is not None:
            self._bind_external_drafter(draft_record, draft_loader, type(self).descriptor)
        if self.weight_stream is not None:
            self.weight_stream.begin_serving()

    def close(self):
        stream = getattr(self, "weight_stream", None)
        if stream is not None:
            stream.close()
            self.weight_stream = None
        super().close()

    def prompt_tokens(self, request):
        if "_mlx2_prompt_tokens" in request:
            return list(request["_mlx2_prompt_tokens"])
        return super().prompt_tokens(request)

    def prepare_multimodal_request(self, request, *, file_loader=None):
        from ..multimodal import media_fingerprint, resolve_media
        from .mlx_vlm import _plain_messages, _source
        from . import nemotron35_media as vm
        from .nemotron35_media import PROCESSOR_REVISION, preprocess_image

        if request.get("video_pruning_rate", 0) not in (0, 0.0):
            raise ValueError(
                "Super VL serves video with every token kept (video_pruning_rate 0); "
                "EVS pruning is not implemented"
            )
        media, replacements, images, videos, order, counts = [], [], [], [], [], []
        for message in request["messages"]:
            content = message.get("content")
            # Check string and text-list content before injecting placeholders.
            texts = (
                [content]
                if isinstance(content, str)
                else [
                    part.get("text", "")
                    for part in content or []
                    if part.get("type") == "text"
                ]
            )
            if any(
                marker in text
                for text in texts
                for marker in ("<image>", "<img>", "</img>")
            ):
                raise ValueError("Super VL text must not contain media placeholders")
            for part in content if isinstance(content, list) else []:
                if part["type"] == "text":
                    continue
                if part["type"] in {"input_video", "video_url"}:
                    value = resolve_media(
                        _source(part, "video"), kind="video", file_loader=file_loader,
                        fps=vm.VIDEO_SAMPLE_FPS, max_frames=vm.VIDEO_MAX_FRAMES,
                    )
                    frames = list(value.value)
                    pixels, per_tubelet = vm.preprocess_video(
                        frames,
                        mean=self.source_config["norm_mean"],
                        std=self.source_config["norm_std"],
                    )
                    meta = value.metadata
                    replacements.append(vm.video_prompt(
                        len(frames), per_tubelet,
                        source_fps=meta.get("source_fps"),
                        indices=meta.get("sampled_indices"),
                    ))
                    tubelets = -(-len(frames) // vm.VIDEO_TEMPORAL_PATCH)
                    media.append(value)
                    videos.append(pixels)
                    order.append("video")
                    counts.append(tubelets * per_tubelet)
                    continue
                if part["type"] not in {"image_url", "input_image"}:
                    raise ValueError("Super VL serving accepts images and video only")
                value = resolve_media(
                    _source(part, "image"), kind="image", file_loader=file_loader
                )
                pixels, count = preprocess_image(
                    value.value,
                    mean=self.source_config["norm_mean"],
                    std=self.source_config["norm_std"],
                )
                media.append(value)
                images.append(pixels)
                order.append("image")
                counts.append(count)
                replacements.append("<img>" + "<image>" * count + "</img>")
        messages = _plain_messages(request["messages"], replacements)
        if not media:
            return {**request, "messages": messages}
        tokens = super().prompt_tokens({**request, "messages": messages})
        positions = [index for index, token in enumerate(tokens) if token == 18]
        if len(positions) != sum(counts):
            raise ValueError("Super VL rendered image tokens/features differ")
        return {
            **request,
            "messages": messages,
            "_mlx2_prompt_tokens": tokens,
            "sampling_profile": request.get("sampling_profile", "vision"),
            "_mlx2_prefill_inputs": (
                {"image_pixels": images} if not videos
                else {"image_pixels": images, "video_pixels": videos, "media_order": order}
            ),
            "_mlx2_media_token_end": positions[-1] + 1,
            "_mlx2_media_fingerprint": media_fingerprint(
                media,
                policy={
                    "processor": PROCESSOR_REVISION,
                    "artifact": self.identity["fingerprint"],
                    **({"video": {
                        "processor": vm.VIDEO_PROCESSOR_REVISION,
                        "fps": vm.VIDEO_SAMPLE_FPS, "max_frames": vm.VIDEO_MAX_FRAMES,
                    }} if videos else {}),
                },
            ),
        }

    def validate_prefill_inputs(self, request, remaining_tokens, prefill_input):
        expected = sum(token == 18 for token in request["_mlx2_prompt_tokens"])
        observed = sum(token == 18 for token in remaining_tokens)
        if observed == 0:
            return {}
        if observed != expected:
            raise ValueError("Super VL cannot resume in the middle of an image span")
        return prefill_input

    def profile_name(self, mtp):
        bits = self.conversion["expert_bits"]
        precision = self.conversion.get("precision") or (
            f"affine{bits}" if bits is not None else "bf16"
        )
        if getattr(self, "nvfp4_dense", None) is not None:
            # Distinct numerics profiles of the NVFP4 export.
            precision += f"-{self.nvfp4_dense}dense-{self.nvfp4_expert_scale}scale"
        return f"nemotron35-super-vl-{precision}-apcv2-" + (
            f"mtp{self._num_draft}" if mtp else "ordinary"
        )

    def _validate_session_width(self, width):
        if type(width) is not int or width not in {1, 2, 4}:
            raise ValueError(
                "Super VL requires one session, or B=2/B=4 for exact progressive external verification"
            )
        if width in {2, 4} and not (
            self.draft_model is not None
            and self.external_policy.get("progressive_verification_tile") is not None
            and self.external_policy.get("progressive_verification_fidelity", "exact")
            == "exact"
        ):
            raise ValueError(
                "Super VL B=2/B=4 requires exact progressive external verification"
            )
        return width

    def _super_execution_config(self, *, max_lanes, prefill_step):
        self._validate_session_width(max_lanes)
        if self.draft_model is not None:
            return self._external_execution_config(
                max_lanes=max_lanes, prefill_step=prefill_step
            )
        return super().execution_config(max_lanes=max_lanes, prefill_step=prefill_step)

    def _tree_node_budgets(self):
        return {1: int(self.external_policy.get("ast_tile_rows", 16)) - 1}

    def create_external_batch(self, **kwargs):
        self._validate_session_width(kwargs.get("completion_batch_size", 1))
        if getattr(self, "tree_route", False):
            from ..runtime.hybrid_tree_executor import HybridTreeExecutor
            from ..runtime.nemotron_tree_executor import NemotronTreeExecutor

            if not callable(getattr(self.draft_model, "propose_tree", None)):
                raise ValueError("Super VL tree verify needs a drafter with propose_tree")
            serial = self.external_policy.get("ast_target_executor", "packed") == "serial"
            kwargs.setdefault("dynamic_singleton_tree", True)
            kwargs.setdefault("dynamic_tree_max_width", 1)
            kwargs.setdefault("dynamic_tree_overflow", "chain")
            # The runtime's tree-tile cost controller (its AST defaults: three
            # ordinary probes, 0.97 loss ratio over four rounds) cost ~7% on
            # short GSM8K requests (timing1: 134 probes, 36 parked rounds in
            # 570), so the tree route ran without it.  At long context the
            # drafter's acceptance falls (qualify-super-vl-nvfp4-20261110:
            # 7.4 tokens/round at 4K, 2.7 at 32K) and the tree loses to plain
            # decode (26 vs 40 tok/s).  ``MLX2_SUPER_TREE_COST_HANDOFF``:
            # ``guarded`` (default, Pierre 2026-11-11) parks only a clear loss
            # (below 80% of ordinary over an 8-round window after two
            # ordinary and two speculative samples, two loss decisions, one
            # settle round) and re-probes: on the ladder's 32K content 37-39 tok/s
            # (ordinary ~40, tree 25-27; cell32k-guarded2 at 019108632), GSM8K
            # first 30 62.1 vs 65.1 (guard1); ``off`` keeps the tree
            # unconditionally.
            handoff, mode, defaulted = _super_tree_cost_handoff(
                type(self).TREE_COST_HANDOFF_DEFAULT
            )
            adapter_owned = "cost_per_commit_handoff" not in kwargs
            kwargs.setdefault("cost_per_commit_handoff", handoff)
            self._tree_cost_handoff = {
                "mode": mode if adapter_owned else "caller",
                "authority": (
                    ("adapter_default_super_tree_" + mode if defaulted else "operator_env")
                    if adapter_owned else "caller"
                ),
            }
            if "target_tree_executor_provider" not in kwargs:
                kwargs["target_tree_executor_provider"] = (
                    HybridTreeExecutor.create() if serial
                    else NemotronTreeExecutor.create(selected=True)
                )
            self._tree_provider = kwargs["target_tree_executor_provider"]
            kwargs.setdefault("tree_node_budget_by_lanes", self._tree_node_budgets())
            if "ast_adaptive_rows" in self.external_policy:
                kwargs.setdefault("ast_adaptive_rows", self.external_policy["ast_adaptive_rows"])
        batch = super().create_external_batch(**kwargs)
        selection = getattr(self, "_tree_cost_handoff", None)
        if (
            getattr(self, "tree_route", False)
            and selection is not None
            and selection["authority"].startswith("adapter_default")
            and getattr(batch, "cost_per_commit_handoff", None) is not None
        ):
            # The receipt says who chose the controller: the adapter default,
            # not an operator policy.
            batch.cost_per_commit_handoff_authority = selection["authority"]
        return batch

    def execution_config(self, *, max_lanes, prefill_step):
        config = self._super_execution_config(max_lanes=max_lanes, prefill_step=prefill_step)
        if getattr(self, "tree_route", False):
            config.update(
                batch_size_route=NEMOTRON_TREE_ROUTE,
                dynamic_tree_overflow="chain",
                tree_node_budget_by_lanes=self._tree_node_budgets(),
                ast_target_executor=self.external_policy.get("ast_target_executor", "packed"),
            )
        return config

    def execution_numerics_contract(self):
        contract = {
            "router_compute": "float32",
            "recurrent_state": "float32",
            "time_step_limit": [0.0, "infinity"],
            "source_revision": self.conversion.get("source_revision", SOURCE_REVISION),
            "quantization_recipe": self.conversion["recipe"],
            "qualification": "unqualified",
        }
        if getattr(self, "tree_route", False):
            executor = self.external_policy.get("ast_target_executor", "packed")
            if executor == "serial":
                from ..runtime.hybrid_tree_executor import PROTOCOL as SERIAL_PROTOCOL

                law = {"law": SERIAL_PROTOCOL}
            else:
                from ..runtime.models.nemotron_mamba_tree import TREE_KERNEL_LAW
                from ..runtime.nemotron_tree_executor import PROTOCOL, TREE_LAW

                law = {"law": TREE_LAW, "protocol": PROTOCOL, "kernel_law": TREE_KERNEL_LAW}
            contract["tree_verify"] = {
                **law,
                "executor": executor,
                "tile_rows": int(self.external_policy.get("ast_tile_rows", 16)),
            }
        return contract

    def nemotron_lilicorr_target_contract(self, draft_record):
        """Bind an inspected owned-head companion to exact Super target taps."""
        taps = tuple(draft_record["args"].target_layer_ids)
        text = draft_record["target"]["text_config"]
        if draft_record["target_fingerprint"] != self.identity["fingerprint"]:
            raise ValueError("Nemotron LiLiCorr target revision changed")
        return {
            "schema": "mlx2.nemotron-lilicorr-target.v1",
            "target_model_type": "nemotron_h_omni",
            "target_fingerprint": self.identity["fingerprint"],
            "target_layer_ids": taps,
            "target_feature_width": len(taps) * int(text["hidden_size"]),
            "capture": "post-layer-pre-final-norm",
            "ordinary_reference_preserved": True,
            "speculative_rollback": "exact",
            "owned_lm_head": True,
            "owned_final_norm": True,
            "qualified": False,
        }

    def _tree_verify_diagnostics(self):
        status = (
            self.model.tree_status() if hasattr(getattr(self, "model", None), "tree_status")
            else {}
        )
        selected = bool(getattr(self, "tree_route", False))
        provider = getattr(self, "_tree_provider", None)
        provider_receipt = provider.receipt() if provider is not None else {}
        provider_receipt.pop("source_identity", None)
        return {
            "schema": "mlx2.nemotron35-super-tree-verify.v1",
            "implemented": True,
            "qualified": False,
            "cost_handoff": getattr(self, "_tree_cost_handoff", None),
            "selected": selected,
            "observed_used": selected and int(provider_receipt.get("dispatches", 0)) > 0,
            "provider": provider_receipt or None,
            "route": NEMOTRON_TREE_ROUTE if selected else None,
            "authority": getattr(self, "tree_route_authority", None),
            "executor": (
                self.external_policy.get("ast_target_executor", "packed") if selected else None
            ),
            **status,
        }

    def diagnostics(self):
        gather_counts = {}
        for switch in getattr(self, "_super_expert_gathers", ()):
            for name, count in switch.sort_status()["counts"].items():
                gather_counts[name] = gather_counts.get(name, 0) + int(count)
        conv_counts = {}
        for mixer in getattr(self, "_super_mamba_mixers", ()):
            for name, count in mixer.conv_status()["counts"].items():
                conv_counts[name] = conv_counts.get(name, 0) + int(count)
        norm_counts = {}
        for mixer in getattr(self, "_super_mamba_mixers", ()):
            for name, count in mixer.norm.status()["counts"].items():
                norm_counts[name] = norm_counts.get(name, 0) + int(count)
        ssm_counts = {}
        ssm_refusal_reasons = {}
        for mixer in getattr(self, "_super_mamba_mixers", ()):
            status = mixer.ssm_status()
            for name, count in status["counts"].items():
                ssm_counts[name] = ssm_counts.get(name, 0) + int(count)
            for name, count in status["refusal_reasons"].items():
                ssm_refusal_reasons[name] = ssm_refusal_reasons.get(name, 0) + int(
                    count
                )
        projection_counts = {}
        for mixer in getattr(self, "_super_mamba_mixers", ()):
            for name, count in mixer.projection_status()["counts"].items():
                projection_counts[name] = projection_counts.get(name, 0) + int(count)
        relu2_down_counts = {}
        for mixer in getattr(self, "_super_moe_mixers", ()):
            for name, count in mixer.relu2_down_status()["counts"].items():
                relu2_down_counts[name] = relu2_down_counts.get(name, 0) + int(count)
        return {
            "architecture": "nemotron-h-mamba-moe-radio",
            "layout": self.layout,
            "mtp_head_present": True,
            "qualification": "unqualified",
            "conversion": {
                key: self.conversion[key]
                for key in ("recipe", "expert_bits", "activation_aware")
            },
            "external_draft": self.draft_model is not None,
            "nvfp4_export": (
                None
                if getattr(self, "nvfp4_dense", None) is None
                else {
                    "schema": "mlx2.nemotron35-super-nvfp4-export.v1",
                    "implemented": True,
                    "qualified": False,
                    "dense": self.nvfp4_dense,
                    "vendor_fp8_modules": len(getattr(self, "vendor_fp8_modules", ())),
                    "expert_scale": self.nvfp4_expert_scale,
                    "nvfp4_modules": len(getattr(self, "nvfp4_modules", ())),
                    "moe_nax_gather_defaulted": getattr(self, "nvfp4_nax_defaulted", False),
                }
            ),
            "moe_nax_gather": _moe_nax_gather_report(
                getattr(self, "moe_nax_gather", "off")
            ),
            "row_exact": (
                self.model.row_exact_status()
                if hasattr(self.model, "row_exact_status")
                else None
            ),
            "kapitsa_prefill": _kapitsa_prefill_report(),
            "expert_gather_sort": {
                "schema": "mlx2.nemotron35-super-expert-gather-sort.v1",
                "implemented": True,
                "qualified": False,
                "selected": self.expert_gather_sort != "auto",
                "observed_used": gather_counts.get("unsorted_decode", 0) > 0,
                "policy": self.expert_gather_sort,
                "modules": len(getattr(self, "_super_expert_gathers", ())),
                "counts": gather_counts,
            },
            "mamba_conv": {
                "schema": "mlx2.nemotron35-super-mamba-conv.v1",
                "implemented": True,
                "qualified": False,
                "selected": self.mamba_conv != "auto",
                "observed_used": any(
                    conv_counts.get(name, 0) > 0
                    for name in ("fused_b1", "fused_b1_compact", "fused_short")
                ),
                "policy": self.mamba_conv,
                "modules": len(getattr(self, "_super_mamba_mixers", ())),
                "counts": conv_counts,
            },
            "mamba_norm": {
                "schema": "mlx2.nemotron35-super-mamba-gate-group-rms.v1",
                "implemented": True,
                "qualified": False,
                "selected": self.mamba_norm != "auto",
                "observed_used": any(
                    norm_counts.get(name, 0) > 0 for name in ("fused_b1", "fused_short")
                ),
                "policy": self.mamba_norm,
                "modules": len(getattr(self, "_super_mamba_mixers", ())),
                "counts": norm_counts,
            },
            "mamba_ssm": {
                "schema": "mlx2.nemotron35-super-mamba-ssm-short-dt.v2",
                "implemented": True,
                "qualified": False,
                "selected": self.mamba_ssm != "auto",
                "observed_used": any(
                    ssm_counts.get(name, 0) > 0
                    for name in ("fused_short", "fused_short_gate")
                ),
                "policy": self.mamba_ssm,
                "modules": len(getattr(self, "_super_mamba_mixers", ())),
                "counts": ssm_counts,
                "refusal_reasons": ssm_refusal_reasons,
            },
            "mamba_projection": {
                "schema": "mlx2.nemotron35-super-mamba-varlen-projection.v1",
                "implemented": True,
                "qualified": False,
                "selected": self.mamba_projection != "auto",
                "observed_used": any(
                    projection_counts.get(name, 0) > 0
                    for name in (
                        "in_proj_compaction_calls",
                        "out_proj_compaction_calls",
                        "in_proj_short_kernel",
                        "out_proj_short_kernel",
                    )
                ),
                "policy": self.mamba_projection,
                "modules": len(getattr(self, "_super_mamba_mixers", ())),
                "counts": projection_counts,
            },
            "mamba_segment": {
                "schema": "mlx2.nemotron35-super-mamba-segment.v1",
                "implemented": True,
                "qualified": False,
                "selected": self.mamba_segment != "auto",
                "observed_used": any(
                    self.model.short_segment_status()["counts"].get(name, 0) > 0
                    for name in (
                        "layerwise_short_exact",
                        "layerwise_short_exact_routed_batch",
                    )
                ),
                **self.model.short_segment_status(),
            },
            "moe_relu2_down": {
                "schema": "mlx2.nemotron35-super-relu2-down-fp32-top22.v1",
                "implemented": True,
                "qualified": False,
                "selected": self.moe_relu2_down != "auto",
                "observed_used": relu2_down_counts.get("fused_b1", 0) > 0,
                "policy": self.moe_relu2_down,
                "modules": len(getattr(self, "_super_moe_mixers", ())),
                "counts": relu2_down_counts,
            },
            "tree_verify": self._tree_verify_diagnostics(),
            "execution_limits": dict(EXECUTION_LIMITS),
            "image_prefill": "b1-candidate",
            "video": "prefill-b1-candidate-no-evs",
        }
