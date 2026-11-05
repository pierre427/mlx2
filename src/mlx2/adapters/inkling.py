"""Inkling Small mixed-2bit artifact gate and native text-serving adapter."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import struct
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .external_draft_policy import ExternalDraftAdapterMixin
from .ordinary_artifact import inspect_indexed_artifact

REPOSITORY = "Sawfwair/Inkling-Small-MLX-Mixed-2bit"
REVISION = "22c95fc2e30400a893dc9ac9fe4df17f306362ae"
CACHE_LAYOUT = "inkling-kv-four-shortconv-v1"
DESCRIPTOR = ModelDescriptor(
    model_type="inkling",
    family="inkling-small",
    variant="mixed-2bit-text",
    state_planes=frozenset(
        {
            StatePlane.ATTENTION_KV,
            StatePlane.RECURRENT,
            StatePlane.RNG,
            StatePlane.TRANSCRIPT,
        }
    ),
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.STREAMING,
            Capability.REASONING,
            Capability.CONTINUOUS_BATCH,
            Capability.PREFIX_REUSE,
            Capability.APC_V2,
            Capability.LAYERED_CACHE,
            Capability.PROMPT_LOOKUP,
        }
    ),
    cache_layout=CACHE_LAYOUT,
    metadata={
        "execution": "mlx2.adapters.inkling.InklingAdapter",
        "qualification": "pending",
        "scope": "native text; four causal shortconv states per layer; mixed affine q2/BF16",
        "ordinary_reference": "implemented",
        "native_mtp": "no MTP tensors in conversion",
    },
)


# Native MTP sidecar: the released mixed-2bit conversion ships no MTP tensors;
# scripts/convert_inkling_mtp_q8.py (mlx2 f5a172d19) converts the BF16 head of
# thinkingmachines/Inkling-Small to affine q8/group-64 beside the target.
MTP_SIDECAR_SHA256 = "dff1cfe7d598d4a5bef7187a896c0840ea1fb85cc96b2d0089f23d10e1a1b1c1"
MTP_SIDECAR_SCHEMA = "mlx2.inkling-mtp-q8-conversion.v1"
MTP_SOURCE_REPOSITORY = "thinkingmachines/Inkling-Small"
MTP_SOURCE_REVISION = "b2d4f225a02032c5d154bff748ab5a00c5ca26e4"
MTP_SOURCE_SHA256 = "d286dd21cb982a0052d24ee0077ec6fc38f5a766dc6953e6c4b85c6473bfa7b3"
MTP_MAX_DEPTHS = 8
NATIVE_MTP_DRAFT_NAMES = frozenset({"native_mtp"})
# Draft-only LM head copies: (bits, group size); None keeps the target head.
MTP_DRAFT_HEADS = {"bf16": None, "q8": (8, 64), "q4": (4, 64)}
_MTP_PROJECTIONS = (
    "input_proj",
    "transformer_block.attn.wq_du",
    "transformer_block.attn.wk_dv",
    "transformer_block.attn.wv_dv",
    "transformer_block.attn.wr_du",
    "transformer_block.attn.wo_ud",
    "transformer_block.mlp.w13_dn",
    "transformer_block.mlp.w2_md",
)
_MTP_PLAIN = (
    "embed_norm.weight",
    "hidden_norm.weight",
    "transformer_block.attn.k_norm.weight",
    "transformer_block.attn.q_norm.weight",
    "transformer_block.attn.k_sconv.weight",
    "transformer_block.attn.v_sconv.weight",
    "transformer_block.attn.rel_logits_proj.proj",
    "transformer_block.attn_norm.weight",
    "transformer_block.mlp_norm.weight",
    "transformer_block.attn_sconv.weight",
    "transformer_block.mlp_sconv.weight",
    "transformer_block.mlp.global_scale",
)
_SHA256_MEMO = {}


def mtp_sidecar_tensor_names(depths=MTP_MAX_DEPTHS):
    names = set()
    for depth in range(depths):
        base = f"model.mtp.layers.{depth}."
        names.update(base + name for name in _MTP_PLAIN)
        names.update(
            f"{base}{name}.{leaf}"
            for name in _MTP_PROJECTIONS
            for leaf in ("weight", "scales", "biases")
        )
    return names


def _file_sha256(path):
    """Full-file SHA-256, memoized per process on the file's stat identity."""
    info = path.stat()
    key = (
        str(path.resolve()),
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )
    if key not in _SHA256_MEMO:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                digest.update(block)
        _SHA256_MEMO[key] = digest.hexdigest()
    return _SHA256_MEMO[key]


def inspect_mtp_sidecar(path, *, depths, target_config):
    """Validate the q8 MTP sidecar against the target and its pinned SHA-256.

    Fails closed on a missing directory/file, a conversion record or config
    that differs from the target's ``mtp_config``, an unexpected tensor set,
    a depth count outside ``1..num_nextn_predict_layers``, or a payload whose
    full-file SHA-256 differs from ``MTP_SIDECAR_SHA256``.
    """
    if not isinstance(path, (str, os.PathLike)) or not str(path):
        raise ValueError("Inkling native MTP requires an mtp_sidecar directory")
    root = Path(path).expanduser()
    if not root.is_dir():
        raise ValueError(f"Inkling MTP sidecar directory is missing: {root}")
    files = {name: root / name for name in ("mtp.safetensors", "config.json", "conversion.json")}
    missing = sorted(name for name, item in files.items() if not item.is_file())
    if missing:
        raise ValueError(f"Inkling MTP sidecar is missing {missing} in {root}")
    config = json.loads(files["config.json"].read_text())
    conversion = json.loads(files["conversion.json"].read_text())
    target_mtp = (target_config or {}).get("mtp_config")
    mtp_config = config.get("mtp_config")
    if not isinstance(target_mtp, dict) or mtp_config != target_mtp:
        raise ValueError("Inkling MTP sidecar mtp_config differs from the target's")
    available = mtp_config.get("num_nextn_predict_layers")
    if type(available) is not int or not 1 <= available <= MTP_MAX_DEPTHS:
        raise ValueError("Inkling MTP sidecar depth count is unsupported")
    if type(depths) is not int or not 1 <= depths <= available:
        raise ValueError(
            f"Inkling mtp_depths must be an integer from 1 to {available}"
        )
    if mtp_config.get("chain_hidden_post_norm") is not False:
        raise ValueError("Inkling MTP chain norm is not implemented")
    local = mtp_config.get("local_layer_ids")
    if not isinstance(local, list) or any(
        type(i) is not int or not 0 <= i < available for i in local
    ):
        raise ValueError("Inkling MTP local_layer_ids are malformed")
    law = {"bits": 8, "group_size": 64, "mode": "affine"}
    quantization = config.get("quantization", {})
    expected_modules = sorted(
        f"model.mtp.layers.{depth}.{name}"
        for depth in range(available)
        for name in _MTP_PROJECTIONS
    )
    if (
        any(quantization.get(key) != value for key, value in law.items())
        or sorted(quantization.get("modules", ())) != expected_modules
    ):
        raise ValueError("Inkling MTP sidecar must be affine q8/group-64 projections")
    source = conversion.get("source", {})
    if (
        conversion.get("schema") != MTP_SIDECAR_SCHEMA
        or source.get("repository") != MTP_SOURCE_REPOSITORY
        or source.get("revision") != MTP_SOURCE_REVISION
        or source.get("sha256") != MTP_SOURCE_SHA256
        or conversion.get("output_sha256") != MTP_SIDECAR_SHA256
        or {k: conversion.get("conversion", {}).get(k) for k in law} != law
    ):
        raise ValueError("Inkling MTP sidecar conversion record differs from the pin")
    with files["mtp.safetensors"].open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if size > 32 << 20:
            raise ValueError("Inkling MTP safetensors header is unreasonably large")
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    if set(header) != mtp_sidecar_tensor_names(available):
        raise ValueError("Inkling MTP sidecar tensor roles differ from q8 layout")
    digest = _file_sha256(files["mtp.safetensors"])
    if digest != MTP_SIDECAR_SHA256:
        raise ValueError(
            f"Inkling MTP sidecar SHA-256 {digest} differs from the pinned "
            f"{MTP_SIDECAR_SHA256}"
        )
    return {
        "path": str(root.resolve()),
        "file": str(files["mtp.safetensors"].resolve()),
        "sha256": digest,
        "depths": depths,
        "available_depths": available,
        "local_layer_ids": list(local),
        "quantization": law,
        "source_repository": MTP_SOURCE_REPOSITORY,
        "source_revision": MTP_SOURCE_REVISION,
        "source_sha256": MTP_SOURCE_SHA256,
        "converter_revision": conversion.get("source_revision"),
        "qualification": "unqualified",
    }


def text_tensor_specs():
    """Exact released text tensor shapes/dtypes, independent of MLX imports."""
    specs = {
        "language_model.lm_head.weight": ("BF16", [201024, 4096]),
        "language_model.model.embed_tokens.weight": ("BF16", [201024, 4096]),
        "language_model.model.embed_norm.weight": ("BF16", [4096]),
        "language_model.model.norm.weight": ("BF16", [4096]),
    }
    for i in range(42):
        base = f"language_model.model.layers.{i}."
        local = (i + 1) % 6 != 0
        shape = {
            "attn_sconv.conv.weight": [4096, 4, 1],
            "mlp_sconv.conv.weight": [4096, 4, 1],
            "input_layernorm.weight": [4096],
            "post_attention_layernorm.weight": [4096],
            "self_attn.k_norm.weight": [128],
            "self_attn.q_norm.weight": [128],
            "self_attn.k_sconv.conv.weight": [1024, 4, 1],
            "self_attn.v_sconv.conv.weight": [1024, 4, 1],
            "self_attn.rel_proj": [16, 512 if local else 1024],
            **{
                f"self_attn.{p}_proj.weight": [width, 4096]
                for p, width in (
                    ("q", 4096),
                    ("k", 1024),
                    ("v", 1024),
                    ("r", 512),
                    ("o", 4096),
                )
            },
            "mlp.global_scale": [1],
        }
        if i < 2:
            shape.update(
                {
                    f"mlp.{p}_proj.weight": dims
                    for p, dims in (
                        ("gate", [16384, 4096]),
                        ("up", [16384, 4096]),
                        ("down", [4096, 16384]),
                    )
                }
            )
        else:
            shape.update(
                {
                    "mlp.gate_weight": [258, 4096],
                    "mlp.e_score_correction_bias": [256],
                    "mlp.shared_experts.gate_proj.weight": [2, 2048, 4096],
                    "mlp.shared_experts.up_proj.weight": [2, 2048, 4096],
                    "mlp.shared_experts.down_proj.weight": [2, 4096, 2048],
                }
            )
            for p in ("gate", "up", "down"):
                out_dim, in_dim = (4096, 2048) if p == "down" else (2048, 4096)
                for leaf, divisor in (("weight", 16), ("scales", 128), ("biases", 128)):
                    specs[base + f"mlp.switch_mlp.{p}_proj.{leaf}"] = (
                        "U32" if leaf == "weight" else "BF16",
                        [256, out_dim, in_dim // divisor],
                    )
        for name, dims in shape.items():
            dtype = (
                "F32"
                if (
                    i >= 2
                    and name in ("mlp.e_score_correction_bias", "mlp.global_scale")
                )
                else "BF16"
            )
            specs[base + name] = (dtype, dims)
    return specs


def inspect_artifact(model_path):
    artifact = inspect_indexed_artifact(model_path)
    config, weights = artifact["config"], artifact["weight_map"]
    expected = {
        "hidden_size": 4096,
        "num_hidden_layers": 42,
        "vocab_size": 201024,
        "unpadded_vocab_size": 200058,
        "moe_intermediate_size": 2048,
        "dense_intermediate_size": 16384,
        "dense_mlp_idx": 2,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "n_shared_experts": 2,
        "sconv_kernel_size": 4,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "swa_num_attention_heads": 32,
        "swa_num_key_value_heads": 8,
        "swa_head_dim": 128,
        "sliding_window_size": 512,
        "d_rel": 16,
        "rel_extent": 1024,
        "gate_activation": "sigmoid",
        "route_scale": 8.0,
        "shared_expert_sink": True,
        "norm_after_topk": True,
        "use_embed_norm": True,
        "use_sconv": True,
        "use_gate_bias": True,
        "use_global_scale": True,
        "q_bias": False,
        "o_bias": False,
        "rms_norm_eps": 1e-6,
        "logits_mup_width_multiplier": 16.0,
        "log_scaling_alpha": 0.1,
        "log_scaling_n_floor": 128000,
        "model_max_length": 1048576,
        "final_logit_softcapping": None,
        "local_layer_ids": [i for i in range(42) if (i + 1) % 6],
    }
    text = config.get("text_config", {})
    if config.get("model_type") != "inkling" or config.get("architectures") != [
        "InklingForConditionalGeneration"
    ]:
        raise ValueError("not an Inkling conditional-generation artifact")
    if any(text.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Inkling Small topology differs from the supported mixed-2bit checkpoint"
        )
    for key in ("layer_types", "mlp_layer_types"):
        if key in text:
            raise ValueError("Inkling layer-order overrides are not supported")
    quant = config.get("quantization")
    law = {"bits": 2, "group_size": 128, "mode": "affine"}
    modules = {
        f"language_model.model.layers.{i}.mlp.switch_mlp.{proj}"
        for i in range(2, 42)
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    if not isinstance(quant, dict) or any(
        quant.get(key) != value for key, value in law.items()
    ):
        raise ValueError("Inkling requires affine 2-bit/group-128 routed experts")
    if (
        set(quant) != set(law) | {"scope"} | modules
        or quant.get("scope") != "routed_experts"
    ):
        raise ValueError(
            "Inkling quantization must cover exactly the 120 routed projections"
        )
    if any(quant[name] != law for name in modules):
        raise ValueError("Inkling mixed-precision expert override differs")
    if config.get("quantization_config", quant) != quant:
        raise ValueError("Inkling quantization aliases disagree")
    specs = text_tensor_specs()
    actual = {name for name in weights if name.startswith("language_model.")}
    if actual != set(specs):
        raise ValueError("Inkling text tensor roles are incomplete or unexpected")
    packed = {name[: -len(".scales")] for name in weights if name.endswith(".scales")}
    if packed != modules or any(
        name + suffix not in weights
        for name in modules
        for suffix in (".weight", ".scales", ".biases")
    ):
        raise ValueError(
            "Inkling packed tensor roles differ from quantization metadata"
        )
    if any("mtp" in name for name in weights):
        raise ValueError("Inkling mixed-2bit route does not load MTP tensors")
    # Validate every index/header binding without materializing 84 GB of tensors.
    seen = set()
    path = Path(artifact["identity"]["path"])
    for name in sorted(set(weights.values())):
        with (path / name).open("rb") as handle:
            raw = handle.read(8)
            size = struct.unpack("<Q", raw)[0]
            if size > 32 << 20:
                raise ValueError("Inkling safetensors header is unreasonably large")
            header = json.loads(handle.read(size))
        intervals = []
        data_bytes = (path / name).stat().st_size - 8 - size
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            if key in seen or weights.get(key) != name:
                raise ValueError("Inkling index/header closure differs")
            seen.add(key)
            shape, dtype, offsets = (
                entry.get("shape"),
                entry.get("dtype"),
                entry.get("data_offsets"),
            )
            sizes = {"BF16": 2, "F32": 4, "U32": 4}
            if (
                not isinstance(shape, list)
                or any(type(n) is not int or n < 0 for n in shape)
                or dtype not in sizes
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(n) is not int for n in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= data_bytes
                or offsets[1] - offsets[0] != math.prod(shape) * sizes[dtype]
            ):
                raise ValueError(f"Inkling tensor payload bounds differ: {key}")
            intervals.append(tuple(offsets))
            if key in specs:
                dtype, shape = specs[key]
                if entry.get("shape") != shape or entry.get("dtype") != dtype:
                    raise ValueError(f"Inkling tensor shape/dtype differs: {key}")
        intervals.sort()
        if any(a[1] > b[0] for a, b in pairwise(intervals)):
            raise ValueError("Inkling tensor payloads overlap")
    if seen != set(weights):
        raise ValueError("Inkling indexed tensors are missing from shard headers")
    artifact.update(text_config=text, has_mtp=False, qualification="pending")
    receipt_path = path / ".hf-download-complete.json"
    attested = False
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        attested = (
            receipt.get("repo") == REPOSITORY and receipt.get("revision") == REVISION
        )
    artifact["identity"].update(
        source_repository=REPOSITORY,
        source_revision=REVISION if attested else "unattested-local-copy",
    )
    digest = hashlib.sha256(artifact["identity"]["fingerprint"].encode())
    extra = path / "special_tokens_map.json"
    if extra.is_file():
        digest.update(extra.name.encode())
        digest.update(extra.read_bytes())
    artifact["identity"]["fingerprint"] = digest.hexdigest()
    return artifact


def validate_policy(value):
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError("Inkling execution policy must be an object")
    value = dict(value)
    executor = value.pop("inkling_tree_executor", "serial")
    if executor not in ("serial", "packed"):
        raise ValueError("Inkling tree executor must be serial or packed")
    kernels = {
        name: value.pop(name, False)
        for name in (
            "inkling_fused",
            "inkling_q2_down_combine",
            "inkling_grouped_q2",
            "inkling_tree_kernels",
            "inkling_tree_cohort",
            "inkling_shortconv_prefill",
        )
    }
    if any(type(v) is not bool for v in kernels.values()):
        raise TypeError("Inkling kernel options must be booleans")
    if executor != "packed" and (
        kernels["inkling_tree_kernels"] or kernels["inkling_tree_cohort"]
    ):
        raise ValueError(
            "Inkling tree kernels and cohort packing require the packed executor"
        )
    mode = value.pop("inkling_moe_nax", None)
    if mode not in (None, "off", "gather", "fused"):
        raise ValueError("Inkling MoE NAX mode must be off, gather or fused")
    kernels["inkling_moe_nax"] = mode
    kernels["inkling_tree_executor"] = executor
    attention = value.pop("inkling_attention", "reference")
    if attention not in ("reference", "decode", "all", "nax", "mapped"):
        raise ValueError(
            "Inkling attention must be reference, decode, all, nax or mapped"
        )
    kernels["inkling_attention"] = attention
    if not value:
        if executor != "serial":
            raise ValueError(
                "packed Inkling executor requires a prompt-lookup AST route"
            )
        return kernels, {}
    supported = {
        "draft_model",
        "num_draft",
        "ast_topology",
        "ast_tiles",
        "ast_tile_rows",
        "ast_multilane_tile_rows",
        "pairwise_selection",
        "tensorfold_cohort_limit",
        "ast_adaptive_rows",
        "batch_size_route",
        "cost_per_commit_handoff",
        "mtp_sidecar",
        "mtp_depths",
        "mtp_draft_head",
        "mtp_tree_pipeline",
    }
    if set(value) - supported:
        raise ValueError(
            f"unsupported Inkling AST controls: {sorted(set(value) - supported)}"
        )
    if value.get("ast_tiles", 1) != 1 or not 4 <= value.get("ast_tile_rows", 16) <= 16:
        raise ValueError("Inkling AST admits one tile of 4..16 rows")
    if (
        value.get("batch_size_route", "tree15_b1_b4_chain_b5plus_v1")
        != "tree15_b1_b4_chain_b5plus_v1"
    ):
        raise ValueError("Inkling AST supports its bounded B1..B4 route only")
    from .qwen38_27b import PROMPT_LOOKUP_DRAFT_NAMES, _native_mtp_tree_policy

    draft = value.get("draft_model")
    sidecar = {
        key: value.pop(key)
        for key in ("mtp_sidecar", "mtp_depths", "mtp_draft_head", "mtp_tree_pipeline")
        if key in value
    }
    if draft in NATIVE_MTP_DRAFT_NAMES:
        # Draft logits only; the target verifies with its own BF16 head, so
        # greedy output cannot change, only acceptance and draft cost.
        if sidecar.setdefault("mtp_draft_head", "bf16") not in MTP_DRAFT_HEADS:
            raise ValueError(
                f"Inkling mtp_draft_head must be one of {sorted(MTP_DRAFT_HEADS)}"
            )
        # Queue the next round's MTP draft on a draft-plane copy while the
        # host commits (the engine's early tree prelaunch).  Off: exact and
        # engaged on Inkling-Small-MLX-Mixed-2bit@6aff5cd5 (238/240 rounds
        # adopted at 16K B1), but neutral -- draft build left the critical
        # path (7.9 -> 0.2 ms/round) and ~3.4 ms of drafter encode/submit
        # plus ~4 ms of longer target wait and commit took its place
        # (rounds 78.1 vs 77.9 ms, 2026-11-04, mlx2 43397e5b9).
        if type(sidecar.setdefault("mtp_tree_pipeline", False)) is not bool:
            raise ValueError("Inkling mtp_tree_pipeline must be a boolean")
        if not isinstance(sidecar.get("mtp_sidecar"), str) or not sidecar["mtp_sidecar"]:
            raise ValueError("Inkling native MTP requires an mtp_sidecar directory")
        depths = sidecar.setdefault("mtp_depths", MTP_MAX_DEPTHS)
        if type(depths) is not int or not 1 <= depths <= MTP_MAX_DEPTHS:
            raise ValueError(
                f"Inkling mtp_depths must be an integer from 1 to {MTP_MAX_DEPTHS}"
            )
        num_draft = value.setdefault("num_draft", min(depths, 5))
        if type(num_draft) is not int or not 1 <= num_draft <= depths:
            raise ValueError(
                "Inkling native MTP num_draft must be an integer from 1 to "
                f"mtp_depths ({depths}); a depth is never reused"
            )
        policy = _native_mtp_tree_policy(value)
        return kernels, {**policy, **sidecar}
    if sidecar:
        raise ValueError(
            "mtp_sidecar, mtp_depths, mtp_draft_head and mtp_tree_pipeline require "
            "draft_model native_mtp"
        )
    if draft not in PROMPT_LOOKUP_DRAFT_NAMES:
        raise ValueError(
            "Inkling AST requires the model-free prompt-lookup drafter or "
            "draft_model native_mtp with an mtp_sidecar"
        )
    return kernels, _native_mtp_tree_policy(value)


class InklingAdapter(ExternalDraftAdapterMixin):
    descriptor = DESCRIPTOR
    default_route = "ordinary"
    prompt_lookup_default = False
    reasoning_effort_semantics = "reasoning_strength_0_0.99"
    ast_tree_tiles = True
    ast_tree_tile_dtype_gate = False  # adapter owns mixed FP32 convolution-state math
    EXTERNAL_DEFAULT_NUM_DRAFT = 15
    EXTERNAL_ROUTE_TAG = "inkling-pld-ast-v1"
    EXTERNAL_PROFILE = "inkling-apcv2-pld-ast"

    def __init__(self, model_path, *, execution_policy=None, weight_streaming=None):
        from .process_globals import guarded_construction

        try:
            guarded_construction(
                self,
                lambda: self._initialize(
                    model_path, execution_policy, weight_streaming
                ),
            )
        except BaseException:
            self.close()
            raise

    def _initialize(self, model_path, execution_policy, weight_streaming):
        if weight_streaming is not None:
            raise ValueError("Inkling has no qualified expert-streaming loader")
        self.kernel_policy, external = validate_policy(execution_policy)
        artifact = inspect_artifact(model_path)
        self.config, self.identity, self.layout = (
            artifact["config"],
            artifact["identity"],
            CACHE_LAYOUT,
        )
        sidecar = None
        if external.get("draft_model") in NATIVE_MTP_DRAFT_NAMES:
            # Fail closed on the sidecar before the 84 GB target is loaded.
            sidecar = inspect_mtp_sidecar(
                external["mtp_sidecar"],
                depths=external["mtp_depths"],
                target_config=self.config,
            )
        from ..process_env import (
            PROCESS_NUMERICS,
            assert_process_profile_applied,
            require_process_numerics,
        )

        require_process_numerics("Inkling native profile")
        self.environment = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            **PROCESS_NUMERICS,
        }
        os.environ.update(self.environment)
        assert_process_profile_applied("Inkling native profile")
        from ..runtime.models import moe_nax_gather, switch_layers
        from .process_globals import MOE_NAX_GATHER, MOE_RHS_PAD_POLICY, claim

        if (
            self.kernel_policy["inkling_attention"] == "nax"
            and not moe_nax_gather.nax_host()
        ):
            raise ValueError("Inkling NAX attention requires an M5 NAX-capable device")
        mode = self.kernel_policy.get("inkling_moe_nax")
        mode = os.environ.get("MLX2_MOE_NAX_GATHER", "off") if mode is None else mode
        bits = moe_nax_gather.affine_bits_from_env()
        if mode != "off" and not os.environ.get("MLX2_MOE_NAX_GATHER_BITS"):
            bits = tuple(sorted({*bits, 2}))
        self.kernel_policy.update(inkling_moe_nax=mode, moe_affine_bits=list(bits))
        self._process_claim = claim(
            self,
            "Inkling native adapter",
            {
                MOE_NAX_GATHER: (mode, moe_nax_gather.set_mode),
                MOE_RHS_PAD_POLICY: ("floor", switch_layers.set_pad_policy),
                "moe_nax_affine_bits": (bits, moe_nax_gather.set_affine_bits),
            },
        )
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer

        from ..runtime.models.inkling import Model, ModelArgs
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        self.model = Model(ModelArgs.from_dict(artifact["text_config"]))
        path = Path(self.identity["path"])
        weights = self.model.sanitize(
            load_shards_evicting(
                [path / name for name in sorted(set(artifact["weight_map"].values()))]
            )
        )
        quant = self.config["quantization"]

        def predicate(name, module):
            override = quant.get("language_model." + name)
            return override if name + ".scales" in weights else False

        nn.quantize(
            self.model, group_size=128, bits=2, mode="affine", class_predicate=predicate
        )
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        self.model.configure_kernels(
            fused=self.kernel_policy.get("inkling_fused", False),
            down_combine=self.kernel_policy.get("inkling_q2_down_combine", False),
            attention=self.kernel_policy["inkling_attention"],
            grouped=self.kernel_policy["inkling_grouped_q2"],
            tree_kernels=self.kernel_policy["inkling_tree_kernels"],
            tree_cohort=self.kernel_policy["inkling_tree_cohort"],
            shortconv_prefill=self.kernel_policy["inkling_shortconv_prefill"],
        )
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        # End-message separates thought/tool/text messages; only end-sampling ends the turn.
        terminal = tokenizer.convert_tokens_to_ids("<|content_model_end_sampling|>")
        if terminal != self.config["eos_token_id"]:
            raise ValueError("Inkling end-sampling token differs from checkpoint EOS")
        self.tokenizer = TokenizerWrapper(
            tokenizer,
            detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=[terminal],
        )
        self.max_context = int(artifact["text_config"]["model_max_length"])
        self.draft_model = None
        self.mtp_sidecar = None
        if sidecar is not None:
            self._bind_native_mtp(external, sidecar)
        elif external:
            from ..runtime.drafters.prompt_lookup_draft import (
                load_prompt_lookup_drafter,
            )
            from .qwen38_27b import _prompt_lookup_record

            self.external_policy = external
            record = _prompt_lookup_record(external, self.identity["fingerprint"])
            self.identity = {
                **self.identity,
                "draft_revision": record["draft_revision"],
                "target_revision": record["target_revision"],
            }
            self._bind_external_drafter(
                record, load_prompt_lookup_drafter, self.descriptor
            )

    def _bind_native_mtp(self, external, sidecar):
        """Attach the pinned q8 sidecar head and bind the cascaded drafter.

        Implemented, unqualified.  The sidecar SHA-256 and loaded depth count
        enter the draft revision, hence the route identity and the APCv2
        layout namespace; prompt lookup is untouched by this branch.
        """
        from functools import partial

        import mlx.core as mx

        from ..runtime.drafters.inkling_mtp import (
            inspect_cascaded_mtp_record,
            load_cascaded_mtp_drafter,
        )
        from ..runtime.models.inkling.mtp import build_mtp

        external = dict(external)
        for key in ("mtp_sidecar", "mtp_depths"):
            external.pop(key)
        draft_head = external.pop("mtp_draft_head", "bf16")
        tree_pipeline = external.pop("mtp_tree_pipeline", False)
        weights = mx.load(sidecar["file"])
        head = build_mtp(
            self.model.config,
            weights,
            depths=sidecar["depths"],
            local_layer_ids=sidecar["local_layer_ids"],
            quantization=sidecar["quantization"],
        )
        mx.eval(head.parameters())
        weights.clear()
        mx.clear_cache()
        self.model.attach_mtp(head)
        self.model.attach_mtp_draft_head(MTP_DRAFT_HEADS[draft_head])
        self.model.select_external_feature_source("mtp_seed")
        self.mtp_sidecar = sidecar
        self.external_policy = external
        record = inspect_cascaded_mtp_record(
            num_draft=external["num_draft"],
            depths=sidecar["depths"],
            sidecar_sha256=sidecar["sha256"],
            target_fingerprint=self.identity["fingerprint"],
            draft_head=draft_head,
        )
        receipt = {
            key: sidecar[key]
            for key in ("sha256", "depths", "source_repository", "source_revision", "qualification")
        }
        receipt["draft_head"] = draft_head
        self.identity = {
            **self.identity,
            "draft_revision": record["draft_revision"],
            "target_revision": record["target_revision"],
            "mtp_sidecar": receipt,
        }
        descriptor = replace(
            self.descriptor,
            metadata={
                **(self.descriptor.metadata or {}),
                "native_mtp": "q8 sidecar head attached; cascaded external drafter "
                "(implemented, unqualified)",
                "mtp_sidecar": receipt,
            },
        )
        self._bind_external_drafter(
            record,
            partial(load_cascaded_mtp_drafter, sidecar=receipt, tree_pipeline=tree_pipeline),
            descriptor,
        )

    def close(self):
        if getattr(self, "_process_claim", None) is not None:
            from .process_globals import release

            release(self)

    def execution_numerics_contract(self):
        return {
            "operation": "inkling-native-v1",
            "kernel_policy": self.kernel_policy,
            "source_sha256": {
                name: hashlib.sha256(
                    (
                        Path(__file__).parents[1] / "runtime/models/inkling" / name
                    ).read_bytes()
                ).hexdigest()
                for name in (
                    "language.py",
                    "attention.py",
                    "attention_nax.py",
                    "cache.py",
                    "config.py",
                    "tree.py",
                    "experts.py",
                    "ancestry.py",
                    "tree_attention.py",
                    "mtp.py",
                    "../qwen36_moe_grouped.py",
                    "../qwen4_routed_decode.py",
                )
            },
            "qualification": "unqualified",
        }

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("Inkling conversion contains no MTP head")
        return (
            self.EXTERNAL_PROFILE
            if self.draft_model is not None
            else "inkling-apcv2-ordinary"
        )

    def execution_config(self, *, max_lanes, prefill_step):
        if self.draft_model is not None:
            return self._external_execution_config(
                max_lanes=max_lanes, prefill_step=prefill_step
            )
        return {
            "persistent": True,
            "num_draft": 0,
            "backend": "ordinary",
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": False,
            "segment_aware_cohort_size": max_lanes,
        }

    @staticmethod
    def prefill_step_default():
        return 128

    def cache_budget(self, *, mtp):
        from .inkling_memory import InklingCacheBudget

        return InklingCacheBudget.from_config(self.config, mtp=mtp)

    def create_external_batch(self, **kwargs):
        from ..runtime.hybrid_tree_executor import HybridTreeExecutor
        from ..runtime.inkling_tree_executor import InklingTreeExecutor
        from .qwen38_27b import _tile_node_budgets

        provider = (
            InklingTreeExecutor
            if self.kernel_policy.get("inkling_tree_executor") == "packed"
            else HybridTreeExecutor
        )
        kwargs.setdefault("dynamic_singleton_tree", True)
        kwargs.setdefault("dynamic_tree_max_width", 4)
        kwargs.setdefault("dynamic_tree_overflow", "cohort")
        kwargs.setdefault(
            "tree_node_budget_by_lanes", _tile_node_budgets(self.external_policy, 4)
        )
        kwargs.setdefault(
            "tensorfold_cohort_limit",
            self.external_policy.get("tensorfold_cohort_limit", 4),
        )
        kwargs.setdefault("target_tree_executor_provider", provider.create())
        for key in ("ast_topology", "ast_adaptive_rows", "pairwise_selection"):
            if key in self.external_policy:
                kwargs.setdefault(key, self.external_policy[key])
        return super().create_external_batch(**kwargs)

    def bind_serving_lane_matmul(self, receipt):
        if self.kernel_policy.get("inkling_tree_executor") != "packed":
            return
        from mlx import nn

        from ..runtime.lane import installed_lane

        missing = []
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if (
                installed_lane(module) is None
                or getattr(module, "_lane_min_rows", None) != 1
                or getattr(module, "_lane_max_rows", 0) < 16
            ):
                missing.append(name)
        if not receipt or missing:
            raise ValueError(
                "Packed Inkling AST needs --lane-matmul exact covering all dense projections; "
                f"uncovered: {missing[:4]}"
            )
        self.packed_projection_law = receipt.get("law_id")

    def prompt_tokens(self, request):
        if "messages" not in request:
            return self.tokenizer.encode(request["prompt"], add_special_tokens=False)
        return self.tokenizer.apply_chat_template(
            self._messages(request),
            tokenize=True,
            add_generation_prompt=True,
            reasoning_effort=self._reasoning_effort(request),
        )

    def render_prompt(self, request):
        if "messages" not in request:
            return request["prompt"]
        return self.tokenizer.apply_chat_template(
            self._messages(request),
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=self._reasoning_effort(request),
        )

    @staticmethod
    def _reasoning_effort(request):
        # The request normalizer keeps ``enable_thinking=False`` without
        # adding an effort; the template's lowest level is "none".
        effort = request.get("reasoning_effort")
        if effort is None:
            return "none" if request.get("enable_thinking") is False else "high"
        return effort

    @staticmethod
    def _messages(request):
        # tool_choice "none" asks for ordinary text: render without the tools.
        if request.get("tools") and request.get("tool_choice", "auto") != "none":
            raise ValueError("Inkling tool routing is not implemented")
        # ``thinking_enabled`` is False (content markers select the channel),
        # so serving would silently skip an explicit budget: refuse it.
        if request.get("thinking_budget"):
            raise ValueError("Inkling thinking budgets are not implemented")
        if request.get("thinking_steer_alpha"):
            raise ValueError("Inkling thinking steering is not implemented")
        messages = copy.deepcopy(request["messages"])
        for message in messages:
            content = message.get("content")
            if isinstance(content, list) and any(
                part.get("type") not in ("text", "input_text")
                for part in content
                if isinstance(part, dict)
            ):
                raise ValueError("Inkling native text adapter does not accept media")
        return messages

    @staticmethod
    def thinking_enabled(request):
        return False  # generated content markers select the channel

    def output_parser(self, request):
        from .inkling_output import InklingOutputParser

        return InklingOutputParser(
            chat="messages" in request, stops=request.get("stop", ())
        )

    def diagnostics(self):
        from ..runtime.models import moe_nax_gather

        return {
            "moe_nax_gather": moe_nax_gather.status(),
            "family": "inkling-small",
            "qualification": "unqualified",
            "adapter_default_route": "ordinary",
            "ordinary_reference_preserved": True,
            "cache_layout": self.layout,
            "kernel_policy": self.kernel_policy,
            "kernel_counters": self.model.kernel_receipt(),
            "target_forward_observed": self.model.completed_forwards > 0,
            "ast": {
                "implemented": True,
                "provider": self.kernel_policy.get("inkling_tree_executor", "serial"),
                "qualified": False,
            },
            "tensorfold": {
                "implemented": True,
                "selection_authority": "server_lane_receipt",
                "qualified": False,
            },
            "media": "weights downloaded; native text route refuses media",
            "native_mtp": {
                "implemented": True,
                "selected": self.mtp_sidecar is not None,
                "qualified": False,
                "sidecar": None
                if self.mtp_sidecar is None
                else self.identity.get("mtp_sidecar"),
                "drafter_stats": None
                if self.mtp_sidecar is None
                else getattr(self.draft_model, "stats", None),
            },
        }
