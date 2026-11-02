"""Exact-weight GLM-5.3 REAP50 packed ordinary B1 text serving."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..sampling_defaults import SamplingDefaults, VendorSampling
from .external_draft_policy import ExternalDraftAdapterMixin
from .glm53_gguf_artifact import (
    FILENAME,
    MODEL_TYPE,
    REFERENCE,
    aligned_tokenizer,
    inspect_artifact,
)
from .ordinary_text import OrdinaryTextAdapter, _template_history

CACHE_LAYOUT = "glm53-gguf-unpadded-b1-v1"
DESCRIPTOR = ModelDescriptor(
    model_type=MODEL_TYPE,
    family="glm53-next",
    variant="reap50-q3-k-m-packed-ordinary",
    state_planes=frozenset(
        {
            StatePlane.ATTENTION_KV,
            StatePlane.RECURRENT,
            StatePlane.SPARSE_INDEX,
            StatePlane.RNG,
            StatePlane.TRANSCRIPT,
        }
    ),
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.STREAMING,
            Capability.REASONING,
            Capability.PREFIX_REUSE,
            Capability.APC_V2,
            Capability.LAYERED_CACHE,
            Capability.EXTERNAL_DRAFT,
        }
    ),
    cache_layout=CACHE_LAYOUT,
    metadata={
        "execution": "mlx2.adapters.glm53_gguf.Glm53GGUFAdapter",
        "qualification": "unqualified",
        "scope": "exact GGUF blocks; B1 ordinary and AST; embedded MTP or external draft heads; unqualified",
    },
)
SAMPLING = VendorSampling.single(
    SamplingDefaults(
        temperature=1.0,
        top_p=0.95,
        top_k=0,
        source="GGUF general.sampling.temp/top_p; neutral top_k",
    ),
    model="GLM-5.3-Flash-REAP50-Q3_K_M",
)


# Sparse DSA policy keys -> values, first = today's law (the model's
# SPARSE_POLICIES under the adapter's glm53_ prefix).
SPARSE_KEYS = {
    "glm53_indexer_pools": ("recompute", "recompute_fixed_order", "incremental_v1", "incremental_v1_audit"),
    "glm53_indexer_select": ("argsort", "topk_v1", "topk_v1_kernel"),
    "glm53_sparse_attention": ("mask_reference", "gathered_v1", "fused_decode_v1", "fused_v2"),
    # gathered_v2_from_<n>: the fused_v2 prompt law from position n, for n in
    # exactly the nine enumerated thresholds (docs/design/glm53-speed-round2.md 2.1).
    "glm53_sparse_prefill": (
        "dense_reference", "gathered_v1_8k", "gathered_v1_16k", "gathered_v1_all",
        *(f"gathered_v2_from_{n}" for n in (2049, 3072, 4096, 5120, 6144, 7168, 8192, 12288, 16384)),
    ),
    "glm53_tree_rows_attention": ("per_row", "batched_v1"),
    "glm53_tree_frontier": ("clone", "tails_v1"),
}
# Adapter defaults for the sparse DSA keys (Pierre, 2026-10-31: "All of arm4"),
# selected unqualified. On GLM-5.3-Flash-REAP50-Q3_K_M.gguf@5a865918 at
# 5385578a2 these decode 26.7/27.2/26.6 tok/s at 4K/16K/32K against
# 27.3/22.1/19.9 for today's law, tokens exact at every context; prefill
# 398/266/223 against 417/288/191 tok/s with a 14 GB lower 32K peak. The
# first value of each SPARSE_KEYS entry stays today's law, selectable by
# naming it. Fused decode stays off (slower than gathered at 16K); batched
# verify rows cannot be exact on Metal (one-query SDPA is another kernel).
SPARSE_DEFAULTS = {
    "glm53_indexer_pools": "incremental_v1",
    "glm53_indexer_select": "topk_v1_kernel",
    "glm53_sparse_attention": "gathered_v1",
    "glm53_sparse_prefill": "gathered_v1_8k",
    "glm53_tree_rows_attention": "per_row",
    "glm53_tree_frontier": "clone",
}


def _check_sparse_combination(policy):
    """Refuse sparse DSA combinations whose law is undefined or not built.

    Model-level rules (selection vs pooling vs consumer) live in
    glm53_gguf.check_sparse_policy; this adds the layered-tree requirements
    and fails closed for tree values the layered tree does not implement.
    """
    from ..runtime.models.glm53_gguf import check_sparse_policy

    check_sparse_policy({key.removeprefix("glm53_"): policy[key] for key in SPARSE_KEYS})
    layered = policy["glm53_ast_target"] == "layered"
    if policy["glm53_tree_rows_attention"] == "batched_v1" and (
        not layered or policy["glm53_sparse_attention"] == "mask_reference"
    ):
        raise ValueError(
            "glm53_tree_rows_attention=batched_v1 requires glm53_ast_target=layered "
            "and a gathered glm53_sparse_attention"
        )
    if policy["glm53_tree_frontier"] == "tails_v1" and (
        not layered
        or policy["glm53_indexer_pools"] not in ("incremental_v1", "incremental_v1_audit")
        or policy["glm53_indexer_select"] == "argsort"
        or policy["glm53_sparse_attention"] == "mask_reference"
    ):
        raise ValueError(
            "glm53_tree_frontier=tails_v1 requires glm53_ast_target=layered, incremental pools, "
            "topk_v1 selection and a gathered glm53_sparse_attention"
        )
    from . import glm53_layer_tree

    for key, supported in (
        ("glm53_tree_rows_attention", "SUPPORTED_TREE_ROWS_ATTENTION"),
        ("glm53_tree_frontier", "SUPPORTED_TREE_FRONTIER"),
    ):
        # The layered tree declares the values it implements; a value it
        # does not declare would silently run the default, so it is refused.
        if policy[key] not in getattr(glm53_layer_tree, supported, SPARSE_KEYS[key][:1]):
            raise ValueError(f"{key}={policy[key]} is not implemented by the layered tree")
    for key, value in (("glm53_sparse_attention", policy["glm53_sparse_attention"]),
                       ("glm53_sparse_prefill", policy["glm53_sparse_prefill"])):
        if value in ("fused_decode_v1", "fused_v2") or value.startswith("gathered_v2_from_"):
            import mlx.core as mx

            if not mx.metal.is_available():
                raise ValueError(f"{key}={value} requires Metal; there is no fallback law")


def _check_mla_fused(policy):
    """glm53_mla_fused=decode_v1 fuses the arm4 sparse laws' decode glue only.

    Under any other pooling, selection or decode consumer it would never
    engage, so the combination is refused rather than silently declined.
    """
    if policy.get("glm53_mla_fused", "reference") != "decode_v1":
        return
    if (
        policy["glm53_indexer_pools"] != "incremental_v1"
        or policy["glm53_indexer_select"] != "topk_v1_kernel"
        or policy["glm53_sparse_attention"] not in ("gathered_v1", "fused_decode_v1", "fused_v2")
    ):
        raise ValueError(
            "glm53_mla_fused=decode_v1 requires glm53_indexer_pools=incremental_v1, "
            "glm53_indexer_select=topk_v1_kernel and a gathered glm53_sparse_attention"
        )
    import mlx.core as mx

    if not mx.metal.is_available():
        raise ValueError("glm53_mla_fused=decode_v1 requires Metal")


def _mla_fused_default(policy):
    """decode_v1 when the resolved sparse laws and device let it engage."""
    resolved = {key: policy.get(key, SPARSE_DEFAULTS[key]) for key in SPARSE_KEYS}
    if (
        resolved["glm53_indexer_pools"] != "incremental_v1"
        or resolved["glm53_indexer_select"] != "topk_v1_kernel"
        or resolved["glm53_sparse_attention"] not in ("gathered_v1", "fused_decode_v1", "fused_v2")
    ):
        return "reference"
    import mlx.core as mx

    return "decode_v1" if mx.metal.is_available() else "reference"


def _mla_fused_status(kernel_policy):
    if kernel_policy.get("glm53_mla_fused", "reference") == "reference":
        return "not selected"
    from ..runtime.models import glm53_mla_fused

    return glm53_mla_fused.route_status()


def _compile_regions_status(kernel_policy):
    if kernel_policy.get("glm53_compile_regions", "off") == "off":
        return "not selected"
    from ..runtime.models import glm53_compile_regions

    return glm53_compile_regions.route_status()


def _hc_fused_status(kernel_policy):
    policy = kernel_policy.get("glm53_hc_fused", "reference")
    if policy == "reference":
        return "not selected"
    from ..runtime.models import glm53_hc_fused

    return glm53_hc_fused.route_status(policy)


def _sparse_route_status():
    from ..runtime.models import (
        glm53_gguf_indexer,
        glm53_kv_append,
        glm53_sparse_attention,
    )

    return {
        "indexer": glm53_gguf_indexer.route_status(),
        "attention": glm53_sparse_attention.route_status(),
        "kv_append": glm53_kv_append.route_status(),
    }


class Glm53GGUFAdapter(ExternalDraftAdapterMixin, OrdinaryTextAdapter):
    descriptor = DESCRIPTOR
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "glm53-reap50-q3-k-m-gguf-ordinary-b1-v1"
    prompt_lookup_default = False
    ast_tree_tiles = True
    # Native packed FP32 execution has its own exact-row contract, independent
    # of the affine/BF16 TensorFold dtype gate used by other adapters.
    ast_tree_tile_dtype_gate = False
    EXTERNAL_PROFILE = "glm53-gguf-apcv2-ast-b1-v1"
    EXTERNAL_ROUTE_TAG = "glm53-exact-row-tree-v1"
    EXTERNAL_DEFAULT_NUM_DRAFT = 15
    apc_cache_headroom_guard = True

    def __init__(self, model_path, *, execution_policy=None):
        from .process_globals import guarded_construction

        self.weights = None
        self.draft_model = None
        self.external_policy = {}
        try:
            guarded_construction(self, lambda: self._load(model_path, execution_policy))
        except BaseException:
            self.close()
            raise

    def _load(self, model_path, policy):
        from ..process_env import (
            PROCESS_NUMERICS,
            assert_process_profile_applied,
            clear_inherited_profile,
            require_process_numerics,
        )

        # Refuse conflicting process numerics before any drafter or artifact
        # inspection, as the DeepSeek V4 profile does.
        require_process_numerics("the GLM-5.3 native GGUF profile")
        clear_inherited_profile(())
        self.environment = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            **PROCESS_NUMERICS,
        }
        os.environ.update(self.environment)
        assert_process_profile_applied("the GLM-5.3 native GGUF profile")
        policy = policy or {}
        if not isinstance(policy, dict):
            raise TypeError("GLM-5.3 execution policy must be an object")
        kernel_keys = {
            "glm53_math",
            "glm53_hc",
            "glm53_ast_target",
            "glm53_draft_activation",
            "glm53_dense_prefill",
            "glm53_expert_prefill",
            "glm53_row_kernel",
            "glm53_shared_fold",
            "glm53_kda_group",
            "glm53_mla_group",
            "glm53_tree_batch",
            "glm53_compile_regions",
            "glm53_router",
            "glm53_kda_small",
            "glm53_kda_fused",
            "glm53_kv_append",
            "glm53_hc_fused",
            "glm53_mla_fused",
            *SPARSE_KEYS,
        }
        external = {k: v for k, v in policy.items() if k not in kernel_keys}
        drafter_kind = None
        draft_record = None
        if external:
            from .qwen38_27b import PROMPT_LOOKUP_DRAFT_NAMES, _native_mtp_tree_policy

            allowed = {
                "draft_model",
                "num_draft",
                "ast_topology",
                "ast_tiles",
                "ast_tile_rows",
                "ast_adaptive_rows",
                "cost_per_commit_handoff",
                "batch_size_route",
                "pairwise_selection",
            }
            if set(external) - allowed or not isinstance(
                external.get("draft_model"), str
            ):
                raise ValueError("unsupported GLM-5.3 AST policy")
            draft_path = external["draft_model"]
            if draft_path in PROMPT_LOOKUP_DRAFT_NAMES:
                drafter_kind = "prompt_lookup"
            elif (
                draft_path == "mtp"
                or Path(draft_path).expanduser().resolve()
                == Path(model_path).expanduser().resolve()
            ):
                drafter_kind = "mtp"
                external.setdefault("num_draft", 3)
            else:
                from .glm53_drafters import inspect_glm53_drafter

                draft_record = inspect_glm53_drafter(draft_path, model_path)
                drafter_kind = "dflash"
                external.setdefault("num_draft", 7)
            # The observed B1 profile is one four-row chain. Larger trees
            # remain explicit, budgeted and unqualified.
            external.setdefault("ast_tile_rows", 4)
            external.setdefault("ast_topology", "chain")
            self.external_policy = _native_mtp_tree_policy(external)
            if (
                self.external_policy["batch_size_route"]
                != "tree15_b1_b4_chain_b5plus_v1"
            ):
                raise ValueError("unsupported GLM-5.3 AST batch route")
        policy = {k: v for k, v in policy.items() if k in kernel_keys}
        if (
            not isinstance(policy, dict)
            or set(policy) - kernel_keys
            or policy.get("glm53_math", "ordinary") not in ("ordinary", "reference")
            or policy.get("glm53_hc", "reference") not in ("reference", "fused_fp32", "mega_v1")
            or policy.get("glm53_ast_target", "serial") not in ("serial", "layered")
            or policy.get("glm53_draft_activation", "target")
            not in ("target", "bfloat16")
            or policy.get("glm53_dense_prefill", "reference")
            not in ("reference", "nax_fp16", "nax_fp16_v2", "nax_fp16_v2d")
            or policy.get("glm53_expert_prefill", "reference")
            not in ("reference", "segmented_nax_fp16", "segmented_nax_fp16_v2")
            or policy.get("glm53_row_kernel", "reference")
            not in ("reference", "kblock_rows", "kblock_exact")
            or policy.get("glm53_shared_fold", "off") not in ("off", "kblock")
            or policy.get("glm53_kda_group", "off") not in ("off", "kblock")
            or policy.get("glm53_mla_group", "off") not in ("off", "kblock")
            or policy.get("glm53_tree_batch", "off") not in ("off", "kblock_v1")
            or policy.get("glm53_compile_regions", "v1") not in ("off", "v1")
            or policy.get("glm53_router", "fused_v1") not in ("reference", "fused_v1", "fused_v2")
            or policy.get("glm53_kda_small", "fused_v1") not in ("reference", "fused_v1")
            or policy.get("glm53_kda_fused", "decode_v1") not in ("reference", "decode_v1")
            or policy.get("glm53_kv_append", "reference") not in ("reference", "rotate_v1")
            or policy.get("glm53_hc_fused", "reference") not in ("reference", "v1", "tail_v1")
            or (policy.get("glm53_hc_fused", "reference") != "reference"
                and policy.get("glm53_hc", "reference") != "mega_v1")
            or policy.get("glm53_mla_fused", "decode_v1") not in ("reference", "decode_v1")
            or any(policy.get(key, SPARSE_DEFAULTS[key]) not in values
                   for key, values in SPARSE_KEYS.items())
        ):
            raise ValueError("unsupported GLM-5.3 packed execution policy")
        self.kernel_policy = {
            "glm53_math": policy.get("glm53_math", "ordinary"),
            "glm53_hc": policy.get("glm53_hc", "reference"),
            "glm53_ast_target": policy.get("glm53_ast_target", "serial"),
            "glm53_draft_activation": policy.get("glm53_draft_activation", "target"),
            # Prompt-only tiled K-block prefill: FP16 operands, a distinct
            # numerical law from the packed FP32 reference; explicit, default-off.
            "glm53_dense_prefill": policy.get("glm53_dense_prefill", "reference"),
            "glm53_expert_prefill": policy.get("glm53_expert_prefill", "reference"),
            # K-block row kernels (decode, verification, untiled rows): same
            # element values as the reference, FP32, a different sum order.
            "glm53_row_kernel": policy.get("glm53_row_kernel", "reference"),
            # Shared expert as slot TOPK of the routed K-block launches; its
            # activation uses the routed kernel's sigmoid expression.
            "glm53_shared_fold": policy.get("glm53_shared_fold", "off"),
            # KDA input projections in two multi-weight launches; each output
            # equals its kblock_rows projection exactly.
            "glm53_kda_group": policy.get("glm53_kda_group", "off"),
            # MLA input projections (and the selecting indexer's) in grouped
            # K-block launches; outputs equal the separate projections.
            "glm53_mla_group": policy.get("glm53_mla_group", "off"),
            # Multi-row (tree verify) K-block work: token-shared dense rows,
            # expert-union MoE and a 5-layer barrier; every row equals its
            # one-row computation bit for bit.
            "glm53_tree_batch": policy.get("glm53_tree_batch", "off"),
            # B1 decode layer bodies (KDA layers whole, MLA FFN halves) traced
            # once under mx.compile and replayed: host graph build only; the
            # replay gate requires bitwise logits and state (glm53_compile_regions).
            # Default (Pierre, 2026-10-31, "set those defaults"), unqualified:
            # interleaved same-process replay of recorded 2K/32K traces on
            # GLM-5.3-Flash-REAP50-Q3_K_M.gguf@5a865918 (1a96bb063), pipelined
            # step 29.7 -> 26.7 ms with compile_regions, -> 28.5 with kda_fused,
            # -> 25.8 ms with both (-13%); tokens identical. "off"/"reference" opt out.
            "glm53_compile_regions": policy.get("glm53_compile_regions", "v1"),
            # MoE routing glue (sigmoid, bias, stable top-8, gather, x2.5
            # normalize) in one Metal launch per layer; ids and weights equal
            # the reference op chain bit for bit. Ordinary trunk and MTP head
            # rows; the layered tree's per-row routing keeps the op chain.
            # Default (Pierre, 2026-10-31), unqualified: with glm53_kda_small,
            # +1.0% decode / +1.6% pipelined at 8K over 3 cooldown-gated reps
            # on GLM-5.3-Flash-REAP50-Q3_K_M.gguf@5a865918 (af45e83af), tokens
            # identical; Metal bitwise tests pass. "reference" keeps the op chain.
            # "fused_v2" also folds the ffn_gate_inp F32 gemv (MLX gemv order)
            # into the router launch for B1 single-token rows: one launch per
            # MoE layer, ids and weights bit for bit. Opt-in, unqualified. Left
            # off: not helpful (interleaved replay, GLM-5.3-Flash-REAP50-Q3_K_M.gguf@
            # 5a865918, 2026-11-01): with glm53_hc_fused, dispatches per decode step
            # fell 1,077 -> 900 (v1) / 810 (tail_v1) but the step did not move
            # (32K 25.83 -> 25.85/25.73 ms, 2K 26.31 -> 26.03/26.36 ms, tokens
            # identical); these launches already overlap inside compiled regions.
            "glm53_router": policy.get("glm53_router", "fused_v1"),
            # KDA gate epilogue (decay, beta and gate sigmoids) in one Metal
            # launch per layer, bit for bit. The small KDA projections
            # themselves are already grouped by glm53_kda_group=kblock.
            # Default with glm53_router (same evidence); "reference" opts out.
            "glm53_kda_small": policy.get("glm53_kda_small", "fused_v1"),
            # The whole post-projection KDA body of one B1 decode token (conv
            # window, depthwise conv, SiLU, q/k L2 norms, gates, delta rule,
            # norm x gate) in one Metal launch per layer, bit for bit
            # (glm53_kda_fused); prefill and other shapes keep the op chain.
            # Default with glm53_compile_regions (same evidence above).
            "glm53_kda_fused": policy.get("glm53_kda_fused", "decode_v1"),
            # Sparse DSA (docs/design/glm53-sparse-dsa.md): pooled-key plane,
            # selection law, decode/verify consumer, prompt consumer and the
            # layered tree's rows and frontier. The first value of each is
            # today's law; SPARSE_DEFAULTS is the unqualified default.
            **{key: policy.get(key, SPARSE_DEFAULTS[key]) for key in SPARSE_KEYS},
            # Storage of the MLA latent/raw planes (glm53_kv_append).
            # "rotate_v1" rotates RING_BUFFERS full buffers so a pipelined
            # append never copies the plane (synthetic bench: 4.4 -> 0.5 ms per
            # step at 32K). Left off: not helpful on the real model
            # (GLM-5.3-Flash-REAP50-Q3_K_M.gguf@5a865918, 2026-10-31 replays):
            # the reference never copies there (per-step peak ~10 MB, not ~1 GB)
            # because a ~6000-kernel step retires step N's reads of a plane
            # before step N+1 encodes its append, so MLX donates the buffer;
            # rotate_v1 then costs ~+1 ms per step and K x the plane memory.
            # Revisit only if fusion shrinks a decode step to fit MLX's
            # in-flight queue (the replay's pipelined peak column shows it).
            "glm53_kv_append": policy.get("glm53_kv_append", "reference"),
            # HC boundary launches under glm53_hc=mega_v1 (glm53_hc_fused):
            # "v1" folds the previous half's expand, the flat norm and the F32
            # mixing gemv into one launch before the mega post kernel;
            # "tail_v1" also runs the post step in that launch's last
            # threadgroup. B1 single-token rows, bit for bit. Left
            # off: not helpful (interleaved replay, GLM-5.3-Flash-REAP50-Q3_K_M.gguf@
            # 5a865918, 2026-11-01): with glm53_hc_fused, dispatches per decode step
            # fell 1,077 -> 900 (v1) / 810 (tail_v1) but the step did not move
            # (32K 25.83 -> 25.85/25.73 ms, 2K 26.31 -> 26.03/26.36 ms, tokens
            # identical); these launches already overlap inside compiled regions.
            "glm53_hc_fused": policy.get("glm53_hc_fused", "reference"),
            # B1 decode MLA glue on the 11 sparse-MLA layers (glm53_mla_fused):
            # "decode_v1" runs the norms + raw indexer row, the pool-completion
            # pooling, the pool scores and the gather + mask in four kernels
            # around the unchanged projections, appends, select kernel, q x
            # pooled matmul and SDPA; bit for bit (first-use gated). Requires
            # the arm4 sparse laws. Default (Pierre, 2026-11-01), unqualified,
            # wherever it can engage (arm4 sparse laws and Metal; otherwise the
            # default is "reference"): interleaved replay on
            # GLM-5.3-Flash-REAP50-Q3_K_M.gguf@5a865918 (a6b4dd417), 32K 26.50 ->
            # 25.85 ms, 2K 27.20 -> 26.52 ms per step, tokens identical, and the
            # ~2.2 ms every-4th-token pool-completion step gone.
            "glm53_mla_fused": policy.get("glm53_mla_fused", _mla_fused_default(policy)),
        }
        _check_sparse_combination(self.kernel_policy)
        _check_mla_fused(self.kernel_policy)
        if (
            self.kernel_policy["glm53_draft_activation"] != "target"
            and drafter_kind != "dflash"
        ):
            raise ValueError("BF16 draft activations require an external DFlash head")
        self.profile += (
            ":"
            + self.kernel_policy["glm53_math"]
            + ":hc="
            + self.kernel_policy["glm53_hc"]
            + ":ast_target="
            + self.kernel_policy["glm53_ast_target"]
            + ":draft_activation="
            + self.kernel_policy["glm53_draft_activation"]
        )
        # Appended only off the reference law, so existing profiles (and the
        # APCv2 namespaces derived from them) keep their identity.
        for key, name in (("glm53_dense_prefill", "dense_prefill"),
                          ("glm53_expert_prefill", "expert_prefill"),
                          ("glm53_row_kernel", "row_kernel"),
                          ("glm53_shared_fold", "shared_fold"),
                          ("glm53_kda_group", "kda_group"),
                          ("glm53_mla_group", "mla_group"),
                          ("glm53_tree_batch", "tree_batch"),
                          ("glm53_router", "router"),
                          ("glm53_kda_small", "kda_small"),
                          ("glm53_kda_fused", "kda_fused"),
                          ("glm53_kv_append", "kv_append"),
                          ("glm53_hc_fused", "hc_fused"),
                          ("glm53_mla_fused", "mla_fused")):
            if self.kernel_policy[key] not in ("reference", "off"):
                self.profile += ":" + name + "=" + self.kernel_policy[key]
        for key, values in SPARSE_KEYS.items():
            if self.kernel_policy[key] != values[0]:
                self.profile += ":" + key.removeprefix("glm53_") + "=" + self.kernel_policy[key]
        if self.kernel_policy["glm53_compile_regions"] != "off":
            self.profile += ":compile_regions=" + self.kernel_policy["glm53_compile_regions"]
        artifact = inspect_artifact(model_path)
        self.identity = artifact["identity"]
        self.config = artifact["config"]
        self.metadata = artifact["metadata"]
        source_root = Path(__file__).resolve().parents[3]
        paths = (
            "src/mlx2/adapters/glm53_gguf.py",
            "src/mlx2/adapters/glm53_layer_tree.py",
            "src/mlx2/adapters/glm53_drafters.py",
            "src/mlx2/adapters/dflash2.py",
            "src/mlx2/runtime/drafters/dflash_base.py",
            "src/mlx2/runtime/drafters/dflash2.py",
            "src/mlx2/runtime/drafters/dflash2_config.py",
            "src/mlx2/runtime/models/glm53_gguf.py",
            "src/mlx2/runtime/models/glm53_gguf_mtp.py",
            "src/mlx2/runtime/models/glm53_gguf_indexer.py",
            "src/mlx2/runtime/models/glm53_sparse_attention.py",
            # Kernel files bound unconditionally (round 2, WP5): they define
            # the served arithmetic, so a kernel-only edit must move the
            # numerics revision and the APCv2 identity.
            "src/mlx2/runtime/models/glm53_sparse_attention_v2.py",
            "src/mlx2/runtime/gguf/kblock_metal.py",
            "src/mlx2/runtime/gguf/kblock_prefill_metal.py",
            "src/mlx2/runtime/models/glm53_kv_append.py",
            "src/mlx2/runtime/models/glm53_kv_append_geometry.py",
            "src/mlx2/runtime/models/glm53_glue_metal.py",
            "src/mlx2/runtime/models/glm53_kda_fused.py",
            "src/mlx2/runtime/models/gated_delta.py",
            "src/mlx2/runtime/models/precise_ops.py",
            "src/mlx2/runtime/models/glm53_compile_regions.py",
            "src/mlx2/runtime/models/glm53_hc_mega.py",
            "src/mlx2/runtime/models/glm53_hc_fused.py",
            "src/mlx2/runtime/models/glm53_mla_fused.py",
            "src/mlx2/runtime/cow_cache.py",
            "src/mlx2/runtime/recurrent_state_codec.py",
            "src/mlx2/runtime/gguf/reader.py",
            "src/mlx2/runtime/gguf/native_read_view.py",
            "src/mlx2/runtime/models/deepseek_v4_hc.py",
            "src/mlx2/runtime/gguf/k_metal.py",
            "src/mlx2/runtime/gguf/packed_metal.py",
            "src/mlx2/runtime/gguf/weights.py",
        )
        self.source_identity = {
            path: hashlib.sha256((source_root / path).read_bytes()).hexdigest()
            for path in paths
        }
        self.numerics_revision = hashlib.sha256(
            json.dumps(
                [self.profile, self.source_identity],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self.layout = CACHE_LAYOUT + ":" + self.numerics_revision
        self.max_context = self.config["max_position_embeddings"]
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper

        root = Path(self.identity["path"])
        data = aligned_tokenizer(
            json.loads((root / REFERENCE).read_text()), self.metadata
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_str(json.dumps(data)),
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
            chat_template=self.metadata["tokenizer.chat_template"],
            model_max_length=self.max_context,
            clean_up_tokenization_spaces=False,
        )
        # These exact IDs/spellings are part of this sealed GGUF's vocabulary.
        # The checkpoint closes most turns by opening a new <|assistant|> turn
        # rather than <|user|> (20/24 ladder rows, 2026-10-31), so that role
        # token ends generation too; without it every reply ran to max_tokens
        # repeating itself.
        if [
            tokenizer.convert_tokens_to_ids(t)
            for t in ("<|endoftext|>", "<|user|>", "<|assistant|>", "<|observation|>")
        ] != [154820, 154827, 154828, 154829]:
            raise ValueError("GLM-5.3 control-token IDs changed")
        self.tokenizer = TokenizerWrapper(
            tokenizer,
            detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=[154820, 154827, 154828, 154829],
        )
        from ..runtime.gguf.weights import GGUFWeights
        from ..runtime.models.glm53_gguf import Glm53GGUFModel

        self.weights = GGUFWeights(
            root / FILENAME,
            expected_stat=artifact["weight_stat"],
            native_read_view=(root / "mlx2-native-packed.safetensors")
            if (root / "mlx2-native-packed.safetensors").exists()
            else None,
            source_sha256=artifact["identity"]["weight_sha256"],
            dense_prefill_policy=self.kernel_policy["glm53_dense_prefill"],
            expert_prefill_policy=self.kernel_policy["glm53_expert_prefill"],
            row_policy=self.kernel_policy["glm53_row_kernel"],
        )
        self.model = Glm53GGUFModel(
            self.config,
            self.weights,
            identity=self.identity["fingerprint"]
            + ":"
            + self.profile
            + ":"
            + self.numerics_revision,
            math_policy=self.kernel_policy["glm53_math"],
            hc_fused=self.kernel_policy["glm53_hc"] in ("fused_fp32", "mega_v1"),
            hc_mega=self.kernel_policy["glm53_hc"] == "mega_v1",
            shared_fold=self.kernel_policy["glm53_shared_fold"] == "kblock",
            kda_group=self.kernel_policy["glm53_kda_group"] == "kblock",
            mla_group=self.kernel_policy["glm53_mla_group"] == "kblock",
            router=self.kernel_policy["glm53_router"],
            kda_small=self.kernel_policy["glm53_kda_small"],
            kda_fused=self.kernel_policy["glm53_kda_fused"],
            hc_fused_policy=self.kernel_policy["glm53_hc_fused"],
            mla_fused=self.kernel_policy["glm53_mla_fused"],
            kv_append=self.kernel_policy["glm53_kv_append"],
            **{key.removeprefix("glm53_"): self.kernel_policy[key] for key in SPARSE_KEYS},
        )
        if self.kernel_policy["glm53_tree_batch"] == "kblock_v1":
            self.weights.row_token_share = True
            self.weights.expert_union = True
            self.model.tree_eval_every = 5
        self.model.compile_regions = self.kernel_policy["glm53_compile_regions"]
        if drafter_kind == "mtp":
            self.model.enable_mtp()
        self.model.preload().eval()
        from ..runtime.gguf.mlx_storage import stamp

        if stamp(root / FILENAME) != artifact["weight_stat"]:
            raise ValueError("GLM-5.3 original GGUF changed during loading")
        if external:
            from ..runtime.drafters.prompt_lookup_draft import (
                load_prompt_lookup_drafter,
            )
            from .qwen38_27b import _prompt_lookup_record

            if drafter_kind == "prompt_lookup":
                record = _prompt_lookup_record(
                    self.external_policy, self.identity["fingerprint"]
                )
                loader = load_prompt_lookup_drafter
            elif drafter_kind == "mtp":
                from ..runtime.drafters.native_mtp import load_native_mtp_drafter

                record = _prompt_lookup_record(
                    self.external_policy, self.identity["fingerprint"]
                )
                record["fingerprint"] = (
                    "glm53-embedded-nextn-v1:" + self.identity["fingerprint"]
                )
                loader = load_native_mtp_drafter
            else:
                from .glm53_drafters import load_glm53_drafter

                record = draft_record

                def loader(record, target):
                    return load_glm53_drafter(
                        record,
                        target,
                        activation=self.kernel_policy["glm53_draft_activation"],
                    )

            self.drafter_kind = drafter_kind
            self._bind_external_drafter(record, loader, DESCRIPTOR)

    def execution_numerics_contract(self):
        return {
            "schema": "glm53-packed-execution-v1",
            "numerics_revision": self.numerics_revision,
            "source_identity": dict(self.source_identity),
            **self.kernel_policy,
            "weight_representation": "original-GGUF-blocks",
            "state": "fp32-vector-KDA-and-latent512-raw-indexer256",
            "tree": "ordinary-B1-per-parent-node",
            "features": self.model.external_feature_source,
        }

    @staticmethod
    def planning_cache_budget(config, *, mtp):
        from .glm53_gguf_memory import Glm53CacheBudget

        # Runs before any policy is bound: book every sparse workspace and the
        # pooled plane, so planning is conservative by construction.
        return Glm53CacheBudget.from_config(config, mtp=mtp, maximum=True)

    def cache_budget(self, *, mtp):
        from .glm53_gguf_memory import Glm53CacheBudget
        from .qwen38_27b import _tile_node_budgets

        rows = (
            0
            if self.draft_model is None
            else _tile_node_budgets(self.external_policy, 1)[1] + 1
        )
        return Glm53CacheBudget.from_config(
            self.config, mtp=mtp, tree_rows=rows, kernel_policy=self.kernel_policy
        )

    def attention_topology(self):
        from ..runtime.attention_topology import (
            INDEXED_KV,
            RECURRENT,
            AttentionTopology,
        )

        return AttentionTopology(
            state_revision=self.identity["fingerprint"],
            cache_layout=self.layout,
            layers=tuple(
                RECURRENT if k == "linear_attention" else INDEXED_KV
                for k in self.config["layer_types"]
            ),
            indexed_layout="glm53-latent512-raw-indexer256-fp32-v1",
            recurrent_layout="glm53-kda-conv3-fp32-vector-decay-v1",
        )

    def create_external_batch(self, **kwargs):
        from .glm53_ast_target import Glm53TargetTreeExecutor
        from .qwen38_27b import _tile_node_budgets

        if kwargs.get("completion_batch_size", 1) != 1:
            raise ValueError("GLM-5.3 AST owns exactly one request")
        kwargs.setdefault("dynamic_singleton_tree", True)
        kwargs.setdefault("dynamic_tree_max_width", 1)
        kwargs.setdefault(
            "target_tree_executor_provider",
            Glm53TargetTreeExecutor(
                self.model, implementation=self.kernel_policy["glm53_ast_target"]
            ),
        )
        kwargs.setdefault(
            "tree_node_budget_by_lanes", _tile_node_budgets(self.external_policy, 1)
        )
        for key in ("ast_topology", "ast_adaptive_rows"):
            if key in self.external_policy:
                kwargs.setdefault(key, self.external_policy[key])
        return super().create_external_batch(**kwargs)

    def execution_config(self, *, max_lanes, prefill_step):
        if max_lanes != 1:
            raise ValueError("GLM-5.3 packed route supports physical width one")
        if self.draft_model is not None:
            return self._external_execution_config(
                max_lanes=max_lanes, prefill_step=prefill_step
            )
        return super().execution_config(max_lanes=max_lanes, prefill_step=prefill_step)

    def render_prompt(self, request):
        if "messages" not in request:
            return request["prompt"]
        text = self.tokenizer.apply_chat_template(
            _template_history(request["messages"]),
            add_generation_prompt=True,
            tokenize=False,
        )
        # The exact GGUF template always opens <think> and ignores enable_thinking.
        if not request.get("enable_thinking", False):
            if not text.endswith("<think>"):
                raise ValueError("GLM-5.3 chat template generation boundary changed")
            text += "</think>"
        return text

    def prompt_tokens(self, request):
        return self.tokenizer.encode(
            self.render_prompt(request), add_special_tokens=False
        )

    def thinking_close_token_ids(self):
        # The template opens <think> at the generation boundary; a bounded
        # thinking_budget forces this close token (refused without it).
        ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        return (int(ids[0]),) if len(ids) == 1 else None

    def output_parser(self, request):
        from ..output import OutputParser, callable_tools

        if callable_tools(request):
            raise ValueError("GLM-5.3 packed route does not implement tools")
        return OutputParser(
            chat="messages" in request,
            thinking=bool(request.get("enable_thinking", False)),
            tools=None,
            stops=request.get("stop", ()),
        )

    def diagnostics(self):
        return {
            "architecture": "glm5-next",
            "route": "ast_" + self.drafter_kind
            if self.draft_model is not None
            else "ordinary",
            "qualification": "unqualified",
            "checkpoint": FILENAME
            + "@"
            + self.identity.get("target_fingerprint", self.identity["fingerprint"]),
            "weights": "native packed Q3_K/Q4_K/Q4_0/Q5_K/Q6_K with original F32 tensors",
            "requantization": False,
            "physical_width": 1,
            "supported_context": self.max_context,
            "execution_policy": dict(self.kernel_policy),
            "numerics_revision": self.numerics_revision,
            "source_identity": dict(self.source_identity),
            "load": dict(self.model.load_receipt),
            "scheduled_calls": dict(self.model.calls),
            "scheduled_packed_calls": {
                "dense": dict(self.weights.scheduled_dense_calls),
                "grouped": dict(self.weights.scheduled_grouped_calls),
                "expert": dict(self.weights.scheduled_expert_calls),
            }
            if self.weights is not None
            else None,
            "scheduled_call_observation": "scheduling counters; GPU completion requires a run receipt",
            "sparse_dsa": _sparse_route_status(),
            "compile_regions": _compile_regions_status(self.kernel_policy),
            "hc_fused": _hc_fused_status(self.kernel_policy),
            "mla_fused": _mla_fused_status(self.kernel_policy),
            "embedded_mtp": "selected"
            if self.model.mtp is not None
            else "not selected",
            "drafter": getattr(self, "drafter_kind", None),
            "vision": "not present",
        }

    def close(self):
        try:
            if getattr(self, "model", None) is not None:
                # Compiled layer bodies and host memos capture weight arrays.
                from ..runtime.models import glm53_compile_regions

                glm53_compile_regions.release(self.model)
            if self.weights is not None:
                self.weights.close()
                self.weights = None
        finally:
            super().close()
