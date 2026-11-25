"""Inkling 2 Mini (``inkling2_loop``) native text adapter.

Original mlx2 adapter; model math in ``runtime/models/inkling2`` (mined from
Blaizzy/mlx-vlm@111ac668, see provenance/inkling2-mini.json).  Unqualified.

The checkpoint is served as published: BF16 dense paths, FP32 attention
residual queries/routing bias, NVFP4 routed experts (layers 0-19, ModelOpt
``amax / outer_scale_denominator`` per-expert global scale) and MXFP8 routed
experts (layers 20-23).  Activations stay floating point (W4A16 / W8A16); the
vendor runtime's activation quantization is not reproduced by default, so
numerical parity with the reference runtime is not claimed.  The opt-in
``inkling2_moe_act_quant`` fake-quantizes the expert activations to the
checkpoint's declared contract as an accuracy experiment (not faster).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..sampling_defaults import GENERATION_CONFIG, SamplingDefaults, VendorSampling
from .external_draft_policy import ExternalDraftAdapterMixin
from .ordinary_artifact import inspect_indexed_artifact
from .ordinary_text import OrdinaryTextAdapter

REPOSITORY = "thinkingmachines/Inkling-2-Mini"
CACHE_LAYOUT = "inkling2-kv-four-shortconv-v1"
NUMERICS = "inkling2-native-w4a16-v1"
# inkling2_moe_act_quant on: expert activations fake-quantized (W4A4 / W8A8
# numerics with BF16 matmuls); a distinct APCv2 numerics identity.
NUMERICS_W4A4_FAKE = "inkling2-native-w4a4-fake-v1"
# inkling2_kv_q8 on: the global visits' K/V stored as int8 (approximate
# state); its own APCv2 numerics identity, alone and with moe_act_quant.
NUMERICS_KV_Q8 = "inkling2-native-w4a16-kvq8-v2"
NUMERICS_W4A4_FAKE_KV_Q8 = "inkling2-native-w4a4-fake-kvq8-v2"
# The approximate-state operation name inkling2_kv_q8 publishes under.
KV_Q8_OPERATION = "inkling2_kv_q8"


# Approximate state may only be published through an explicitly qualified
# approximate operation (AGENTS.md). inkling2_kv_q8 is unqualified, so a Q8
# route refuses APCv2 reuse and publication outright: no store, no lookup,
# no disk tier, no session prefetch, no prefill handoff (the engine's
# route-level apcv2_reuse_disabled_reason), and ``apc_publication_refusal``
# declines any int8 lane that reaches a store anyway. The numerics operation
# and the "-kvq8v2" layout already keep Q8 entries apart from BF16 ones, so
# this flag is the single switch a qualification would flip.
KV_Q8_PUBLICATION_QUALIFIED = False
KV_Q8_APC_REASON = "inkling2_kv_q8_unqualified_approximate_state"


def kv_q8_apc_refusal(policy):
    """Why a kernel policy may not reuse or publish APCv2 state, or None."""
    if policy.get("kv_q8") and not KV_Q8_PUBLICATION_QUALIFIED:
        return KV_Q8_APC_REASON
    return None


def _holds_q8_rows(prompt_cache):
    """Structural check: some plane holds int8 (kv_q8) K/V rows."""
    import mlx.core as mx

    stack = list(prompt_cache or ())
    while stack:
        plane = stack.pop()
        children = getattr(plane, "caches", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
            continue
        keys = getattr(plane, "keys", None)
        if isinstance(keys, mx.array) and keys.dtype == mx.int8:
            return True
    return False


def numerics_operation(policy):
    """The APCv2 numerics operation of an Inkling 2 kernel policy."""
    if policy.get("kv_q8"):
        return NUMERICS_W4A4_FAKE_KV_Q8 if policy.get("moe_act_quant") else NUMERICS_KV_Q8
    return NUMERICS_W4A4_FAKE if policy.get("moe_act_quant") else NUMERICS
# generation_config.json of thinkingmachines/Inkling-2-Mini: do_sample, T 1.0.
SAMPLING = VendorSampling.single(
    SamplingDefaults(temperature=1.0, top_p=1.0, source=GENERATION_CONFIG,
                     note="do_sample=true; no top_p/top_k in generation_config"),
    model=REPOSITORY,
)
DESCRIPTOR = ModelDescriptor(
    model_type="inkling2",
    family="inkling2",
    variant="mini-loop-nvfp4-mxfp8-text",
    state_planes=frozenset(
        {StatePlane.ATTENTION_KV, StatePlane.RECURRENT, StatePlane.RNG, StatePlane.TRANSCRIPT}
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
        }
    ),
    cache_layout=CACHE_LAYOUT,
    metadata={
        "execution": "mlx2.adapters.inkling2.Inkling2Adapter",
        "qualification": "pending",
        "scope": "native text; 24 layers / 48 visits; four shortconv states per visit",
        "ordinary_reference": "implemented",
        "dspark": "bundled mtp.safetensors; opt-in external draft (implemented, unqualified)",
        "vision": "weights present; native text route refuses media",
    },
)

# Topology this port implements; anything else refuses before loading.
_EXPECTED = {
    "model_type": "inkling2_loop_text",
    "hidden_size": 2304,
    "num_hidden_layers": 24,
    "vocab_size": 201024,
    "unpadded_vocab_size": 200058,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "swa_num_attention_heads": 24,
    "swa_num_key_value_heads": 8,
    "swa_head_dim": 128,
    "sliding_window_size": 1024,
    "d_rel": 16,
    "rel_extent": 1024,
    "sconv_kernel_size": 4,
    "n_routed_experts": 128,
    "num_experts_per_tok": 8,
    "n_shared_experts": 0,
    "intermediate_size": 1152,
    "use_gate_bias": True,
    "q_bias": False,
    "o_bias": False,
    "use_value_norm_no_gamma": True,
    "attn_res_normalize_values": True,
    "n_layers_per_attn_res_block": 6,
}
_LOOP = {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": False}
NVFP4, MXFP8 = "nvfp4_nearest", "mxfp8_nearest"


def _quant_groups(path: Path) -> dict[str, str]:
    """Expert tensor -> weight transform, from hf_quant_config.json."""
    config = json.loads((path / "hf_quant_config.json").read_text())
    if config.get("quant_method") != "inkling2":
        raise ValueError("Inkling 2 hf_quant_config.json quant_method must be inkling2")
    transforms = {}
    for group in config["model"]["parameter_groups"]:
        kind = group.get("weight_transform")
        if kind == "cast":
            continue
        if kind not in (NVFP4, MXFP8):
            raise ValueError(f"Inkling 2 weight transform {kind!r} is not implemented")
        for name in group["tensors"]:
            transforms[name] = kind
    return transforms


def inspect_artifact(model_path) -> dict:
    """Fail-closed topology/tensor check with no MLX import."""
    artifact = inspect_indexed_artifact(model_path)
    config = artifact["config"]
    if config.get("model_type") != "inkling2_loop":
        raise ValueError("artifact is not Inkling 2 (inkling2_loop)")
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("Inkling 2 config has no text_config")
    for key, value in _EXPECTED.items():
        if text.get(key) != value:
            raise ValueError(f"Inkling 2 {key}={text.get(key)!r} is not implemented")
    if text.get("loop_config") != _LOOP:
        raise ValueError(f"Inkling 2 loop_config {text.get('loop_config')!r} is not implemented")
    local = text.get("local_layer_ids")
    if not isinstance(local, list) or sorted(local) != [
        i for i in range(24) if (i + 1) % 6
    ]:
        raise ValueError("Inkling 2 local/global layer pattern is not implemented")
    path = Path(artifact["identity"]["path"])
    transforms = _quant_groups(path)
    weights = artifact["weight_map"]
    required = {"model.llm.embed.weight", "model.llm.unembed.weight", "model.llm.norm.weight",
                "model.llm.attn_res.output_query", "model.llm.middle_cycle_cross_pass_queries",
                "model.llm.middle_cycle_cross_pass_kv_queries"}
    expert_kinds = {}
    for i in range(24):
        p = f"model.llm.layers.{i}."
        required |= {p + "attn.wq_du.weight", p + "attn.attn_sink_proj.weight",
                     p + "attn.rel_logits_proj.proj", p + "mlp.gate.weight", p + "mlp.gate.bias",
                     p + "attn_sconv.weight", p + "mlp_sconv.weight"}
        kinds = {transforms.get(p + "mlp.experts." + n) for n in ("w13_weight", "w2_weight")}
        if len(kinds) != 1 or None in kinds:
            raise ValueError(f"Inkling 2 layer {i} experts have no single declared format")
        kind = kinds.pop()
        expert_kinds[i] = kind
        for n in ("w13_weight", "w2_weight"):
            required |= {p + f"mlp.experts.{n}", p + f"mlp.experts.{n}_scale"}
            if kind == NVFP4:
                required |= {p + f"mlp.experts.{n}_amax",
                             p + f"mlp.experts.{n}_outer_scale_denominator"}
        if 6 <= i < 18:
            required.add(p + "controllers.attn.down_weight")
    missing = sorted(required - weights.keys())
    if missing:
        raise ValueError(f"Inkling 2 tensors are incomplete: {missing[0]}")
    artifact.update(text_config=text, expert_kinds=expert_kinds,
                    has_mtp=(path / "mtp.safetensors").is_file())
    return artifact


# Opt-in execution switches; every one defaults off and is unqualified.
KERNEL_OPTIONS = {
    # Row-stacked q/r/gate/sink(/k/v) input projection, one GEMV when a
    # visit's q/k/v inputs coincide (implemented 2026-11-20; A/B pending).
    "inkling2_fused_projections": "fused_projections",
    # BF16-weight FP32-accumulate head GEMV instead of the cached FP32 head
    # copy (1.85 GB resident, re-read every token); summation order differs.
    "inkling2_head_kernel": "head_kernel",
    # One-token decode passes the per-query sink to SDPA ``sinks`` (one SDPA
    # per row) instead of concatenating a zero K/V column, which copied the
    # whole K/V on all 48 visits every token (kapitsa/llama.cpp/Rapid-MLX do
    # the same; implemented 2026-11-21, A/B pending). GPU reduction
    # partition differs (S vs S+1 keys).
    "inkling2_sink_sdpa": "sink_sdpa",
    # Zero sink K/V row written into the KV cache's spare capacity past
    # offset and read as a view, instead of concatenated onto K/V (a full
    # K/V copy per visit; 2.1 GB per decode token over the 8 global visits
    # at 64K, and the likely source of the freed-buffer pool growth past 32K
    # in ladder 8e8e3f5b1). Bit-identical by construction (same values, same
    # order); plain KVCache only, anything else falls back. Covers prefill,
    # decode and DSpark verify rows. Implemented 2026-11-21, GPU A/B pending.
    # An append that leaves the buffer exactly full (every 512-token prefill
    # chunk over the 256-token growth) grows it by one cache step instead of
    # falling back to the concatenation (2026-11-22, language.SINK_SLOT_SPARE;
    # sink_slot_grows and sink_slot_fallback_reasons in the kernel receipt).
    "inkling2_sink_slot": "sink_slot",
    # Prefill/verify attention (rows > 1) through one FP32 flash kernel with
    # the relative bias, window and sink in-kernel, replacing the dense bias
    # mask chunks (implemented 2026-11-21, GPU-only; decode keeps SDPA).
    # Tried 2026-11-21 on Inkling-2-Mini@440956c8 (86aa3db9e): correct (Metal
    # parity; 4096-token argmax 3983/4096; first 32 of 113 splits near ties, 81 unclassified) but
    # NOT faster. Prefill ABAB tok/s off vs flash: 4K 1260 vs 1026, 16K 998 vs
    # 770, 64K 405 vs 389. FP32 simdgroup MMA plus per-query-head K/V tile
    # loads (no GQA sharing) cost more than the dense-mask chunks save. Left
    # off; a faster version needs BF16 MMA with FP32 accumulation, K/V tiles
    # shared across the GQA group, and a profile first.
    "inkling2_flash_prefill": "flash_prefill",
    # Short query blocks (1 <= L < 32: DSpark verify rows L = 8 and one-token
    # decode) through a GQA-packed split-K flash-decoding kernel ported from
    # jundot/omlx PR 5511 (runtime/models/inkling2/splitk.py, Apache-2.0):
    # all G*L rows of a KV head share one K/V read, keys split across
    # threadgroups and merged, bias/window/padding/sink in-kernel, no mask
    # and no sink column. Replaces the dense-mask chunks (verify) and SDPA
    # (decode; it takes precedence over inkling2_sink_sdpa). Any block of
    # 1..31 rows engages, so short prefill chunks (tails, APCv2 warm
    # suffixes, small DSpark prefill steps) do too: cold and warm runs of one
    # prompt can then differ numerically. GPU, BF16 (FP32 for test models),
    # KVCache/ring caches; declines are counted by reason. Near-exact (FP16
    # tile staging, different reduction order). Its source is in the APCv2
    # numerics contract when on.
    # Implemented 2026-11-21, GPU tests and A/B pending.
    # Sliding rings reach the kernels as their physical buffers plus a start
    # row (splitk.RING_DIRECT, 2026-11-22) instead of a chronological copy
    # made every decode step once wrapped; the kernels walk the same logical
    # keys, so it is bitwise the copy. Declines still copy (counted in
    # verify_splitk_ring). RING_DIRECT is DEFAULT ON (Pierre 2026-11-23, Metal
    # bitwise + decode +3.5-5.4%; see splitk.py); splitk.VECTOR_LOADS stays off
    # (no decode gain).
    "inkling2_verify_splitk": "verify_splitk",
    # Sparse MoE in fewer launches, bit-identical (implemented 2026-11-21,
    # runtime/models/inkling2/moe_fast.py). DEFAULT ON (Pierre 2026-11-21)
    # with the next two, after the GPU check on Inkling-2-Mini@440956c8 at
    # eed5a8160: 4096-token prefill logits bitwise equal, greedy 4/4 and
    # decode tokens equal, prefill +28% at 4K and +23% at 16K: compiled
    # routing, one gather over a stacked [gate; up] table (the gate/up
    # modules become row views of it, so memory does not grow), the shared
    # NVFP4 w13 scale and the softcapped SwiGLU in one kernel, and the down
    # scale with the weighted combine. Decode and prefill.
    "inkling2_moe_fused": "moe_fused",
    # Multi-token sorted calls (prefill chunks, verify windows), GPU, needs
    # inkling2_moe_fused (default on; unset, it follows inkling2_moe_fused): the weighted
    # combine reads the sorted down rows through inv_order
    # (qwen4_moe_weighted_sum, MLX column-reduce order replayed), so the
    # unsorted [T, K, D] copy and the FP32 product are never written.
    "inkling2_moe_weighted_sum": "moe_weighted_sum",
    # Default OFF, opt-in: renormalize the picked MoE gate weights in the
    # log domain (softmax of log-sigmoid; SGLang #50525 idea) instead of
    # sigmoid / (sum + 1e-20), which zeroes the MoE output when every picked
    # logit is <= ~-90.  Same expert choice; not bitwise equal to the default
    # (Pierre decides the default).  Refused with inkling2_moe_decode_kernel,
    # inkling2_moe_panel_kernel and inkling2_native_replay.
    # Tested 2026-11-23 (Pierre: "test it first"; run
    # w1122-44fe2a014-mtplx-smoke-logdomain-a1, Inkling-2-Mini@440956c8):
    # NOT HELPFUL, LEAVE OFF.  Over 199,680 routing rows of real text (4K
    # prefill + 64 decode) the smallest picked sum was 0.153 and the smallest
    # top picked logit -3.1, nowhere near the ~-90 underflow it guards; with
    # it on, logits are never bitwise (prefill argmax 97%, first divergence a
    # same-top-two near tie, margins 0.036/0.030) and speed is unchanged.
    "inkling2_moe_logdomain_renorm": "moe_logdomain_renorm",
    # Prefill only, M5 only: NVFP4 sorted expert gathers through the
    # segmented NAX kernel (moe_nax_gather, canary-proven bit-identical to
    # the stock sorted kernel). The mode is a process global, claimed at
    # load; MXFP8 layers 20-23 have no NAX loader and keep the stock kernel.
    # Unset = auto (default): on for an M5 NAX host, off elsewhere; an
    # explicit true on a non-M5 host still fails closed.
    "inkling2_moe_nax_gather": "moe_nax_gather",
    # Launch-count fusions (runtime/models/inkling2/fuse.py; implemented
    # 2026-11-22, default off, GPU A/B pending). Decode is dispatch-bound
    # (33 ms per token at 1K against a ~12 ms weight floor), so these cut
    # kernels without changing the arithmetic. The first four are intended
    # bit-identical (CPU tiny model bitwise: prefill, decode, B1/B2, ring
    # wrap, FP32 and BF16).
    # q/k/v loop-message controllers and input norms as batched chains
    # (B1 decode), shared norms and compiled blends otherwise.
    "inkling2_fuse_controllers": "fuse_controllers",
    # Attention-residual mixes read once-per-forward FP32 copies of the
    # BF16 blocks instead of promoting every block on every mix.
    "inkling2_fuse_residual": "fuse_residual",
    # Short conv: next state as a view of the context (no second concat),
    # epilogue plus the decoder's residual round trip as one compiled kernel,
    # the dynamic conv's lag loop compiled.
    "inkling2_fuse_sconv": "fuse_sconv",
    # Relative-bias mask indices once per forward, ring read plus zero sink
    # row as one concat, tau skipped below the floor (exactly 1) and
    # compiled above, compiled output gate, direct BF16 value norm.
    "inkling2_fuse_attention": "fuse_attention",
    # Near-exact GPU kernels, kept separate (own summation order):
    # one-launch static short conv (<= 8 rows, needs inkling2_fuse_sconv;
    # default on since 2026-11-22: near-exact by construction, observed
    # bitwise on the checkpoint's tested shapes, see KERNEL_DEFAULTS)
    "inkling2_fuse_sconv_kernel": "fuse_sconv_kernel",
    # and one-launch one-row attention-residual mixes and loop messages.
    "inkling2_fuse_residual_kernel": "fuse_residual_kernel",
    # Added 2026-11-22 from omlx#5538 and Rapid-MLX #5918-#5920
    # (reimplemented; default off, GPU A/B pending). Intended bit-identical:
    # cached lhs indices for unsorted expert gathers (no arange per call),
    "inkling2_fuse_gather_lhs": "fuse_gather_lhs",
    # FP32 copies of RMSNorm weights for FP32 inputs (no per-call weight cast),
    "inkling2_fuse_norm_fp32": "fuse_norm_fp32",
    # one log-tau per forward for the global visits (above the 8K floor).
    "inkling2_fuse_shared_tau": "fuse_shared_tau",
    # Near-exact Metal kernels (own summation order): the global-layer dynamic
    # k/v conv in one launch (needs inkling2_fuse_sconv),
    "inkling2_fuse_dconv_kernel": "fuse_dconv_kernel",
    # the loop-message forget/gain GEMVs + tanh + blend in one launch for
    # one-row calls (needs inkling2_fuse_controllers),
    "inkling2_fuse_message_kernel": "fuse_message_kernel",
    # the one-token fused MoE forward compiled per shape (needs moe_fused),
    # GPU only. The first GPU try failed "[compile] ... uncaptured inputs":
    # the state was passed as fresh parameters() dictionaries while the
    # traced code read the modules (fixed in cba0c1602). The 2026-11-22 fix
    # on top: a trace hands the arrays back to the modules as new Python
    # objects, which made the identity-keyed fused build restack gate/up and
    # retrace on every token; they are now re-stamped (one trace per layer),
    # and the FP32 router copy is an argument cast outside the trace. CPU:
    # traced and bitwise equal to the eager fused path (NVFP4/MXFP8 tiny
    # model). GPU 2026-11-22 (Inkling-2-Mini@440956c8, see the
    # inkling2_compile_decode_blocks row of the decisions TSV): bitwise,
    # but no decode gain (the launches it cannot fuse -- gathers, index
    # aranges, reductions -- remain; the epilogues it fuses were already
    # compiled). Not helpful; leave off.
    "inkling2_compile_decode_blocks": "compile_decode_blocks",
    # and the logit softcap compiled into one kernel on the GPU.
    "inkling2_head_softcap_compiled": "head_softcap_compiled",
    # The one-token NVFP4 routed-expert forward in two Metal launches
    # (runtime/models/inkling2/moe_decode_kernel.py; implemented 2026-11-22,
    # default off, needs inkling2_moe_fused). The shipped router ops, then a
    # gate/up gather + scale + softcapped SwiGLU kernel and a down gather +
    # scale + routing-weight normalization + weighted-sum kernel, written to
    # MLX's one-token gather lane order and the shipped rounding points so it
    # can be bitwise (the Metal parity sweep picks the contraction and
    # route-sum order). ~6 launches per decode MoE call instead of ~16.
    # GPU only; other shapes, top_k > 8, MXFP8 layers and act-quant decline, counted
    # (moe_fast_counts decode_kernel / decode_kernel_declined:*).
    # GPU 2026-11-22, Inkling-2-Mini@440956c8, M5 Max, contended (sanity):
    # see the inkling2_moe_decode_kernel rows of the decisions TSV. Exact
    # with the module defaults (contraction default, sequential route sum,
    # the default from 09260f5d9; full run at 6b0946589: bitwise, decode
    # -3.4%/-3.1% at 1K/16K), small gain. Night retest 98883cc60 (2026-11-24,
    # quiet GPU): logits bitwise 64/64 at 1K and 16K, +3.4%/+3.0% decode.
    # DEFAULT ON (Pierre 2026-11-25); see KERNEL_DEFAULTS.
    "inkling2_moe_decode_kernel": "moe_decode_kernel",
    # Expert-grouped verify panels (runtime/models/inkling2/moe_panel_kernel.py;
    # implemented 2026-11-23 from the MTPLX sweep candidate 1, mechanism of
    # MTPLX fast/experts.py, Apache-2.0; default off, needs
    # inkling2_moe_fused, GPU pending). 2-32 row MoE calls (DSpark verify,
    # short prefill tails) after the shipped router ops: a plan launch lists
    # the panel's distinct experts and their member pairs, then gate/up and
    # down products read each distinct expert once for all of its rows
    # (a 4-row top-8 panel names ~19 experts for 32 pairs), and a combine
    # launch. NVFP4 (layers 0-19) and MXFP8 (20-23). Intended bitwise to the
    # shipped W4A16 route: each pair keeps MLX's gather_qmv lane order (the
    # Metal test sweeps the two dispatch details, QDOT_CONTRACT and
    # FAST_K_RULE). Calls where the shipped route would stream
    # (gather_qmm_rhs), one-token calls and act-quant decline, counted
    # (moe_fast_counts panel_kernel / panel_kernel_declined:*).
    # First GPU sanity A/B (run w1122-3ee59f64e-mtplx-smoke-moe-a1,
    # Inkling-2-Mini@440956c8, M5 Max, main 3ee59f64e, 1K context, 3 rounds,
    # with PANEL_OVER_WEIGHTED_SUM on): bitwise at 2 and 8 rows, 48 panel
    # calls per verify forward, but SLOWER: median verify forward 0.0389 ->
    # 0.0440 s at 2 rows (+13%) and 0.0572 -> 0.0734 s at 8 rows (+28%).
    # Leave off; pending the overnight A/B.
    "inkling2_moe_panel_kernel": "moe_panel_kernel",
    # MXFP8 expert tables (layers 20-23) on the NAX sorted gather (M5 tensor
    # units, prefill only; implemented 2026-11-23, MTPLX sweep candidate 10,
    # default off, needs inkling2_moe_nax_gather, GPU pending). A MXFP8
    # loader (E4M3 codes x E8M0 group scales, decoded to BF16 as MLX's
    # QuantizedBlockLoader does) in moe_nax_gather; each instantiation is
    # canary-proven bitwise against the stock sorted gather_qmm before use,
    # else the stock kernel runs (fallback counted).
    "inkling2_moe_nax_mxfp8": "moe_nax_mxfp8",
    # W4A4 accuracy experiment (implemented 2026-11-21, default off, GPU A/B
    # pending; scripts/research/inkling2_w4a4_ab.py). Fake-quantizes the
    # routed experts' inputs as hf_quant_config.json's
    # required_activation_quantization declares: NVFP4 four-over-six (MAE)
    # per-token on layers 0-19, MXFP8 block-32 on layers 20-23, both the
    # gate/up input and the SwiGLU output (runtime/models/inkling2/
    # act_quant.py). The router and every non-expert tensor stay
    # unquantized. NOT a speed path: Metal has no FP4 x FP4 units, so the
    # matmuls are unchanged and the quantize-dequantize is extra work. It
    # changes numerics: its source joins the APCv2 numerics contract and the
    # operation becomes NUMERICS_W4A4_FAKE.
    "inkling2_moe_act_quant": "moe_act_quant",
    # Int8 K/V for the 8 global visits (implemented 2026-11-22, default off,
    # unqualified, approximate; runtime/models/inkling2/kv_q8.py, mechanism
    # from g13-engine c91c0180, MIT, no code copied). Symmetric codes per
    # 64-channel group with FP16 scales, stored inline as one int8 row per
    # token (v2: codes, scales, zero pad to 16 bytes = 144 bytes instead of
    # 256 at D = 128; 16-byte vector loads in the kernels), so plain KVCache /
    # BatchKVCache semantics (trim rollback, merge, APCv2 freeze/restore)
    # carry over. GPU 1..31-row blocks (decode, DSpark verify) read the codes
    # in split-K kernels with splitk.py's bias/window/padding/sink law;
    # longer prefill chunks run the same kernel in 31-row query tiles; the
    # CPU and declined geometry dequantize key blocks of <= 4096 keys with
    # an online softmax. Nothing dequantizes the whole prefix (receipt:
    # kv_q8_full_prefix_dequant stays 0). Sliding rings stay BF16. It changes
    # outputs: its own numerics operation (NUMERICS_KV_Q8), APCv2 layout
    # (suffix "-kvq8v2") and source hashes, and prompt logprobs are refused.
    # Unqualified approximate state: APCv2 reuse and publication are refused
    # on the route (KV_Q8_PUBLICATION_QUALIFIED, kv_q8_apc_refusal).
    # Combines with every other switch; GPU tests and A/B pending
    # (tests/test_inkling2_kv_q8_metal.py, scripts/research/inkling2_kv_q8_ab.py).
    "inkling2_kv_q8": "kv_q8",
    # Prefill attention slices (2026-11-22 audit; default off, GPU A/B
    # pending). With flash prefill off, multi-row attention runs masked SDPA
    # in row slices sized by a 64 MiB mask budget (inkling2_prefill_mask_mb),
    # each slice re-reading the whole K/V. The budget counts 4 bytes per mask
    # element but the mask is BF16 on the checkpoint, so slices are half the
    # intended size (at 128K: ~103 slices x 512 MiB of K/V per global visit
    # per 512-token chunk). This counts the mask's element size instead:
    # twice the rows per slice, half the K/V passes, twice the transient mask
    # memory (the same as inkling2_prefill_mask_mb 128 on BF16). Near-exact:
    # the values are the same, but the slice shapes change, so SDPA may pick
    # another kernel tile and reduce in another order. No-op for FP32.
    "inkling2_prefill_mask_bf16_budget": "prefill_mask_bf16_budget",
    # The same masked slices append each query's sink logit with a second
    # full-width concatenation (one more write and read of the whole mask
    # per slice). This builds the mask with the sink column included: the
    # sink logit is an extra bias-table entry the sink column's gather index
    # points at (runtime/models/inkling2/prefill_attention.py). The mask
    # values are copies of the reference's, so it is bit-identical (CPU tiny
    # model: bitwise, B1/B2, left padding, FP32/BF16, sliding and global
    # layers); default off until the GPU A/B measures the saving.
    "inkling2_prefill_mask_with_sink": "prefill_mask_with_sink",
    # inkling2_head_kernel with 16-byte (uint4) weight loads instead of 8-byte
    # (uint2); needs inkling2_head_kernel. Every lane still sums the same
    # four-value groups in the same order (a 16-byte load holds two lanes'
    # groups, so two simd_shuffles redistribute them; the per-group
    # arithmetic is unchanged), so it is intended bit-identical to the
    # narrow kernel; the Metal test checks that. The head kernel itself is
    # a future-default candidate: it reads the 0.92 GB BF16 head per decode
    # step where the default FP32 copy reads ~1.85 GB. Default off (2026-11-22
    # audit); the shuffles may cost what the wider loads save, GPU A/B pending.
    "inkling2_head_kernel_wide": "head_kernel_wide",
    # Global-layer prefill/verify attention (rows > 1) through a tiled
    # flash-style Metal kernel (runtime/models/inkling2/prefill_attention.py,
    # original; mechanism after omlx#5526 flash_banded_attention): relative
    # bias, padding, sink and log-tau-scaled q/r in-kernel with an online
    # softmax, K/V read once per query tile for all G query heads of a KV
    # head (packed rows interleave the heads), no mask. At 128K the masked
    # slices read K/V ~100 times per global visit per 512-token chunk
    # (~412 GiB); here it is ceil(512 * G / 32) tile passes. BF16 staged in
    # FP16 with FP32 accumulation; near-exact (CPU emulation of the kernel's
    # tiling vs the reference: see tests/test_inkling2_nonbitwise_cpu.py).
    # Improves on inkling2_flash_prefill (FP32 staging, one head per tile,
    # slower: 4K 1260 -> 1026 tok/s); the two are alternatives. Short blocks
    # (< 32 rows) still go to inkling2_verify_splitk first. Sliding layers
    # keep the masked path. GPU only, declines counted. Default off: Metal
    # tests and the GPU A/B are pending (omlx found MLX's masked SDPA faster
    # than its flash kernel for full attention on M5 NAX hosts).
    "inkling2_prefill_tiled_attention": "prefill_tiled_attention",
    # Prompt attention (rows > 1, sliding and global visits) in one Metal
    # launch per visit and chunk (runtime/models/inkling2/prefill_onelaunch.py,
    # clean-room implementation 2026-11-23 of the one-launch spec; no MTPLX
    # code): Q/K/V, bias profiles and sinks read in place through their
    # strides, ring caches read through their (rot, size) mapping, no mask,
    # bias plane or sink row (memory: the output only). FP32 operands,
    # probabilities and online softmax from the sink; key tiles aligned to
    # each sequence's own token frame, so a row's output is bitwise
    # independent of the chunk width, its place in the chunk and left
    # padding. Two forms of the same function: Metal 4 tensor ops on M5-class
    # GPUs (macOS 26.2+), simdgroup 8x8 FP32 MMA elsewhere
    # (MLX2_INKLING2_ONELAUNCH_FORM=auto|simd|tensor|off). Each (form, dtype,
    # window kind) passes a float64 self-check on first use or the route
    # declines (first_use). Blocks under 32 rows still go to
    # inkling2_verify_splitk first; kv_q8 global visits, fused-ring and
    # deferred-commit views and unmapped rings decline. All declines counted
    # (kernel_receipt()["prefill_onelaunch"]). Near-exact against the composed
    # path (different reduction order, more accurate than it on synthetic
    # inputs). Alternative to inkling2_prefill_tiled_attention and
    # inkling2_flash_prefill. Default off: Metal tests, model-path check and
    # the A/B are pending.
    "inkling2_prefill_onelaunch_attention": "prefill_onelaunch_attention",
    # One-token decode front end of every attention visit in one Metal launch
    # (runtime/models/inkling2/attn_decode.py, original, 2026-11-22): the k/v
    # short convolutions and next conv states, the q/k/r L2 clips, the value
    # RMS norm, the global visits' log-position scale, the relative-bias rows
    # (r @ rel_proj) and the FP32 sink, between the input projections and the
    # K/V append; the attention stays the split-K kernel. B1 one-token calls
    # only, on KVCache/RotatingKVCache (direct API) and the one-row
    # BatchKVCache/BatchRotatingKVCache a served lane is merged into (their
    # device left_padding passed on as the composed path passes it); requires
    # inkling2_verify_splitk. Everything else declines (counted by reason,
    # kernel_counters["attn_decode"]) to the composed path, including the
    # fused-ring SDPA route (fuse_attention on, sink_sdpa off).
    # Near-exact (the reductions sum in the kernel's order). GPU only.
    # Default off (2026-11-22, Inkling-2-Mini@440956c8; evidence in
    # docs/research/model-feature-decisions.tsv): engages on every served
    # one-row visit; tokens exact (direct 1K/16K, oracle prompts) or
    # a near-tie candidate (served summary, token 72: same top two, margins
    # .009/.073 nat; unclassified, the response had no functional oracle);
    # logits not bitwise
    # (the bias rows sum in the kernel's order). Launches per call 38 -> 22
    # sliding, 55 -> 20 global (~900 fewer per token), but decode gains only
    # ~1.5-5% (sanity A/Bs, contended host): the removed launches cost ~1 us
    # each, so the attention share is mostly elsewhere (weight reads, the
    # split-K kernel, the cache appends, the ring copy). Not a default unless
    # Pierre decides otherwise.
    "inkling2_attn_decode_kernel": "attn_decode_kernel",
    # Deferred commit (MTPLX fast/state.py mechanism, 2026-11-23; default
    # off, unqualified, CPU tests only, GPU pending): while a verify
    # transaction speculates, the 40 sliding-window ring visits write nothing
    # (RotatingKVCache ``update_and_fetch(deferred=True)`` returns the
    # chronological K/V the rollback path's concatenation holds, the same
    # values in the same order) and the commit writes only the kept rows in
    # place; rejected rows cost nothing. Replaces the per-forward window
    # snapshot copy, the concatenation into the cache and the restore +
    # replay on reject. The four conv histories commit the kept prefix by
    # slicing instead of the masked gather. Bitwise against the rollback path
    # (tokens and committed state; tests/test_inkling2_deferred_commit_cpu.py).
    # Counters: kernel_counters["verify_deferred_commit"].
    "inkling2_verify_deferred_commit": "verify_deferred_commit",
    # Visit length check (MTPLX #710, 2026-11-23; default off): before every
    # forward, every visit's B1 KV plane must hold the same committed length,
    # so an APCv2 restore that landed visits at different positions fails
    # closed instead of decoding from mixed state. Host integers only; no
    # output change. Counters: kernel_counters["visit_length_check"].
    "inkling2_visit_length_check": "visit_length_check",
    # MTPLX ingest 2026-11-23 (candidate 12; runtime/models/inkling2/
    # prefill_route.py, original code): mx.async_eval of the residual stream
    # every 4 visits inside a prompt forward (>= 64 rows), so the GPU starts
    # the chunk while Python builds the rest. Same graph and kernels:
    # bit-identical (CPU test; GPU pending). Counter: prefill_handover_calls.
    # Default off.
    "inkling2_prefill_handover": "prefill_handover",
    # MTPLX launch-count ingest (2026-11-23; mechanisms from
    # youssofal/MTPLX@3890be55, Apache-2.0, reimplemented; provenance/
    # inkling2-mtplx-launch.json). Default off, unqualified, GPU A/B pending.
    # Each one is bitwise by construction against the path it replaces, and
    # proves it: the first call of every shape evaluates both and compares
    # them bit for bit (runtime/models/inkling2/library_kernels.py
    # run_proved); the proving call returns the composed result, a refused
    # shape keeps the composed path. So outputs equal the option-off route
    # whether or not the launch engages; the counters say whether it did
    # (kernel_receipt: calls / proved / refused / declined:<why>).
    # A loop pass's 12 message reads as one scores product, one softmax and
    # one mix (runtime/models/inkling2/pass_messages.py; any device).
    "inkling2_pass_messages": "pass_messages",
    # A visit's six input projections in one launch of MLX's own product
    # code (GEMVKernel for 1 row, GemvWide for 2..8 BF16 rows, compiled from
    # the installed headers; stage_a.py). GPU only.
    "inkling2_stage_a_kernel": "stage_a_kernel",
    # The attention output gate inside the split-K merge launch (MLX's own
    # Sigmoid functor; attn_output.py, splitk.merge_gated). Needs
    # inkling2_verify_splitk. GPU only. Not on visits that
    # inkling2_attn_decode_kernel takes (it calls the split-K kernel itself).
    "inkling2_gate_merge": "gate_merge",
    # o_proj + the attention short conv + the residual add in one launch
    # (attn_output.py). Needs inkling2_fuse_sconv and
    # inkling2_fuse_sconv_kernel (the path it is proven against). GPU only.
    "inkling2_attn_output_kernel": "attn_output_kernel",
    # Token embedding gather + embed norm in one launch (frame_kernels.py).
    "inkling2_embed_kernel": "embed_kernel",
    # Recorded-step native replay (2026-11-23; clean-room implementation from
    # a functional spec, provenance/inkling2-native-replay-cleanroom.json).
    # Default off, unqualified. On: one-row ordinary decode runs through the
    # fixed-capacity step (runtime/models/inkling2/capacity_state.py: planes
    # padded in place to a constant geometry, round integers as uint32
    # arrays) and offers it to the replay engine (runtime/replay). Near-exact
    # against the ordinary route (the step reads masked columns, so its
    # reductions run over more terms). Without an installed native module
    # the engine declines every call as module_missing and the MLX step
    # serves. Receipt: diagnostics()["native_replay"]. Refused with
    # inkling2_kv_q8 and inkling2_attn_decode_kernel.
    "inkling2_native_replay": "native_replay",
}
# Defaults for KERNEL_OPTIONS keys a policy leaves unset (everything else is
# False). None for the NAX gather means "auto", resolved at load.
#
# The composed set is DEFAULT ON (Pierre 2026-11-22, "full set"): sink SDPA
# (one-token decode), sink slot (multi-row calls), the four exact launch-count
# fusions and split-K short-block attention. Basis, Inkling-2-Mini@440956c8:
# ladder 028602cc6 (composed + split-K) decode +5% to +76% (ordinary) and
# +23% to +92% (DSpark) at 1K-128K against reference ordinary_b1, oracle
# 24/24, cold/warm exact, 12/12 splits near_tie_equivalent; fusions bitwise
# on Metal and on the checkpoint. Sink SDPA and split-K are near-exact (their
# reduction order differs), so the default route is near-tie, not bitwise,
# equal to the reference; set any of them false to get the reference back.
#
# fuse_sconv_kernel is DEFAULT ON too (Pierre 2026-11-22, "leave
# fuse_sconv_kernel default enabled"), on top of fuse_sconv. Basis,
# Inkling-2-Mini@440956c8, fuse A/B 16eed37f5 against the defaults
# (qualification/runs/inkling2-gpu-20261122/16eed37f5-kernsconv): decode
# logits bitwise on 128/128 steps and last prefill logits bitwise at 1K and
# 16K; decode rate unchanged (33.8 vs 33.0, 32.2 vs 30.6, 32.0 vs 31.9 tok/s
# after the warm-up block), prefill unchanged. It engages on GPU calls of up
# to 8 rows (decode and DSpark verify; the A/B covered decode only) and falls
# back elsewhere; unset it
# follows fuse_sconv, so inkling2_fuse_sconv false turns both off.
# fuse_residual_kernel stays off: A/B 16eed37f5 (16eed37f5-kernres) was not
# bitwise (max |dlogit| 0.08 at 1K, 0.55 at 16K, argmax 125/128 with the 3
# margins not captured), decode within noise, prefill 12% slower at 1K.
# W4A4 (moe_act_quant) stays off.
# head_kernel is DEFAULT ON (Pierre 2026-11-25, after the Rapid-MLX gap A/B
# qualification/runs/inkling2-gpu-20261125/566479abc-gaplevers): it reads the
# BF16 unembedding with FP32 accumulation instead of a cached FP32 copy (1.85
# GB per token), decode +6.2% at 1K and +6.9% at 16K, argmax 64/64, max logit
# difference 3.8e-6 (summation order only). It falls back to the FP32 GEMM
# off GPU, for non-BF16 heads, or widths not a multiple of 128; the reference
# policy keeps the FP32 GEMM for mlx-vlm parity. head_kernel_wide stays off.
# moe_decode_kernel is DEFAULT ON (Pierre 2026-11-25, after the night retest
# 98883cc60 inkk1123-98883cc60-n1124-retest, Inkling-2-Mini@440956c8): decode
# logits bitwise on 64/64 steps at 1K and 16K, prefill last logits equal,
# decode +3.4% (1K) / +3.0% (16K); earlier 6b0946589 also bitwise, -3.4%/-3.1%
# ms/step. Unset, it follows moe_fused and stays off with moe_logdomain_renorm;
# MXFP8 layers, act-quant and other shapes decline (counted). The reference
# policy turns it off.
KERNEL_DEFAULTS = {"moe_fused": True, "moe_weighted_sum": None, "moe_nax_gather": None,
                   "sink_sdpa": True, "sink_slot": True, "verify_splitk": True,
                   "fuse_controllers": True, "fuse_residual": True, "fuse_sconv": True,
                   "fuse_attention": True, "fuse_sconv_kernel": True, "head_kernel": True,
                   "moe_decode_kernel": True}
# The policy that selects the reference path for every switch the defaults
# turn on (comparison scripts and the ladder's reference arm start from it).
REFERENCE_POLICY = {key: False for key, name in KERNEL_OPTIONS.items()
                    if KERNEL_DEFAULTS.get(name) is True
                    and (not name.startswith("moe_") or name == "moe_decode_kernel")}
# Varloop (Pierre 2026-11-21): run the middle
# block N times instead of the trained 3. The checkpoint has is_varloop
# false, so every N != 3 is out of distribution; it is a research knob, not a
# serving option. Incompatible with DSpark (its taps assume 48 visits).
LOOP_PASSES_OPTION = "inkling2_loop_passes"
# Hard no-thinking (Pierre 2026-11-21): with
# this policy on, a request that turns thinking off (enable_thinking false,
# reasoning_effort "none", or thinking_budget 0) gets the assistant header and
# text marker appended to the prompt, so the model starts in the text channel
# instead of at the header where it may open <|content_thinking|>. Default
# off: without it "none" stays the template's soft hint.
HARD_NO_THINKING_OPTION = "inkling2_hard_no_thinking"
TEXT_PREFILL = ("<|message_model|>", "<|content_text|>")
# Prefill mask budget (MiB; default 64). Long-context prefill builds the
# relative-bias mask in row chunks of this many bytes and runs one SDPA per
# chunk over the whole K/V: at 128K a 64 MiB budget is ~5 query rows per call,
# so a 512-token chunk re-reads K/V ~100 times per global visit. A larger
# budget trades transient memory for fewer passes; rows per call change, so
# SDPA kernel choice may too (near-exact). Opt-in pending an A/B.
PREFILL_MASK_OPTION = "inkling2_prefill_mask_mb"
# DSpark prefill allocator reclaim. Ordinary prefill ends every chunk with
# mx.clear_cache() (MLX2_PREFILL_CLEAR_CACHE=always); the external-draft
# engine does so only with its default-off prefill_allocator_reclaim. Each
# Inkling prefill chunk needs buffers of a new size (the key count grows), so
# without the clear the MLX pool grows with context: on the real checkpoint
# at 512-token chunks it held 41.5 GiB at 27K keys (target only and with the
# drafter alike) against 0 GiB with the clear, at the same ~20 GiB active
# (qualification/runs/inkling2-gpu-20261121/9f49b58b9-pool). The DSpark B1
# ladder at 32K was stopped by the memory watchdog at an 84.6 GiB footprint.
# The clear is exact (allocator only). DEFAULT ON for the DSpark route (Pierre
# 2026-11-21, "default if it passes"): ladder 8e8e3f5b1 on Inkling-2-Mini
# @440956c8 ran DSpark B1 1K-128K with no memwatch stop (footprint 29.7 GB at
# 32K, 35.0 at 64K, 46.0 at 128K), oracle 12/12, all 6 splits from ordinary
# near_tie_equivalent. Unset = on with DSpark, off otherwise; false opts out.
PREFILL_RECLAIM_OPTION = "inkling2_dspark_prefill_reclaim"
# DSpark pre-drafting: queue the next chain block before the host reads the
# verify acceptance (ExternalDraftBatchGenerator ``predraft``). Opt-in,
# unqualified, greedy lanes only; exact by adoption at the matching boundary
# (tests/test_inkling2_dspark_predraft_cpu.py).
PREDRAFT_OPTION = "inkling2_dspark_predraft"
# DSpark drafter options, all opt-in and DSpark-only (mechanisms from
# g13-engine c91c0180, re-implemented; docs/PROVENANCE.md). They change
# draft-side work only; the target verifies every proposal.
# kv_headroom: the draft block's K/V written in place on global layers
# instead of concatenated onto the whole draft context (bitwise the same).
# fold_q8: the Markov head's folded lm_head @ hidden_out matrix in MLX affine
# 8-bit (about half the bytes per Markov step); drafts may differ.
# cost_depth: draft depth per round from a verify cost table (true: timed at
# load; or an object with verify_ms/draft_ms), through the existing cohort
# adaptive_verification policy; see runtime/drafters/inkling2_dspark_cost.py.
# MTPLX ingest 2026-11-23, candidate 9 (adapters/inkling2_prefill_plan.py,
# original code): prefill chunks of up to 2,048 rows for chunk starts below
# 18,432 keys (never crossing it) and 512 rows past it, a deterministic
# function of the chunk's absolute depth fixed at load (no run-time memory
# input), so a read resumed from a restored prefix runs the cold read's
# chunks whenever a checkpoint cut the cold read there (as on the 512 route).
# The schedule is part of the APCv2 namespace. Declines at load (step 512,
# 512 namespace) when free GPU memory is below 12 GiB. Admission charges the
# largest chunk the schedule runs; nothing is narrowed at run time. Ordinary
# route only (the DSpark prefill slicer has no schedule hook yet: refused). Implies
# inkling2_prefill_transient_pricing. Default off; the width changes GEMM/MoE
# tiles, so logits differ from the 512-row route.
PREFILL_WIDE_OPTION = "inkling2_prefill_wide_chunk"
# MTPLX ingest 2026-11-23, candidate 11 (Inkling2CacheBudget.transient_terms):
# admission charges each prefill chunk's working memory, derived from the
# shapes (residual states, one visit's projections, the mask slices and
# index planes, the routed experts, the global K/V growth copy). Not purely
# additive at the default width (it adds bytes to every arriving lane's
# requirement, so a host near its limit can defer or refuse a lane it admits
# today), hence default off.
PREFILL_PRICING_OPTION = "inkling2_prefill_transient_pricing"
# PR sweep 2026-11-25 (raullenchai/Rapid-MLX#5953, idea only; runtime/models/
# inkling2/prefill_route.py): a one-row decode forward submits the residual
# stream with mx.async_eval after every visit. Same graph and kernels:
# bit-identical (CPU test; GPU A/B pending). Not combined with
# inkling2_native_replay (its recorded step is replayed, not rebuilt).
# Counter: decode_handover_calls. Default off, unqualified.
DECODE_HANDOVER_OPTION = "inkling2_decode_handover"
DSPARK_OPTIONS = {"inkling2_dspark_kv_headroom": "kv_headroom",
                  "inkling2_dspark_fold_q8": "fold_q8",
                  "inkling2_dspark_cost_depth": "cost_depth"}


def dspark_options(external):
    """Pop and validate the DSpark drafter options from ``external``."""
    from ..runtime.drafters import inkling2_dspark_cost

    options = {}
    for key, name in DSPARK_OPTIONS.items():
        if key not in external:
            continue
        value = external.pop(key)
        if name == "cost_depth":
            options[name] = inkling2_dspark_cost.parse(value)
            if options[name] is not None and external.get("adaptive_verification") is not None:
                raise ValueError(f"Inkling 2 {key} replaces adaptive_verification; set one")
            continue
        if type(value) is not bool:
            raise TypeError(f"Inkling 2 {key} must be a boolean")
        options[name] = value
    if any(options.values()) and not external.get("draft_model"):
        raise ValueError(f"Inkling 2 {sorted(DSPARK_OPTIONS)} require the DSpark route")
    return options


# validate_policy keys the adapter consumes itself; every other kernel key is a
# Model.configure_kernels keyword. Scripts and tests that configure a model from
# validate_policy go through model_kernel_options so a new adapter-only key
# cannot break them (2026-11-23: prefill_wide_chunk broke three callers).
ADAPTER_ONLY_KERNEL_KEYS = ("hard_no_thinking", "dspark_prefill_reclaim", "dspark_predraft",
                            "prefill_wide_chunk", "prefill_transient_pricing",
                            "decode_handover")


def model_kernel_options(kernels):
    """``validate_policy``'s kernel dict without the adapter-only keys."""
    return {k: v for k, v in kernels.items() if k not in ADAPTER_ONLY_KERNEL_KEYS}


def validate_policy(value):
    """``(kernel switches, external-draft keys)`` from an execution policy."""
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError("Inkling 2 execution policy must be an object")
    value = dict(value)
    kernels = {}
    defaulted = set()
    for key, name in KERNEL_OPTIONS.items():
        # The default applies only when the key is absent; a present value
        # (null included) must be a boolean.
        if key not in value:
            kernels[name] = KERNEL_DEFAULTS.get(name, False)
            defaulted.add(name)
            continue
        flag = value.pop(key)
        if type(flag) is not bool:
            raise TypeError(f"Inkling 2 {key} must be a boolean")
        kernels[name] = flag
    if kernels["moe_weighted_sum"] is None:
        kernels["moe_weighted_sum"] = kernels["moe_fused"]
    hard = value.pop(HARD_NO_THINKING_OPTION, False)
    if type(hard) is not bool:
        raise TypeError(f"Inkling 2 {HARD_NO_THINKING_OPTION} must be a boolean")
    kernels["hard_no_thinking"] = hard
    reclaim = None  # unset: resolved at load (on with DSpark)
    if PREFILL_RECLAIM_OPTION in value:
        reclaim = value.pop(PREFILL_RECLAIM_OPTION)
        if type(reclaim) is not bool:
            raise TypeError(f"Inkling 2 {PREFILL_RECLAIM_OPTION} must be a boolean")
    kernels["dspark_prefill_reclaim"] = reclaim
    predraft = value.pop(PREDRAFT_OPTION, False)
    if type(predraft) is not bool:
        raise TypeError(f"Inkling 2 {PREDRAFT_OPTION} must be a boolean")
    kernels["dspark_predraft"] = predraft
    decode_handover = value.pop(DECODE_HANDOVER_OPTION, False)
    if type(decode_handover) is not bool:
        raise TypeError(f"Inkling 2 {DECODE_HANDOVER_OPTION} must be a boolean")
    if decode_handover and kernels.get("native_replay"):
        raise ValueError(f"Inkling 2 {DECODE_HANDOVER_OPTION} is not combined with "
                         "inkling2_native_replay")
    kernels["decode_handover"] = decode_handover
    wide = value.pop(PREFILL_WIDE_OPTION, False)
    if type(wide) is not bool:
        raise TypeError(f"Inkling 2 {PREFILL_WIDE_OPTION} must be a boolean")
    pricing = value.pop(PREFILL_PRICING_OPTION, None)
    if pricing is not None and type(pricing) is not bool:
        raise TypeError(f"Inkling 2 {PREFILL_PRICING_OPTION} must be a boolean")
    if wide and pricing is False:
        raise ValueError(f"Inkling 2 {PREFILL_WIDE_OPTION} requires {PREFILL_PRICING_OPTION}")
    kernels["prefill_wide_chunk"] = wide
    kernels["prefill_transient_pricing"] = bool(wide if pricing is None else pricing)
    mask_mb = value.pop(PREFILL_MASK_OPTION, None)
    if mask_mb is not None and type(mask_mb) is not int:
        raise TypeError(f"Inkling 2 {PREFILL_MASK_OPTION} must be an integer")
    if mask_mb is not None and not 16 <= mask_mb <= 1024:
        raise ValueError(f"Inkling 2 {PREFILL_MASK_OPTION} must be 16..1024")
    kernels["prefill_mask_mb"] = mask_mb
    passes = value.pop(LOOP_PASSES_OPTION, None)
    if passes is not None and (type(passes) is not int or not 1 <= passes <= 8):
        raise TypeError(f"Inkling 2 {LOOP_PASSES_OPTION} must be an integer 1..8")
    kernels["loop_passes"] = passes
    if "moe_decode_kernel" in defaulted and (not kernels["moe_fused"]
                                             or kernels["moe_logdomain_renorm"]):
        kernels["moe_decode_kernel"] = False  # the default follows moe_fused
    if "fuse_sconv_kernel" in defaulted and not kernels["fuse_sconv"]:
        kernels["fuse_sconv_kernel"] = False  # the default follows fuse_sconv
    if kernels["fuse_sconv_kernel"] and not kernels["fuse_sconv"]:
        raise ValueError("Inkling 2 inkling2_fuse_sconv_kernel requires inkling2_fuse_sconv")
    if kernels["fuse_dconv_kernel"] and not kernels["fuse_sconv"]:
        raise ValueError("Inkling 2 inkling2_fuse_dconv_kernel requires inkling2_fuse_sconv")
    if kernels["fuse_message_kernel"] and not kernels["fuse_controllers"]:
        raise ValueError("Inkling 2 inkling2_fuse_message_kernel requires inkling2_fuse_controllers")
    if kernels["prefill_onelaunch_attention"] and (kernels["prefill_tiled_attention"]
                                                   or kernels["flash_prefill"]):
        raise ValueError("Inkling 2 inkling2_prefill_onelaunch_attention, "
                         "inkling2_prefill_tiled_attention and inkling2_flash_prefill "
                         "are alternatives")
    if kernels["prefill_tiled_attention"] and kernels["flash_prefill"]:
        raise ValueError("Inkling 2 inkling2_prefill_tiled_attention and inkling2_flash_prefill "
                         "are alternatives")
    if kernels["head_kernel_wide"] and not kernels["head_kernel"]:
        raise ValueError("Inkling 2 inkling2_head_kernel_wide requires inkling2_head_kernel")
    if kernels["attn_decode_kernel"] and not kernels["verify_splitk"]:
        raise ValueError("Inkling 2 inkling2_attn_decode_kernel requires inkling2_verify_splitk")
    if kernels["gate_merge"] and not kernels["verify_splitk"]:
        raise ValueError("Inkling 2 inkling2_gate_merge requires inkling2_verify_splitk")
    if kernels["attn_output_kernel"] and not (kernels["fuse_sconv"]
                                              and kernels["fuse_sconv_kernel"]):
        raise ValueError("Inkling 2 inkling2_attn_output_kernel requires inkling2_fuse_sconv "
                         "and inkling2_fuse_sconv_kernel")
    if kernels["compile_decode_blocks"] and not kernels["moe_fused"]:
        raise ValueError("Inkling 2 inkling2_compile_decode_blocks requires inkling2_moe_fused")
    if kernels["moe_decode_kernel"] and not kernels["moe_fused"]:
        raise ValueError("Inkling 2 inkling2_moe_decode_kernel requires inkling2_moe_fused")
    if kernels["moe_panel_kernel"] and not kernels["moe_fused"]:
        raise ValueError("Inkling 2 inkling2_moe_panel_kernel requires inkling2_moe_fused")
    if kernels["moe_nax_mxfp8"] and kernels["moe_nax_gather"] is False:
        # Unset moe_nax_gather is resolved at load (configure_kernels
        # refuses moe_nax_mxfp8 when it resolves off).
        raise ValueError("Inkling 2 inkling2_moe_nax_mxfp8 requires inkling2_moe_nax_gather")
    if kernels["moe_weighted_sum"] and not kernels["moe_fused"]:
        raise ValueError("Inkling 2 inkling2_moe_weighted_sum requires inkling2_moe_fused")
    for other in ("moe_decode_kernel", "moe_panel_kernel", "native_replay"):
        if kernels["moe_logdomain_renorm"] and kernels[other]:
            raise ValueError(f"Inkling 2 inkling2_moe_logdomain_renorm is not combined with "
                             f"inkling2_{other}")
    for other in ("kv_q8", "attn_decode_kernel"):
        if kernels["native_replay"] and kernels[other]:
            raise ValueError(f"Inkling 2 inkling2_native_replay is not combined with "
                             f"inkling2_{other}")
    return kernels, value


def _deinterleave(v):
    """Fused ``[..., 2*inter, in]`` gate/up rows (gate, up alternating) -> halves."""
    import mlx.core as mx

    *lead, rows, cols = v.shape
    w = v.reshape(*lead, rows // 2, 2, cols)
    return mx.contiguous(w[..., 0, :]), mx.contiguous(w[..., 1, :])


def install_experts(model, experts: dict, kinds: dict) -> dict:
    """Replace each layer's SwitchGLU projections with its checkpoint format.

    NVFP4: ``ScaledNVFP4SwitchLinear`` (E2M1 x E4M3/16 x FP32 per-expert
    ``amax / outer_scale_denominator``).  MXFP8: ``QuantizedSwitchLinear``
    mode ``mxfp8`` (E4M3 x E8M0/32).  Gate/up arrive row-interleaved.
    """
    import mlx.core as mx

    from ..runtime.models.switch_layers import QuantizedSwitchLinear
    from ..runtime.nvfp4_linear import ScaledNVFP4SwitchLinear

    counts = {NVFP4: 0, MXFP8: 0}
    cfg = model.config
    hidden, inter, num = cfg.hidden_size, cfg.intermediate_size, cfg.n_routed_experts
    for layer, tensors in sorted(experts.items()):
        kind = kinds[layer]
        mlp = model.model.layers[layer].mlp.switch_mlp
        for name, projections, (in_dims, out_dims) in (
            ("w13_weight", ("gate_proj", "up_proj"), (hidden, inter)),
            ("w2_weight", ("down_proj",), (inter, hidden)),
        ):
            weight = tensors.pop(name).view(mx.uint32)
            scales = tensors.pop(name + "_scale")
            parts = (
                list(zip(_deinterleave(weight), _deinterleave(scales)))
                if len(projections) == 2
                else [(weight, scales)]
            )
            global_scale = None
            if kind == NVFP4:
                amax = tensors.pop(name + "_amax").astype(mx.float32)
                denominator = tensors.pop(name + "_outer_scale_denominator").astype(mx.float32)
                global_scale = (amax / denominator).reshape(num)
            for projection, (w, s) in zip(projections, parts):
                if kind == NVFP4:
                    module = ScaledNVFP4SwitchLinear(in_dims, out_dims, num)
                    module.update({"weight": w, "scales": s, "weight_scale_2": global_scale})
                else:
                    module = QuantizedSwitchLinear(
                        in_dims, out_dims, num, bias=False, group_size=32, bits=8, mode="mxfp8"
                    )
                    module.weight, module.scales, module.biases = w, s, None
                    module.freeze()
                if module.weight.shape != (num, out_dims, in_dims * (4 if kind == NVFP4 else 8) // 32):
                    raise ValueError(f"Inkling 2 layer {layer} {projection} weight shape {module.weight.shape}")
                setattr(mlp, projection, module)
            counts[kind] += 1
        if tensors:
            raise ValueError(f"Inkling 2 layer {layer} has unused expert tensors {sorted(tensors)}")
    return {"nvfp4_layers": counts[NVFP4] // 2, "mxfp8_layers": counts[MXFP8] // 2}


def load_model(path, artifact):
    import mlx.core as mx

    from ..runtime.models.inkling2 import Model, TextConfig
    from ..runtime.ubc_evict import load_shards_evicting

    path = Path(path)
    model = Model(TextConfig.from_dict(artifact["text_config"]))

    def text_only(shard):
        return {k: v for k, v in shard.items() if k.startswith("model.llm.")}

    shards = [path / name for name in sorted(set(artifact["weight_map"].values()))]
    raw = load_shards_evicting(shards, sanitize=text_only)
    weights, experts = model.sanitize(raw)
    raw.clear()
    receipt = install_experts(model, experts, artifact["expert_kinds"])
    from mlx.utils import tree_flatten

    # Installed experts carry their own arrays; strict load checks the rest.
    weights.update((k, v) for k, v in tree_flatten(model.parameters()) if ".switch_mlp." in k)
    model.load_weights(list(weights.items()), strict=True)
    weights.clear()
    model.eval()
    mx.eval(model.parameters())
    mx.clear_cache()
    return model, receipt


DSPARK_DRAFT_NAMES = frozenset({"embedded_dspark"})


def dspark_record(path, artifact):
    """External-draft record for the bundled DSpark (header-only inspection)."""
    from types import SimpleNamespace

    from .inkling2_dspark import inspect_drafter

    inner = inspect_drafter(path, artifact["config"])
    return {
        "fingerprint": inner["draft_revision"],
        "draft_revision": inner["draft_revision"],
        "target_revision": artifact["identity"]["fingerprint"],
        "config": {"architectures": ["Inkling2DSpark"]},
        "args": SimpleNamespace(block_size=inner["block_size"], minimum_proposal_length=1),
        "dspark": inner,
    }



def _zero_thinking_budget(request):
    budget = request.get("thinking_budget")
    return type(budget) is int and budget == 0


def _numeric_effort(effort):
    """A numeric ``reasoning_effort`` rounded half-up to 2 decimals.

    The template prints the number as given (``0.333`` -> "0.333"); the
    reference renderers (vLLM #67325, SGLang #50590) write 2 decimals.  The
    template's range [0.0, 0.99] is checked here after rounding, so an
    out-of-range or non-finite value is a clear 400, not a template error.
    Named efforts ("low", "none", ...) pass through to the template's map.
    """
    import math
    from decimal import ROUND_HALF_UP, Decimal

    if isinstance(effort, bool) or not isinstance(effort, (int, float)):
        if isinstance(effort, bool):
            raise ValueError("Inkling 2 reasoning_effort must be a level name or a number")
        return effort
    if isinstance(effort, float) and not math.isfinite(effort):
        raise ValueError("Inkling 2 reasoning_effort must be finite")
    # Reject large values before converting to float or quantizing to cents.
    if not -1 < effort < 1:
        raise ValueError("Inkling 2 reasoning_effort must be in [0.0, 0.99]")
    rounded = float(Decimal(repr(float(effort))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    if not 0.0 <= rounded <= 0.99:
        raise ValueError("Inkling 2 reasoning_effort must be in [0.0, 0.99]")
    return rounded


def _tool_arguments(text):
    """OpenAI clients send ``function.arguments`` as a JSON string; the
    checkpoint template accepts only a parsed object and raised mid-render
    (sweep 2026-11-23, vLLM #67325 idea).  Malformed JSON fails closed."""
    try:
        value = json.loads(text) if text.strip() else {}
    except (ValueError, RecursionError) as error:
        raise ValueError("Inkling 2 tool call history arguments must be a JSON object") from error
    if not isinstance(value, dict):
        raise ValueError("Inkling 2 tool call history arguments must be a JSON object")
    return value

class Inkling2Adapter(ExternalDraftAdapterMixin, OrdinaryTextAdapter):
    EXTERNAL_DEFAULT_NUM_DRAFT = 7
    EXTERNAL_ROUTE_TAG = "inkling2-dspark-v1"
    EXTERNAL_PROFILE = "inkling2-apcv2-dspark"
    descriptor = DESCRIPTOR
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "inkling2-apcv2-ordinary"
    reasoning_effort_semantics = "reasoning_strength_0_0.99"
    # The DSpark route's prefill runs in the generic ExternalDraftBatchGenerator,
    # whose slices end on planned APCv2 boundaries and snapshot the target
    # rings, conv state and DSpark draft sidecar
    # (ExternalDraftBatchGenerator._capture_state_boundary); a resume from such
    # a checkpoint is greedy-exact against the cold prompt and ordinary decode,
    # ring wrapped (tests/test_inkling2_dspark_cpu.py::
    # test_engine_interior_checkpoint_resumes_like_the_cold_prompt). Without it
    # DSpark could reuse only an exact whole-prefix entry: its draft sidecar is
    # state at the entry's end, so the trimmed hits the ordinary route serves a
    # shared prefix from are refused, and GSM8K's 5-shot prefix hit 0/40
    # (qualification/runs/inkling2-gpu-20261122/16eed37f5-dsgsm8k). Own
    # __dict__ only (serving.external_state_checkpoints).
    external_apc_state_checkpoints = True

    def __init__(self, model_path, *, execution_policy=None, weight_streaming=None):
        from .process_globals import guarded_construction

        try:
            guarded_construction(
                self, lambda: self._initialize(model_path, execution_policy, weight_streaming)
            )
        except BaseException:
            self.close()
            raise

    def _initialize(self, model_path, execution_policy, weight_streaming):
        if weight_streaming is not None:
            raise ValueError("Inkling 2 declares no weight streaming mode")
        self.kernel_policy, external = validate_policy(execution_policy)
        # Opt-in community DFlash2 route (adapters/inkling2_dflash2.py):
        # selected only by a draft_model path holding a DFlash2 artifact.
        dflash2_route = self._select_dflash2_route(external)
        self.dspark_options = dspark_options(external)
        if dflash2_route and self.dspark_options:
            self._dflash2_refusal("dspark_option")
            raise ValueError(f"Inkling 2 {sorted(DSPARK_OPTIONS)} require the DSpark route")
        cost_depth = self.dspark_options.pop("cost_depth", None)
        self.dspark_cost_depth = None
        self._parse_external_policy(external, family="Inkling 2")
        if external and self.kernel_policy["loop_passes"] not in (None, _LOOP["num_passes"]):
            raise ValueError("Inkling 2 DSpark taps assume the trained loop count")
        if self.kernel_policy["dspark_prefill_reclaim"] and not external:
            raise ValueError(f"Inkling 2 {PREFILL_RECLAIM_OPTION} requires the DSpark route")
        if self.kernel_policy["dspark_predraft"] and not external:
            raise ValueError(f"Inkling 2 {PREDRAFT_OPTION} requires the DSpark route")
        if self.kernel_policy["dspark_predraft"] and (
                cost_depth is not None or self.dspark_options.get("kv_headroom")):
            # No engine-equality test covers these pairings yet (integration
            # review 2026-11-22): cost depth would discard every shorter-round
            # pre-draft, and headroom writes into the pre-draft's cache copy.
            raise ValueError(f"Inkling 2 {PREDRAFT_OPTION} is not yet combined with "
                             "inkling2_dspark_cost_depth or inkling2_dspark_kv_headroom")
        if self.kernel_policy["prefill_wide_chunk"] and external:
            raise ValueError(f"Inkling 2 {PREFILL_WIDE_OPTION} is ordinary-route only (the DSpark "
                             "prefill slicer has no adapter width plan)")
        if dflash2_route and self.kernel_policy["dspark_predraft"]:
            self._dflash2_refusal("dspark_option")
            raise ValueError(f"Inkling 2 {PREDRAFT_OPTION} requires the DSpark route")
        if self.kernel_policy["dspark_prefill_reclaim"] is None:
            # On by default for DSpark only; the DFlash2 route takes it explicitly.
            self.kernel_policy["dspark_prefill_reclaim"] = bool(external) and not dflash2_route
        if external and not dflash2_route and external.get("draft_model") not in DSPARK_DRAFT_NAMES:
            raise ValueError(f"Inkling 2 draft_model must be one of {sorted(DSPARK_DRAFT_NAMES)} "
                             "or the path of the Inkling-2-Mini-DFlash2 drafter")
        artifact = inspect_artifact(model_path)
        record = None
        if dflash2_route:
            from . import inkling2_dflash2

            if self._composition_long_copies():
                self._dflash2_refusal("unsupported_policy_key")
                raise ValueError("Inkling 2 DFlash2 route refuses prompt_lookup_max_span "
                                 "(copies past the trained block)")
            # Fail closed on the drafter before the target is loaded.
            try:
                record = inkling2_dflash2.inspect_drafter(
                    external["draft_model"], target_path=artifact["identity"]["path"],
                    text_config=artifact["text_config"],
                    target_fingerprint=artifact["identity"]["fingerprint"],
                    embed_norm=self.dflash2_embed_norm)
            except inkling2_dflash2.DFlash2Refusal as error:
                self._dflash2_refusal(error.reason, counted=True)
                raise
            try:
                self._check_num_draft(record)
            except ValueError:
                self._dflash2_refusal("num_draft")
                raise
        elif external:
            if not artifact["has_mtp"]:
                raise ValueError("Inkling 2 artifact carries no mtp.safetensors")
            # Fail closed on the drafter before the target is loaded.
            record = dspark_record(artifact["identity"]["path"], artifact)
            self._check_num_draft(record)
        self.config, self.identity, self.layout = (
            artifact["config"], artifact["identity"], CACHE_LAYOUT,
        )
        from ..process_env import (
            PROCESS_NUMERICS,
            assert_process_profile_applied,
            clear_inherited_profile,
            require_process_numerics,
        )

        self.moe_nax_gather_auto = self.kernel_policy["moe_nax_gather"] is None
        if self.moe_nax_gather_auto:
            from ..runtime.models import moe_nax_gather

            self.kernel_policy["moe_nax_gather"] = moe_nax_gather.nax_host()
        if self.kernel_policy["moe_nax_gather"]:
            from ..runtime.models import moe_nax_gather

            from .process_globals import claim_moe

            if not moe_nax_gather.nax_host():
                raise ValueError("Inkling 2 inkling2_moe_nax_gather requires an M5 NAX device")
            # Prefill-only and bit-identical to the stock sorted kernel; the
            # pad floor stays the default.
            claim_moe(self, "Inkling 2 adapter", "gather")
        else:
            from .process_globals import claim_stock_moe

            # The sorted gathers read the NAX mode and pad policy globals;
            # hold the stock selections so another live adapter cannot
            # change them underneath (as Kimi Linear does).
            claim_stock_moe(self, "Inkling 2 adapter")
        require_process_numerics("Inkling 2 native profile")
        self.environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", **PROCESS_NUMERICS}
        clear_inherited_profile(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_"))
        os.environ.update(self.environment)
        assert_process_profile_applied("Inkling 2 native profile")
        from transformers import AutoTokenizer

        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper

        path = Path(self.identity["path"])
        self.model, self.expert_receipt = load_model(path, artifact)
        for key in ADAPTER_ONLY_KERNEL_KEYS:
            setattr(self, key, self.kernel_policy.pop(key))
        self.model.configure_kernels(**self.kernel_policy)
        if self.decode_handover:
            from ..runtime.models.inkling2 import prefill_route

            prefill_route.configure_decode(self.model, decode_handover=True)
        self.prefill_schedule = None
        if self.prefill_wide_chunk:
            self._install_prefill_schedule()
        if self.kernel_policy.get("native_replay"):
            # The engine keys its programs by the weights identity too.
            self.model._native_replay.replay.weights_identity = self.checkpoint
        self.layout = self.model.apc_v2_layout
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        # <|end_message|> separates thought/text messages; only end-sampling ends the turn.
        terminal = tokenizer.convert_tokens_to_ids("<|content_model_end_sampling|>")
        if terminal != self.config["eos_token_id"]:
            raise ValueError("Inkling 2 end-sampling token differs from checkpoint EOS")
        self.tokenizer = TokenizerWrapper(
            tokenizer, detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=[terminal]
        )
        self.max_context = int(artifact["text_config"]["model_max_length"])
        self.has_mtp = artifact["has_mtp"]
        if dflash2_route:
            self._bind_dflash2(record)
        elif record is not None:
            from functools import partial

            from ..runtime.drafters.inkling2_dspark_external import load_external_drafter

            self.identity = {**self.identity, "draft_revision": record["draft_revision"],
                             "target_revision": record["target_revision"]}
            self._bind_external_drafter(
                record,
                lambda rec, model: load_external_drafter(
                    rec["dspark"], model, num_draft=self._external_num_draft(),
                    options=self.dspark_options,
                    cap_to_block=self._composition_long_copies() or self._prompt_copy_long(),
                    clamp_requests=self._prompt_copy_long()),
                self.descriptor,
            )
            if cost_depth is not None:
                self.dspark_cost_depth = self._dspark_cost_policy(cost_depth)

    def _num_draft_ceiling(self, record):
        """Block size minus one, or the composition's copy span when an explicit
        ``prompt_lookup_max_span`` is set (the drafter backend stays capped to
        its block; only prompt-lookup copies use the longer rounds)."""
        ceiling = super()._num_draft_ceiling(record)
        if self._composition_long_copies():
            span = int(self.external_policy["proposal_composition"]["prompt_lookup_max_span"])
            ceiling = max(ceiling, span)
        if self._prompt_copy_long():
            from ..runtime.prompt_copy_round import PromptCopyPolicy

            span = PromptCopyPolicy.from_value(self.external_policy["prompt_copy_round"]).max_tokens
            ceiling = max(ceiling, span)
        return ceiling

    def _prompt_copy_long(self):
        """An explicit ``prompt_copy_round`` (opt-in): its copies may run past
        the drafter block (MTPLX copies up to 24 tokens), so ``num_draft`` may
        rise to its ``max_tokens``; the DSpark backend stays capped to its
        block and clamps longer requests (``clamp_requests``)."""
        from ..runtime.prompt_copy_round import PromptCopyPolicy

        value = (getattr(self, "external_policy", None) or {}).get("prompt_copy_round")
        return PromptCopyPolicy.from_value(value) is not None

    def _composition_long_copies(self):
        """An explicit composition with ``prompt_lookup_max_span`` (opt-in)."""
        value = (getattr(self, "external_policy", None) or {}).get("proposal_composition")
        return isinstance(value, dict) and int(value.get("prompt_lookup_max_span", 0) or 0) > 0

    def _select_dflash2_route(self, external):
        """Whether ``external`` selects the community DFlash2 drafter (opt-in).

        Selected only when ``draft_model`` is a path whose config declares a
        DFlash2DraftModel; pops the route's own diagnostic key and refuses
        every policy key the route does not implement.  Counted refusals go
        to ``dflash2_refusals`` (adapter class, so a failed construction
        still leaves its count)."""
        from . import inkling2_dflash2

        self.dflash2_route = False
        self.dflash2_embed_norm = True
        draft = external.get("draft_model")
        embed_norm = external.pop(inkling2_dflash2.EMBED_NORM_OPTION, None)
        selected = (isinstance(draft, str) and draft not in DSPARK_DRAFT_NAMES
                    and inkling2_dflash2.is_dflash2_artifact(draft))
        if embed_norm is not None and not selected:
            self._dflash2_refusal("embed_norm_without_route")
            raise ValueError(f"Inkling 2 {inkling2_dflash2.EMBED_NORM_OPTION} requires the "
                             "DFlash2 route")
        if not selected:
            return False
        if embed_norm is not None and type(embed_norm) is not bool:
            self._dflash2_refusal("embed_norm_type")
            raise TypeError(f"Inkling 2 {inkling2_dflash2.EMBED_NORM_OPTION} must be a boolean")
        extra = sorted(set(external) - inkling2_dflash2.ALLOWED_POLICY_KEYS - set(DSPARK_OPTIONS))
        if extra:
            self._dflash2_refusal("unsupported_policy_key")
            raise ValueError(f"Inkling 2 DFlash2 route does not implement {extra} (chain "
                             f"route only: {sorted(inkling2_dflash2.ALLOWED_POLICY_KEYS)})")
        self.dflash2_route = True
        self.dflash2_embed_norm = embed_norm is not False
        return True

    def _bind_dflash2(self, record):
        from . import inkling2_dflash2

        self.EXTERNAL_ROUTE_TAG = inkling2_dflash2.ROUTE_TAG
        self.identity = {**self.identity, "draft_revision": record["draft_revision"],
                         "target_revision": record["target_revision"]}
        try:
            # The record fixes the embedding law (and with it the draft
            # fingerprint, so the diagnostic arm has its own identity/layout).
            self._bind_external_drafter(record, inkling2_dflash2.load_drafter, self.descriptor)
        except inkling2_dflash2.DFlash2Refusal as error:
            self._dflash2_refusal(error.reason, counted=True)
            raise
        except (ValueError, TypeError, OSError):
            self._dflash2_refusal("bind")
            raise
        self.profile_name = self._dflash2_profile_name
        self.dflash2_record_config = record["config"]
        self.dflash2_record_receipt = {k: record[k] for k in (
            "checkpoint", "fingerprint", "config_sha256", "source_repository", "source_revision",
            "tap_convention", "embedding_law", "embed_norm", "draft_revision")}
        self.descriptor = replace(
            self.descriptor, cache_layout=self.layout,
            metadata={**self.descriptor.metadata, "qualification": "unqualified",
                      "speculative_route": "external_inkling2_dflash2_chain"})

    def _dflash2_refusal(self, reason, *, counted=False):
        """Record one DFlash2 refusal on this adapter (and process-wide)."""
        from .inkling2_dflash2 import REFUSALS

        if not counted:
            REFUSALS[reason] += 1
        mine = self.__dict__.setdefault("dflash2_refusals", {})
        mine[reason] = mine.get(reason, 0) + 1

    @staticmethod
    def _dflash2_profile_name(mtp):
        if mtp:
            raise ValueError("External draft is not native MTP")
        from .inkling2_dflash2 import PROFILE

        return PROFILE

    @property
    def checkpoint(self):
        fingerprint = self.identity.get("target_fingerprint", self.identity["fingerprint"])
        return f"{Path(self.identity['path']).name}@{fingerprint}"

    def bind_prefill_step(self, prefill_step):
        self._prefill_step = int(prefill_step)
        schedule = getattr(self, "prefill_schedule", None)
        if schedule is not None and schedule.engaged:
            # prefill_wide_chunks counts lanes in forwards of the resolved step.
            from ..runtime.models.inkling2 import prefill_route

            step = int(prefill_step)
            prefill_route.set_wide_rows(self.model, step if step > schedule.narrow else None)

    def _install_prefill_schedule(self, headroom_bytes=None):
        """inkling2_prefill_wide_chunk: decide the route once at load and, if
        it engages, declare the width schedule on the model (the generic
        ``_mlx2_prefill_rows_at`` hook)."""
        from ..runtime.models.inkling2 import prefill_route
        from .inkling2_prefill_plan import WIDE_ROWS, WidePrefillSchedule, gpu_headroom_bytes

        if headroom_bytes is None:
            headroom_bytes = gpu_headroom_bytes()
        schedule = WidePrefillSchedule(headroom_bytes=headroom_bytes)
        self.prefill_schedule = schedule
        if schedule.engaged:
            object.__setattr__(self.model, "_mlx2_prefill_rows_at", schedule.rows_at)
            prefill_route.set_wide_rows(self.model, getattr(self, "_prefill_step", WIDE_ROWS))
        return schedule

    def _wide_schedule(self):
        schedule = getattr(self, "prefill_schedule", None)
        return schedule if schedule is not None and schedule.engaged else None

    def prefill_schedule_identity(self, prefill_step):
        """The width schedule's APCv2 namespace term (None: fixed step)."""
        schedule = self._wide_schedule()
        return None if schedule is None else schedule.identity(prefill_step)

    def cache_budget(self, *, mtp):
        """Per-lane cache bound (adapters/inkling2_memory.py) for lane admission
        and the APCv2 checkpoint planners (rolling, junction, interior).

        Added 2026-11-22 so the DSpark route can capture shared-prefix
        boundaries: without a projection admission plans no hybrid checkpoint
        (junctions were planned and dropped on 39/40 GSM8K prompts,
        qualification/runs/inkling2-gpu-20261122/3c69f31b2-dsjunc). It also
        replaces the generic unmeasured lane envelope on both routes and turns
        on the route-default rolling prefill checkpoints."""
        if mtp:
            raise ValueError("Inkling 2 has no native MTP route")
        draft = None
        if getattr(self, "draft_model", None) is not None and getattr(self, "dflash2_route", False):
            from .inkling2_dflash2 import budget_draft

            draft = budget_draft(self.dflash2_record_config)
        elif getattr(self, "draft_model", None) is not None:
            draft = self.config["mtp_config"]
        # The transient is the largest chunk the schedule runs at this depth
        # (Inkling2CacheBudget.prefill_transient_bytes); never narrowed.
        return self._lane_budget(
            prefill_rows=getattr(self, "_prefill_step", self.prefill_step_default()), draft=draft)

    def _lane_budget(self, *, prefill_rows, draft=None):
        from .inkling2_memory import Inkling2CacheBudget

        policy = getattr(self, "kernel_policy", {})
        return Inkling2CacheBudget.from_config(
            self.model.config, visits=self.model.model.layer_indices, draft=draft,
            prefill_rows=prefill_rows,
            num_draft=self._external_num_draft() if draft is not None else 0,
            kv_q8=bool(policy.get("kv_q8")),
            transient_pricing=bool(getattr(self, "prefill_transient_pricing", False)),
            mask_budget_mb=policy.get("prefill_mask_mb"),
            fixed_capacity_step=(_capacity_step() if policy.get("native_replay") else 0),
            schedule=(None if self._wide_schedule() is None
                      else (self.prefill_schedule.crossover, self.prefill_schedule.narrow)),
        )

    def execution_numerics_contract(self):
        root = Path(__file__).parents[1] / "runtime/models"
        names = ("inkling2/language.py", "inkling2/config.py", "inkling/cache.py",
                 "inkling/language.py", "switch_layers.py", "../nvfp4_linear.py")
        policy = getattr(self, "kernel_policy", {})
        if policy.get("moe_fused"):
            names += ("inkling2/moe_fast.py",)
        if policy.get("moe_decode_kernel"):
            names += ("inkling2/moe_decode_kernel.py",)
        if policy.get("moe_panel_kernel"):
            # It reuses moe_decode_kernel's router and contract switches.
            names += ("inkling2/moe_panel_kernel.py", "inkling2/moe_decode_kernel.py")
        if policy.get("moe_weighted_sum"):
            names += ("qwen4_moe_weighted_sum.py",)
        if policy.get("moe_nax_gather"):
            names += ("moe_nax_gather.py",)
        if policy.get("verify_splitk"):
            names += ("inkling2/splitk.py",)
        if any(policy.get(k) for k in ("fuse_controllers", "fuse_residual", "fuse_sconv",
                                       "fuse_attention", "fuse_norm_fp32", "fuse_shared_tau")):
            names += ("inkling2/fuse.py",)
        if any(policy.get(k) for k in ("fuse_sconv_kernel", "fuse_residual_kernel",
                                       "fuse_dconv_kernel", "fuse_message_kernel")):
            names += ("inkling2/fuse_kernels.py",)
        if policy.get("fuse_gather_lhs"):
            names += ("gather_lhs.py",)
        if policy.get("moe_act_quant"):
            names += ("inkling2/act_quant.py",)
        if policy.get("kv_q8"):
            # The kernels reuse splitk.py's merge kernel and split heuristics.
            names += ("inkling2/kv_q8.py", "inkling2/splitk.py")
        if any(policy.get(k) for k in ("prefill_mask_with_sink", "prefill_tiled_attention")):
            names += ("inkling2/prefill_attention.py",)
        if policy.get("prefill_handover"):
            names += ("inkling2/prefill_route.py",)
        if policy.get("attn_decode_kernel"):
            names += ("inkling2/attn_decode.py",)
        if policy.get("prefill_onelaunch_attention"):
            names += ("inkling2/prefill_onelaunch.py",)
        if policy.get("native_replay"):
            # The fixed-capacity step and the engine that may serve it.
            names += ("inkling2/capacity_state.py", "../replay/engine.py",
                      "../replay/loader.py")
        contract = {
            "operation": numerics_operation(policy),
            "kernel_policy": dict(getattr(self, "kernel_policy", {})),
            "source_sha256": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
            },
            "qualification": "unqualified",
        }
        if policy.get("kv_q8"):
            contract["approximate_state"] = kv_q8_receipt(True)
        return contract

    def execution_config(self, *, max_lanes, prefill_step):
        config = self._route_execution_config(max_lanes=max_lanes, prefill_step=prefill_step)
        reason = kv_q8_apc_refusal(getattr(self, "kernel_policy", {}))
        if reason is not None:
            config = {**config, "apcv2_reuse_disabled_reason": reason}
        return config

    def apc_publication_refusal(self, prompt_cache, tokens):
        """Decline APCv2 publication of kv_q8 (approximate, unqualified) lanes.

        Second line behind the route-level refusal in ``execution_config``:
        refuses by policy and, structurally, any lane holding int8 rows.
        Declines are counted (``diagnostics()["kv_q8"]``)."""
        reason = kv_q8_apc_refusal(getattr(self, "kernel_policy", {}))
        if reason is None and not KV_Q8_PUBLICATION_QUALIFIED and _holds_q8_rows(prompt_cache):
            reason = KV_Q8_APC_REASON
        if reason is not None:
            self.__dict__["kv_q8_apc_declines"] = self.__dict__.get("kv_q8_apc_declines", 0) + 1
        return reason

    def _route_execution_config(self, *, max_lanes, prefill_step):
        self.bind_prefill_step(prefill_step)
        if getattr(self, "draft_model", None) is not None:
            if max_lanes != 1:
                # Verify over sliding-window rings beside recurrent state is
                # B1 only (HybridVerifyRows); fail at startup, not mid-round.
                if getattr(self, "dflash2_route", False):
                    self._dflash2_refusal("max_lanes")
                raise ValueError("Inkling 2 DSpark serving is B1 only: start with --max-lanes 1")
            return self._external_execution_config(max_lanes=max_lanes, prefill_step=prefill_step)
        return {
            "persistent": True,
            "num_draft": 0,
            "backend": "ordinary",
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": False,
            "segment_aware_cohort_size": max_lanes,
        }

    def _dspark_cost_policy(self, settings):
        """``inkling2_dspark_cost_depth`` as a cohort adaptive_verification value."""
        from ..runtime.acceptance_estimator import AdaptiveVerificationPolicy
        from ..runtime.drafters import inkling2_dspark_cost

        count = self._external_num_draft()
        if "verify_ms" in settings:
            verify, draft, source = settings["verify_ms"], settings["draft_ms"], "inkling2_policy_table"
        else:
            backend = self.draft_model
            while hasattr(backend, "backend"):
                backend = backend.backend
            verify, draft = inkling2_dspark_cost.measure_costs(
                self.model, backend.drafter, count + 1,
                context=settings["measure_context"], repeats=settings["measure_repeats"],
            )
            source = "inkling2_measured_at_load"
        value = inkling2_dspark_cost.adaptive_verification(
            settings, count, verify_ms=verify, draft_ms=draft, source=source)
        AdaptiveVerificationPolicy.from_value(value, count)  # fail closed at load
        return value

    def create_external_batch(self, **kwargs):
        if getattr(self, "dspark_prefill_reclaim", False):
            kwargs.setdefault("prefill_allocator_reclaim", True)
        if getattr(self, "dspark_predraft", False):
            kwargs.setdefault("predraft", True)
        if getattr(self, "dspark_cost_depth", None) is not None:
            kwargs.setdefault("adaptive_verification", self.dspark_cost_depth)
        return super().create_external_batch(**kwargs)

    def prefill_step_default(self):
        """512 rows; 2,048 (the schedule's widest chunk) when
        inkling2_prefill_wide_chunk engaged at load."""
        if self._wide_schedule() is not None:
            from .inkling2_prefill_plan import WIDE_ROWS

            return WIDE_ROWS
        return 512

    def prefill_step_reference(self):
        """The step today's APCv2 keys are built at; any other step, and the
        wide schedule, bind the namespace (serving.prefill_step_namespace_term)."""
        return 512

    # ----- prompts and output -----

    @staticmethod
    def _reasoning_effort(request):
        # The template's lowest level is "none" (0.0), which the card says does
        # not fully disable reasoning; unset renders the template default 0.9.
        effort = request.get("reasoning_effort")
        if _zero_thinking_budget(request):
            # A zero budget is the template's lowest level, not a hard cap:
            # the model may still open a thinking block (kapitsa #1380).
            if effort not in (None, "none") or request.get("enable_thinking") is True:
                raise ValueError(
                    "Inkling 2 thinking_budget 0 conflicts with reasoning_effort/enable_thinking"
                )
            return "none"
        if effort is None:
            return "none" if request.get("enable_thinking") is False else None
        if request.get("enable_thinking") is False and effort != "none":
            raise ValueError("Inkling 2 enable_thinking false conflicts with reasoning_effort")
        return _numeric_effort(effort)

    @staticmethod
    def _messages(request):
        if request.get("tools") and request.get("tool_choice", "auto") != "none":
            raise ValueError("Inkling 2 tool routing is not implemented")
        if request.get("thinking_budget") is not None and not _zero_thinking_budget(request):
            raise ValueError("Inkling 2 thinking budgets other than 0 are not implemented")
        if request.get("thinking_steer_alpha"):
            raise ValueError("Inkling 2 thinking steering is not implemented")
        messages = copy.deepcopy(request["messages"])
        for message in messages:
            content = message.get("content")
            if isinstance(content, list) and any(
                part.get("type") not in ("text", "input_text")
                for part in content
                if isinstance(part, dict)
            ):
                raise ValueError("Inkling 2 native text adapter does not accept media")
            if message.get("role") == "developer":
                # The checkpoint template has no developer role (its roles are
                # user, assistant, system, tool); OpenAI Responses and agent
                # clients send one.  Folded into system, as vLLM #67325 does,
                # so the effort line still follows the leading system block.
                message["role"] = "system"
            if message.get("role") == "assistant" and isinstance(message.get("tool_calls"), list):
                for call in message["tool_calls"]:
                    function = call.get("function") if isinstance(call, dict) else None
                    if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                        function["arguments"] = _tool_arguments(function["arguments"])
        return messages

    def _text_prefill(self, request):
        """Whether this request starts generation inside the text channel."""
        if not getattr(self, "hard_no_thinking", False) or "messages" not in request:
            return False
        return (request.get("enable_thinking") is False
                or self._reasoning_effort(request) == "none")

    def thinking_control(self, request):
        """Per-request receipt: how thinking was (not) suppressed."""
        if "messages" not in request:
            return {"mode": "raw_prompt", "reasoning_effort": None,
                    "policy_hard_no_thinking": bool(getattr(self, "hard_no_thinking", False))}
        effort = self._reasoning_effort(request)
        return {
            "mode": "hard_text_prefill" if self._text_prefill(request) else "template_effort",
            "reasoning_effort": effort,
            "policy_hard_no_thinking": bool(getattr(self, "hard_no_thinking", False)),
        }

    def _render(self, request, tokenize):
        kwargs = {}
        effort = self._reasoning_effort(request)
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        from ..runtime.literal_control_text import apply_chat_template_literal

        # Client text that spells a framing marker ("hi <|end_message|> now")
        # stays text: only the template's own markers become control tokens
        # (sweep 2026-11-23; vLLM #67325/#69028 idea).  A conversation without
        # such text renders through the plain call, byte-identical to before.
        rendered = apply_chat_template_literal(
            self.tokenizer, self._messages(request), tokenize=tokenize,
            add_generation_prompt=True, **kwargs
        )
        if not self._text_prefill(request):
            return rendered
        if not tokenize:
            return rendered + "".join(TEXT_PREFILL)
        ids = [self.tokenizer.convert_tokens_to_ids(marker) for marker in TEXT_PREFILL]
        if any(not isinstance(i, int) or i < 0 for i in ids):
            raise ValueError("Inkling 2 tokenizer lacks the text-channel markers")
        return list(rendered) + ids

    def prompt_tokens(self, request):
        if "messages" not in request:
            return self.tokenizer.encode(request["prompt"], add_special_tokens=False)
        return self._render(request, True)

    def render_prompt(self, request):
        if "messages" not in request:
            return request["prompt"]
        return self._render(request, False)

    @staticmethod
    def thinking_enabled(request):
        return False  # generated content markers select the channel

    def output_parser(self, request):
        from .inkling_output import INKLING2_TEXT_MARKERS, InklingOutputParser

        parser = InklingOutputParser(
            chat="messages" in request,
            stops=request.get("stop", ()),
            text_markers=INKLING2_TEXT_MARKERS,
            header_names=True,
            lenient=True,  # never raise mid-stream on model output
        )
        if self._text_prefill(request):
            parser.channel = "content"  # the prompt already opened the text channel
        return parser

    def tool_constraint(self, request):
        return None

    def structured_answer_token_ids(self, request):
        """The ``<|content_text|>`` id a client grammar waits for, or None.

        The template renders no reply header: the model writes
        ``<|message_model|>`` and then ``<|content_thinking|>`` or
        ``<|content_text|>`` itself. A ``response_format``/``grammar`` bound
        from the first generated token excludes those special ids, so the
        model would have to answer with no header at all, off its trained
        framing (omlx#5635 measured 17/71 correct extractions this way on
        Inkling 2 Mini). The grammar therefore binds once the text channel is
        open; the answer then ends at end-sampling, as an unconstrained
        reply does. None when the prompt already opened the text channel
        (``inkling2_hard_no_thinking``), for a raw completion, or when the
        marker does not encode to one id that decodes back exactly."""
        if "messages" not in request or self._text_prefill(request):
            return None
        marker = TEXT_PREFILL[1]
        try:
            ids = [int(token) for token in self.tokenizer.encode(marker, add_special_tokens=False)]
            if len(ids) != 1 or self.tokenizer.decode(ids) != marker:
                return None
        except Exception:  # noqa: BLE001 - undeclared marker, not a load failure
            return None
        return tuple(ids)

    def structured_envelope_token_ids(self):
        """``((), (<|end_message|>,))``: a constrained answer may close its
        text block once the grammar is complete, as the model does unconstrained
        (omlx#5635 counted <|end_message|> after every answer), before
        end-sampling. Nothing opens inside the answer: the header comes before
        the ``structured_answer_token_ids`` deferral point. None when the marker
        does not encode to one id that decodes back exactly."""
        marker = "<|end_message|>"
        try:
            ids = [int(token) for token in self.tokenizer.encode(marker, add_special_tokens=False)]
            if len(ids) != 1 or self.tokenizer.decode(ids) != marker:
                return None
        except Exception:  # noqa: BLE001 - undeclared marker, not a load failure
            return None
        return (), tuple(ids)

    def _native_replay_receipt(self):
        from ..runtime.models.inkling2 import capacity_state

        model = getattr(self, "model", None)
        if model is None or not getattr(self, "kernel_policy", {}).get("native_replay"):
            return {"selected": False, "qualified": False}
        return {"selected": True, "qualified": False, **capacity_state.receipt(model)}

    def diagnostics(self):
        return {
            "family": "inkling2",
            "route": "ordinary",
            "qualification": "unqualified",
            "checkpoint": self.checkpoint if getattr(self, "identity", None) else None,
            "adapter_default_route": "ordinary",
            "ordinary_reference_preserved": True,
            "cache_layout": self.layout,
            "numerics": numerics_operation(getattr(self, "kernel_policy", {})),
            "kv_q8": {**kv_q8_receipt(bool(getattr(self, "kernel_policy", {}).get("kv_q8")),
                                      layout=getattr(self, "layout", None)),
                      "apcv2_publication_declines": self.__dict__.get("kv_q8_apc_declines", 0)},
            "kernel_policy": dict(getattr(self, "kernel_policy", {})),
            "moe_nax_gather_auto": bool(getattr(self, "moe_nax_gather_auto", False)),
            "kernel_counters": (self.model.kernel_receipt() if getattr(self, "model", None) else None),
            "prefill_schedule": (self.prefill_schedule.receipt()
                                 if getattr(self, "prefill_schedule", None) else {"selected": False}),
            "prefill_transient_pricing": bool(getattr(self, "prefill_transient_pricing", False)),
            "loop": (self.model.loop_receipt() if getattr(self, "model", None) else None),
            "hard_no_thinking": bool(getattr(self, "hard_no_thinking", False)),
            "decode_handover": {"selected": bool(getattr(self, "decode_handover", False)),
                                "qualified": False},
            "native_replay": self._native_replay_receipt(),
            "experts": getattr(self, "expert_receipt", None),
            "target_forward_observed": bool(getattr(getattr(self, "model", None), "completed_forwards", 0)),
            "dspark": _dspark_receipt(self),
            "media": "weights present; native text route refuses media",
            # Present only when this adapter selected the opt-in DFlash2
            # route or recorded a refusal of it, so other routes' receipts
            # are unchanged.
            **({"dflash2": _dflash2_receipt(self)} if _dflash2_seen(self) else {}),
        }


def _dspark_receipt(self):
    if getattr(self, "dflash2_route", False):
        # The DFlash2 route's drafter counters live under "dflash2".
        return {"implemented": True, "selected": False, "qualified": False,
                "weights_present": bool(getattr(self, "has_mtp", False)),
                "not_selected_reason": "dflash2_route_selected"}
    return {
        "implemented": True,
        "selected": getattr(self, "draft_model", None) is not None,
        "qualified": False,
        "weights_present": bool(getattr(self, "has_mtp", False)),
        "num_draft": self._external_num_draft() if getattr(self, "draft_model", None) is not None else None,
        # prompt_copy_round only: the engine depth above is the copy
        # span; DSpark rounds draft at most this many tokens.
        **({"dspark_draft_depth": _dspark_backend_depth(self.draft_model)}
           if getattr(self, "draft_model", None) is not None
           and getattr(self, "_prompt_copy_long", lambda: False)() else {}),
        "stats": _drafter_stats(getattr(self, "draft_model", None)),
        "prefill_reclaim": bool(getattr(self, "dspark_prefill_reclaim", False)),
        "predraft": bool(getattr(self, "dspark_predraft", False)),
        "options": _drafter_options(getattr(self, "draft_model", None)),
        "cost_depth": getattr(self, "dspark_cost_depth", None),
    }


def _dflash2_seen(adapter):
    return bool(getattr(adapter, "dflash2_route", False)) or bool(
        adapter.__dict__.get("dflash2_refusals"))


def _dflash2_receipt(adapter):
    """Route receipt of the opt-in community DFlash2 drafter."""
    from . import inkling2_dflash2

    selected = bool(getattr(adapter, "dflash2_route", False)) and getattr(
        adapter, "draft_model", None) is not None
    # observed-used is the engine's speculative counters, not this receipt.
    out = {"implemented": True, "selected": selected, "qualified": False,
           "refusals": dict(sorted(adapter.__dict__.get("dflash2_refusals", {}).items())),
           "process_refusals": dict(sorted(inkling2_dflash2.REFUSALS.items()))}
    if selected:
        draft = adapter.draft_model
        backend = draft
        while hasattr(backend, "backend"):
            backend = backend.backend
        out.update(num_draft=adapter._external_num_draft(),
                   record=getattr(adapter, "dflash2_record_receipt", None),
                   drafter=getattr(backend, "inkling2_dflash2_receipt", None),
                   settings=getattr(draft, "receipt_settings", None),
                   embed_norm=bool(getattr(adapter, "dflash2_embed_norm", True)),
                   prefill_reclaim=bool(getattr(adapter, "dspark_prefill_reclaim", False)))
    return out


def _capacity_step():
    from ..runtime.models.inkling2 import capacity_state

    return capacity_state.CAPACITY_STEP


def kv_q8_receipt(selected, *, layout=None):
    """Route receipt of ``inkling2_kv_q8``: approximate, unqualified."""
    from ..runtime.models.inkling2 import kv_q8

    out = {"operation": KV_Q8_OPERATION, "implemented": True, "selected": bool(selected),
           "qualified": False, "fidelity": "approximate" if selected else "exact",
           "scope": "8 global attention visits; sliding-window rings stay BF16",
           "format": kv_q8.FORMAT, "group_size": kv_q8.GROUP, "scale_dtype": "float16",
           "layout_suffix": kv_q8.LAYOUT_SUFFIX}
    reason = kv_q8_apc_refusal({"kv_q8": bool(selected)})
    out["apcv2_publication"] = {"enabled": reason is None, "reason": reason,
                                "qualified_switch": KV_Q8_PUBLICATION_QUALIFIED}
    if layout is not None:
        out["cache_layout"] = layout
    return out


def _dspark_backend_depth(draft):
    """The DSpark backend's own draft depth (below any wrapper)."""
    while hasattr(draft, "backend"):
        draft = draft.backend
    depth = getattr(draft, "num_draft", None)
    return int(depth) if depth is not None else None


def _drafter_stats(draft):
    while draft is not None and hasattr(draft, "backend"):
        draft = draft.backend
    return dict(getattr(draft, "stats", {}) or {}) if draft is not None else None


def _drafter_options(draft):
    while draft is not None and hasattr(draft, "backend"):
        draft = draft.backend
    drafter = getattr(draft, "drafter", None)
    return getattr(drafter, "option_receipt", None)
