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
    "inkling2_compile_decode_blocks": "compile_decode_blocks",
    # and the logit softcap compiled into one kernel on the GPU.
    "inkling2_head_softcap_compiled": "head_softcap_compiled",
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
KERNEL_DEFAULTS = {"moe_fused": True, "moe_weighted_sum": None, "moe_nax_gather": None,
                   "sink_sdpa": True, "sink_slot": True, "verify_splitk": True,
                   "fuse_controllers": True, "fuse_residual": True, "fuse_sconv": True,
                   "fuse_attention": True, "fuse_sconv_kernel": True}
# The policy that selects the reference path for every switch the defaults
# turn on (comparison scripts and the ladder's reference arm start from it).
REFERENCE_POLICY = {key: False for key, name in KERNEL_OPTIONS.items()
                    if KERNEL_DEFAULTS.get(name) is True and not name.startswith("moe_")}
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
    if "fuse_sconv_kernel" in defaulted and not kernels["fuse_sconv"]:
        kernels["fuse_sconv_kernel"] = False  # the default follows fuse_sconv
    if kernels["fuse_sconv_kernel"] and not kernels["fuse_sconv"]:
        raise ValueError("Inkling 2 inkling2_fuse_sconv_kernel requires inkling2_fuse_sconv")
    if kernels["fuse_dconv_kernel"] and not kernels["fuse_sconv"]:
        raise ValueError("Inkling 2 inkling2_fuse_dconv_kernel requires inkling2_fuse_sconv")
    if kernels["fuse_message_kernel"] and not kernels["fuse_controllers"]:
        raise ValueError("Inkling 2 inkling2_fuse_message_kernel requires inkling2_fuse_controllers")
    if kernels["compile_decode_blocks"] and not kernels["moe_fused"]:
        raise ValueError("Inkling 2 inkling2_compile_decode_blocks requires inkling2_moe_fused")
    if kernels["moe_weighted_sum"] and not kernels["moe_fused"]:
        raise ValueError("Inkling 2 inkling2_moe_weighted_sum requires inkling2_moe_fused")
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
        self._parse_external_policy(external, family="Inkling 2")
        if external and self.kernel_policy["loop_passes"] not in (None, _LOOP["num_passes"]):
            raise ValueError("Inkling 2 DSpark taps assume the trained loop count")
        if self.kernel_policy["dspark_prefill_reclaim"] and not external:
            raise ValueError(f"Inkling 2 {PREFILL_RECLAIM_OPTION} requires the DSpark route")
        if self.kernel_policy["dspark_prefill_reclaim"] is None:
            self.kernel_policy["dspark_prefill_reclaim"] = bool(external)
        if external and external.get("draft_model") not in DSPARK_DRAFT_NAMES:
            raise ValueError(f"Inkling 2 draft_model must be one of {sorted(DSPARK_DRAFT_NAMES)}")
        artifact = inspect_artifact(model_path)
        record = None
        if external:
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
        self.hard_no_thinking = self.kernel_policy.pop("hard_no_thinking")
        self.dspark_prefill_reclaim = self.kernel_policy.pop("dspark_prefill_reclaim")
        self.model.configure_kernels(**self.kernel_policy)
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
        if record is not None:
            from functools import partial

            from ..runtime.drafters.inkling2_dspark_external import load_external_drafter

            self.identity = {**self.identity, "draft_revision": record["draft_revision"],
                             "target_revision": record["target_revision"]}
            self._bind_external_drafter(
                record,
                lambda rec, model: load_external_drafter(
                    rec["dspark"], model, num_draft=self._external_num_draft()),
                self.descriptor,
            )

    @property
    def checkpoint(self):
        fingerprint = self.identity.get("target_fingerprint", self.identity["fingerprint"])
        return f"{Path(self.identity['path']).name}@{fingerprint}"

    def bind_prefill_step(self, prefill_step):
        self._prefill_step = int(prefill_step)

    def cache_budget(self, *, mtp):
        """Per-lane cache bound (adapters/inkling2_memory.py) for lane admission
        and the APCv2 checkpoint planners (rolling, junction, interior).

        Added 2026-11-22 so the DSpark route can capture shared-prefix
        boundaries: without a projection admission plans no hybrid checkpoint
        (junctions were planned and dropped on 39/40 GSM8K prompts,
        qualification/runs/inkling2-gpu-20261122/3c69f31b2-dsjunc). It also
        replaces the generic unmeasured lane envelope on both routes and turns
        on the route-default rolling prefill checkpoints."""
        from .inkling2_memory import Inkling2CacheBudget

        if mtp:
            raise ValueError("Inkling 2 has no native MTP route")
        draft = None
        if getattr(self, "draft_model", None) is not None:
            draft = self.config["mtp_config"]
        return Inkling2CacheBudget.from_config(
            self.model.config, visits=self.model.model.layer_indices, draft=draft,
            prefill_rows=getattr(self, "_prefill_step", self.prefill_step_default()),
            num_draft=self._external_num_draft() if draft is not None else 0,
        )

    def execution_numerics_contract(self):
        root = Path(__file__).parents[1] / "runtime/models"
        names = ("inkling2/language.py", "inkling2/config.py", "inkling/cache.py",
                 "inkling/language.py", "switch_layers.py", "../nvfp4_linear.py")
        policy = getattr(self, "kernel_policy", {})
        if policy.get("moe_fused"):
            names += ("inkling2/moe_fast.py",)
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
        return {
            "operation": NUMERICS_W4A4_FAKE if policy.get("moe_act_quant") else NUMERICS,
            "kernel_policy": dict(getattr(self, "kernel_policy", {})),
            "source_sha256": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
            },
            "qualification": "unqualified",
        }

    def execution_config(self, *, max_lanes, prefill_step):
        self.bind_prefill_step(prefill_step)
        if getattr(self, "draft_model", None) is not None:
            if max_lanes != 1:
                # Verify over sliding-window rings beside recurrent state is
                # B1 only (HybridVerifyRows); fail at startup, not mid-round.
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

    def create_external_batch(self, **kwargs):
        if getattr(self, "dspark_prefill_reclaim", False):
            kwargs.setdefault("prefill_allocator_reclaim", True)
        return super().create_external_batch(**kwargs)

    @staticmethod
    def prefill_step_default():
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
        return effort

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
        rendered = self.tokenizer.apply_chat_template(
            self._messages(request), tokenize=tokenize, add_generation_prompt=True, **kwargs
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
        )
        if self._text_prefill(request):
            parser.channel = "content"  # the prompt already opened the text channel
        return parser

    def tool_constraint(self, request):
        return None

    def diagnostics(self):
        return {
            "family": "inkling2",
            "route": "ordinary",
            "qualification": "unqualified",
            "checkpoint": self.checkpoint if getattr(self, "identity", None) else None,
            "adapter_default_route": "ordinary",
            "ordinary_reference_preserved": True,
            "cache_layout": self.layout,
            "numerics": (NUMERICS_W4A4_FAKE if getattr(self, "kernel_policy", {}).get("moe_act_quant")
                         else NUMERICS),
            "kernel_policy": dict(getattr(self, "kernel_policy", {})),
            "moe_nax_gather_auto": bool(getattr(self, "moe_nax_gather_auto", False)),
            "kernel_counters": (self.model.kernel_receipt() if getattr(self, "model", None) else None),
            "loop": (self.model.loop_receipt() if getattr(self, "model", None) else None),
            "hard_no_thinking": bool(getattr(self, "hard_no_thinking", False)),
            "experts": getattr(self, "expert_receipt", None),
            "target_forward_observed": bool(getattr(getattr(self, "model", None), "completed_forwards", 0)),
            "dspark": {
                "implemented": True,
                "selected": getattr(self, "draft_model", None) is not None,
                "qualified": False,
                "weights_present": bool(getattr(self, "has_mtp", False)),
                "num_draft": self._external_num_draft() if getattr(self, "draft_model", None) is not None else None,
                "stats": _drafter_stats(getattr(self, "draft_model", None)),
                "prefill_reclaim": bool(getattr(self, "dspark_prefill_reclaim", False)),
            },
            "media": "weights present; native text route refuses media",
        }


def _drafter_stats(draft):
    while draft is not None and hasattr(draft, "backend"):
        draft = draft.backend
    return dict(getattr(draft, "stats", {}) or {}) if draft is not None else None
