# SPDX-License-Identifier: Apache-2.0
# Ported from the Hy4-flash reference implementations (transformers PR 50903,
# vLLM PR 68296); MLA absorption mined from mlx2's xing4_0.py.  See
# provenance/hy-v4-flash.json.
"""Tencent Hy4-flash (``hy_v4_flash``) text backbone: ordinary-decode reference.

Topology (``config.json`` of tencent/Hy4-flash@d88c9dfd):

* 42 decoder layers on ``hc_mult`` = 4 identity hyper-connection streams
  (iHC: sigmoid pre/post gates, no Sinkhorn ``comb``), sandwich norms;
* gated MLA attention with learned sinks in two shapes: ``indexed_attention``
  layers (32 heads, RoPE theta 1e7) attend to the tokens a compressed
  indexer recalls, ``sliding_attention`` layers (64 heads, theta 1e4) to the
  last ``sliding_window`` keys;
* layer 0 dense MLP, layers 1..41 latent MoE (512 experts, top-16 sigmoid
  router with correction bias, routed experts on a 1536-wide latent, one
  shared expert, clamped SwiGLU);
* Engram hashed bi/tri-gram memories added to the streams before layers
  1, 11, 21, 31 (config ``engram_layer_ids`` are 1-based: 2, 12, 22, 32).

Everything is written from the reference math; the transcription notes that
matter numerically are at each block.

Image input (``hy_v4_flash_vision``): a media prefill passes
``pixel_values``/``image_grid_thw``; the tower's features replace the image
placeholder embeddings and every layer cache records the chunk's vision spans
(``mrope_spans``), whose 3D interleaved M-RoPE positions the attention and
indexer read.  Chunks that touch no span keep the scalar fast-RoPE path at
``slot + delta`` (``delta`` 0 for text-only requests, so their math is
unchanged).  Video spans fail closed.

The attention is MLA in absorbed form (one latent key/value head, per-head
``embed_q`` / ``unembed_out``), which is the reference ``expand_kv`` math up
to rounding.  Sinks are applied as a zero-valued extra key whose logit is the
head's sink.  Sliding layers keep ``sliding_window - 1 + ROLLBACK`` latents so
speculative rollback stays exact.

This module also holds the DFly2 drafter (``HY4FlashDFly``): five dense MLA
layers that read the target's latent KV and its indexer selections.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..cache_operations import CacheOperationsContract, CachePlaneDeclaration
from .base import BaseModelArgs
from .cache import _BaseCache
from .hy_v4_flash_fp8kv import FORMATS as LATENT_KV_FORMATS
from .hy_v4_flash_fp8kv import ROW_REVISION as FP8_ROW_REVISION
from .hy_v4_flash_fp8kv import pack_rows as fp8_pack_rows
from .hy_v4_flash_fp8kv import row_bytes as fp8_row_bytes
from .hy_v4_flash_fp8kv import unpack_rows as fp8_unpack_rows
from . import hy_v4_flash_union_prefill as _union_prefill
from .hy_v4_flash_sparse_prefill import sparse_prefill_attend
from .hy_v4_flash_sparse_prefill import fp8_unsupported_reason as _sparse_fp8_reason
from .hy_v4_flash_sparse_prefill import sparse_prefill_attend_fp8 as _sparse_prefill_attend_fp8
from .hy_v4_flash_sparse_prefill import unsupported_reason as sparse_unsupported_reason
from .hy_v4_flash_vision import (chunk_positions, find_spans, mrope_planes, slot_positions,
                                 span_end)
from . import hy_v4_flash_fusions as _fusions
from . import hy_v4_flash_resident as _resident
from . import moe_nax_gather as _moe_nax
from . import switch_read_once as _read_once
from .switch_layers import SwitchGLU

INDEXED = "indexed_attention"
SLIDING = "sliding_attention"
_LEGACY_LAYER_TYPES = {
    "deepseek_sparse_attention": INDEXED,
    "indexed_attention": INDEXED,
    "full_attention": INDEXED,
    "sliding_attention": SLIDING,
}
# Speculative rollback bound for the sliding layers' retained latents.
ROLLBACK = 64
# Query rows above this use one shared key set with a mask instead of
# gathering each row's recalled tokens.
GATHER_MAX_QUERY = 64
# Sparse prefill engages only once the history holds at least this many keys
# (micro-bench: 1.2-3.9x at 8K, 5-12x at 32K; short prompts favour dense SDPA).
SPARSE_PREFILL_MIN_KEYS = 4096
# Bound on [rows, heads, keys] fp32 scores per attention chunk.
SCORE_BUDGET = 1 << 27
# Evaluate each score block before building the next, so a chunked attention
# or indexer forward holds one block of ``SCORE_BUDGET`` elements, not all of
# them.  Scheduling only (bit-identical outputs); single-block forwards
# (decode, verify, short prefill) never evaluate early.  The cache budget
# (``adapters/hy_v4_flash_memory``) charges unbounded blocks when it is off.
BOUNDED_SCORE_BLOCKS = True
# A media prefill arrives as one chunk of the whole prompt; its layers run in
# slices of this many tokens so the per-chunk masks stay bounded.
MEDIA_PREFILL_STEP = 2048
# Routed-expert counts at which ``qwen4_moe_weighted_sum`` was verified bit for
# bit against the eager combine on Metal for this model's shapes (fp32 router
# weights, bf16 latent rows; tests/test_hy_v4_flash_prefill_exact_metal.py).
# Hy4-flash routes 16; ``moe_top_k`` overrides fall back to the eager tail.
SORTED_SUM_TOP_K = (16,)
# ``prefill_tail_skip`` FAMILY_DEFAULTS key (default on, Pierre 2026-11-21).
TAIL_SKIP_FAMILY = "hy_v4_flash"


def _normalize_layer_type(value: str) -> str:
    try:
        return _LEGACY_LAYER_TYPES[value]
    except KeyError:
        raise ValueError(f"unsupported Hy4-flash layer type {value!r}") from None


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "hy_v4_flash"
    vocab_size: int = 120832
    hidden_size: int = 3072
    intermediate_size: int = 13824
    moe_intermediate_size: int = 1536
    routed_expert_hidden_size: Optional[int] = 1536
    num_hidden_layers: int = 42
    num_attention_heads: int = 32
    swa_num_attention_heads: int = 64
    q_lora_rank: int = 1024
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    swa_qk_nope_head_dim: int = 192
    swa_qk_rope_head_dim: int = 64
    swa_v_head_dim: int = 256
    layer_types: List[str] = field(default_factory=list)
    mlp_layer_types: List[str] = field(default_factory=list)
    rope_parameters: Dict[str, Any] = field(default_factory=dict)
    sliding_window: int = 512
    rms_norm_eps: float = 1e-5
    n_routed_experts: int = 512
    num_experts_per_tok: int = 16
    n_shared_experts: int = 1
    routed_scaling_factor: float = 3.998
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    swiglu_limit: float = 10.0
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_magnitude: float = 2.0
    compressed_indexer_compress_ratio: int = 4
    compressed_indexer_num_attention_heads: int = 16
    compressed_indexer_head_dim: int = 128
    compressed_indexer_group_topk: int = 512
    engram_layer_ids: Optional[List[int]] = None
    engram_max_ngram_size: int = 3
    engram_n_head_per_ngram: int = 8
    engram_n_embed_per_ngram: int = 768
    engram_layer_head_vocab_sizes: Optional[Dict[str, List[List[int]]]] = None
    engram_pad_id: int = 120002
    engram_fp8_embedding: bool = True
    engram_seed: int = 0
    engram_lookup_table_size: int = 120818
    # Engram norm/fp8 profile; see EngramLayer (default: the transformers port).
    engram_precision: str = "transformers"
    # Latent KV row format: "native" (the activation dtype, exact) or "fp8"
    # (approximate, opt-in, unqualified; see hy_v4_flash_fp8kv).
    latent_kv: str = "native"
    learnable_sink: bool = True
    tie_word_embeddings: bool = False
    enable_lm_head_fp32: bool = True
    max_position_embeddings: int = 1048576
    image_token_id: Optional[int] = None
    video_token_id: Optional[int] = None
    # DFly2 drafter (top-level ``dfly_*`` keys of the composite config).
    dfly_num_hidden_layers: int = 0
    dfly_mask_token_id: Optional[int] = None
    dfly_target_index_layer: Optional[List[int]] = None
    dfly_target_kv_layers: Optional[List[int]] = None
    dfly_target_index_topk: Optional[int] = None
    dfly_context_local_window: Optional[int] = None
    dfly_mtp_fusion: str = "slot0"
    dfly_enable_hidden_correction: bool = True
    num_speculative_tokens: int = 5

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = [
                INDEXED if i < 2 or (i - 1) % 4 == 0 else SLIDING
                for i in range(self.num_hidden_layers)
            ]
        self.layer_types = [_normalize_layer_type(t) for t in self.layer_types]
        if not self.mlp_layer_types:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
        if (len(self.layer_types) != self.num_hidden_layers
                or len(self.mlp_layer_types) != self.num_hidden_layers):
            raise ValueError("Hy4-flash layer pattern length mismatch")
        rope = {}
        for key, value in (self.rope_parameters or {}).items():
            rope[_normalize_layer_type(key)] = dict(value)
        for kind, theta in ((INDEXED, 1e7), (SLIDING, 1e4)):
            rope.setdefault(kind, {"rope_type": "default", "rope_theta": theta})
            if rope[kind].get("rope_type", "default") != "default":
                raise ValueError("Hy4-flash supports only default RoPE")
        self.rope_parameters = rope
        if self.n_group != 1 or self.topk_group != 1:
            raise ValueError("Hy4-flash router groups other than 1 are not implemented")
        if self.latent_kv not in LATENT_KV_FORMATS:
            raise ValueError(f"Hy4-flash latent_kv must be one of {LATENT_KV_FORMATS}")

    @property
    def candidate_topk(self) -> int:
        return self.compressed_indexer_group_topk * self.compressed_indexer_compress_ratio

    @property
    def engram_layers(self) -> Dict[int, int]:
        """Decoder layer index -> 1-based Engram layer id."""
        return {lid - 1: lid for lid in (self.engram_layer_ids or [])}

    @classmethod
    def from_hf_config(cls, config: dict) -> "ModelArgs":
        """Flatten the composite ``HYV4FlashConfig`` (text_config + dfly keys)."""
        text = dict(config.get("text_config") or config)
        for key in ("image_token_id", "video_token_id"):
            if key in config:
                text.setdefault(key, config[key])
        for key, value in config.items():
            if key.startswith("dfly_"):
                text[key] = value
        spec = config.get("speculative_config") or {}
        if "num_speculative_tokens" in spec:
            text["num_speculative_tokens"] = int(spec["num_speculative_tokens"])
        return cls.from_dict(text)


# ----------------------------------------------------------------- numerics --

TORCH_RMS_EPS = float(np.finfo(np.float32).eps)


def rms_norm(x, weight, eps):
    return mx.fast.rms_norm(x, weight, eps)


def torch_rms_norm(x, weight=None):
    """``torch.nn.RMSNorm(dims)`` with ``eps=None``, as the Engram layers use it.

    On bf16/fp16 input torch takes the epsilon of the fp32 opmath type
    (``finfo(float32).eps``, not the input dtype's 2**-7 for bf16) and applies
    the weight in fp32 before the one rounding to the input dtype (checked on
    torch 2.8 and 2.14; Nace.AI's Hy4 port agrees).  ``mx.fast.rms_norm`` on
    bf16 rounds before it scales, which is the model's own ``HYV4FlashRMSNorm``
    order, so here the whole norm runs in fp32 and is rounded once.
    ``weight`` may be ``[..., d]`` (stacked per-head gammas); None means 1.
    """
    dtype = x.dtype
    xf = x.astype(mx.float32)
    if weight is None or weight.ndim == 1:
        w = None if weight is None else weight.astype(mx.float32)
        return mx.fast.rms_norm(xf, w, TORCH_RMS_EPS).astype(dtype)
    y = mx.fast.rms_norm(xf, None, TORCH_RMS_EPS) * weight.astype(mx.float32)
    return y.astype(dtype)


def inv_frequencies(base: float, dim: int) -> mx.array:
    return 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))


def rope_tables(positions: mx.array, inv_freq: mx.array, dtype):
    """cos/sin ``[..., L, dim/2]``; cast to the activation dtype as the reference does."""
    freqs = positions.astype(mx.float32)[..., None] * inv_freq
    return mx.cos(freqs).astype(dtype), mx.sin(freqs).astype(dtype)


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Interleaved (adjacent-pair) rotation of the whole last axis, in fp32.

    The reference rotates ``(x[2i], x[2i+1])`` by frequency ``i``
    (``rotate_half`` on even/odd lanes, ``cos[..., :d/2].repeat_interleave``).
    ``cos``/``sin`` broadcast against ``x[..., d/2]``.
    """
    dtype = x.dtype
    pairs = x.astype(mx.float32).reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    c, s = cos.astype(mx.float32), sin.astype(mx.float32)
    out = mx.stack([a * c - b * s, b * c + a * s], axis=-1)
    return out.reshape(x.shape).astype(dtype)


def mrope_rotate(x: mx.array, positions: np.ndarray, base: float, section) -> mx.array:
    """Interleaved 3D M-RoPE of the whole last axis of ``x [..., L, d]``.

    ``positions`` is ``[3, L]`` (T, H, W); frequency ``i`` reads the plane
    ``mrope_planes`` assigns it (``recomposition_frequencies``).  With all
    three planes equal this is ``mx.fast.rope(traditional=True)``.
    """
    if section is None:
        raise ValueError("Hy4-flash vision positions need an mrope_section")
    d = x.shape[-1]
    plane = mrope_planes(d // 2, section)
    pos = np.asarray(positions, dtype=np.int64)[plane].T.astype(np.float32)  # [L, d/2]
    freqs = mx.array(pos) * inv_frequencies(base, d)
    return apply_rope(x, mx.cos(freqs), mx.sin(freqs))


def fp8_quantize(x: mx.array):
    """``(E4M3 codes uint8, per-vector scale in x.dtype)`` of the Engram round trip.

    The scale is ``amax / 448`` over the last axis, computed in ``x.dtype``.
    The converter's ``fp8_e4m3_row`` Engram tables are written with this very
    function, so decoding them (``codes * scale``) reproduces
    ``fp8_round_trip`` bit for bit when the model runs in the same dtype.
    """
    scale = mx.maximum(mx.max(mx.abs(x), axis=-1, keepdims=True), 1e-12) / 448.0
    scale = scale.astype(x.dtype)
    return mx.to_fp8(mx.clip(x / scale, -448.0, 448.0)), scale


def fp8_round_trip(x: mx.array) -> mx.array:
    """Per-vector E4M3 quantize/dequantize, as Engram rows were trained to be read."""
    q, scale = fp8_quantize(x)
    return mx.from_fp8(q, dtype=x.dtype) * scale


# --------------------------------------------------------------------- MLP --

class MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int, *, post_norm: bool, eps: float):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        if post_norm:
            self.post_fc2_norm = nn.RMSNorm(hidden, eps=eps)

    def __call__(self, x):
        y = self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
        norm = getattr(self, "post_fc2_norm", None)
        return norm(y) if norm is not None else y


@lru_cache(maxsize=None)
def _clamped_swiglu_kernel(limit: float):
    def fn(up, gate):
        return nn.silu(mx.minimum(gate, limit)) * mx.clip(up, -limit, limit)

    return mx.compile(fn)


class ClampedSwiGLU(nn.Module):
    """``silu(min(gate, l)) * clip(up, -l, l)`` (HYV4FlashExperts._apply_gate), compiled."""

    def __init__(self, limit: float):
        super().__init__()
        self.limit = float(limit)

    def __call__(self, up, gate):
        return _clamped_swiglu_kernel(self.limit)(up, gate)


class Router(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.scale = args.routed_scaling_factor
        self.norm = args.norm_topk_prob
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((args.n_routed_experts,), dtype=mx.float32)

    def __call__(self, x):
        # fp32 weight: the reference expression; a BF16-resident weight (opt-in
        # ``resident_bf16``, H5) is widened at use (hy_v4_flash_resident).
        logits = _resident.f32_matmul(x, self.weight, kernel=getattr(self, "resident_gemv", False))
        scores = mx.sigmoid(logits)
        choice = scores + self.e_score_correction_bias.astype(mx.float32)
        k = self.top_k
        # Discrete ids carry no gradient (value-neutral; the weights stay differentiable).
        indices = mx.stop_gradient(mx.argpartition(-choice, kth=k - 1, axis=-1)[..., :k].astype(mx.int32))
        weights = mx.take_along_axis(scores, indices, axis=-1)
        if self.norm:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        return indices, weights * self.scale


class Q2ZeroSwitchLinear(nn.Module):
    """Routed experts in the third-party ``q2_0`` mode (2-bit, no stored biases).

    ``q2_0`` (Nace.AI Hy4-flash pack, provenance ``hy-v4-flash.json``) is MLX
    affine 2-bit with ``biases = -scales``: a weight is ``scale * (q - 1)``,
    ``q`` in 0..3, 16 codes per little-endian U32, low bits first.  MLX 0.32.4
    has no ``q2_0`` mode, so the pack's F16 scales are widened to fp32 at load
    (``q2_0_planes``) and the gather runs as ``gather_qmm(bits=2, affine)``
    with fp32 input, scales and biases, the result cast back to the input
    dtype (the pack's own execution).  Same call contract as
    ``QuantizedSwitchLinear``.
    """

    mode = "affine"
    source_mode = "q2_0"

    def __init__(self, input_dims: int, output_dims: int, num_experts: int, group_size: int = 64):
        super().__init__()
        if input_dims % group_size or input_dims % 16:
            raise ValueError("q2_0 experts need an input width divisible by the group size and 16")
        self.weight = mx.zeros((num_experts, output_dims, input_dims // 16), dtype=mx.uint32)
        self.scales = mx.zeros((num_experts, output_dims, input_dims // group_size), dtype=mx.float32)
        self.biases = mx.zeros_like(self.scales)
        self.group_size = group_size
        self.bits = 2
        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        from .switch_layers import _quantized_gather_tail_policy

        # Same sorted-gather workaround as QuantizedSwitchLinear: a sorted
        # call whose width misses the sorted kernel's K tile runs unsorted.
        policy = _quantized_gather_tail_policy("affine", int(x.shape[-1]), sorted_indices)
        return mx.gather_qmm(
            x.astype(mx.float32), self["weight"], self["scales"], self["biases"],
            rhs_indices=indices, transpose=True, group_size=self.group_size, bits=2,
            mode="affine", sorted_indices=sorted_indices and policy == "native").astype(x.dtype)


def q2_0_planes(weights: dict, stem: str) -> None:
    """In place: a ``q2_0`` module's U32 codes + F16 scales -> fp32 affine planes."""
    if stem + ".biases" in weights:
        raise ValueError(f"{stem}: q2_0 stores no biases (they are -scales)")
    w, scales = weights.get(stem + ".weight"), weights.get(stem + ".scales")
    if w is None or scales is None or w.dtype != mx.uint32 or w.shape[:-1] != scales.shape[:-1]:
        raise ValueError(f"{stem}: q2_0 needs U32 codes and per-group scales")
    scales = scales.astype(mx.float32)
    weights[stem + ".scales"] = scales
    weights[stem + ".biases"] = -scales


class MoE(nn.Module):
    """Latent MoE: router and shared expert see the full width, experts a latent."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        latent = args.routed_expert_hidden_size or args.hidden_size
        self.gate = Router(args)
        self.experts = SwitchGLU(latent, args.moe_intermediate_size, args.n_routed_experts,
                                 activation=ClampedSwiGLU(args.swiglu_limit))
        self.shared_experts = MLP(args.hidden_size,
                                  args.moe_intermediate_size * args.n_shared_experts,
                                  post_norm=False, eps=args.rms_norm_eps)
        if args.routed_expert_hidden_size is not None:
            self.routed_expert_down_proj = nn.Linear(args.hidden_size, latent, bias=False)
            self.routed_expert_up_proj = nn.Linear(latent, args.hidden_size, bias=False)
        self.combine_output_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Opt-in read-once small-row expert kernels (``moe_expert_read_once``,
        # runtime/models/switch_read_once).  Observed use and declines are
        # counted per block; a declined call runs ``self.experts``.
        self.expert_read_once = False
        self.read_once_calls = 0
        self.read_once_declines = {}
        # Sorted-order weighted sum (``moe_sorted_weighted_sum``, default on,
        # Pierre 2026-11-21): on a sorted Metal gather the routed rows are
        # combined in sorted order by ``qwen4_moe_weighted_sum`` (bit-identical
        # to the eager unsort + product + sum; tests/
        # test_hy_v4_flash_prefill_exact_metal.py), so the unsorted [T, K, D]
        # copy and the fp32 product are never written.  Every other call (CPU,
        # unsorted decode, read-once, a wrapped bank, an unverified top-k)
        # runs the eager tail; counted per reason.
        self.sorted_weighted_sum = True
        self.sorted_sum_calls = 0
        self.sorted_sum_declines = {}

    def _decline_sorted_sum(self, reason):
        self.sorted_sum_declines[reason] = self.sorted_sum_declines.get(reason, 0) + 1

    def _sorted_combine(self, routed_in, indices, weights):
        """The combined routed latent ``[..., latent]`` (fp32), or None (eager tail)."""
        from . import qwen4_moe_weighted_sum as _ws
        from .switch_layers import GateUpFusedSwitchGLU, _scatter_unsort

        if int(indices.size) < _ws.MIN_ROUTED_ROWS:
            # Decode and short verify forwards: the kernel never admits them;
            # no host probe and no counter on the hot path.
            return None
        if type(self.experts) not in (SwitchGLU, GateUpFusedSwitchGLU):
            self._decline_sorted_sum("experts_type")
            return None
        if not _ws.runtime_supported():
            self._decline_sorted_sum("not_metal")
            return None
        if int(indices.shape[-1]) not in SORTED_SUM_TOP_K:
            self._decline_sorted_sum("top_k_unsupported")
            return None
        out = self.experts(routed_in, indices, keep_sorted=True)
        if not isinstance(out, tuple):
            self._decline_sorted_sum("unsorted_gather")
            return out, False
        rows, inv_order = out
        admission = _ws.admit_moe_weighted_sum(
            x_sorted=rows, inv_order=inv_order, scores=weights, indices=indices, do_sort=True,
            training=self.training, supported_top_k=SORTED_SUM_TOP_K)
        if not admission.accepted:
            # Stable key: the reason's leading words, without shapes or counts.
            self._decline_sorted_sum("admission_" + "_".join(admission.reason.split()[:2]))
            # Exactly the unsort SwitchGLU would have run.
            return _scatter_unsort(rows, inv_order, indices.shape).squeeze(-2), False
        self.sorted_sum_calls += 1
        return _ws.moe_weighted_sum(rows, inv_order, weights), True

    def __call__(self, x):
        indices, weights = self.gate(x)
        down = getattr(self, "routed_expert_down_proj", None)
        routed_in = down(x) if down is not None else x
        y = None
        if self.expert_read_once:
            reason = _read_once.refusal(self.experts, routed_in, indices)
            if reason is None:
                y = _read_once.switch_glu(self.experts, routed_in, indices)
                self.read_once_calls += 1
            else:
                _read_once._refuse(reason)
                self.read_once_declines[reason] = self.read_once_declines.get(reason, 0) + 1
        combined = None
        if y is None and self.sorted_weighted_sum:
            got = self._sorted_combine(routed_in, indices, weights)
            if got is not None:
                y, summed = got
                if summed:
                    combined = y.astype(x.dtype)
        if combined is None:
            if y is None:
                y = self.experts(routed_in, indices)
            combined = (y * weights[..., None]).sum(axis=-2).astype(x.dtype)
        y = combined
        up = getattr(self, "routed_expert_up_proj", None)
        if up is not None:
            y = up(y)
        return self.combine_output_norm(y + self.shared_experts(x))


# -------------------------------------------------------- MLA + indexer --

class MultiLinear(nn.Module):
    """Per-head linear ``[H, out, in]`` (mined from xing4_0.MultiLinear)."""

    def __init__(self, input_dims: int, output_dims: int, num_heads: int):
        super().__init__()
        self.weight = mx.zeros((num_heads, output_dims, input_dims))

    def __call__(self, x, transpose=True):
        return x @ (self.weight.swapaxes(-1, -2) if transpose else self.weight)


def mla_attend(q, kv, mask, sinks, scale, rank):
    """Fused absorbed MLA with learned sinks over one latent KV head.

    ``q [N, H, L, r+p]`` (latent query ‖ roped pe), ``kv [N, 1, T, r+p]``
    (normed latent ‖ roped key pe, the cache buffer itself): the same buffer
    is key and value, so no K/V copy is made; the value's trailing ``p``
    output lanes are dropped.  ``mask`` is None, ``"causal"`` or bool
    ``[N|1, 1, L, T]``.  ``mx.fast.scaled_dot_product_attention`` applies the
    sinks as the reference's extra softmax column (its softmax is fp32).
    Queries are chunked so a score block stays under ``SCORE_BUDGET``.
    Returns ``[N, H, L, r]``.
    """
    N, H, L, _ = q.shape
    T = kv.shape[2]
    sinks = sinks.astype(q.dtype)
    rows = L if isinstance(mask, str) else max(1, SCORE_BUDGET // max(1, N * H * T))
    if rows >= L:
        return mx.fast.scaled_dot_product_attention(q, kv, kv, scale=scale, mask=mask, sinks=sinks)[..., :rank]
    outs = []
    for a in range(0, L, rows):
        b = min(L, a + rows)
        m = mask if mask is None else mask[..., a:b, :]
        outs.append(mx.fast.scaled_dot_product_attention(
            q[:, :, a:b], kv, kv, scale=scale, mask=m, sinks=sinks)[..., :rank])
        if BOUNDED_SCORE_BLOCKS:
            # A lazy graph runs every chunk's score, mask and softmax buffers
            # side by side (measured on CPU: 12.3 GiB for a 2048-row chunk at
            # 16K, against 1.66 GiB evaluated per block), so the chunking
            # bounded nothing.  Same kernels and shapes: bit-identical.
            mx.eval(outs[-1])
    return mx.concatenate(outs, axis=2)


def sliding_band_attend(q, kv, first, start, window, sinks, scale, rank, block):
    """Sliding-window prefill over each query block's band of keys only.

    ``q [N, H, L, r+p]`` for positions ``start..start+L-1``; ``kv [N, 1, T,
    r+p]`` the rows of positions ``first..start+L-1`` (``first`` at or
    before ``start - window + 1`` when that history exists).  Query rows
    ``[a, b)`` can see only positions ``[start + a - window + 1, start + b)``
    (the previous ``window - 1`` keys, cached ring rows included) plus the
    sink column, so each block of ``block`` rows runs ``mla_attend`` over
    ``block + window - 1`` keys instead of all ``T``: the full masked
    ``L x T`` square becomes a band (the Ripple ``window_blocked_attention``
    mechanism, ``ricursive.py``, with sinks and latent rows).  Every row sees
    the same keys, mask and sink as ``mla_attend`` over the whole slice; the
    masked keys it drops contributed exactly zero, but the softmax and value
    reductions run over a shorter length, so the result is numerically
    equivalent, not bit-identical (``tests/test_hy_v4_flash_prefill_scaling_cpu.py``).
    A chunk of at most ``block`` rows is the stock call unchanged.

    With ``BOUNDED_SCORE_BLOCKS`` the pending blocks are evaluated once their
    score elements would exceed ``SCORE_BUDGET`` (the bound ``mla_attend``
    keeps), so a band forward syncs about ``L * (block + window) / (L * T)``
    as often as the square one, and never when it fits one budget.
    Returns ``[N, H, L, r]``.
    """
    N, H, L, _ = q.shape
    if L <= block:
        lo = max(first, start - window + 1)
        kpos = mx.arange(lo, start + L)
        qpos = mx.arange(start, start + L)
        allowed = (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] > qpos[:, None] - window)
        return mla_attend(q, kv[:, :, lo - first:], allowed[None, None], sinks, scale, rank)
    outs, pending, live = [], [], 0
    for a in range(0, L, block):
        b = min(L, a + block)
        k0 = max(first, start + a - window + 1)
        k1 = start + b
        elements = N * H * (b - a) * (k1 - k0)
        if BOUNDED_SCORE_BLOCKS and pending and live + elements > SCORE_BUDGET:
            mx.eval(pending)
            pending, live = [], 0
        qpos = mx.arange(start + a, start + b)
        kpos = mx.arange(k0, k1)
        allowed = (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] > qpos[:, None] - window)
        out = mla_attend(q[:, :, a:b], kv[:, :, k0 - first:k1 - first], allowed[None, None],
                         sinks, scale, rank)
        outs.append(out)
        pending.append(out)
        live += elements
    return mx.concatenate(outs, axis=2)


def gathered_indexed_attend(q, kv, recalled, sinks, scale, rank, rows=None):
    """Indexed attention for short chunks: one gathered key set per query row.

    ``q [B, H, L, r+p]``, ``kv [B, T, r+p]`` stored rows, ``recalled [B, L, K]``
    causal ids (-1 empty); ``rows`` turns the gathered storage rows into
    attention rows (fp8 caches dequantize only the recalled rows).
    """
    B, _, L, D = q.shape
    K = recalled.shape[-1]
    gathered = mx.take_along_axis(kv[:, None], mx.maximum(recalled, 0)[..., None], axis=2)
    if rows is not None:
        gathered = rows(gathered)
    out = mla_attend(q.transpose(0, 2, 1, 3).reshape(B * L, -1, 1, D),
                     gathered.reshape(B * L, 1, K, -1),
                     (recalled >= 0).reshape(B * L, 1, 1, K), sinks, scale, rank)
    return out.reshape(B, L, -1, rank).transpose(0, 2, 1, 3)


def dense_indexed_attend(q, kv, recalled, sinks, scale, rank, qpos, kpos):
    """Indexed prefill reference: every key with a ``[B, L, T]`` hit mask.

    ``kv [B, T, r+p]`` attention rows at positions ``kpos``; queries at
    ``qpos``.  ``O(L * T)`` per chunk (the default; the opt-in
    ``sparse_prefill_attention`` kernel reads only the recalled rows).
    """
    B, L, _ = recalled.shape
    T = kv.shape[1]
    hit = mx.zeros((B, L, T + 1), dtype=mx.bool_)
    slot = mx.where(recalled >= 0, recalled, T)
    hit = mx.put_along_axis(hit, slot, mx.array(True), axis=-1)[..., :T]
    allowed = hit & (kpos[None, None, :] <= qpos[None, :, None])
    return mla_attend(q, kv[:, None], allowed[:, None], sinks, scale, rank)


class CompressedIndexer(nn.Module):
    """Gated-pool compressed indexer (``HYV4FlashCompressedIndexer``).

    Token keys are cached; a group's pooled key is computed once it is
    complete and cached (fp32 copy of the reference's bf16 value) with the
    RoPE phase of its first token.  A query recalls ``group_topk - 1``
    complete earlier groups plus its own group.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        d = args.compressed_indexer_head_dim
        self.ratio = args.compressed_indexer_compress_ratio
        self.num_heads = args.compressed_indexer_num_attention_heads
        self.head_dim = d
        self.group_topk = args.compressed_indexer_group_topk
        self.rope_dim = args.qk_rope_head_dim
        self.base = float(args.rope_parameters[INDEXED]["rope_theta"])
        self.section = args.rope_parameters[INDEXED].get("mrope_section")
        self.eps = args.rms_norm_eps
        self.query_projection = nn.Linear(args.q_lora_rank, self.num_heads * d, bias=False)
        self.head_weight_projection = nn.Linear(args.hidden_size, self.num_heads, bias=False)
        self.token_key_projection = nn.Linear(args.hidden_size, d, bias=False)
        self.token_key_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.key_value_projection = nn.Linear(d, d, bias=False)
        self.key_gate_projection = nn.Linear(d, d, bias=False)
        self.key_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.intra_group_ape = mx.zeros((self.ratio, d), dtype=mx.float32)

    def _rope_tail(self, x, offset, scale=1.0):
        """Rotate the trailing ``rope_dim`` lanes of ``x [..., L, d]`` (positions on -2).

        ``offset`` is a scalar start or ``[3, L]`` vision positions.
        """
        head, tail = x[..., : -self.rope_dim], x[..., -self.rope_dim:]
        if isinstance(offset, np.ndarray):
            tail = mrope_rotate(tail, offset, self.base, self.section)
        else:
            tail = mx.fast.rope(tail, self.rope_dim, traditional=True, base=self.base,
                                scale=scale, offset=offset)
        return mx.concatenate([head, tail], axis=-1)

    def pool(self, token_keys, first_group: int, positions=None):
        """``[B, g*ratio, d]`` complete groups -> roped pooled keys ``[B, g, d]`` fp32."""
        B, n, d = token_keys.shape
        g = n // self.ratio
        grouped = token_keys.reshape(B, g, self.ratio, d).astype(mx.float32)
        values = grouped @ self.key_value_projection.weight.T
        gates = grouped @ self.key_gate_projection.weight.T + self.intra_group_ape
        pooled = (values * mx.softmax(gates, axis=2)).sum(axis=2)
        pooled = rms_norm(pooled, self.key_norm.weight.astype(mx.float32), self.eps).astype(token_keys.dtype)
        # Group j's phase is that of its first token, position j * ratio
        # (``positions`` [3, g] of those tokens when vision spans exist).
        # Opt-in ``resident_bf16`` indexer_keys (H5): bf16 activations give
        # bf16-exact keys, cached bf16 instead of as an fp32 copy.
        key_dtype = mx.bfloat16 if (getattr(self, "lowp_keys", False) and token_keys.dtype == mx.bfloat16) \
            else mx.float32
        if key_dtype != mx.float32:
            self.lowp_key_groups = getattr(self, "lowp_key_groups", 0) + g
        if positions is not None:
            return self._rope_tail(pooled, positions).astype(key_dtype)
        return self._rope_tail(pooled, first_group, scale=float(self.ratio)).astype(key_dtype)

    def _fused_logits(self, q, group_keys, w, own):
        """Negated masked logits of the opt-in ``indexer_fused_scores`` kernel (v1, or the H6 simd layout)."""
        if getattr(self, "fused_scores_variant", "v1") == "simd":
            reason = _fusions.indexer_simd_unsupported_reason(self.num_heads, self.head_dim)
            if reason is None:
                self.fused_score_simd_calls = getattr(self, "fused_score_simd_calls", 0) + 1
                return _fusions.indexer_logits_simd(q, group_keys, w, own, negate=True)
            declines = getattr(self, "fused_score_simd_declines", None)
            if declines is None:
                declines = self.fused_score_simd_declines = {}
            declines[reason] = declines.get(reason, 0) + 1
        return _fusions.indexer_logits_metal(q, group_keys, w, own, negate=True)

    def select(self, hidden, q_resid, start: int, group_keys, where=None):
        """Recalled token slots ``[B, L, group_topk*ratio]`` int32, -1 unused.

        Unused entries are the reference's ``-inf`` picks (groups at or after
        the query's own group, which the causal mask removes there).  The
        own group is always listed; callers still apply causality.
        """
        B, L, _ = hidden.shape
        G = group_keys.shape[1]
        q = self.query_projection(q_resid).reshape(B, L, self.num_heads, self.head_dim)
        # Query phases follow positions; groups (own/causal) follow slots.
        q = self._rope_tail(q.swapaxes(1, 2), start if where is None else where).astype(mx.float32)
        w = self.head_weight_projection(hidden).astype(mx.float32) * (
            self.num_heads ** -0.5 * self.head_dim ** -0.5)
        own_all = (mx.arange(start, start + L) // self.ratio).astype(mx.int32)  # [L]
        k = min(self.group_topk - 1, G)
        # Opt-in ``indexer_fused_scores``: the logits come straight from one
        # Metal kernel (no [B, heads, rows, G] block), so a chunk may hold
        # ``heads`` times more rows under the same budget.
        fused = (getattr(self, "fused_scores", False) and G > 0 and self.head_dim % 4 == 0
                 and _fusions.on_gpu())
        if fused:
            self.fused_score_calls = getattr(self, "fused_score_calls", 0) + 1
        # Score in query chunks: [B, heads, rows, G] fp32 stays under the budget.
        rows = max(1, SCORE_BUDGET // max(1, B * (1 if fused else self.num_heads) * max(G, 1)))
        picks = []
        for a in range(0, L, rows):
            b = min(L, a + rows)
            own = own_all[a:b]
            own_col = mx.broadcast_to(own[None, :, None], (B, b - a, 1))
            if fused:
                # The kernel writes the negated masked logits (negation is
                # exact, so argpartition sees the reference's operand order);
                # a pick is valid iff it lies before the own group.
                neg = self._fused_logits(q[:, :, a:b], group_keys, w[:, a:b], own)
                if k > 0:
                    picked = mx.argpartition(neg, kth=k - 1, axis=-1)[..., :k].astype(mx.int32)
                    picked = mx.where(picked < own[None, :, None], picked, -1)
                    picks.append(mx.concatenate([picked, own_col], axis=-1))
                else:
                    picks.append(own_col)
                if BOUNDED_SCORE_BLOCKS and rows < L:
                    mx.eval(picks[-1])
                continue
            scores = nn.relu(q[:, :, a:b] @ group_keys[:, None].swapaxes(-1, -2))
            logits = (w[:, a:b].swapaxes(1, 2)[..., None] * scores).sum(axis=1)  # [B, l, G]
            valid = mx.arange(G)[None, None, :] < own[None, :, None]
            logits = mx.where(valid, logits, -mx.inf)
            if k > 0:
                picked = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k].astype(mx.int32)
                picked = mx.where(mx.take_along_axis(valid, picked, axis=-1), picked, -1)
                picks.append(mx.concatenate([picked, own_col], axis=-1))
            else:
                picks.append(own_col)
            if BOUNDED_SCORE_BLOCKS and rows < L:
                mx.eval(picks[-1])  # one score block live at a time (see mla_attend)
        groups = picks[0] if len(picks) == 1 else mx.concatenate(picks, axis=1)
        offsets = mx.arange(self.ratio, dtype=mx.int32)
        tokens = groups[..., None] * self.ratio + offsets
        tokens = mx.where(groups[..., None] >= 0, tokens, -1)
        # Discrete selection: no gradient flows through the picked ids (value-neutral; MLX refuses a VJP
        # through gather indices, which the aMoEba carve's layer-wise backward would otherwise hit).
        return mx.stop_gradient(tokens.reshape(B, L, -1).astype(mx.int32))


class Attention(nn.Module):
    """Gated MLA with learned sinks, sandwich post-projection norm."""

    def __init__(self, args: ModelArgs, kind: str, *, indexer: bool):
        super().__init__()
        sliding = kind == SLIDING
        self.kind = kind
        self.num_heads = args.swa_num_attention_heads if sliding else args.num_attention_heads
        self.qk_nope = args.swa_qk_nope_head_dim if sliding else args.qk_nope_head_dim
        self.qk_rope = args.swa_qk_rope_head_dim if sliding else args.qk_rope_head_dim
        self.v_dim = args.swa_v_head_dim if sliding else args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.scale = (self.qk_nope + self.qk_rope) ** -0.5
        self.window = args.sliding_window
        self.candidate_topk = args.candidate_topk
        self.base = float(args.rope_parameters[kind]["rope_theta"])
        self.section = args.rope_parameters[kind].get("mrope_section")
        H = self.num_heads
        self.q_a_proj = nn.Linear(args.hidden_size, args.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(args.q_lora_rank, H * (self.qk_nope + self.qk_rope), bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(args.hidden_size, self.kv_lora_rank + self.qk_rope, bias=False)
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
        # kv_b_proj is folded into these at sanitize time.
        self.embed_q = MultiLinear(self.qk_nope, self.kv_lora_rank, H)
        self.unembed_out = MultiLinear(self.kv_lora_rank, self.v_dim, H)
        self.linear_gate = nn.Linear(args.hidden_size, H * self.v_dim, bias=False)
        self.o_proj = nn.Linear(H * self.v_dim, args.hidden_size, bias=False)
        self.post_proj_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.learnable_sink_param = mx.zeros((H,), dtype=mx.float32)
        # Opt-in band attention for sliding prefill (``sliding_band_attend``):
        # query rows per block, 0 = off (the whole masked slice, the default).
        # A plain attribute, not a parameter.
        self.sliding_attention_blocks = 0
        self.band_forwards = 0  # sliding forwards that took the band (receipts)
        # Opt-in sparse prefill kernel for indexed chunks of more than
        # GATHER_MAX_QUERY rows (``hy_v4_flash_sparse_prefill``); off = the
        # dense masked reference.  Plain attributes, not parameters.
        self.sparse_prefill_attention = False
        self.sparse_prefill_forwards = 0  # indexed forwards that took the kernel (receipts)
        self.sparse_prefill_declines = {}  # reason -> forwards that fell back to dense
        # fp8 latent rows (H3): forwards whose packed rows the kernel dequantized
        # itself, and why others unpacked first (the native kernel still ran).
        self.sparse_prefill_fp8_forwards = 0
        self.sparse_prefill_fp8_declines = {}
        # Opt-in H4 union prefill (``sparse_prefill_union_rows``): query rows
        # per union sub-chunk, 0 = off.  Plain attributes, not parameters.
        self.sparse_prefill_union_rows = 0
        self.sparse_prefill_union_forwards = 0
        self.sparse_prefill_union_declines = {}
        if indexer:
            self.compressed_indexer = CompressedIndexer(args)

    def rope(self, x, offset):
        if isinstance(offset, np.ndarray):  # [3, L] vision-span positions
            return mrope_rotate(x, offset, self.base, self.section)
        return mx.fast.rope(x, self.qk_rope, traditional=True, base=self.base, scale=1.0, offset=offset)

    def project(self, x, offset):
        """``q [B, H, L, r+p]``, ``q_resid``, and this step's key rows ``[B, L, r+p]``."""
        q, q_resid = self.project_query(x, offset)
        return q, q_resid, self.project_keys(x, offset)

    def project_query(self, x, offset):
        """``q [B, H, L, r+p]`` and ``q_resid`` (``q_a_proj`` leads: the fused input GEMM's order)."""
        B, L, _ = x.shape
        H = self.num_heads
        q_resid = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_resid).reshape(B, L, H, self.qk_nope + self.qk_rope).transpose(0, 2, 1, 3)
        q_lat = self.embed_q(q[..., : self.qk_nope])  # [B, H, L, r]
        q_pe = self.rope(q[..., self.qk_nope:], offset)
        return mx.concatenate([q_lat, q_pe.astype(q_lat.dtype)], axis=-1), q_resid

    def project_keys(self, x, offset):
        """This step's latent ‖ RoPE key rows ``[B, L, r+p]`` (the cached state)."""
        kv = self.kv_a_proj_with_mqa(x)
        latent = self.kv_a_layernorm(kv[..., : self.kv_lora_rank])
        k_pe = self.rope(kv[:, None, :, self.kv_lora_rank:], offset)[:, 0]
        return mx.concatenate([latent, k_pe.astype(latent.dtype)], axis=-1)

    # -- prompt-chunk tail (prefill_tail_skip tier 2) ----------------------
    def tail_row(self, x, cache):
        """The last row's block output ``[B, 1, D]``; every state write over all rows.

        Indexed layers only, and only for a chunk whose every lane is full
        (the checkpoint-row claim guarantees it): key rows, indexer token
        keys and pooled group keys are computed and appended over every row
        by the same ops as ``_attend`` (``project_keys``, the token-key
        projection and ``append_index_keys``), so the cache is bitwise the
        full forward's.  The query, the indexer selection, attention and the
        output projection run on the last row, whose output is the only one
        anything reads (``Model._tail_refusal`` keeps the full chunk for
        every other reader).
        """
        last = x[:, -1:]
        rows = getattr(cache, "rows", None)
        if rows is None:
            return self.finish(last, self._attend_tail(x, last, cache))
        L = x.shape[1]
        outs = []
        fused = self.get("fused_input_proj") if "fused_input_proj" in self else None
        if fused is not None:
            fused.bypass = True
        try:
            for i, (row, (start, count)) in enumerate(zip(rows, cache.layout(L))):
                if start != 0 or count != L:
                    raise RuntimeError("Hy4-flash tail row needs every lane full (claim invariant)")
                outs.append(self._attend_tail(x[i : i + 1], last[i : i + 1], row))
        finally:
            if fused is not None:
                fused.bypass = False
                fused.release()
        return self.finish(last, mx.concatenate(outs, axis=0))

    def _attend_tail(self, x, last, cache):
        if self.kind != INDEXED or cache.mrope_spans or cache.record:
            raise RuntimeError("Hy4-flash tail row: indexed, text-only, unrecorded caches only")
        L = x.shape[1]
        start = cache.offset
        qpos = start + L - 1
        q, q_resid = self.project_query(last, qpos)
        keys = self.project_keys(x, start)
        kv, first = cache.append(keys)
        sinks = self.learnable_sink_param
        r = self.kv_lora_rank
        indexer = self.compressed_indexer
        token_keys = indexer.token_key_norm(indexer.token_key_projection(x))
        group_keys = cache.append_index_keys(token_keys, indexer)
        if start + L <= self.candidate_topk:
            # The last row sees every key (indexed caches keep all from 0).
            return mla_attend(q, cache.latent_rows(kv, keys.dtype)[:, None], None, sinks, self.scale, r)
        recalled = indexer.select(last, q_resid, qpos, group_keys)  # [B, 1, K]
        recalled = mx.where(recalled <= qpos, recalled, -1)
        return gathered_indexed_attend(q, kv, recalled, sinks, self.scale, r,
                                       rows=lambda g: cache.latent_rows(g, keys.dtype))

    def finish(self, x, latent_out):
        """``latent_out [B, H, L, r]`` -> block output."""
        B, L = x.shape[:2]
        out = self.unembed_out(latent_out).transpose(0, 2, 1, 3)  # [B, L, H, v]
        gate = self.linear_gate(x).reshape(B, L, self.num_heads, self.v_dim)
        if "fused_input_proj" in self:
            # Opt-in ``mla_fused_input_proj``: the sigmoid-gate multiply as one compiled kernel.
            gated = _fusions.gated_output(out, gate)
        else:
            gated = out * mx.sigmoid(gate)
        return self.post_proj_norm(self.o_proj(gated.reshape(B, L, -1)))

    def __call__(self, x, cache):
        rows = getattr(cache, "rows", None)
        if rows is None:
            return self.finish(x, self._attend(x, cache))
        # A batch of lanes: projections, MoE and the head run batched; each
        # lane's attention reads only its own latent history and its own
        # (start, count) span of a ragged chunk.
        L = x.shape[1]
        outs = []
        fused = self.get("fused_input_proj") if "fused_input_proj" in self else None
        if fused is not None:
            # Lane slices are separate inputs: each projection runs its own
            # segment (exactly the split route), so no lane computes the
            # gate rows that ``finish`` recomputes on the whole batch.
            fused.bypass = True
        try:
            for i, (row, (start, count)) in enumerate(zip(rows, cache.layout(L))):
                if count == 0:
                    outs.append(mx.zeros((1, self.num_heads, L, self.kv_lora_rank), dtype=x.dtype))
                    continue
                out = self._attend(x[i : i + 1, start : start + count], row)
                if count < L:
                    out = mx.pad(out, [(0, 0), (0, 0), (start, L - start - count), (0, 0)])
                outs.append(out)
        finally:
            if fused is not None:
                fused.bypass = False
                fused.release()
        return self.finish(x, mx.concatenate(outs, axis=0))

    def _attend(self, x, cache):
        L = x.shape[1]
        start = cache.offset
        spans = cache.mrope_spans
        # RoPE positions; masks, windows and indexer groups stay on slots.
        where = chunk_positions(spans, start, L) if spans else start
        q, q_resid, keys = self.project(x, where)
        # [B, T, r+p] (native) or packed fp8 rows; absolute position of kv[:, 0].
        # Queries stay in the model dtype; only stored rows are approximate.
        kv, first = cache.append(keys)
        T = kv.shape[1]
        sinks = self.learnable_sink_param
        r = self.kv_lora_rank
        qpos = mx.arange(start, start + L)
        kpos = mx.arange(first, first + T)
        if self.kind == SLIDING:
            # Attend only the keys this chunk can see.  The ring also keeps
            # ROLLBACK older rows whose count depends on how the history was
            # chunked or trimmed; masking them out is exact, but it changes
            # the SDPA reduction length, so an APCv2 landing would not repeat
            # the cold run bit for bit.
            lo = max(first, start - self.window + 1)
            block = self.sliding_attention_blocks
            if block and L > block:
                # Opt-in band: each query block reads only its own window.
                self.band_forwards += 1
                kv = cache.latent_rows(kv[:, lo - first :], keys.dtype)
                return sliding_band_attend(q, kv[:, None], lo, start, self.window, sinks,
                                           self.scale, r, block)
            kv, kpos = kv[:, lo - first :], mx.arange(lo, start + L)
            kv = cache.latent_rows(kv, keys.dtype)
            if L == 1:
                # One query: the slice above is exactly its visible window,
                # so the mask would be all true; skip building it (decode).
                return mla_attend(q, kv[:, None], None, sinks, self.scale, r)
            allowed = (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] > qpos[:, None] - self.window)
            return mla_attend(q, kv[:, None], allowed[None, None], sinks, self.scale, r)
        indexer = self.compressed_indexer
        token_keys = indexer.token_key_norm(indexer.token_key_projection(x))
        group_keys = cache.append_index_keys(token_keys, indexer)
        if start + L <= self.candidate_topk:
            cache.record_selection(None, start, L)
            mask = None if L == 1 else "causal"
            kv = cache.latent_rows(kv, keys.dtype)
            return mla_attend(q, kv[:, None], mask, sinks, self.scale, r)
        recalled = indexer.select(x, q_resid, start, group_keys,
                                  where if spans else None)  # [B, L, K]
        recalled = mx.where(recalled <= qpos[None, :, None], recalled, -1)
        cache.record_selection(recalled, start, L)
        if L <= GATHER_MAX_QUERY:
            return gathered_indexed_attend(q, kv, recalled, sinks, self.scale, r,
                                           rows=lambda g: cache.latent_rows(g, keys.dtype))
        union = self.sparse_prefill_union_rows
        if union:
            # Opt-in H4 union route (``hy_v4_flash_union_prefill``): sub-chunks of
            # ``union`` queries attend the union of their recalled rows, read
            # once.  Numerically equivalent to the dense path; GPU A/B pending.
            reason = ("short_context" if int(kv.shape[1]) < SPARSE_PREFILL_MIN_KEYS
                      else _union_prefill.unsupported_reason(lanes=q.shape[0], first=first))
            if reason is None:
                self.sparse_prefill_union_forwards += 1
                return _union_prefill.union_indexed_attend(
                    q, kv, recalled, sinks, self.scale, r, qpos, union, attend=mla_attend,
                    rows=lambda g: cache.latent_rows(g, keys.dtype),
                    bounded=BOUNDED_SCORE_BLOCKS, budget=SCORE_BUDGET)
            self.sparse_prefill_union_declines[reason] = self.sparse_prefill_union_declines.get(reason, 0) + 1
        packed = kv
        kv = None if getattr(cache, "approximate", False) else cache.latent_rows(packed, keys.dtype)
        if self.sparse_prefill_attention and int(packed.shape[1]) < SPARSE_PREFILL_MIN_KEYS:
            # Short history: the 2048 recalled keys are most of the keys, so the
            # fused dense SDPA does the same work faster (2026-11-22 bisect:
            # ~1K-token GSM8K prompts were slower with the kernel).
            self.sparse_prefill_declines["short_context"] = self.sparse_prefill_declines.get("short_context", 0) + 1
        elif self.sparse_prefill_attention:
            # The kernel reads id i at kv[i - first]; the dense and gather
            # references index kv with the id itself, so they agree only at
            # first == 0 (every indexed cache: only sliding caches advance
            # start); any other forward is a counted fallback.  It also needs
            # distinct ids per row, which select() guarantees (distinct
            # groups, each expanded to distinct tokens); the dense hit mask
            # would dedup a repeat, the kernel would count it twice.
            reason = sparse_unsupported_reason(q.shape[1], q.shape[-1], r, q.dtype,
                                               rows=L, lanes=q.shape[0], first=first)
            if reason is None and kv is None:
                # H3 (fp8 rows): stage the recalled packed rows and dequantize
                # them in the kernel instead of unpacking all T rows first.
                # Bitwise the unpack-then-kernel route (pow2 scales; the
                # variant's dispatch probe checks it), so it replaces that
                # route inside the opt-in fp8 format.  GPU A/B pending.
                fp8_reason = _sparse_fp8_reason(q.shape[1], q.shape[-1], r, q.dtype, row_bytes=int(packed.shape[-1]),
                                                rows=L, lanes=q.shape[0], first=first)
                if fp8_reason is None:
                    self.sparse_prefill_forwards += 1
                    self.sparse_prefill_fp8_forwards += 1
                    return _sparse_prefill_attend_fp8(q, packed, recalled, sinks, self.scale, r,
                                                      start=start, first=first)
                # Counted apart: the native kernel below still serves this forward.
                self.sparse_prefill_fp8_declines[fp8_reason] = self.sparse_prefill_fp8_declines.get(fp8_reason, 0) + 1
            if kv is None:
                kv = cache.latent_rows(packed, keys.dtype)
            if reason is None:
                # Opt-in: each query reads only its recalled rows (one Metal kernel).
                self.sparse_prefill_forwards += 1
                return sparse_prefill_attend(q, kv, recalled, sinks, self.scale, r,
                                             start=start, first=first)
            self.sparse_prefill_declines[reason] = self.sparse_prefill_declines.get(reason, 0) + 1
        if kv is None:
            kv = cache.latent_rows(packed, keys.dtype)
        return dense_indexed_attend(q, kv, recalled, sinks, self.scale, r, qpos, kpos)


# ------------------------------------------------- hyper-connections --

@lru_cache(maxsize=None)
def _hc_pre_kernel(n: int, eps: float, hc_eps: float, magnitude: float):
    """Compiled iHC boundary: fp32 mixes, sigmoid pre/post, stream collapse."""

    def fn(streams, hc_fn, base, scale):
        flat = streams.reshape(*streams.shape[:-2], -1).astype(mx.float32)
        mixes = (flat @ hc_fn.T) * mx.rsqrt(mx.mean(mx.square(flat), axis=-1, keepdims=True) + eps)
        pre = mx.sigmoid(mixes[..., :n] * scale[0] + base[:n]) + hc_eps
        post = magnitude * mx.sigmoid(mixes[..., n:] * scale[1] + base[n:]) + hc_eps
        collapsed = (pre[..., None] * streams.astype(mx.float32)).sum(axis=-2)
        return post, collapsed.astype(streams.dtype)

    return mx.compile(fn)


@lru_cache(maxsize=None)
def _hc_head_kernel(eps: float, hc_eps: float):
    def fn(streams, hc_fn, base, scale):
        flat = streams.reshape(*streams.shape[:-2], -1).astype(mx.float32)
        mixes = (flat @ hc_fn.T) * mx.rsqrt(mx.mean(mx.square(flat), axis=-1, keepdims=True) + eps)
        pre = mx.sigmoid(mixes * scale + base) + hc_eps
        return (pre[..., None] * streams.astype(mx.float32)).sum(axis=-2).astype(streams.dtype)

    return mx.compile(fn)


@contextmanager
def refed_rows(model, prefix):
    """Mark each batch row's first ``prefix[b]`` positions as re-fed for one forward.

    ``verify_phase_align`` re-feeds a child chain's root path, positions the
    root chain's row of the same forward also holds.  The layer-output
    readers (calculator tiles, ``_reader_inputs``) see re-fed positions as
    padding (id -1), so a position never matches twice in one forward and
    the root chain's row stays the one that decides.  ``None`` / all zero:
    nothing is marked.
    """
    if not prefix or not any(prefix) or not hasattr(model, "layer_output_readers"):
        yield
        return
    saved = getattr(model, "_reader_refed_prefix", None)
    object.__setattr__(model, "_reader_refed_prefix", [int(n) for n in prefix])
    try:
        yield
    finally:
        object.__setattr__(model, "_reader_refed_prefix", saved)


@mx.compile
def _hc_merge(post, branch, residual):
    out = post[..., None] * branch.astype(mx.float32)[..., None, :] + residual.astype(mx.float32)
    return out.astype(residual.dtype)


def _unweighted_inverse_rms(flat, eps):
    return mx.rsqrt(mx.mean(mx.square(flat), axis=-1, keepdims=True) + eps)


class HCPre(nn.Module):
    def __init__(self, args: ModelArgs, outputs: int):
        super().__init__()
        n = args.hc_mult
        self.hc_fn = mx.zeros((outputs, n * args.hidden_size), dtype=mx.float32)
        self.hc_base = mx.zeros((outputs,), dtype=mx.float32)
        self.hc_scale = mx.zeros((2 if outputs == 2 * n else 1,), dtype=mx.float32)


class HyperConnection(nn.Module):
    """iHC boundary: ``post, collapse = hc(streams)`` (HYV4FlashHyperConnection)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n = args.hc_mult
        self.eps = args.rms_norm_eps
        self.hc_eps = args.hc_eps
        self.magnitude = args.hc_magnitude
        self.hc_pre = HCPre(args, 2 * args.hc_mult)

    def __call__(self, streams):
        p = self.hc_pre
        kernel = _hc_pre_kernel(self.n, float(self.eps), float(self.hc_eps), float(self.magnitude))
        return kernel(streams, p.hc_fn.astype(mx.float32), p.hc_base.astype(mx.float32),
                      p.hc_scale.astype(mx.float32))


class HyperHead(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        n = args.hc_mult
        self.eps = args.rms_norm_eps
        self.hc_eps = args.hc_eps
        self.hc_head_fn = mx.zeros((n, n * args.hidden_size), dtype=mx.float32)
        self.hc_head_base = mx.zeros((n,), dtype=mx.float32)
        self.hc_head_scale = mx.zeros((1,), dtype=mx.float32)

    def __call__(self, streams):
        kernel = _hc_head_kernel(float(self.eps), float(self.hc_eps))
        return kernel(streams, self.hc_head_fn.astype(mx.float32), self.hc_head_base.astype(mx.float32),
                      self.hc_head_scale.astype(mx.float32))


def _dispatch(x):
    """Hand the graph built so far to the device (``async_eval``: a scheduling hint, same numerics). Inside a
    graph transformation (``vjp``/``grad``, e.g. the aMoEba carve's whole-graph gradient) MLX refuses
    ``async_eval``; the hint is then skipped."""
    try:
        mx.async_eval(x)
    except ValueError:
        pass


def hc_merge(post, branch, residual):
    return _hc_merge(post, branch, residual)


# ------------------------------------------------------------------ Engram --

class EngramHasher:
    """Host-side n-gram hashing of one Engram layer (``hash_ngrams``), exact int64."""

    def __init__(self, lookup_table: np.ndarray, head_vocab_sizes: Sequence[Sequence[int]],
                 *, layer_id: int, seed: int, max_ngram: int, heads_per_ngram: int, pad_id: int):
        table = np.asarray(lookup_table).astype(np.int64).reshape(-1)
        if table.size == 0 or not 0 <= pad_id < table.size:
            raise ValueError("Engram lookup table must cover the pad id")
        self.table = table
        self.pad_id = int(pad_id)
        self.max_ngram = int(max_ngram)
        self.heads = int(heads_per_ngram)
        sizes = [int(s) for order in head_vocab_sizes for s in order]
        if len(sizes) != (self.max_ngram - 1) * self.heads:
            raise ValueError("Engram head vocabulary layout mismatch")
        self.moduli = np.asarray(sizes, dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.moduli)[:-1]]).astype(np.int64)
        self.rows = int(self.moduli.sum())
        compressed = int(table.max()) + 1
        half_bound = max(1, np.iinfo(np.int64).max // compressed // 2)
        rng = np.random.default_rng(int(seed) + 10007 * int(layer_id))
        drawn = rng.integers(0, half_bound, size=self.max_ngram, dtype=np.int64)
        self.multipliers = drawn * 2 + 1

    def hash_ids(self, ngram_tokens: np.ndarray) -> np.ndarray:
        """``[B, L, max_ngram]`` token ids -> per-head ids ``[B, L, heads*orders]``."""
        tokens = np.asarray(ngram_tokens, dtype=np.int64)
        size = self.table.size
        compressed = np.where(tokens < size, self.table[np.minimum(tokens, size - 1)],
                              self.table[self.pad_id])
        mixed = compressed * self.multipliers
        out = []
        for n in range(2, self.max_ngram + 1):
            mix = mixed[..., 0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, mixed[..., k])
            start = (n - 2) * self.heads
            out.append(mix[..., None] % self.moduli[start:start + self.heads])
        return np.concatenate(out, axis=-1)

    def table_rows(self, ngram_tokens: np.ndarray) -> np.ndarray:
        """Flat table row of each head's id (head offset added)."""
        return self.hash_ids(ngram_tokens) + self.offsets


class ResidentEngramRows:
    """Engram table held in memory (tests, small checkpoints)."""

    def __init__(self, weight: mx.array):
        self.weight = weight
        self.rows = int(weight.shape[0])

    def read(self, rows: np.ndarray) -> mx.array:
        return self.weight[mx.array(rows.astype(np.int64))]

    def prefetch(self, rows, *, required: bool = False) -> int:
        return 0


ENGRAM_PRECISIONS = ("transformers", "vllm")


def engram_gate(key, query, hidden: int, *, transformers: bool, torch_round: bool = False):
    """``sigmoid(signed_sqrt((key * query).sum(-1) * hidden**-0.5))`` over the last axis.

    Default (``engram_gate_rounding: fp32``): fp32 products and sum, scaled,
    rounded once to the activation dtype in the ``transformers`` profile
    (``vllm`` keeps fp32).  ``torch_round`` (opt-in, ``transformers`` only):
    torch's rounding points in the reference's
    ``(key * query).sum(dim=-1) * self.hidden_size**-0.5`` on bf16 tensors:
    the product is a bf16 tensor (rounded per element), the sum accumulates
    in fp32 and rounds, the scalar multiply runs in fp32 and rounds again;
    and torch's bf16 ``sigmoid`` computes in fp32 and rounds once, where
    MLX's bf16 sigmoid rounds its inner steps (it differed on 246 of 2048
    rows given identical inputs), so here it runs in fp32.  Measured against
    torch 2.14 on CPU (tests/test_hy_v4_flash_fusions_cpu.py): before the
    sigmoid the default differs from torch on about half of random bf16
    rows, rounding after the sum and the scale only (omlx #5469's fix) on
    about 40%, this on about 1 in 4000 (fp32 summation order).
    """
    dtype = key.dtype
    if transformers and torch_round:
        g = (key * query).astype(mx.float32).sum(axis=-1).astype(dtype)
        g = (g.astype(mx.float32) * hidden ** -0.5).astype(dtype)
        s = mx.sqrt(mx.maximum(mx.abs(g), 1e-6)) * mx.sign(g)
        return mx.sigmoid(s.astype(mx.float32)).astype(dtype)
    else:
        g = (key.astype(mx.float32) * query.astype(mx.float32)).sum(axis=-1) * hidden ** -0.5
        if transformers:
            g = g.astype(dtype)
    return mx.sigmoid(mx.sqrt(mx.maximum(mx.abs(g), 1e-6)) * mx.sign(g))


class EngramLayer(nn.Module):
    """Engram memory residual on the HC streams (HYV4FlashEngramLayer).

    Two precision profiles; the references disagree on two steps:

    * ``transformers`` (default, Pierre 2026-11-18): the transformers port.
      Rows take the per-vector E4M3 round trip when ``engram_fp8_embedding``
      is set ("as Engram embeddings were trained to be read"), and the value
      branch is combined in the activation dtype after the unified value norm
      rounds.
    * ``vllm``: vLLM ``FlashEngram``.  A bf16 table is read without the
      round trip (an FP8 table already carries it), and the value norm and
      branch combine stay in fp32.

    The per-head lookup norms, ``norm_key`` and ``norm_query`` are
    ``torch.nn.RMSNorm(eps=None)`` in both references: fp32 epsilon, weight
    applied in fp32, one rounding (``torch_rms_norm``).  The unified value
    norm is that op in ``transformers``; ``vllm`` keeps it unrounded in fp32
    with the same epsilon.  Until 2026-11-20 the ``transformers`` profile, and
    ``norm_key``/``norm_query`` in both, used the bf16 epsilon 2**-7 with
    round-then-scale, a misreading of torch's documentation.

    Table formats (``bind_engram`` sets ``table_round_trip_dtype``): an
    ``fp8_e4m3_row`` table (converter ``--engram-formats``) already carries the
    per-vector round trip, computed in its declared dtype.  When that dtype
    is the activation dtype the ``transformers`` profile uses the decoded rows
    as they are, which equals the round trip of the bf16 table bit for bit;
    in any other dtype it applies the round trip again (a second, approximate
    quantization).  ``vllm`` reads an FP8 table's rows as decoded, as vLLM
    does.  ``bf16`` and ``affine4_g32`` tables carry no round trip.
    """

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        if args.engram_precision not in ENGRAM_PRECISIONS:
            raise ValueError(f"Engram precision must be one of {ENGRAM_PRECISIONS}")
        self.layer_id = layer_id
        self.precision = args.engram_precision
        self.hidden = args.hidden_size
        self.n = args.hc_mult
        self.fp8 = bool(args.engram_fp8_embedding)
        self.head_dim = args.engram_n_embed_per_ngram // args.engram_n_head_per_ngram
        self.num_heads = (args.engram_max_ngram_size - 1) * args.engram_n_head_per_ngram
        width = (args.engram_max_ngram_size - 1) * args.engram_n_embed_per_ngram
        self.lookup_per_head_norms = [nn.RMSNorm(self.head_dim) for _ in range(self.num_heads)]
        self.value_proj = nn.Linear(width, self.hidden, bias=True)
        self.key_projs = [nn.Linear(width, self.hidden, bias=True) for _ in range(self.n)]
        self.norm_key = [nn.RMSNorm(self.hidden) for _ in range(self.n)]
        self.norm_query = [nn.RMSNorm(self.hidden) for _ in range(self.n)]
        self.unified_pre_gamma = mx.ones((self.hidden,))
        self.unified_per_hc_scales = [mx.zeros((self.hidden,)) for _ in range(self.n)]
        self.lookup_table = mx.arange(args.engram_lookup_table_size, dtype=mx.int32)
        # Dtype in which the bound table already took the E4M3 round trip.
        self.table_round_trip_dtype = None

    def __call__(self, streams, rows, mask):
        """``streams [B, L, n, D]``, ``rows [B, L, heads, head_dim]`` (fp32), ``mask [B, L]``."""
        dtype = streams.dtype
        transformers = self.precision == "transformers"
        x = rows.astype(dtype)
        carried = self.table_round_trip_dtype is not None and self.table_round_trip_dtype == dtype
        if self.fp8 and transformers and not carried:
            x = fp8_round_trip(x)
        gammas = mx.stack([norm.weight for norm in self.lookup_per_head_norms])
        x = torch_rms_norm(x, gammas)
        x = x.reshape(*x.shape[:-2], -1)
        torch_round = transformers and getattr(self, "gate_rounding", "fp32") == "torch"
        if torch_round:
            self.gate_calls = getattr(self, "gate_calls", 0) + 1

        def gate_of(key, query):
            return engram_gate(key, query, self.hidden, transformers=transformers, torch_round=torch_round)

        if getattr(self, "fused", False):
            # Opt-in ``engram_fused_proj``: one GEMM for the n key projections
            # and the value projection; stacked norms; the gate on [B, L, n].
            parts = self.fused_proj.segments(x)
            value = parts[self.n]
            keys = mx.stack(parts[: self.n], axis=-2)  # [B, L, n, D]
            if _fusions.on_gpu():
                # One norm per operand with stacked gammas (bitwise the
                # per-stream norms on Metal, tests).  MLX's CPU rms_norm with
                # a weight rounds fp32 differently from norm-then-scale, so
                # the CPU keeps the per-stream calls.
                key = torch_rms_norm(keys, self.stacked_norm_key)
                query = torch_rms_norm(streams, self.stacked_norm_query)
            else:
                key = mx.stack([torch_rms_norm(keys[..., c, :], self.norm_key[c].weight)
                                for c in range(self.n)], axis=-2)
                query = mx.stack([torch_rms_norm(streams[..., c, :], self.norm_query[c].weight)
                                  for c in range(self.n)], axis=-2)
            gate = gate_of(key, query)[..., None]  # [B, L, n, 1]
        else:
            gates = []
            for c in range(self.n):
                key = torch_rms_norm(self.key_projs[c](x), self.norm_key[c].weight)
                query = torch_rms_norm(streams[..., c, :], self.norm_query[c].weight)
                gates.append(gate_of(key, query))
            gate = mx.stack(gates, axis=-1)[..., None]  # [B, L, n, 1]
            value = self.value_proj(x)
        scales = mx.stack(self.unified_per_hc_scales)  # [n, D]
        if transformers:
            main = self.unified_pre_gamma.astype(dtype) * torch_rms_norm(value)
            out = gate.astype(dtype) * main[..., None, :] + scales.astype(dtype) * main[..., None, :]
        else:
            value = value.astype(mx.float32)
            signal = (self.unified_pre_gamma.astype(mx.float32) * value
                      * mx.rsqrt(mx.mean(mx.square(value), axis=-1, keepdims=True) + TORCH_RMS_EPS))
            out = (gate + scales.astype(mx.float32)) * signal[..., None, :]
        out = mx.where(mask[..., None, None], out, 0.0).astype(dtype)
        return streams + out


# ------------------------------------------------------------------- cache --

_STEP = 256


def _buf_append(obj, p, x):
    """Append ``x [B, n, d]`` to buffer ``p`` of ``obj`` (``p_data/p_head/p_end``).

    Writes only at or past ``p_end`` into the existing buffer; growth (and,
    for a moving head, compaction) allocates a fresh buffer object, so rows
    below the old fill level of the old object are never rewritten.
    """
    data, head, end = getattr(obj, p + "_data"), getattr(obj, p + "_head"), getattr(obj, p + "_end")
    n = x.shape[1]
    if data is None or end + n > data.shape[1]:
        live = end - head
        grow = ((live + n + _STEP - 1) // _STEP) * _STEP
        new = mx.zeros((x.shape[0], max(grow, live + n), x.shape[2]), dtype=x.dtype)
        if data is not None and live:
            new[:, :live] = data[:, head:end]
        data, head, end = new, 0, live
    data[:, end : end + n] = x
    end += n
    setattr(obj, p + "_data", data)
    setattr(obj, p + "_head", head)
    setattr(obj, p + "_end", end)
    return data[:, head:end]


def _buf_view(obj, p):
    data = getattr(obj, p + "_data")
    return None if data is None else data[:, getattr(obj, p + "_head") : getattr(obj, p + "_end")]


def _buf_len(obj, p):
    return getattr(obj, p + "_end") - getattr(obj, p + "_head")


def _buf_load(obj, p, array):
    setattr(obj, p + "_data", array)
    setattr(obj, p + "_head", 0)
    setattr(obj, p + "_end", 0 if array is None else int(array.shape[1]))


def _buf_init(obj, p):
    setattr(obj, p + "_data", None)
    setattr(obj, p + "_head", 0)
    setattr(obj, p + "_end", 0)


class HY4FlashLayerCache(_BaseCache):
    """One lane's MLA history for one decoder layer.

    Buffers (``<name>_data`` ``[B, T, d]`` with ``_head``/``_end`` fill
    marks, flat on the cache so the engine can borrow them):

    * indexed layers: ``kv`` (normed latent ‖ roped key pe, read in place as
      attention key and value), ``tk`` (indexer token keys) and ``gk``
      (pooled group keys, fp32).  All three are append-only: writes land at
      or past ``_end``, so they are declared in
      ``_RECOVERY_APPEND_ONLY_FIELDS`` and recovery points / tree forks
      borrow them instead of aliasing (no copy on the next append).
    * sliding layers: ``ring`` retains ``window - 1 + ROLLBACK`` rows; its
      head moves and compaction rewrites rows, so it is not declared.

    ``selections`` maps an absolute position to the indexer's recalled slots
    for that query (``[B, K]``; ``None`` for a dense step), kept for the last
    ``ROLLBACK`` positions when ``record`` is set (DFly2 Index Share).
    ``token_tail`` (Engram layers only) holds the last token ids: enough
    that an Engram layer never bounds a rollback tighter than the sliding
    ring does (``Model.engram_tail_keep``).
    ``mrope_spans`` holds the lane's vision spans ``(slot, position, t, h, w)``
    (empty for text-only lanes); RoPE positions of later slots follow them.
    """

    cache_operations_contract = CacheOperationsContract(
        family="indexed_kv",
        topology="append_only",
        state_revision="hy4-flash-latent-rope-indexer-v2",
        cache_layout="hy4-flash-private-rows-v2",
        numerics="absorbed-mla-fused-sdpa-sinks-visible-window",
        planes=(CachePlaneDeclaration("attention_kv", ("state",), ("meta_state",)),),
        exact_prefix=True,
        restores_at_checkpoints=True,
        rollback="kv_trim",
    )
    _RECOVERY_APPEND_ONLY_FIELDS = (("kv_data", -2, "kv_end"), ("tk_data", -2, "tk_end"),
                                    ("gk_data", -2, "gk_end"))

    def __init__(self, kind: str = INDEXED, *, window: int = 0, ratio: int = 4,
                 record: bool = False, tail: int = 0):
        self.kind = kind
        self.window = int(window)
        self.ratio = int(ratio)
        self.offset = 0
        self.start = 0  # absolute position of the first retained kv row
        for p in self._buffers():
            _buf_init(self, p)
        self.record = bool(record)
        self.selections = {}
        self.tail_size = int(tail)
        self.token_tail = ()  # tuple of per-row token tuples (clone-safe)
        self.mrope_spans = ()  # tuple of vision spans (clone-safe)

    def _buffers(self):
        return ("kv", "tk", "gk") if self.kind == INDEXED else ("ring",)

    @property
    def _kv(self):
        return "kv" if self.kind == INDEXED else "ring"

    # -- appends -------------------------------------------------------------
    def defer_append_only_to(self, root):
        """Take this fork's append-only buffers from ``root`` at first use.

        ``verify_phase_align`` runs a lane's root chain (which borrows the
        live append-only buffers and writes past their fill level in place)
        and its child chains in ONE forward.  A child fork that aliased those
        buffers before the forward would hold a second reference to them, so
        the root's write could no longer reuse the buffer and would copy the
        whole history.  Instead the fork drops its references now and takes
        ``root``'s buffers when it first appends -- after ``root``'s write in
        the same layer, because a batch forward attends its rows in order --
        and its own write then copies on write, exactly as a wave-path child
        chain (cloned from its parent's finished state) does.  The fork reads
        only rows below its own fill marks, which ``root`` never rewrites.
        Sliding rings are never lent, so they stay aliased.
        """
        if self.kind != INDEXED:
            return
        if root.kind != INDEXED or any(getattr(root, p + "_head") != getattr(self, p + "_head")
                                       for p in ("kv", "tk", "gk")):
            raise ValueError("Hy4-flash deferred fork needs the root of the same lane and layer")
        for name, _, _ in self._RECOVERY_APPEND_ONLY_FIELDS:
            setattr(self, name, None)
        self._deferred_root = root

    def _resolve_deferred(self):
        root = self.__dict__.pop("_deferred_root", None)
        if root is None:
            return
        for name, _, _ in self._RECOVERY_APPEND_ONLY_FIELDS:
            data = getattr(root, name)
            setattr(self, name, None if data is None else mx.stop_gradient(data))

    def append(self, keys):
        """Append ``[B, L, r+p]``; return the rows this step attends and their first position."""
        self._resolve_deferred()
        L = keys.shape[1]
        rows = _buf_append(self, self._kv, keys)
        first = self.start
        self.offset += L
        if self.kind == SLIDING:
            excess = _buf_len(self, "ring") - (self.window - 1 + ROLLBACK)
            if excess > 0:
                self.ring_head += excess
                self.start += excess
        return rows, first

    def history(self, upto: int):
        """Indexed layers: stored ``kv`` rows for positions ``[0, upto)``.

        Rows are in the cache's storage format; ``latent_rows`` turns them
        (or a gather of them) into attention rows.
        """
        self._resolve_deferred()
        return _buf_view(self, "kv")[:, :upto]

    # Native rows are read in place: no copy, no cast.
    latent_kv = "native"
    approximate = False

    def latent_rows(self, rows, dtype):
        return rows

    def append_index_keys(self, token_keys, indexer):
        self._resolve_deferred()
        keys = _buf_append(self, "tk", token_keys)
        complete = keys.shape[1] // self.ratio
        have = _buf_len(self, "gk")
        if complete > have:
            positions = (slot_positions(self.mrope_spans, np.arange(have, complete) * self.ratio)
                         if self.mrope_spans else None)
            pooled = indexer.pool(keys[:, have * self.ratio : complete * self.ratio], have,
                                  positions=positions)
            _buf_append(self, "gk", pooled)
        return _buf_view(self, "gk") if _buf_len(self, "gk") else mx.zeros(
            (token_keys.shape[0], 0, token_keys.shape[2]), dtype=mx.float32)

    def record_selection(self, selection, start: int, rows: int):
        if not self.record:
            return
        first = max(0, rows - ROLLBACK)
        for j in range(first, rows):
            self.selections[start + j] = None if selection is None else selection[:, j]
        floor = start + rows - ROLLBACK
        for pos in [p for p in self.selections if p < floor]:
            del self.selections[pos]

    def selection_at(self, position: int):
        """Recalled slots ``[B, K]`` of the query at ``position``; None if dense.

        Raises KeyError when the position was not recorded (``record`` off,
        or older than the retained ``ROLLBACK`` window).
        """
        return self.selections[position]

    # -- batching / forks ------------------------------------------------
    def clone(self, *, lend: bool = False):
        """Independent owner.

        By default every buffer is aliased descriptor-only (MLX copies on the
        first write).  ``lend=True`` hands the append-only buffers' objects
        themselves to the fork: it writes only past this cache's fill level,
        which this cache never reads, so a fork that is discarded leaves this
        cache exact and its appends copy nothing.  At most one lent fork may
        be alive at a time (the tree executor's root chain).
        """
        self._resolve_deferred()
        obj = object.__new__(type(self))
        obj.__dict__.update(self.__dict__)
        lent = {name for name, _, _ in self._RECOVERY_APPEND_ONLY_FIELDS} if lend else set()
        for p in self._buffers():
            name = p + "_data"
            data = getattr(self, name)
            if data is not None and name not in lent:
                setattr(obj, name, mx.stop_gradient(data))
        obj.selections = {k: None if v is None else mx.stop_gradient(v)
                          for k, v in self.selections.items()}
        return obj

    @classmethod
    def merge(cls, caches):
        return HY4FlashBatchLayerCache.merge(caches)

    def can_trim(self, n):
        """Whether rolling back ``n`` positions leaves an exact cache.

        Every plane that keeps a bounded history must still cover what the
        next forward reads at the new length:

        * sliding layers: the last ``window - 1`` keys (exactness);
        * Engram layers: the last ``tail_size`` token ids, the next n-gram's
          history (exactness; an Engram cache that has seen tokens but holds
          no tail is refused).

        DFly2 Index Share selections need no guard: the drafter reads only
        the selection of the last committed token, and a landing always
        re-feeds at least one token, which records it afresh.

        APCv2 asks this before a branch landing (``preflight_trim``); a
        refused landing falls back to a shallower entry or a cold prefill.
        """
        n = min(int(n), self.offset)
        if n <= 0:
            return True
        new = self.offset - n
        if self.kind == SLIDING and new - self.start < min(new, self.window - 1):
            return False
        if self.tail_size and new > 0 and (
                not self.token_tail
                or any(len(row) - n < min(new, self.tail_size) for row in self.token_tail)):
            return False
        if n > 0 and any(s[0] < new < span_end(s) for s in self.mrope_spans):
            return False  # never cut a vision span in two
        return True

    def preflight_trim(self, n):
        """APCv2's branch-landing check: refuse a rollback past the retained window."""
        return self.can_trim(n)

    # -- cache protocol ----------------------------------------------------
    def is_trimmable(self):
        return True

    def trim(self, n):
        from .cache import _note_recovery_rewind

        self._resolve_deferred()
        n = min(int(n), self.offset)
        if n <= 0:
            return 0
        new = self.offset - n
        if not self.can_trim(n):
            raise ValueError("Hy4-flash cache cannot roll back past its retained window "
                             "or into a vision span")
        self.offset = new
        self.mrope_spans = tuple(s for s in self.mrope_spans if s[0] < new)
        kv = self._kv
        setattr(self, kv + "_end", getattr(self, kv + "_head") + (new - self.start))
        if self.kind == INDEXED:
            self.tk_end = min(self.tk_end, new)
            self.gk_end = min(self.gk_end, new // self.ratio)
        if self.token_tail:
            self.token_tail = tuple(row[: max(0, len(row) - n)] for row in self.token_tail)
        for pos in [p for p in self.selections if p >= new]:
            del self.selections[pos]
        _note_recovery_rewind(self)
        return n

    def size(self):
        return self.offset

    def empty(self):
        return self.offset == 0

    @property
    def nbytes(self):
        return sum(getattr(self, p + "_data").nbytes for p in self._buffers()
                   if getattr(self, p + "_data") is not None)

    # -- serialization ------------------------------------------------------
    # state: buffer views, then the recorded selections (oldest first);
    # meta_state: geometry, fill, Engram tail and the selection positions.
    @property
    def state(self):
        self._resolve_deferred()
        arrays = [_buf_view(self, p) for p in self._buffers()]
        arrays += [v for _, v in sorted(self.selections.items()) if v is not None]
        return arrays

    @state.setter
    def state(self, value):
        value = list(value)
        buffers = self._buffers()
        for p, array in zip(buffers, value):
            _buf_load(self, p, array)
        dense = set(getattr(self, "_pending_dense", ()))
        arrays = iter(value[len(buffers):])
        self.selections = {pos: (None if pos in dense else next(arrays))
                           for pos in getattr(self, "_pending_positions", ())}
        self.__dict__.pop("_pending_dense", None)
        self.__dict__.pop("_pending_positions", None)

    @property
    def meta_state(self):
        tail = ";".join(",".join(str(int(t)) for t in row) for row in self.token_tail)
        positions = sorted(self.selections)
        meta = (self.kind, str(self.window), str(self.ratio), str(self.offset), str(self.start),
                str(self.tail_size), str(len(self.token_tail)), tail, str(int(self.record)),
                ",".join(str(p) for p in positions),
                ",".join(str(p) for p in positions if self.selections[p] is None))
        if self.mrope_spans:  # text-only lanes keep the 11-field layout
            meta += (";".join(",".join(str(int(v)) for v in s) for s in self.mrope_spans),)
        return meta

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        obj.meta_state = meta_state
        obj.state = state
        return obj

    @meta_state.setter
    def meta_state(self, value):
        value = tuple(value)
        spans = value[11] if len(value) == 12 else ""
        (kind, window, ratio, offset, start, tail_size, rows, tail, record,
         positions, dense) = value[:11] if len(value) == 12 else value
        self.mrope_spans = tuple(tuple(int(v) for v in s.split(",")) for s in spans.split(";") if s)
        if any(len(s) != 5 for s in self.mrope_spans):
            raise ValueError("invalid Hy4-flash vision span state")
        self.kind, self.window, self.ratio = kind, int(window), int(ratio)
        self.offset, self.start = int(offset), int(start)
        self.record = bool(int(record))
        self.tail_size = int(tail_size)
        self.token_tail = tuple(tuple(int(t) for t in row.split(",") if t) for row in tail.split(";")) \
            if int(rows) else ()
        self._pending_positions = tuple(int(p) for p in positions.split(",") if p)
        self._pending_dense = tuple(int(p) for p in dense.split(",") if p)
        self.selections = {}
        for p in self._buffers():
            if not hasattr(self, p + "_data"):
                _buf_init(self, p)


class HY4FlashFP8LayerCache(HY4FlashLayerCache):
    """Opt-in approximate variant: ``kv`` / ``ring`` hold packed fp8 rows.

    Each row is ``hy_v4_flash_fp8kv.pack_rows`` of the native row (E4M3
    latent in 128-lane tiles with pow2 fp32 scales, bf16 RoPE tail): 656
    bytes for Hy4-flash against 1152 bf16.  Buffers, fill marks, append-only
    declarations, lending, trim and serialization are the native ones; only
    the bytes in a row differ, so rollback stays exact on the packed buffer.
    The indexer buffers (``tk``, ``gk``) stay native.  Approximate state: a
    distinct revision and layout with ``exact_prefix=False``; the adapter never declares
    APCv2 for these rows and declares no approximate APC operation, so they are
    never published as exact prefixes.
    """

    cache_operations_contract = CacheOperationsContract(
        family="indexed_kv",
        topology="append_only",
        state_revision="hy4-flash-latent-fp8-rope-bf16-indexer-v1",
        cache_layout="hy4-flash-private-rows-fp8-ds-mla-v1",
        numerics="approximate-fp8-e4m3-latent-pow2-tile-scales-bf16-rope-dequant-sdpa-sinks-visible-window",
        planes=(CachePlaneDeclaration("attention_kv", ("state",), ("meta_state",)),),
        exact_prefix=False,
        restores_at_checkpoints=True,
        rollback="kv_trim",
    )
    # Same write pattern as the native rows (only the row bytes differ), so
    # recovery points may borrow the append-only buffers.  Declared again:
    # recovery reads only a class's own declaration.
    _RECOVERY_APPEND_ONLY_FIELDS = HY4FlashLayerCache._RECOVERY_APPEND_ONLY_FIELDS
    latent_kv = "fp8"
    approximate = True

    def __init__(self, kind: str = INDEXED, *, rank: int, rope: int, **kwargs):
        super().__init__(kind, **kwargs)
        self.rank, self.rope = int(rank), int(rope)
        fp8_row_bytes(self.rank, self.rope)  # validates the tile geometry

    def append(self, keys):
        if keys.shape[-1] != self.rank + self.rope:
            raise ValueError("Hy4-flash fp8 latent row width disagrees with the cache geometry")
        return super().append(fp8_pack_rows(keys, self.rank))

    def latent_rows(self, rows, dtype):
        return fp8_unpack_rows(rows, self.rank, dtype)

    @property
    def meta_state(self):
        return (*HY4FlashLayerCache.meta_state.fget(self), FP8_ROW_REVISION, str(self.rank), str(self.rope))

    @meta_state.setter
    def meta_state(self, value):
        value = tuple(value)
        # Native fields (11, or 12 with vision spans), then the fp8 trailer.
        if len(value) not in (14, 15) or value[-3] != FP8_ROW_REVISION:
            raise ValueError("invalid Hy4-flash fp8 latent cache revision")
        HY4FlashLayerCache.meta_state.fset(self, value[:-3])
        self.rank, self.rope = int(value[-2]), int(value[-1])
        fp8_row_bytes(self.rank, self.rope)

    def _load_state(self, value):
        value = list(value)
        width = fp8_row_bytes(self.rank, self.rope)
        array = value[0] if value else None
        if array is not None and (array.dtype != mx.uint8 or array.shape[-1] != width):
            raise ValueError("Hy4-flash fp8 latent state is not packed fp8 rows")
        HY4FlashLayerCache.state.fset(self, value)

    state = property(HY4FlashLayerCache.state.fget, _load_state)


LAYER_CACHE_TYPES = (HY4FlashLayerCache, HY4FlashFP8LayerCache)


def layer_cache_type(meta_state):
    """Row cache class of a serialized row (fp8 rows carry the row revision)."""
    meta = tuple(meta_state)
    if len(meta) in (14, 15) and meta[-3] == FP8_ROW_REVISION:
        return HY4FlashFP8LayerCache
    if len(meta) in (11, 12):  # 12: with vision spans
        return HY4FlashLayerCache
    raise ValueError("unknown Hy4-flash layer cache metadata")


class HY4FlashBatchLayerCache(_BaseCache):
    """A batch of lanes for one layer: one private ``HY4FlashLayerCache`` per row.

    Projections, MoE, Engram and the head run on the joined ``[B, L]`` rows;
    each lane's attention reads only its own latent history (no padded joint
    slab, no cross-row mask).  ``prepare(lengths=, right_padding=)`` marks a
    ragged chunk; ``layout(width)`` gives each row its ``(start, count)``
    inside the chunk and the model calls ``advance(width)`` after a forward.
    """

    def cache_operations_contract(self):
        """Exact batch ABI, or the approximate one over fp8 latent rows."""
        if self.rows and type(self.rows[0]) is HY4FlashFP8LayerCache:
            return FP8_BATCH_OPERATIONS
        return BATCH_OPERATIONS

    def __init__(self, rows):
        rows = list(rows)
        if not rows or any(type(r) not in LAYER_CACHE_TYPES for r in rows):
            raise ValueError("Hy4-flash batch needs HY4FlashLayerCache rows")
        if len({id(r) for r in rows}) != len(rows):
            raise ValueError("batch rows must have independent owners")
        kinds = {(type(r), r.kind, r.window, r.ratio, getattr(r, "rank", None)) for r in rows}
        if len(kinds) != 1:
            raise ValueError("Hy4-flash batch rows come from different layers or latent formats")
        self.rows = rows
        self._remaining = None
        self._left = None

    # -- shape ---------------------------------------------------------------
    @property
    def offset(self):
        return self.rows[0].offset if len(self.rows) == 1 else mx.array(
            [r.offset for r in self.rows], mx.int32)

    @property
    def left_padding(self):
        return mx.zeros((len(self.rows),), mx.int32)

    @property
    def host_lengths(self):
        return [r.offset for r in self.rows]

    @property
    def nbytes(self):
        return sum(r.nbytes for r in self.rows)

    def size(self):
        return max((r.offset for r in self.rows), default=0)

    def empty(self):
        return all(r.offset == 0 for r in self.rows)

    # -- state ---------------------------------------------------------------
    @property
    def state(self):
        return tuple(tuple(r.state) for r in self.rows)

    @state.setter
    def state(self, values):
        if len(values) != len(self.rows):
            raise ValueError("batch state row mismatch")
        for row, value in zip(self.rows, values):
            row.state = value

    @property
    def meta_state(self):
        import json

        return ("hy4-flash-batch-v1", json.dumps([list(r.meta_state) for r in self.rows]))

    @classmethod
    def from_state(cls, state, meta_state):
        import json

        if len(meta_state) != 2 or meta_state[0] != "hy4-flash-batch-v1":
            raise ValueError("invalid Hy4-flash batch cache revision")
        metas = json.loads(meta_state[1])
        if len(metas) != len(state):
            raise ValueError("batch state/metadata row mismatch")
        return cls([layer_cache_type(m).from_state(v, tuple(m)) for v, m in zip(state, metas)])

    # -- lane set ------------------------------------------------------------
    @classmethod
    def merge(cls, caches):
        rows = []
        for cache in caches:
            rows.extend(cache.rows if isinstance(cache, HY4FlashBatchLayerCache) else [cache])
        return cls([r.clone() for r in rows])

    def extract(self, index):
        if type(index) is not int or not 0 <= index < len(self.rows):
            raise ValueError("invalid batch row index")
        return self.rows[index].clone()

    def filter(self, keep):
        keep = list(keep)
        if len(set(keep)) != len(keep) or any(type(i) is not int or not 0 <= i < len(self.rows) for i in keep):
            raise ValueError("invalid batch row filter")
        self.rows = [self.rows[i] for i in keep]
        if self._remaining is not None:
            self._remaining = [self._remaining[i] for i in keep]
        if self._left is not None:
            self._left = [self._left[i] for i in keep]

    def extend(self, other):
        added = [r.clone() for r in (other.rows if isinstance(other, HY4FlashBatchLayerCache) else [other])]
        self.rows = type(self)(self.rows + added).rows  # same row-class / format checks
        self._remaining = self._left = None

    # -- ragged chunks -----------------------------------------------------
    def prepare(self, *, lengths=None, left_padding=None, right_padding=None):
        count = len(self.rows)

        def vector(value, label):
            if value is None:
                return None
            result = [int(v) for v in (value.tolist() if hasattr(value, "tolist") else value)]
            if len(result) != count or any(n < 0 for n in result):
                raise ValueError("invalid batch " + label)
            return result

        remaining = vector(lengths, "lengths")
        right = vector(right_padding, "right padding")
        if right is not None and remaining is None:
            raise ValueError("right padding requires logical lengths")
        self._remaining, self._left = remaining, vector(left_padding, "left padding")

    def layout(self, width):
        left = self._left or [0] * len(self.rows)
        remaining = self._remaining or [width] * len(self.rows)
        out = []
        for i in range(len(self.rows)):
            start = min(left[i], width)
            out.append((start, min(width - start, remaining[i])))
        return out

    def advance(self, width):
        layout = self.layout(width)
        if self._remaining is not None:
            self._remaining = [n - c for n, (_, c) in zip(self._remaining, layout)]
        if self._left is not None:
            self._left = [max(0, n - width) for n in self._left]

    def finalize(self):
        if (self._remaining is not None and any(self._remaining)) or (self._left is not None and any(self._left)):
            raise ValueError("Hy4-flash batch prefill has incomplete rows")
        self._remaining = self._left = None

    # -- rollback ------------------------------------------------------------
    def is_trimmable(self):
        return True

    def trim(self, n):
        if not all(r.can_trim(n) for r in self.rows):
            raise ValueError("Hy4-flash cache cannot roll back past its retained window")
        return min(r.trim(n) for r in self.rows)

    def preflight_trim(self, n):
        """APCv2's branch-landing check for a lane batch: every row must admit it."""
        return all(r.can_trim(n) for r in self.rows)

    def supports_ragged_trim(self):
        return True

    def _ragged(self, n):
        n = [int(v) for v in (n.tolist() if hasattr(n, "tolist") else n)]
        if len(n) != len(self.rows) or any(v < 0 for v in n):
            raise ValueError("ragged trim needs one nonnegative count per row")
        if not all(r.can_trim(v) for r, v in zip(self.rows, n)):
            raise ValueError("Hy4-flash cache cannot roll back past its retained window")
        return n

    def preflight_ragged_trim(self, n, *, validate: bool = True):
        self._ragged(n)

    def trim_ragged(self, n, *, validate: bool = True):
        for row, count in zip(self.rows, self._ragged(n)):
            row.trim(count)

    def close(self):
        self.rows = []
        self._remaining = self._left = None


BATCH_OPERATIONS = CacheOperationsContract(
    family="indexed_kv",
    topology="append_only",
    state_revision="hy4-flash-latent-rope-indexer-v1",
    cache_layout="hy4-flash-private-rows-batch-v1",
    numerics="absorbed-mla-fused-sdpa-sinks-visible-window",
    planes=(CachePlaneDeclaration("attention_kv", ("state",), ("meta_state",)),),
    exact_prefix=True,
    restores_at_checkpoints=True,
    rollback="kv_trim",
)
FP8_BATCH_OPERATIONS = CacheOperationsContract(
    family="indexed_kv",
    topology="append_only",
    state_revision="hy4-flash-latent-fp8-rope-bf16-indexer-v1",
    cache_layout="hy4-flash-private-rows-fp8-ds-mla-batch-v1",
    numerics="approximate-fp8-e4m3-latent-pow2-tile-scales-bf16-rope-dequant-sdpa-sinks-visible-window",
    planes=(CachePlaneDeclaration("attention_kv", ("state",), ("meta_state",)),),
    exact_prefix=False,
    restores_at_checkpoints=True,
    rollback="kv_trim",
)


# ------------------------------------------------------------------- model --

class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, index: int):
        super().__init__()
        kind = args.layer_types[index]
        self.kind = kind
        self.self_attn = Attention(args, kind, indexer=kind == INDEXED)
        if args.mlp_layer_types[index] == "sparse":
            self.mlp = MoE(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size, post_norm=True, eps=args.rms_norm_eps)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hc_attn_layer = HyperConnection(args)
        self.hc_mlp_layer = HyperConnection(args)
        layer_id = args.engram_layers.get(index)
        if layer_id is not None:
            self.engram_layer = EngramLayer(args, layer_id)

    def __call__(self, streams, cache, engram=None):
        if engram is not None and "engram_layer" in self:
            streams = self.engram_layer(streams, *engram)
        seam = getattr(self, "hc_seam", None)
        if seam is not None:
            # Opt-in ``hc_seam_fused``: hc pre + input norm at the entry, and
            # merge + hc pre + norm between attention and MLP, fused.
            post, x = seam.pre(self.hc_attn_layer, self.input_layernorm, streams)
            x = self.self_attn(x, cache)
            streams, post, x = seam.merge_pre(self.hc_mlp_layer, self.post_attention_layernorm,
                                              post, x, streams)
            return hc_merge(post, self.mlp(x), streams)
        post, x = self.hc_attn_layer(streams)
        x = self.self_attn(self.input_layernorm(x), cache)
        streams = hc_merge(post, x, streams)
        post, x = self.hc_mlp_layer(streams)
        x = self.mlp(self.post_attention_layernorm(x))
        return hc_merge(post, x, streams)

    def tail_row(self, streams, cache, engram=None):
        """``__call__`` for the last row only, after every state write over all rows.

        Engram, the attention-side iHC boundary and the input norm run over
        every row (the attention input feeds the cached keys); attention
        returns the last row (``Attention.tail_row``); the merge, the MLP-side
        boundary, the MLP/MoE and the final merge run on that row.  All of
        them are row-wise, so the row is the full forward's up to the
        attention's own row-count rounding.
        """
        if engram is not None and "engram_layer" in self:
            streams = self.engram_layer(streams, *engram)
        seam = getattr(self, "hc_seam", None)
        if seam is not None:
            post, x = seam.pre(self.hc_attn_layer, self.input_layernorm, streams)
            x = self.self_attn.tail_row(x, cache)
            streams, post, x = seam.merge_pre(self.hc_mlp_layer, self.post_attention_layernorm,
                                              post[:, -1:], x, streams[:, -1:])
            return hc_merge(post, self.mlp(x), streams)
        post, x = self.hc_attn_layer(streams)
        x = self.self_attn.tail_row(self.input_layernorm(x), cache)
        streams = hc_merge(post[:, -1:], x, streams[:, -1:])
        post, x = self.hc_mlp_layer(streams)
        x = self.mlp(self.post_attention_layernorm(x))
        return hc_merge(post, x, streams)


class HY4FlashModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.hc_head = HyperHead(args)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)


class Model(nn.Module):
    """Hy4-flash text model.  ``__call__`` returns logits (fp32 head)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = HY4FlashModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.engram_hashers: Dict[int, EngramHasher] = {}
        self.engram_rows: Dict[int, Any] = {}
        self.record_layers: frozenset = frozenset()
        # ``host`` (default): n-grams hashed in NumPy after ``np.array(inputs)``.
        # ``device`` (opt-in, unqualified, exact): hashed by MLX ops on the caller's
        # stream; see ``_hidden_device_hash``.
        self.engram_hash = "host"
        self._device_hasher = None
        self.engram_hash_stats = {"host_forwards": 0, "device_forwards": 0, "device_rows": 0}
        # Trunk forwards (every route reaches ``_layers_from``): observed_used
        # evidence for load-time weight policies such as ``amoeba_mask``.
        self.trunk_stats = {"forwards": 0}
        # Opt-in (adapter policy ``moe_nax``), default off: open
        # ``moe_nax_gather.forward_scope`` around every trunk forward, so the
        # NAX sorted gather (when its process mode is on) may serve prefill
        # forwards.  The adapter admits it only on the ordinary route.
        self.moe_nax_prefill = False
        # Load-time record of the opt-in fused gate/up expert banks
        # (``fuse_routed_gate_up``); None when it was not selected.
        # A plain attribute, not a module child (it holds no parameters).
        object.__setattr__(self, "moe_gate_up_fusion", None)
        # Layer-output readers (opt-in; empty by default, so forwards are
        # unchanged): ``{layer: fn(layer, streams, ids, prefix)}`` called
        # after that decoder layer of every text forward through ``hidden``.
        # ``streams`` is the layer's ``[B, L, hc_mult, D]`` output (lazy),
        # ``ids`` the forward's host ``int64 [B, L]`` token ids with positions
        # outside a row's span (batch padding) set to -1, and ``prefix`` each
        # row's committed token ids before the forward (the Engram token tail;
        # ``None`` without Engram layers).  Readers must not change ``streams``.
        # Media prefill chunks call no reader.  Plain attribute, not a child.
        object.__setattr__(self, "layer_output_readers", {})
        # Prompt-chunk head trim (adapter policy ``prefill_head_trim``, default
        # on, Pierre 2026-11-21): inside a checkpoint-row request only each
        # lane's checkpoint row goes through the fp32 head, padded to the
        # smallest row count whose kernel is bitwise the full chunk's
        # (``calibrate_head_rows``).  Plain attributes, not parameters.
        self.prefill_head_trim = True
        # Prompt-chunk last-layer tail (policy ``prefill_tail_skip``, default
        # on; prefill_tail_skip tier 2, family ``hy_v4_flash``).
        self.prefill_tail_skip = True
        self.prefill_exact_stats = {"head_trim_forwards": 0, "head_rows_skipped": 0,
                                    "head_trim_declines": {}, "tail_forwards": 0,
                                    "tail_rows_skipped": 0}
        object.__setattr__(self, "_head_row_floors", {})

    @property
    def sliding_attention_min_block(self) -> int:
        """Smallest admitted band block: the sliding window, and never below
        ``ROLLBACK``.  A forward takes the band only when it has more rows
        than one block, and no verify or tree forward can (a speculative
        round must stay trimmable, so it has at most ``ROLLBACK`` rows), so
        verify numerics never change; only prefill chunks do."""
        return max(int(self.args.sliding_window), ROLLBACK)

    def set_sliding_attention_blocks(self, rows: int) -> None:
        """Opt-in band attention on every sliding layer (``rows`` per query
        block; 0 restores the default whole-slice attention)."""
        if type(rows) is not int or rows < 0:
            raise ValueError("sliding_attention_blocks must be 0 (off) or a positive row count")
        if rows and rows < self.sliding_attention_min_block:
            raise ValueError(
                f"sliding_attention_blocks must be at least {self.sliding_attention_min_block} "
                "(the sliding window, at least ROLLBACK): smaller blocks would reach verify forwards")
        for layer in self.model.layers:
            if layer.self_attn.kind == SLIDING:
                layer.self_attn.sliding_attention_blocks = rows

    def set_moe_expert_read_once(self, on: bool) -> None:
        """Opt-in read-once small-row routed-expert kernels on every MoE layer
        (False: the stock SwitchGLU gathers, the default)."""
        if type(on) is not bool:
            raise ValueError("moe_expert_read_once must be a boolean")
        for layer in self.model.layers:
            if isinstance(layer.mlp, MoE):
                layer.mlp.expert_read_once = on

    def moe_read_once_status(self) -> dict:
        """Per-model counters of the read-once expert route (selected vs observed_used)."""
        moes = [layer.mlp for layer in self.model.layers if isinstance(layer.mlp, MoE)]
        calls = sum(m.read_once_calls for m in moes)
        declines = {}
        for m in moes:
            for reason, n in m.read_once_declines.items():
                declines[reason] = declines.get(reason, 0) + n
        return {"selected": any(m.expert_read_once for m in moes), "layer_calls": calls,
                "declined_layer_calls": declines, "observed_used": calls > 0}

    def set_sparse_prefill_union_rows(self, rows: int) -> None:
        """Opt-in H4 union prefill on every indexed layer (``rows`` queries per
        union sub-chunk; 0 restores the default routes)."""
        if type(rows) is not int or rows < 0:
            raise ValueError("sparse_prefill_union_rows must be 0 (off) or a positive row count")
        for layer in self.model.layers:
            if layer.self_attn.kind == INDEXED:
                layer.self_attn.sparse_prefill_union_rows = rows

    def set_sparse_prefill_attention(self, on: bool) -> None:
        """Opt-in sparse prefill kernel on every indexed layer (False: the
        dense masked reference, the default)."""
        if type(on) is not bool:
            raise ValueError("sparse_prefill_attention must be a boolean")
        for layer in self.model.layers:
            if layer.self_attn.kind == INDEXED:
                layer.self_attn.sparse_prefill_attention = on

    @property
    def sparse_prefill_attention(self) -> bool:
        return any(layer.self_attn.sparse_prefill_attention for layer in self.model.layers
                   if layer.self_attn.kind == INDEXED)

    @property
    def sliding_attention_blocks(self) -> int:
        rows = {layer.self_attn.sliding_attention_blocks for layer in self.model.layers
                if layer.self_attn.kind == SLIDING}
        return max(rows) if rows else 0

    # -- engram ----------------------------------------------------------
    def bind_engram(self, rows_by_layer: Dict[int, Any]):
        """Build the hashers from the loaded lookup tables and bind row readers.

        ``rows_by_layer`` maps each Engram decoder layer index to an object
        with ``read(flat_rows) -> mx.array [n, head_dim]``: a
        ``ResidentEngramRows`` or an SSD ``EngramTableRows``.
        """
        a = self.args
        sizes = a.engram_layer_head_vocab_sizes or {}
        if set(rows_by_layer) != set(a.engram_layers):
            raise ValueError("every Engram layer needs exactly one table reader")
        self.engram_hashers, self.engram_rows = {}, {}
        for index, layer_id in a.engram_layers.items():
            layer = self.model.layers[index].engram_layer
            hasher = EngramHasher(
                np.array(layer.lookup_table), sizes[str(layer_id)], layer_id=layer_id,
                seed=a.engram_seed, max_ngram=a.engram_max_ngram_size,
                heads_per_ngram=a.engram_n_head_per_ngram, pad_id=a.engram_pad_id)
            reader = rows_by_layer[index]
            rows = getattr(reader, "rows", None)
            if rows is None and hasattr(reader, "weight"):
                rows = reader.weight.shape[0]
            if rows is not None and int(rows) != hasher.rows:
                raise ValueError(f"Engram layer {index} table has {rows} rows, hash layout needs {hasher.rows}")
            carried = getattr(reader, "fp8_round_trip_dtype", None)
            if carried is not None and not a.engram_fp8_embedding:
                raise ValueError("an FP8 Engram table needs engram_fp8_embedding (it carries the round trip)")
            activation = embedding_dtype(self.model.embed_tokens)
            if carried is not None and carried != activation:
                # The model would quantize the rows a second time: refuse
                # rather than serve an unlabelled approximation.
                raise ValueError(f"FP8 Engram table took its round trip in {carried}, "
                                 f"the model runs in {activation}")
            layer.table_round_trip_dtype = carried
            self.engram_hashers[index] = hasher
            self.engram_rows[index] = reader
        self._device_hasher = None
        if self.engram_hash == "device":
            self.device_hasher  # refuse a geometry it cannot hash exactly at load, not at decode

    @property
    def device_hasher(self):
        """The stacked device hasher of every Engram layer (``engram_hash: device``)."""
        if self._device_hasher is None:
            from .hy_v4_flash_device_hash import DeviceEngramHasher

            self._device_hasher = DeviceEngramHasher(
                [self.engram_hashers[i] for i in sorted(self.engram_hashers)])
        return self._device_hasher

    @staticmethod
    def pop_engram_tables(weights: dict) -> Dict[int, Any]:
        """Remove the (sanitized) resident Engram tables from ``weights``."""
        out = {}
        for key in list(weights):
            if key.endswith("engram_layer.multi_head_embedding_weight"):
                out[int(key.split(".")[2])] = weights.pop(key)
        return out

    def _ngram_tokens(self, tokens: np.ndarray, start: int, tail: np.ndarray,
                      document_positions: Optional[np.ndarray] = None) -> np.ndarray:
        """Each position's token with the ``max_ngram - 1`` before it; pad before 0.

        ``document_positions`` (the temporal M-RoPE plane, given only when the
        lane has vision spans) replaces the slot as the reference's
        ``document_offsets`` (``build_ngram_tokens``).
        """
        a = self.args
        n = a.engram_max_ngram_size
        B, L = tokens.shape
        history = np.concatenate([tail[:, -(n - 1):] if tail.size else np.zeros((B, 0), np.int64),
                                  tokens], axis=1)
        lead = history.shape[1] - L
        out = np.full((B, L, n), a.engram_pad_id, dtype=np.int64)
        for k in range(n):
            pos = np.arange(L) + start if document_positions is None else document_positions
            src = np.arange(L) + lead - k
            ok = (pos >= k) & (src >= 0)
            out[:, ok, k] = history[:, src[ok]]
        return out

    @property
    def engram_tail_keep(self) -> int:
        """Token ids each Engram cache keeps: the n-gram history plus the
        sliding ring's ``window - 1 + ROLLBACK``, so any rollback the ring
        admits (all of them before it wraps) still leaves the next n-gram
        its history; the tail never refuses an APCv2 landing the window allows."""
        a = self.args
        window = a.sliding_window - 1 if SLIDING in a.layer_types else 0
        return a.engram_max_ngram_size - 1 + window + ROLLBACK

    def engram_inputs(self, tokens: np.ndarray, start: int, caches) -> Dict[int, tuple]:
        """Per-layer ``(rows [B, L, heads, hd] fp32, mask [B, L])`` for one lane set.

        ``caches`` are row caches sharing ``start``; their ``token_tail``
        supplies the n-gram history before this chunk.  Every layer's rows
        are submitted to its reader before any is read, so the four tables'
        reads overlap instead of running one after another.
        """
        plan = self._engram_plan(tokens, start, caches)
        return {index: resolve() for index, resolve in plan.items()}

    def _engram_plan(self, tokens: np.ndarray, start: int, caches, hashed=None) -> Dict[int, Any]:
        """Hash one lane set's rows, submit every layer's reads, return resolvers.

        ``{layer: resolve() -> (rows, mask)}``; the token tails are updated
        here, so the plan is made exactly once per forward.  ``resolve``
        waits only for its own layer's reads.  ``hashed`` (``{layer: rows
        [B, L, heads]}``) are rows the device hasher already computed; they
        equal ``hasher.table_rows`` of this chunk's n-grams.
        """
        if not self.args.engram_layers:
            return {}
        if set(self.engram_hashers) != set(self.args.engram_layers):
            raise ValueError("Hy4-flash Engram tables are not bound (Model.bind_engram)")
        a = self.args
        first = min(self.engram_hashers)
        tail_rows = caches[first].token_tail
        tail = np.array(tail_rows, dtype=np.int64) if tail_rows and len(tail_rows[0]) else \
            np.zeros((tokens.shape[0], 0), np.int64)
        ngrams = None
        if hashed is None:
            spans = getattr(caches[first], "mrope_spans", ())
            document = (slot_positions(spans, np.arange(start, start + tokens.shape[1]))[0]
                        if spans else None)
            ngrams = self._ngram_tokens(tokens, start, tail, document)
        mask = tokens != a.engram_pad_id
        for special in (a.image_token_id, a.video_token_id):
            if special is not None:
                mask &= tokens != special
        out = {}
        # Queue optional OS read-ahead for later tables before issuing the first
        # required prefetch. With hints off, hashing and I/O still overlap.
        advised = [index for index in self.engram_hashers if index != first
                   and getattr(self.engram_rows[index], "read_advise", False)]
        if advised:
            if hashed is None:
                hashed = {index: hasher.table_rows(ngrams) for index, hasher in self.engram_hashers.items()}
            for index in advised:
                self.engram_rows[index].advise(hashed[index].reshape(-1))
        mask_mx = mx.array(mask)
        for index, hasher in self.engram_hashers.items():
            rows = hasher.table_rows(ngrams) if hashed is None else hashed[index]  # [B, L, heads]
            reader = self.engram_rows[index]
            prefetch = getattr(reader, "prefetch", None)
            if callable(prefetch):
                # Required: every row is read by ``resolve`` (never declined).
                prefetch(rows.reshape(-1), required=True)

            def resolve(reader=reader, rows=rows):
                B, L, heads = rows.shape
                data = reader.read(rows.reshape(-1)).reshape(B, L, heads, -1)
                return data.astype(mx.float32), mask_mx

            out[index] = resolve
        keep = self.engram_tail_keep
        combined = np.concatenate([tail, tokens], axis=1)[:, -keep:]
        packed = tuple(tuple(int(t) for t in row) for row in combined.tolist())
        for index in self.engram_hashers:
            caches[index].token_tail = packed
        return out

    def _batched_engram(self, tokens: np.ndarray, layout, cache) -> Dict[int, tuple]:
        """Engram inputs for a lane batch: each row with its own tail and span."""
        plan = self._batched_engram_plan(tokens, layout, cache)
        return {index: resolve() for index, resolve in plan.items()}

    def _batched_engram_plan(self, tokens: np.ndarray, layout, cache, hashed=None) -> Dict[int, Any]:
        """Every lane's reads for every layer submitted first; per-layer resolvers.

        ``hashed`` (device hashing) holds one ``{layer: rows}`` per lane
        with a nonzero count, in lane order.
        """
        if not self.args.engram_layers:
            return {}
        B, L = tokens.shape
        parts = []
        hashed = iter(hashed) if hashed is not None else None
        for b, (start, count) in enumerate(layout):
            lane = [c.rows[b] for c in cache]
            if count == 0:
                parts.append(None)
                continue
            parts.append((start, count, self._engram_plan(
                tokens[b : b + 1, start : start + count], lane[0].offset, lane,
                None if hashed is None else next(hashed))))
        return {index: (lambda index=index: self._batched_resolve(index, parts, L))
                for index in self.args.engram_layers}

    def _batched_resolve(self, index, parts, L):
        rows, masks = [], []
        for part in parts:
            if part is None:
                rows.append(None)
                masks.append(mx.zeros((1, L), dtype=mx.bool_))
                continue
            start, count, value = part
            data, mask = value[index]()
            pad = [(0, 0), (start, L - start - count)]
            rows.append(mx.pad(data, pad + [(0, 0), (0, 0)]))
            masks.append(mx.pad(mask, pad))
        shape = next(r for r in rows if r is not None).shape[1:]
        rows = [r if r is not None else mx.zeros((1, *shape), mx.float32) for r in rows]
        return mx.concatenate(rows, axis=0), mx.concatenate(masks, axis=0)

    # -- forward -----------------------------------------------------------
    def make_cache(self):
        a = self.args
        caches = []
        fp8 = a.latent_kv == "fp8"
        for i, kind in enumerate(a.layer_types):
            options = dict(window=a.sliding_window, ratio=a.compressed_indexer_compress_ratio,
                           record=i in self.record_layers,
                           tail=a.engram_max_ngram_size - 1 if i in a.engram_layers else 0)
            if fp8:
                rope = a.swa_qk_rope_head_dim if kind == SLIDING else a.qk_rope_head_dim
                caches.append(HY4FlashFP8LayerCache(kind, rank=a.kv_lora_rank, rope=rope, **options))
            else:
                caches.append(HY4FlashLayerCache(kind, **options))
        return caches

    # -- vision ------------------------------------------------------------
    def attach_vision(self, tower) -> None:
        """Bind a loaded ``VisionTower`` (after the text weights are loaded)."""
        if tower.args.out_hidden_size != self.args.hidden_size:
            raise ValueError("Hy4-flash vision tower width differs from the text width")
        if self.args.image_token_id is None:
            raise ValueError("Hy4-flash config has no image_token_id")
        self.vision_tower = tower

    @property
    def has_vision(self) -> bool:
        return "vision_tower" in self

    def _vision_features(self, tokens: np.ndarray, cache, pixel_values, image_grid_thw):
        """Encode this chunk's images.

        Returns ``(features [n_image_tokens, D] in placeholder order, the lane
        row caches, the lane's prior spans, its spans after this chunk)``.
        """
        a = self.args
        if not self.has_vision:
            raise ValueError("Hy4-flash vision tower is not loaded; image input is unavailable")
        if image_grid_thw is None:
            raise ValueError("Hy4-flash pixel_values need image_grid_thw")
        if tokens.shape[0] != 1:
            raise ValueError("Hy4-flash image prefill runs one lane at a time (B=1)")
        lanes = []
        for c in cache:
            rows = getattr(c, "rows", None)
            if rows is not None and len(rows) != 1:
                raise ValueError("Hy4-flash image prefill runs one lane at a time (B=1)")
            lanes.append(rows[0] if rows is not None else c)
        grids = np.asarray(image_grid_thw.tolist() if hasattr(image_grid_thw, "tolist")
                           else image_grid_thw, dtype=np.int64).reshape(-1, 3)
        start, prior = lanes[0].offset, lanes[0].mrope_spans
        tower = self.vision_tower
        new = find_spans(tokens[0], start, a.image_token_id, grids.tolist(),
                         tower.args.spatial_merge_size, prior)
        features = tower(pixel_values, grids.tolist())
        count = int(np.sum(tokens == a.image_token_id))
        if features.shape != (count, a.hidden_size):
            raise ValueError("Hy4-flash image features and image tokens do not match")
        stats = self.__dict__.setdefault("vision_stats", {"prefills": 0, "images": 0, "image_tokens": 0})
        stats["prefills"] += 1
        stats["images"] += len(new)
        stats["image_tokens"] += count
        return features, lanes, prior, prior + new

    def _media_hidden(self, inputs, tokens, cache, features, lanes, prior, spans):
        """The media prefill chunk: features scattered over the placeholders,
        spans recorded on every lane cache, layers run in bounded slices."""
        a = self.args
        L = tokens.shape[1]
        batched = hasattr(cache[0], "rows")
        if batched and cache[0].layout(L) != [(0, L)]:
            raise ValueError("Hy4-flash image prefill needs an unpadded B=1 chunk")
        h = self.model.embed_tokens(inputs)
        # ``masked_scatter``: features replace the placeholders in order.
        h[0, mx.array(np.flatnonzero(tokens[0] == a.image_token_id))] = features.astype(h.dtype)
        for lane in lanes:
            lane.mrope_spans = spans
        outs = []
        try:
            for s in range(0, L, MEDIA_PREFILL_STEP):
                e = min(L, s + MEDIA_PREFILL_STEP)
                engram = self.engram_inputs(tokens[:, s:e], lanes[0].offset, lanes)
                x = h[:, s:e]
                streams = mx.broadcast_to(x[:, :, None, :], (*x.shape[:2], a.hc_mult, x.shape[-1]))
                for i, layer in enumerate(self.model.layers):
                    streams = layer(streams, lanes[i], engram.get(i))
                outs.append(self.model.norm(self.model.hc_head(streams)))
                if e < L:
                    mx.eval(outs[-1], [c.state for c in lanes])
        except BaseException:
            for lane in lanes:
                lane.mrope_spans = prior
            raise
        if batched:
            for c in cache:
                c.advance(L)
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)

    def hidden(self, inputs: mx.array, cache=None, *, pixel_values=None, image_grid_thw=None,
               tail_row: bool = False):
        """Post-norm final hidden ``[B, L, D]`` (DFly2's target feature).

        ``pixel_values`` ``[patches, C*p*p]`` with ``image_grid_thw`` (one
        ``(t, h, w)`` patch grid per image) is the media prefill; image
        placeholder tokens without them, and any video token, fail closed.
        """
        if cache is None:
            cache = self.make_cache()
        if self.moe_nax_prefill:
            # A no-op while the NAX mode is off or an outer forward decided.
            with _moe_nax.forward_scope(inputs, cache):
                return self._hidden_route(inputs, cache, pixel_values, image_grid_thw, tail_row)
        return self._hidden_route(inputs, cache, pixel_values, image_grid_thw, tail_row)

    def _hidden_route(self, inputs, cache, pixel_values, image_grid_thw, tail_row=False):
        # ``tail_row`` (prompt-chunk tail, ``__call__`` only): the last layer
        # returns each lane's last row after writing its state over all rows.
        if tail_row and pixel_values is not None:
            raise ValueError("Hy4-flash tail row is text-only")
        if (self.engram_hash == "device" and pixel_values is None and self.engram_hashers
                and inputs.ndim == 2 and inputs.shape[1] <= ROLLBACK):
            return self._hidden_device_hash(inputs, cache, tail_row)
        return self._hidden(inputs, cache, pixel_values, image_grid_thw, tail_row)

    def _hidden(self, inputs, cache, pixel_values, image_grid_thw, tail_row=False):
        a = self.args
        # Host n-gram hashing needs the token ids: this waits for whatever
        # produced ``inputs`` (at decode, the previous step's sampler).
        tokens = np.array(inputs).astype(np.int64)
        if a.video_token_id is not None and np.any(tokens == a.video_token_id):
            raise ValueError("Hy4-flash video input is not implemented")
        if pixel_values is not None:
            return self._media_hidden(inputs, tokens, cache,
                                      *self._vision_features(tokens, cache, pixel_values, image_grid_thw))
        if a.image_token_id is not None and np.any(tokens == a.image_token_id):
            raise ValueError("Hy4-flash image tokens need their pixel_values (media prefill)")
        batched = hasattr(cache[0], "rows")
        if batched and tokens.shape[0] != len(cache[0].rows):
            raise ValueError("Hy4-flash batch cache rows disagree with the input batch")
        # Before the Engram plan, which moves the token tails past this chunk.
        reading = self._reader_inputs(tokens, cache, batched) if self.layer_output_readers else None
        if batched:
            engram = self._batched_engram_plan(tokens, cache[0].layout(tokens.shape[1]), cache)
        else:
            engram = self._engram_plan(tokens, cache[0].offset, cache)
        if engram:
            self.engram_hash_stats["host_forwards"] += 1
        h = self.model.embed_tokens(inputs)
        streams = mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], a.hc_mult, h.shape[-1]))
        return self._layers_from(streams, cache, engram, 0, tokens.shape[1], batched, reading,
                                 tail_row=tail_row)

    def _reader_inputs(self, tokens: np.ndarray, cache, batched: bool):
        """``(ids, prefix)`` for the layer-output readers of one forward.

        ``ids``: the host token ids with every position outside its row's
        span (batch padding) set to -1.  ``prefix``: each row's committed
        tokens before the chunk, from the first Engram layer's token tail
        (read before this forward's plan moves it), or None.
        """
        ids = tokens.astype(np.int64, copy=True)
        L = tokens.shape[1]
        if batched:
            keep = np.zeros(ids.shape, dtype=bool)
            refed = getattr(self, "_reader_refed_prefix", None)
            for b, (start, count) in enumerate(cache[0].layout(L)):
                skip = min(count, refed[b]) if refed is not None and b < len(refed) else 0
                keep[b, start + skip:start + count] = True
            ids[~keep] = -1
        prefix = None
        if self.args.engram_layers:
            first = min(self.args.engram_layers)
            if batched:
                prefix = [list(r.token_tail[0]) if r.token_tail else [] for r in cache[first].rows]
            else:
                tail = cache[first].token_tail
                prefix = ([list(row) for row in tail] if tail and len(tail) == ids.shape[0]
                          else [[] for _ in range(ids.shape[0])])
        return ids, prefix

    def _read_layer_output(self, i, streams, reading):
        reader = self.layer_output_readers.get(i)
        if reader is not None and reading is not None:
            reader(i, streams, *reading)

    def _layers_from(self, streams, cache, engram, begin: int, L: int, batched: bool, reading=None,
                     *, tail_row: bool = False):
        """Run decoder layers ``begin..`` and the head; dispatch at Engram boundaries.

        The rows of every Engram layer were submitted before the first one
        ran.  After the first Engram layer, and before each later one waits
        for its rows, the layers built so far are handed to the GPU
        (``async_eval``: same graph, same numerics), so the device works
        while Python builds the next segment.  ``reading`` feeds the
        ``layer_output_readers`` (``_reader_inputs``).
        """
        first_engram = min(engram) if engram else None
        self.trunk_stats["forwards"] += 1
        last = len(self.model.layers) - 1
        for i in range(begin, len(self.model.layers)):
            layer = self.model.layers[i]
            resolve = engram.get(i)
            if resolve is not None and i != first_engram:
                _dispatch(streams)
            run = layer.tail_row if tail_row and i == last else layer
            streams = run(streams, cache[i], resolve() if resolve is not None else None)
            if self.layer_output_readers:
                self._read_layer_output(i, streams, reading)
            if i == first_engram and len(self.model.layers) - i > 1:
                _dispatch(streams)
        if batched:
            for c in cache:
                c.advance(L)
        return self.model.norm(self.model.hc_head(streams))

    # -- device Engram hashing (opt-in ``engram_hash: device``) ------------
    def _device_ngrams(self, ids: mx.array, caches, start: int) -> mx.array:
        """``ids [r, L]`` (int32, device) -> n-gram tokens ``[r*L, max_ngram]``.

        The same gather as ``_ngram_tokens``: the index plan depends only on
        host metadata (slot, tail length, vision spans); the history is the
        lane's host token tail uploaded next to the device ids.
        """
        a = self.args
        n = a.engram_max_ngram_size
        r, L = ids.shape
        first = caches[min(self.engram_hashers)]
        tail_rows = first.token_tail
        tail = np.array(tail_rows, dtype=np.int64) if tail_rows and len(tail_rows[0]) else \
            np.zeros((r, 0), np.int64)
        tail = tail[:, -(n - 1):] if tail.size else np.zeros((r, 0), np.int64)
        spans = getattr(first, "mrope_spans", ())
        document = (slot_positions(spans, np.arange(start, start + L))[0] if spans else None)
        lead = tail.shape[1]
        index = np.full((L, n), -1, dtype=np.int64)
        for k in range(n):
            pos = np.arange(L) + start if document is None else document
            src = np.arange(L) + lead - k
            ok = (pos >= k) & (src >= 0)
            index[ok, k] = src[ok]
        history = ids if lead == 0 else mx.concatenate([mx.array(tail.astype(np.int32)), ids], axis=1)
        gathered = mx.take(history, mx.array(np.maximum(index, 0).reshape(-1).astype(np.int32)), axis=1)
        out = mx.where(mx.array(index >= 0)[None], gathered.reshape(r, L, n), a.engram_pad_id)
        return out.reshape(r * L, n)

    def _hidden_device_hash(self, inputs: mx.array, cache, tail_row: bool = False):
        """Text forward (``L <= ROLLBACK``) with the n-grams hashed on device.

        Order: the n-gram rows of every lane and Engram layer are computed
        and dispatched; the layers before the first Engram layer are built
        and dispatched; then one host read of ``rows ‖ token ids`` (the
        forward's only wait) feeds the fail-closed token checks, the token
        tails, the masks and the SSD row reads, exactly as on the host
        path.  Same rows, same graph: exact.

        The prefix layers append to their caches before the token checks
        run.  Any failure up to the Engram plan (a refused video or image
        token, a reader error while the row reads are submitted) restores
        those caches' saved fields, so the call leaves them as the host path
        would: untouched.  A reader error raised later, when a layer resolves
        its rows, leaves the earlier layers appended on both paths; the
        ReadPool is closed by then (``ReadPool.fail``), so the session ends.  Appends write
        only past the saved fill level or into a fresh buffer object
        (``_buf_append``), so the saved fields are the whole prior state.
        """
        a = self.args
        B, L = inputs.shape
        batched = hasattr(cache[0], "rows")
        if batched:
            if B != len(cache[0].rows):
                raise ValueError("Hy4-flash batch cache rows disagree with the input batch")
            layout = cache[0].layout(L)
            lanes = [(b, b + 1, start, count, [c.rows[b] for c in cache])
                     for b, (start, count) in enumerate(layout) if count]
        else:
            layout = None
            lanes = [(0, B, 0, L, cache)]
        hasher = self.device_hasher
        first = min(self.engram_hashers)
        ids = inputs.astype(mx.int32)
        parts = [self._device_ngrams(ids[lo:hi, start:start + count], caches, caches[0].offset)
                 for lo, hi, start, count, caches in lanes]
        ngrams = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
        rows = hasher(ngrams)  # [engram layers, N, heads]
        combined = mx.concatenate([rows.reshape(-1), ids.reshape(-1)])
        mx.async_eval(combined)
        prefix = [c for i in range(first) for c in (cache[i].rows if batched else [cache[i]])]
        saved = [(c, dict(c.__dict__), dict(c.selections)) for c in prefix]
        early = []  # (layer, streams) of read layers before the host ids exist
        try:
            h = self.model.embed_tokens(inputs)
            streams = mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], a.hc_mult, h.shape[-1]))
            for i in range(first):
                streams = self.model.layers[i](streams, cache[i], None)
                if i in self.layer_output_readers:
                    early.append((i, streams))
            if first:
                _dispatch(streams)
            host = np.array(combined)  # the forward's one host wait
            flat = host[: rows.size].reshape(rows.shape).astype(np.int64)
            tokens = host[rows.size:].reshape(B, L).astype(np.int64)
            if a.video_token_id is not None and np.any(tokens == a.video_token_id):
                raise ValueError("Hy4-flash video input is not implemented")
            if a.image_token_id is not None and np.any(tokens == a.image_token_id):
                raise ValueError("Hy4-flash image tokens need their pixel_values (media prefill)")
            # Before the Engram plan, which moves the token tails past this chunk.
            reading = self._reader_inputs(tokens, cache, batched) if self.layer_output_readers else None
            order = sorted(self.engram_hashers)
            hashed, offset = [], 0
            for lo, hi, _, count, _ in lanes:
                size = (hi - lo) * count
                hashed.append({index: flat[j, offset:offset + size].reshape(hi - lo, count, -1)
                               for j, index in enumerate(order)})
                offset += size
            if batched:
                engram = self._batched_engram_plan(tokens, layout, cache, hashed)
            else:
                engram = self._engram_plan(tokens, cache[first].offset, cache, hashed[0])
        except BaseException:
            for c, fields, selections in saved:
                c.__dict__.clear()
                c.__dict__.update(fields)
                c.selections = selections
            raise
        stats = self.engram_hash_stats
        stats["device_forwards"] += 1
        stats["device_rows"] += int(ngrams.shape[0])
        for i, early_streams in early:
            self._read_layer_output(i, early_streams, reading)
        return self._layers_from(streams, cache, engram, first, L, batched, reading,
                                 tail_row=tail_row)

    # -- speculative / AST target contract --------------------------------
    # Taps are the post-norm final hidden (DFly2's target feature); only the
    # last layer can be captured.
    external_feature_source = "final_hidden"
    supports_speculative_rollback = True

    @property
    def external_feature_width(self):
        return self.args.hidden_size

    @property
    def speculative_args(self):
        from types import SimpleNamespace

        a = self.args
        return SimpleNamespace(
            vocab_size=a.vocab_size, hidden_size=a.hidden_size,
            num_attention_heads=a.num_attention_heads, num_key_value_heads=1,
            head_dim=a.kv_lora_rank + a.qk_rope_head_dim, num_hidden_layers=a.num_hidden_layers,
            num_experts_per_tok=a.num_experts_per_tok, intermediate_size=a.intermediate_size,
            moe_intermediate_size=a.moe_intermediate_size)

    def forward_with_taps(self, inputs, cache, capture_layers, *, body_only=False):
        layers = tuple(capture_layers)
        if layers != (self.args.num_hidden_layers - 1,):
            raise ValueError("Hy4-flash taps only the final post-norm hidden (last layer)")
        hidden = self.hidden(inputs, cache)
        return (None if body_only else self.logits(hidden)), hidden

    def prefill_body(self, inputs, cache, capture_layers):
        return self.forward_with_taps(inputs, cache, capture_layers, body_only=True)[1]

    def media_prefill_body(self, inputs, cache, capture_layers, *, pixel_values=None,
                           image_grid_thw=None, **unknown):
        """One isolated media prefill chunk on the external-draft engine.

        The same ``hidden(pixel_values=, image_grid_thw=)`` forward the
        ordinary route runs; the lane decodes plain (the adapter declares
        ``multimodal_plain_decode_routes``), so only the final hidden is
        returned, as ``prefill_body`` does.
        """
        if unknown:
            raise ValueError(f"Hy4-flash media prefill does not take {sorted(unknown)}")
        if pixel_values is None or image_grid_thw is None:
            raise ValueError("Hy4-flash media prefill needs pixel_values and image_grid_thw")
        layers = tuple(capture_layers)
        if layers != (self.args.num_hidden_layers - 1,):
            raise ValueError("Hy4-flash taps only the final post-norm hidden (last layer)")
        return self.hidden(inputs, cache, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    def logits(self, hidden):
        head = self.lm_head
        if not self.args.enable_lm_head_fp32:
            return head(hidden).astype(mx.float32)
        if "scales" not in head:
            return _resident.f32_matmul(hidden, head.weight, kernel=getattr(self, "resident_gemv", False))
        # A quantized head keeps the fp32 contract: fp32 input, scales and
        # biases (``sanitize`` stores them fp32 once), fp32 accumulation.
        biases = head.get("biases")
        out = mx.quantized_matmul(
            hidden.astype(mx.float32), head["weight"], head["scales"].astype(mx.float32),
            None if biases is None else biases.astype(mx.float32), transpose=True,
            group_size=head.group_size, bits=head.bits, mode=head.mode)
        if "bias" in head:
            out = out + head["bias"].astype(mx.float32)
        return out

    def __call__(self, inputs: mx.array, cache=None, *, pixel_values=None, image_grid_thw=None,
                 **_unused):
        for key in ("pixel_values_videos", "video_grid_thw"):
            if _unused.get(key) is not None:
                raise ValueError("Hy4-flash video input is not implemented")
        if pixel_values is None and self._claim_tail(inputs, cache):
            from ..prefill_tail_skip import STATS as _TAIL_STATS

            _TAIL_STATS["hy_v4_flash_tail_split"] += 1
            stats = self.prefill_exact_stats
            stats["tail_forwards"] += 1
            stats["tail_rows_skipped"] += int(inputs.shape[0]) * (int(inputs.shape[1]) - 1)
            return self.logits(self.hidden(inputs, cache, tail_row=True))
        return self._prompt_head(self.hidden(inputs, cache, pixel_values=pixel_values,
                                             image_grid_thw=image_grid_thw))

    # -- prompt-chunk head trim and tail (exact defaults) -------------------
    # Candidate row counts for the head floor, smallest first; references are
    # one 256-row lane and two 160-row lanes (above every kernel switch
    # measured: Metal q8 matrix kernel from 33-65 rows, Accelerate sgemm from
    # 2-32 rows on the tiny head).
    HEAD_ROW_CANDIDATES = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128)
    HEAD_ROW_REFERENCES = ((1, 256), (2, 160))

    def head_row_floor(self) -> int:
        """Rows the trimmed head pads to on this device (0: never trim); memoized."""
        key = (str(mx.default_device()), "scales" in self.lm_head, bool(self.args.enable_lm_head_fp32))
        floor = self._head_row_floors.get(key)
        if floor is None:
            floor = self._head_row_floors[key] = self.calibrate_head_rows()
        return floor

    def calibrate_head_rows(self) -> int:
        """Smallest candidate row count at which ``logits`` of rows padded with zeros
        is bitwise ``logits`` of the same rows inside a full chunk, on this device.

        The head's kernel (MLX vector vs matrix GEMM, Accelerate's sgemm
        blocking) depends on the row count, so one row through the head is
        not bitwise the same row of a 2048-row chunk.  At and above the
        returned floor the kernels computed each row the same way.  Random
        rows at the real widths, two reference shapes, three rows each
        (first, middle, last); evidence, not proof, like
        ``FusedLinear.self_check``.  0 when no candidate matches (the head
        then always runs full).
        """
        D = self.args.hidden_size
        dtype = embedding_dtype(self.model.embed_tokens)
        refs = []
        for seed, shape in enumerate(self.HEAD_ROW_REFERENCES):
            x = mx.random.normal((*shape, D), key=mx.random.key(7919 + seed)).astype(dtype)
            flat = x.reshape(-1, D)
            n = flat.shape[0]
            picks = [0, n // 2, n - 1]
            refs.append((flat, picks, self.logits(x).reshape(n, -1)))
        for rows in self.HEAD_ROW_CANDIDATES:
            same = True
            for flat, picks, ref in refs:
                k = min(rows, len(picks))
                chosen = mx.array(picks[:k], dtype=mx.int32)
                padded = mx.concatenate([flat[chosen], mx.zeros((rows - k, D), dtype=flat.dtype)])
                out = self.logits(padded[None])[0, :k]
                if not mx.array_equal(out, ref[chosen]).item():
                    same = False
                    break
            if same:
                return rows
        return 0

    def _decline_head_trim(self, reason):
        d = self.prefill_exact_stats["head_trim_declines"]
        d[reason] = d.get(reason, 0) + 1

    def _prompt_head(self, hidden):
        """The vocabulary head of a forward; checkpoint rows only inside a prompt request.

        Outside a checkpoint-row request (decode, verify, scoring, taps) and
        whenever trimming is off or cannot be bitwise, every row runs.
        Inside one, each lane's checkpoint row is gathered, padded with zero
        rows to ``max(head_row_floor(), lanes)`` and projected; the result
        is ``[B, 1, V]``, bitwise the full head's rows (calibrated floor).
        """
        from ..prefill_checkpoint_rows import open_checkpoint_rows, select_checkpoint_rows

        from ..prefill_tail_skip import tail_skip_mode

        if not self.prefill_head_trim or hidden.ndim != 3 or hidden.shape[1] <= 1:
            return self.logits(hidden)
        if tail_skip_mode(TAIL_SKIP_FAMILY) == "0":
            # MLX2_PREFILL_TAIL_SKIP=0 turns off the head trim too, as on the
            # other families, so an env-only A/B compares against every row.
            return self.logits(hidden)
        B, L, D = hidden.shape
        if open_checkpoint_rows(B, L) is None:
            return self.logits(hidden)
        floor = self.head_row_floor()
        if floor == 0:
            self._decline_head_trim("no_bitwise_floor")
            return self.logits(hidden)
        rows = max(floor, B)
        if B * L <= rows:
            self._decline_head_trim("chunk_within_floor")
            return self.logits(hidden)
        kept = select_checkpoint_rows(hidden).reshape(B, D)
        if rows > B:
            kept = mx.concatenate([kept, mx.zeros((rows - B, D), dtype=kept.dtype)])
        stats = self.prefill_exact_stats
        stats["head_trim_forwards"] += 1
        stats["head_rows_skipped"] += B * L - rows
        return self.logits(kept[None])[0, :B].reshape(B, 1, -1)

    _TAIL_MODULES = None

    @classmethod
    def _tail_module_types(cls):
        """Module classes the last layer may hold for the tail split (stock and fused)."""
        if cls._TAIL_MODULES is None:
            from .switch_layers import (GateUpFusedSwitchGLU, QuantizedSwitchLinear, SwitchGLU,
                                        SwitchLinear)

            cls._TAIL_MODULES = frozenset({
                DecoderLayer, Attention, CompressedIndexer, MoE, MLP, Router, HyperConnection, HCPre,
                MultiLinear, ClampedSwiGLU, Q2ZeroSwitchLinear, EngramLayer, nn.Linear,
                nn.QuantizedLinear, nn.RMSNorm, SwitchGLU, GateUpFusedSwitchGLU, SwitchLinear,
                QuantizedSwitchLinear, _fusions.FusedLinear, _fusions.SegmentLinear})
        return cls._TAIL_MODULES

    def _tail_refusal(self, inputs, cache):
        """Why this prompt chunk keeps the full last layer, or None."""
        from ..prefill_tail_skip import structural_refusal

        layers = self.model.layers
        last = len(layers) - 1
        if cache is None or len(cache) != len(layers):
            return "hy_v4_flash_cache_layout"
        layer = layers[last]
        if layer.self_attn.kind != INDEXED:
            return "hy_v4_flash_last_layer_sliding"
        allowed = self._tail_module_types()
        if any(type(m) not in allowed for _, m in layer.named_modules()):
            # Capture hooks, TensorFold rows and any other wrapper.
            return "hy_v4_flash_wrapped_last_layer"
        reason = structural_refusal(inputs, cache, trunk=self.model, trunk_class=HY4FlashModel)
        if reason is not None:
            return reason
        if last in self.layer_output_readers:
            # A calculator tile / aMoEba capture reads every row of the last layer.
            return "hy_v4_flash_last_layer_reader"
        if any(getattr(self, name, None) is not None for name in (
                "_mlx2_weight_overlays", "_mlx2_layer_excision", "_mlx2_expert_masks",
                "_mlx2_ablation_overlays", "_mlx2_amoeba_rows")):
            return "hy_v4_flash_weight_overlay"
        entry = cache[last]
        lanes = getattr(entry, "rows", None) or [entry]
        if any(type(c) not in LAYER_CACHE_TYPES for c in lanes):
            return "hy_v4_flash_cache_type"
        if any(c.record for c in lanes):
            # DFly2 Index Share reads the last ROLLBACK rows' selections.
            return "hy_v4_flash_index_share_record"
        if any(c.mrope_spans for c in lanes):
            return "hy_v4_flash_mrope_spans"
        if hasattr(entry, "layout"):
            L = int(inputs.shape[1])
            if any(span != (0, L) for span in entry.layout(L)):
                # Left padding (or any lane not covering the whole chunk):
                # Attention.tail_row needs every lane full.
                return "hy_v4_flash_lane_layout"
        fused = layer.self_attn.get("fused_input_proj") if "fused_input_proj" in layer.self_attn else None
        if (fused is not None and not hasattr(entry, "rows")
                and int(inputs.shape[0]) * int(inputs.shape[1]) <= fused.max_rows):
            # The full forward takes its key rows from the one fused input
            # GEMM here; the tail's split key projection is only self-checked
            # equal to it, so a chunk this short keeps the reference.
            return "hy_v4_flash_fused_input_rows"
        return None

    def _claim_tail(self, inputs, cache) -> bool:
        """Claim the open checkpoint-row request for the last-layer tail split.

        Only this plain prompt call reaches it: taps (``forward_with_taps``),
        ``prefill_body``, media prefill and verify forwards call ``hidden``
        directly or open no request, so every reader of all rows keeps the
        full forward.  ``MLX2_PREFILL_TAIL_SKIP=0`` or policy
        ``prefill_tail_skip: false`` turns it off.
        """
        from ..prefill_checkpoint_rows import final_row_checkpoint_open
        from ..prefill_tail_skip import claim_adapter_tail, tail_skip_mode

        if (not self.prefill_tail_skip or getattr(inputs, "ndim", 0) != 2 or inputs.shape[1] <= 1
                or tail_skip_mode(TAIL_SKIP_FAMILY) != "1"):
            return False
        if not final_row_checkpoint_open(inputs.shape[0], inputs.shape[1]):
            # No claimable request (decode, verify, scoring, ragged lanes):
            # skip the refusal walk on the hot path.
            return False
        return claim_adapter_tail(inputs, kind="tail", refuse=self._tail_refusal(inputs, cache),
                                  family=TAIL_SKIP_FAMILY)

    def prefill_exact_status(self) -> dict:
        """Counters of the head trim, the tail split and the sorted MoE combine."""
        moes = [layer.mlp for layer in self.model.layers if isinstance(layer.mlp, MoE)]
        declines = {}
        for m in moes:
            for reason, n in m.sorted_sum_declines.items():
                declines[reason] = declines.get(reason, 0) + n
        stats = self.prefill_exact_stats
        return {
            "head_trim": {"selected": bool(self.prefill_head_trim),
                          "row_floor": {f"{k[0]}|quantized={k[1]}|fp32={k[2]}": v
                                        for k, v in self._head_row_floors.items()},
                          "forwards": stats["head_trim_forwards"],
                          "rows_skipped": stats["head_rows_skipped"],
                          "declined": dict(stats["head_trim_declines"]),
                          "observed_used": stats["head_trim_forwards"] > 0},
            "tail_skip": {"selected": bool(self.prefill_tail_skip),
                          "forwards": stats["tail_forwards"],
                          "rows_skipped": stats["tail_rows_skipped"],
                          "observed_used": stats["tail_forwards"] > 0},
            "sorted_weighted_sum": {"selected": any(m.sorted_weighted_sum for m in moes),
                                    "layer_calls": sum(m.sorted_sum_calls for m in moes),
                                    "declined_layer_calls": declines,
                                    "observed_used": any(m.sorted_sum_calls for m in moes)},
        }

    def set_moe_sorted_weighted_sum(self, on: bool) -> None:
        """Sorted-order MoE combine on every MoE layer (False: the eager tail)."""
        if type(on) is not bool:
            raise ValueError("moe_sorted_weighted_sum must be a boolean")
        for layer in self.model.layers:
            if isinstance(layer.mlp, MoE):
                layer.mlp.sorted_weighted_sum = on

    @property
    def layers(self):
        return self.model.layers

    # -- opt-in fused gate/up expert banks ---------------------------------
    def fuse_routed_gate_up(self, weights: dict) -> dict:
        """Swap each routed ``SwitchGLU`` whose gate and up banks qualify for
        a ``GateUpFusedSwitchGLU`` and fuse its planes in ``weights``.

        Call after quantization (and the q2_0 install), before the strict
        load.  A bank that does not qualify (a ``q2_0`` bank, mixed formats,
        an unquantized bank) keeps the split route; its reason is recorded.
        Returns (and keeps as ``moe_gate_up_fusion``) the load record.
        """
        from .switch_layers import (
            GateUpFusedSwitchGLU, fused_gate_up_planes, gate_up_fusion_refusal)

        fused, refused = [], {}
        for index, layer in enumerate(self.model.layers):
            moe = layer.mlp
            if not isinstance(moe, MoE):
                continue
            stem = f"model.layers.{index}.mlp.experts"
            reason = gate_up_fusion_refusal(moe.experts)
            if reason is None and f"{stem}.gate_proj.scales" not in weights:
                reason = "planes_missing"
            if reason is not None:
                refused[index] = reason
                continue
            twin = GateUpFusedSwitchGLU.from_switch_glu(moe.experts)
            fused_gate_up_planes(weights, stem)
            moe.experts = twin
            fused.append(index)
        record = {"fused_layers": fused, "refused_layers": refused}
        object.__setattr__(self, "moe_gate_up_fusion", record)
        return record

    def moe_gate_up_status(self) -> dict:
        """Route record of the fused gate/up banks (selected vs observed_used)."""
        from .switch_layers import GateUpFusedSwitchGLU

        record = self.moe_gate_up_fusion
        calls = sum(layer.mlp.experts._fused_calls for layer in self.model.layers
                    if isinstance(layer.mlp, MoE) and type(layer.mlp.experts) is GateUpFusedSwitchGLU)
        return {
            "implemented": True, "qualified": False, "selected": record is not None,
            "fused_layers": list(record["fused_layers"]) if record else [],
            "refused_layers": dict(record["refused_layers"]) if record else {},
            "fused_calls": calls, "observed_used": calls > 0,
            "exactness": "bit-identical to the split gate/up gathers on CPU (tests); GPU unmeasured",
        }

    # -- weights -----------------------------------------------------------
    def sanitize(self, weights: dict) -> dict:
        """Checkpoint names -> module names.

        Drops ``vit.*`` (the vision tower loads separately, ``attach_vision``) and ``model.dfly_layers.*`` (loaded by
        ``HY4FlashDFly``); splits packed expert ``gate_up_proj``; folds
        ``kv_b_proj`` into per-head ``embed_q`` / ``unembed_out``; moves the
        Engram tables to ``multi_head_embedding_weight``; keeps the
        reference's fp32 modules in fp32.
        """
        a = self.args
        out = {}
        for key, value in weights.items():
            if key.startswith(("vit.", "model.dfly_layers.", "model.vision_tower", "mtp.")):
                continue
            if key.startswith("model.language_model."):
                key = "model." + key[len("model.language_model."):]
            # mlx-lm-style packs (e.g. the Nace.AI MLX pack) name the routed
            # experts ``mlp.switch_mlp``; mlx2's module is ``mlp.experts``.
            key = switch_mlp_name(key)
            if ".self_attn.kv_b_proj." in key and not key.endswith("kv_b_proj.weight"):
                raise ValueError(f"{key}: a quantized kv_b_proj cannot be folded into embed_q/unembed_out")
            if key.endswith("mlp.experts.gate_up_proj"):
                inter = value.shape[1] // 2
                stem = key[: -len("gate_up_proj")]
                out[stem + "gate_proj.weight"] = value[:, :inter]
                out[stem + "up_proj.weight"] = value[:, inter:]
                continue
            if key.endswith("mlp.experts.down_proj"):
                out[key + ".weight"] = value
                continue
            if key.endswith("self_attn.kv_b_proj.weight"):
                stem = key[: -len("kv_b_proj.weight")]
                index = int(key.split(".")[2])
                kind = a.layer_types[index]
                out.update(fold_kv_b(stem, value, a, kind))
                continue
            if key.endswith("engram_layer.multi_head_embedding.embedding.weight"):
                key = key.replace("multi_head_embedding.embedding.weight", "multi_head_embedding_weight")
            out[key] = value
        for key in list(out):
            if not any(tag in key for tag in _FP32_KEYS):
                continue
            stem = key[: -len(".weight")] if key.endswith(".weight") else None
            if stem is not None and stem + ".scales" in out:
                # Packed codes stay as they are.  A quantized LM head under
                # enable_lm_head_fp32 keeps fp32 scales/biases instead.
                if stem == "lm_head" and a.enable_lm_head_fp32:
                    for part in (".scales", ".biases"):
                        if stem + part in out:
                            out[stem + part] = out[stem + part].astype(mx.float32)
                continue
            out[key] = out[key].astype(mx.float32)
        return out

    @property
    def quant_predicate(self):
        def predicate(path, module):
            if path.endswith("mlp.gate") or any(tag in path for tag in (
                    "engram_layer", "compressed_indexer.key_", "hc_", "embed_q", "unembed_out")):
                return False
            return True
        return predicate


# Stored fp32 once at load so no step re-reads a bf16 copy to upcast it: the
# reference keeps these modules fp32 (``_keep_in_fp32_modules_strict``; the
# LM head is fp32 under ``enable_lm_head_fp32``; the router computes in fp32).
_FP32_KEYS = ("e_score_correction_bias", "hc_attn_layer.", "hc_mlp_layer.", "hc_head.",
              "learnable_sink_param", "key_value_projection", "key_gate_projection",
              "intra_group_ape", "mlp.gate.weight", "lm_head.weight")


def switch_mlp_name(key: str) -> str:
    """``...mlp.switch_mlp.<proj>...`` (mlx-lm naming) -> ``...mlp.experts.<proj>...``."""
    return key.replace(".mlp.switch_mlp.", ".mlp.experts.") if ".mlp.switch_mlp." in key else key


def embedding_dtype(embedding) -> mx.Dtype:
    """Activation dtype an (optionally quantized) embedding returns."""
    return embedding["scales"].dtype if "scales" in embedding else embedding.weight.dtype


def fold_kv_b(stem: str, weight: mx.array, args: ModelArgs, kind: str) -> dict:
    """``kv_b_proj [H*(nope+v), r]`` -> ``embed_q [H, r, nope]``, ``unembed_out [H, v, r]``."""
    sliding = kind == SLIDING
    H = args.swa_num_attention_heads if sliding else args.num_attention_heads
    nope = args.swa_qk_nope_head_dim if sliding else args.qk_nope_head_dim
    v = args.swa_v_head_dim if sliding else args.v_head_dim
    w = weight.reshape(H, nope + v, -1)
    return {stem + "embed_q.weight": w[:, :nope, :].swapaxes(-1, -2),
            stem + "unembed_out.weight": w[:, nope:, :]}


# ------------------------------------------------------------------- DFly2 --

class DFlyFusion(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.enorm = nn.RMSNorm(d, eps=eps)
        self.hnorm = nn.RMSNorm(d, eps=eps)
        self.eh_proj = nn.Linear(2 * d, d, bias=False)
        self.gate = mx.zeros((d,))

    def __call__(self, anchors, target_hidden):
        fused = self.eh_proj(mx.concatenate([self.enorm(anchors), self.hnorm(target_hidden)], axis=-1))
        return anchors + self.gate * fused


class DFlyHiddenCorrection(nn.Module):
    def __init__(self, d: int, intermediate: int, eps: float):
        super().__init__()
        self.hidden_norm = nn.RMSNorm(d, eps=eps)
        self.embed_norm = nn.RMSNorm(d, eps=eps)
        self.gate_proj = nn.Linear(2 * d, intermediate, bias=False)
        self.up_proj = nn.Linear(2 * d, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, d, bias=False)

    def __call__(self, hidden, previous_embed):
        x = mx.concatenate([self.hidden_norm(hidden), self.embed_norm(previous_embed)], axis=-1)
        return hidden + self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlyLayer(nn.Module):
    """Pre-norm residual block: indexed-shape MLA (no indexer) + dense MLP."""

    def __init__(self, args: ModelArgs, intermediate: int):
        super().__init__()
        self.self_attn = Attention(args, INDEXED, indexer=False)
        self.mlp = MLP(args.hidden_size, intermediate, post_norm=False, eps=args.rms_norm_eps)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, h, anchor: int, history, history_valid):
        """``history [B, K, r+p]`` target rows (KV Share), ``history_valid [B, K]``."""
        attn = self.self_attn
        x = self.input_layernorm(h)
        q, _, keys = attn.project(x, anchor)
        B, W = h.shape[:2]
        K = history.shape[1]
        kv = mx.concatenate([history.astype(keys.dtype), keys], axis=1)
        block = mx.arange(W)
        causal = block[None, :] <= block[:, None]
        allowed = mx.concatenate([
            mx.broadcast_to(history_valid[:, None, :], (B, W, K)),
            mx.broadcast_to(causal[None], (B, W, W))], axis=-1)
        out = mla_attend(q, kv[:, None], allowed[:, None], attn.learnable_sink_param,
                         attn.scale, attn.kv_lora_rank)
        h = h + attn.finish(x, out)
        return h + self.mlp(self.post_attention_layernorm(h))


class HY4FlashDFly(nn.Module):
    """DFly2 drafter: KV Share + Index Share over the target's sparse layers.

    One query block per round: ``[bonus, mask x N]`` at positions
    ``anchor .. anchor+N``.  Slot 0 is fused with the target's post-norm
    hidden of the last accepted token.  Draft layer ``i`` attends to the
    target latent history of ``target_kv_layers[i]`` at the positions the
    target's indexer selected (``target_index_layer[i]``) for that token,
    plus the block causally.  Each mask slot's output is corrected with the
    previous token's embedding and read through the target's LM head.
    """

    def __init__(self, args: ModelArgs, *, intermediate: int, correction: int):
        super().__init__()
        if args.dfly_num_hidden_layers < 1:
            raise ValueError("checkpoint declares no DFly2 layers")
        if args.dfly_mtp_fusion != "slot0" or not args.dfly_enable_hidden_correction:
            raise ValueError("DFly2 requires slot0 fusion and hidden correction")
        n = args.dfly_num_hidden_layers
        self.index_layers = list(args.dfly_target_index_layer or [])
        self.kv_layers = list(args.dfly_target_kv_layers or [])
        if len(self.index_layers) != n or len(self.kv_layers) != n:
            raise ValueError("DFly2 needs one target index/KV layer per draft layer")
        for i in self.index_layers + self.kv_layers:
            if not 1 <= i < args.num_hidden_layers or args.layer_types[i] != INDEXED:
                raise ValueError(f"DFly2 target layer {i} must be an indexed (sparse) layer")
        if len(set(self.kv_layers)) != n:
            raise ValueError("DFly2 requires distinct target KV layers")
        if args.dfly_mask_token_id is None or not 0 <= args.dfly_mask_token_id < args.vocab_size:
            raise ValueError("DFly2 mask token is outside the vocabulary")
        self.args = args
        self.mask_token = int(args.dfly_mask_token_id)
        self.local_window = int(args.dfly_context_local_window or 0)
        self.layers = [DFlyLayer(args, intermediate) for _ in range(n)]
        self.final_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mtp_fusion = DFlyFusion(args.hidden_size, args.rms_norm_eps)
        self.hidden_correction = DFlyHiddenCorrection(args.hidden_size, correction, args.rms_norm_eps)

    @staticmethod
    def geometry(weights: dict) -> tuple[int, int]:
        inter = weights["model.dfly_layers.0.mlp.gate_proj.weight"].shape[0]
        corr = weights["model.dfly_layers.hidden_correction.gate_proj.weight"].shape[0]
        return int(inter), int(corr)

    def sanitize(self, weights: dict) -> dict:
        prefix = "model.dfly_layers."
        out = {}
        for key, value in weights.items():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            if name.startswith("confidence_head."):
                continue
            if name.split(".", 1)[0].isdigit():
                name = "layers." + name
            if name.endswith("self_attn.kv_b_proj.weight"):
                out.update(fold_kv_b(name[: -len("kv_b_proj.weight")], value, self.args, INDEXED))
                continue
            if "learnable_sink_param" in name:
                value = value.astype(mx.float32)
            out[name] = value
        return out

    def history(self, caches, layer: int, anchor: int, dtype=None):
        """Target rows draft layer ``layer`` reads: those the target's indexer
        recalled for the last committed token (position ``anchor - 1``), all
        before the anchor.  Stored fp8 rows are dequantized to ``dtype``
        after the gather (native rows are returned as stored)."""
        source = caches[self.kv_layers[layer]]
        rows_all = source.history(anchor)  # [B, anchor, r+p] stored rows
        selection = caches[self.index_layers[layer]].selection_at(anchor - 1)
        window_start = anchor - self.local_window
        dtype = mx.float32 if dtype is None else dtype
        if selection is None:
            return source.latent_rows(rows_all, dtype), mx.ones(rows_all.shape[:2], dtype=mx.bool_)
        rows = mx.where((selection >= 0) & (selection < window_start), selection, -1)
        if self.local_window:
            local = mx.arange(window_start, anchor)
            local = mx.broadcast_to(mx.where(local >= 0, local, -1)[None], (rows.shape[0], local.size))
            rows = mx.concatenate([rows, local], axis=-1)
        gathered = mx.take_along_axis(rows_all, mx.maximum(rows, 0)[..., None], axis=1)
        return source.latent_rows(gathered, dtype), rows >= 0

    def propose(self, target: Model, caches, bonus: mx.array, target_hidden: mx.array,
                *, anchor: int, num_draft: int, choose=None):
        """Draft tokens ``[B, num_draft]`` and their logits ``[B, num_draft, V]`` (fp32).

        ``choose(step, logits [B, V]) -> tokens [B]`` picks each draft token
        (default greedy argmax; vLLM's ``probabilistic`` draws here), which the
        next step's hidden correction then reads.  ``anchor`` is the bonus
        token's position (the target cache's committed length).
        """
        h = self.block_hidden(target, caches, bonus, target_hidden, anchor=anchor, num_draft=num_draft)
        previous = bonus.reshape(-1).astype(mx.int32)
        drafted, logits = [], []
        for step in range(num_draft):
            row = self.step_logits(target, h[:, step + 1], previous)
            previous = (mx.argmax(row, axis=-1) if choose is None else choose(step, row)).astype(mx.int32)
            drafted.append(previous)
            logits.append(row)
        return mx.stack(drafted, axis=1), mx.stack(logits, axis=1)

    def block_hidden(self, target: Model, caches, bonus: mx.array, target_hidden: mx.array,
                     *, anchor: int, num_draft: int) -> mx.array:
        """The round's one block forward: final-normed ``[B, num_draft + 1, D]``.

        It reads only the bonus token, the target feature and the target
        caches, never the drafts, so every tree node of a round shares it.
        """
        B = bonus.shape[0]
        ids = mx.concatenate([bonus.reshape(B, 1).astype(mx.int32),
                              mx.full((B, num_draft), self.mask_token, dtype=mx.int32)], axis=1)
        h = target.model.embed_tokens(ids)
        slot0 = self.mtp_fusion(h[:, :1], target_hidden.reshape(B, 1, -1).astype(h.dtype))
        h = mx.concatenate([slot0, h[:, 1:]], axis=1)
        for i, layer in enumerate(self.layers):
            rows, valid = self.history(caches, i, anchor, dtype=h.dtype)
            h = layer(h, anchor, rows, valid)
        return self.final_norm(h)

    def step_logits(self, target: Model, slot_hidden: mx.array, previous: mx.array) -> mx.array:
        """Draft logits ``[K, V]`` (fp32) of mask-slot outputs ``[K, D]`` given
        each row's previous token ``[K]`` (hidden correction + LM head).  Rows
        are independent, so one call scores any set of tree nodes."""
        embed = target.model.embed_tokens(previous.astype(mx.int32))
        return target.logits(self.hidden_correction(slot_hidden, embed))


def dfly_greedy_generate(target: Model, drafter: HY4FlashDFly, prompt: Sequence[int], *,
                         max_tokens: int, num_draft: int = 5, eos_ids=(), prefill_step: int = 512):
    """Exact greedy DFly2 speculative decode for one request (B = 1).

    Every committed token is the target's own greedy choice, so the output
    equals ordinary greedy decode up to near-tie numerics of the verify
    shape.  Returns ``(tokens, receipt)``; the receipt carries vLLM-style
    ``verify_accept_hist`` (accepted drafts per verify round) and
    ``verify_depth_hist`` (drafted positions per round, the per-position
    acceptance denominator; the last rounds draft fewer than ``num_draft``).
    """
    if num_draft < 1:
        raise ValueError("DFly2 num_draft must be positive")
    record = set(drafter.index_layers)
    previous_record = target.record_layers
    target.record_layers = frozenset(record)
    try:
        caches = target.make_cache()
        prompt = list(prompt)
        hidden = None
        for s in range(0, len(prompt), prefill_step):
            hidden = target.hidden(mx.array([prompt[s:s + prefill_step]]), caches)
        source_row = hidden.shape[1] - 1
        last_hidden = hidden[:, source_row]
        bonus = int(mx.argmax(target.logits(last_hidden), axis=-1).item())
        out = [bonus]
        hist = [0] * (num_draft + 1)
        depth_hist = [0] * (num_draft + 1)
        rounds = 0
        eos = set(int(e) for e in eos_ids)
        while len(out) < max_tokens and bonus not in eos:
            anchor = caches[0].offset
            depth = min(num_draft, max_tokens - len(out))
            drafts, _ = drafter.propose(target, caches, mx.array([bonus]), last_hidden,
                                        anchor=anchor, num_draft=num_draft)
            drafts = [int(t) for t in np.array(drafts[0])[:depth]]
            block = mx.array([[bonus] + drafts])
            hidden = target.hidden(block, caches)
            choice = [int(t) for t in np.array(mx.argmax(target.logits(hidden), axis=-1)[0])]
            accepted = 0
            while accepted < depth and drafts[accepted] == choice[accepted]:
                accepted += 1
            committed = drafts[:accepted]
            stop = next((j for j, t in enumerate(committed) if t in eos), None)
            if stop is not None:
                committed = committed[: stop + 1]
                accepted = stop + 1
            rounds += 1
            hist[accepted] += 1
            depth_hist[depth] += 1
            for c in caches:
                c.trim(len(block[0]) - (accepted + 1))
            out.extend(committed)
            source_row = accepted
            last_hidden = hidden[:, source_row]
            if stop is not None:
                break
            bonus = choice[accepted]
            out.append(bonus)
        out = out[:max_tokens]
        receipt = {
            "route": "dfly2", "qualified": False, "num_draft": num_draft,
            "latent_kv": target.args.latent_kv,
            "approximate_state": target.args.latent_kv != "native",
            "verify_rounds": rounds, "verify_accept_hist": hist, "verify_depth_hist": depth_hist,
            "mean_acceptance_length": (1 + sum(i * n for i, n in enumerate(hist)) / rounds) if rounds else None,
        }
        return out, receipt
    finally:
        target.record_layers = previous_record
