# SPDX-License-Identifier: MIT
"""MLX model adapter for inclusionAI/Ling-3.0-flash.

Mined from TensorFold/Ling-3.0-flash-oQ4 at the revision recorded in
``provenance/ling3-ubtree.json`` (MIT).  Imports are mapped onto mlx2's
self-contained runtime and target taps are added for the external drafter.
"""

import math
import os
import threading
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
from mlx import nn

from ..prefill_checkpoint_rows import select_checkpoint_rows
from ..prefill_tail_skip import STATS, claim_adapter_tail, structural_refusal
from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_update
from ..technologies import linear_recurrence as _lr
from .rope_utils import initialize_rope
from . import ling_rowwise, moe_nax_gather
from .switch_layers import SwiGLU, SwitchGLU
from .xing4_0 import MultiLinear
from ..technologies.drafter_blocks import NEXTN_FUSION, fuse_mtp_module
from ..technologies import mla as _mla
from ..technologies.moe_router import cached_spec, route as _route_reference


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    n_group: int
    topk_group: int
    first_k_dense_replace: int
    layer_group_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    routed_scaling_factor: float
    head_dim: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    # Absent from Ling-3.2-air's text_config; the reference default is 1.
    num_shared_experts: int = 1
    short_conv_kernel_size: int = 4
    moe_shared_expert_intermediate_size: int | None = None
    q_lora_rank: int | None = None
    rope_interleave: bool = True
    rope_scaling: dict[str, float | str] | None = None
    use_qkv_bias: bool = False
    use_bias: bool = False
    norm_topk_prob: bool = True
    score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = True
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = 0
    no_kda_lora: bool = False
    kda_safe_gate: bool = False
    kda_lower_bound: float | None = None
    gated_attention_proj_granularity_type: str | None = None
    # Ling-3.2-air: per-layer clamped SwiGLU (0 = no clamp) and an fp32 router.
    expert_swiglu_limit_list: list[float] | None = None
    share_expert_swiglu_limit_list: list[float] | None = None
    router_dtype: str | None = None
    # Adapter opt-in, never read from a checkpoint config: the Ling-3.2-air
    # vendor laws (clamped SwiGLU, fp32 router and route weights, -inf group
    # masking). Ling 3 Flash's oQ4 config also lists router_dtype/limit lists
    # but was served (and qualified) without them, with a quantized router.
    vendor_v32_laws: bool = False
    # Ling-3.2-air VL: [T, H, W] rotary frequency sections of 3D M-RoPE (ling_mrope).
    mrope_section: list | None = None


def swiglu_limit(values: list[float] | None, layer_idx: int | None) -> float | None:
    """The vendor's ``_get_swiglu_limit``: out-of-range or 0 means no clamp."""
    if values is None or layer_idx is None or not 0 <= layer_idx < len(values):
        return None
    return float(values[layer_idx]) or None


def clamped_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    """``min(silu(gate), L) * clip(up, -L, L)``: clamp AFTER the activation.

    Not the GLM/gpt-oss pre-activation order (those clamp the gate logit);
    transformers #50574 and sglang #52593 apply exactly this law for Ling.
    """
    return mx.minimum(nn.silu(gate), limit) * mx.clip(up, -limit, limit)


class ClampedSwiGLU(nn.Module):
    """SwitchGLU activation with Ling's post-activation clamp."""

    def __init__(self, limit: float):
        super().__init__()
        self.swiglu_limit = float(limit)

    def __call__(self, x, gate):
        return clamped_swiglu(gate, x, self.swiglu_limit)


# MLX2_LING_PREFETCH_CACHE_OFFSETS=1 (default off; Pierre 2026-11-26: exact,
# A/B before any default).  Off, every MLA layer visit reads its size-1
# ``cache.offset`` with ``.item()`` (Phase 0: 8.8 host reads per Ling MTP d2
# round).  On, a trunk forward reads every global layer's size-1 integer
# offset in one ``tolist()`` at its start and each layer's first visit takes
# its own value from that table.  Exact: a layer's cache offset only moves in
# that layer's own ``update_and_fetch``, after its read, so the value at the
# forward's start is the value at the visit.  An entry is consumed on use, so
# a cache visited twice in one forward reads the live offset the second time.
# GPU A/B 2026-11-26 @8835a3a6e (Ling-3.2-air e4r8, MTP d2, one lane, no
# thermal hold): GSM8K100 token-exact vs off (acc 0.96 both); syncs/round
# 11.6 -> 5.8 (blocking 2.0 -> 3.0: the one tolist waits where the item()s
# overlapped); quiet 1.1K/17.7K 48.5/26.3 vs 43.5/24.7 tok/s, GSM8K decode
# median 58.3 vs 55.1 (qualification/runs/specround-hostsync-20261126/ling).
_PREFETCH_OFFSETS_ENV = "MLX2_LING_PREFETCH_CACHE_OFFSETS"
_offset_table = threading.local()
# Engagement counters (process-wide): forwards that prefetched, offsets read.
OFFSET_PREFETCH_STATS: Counter = Counter()


def _prefetchable_offset(cache):
    offset = getattr(cache, "offset", None) if cache is not None else None
    if (
        isinstance(offset, mx.array) and offset.size == 1
        and mx.issubdtype(offset.dtype, mx.integer)
    ):
        return offset
    return None


@contextmanager
def _prefetched_cache_offsets(layers, cache):
    """Read the global layers' size-1 cache offsets of one forward at once."""
    if (
        os.environ.get(_PREFETCH_OFFSETS_ENV, "0") != "1"
        or cache is None
        or getattr(_offset_table, "table", None) is not None
    ):
        yield
        return
    pending = []
    for layer, layer_cache in zip(layers, cache):
        if not getattr(layer, "is_global", False):
            continue
        offset = _prefetchable_offset(layer_cache)
        if offset is not None:
            pending.append((layer_cache, offset))
    # Shared offset objects could move under another layer's in-place
    # update; mixed dtypes would change ``.item()``'s Python type.  Both keep
    # the per-visit reads.
    if (
        len(pending) < 2
        or len({offset.dtype for _, offset in pending}) != 1
        or len({id(offset) for _, offset in pending}) != len(pending)
        or len({id(layer_cache) for layer_cache, _ in pending}) != len(pending)
    ):
        yield
        return
    values = mx.concatenate([offset.reshape(1) for _, offset in pending]).tolist()
    OFFSET_PREFETCH_STATS["forwards"] += 1
    OFFSET_PREFETCH_STATS["offsets"] += len(pending)
    _offset_table.table = {
        id(layer_cache): (layer_cache, value)
        for (layer_cache, _), value in zip(pending, values)
    }
    try:
        yield
    finally:
        _offset_table.table = None


def _cache_offset(cache: Any | None) -> int | mx.array:
    table = getattr(_offset_table, "table", None)
    if table and cache is not None:
        entry = table.pop(id(cache), None)
        if entry is not None and entry[0] is cache:
            return entry[1]
    offset = cache.offset if cache is not None else 0
    if isinstance(offset, mx.array):
        if offset.size == 1:
            return offset.item()
        return mx.array(offset.tolist(), dtype=offset.dtype)
    return offset


class MLP(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        intermediate_size: int | None = None,
        swiglu_limit: float | None = None,
    ):
        super().__init__()
        hidden = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(args.hidden_size, hidden, bias=args.use_bias)
        self.up_proj = nn.Linear(args.hidden_size, hidden, bias=args.use_bias)
        self.down_proj = nn.Linear(hidden, args.hidden_size, bias=args.use_bias)
        # Plain attribute (not a parameter): fused MLP paths must check it.
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = self.gate_proj(x), self.up_proj(x)
        if self.swiglu_limit is None:
            return self.down_proj(swiglu(gate, up))
        return self.down_proj(clamped_swiglu(gate, up, self.swiglu_limit))


class MultiLatentAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.q_lora_rank = args.q_lora_rank
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.v_head_dim = args.v_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.gate_type = args.gated_attention_proj_granularity_type
        self.scale = self.qk_head_dim**-0.5
        self.ling_fused_mla = False
        self.ling_absorbed_verify_rows = False
        # Diagnostic per-row verify law (ling_rowwise, component ``mla``). Off.
        self.ling_rowwise = False
        self.ling_calls = Counter()

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                args.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = nn.Linear(
                args.hidden_size,
                self.q_lora_rank,
                bias=args.use_qkv_bias,
            )
            self.q_a_layernorm = nn.RMSNorm(
                self.q_lora_rank,
                eps=args.rms_norm_eps,
            )
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=args.use_qkv_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(
            self.kv_lora_rank,
            eps=args.rms_norm_eps,
        )
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim,
            self.kv_lora_rank,
            self.num_heads,
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank,
            self.v_head_dim,
            self.num_heads,
        )

        if self.gate_type is None:
            self.g_proj = None
        elif self.gate_type == "head_wise":
            self.g_proj = nn.Linear(args.hidden_size, self.num_heads, bias=False)
        elif self.gate_type == "element_wise":
            self.g_proj = nn.Linear(
                args.hidden_size,
                self.num_heads * self.v_head_dim,
                bias=False,
            )
        else:
            raise ValueError(f"Unsupported attention gate type: {self.gate_type}")

        self.dense = nn.Linear(
            self.num_heads * self.v_head_dim,
            args.hidden_size,
            bias=args.use_qkv_bias,
        )

        # yarn mscale whenever mscale_all_dim and factor > 1, applied as
        # scale *= mscale * mscale (mscale_rule/mscale_order of the spec).
        self.mla_spec = _mla.MLASpec.from_config(
            args, mscale_rule="any_rope_type", mscale_order="squared",
            absorbed_rule="single_row", absorbed_layout="broadcast",
            pe_form="score_bias", sdpa_route="base",
        )
        self.scale = self.mla_spec.softmax_scale

        self.rope_base = float(args.rope_theta)
        # 3D M-RoPE here is the unscaled pair-interleaved law only (no YaRN).
        self.mrope_section = (tuple(args.mrope_section)
                              if args.mrope_section is not None and args.rope_interleave
                              and args.rope_scaling is None else None)
        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=args.rope_interleave,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
        *,
        tree=None,
        rope_positions=None,
    ) -> mx.array:
        batch, length, _ = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(batch, length, self.num_heads, self.qk_head_dim)
        q = q.transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(
            compressed_kv,
            [self.kv_lora_rank],
            axis=-1,
        )
        k_pe = k_pe.reshape(batch, length, 1, self.qk_rope_head_dim)
        k_pe = k_pe.transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = _cache_offset(cache)
        if rope_positions is None and tree is None and self._rowwise_rows(length):
            q_pe = self._rope_rows(q_pe, offset)
            k_pe = self._rope_rows(k_pe, offset)
        elif rope_positions is None:
            q_pe = self.rope(q_pe, offset=offset) if tree is None else tree.rope(self.rope, q_pe, offset)
            k_pe = self.rope(k_pe, offset=offset) if tree is None else tree.rope(self.rope, k_pe, offset)
        else:
            q_pe, k_pe = self._rope_at(q_pe, k_pe, rope_positions, tree)

        kv_latent = mx.expand_dims(kv_latent, axis=1)
        segmented = tree is None and cache is not None and hasattr(cache, "row_views")
        if cache is not None and not segmented:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        if segmented:
            # Segmented batch (true-batched self-MTP / multi-lane verify): the
            # cache appends each lane's rows to that lane's own history and
            # returns no dense K/V; every lane attends its own latent through
            # the same branch a width-one call of its valid length takes.
            output = self._attend_segmented_rows(q_nope, q_pe, kv_latent, k_pe, mask, cache)
        elif tree is not None:
            output = tree.mla(self, q_nope, q_pe, kv_latent, k_pe, cache, int(offset))
        else:
            output = self._attend_latent(q_nope, q_pe, kv_latent, k_pe, mask, cache)

        output = output.transpose(0, 2, 1, 3)
        if self.g_proj is not None:
            gate = mx.sigmoid(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
            if self.gate_type == "head_wise":
                output = output * gate[..., None]
            else:
                output = output * gate.reshape(
                    batch,
                    length,
                    self.num_heads,
                    self.v_head_dim,
                )

        output = output.reshape(batch, length, -1)
        return self.dense(output)

    def _rope_at(self, q_pe, k_pe, positions, tree):
        """Request-owned positions (Ling-3.2-air vision mode, ``ling_mrope``).

        An ``int`` or per-row ``mx.array`` replaces the cache offset (text after an
        image: ``slot + delta``); a ``[3, L]`` array is 3D M-RoPE for one row.
        """
        if tree is not None:
            raise ValueError("Ling tree verification does not take request-owned RoPE positions")
        if isinstance(positions, np.ndarray):
            if q_pe.shape[0] != 1:
                raise ValueError("Ling 3D M-RoPE positions are one request's rows")
            from .ling_mrope import mrope_rotate

            base, section = self.rope_base, self.mrope_section
            if section is None:
                raise ValueError("this Ling artifact has no supported 3D M-RoPE law "
                                 "(needs mrope_section, rope_interleave and no rope_scaling)")
            if positions.ndim != 2 or positions.shape != (3, q_pe.shape[-2]):
                raise ValueError("Ling 3D M-RoPE positions must be [3, L]")
            # Runs whose three planes are equal and consecutive are 1D RoPE: they
            # take fast.rope at their offset, bitwise the text path. Only image
            # rows (unequal planes) use the 3D rotation.
            equal = (positions[0] == positions[1]) & (positions[1] == positions[2])
            qs, ks, start, length = [], [], 0, positions.shape[1]
            while start < length:
                stop = start + 1
                if equal[start]:
                    while (stop < length and equal[stop]
                           and positions[0, stop] == positions[0, stop - 1] + 1):
                        stop += 1
                    offset = int(positions[0, start])
                    qs.append(self.rope(q_pe[..., start:stop, :], offset=offset))
                    ks.append(self.rope(k_pe[..., start:stop, :], offset=offset))
                else:
                    while stop < length and not equal[stop]:
                        stop += 1
                    rows = positions[:, start:stop]
                    qs.append(mrope_rotate(q_pe[..., start:stop, :], rows, base=base, section=section))
                    ks.append(mrope_rotate(k_pe[..., start:stop, :], rows, base=base, section=section))
                start = stop
            if len(qs) == 1:
                return qs[0], ks[0]
            return mx.concatenate(qs, axis=-2), mx.concatenate(ks, axis=-2)
        return self.rope(q_pe, offset=positions), self.rope(k_pe, offset=positions)

    def _rowwise_rows(self, length):
        return (self.ling_rowwise and 1 < length <= ling_rowwise.MAX_ROWS
                and ling_rowwise.active())

    def _rope_rows(self, x, offset):
        return mx.concatenate(
            [self.rope(x[:, :, i:i + 1], offset=offset + i) for i in range(x.shape[2])], axis=2)

    def _rowwise_decline(self, reason):
        self.ling_calls["mla_rowwise_declined"] += 1
        self.ling_calls["mla_rowwise_declined:" + reason] += 1
        return None

    def _attend_rows(self, q_nope, q_pe, kv_latent, k_pe, mask, cache):
        """Per-row verify law: row ``i`` attends its own causal key prefix
        through the one-row (decode) branch.  None declines (counted)."""
        length, keys = q_nope.shape[2], kv_latent.shape[-2]
        if isinstance(mask, str):
            if mask != "causal":
                return self._rowwise_decline("mask kind")
            ends = [keys - length + i + 1 for i in range(length)]
        elif mask is None:
            return self._rowwise_decline("no mask")
        else:
            rows = np.array(mask.astype(mx.bool_))
            rows = rows.reshape(-1, rows.shape[-2], rows.shape[-1])
            if rows.shape[1:] != (length, keys) or not (rows == rows[:1]).all():
                return self._rowwise_decline("mask shape")
            ends = []
            for row in rows[0]:
                end = int(row.sum())
                if end == 0 or not row[:end].all():
                    return self._rowwise_decline("non-prefix mask")
                ends.append(end)
        pieces = [
            self._attend_latent(q_nope[:, :, i:i + 1], q_pe[:, :, i:i + 1],
                                kv_latent[..., :end, :], k_pe[..., :end, :], None, cache)
            for i, end in enumerate(ends)
        ]
        self.ling_calls["mla_rowwise_rows"] += length
        return mx.concatenate(pieces, axis=2)

    def _attend_latent(self, q_nope, q_pe, kv_latent, k_pe, mask, cache):
        """Reference latent attention over dense history -> [B, heads, L, v].

        Kimi's split kernel consumes the shared channels after position
        encoding. Ling supplies rotated Q/K here, preserving its RoPE law.
        """
        length = q_nope.shape[2]
        if self._rowwise_rows(length):
            output = self._attend_rows(q_nope, q_pe, kv_latent, k_pe, mask, cache)
            if output is not None:
                return output
        if length == 1:
            fused_output = self.fused_latent_output(q_nope, q_pe, kv_latent, k_pe, mask)
            if fused_output is not None:
                return fused_output
        # Opt-in attribution law (ling_optimizations.absorbed_verify_rows): short
        # multi-row forwards (speculative verify windows, 2-16 rows) take the
        # one-row absorbed formula instead of the expanded one, so each verify
        # row follows the decode step's arithmetic order.
        absorbed = length == 1 or (self.ling_absorbed_verify_rows and length <= 16)
        output = _mla.score_bias_attend(
            self.mla_spec, q_nope, q_pe, kv_latent, k_pe, mask,
            embed_q=self.embed_q, unembed_out=self.unembed_out, absorbed=absorbed,
            scale=self.scale, cache=cache,
        )
        if absorbed and length > 1:
            self.ling_calls["mla_absorbed_verify_rows"] += 1
        self.ling_calls["mla_reference"] += 1
        return output

    def _attend_segmented_rows(self, q_nope, q_pe, kv_latent, k_pe, mask, cache):
        from ..segmented_plain_kv import SegmentedBatchKVCache

        # Refuse before the append, so an unsupported cache is left untouched.
        if type(cache) is not SegmentedBatchKVCache:
            raise NotImplementedError("segmented Ling MLA supports plain latent rows only")
        if mask is not None and not isinstance(mask, mx.array):
            raise ValueError("segmented Ling MLA requires an explicit array mask")
        cache.update_and_fetch(kv_latent, k_pe)
        batch, heads, length, _ = q_nope.shape
        pieces = []
        for index, valid, row_latent, row_pe, row_mask in cache.row_views(mask):
            if not valid:
                pieces.append(mx.zeros((1, heads, length, self.v_head_dim), dtype=q_nope.dtype))
                continue
            # A one-query row's slice of a plain segmented mask is all True:
            # pass None, as KVCache.make_mask does at width one, so the fused
            # decode kernel admits the same rows it would there.
            piece = self._attend_latent(
                q_nope[index:index + 1, :, :valid], q_pe[index:index + 1, :, :valid],
                row_latent, row_pe, None if valid == 1 else row_mask, None,
            )
            if valid < length:
                piece = mx.pad(piece, [(0, 0), (0, 0), (0, length - valid), (0, 0)])
            pieces.append(piece)
        cache.note_attention()
        self.ling_calls["mla_segmented_rows"] += batch
        return mx.concatenate(pieces, axis=0)

    def tail(self, x: mx.array, mask: mx.array | None = None, cache: KVCache | None = None):
        """Prompt-chunk tail: the latent/positional cache write over every row,
        the query and attention output for the last row only.

        The cache write (``kv_a_proj_with_mqa``, ``kv_a_layernorm``, the key
        RoPE and ``update_and_fetch``) is the full forward's, so the cache is
        bitwise identical.  The single query attends through the absorbed
        latent form the decode step uses (``embed_q`` on the query,
        ``unembed_out`` on the output) instead of expanding every cached
        latent into per-head keys and values.
        """
        batch, length, _ = x.shape
        last = x[:, -1:]
        if self.q_lora_rank is None:
            q = self.q_proj(last)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(last)))
        q = q.reshape(batch, 1, self.num_heads, self.qk_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(batch, length, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = _cache_offset(cache)
        q_pe = self.rope(q_pe, offset=offset + (length - 1))
        k_pe = self.rope(k_pe, offset=offset)
        kv_latent = mx.expand_dims(kv_latent, axis=1)
        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        last_mask = mask[..., -1:, :] if mask is not None and not isinstance(mask, str) else None
        output = _mla.absorbed_attention(
            self.mla_spec, self.embed_q(q_nope), q_pe, kv_latent, k_pe, last_mask,
            scale=self.scale, cache=cache,
        )
        output = self.unembed_out(output).transpose(0, 2, 1, 3)
        self.ling_calls["mla_tail"] += 1
        if self.g_proj is not None:
            gate = mx.sigmoid(self.g_proj(last).astype(mx.float32)).astype(output.dtype)
            if self.gate_type == "head_wise":
                output = output * gate[..., None]
            else:
                output = output * gate.reshape(batch, 1, self.num_heads, self.v_head_dim)
        return self.dense(output.reshape(batch, 1, -1))

    def fused_latent_output(self, q_nope, q_pe, latent, positional, mask=None):
        if not self.ling_fused_mla or self.training or self.scale != 192**-0.5:
            return None
        from .kimi_linear_mla_fused import admits, attend

        if q_nope.shape != (1, 32, 1, 128) or mask is not None:
            return None
        absorbed = self.embed_q(q_nope)
        if not admits(absorbed, q_pe, latent, positional, mask):
            return None
        self.ling_calls["mla_fused_decode"] += 1
        return self.unembed_out(attend(absorbed, q_pe, latent, positional).astype(q_nope.dtype))


class ShortConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            bias=False,
            groups=channels,
            padding=0,
        )

    def __call__(
        self,
        x: mx.array,
        state: mx.array | None,
        mask: mx.array | None,
        lengths: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        return _lr.short_conv(x, state, mask, lengths, self.conv, kernel_size=self.kernel_size)


class KimiDeltaAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        # Structural recurrence contract shared by the explicit chunk32
        # installer; vector gates retain their own per-channel decay math.
        self.num_k_heads = self.num_v_heads = self.num_heads
        self.head_k_dim = self.head_v_dim = self.head_dim
        self.conv_kernel = args.short_conv_kernel_size
        self.projection_dim = self.num_heads * self.head_dim
        self.no_kda_lora = args.no_kda_lora
        # The released recurrent reference passes this bound into the gate.
        # Omitting it changes the recurrence even though every shape matches.
        self.lower_bound = args.kda_lower_bound
        self.scale = float(self.head_dim) ** -0.5
        # Ling's KDA laws: rms q/k with eps 1e-6 on the MEAN, the bounded
        # sigmoid gate (unbounded softplus when the checkpoint has no bound).
        self.recurrence_spec = _lr.LinearRecurrenceSpec.from_config(
            args, qk_norm="rms_mean", core_scan=True)
        self.ling_merged_conv = False
        self.ling_fused_norm = False
        self.ling_kda_fused = False
        # Diagnostic per-row verify law (ling_rowwise, component ``kda``). Off.
        self.ling_rowwise = False
        self.ling_calls = Counter()

        hidden = args.hidden_size
        self.q_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.q_conv1d = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.k_conv1d = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.v_conv1d = ShortConv1d(self.projection_dim, self.conv_kernel)

        if self.no_kda_lora:
            self.f_proj = nn.Linear(hidden, self.projection_dim, bias=False)
            self.g_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        else:
            self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
            self.f_b_proj = nn.Linear(
                self.head_dim,
                self.projection_dim,
                bias=False,
            )
            self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(
                self.head_dim,
                self.projection_dim,
                bias=False,
            )

        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.A_log = mx.log(
            mx.random.uniform(low=1.0, high=16.0, shape=(self.num_heads,))
        )
        self.dt_bias = mx.zeros((self.projection_dim,))
        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: ArraysCache | None = None,
        *,
        tree=None,
    ) -> mx.array:
        if tree is not None:
            return tree.kda(self, x, cache)
        batch, length, _ = x.shape
        dtype = x.dtype

        if cache is not None:
            q_state, k_state, v_state, recurrent_state = cache
            lengths = cache.lengths
        else:
            q_state = k_state = v_state = recurrent_state = None
            lengths = None

        if q_state is None:
            state = mx.zeros(
                (batch, self.conv_kernel - 1, self.projection_dim),
                dtype=dtype,
            )
            q_state = k_state = v_state = state

        spans = None
        if cache is not None and cache.speculating:
            spans = cache.rollback_spans(length, mask)
        snapshot = [q_state, k_state, v_state, recurrent_state]
        raw = (self.q_proj(x), self.k_proj(x), self.v_proj(x))
        if self.ling_kda_fused and not self.training:
            fused = self.fused_decode(x, raw, snapshot, mask, cache, spans)
            if fused is not None:
                return fused
        rowwise = (self.ling_rowwise and 1 < length <= ling_rowwise.MAX_ROWS
                   and ling_rowwise.active())
        if rowwise:
            (q, k, v), (q_state, k_state, v_state) = self._convolve_rows(
                raw, (q_state, k_state, v_state), mask, lengths,
            )
        else:
            (q, k, v), (q_state, k_state, v_state) = self.convolve_qkv(
                raw, (q_state, k_state, v_state), mask, lengths,
            )

        if cache is not None:
            cache[0] = q_state
            cache[1] = k_state
            cache[2] = v_state

        q = q.reshape(batch, length, self.num_heads, self.head_dim)
        k = k.reshape(batch, length, self.num_heads, self.head_dim)
        v = v.reshape(batch, length, self.num_heads, self.head_dim)

        if rowwise:
            # The CPU rms_norm is not row-invariant at every width; keep the
            # one-row reduction for the per-row law.
            q = (self.scale**2) * ling_rowwise.rows(
                lambda row: mx.fast.rms_norm(row, None, 1e-6), q, axis=1)
            k = self.scale * ling_rowwise.rows(
                lambda row: mx.fast.rms_norm(row, None, 1e-6), k, axis=1)
        else:
            q, k = _lr.normalize_qk(self.recurrence_spec, q, k)

        if self.no_kda_lora:
            decay_logits = self.f_proj(x)
        else:
            decay_logits = self.f_b_proj(self.f_a_proj(x))
        decay_logits = decay_logits.reshape(
            batch,
            length,
            self.num_heads,
            self.head_dim,
        )
        beta_logits = self.b_proj(x).reshape(batch, length, self.num_heads)

        if spans is not None:
            self._record_rollback(
                cache, length, spans, snapshot, raw, mask,
                (q, k, v, decay_logits, beta_logits), rowwise=rowwise,
            )

        if rowwise:
            output, recurrent_state = self._scan_rows(
                (q, k, v, decay_logits, beta_logits), recurrent_state, mask,
            )
            self.ling_calls["kda_rowwise_rows"] += length
        else:
            output, recurrent_state = self._gated_delta_update(
                q,
                k,
                v,
                decay_logits,
                beta_logits,
                self.A_log.reshape(self.num_heads, 1),
                self.dt_bias.reshape(self.num_heads, self.head_dim),
                state=recurrent_state,
                mask=mask,
                use_kernel=not self.training,
                lower_bound=self.lower_bound,
                prefill_chunk_size=(getattr(self, "_prefill_scan_chunk", 0)
                    if length >= 17 and not bool(getattr(cache, "speculating", False)) else 0),
                prefill_stats=getattr(self, "_prefill_scan_stats", None),
            )

        if cache is not None:
            cache[3] = recurrent_state
            cache.advance(length)

        if self.no_kda_lora:
            gate = self.g_proj(x)
        else:
            gate = self.g_b_proj(self.g_a_proj(x))
        gate = gate.reshape(batch, length, self.num_heads, self.head_dim)
        output = self.output_norm_gate(output, gate)
        return self.o_proj(output.reshape(batch, length, -1))

    def _convolve_rows(self, raw, states, mask, lengths):
        """Per-row verify law: each row through the one-row convolution."""
        outputs = ([], [], [])
        for i in range(raw[0].shape[1]):
            row_mask = None if mask is None else mask[:, i:i + 1]
            row_lengths = None if lengths is None else mx.clip(lengths - i, 0, 1)
            values, states = self.convolve_qkv(
                tuple(value[:, i:i + 1] for value in raw), states, row_mask, row_lengths,
            )
            for bucket, value in zip(outputs, values):
                bucket.append(value)
        return tuple(mx.concatenate(bucket, axis=1) for bucket in outputs), states

    def _scan_rows(self, scan, state, mask):
        """Per-row verify law: the gated-delta scan one row per call."""
        a_log = self.A_log.reshape(self.num_heads, 1)
        dt_bias = self.dt_bias.reshape(self.num_heads, self.head_dim)
        outputs = []
        for i in range(scan[0].shape[1]):
            output, state = self._gated_delta_update(
                *(value[:, i:i + 1] for value in scan), a_log, dt_bias,
                state=state, mask=None if mask is None else mask[:, i:i + 1],
                use_kernel=not self.training, lower_bound=self.lower_bound,
                prefill_chunk_size=0,
                prefill_stats=getattr(self, "_prefill_scan_stats", None),
            )
            outputs.append(output)
        return mx.concatenate(outputs, axis=1), state

    def _kda_fused_decline(self, reason):
        self.ling_calls["kda_fused_declined"] += 1
        self.ling_calls["kda_fused_declined:" + reason] += 1
        return None

    def fused_decode(self, x, raw, snapshot, mask, cache, spans):
        """One-launch B1 decode body (``ling_kda_fused.decode_v1``) or None.

        Declines (counted per reason) whenever the eager chain must run:
        speculation (the APCv2 rollback recorder replays the eager scan
        operands), masks, ragged lengths, batch or prompt rows, the unbounded
        gate law and any operand the kernel does not reproduce bit for bit.
        """
        from . import differentiable
        from .ling_kda_fused import ling_kda_fused_decode, ling_kda_fused_refusal

        if differentiable.routed("kda_fused"):
            # A Metal launch has no VJP; reference_path() promises pure MLX ops.
            return self._kda_fused_decline("differentiable reference")
        if cache is None:
            return self._kda_fused_decline("no cache")
        if x.shape[:2] != (1, 1):
            return self._kda_fused_decline("not B1 one-token decode")
        if mask is not None:
            return self._kda_fused_decline("mask")
        if cache.speculating or spans is not None:
            return self._kda_fused_decline("speculating")
        if cache.lengths is not None:
            return self._kda_fused_decline("ragged lengths")
        if self.lower_bound is None:
            return self._kda_fused_decline("unbounded gate law")
        if self.conv_kernel != 4:
            return self._kda_fused_decline("conv kernel size")
        state = snapshot[3]
        if state is None:
            state = mx.zeros((1, self.num_heads, self.head_dim, self.head_dim), dtype=mx.float32)
        if self.no_kda_lora:
            decay, gate = self.f_proj(x), self.g_proj(x)
        else:
            decay = self.f_b_proj(self.f_a_proj(x))
            gate = self.g_b_proj(self.g_a_proj(x))
        operands = dict(
            raw=raw, convs=tuple(snapshot[:3]),
            weights=(self.q_conv1d.conv.weight, self.k_conv1d.conv.weight,
                     self.v_conv1d.conv.weight),
            a=decay, b=self.b_proj(x), A_log=self.A_log, dt_bias=self.dt_bias,
            state=state, gate=gate, norm=self.o_norm.weight,
        )
        refusal = ling_kda_fused_refusal(**operands)
        if refusal is not None:
            return self._kda_fused_decline(refusal)
        output, convs, state = ling_kda_fused_decode(
            **operands, rms_eps=self.o_norm.eps, lower_bound=self.lower_bound)
        cache[0], cache[1], cache[2] = convs
        cache[3] = state
        cache.advance(1)
        self.ling_calls["kda_fused"] += 1
        return self.o_proj(output)

    def convolve_qkv(self, raw, states, mask=None, lengths=None):
        # The real layer0 A/B won at one row and lost at 32/128 prompt rows.
        # Keep prompt convolutions on the ordinary path. Tree ancestry windows
        # arrive as independent batch rows with sequence length one.
        merge = (self.ling_merged_conv and raw[0].shape[1] == 1
                 and mask is None and lengths is None and not self.training)
        if merge:
            weights = (self.q_conv1d.conv.weight, self.k_conv1d.conv.weight, self.v_conv1d.conv.weight)
            # A channel stack must not promote otherwise independent branches.
            # The loader preserves per-tensor dtypes, including mixed artifacts.
            merge = all(value.dtype == raw[0].dtype for value in (*raw, *weights)) and all(
                state is None or state.dtype == raw[0].dtype for state in states
            )
        if merge:
            inputs = mx.concatenate(raw, axis=-1)
            history = mx.concatenate([
                mx.zeros((value.shape[0], self.conv_kernel-1, value.shape[-1]), dtype=value.dtype)
                if state is None else state for value, state in zip(raw, states, strict=True)
            ], axis=-1)
            full = mx.concatenate([history, inputs], axis=1)
            cached = getattr(self, "_ling_conv_weights", None)
            if cached is None or any(a is not b for a, b in zip(cached[0], weights, strict=True)):
                cached = (weights, mx.concatenate(weights, axis=0))
                object.__setattr__(self, "_ling_conv_weights", cached)
            weight = cached[1]
            result = nn.silu(mx.conv1d(full, weight, groups=3*self.projection_dim))
            tails = mx.contiguous(full[:, -(self.conv_kernel-1):])
            self.ling_calls["merged_conv"] += 1
            return mx.split(result, 3, axis=-1), mx.split(tails, 3, axis=-1)
        self.ling_calls["separate_conv"] += 1
        values = [module(value, state, mask, lengths) for module, value, state in
                  zip((self.q_conv1d, self.k_conv1d, self.v_conv1d), raw, states, strict=True)]
        return tuple(item[0] for item in values), tuple(item[1] for item in values)

    def output_norm_gate(self, output, gate):
        if self.ling_fused_norm and not self.training and not _lr.kernel_refused("kda_fused"):
            from .ling_fused import norm_gate

            result = norm_gate(output, gate, self.o_norm.weight, self.o_norm.eps)
            if result is not None:
                self.ling_calls["fused_norm"] += 1
                return result
            self.ling_calls["fused_norm_declined"] += 1
        self.ling_calls["eager_norm"] += 1
        return self.o_norm(output) * mx.sigmoid(gate)

    def _gated_delta_update(self, *args, **kwargs):
        return gated_delta_update(*args, **kwargs)

    def _record_rollback(self, cache, length, spans, snapshot, raw, mask, scan, *, rowwise=False):
        """Retain all three conv histories and the accepted KDA successor.

        Speculative forwards disable reordered chunk scans. Replay therefore
        uses exactly the forward's ordinary recurrence and bounded gate law.
        The cache owns per-row valid depths, including right-padded slabs.
        """
        keep = self.conv_kernel - 1
        padded = [
            mx.concatenate([
                history, value if mask is None else mx.where(mask[..., None], value, 0)
            ], axis=1)
            for history, value in zip(snapshot[:3], raw)
        ]
        use_kernel = not self.training
        a_log = self.A_log.reshape(self.num_heads, 1)
        dt_bias = self.dt_bias.reshape(self.num_heads, self.head_dim)
        lower_bound = self.lower_bound

        def rollback(m):
            if rowwise and m > 0:
                _, state = self._scan_rows(
                    tuple(value[:, :m] for value in scan), snapshot[3],
                    None if mask is None else mask[:, :m],
                )
                return [*[mx.contiguous(value[:, m:m + keep]) for value in padded], state]
            _, state = self._gated_delta_update(
                *(value[:, :m] for value in scan), a_log, dt_bias,
                state=snapshot[3],
                mask=None if mask is None else mask[:, :m],
                use_kernel=use_kernel, lower_bound=lower_bound,
            )
            return [*[mx.contiguous(value[:, m:m + keep]) for value in padded], state]

        cache.record_rollback(length, rollback, snapshot, depths=spans or None)


@mx.compile
def group_expert_select(
    gates: mx.array,
    expert_bias: mx.array | None,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    score_function: str,
    group_mask_neg_inf: bool = False,
) -> tuple[mx.array, mx.array]:
    if score_function not in ("sigmoid", "softmax"):
        raise ValueError(f"Unsupported MoE score function: {score_function}")
    # Vendor masks dropped groups with -inf: with a negative expert_bias a
    # kept expert can score below 0, so a 0.0 fill could pick a dropped one.
    # 0.0 stays the legacy (non-vendor) law: RouterSpec.group_fill.
    spec = cached_spec(
        top_k=top_k, score_func=score_function, score_dtype="float32",
        use_bias=expert_bias is not None, n_group=n_group, topk_group=topk_group,
        group_fill=-math.inf if group_mask_neg_inf else 0.0,
        topk_method="argpartition_neg", norm_topk_prob=bool(norm_topk_prob),
        norm_eps=1e-20, norm_min_top_k=2,
        routed_scaling_factor=float(routed_scaling_factor),
    )
    return _route_reference(gates, spec, expert_bias, out_dtype=gates.dtype)


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling_factor = args.routed_scaling_factor
        self.score_function = args.score_function
        self.gate_proj = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        # sglang #44715/#38719: a bf16 gate GEMM perturbs top-k; Ling-3.2-air
        # declares router_dtype fp32 and keeps mlp.gate out of quantization.
        self.router_fp32 = args.vendor_v32_laws and args.router_dtype == "fp32"
        self.group_mask_neg_inf = args.vendor_v32_laws
        self.expert_bias = (
            mx.zeros((args.num_experts,), dtype=mx.float32)
            if args.moe_router_enable_expert_bias
            else None
        )

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        if self.router_fp32:
            if "scales" in self.gate_proj:
                raise ValueError("an fp32 Ling router must not be quantized")
            logits = x.astype(mx.float32) @ self.gate_proj.weight.astype(mx.float32).T
        else:
            logits = self.gate_proj(x)
        indices, weights = group_expert_select(
            logits,
            self.expert_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
            self.score_function,
            self.group_mask_neg_inf,
        )
        # Vendor: expert outputs are scaled by the fp32 topk_weight, then cast.
        return indices, weights if self.router_fp32 else weights.astype(x.dtype)


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int | None = None):
        super().__init__()
        laws = args.vendor_v32_laws
        self.fp32_route_weights = laws
        self.expert_swiglu_limit = (
            swiglu_limit(args.expert_swiglu_limit_list, layer_idx) if laws else None
        )
        shared_limit = swiglu_limit(args.share_expert_swiglu_limit_list, layer_idx) if laws else None
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            activation=(
                ClampedSwiGLU(self.expert_swiglu_limit)
                if self.expert_swiglu_limit is not None else SwiGLU()
            ),
            bias=args.use_bias,
        )
        self.gate = Gate(args)
        self.ling_routed_decode = False
        # Diagnostic per-row verify law (ling_rowwise, component ``moe``). Off.
        self.ling_rowwise = False
        shared_size = (
            args.moe_shared_expert_intermediate_size or args.moe_intermediate_size
        ) * args.num_shared_experts
        self.shared_experts = (
            MLP(args, intermediate_size=shared_size, swiglu_limit=shared_limit)
            if args.num_shared_experts > 0
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        if (self.ling_rowwise and 1 < x.shape[-2] <= ling_rowwise.MAX_ROWS
                and ling_rowwise.active()):
            ling_rowwise.STATS["moe_rows"] += x.shape[-2]
            return ling_rowwise.rows(self._forward, x)
        return self._forward(x)

    def _forward(self, x: mx.array) -> mx.array:
        indices, weights = self.gate(x)
        output = None
        if self.ling_routed_decode:
            # Opt-in (ling_optimizations.routed_moe_decode): one-token decode
            # through the device-routed gate_up/down kernels; None declines.
            from . import ling_routed_decode

            output = ling_routed_decode.switch_glu(self.switch_mlp, x, indices)
        if output is None:
            output = self.switch_mlp(x, indices)
        output = (output * weights[..., None]).sum(axis=-2)
        if self.fp32_route_weights:
            output = output.astype(x.dtype)
        if self.shared_experts is not None:
            output = output + self.shared_experts(x)
        return output


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        n_layers = args.num_hidden_layers
        group_size = args.layer_group_size
        self.is_global = (layer_idx + 1) % group_size == 0 or layer_idx >= (
            n_layers // group_size
        ) * group_size
        self.attention = (
            MultiLatentAttention(args) if self.is_global else KimiDeltaAttention(args)
        )
        self.mlp = (
            SparseMoeBlock(args, layer_idx)
            if args.num_experts is not None and layer_idx >= args.first_k_dense_replace
            else MLP(args)
        )
        self.input_layernorm = nn.RMSNorm(
            args.hidden_size,
            eps=args.rms_norm_eps,
        )
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size,
            eps=args.rms_norm_eps,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
        *,
        tree=None,
        rope_positions=None,
    ) -> mx.array:
        if rope_positions is not None and self.is_global:
            attention_output = self.attention(
                self.input_layernorm(x), mask, cache, tree=tree, rope_positions=rope_positions,
            )
        else:
            attention_output = self.attention(
                self.input_layernorm(x),
                mask,
                cache,
                tree=tree,
            )
        hidden = x + attention_output
        mlp_output = self.mlp(self.post_attention_layernorm(hidden))
        return hidden + mlp_output

    def tail(self, x: mx.array, mask: mx.array | None = None, cache: Any | None = None):
        """Prompt-chunk tail (MLX2_PREFILL_TAIL_SKIP=1): one output row per lane."""
        hidden = x[:, -1:] + self.attention.tail(self.input_layernorm(x), mask, cache)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class LanguageModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx) for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._attention_idx = next(
            (i for i, layer in enumerate(self.layers) if layer.is_global),
            0,
        )
        self._kda_idx = next(
            (i for i, layer in enumerate(self.layers) if not layer.is_global),
            0,
        )

    @property
    def embed_tokens(self):
        """Common target-embedding seam used by external drafters."""

        return self.word_embeddings

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        *,
        tail_row: bool = False,
        inputs_embeds: mx.array | None = None,
        rope_positions=None,
    ) -> mx.array:
        # Admits the NAX sorted gather for prompt chunks when the adapter
        # claimed it (bit-identical to stock sorted gather_qmm); a no-op
        # under claim_stock_moe and for decode rows.
        shape = inputs if inputs_embeds is None else inputs_embeds
        with moe_nax_gather.forward_scope(inputs, cache), ling_rowwise.scope(shape, cache), \
                _prefetched_cache_offsets(self.layers, cache):
            return self._forward(inputs, cache, tail_row=tail_row,
                                 inputs_embeds=inputs_embeds, rope_positions=rope_positions)

    def _forward(self, inputs, cache=None, *, tail_row=False, inputs_embeds=None,
                 rope_positions=None):
        """``inputs_embeds`` / ``rope_positions``: Ling-3.2-air vision mode only
        (image features scattered into the embeddings; request-owned RoPE)."""
        if (inputs_embeds is not None or rope_positions is not None) and tail_row:
            raise ValueError("Ling vision-mode rows do not take the prompt tail path")
        hidden = self.word_embeddings(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)

        attention_mask = create_attention_mask(
            hidden,
            cache[self._attention_idx],
            return_array=True,
        )
        kda_mask = create_ssm_mask(hidden, cache[self._kda_idx])

        last = len(self.layers) - 1
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            mask = attention_mask if layer.is_global else kda_mask
            if tail_row and index == last:
                hidden = layer.tail(hidden, mask=mask, cache=layer_cache)
            elif rope_positions is not None:
                hidden = layer(hidden, mask=mask, cache=layer_cache, rope_positions=rope_positions)
            else:
                hidden = layer(hidden, mask=mask, cache=layer_cache)
        return self.norm(hidden)


class MTPLayer(nn.Module):
    """Ling NEXTN layer: ``final_layernorm(block(eh_proj([enorm(e); hnorm(h)])))``.

    The vendor feeds the trunk's POST-``model.norm`` hidden (not the pre-norm
    residual DeepSeek uses) and the embedding of the following token; the
    block is gated MLA (``mtp_use_kda: false``) plus the routed MoE, and the
    head is the shared ``lm_head``.  Its MLA cache holds pair (h_p, t_{p+1})
    at offset p, the vendor's position_ids for that pair.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.is_global = True
        eps = args.rms_norm_eps
        self.enorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.hnorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.eh_proj = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.attention = MultiLatentAttention(args)
        # Index ``num_hidden_layers`` is past the limit lists: no clamp (vendor).
        self.mlp = SparseMoeBlock(args, args.num_hidden_layers)
        self.final_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)

    def __call__(self, embeds, hidden, mask=None, cache=None):
        x = fuse_mtp_module(NEXTN_FUSION, self, embeds, hidden)
        x = x + self.attention(self.input_layernorm(x), mask, cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return self.final_layernorm(x)


class MTP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        if args.num_nextn_predict_layers != 1:
            raise ValueError("the Ling MTP head runs exactly one NEXTN layer")
        self.layers = [MTPLayer(args)]


class Model(nn.Module):
    # ArraysCache record/trim plus KVCache trim are the generic
    # HybridVerifyRows exact transaction contract.  This capability is owned
    # by the cache topology, independent of the Ling adapter or route name.
    supports_speculative_rollback = True

    # Adapter-selected external feature: ``None`` (post-block taps) or
    # ``"mtp_seed"`` (post-final-norm hidden for the native MTP head drafter).
    external_feature_source = None

    def __init__(self, args: ModelArgs, *, mtp: bool = False):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LanguageModel(args)
        self.mtp = MTP(args) if mtp else None
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.hidden_size,
                args.vocab_size,
                bias=False,
            )

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
    ) -> mx.array:
        with ling_rowwise.scope(inputs, cache):
            if inputs.ndim == 2 and inputs.shape[1] > 1 and self._claim_tail(inputs, cache):
                STATS["ling_tail_split"] += 1
                output = self.model(inputs, cache, tail_row=True)
            else:
                # Prompt chunks project only each lane's checkpoint row through
                # the head (prefill_checkpoint_rows); identity outside a request.
                output = select_checkpoint_rows(self.model(inputs, cache))
            if self.args.tie_word_embeddings:
                return self.model.word_embeddings.as_linear(output)
            return self.lm_head(output)

    def _claim_tail(self, inputs, cache=None):
        """Opt-in last-layer tail (MLX2_PREFILL_TAIL_SKIP=1, default off).

        Claims an open prompt-chunk checkpoint request whose rows are every
        lane's last; the forward then returns one row per lane.  Declines
        (full chunk, nothing claimed) for a replaced last layer or attention
        and a recurrent (KDA) last layer (impossible for the layer-group rule
        today, which makes the last layer global).  A cache's own bucketed
        decode attention may serve the single tail query; it reads cache
        state and never writes it.
        """
        last = self.model.layers[-1]
        refuse = None
        if type(last) is not DecoderLayer or type(last.attention) not in (
            MultiLatentAttention, KimiDeltaAttention
        ):
            refuse = "wrapped_last_layer"
        elif not last.is_global:
            refuse = "recurrent_last_layer"
        else:
            refuse = structural_refusal(
                inputs, cache, trunk=self.model, trunk_class=LanguageModel,
                parts=((last.mlp, (SparseMoeBlock, MLP)),),
            )
        return claim_adapter_tail(inputs, kind="tail", refuse=refuse)

    @property
    def layers(self):
        return self.model.layers

    @property
    def speculative_args(self):
        return self.args

    def logits(self, hidden: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.model.word_embeddings.as_linear(hidden)
        return self.lm_head(hidden)

    def make_mtp_cache(self) -> list[Any]:
        if self.mtp is None:
            raise ValueError("this Ling artifact has no loaded MTP head")
        return [KVCache()]

    def mtp_step(self, hidden, tokens, mtp_cache):
        """One MTP forward over S positions (the shared native-MTP ABI).

        hidden: [B, S, H] post-``model.norm`` hiddens at positions p..p+S-1
        (or a previous ``mtp_step``'s post hidden when chaining depth);
        tokens: [B, S] the tokens at p+1..p+S.  Returns (logits, post hidden).
        """
        if self.mtp is None:
            raise ValueError("this Ling artifact has no loaded MTP head")
        layer = self.mtp.layers[0]
        embeds = self.model.word_embeddings(tokens)
        mask = create_attention_mask(embeds, mtp_cache[0], return_array=True)
        post = layer(embeds, hidden, mask=mask, cache=mtp_cache[0])
        return self.logits(post), post

    def select_external_feature_source(self, source):
        if source not in (None, "mtp_seed"):
            raise ValueError("unsupported Ling external feature source")
        if source == "mtp_seed" and self.mtp is None:
            raise ValueError("the MTP seed feature requires a loaded MTP head")
        object.__setattr__(self, "external_feature_source", source)

    @property
    def external_feature_width(self):
        if self.external_feature_source == "mtp_seed":
            return int(self.args.hidden_size)
        return None

    def forward_with_taps(
        self,
        inputs,
        cache,
        capture_layers,
        *,
        body_only=False,
        last_logits_only=False,
    ):
        """Return logits and concatenated post-layer residual-stream taps."""

        with ling_rowwise.scope(inputs, cache):
            return self._forward_with_taps_scoped(
                inputs, cache, capture_layers, body_only=body_only,
                last_logits_only=last_logits_only)

    def _forward_with_taps_scoped(self, inputs, cache, capture_layers, *, body_only,
                                  last_logits_only):
        capture_layers = tuple(int(value) for value in capture_layers)
        if self.external_feature_source == "mtp_seed":
            if capture_layers != (len(self.layers) - 1,):
                raise ValueError("the MTP seed feature is the last layer's post-norm hidden")
            if body_only and last_logits_only:
                raise ValueError("body_only and last_logits_only are mutually exclusive")
            hidden = self.model(inputs, cache)
            if body_only:
                return None, hidden
            return self.logits(hidden[:, -1:] if last_logits_only else hidden), hidden
        if (
            not capture_layers
            or tuple(sorted(set(capture_layers))) != capture_layers
            or capture_layers[0] < 0
            or capture_layers[-1] >= len(self.layers)
        ):
            raise ValueError("invalid Ling target capture layers")
        if body_only and last_logits_only:
            raise ValueError("body_only and last_logits_only are mutually exclusive")
        with moe_nax_gather.forward_scope(inputs, cache), \
                _prefetched_cache_offsets(self.layers, cache):
            return self._forward_with_taps(inputs, cache, capture_layers, body_only=body_only,
                                           last_logits_only=last_logits_only)

    def _forward_with_taps(self, inputs, cache, capture_layers, *, body_only, last_logits_only):
        hidden = self.model.word_embeddings(inputs)
        cache = [None] * len(self.layers) if cache is None else cache
        attention_mask = create_attention_mask(
            hidden, cache[self.model._attention_idx], return_array=True
        )
        kda_mask = create_ssm_mask(hidden, cache[self.model._kda_idx])
        wanted = frozenset(capture_layers)
        taps = []
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            mask = attention_mask if layer.is_global else kda_mask
            hidden = layer(hidden, mask=mask, cache=layer_cache)
            if index in wanted:
                taps.append(hidden)
        if len(taps) != len(capture_layers):
            raise RuntimeError("Ling target tap count does not match capture layers")
        features = mx.concatenate(taps, axis=-1)
        hidden = self.model.norm(hidden)
        if body_only:
            return None, features
        projected = hidden[:, -1:] if last_logits_only else hidden
        logits = (
            self.model.word_embeddings.as_linear(projected)
            if self.args.tie_word_embeddings
            else self.lm_head(projected)
        )
        return logits, features

    def prefill_body(self, inputs, cache, capture_layers):
        return self.forward_with_taps(inputs, cache, capture_layers, body_only=True)[1]

    def forward_tree_with_taps(self, inputs, tree, capture_layers):
        """One layer traversal for all nodes, using parent-indexed state math."""
        capture_layers = tuple(capture_layers)
        if (
            inputs.shape != (1, len(tree.parents)) or not capture_layers
            or tuple(sorted(set(capture_layers))) != capture_layers
            or capture_layers[0] < 0 or capture_layers[-1] >= len(self.layers)
        ):
            raise ValueError("invalid Ling tree input or capture layers")
        hidden = self.model.word_embeddings(inputs)
        taps = []
        for index, (layer, cache) in enumerate(zip(self.layers, tree.private, strict=True)):
            hidden = layer(hidden, cache=cache, tree=tree)
            if index in capture_layers:
                taps.append(hidden)
        output = self.model.norm(hidden)
        logits = self.logits(output)
        if self.external_feature_source == "mtp_seed":
            return logits, output
        return logits, mx.concatenate(taps, axis=-1)

    def make_cache(self) -> list[Any]:
        return [
            KVCache() if layer.is_global else ArraysCache(size=4)
            for layer in self.layers
        ]

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        n_layers = self.args.num_hidden_layers
        weights = {
            key: value
            for key, value in weights.items()
            if not (
                key.startswith("model.layers.")
                and key.split(".")[2].isdigit()
                and int(key.split(".")[2]) >= n_layers
            )
        }

        if self.mtp is None:
            weights = {k: v for k, v in weights.items() if not k.startswith("mtp.")}
        else:
            kv_b_key = "mtp.layers.0.attention.kv_b_proj.weight"
            if kv_b_key in weights:
                key_weight, value_weight = self._split_kv_b(weights.pop(kv_b_key))
                weights["mtp.layers.0.attention.embed_q.weight"] = key_weight
                weights["mtp.layers.0.attention.unembed_out.weight"] = value_weight
            gate_key = "mtp.layers.0.mlp.gate.weight"
            if gate_key in weights:
                weights["mtp.layers.0.mlp.gate.gate_proj.weight"] = weights.pop(gate_key)

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for layer_idx, layer in enumerate(self.layers):
            prefix = f"model.layers.{layer_idx}"

            if isinstance(layer.mlp, SparseMoeBlock):
                for projection in ("gate_proj", "down_proj", "up_proj"):
                    for suffix in ("weight", "scales", "biases"):
                        first = f"{prefix}.mlp.experts.0.{projection}.{suffix}"
                        if first in weights:
                            stacked = [
                                weights.pop(
                                    f"{prefix}.mlp.experts.{expert}.{projection}.{suffix}"
                                )
                                for expert in range(self.args.num_experts)
                            ]
                            weights[
                                f"{prefix}.mlp.switch_mlp.{projection}.{suffix}"
                            ] = mx.stack(stacked)

                gate_key = f"{prefix}.mlp.gate.weight"
                if gate_key in weights:
                    weights[f"{prefix}.mlp.gate.gate_proj.weight"] = weights.pop(
                        gate_key
                    )

            if layer.is_global:
                kv_b_key = f"{prefix}.attention.kv_b_proj.weight"
                if kv_b_key in weights:
                    key_weight, value_weight = self._split_kv_b(weights.pop(kv_b_key))
                    weights[f"{prefix}.attention.embed_q.weight"] = key_weight
                    weights[f"{prefix}.attention.unembed_out.weight"] = value_weight
            else:
                for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                    source = f"{prefix}.attention.{name}.weight"
                    if source in weights:
                        value = weights.pop(source)
                        if value.ndim == 3:
                            value = value.moveaxis(2, 1)
                        weights[f"{prefix}.attention.{name}.conv.weight"] = value

        return weights

    def _split_kv_b(self, value):
        combined = self.args.qk_nope_head_dim + self.args.v_head_dim
        value = value.reshape(self.args.num_attention_heads, combined, -1)
        key_weight = mx.contiguous(value[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2))
        value_weight = mx.contiguous(value[:, self.args.qk_nope_head_dim :, :])
        return key_weight, value_weight

    @property
    def quant_predicate(self):
        def predicate(path: str, _module: nn.Module):
            if path.endswith("mlp.gate.gate_proj"):
                if self.args.vendor_v32_laws and self.args.router_dtype == "fp32":
                    return False
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if path.endswith(("A_log", "dt_bias")):
                return False
            return "expert_bias" not in path

        return predicate
