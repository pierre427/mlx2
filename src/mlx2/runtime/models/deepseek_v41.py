# SPDX-License-Identifier: MIT
"""Adapter-owned V4.1 ordinary B1 model, using V4 mechanisms where laws agree.

Checkpoint weights remain native packed blocks. Shared CSA2 pools, staggered
HC, Engram and QAT differ from V4 and are implemented here. All policies are
unqualified until the exact artifact passes the deferred model campaign.
"""

import hashlib
import json
from contextlib import nullcontext
from dataclasses import asdict

import mlx.core as mx
import numpy as np

from ..prefill_checkpoint_rows import select_checkpoint_rows
from .deepseek_v4 import (
    Rotary,
    _sinkhorn_mix_fp32,
    attend,
    hc_expand,
    limited_swiglu,
    route_experts,
    stable_pool_topk,
)
from .deepseek_v41_cache import DeepSeekV41LayerCache
from .deepseek_v41_quant import quantize

NUMERICS_REVISION = "mlx2.v41-packed-bf16-staggered-csa2-engram.v1"
INDEX_POLICIES = ("reference", "finite_all_retained")
SELECTED_KV_POLICIES = ("reference", "decode_reuse")
KV_STORAGE_POLICIES = ("reference", "native_packed")
CANDIDATE_SCORE_POLICIES = ("reference", "candidate_only")
CED_POLICIES = ("reference", "dependency_exact")
# Selected-attention laws with a compatible packed consumer: the reference
# bounded packed gather, or exact bounded retained-pool decode staging.
PACKED_ATTENTION_MATH_POLICIES = ("reference", "all_retained_broadcast", "omlx_nax_bf16_online_v1")


def candidate_blocks(scores, lens, count, block, *, finite_all_retained=False):
    """Chronological tie break, newest visible block pinned, unreachable dropped."""
    width = scores.shape[-1]
    if finite_all_retained and (width + block - 1) // block <= count:
        # This explicit profile requires finite reachable scores. The mask
        # marks whole reachable blocks, including the newest partial block;
        # the separate token-causal mask still excludes its future rows.
        return (mx.arange(width) // block * block)[None, None] < lens
    pad = (-width) % block
    padded = (
        mx.concatenate((scores, mx.full((*scores.shape[:-1], pad), -mx.inf)), -1)
        if pad
        else scores
    )
    blocks = mx.max(padded.reshape(*scores.shape[:-1], -1, block), -1)
    newest = (lens - 1) // block
    blocks = mx.where(mx.arange(blocks.shape[-1])[None, None] == newest, mx.inf, blocks)
    indices = stable_pool_topk(blocks, min(count, blocks.shape[-1]))
    picked = mx.take_along_axis(blocks, indices, -1) > -mx.inf
    keep = mx.zeros(blocks.shape, mx.bool_)
    keep = mx.put_along_axis(keep, indices, picked, -1)
    return mx.repeat(keep, block, axis=-1)[..., :width]


def _selected_rows(pooled, indices):
    return pooled[0, indices[0]][None]


def decode_selected_kv(source, indices, valid, shared, *, counters=None):
    """One forward-local gather, invalidated by owner or selection changes.

    Keep strong references instead of bare Python/Metal addresses. Source
    prefixes are append-only; count/offset and backing identity distinguish
    growth, restoration and verifier-visible frontier changes. Nothing is
    retained on the model or persistent cache after the forward returns.
    """
    packed = getattr(source, "packed_pooled", None)
    pooled = packed if packed is not None else source.pooled
    dtype = mx.float32 if packed is not None else (None if pooled is None else pooled.dtype)
    if (
        pooled is None
        or len(pooled.shape) != 3
        or pooled.shape[0] != 1
        or indices.ndim != 3
        or indices.shape[:2] != (1, 1)
        or valid.shape != indices.shape
        or valid.dtype != mx.bool_
        or indices.dtype not in (mx.int32, mx.uint32)
        or not 0 < indices.shape[-1] <= min(512, pooled.shape[1])
    ):
        raise ValueError("V4.1 decode gather requires B1 width-one selected rows")
    geometry = (pooled.shape, dtype, indices.shape, indices.dtype,
                valid.shape, valid.dtype, source.offset, source._pool_count)
    generation = shared.get("_index_generation", 0)
    entry = shared.get("_decode_selected_kv")
    if (entry is not None and entry["source"] is source
            and entry["pool"] is source._pooled
            and entry["indices"] is indices and entry["valid"] is valid
            and entry["geometry"] == geometry
            and entry["generation"] == generation):
        if counters is not None:
            counters["decode_gather_hit"] = counters.get("decode_gather_hit", 0) + 1
        return entry["value"]
    # A failed new gather must not leave an old selection available to reuse.
    shared.pop("_decode_selected_kv", None)
    value = source.gather_pooled(indices) if packed is not None else _selected_rows(pooled, indices)
    shared["_decode_selected_kv"] = {
        "source": source, "pool": source._pooled, "indices": indices,
        "valid": valid, "geometry": geometry, "generation": generation,
        "value": value,
    }
    if counters is not None:
        counters["decode_gather_build"] = counters.get("decode_gather_build", 0) + 1
        if packed is not None:
            counters["packed_selected_staging_max_rows"] = max(counters.get("packed_selected_staging_max_rows", 0), indices.shape[-1])
    return value


def selected_attention(
    q, local, local_mask, pooled, indices, valid, sinks, *, workspace_bytes=16 << 20,
    gathered=None, trusted_broadcast=False, packed_pooled=None, counters=None,
):
    """HySparse2-style bounded selected-row gather, no full-context score/mask.

    Attention visits the same local then chronological compressed rows as the
    ordinary V4 reference. Tile bounds include gathered vectors and scores.
    """
    if packed_pooled is not None and (pooled is not None or trusted_broadcast):
        raise ValueError("packed selected attention requires one native source and reference math")
    if pooled is None and packed_pooled is None:
        if gathered is not None:
            raise ValueError("V4.1 reused gather requires a pooled source")
        return attend(q, local, local_mask, sinks=sinks)
    heads, length, dim = q.shape[1:]
    count = indices.shape[-1]
    if gathered is not None and (
        length != 1 or gathered.shape != (1, 1, count, dim)
        or gathered.dtype != (mx.float32 if packed_pooled is not None else pooled.dtype)
    ):
        raise ValueError("V4.1 reused gather must match width-one attention")
    per_row = 4 * (heads * (local.shape[2] + count) + count * dim + 2 * heads * dim)
    tile = max(1, min(128, workspace_bytes // max(1, per_row)))
    output = []
    for first in range(0, length, tile):
        last = min(length, first + tile)
        query = q[:, :, first:last].astype(mx.float32) * dim**-0.5
        if gathered is not None:
            rows = gathered
        elif packed_pooled is not None:
            from .deepseek_v41_packed_qat import gather_decode

            rows = gather_decode(packed_pooled, indices[:, first:last])
            if counters is not None:
                staged = (last-first)*count
                counters["packed_selected_decode_rows"] = counters.get("packed_selected_decode_rows", 0) + staged
                counters["packed_selected_staging_max_rows"] = max(counters.get("packed_selected_staging_max_rows", 0), staged)
        else:
            rows = (mx.broadcast_to(pooled[:, None], (1, last-first, count, dim))
                    if trusted_broadcast and length > 1 and count == pooled.shape[1] <= 512
                    else _selected_rows(pooled, indices[:, first:last]))
        ls = query @ mx.swapaxes(local.astype(mx.float32), -1, -2)
        ls = mx.where(local_mask[..., first:last, :], ls, -mx.inf)
        ps = mx.einsum("bhld,blkd->bhlk", query, rows)
        ps = mx.where(valid[:, None, first:last], ps, -mx.inf)
        norm = mx.logaddexp(
            mx.logsumexp(ls, -1, keepdims=True), mx.logsumexp(ps, -1, keepdims=True)
        )
        norm = mx.logaddexp(norm, sinks[None, :, None, None].astype(mx.float32))
        result = mx.exp(ls - norm) @ local.astype(mx.float32)
        result += mx.einsum("bhlk,blkd->bhld", mx.exp(ps - norm), rows)
        output.append(result.astype(q.dtype))
    return mx.concatenate(output, 2)


class Attention:
    def __init__(self, model, layer):
        self.model, self.c, self.w, self.layer = model, model.args, model.weights, layer
        c = self.c
        self.prefix = f"blk.{layer}."
        self.ratio = c.compress_ratios[layer]
        self.source = layer in c.kv_source_layer_ids
        self.index_source = layer in c.index_source_layer_ids
        self.rope = Rotary(
            c.qk_rope_head_dim,
            c.compress_rope_theta if self.ratio else c.rope_theta,
            c.rope_scaling if self.ratio else None,
        )
        self._frame_rope_key = (
            c.qk_rope_head_dim,
            c.compress_rope_theta if self.ratio else c.rope_theta,
            json.dumps(c.rope_scaling if self.ratio else None, sort_keys=True),
        )
        self.q_norm = self.w.array(self.prefix + "attn_q_a_norm.weight")
        self.kv_norm = self.w.array(self.prefix + "attn_kv_a_norm.weight")
        self.sinks = self.w.array(self.prefix + "attn_sinks.weight")

    def linear(self, x, name, *, rounded=True):
        return self.model.linear(x, self.prefix + name + ".weight", rounded=rounded)

    def _phase(self, shared, x, offset, positions, inverse):
        geometry = None if shared is None else shared.get("_source_prefill_geometry")
        if geometry is None:
            return None
        from .deepseek_v41_frame_geometry import phase_operands
        return phase_operands(geometry, mx, self.rope, self._frame_rope_key, x, offset,
                              positions=positions, inverse=inverse)

    def rotate(self, x, offset=0, *, positions=None, inverse=False, shared=None):
        phase = self._phase(shared, x, offset, positions, inverse)
        if getattr(self.model,"pointwise_policy","reference") == "fused":
            from .deepseek_v41_fused import rotate, self_check
            self_check()
            return rotate(x,self.rope,offset,positions=positions,inverse=inverse,phase=phase)
        if phase is not None:
            from .deepseek_v41_frame_geometry import apply_rotary
            return self.model.round(apply_rotary(mx, x, self.rope, phase))
        return self.model.round(
            self.rope(x, offset, positions=positions, inverse=inverse)
        )

    def rotate_quantized(self,x,kind,offset=0,*,positions=None,inverse=False,shared=None):
        m=self.model
        if (getattr(m,"attention_epilogue_policy","reference") == "fused"
                and m.qat_policy == "fused_fp32" and self.c.simulate_quantization):
            from .deepseek_v41_fused import epilogue_check, rotate_qat
            epilogue_check()
            phase = self._phase(shared, x, offset, positions, inverse)
            return rotate_qat(x,self.rope,kind,offset,positions=positions,inverse=inverse,phase=phase)
        return m.qat(self.rotate(x,offset,positions=positions,inverse=inverse,shared=shared),kind)

    def rotate_packed(self, x, kind, offset=0, *, positions=None, shared=None):
        from .deepseek_v41_packed_qat import produce

        return produce(self.rotate(x, offset, positions=positions, shared=shared), kind,
                       codec=getattr(self.model, "packed_codec_policy", "reference"))

    def publish(self, x, state, start, shared=None):
        c, m, ratio = self.c, self.model, self.ratio
        if ratio == 2 and m.can_pair(x, self.prefix+"attn_compressor_kv.weight", self.prefix+"attn_compressor_gate.weight"):
            kv, gate = self.w.paired_project(x, self.prefix+"attn_compressor_kv.weight", self.prefix+"attn_compressor_gate.weight", rounded=False)
        else:
            kv = self.linear(x, "attn_compressor_kv", rounded=ratio == 1)
            gate = self.linear(x, "attn_compressor_gate", rounded=False) if ratio == 2 else None
        if ratio == 2:
            if state.buf_kv is not None:
                kv = mx.concatenate((state.buf_kv, kv), 1)
                gate = mx.concatenate((state.buf_gate, gate), 1)
            complete = kv.shape[1] // ratio
            usable = complete * ratio
            state.buf_kv = mx.asarray(kv[:, usable:], copy=True)
            state.buf_gate = mx.asarray(gate[:, usable:], copy=True)
            kv, gate = kv[:, :usable], gate[:, :usable]
            if complete:
                kv = mx.sum(
                    kv.reshape(1, complete, ratio, c.head_dim)
                    * mx.softmax(gate.reshape(1, complete, ratio, c.head_dim), axis=2),
                    axis=2,
                )
                kv = m.round(kv)
            else:
                kv = mx.zeros((1, 0, c.head_dim), mx.float32)
        else:
            complete = x.shape[1]
            state.buf_kv = state.buf_gate = mx.zeros((1, 0, c.head_dim), mx.float32)
        if complete:
            latent = m.norm(
                kv, self.w.array(self.prefix + "attn_compressor_norm.weight")
            )
            positions = tuple(
                range(
                    (start // ratio) * ratio, (start // ratio + complete) * ratio, ratio
                )
            )
            key = m.norm(
                self.linear(latent, "indexer.attn_k"),
                self.w.array(self.prefix + "indexer.k_norm.weight"),
            )
            if getattr(m, "kv_storage_policy", "reference") == "native_packed":
                from .deepseek_v41_packed_qat import PackedQAT

                key = self.rotate_packed(key[:, None], "index_fp4", positions=positions, shared=shared)
                pooled = self.rotate_packed(latent[:, None], "compressed_fp4", positions=positions, shared=shared)
                state.append_pools_packed(
                    PackedQAT(pooled.codes.squeeze(1), pooled.scales.squeeze(1), pooled.kind),
                    PackedQAT(key.codes.squeeze(1), key.scales.squeeze(1), key.kind),
                )
            else:
                key = self.rotate_quantized(key[:, None], "index_fp4",positions=positions,shared=shared).squeeze(1)
                pooled = self.rotate_quantized(latent[:, None], "compressed_fp4",positions=positions,shared=shared).squeeze(1)
                state.append_pools(pooled, key)
        else:
            if getattr(m, "kv_storage_policy", "reference") == "native_packed":
                from .deepseek_v41_packed_qat import PackedQAT

                state.append_pools_packed(
                    PackedQAT(mx.zeros((1, 0, c.head_dim // 2), mx.uint8),
                              mx.zeros((1, 0, c.head_dim // 16), mx.uint8), "compressed_fp4"),
                    PackedQAT(mx.zeros((1, 0, c.index_head_dim // 2), mx.uint8),
                              mx.zeros((1, 0, c.index_head_dim // 32), mx.uint8), "index_fp4"),
                )
            else:
                state.append_pools(
                    mx.zeros((1, 0, c.head_dim), mx.float32),
                    mx.zeros((1, 0, c.index_head_dim), mx.float32),
                )

    def select(self, x, qr, source, start, shared):
        """Bound head-score scratch by tiling query rows; selection remains exact."""
        c, m = self.c, self.model
        length = x.shape[1]
        count = source._pool_count if hasattr(source, "_pool_count") else source.index_keys.shape[1]
        shared["_index_generation"] = shared.get("_index_generation", 0) + 1
        shared.pop("_decode_selected_kv", None)
        shared.pop("indices", None)
        shared.pop("selected_mask", None)
        shared.pop("_all_retained_chronological", None)
        if not count:
            shared["indices"] = mx.zeros((1, length, 0), mx.int32)
            shared["selected_mask"] = mx.zeros((1, length, 0), mx.bool_)
            if self.layer == c.candidate_source_layer_id:
                shared["candidates"] = mx.zeros((1, length, 0), mx.bool_)
            return
        finite = m.index_policy == "finite_all_retained"
        all_blocks = (count + c.candidate_block_size - 1) // c.candidate_block_size <= c.candidate_topk_blocks
        if finite and count <= c.index_topk and (
            self.layer != c.candidate_source_layer_id or all_blocks
        ):
            # Replacing score work is valid only under this explicit profile's
            # finite-unmasked-score contract; no host readback is introduced.
            # Match stable_pool_topk's argpartition index dtype, not the
            # int32 positions used internally by its chronological ranking.
            positions = mx.arange(count, dtype=mx.uint32)[None, None]
            lens = ((mx.arange(start, start + length) + 1) // self.ratio)[None, :, None]
            shared["indices"] = mx.broadcast_to(positions, (1, length, count))
            visible = positions < lens
            if self.layer == c.candidate_source_layer_id:
                shared["candidates"] = (positions // c.candidate_block_size * c.candidate_block_size) < lens
            elif self.layer > c.candidate_source_layer_id >= 0:
                prior = shared.get("candidates")
                if prior is None:
                    raise ValueError("V4.1 candidate source failed to publish")
                visible = visible & prior
            shared["selected_mask"] = visible
            shared["_all_retained_chronological"] = (id(shared["indices"]), count, length, start,
                                                     shared["_index_generation"])
            m.scheduled_kv_calls["all_retained_index"] = m.scheduled_kv_calls.get("all_retained_index", 0) + 1
            if self.layer == c.candidate_source_layer_id:
                m.scheduled_kv_calls["candidate_rank_bypass"] = m.scheduled_kv_calls.get("candidate_rank_bypass", 0) + 1
            return
        query = (
            self.linear(qr, "indexer.attn_q_b")
            .reshape(1, length, c.index_n_heads, c.index_head_dim)
            .transpose(0, 2, 1, 3)
        )
        query = self.rotate_quantized(query,"index_fp4",start,shared=shared)
        weight = self.linear(x, "indexer.proj") * (
            c.index_head_dim**-0.5 * c.index_n_heads**-0.5
        )
        candidate_only = (getattr(m, "candidate_score_policy", "reference") == "candidate_only"
                          and self.layer > c.candidate_source_layer_id >= 0)
        capacity = min(count, c.candidate_topk_blocks * c.candidate_block_size)
        per_row = ((c.index_n_heads + c.index_head_dim) * capacity + count
                   if candidate_only else c.index_n_heads * count)
        tile = max(1, min(128, (16 << 20) // max(1, per_row * 4)))
        selected, masks = [], []
        prior_candidates = shared.get("candidates")
        if candidate_only and (prior_candidates is None or prior_candidates.shape != (1, length, count)):
            raise ValueError("V4.1 candidate source failed to publish matching rows")
        packed_index = getattr(source, "packed_index_keys", None)
        all_keys = None if candidate_only or packed_index is not None else source.index_keys
        for first in range(0, length, tile):
            last = min(length, first + tile)
            if candidate_only:
                prior = prior_candidates[:, first:last]
                positions = mx.arange(count, dtype=mx.uint32)[None, None]
                # Fixed-capacity chronological gather. Absent candidates are
                # padding only and are masked before global tie-breaking.
                ids = mx.sort(mx.argsort(mx.where(prior, positions, count + positions), axis=-1)[..., :capacity], axis=-1)
                keys = (source.gather_index_keys(ids) if hasattr(source, "gather_index_keys")
                        else source.index_keys[0, ids[0]][None])
                small = mx.einsum("bhsd,bstd->bsht", query[:, :, first:last], keys)
                small = mx.sum(mx.maximum(small, 0) * weight[:, first:last, :, None], 2)
                small = mx.where(mx.take_along_axis(prior, ids, -1), small, -mx.inf)
                score = mx.put_along_axis(mx.full((1, last-first, count), -mx.inf), ids, small, -1)
                m.scheduled_kv_calls["candidate_only_score_rows"] = m.scheduled_kv_calls.get("candidate_only_score_rows", 0) + (last-first)*capacity
                m.scheduled_kv_calls["candidate_only_unscored_rows"] = m.scheduled_kv_calls.get("candidate_only_unscored_rows", 0) + (last-first)*(count-capacity)
            elif packed_index is not None:
                from .deepseek_v41_packed_qat import decode

                parts = []
                for key_first in range(0, count, 4096):
                    key_last = min(count, key_first + 4096)
                    keys = decode(packed_index.sliced((slice(None), slice(key_first, key_last))))
                    part = mx.einsum("bhsd,btd->bsht", query[:, :, first:last], keys)
                    parts.append(mx.sum(mx.maximum(part, 0) * weight[:, first:last, :, None], 2))
                    m.scheduled_kv_calls["packed_index_staging_max_rows"] = max(m.scheduled_kv_calls.get("packed_index_staging_max_rows", 0), key_last-key_first)
                score = mx.concatenate(parts, -1)
            else:
                score = mx.einsum("bhsd,btd->bsht", query[:, :, first:last], all_keys)
                score = mx.sum(mx.maximum(score, 0) * weight[:, first:last, :, None], 2)
            lens = ((mx.arange(start + first, start + last) + 1) // self.ratio)[
                None, :, None
            ]
            score = mx.where(mx.arange(count)[None, None] < lens, score, -mx.inf)
            if self.layer == c.candidate_source_layer_id:
                masks.append(
                    candidate_blocks(
                        score, lens, c.candidate_topk_blocks, c.candidate_block_size,
                        finite_all_retained=finite,
                    )
                )
                if finite and all_blocks:
                    m.scheduled_kv_calls["candidate_rank_bypass"] = m.scheduled_kv_calls.get("candidate_rank_bypass", 0) + 1
            elif self.layer > c.candidate_source_layer_id >= 0:
                if prior_candidates is None:
                    raise ValueError("V4.1 candidate source failed to publish")
                score = mx.where(prior_candidates[:, first:last], score, -mx.inf)
            indices = stable_pool_topk(score, min(c.index_topk, count))
            # Invalid selections remain in range for gather, and carry a separate mask.
            visible = mx.take_along_axis(score, indices, -1) > -mx.inf
            selected.append((indices, visible))
        shared["indices"] = mx.concatenate([v[0] for v in selected], 1)
        shared["selected_mask"] = mx.concatenate([v[1] for v in selected], 1)
        if masks:
            shared["candidates"] = mx.concatenate(masks, 1)

    def __call__(self, x, state, cache, shared):
        c, m, length, start = self.c, self.model, x.shape[1], state.offset
        if (getattr(m, "kv_storage_policy", "reference") == "native_packed"
                and getattr(m, "source_attention_math_policy", "reference") not in PACKED_ATTENTION_MATH_POLICIES):
            raise ValueError("native packed KV requires reference, broadcast or oMLX selected attention math")
        if m.can_pair(x, self.prefix+"attn_q_a.weight", self.prefix+"attn_kv.weight"):
            qa, kva = self.w.paired_project(x,self.prefix+"attn_q_a.weight",self.prefix+"attn_kv.weight")
        else:
            qa, kva = self.linear(x, "attn_q_a"), None
        qr = m.norm(qa, self.q_norm)
        q = (
            self.linear(qr, "attn_q_b")
            .reshape(1, length, c.num_attention_heads, c.head_dim)
            .transpose(0, 2, 1, 3)
        )
        q = self.rotate(q, start, shared=shared)  # V4.1 intentionally has no per-head query RMSNorm.
        kv = m.norm(self.linear(x, "attn_kv") if kva is None else kva, self.kv_norm)[:, None]
        if getattr(m, "kv_storage_policy", "reference") == "native_packed":
            local = state.update_local_packed(self.rotate_packed(kv, "fp8", start, shared=shared))
            m.scheduled_kv_calls["native_packed_local_rows"] = m.scheduled_kv_calls.get("native_packed_local_rows", 0) + length
        else:
            kv = self.rotate_quantized(kv,"fp8",start,shared=shared)
            local = state.update_local(kv)
        geometry = shared.get("_source_prefill_geometry")
        if geometry is None:
            queries = mx.arange(start, start + length)[:, None]
            keys = mx.arange(start + length - local.shape[2], start + length)
            mask = ((queries >= keys) & (queries < keys + c.sliding_window))[None, None]
        else:
            from .deepseek_v41_frame_geometry import local_mask
            mask = local_mask(geometry, mx, start, length, local.shape[2], c.sliding_window)
        pooled = packed_pooled = indices = valid = None
        if self.ratio:
            owner = cache[c.kv_source_for(self.layer)]
            if self.source:
                if geometry is None:
                    self.publish(x, state, start)
                else:
                    self.publish(x, state, start, shared=shared)
            # Binding to the source row works even when a ratio2 group is incomplete.
            if self.index_source:
                self.select(x, qr, owner, start, shared)
            packed_pooled = getattr(owner, "packed_pooled", None)
            pooled = None if packed_pooled is not None else owner.pooled
            source_pool = packed_pooled if packed_pooled is not None else pooled
            if source_pool is None:
                raise ValueError("V4.1 compressed source failed to publish")
            if source_pool.shape[1]:
                indices = shared.get("indices")
                valid = shared.get("selected_mask")
                if indices is None or valid is None:
                    raise ValueError("V4.1 index source failed to publish")
            else:
                pooled = packed_pooled = None
            if packed_pooled is not None and getattr(m, "source_attention_math_policy", "reference") != "reference":
                from .deepseek_v41_packed_cache import stage_retained_pool
                pooled, packed_pooled = stage_retained_pool(m, packed_pooled, indices, length, start, shared)
        trusted_retained = (m.index_policy == "finite_all_retained" and length > 1 and pooled is not None
                     and pooled.shape[1] <= 512
                     and shared.get("_all_retained_chronological") ==
                     (id(indices), pooled.shape[1], length, start, shared.get("_index_generation")))
        broadcast = trusted_retained and getattr(m, "source_attention_math_policy", "reference") in ("all_retained_broadcast", "omlx_nax_bf16_online_v1", "source_short_decode_joint_f32_v1", "source_short_decode_grouped_f32_v1")
        short_policy=getattr(m,'source_attention_math_policy','reference') in (
            'source_short_decode_joint_f32_v1','source_short_decode_grouped_f32_v1')
        short_gather=(decode_selected_kv(owner,indices,valid,shared,counters=m.scheduled_kv_calls)
                      if short_policy and pooled is not None and length==1 and m.selected_kv_policy=='decode_reuse' else None)
        def reference_attention():
            if getattr(m, "source_attention_math_policy", "reference") != "reference":
                key = "source_all_retained_broadcast_attention" if broadcast else "source_attention_math_reference"
                m.scheduled_kv_calls[key] = m.scheduled_kv_calls.get(key, 0) + 1
                if broadcast:
                    key = "source_broadcast_avoided_gather_logical_bytes"
                    m.scheduled_kv_calls[key] = m.scheduled_kv_calls.get(key, 0) + length*pooled.shape[1]*c.head_dim*4
            return selected_attention(
                q, local, mask, pooled, indices, valid, self.sinks,
                gathered=(short_gather if short_policy else decode_selected_kv(owner, indices, valid, shared,
                                            counters=m.scheduled_kv_calls)
                          if (pooled is not None or packed_pooled is not None) and length == 1
                          and m.selected_kv_policy == "decode_reuse" else None),
                **({"packed_pooled": packed_pooled, "counters": m.scheduled_kv_calls}
                   if packed_pooled is not None else {}),
                **({"trusted_broadcast": True} if broadcast else {}),
            )
        if getattr(m, "source_attention_math_policy", "reference") == "omlx_nax_bf16_online_v1":
            from .deepseek_v41_source_attention import dispatch
            raw_attention = dispatch(q, local, mask, pooled, indices, valid, self.sinks,
                                     trusted_retained=trusted_retained, reference=reference_attention,
                                     counters=m.scheduled_kv_calls)
        elif getattr(m, "source_attention_math_policy", "reference") in ('source_short_decode_joint_f32_v1','source_short_decode_grouped_f32_v1'):
            from .deepseek_v41_short_attention import dispatch
            raw_attention = dispatch(q,local,mask,pooled,indices,valid,self.sinks,
                policy=m.source_attention_math_policy,
                gathered=short_gather,
                reference=reference_attention,counters=m.scheduled_kv_calls)
        else:
            raw_attention = reference_attention()
        out = m.round(raw_attention)
        fused_output = getattr(m,"attention_epilogue_policy","reference") == "fused"
        out = (self.rotate_quantized(out,"fp8",start,inverse=True,shared=shared) if fused_output
               else self.rotate(out,start,inverse=True,shared=shared))
        out = (
            out.reshape(
                1, c.o_groups, c.num_attention_heads // c.o_groups, length, c.head_dim
            )
            .transpose(0, 3, 1, 2, 4)
            .reshape(1, length, c.o_groups, -1)
        )
        # The grouped output activation QAT exists in the official V4.1 graph.
        if not fused_output:
            out = m.qat(out, "fp8")
        out = m.round(
            self.w.grouped(out, self.prefix + "attn_output_a.weight", groups=c.o_groups)
        )
        return self.linear(out.reshape(1, length, -1), "attn_output_b")


class Block:
    def __init__(self, model, layer):
        self.model, self.c, self.w, self.layer = model, model.args, model.weights, layer
        self.prefix = f"blk.{layer}."
        self.attn = Attention(model, layer)
        self.hc = {
            kind: tuple(
                self.w.array(self.prefix + f"hc_{kind}_{part}.weight").astype(
                    mx.float32
                )
                for part in ("fn", "scale", "base")
            )
            for kind in ("attn", "ffn")
        }
        self.attn_norm = self.w.array(self.prefix + "attn_norm.weight")
        self.ffn_norm = self.w.array(self.prefix + "ffn_norm.weight")
        self.correction = self.w.array(self.prefix + "exp_probs_b.bias")

    def mixes(self, h, kind):
        c = self.c
        fn, scale, base = self.hc[kind]
        flat = h.reshape(*h.shape[:2], -1).astype(mx.float32)
        # V4.1 normalizes after projection; V4's pre-normalized GEMM differs.
        mixes = (flat @ fn.T) * mx.rsqrt(
            mx.mean(mx.square(flat), -1, keepdims=True) + c.rms_norm_eps
        )
        if getattr(self.model, "hc_coeff_policy", "reference") == "source_fused_fp32":
            rows = mixes.size // mixes.shape[-1]
            if c.hc_mult == 4 and 1 <= rows <= 1000:
                from .deepseek_v41_source_hc import split

                result = split(mixes, scale, base, iterations=c.hc_sinkhorn_iters, eps=c.hc_eps)
                counts = self.model.scheduled_hc_calls
                counts["source_fused_fp32"] = counts.get("source_fused_fp32", 0) + 1
                return result
            # The explicit Source coefficient policy does not change residual
            # expansion or inherit the numerically different legacy split.
            result = _sinkhorn_mix_fp32(
                mixes, scale, base, c.hc_mult, c.hc_sinkhorn_iters, c.hc_eps
            )
            counts = self.model.scheduled_hc_calls
            counts["reference_fallback"] = counts.get("reference_fallback", 0) + 1
            return result
        if self.model.hc_policy == "fused_fp32":
            from .deepseek_v4_hc import split_sinkhorn

            return split_sinkhorn(
                mixes, scale, base, iterations=c.hc_sinkhorn_iters, eps=c.hc_eps
            )
        return _sinkhorn_mix_fp32(
            mixes, scale, base, c.hc_mult, c.hc_sinkhorn_iters, c.hc_eps
        )

    def _mega(self, kind, h, pre):
        """Opt-in single-pass Source mHC for one decode row, else None."""
        m, c = self.model, self.c
        if getattr(m, "source_hc_mega_policy", "reference") != "mega_v1":
            return None
        from .deepseek_v41_source_hc_mega import can_pre, pre as mega_pre, self_check
        fn, scale, base = self.hc[kind]
        weight = self.attn_norm if kind == "attn" else self.ffn_norm
        counts = m.scheduled_hc_calls
        if not can_pre(h, pre, fn, scale, base, weight):
            counts["source_hc_mega_pre_reference"] = counts.get("source_hc_mega_pre_reference", 0) + 1
            return None
        options = dict(iterations=c.hc_sinkhorn_iters, hc_eps=c.hc_eps, rms_eps=c.rms_norm_eps)
        self_check(**options)
        counts["source_hc_mega_pre"] = counts.get("source_hc_mega_pre", 0) + 1
        return mega_pre(h, pre, self.hc[kind], weight, **options)

    def _mega_store(self, branch, h, post, comb, shared_out=None):
        m = self.model
        if getattr(m, "source_hc_mega_policy", "reference") != "mega_v1":
            return None
        from .deepseek_v41_source_hc_mega import can_store, store
        counts = m.scheduled_hc_calls
        key = "source_hc_mega_store" + ("_shared" if shared_out is not None else "")
        if not can_store(branch, h, post, comb, shared_out):
            counts[key + "_reference"] = counts.get(key + "_reference", 0) + 1
            return None
        counts[key] = counts.get(key, 0) + 1
        return store(branch, h, post, comb, shared_out)

    def attention_half(self, h, pre, state, cache, shared):
        m = self.model
        mega = getattr(m, "source_hc_mega_policy", "reference") == "mega_v1"
        fused = self._mega("attn", h, pre) if mega else None
        if fused is None:
            attn_pre, post, comb = self.mixes(h, "attn")
            x = m.norm(m.collapse(h, pre), self.attn_norm)
        else:
            attn_pre, post, comb, x = fused
        x = self.attn(x, state, cache, shared)
        collapsed = None
        stored = self._mega_store(x, h, post, comb) if mega else None
        if stored is not None:
            h = stored
            fused = self._mega("ffn", h, attn_pre)
            if fused is not None:
                return (h, fused[3], *fused[:3])
        elif getattr(m,"source_hc_boundary_policy","reference") == "expand_collapse_seq":
            from .deepseek_v41_source_hc_boundary import (
                can_expand_collapse,
                expand_collapse,
            )
            if can_expand_collapse(x,h,post,comb,attn_pre):
                h,collapsed=expand_collapse(x,h,post,comb,attn_pre)
                key="source_hc_boundary_expand_collapse"
            else:
                h=m.expand(x,h,post,comb)
                key="source_hc_boundary_reference"
            m.scheduled_hc_calls[key]=m.scheduled_hc_calls.get(key,0)+1
        else:
            h = m.expand(x, h, post, comb)
        ffn_pre, post, comb = self.mixes(h, "ffn")
        x = m.norm(m.collapse(h, attn_pre)if collapsed is None else collapsed, self.ffn_norm)
        return h, x, ffn_pre, post, comb

    def router(self, x, image_mask=None):
        m, c = self.model, self.c
        route = route_experts
        correction = self.correction
        if image_mask is not None:
            image_mask = np.asarray(image_mask)
            if image_mask.dtype != np.bool_ or image_mask.shape != (x.shape[1],):
                raise ValueError("V4.1 image routing mask must match query rows")
            visual = self.w.array(self.prefix + "exp_probs_b_vl.bias").astype(mx.float32)
            if visual.shape != self.correction.shape:
                raise ValueError("V4.1 visual router bias topology mismatch")
            correction = mx.where(mx.array(image_mask)[None, :, None], visual, self.correction)
        if image_mask is None and getattr(m,"pointwise_policy","reference") == "fused":
            from .deepseek_v41_fused import router, self_check
            self_check()
            route = router
        return route(
            m.linear(x, self.prefix + "ffn_gate_inp.weight", rounded=False),
            top_k=c.num_experts_per_tok,
            scale=c.routed_scaling_factor,
            correction=correction,
            normalize=c.norm_topk_prob,
        )

    def shared_expert(self, x, *, ffn_input_token=None):
        m, c = self.model, self.c
        if (ffn_input_token is None and getattr(self.w,"shared_policy","reference") == "fused"
                and self.w.can_fuse_dense(x,self.prefix+"ffn_gate_shexp.weight",self.prefix+"ffn_up_shexp.weight")):
            from ..gguf.v41_dense import shared_middle
            middle=shared_middle(x,self.w.packed(self.prefix+"ffn_gate_shexp.weight"),self.w.packed(self.prefix+"ffn_up_shexp.weight"))
            self.w.scheduled_dense_calls["v41_shared_swiglu"] = self.w.scheduled_dense_calls.get("v41_shared_swiglu",0)+1
            return m.linear(middle,self.prefix+"ffn_down_shexp.weight")
        gate, up = self.w.gate_up(
            x,
            self.prefix + "ffn_gate_shexp.weight",
            self.prefix + "ffn_up_shexp.weight",
            **({'ffn_input_token':ffn_input_token}if ffn_input_token is not None else{}),
        )
        gate, up = m.round(gate), m.round(up)
        shared_out = m.linear(
            m.round(limited_swiglu(gate, up, c.swiglu_limit)),
            self.prefix + "ffn_down_shexp.weight",
        )
        return shared_out

    def finish(self, h, routed, shared_out, post, comb):
        m = self.model
        if getattr(m, "source_hc_mega_policy", "reference") == "mega_v1":
            stored = self._mega_store(routed, h, post, comb, shared_out)
            if stored is not None:
                return stored
        if getattr(m,"hc_epilogue_policy","reference") == "fused" and m.hc_policy == "fused_fp32":
            from .deepseek_v41_fused import epilogue_check, expand
            epilogue_check()
            return expand(routed,h,post,comb,shared_out)
        return m.round(hc_expand(m.round(routed + shared_out), h, post, comb, policy=m.hc_policy))

    def __call__(self, h, pre, state, cache, shared):
        c = self.c
        h, x, ffn_pre, post, comb = self.attention_half(h, pre, state, cache, shared)
        image_mask = shared.get("_image_mask")
        indices, scores = self.router(x) if image_mask is None else self.router(x, image_mask=image_mask)
        overlap = (getattr(self.w, "expert_io_policy", "blocking") == "overlap"
                   and self.w.expert_stream is not None)
        token=None
        factory=getattr(self.w,'ffn_input_token',None)
        if factory is not None:
            token=factory(x,self.prefix[:-1])
        ticket=None
        def routed_experts():
            return self.w.moe(
                token.flat_input if token is not None else x.reshape(-1, c.hidden_size),
                indices.reshape(-1, c.num_experts_per_tok),
                scores.reshape(-1, c.num_experts_per_tok),
                self.prefix[:-1],
                **({"ticket": ticket} if ticket is not None else {}),
                **({'ffn_input_token':token}if token is not None else{}),
            ).reshape(x.shape)
        try:
            if overlap:
                # A provider may use the exact router input only for early-read hints.
                hint={'route_hidden':x}if getattr(self.w.expert_stream,'accepts_route_hidden',False)else{}
                ticket=self.w.expert_stream.prepare(self.prefix[:-1], indices.reshape(-1,c.num_experts_per_tok),**hint)
            if getattr(self.w,'source_submission_policy','reference') == 'shared_first_v1':
                if overlap:
                    raise ValueError('Source shared-first does not admit asynchronous expert tickets')
                from ...adapters.deepseek_v41_source_submission import (
                    shared_before_route,
                )
                routed, shared_out = shared_before_route(
                    lambda: self.shared_expert(x,**({'ffn_input_token':token}if token is not None else{})),
                    routed_experts, submit=mx.async_eval, counters=self.w.scheduled_dense_calls)
            else:
                routed = None if overlap else routed_experts()
                shared_out = self.shared_expert(x,**({'ffn_input_token':token}if token is not None else{}))
                if overlap:
                    mx.async_eval(shared_out)
                    routed = routed_experts()
        except BaseException:
            if ticket is not None:
                self.w.expert_stream.cancel(ticket)
            raise
        finally:
            if token is not None:token.close()
        return self.finish(h, routed, shared_out, post, comb), ffn_pre


class DeepSeekV41Model:
    def __init__(
        self,
        config,
        weights,
        *,
        identity,
        hasher=None,
        engram_tables=None,
        hc_policy="reference",
        hc_coeff_policy="reference",
        qat_policy="reference",
        index_policy="reference",
        selected_kv_policy="reference",
        source_attention_math_policy="reference",
        source_hc_boundary_policy="reference",
        source_hc_mega_policy="reference",
        kv_storage_policy="reference",
        candidate_score_policy="reference",
        ced_policy="reference",
        context_limit=None,
    ):
        if hc_policy not in ("reference", "fused_fp32"):
            raise ValueError("unsupported V4.1 HC policy")
        if hc_coeff_policy not in ("reference", "source_fused_fp32"):
            raise ValueError("unsupported V4.1 Source HC coefficient policy")
        if hc_coeff_policy != "reference" and hc_policy != "reference":
            raise ValueError("Source HC coefficients cannot combine with legacy fused HC")
        if qat_policy not in ("reference", "fused_fp32"):
            raise ValueError("unknown V4.1 QAT policy")
        if kv_storage_policy not in KV_STORAGE_POLICIES or candidate_score_policy not in CANDIDATE_SCORE_POLICIES or ced_policy not in CED_POLICIES:
            raise ValueError("unknown V4.1 CSA2/CED policy")
        if kv_storage_policy == "native_packed" and (not config.simulate_quantization or config.activation_dtype != "bfloat16"):
            raise ValueError("native packed QAT requires simulated QAT and BF16 activation boundaries")
        if kv_storage_policy == "native_packed" and source_attention_math_policy not in PACKED_ATTENTION_MATH_POLICIES:
            raise ValueError("native packed KV requires reference, broadcast or oMLX selected attention math")
        # Fused QAT producers emit original codes through the exact fused codec.
        self.packed_codec_policy = ("fused_metal_v1" if kv_storage_policy == "native_packed"
                                    and qat_policy == "fused_fp32" else "reference")
        self.kv_storage_policy = kv_storage_policy
        self.candidate_score_policy = candidate_score_policy
        self.ced_policy = ced_policy
        if index_policy not in INDEX_POLICIES or selected_kv_policy not in SELECTED_KV_POLICIES:
            raise ValueError("unknown V4.1 index or selected-KV policy")
        if source_attention_math_policy not in ("reference", "all_retained_broadcast", "omlx_nax_bf16_online_v1", "source_short_decode_joint_f32_v1", "source_short_decode_grouped_f32_v1"):
            raise ValueError("unknown Source attention math policy")
        if source_attention_math_policy != "reference" and index_policy != "finite_all_retained":
            raise ValueError("Source attention broadcast requires finite_all_retained index policy")
        if source_hc_boundary_policy not in ("reference","expand_collapse_seq"):
            raise ValueError("unknown Source HC boundary policy")
        if source_hc_boundary_policy != "reference" and (hc_policy!="reference"or config.activation_dtype!="bfloat16"):
            raise ValueError("Source HC boundary requires reference HC and BF16 boundaries")
        if source_hc_mega_policy not in ("reference", "mega_v1"):
            raise ValueError("unknown Source HC mega policy")
        if source_hc_mega_policy != "reference" and (
                hc_policy != "reference" or hc_coeff_policy != "source_fused_fp32"
                or source_hc_boundary_policy != "reference"
                or config.activation_dtype != "bfloat16" or config.hc_mult != 4):
            raise ValueError("Source HC mega requires Source fused coefficients, reference HC/boundary and BF16 HC4")
        self.source_hc_mega_policy = source_hc_mega_policy
        self.index_policy, self.selected_kv_policy = index_policy, selected_kv_policy
        self.source_attention_math_policy = source_attention_math_policy
        self.source_hc_boundary_policy = source_hc_boundary_policy
        self.scheduled_kv_calls = {}
        self.hc_coeff_policy = hc_coeff_policy
        self.scheduled_hc_calls = {}
        self.qat_policy = qat_policy
        self.args, self.weights, self.hc_policy = config, weights, hc_policy
        if context_limit is None:
            context_limit = min(config.max_position_embeddings, 65536)
        if (type(context_limit) is not int or not 1 <= context_limit <= min(config.max_position_embeddings, 1048576)):
            raise ValueError("V4.1 context limit must fit configured positions and the 1M route bound")
        if context_limit > 65536 and (kv_storage_policy != "native_packed" or candidate_score_policy != "candidate_only"):
            raise ValueError("V4.1 context above 65536 requires native packed KV and candidate-only scoring")
        self.context_limit = context_limit
        self.identity = (
            identity
            + ":"
            + NUMERICS_REVISION
            + ":hc="
            + hc_policy
            + ":hc_coeff=" + hc_coeff_policy
            + ":qat="
            + qat_policy
            + ":index=" + index_policy + ":selected_kv=" + selected_kv_policy
        )
        if source_attention_math_policy != "reference":
            self.identity += ":source_attention_math=" + source_attention_math_policy
        if source_attention_math_policy == "omlx_nax_bf16_online_v1":
            from .deepseek_v41_source_attention import implementation_identity
            self.identity += ":source_attention_implementation=" + implementation_identity()
        if source_attention_math_policy in ('source_short_decode_joint_f32_v1','source_short_decode_grouped_f32_v1'):
            from .deepseek_v41_short_attention import implementation_identity
            self.identity += ':source_attention_implementation='+implementation_identity()
        if source_hc_boundary_policy != "reference":
            self.identity += ":source_hc_boundary=" + source_hc_boundary_policy
        if source_hc_mega_policy != "reference":
            from .deepseek_v41_source_hc_mega import implementation_identity
            self.identity += ":source_hc_mega=" + source_hc_mega_policy + ":" + implementation_identity()
        for name, value in (("kv_storage", kv_storage_policy), ("candidate_score", candidate_score_policy), ("ced", ced_policy)):
            if value != "reference":
                self.identity += ":" + name + "=" + value
        if self.packed_codec_policy != "reference":
            from .deepseek_v41_packed_fused import implementation_identity as packed_codec_identity
            self.identity += (":packed_codec=" + self.packed_codec_policy
                              + ":packed_codec_implementation=" + packed_codec_identity())
        if context_limit != min(config.max_position_embeddings, 65536):
            self.identity += ":context_limit=" + str(context_limit)
        execution_geometry = {
            "config": asdict(config),
            "compute": getattr(weights, "compute_policy", "reference"),
            "expert_prefill": getattr(weights, "expert_prefill_policy", "reference"),
            "dense_prefill": getattr(weights, "dense_prefill_policy", "reference"),
            "expert_storage": "bounded_native_rows" if getattr(weights, "expert_stream", None) is not None else "resident",
        }
        self.identity += (
            ":geometry="
            + hashlib.sha256(
                json.dumps(
                    execution_geometry, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
        )
        self.hasher, self.engram_tables = hasher, engram_tables or {}
        if set(self.engram_tables) != set(config.engram_layer_ids) or (
            config.engram_layer_ids and hasher is None
        ):
            raise ValueError("V4.1 requires every Engram table and its exact token map")
        self.layers = [Block(self, i) for i in range(config.num_hidden_layers)]
        self.norm_weight = weights.array("output_norm.weight")
        self.eval_interval = 1
        self._weights_parked = False
        self._weight_executions = 0

    def _weight_scope(self):
        scope = getattr(self.weights, "dense_window", None)
        return scope() if scope is not None else nullcontext()

    def _weight_eval(self, *values):
        evaluate = getattr(self.weights, "evaluate_dense_window", None)
        if evaluate is not None:
            evaluate(*values)

    def release_weight_references(self):
        """Drop every model-owned weight anchor at a quiescent boundary."""
        if self._weight_executions:
            raise ValueError("cannot park model during an active forward")
        if self._weights_parked:
            return
        self.layers = []
        self.norm_weight = None
        self._weights_parked = True

    def restore_weight_references(self):
        if self._weight_executions:
            raise ValueError("cannot activate model during an active forward")
        if not self._weights_parked:
            return
        layers = [Block(self, i) for i in range(self.args.num_hidden_layers)]
        norm = self.weights.array("output_norm.weight")
        self.layers, self.norm_weight = layers, norm
        self._weights_parked = False
        try:
            # Resident policy restores its admitted payload; paged policy
            # restores only anchors and leaves dense payloads demand-loaded.
            self.preload().eval()
        except BaseException:
            self.layers, self.norm_weight = [], None
            self._weights_parked = True
            raise

    def round(self, x):
        if getattr(self,"pointwise_policy","reference") == "fused" and self.args.activation_dtype == "bfloat16":
            from .deepseek_v41_fused import rounded, self_check
            self_check()
            return rounded(x)
        return x.astype(getattr(mx, self.args.activation_dtype)).astype(mx.float32)

    def norm(self, x, weight):
        return self.round(
            mx.fast.rms_norm(
                x.astype(mx.float32), weight.astype(mx.float32), self.args.rms_norm_eps
            )
        )

    def linear(self, x, name, *, rounded=True):
        if (getattr(self.weights,"dense_epilogue_policy","reference") == "fused"
                and self.weights.can_fuse_dense(x,name)):
            return self.weights.fused_project(x,name,rounded=rounded)
        result = self.weights.linear(name)(x)
        return self.round(result) if rounded else result.astype(mx.float32)

    def can_pair(self, x, *names):
        return (getattr(self.weights,"dense_pair_policy","reference") == "fused"
                and self.weights.can_fuse_dense(x,*names))

    def expand(self,branch,residual,post,comb):
        if getattr(self,"hc_epilogue_policy","reference") == "fused" and self.hc_policy == "fused_fp32":
            from .deepseek_v41_fused import epilogue_check, expand
            epilogue_check()
            return expand(branch,residual,post,comb)
        return self.round(hc_expand(branch,residual,post,comb,policy=self.hc_policy))

    def qat(self, x, kind):
        if not self.args.simulate_quantization:
            return x
        function = quantize
        if self.qat_policy == "fused_fp32":
            from .deepseek_v41_quant import fused_quantize

            function = fused_quantize
        return self.round(function(x, kind))

    def collapse(self, h, pre):
        return self.round(mx.sum(h.astype(mx.float32) * pre[..., None], -2))

    def engram(self, h, layer, hashes, *, rows=None):
        c, p = self.args, f"blk.{layer}."
        rows = mx.array(self.engram_tables[layer].read(hashes)) if rows is None else rows
        kv = self.linear(rows.reshape(*h.shape[:2], -1), p + "engram_kv.weight")
        key = kv[..., : c.hc_mult * c.hidden_size].reshape(h.shape)
        value = kv[..., c.hc_mult * c.hidden_size :]
        qweight = self.weights.array(p + "engram_q_norm.weight")
        kweight = self.weights.array(p + "engram_k_norm.weight")
        rstd = mx.rsqrt(mx.mean(mx.square(h), -1) + c.rms_norm_eps) * mx.rsqrt(
            mx.mean(mx.square(key), -1) + c.rms_norm_eps
        )
        dot = mx.sum(h * (qweight * kweight) * key, -1) * rstd * c.hidden_size**-0.5
        mag = mx.sqrt(mx.maximum(mx.abs(dot), 1e-6))
        gate = mx.sigmoid(mx.where((dot.view(mx.uint32) >> 31) != 0, -mag, mag))
        return self.round(h + gate[..., None] * value[..., None, :])

    def make_cache(self):
        c = self.args
        cache_class = self.cache_class
        return [
            cache_class(
                r,
                xp=mx,
                identity=self.identity,
                head_dim=c.head_dim,
                index_head_dim=c.index_head_dim,
                window_size=c.sliding_window,
                source=i in c.kv_source_layer_ids,
                history=i == 0 and bool(c.engram_layer_ids),
                compressed_vocab=c.engram_compressed_vocab_size,
            )
            for i, r in enumerate(c.compress_ratios)
        ]

    @property
    def cache_class(self):
        if getattr(self, "kv_storage_policy", "reference") == "native_packed":
            from .deepseek_v41_packed_cache import DeepSeekV41PackedLayerCache

            return DeepSeekV41PackedLayerCache
        return DeepSeekV41LayerCache

    def validate_forward(self, inputs, cache, logits_to_keep=None):
        if self._weights_parked:
            raise ValueError("V4.1 model weights are parked")
        c = self.args
        if (
            inputs.ndim != 2
            or inputs.shape[0] != 1
            or not 0 < inputs.shape[1] <= 1024
            or inputs.dtype not in (mx.int32, mx.uint32, mx.int64)
        ):
            raise ValueError("V4.1 ordinary requires integer B1 tokens, rows1..1024")
        if logits_to_keep is not None and (
            type(logits_to_keep) is not int or logits_to_keep < 0
        ):
            raise ValueError("invalid V4.1 logits_to_keep")
        cache = self.make_cache() if cache is None else cache
        if len(cache) != len(self.layers) or len({id(s) for s in cache}) != len(cache):
            raise ValueError("V4.1 cache layer count/ownership mismatch")
        for i, state in enumerate(cache):
            if (
                type(state) is not self.cache_class
                or state.identity != self.identity
                or state.ratio != c.compress_ratios[i]
                or state.source != (i in c.kv_source_layer_ids)
                or state.history != (i == 0 and bool(c.engram_layer_ids))
                or state.pending
                or state.head_dim != c.head_dim
                or state.index_head_dim != c.index_head_dim
                or state.window_size != c.sliding_window
                or state.compressed_vocab != c.engram_compressed_vocab_size
            ):
                raise ValueError(
                    "V4.1 cache revision/identity/topology/frontier mismatch"
                )
            state.validate_state()
        if (
            len({s.offset for s in cache}) != 1
            or cache[0].offset + inputs.shape[1] > self.context_limit
        ):
            raise ValueError("V4.1 cache prefix or context mismatch")
        ids = np.asarray(inputs)
        if np.any(ids < 0) or np.any(ids >= c.vocab_size):
            raise ValueError("V4.1 token ID outside vocabulary")
        return cache, ids

    def _forward_impl(
        self,
        inputs,
        cache=None,
        *,
        logits_to_keep=None,
        capture_layers=(),
        body_only=False,
        dependency_prefill=False,
        image_inputs=None,
        embedding_override=None,
        image_mask=None,
        head_hidden_only=False,
    ):
        c = self.args
        cache, ids = self.validate_forward(inputs, cache, logits_to_keep)
        if image_inputs is not None:
            if embedding_override is not None or image_mask is not None:
                raise ValueError("V4.1 duplicate visual embedding input")
            prepare = getattr(self, "vision_inputs", None)
            if prepare is None:
                raise ValueError("V4.1 vision input provider unavailable")
            embedding_override, image_mask = prepare(inputs, cache, image_inputs)
        if embedding_override is not None or image_mask is not None:
            if capture_layers or self.ced_policy != "reference" or dependency_prefill:
                raise ValueError("V4.1 vision requires ordinary full-depth prefill without DSpark taps")
            image_mask = np.asarray(image_mask)
            if (embedding_override is None or embedding_override.shape != (1, inputs.shape[1], c.hidden_size)
                    or embedding_override.dtype != mx.float32 or image_mask.dtype != np.bool_
                    or image_mask.shape != (inputs.shape[1],)):
                raise ValueError("V4.1 visual embeddings/mask geometry mismatch")
        if dependency_prefill and (capture_layers or body_only or logits_to_keep != 1):
            raise ValueError("dependency CED requires final-row logits without taps")
        starts = (0,) * len(self.layers)
        base_offset = 0
        if dependency_prefill:
            from .deepseek_v41_ced import dependency_starts, trim_shared

            starts = dependency_starts(c, inputs.shape[1])
            base_offset = cache[0].offset
        begin_frame=getattr(self.weights,'begin_source_frame',None)
        if begin_frame is not None and getattr(self.weights,'source_row_packet_policy','reference')!='reference':
            begin_frame(cache[0].offset,inputs.shape[1])
        hashes = tail = None
        stream = getattr(self.weights, "expert_stream", None)
        if stream is not None and stream.atlas is not None:
            from ..expert_stream_atlas import ExpertStreamAtlas

            stream.begin_context(ExpertStreamAtlas.keys(self.hasher.token_map[ids[0]],
                                                       np.asarray(cache[0].hash_tail)[::-1]))
        if self.hasher is not None:
            hashes, tail = self.hasher(ids[0], np.asarray(cache[0].hash_tail),
                                       **({"mask": ~image_mask} if image_mask is not None else {}))
            for index, layer in enumerate(c.engram_layer_ids):
                self.engram_tables[layer].prefetch(hashes[:, index])
        for state in cache:
            state._in_forward = True
        with self._weight_scope():
            x = self.round(self.weights.embedding(inputs.astype(mx.uint32))
                           if embedding_override is None else embedding_override)
            h = mx.broadcast_to(
                x[:, :, None], (1, inputs.shape[1], c.hc_mult, c.hidden_size)
            )
            pre = mx.broadcast_to(mx.array([1.0] + [0.0] * (c.hc_mult - 1)), h.shape[:-1])
            self._weight_eval(h, pre)
        shared = {}
        if image_mask is not None:
            shared["_image_mask"] = image_mask
        from .deepseek_v41_frame_geometry import frame_geometry
        geometry = frame_geometry(self.weights, inputs.shape[1], self.scheduled_kv_calls)
        if geometry is not None:
            shared["_source_prefill_geometry"] = geometry
        taps = []
        active_start = 0
        for i, (block, state) in enumerate(zip(self.layers, cache)):
            if starts[i] != active_start:
                delta = starts[i] - active_start
                h, pre = h[:, delta:], pre[:, delta:]
                trim_shared(shared, delta)
                active_start = starts[i]
            if active_start:
                # The gap's outputs are outside every future dependency. Its
                # first W-1 query outputs are warmup, discarded by the next
                # layer's shrink. The retained W local keys are all exact.
                state.keys = None
                state.offset = base_offset + active_start
            with self._weight_scope():
                if i in c.engram_layer_ids:
                    enriched = self.engram(h, i, hashes[active_start:, c.engram_layer_ids.index(i)])
                    h = (enriched if image_mask is None else
                         mx.where(mx.array(image_mask)[None, :, None, None], h, enriched))
                if i in capture_layers:
                    taps.append(self.round(mx.mean(h, axis=-2)))
                evaluate_block = getattr(self.weights, "evaluate_block", None)
                if evaluate_block is not None:
                    # Adapter-owned transactions must complete and resolve every
                    # missing weight before another block consumes their output.
                    h, pre = evaluate_block(
                        self, block, h, pre, state, cache, shared, taps=taps
                    )
                else:
                    h, pre = block(h, pre, state, cache, shared)
                    if self.eval_interval and mx.default_device() == mx.gpu:
                        bank = getattr(self.weights, 'source_resident_bank_session', None)
                        bank_decode = (inputs.shape[1] == 1 and bank is not None
                                       and getattr(bank, 'active', False))
                        submission = getattr(bank, 'submission_policy', 'per_layer_async_v1')
                        deferred = (bank_decode and (submission == 'whole_forward_v1'
                                    or submission == 'layer_group_2_v1' and (i + 1) % 2 != 0
                                    or submission == 'layer_group_4_v1' and (i + 1) % 4 != 0))
                        if bank_decode and isinstance(getattr(bank, 'stats', None), dict):
                            counter = 'layer_async_deferred' if deferred else 'layer_async_submitted'
                            bank.stats[counter] = bank.stats.get(counter, 0) + 1
                        if not deferred:
                            mx.async_eval(h, pre)
                    self._weight_eval(h, pre, [v.state for v in cache], shared, taps)
        if dependency_prefill:
            self.scheduled_kv_calls["ced_prefill_calls"] = self.scheduled_kv_calls.get("ced_prefill_calls", 0) + 1
            self.scheduled_kv_calls["ced_skipped_layer_rows"] = self.scheduled_kv_calls.get("ced_skipped_layer_rows", 0) + sum(starts)
        if tail is not None:
            cache[0].hash_tail = mx.array(tail, mx.int32)
        features = mx.concatenate(taps, -1) if taps else None
        with self._weight_scope():
            hidden = self.norm(self.collapse(h, pre), self.norm_weight)
            if dependency_prefill:
                from ..prefill_checkpoint_rows import (
                    select_checkpoint_rows_from_final_suffix,
                )

                hidden = select_checkpoint_rows_from_final_suffix(hidden, full_width=inputs.shape[1])
            else:
                hidden = (
                select_checkpoint_rows(hidden)
                if logits_to_keep is None
                else hidden[:, -logits_to_keep:]
                if logits_to_keep
                else hidden
                )
            logits = (hidden if head_hidden_only else None if body_only else
                      self.linear(hidden, "output.weight", rounded=False))
            # Include the current frontier after the Engram tail assignment;
            # deferred adapter work must finish every branch before publication.
            self._weight_eval(logits, features, hidden, [v.state for v in cache], shared, taps)
        for state in cache:
            state.validate_state()
        for state in cache:
            state._in_forward = False
        if stream is not None:
            stream.end_context()
        return logits, features

    def _forward(self, inputs, cache=None, *, logits_to_keep=None, capture_layers=(), body_only=False,
                 dependency_prefill=False, image_inputs=None, embedding_override=None, image_mask=None,
                 head_hidden_only=False):
        if (getattr(self,'_mlx2_ds41_ast_active',None) is not None
                or getattr(self,'_mlx2_ds41_ast_poisoned',False)):
            raise ValueError('Public Source forward cannot enter an active or poisoned AST transaction')
        if getattr(self.weights,'_source_current_frame',None) is not None:
            raise ValueError('Source model cannot enter a nested forward frame')
        bank_session=getattr(self.weights,'source_resident_bank_session',None)
        if bank_session is not None and getattr(self, "kv_storage_policy", "reference") == "native_packed":
            raise ValueError("native packed KV cannot enter a reference-cache bank transaction")
        if bank_session is not None and (bank_session.active or bank_session.replaying):
            raise ValueError('Public Source forward cannot reenter an active bank transaction')
        options = {"logits_to_keep": logits_to_keep, "capture_layers": capture_layers, "body_only": body_only,
                   "dependency_prefill": dependency_prefill, "image_inputs": image_inputs,
                   "embedding_override": embedding_override, "image_mask": image_mask,
                   "head_hidden_only": head_hidden_only}
        if not body_only and bank_session is not None and bank_session.selected(inputs,cache):
            return bank_session.forward(self,inputs,cache,
                lambda private:self._forward_ordinary(inputs,private,**options))
        return self._forward_ordinary(inputs,cache,**options)

    def _forward_ordinary(self, inputs, cache=None, *, logits_to_keep=None, capture_layers=(), body_only=False,
                          dependency_prefill=False, image_inputs=None, embedding_override=None, image_mask=None,
                          head_hidden_only=False):
        if (getattr(self,'_mlx2_ds41_ast_active',None) is not None
                or getattr(self,'_mlx2_ds41_ast_poisoned',False)):
            raise ValueError('Ordinary Source forward cannot enter an active or poisoned AST transaction')
        # Private bank execution/replay enters the ordinary implementation
        # directly, so diagnostic wrappers around _forward run exactly once.
        if getattr(self.weights,'_source_current_frame',None) is not None:
            raise ValueError('Source model cannot enter a nested forward frame')
        self._weight_executions += 1
        try:
            execution = getattr(self.weights, "execution_scope", None)
            rows_execution = getattr(self.weights, "execution_scope_for_rows", None)
            scope = (rows_execution(inputs.shape[1]) if callable(rows_execution)
                     else execution() if execution is not None else nullcontext())
            with scope:
                return self._forward_impl(inputs, cache, logits_to_keep=logits_to_keep,
                                          capture_layers=capture_layers, body_only=body_only,
                                          dependency_prefill=dependency_prefill, image_inputs=image_inputs,
                                          embedding_override=embedding_override, image_mask=image_mask,
                                          head_hidden_only=head_hidden_only)
        except BaseException as error:
            discard = getattr(self, "discard_engram_prefetch", None)
            if discard is not None:
                discard(error)
            raise
        finally:
            end_frame=getattr(self.weights,'end_source_frame',None)
            if end_frame is not None:
                end_frame()
            self._weight_executions -= 1

    def discard_engram_prefetch(self, error=None):
        """A forward that raised before its Engram reads must not leave a
        prefetch window owned; later forwards and park() would refuse."""
        for table in (getattr(self, "engram_tables", None) or {}).values():
            discard = getattr(table, "discard_prefetch", None)
            if discard is None:
                continue
            try:
                discard()
            except BaseException as cleanup:  # noqa: BLE001 - keep the forward's failure
                if error is None:
                    raise
                error.add_note("Engram prefetch discard also failed: " + repr(cleanup))

    def __call__(self, inputs, cache=None, *, logits_to_keep=None, image_inputs=None):
        return self._forward(inputs, cache, logits_to_keep=logits_to_keep,
                             **({"image_inputs": image_inputs} if image_inputs is not None else {}))[0]

    def forward_with_taps(self, inputs, cache, capture_layers, *, body_only=False):
        capture_layers = tuple(capture_layers)
        if (
            not capture_layers
            or tuple(sorted(set(capture_layers))) != capture_layers
            or any(
                type(i) is not int or not 0 <= i < len(self.layers)
                for i in capture_layers
            )
        ):
            raise ValueError("V4.1 target taps require ordered valid layer IDs")
        return self._forward(
            inputs, cache, capture_layers=capture_layers, body_only=body_only
        )

    def prefill_body(self, inputs, cache, capture_layers):
        from ..gguf.prefill_scope import dense_prefill_context
        from ..gguf.segmented_prefill import expert_prefill_context

        with (
            dense_prefill_context(
                self.weights.dense_prefill_policy, tokens=inputs.shape[1]
            ),
            expert_prefill_context(
                getattr(self.weights, "expert_prefill_policy", "reference"),
                tokens=inputs.shape[1],
            ),
        ):
            return self.forward_with_taps(
                inputs, cache, capture_layers, body_only=True
            )[1]

    def prefill_forward(self, inputs, cache=None, *, image_inputs=None):
        from ..gguf.prefill_scope import dense_prefill_context
        from ..gguf.segmented_prefill import expert_prefill_context

        if image_inputs is not None:
            if self.ced_policy != "reference":
                raise ValueError("V4.1 vision requires ordinary full-depth prefill")
            if (inputs.ndim != 2 or inputs.shape[0] != 1 or not inputs.shape[1]
                    or inputs.shape[1] > self.context_limit):
                raise ValueError("invalid V4.1 visual prompt geometry")
            cache = self.make_cache() if cache is None else cache
            prepare = getattr(self, "vision_inputs", None)
            if prepare is None:
                raise ValueError("V4.1 vision input provider unavailable")
            embeddings, image_mask = prepare(inputs, cache, image_inputs)
            if (embeddings.shape != (1, inputs.shape[1], self.args.hidden_size)
                    or embeddings.dtype != mx.float32 or np.asarray(image_mask).dtype != np.bool_
                    or np.asarray(image_mask).shape != (inputs.shape[1],)):
                raise ValueError("V4.1 visual embeddings/mask geometry mismatch")
            image_mask = np.asarray(image_mask)
            hidden = []
            for first in range(0, inputs.shape[1], 1024):
                last = min(inputs.shape[1], first + 1024)
                with dense_prefill_context(self.weights.dense_prefill_policy, tokens=last-first), expert_prefill_context(
                    getattr(self.weights, "expert_prefill_policy", "reference"), tokens=last-first
                ):
                    value, _ = self._forward(inputs[:, first:last], cache, logits_to_keep=0,
                                             embedding_override=embeddings[:, first:last],
                                             image_mask=image_mask[first:last], head_hidden_only=True)
                    hidden.append(value)
            with self._weight_scope():
                output = self.linear(select_checkpoint_rows(mx.concatenate(hidden, 1)), "output.weight", rounded=False)
                self._weight_eval(output, [v.state for v in cache])
            return output
        with (
            dense_prefill_context(
                self.weights.dense_prefill_policy, tokens=inputs.shape[1]
            ),
            expert_prefill_context(
                getattr(self.weights, "expert_prefill_policy", "reference"),
                tokens=inputs.shape[1],
            ),
        ):
            if self.ced_policy == "dependency_exact":
                return self._forward(inputs, cache, logits_to_keep=1, dependency_prefill=True)[0]
            return self(inputs, cache)

    def verify_rows(self, inputs, cache, capture_layers=()):
        if not 1 <= inputs.shape[1] <= 6:
            raise ValueError(
                "V4.1 verifier supports a B1 ordinary chain of1..6"
            )
        if capture_layers:
            values = [self.forward_with_taps(inputs[:, i:i+1], cache, capture_layers)
                      for i in range(inputs.shape[1])]
            return mx.concatenate([v[0] for v in values], 1), mx.concatenate([v[1] for v in values], 1)
        if getattr(self, "verify_rows_policy", "serial") == "layerwise":
            return self.verify_layerwise(inputs, cache)[0], None
        return mx.concatenate(
            [
                self(inputs[:, i : i + 1], cache, logits_to_keep=0)
                for i in range(inputs.shape[1])
            ],
            1,
        ), None

    def verify_layerwise(self, inputs, cache, *, retain_frontiers=False):
        self._weight_executions += 1
        try:
            return self._verify_layerwise_impl(inputs, cache, retain_frontiers=retain_frontiers)
        except BaseException as error:
            discard = getattr(self, "discard_engram_prefetch", None)
            if discard is not None:
                discard(error)
            raise
        finally:
            self._weight_executions -= 1

    def _verify_layerwise_impl(self, inputs, cache, *, retain_frontiers=False):
        """Layer-major chain: width-one attention/dense math, grouped exact MoE.

        Each row sees precisely its ordinary source-pool extent. Grouping only
        the independent expert projections removes repeated routing barriers
        without changing causal state, reductions or rejected-prefix commits.
        """
        from copy import copy

        from .deepseek_v4_cache import _copy

        if getattr(self, "kv_storage_policy", "reference") == "native_packed" and retain_frontiers:
            raise ValueError("native packed KV requires snapshot/replay verification commits")

        if not 1 <= inputs.shape[1] <= 6:
            raise ValueError("layerwise verification requires one to six rows")
        cache, ids = self.validate_forward(inputs, cache, 0)
        c, width = self.args, inputs.shape[1]
        hashes, tails = None, []
        if self.hasher:
            tail = np.asarray(cache[0].hash_tail)
            hashes, _ = self.hasher(ids[0], tail)
            for token in ids[0]:
                _, tail = self.hasher(np.asarray([token]), tail)
                tails.append(mx.array(tail, mx.int32))
            for j, layer in enumerate(c.engram_layer_ids):
                self.engram_tables[layer].prefetch(hashes[:, j])
        stream = getattr(self.weights, "expert_stream", None)
        if stream is not None and stream.atlas is not None:
            from ..expert_stream_atlas import ExpertStreamAtlas
            stream.begin_context(ExpertStreamAtlas.keys(self.hasher.token_map[ids[0]],
                                                       np.asarray(cache[0].hash_tail)[::-1]))
        hidden, pre = [], []
        with self._weight_scope():
            for row in range(width):
                x = self.round(self.weights.embedding(inputs[:, row:row+1].astype(mx.uint32)))
                hidden.append(mx.broadcast_to(x[:, :, None], (1, 1, c.hc_mult, c.hidden_size)))
                pre.append(mx.broadcast_to(mx.array([1.0]+[0.0]*(c.hc_mult-1)), (1, 1, c.hc_mult)))
            self._weight_eval(hidden, pre)
        shared = [{} for _ in range(width)]
        frontiers = [[] for _ in range(width-1)] if retain_frontiers else []
        for state in cache:
            state._in_forward = True
        for layer, (block, state) in enumerate(zip(self.layers, cache)):
            with self._weight_scope():
                engram = None
                if layer in c.engram_layer_ids:
                    index = c.engram_layer_ids.index(layer)
                    engram = mx.array(self.engram_tables[layer].read(hashes[:, index]))
                staged, routes, scores = [], [], []
                for row in range(width):
                    if engram is not None:
                        hidden[row] = self.engram(hidden[row], layer, None, rows=engram[row:row+1])
                    visible = cache
                    if state.ratio and not state.source:
                        owner_index = c.kv_source_for(layer)
                        owner = copy(cache[owner_index])
                        owner.offset = state.offset+1
                        owner._pool_count = owner.offset//owner.ratio
                        visible = list(cache)
                        visible[owner_index] = owner
                    h, x, pre[row], post, comb = block.attention_half(hidden[row], pre[row], state, visible, shared[row])
                    indices, weight = block.router(x)
                    staged.append((h, x, post, comb))
                    routes.append(indices.reshape(1, c.num_experts_per_tok))
                    scores.append(weight.reshape(1, c.num_experts_per_tok))
                    if layer == 0 and tails:
                        state.hash_tail = tails[row]
                    if retain_frontiers and row < width-1:
                        state.validate_state()
                        local = _copy(mx, (state.keys, state.buf_kv, state.buf_gate, state.hash_tail))
                        frontiers[row].append((state.meta_state, local, state._pool_count, state._pooled is not None))
                indices, score = mx.concatenate(routes), mx.concatenate(scores)
                x = mx.concatenate([item[1].reshape(1, c.hidden_size) for item in staged])
                overlap = stream is not None and getattr(self.weights, "expert_io_policy", "blocking") == "overlap"
                ticket = stream.prepare(block.prefix[:-1], indices) if overlap else None
                try:
                    shared_out = [block.shared_expert(item[1]) for item in staged]
                    if overlap:
                        mx.async_eval(shared_out)
                    routed = self.weights.moe(x, indices, score, block.prefix[:-1],
                                              **({"ticket": ticket} if ticket is not None else {}))
                except BaseException:
                    if ticket is not None:
                        stream.cancel(ticket)
                    raise
                for row, (h, _, post, comb) in enumerate(staged):
                    hidden[row] = block.finish(h, routed[row:row+1][None], shared_out[row], post, comb)
                if self.eval_interval and mx.default_device() == mx.gpu:
                    mx.async_eval(hidden, pre)
                self._weight_eval(hidden, pre, [v.state for v in cache], shared, frontiers)
        with self._weight_scope():
            logits = [self.linear(self.norm(self.collapse(h, p), self.norm_weight), "output.weight", rounded=False)
                      for h, p in zip(hidden, pre)]
            self._weight_eval(logits)
        for state in cache:
            state.validate_state()
            state._in_forward = False
        stats = getattr(self, "verified_commit_stats", None)
        if stats is not None:
            stats["layerwise_calls"] = stats.get("layerwise_calls", 0)+1
            stats["layerwise_rows"] = stats.get("layerwise_rows", 0)+width
        if stream is not None:
            stream.end_context()
        return mx.concatenate(logits, 1), frontiers

    def named_modules(self):
        yield "", self

    def parameters(self):
        native = getattr(self.weights, "_dense", {})
        packed = getattr(self.weights, "_packed", {})
        linears = getattr(self.weights, "native_fp32_parameters", dict)()
        return {
            "packed": {k: v.data for k, v in packed.items()},
            "native": native,
            "linears": linears,
            "hc": {str(i): block.hc for i, block in enumerate(self.layers)},
            "rotary": {
                str(i): block.attn.rope.inv for i, block in enumerate(self.layers)
            },
        }

    def preload(self):
        if self._weights_parked:raise ValueError("V4.1 model weights are parked")
        for name, tensor in self.weights.tensors.items():
            if ".engram_embd." in name:
                continue
            pages = getattr(self.weights, "dense_pages", None)
            if pages is not None and name in pages.pages:
                continue
            if tensor.ggml_type == 19 and getattr(self.weights, "expert_stream", None) is not None:
                continue
            if name == "token_embd.weight":
                self.weights.prepare_embedding(name)
            elif tensor.ggml_type in (8, 19):
                self.weights.packed(name)
            elif len(tensor.shape) == 2 and not any(
                part in name for part in ("hc_", "engram_q_norm", "engram_k_norm")
            ):
                self.weights.linear(name)
            else:
                self.weights.array(name)
        self.weights.preload_native_linear_fp32()
        return self

    def eval(self):
        mx.eval(self.parameters())
        return self
