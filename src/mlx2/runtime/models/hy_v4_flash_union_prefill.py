# SPDX-License-Identifier: Apache-2.0
"""Hy4-flash union prefill for indexed layers (opt-in ``sparse_prefill_union_rows``).

Original mlx2 code; the mechanism is Maxwell's long-chunk ``union`` route
(``maxwell_moe.Attention._sparse_union``: per query sub-chunk, gather the
union of the routed blocks once and attend it with a membership mask).

Audit H4 (2026-11-22).  The sparse prefill kernel
(``hy_v4_flash_sparse_prefill``) runs one threadgroup per query row, and each
row fetches its ``K = 2048`` recalled latent rows on its own: about 4.8 GB
of row reads per indexed layer per 2048-row chunk, although neighbouring
queries recall mostly the same groups.  This route splits a chunk into
sub-chunks of ``rows`` queries, gathers the union of their recalled rows
once (dequantizing only those rows for fp8 caches), and runs the fused SDPA
(with sinks) over the union with a ``[rows, U]`` mask: query ``l`` sees
exactly its recalled ids at or before its position, the dense reference's
key set (``hy_v4_flash.dense_indexed_attend``; duplicates count once there
and here).  Masked keys contribute exactly zero, but the softmax and value
reductions run over ``U`` instead of ``T`` keys, so the result is
numerically equivalent to the dense path, not bit-identical (each
``rows`` value is its own APCv2 namespace).

One host read per layer chunk (the union sizes), like Maxwell.  Lanes run
one at a time (``B == 1``); indexed caches start at position 0
(``first == 0``), otherwise the forward declines with a counted reason.
GPU A/B pending: it trades per-row reads for SDPA over the union, which wins
only while the union of ``rows`` queries stays well under ``rows * K``.
"""

from __future__ import annotations

import mlx.core as mx

STATS = {"union_forwards": 0, "union_subchunks": 0, "union_rows": 0, "recalled_rows": 0}


def union_stats(*, reset: bool = False) -> dict:
    """Host-side counters: forwards, sub-chunks, union rows gathered and the
    per-row recalled slots they replace (``recalled_rows / union_rows`` is the
    read reduction)."""
    snapshot = dict(STATS)
    if reset:
        for key in STATS:
            STATS[key] = 0
    return snapshot


def unsupported_reason(*, lanes: int, first: int) -> str | None:
    if lanes != 1:
        return "lanes"
    if first != 0:
        return "kv_first_nonzero"
    return None


def union_indexed_attend(q, kv, recalled, sinks, scale, rank, qpos, rows_per_union: int, *,
                         attend, rows=None, bounded: bool = True, budget: int = 1 << 27):
    """``q [1, H, L, D]``, stored rows ``kv [1, T, row]``, ``recalled [1, L, K]``
    ids (-1 empty), ``qpos [L]`` -> ``[1, H, L, rank]``.

    ``attend`` is ``hy_v4_flash.mla_attend``; ``rows`` turns gathered stored
    rows into attention rows (fp8 caches dequantize only the union).  With
    ``bounded``, pending sub-chunks are evaluated once their score elements
    would pass ``budget`` (the model's ``SCORE_BUDGET`` contract).
    """
    _, H, L, _ = q.shape
    T = int(kv.shape[1])
    K = int(recalled.shape[-1])
    step = max(1, min(int(rows_per_union), L))
    n = -(-L // step)
    rec = recalled[0]
    pad = n * step - L
    if pad:
        rec = mx.concatenate([rec, mx.full((pad, K), -1, dtype=rec.dtype)], axis=0)
    causal = rec <= qpos_pad(qpos, pad)[:, None]
    slot = mx.where((rec >= 0) & causal & (rec < T), rec, T)          # [n*step, K]
    member = mx.zeros((n, T + 1), dtype=mx.bool_)
    member = mx.put_along_axis(member, slot.reshape(n, step * K), mx.array(True), axis=-1)[:, :T]
    counts = member.sum(axis=-1).tolist()                              # one host read
    ids = mx.arange(T, dtype=mx.int32)
    order = mx.argsort(mx.where(member, ids, T + ids), axis=-1).astype(mx.int32)  # members first, ascending
    STATS["union_forwards"] += 1
    outs, pending, live = [], [], 0
    for i in range(n):
        s0, s1 = i * step, min(L, (i + 1) * step)
        u = int(counts[i])
        STATS["union_subchunks"] += 1
        STATS["union_rows"] += u
        STATS["recalled_rows"] += (s1 - s0) * K
        if u == 0:
            outs.append(mx.zeros((1, H, s1 - s0, rank), dtype=q.dtype))
            continue
        elements = H * (s1 - s0) * u
        if bounded and pending and live + elements > budget:
            mx.eval(pending)
            pending, live = [], 0
        tok = order[i, :u]                                             # [U] ascending ids
        keys = mx.take(kv[0], tok, axis=0)                             # [U, row]
        if rows is not None:
            keys = rows(keys)
        hit = mx.zeros((s1 - s0, T + 1), dtype=mx.bool_)
        hit = mx.put_along_axis(hit, slot[s0:s1], mx.array(True), axis=-1)
        mask = mx.take(hit, tok, axis=-1)                              # [c, U]: hit & causal & in range
        out = attend(q[:, :, s0:s1], keys[None, None], mask[None, None], sinks, scale, rank)
        outs.append(out)
        pending.append(out)
        live += elements
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


def qpos_pad(qpos, pad: int):
    if not pad:
        return qpos
    return mx.concatenate([qpos, mx.full((pad,), -1, dtype=qpos.dtype)])
