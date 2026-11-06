# SPDX-License-Identifier: MIT
# Indexer math ported from higa2515/Qwen3.8-27B-QSA-BF16 research/qsa
# (Apache-2.0); see provenance/qsa-sidecar-27b.json.
"""Research QSA sidecar: learned block indexers on a dense full-attention model.

A QSA sidecar adds a small learned indexer to each full-attention layer of an
otherwise unchanged checkpoint.  The indexer scores 4-token key blocks; every
query attends to its top ``budget / block_size`` complete blocks plus the
tokens of its own unfinished block.  The KV cache is unchanged.

Indexer (per layer, from the reference ``QSAIndexer``):

* ``q = rope64(rmsnorm_zc(x @ Wq^T))`` with 4 heads of 128;
* ``k = rmsnorm_zc(x @ Wk^T)`` (one shared head), mean-pooled over each
  complete block, ``rmsnorm_zc`` again, then rope64 at the block start;
* ``score[t, j] = sum_h relu(q[t, h] . kblock[j])`` over blocks with
  ``4j + 3 <= t``.

``rmsnorm_zc`` scales by ``1 + w`` (zero-centred).  RoPE is the main
attention's: first 64 channels, non-traditional, base 1e7.  Scores run in
float32, as the reference does.

This module is a research probe, not a serving route.  It refuses batches,
rewinds and mixed packing instead of guessing; serving integration (APCv2
state, batching, MTP) is separate work.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import mlx.core as mx

BLOCK = 4
KEY_BLOCKS_MIN_CAPACITY = 256  # pooled-key rows first allocated per layer
MODES = ("dense", "selfdense", "window", "quest", "mask", "nax", "gather")

STATUS: Counter = Counter()


class QSASidecarRefused(RuntimeError):
    """The sidecar cannot serve this call shape exactly."""


def _zc_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    return mx.fast.rms_norm(x, 1.0 + weight, eps)


def _rope(x: mx.array, *, dims: int, base: float, offset: int, scale: float = 1.0):
    # x [..., L, D]: positions are ``scale * (offset + i)`` along axis -2.
    return mx.fast.rope(
        x, dims, traditional=False, base=base, scale=scale, offset=offset
    )


class LayerIndexer:
    """One layer's indexer weights plus its incremental pooled-key state."""

    def __init__(self, weights: dict, *, heads: int, head_dim: int, rope_dim: int,
                 rope_base: float, eps: float):
        self.wq = weights["q_proj.weight"].astype(mx.float32)
        self.wk = weights["k_proj.weight"].astype(mx.float32)
        self.q_norm = weights["q_norm"].astype(mx.float32)
        self.k_norm = weights["k_norm"].astype(mx.float32)
        self.k_block_norm = weights["k_block_norm"].astype(mx.float32)
        # ``rmsnorm_zc`` scales by ``1 + w``: form it once, not on every call.
        self.q_scale = 1.0 + self.q_norm
        self.k_scale = 1.0 + self.k_norm
        self.k_block_scale = 1.0 + self.k_block_norm
        self.heads = heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.rope_base = rope_base
        self.eps = eps
        if self.wq.shape[0] != heads * head_dim or self.wk.shape[0] != head_dim:
            raise ValueError("indexer weights do not match the declared geometry")
        self.reset()

    def reset(self):
        self.tokens = 0  # positions folded into blocks + tail
        self.n_blocks = 0
        # [capacity, D] pooled, normed, roped keys; rows [:n_blocks] are live.
        # Capacity grows geometrically and appends write in place, so a decode
        # step no longer re-copies the whole pooled history.
        self._kbuf: Optional[mx.array] = None
        self.tail: Optional[mx.array] = None  # [<4, D] normed raw keys

    @property
    def blocks(self) -> Optional[mx.array]:
        """[nb, D] pooled, normed, roped block keys (a view of the buffer)."""
        if not self.n_blocks:
            return None
        return self._kbuf[: self.n_blocks]

    def queries(self, x: mx.array, offset: int) -> mx.array:
        """``x`` [L, hidden] -> roped queries [H, L, D] (float32)."""
        length = x.shape[0]
        q = (x @ self.wq.T).reshape(length, self.heads, self.head_dim)
        q = mx.fast.rms_norm(q, self.q_scale, self.eps).transpose(1, 0, 2)
        return _rope(q, dims=self.rope_dim, base=self.rope_base, offset=offset)

    def append_keys(self, x: mx.array, offset: int) -> None:
        if offset != self.tokens:
            raise QSASidecarRefused(
                f"indexer state holds {self.tokens} positions, call starts at {offset}"
            )
        k = mx.fast.rms_norm(x @ self.wk.T, self.k_scale, self.eps)
        if self.tail is not None:
            k = mx.concatenate([self.tail, k], axis=0)
        first_block = self.tokens // BLOCK
        complete = k.shape[0] // BLOCK
        if complete:
            pooled = k[: complete * BLOCK].reshape(complete, BLOCK, -1).mean(axis=1)
            pooled = mx.fast.rms_norm(pooled, self.k_block_scale, self.eps)
            # block j starts at position 4j: rope position = 4 * (first + i).
            pooled = _rope(
                pooled[None], dims=self.rope_dim, base=self.rope_base,
                offset=first_block, scale=float(BLOCK),
            )[0]
            self._store_blocks(pooled)
        rest = k[complete * BLOCK:]
        self.tail = rest if rest.shape[0] else None
        self.tokens += x.shape[0]

    def _store_blocks(self, pooled: mx.array) -> None:
        start = self.n_blocks
        stop = start + pooled.shape[0]
        capacity = 0 if self._kbuf is None else self._kbuf.shape[0]
        if stop > capacity:
            # Doubling: every pooled row is re-copied O(1) times amortized.
            grown = max(stop, 2 * capacity, KEY_BLOCKS_MIN_CAPACITY)
            parts = [pooled, mx.zeros((grown - stop, pooled.shape[-1]), pooled.dtype)]
            if start:
                parts.insert(0, self._kbuf[:start])
            self._kbuf = mx.concatenate(parts, axis=0)
        else:
            # Slice update: donated (in place) once the previous step's views
            # of the buffer have been evaluated and released.
            self._kbuf[start:stop] = pooled
        self.n_blocks = stop

    def scores(self, q: mx.array, q_chunk: int = 512) -> mx.array:
        """[H, L, D] queries -> [L, nb] block scores (invalid blocks not masked)."""
        kb = self.blocks
        out = []
        for start in range(0, q.shape[1], q_chunk):
            part = q[:, start:start + q_chunk]
            acc = None
            for h in range(self.heads):
                s = mx.maximum(part[h] @ kb.T, 0.0)
                acc = s if acc is None else acc + s
            out.append(acc)
        return out[0] if len(out) == 1 else mx.concatenate(out, axis=0)


class QSASidecar:
    """All layers' indexers plus run mode, budget and receipts."""

    def __init__(self, path: str | os.PathLike, *, budget: int = 2048,
                 mode: str = "nax", rope_base: float = 1e7, rope_dim: int = 64,
                 heads: int = 4, head_dim: int = 128, eps: float = 1e-6):
        if mode not in MODES:
            raise ValueError(f"QSA sidecar mode must be one of {MODES}")
        if budget % BLOCK:
            raise ValueError("budget must be a multiple of the block size")
        path = Path(path)
        file = path / "qsa-indexers.safetensors" if path.is_dir() else path
        raw = mx.load(str(file))
        layers: dict[int, dict] = {}
        for name, value in raw.items():
            parts = name.split(".")
            layer = int(parts[parts.index("layers") + 1])
            key = ".".join(parts[parts.index("_qsa_indexer") + 1:])
            layers.setdefault(layer, {})[key] = value
        self.indexers = {
            layer: LayerIndexer(w, heads=heads, head_dim=head_dim, rope_dim=rope_dim,
                                rope_base=rope_base, eps=eps)
            for layer, w in sorted(layers.items())
        }
        self.file = str(file)
        self.budget = budget
        self.block_topk = budget // BLOCK
        self.mode = mode
        # Fused Metal selection (qsa_select.py) instead of MLX score/argpartition.
        self.fused_select = False
        # Training-free Quest control (mode "quest"): window, sink, budget, page.
        self.quest = {"window": 256, "sink": 256, "budget": 1024, "page": 32}

    def reset(self):
        for indexer in self.indexers.values():
            indexer.reset()

    def identity(self) -> dict:
        return {"file": self.file, "layers": sorted(self.indexers),
                "budget": self.budget, "mode": self.mode,
                "fused_select": self.fused_select}


def _selection(indexer: LayerIndexer, x: mx.array, offset: int, topk: int,
               fused: bool = False):
    """Top-k complete blocks per query, or ``None`` when selection is dense.

    Returns ``(ids [L, k] int32, valid [L, k] bool, q_pos [L] int32)``.
    """
    length = x.shape[0]
    indexer.append_keys(x, offset)
    n_blocks = indexer.n_blocks
    if n_blocks <= topk:
        return None  # every complete block selected: plain causal attention
    q = indexer.queries(x, offset)
    q_pos = mx.arange(offset, offset + length, dtype=mx.int32)
    if fused:
        from .qsa_select import fused_select

        (ids, counts) = fused_select(q, indexer.blocks, q_pos, topk)
        picked_valid = mx.arange(topk, dtype=mx.int32)[None, :] < counts[:, None]
        return ids, picked_valid, q_pos, n_blocks, counts
    scores = indexer.scores(q)  # [L, nb]
    starts = mx.arange(n_blocks, dtype=mx.int32) * BLOCK
    valid = (starts[None, :] + (BLOCK - 1)) <= q_pos[:, None]
    scores = mx.where(valid, scores, -mx.inf)
    ids = mx.argpartition(-scores, kth=topk - 1, axis=-1)[:, :topk].astype(mx.int32)
    picked_valid = mx.take_along_axis(valid, ids, axis=-1)
    return ids, picked_valid, q_pos, n_blocks, None


def _dense_mask(ids, picked_valid, q_pos, n_blocks, total):
    """Reference boolean visibility [1, 1, L, total] for a selection."""
    length = ids.shape[0]
    onehot = mx.zeros((length, n_blocks), dtype=mx.bool_)
    onehot = mx.put_along_axis(onehot, ids, picked_valid, axis=-1)
    token = mx.arange(total, dtype=mx.int32)
    block_of = token // BLOCK
    in_complete = block_of < n_blocks
    sel = mx.take(onehot, mx.minimum(block_of, n_blocks - 1), axis=1) & in_complete[None]
    causal = token[None, :] <= q_pos[:, None]
    tail = (block_of[None, :] == (q_pos // BLOCK)[:, None]) & causal
    return ((sel | tail) & causal)[None, None]


def _quest_mask(queries, keys, offset, *, window, sink, budget, page, q_chunk=512):
    """Training-free Quest selection (arXiv 2406.10774) with a sink and a recent
    window, as in arXiv 2611.06133; implemented from the papers, no code copied.

    Each complete ``page``-token page of cached (roped) keys is summarised by
    its per-channel min and max.  The page bound for a query head is
    ``sum_c max(q_c * min_c, q_c * max_c) = relu(q) . max + min(q, 0) . min``;
    it is summed over every query head (so over KV heads), giving one shared
    page set per query token.  The query sees the union of the first ``sink``
    tokens, the last ``window`` tokens and the top ``budget // page`` pages
    (pages lying wholly inside the sink or window are not candidates, so the
    budget buys new tokens).  Selection is refreshed on every call.

    Returns boolean visibility ``[1, 1, L, total]``.
    """
    (_, n_heads, length, dim) = queries.shape
    (_, n_kv, total, _) = keys.shape
    group = n_heads // n_kv
    n_pages = total // page
    q_pos = mx.arange(offset, offset + length, dtype=mx.int32)
    token = mx.arange(total, dtype=mx.int32)
    base = (token[None, :] < sink) | (token[None, :] > (q_pos[:, None] - window))
    if n_pages == 0 or budget <= 0:
        vis = base
    else:
        kp = keys[0, :, : n_pages * page].astype(mx.float32).reshape(n_kv, n_pages, page, dim)
        (kmin, kmax) = (kp.min(axis=2), kp.max(axis=2))  # [n_kv, P, D]
        q = queries[0].astype(mx.float32).reshape(n_kv, group, length, dim)
        pos = mx.maximum(q, 0.0).sum(axis=1)  # [n_kv, L, D]
        neg = mx.minimum(q, 0.0).sum(axis=1)
        starts = mx.arange(n_pages, dtype=mx.int32) * page
        ends = starts + page - 1
        top = min(n_pages, budget // page)
        rows = []
        for start in range(0, length, q_chunk):
            stop = min(start + q_chunk, length)
            qp = q_pos[start:stop, None]
            score = None
            for g in range(n_kv):
                part = pos[g, start:stop] @ kmax[g].T + neg[g, start:stop] @ kmin[g].T
                score = part if score is None else score + part
            candidate = (ends[None, :] <= qp) & (ends[None, :] >= sink) & (
                starts[None, :] <= qp - window
            )
            score = mx.where(candidate, score, -mx.inf)
            ids = mx.argpartition(-score, kth=top - 1, axis=-1)[:, :top]
            ok = mx.take_along_axis(candidate, ids, axis=-1)
            chosen = mx.put_along_axis(
                mx.zeros((stop - start, n_pages), dtype=mx.bool_), ids, ok, axis=-1
            )
            page_of = mx.minimum(token // page, n_pages - 1)
            in_page = (token < n_pages * page)[None, :] & mx.take(chosen, page_of, axis=1)
            rows.append(in_page)
        vis = base | (rows[0] if len(rows) == 1 else mx.concatenate(rows, axis=0))
    causal = token[None, :] <= q_pos[:, None]
    return (vis & causal)[None, None]


def _compact(ids, picked_valid, q_pos, total, counts=None):
    from .qwen4_exp import QSACompactBlocks, _compact_qsa_block_ids

    if counts is not None:  # fused selection: already sorted and packed
        (block_ids, counts) = (ids[None], counts[None])
    else:
        (block_ids, counts) = _compact_qsa_block_ids(
            ids[None], picked_valid[None], n_blocks=total // BLOCK
        )
    tail_stop = (q_pos + 1)[None]
    return QSACompactBlocks(
        block_ids=block_ids, block_counts=counts,
        tail_start=tail_stop // BLOCK * BLOCK, tail_stop=tail_stop,
        left_padding=None, block_size=BLOCK, physical_width=total, causal_mask=None,
    )


def _nax(queries, keys, values, compact, scale):
    from .qwen4_qsa_nax import compact_blocks_to_kernel_inputs, nax_qsa_attention

    (ids, counts, n_sel, u_width, q_pos, left_pad, total) = (
        compact_blocks_to_kernel_inputs(compact)
    )
    out = nax_qsa_attention(
        queries, keys, values, ids, counts, n_sel, q_pos, left_pad,
        scale=scale, u_width=u_width, total=total, n_kv_heads=keys.shape[1],
    )
    return out.astype(queries.dtype)


def _gather(queries, keys, values, ids, picked_valid, q_pos, scale):
    """Decode-only: one shared selection (L == 1) gathered, then plain SDPA."""
    if queries.shape[2] != 1:
        raise QSASidecarRefused("gather mode serves single-token decode only")
    total = keys.shape[2]
    tail_start = (q_pos[0] // BLOCK) * BLOCK
    block_tok = (ids[0][:, None] * BLOCK + mx.arange(BLOCK, dtype=mx.int32)).reshape(-1)
    tail_tok = tail_start + mx.arange(BLOCK, dtype=mx.int32)
    tokens = mx.concatenate([block_tok, tail_tok])
    # A query at a block end owns a complete block that may also be selected:
    # drop the tail copy then, so no key is counted twice.
    own = q_pos[0] // BLOCK
    dup = mx.any((ids[0] == own) & picked_valid[0])
    ok = mx.concatenate([
        mx.repeat(picked_valid[0], BLOCK),
        (tail_tok <= q_pos[0]) & ~dup,
    ])
    tokens = mx.clip(tokens, 0, total - 1)
    k = mx.take(keys, tokens, axis=2)
    v = mx.take(values, tokens, axis=2)
    return mx.fast.scaled_dot_product_attention(
        queries, k, v, scale=scale, mask=ok[None, None, None, :]
    )


def install(model, sidecar: QSASidecar) -> list[int]:
    """Attach ``sidecar`` to the matching full-attention layers of ``model``."""
    from .qwen38_27b import Qwen3NextAttention

    text = getattr(model, "language_model", model)
    body = getattr(text, "model", text)
    installed = []
    for index, layer in enumerate(body.layers):
        if index not in sidecar.indexers:
            continue
        attention = getattr(layer, "self_attn", None)
        if not isinstance(attention, Qwen3NextAttention):
            raise ValueError(f"layer {index} has no full attention for an indexer")
        object.__setattr__(attention, "_qsa_sidecar", sidecar)
        object.__setattr__(attention, "_qsa_layer", index)
        attention.__class__ = _qsa_class(type(attention))
        installed.append(index)
    missing = sorted(set(sidecar.indexers) - set(installed))
    if missing:
        raise ValueError(f"indexer layers without attention: {missing}")
    return installed


_CLASSES: dict[type, type] = {}


def _qsa_class(base: type) -> type:
    if base in _CLASSES or getattr(base, "_is_qsa_sidecar", False):
        return _CLASSES.get(base, base)

    class QSASidecarAttention(base):
        _is_qsa_sidecar = True

        def __call__(self, x, mask=None, cache=None):
            sidecar = self._qsa_sidecar
            if sidecar.mode == "dense":
                return base.__call__(self, x, mask, cache)
            return self._qsa_call(x, mask, cache, sidecar)

        def mixed(self, x, parts):
            raise QSASidecarRefused("QSA sidecar does not serve mixed packing")

        def _qsa_call(self, x, mask, cache, sidecar):
            (B, L, _) = x.shape
            if B != 1 or cache is None:
                raise QSASidecarRefused("QSA sidecar serves one cached sequence")
            offset = int(cache.offset)
            q_proj, keys, values = self._qkv(x)
            (queries, gate) = mx.split(
                q_proj.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
            )
            gate = gate.reshape(B, L, -1)
            queries = self.q_norm(queries).transpose(0, 2, 1, 3)
            keys = self.k_norm(
                keys.reshape(B, L, self.num_key_value_heads, -1)
            ).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            (keys, values) = cache.update_and_fetch(keys, values)
            total = keys.shape[2]
            if total != offset + L:
                raise QSASidecarRefused("cache width does not match its offset")
            if sidecar.mode == "quest":
                from .base import scaled_dot_product_attention
                from .precise_ops import gate_sigmoid

                vis = _quest_mask(queries, keys, offset, **sidecar.quest)
                STATUS["quest_mask_rows"] += L
                output = mx.fast.scaled_dot_product_attention(
                    queries, keys, values, scale=self.scale, mask=vis
                )
                gate = gate_sigmoid(gate).reshape(B, L, self.num_attention_heads, -1)
                output = (output.transpose(0, 2, 1, 3) * gate).reshape(B, L, -1)
                return self.o_proj(output)
            indexer = sidecar.indexers[self._qsa_layer]
            picked = _selection(
                indexer, x[0].astype(mx.float32), offset, sidecar.block_topk,
                fused=sidecar.fused_select,
            )
            if sidecar.mode == "selfdense":
                picked = None  # control: this path, every block selected
            elif sidecar.mode == "window" and picked is not None:
                # control: same budget, most recent complete blocks only
                (_, _, q_pos, n_blocks, _) = picked
                newest = (q_pos + 1) // BLOCK - 1
                ids = newest[:, None] - mx.arange(sidecar.block_topk, dtype=mx.int32)[None]
                picked = (mx.maximum(ids, 0), ids >= 0, q_pos, n_blocks, None)
            from .base import scaled_dot_product_attention
            from .precise_ops import gate_sigmoid

            if picked is None:
                STATUS["dense_by_construction"] += 1
                output = scaled_dot_product_attention(
                    queries, keys, values, cache=cache, scale=self.scale, mask=mask
                )
            else:
                (ids, picked_valid, q_pos, n_blocks, counts) = picked
                mode = "nax" if sidecar.mode == "window" else sidecar.mode
                if mode == "gather" and L > 1:
                    mode = "nax"
                if mode == "mask":
                    vis = _dense_mask(ids, picked_valid, q_pos, n_blocks, total)
                    output = mx.fast.scaled_dot_product_attention(
                        queries, keys, values, scale=self.scale, mask=vis
                    )
                elif mode == "nax":
                    output = _nax(
                        queries, keys, values, _compact(ids, picked_valid, q_pos, total, counts),
                        self.scale,
                    )
                else:
                    output = _gather(
                        queries, keys, values, ids, picked_valid, q_pos, self.scale
                    )
                STATUS[f"sparse_{mode}_rows"] += L
                if counts is not None:
                    STATUS["fused_select_rows"] += L
            # Gate the [B, L, H, D] view directly: the elementwise multiply
            # writes a row-contiguous result, so the flatten is a view and the
            # separate transpose copy (L x H x D per layer) is gone.
            gate = gate_sigmoid(gate).reshape(B, L, self.num_attention_heads, -1)
            output = (output.transpose(0, 2, 1, 3) * gate).reshape(B, L, -1)
            return self.o_proj(output)

    QSASidecarAttention.__name__ = f"QSASidecar{base.__name__}"
    _CLASSES[base] = QSASidecarAttention
    return QSASidecarAttention


def status(reset: bool = False) -> dict:
    snapshot = dict(STATUS)
    if reset:
        STATUS.clear()
    return snapshot
