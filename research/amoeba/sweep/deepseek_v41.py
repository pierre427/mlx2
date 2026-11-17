"""DeepSeek V4.1 sweep family: the ordinary mlx2 V4.1 layer code, run one layer at a time.

Everything numerical is the in-tree route (src/mlx2/runtime/models/deepseek_v41.py ``Block``,
``Attention``, ``engram``, the Source weights and their LFU expert store); this module only
schedules it layer-major and carries the per-text state a later layer reads:

- ``h``    [B, T, hc, d]  hyper-connection residual streams (stored bf16 when the activation
                          dtype is bf16: every producer rounds through ``model.round`` first,
                          so the bf16 copy is lossless)
- ``pre``  [B, T, hc]     the next block's collapse weights (f32)
- ``ids``  [B, T]         tokens (Engram hashes are recomputed from them)
- ``pool`` / ``pool_keys`` [B, P, head] / [B, P, index_head]  the latest KV source layer's
                          compressed pool and indexer keys (CSA2 consumers read the source layer's
                          cache, not their own)

Index selection (``shared["indices"]`` etc.) is not carried. Under the ``finite_all_retained``
index policy and a pool no longer than ``index_topk`` (always, for 64-token probes), the
ordinary ``Attention.select`` takes its all-retained branch, which reads only the pool extent
and the row geometry; the family replays that exact call before each consumer layer. Any other
geometry fails closed.

Attention runs per text (the ordinary route is B1). The FFN half (router, shared expert, routed
experts, HC epilogue) is token-wise, so up to ``batch_rows`` rows of several texts go through
one ``Block.__call__`` with the attention half already computed; ``ffn_texts=1`` makes every
call the same shape as the ordinary forward.

Units are the 384 routed experts per layer. The per-route signal is the norm of that route's
weighted expert output row (``parts`` of the Source ordered combine: the routed weight enters
before down_proj, so this is ``w * ||row||`` up to the FP8 QAT of the middle activation).
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class _Tok:
    def __init__(self, tokenizer):
        self.t = tokenizer

    def encode(self, text):
        for t in (self.t, getattr(self.t, "_tokenizer", None)):
            if t is None or not hasattr(t, "encode"):
                continue
            try:
                out = t.encode(text, add_special_tokens=False)
            except TypeError:
                out = t.encode(text)
            return list(getattr(out, "ids", out))
        raise TypeError("tokenizer has no encode")

    def decode(self, ids):
        return self.t.decode(list(ids))


class _CacheView:
    """The ``cache`` list Attention indexes: only the layer itself and its KV source exist."""

    def __init__(self, items):
        self.items = items

    def __getitem__(self, i):
        return self.items[i]


def route_signal_from_parts(parts, values, order):
    """Per-route norm [rows, k] of the expert-sorted ``parts`` of an ordered Source combine.

    Same slot map as ``deepseek_v41_source_math.ordered_offsets``: original route (r, j) lives at
    packed row ``inverse[r * k + j]``."""
    import mlx.core as mx
    packed = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
    norms = mx.linalg.norm(packed.astype(mx.float32), axis=-1)
    order = np.asarray(order)
    inverse = np.empty(order.size, dtype=np.uint32)
    inverse[order] = np.arange(order.size, dtype=np.uint32)
    return norms[mx.array(inverse.reshape(values.shape))]


def read_engram_rows(table, ids, out, *, gap_bytes=256 << 10, span_bytes=64 << 20):
    """Decode Engram rows ``ids`` (sorted unique) of an ordinary ``EngramRows`` table into ``out``
    (uint16 [U, 256], the bf16 bits ``decode_engram_rows`` rounds to), reading each plane in long
    sequential spans: two rows share a span when their gap is below ``gap_bytes``."""
    import fcntl
    import os
    from mlx2.runtime.models.deepseek_v41_engram import decode_engram_rows
    if table.affine:
        raise ValueError("affine Engram planes are not supported by the sweep scan")
    if len(ids) and (ids[0] < 0 or ids[-1] >= table.rows or np.any(np.diff(ids) <= 0)):
        raise ValueError("Engram scan ids must be sorted unique rows of the table")
    planes = ([(table.offset, 264, slice(0, 264))] if table.scale_offset is None else
              [(table.offset, 256, slice(0, 256)), (table.scale_offset, 8, slice(256, 264))])
    fd = os.open(table.path, os.O_RDONLY)
    stats = {"engram_read_bytes": 0, "engram_spans": 0}
    try:
        if os.uname().sysname == "Darwin":
            fcntl.fcntl(fd, 48, 1)       # F_NOCACHE, as the ordinary reader
        st = os.fstat(fd)
        from mlx2.runtime.models.deepseek_v41_engram import file_stamp
        if file_stamp(st) != table.expected_stat:
            raise ValueError("Engram source changed after sealing")
        step = 1 << 16
        for a in range(0, len(ids), step):
            chunk = ids[a:a + step]
            raw = np.empty((len(chunk), 264), np.uint8)
            for base, size, cols in planes:
                offs = base + chunk.astype(np.int64) * size
                i = 0
                while i < len(chunk):
                    j = i + 1
                    while (j < len(chunk) and offs[j] - (offs[j - 1] + size) <= gap_bytes
                           and offs[j] + size - offs[i] <= span_bytes):
                        j += 1
                    start, end = int(offs[i]), int(offs[j - 1] + size)
                    buf = bytearray()
                    while len(buf) < end - start:
                        block = os.pread(fd, end - start - len(buf), start + len(buf))
                        if not block:
                            raise OSError("short Engram span read")
                        buf += block
                    view = np.frombuffer(bytes(buf), np.uint8)
                    rel = (offs[i:j] - start)[:, None] + np.arange(size)[None]
                    raw[i:j, cols] = view[rel]
                    stats["engram_read_bytes"] += end - start
                    stats["engram_spans"] += 1
                    i = j
            dec = decode_engram_rows(raw)
            out[a:a + len(chunk)] = (np.ascontiguousarray(dec).view(np.uint32) >> 16).astype(np.uint16)
    finally:
        os.close(fd)
    return stats


class DeepSeekV41Family:
    batch_rows = 1024
    unit = ("per-text mean over positions >= 1 of the routed-expert signal: norm of the weighted "
            "expert output row (routed weight x row norm), accumulated per expert id; fp16")

    def __init__(self, model, tok, *, checkpoint, fingerprint, signal="auto", ffn_texts=None,
                 adapter=None, policy=None):
        self.m, self.c = model, model.args
        self.adapter, self.policy = adapter, policy
        self.tok = tok
        self.checkpoint, self.fingerprint = checkpoint, fingerprint
        self.L = self.c.num_hidden_layers
        self.widths = [int(self.c.n_routed_experts)] * self.L
        self.kinds = ["experts"] * self.L
        self.k = int(self.c.num_experts_per_tok)
        self.ffn_texts = ffn_texts
        stream = getattr(model.weights, "expert_stream", None)
        self.signal = ("combine" if stream is not None else "probe") if signal == "auto" else signal
        if self.signal not in ("combine", "probe"):
            raise ValueError("signal must be auto, combine or probe")
        self.counters = {"select_replays": 0, "combine_captures": 0, "probe_calls": 0}
        self.check_signal = False
        self._engram = {}           # layer -> (sorted unique row ids, decoded bf16 rows memmap, path)
        if model.index_policy != "finite_all_retained":
            raise ValueError("sweep needs index_policy finite_all_retained (selection replay)")

    # ------------------------------------------------------------------ geometry helpers
    def _bf16(self):
        return self.c.activation_dtype == "bfloat16"

    def _new_cache(self, l):
        c, m = self.c, self.m
        return m.cache_class(c.compress_ratios[l], xp=__import__("mlx.core").core, identity=m.identity,
                             head_dim=c.head_dim, index_head_dim=c.index_head_dim,
                             window_size=c.sliding_window, source=l in c.kv_source_layer_ids,
                             history=l == 0 and bool(c.engram_layer_ids),
                             compressed_vocab=c.engram_compressed_vocab_size)

    def _owner(self, src, T, pool, keys):
        state = self._new_cache(src)
        state.offset = T
        state.state = (None, None, None, pool, keys, None)
        return state

    def _hashes(self, ids_row):
        tail = np.full((3,), -1, np.int32)
        hashes, _ = self.m.hasher(np.asarray(ids_row), tail)
        return hashes

    def _replay_selection(self, l, T, owner, shared):
        """Re-run the ordinary all-retained ``select`` calls whose result layer ``l`` reads."""
        import mlx.core as mx
        c = self.c
        count = owner.index_keys.shape[1]
        if count > c.index_topk or -(-count // c.candidate_block_size) > c.candidate_topk_blocks:
            raise ValueError("sweep selection replay needs the all-retained geometry")
        src = c.kv_source_for(l)
        prior = [i for i in c.index_source_layer_ids if i <= l]
        if not prior:
            raise ValueError(f"V4.1 layer {l} has no index source")
        last = prior[-1]
        cand = c.candidate_source_layer_id
        calls = []
        if cand >= 0 and last > cand and cand != l:
            calls.append(cand)
        if last != l and last not in calls:
            calls.append(last)
        dummy = mx.zeros((1, T, 1), mx.float32)
        for i in calls:
            if c.kv_source_for(i) != src:
                raise ValueError("sweep selection replay: index source reads another pool")
            self.m.layers[i].attn.select(dummy, None, owner, 0, shared)
            self.counters["select_replays"] += 1

    # ------------------------------------------------------------------ interface
    def embed(self, ids):
        import mlx.core as mx
        m, c = self.m, self.c
        ids = ids.astype(mx.int32)
        B, T = ids.shape
        with m._weight_scope():
            x = m.round(m.weights.embedding(ids.astype(mx.uint32)))
            h = mx.broadcast_to(x[:, :, None], (B, T, c.hc_mult, c.hidden_size))
            pre = mx.broadcast_to(mx.array([1.0] + [0.0] * (c.hc_mult - 1)), (B, T, c.hc_mult))
            pre = pre.astype(mx.float32)
        return {"ids": ids, "h": h.astype(mx.bfloat16) if self._bf16() else h, "pre": pre}

    @contextmanager
    def _route_capture(self, sink):
        """Collect one per-route signal [rows, k] per routed-expert call into ``sink``."""
        import mlx.core as mx
        w = self.m.weights
        original = w.moe
        if self.signal == "combine":
            from mlx2.adapters import deepseek_v41_source_math as math_mod
            combine = math_mod.ordered_combine
            seen = []

            def ordered_combine(parts, values, order, **kw):
                seen.append(route_signal_from_parts(parts, values, order))
                self.counters["combine_captures"] += 1
                return combine(parts, values, order, **kw)

            def moe(x, ids, scores, prefix, **kw):
                before = len(seen)
                out = original(x, ids, scores, prefix, **kw)
                if len(seen) != before + 1:
                    raise RuntimeError(f"{prefix}: routed call bypassed the ordered combine "
                                       f"({len(seen) - before} captures); unsupported expert path")
                sig = seen.pop()
                sink.append((ids, sig))
                if self.check_signal:
                    # independent per-slot probe: the routed call with one slot's weight kept
                    k = ids.shape[-1]
                    probe = []
                    for j in range(k):
                        one = (mx.arange(k) == j).astype(scores.dtype)
                        probe.append(mx.linalg.norm(original(x, ids, scores * one, prefix, **kw).astype(mx.float32), axis=-1))
                        seen.pop()
                    probe = mx.stack(probe, -1)
                    rel = float(mx.max(mx.abs(probe - sig)) / mx.maximum(mx.max(sig), 1e-30))
                    self.counters["check_signal_max_rel"] = max(self.counters.get("check_signal_max_rel", 0.0), rel)
                    self.counters["check_signal_calls"] = self.counters.get("check_signal_calls", 0) + 1
                return out
            math_mod.ordered_combine = ordered_combine
            w.moe = moe
            try:
                yield
            finally:
                math_mod.ordered_combine = combine
                del w.moe
        else:
            def moe(x, ids, scores, prefix, **kw):
                out = original(x, ids, scores, prefix, **kw)
                k = ids.shape[-1]
                sig = []
                for j in range(k):
                    one = (mx.arange(k) == j).astype(scores.dtype)
                    sig.append(mx.linalg.norm(original(x, ids, scores * one, prefix, **kw).astype(mx.float32), axis=-1))
                    self.counters["probe_calls"] += 1
                sink.append((ids, mx.stack(sig, -1)))
                return out
            w.moe = moe
            try:
                yield
            finally:
                del w.moe

    def run_layer(self, l, carry):
        import mlx.core as mx
        m, c = self.m, self.c
        block = m.layers[l]
        ids = carry["ids"]
        B, T = ids.shape
        h = carry["h"].astype(mx.float32)
        pre = carry["pre"]
        ratio = c.compress_ratios[l]
        src = c.kv_source_for(l) if ratio else None
        out = {"ids": ids}
        with m._weight_scope():
            if l in c.engram_layer_ids:
                idx = c.engram_layer_ids.index(l)
                ids_np = np.asarray(ids)
                # same grouping as the FFN half: ffn_texts=1 keeps the ordinary M = T projection
                eg = self.ffn_texts or B
                parts = []
                for g in range(0, B, eg):
                    n = min(eg, B - g)
                    hashes = np.concatenate([self._hashes(ids_np[b])[:, idx] for b in range(g, g + n)], 0)
                    rows = self._engram_lookup(l, hashes)
                    parts.append(m.engram(h[g:g + n].reshape(1, n * T, *h.shape[2:]), l, hashes,
                                          **({"rows": rows} if rows is not None else {})).reshape(n, T, *h.shape[2:]))
                h = mx.concatenate(parts, 0)
            halves, pools, keys = [], [], []
            for b in range(B):
                state = self._new_cache(l)
                items = {l: state}
                shared = {}
                if ratio:
                    if src != l:
                        owner = self._owner(src, T, carry["pool"][b:b + 1], carry["pool_keys"][b:b + 1])
                        items[src] = owner
                    cand = c.candidate_source_layer_id
                    if l not in c.index_source_layer_ids or 0 <= cand < l:
                        if src == l:
                            raise ValueError(f"layer {l}: a KV source that reads a prior selection "
                                             "cannot be replayed before its own publish")
                        self._replay_selection(l, T, owner, shared)
                state._in_forward = True
                halves.append(block.attention_half(h[b:b + 1], pre[b:b + 1], state, _CacheView(items), shared))
                state._in_forward = False
                if ratio and src == l:
                    pools.append(state.pooled)
                    keys.append(state.index_keys)
            per = max(1, self.batch_rows // T)
            if self.ffn_texts:
                per = min(per, self.ffn_texts)
            hs, ps, sink = [], [], []
            with self._route_capture(sink):
                for g in range(0, B, per):
                    part = halves[g:g + per]
                    cat = tuple(mx.concatenate([p[i] for p in part], axis=1) for i in range(5))
                    block.attention_half = lambda *a, _cat=cat, **k: _cat
                    try:
                        hn, pn = block(cat[0], None, None, None, {})
                    finally:
                        del block.attention_half
                    hs.append(hn.reshape(len(part), T, *hn.shape[2:]))
                    ps.append(pn.reshape(len(part), T, *pn.shape[2:]))
            h = mx.concatenate(hs, 0)
            pre = mx.concatenate(ps, 0).astype(mx.float32)
            inds = mx.concatenate([i.reshape(-1, T, self.k) for i, _ in sink], 0)
            sig = mx.concatenate([s.reshape(-1, T, self.k) for _, s in sink], 0)
            out["h"] = h.astype(mx.bfloat16) if self._bf16() else h
            out["pre"] = pre
            if ratio and src == l:
                out["pool"], out["pool_keys"] = mx.concatenate(pools, 0), mx.concatenate(keys, 0)
            elif "pool" in carry:
                out["pool"], out["pool_keys"] = carry["pool"], carry["pool_keys"]
            payload = {"inds": inds.astype(mx.int32), "signal": sig.astype(mx.float32)}
            m._weight_eval(out, payload)
        mx.eval(out, payload)
        return out, payload

    def prepare_layer(self, l, seqs, scratch):
        """Engram layers: read every row the corpus needs in file order, once.

        The ordinary reader issues one uncached pread per row and plane, which on a spinning or
        USB-2 disk is ~10 ms each (~24 rows per token). This reads the same bytes as a few long
        sequential spans and decodes them with the ordinary ``decode_engram_rows``."""
        c = self.c
        if l not in c.engram_layer_ids or not self.m.engram_tables or len(seqs) < self.engram_scan_min_texts:
            return None
        idx = c.engram_layer_ids.index(l)
        ids = np.unique(np.concatenate([self._hashes(np.asarray(s, np.int32))[:, idx].reshape(-1) for s in seqs]))
        table = self.m.engram_tables[l]
        path = Path(scratch) / f"engram_L{l:02d}.u16.npy"
        rows = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=(len(ids), 256))
        stats = read_engram_rows(table, ids, rows)
        rows.flush()
        self._engram[l] = (ids, np.load(path, mmap_mode="r"), path)
        return {"engram_unique_rows": int(len(ids)), **stats}

    engram_scan_min_texts = 64

    def _engram_lookup(self, l, hashes):
        import mlx.core as mx
        got = self._engram.get(l)
        if got is None:
            return None
        ids, rows, _ = got
        flat = np.asarray(hashes).reshape(-1).astype(np.int64)
        pos = np.searchsorted(ids, flat)
        if np.any(pos >= len(ids)) or np.any(ids[np.minimum(pos, len(ids) - 1)] != flat):
            raise RuntimeError(f"Engram L{l}: row missing from the prepared scan")
        vals = (np.asarray(rows[pos]).astype(np.uint32) << 16).view(np.float32)
        return mx.array(vals.reshape(*np.asarray(hashes).shape, 256))

    def release_layer(self, l):
        import mlx.core as mx
        got = self._engram.pop(l, None)
        if got is not None:
            path = got[2]
            del got
            Path(path).unlink(missing_ok=True)
        stream = getattr(self.m.weights, "expert_stream", None)
        if stream is not None:
            protected = set(getattr(stream, "_leased", {})) | set(getattr(stream, "_pending_uploads", {}))
            for key in [k for k in list(stream._cache) if k[0] == l and k not in protected]:
                stream._adaptive_evict_key(key)
        mx.clear_cache()

    def final_logits(self, carry):
        import mlx.core as mx
        m = self.m
        with m._weight_scope():
            h = carry["h"].astype(mx.float32)
            B, T = h.shape[:2]
            hidden = m.norm(m.collapse(h.reshape(1, B * T, *h.shape[2:]), carry["pre"].reshape(1, B * T, -1)),
                            m.norm_weight)
            logits = m.linear(hidden, "output.weight", rounded=False)
            m._weight_eval(logits)
        return logits.reshape(B, T, -1)

    def reference_logits(self, ids, layers=None):
        """The ordinary forward, one text at a time (the route is B1). ``layers=k`` runs the same
        ordinary forward on the first k blocks only (then the final norm and head), the reference
        for a k-layer sweep prefix."""
        import mlx.core as mx
        m = self.m
        rows = []
        full = m.layers
        k = len(full) if layers is None else int(layers)
        try:
            m.layers = full[:k]
            for b in range(ids.shape[0]):
                cache = m.make_cache()[:k]
                try:
                    logits, _ = m._forward(ids[b:b + 1].astype(mx.int32), cache, logits_to_keep=0)
                    mx.eval(logits)
                finally:
                    # a truncated forward prefetches Engram rows for layers it never reaches
                    m.discard_engram_prefetch()
                rows.append(logits)
        finally:
            m.layers = full
        return mx.concatenate(rows, 0)

    def stream_receipt(self):
        stream = getattr(self.m.weights, "expert_stream", None)
        if stream is None:
            return None
        r = stream.receipt()
        return {"counters": r.get("counters"), "cache_payload_bytes": r.get("cache_payload_bytes"),
                "cache_budget_bytes": r.get("cache_budget_bytes")}

    def close(self):
        if self.adapter is not None:
            self.adapter.close()


def load(model_dir, *, expert_budget_gb=24, signal="auto", ffn_texts=None, policy_overrides=None):
    """Load the Source checkpoint through the Source adapter on its ordinary default bundle,
    with exact original MXFP4 experts streamed through the LFU store (``original_lfu``)."""
    import mlx.core as mx
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter
    from mlx2.runtime.streamed_load import WeightStreamRequest
    from mlx2.serving import preload_admission_limit_bytes
    mx.set_default_device(mx.gpu)
    budget = int(expert_budget_gb * 1e9)
    policy = dict(DeepSeekV41SourceAdapter.default_route_execution_policy["ordinary"])
    policy["deepseek_source_experts"] = {"mode": "original_lfu", "budget_bytes": budget}
    policy.update(policy_overrides or {})
    if policy.get("deepseek_source_prefill_batch", "reference") == "gate_up_down":
        raise ValueError("gate_up_down combines through combine_mapped; the sweep reads the ordered combine")
    adapter = DeepSeekV41SourceAdapter(str(model_dir), execution_policy=policy,
        weight_streaming=WeightStreamRequest(mode="moe_experts", budget_bytes=budget,
            admission_limit_bytes=preload_admission_limit_bytes(), read_workers=2, bypass_page_cache=True))
    fp = adapter.source_artifact["identity"]["fingerprint"]
    name = Path(model_dir).name
    fam = DeepSeekV41Family(adapter.model, _Tok(adapter.tokenizer), checkpoint=f"{name}@{fp[:8]}",
                            fingerprint=fp, signal=signal, ffn_texts=ffn_texts, adapter=adapter,
                            policy=json.loads(json.dumps(policy)))
    return fam
