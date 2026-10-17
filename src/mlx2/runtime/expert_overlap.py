"""Experimental gated overlap against mlx2's actual reader/LRU/gather path.

Opt-in only; the ordinary streamed projection remains the default. Sushi MIT expert_stream.zig
inspired the all-hit speculation; this uses immutable per-call MLX stacks.
"""

from collections import deque


class GatedOverlap:
    def __init__(self, module):
        self.module = module
        self.history = deque(maxlen=32)
        self.counts = dict(attempts=0, accepted=0, rebuilds=0, declined=0)

    def __call__(self, x, indices, sorted_indices=False):
        import mlx.core as mx
        import numpy as np

        m = self.module
        cache = m._stream_cache
        hot = sorted(cache._entries)
        eligible = (
            mx.default_device() == mx.gpu
            and not sorted_indices
            and x.dtype in (mx.float16, mx.bfloat16)
            and m.mode == "affine"
            and m.bits == 4
            and m.group_size == 64
            and 0 < indices.size <= 32
            and 0 < len(hot) <= 2 * indices.size
            and len(self.history) >= 8
            and sum(self.history) / len(self.history) >= 0.9
        )
        if not eligible:
            self.counts["declined"] += 1
            host = np.asarray(indices, dtype=np.int64)
            unique = np.unique(host)
            self.history.append(all(int(e) in cache._entries for e in unique))
            return m._synchronous(
                x, indices, sorted_indices=sorted_indices, host=host, unique=unique
            )
        self.counts["attempts"] += 1
        # Retain immutable expert arrays while the asynchronous candidate runs.
        # No state or LRU mutation occurs until its ID mapping is validated.
        rows = [cache._entries[e] for e in hot]
        table = np.full(m.num_experts, -1, dtype=np.int32)
        table[hot] = np.arange(len(hot), dtype=np.int32)
        bounded_ids = mx.clip(indices, 0, m.num_experts - 1)
        mapped = mx.array(table)[bounded_ids]
        safe = mx.maximum(mapped, 0).astype(mx.uint32)
        weight, scales = [mx.stack([r[k] for r in rows]) for k in (0, 1)]
        biases = mx.stack([r[2] for r in rows]) if len(rows[0]) > 2 else None
        candidate = mx.gather_qmm(
            x,
            weight,
            scales,
            biases,
            rhs_indices=safe,
            transpose=True,
            group_size=m.group_size,
            bits=m.bits,
            mode=m.mode,
            sorted_indices=False,
        )
        if "bias" in m:
            candidate = candidate + mx.expand_dims(m["bias"][bounded_ids], -2)
        stats = cache.stats
        prior = stats._group_transient if cache.group != stats._group else 0
        # The bank is resident rows only; this runs before acquire, so the
        # previous call's ephemeral count does not apply.
        cache.note_gather(len(hot), ephemeral=0)
        # Before the host index read, the preceding layer may still retain its
        # copies. This extra bank is reserved by install_expert_streaming.
        if prior:
            transient = prior + len(hot) * cache.reader.expert_bytes
            stats.peak_transient_bytes = max(stats.peak_transient_bytes, transient)
            stats.peak_tracked_bytes = max(
                stats.peak_tracked_bytes, stats.resident_bytes + transient
            )
        mx.async_eval(indices, candidate)
        host = np.asarray(indices, dtype=np.int64)
        unique = np.unique(host)
        hit = all(int(e) in cache._entries for e in unique)
        self.history.append(hit)
        if hit:
            cache.acquire([int(e) for e in unique])
            if m._stream_collector is not None:
                m._stream_collector.observe(m._stream_layer, unique)
            self.counts["accepted"] += 1
            return candidate
        # Finish and release the rejected bank before the normal read/gather
        # path allocates another, so the existing transient reservation holds.
        mx.eval(candidate)
        del candidate, weight, scales, biases, rows, mapped, safe
        self.counts["rebuilds"] += 1
        self.history.clear()  # a miss requires a fresh warm history
        return m._synchronous(
            x, indices, sorted_indices=sorted_indices, host=host, unique=unique
        )
