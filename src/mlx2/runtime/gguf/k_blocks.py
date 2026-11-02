"""Bounded GGML K-block oracle. See provenance/gguf-k-quants.json.

Vectorized scalar indexing follows the physical byte layout, independently of
the separate gguf Python reference used by tests. No MLX import or GPU work.
"""

from __future__ import annotations

import numpy as np

SIZES = {2: (32, 18), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210)}


def decode_k_blocks(data, kind):
    count, size = SIZES[kind]
    p = np.frombuffer(data, dtype=np.uint8).reshape(-1, size)
    i = np.arange(count)

    def half(offset):
        return p[:, offset : offset + 2].copy().view("<f2").astype(np.float32)

    if kind == 2:
        q = ((p[:, 2 + i % 16] >> (4 * (i // 16))) & 15).astype(np.int32) - 8
        return (half(0) * q.astype(np.float32)).reshape(-1)
    if kind == 11:
        group = i // 16
        lo = (p[:, 96 + group % 8] >> (4 * (group // 8))) & 15
        hi = (p[:, 104 + group % 4] >> (2 * (group // 4))) & 3
        scale = (lo | (hi << 4)).astype(np.int32) - 32
        q = ((p[:, 32 + (i // 128) * 32 + i % 32] >> (2 * ((i // 32) % 4))) & 3).astype(
            np.int32
        )
        q -= 4 * (1 - ((p[:, i % 32] >> (i // 32)) & 1).astype(np.int32))
        return (half(108) * scale.astype(np.float32) * q.astype(np.float32)).reshape(-1)
    if kind in (12, 13):
        group = i // 32
        sc = p[:, 4 + group % 4]
        mn = p[:, 8 + group % 4]
        aux = p[:, 12 + group % 4]
        scale = np.where(group < 4, sc & 63, (aux & 15) | ((sc >> 2) & 48))
        minimum = np.where(group < 4, mn & 63, (aux >> 4) | ((mn >> 2) & 48))
        qoff = 16 if kind == 12 else 48
        q = (p[:, qoff + (i // 64) * 32 + i % 32] >> (4 * ((i // 32) % 2))) & 15
        if kind == 13:
            q = q | (((p[:, 16 + i % 32] >> (i // 32)) & 1) << 4)
        return (
            half(0) * scale.astype(np.float32) * q.astype(np.float32)
            - half(2) * minimum.astype(np.float32)
        ).reshape(-1)
    ql = (p[:, (i // 128) * 64 + i % 64] >> (4 * ((i // 64) % 2))) & 15
    qh = (p[:, 128 + (i // 128) * 32 + i % 32] >> (2 * ((i // 32) % 4))) & 3
    q = (ql | (qh << 4)).astype(np.int32) - 32
    scales = p[:, 192 + i // 16].view(np.int8).astype(np.float32)
    return (half(208) * scales * q.astype(np.float32)).reshape(-1)
