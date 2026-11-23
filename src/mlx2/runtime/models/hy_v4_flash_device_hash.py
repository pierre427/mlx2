# SPDX-License-Identifier: Apache-2.0
"""Device-side Engram n-gram hashing for Hy4-flash (opt-in ``engram_hash: device``).

``EngramHasher`` (``hy_v4_flash``) hashes on the host in int64: each n-gram
token is mapped through the layer's lookup table to a compressed id ``c``,
multiplied by an odd per-position multiplier ``m_k`` (``c * m_k < 2**63`` by
the reference's choice of bound), the products of one n-gram are XORed, and
the 64-bit result is reduced modulo each head's prime ``M``; the head offset
gives the flat table row.

This module computes the same rows with MLX ops that never rely on int64 or
on integer wrap-around, so it is exact on any device:

* ``c`` (< 2**32) and ``m_k`` (< 2**63) are split into 16-bit limbs held in
  uint32; every limb product is below 2**32, and each 16-bit result column
  sums at most four 16-bit halves, so nothing overflows;
* the four 16-bit product limbs are XORed limb-wise (XOR is limb-local);
* the 64-bit value is reduced by Horner's rule over its eight bytes,
  ``acc = (acc * 256 + byte) % M``, which stays below 2**32 when
  ``M <= 2**24`` (Hy4-flash heads are ~10.3M < 16.8M); larger moduli are
  refused at construction rather than computed approximately.

All Engram layers are hashed in one vectorized call (tables stacked) on the
caller's stream.  No side stream: a cross-stream dependency on the sampled
token would be bridged by a Fence, which under ``MLX_METAL_FAST_SYNCH=1``
can wedge the GPU (``runtime/generation_stream.py``), and an MLX stream is
bound to the thread that created it.  On one in-order stream the read-back
waits for the work queued before the hash (at decode: the previous step,
which it needs anyway), not for what the forward queues after it.

Original mlx2 code; nothing mined.
"""

from __future__ import annotations

from typing import Sequence

import mlx.core as mx
import numpy as np

MAX_MODULUS = 1 << 24
_LIMB = 0xFFFF


def _limbs16(value: int) -> list[int]:
    return [(value >> (16 * i)) & _LIMB for i in range(4)]


class DeviceEngramHasher:
    """All Engram layers' flat table rows from ``[N, max_ngram]`` n-gram tokens.

    ``hashers`` are the host ``EngramHasher`` objects in a fixed layer order;
    ``__call__`` returns int32 ``[layers, N, orders*heads]`` rows equal to
    ``hasher.table_rows`` of each layer.
    """

    def __init__(self, hashers: Sequence):
        hashers = list(hashers)
        if not hashers:
            raise ValueError("device Engram hashing needs at least one layer")
        first = hashers[0]
        n, heads = first.max_ngram, first.heads
        size = first.table.size
        for h in hashers:
            if (h.max_ngram, h.heads, h.table.size) != (n, heads, size):
                raise ValueError("device Engram hashing needs one n-gram/head/table geometry")
        if n < 2:
            raise ValueError("Engram max_ngram must be at least 2")
        tables = np.stack([h.table for h in hashers])
        if tables.min() < 0 or tables.max() >= 1 << 32:
            raise ValueError("Engram compressed ids must fit 32 bits for device hashing")
        compressed_max = tables.max(axis=1)  # per layer: the hasher's ``compressed - 1``
        multipliers = np.stack([h.multipliers for h in hashers])  # [layers, n]
        for c_max, row in zip(compressed_max.tolist(), multipliers.tolist()):
            if any(m <= 0 or m >= 1 << 63 or int(c_max) * m >= 1 << 63 for m in row):
                # The host product would overflow int64; the limbs would not
                # reproduce its wrap, so refuse instead of differing.
                raise ValueError("Engram multiplier bound exceeds the exact device range")
        moduli = np.stack([h.moduli for h in hashers])  # [layers, orders*heads]
        offsets = np.stack([h.offsets for h in hashers])
        if moduli.min() <= 0 or moduli.max() > MAX_MODULUS:
            raise ValueError(f"device Engram hashing needs head moduli in 1..{MAX_MODULUS}")
        if (offsets + moduli).max() >= 1 << 31:
            raise ValueError("Engram flat rows must fit int32 for device hashing")
        self.layers = len(hashers)
        self.max_ngram, self.heads, self.table_size = n, heads, size
        self.orders = n - 1
        L, O = self.layers, self.orders
        self._tables = mx.array(tables.astype(np.uint32))  # [L, size]
        self._pad = mx.array(np.stack([h.table[h.pad_id] for h in hashers]).astype(np.uint32))  # [L]
        limbs = np.array([[_limbs16(int(m)) for m in row] for row in multipliers.tolist()],
                         dtype=np.uint32)  # [L, n, 4]
        self._mult = mx.array(limbs)
        self._moduli = mx.array(moduli.reshape(L, O, heads).astype(np.uint32))
        self._offsets = mx.array(offsets.reshape(L, O, heads).astype(np.int32))
        self._fn = mx.compile(self._rows)
        self.calls = 0

    # ``tokens [N, n]`` int32 -> rows ``[L, N, O*heads]`` int32.
    def _rows(self, tokens, tables, pad, mult, moduli, offsets):
        L, n, O, size = self.layers, self.max_ngram, self.orders, self.table_size
        N = tokens.shape[0]
        # Compressed ids: ``table[t]`` when ``t < size`` else the pad's.
        clipped = mx.minimum(tokens, size - 1).astype(mx.uint32).reshape(-1)  # [N*n]
        comp = mx.take(tables, clipped, axis=1).reshape(L, N, n)
        inside = (tokens < size)[None]
        comp = mx.where(inside, comp, pad[:, None, None])
        c = (comp & _LIMB, comp >> 16)  # 16-bit limbs of c, [L, N, n]
        m = [mult[:, None, :, j] for j in range(4)]  # [L, 1, n]
        cols = [None] * 6
        for i in range(2):
            for j in range(4):
                p = c[i] * m[j]  # < 2**32
                for k, part in ((i + j, p & _LIMB), (i + j + 1, p >> 16)):
                    cols[k] = part if cols[k] is None else cols[k] + part
        limbs, carry = [], None
        for k in range(4):  # the product is < 2**63: columns 4 and 5 end as zero
            v = cols[k] if carry is None else cols[k] + carry
            limbs.append(v & _LIMB)
            carry = v >> 16
        # Running XOR over the n-gram positions: orders 2..n.
        acc = [limb[..., 0] for limb in limbs]
        mixes = []
        for k in range(1, n):
            acc = [a ^ limb[..., k] for a, limb in zip(acc, limbs)]
            mixes.append(mx.stack(acc, axis=-1))  # [L, N, 4]
        mix = mx.stack(mixes, axis=2)  # [L, N, O, 4]
        mod = moduli[:, None]  # [L, 1, O, heads]
        rem = mx.zeros((L, N, O, self.heads), dtype=mx.uint32)
        for limb in (3, 2, 1, 0):
            for shift in (8, 0):
                byte = (mix[..., limb] >> shift) & 0xFF
                rem = (rem * 256 + byte[..., None]) % mod
        rows = rem.astype(mx.int32) + offsets[:, None]
        return rows.reshape(L, N, O * self.heads)

    def __call__(self, tokens: mx.array) -> mx.array:
        """``tokens [N, max_ngram]`` (any integer dtype, ids >= 0) -> int32 rows."""
        if tokens.ndim != 2 or tokens.shape[1] != self.max_ngram:
            raise ValueError("device Engram hashing takes [N, max_ngram] n-gram tokens")
        self.calls += 1
        return self._fn(tokens.astype(mx.int32), self._tables, self._pad, self._mult,
                        self._moduli, self._offsets)
