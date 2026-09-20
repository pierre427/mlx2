"""Compact draft laws: a proposed block plus the top-C law each token came from.

Original mlx2 code; the batched pairwise walk that produces it follows the
Splash DFlash selector design (provenance/splash-02-dflash-pair-select.json).
A DFlash2 proposal law has support only on the selector's candidates, so the
block carries ``[B, K, C]`` candidate ids and probabilities instead of ``K``
dense vocabulary rows.  ``dense_laws`` rebuilds exactly the float64 rows the
host verifier consumes.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class DraftBlock:
    tokens: mx.array  # [B, K] int32
    cand_ids: mx.array  # [B, K, C] int32
    cand_q: mx.array  # [B, K, C] float32; row sums are 1
    lengths: tuple[int, ...]  # proposed tokens per row (<= K)

    def token_lists(self) -> list[list[int]]:
        tokens = np.asarray(self.tokens)
        return [
            [int(value) for value in tokens[row, :length]]
            for row, length in enumerate(self.lengths)
        ]

    def dense_laws(self, vocab: int) -> list[list[np.ndarray]]:
        ids = np.asarray(self.cand_ids)
        q = np.asarray(self.cand_q.astype(mx.float32)).astype(np.float64)
        laws = []
        for row, length in enumerate(self.lengths):
            row_laws = []
            for position in range(length):
                law = np.zeros(int(vocab), dtype=np.float64)
                law[ids[row, position]] = q[row, position]
                row_laws.append(law)
            laws.append(row_laws)
        return laws


__all__ = ["DraftBlock"]
