"""Exact tile verifier used by default-off external-draft experiments.

This module owns only the proposal acceptance law.  Target execution, cache
transactions, and publication remain with the external executor so a caller
can stage every tile on request-private state before atomically promoting it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .speculative_sampling import (
    RequestRNG,
    VerifiedBlock,
    probability,
    verify_proposals,
)

PROGRESSIVE_EXTERNAL_VERIFY_VERSION = "exact-linear-b1-private-promotion-v1"
PROGRESSIVE_TARGET_PROTOCOL_VERSION = "row-exact-b1s1-cache-state-v1"


def verify_proposal_tile(
    tokens: Sequence[int],
    proposal_laws: Sequence[np.ndarray],
    target_laws: Sequence[np.ndarray],
    rng: RequestRNG,
    *,
    final: bool,
) -> VerifiedBlock:
    """Verify one proposal slice without an intermediate bonus draw.

    A non-final, fully accepted tile returns only its proposals.  It consumes
    the same ordered acceptance uniforms as ordinary token-wise verification,
    leaving the next proposal row and the sole bonus draw to a later tile.
    """

    tokens = tuple(tokens)
    q = tuple(probability(law) for law in proposal_laws)
    p = tuple(probability(law) for law in target_laws)
    if final:
        return verify_proposals(tokens, q, p, rng)
    if not tokens or len(q) != len(tokens) or len(p) != len(tokens):
        raise ValueError("non-final tile needs one target law per proposal token")
    if any(law.shape != p[0].shape for law in (*q, *p)):
        raise ValueError("tile vocabulary mismatch")
    for token, law in zip(tokens, q):
        if not 0 <= token < len(law) or law[token] <= 0:
            raise ValueError("proposed token has zero proposal probability")

    emitted: list[int] = []
    target_rows: list[np.ndarray] = []
    for index, token in enumerate(tokens):
        if rng.uniform() < min(1.0, p[index][token] / q[index][token]):
            emitted.append(token)
            target_rows.append(p[index])
            continue
        residual = np.maximum(p[index] - q[index], 0)
        correction = rng.sample(residual if residual.sum() > 0 else p[index])
        emitted.append(correction)
        target_rows.append(p[index])
        return VerifiedBlock(
            index,
            tuple(emitted),
            tuple(target_rows),
            True,
        )
    return VerifiedBlock(
        len(tokens),
        tuple(emitted),
        tuple(target_rows),
        False,
    )
