"""Exact tile verifier used by default-off external-draft experiments.

This module owns only the proposal acceptance law.  Target execution, cache
transactions, and publication remain with the external executor so a caller
can stage every tile on request-private state before atomically promoting it.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path

import numpy as np

from .speculative_sampling import (
    RequestRNG,
    VerifiedBlock,
    probability,
    verify_proposals,
)

PROGRESSIVE_EXTERNAL_VERIFY_VERSION = "exact-linear-b1-private-promotion-v1"
PROGRESSIVE_TARGET_PROTOCOL_VERSION = "row-exact-b1s1-cache-state-v1"
PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION = (
    "near-tie-bounded-b1s1-cache-state-v1"
)
PROGRESSIVE_FIDELITY_EXACT = "exact"
PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED = "near_tie_bounded"
PROGRESSIVE_FIDELITIES = frozenset(
    {PROGRESSIVE_FIDELITY_EXACT, PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED}
)
PROGRESSIVE_APPROXIMATE_QUALIFICATION_GATE_SCHEMA = (
    "mlx2.qwen38-approx-progressive-qualification-gate.v1"
)
_ROOT = Path(__file__).resolve().parents[3]
_ALGORITHM_SOURCE_PATHS = (
    "src/mlx2/runtime/progressive_external_verify.py",
    "src/mlx2/runtime/external_speculative.py",
    "src/mlx2/adapters/external_draft_policy.py",
)


def _source_hashes(paths):
    result = {}
    for name in paths:
        path = _ROOT / name
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def progressive_target_protocol_binding(model, fidelity):
    """Resolve one explicitly selected target fidelity and bind its sources.

    Exact admission retains the original singular model declaration.  The
    approximate candidate uses a separate declaration map, so its existence
    cannot weaken or alias the row-exact gate.
    """

    if fidelity not in PROGRESSIVE_FIDELITIES:
        raise ValueError("unsupported progressive verification fidelity")
    target_sources = ()
    if fidelity == PROGRESSIVE_FIDELITY_EXACT:
        protocol = getattr(model, "progressive_external_verify_protocol", None)
        if protocol != PROGRESSIVE_TARGET_PROTOCOL_VERSION:
            raise ValueError(
                "progressive verification requires a row-exact progressive "
                "target protocol; the target adapter must implement the "
                "row-exact B1 state-promotion protocol"
            )
    else:
        declarations = getattr(
            model, "progressive_external_approximate_protocols", None
        )
        if not isinstance(declarations, dict):
            raise ValueError(
                "near-tie-bounded progressive verification requires an "
                "explicit target protocol declaration"
            )
        declaration = declarations.get(
            PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION
        )
        if not isinstance(declaration, dict):
            raise ValueError(
                "near-tie-bounded progressive verification requires an "
                "explicit target protocol declaration"
            )
        if declaration.get("fidelity") != fidelity:
            raise ValueError("progressive target protocol fidelity mismatch")
        target_sources = declaration.get("source_paths")
        if (
            not isinstance(target_sources, (list, tuple))
            or not target_sources
            or any(not isinstance(path, str) or not path for path in target_sources)
        ):
            raise ValueError("progressive target protocol has no source binding")
        protocol = PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION
    source_paths = tuple(dict.fromkeys((*_ALGORITHM_SOURCE_PATHS, *target_sources)))
    return {
        "schema": "mlx2.progressive-external-target-binding.v1",
        "algorithm": PROGRESSIVE_EXTERNAL_VERIFY_VERSION,
        "target_protocol": protocol,
        "fidelity": fidelity,
        "source_hashes": _source_hashes(source_paths),
        "qualification_gate_schema": (
            PROGRESSIVE_APPROXIMATE_QUALIFICATION_GATE_SCHEMA
            if fidelity == PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED
            else None
        ),
    }


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
