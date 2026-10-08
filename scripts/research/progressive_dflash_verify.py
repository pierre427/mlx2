#!/usr/bin/env python3
"""Research-only progressive target verification for one DFlash proposal.

The drafter produces one complete linear proposal and its exact per-position
laws before this module is called.  The target then verifies that same proposal
in bounded tiles.  A fully accepted non-final tile does *not* draw a bonus;
only a rejection or the final tile draws the correction/bonus token.

This module deliberately has no serving, APCv2, scheduler, or model imports.
Callers must give it a request-private target cache and a ``prepare_stage``
callback returning a transactional target forward.  It is a CPU research seam,
not a selectable route.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mlx2.runtime.speculative_sampling import (
    RequestRNG,
    VerifiedBlock,
    probability,
    verify_proposals,
)


@dataclass(frozen=True)
class PreparedStage:
    """One private target forward and its rollback-capable transaction."""

    target_laws: tuple[np.ndarray, ...]
    features: Any
    transaction: Any


@dataclass(frozen=True)
class ProgressiveVerificationResult:
    """Materialized private result; callers decide whether to publish it."""

    emitted: tuple[int, ...]
    accepted: int
    committed_inputs: tuple[int, ...]
    cache: Any
    feature_parts: tuple[Any, ...]
    rng_state: dict[str, Any]
    target_rows: int
    target_launches: int
    rejected: bool
    verification_tile: int


PrepareStage = Callable[[Any, tuple[int, ...], tuple[int, ...]], PreparedStage]


def _validate_proposal(tokens: Sequence[int], laws: Sequence[np.ndarray]) -> None:
    if not tokens or len(tokens) != len(laws):
        raise ValueError("progressive verification needs one law per proposal token")
    normalized = tuple(probability(law) for law in laws)
    vocab = len(normalized[0])
    if any(len(law) != vocab for law in normalized):
        raise ValueError("proposal vocabulary mismatch")
    for token, law in zip(tokens, normalized):
        if type(token) is not int or not 0 <= token < vocab or law[token] <= 0:
            raise ValueError("proposed token has zero proposal probability")


def verify_proposal_tile(
    tokens: Sequence[int],
    proposal_laws: Sequence[np.ndarray],
    target_laws: Sequence[np.ndarray],
    rng: RequestRNG,
    *,
    final: bool,
) -> VerifiedBlock:
    """Verify one proposal slice without an intermediate bonus draw.

    The final form delegates to the ordinary exact verifier.  The non-final
    form has one target law per proposal token.  It consumes the same ordered
    acceptance uniforms and residual draw as token-wise verification, but a
    full accept returns only the proposals and leaves the next proposal row to
    the following target stage.
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


def progressive_verify(
    tokens: Sequence[int],
    proposal_laws: Sequence[np.ndarray],
    *,
    cache: Any,
    anchor: int,
    history: Sequence[int],
    rng: RequestRNG,
    verification_tile: int,
    prepare_stage: PrepareStage,
) -> ProgressiveVerificationResult:
    """Verify one precomputed proposal through request-private target stages."""

    tokens = tuple(tokens)
    laws = tuple(proposal_laws)
    _validate_proposal(tokens, laws)
    if type(anchor) is not int or anchor < 0:
        raise ValueError("anchor must be a nonnegative token id")
    if (
        type(verification_tile) is not int
        or not 1 <= verification_tile <= len(tokens)
    ):
        raise ValueError("verification tile must be within the proposal length")

    local_history = list(history)
    committed_inputs: list[int] = []
    emitted: list[int] = []
    feature_parts: list[Any] = []
    accepted = 0
    target_rows = 0
    launches = 0
    offset = 0
    rejected = False

    while offset < len(tokens):
        remaining = len(tokens) - offset
        final = remaining <= verification_tile
        count = remaining if final else verification_tile
        stage_tokens = tokens[offset : offset + count]
        stage_laws = laws[offset : offset + count]
        # A non-final target stage predicts T proposals from T input rows:
        # anchor, followed by the first T-1 proposals.  The Tth proposal stays
        # pending as the next anchor after a full accept.  The final stage adds
        # every remaining proposal input so its final row supplies the bonus.
        inputs = (
            (anchor, *stage_tokens)
            if final
            else (anchor, *stage_tokens[:-1])
        )
        prepared = prepare_stage(cache, inputs, tuple(local_history))
        transaction = prepared.transaction
        expected_laws = count + int(final)
        if len(prepared.target_laws) != expected_laws:
            if not transaction.closed:
                transaction.abort()
            raise ValueError("target stage returned the wrong law count")
        if int(prepared.features.shape[1]) != len(inputs):
            if not transaction.closed:
                transaction.abort()
            raise ValueError("target feature rows do not match staged inputs")
        launches += 1
        target_rows += len(inputs)
        try:
            outcome = verify_proposal_tile(
                stage_tokens,
                stage_laws,
                prepared.target_laws,
                rng,
                final=final,
            )
            consumed = min(outcome.accepted + 1, len(outcome.emitted))
            cache = transaction.commit(accepted_lengths=[consumed])[0]
        except BaseException:
            if not transaction.closed:
                transaction.abort()
            raise

        feature_parts.append(prepared.features[:, :consumed])
        committed = inputs[:consumed]
        committed_inputs.extend(committed)
        local_history.extend(committed)
        emitted.extend(outcome.emitted)
        accepted += outcome.accepted
        anchor = int(outcome.emitted[-1])
        if outcome.rejected:
            rejected = True
            break
        offset += count
        if final:
            break

    return ProgressiveVerificationResult(
        emitted=tuple(emitted),
        accepted=accepted,
        committed_inputs=tuple(committed_inputs),
        cache=cache,
        feature_parts=tuple(feature_parts),
        rng_state=rng.snapshot(),
        target_rows=target_rows,
        target_launches=launches,
        rejected=rejected,
        verification_tile=verification_tile,
    )


def fixed_verify(
    tokens: Sequence[int],
    proposal_laws: Sequence[np.ndarray],
    *,
    cache: Any,
    anchor: int,
    history: Sequence[int],
    rng: RequestRNG,
    prepare_stage: PrepareStage,
) -> ProgressiveVerificationResult:
    """One-forward reference for the same precomputed proposal."""

    tokens = tuple(tokens)
    laws = tuple(proposal_laws)
    _validate_proposal(tokens, laws)
    inputs = (anchor, *tokens)
    prepared = prepare_stage(cache, inputs, tuple(history))
    transaction = prepared.transaction
    try:
        outcome = verify_proposal_tile(
            tokens,
            laws,
            prepared.target_laws,
            rng,
            final=True,
        )
        consumed = min(outcome.accepted + 1, len(outcome.emitted))
        cache = transaction.commit(accepted_lengths=[consumed])[0]
    except BaseException:
        if not transaction.closed:
            transaction.abort()
        raise
    return ProgressiveVerificationResult(
        emitted=tuple(outcome.emitted),
        accepted=outcome.accepted,
        committed_inputs=tuple(inputs[:consumed]),
        cache=cache,
        feature_parts=(prepared.features[:, :consumed],),
        rng_state=rng.snapshot(),
        target_rows=len(inputs),
        target_launches=1,
        rejected=outcome.rejected,
        verification_tile=len(tokens),
    )


def _plan() -> dict[str, Any]:
    return {
        "schema": "mlx2.dflash-progressive-verify.v1",
        "mode": "cpu_research_harness",
        "mechanism": "one_precomputed_proposal_progressive_target_tiles",
        "constraints": {
            "batch_width": 1,
            "verification": "token_wise_exact",
            "logits_processors": False,
            "apcv2_publication": False,
            "state_visibility": "request_private_until_caller_promotion",
            "intermediate_bonus_draw": False,
        },
        "research_harness_state": {
            "implemented": True,
            "qualified": False,
            "selected": False,
            "observed_used": False,
            "performance_claim": False,
        },
        "serving_route_state": {
            "implemented": False,
            "qualified": False,
            "selected": False,
            "observed_used": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    payload = _plan()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
