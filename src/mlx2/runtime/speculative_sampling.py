"""Exact proposal-distribution verification, independent of tensors/devices.

Original mlx2 code. q is the distribution actually used by the drafter,
including its candidate selector; p is the fully transformed target law.
"""
import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


def probability(values):
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("Invalid probability distribution")
    return p / p.sum()


def softmax(scores, temperature=1.0):
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 1 or np.any(np.isnan(x)) or not np.any(np.isfinite(x)):
        raise ValueError("Invalid selector scores")
    if temperature == 0:
        result = np.zeros_like(x); result[np.argmax(x)] = 1; return result
    if temperature < 0:
        raise ValueError("Temperature cannot be negative")
    x = x / temperature
    return probability(np.exp(x - np.max(x)))


class RequestRNG:
    """A lane-owned stream; membership never changes another lane's draws."""
    def __init__(self, seed=0, *, state=None):
        self.generator = np.random.Generator(np.random.PCG64(seed))
        self.draws = 0
        if state is not None:
            self.generator.bit_generator.state = copy.deepcopy(state["generator"])
            self.draws = int(state["draws"])

    def snapshot(self):
        return {"generator": copy.deepcopy(self.generator.bit_generator.state), "draws": self.draws}

    def uniform(self):
        self.draws += 1
        return float(self.generator.random())

    def sample(self, p):
        p = probability(p)
        return min(int(np.searchsorted(np.cumsum(p), self.uniform(), side="right")), len(p)-1)


@dataclass(frozen=True)
class FLyVerificationPolicy:
    """Default-off entropy-gated deferred verification (vllm#53987 design)."""

    enabled: bool = False
    entropy_threshold: float = 2.0
    window: int = 2
    min_prob: float = 0.01

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("FLy enabled must be boolean")  # noqa: TRY004
        if (
            isinstance(self.entropy_threshold, bool)
            or not isinstance(self.entropy_threshold, (int, float))
            or not math.isfinite(float(self.entropy_threshold))
            or self.entropy_threshold < 0
        ):
            raise ValueError("FLy entropy_threshold must be finite and nonnegative")
        if (
            isinstance(self.window, bool)
            or not isinstance(self.window, int)
            or self.window < 1
        ):
            raise ValueError("FLy window must be a positive integer")
        if (
            isinstance(self.min_prob, bool)
            or not isinstance(self.min_prob, (int, float))
            or not math.isfinite(float(self.min_prob))
            or not 0 <= self.min_prob <= 1
        ):
            raise ValueError("FLy min_prob must be finite and in [0, 1]")

    @classmethod
    def from_value(cls, value):
        if value is None or value is False:
            return cls()
        if value is True:
            return cls(enabled=True)
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(  # noqa: TRY004
                "fly_verification must be a boolean or object"
            )
        allowed = {"enabled", "entropy_threshold", "window", "min_prob"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown FLy verification settings: {sorted(unknown)}")
        return cls(**dict(value))

    def as_dict(self):
        return {
            "enabled": self.enabled,
            "entropy_threshold": float(self.entropy_threshold),
            "window": self.window,
            "min_prob": float(self.min_prob),
        }


def apply_fly_relaxation(tokens, targets, ordinary_accepts, policy):
    """Return ``(accepted_prefix, relaxed_count)`` under the FLy rule.

    A failed position is relaxed only when its target entropy/probability gates
    pass and all W following proposal decisions pass the ordinary verifier.
    Those W tokens are therefore accepted normally; scanning resumes after the
    deferred window. Processed/structured laws are excluded by the caller.
    """
    policy = FLyVerificationPolicy.from_value(policy)
    tokens = tuple(int(token) for token in tokens)
    targets = tuple(probability(law) for law in targets)
    ordinary_accepts = tuple(bool(value) for value in ordinary_accepts)
    if len(tokens) != len(targets) or len(tokens) != len(ordinary_accepts):
        raise ValueError("FLy verification vectors must have equal length")
    if not policy.enabled:
        accepted = 0
        while accepted < len(tokens) and ordinary_accepts[accepted]:
            accepted += 1
        return accepted, 0
    index = 0
    relaxed = 0
    while index < len(tokens):
        if ordinary_accepts[index]:
            index += 1
            continue
        stop = index + 1 + policy.window
        law = targets[index]
        positive = law[law > 0]
        entropy = float(-np.sum(positive * np.log(positive)))
        if (
            stop > len(tokens)
            or entropy < policy.entropy_threshold
            or law[tokens[index]] < policy.min_prob
            or not all(ordinary_accepts[index + 1 : stop])
        ):
            break
        relaxed += 1
        index = stop
    return index, relaxed


@dataclass(frozen=True)
class VerifiedBlock:
    accepted: int
    emitted: tuple[int, ...]
    target_probabilities: tuple[np.ndarray, ...]
    rejected: bool
    relaxed_accepts: int = 0


def verify_proposals(tokens, proposals, targets, rng, *, fly_verification=None):
    """Accept q draws with min(1,p[x]/q[x]); rejection draws normalized(p-q)+.

    targets contains K+1 conditional target laws; the final is the bonus law.
    Every law is checked before draws/state changes. The caller atomically
    commits anchor+accepted proposal inputs, leaving correction/bonus pending.
    """
    tokens = tuple(int(t) for t in tokens)
    q = tuple(probability(v) for v in proposals)
    p = tuple(probability(v) for v in targets)
    if len(q) != len(tokens) or len(p) != len(tokens)+1:
        raise ValueError("Proposal/target count mismatch")
    if any(v.shape != p[0].shape for v in (*q, *p)):
        raise ValueError("Vocabulary mismatch")
    for token, law in zip(tokens, q):
        if not 0 <= token < len(law) or law[token] <= 0:
            raise ValueError("Proposed token has zero proposal probability")
    fly = FLyVerificationPolicy.from_value(fly_verification)
    if fly.enabled:
        ordinary_accepts = []
        for i, token in enumerate(tokens):
            ordinary_accepts.append(
                rng.uniform() < min(1.0, p[i][token] / q[i][token])
            )
        accepted, relaxed = apply_fly_relaxation(
            tokens, p[: len(tokens)], ordinary_accepts, fly
        )
        emitted = list(tokens[:accepted])
        laws = list(p[:accepted])
        if accepted < len(tokens):
            residual = np.maximum(p[accepted] - q[accepted], 0)
            if residual.sum() <= 0:
                raise ArithmeticError("Rejected proposal has no residual mass")
            emitted.append(rng.sample(residual)); laws.append(p[accepted])
            return VerifiedBlock(
                accepted, tuple(emitted), tuple(laws), True, relaxed
            )
        emitted.append(rng.sample(p[-1])); laws.append(p[-1])
        return VerifiedBlock(
            len(tokens), tuple(emitted), tuple(laws), False, relaxed
        )
    emitted, laws = [], []
    for i, token in enumerate(tokens):
        if rng.uniform() < min(1.0, p[i][token] / q[i][token]):
            emitted.append(token); laws.append(p[i]); continue
        residual = np.maximum(p[i] - q[i], 0)
        if residual.sum() <= 0:
            raise ArithmeticError("Rejected proposal has no residual mass")
        emitted.append(rng.sample(residual)); laws.append(p[i])
        return VerifiedBlock(i, tuple(emitted), tuple(laws), True)
    emitted.append(rng.sample(p[-1])); laws.append(p[-1])
    return VerifiedBlock(len(tokens), tuple(emitted), tuple(laws), False)


def verify_compact_proposals(
    tokens, candidate_ids, candidate_probs, targets, rng, *, fly_verification=None
):
    """Verify pairwise DFlash proposals without expanding sparse q to vocabulary size.

    Keep the dense verifier as the oracle for host and processor routes.  The
    proposal rows contain unique vocabulary IDs; target laws remain dense.
    Validation finishes before the first RNG draw, as in ``verify_proposals``.
    """
    tokens = tuple(int(token) for token in tokens)
    ids = tuple(np.asarray(row) for row in candidate_ids)
    q = tuple(probability(row) for row in candidate_probs)
    p = tuple(probability(row) for row in targets)
    if len(q) != len(tokens) or len(ids) != len(tokens) or len(p) != len(tokens) + 1:
        raise ValueError("Proposal/target count mismatch")
    vocab = len(p[0])
    if any(row.shape != p[0].shape for row in p):
        raise ValueError("Vocabulary mismatch")
    selected_q = []
    for token, row_ids, row_q in zip(tokens, ids, q):
        if (
            row_ids.ndim != 1 or not np.issubdtype(row_ids.dtype, np.integer)
            or row_ids.shape != row_q.shape
            or np.any(row_ids < 0) or np.any(row_ids >= vocab)
            or len(np.unique(row_ids)) != len(row_ids)
        ):
            raise ValueError("Vocabulary mismatch")
        matches = np.flatnonzero(row_ids == token)
        selected = float(row_q[matches[0]]) if len(matches) else 0.0
        if not 0 <= token < vocab or selected <= 0:
            raise ValueError("Proposed token has zero proposal probability")
        selected_q.append(selected)

    def residual_at(index):
        residual = p[index].copy()
        residual[ids[index]] -= q[index]
        np.maximum(residual, 0, out=residual)
        if residual.sum() <= 0:
            raise ArithmeticError("Rejected proposal has no residual mass")
        return residual

    fly = FLyVerificationPolicy.from_value(fly_verification)
    if fly.enabled:
        ordinary_accepts = [
            rng.uniform() < min(1.0, p[i][token] / selected_q[i])
            for i, token in enumerate(tokens)
        ]
        accepted, relaxed = apply_fly_relaxation(
            tokens, p[: len(tokens)], ordinary_accepts, fly
        )
        emitted = list(tokens[:accepted])
        laws = list(p[:accepted])
        if accepted < len(tokens):
            emitted.append(rng.sample(residual_at(accepted)))
            laws.append(p[accepted])
            return VerifiedBlock(accepted, tuple(emitted), tuple(laws), True, relaxed)
        emitted.append(rng.sample(p[-1])); laws.append(p[-1])
        return VerifiedBlock(len(tokens), tuple(emitted), tuple(laws), False, relaxed)
    emitted, laws = [], []
    for i, token in enumerate(tokens):
        if rng.uniform() < min(1.0, p[i][token] / selected_q[i]):
            emitted.append(token); laws.append(p[i]); continue
        emitted.append(rng.sample(residual_at(i))); laws.append(p[i])
        return VerifiedBlock(i, tuple(emitted), tuple(laws), True)
    emitted.append(rng.sample(p[-1])); laws.append(p[-1])
    return VerifiedBlock(len(tokens), tuple(emitted), tuple(laws), False)
