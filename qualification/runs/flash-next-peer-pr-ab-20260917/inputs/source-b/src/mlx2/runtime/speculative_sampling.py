"""Exact proposal-distribution verification, independent of tensors/devices.

Original mlx2 code. q is the distribution actually used by the drafter,
including its candidate selector; p is the fully transformed target law.
"""
from dataclasses import dataclass
import copy
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
class VerifiedBlock:
    accepted: int
    emitted: tuple[int, ...]
    target_probabilities: tuple[np.ndarray, ...]
    rejected: bool


def verify_proposals(tokens, proposals, targets, rng):
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
