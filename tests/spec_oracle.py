"""Exhaustive small-vocab oracle for mlx2 speculative verifiers.

Original mlx2 test code (2026-10-06 speculative sweep).

Every random draw the code under test makes is replaced by a *symbolic*
draw whose outcome is enumerated: a uniform becomes an interval that is
split at each comparison, and a categorical draw over a law becomes one
branch per support token (weight = that law's mass).  Running the real
verifier code under a depth-first enumeration of every branch gives the
EXACT output distribution of the implementation (no Monte Carlo).

Each round's emitted tokens are completed to K+1 tokens with the target's
own conditionals; speculative sampling is exact iff that completed joint
equals the target joint over K+1 tokens.
"""
from __future__ import annotations

import itertools

import numpy as np


class _Need(Exception):
    def __init__(self, n):
        self.n = n


class Explorer:
    def __init__(self, script):
        self.script = list(script)
        self.pos = 0
        self.weight = 1.0

    def choose(self, weights):
        if self.pos < len(self.script):
            i = self.script[self.pos]
            self.pos += 1
            self.weight *= weights[i]
            return i
        raise _Need(len(weights))


class U:
    """Symbolic uniform on [lo, hi)."""

    def __init__(self, ex):
        self.ex, self.lo, self.hi = ex, 0.0, 1.0

    def _lt(self, t):  # P(u < t)
        t = float(t)
        if t <= self.lo:
            return False
        if t >= self.hi:
            return True
        width = self.hi - self.lo
        w_true = (t - self.lo) / width
        i = self.ex.choose([w_true, 1.0 - w_true])
        if i == 0:
            self.hi = t
            return True
        self.lo = t
        return False

    # u < t and u <= t differ on a null set only, so the exhaustive law
    # check cannot tell them apart; but a finite RNG does return 0.0, and
    # ``u <= 0`` then accepts a probability-zero event.  Verifiers compare
    # uniforms strictly, and the oracle refuses the non-strict forms.
    def __lt__(self, t):
        return self._lt(t)

    def __le__(self, t):
        raise TypeError("compare a uniform strictly (u < t); u <= t passes at u == 0.0")

    def __gt__(self, t):
        return not self._lt(t)

    def __ge__(self, t):
        raise TypeError("compare a uniform strictly (u < t); u >= t is u < t negated")

    def __float__(self):
        raise TypeError("symbolic uniform used numerically")

    def item(self):
        return self


def branch_law(ex, law):
    law = np.asarray(law, dtype=np.float64)
    law = law / law.sum()
    support = np.flatnonzero(law > 0)
    i = ex.choose([float(law[j]) for j in support])
    return int(support[i])


class SymRequestRNG:
    """Drop-in for speculative_sampling.RequestRNG."""

    def __init__(self, ex):
        self.ex = ex
        self.draws = 0

    def uniform(self):
        self.draws += 1
        return U(self.ex)

    def sample(self, p):
        from mlx2.runtime.speculative_sampling import probability

        self.draws += 1
        return branch_law(self.ex, probability(p))


def enumerate_outcomes(run):
    """run(ex) -> hashable outcome. Returns {outcome: probability}."""
    out = {}
    stack = [[]]
    while stack:
        script = stack.pop()
        ex = Explorer(script)
        try:
            result = run(ex)
        except _Need as need:
            for i in range(need.n):
                stack.append(script + [i])
            continue
        out[result] = out.get(result, 0.0) + ex.weight
    return out


# ---------------------------------------------------------------- problems
class Problem:
    """Random Markov target P(.|prefix) and draft Q(.|prefix), vocab V."""

    def __init__(self, V, K, seed, *, draft="random", target="random",
                 zero_frac=0.3):
        self.V, self.K = V, K
        self.rng = np.random.default_rng(seed)
        self._p, self._q = {}, {}
        self.draft, self.target, self.zero_frac = draft, target, zero_frac

    def _law(self, kind):
        V, r = self.V, self.rng
        if kind == "onehot":
            law = np.zeros(V)
            law[r.integers(V)] = 1
            return law
        law = r.dirichlet(np.full(V, 0.7))
        if kind == "masked":
            mask = r.random(V) < self.zero_frac
            mask[r.integers(V)] = False
            law[mask] = 0
        return law / law.sum()

    def P(self, prefix):
        prefix = tuple(prefix)
        if prefix not in self._p:
            self._p[prefix] = self._law(self.target)
        return self._p[prefix]

    def Q(self, prefix):
        prefix = tuple(prefix)
        if prefix not in self._q:
            if self.draft == "equal":
                self._q[prefix] = self.P(prefix).copy()
            else:
                self._q[prefix] = self._law(self.draft)
        return self._q[prefix]

    def target_joint(self, n):
        out = {}
        for seq in itertools.product(range(self.V), repeat=n):
            w = 1.0
            for i in range(n):
                w *= self.P(seq[:i])[seq[i]]
                if w == 0:
                    break
            if w:
                out[seq] = w
        return out

    def complete(self, emitted_dist, n):
        """Extend each emitted prefix to n tokens with the target law."""
        out = {}
        for emitted, w in emitted_dist.items():
            emitted = tuple(emitted)[:n]
            frontier = {emitted: w}
            while True:
                nxt = {}
                done = True
                for seq, sw in frontier.items():
                    if len(seq) >= n:
                        nxt[seq] = nxt.get(seq, 0) + sw
                        continue
                    done = False
                    law = self.P(seq)
                    for t in np.flatnonzero(law > 0):
                        s2 = seq + (int(t),)
                        nxt[s2] = nxt.get(s2, 0) + sw * law[t]
                frontier = nxt
                if done:
                    break
            for seq, sw in frontier.items():
                out[seq] = out.get(seq, 0) + sw
        return out

    def drafts(self):
        """All draft sequences x with their draft probability."""
        for seq in itertools.product(range(self.V), repeat=self.K):
            w = 1.0
            for i in range(self.K):
                w *= self.Q(seq[:i])[seq[i]]
                if w == 0:
                    break
            if w:
                yield seq, w


def tv(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)


def round_distribution(problem, verify_one):
    """verify_one(ex, draft, proposals, targets) -> emitted tuple."""
    K = problem.K
    total = {}
    for draft, wq in problem.drafts():
        proposals = [problem.Q(draft[:i]) for i in range(K)]
        targets = [problem.P(draft[:i]) for i in range(K + 1)]
        dist = enumerate_outcomes(
            lambda ex: tuple(verify_one(ex, draft, proposals, targets))  # noqa: B023
        )
        for k, v in dist.items():
            total[k] = total.get(k, 0) + wq * v
    return total


def check(problem, verify_one):
    emitted = round_distribution(problem, verify_one)
    mass = sum(emitted.values())
    completed = problem.complete(emitted, problem.K + 1)
    return mass, tv(completed, problem.target_joint(problem.K + 1))
