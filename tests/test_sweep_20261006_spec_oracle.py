"""Exhaustive exactness oracle for every speculative verifier.

Each test enumerates every RNG branch of the real verifier code
(``tests/spec_oracle.py``) and compares the completed output law with the
target joint law.  Ported from the 2026-10-06 speculative sweep; the
self-MTP ``block`` rule case is the only coverage of ``_block_verify``.
"""
from __future__ import annotations

import builtins
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from spec_oracle import (
    Problem,
    SymRequestRNG,
    U,
    branch_law,
    check,
    enumerate_outcomes,
    tv,
)

TOL = 1e-9
CASES = [
    # (V, K, draft kind, target kind)
    (4, 2, "random", "random"),
    (4, 2, "masked", "masked"),
    (4, 2, "onehot", "random"),   # greedy drafter = delta q
    (4, 2, "onehot", "masked"),
    (4, 2, "random", "onehot"),   # greedy target
    (4, 2, "equal", "random"),    # p == q (zero residual mass)
    (5, 3, "random", "masked"),
    (3, 3, "masked", "random"),
]


def _ids(case):
    return "V{}K{}-{}-{}".format(*case)


# ------------------------------------------------- external dense verifier
@pytest.mark.parametrize("case", CASES, ids=_ids)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_verify_proposals_exact(case, seed):
    from mlx2.runtime.speculative_sampling import verify_proposals

    V, K, d, t = case
    problem = Problem(V, K, seed, draft=d, target=t)

    def verify(ex, draft, proposals, targets):
        return verify_proposals(draft, proposals, targets, SymRequestRNG(ex)).emitted

    mass, err = check(problem, verify)
    assert abs(mass - 1) < TOL and err < TOL, (mass, err)


# ------------------------------------------------ external compact verifier
@pytest.mark.parametrize("case", CASES, ids=_ids)
@pytest.mark.parametrize("seed", [0, 1])
def test_verify_compact_proposals_exact(case, seed):
    from mlx2.runtime.speculative_sampling import verify_compact_proposals

    V, K, d, t = case
    problem = Problem(V, K, seed, draft=d, target=t)

    def verify(ex, draft, proposals, targets):
        ids, probs = [], []
        for q in proposals:
            # sparse support in shuffled order, including a zero-q entry
            support = np.flatnonzero(q > 0).tolist()
            extra = [i for i in range(V) if i not in support][:1]
            order = support + extra
            if d != "equal":  # equal: see test_zero_residual_crash
                np.random.default_rng(len(order)).shuffle(order)
            ids.append(np.array(order, dtype=np.int64))
            probs.append(np.array([q[i] for i in order]))
        return verify_compact_proposals(
            draft, ids, probs, targets, SymRequestRNG(ex)
        ).emitted

    mass, err = check(problem, verify)
    assert abs(mass - 1) < TOL and err < TOL, (mass, err)


# ------------------------------------------------ FLy is approximate (label)
def test_fly_is_not_exact_and_is_default_off():
    from mlx2.runtime.speculative_sampling import (
        FLyVerificationPolicy,
        verify_proposals,
    )

    assert FLyVerificationPolicy().enabled is False
    problem = Problem(4, 3, 3, draft="random", target="random")
    policy = FLyVerificationPolicy(enabled=True, entropy_threshold=0.0,
                                   window=1, min_prob=0.0)

    def verify(ex, draft, proposals, targets):
        return verify_proposals(draft, proposals, targets, SymRequestRNG(ex),
                                fly_verification=policy).emitted

    _, err = check(problem, verify)
    assert err > 1e-3  # documents: FLy changes the output law


# ---------------------------------------------------------- tree walk (TF)
def _make_external_tree_verifier(problem):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    gen = ExternalDraftBatchGenerator.__new__(ExternalDraftBatchGenerator)
    gen.stops = set()
    gen.tree_gates = {"logprobs_on_request": False}
    gen.scheduler_stats = {}
    gen._target_law = (
        lambda lane, logits, history, reachable=True, response_rows=None:
        problem.P(tuple(history[len(lane.base):]))
    )
    return gen


def _random_tree(rng, V, nodes, depth):
    """Random proposal tree: unique children per parent, depth-bounded."""
    tokens, parents, depths = [], [], []
    for _ in range(200):
        if len(tokens) == nodes:
            break
        parent = int(rng.integers(-1, len(tokens))) if tokens else -1
        pd = 0 if parent < 0 else depths[parent]
        if pd >= depth:
            continue
        used = {tokens[i] for i, p in enumerate(parents) if p == parent}
        free = [t for t in range(V) if t not in used]
        if not free:
            continue
        tokens.append(int(rng.choice(free)))
        parents.append(parent)
        depths.append(pd + 1)
    return tokens, parents


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("target", ["random", "masked", "onehot"])
def test_tree_walk_exact(seed, target):
    from mlx2.runtime.external_speculative import TreeDraftRow

    V, D = 4, 3
    problem = Problem(V, D, seed, target=target)
    gen = _make_external_tree_verifier(problem)
    rng = np.random.default_rng(100 + seed)
    tokens, parents = _random_tree(rng, V, nodes=7, depth=D)
    block = TreeDraftRow(tokens, parents)

    def run(ex):
        lane = SimpleNamespace(anchor=-1, base=[11, 12], history=None,
                               rng=SymRequestRNG(ex), sampling={"emit_logprobs": False})
        lane.history = list(lane.base)
        # history = base + [anchor] + path tokens; strip base+anchor below
        gen._target_law = (
            lambda ln, logits, history, reachable=True, response_rows=None:
            problem.P(tuple(history[len(ln.base) + 1:]))
        )
        decision = gen._verify_tree(lane, block, np.zeros((1, 64)))
        return tuple(decision.emitted)

    emitted = enumerate_outcomes(run)
    # tree is deterministic (no draft weight); compare to target joint
    n = max(len(e) for e in emitted)
    completed = problem.complete(emitted, n)
    assert tv(completed, problem.target_joint(n)) < TOL


# --------------------------------------------- self-MTP verifiers (MLX code)
class _SymCategorical:
    def __init__(self, ex, logits):
        self.ex = ex
        self.logits = np.asarray(logits.astype(mx.float32), dtype=np.float64)

    def _draw(self, row):
        x = row - row.max()
        law = np.exp(x)
        return branch_law(self.ex, law / law.sum())

    def item(self):
        assert self.logits.ndim == 1
        return self._draw(self.logits)

    def tolist(self):
        if self.logits.ndim == 1:
            return self._draw(self.logits)
        return [self._draw(r) for r in self.logits]


class _SymUniforms:
    def __init__(self, ex, k):
        self.values = [U(ex) for _ in range(k)]

    def __getitem__(self, i):
        return self.values[i]

    def tolist(self):
        return list(self.values)


class _MXProxy:
    def __init__(self, ex):
        self._ex = ex
        outer = self

        class _Random:
            def categorical(self, logits, key=None):
                return _SymCategorical(outer._ex, logits)

            def uniform(self, *a, shape=(), key=None, **k):
                assert tuple(shape) == (), shape
                return U(outer._ex)

        self.random = _Random()

    def eval(self, *args):
        real = [a for a in args if isinstance(a, (mx.array, list, tuple))]
        flat = []
        for a in real:
            if isinstance(a, (list, tuple)):
                flat.extend(x for x in a if isinstance(x, mx.array))
            else:
                flat.append(a)
        if flat:
            mx.eval(*flat)

    def __getattr__(self, name):
        return getattr(mx, name)


def _float(x):
    return x if isinstance(x, U) else builtins.float(x)


def _install(monkeypatch, hs, ex):
    monkeypatch.setattr(hs, "mx", _MXProxy(ex))
    monkeypatch.setattr(hs, "_draw_mtp_acceptance_uniforms",
                        lambda k, rng=None: _SymUniforms(ex, k))
    monkeypatch.setattr(hs, "float", _float, raising=False)
    monkeypatch.setattr(hs, "draw_key", lambda rng: None)
    import mlx2.runtime.generate as gen
    monkeypatch.setattr(gen, "_invalid_output_reason", lambda t, lp: None)


def _logs(law):
    with np.errstate(divide="ignore"):
        return mx.array(np.log(np.asarray(law, dtype=np.float64)).astype(np.float32))


def _hybrid_check(monkeypatch, problem, rule):
    import mlx2.runtime.hybrid_speculative as hs

    def verify(ex, draft, proposals, targets):
        _install(monkeypatch, hs, ex)
        logprobs = mx.stack([_logs(p) for p in targets])
        draft_logprobs = [_logs(q) for q in proposals]
        drafts = list(draft)
        if rule == "block":
            n, bonus = hs._block_verify(logprobs, draft_logprobs, drafts, 1.0)
        elif rule == "batched_residual":
            n, bonus = hs._batched_residual_verify(logprobs, draft_logprobs, drafts, 1.0)
        elif rule == "residual":  # inline loop of _propose_batched_self_mtp_round
            k = len(drafts)
            n = 0
            while n < k and hs._accept_sampled_draft(
                logprobs[n], draft_logprobs[n], drafts[n]
            ):
                n += 1
            if n < k:
                bonus = hs._residual_sample(logprobs[n], draft_logprobs[n], 1.0)
            else:
                bonus = hs._sample_from_logprobs(logprobs[n], 1.0)
        elif rule == "exact":  # inline loop of the "exact" accept rule
            sampled = hs.mx.random.categorical(logprobs).tolist()
            n = 0
            while n < len(drafts) and sampled[n] == drafts[n]:
                n += 1
            bonus = int(sampled[n])
        else:
            raise AssertionError(rule)
        return tuple(drafts[:n]) + (int(bonus),)

    return check(problem, verify)


# float32 laws: tolerance reflects f32 log/exp round trip only.
F32_TOL = 2e-5
HYBRID_CASES = [c for c in CASES if c[3] != "onehot"]


@pytest.mark.parametrize("rule", ["block", "batched_residual", "residual", "exact"])
@pytest.mark.parametrize("case", HYBRID_CASES, ids=_ids)
@pytest.mark.parametrize("seed", [0, 1])
def test_self_mtp_rules_exact(monkeypatch, rule, case, seed):
    V, K, d, t = case
    problem = Problem(V, K, seed, draft=d, target=t)
    mass, err = _hybrid_check(monkeypatch, problem, rule)
    assert abs(mass - 1) < 1e-6 and err < F32_TOL, (mass, err)


def test_oracle_has_teeth():
    """Falsifier: two classic wrong verifiers must be detected."""
    problem = Problem(4, 2, 0)

    def correction_from_target(ex, draft, proposals, targets):
        rng = SymRequestRNG(ex)
        out = []
        for i, x in enumerate(draft):
            if rng.uniform() < min(1.0, targets[i][x] / proposals[i][x]):
                out.append(x); continue
            out.append(rng.sample(targets[i])); return out
        out.append(rng.sample(targets[-1])); return out

    def no_min_ratio_greedy_accept(ex, draft, proposals, targets):
        rng = SymRequestRNG(ex)
        out = []
        for i, x in enumerate(draft):
            if targets[i][x] >= proposals[i][x] * 0.5:
                out.append(x); continue
            out.append(rng.sample(np.maximum(targets[i] - proposals[i], 0) + 1e-300)); return out
        out.append(rng.sample(targets[-1])); return out

    for wrong in (correction_from_target, no_min_ratio_greedy_accept):
        _, err = check(problem, wrong)
        assert err > 1e-3, wrong.__name__


# ------------------------------------------- SPEC-07: zero residual mass
class _AlwaysReject:
    """Uniforms at the top of [0, 1): every ratio below one rejects."""

    def uniform(self):
        return 1 - 2 ** -53

    def sample(self, law):
        return int(np.argmax(law))


def _ulp_equal_laws():
    p = np.array([0.3949407129162019, 0.5922363751276989,
                  0.011504765988698002, 0.0013181459674011687])
    q = p.copy()
    q[0] = np.nextafter(q[0], 2)
    return p, q


@pytest.mark.parametrize("fly", [None, {"enabled": True, "window": 1}])
def test_zero_residual_mass_samples_the_target_law(fly):
    """A rounding-only rejection draws from p instead of escaping the lane."""
    from mlx2.runtime.speculative_sampling import verify_proposals

    p, q = _ulp_equal_laws()
    block = verify_proposals([0], [q], [p, p], _AlwaysReject(), fly_verification=fly)
    assert block.rejected and block.accepted == 0
    assert block.emitted == (int(np.argmax(p)),)


def test_compact_zero_residual_mass_samples_the_target_law():
    from mlx2.runtime.speculative_sampling import verify_compact_proposals

    p, q = _ulp_equal_laws()
    ids = [np.arange(len(q))]
    block = verify_compact_proposals([0], ids, [q], [p, p], _AlwaysReject())
    assert block.rejected and block.emitted == (int(np.argmax(p)),)
