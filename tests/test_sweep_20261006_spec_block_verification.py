"""Block verification for the linear external chain (Sun et al. 2024, Alg. 2).

Exhaustive oracle (``tests/spec_oracle.py``): every RNG branch of the real
verifier is enumerated, so the completed output law is compared with the
target joint exactly, and E[tau] is computed exactly, not sampled.
"""
import math
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from spec_oracle import Problem, SymRequestRNG, check, enumerate_outcomes

from mlx2.runtime.speculative_sampling import (
    FLyVerificationPolicy,
    RequestRNG,
    verify_block_proposals,
    verify_compact_block_proposals,
    verify_proposals,
)

TOL = 1e-9
KINDS = [
    ("random", "random"),
    ("masked", "masked"),   # grammar / top-k zeros on both sides
    ("onehot", "random"),   # point-mass drafts (PLD rows, greedy drafter)
    ("random", "onehot"),   # greedy target
    ("equal", "random"),    # p == q
    ("mixed", "masked"),    # composition: point-mass rows beside stochastic ones
]
CASES = [(V, K, d, t) for V, K in ((3, 2), (4, 3), (5, 2), (3, 4)) for d, t in KINDS]


class MixedProblem(Problem):
    """Draft rows alternate between a point mass and a stochastic law."""

    def Q(self, prefix):
        prefix = tuple(prefix)
        if prefix not in self._q:
            self._q[prefix] = self._law("onehot" if len(prefix) % 2 else "random")
        return self._q[prefix]


def _problem(V, K, draft, target, seed):
    if draft == "mixed":
        return MixedProblem(V, K, seed, draft="random", target=target)
    return Problem(V, K, seed, draft=draft, target=target)


def _ids(case):
    return "V{}K{}-{}-{}".format(*case)


def _dense(ex, draft, proposals, targets):
    return verify_block_proposals(draft, proposals, targets, SymRequestRNG(ex))


def _compact(ex, draft, proposals, targets):
    ids, probs = [], []
    for position, q in enumerate(proposals):
        support = np.flatnonzero(q > 0).tolist()
        extra = [i for i in range(len(q)) if i not in support][:1]
        order = support + extra
        np.random.default_rng(len(order) + position).shuffle(order)
        ids.append(np.array(order, dtype=np.int64))
        probs.append(np.array([q[i] for i in order]))
    return verify_compact_block_proposals(draft, ids, probs, targets, SymRequestRNG(ex))


@pytest.mark.parametrize("verify", [_dense, _compact], ids=["dense", "compact"])
@pytest.mark.parametrize("case", CASES, ids=_ids)
@pytest.mark.parametrize("seed", [0, 1])
def test_block_verification_is_exact(verify, case, seed):
    problem = _problem(*case, seed)
    mass, err = check(problem, lambda *args: verify(*args).emitted)
    assert abs(mass - 1) < TOL and err < TOL, (mass, err)


def _expected_accepted(problem, verify):
    total = 0.0
    for draft, weight in problem.drafts():
        q = [problem.Q(draft[:i]) for i in range(problem.K)]
        p = [problem.P(draft[:i]) for i in range(problem.K + 1)]
        outcomes = enumerate_outcomes(lambda ex: verify(ex, draft, q, p).accepted)  # noqa: B023 - called in-loop
        total += weight * sum(w * accepted for accepted, w in outcomes.items())
    return total


def _token(ex, draft, proposals, targets):
    return verify_proposals(draft, proposals, targets, SymRequestRNG(ex))


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_block_never_accepts_less_and_equals_token_wise_for_point_masses(case):
    problem = _problem(*case, 3)
    token = _expected_accepted(problem, _token)
    block = _expected_accepted(problem, _dense)
    assert block >= token - TOL
    if case[2] == "onehot":
        assert block == pytest.approx(token, abs=TOL)


def test_block_gain_on_stochastic_drafts():
    """The design's exact gain table: about +9% (V4 K3) to +18% (V5 K4)."""
    gains = []
    for V, K in ((4, 3), (5, 4)):
        token = block = 0.0
        for seed in range(4):
            problem = Problem(V, K, seed)
            token += _expected_accepted(problem, _token)
            block += _expected_accepted(problem, _dense)
        gains.append(block / token - 1)
    assert gains[0] > 0.05 and gains[1] > 0.10, gains


# ------------------------------------------------------------- falsifiers
def _reference(ex, draft, q, p, *, drop_one_minus_p=False, unscaled=False, first=False):
    rng = SymRequestRNG(ex)
    count = len(draft)
    uniforms = [rng.uniform() for _ in range(count)]
    mass, masses, tau, failed = 1.0, [1.0], 0, False
    for i in range(count):
        mass = min(1.0, mass * p[i][draft[i]] / q[i][draft[i]])
        masses.append(mass)
        if i == count - 1:
            passing = mass
        else:
            residual = np.maximum(mass * p[i + 1] - q[i + 1], 0).sum()
            denominator = residual + (0.0 if drop_one_minus_p else 1.0 - mass)
            passing = 1.0 if denominator <= 0 else residual / denominator
        if first:
            if failed:
                continue
            if uniforms[i] < passing:
                tau = i + 1
            else:
                failed = True
        elif uniforms[i] < passing:
            tau = i + 1
    if tau == count:
        return tuple(draft) + (rng.sample(p[count]),)
    scale = 1.0 if unscaled else masses[tau]
    residual = np.maximum(scale * p[tau] - q[tau], 0)
    return tuple(draft[:tau]) + (rng.sample(residual if residual.sum() > 0 else p[tau]),)


@pytest.mark.parametrize(
    "variant", [{"drop_one_minus_p": True}, {"unscaled": True}, {"first": True}],
    ids=["h-without-1-minus-P", "residual-without-P-scale", "tau-first-pass"],
)
def test_oracle_detects_wrong_block_rules(variant):
    errors = []
    for seed in range(3):
        problem = Problem(4, 3, seed)
        errors.append(check(problem, lambda ex, d, q, p: _reference(ex, d, q, p, **variant))[1])
    assert max(errors) > 1e-3, errors
    # The faithful reference itself is exact.
    assert check(Problem(4, 3, 0), lambda ex, d, q, p: _reference(ex, d, q, p))[1] < TOL


# ---------------------------------------------------------- RNG contract
class _CountingRNG(RequestRNG):
    def sample(self, law):
        self.categorical = getattr(self, "categorical", 0) + 1
        return super().sample(law)


@pytest.mark.parametrize("count", [0, 1, 4])
def test_block_draws_k_uniforms_then_one_categorical(count):
    rng_laws = np.random.default_rng(count)
    p = [rng_laws.dirichlet(np.ones(6)) for _ in range(count + 1)]
    q = [rng_laws.dirichlet(np.ones(6)) for _ in range(count)]
    tokens = [int(np.argmax(law)) for law in q]
    rng = _CountingRNG(9)
    verify_block_proposals(tokens, q, p, rng)
    assert rng.draws == count + 1 and rng.categorical == 1


def test_block_validates_before_the_first_draw():
    p = [np.full(4, 0.25)] * 3
    q = [np.array([0.0, 0.5, 0.5, 0.0]), np.full(4, 0.25)]
    for verify, proposals in (
        (verify_block_proposals, (q,)),
        (verify_compact_block_proposals, ([np.arange(4)] * 2, q)),
    ):
        rng = RequestRNG(1)
        with pytest.raises(ValueError, match="zero proposal probability"):
            verify([0, 1], *proposals, p, rng)
        assert rng.draws == 0


# ------------------------------------------------- _verify integration
def _block_generator(problem, unprocessed):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    gen = ExternalDraftBatchGenerator.__new__(ExternalDraftBatchGenerator)
    gen.mx = mx
    gen.stops = set()
    gen.fly_verification = FLyVerificationPolicy()
    gen.exact_verification = "block"

    def target_law(lane, logits, history, reachable=True, response_rows=None, greedy_token=False,
                   *, history_suffix=()):
        history = [*history, *history_suffix]
        prefix = tuple(history[len(lane.base) + 1:])
        if response_rows is not None:
            response_rows.append(None)
        # Past a draft the processed target forbids, the real route skips the
        # processors: model that by an unrelated (unprocessed) law.
        return problem.P(prefix) if reachable else unprocessed(prefix)

    gen._target_law = target_law
    return gen


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_verify_block_route_is_exact_with_forbidden_drafts_and_unprocessed_rows(seed):
    from mlx2.runtime.external_speculative import HostDraftRow

    problem = Problem(4, 3, seed, draft="random", target="masked", zero_frac=0.4)
    noise = Problem(4, 3, seed + 50)
    gen = _block_generator(problem, noise.P)

    def verify(ex, draft, proposals, targets):
        lane = SimpleNamespace(
            uid=0, anchor=-1, base=[11, 12], history=[11, 12], processors=[object()],
            rng=SymRequestRNG(ex),
            sampling={"sampling_temp": 0.7, "emit_logprobs": False},
        )
        (decision,) = gen._verify(
            [lane], [HostDraftRow(list(draft), list(proposals))], mx.zeros((1, 4, 4))
        )
        assert lane.block_verified_rounds == 1
        return decision.emitted

    mass, err = check(problem, verify)
    assert abs(mass - 1) < TOL and err < TOL, (mass, err)


def test_block_mode_refuses_routes_it_cannot_decide_for():
    from test_dflash_pair_select import generator, tiny

    m, d = tiny(vocab=64, top_k=16, block_size=8)
    with pytest.raises(ValueError, match="block verification requires"):
        generator(m, d, exact_verification="block", fly_verification=True)
    with pytest.raises(ValueError, match="block verification requires"):
        generator(m, d, exact_verification="block",
                  adaptive_verification={"verification_costs": [1.0, 1.1, 1.2, 1.3]})
    with pytest.raises(ValueError, match="exact_verification must be"):
        generator(m, d, exact_verification="tree")


@pytest.mark.parametrize("temp", [0.0, 0.8])
def test_block_mode_serves_and_reports_the_rule(temp):
    from test_dflash_pair_select import drain, generator, tiny

    m, d = tiny(vocab=64, top_k=16, block_size=8)
    prompts = [[1, 2, 3, 4], [5, 6, 7, 8]]
    runs = {}
    for mode in ("token", "block"):
        b = generator(m, d, exact_verification=mode)
        b.insert(prompts, max_tokens=[10, 10],
                 sampling_configs=[{"sampling_temp": temp}] * 2)
        runs[mode] = drain(b)
    output, receipts = runs["block"]
    receipt = receipts[0]
    assert receipt["exact_verification"]["selected"] == "block"
    if temp == 0:
        # Greedy rows verify by token compare under either rule.
        assert output == runs["token"][0]
        assert receipt["verification"] == "exact"
    else:
        assert receipt["verification"] == "exact_block"
        assert receipt["exact_verification"]["block_verified_rounds"] > 0
    assert "exact_verification" not in runs["token"][1][0]


# ------------------------------------------------- zero uniform (codex P1)
class _ZeroRNG(RequestRNG):
    """A finite RNG whose uniform draws are exactly 0.0, its smallest value."""

    def uniform(self):
        self.draws += 1
        return 0.0


@pytest.mark.parametrize("verify", ["dense", "compact"])
@pytest.mark.parametrize("draft_law", [[1.0, 0.0], [0.5, 0.5]], ids=["onehot", "uniform"])
def test_block_never_accepts_a_draft_the_target_forbids_when_u_is_zero(verify, draft_law):
    # Target law [0, 1] gives the draft 0 no mass: P_1 = 0 and the pass
    # probability is 0, so even u == 0.0 must reject (``u < h``, as the
    # token-wise rule ``u < p/q`` does).
    q = [np.array(draft_law)]
    p = [np.array([0.0, 1.0]), np.array([0.5, 0.5])]
    if verify == "dense":
        result = verify_block_proposals([0], q, p, _ZeroRNG(0))
    else:
        result = verify_compact_block_proposals([0], [np.arange(2)], q, p, _ZeroRNG(0))
    assert result.accepted == 0
    assert result.emitted == (1,)
    token = verify_proposals([0], q, p, _ZeroRNG(0))
    assert token.accepted == 0 and token.emitted == (1,)


def test_block_zero_uniform_lets_no_later_row_rescue_a_forbidden_middle_draft():
    # A forbidden middle draft zeroes P from that row on; no later position
    # may rescue it at u == 0.0.
    q = [np.array([0.5, 0.5])] * 3
    p = [np.array([0.5, 0.5]), np.array([1.0, 0.0]), np.array([0.5, 0.5]),
         np.array([0.5, 0.5])]
    result = verify_block_proposals([0, 1, 0], q, p, _ZeroRNG(0))
    assert result.accepted == 1
    assert result.emitted[:1] == (0,)


def test_self_mtp_accept_rules_reject_a_forbidden_draft_at_u_zero(monkeypatch):
    """The self-MTP rules had the same ``u <= h`` comparison."""
    from mlx2.runtime import hybrid_speculative as hs

    monkeypatch.setattr(hs, "_draw_mtp_acceptance_uniforms", lambda k, rng=None: mx.zeros((k,)))
    monkeypatch.setattr(mx.random, "uniform", lambda *a, **k: mx.zeros(k.get("shape", ())))
    neg = -1e30
    target = mx.array([[neg, 0.0], [math.log(0.5), math.log(0.5)]])
    draft = [mx.array([0.0, neg])]
    tau, _bonus = hs._block_verify(target, draft, [0], 1.0)
    assert tau == 0
    assert not hs._accept_sampled_draft(target[0], draft[0], 0)
    accepted, _bonus = hs._batched_residual_verify(target, draft, [0], 1.0)
    assert accepted == 0
