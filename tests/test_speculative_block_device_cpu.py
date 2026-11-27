"""Device block verification == the host block verifier's decision and law."""
import itertools

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.speculative_block_device import block_decide
from mlx2.runtime.speculative_sampling import _block_accept


class _Uniforms:
    def __init__(self, values):
        self.values = list(values)

    def uniform(self):
        return self.values.pop(0)


def _laws(rng, rows, vocab, *, forbid=False):
    laws = rng.dirichlet(np.ones(vocab) * 0.7, size=rows)
    if forbid:
        laws[:, 0] = 0
        laws /= laws.sum(-1, keepdims=True)
    return laws


@pytest.mark.parametrize("count,vocab", [(1, 3), (2, 4), (4, 5), (6, 7)])
def test_decision_matches_host_block_accept(count, vocab):
    rng = np.random.default_rng(count * 31 + vocab)
    keys = mx.random.split(mx.random.key(1), count + 1)
    for trial in range(60):
        p = _laws(rng, count + 1, vocab, forbid=trial % 5 == 0)
        q = _laws(rng, count, vocab)
        if trial % 7 == 0:
            q[: count // 2 + 1] = p[: count // 2 + 1]          # p = q rows
        drafts = [int(rng.choice(vocab, p=q[i])) for i in range(count)]
        uniforms = rng.random(count)

        def residual_mass(index, mass):
            return float(np.maximum(mass * p[index] - q[index], 0).sum())

        tau, coverage = _block_accept(drafts, [q[i][t] for i, t in enumerate(drafts)],
                                      residual_mass, p, _Uniforms(uniforms))
        with np.errstate(divide="ignore"):
            accepted, last, got = block_decide(
                mx, mx.array(np.log(p), dtype=mx.float32), mx.array(np.log(q), dtype=mx.float32),
                mx.array(drafts, dtype=mx.int32), mx.array(uniforms, dtype=mx.float32),
                keys[:count], keys[count])
        mx.eval(accepted, last, got)
        assert int(accepted) == tau, (trial, drafts, uniforms)
        assert float(got) == pytest.approx(coverage, rel=1e-5, abs=1e-6)
        if tau < count:
            residual = np.maximum(coverage * p[tau] - q[tau], 0)
            support = residual if residual.sum() > 1e-6 else p[tau]
            assert support[int(last)] > 0
        else:
            assert p[count][int(last)] > 0


@pytest.mark.parametrize("small", [7e-8, 1.5e-16])
def test_full_coverage_passes_even_with_tiny_residual(small):
    p = [[1, 0, 0], [1 - small, 0, small], [1, 0, 0]]
    q = [[1, 0, 0], [1 - small, small, 0]]
    _assert_block_decision(p, q, [0, 1], [0.9, 0.5], expected=1)


def test_later_ratios_restore_a_prefix_after_underflow():
    small = np.exp(-400.0)
    rare, certain = [small, 1.0], [1.0, 0.0]
    p = [rare, rare, certain, certain, certain]
    q = [certain, certain, rare, rare]
    _assert_block_decision(p, q, [0] * 4, [0.99] * 4, expected=4)


def _assert_block_decision(p, q, drafts, uniforms, expected):
    p, q = np.asarray(p), np.asarray(q)

    def residual_mass(index, mass):
        return np.maximum(mass * p[index] - q[index], 0).sum()

    tau, coverage = _block_accept(drafts, q[np.arange(len(q)), drafts],
                                  residual_mass, p, _Uniforms(uniforms))
    keys = mx.random.split(mx.random.key(0), len(q) + 1)
    with np.errstate(divide="ignore"):
        device_tau, _, device_coverage = block_decide(
            mx, mx.array(np.log(p), mx.float32), mx.array(np.log(q), mx.float32),
            mx.array(drafts), mx.array(uniforms), keys[:-1], keys[-1])
    assert tau == int(device_tau) == expected
    assert coverage == float(device_coverage) == 1.0


def test_output_law_is_the_target_joint():
    """Drafts ~ q, decide on the device, finish a short round with the next
    round's draw from p_1 (context-free rows): the first two emitted tokens
    must follow p_0 x p_1.  Vectorized trials, Monte Carlo TV bound."""
    rng = np.random.default_rng(7)
    count, vocab, trials = 2, 4, 200_000
    p = _laws(rng, count + 1, vocab)
    q = _laws(rng, count, vocab)
    logp = mx.broadcast_to(mx.array(np.log(p), dtype=mx.float32), (trials, count + 1, vocab))
    logq = mx.broadcast_to(mx.array(np.log(q), dtype=mx.float32), (trials, count, vocab))
    keys = mx.random.split(mx.random.key(11), 2 * count + 3)
    drafts = mx.stack([mx.random.categorical(logq[:, i], key=keys[i]) for i in range(count)],
                      axis=-1).astype(mx.int32)
    uniforms = mx.random.uniform(shape=(trials, count), key=keys[count])
    accepted, last, _ = block_decide(mx, logp, logq, drafts, uniforms,
                                     keys[count + 1: 2 * count + 1], keys[2 * count + 1])
    follow = mx.random.categorical(logp[:, 1], key=keys[2 * count + 2]).astype(mx.int32)
    accepted, last, drafts, follow = (np.array(x) for x in (accepted, last, drafts, follow))
    first = np.where(accepted >= 1, drafts[:, 0], last)
    second = np.where(accepted >= 2, drafts[:, 1], np.where(accepted == 1, last, follow))
    joint = np.zeros((vocab, vocab))
    np.add.at(joint, (first, second), 1)
    joint /= trials
    target = np.outer(p[0], p[1])
    assert 0.5 * np.abs(joint - target).sum() < 0.01
    # Block verification never accepts less than token-wise in expectation.
    token_wise = sum(np.prod([np.minimum(p[i], q[i]).sum() for i in range(k)]) for k in range(1, count + 1))
    assert accepted.mean() >= token_wise - 0.01


def test_shape_validation():
    keys = mx.random.split(mx.random.key(0), 3)
    with pytest.raises(ValueError, match="row counts"):
        block_decide(mx, mx.zeros((2, 3)), mx.zeros((2, 3)), mx.zeros((2,), dtype=mx.int32),
                     mx.zeros((2,)), keys[:2], keys[2])
    with pytest.raises(ValueError, match="width"):
        block_decide(mx, mx.zeros((3, 3)), mx.zeros((2, 4)), mx.zeros((2,), dtype=mx.int32),
                     mx.zeros((2,)), keys[:2], keys[2])
    with pytest.raises(ValueError, match="at least one"):
        block_decide(mx, mx.zeros((1, 3)), mx.zeros((0, 3)), mx.zeros((0,), dtype=mx.int32),
                     mx.zeros((0,)), keys[:0], keys[0])
