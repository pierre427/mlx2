"""Greedy parity and deterministic park/re-entry contracts."""

import pytest

mx = pytest.importorskip("mlx.core")
from mlx2.experimental.hysparse2.batching import ResearchBatcher
from mlx2.experimental.hysparse2.config import Config
from mlx2.experimental.hysparse2.model import Model
from mlx2.experimental.hysparse2.speculation import GreedyMTPReference
from mlx2.runtime.adaptive_policy import CohortAdaptiveMTPDepth


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(42)
    yield
    mx.set_default_device(previous)


def ordinary(model, prompt, count):
    logits, cache = model.prefill(mx.array([prompt]))
    output = []
    for _ in range(count):
        token = int(mx.argmax(logits[0, -1]).item())
        output.append(token)
        logits = model.decode(mx.array([[token]]), cache)
    return output, logits


def policy():
    return CohortAdaptiveMTPDepth(
        max_depth=1,
        adaptive_single_lane=True,
        ewma_alpha=1,
        loss_rounds=1,
        gain_rounds=1,
        park_rounds=2,
        probe_interval=128,
        min_samples_per_depth=128,
    )


def oracle(model, first, cache):
    clone = ResearchBatcher(model)._merge([cache])
    logits = model.decode(mx.array([[first]]), clone)
    return int(mx.argmax(logits[0, -1]).item())


def test_natural_mtp_greedy_parity():
    model = Model(Config.smoke())
    model.eval()
    prompt = [1, 2, 3, 4, 5]
    expected, next_logits = ordinary(model, prompt, 8)
    got, cache, receipt = GreedyMTPReference(model).generate(prompt, max_tokens=8)
    assert got == expected and cache.length == len(prompt) + 8
    final = model._cross(cache.boundary, cache, cache.length - 1)
    assert float(mx.max(mx.abs(final - next_logits)).item()) < 1e-3
    assert receipt["target_verified"] and not receipt["accelerated_verification"]


def test_forced_rejection_parks_then_reentry_accepts():
    model = Model(Config.smoke())
    model.eval()
    attempts = [0]

    def injected(m, first, cache):
        attempts[0] += 1
        correct = oracle(m, first, cache)
        return (correct + 1) % m.config.vocab_size if attempts[0] == 1 else correct

    runner = GreedyMTPReference(model, policy=policy(), proposal_fn=injected)
    prompt = [1, 2, 3, 4, 5]
    expected, _ = ordinary(model, prompt, 8)
    got, cache, receipt = runner.generate(prompt, max_tokens=8)
    assert got == expected and cache.length == len(prompt) + 8
    assert [r["depth"] for r in receipt["rounds"][:4]] == [1, 0, 0, 1]
    assert receipt["rounds"][3]["accepted"]
    assert receipt["controller_counters"]["parks"] == 1
    assert receipt["controller_counters"]["reentries"] == 1
    assert receipt["proposal"] == "injected-test"


def test_budget_one_never_drafts():
    model = Model(Config.smoke())
    model.eval()

    def forbidden(*_):
        raise AssertionError("last token must not draft beyond output budget")

    _, cache, receipt = GreedyMTPReference(model, proposal_fn=forbidden).generate(
        [1, 2], max_tokens=1
    )
    assert cache.length == 3 and receipt["rounds"][0]["depth"] == 0


def test_proposal_decode_is_isolated_from_verified_target_cache():
    model = Model(Config.smoke())
    model.eval()
    prompt = [1, 2, 3, 4, 5]
    expected, expected_logits = ordinary(model, prompt, 8)
    def proposal(m, first, state):
        next_logits = m.decode(mx.array([[first]]), state)
        return int(mx.argmax(next_logits[0, -1]).item())
    got, cache, receipt = GreedyMTPReference(model, proposal_fn=proposal).generate(prompt, max_tokens=8)
    assert got == expected and cache.length == len(prompt) + 8
    actual = model._cross(cache.boundary, cache, cache.length - 1)
    assert float(mx.max(mx.abs(actual - expected_logits)).item()) == 0
    assert receipt["target_verified"]


def test_cache_fork_shares_tensors_but_not_history_or_identity():
    model = Model(Config.smoke())
    model.eval()
    _, cache = model.prefill(mx.array([[1, 2, 3, 4, 5]]))
    fork = cache.fork()
    assert all(a is b for a, b in zip(cache.arrays(), fork.arrays(), strict=True))
    assert fork.boundary is cache.boundary and fork.ple_history is cache.ple_history
    assert fork.owner is cache.owner and fork.resident_bytes() == cache.resident_bytes()
    fork.apcv2_identity["semantic_fingerprint"] = ("modified",)
    assert fork.apcv2_identity != cache.apcv2_identity
    fork = cache.fork()
    before = [len(blocks) for blocks in cache.cross_kv.values()]
    model.decode(mx.array([[6]]), fork)
    assert cache.length == 5 and fork.length == 6
    assert [len(blocks) for blocks in cache.cross_kv.values()] == before
