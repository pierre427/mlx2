"""Device round (``device_round.lazy_draft``) for the EAGLE chain drafters, CPU.

EAGLE-3, EAGLE-4 (chain only) and Cohere EAGLE declare ``draft_lazy``:
(a) the lazy greedy chain equals ``draft_distributions``' tokens for that
lane, leaves the same committed draft KV, drafter stats and lane RNG;
(b) a ``device_round {lazy_draft, decide}`` decode equals the option-off
decode with the same proposed/accepted counts (rounds with accepts and
rejects); (c) what the lazy chain cannot serve bitwise (sampled lanes,
cohorts wider than one, EAGLE-4 processors, tree rounds) declines with a
counted reason and leaves the output unchanged.  Tiny random weights only.
"""
import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

import test_cohere_eagle_cpu as cohere_tests
import test_eagle3_cpu as eagle3_tests
import test_eagle4_cpu as eagle4_tests

from mlx2.runtime.drafters.dflash2 import DRAFT_LAZY_HOST_REDO
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.proposal_composition import LazyDecline
from mlx2.runtime.speculative_device_round import DECLINE_REASONS
from mlx2.runtime.speculative_sampling import RequestRNG

BOTH = {"lazy_draft": True, "decide": True}
PROC = {**BOTH, "processors": True}
PROMPTS = ([1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [9, 3, 9, 3, 9, 3])

FAMILIES = {
    "eagle3": eagle3_tests.tiny,
    "eagle4": eagle4_tests.tiny,
    # Seed 7: random weights whose chains are sometimes accepted.
    "cohere_eagle": lambda: cohere_tests.tiny(seed=7),
}


def feature_width(draft):
    config = draft.config
    if hasattr(config, "feature_width"):
        return int(config.feature_width)
    return int(config.hidden_size)  # Cohere EAGLE-1: the final-norm feature


def committed(draft, key, length=3):
    """A draft cache holding ``length`` committed features (as after a prefill)."""
    cache = draft.make_cache()
    features = mx.random.normal((1, length, feature_width(draft)), key=mx.random.key(key))
    draft.append_context(features, cache, context_tokens=[[3 + i for i in range(length)]])
    mx.eval([c.state for c in cache])
    return cache


def clone(cache):
    import copy

    return copy.deepcopy(cache)


def assert_same_cache(a, b):
    for x, y in zip(a, b, strict=True):
        assert x.offset == y.offset
        for u, v in zip(x.state, y.state, strict=True):
            np.testing.assert_array_equal(np.asarray(u), np.asarray(v))


def _presence(prompt, penalty=2.5):
    from mlx2.runtime.sample_utils import make_logits_processors

    return make_logits_processors(presence_penalty=penalty, presence_context_size=20,
                                  penalty_generation_start=len(prompt))


# -- (a) lazy chain equals the host chain --------------------------------------


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("depth", [1, 2, 4])
@pytest.mark.parametrize("tail", [1, 3])
def test_draft_lazy_equals_host_chain_and_cache(family, depth, tail):
    _, draft = FAMILIES[family]()
    base = committed(draft, key=40 + depth)
    cache_a, cache_b = clone(base), clone(base)
    hidden = mx.random.normal((1, tail, feature_width(draft)), key=mx.random.key(7 * tail))
    anchor = 11
    pairing = [5, 9, 13][:tail - 1] + [anchor]
    rng_a, rng_b = RequestRNG(4), RequestRNG(4)
    before = dict(draft.stats)
    tokens, _laws = draft.draft_distributions(
        [anchor], hidden, draft.batch_caches([cache_a]), depth, [rng_a], [0.0],
        context_tokens=[pairing])
    host = dict(draft.stats)
    extra = {"rng": rng_b} if draft.draft_lazy_consumes_rng else {}
    lazy = draft.draft_lazy(anchor, hidden, cache_b, depth, context_tokens=[pairing], **extra)
    assert lazy.dtype == mx.int32 and lazy.shape == (depth,)
    assert [int(t) for t in lazy.tolist()] == tokens[0]
    assert_same_cache(cache_a, cache_b)
    assert rng_a.snapshot() == rng_b.snapshot()
    # The lazy call books exactly what the host call booked.
    assert {k: draft.stats[k] - host[k] for k in host} == {k: host[k] - before[k] for k in host}


@pytest.mark.parametrize("family", ["eagle3", "cohere_eagle"])
def test_processed_draft_lazy_equals_processed_host_chain(family):
    _, draft = FAMILIES[family]()
    history = [5, 9, 9, 3, 12, 4]
    processors = _presence(history[:2])
    for depth in (1, 3):
        base = committed(draft, key=60 + depth)
        cache_a, cache_b = clone(base), clone(base)
        hidden = mx.random.normal((1, 2, feature_width(draft)), key=mx.random.key(depth))
        pairing = [4, 9]
        rng_a, rng_b = RequestRNG(5), RequestRNG(5)
        tokens, _ = draft.draft_distributions(
            [9], hidden, draft.batch_caches([cache_a]), depth, [rng_a], [0.0],
            context_tokens=[pairing], logits_processors=[processors],
            processor_histories=[history])
        lazy = draft.draft_lazy(9, hidden, cache_b, depth, rng=rng_b, context_tokens=[pairing],
                                logits_processors=processors, processor_history=history)
        assert [int(t) for t in lazy.tolist()] == tokens[0]
        assert_same_cache(cache_a, cache_b)
        assert rng_a.snapshot() == rng_b.snapshot()


@pytest.mark.parametrize("family", ["eagle3", "cohere_eagle"])
def test_processed_dead_end_falls_back_to_the_raw_row_as_the_host(family):
    _, draft = FAMILIES[family]()
    history = [5, 9]

    def dead(tokens, logits):
        return mx.full(logits.shape, -float("inf"), dtype=logits.dtype)

    dead.probe = dead
    base = committed(draft, key=3)
    cache_a, cache_b = clone(base), clone(base)
    hidden = mx.random.normal((1, 1, feature_width(draft)), key=mx.random.key(2))
    tokens, _ = draft.draft_distributions(
        [9], hidden, draft.batch_caches([cache_a]), 3, [RequestRNG(1)], [0.0],
        context_tokens=[[9]], logits_processors=[[dead]], processor_histories=[history])
    lazy = draft.draft_lazy(9, hidden, cache_b, 3, rng=RequestRNG(1), context_tokens=[[9]],
                            logits_processors=[dead], processor_history=history)
    assert [int(t) for t in lazy.tolist()] == tokens[0]


@pytest.mark.parametrize("family", ["eagle3", "cohere_eagle"])
def test_nan_row_is_redone_on_the_host(family):
    _, draft = FAMILIES[family]()

    def poison(tokens, logits):
        # One NaN beside finite scores: the host ``softmax`` refuses the row.
        return mx.where(mx.arange(logits.shape[-1]) == 0, float("nan"), mx.zeros_like(logits))

    poison.probe = poison
    hidden = mx.random.normal((1, 1, feature_width(draft)), key=mx.random.key(2))
    lazy = draft.draft_lazy(9, hidden, committed(draft, key=4), 2, rng=RequestRNG(1),
                            context_tokens=[[9]], logits_processors=[poison],
                            processor_history=[1])
    assert [int(t) for t in lazy.tolist()] == [DRAFT_LAZY_HOST_REDO] * 2
    with pytest.raises(ValueError, match="selector scores"):
        draft.draft_distributions(
            [9], hidden, draft.batch_caches([committed(draft, key=4)]), 2, [RequestRNG(1)],
            [0.0], context_tokens=[[9]], logits_processors=[[poison]],
            processor_histories=[[1]])


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_draft_lazy_refusals_touch_no_state(family):
    _, draft = FAMILIES[family]()
    width = feature_width(draft)
    hidden = mx.random.normal((1, 2, width), key=mx.random.key(1))
    cache = draft.make_cache()
    with pytest.raises(ValueError, match="end with the anchor"):
        draft.draft_lazy(1, hidden, cache, 2, context_tokens=[[3, 4]])
    with pytest.raises(ValueError, match="following each feature"):
        draft.draft_lazy(4, hidden, cache, 2)
    with pytest.raises(LazyDecline):
        draft.draft_lazy(4, hidden[:, :0], cache, 2, context_tokens=[[]])
    with pytest.raises(ValueError, match="host anchor"):
        draft.draft_lazy(mx.array([4]), hidden, cache, 2, context_tokens=[[3, 4]])
    with pytest.raises(ValueError, match="one lane"):
        draft.draft_lazy(4, mx.concatenate([hidden, hidden]), cache, 2,
                         context_tokens=[[3, 4]])
    with pytest.raises(ValueError, match="greedy only"):
        draft.draft_lazy(4, hidden, cache, 2, lambda v: v, context_tokens=[[3, 4]])
    assert all(c.offset == 0 for c in cache)
    assert draft.stats["draft_calls"] == 0


def test_eagle4_lazy_takes_no_processors():
    _, draft = eagle4_tests.tiny()
    assert draft.draft_lazy_processors is False
    hidden = mx.random.normal((1, 1, feature_width(draft)), key=mx.random.key(1))
    with pytest.raises(ValueError, match="no logits processors"):
        draft.draft_lazy(4, hidden, draft.make_cache(), 2, context_tokens=[[4]],
                         logits_processors=_presence([1]), processor_history=[1])


# -- (b) the device round -------------------------------------------------------


def decode(family, prompts=(PROMPTS[0],), steps=16, sampling=None, processors=None,
           num_draft=3, **options):
    target, draft = FAMILIES[family]()
    batch = ExternalDraftBatchGenerator(target, draft_model=draft, binding="test",
                                        num_draft=num_draft, prefill_step_size=3, **options)
    try:
        insert = {}
        if processors is not None:
            insert["logits_processors"] = [processors(p) for p in prompts]
        uids = batch.insert([list(p) for p in prompts], max_tokens=[steps] * len(prompts),
                            sampling_configs=[dict(sampling or {"sampling_temp": 0})
                                              for _ in prompts], **insert)
        got = {uid: [] for uid in uids}
        rngs = {}
        for _ in range(800):
            for uid in uids:
                if uid in batch.lanes:
                    rngs[uid] = batch.lanes[uid].rng.snapshot()
            _, responses = batch.next()
            for response in responses:
                got[response.uid].append(response.token)
            if not batch.lanes:
                break
        else:
            raise AssertionError("scheduler stalled")
        return ([got[uid] for uid in uids], [rngs.get(uid) for uid in uids],
                dict(batch.scheduler_stats), dict(draft.stats))
    finally:
        batch.close()


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("policy", [BOTH, {"lazy_draft": True}])
def test_device_round_greedy_equals_option_off(family, policy):
    for prompt in PROMPTS:
        off, off_rng, off_stats, off_draft = decode(family, prompts=(prompt,))
        on, on_rng, stats, on_draft = decode(family, prompts=(prompt,), device_round=policy)
        assert on == off and on_rng == off_rng
        rounds = stats["external_device_round_rounds"]
        assert rounds > 0
        assert stats["external_device_round_lazy_draft_lanes"] == rounds
        assert stats["external_device_round_host_reads"] == rounds
        assert stats["external_device_round_declines"] == 0, stats[DECLINE_REASONS]
        assert stats["external_context_token_pairings"] == off_stats[
            "external_context_token_pairings"]
        for key in ("proposed_tokens", "accepted_proposals", "external_rounds"):
            assert stats[key] == off_stats[key], key
        assert on_draft == off_draft


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_device_round_sees_accepts_and_rejects(family):
    proposed = accepted = 0
    for prompt in PROMPTS:
        _, _, stats, _ = decode(family, prompts=(prompt,), steps=24, device_round=BOTH)
        proposed += stats["proposed_tokens"]
        accepted += stats["accepted_proposals"]
    assert 0 < accepted < proposed


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_two_lanes_decline_lazy_draft_width_counted(family):
    off, off_rng, off_stats, _ = decode(family, prompts=PROMPTS[:2])
    on, on_rng, stats, _ = decode(family, prompts=PROMPTS[:2], device_round=BOTH)
    assert on == off and on_rng == off_rng
    reasons = stats[DECLINE_REASONS]
    assert reasons.get("lazy_draft_width", 0) > 0
    assert stats["external_device_round_declines"] == sum(reasons.values())
    for key in ("proposed_tokens", "accepted_proposals"):
        assert stats[key] == off_stats[key], key


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_sampled_lane_declines_counted(family):
    # ``lazy_draft`` alone: a declined round is the host round, bit for bit.
    # (With ``decide`` the host-drafted round's verify is the device round's
    # own sampled law, documented as different tokens for a seed.)
    sampling = {"sampling_temp": 0.8, "sampling_seed": 3}
    off, off_rng, _, _ = decode(family, sampling=sampling)
    on, on_rng, stats, _ = decode(family, sampling=sampling, device_round={"lazy_draft": True})
    assert on == off and on_rng == off_rng
    assert stats["external_device_round_rounds"] == 0
    reasons = stats[DECLINE_REASONS]
    assert reasons.get("lazy_draft_sampled_lane", 0) > 0
    assert stats["external_device_round_declines"] == sum(reasons.values())


@pytest.mark.parametrize("family", ["eagle3", "cohere_eagle"])
def test_presence_penalty_lane_keeps_the_device_round_exactly(family):
    off, off_rng, off_stats, _ = decode(family, processors=_presence)
    on, on_rng, stats, _ = decode(family, processors=_presence, device_round=PROC)
    assert on == off and on_rng == off_rng
    assert stats["external_device_round_rounds"] > 0
    assert stats["external_device_round_declines"] == 0, stats[DECLINE_REASONS]
    for key in ("proposed_tokens", "accepted_proposals"):
        assert stats[key] == off_stats[key], key


def test_eagle4_presence_penalty_lane_declines_counted():
    off, _, _, _ = decode("eagle4", processors=_presence)
    on, _, stats, _ = decode("eagle4", processors=_presence, device_round=PROC)
    assert on == off
    assert stats["external_device_round_rounds"] == 0
    assert stats[DECLINE_REASONS].get("lazy_draft_processors", 0) > 0


def test_eagle4_tree_rounds_stay_on_the_tree_path(monkeypatch):
    """tree15 rounds dispatch to ``_tree_round`` before the device round is
    asked; a chain round of a tree15 engine would decline ``tree_topology``."""
    from mlx2.runtime.speculative_device_round import DeviceRound

    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    off, _, off_stats, _ = decode("eagle4")
    on, _, stats, _ = decode("eagle4", device_round=BOTH)
    assert on == off
    assert stats["external_tree_rounds"] == off_stats["external_tree_rounds"] > 0
    assert stats["external_device_round_rounds"] == 0
    assert set(stats[DECLINE_REASONS]) <= {"tree_topology"}
    target, draft = eagle4_tests.tiny()
    batch = ExternalDraftBatchGenerator(target, draft_model=draft, binding="test", num_draft=3,
                                        device_round=BOTH)
    try:
        assert DeviceRound._engine_reason(batch) == "tree_topology"
    finally:
        batch.close()


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_host_redo_restores_and_runs_the_round_on_the_host(family, monkeypatch):
    _target, draft = FAMILIES[family]()
    model_class = type(draft)
    real = model_class.draft_lazy
    calls = {"n": 0}

    def redo_first(self, *args, **kwargs):
        calls["n"] += 1
        tokens = real(self, *args, **kwargs)  # mutates the draft KV, stats and RNG
        if calls["n"] == 1:
            return mx.full(tokens.shape, DRAFT_LAZY_HOST_REDO, dtype=mx.int32)
        return tokens

    off, off_rng, off_stats, off_draft = decode(family)
    monkeypatch.setattr(model_class, "draft_lazy", redo_first)
    on, on_rng, stats, on_draft = decode(family, device_round=BOTH)
    assert on == off and on_rng == off_rng
    reasons = stats[DECLINE_REASONS]
    assert reasons["host_redo"] == 1
    # The redone round's host-drafted laws are not point masses: its verify
    # stays on the host (counted by ``decide``), as for any host-drafted row.
    assert set(reasons) <= {"host_redo", "non_point_mass_proposals"}
    for key in ("proposed_tokens", "accepted_proposals"):
        assert stats[key] == off_stats[key], key
    assert on_draft == off_draft


def test_cohere_steering_lane_declines_counted():
    """North exposes residual taps; only a lane that asks for steering
    keeps the host round (its verify rows are steered there)."""
    def steer(_prompt):
        mx.random.seed(11)
        return [cohere_tests._Steerer(mx.random.normal((16,)) * 8.0)]

    off, _, off_stats, _ = decode("cohere_eagle", processors=steer)
    on, _, stats, _ = decode("cohere_eagle", processors=steer, device_round=PROC)
    assert on == off
    assert off_stats["external_verify_steer_rounds"] > 0
    assert stats["external_device_round_rounds"] == 0
    assert stats[DECLINE_REASONS].get("residual_steer", 0) > 0
