"""Inkling 2 DSpark opt-in options, CPU only.

* ``inkling2_dspark_kv_headroom``: block K/V written in place on the
  drafter's global layers instead of ``mx.concatenate``; bitwise the same.
* ``inkling2_dspark_fold_q8``: the Markov head reads an 8-bit fold; drafts
  may differ, greedy output still equals ordinary decode.
* ``inkling2_dspark_cost_depth``: per-round depth from a verify cost table
  (the cohort ``adaptive_verification`` policy, plus solo rounds and a
  reject cooldown); greedy output is bitwise that of fixed depth, sampled
  lanes keep fixed depth and its RNG draws.

Mechanisms from g13-engine c91c0180 (MIT), re-implemented; see
docs/PROVENANCE.md.
"""

import mlx.core as mx
import pytest

from mlx2.runtime.models.cache import KVCache, RotatingKVCache

from test_inkling2_dspark_cpu import (  # noqa: F401
    TEXT, WIDTH, _config, _cpu, _hidden, _ordinary_greedy, _random, pair,
)


def _recording():
    store = []

    def sample(logits):
        store.append(logits)
        return mx.argmax(logits, -1)

    return store, sample


@pytest.fixture
def headroom(pair):
    """The shared pair; options and counters are reset afterwards."""
    target, draft, _ = pair
    yield target, draft
    draft.configure_options()
    for layer in draft.layers:
        layer.self_attn.kv_headroom_stats.update(writes=0, grows=0, fallbacks=0)


def _engine_greedy(target, backend, prompt, *, num_draft, max_tokens, temperature=0, **kwargs):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    batch = ExternalDraftBatchGenerator(
        target, draft_model=backend, binding="test", num_draft=num_draft,
        prefill_step_size=16, **kwargs,
    )
    try:
        uid = batch.insert([prompt], max_tokens=[max_tokens],
                           sampling_configs=[{"sampling_temp": temperature}])[0]
        got = []
        for _ in range(2000):
            _, responses = batch.next()
            got += [r.token for r in responses if r.uid == uid]
            if not batch.lanes:
                break
        return got, batch
    finally:
        batch.close()


def _cached_rounds(draft, hidden, splits, anchors):
    """Commit ``hidden`` in pieces, drafting a full block after each piece;
    returns every round's tokens and draft logits, and the cache."""
    cache = draft.make_cache()
    tokens, logits = [], []
    start = 0
    for stop, anchor in zip(splits, anchors):
        draft.commit_context(hidden[:, start:stop], cache)
        store, sample = _recording()
        tokens.append(draft.draft_cached(anchor, hidden[:, stop - 1 : stop], cache, 8, sample).tolist())
        logits.append(store)
        start = stop
    return tokens, logits, cache


def _assert_same_rounds(a, b):
    assert a[0] == b[0]
    for rounds_a, rounds_b in zip(a[1], b[1]):
        assert all(mx.array_equal(x, y) for x, y in zip(rounds_a, rounds_b))
    for ca, cb in zip(a[2], b[2]):
        assert type(ca) is type(cb) and ca.offset == cb.offset
        if isinstance(ca, RotatingKVCache):
            pairs = [(ca._temporal_order(x), cb._temporal_order(y))
                     for x, y in ((ca.keys, cb.keys), (ca.values, cb.values))]
        else:
            pairs = [(x[..., : ca.offset, :], y[..., : cb.offset, :])
                     for x, y in ((ca.keys, cb.keys), (ca.values, cb.values))]
        assert all(mx.array_equal(x, y) for x, y in pairs)


def test_kv_headroom_cached_drafts_are_bitwise_across_growth_and_window_wrap(headroom):
    """Long context (crosses the 256-slot KVCache step twice, local windows
    of 4 wrap on every round): every round's draft logits, tokens and the
    committed K/V below the fill level are bitwise equal with headroom on."""
    _, draft = headroom
    hidden = _hidden(1, 600, seed=31)
    # Uneven commits: a prompt, then accepted-prefix sized rows, landing a
    # block write exactly across each capacity boundary.
    splits = [250, 251, 253, 256, 257, 300, 505, 509, 512, 520, 600]
    anchors = [3, 9, 11, 2, 7, 5, 1, 8, 4, 6, 10]
    off = _cached_rounds(draft, hidden, splits, anchors)
    draft.configure_options(kv_headroom=True)
    on = _cached_rounds(draft, hidden, splits, anchors)
    _assert_same_rounds(off, on)
    receipt = draft.option_receipt["kv_headroom"]
    assert receipt["selected"] and receipt["global_layers"] == 1
    assert receipt["writes"] == len(splits) and receipt["grows"] >= 2
    assert receipt["fallbacks"] == 0
    # The block is never committed: the fill level is the context only.
    assert all(c.offset == 600 for c in on[2])
    assert draft.receipt_settings["kv_headroom"] is True


def test_kv_headroom_fused_draft_block_is_bitwise(headroom):
    _, draft = headroom
    hidden = _hidden(1, 300, seed=32)
    runs = []
    for enabled in (False, True):
        draft.configure_options(kv_headroom=enabled)
        cache, rounds = draft.make_cache(), []
        for start, stop, anchor in ((0, 255, 4), (255, 257, 9), (257, 300, 2)):
            store, sample = _recording()
            tokens = draft.draft_block(anchor, hidden[:, start:stop], cache, 8, sample)
            rounds.append((tokens.tolist(), store))
        runs.append(rounds)
    for (ta, la), (tb, lb) in zip(*runs):
        assert ta == tb and all(mx.array_equal(x, y) for x, y in zip(la, lb))
    assert "kv_headroom" not in draft.configure_options().receipt_settings


def test_kv_headroom_falls_back_on_dtype_mismatch(headroom):
    draft = headroom[1].configure_options(kv_headroom=True)
    attention = next(layer.self_attn for layer in draft.layers if layer.self_attn.window is None)
    cache = KVCache()
    cache.update_and_fetch(mx.zeros((1, 2, 3, 32)), mx.zeros((1, 2, 3, 32)))
    k, v = attention._with_block(cache, cache.keys[..., :3, :], cache.values[..., :3, :],
                                 mx.ones((1, 2, 2, 32), mx.bfloat16), mx.ones((1, 2, 2, 32), mx.bfloat16))
    assert k.shape[2] == 5 and attention.kv_headroom_stats["fallbacks"] == 1
    with pytest.raises(TypeError):
        draft.configure_options(kv_headroom=1)


@pytest.mark.parametrize("num_draft", [3, 7])
def test_kv_headroom_engine_greedy_equals_off_and_ordinary_at_long_prompt(headroom, num_draft):
    """Serving engine, 300-token prompt (rings wrapped, draft KV past one
    growth step): identical greedy tokens with headroom on and off, and both
    equal ordinary decode."""
    from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal

    target, draft = headroom
    mx.random.seed(5)
    prompt = [int(t) for t in mx.random.randint(1, 400, (300,)).tolist()]
    expect = _ordinary_greedy(target, prompt)
    runs = {}
    for enabled in (False, True):
        draft.configure_options(kv_headroom=enabled)
        runs[enabled], _ = _engine_greedy(
            target, Inkling2DSparkExternal(draft, num_draft=num_draft), prompt,
            num_draft=num_draft, max_tokens=12,
        )
    assert runs[True] == runs[False] == expect
    assert draft.option_receipt["kv_headroom"]["writes"] > 0



@pytest.mark.parametrize("key", ["inkling2_dspark_kv_headroom", "inkling2_dspark_fold_q8",
                                 "inkling2_dspark_cost_depth"])
def test_option_policy_is_boolean_and_dspark_only(key, monkeypatch, pair64):
    from mlx2.adapters import inkling2
    from mlx2.adapters.inkling2 import Inkling2Adapter, dspark_options
    from mlx2.runtime.drafters import inkling2_dspark_external

    name = inkling2.DSPARK_OPTIONS[key]
    external = {key: True, "draft_model": "embedded_dspark"}
    parsed = dspark_options(external)
    assert external == {"draft_model": "embedded_dspark"}
    if name == "cost_depth":
        assert parsed[name]["reject_cooldown"] == 8 and parsed[name]["allow_solo"] is False
        assert dspark_options({key: False}) == {name: None}
        pytest.skip("cost_depth is not a drafter option (covered below)")
    assert parsed == {name: True}
    assert dspark_options({key: False}) == {name: False}  # off is harmless anywhere
    for bad in (1, None, "true"):
        with pytest.raises(TypeError):
            dspark_options({key: bad, "draft_model": "embedded_dspark"})
    # Refused on the ordinary route, before any artifact is read.
    with pytest.raises(ValueError, match="DSpark route"):
        Inkling2Adapter("/nonexistent", execution_policy={key: True})
    # The loader applies the options to the drafter it wraps.
    _, draft = pair64
    monkeypatch.setattr("mlx2.adapters.inkling2_dspark.load_drafter", lambda record, target: draft)
    try:
        backend = inkling2_dspark_external.load_external_drafter(
            None, None, num_draft=3, options={name: True})
        assert backend.drafter.option_receipt[name]["selected"] is True
        assert inkling2._drafter_options(backend) == draft.option_receipt
    finally:
        draft.configure_options()


@pytest.fixture(scope="module")
def pair64():
    """The fixture pair with markov_rank 64 (rank 16 has no 8-bit group)."""
    from mlx2.runtime.drafters.inkling2_dspark import Inkling2DSparkConfig, Inkling2DSparkDraftModel
    from mlx2.runtime.models.inkling2 import Model, TextConfig

    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    target = Model(TextConfig.from_dict(TEXT))
    _random(target, 0)
    draft = Inkling2DSparkDraftModel(Inkling2DSparkConfig.from_hf(_config(markov_rank=64)))
    _random(draft, 1)
    draft.bind(target)
    mx.set_default_device(old)
    return target, draft


@pytest.fixture
def q8(pair64):
    target, draft = pair64
    yield target, draft
    draft.configure_options()


def test_fold_q8_quantizes_the_fold_and_switches_back_exactly(q8, pair):
    import numpy as np

    target, draft = q8
    head = draft.markov_head
    bf16 = head._fold_weight
    draft.configure_options(fold_q8=True)
    weight, scales, biases, group = head._fold_q
    assert group == 64 and head._fold_weight is bf16
    restored = mx.dequantize(weight, scales, biases, group_size=64, bits=8)
    # Affine 8-bit: every entry within one quantization step of the fold
    # (MLX rounds the group bias too, so not the ideal half step).
    step = np.abs(np.repeat(np.array(scales), 64, axis=1))
    assert np.all(np.abs(np.array(restored) - np.array(bf16)) <= step + 1e-7)
    receipt = draft.option_receipt["fold_q8"]
    assert receipt["selected"] and receipt["engaged"] and receipt["bytes"] < receipt["bf16_bytes"]
    assert draft.receipt_settings["head_fold"].endswith("+mlx-affine-q8")
    # A rebind that refolds requantizes; turning it off is the BF16 fold again.
    draft._fold_source = None
    draft.bind(target)
    assert head._fold_q is not None and head._fold_q[0] is not weight
    draft.configure_options()
    assert head._fold_q is None and draft.receipt_settings["head_fold"] == "fp64-numpy-4096-row-tiles"
    # markov_rank 16 has no supported 8-bit group size: refused, not rounded.
    with pytest.raises(ValueError, match="divisible by 32"):
        pair[1].configure_options(fold_q8=True)
    assert pair[1].markov_head.fold_q8 is False


def test_fold_q8_proposal_agreement_with_bf16_fold(q8):
    """Record how often 8-bit fold proposals differ from the BF16 fold on
    the tiny model (random weights, so this bounds nothing on a checkpoint)."""
    _, draft = q8
    hidden = _hidden(1, 200, seed=41)
    splits = list(range(20, 201, 9))
    anchors = [(7 * i) % 400 + 1 for i in range(len(splits))]
    off = _cached_rounds(draft, hidden, splits, anchors)[0]
    draft.configure_options(fold_q8=True)
    on = _cached_rounds(draft, hidden, splits, anchors)[0]
    positions = [(a, b) for ra, rb in zip(off, on) for a, b in zip(ra[0], rb[0])]
    first = sum(ra[0][0] != rb[0][0] for ra, rb in zip(off, on))
    differing = sum(a != b for a, b in positions)
    print(f"fold_q8 vs bf16: {differing}/{len(positions)} proposal positions differ, "
          f"{first}/{len(off)} first proposals differ")
    assert len(positions) == 7 * len(splits)
    assert differing < len(positions) // 2  # the 8-bit fold is a close copy


@pytest.mark.parametrize("num_draft", [3, 7])
def test_fold_q8_engine_greedy_equals_ordinary(q8, num_draft):
    from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal

    target, draft = q8
    mx.random.seed(6)
    prompt = [int(t) for t in mx.random.randint(1, 400, (60,)).tolist()]
    expect = _ordinary_greedy(target, prompt)
    draft.configure_options(fold_q8=True, kv_headroom=True)
    got, batch = _engine_greedy(target, Inkling2DSparkExternal(draft, num_draft=num_draft), prompt,
                                num_draft=num_draft, max_tokens=12)
    assert got == expect
    assert batch.scheduler_stats["proposed_tokens"] > 0


# ----- inkling2_dspark_cost_depth -----------------------------------------

# ms per verify of rows 1..8; the steep one is the shape g13-engine reports
# for Inkling-2-Mini on an M1 Max (42 ms at 1 row, 133 ms at 8).
STEEP = [42.0, 67.0, 83.0, 95.0, 102.0, 118.0, 129.0, 133.0]
FLAT = [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5]


def _cost_value(num_draft, verify_ms, draft_ms=9.0, **settings):
    from mlx2.runtime.drafters import inkling2_dspark_cost

    parsed = inkling2_dspark_cost.parse({"verify_ms": verify_ms, "draft_ms": draft_ms,
                                         "min_observations": 1, "full_depth_interval": 1000,
                                         **settings})
    return inkling2_dspark_cost.adaptive_verification(
        parsed, num_draft, verify_ms=parsed["verify_ms"], draft_ms=parsed["draft_ms"],
        source="test_table")


def test_cost_depth_policy_validation():
    from mlx2.adapters.inkling2 import Inkling2Adapter, dspark_options
    from mlx2.runtime.acceptance_estimator import AdaptiveVerificationPolicy
    from mlx2.runtime.drafters.inkling2_dspark_cost import parse

    assert parse(False) is None and parse(True)["measure_context"] == 256
    for bad, error in (({"verify_ms": [1.0, 2.0]}, ValueError), ({"draft_ms": 1.0}, ValueError),
                       ({"verify_ms": [1.0], "draft_ms": 1.0}, ValueError),
                       ({"verify_ms": [1.0, 0.0], "draft_ms": 1.0}, ValueError),
                       ({"verify_ms": [1.0, True], "draft_ms": 1.0}, TypeError),
                       ({"reject_cooldown": 65}, ValueError), ({"reject_cooldown": -1}, TypeError),
                       ({"allow_solo": 1}, TypeError), ({"min_gain": 1.0}, ValueError),
                       ({"unknown": 1}, ValueError), (1, TypeError), (None, TypeError)):
        with pytest.raises(error):
            parse(bad)
    value = _cost_value(3, STEEP, allow_solo=True)
    assert value["verification_costs"] == STEEP[:4] and value["greedy_only"] is True
    policy = AdaptiveVerificationPolicy.from_value(value, 3)
    assert policy.cost_source == "test_table" and policy.allow_solo and policy.reject_cooldown == 8
    with pytest.raises(ValueError, match="rows 1..9"):
        _cost_value(8, STEEP)
    # Cohort mode only; the generic policy refuses these settings elsewhere.
    with pytest.raises(ValueError):
        AdaptiveVerificationPolicy.from_value({**value, "mode": "per_request"}, 3)
    with pytest.raises(ValueError, match="replaces adaptive_verification"):
        dspark_options({"inkling2_dspark_cost_depth": True, "draft_model": "embedded_dspark",
                        "adaptive_verification": value})
    with pytest.raises(ValueError, match="DSpark route"):
        Inkling2Adapter("/nonexistent", execution_policy={"inkling2_dspark_cost_depth": True})


def test_choose_depth_cost_model_solo_and_cooldown():
    """The cost model itself, with a pinned estimator state."""
    import numpy as np

    from mlx2.runtime.acceptance_estimator import AdaptiveVerificationPolicy, OnlineAcceptanceEstimator

    policy = AdaptiveVerificationPolicy.from_value(_cost_value(7, STEEP, allow_solo=True), 7)
    estimator = OnlineAcceptanceEstimator(7, reject_cooldown=policy.reject_cooldown)
    estimator.observed_counts[:] = 10
    estimator.feature_counts[:] = 10
    estimator.feature_means[:] = 0.0
    estimator.slope = 0.0
    estimator.rounds = 1  # round 0 is a forced full-depth exploration round

    def depth_at(probability):
        estimator.intercepts[:] = np.log(probability / (1 - probability))
        return policy.choose_depth(estimator, 7, 1)

    # E[commit | K] / (draft + verify[K]): high acceptance pays for depth,
    # low acceptance picks a solo round, and the fixed cap is kept on a tie.
    assert depth_at(0.98) == 7
    assert depth_at(0.75) == 4  # 3.05 / (9 + 102) beats 3.60 / (9 + 133) by > 5%
    assert depth_at(0.02) == 0
    no_solo = AdaptiveVerificationPolicy.from_value(_cost_value(7, STEEP), 7)
    assert no_solo.choose_depth(estimator, 7, 1) == 1  # the default floor keeps observing
    flat = AdaptiveVerificationPolicy.from_value(_cost_value(7, FLAT), 7)
    estimator.intercepts[:] = np.log(0.6 / 0.4)
    assert flat.choose_depth(estimator, 7, 1) == 7
    # Wider cohorts have no measured table: fixed depth.
    assert policy.choose_depth(estimator, 7, 2) == 7
    # Cooldown: two first-proposal rejections start min(8, 2 * 2) solo rounds.
    estimator.intercepts[:] = np.log(0.98 / 0.02)
    for _ in range(2):
        estimator.observe([0.0] * 7, 0, rejected=True)
        estimator.finish_round()
    assert estimator.cooldown == 4 and estimator.cooldowns_started == 1
    assert [policy.choose_depth(estimator, 7, 1) for _ in range(5)] == [0, 0, 0, 0, 7]
    assert estimator.cooldown_rounds == 4
    estimator.observe([0.0] * 7, 3, rejected=True)  # a good round ends the streak
    estimator.finish_round()
    assert estimator.low_streak == 0
    estimator.observe([0.0] * 7, 0, rejected=False)  # a stop is not a rejection
    estimator.finish_round()
    assert estimator.low_streak == 0 and estimator.cooldown == 0
    # Off by default: the generic estimator never schedules a cooldown.
    plain = OnlineAcceptanceEstimator(7)
    for _ in range(4):
        plain.observe([0.0] * 7, 0, rejected=True)
        plain.finish_round()
    assert plain.cooldown == 0


@pytest.mark.parametrize("num_draft", [3, 7])
@pytest.mark.parametrize("table,settings", [
    (STEEP, {}),                                  # cost model picks short rounds, cooldowns
    (STEEP, {"allow_solo": True}),                # cost-chosen solo rounds too
    (FLAT, {}),                                   # cost model keeps the cap
    (FLAT, {"allow_solo": False, "reject_cooldown": 8, "min_observations": 10_000}),  # cooldown only
    (STEEP, {"reject_cooldown": 0}),              # cost model only, no cooldown
])
def test_cost_depth_engine_greedy_is_bitwise_fixed_depth(pair, num_draft, table, settings):
    """Greedy tokens with the cost-scheduled depth equal fixed depth and
    ordinary decode, whatever depths, solo rounds and cooldowns ran."""
    from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal

    target, draft, _ = pair
    mx.random.seed(9)
    prompt = [int(t) for t in mx.random.randint(1, 400, (40,)).tolist()]
    fixed, _ = _engine_greedy(target, Inkling2DSparkExternal(draft, num_draft=num_draft), prompt,
                              num_draft=num_draft, max_tokens=40)
    value = _cost_value(num_draft, table, **settings)
    got, batch = _engine_greedy(target, Inkling2DSparkExternal(draft, num_draft=num_draft), prompt,
                                num_draft=num_draft, max_tokens=40, adaptive_verification=value)
    assert got == fixed
    assert got[:12] == _ordinary_greedy(target, prompt)
    receipt = batch._adaptive_receipt()["adaptive_verification"]
    histogram = {int(k): v for k, v in receipt["depth_histogram"].items()}
    assert receipt["cost_model_source"] == "test_table" and sum(histogram.values()) > 0
    estimator = batch.acceptance_estimator
    print(f"num_draft={num_draft} settings={settings} depths={dict(sorted(histogram.items()))} "
          f"cooldowns={estimator.cooldowns_started} solo={estimator.cooldown_rounds}")
    if settings.get("min_observations") == 10_000:
        # The cost model never engages; every short round is a cooldown.
        assert estimator.cooldown_rounds > 0 and set(histogram) <= {0, num_draft}
    if table is FLAT and not settings:
        assert any(depth == num_draft for depth in histogram)
    if table is STEEP:
        # Rounds cut below the budget-limited fixed depth (cost model or cooldown).
        assert batch.scheduler_stats["external_adaptive_trimmed_rounds"] > 0
        if not settings:
            assert receipt["reject_cooldown"]["cap"] == 8


def test_cost_depth_sampled_lane_keeps_fixed_depth_rng_draws(pair):
    from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal

    target, draft, _ = pair
    mx.random.seed(10)
    prompt = [int(t) for t in mx.random.randint(1, 400, (30,)).tolist()]
    runs = []
    for value in (None, _cost_value(7, STEEP)):
        got, batch = _engine_greedy(target, Inkling2DSparkExternal(draft, num_draft=7), prompt,
                                    num_draft=7, max_tokens=20, temperature=0.8,
                                    adaptive_verification=value)
        runs.append(got)
    assert runs[0] == runs[1]
    # Every round kept the fixed depth (shorter only at the token budget).
    stats = batch.scheduler_stats
    assert stats["external_adaptive_sampled_fixed_rounds"] == stats["external_adaptive_rounds"] > 0
    assert stats["external_adaptive_trimmed_rounds"] == 0


def test_cost_depth_measured_table_is_deterministic_with_a_fake_clock(pair):
    from mlx2.runtime.drafters.inkling2_dspark_cost import measure_costs

    target, draft, _ = pair

    def fake_clock():
        ticks = iter(range(10_000))
        return lambda: next(ticks) / 1024  # every interval reads 1000/1024 ms, exactly

    before = target.completed_forwards
    tables = [measure_costs(target, draft, 4, context=24, repeats=2, clock=fake_clock()) for _ in range(2)]
    tick = 1000 / 1024
    assert tables[0] == tables[1] == ([tick] * 4, tick)
    assert target.completed_forwards == before  # not reported as served traffic


def test_cost_depth_adapter_builds_and_passes_the_policy(pair, monkeypatch):
    from mlx2.adapters import external_draft_policy, inkling2
    from mlx2.adapters.inkling2 import Inkling2Adapter
    from mlx2.runtime.drafters import inkling2_dspark_cost
    from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal

    target, draft, _ = pair
    adapter = object.__new__(Inkling2Adapter)
    adapter.model, adapter.external_policy = target, {"num_draft": 3}
    adapter.draft_model = Inkling2DSparkExternal(draft, num_draft=3)
    table = inkling2_dspark_cost.parse({"verify_ms": STEEP, "draft_ms": 3.0})
    value = adapter._dspark_cost_policy(table)
    assert value["verification_costs"] == STEEP[:4] and value["cost_source"] == "inkling2_policy_table"
    monkeypatch.setattr(inkling2_dspark_cost, "measure_costs",
                        lambda *a, **k: ([5.0, 9.0, 13.0, 17.0], 2.0))
    measured = adapter._dspark_cost_policy(inkling2_dspark_cost.parse(True))
    assert measured["cost_source"] == "inkling2_measured_at_load"
    assert measured["verification_costs"] == [5.0, 9.0, 13.0, 17.0]
    seen = []
    monkeypatch.setattr(external_draft_policy.ExternalDraftAdapterMixin, "create_external_batch",
                        lambda self, **kw: seen.append(kw) or "batch")
    adapter.dspark_cost_depth = measured
    adapter.create_external_batch()
    assert seen[-1]["adaptive_verification"] is measured
    assert inkling2._drafter_options(adapter.draft_model)["fold_q8"]["selected"] is False


def test_predraft_refuses_untested_pairings():
    import pytest as _pytest

    from mlx2.adapters.inkling2 import Inkling2Adapter

    for extra in ({"inkling2_dspark_kv_headroom": True},
                  {"inkling2_dspark_cost_depth": {"verify_ms": [1.0] * 8, "draft_ms": 1.0}}):
        policy = {"draft_model": "embedded_dspark", "num_draft": 7,
                  "inkling2_dspark_predraft": True, **extra}
        with _pytest.raises(ValueError, match="not yet combined"):
            Inkling2Adapter("/nonexistent-inkling2", execution_policy=policy)
