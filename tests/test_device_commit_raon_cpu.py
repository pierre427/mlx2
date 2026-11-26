"""device_round.device_commit (spec-round slice 1a-4 DC-4), CPU only, tiny Raon route.

Design: docs/research/2026-11-25-gpu-resident-spec-rounds-design.md section 4.9.
A tiny random Raon target (4 mHC streams, MLA + RDA layers, MoE) with a
4-stream DFlash2 nd5 drafter (two sliding layers) runs the one-lane device
round with and without ``device_commit``. Pins:

* output tokens, the lane RNG, the composition score/trust states, the
  delivered prompt cache and the sidecar are bitwise those of the
  switch-off device round, with acceptance varied round by round (an
  oracle draft wrapper) and with the default composition + presence penalty;
* the lane enters device form once and leaves it on declines, finishes,
  ordinary/forced rounds, wider cohorts and host redo, and no host reader
  sees device-form caches;
* the per-round recovery point borrows the KV and ring buffers: their
  storage stays in place across rounds (no whole-buffer copy per round),
  and the borrow refuses when it can no longer be exact.
"""

import copy

import mlx.core as mx
import numpy as np
import pytest

from test_external_dflash2_cpu import drain
from test_raon1_mini_adaptive_latch_cpu import NUM_DRAFT, PROMPT, raon_route

from mlx2.runtime import external_speculative as es
from mlx2.runtime.drafters import dflash2
from mlx2.runtime.drafters.device_ring_kv import DeviceRingKVCache
from mlx2.runtime.models.cache import KVCache, RotatingKVCache
from mlx2.runtime.models.device_offset_kv import DeviceOffsetKVCache
from mlx2.runtime.speculative_device_round import (
    DECLINE_REASONS,
    DEVICE_COMMIT_SKIPS,
    DeviceRoundPolicy,
)

mx.set_default_device(mx.cpu)

OFF = {"lazy_draft": True, "decide": True, "processors": True}
ON = {**OFF, "device_commit": True}
LONG = PROMPT + [3, 7, 13, 19, 43, 47, 53, 59, 61, 67, 71, 73]
_DRAFT_LAZY = dflash2.DFlash2DraftModel.draft_lazy  # unwrapped, for the oracle


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _reference(prompt, steps):
    target, _ = raon_route(composed=False)
    cache = target.make_cache()
    logits = target(mx.array([prompt]), cache=cache)
    tokens = []
    for _ in range(steps):
        token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(token)
        logits = target(mx.array([[token]]), cache=cache)
    return tokens


def _oracle(monkeypatch, sequence, holder):
    """Drafts = the target's own continuation, cut at a round-dependent depth.

    The real lazy draft still runs (context append, ring writes, RNG), so the
    device-form drafter is exercised; only its tokens are replaced, and both
    arms see the same replacement for the same lane state.
    """
    real = _DRAFT_LAZY
    rounds = {"n": 0}

    def drafted(self, anchor, hidden, cache, count, *args, **kwargs):
        tokens = real(self, anchor, hidden, cache, count, *args, **kwargs)
        lane = next(iter(holder["engine"].lanes.values()))
        start = len(lane.history) + 1
        want = list(sequence[start:start + count])
        if len(want) < count:
            return tokens
        cut = rounds["n"] % (count + 1)
        rounds["n"] += 1
        if cut < count:
            want[cut] = (want[cut] + 1) % 96
        return mx.array(want, dtype=mx.int32)

    monkeypatch.setattr(dflash2.DFlash2DraftModel, "draft_lazy", drafted)


def _decode(policy, *, prompt=PROMPT, steps=40, composed=False, oracle=None, presence=False,
            stops=None, inspect=None, sampling=None, **options):
    from mlx2.runtime.sample_utils import make_logits_processors

    target, draft = raon_route(composed=composed)
    holder = {}
    if oracle is not None:
        _oracle(oracle, list(prompt) + _reference(prompt, steps), holder)
    options.setdefault("prefill_step_size", 4)
    engine = es.ExternalDraftBatchGenerator(
        target, draft_model=draft, binding="raon-test", num_draft=NUM_DRAFT,
        device_round=policy, **options,
    )
    if stops is not None:
        engine.stops = set(stops)
    holder["engine"] = engine
    processors = ([make_logits_processors(presence_penalty=1.5, presence_context_size=20,
                                          penalty_generation_start=len(prompt))]
                  if presence else None)
    try:
        uid = engine.insert([list(prompt)], max_tokens=[steps],
                            sampling_configs=[sampling or {"sampling_temp": 0.0}],
                            logits_processors=processors)[0]
        got, state, final = [], None, None
        for _ in range(400):
            lane = engine.lanes.get(uid)
            if lane is not None:
                state = (lane.rng.snapshot(), copy.deepcopy(getattr(lane, "proposal_score_counts", None)),
                         copy.deepcopy(getattr(lane, "proposal_trust_states", None)))
                if inspect is not None:
                    inspect(engine, lane)
            _, responses = engine.next()
            for response in responses:
                if response.uid == uid:
                    got.append(response.token)
                    if response.finish_reason:
                        final = response
            if not engine.lanes:
                break
        return got, state, final, dict(engine.scheduler_stats)
    finally:
        engine.close()


def _arrays(value):
    """Every array of a frozen cache list or sidecar, KV read below its offset."""
    out = []

    def visit(item):
        if isinstance(item, mx.array):
            out.append(np.array(item))
        elif type(item) is KVCache and item.keys is not None:
            out.extend([np.array(item.keys[..., :item.offset, :]),
                        np.array(item.values[..., :item.offset, :]), item.offset])
        elif isinstance(item, (list, tuple)):
            for x in item:
                visit(x)
        elif type(item) is RotatingKVCache:
            if item.keys is not None:
                keys = item._temporal_order(item.keys)
                values = item._temporal_order(item.values)
                out.extend([np.array(keys), np.array(values)])
            out.append(item.offset)
        elif hasattr(item, "cache") and isinstance(item.cache, list):
            visit(item.cache)
        elif hasattr(item, "state") and not isinstance(item, (int, float, str)):
            visit(getattr(item, "state"))

    visit(value)
    return out


def _same_arrays(a, b, *, atol=0.0):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        if isinstance(x, np.ndarray):
            assert x.shape == y.shape and x.dtype == y.dtype
            if atol:
                assert np.allclose(x, y, rtol=0, atol=atol)
            else:
                assert np.array_equal(x, y)
        else:
            assert x == y


def _assert_same_finish(off, on):
    """The delivered prompt cache and sidecar, as the switch-off round leaves them.

    Target caches (MLA latent KV, RDA state) and the undrafted tail are
    bitwise. The drafter's sliding context is equal within float32 rounding
    only: the rings project every verify row and the host path only the
    committed ones, and a matmul's rounding depends on its row count (DC-3;
    drafts are held to the near-tie bar, and they decide acceptance only).
    """
    assert off is not None and on is not None
    assert off.finish_reason == on.finish_reason
    assert off.all_tokens == on.all_tokens
    _same_arrays(_arrays(off.prompt_cache), _arrays(on.prompt_cache))
    (off_draft, off_tail), (on_draft, on_tail) = off.cache_sidecar.state, on.cache_sidecar.state
    assert all(type(c) is RotatingKVCache for c in on_draft)
    _same_arrays(_arrays(off_draft), _arrays(on_draft), atol=1e-5)
    assert np.array_equal(np.array(off_tail), np.array(on_tail))
    on.cache_sidecar.validate("raon-test", len(on.all_tokens))
    assert all(type(c) is not DeviceOffsetKVCache for c in on.prompt_cache)


# -- policy -------------------------------------------------------------------------

def test_device_commit_needs_lazy_draft_and_decide_and_refuses_prompt_copy():
    with pytest.raises(ValueError, match="device_commit requires lazy_draft and decide"):
        DeviceRoundPolicy.from_value({"lazy_draft": True, "device_commit": True})
    with pytest.raises(ValueError, match="device_commit requires lazy_draft and decide"):
        DeviceRoundPolicy.from_value({"decide": True, "device_commit": True})
    from mlx2.runtime.speculative_device_round import install

    owner = type("Owner", (), {"scheduler_stats": {}})()
    with pytest.raises(ValueError, match="refused beside prompt_copy_round"):
        install(owner, ON, {"max_tokens": 8})


def test_switch_off_installs_no_device_commit_counters():
    _, _, _, stats = _decode(OFF, steps=6)
    assert "external_device_round_device_commit_rounds" not in stats
    assert DEVICE_COMMIT_SKIPS not in stats


# -- exactness against the switch-off device round -------------------------------------

@pytest.mark.parametrize("composed,presence", [(False, False), (False, True), (True, True)],
                         ids=["plain", "presence", "composed-presence"])
def test_outputs_rng_scores_and_caches_equal_the_switch_off_round(monkeypatch, composed, presence):
    off, off_state, off_final, _ = _decode(OFF, composed=composed, presence=presence,
                                           oracle=monkeypatch)
    on, on_state, on_final, stats = _decode(ON, composed=composed, presence=presence,
                                            oracle=monkeypatch)
    assert on == off and len(on) == 40
    assert on_state == off_state  # lane RNG, lagged scores, trust states
    _assert_same_finish(off_final, on_final)
    rounds = stats["external_device_round_rounds"]
    assert rounds > 0
    assert stats["external_device_round_device_commit_rounds"] == rounds
    assert stats["external_device_round_device_commit_enters"] == 1
    assert stats["external_device_round_device_commit_leaves"] == 1
    assert stats["external_device_round_host_reads"] == rounds
    assert stats["external_device_round_declines"] == 0
    # Acceptance varied: the oracle accepted every depth from 0 to nd5.
    receipt = on_final.speculative_receipt
    assert receipt["accepted"] > 0 and receipt["device_round"]["lane"]["device_commit_rounds"] > 0


def test_random_drafter_and_a_longer_prompt():
    # The random drafter accepts almost nothing: one committed row per round.
    off, off_state, off_final, _ = _decode(OFF, prompt=LONG, steps=30)
    on, on_state, on_final, stats = _decode(ON, prompt=LONG, steps=30)
    assert on == off and on_state == off_state
    _assert_same_finish(off_final, on_final)
    assert stats["external_device_round_device_commit_rounds"] > 0


def test_a_stop_inside_a_round_finishes_from_host_form(monkeypatch):
    sequence = _reference(PROMPT, 40)
    stop = sequence[17]
    off, _, off_final, _ = _decode(OFF, oracle=monkeypatch, stops=[stop])
    on, _, on_final, stats = _decode(ON, oracle=monkeypatch, stops=[stop])
    assert on == off and on[-1] == stop and len(on) < 40
    assert on_final.finish_reason == "stop"
    _assert_same_finish(off_final, on_final)
    assert stats["external_device_round_device_commit_leaves"] == 1


# -- leaving the device form ------------------------------------------------------------

def test_host_redo_restores_device_form_then_runs_on_the_host(monkeypatch):
    off, off_state, off_final, _ = _decode(OFF, presence=True)
    real = dflash2.DFlash2DraftModel.draft_lazy
    calls = []

    def flaky(self, *args, **kwargs):
        tokens = real(self, *args, **kwargs)
        calls.append(1)
        if len(calls) == 4:  # a device-form round asks for a host redo
            return mx.full(tokens.shape, dflash2.DRAFT_LAZY_HOST_REDO, dtype=mx.int32)
        return tokens

    monkeypatch.setattr(dflash2.DFlash2DraftModel, "draft_lazy", flaky)
    on, on_state, on_final, stats = _decode(ON, presence=True)
    assert on == off and on_state == off_state
    _assert_same_finish(off_final, on_final)
    assert stats[DECLINE_REASONS].get("host_redo", 0) == 1
    assert stats["recovery_checkpoint_restores"] == 1
    # Entered, left for the redo's host round, entered again, left at finish.
    assert stats["external_device_round_device_commit_enters"] == 2
    assert stats["external_device_round_device_commit_leaves"] == 2


def test_forced_ordinary_round_leaves_device_form(monkeypatch):
    off, off_state, off_final, _ = _decode(OFF, oracle=monkeypatch)
    real = es.ExternalDraftBatchGenerator._run_round_impl
    seen = {"n": 0}

    def forcing(self, cohort, *, forced_depth=None):
        seen["n"] += 1
        if seen["n"] == 6:
            return real(self, cohort, forced_depth=0)
        return real(self, cohort, forced_depth=forced_depth)

    monkeypatch.setattr(es.ExternalDraftBatchGenerator, "_run_round_impl", forcing)
    on, on_state, on_final, stats = _decode(ON, oracle=monkeypatch)
    monkeypatch.setattr(es.ExternalDraftBatchGenerator, "_run_round_impl", real)
    seen["n"] = 0
    monkeypatch.setattr(es.ExternalDraftBatchGenerator, "_run_round_impl", forcing)
    ref, ref_state, ref_final, _ = _decode(OFF, oracle=monkeypatch)
    assert on == ref and on_state == ref_state
    _assert_same_finish(ref_final, on_final)
    assert stats["external_device_round_device_commit_enters"] == 2
    assert off == ref  # a forced ordinary round changes no greedy token


def test_a_second_lane_declines_the_width_and_leaves_device_form():
    def run(policy):
        target, draft = raon_route(composed=False)
        engine = es.ExternalDraftBatchGenerator(
            target, draft_model=draft, binding="raon-test", num_draft=NUM_DRAFT,
            prefill_step_size=4, device_round=policy)
        try:
            first = engine.insert([list(PROMPT)], max_tokens=[30],
                                  sampling_configs=[{"sampling_temp": 0.0}])[0]
            got = {first: []}
            for step in range(400):
                if step == 6:
                    second = engine.insert([list(reversed(PROMPT))], max_tokens=[10],
                                           sampling_configs=[{"sampling_temp": 0.0}])[0]
                    got[second] = []
                _, responses = engine.next()
                for response in responses:
                    got[response.uid].append(response.token)
                if not engine.lanes:
                    break
            return got, dict(engine.scheduler_stats)
        finally:
            engine.close()

    off, _ = run(OFF)
    on, stats = run(ON)
    assert on == off
    assert stats[DECLINE_REASONS].get("lazy_draft_width", 0) > 0
    assert stats["external_device_round_device_commit_leaves"] >= 2
    assert stats["external_device_round_device_commit_enters"] >= 2


def test_remove_with_prompt_caches_returns_host_form():
    target, draft = raon_route(composed=False)
    engine = es.ExternalDraftBatchGenerator(
        target, draft_model=draft, binding="raon-test", num_draft=NUM_DRAFT,
        prefill_step_size=4, device_round=ON)
    try:
        uid = engine.insert([list(PROMPT)], max_tokens=[40],
                            sampling_configs=[{"sampling_temp": 0.0}])[0]
        for _ in range(8):
            engine.next()
        lane = engine.lanes[uid]
        assert lane.device_commit_form
        assert any(type(c) is DeviceOffsetKVCache for c in lane.cache)
        with pytest.raises(RuntimeError, match="host-form caches"):
            engine._freeze_cache(lane.cache)
        with pytest.raises(RuntimeError, match="host-form caches"):
            engine._sidecar(lane)
        caches = engine.remove([uid], return_prompt_caches=True)
        assert all(type(c) is not DeviceOffsetKVCache for c in caches[uid])
        assert any(type(c) is KVCache for c in caches[uid])
    finally:
        engine.close()


def test_unservable_lanes_keep_the_host_commit():
    from mlx2.runtime import speculative_device_commit as commit

    target, draft = raon_route(composed=False)
    lane = type("Lane", (), {})()
    lane.cache = target.make_cache()
    lane.draft_cache = draft.make_cache()
    lane.tail = mx.zeros((1, 2, 8))
    owner = type("Owner", (), {"draft": draft, "num_draft": NUM_DRAFT})()
    assert commit.lane_reason(owner, lane) is None
    # A pending tail wider than a ring append (num_draft + 1 rows) keeps the
    # host commit; prefill leaves an empty tail and rounds at most nd + 1 rows.
    lane.tail = mx.zeros((1, NUM_DRAFT + 2, 8))
    assert commit.lane_reason(owner, lane) == "long_tail"
    lane.tail = mx.zeros((1, 2, 8))
    lane.draft_cache = [KVCache() for _ in lane.draft_cache]
    assert commit.lane_reason(owner, lane) == "draft_cache"
    lane.draft_cache = draft.make_cache()
    lane.cache = [RotatingKVCache(max_size=4) if type(c) is KVCache else c for c in lane.cache]
    assert commit.lane_reason(owner, lane) == "target_cache"
    lane.cache = target.make_cache()
    # A recurrent row already owned by a speculation window, or a
    # non-recurrent ArraysCache, keeps the host commit (not a failed round).
    from mlx2.runtime.models.cache import ArraysCache

    rda = next(c for c in lane.cache if type(c) is ArraysCache)
    rda.start_speculation()
    assert commit.lane_reason(owner, lane) == "target_cache"
    rda.stop_speculation()
    assert commit.lane_reason(owner, lane) is None
    object.__setattr__(draft.layers[0], "_fused_layer", lambda *a, **k: None)
    assert commit.lane_reason(owner, lane) == "drafter_kernel"


# -- recovery lending: no per-round whole-buffer copy --------------------------------------

def _ptr(array):
    view = np.asarray(array)
    address = view.__array_interface__["data"][0]
    del view
    return address


def test_recovery_points_borrow_kv_and_ring_buffers_in_place():
    # The real drafter (no oracle): its drafts read the rings, so every ring
    # write is evaluated inside the round, while the recovery point is held.
    seen = []

    def inspect(engine, lane):
        if getattr(lane, "device_commit_form", False):
            kv = next(c for c in lane.cache if type(c) is DeviceOffsetKVCache)
            ring = next(c for c in lane.draft_cache if type(c) is DeviceRingKVCache)
            mx.eval(kv.keys, ring.keys)
            seen.append((kv.keys.shape[2], _ptr(kv.keys), _ptr(ring.keys), kv.offset))

    _decode(ON, steps=40, inspect=inspect)
    assert len(seen) > 10
    # Within one KV capacity step the buffers keep their storage round after
    # round: every write was in place despite the per-round recovery point.
    same_capacity = [s for s in seen if s[0] == seen[-1][0]]
    assert len(same_capacity) > 5
    assert len({s[1] for s in same_capacity}) == 1
    assert len({s[2] for s in seen[1:]}) == 1


def test_device_kv_lends_only_at_an_exact_level_and_guards_rewinds():
    from mlx2.runtime.cow_cache import (
        COWCacheError,
        restore_recovery_descriptors,
        snapshot_recovery_descriptors,
    )

    host = KVCache()
    host.update_and_fetch(mx.ones((1, 1, 6, 4)), mx.ones((1, 1, 6, 4)))
    live = DeviceOffsetKVCache.adopt(host)
    snapshot, _, borrowed = snapshot_recovery_descriptors([live])
    assert snapshot[0].keys is None and len(borrowed) == 1
    live.update_and_fetch(mx.zeros((1, 1, 3, 4)), mx.zeros((1, 1, 3, 4)))
    with pytest.raises(RuntimeError, match="no exact recovery level"):
        snapshot_recovery_descriptors([live])
    live.device_commit(mx.array([2], dtype=mx.int32))
    live.observe(8)
    (restored,), _ = restore_recovery_descriptors(snapshot, None, borrowed)
    assert restored.device_mode and restored.bound == 6
    assert mx.array_equal(restored.keys[..., :6, :], mx.ones((1, 1, 6, 4))).item()
    live.observe(5)  # below the borrowed level: a later write could rewrite it
    with pytest.raises(COWCacheError, match="rewound below"):
        restore_recovery_descriptors(snapshot, None, borrowed)


def test_device_ring_borrow_refuses_past_its_slack():
    from mlx2.runtime.cow_cache import (
        COWCacheError,
        restore_recovery_descriptors,
        snapshot_recovery_descriptors,
    )

    ring = DeviceRingKVCache(window=3, slack=2)
    k = mx.arange(8, dtype=mx.float32).reshape(1, 1, 2, 4)
    ring.append(k, k, mx.array([2], dtype=mx.int32))
    tail = mx.zeros((1, 0, 4))
    snapshot, (draft, _), borrowed = snapshot_recovery_descriptors([], ([ring], tail))
    assert draft[0].keys is None and len(borrowed) == 1
    ring.append(k + 100, k + 100, mx.array([1], dtype=mx.int32))
    _, (restored, _) = restore_recovery_descriptors([], (draft, tail), borrowed)
    restored = restored[0]
    assert int(restored.pos.item()) == 2
    visible = np.array(restored.context_mask())
    assert np.array_equal(np.array(restored.keys)[0, 0][visible], np.array(k)[0, 0])
    ring.append(k, k, mx.array([1], dtype=mx.int32))  # 4 rows since the capture > slack 2
    with pytest.raises(COWCacheError, match="more than its slack"):
        restore_recovery_descriptors([], (draft, tail), borrowed)


def test_ring_to_host_holds_the_committed_window():
    ring = DeviceRingKVCache(window=3, slack=2)
    for step in range(4):
        k = mx.full((1, 1, 2, 4), float(step))
        ring.append(k, k, mx.array([1 + step % 2], dtype=mx.int32))
    host = ring.to_host()
    assert type(host) is RotatingKVCache and host.max_size == 3
    assert host.offset == int(ring.pos.item()) and host._idx == 3
    back = DeviceRingKVCache.from_host(host, window=3, slack=2)
    assert np.array_equal(np.array(back.context_mask()).sum(), 3)


def test_a_cost_handoff_engine_keeps_the_host_commit(monkeypatch):
    """A parked lane's batch route reads caches before any round dispatch:
    engines with a reversible cost handoff never take the device form.

    The handoff parks on measured round cost, so both arms run on the
    synthetic clock of the adaptive-latch tests (a speculative round costs
    100 us, an ordinary or parked one 100 ns): on the wall clock the arms
    could park at different rounds and draw different RNG counts."""
    from test_raon1_mini_adaptive_latch_cpu import ast_preset

    clock = {"now": 1}
    real_round = es.ExternalDraftBatchGenerator._round
    monkeypatch.setattr(es.time, "perf_counter_ns", lambda: clock["now"])

    def timed(self, cohort, *, forced_depth=None):
        parked = all(es._speculation_parked(lane) for lane in cohort)
        result = real_round(self, cohort, forced_depth=forced_depth)
        clock["now"] += 100 if forced_depth == 0 or parked else 100_000
        return result

    monkeypatch.setattr(es.ExternalDraftBatchGenerator, "_round", timed)
    off, off_state, _, _ = _decode(OFF, steps=24, cost_per_commit_handoff=ast_preset())
    on, on_state, _, stats = _decode(ON, steps=24, cost_per_commit_handoff=ast_preset())
    assert on == off and on_state == off_state
    assert stats["external_device_round_device_commit_enters"] == 0
    assert stats[DEVICE_COMMIT_SKIPS].get("cost_handoff", 0) > 0


# -- exact block verification ----------------------------------------------------
def test_block_verified_sampled_rounds_keep_the_host_commit():
    """Device block rounds are not validated under a device commit: a sampled
    lane under ``exact_verification: "block"`` with sampled drafts commits on
    the host (counted skip).  (The DFlash2 lazy draft is greedy-only, so on
    this route such rounds decline earlier; the gate is checked directly.)"""
    from types import SimpleNamespace

    from mlx2.runtime.speculative_device_round import DeviceRound

    runtime = DeviceRound.__new__(DeviceRound)
    runtime.policy = DeviceRoundPolicy.from_value({**ON, "sampled_drafts": True})
    lane = SimpleNamespace(sampling={"sampling_temp": 1.0})
    owner = SimpleNamespace(cost_per_commit_handoff=None, exact_verification="block",
                            scheduler_stats={})
    assert runtime._enter_device_commit(owner, [lane], [None]) is False
    assert owner.scheduler_stats[DEVICE_COMMIT_SKIPS] == {"block_verification": 1}


def test_greedy_rounds_under_block_verification_still_commit_on_the_device():
    off = _decode(OFF, exact_verification="block")
    on = _decode(ON, exact_verification="block")
    assert on[0] == off[0]
    assert on[3].get("external_device_round_device_commit_enters", 0) > 0
    assert "block_verification" not in on[3].get(DEVICE_COMMIT_SKIPS, {})
