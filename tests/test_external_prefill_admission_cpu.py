"""External-route prefill admission charges resident KV once, state planes 4x."""

from types import SimpleNamespace

import mlx.core as mx

from mlx2.runtime.external_speculative import (
    ExternalDraftBatchGenerator,
    _prefill_target_reservation,
)
from mlx2.runtime.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache


def _kv(tokens, cls=KVCache):
    cache = cls(512) if cls is RotatingKVCache else cls()
    cache.update_and_fetch(mx.zeros((1, 2, tokens, 8)), mx.zeros((1, 2, tokens, 8)))
    return cache


def _state():
    cache = ArraysCache(1)
    cache[0] = mx.zeros((1, 3, 64))
    return cache


def test_prefill_reservation_charges_kv_once_and_state_four_times():
    kv, ring, state = _kv(256), _kv(64, RotatingKVCache), _state()
    assert kv.nbytes and ring.nbytes and state.nbytes
    expected = kv.nbytes + ring.nbytes + 4 * state.nbytes
    assert _prefill_target_reservation([kv, ring, state]) == expected
    assert _prefill_target_reservation([CacheList(kv, state), ring]) == expected

    class Unknown:
        nbytes = 100

    assert _prefill_target_reservation([Unknown()]) == 400


def _batch(headroom):
    target = SimpleNamespace(num_key_value_heads=2, head_dim=8, num_hidden_layers=2,
                             hidden_size=16, intermediate_size=32, vocab_size=64)
    draft = SimpleNamespace(num_hidden_layers=0, intermediate_size=0, hidden_size=0,
                            num_key_value_heads=0, head_dim=0, feature_width=0)
    return SimpleNamespace(
        memory_headroom=lambda: headroom, _target_args=target,
        draft=SimpleNamespace(config=draft), layers=(), scheduler_stats={},
        continuation_policy=None, continuation_pool_ast_bridge=False,
    )


def test_resident_kv_no_longer_pushes_a_lone_prefill_slice_off_plan(monkeypatch):
    import mlx2.runtime.external_speculative as ext

    cache = [_kv(4096), _kv(4096)]
    lane = SimpleNamespace(cache=cache, draft_cache=[], remaining=[0] * 2049,
                           tail=mx.zeros((1, 0)))
    resident = sum(c.nbytes for c in cache)
    probe = _batch(1 << 40)
    assert ExternalDraftBatchGenerator._admit(probe, [lane], 2048, prefill=True)
    reserved = probe.scheduler_stats["reservation_bytes"]
    # Headroom between the new reservation and the old 4x-resident one: the
    # planned slice is admitted instead of halved off the APCv2 plan.
    headroom = reserved + resident
    assert ExternalDraftBatchGenerator._admit(_batch(headroom), [lane], 2048, prefill=True)
    monkeypatch.setattr(ext, "_prefill_target_reservation",
                        lambda caches: 4 * sum(c.nbytes for c in caches))
    assert not ExternalDraftBatchGenerator._admit(_batch(headroom), [lane], 2048, prefill=True)
    monkeypatch.undo()
    # Decode rounds keep the 4x copy charge.
    decode = _batch(1 << 40)
    ExternalDraftBatchGenerator._admit(decode, [lane], 1)
    assert decode.scheduler_stats["reservation_bytes"] >= 4 * resident


def test_refused_prefill_admission_records_bounded_headroom_terms():
    cache = [_kv(4096)]
    lane = SimpleNamespace(cache=cache, draft_cache=[], remaining=[0] * 2049,
                           tail=mx.zeros((1, 0)), history=[0] * 300)
    batch = _batch(0)
    for _ in range(20):
        assert not ExternalDraftBatchGenerator._admit(batch, [lane], 2048, prefill=True)
    stats = batch.scheduler_stats
    records = stats["prefill_admission_refusals"]
    assert stats["prefill_admission_refusals_total"] == 20
    assert isinstance(records, tuple) and len(records) == 16
    last = records[-1]
    assert (last["depth"], last["append"], last["lanes"], last["headroom"]) == (300, 2048, 1, 0)
    assert last["required"] == stats["reservation_bytes"]
    assert {"host_term", "advisory_term", "active", "cached"} <= set(last["terms"])
    # Decode refusals are not recorded here.
    decode = _batch(0)
    ExternalDraftBatchGenerator._admit(decode, [lane], 1)
    assert "prefill_admission_refusals" not in decode.scheduler_stats


class _Generator(ExternalDraftBatchGenerator):
    _target_args = None  # shadow the model-derived property


def _generator(headroom_fn, settle=None):
    batch = object.__new__(_Generator)
    fake = _batch(0)
    for name, value in vars(fake).items():
        setattr(batch, name, value)
    batch.memory_headroom = headroom_fn
    batch.reclaim_memory = lambda: None
    batch.evict_checkpoint = None
    batch.settle_memory = settle
    batch._settle_once = None
    batch._settle_pending = False
    batch._reclaims_left = 2
    batch._allocator_reclaimed = False
    return batch


def _admit_planned(batch, lane, rows):
    """The serial prefill admission loop up to (not including) halving."""
    admitted = batch._admit([lane], rows, prefill=True)
    while not admitted and batch._reclaim_for_admission():
        admitted = batch._admit([lane], rows, prefill=True)
    return admitted


def test_reclaim_settles_the_footprint_before_the_planned_slice_is_rechecked():
    lane = SimpleNamespace(cache=[_kv(4096)], draft_cache=[], remaining=[0] * 2049,
                           tail=mx.zeros((1, 0)), history=[0] * 300)
    probe = _batch(1 << 40)
    ExternalDraftBatchGenerator._admit(probe, [lane], 2048, prefill=True)
    required = probe.scheduler_stats["reservation_bytes"]
    # Freed Metal pages leave the footprint only once the settle has waited.
    state = {"settled": False, "calls": 0}

    def headroom():
        return required + 1 if state["settled"] else required // 2

    def settle(timeout=None):
        state["calls"] += 1
        state["settled"] = True
        return {"exit": "baseline", "waited": 0.05, "released_bytes": 1 << 30}

    stale = _generator(headroom)
    assert not _admit_planned(stale, lane, 2048)  # would halve off plan
    settled = _generator(headroom, settle)
    assert _admit_planned(settled, lane, 2048)
    assert state["calls"] == 1
    assert settled.scheduler_stats["external_admission_footprint_settles"] == 1


def test_settle_waits_only_when_the_reclaim_alone_falls_short():
    lane = SimpleNamespace(cache=[_kv(4096)], draft_cache=[], remaining=[0] * 2049,
                           tail=mx.zeros((1, 0)), history=[0] * 300)
    probe = _batch(1 << 40)
    ExternalDraftBatchGenerator._admit(probe, [lane], 2048, prefill=True)
    required = probe.scheduler_stats["reservation_bytes"]
    state = {"reclaimed": False, "settles": 0}

    def settle(timeout=None):
        state["settles"] += 1
        return {"exit": "quiet", "waited": 0.3, "released_bytes": 0}

    batch = _generator(lambda: required + 1 if state["reclaimed"] else 0, settle)

    def reclaim():
        state["reclaimed"] = True

    batch.reclaim_memory = reclaim
    assert _admit_planned(batch, lane, 2048)
    assert state["settles"] == 0


def test_multi_lane_prefill_slab_keeps_the_copy_charge():
    lanes = [SimpleNamespace(cache=[_kv(4096)], draft_cache=[], remaining=[0] * 2049,
                             tail=mx.zeros((1, 0))) for _ in range(2)]
    resident = sum(c.nbytes for lane in lanes for c in lane.cache)
    one, two = _batch(1 << 40), _batch(1 << 40)
    ExternalDraftBatchGenerator._admit(one, lanes[:1], 2048, prefill=True)
    ExternalDraftBatchGenerator._admit(two, lanes, 2048, prefill=True)
    per_lane = one.scheduler_stats["reservation_bytes"]
    # Two lanes: each lane's own append terms plus 4x every resident cache.
    assert two.scheduler_stats["reservation_bytes"] >= 2 * (per_lane - resident // 2) + 4 * resident
