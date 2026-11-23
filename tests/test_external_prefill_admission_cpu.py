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


# -- append-only caches that are not KVCache (Hy4-flash latent/indexer) -------


def _hy4_caches(tokens):
    """One indexed (append-only kv/tk/gk) and one sliding (ring) Hy4 layer."""
    from mlx2.runtime.models import hy_v4_flash as H

    indexed = H.HY4FlashLayerCache(H.INDEXED, ratio=4)
    indexed.append(mx.zeros((1, tokens, 24)))
    H._buf_append(indexed, "tk", mx.zeros((1, tokens, 8)))
    H._buf_append(indexed, "gk", mx.zeros((1, tokens // 4, 8), dtype=mx.float32))
    sliding = H.HY4FlashLayerCache(H.SLIDING, window=16)
    sliding.append(mx.zeros((1, 16, 24)))
    return indexed, sliding


def test_declared_append_only_buffers_are_charged_once():
    indexed, sliding = _hy4_caches(512)
    assert indexed.nbytes and sliding.nbytes
    # Indexed latent/indexer buffers are declared append-only: charged once.
    # The sliding ring compacts (rewrites rows) and keeps the 4x charge.
    assert _prefill_target_reservation([indexed]) == indexed.nbytes
    assert _prefill_target_reservation([sliding]) == 4 * sliding.nbytes
    assert _prefill_target_reservation([CacheList(indexed, sliding)]) == (
        indexed.nbytes + 4 * sliding.nbytes
    )

    class Partly:
        _RECOVERY_APPEND_ONLY_FIELDS = (("keys", -2, "offset"),)

        def __init__(self):
            self.keys = mx.zeros((1, 100), dtype=mx.uint8)
            self.nbytes = 100 + 30  # 30 undeclared bytes

    assert _prefill_target_reservation([Partly()]) == 100 + 4 * 30

    class Inherited(Partly):
        """Only the exact type's own declaration counts (cow_cache law)."""

    assert _prefill_target_reservation([Inherited()]) == 4 * 130


def test_long_append_only_prefill_keeps_the_planned_slice(monkeypatch):
    """Scaled-down 128K Hy4 repro: the planned slice is admitted, not halved.

    At 130K tokens the old 4x charge on the Hy4 latent cache (~9.8 GB for a
    one-row slice) exceeded 7.8-9.3 GB of headroom, so the halving loop ran
    512/256-row slices and APCv2 refused every publication of the prompt.
    Headroom here sits between the 1x and 4x reservations.
    """
    import mlx2.runtime.external_speculative as ext

    cache = list(_hy4_caches(4096))
    resident = sum(c.nbytes for c in cache)
    lane = SimpleNamespace(cache=cache, draft_cache=[], remaining=[0] * 2049,
                           tail=mx.zeros((1, 0)), history=[0] * 4096)
    probe = _batch(1 << 40)
    assert ExternalDraftBatchGenerator._admit(probe, [lane], 1024, prefill=True)
    headroom = probe.scheduler_stats["reservation_bytes"] + resident
    assert _admit_planned(_generator(lambda: headroom), lane, 1024)
    monkeypatch.setattr(ext, "_prefill_target_reservation",
                        lambda caches: 4 * sum(c.nbytes for c in caches))
    assert not _admit_planned(_generator(lambda: headroom), lane, 1024)


def test_admission_halving_is_named_in_the_slice_schedule():
    from collections import defaultdict, deque

    from mlx2.runtime.external_speculative import Lane

    batch = object.__new__(_Generator)
    batch.prefill_step = 8
    batch.prefill_step_autoscale = False
    batch.scheduler_stats = defaultdict(int)
    batch.mx = mx
    batch.layers = ()
    batch.model = SimpleNamespace(
        prefill_body=lambda x, cache, layers: mx.zeros((1, x.shape[1], 2))
    )
    batch.decode_time_fairness = SimpleNamespace(enabled=False)
    batch.decode_first = SimpleNamespace(prefill_token_budget=None)
    batch._decode_first_round_mode = "off"
    batch.prefill_allocator_reclaim = False
    batch.external_varlen_prefill = False
    batch.prefill_order = SimpleNamespace(enabled=False)
    batch.capacity = 1
    batch._memory_waiting_uids = set()
    batch._prefill_response = lambda lane, **kw: SimpleNamespace(uid=lane.uid)
    batch._append_context = lambda lane, following: None
    batch._capture_state_boundary = lambda lane: None
    batch._admit = lambda lanes, append, prefill=False: append <= 4
    batch._reclaim_for_admission = lambda: False
    lane = Lane(uid=3, history=[], remaining=deque(range(30)), cache=[],
                draft_cache=[], tail=mx.zeros((1, 0, 2)), rng=None, maximum=4,
                processors=[], sampling={})
    batch.lanes = {3: lane}
    batch._prefill_phase([3])
    receipt = batch.prefill_slice_schedule(3)
    assert receipt["boundaries"] == [4]
    assert receipt["first_deviation"] == 0
    assert receipt["deviation_reason"] == "admission_halving"
    assert (receipt["deviation_planned_rows"], receipt["deviation_executed_rows"]) == (8, 4)
    assert batch.scheduler_stats["prefill_adaptive_slices"] == 1
