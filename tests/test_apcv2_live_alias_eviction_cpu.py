"""Pressure eviction keeps a running lane's unwritten COW prompt boundary.

Hy4-flash 128K DFly2 (2026-11-21, ladder hy4-ladder128-20261121-0d713c0d5-off,
Hy4-flash-MLX-4.6bit-mlx2@bdf83d7e...): the cold request's first decode
admission evicted its own just-published prompt boundary (eviction age < 1 s)
while that boundary still shared the live lane's buffers, so nothing was
freed.  The warm re-send then found only the finish tip, whose draft sidecar
is state at the tip's end, and was discarded as
``external_draft_sidecar_missing``.
"""

from types import SimpleNamespace
import threading

import mlx.core as mx

from mlx2.runtime.apc_v2 import APCKey, APCv2, MTPAPCSidecar
from mlx2.runtime.models.cache import KVCache
from mlx2.serving import live_boundary_aliases


def _state(length, seed=0):
    cache = KVCache()
    values = mx.arange(seed, seed + length, dtype=mx.float32).reshape(1, 1, length, 1)
    cache.update_and_fetch(values, values)
    mx.eval(cache.state)
    return cache


def _sidecar(covered):
    draft = _state(covered, seed=100)
    return MTPAPCSidecar(([draft], mx.zeros((1, 1, 2))), covered)


def test_live_alias_is_skipped_and_counted_and_copies_still_evict():
    apc = APCv2(max_size=4, layout_name="test-kv-v1")
    key = APCKey("alias")
    boundary = [1, 2, 3, 4]
    apc.store(key, boundary, [_state(4)])
    aliases = {(key, tuple(boundary))}
    # The boundary is the only candidate and still aliases its live lane.
    assert not apc.evict_oldest_unleased(live_aliases=aliases)
    assert apc.apc_stats["pressure_live_alias_skips"] == 1
    hit = apc.lookup(key, boundary + [5], allow_disk_restore=False)
    assert hit.cached_tokens == 4
    hit.cache.close()
    # Another entry is still evictable under pressure (the alias is passed
    # over again on the way, and counted again).
    apc.store(key, [9, 9, 9], [_state(3)])
    assert apc.evict_oldest_unleased(live_aliases=aliases)
    assert apc.lookup(key, [9, 9, 9, 1], allow_disk_restore=False).hit is False
    # Once the lane has written (no longer in the alias set) it is evictable.
    assert apc.evict_oldest_unleased()
    assert len(apc) == 0
    assert apc.lifetime_stats()["pressure_live_alias_skips"] == 2


def test_sidecar_boundary_survives_its_lane_so_the_resend_still_hits():
    """Boundary + finish tip of a draft route: the re-send needs the boundary."""
    apc = APCv2(max_size=4, layout_name="test-kv-v1")
    key = APCKey("resend")
    prompt = list(range(1, 9))
    apc.store(key, prompt, [_state(8)], sidecar=_sidecar(8))
    # Pressure during the first decode admission, boundary still aliased.
    assert not apc.evict_oldest_unleased(live_aliases={(key, tuple(prompt))})
    tip = prompt + [20, 21, 22]
    apc.store(key, tip, [_state(len(tip))], sidecar=_sidecar(len(tip)))
    hit = apc.lookup(key, prompt + [0], allow_disk_restore=False, require_sidecar=True)
    assert hit.cached_tokens == 8 and hit.sidecar is not None
    hit.cache.close()


def _external_lane(uid, history):
    from collections import deque

    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator, Lane

    gen = ExternalDraftBatchGenerator.__new__(ExternalDraftBatchGenerator)
    gen._prelaunched = {}
    gen._parked_inflight = None
    lane = Lane(uid=uid, history=list(history), remaining=deque([99]), cache=[],
                draft_cache=[], tail=mx.zeros((1, 0, 2)), rng=None, maximum=4,
                processors=[], sampling={})
    gen.lanes = {uid: lane}
    return gen, lane


def _job(uid, boundary, completion_tokens):
    return SimpleNamespace(uid=uid, apc_live_boundary=boundary,
                           completion_tokens=completion_tokens,
                           cancelled=threading.Event())


def test_real_publish_order_protects_until_the_lane_writes():
    """Serving publishes deferred prompt ends after counting the first token.

    That token is sampled from the prefill logits; its row is not written
    yet, so the boundary must still be protected although
    ``completion_tokens`` is already 1.  The lane's written position decides.
    """
    key = APCKey("order")
    prompt = list(range(1, 9))
    gen, lane = _external_lane(7, prompt)
    job = _job(7, (key, tuple(prompt)), completion_tokens=1)
    apc = APCv2(max_size=4, layout_name="test-kv-v1")
    apc.store(key, prompt, [_state(8)], sidecar=_sidecar(8))
    apc.set_live_alias_provider(lambda: live_boundary_aliases([job], gen))
    assert apc.unleased_resident_nbytes() == 0  # not reclaimable while aliased
    assert not apc.evict_oldest_unleased()
    assert apc.apc_stats["pressure_live_alias_skips"] == 1
    # A dispatched-but-unread or prelaunched round may have written: unknown.
    gen._prelaunched[7] = object()
    assert live_boundary_aliases([job], gen) == set()
    gen._prelaunched.clear()
    # The first decode round commits rows: the lane owns a copy now.
    lane.history.extend([99, 20])
    assert apc.unleased_resident_nbytes() > 0
    assert apc.evict_oldest_unleased()
    assert len(apc) == 0


def test_live_boundary_aliases_filters():
    key = APCKey("jobs")
    gen, _ = _external_lane(1, [1, 2])
    cancelled = _job(1, (key, (1, 2)), 0)
    cancelled.cancelled.set()
    assert live_boundary_aliases([_job(1, (key, (1, 2)), 3)], gen) == {(key, (1, 2))}
    assert live_boundary_aliases([cancelled], gen) == set()
    assert live_boundary_aliases([_job(2, (key, (1, 2)), 0)], gen) == set()  # no lane
    assert live_boundary_aliases([_job(1, None, 0)], gen) == set()
    # A generator that cannot report its written position protects nothing.
    assert live_boundary_aliases([_job(1, (key, (1, 2)), 0)], object()) == set()


def test_copied_entry_still_evicts():
    """Only a copy-on-write entry aliases its lane; a copied one frees memory."""
    apc = APCv2(max_size=4, layout_name="test-kv-v1")
    apc._cow_branching = False
    key = APCKey("copy")
    apc.store(key, [1, 2, 3], [_state(3)])
    assert apc.evict_oldest_unleased(live_aliases={(key, (1, 2, 3))})
    assert apc.apc_stats["pressure_live_alias_skips"] == 0


def test_control_evicting_the_aliased_boundary_leaves_a_sidecarless_landing():
    """The failure shape without the alias set: the re-send lands target-only."""
    apc = APCv2(max_size=4, layout_name="test-kv-v1")
    key = APCKey("control")
    prompt = list(range(1, 9))
    apc.store(key, prompt, [_state(8)], sidecar=_sidecar(8))
    assert apc.evict_oldest_unleased()
    tip = prompt + [20, 21, 22]
    apc.store(key, tip, [_state(len(tip))], sidecar=_sidecar(len(tip)))
    hit = apc.lookup(key, prompt + [0], allow_disk_restore=False, require_sidecar=True)
    assert hit.sidecar is None  # serving discards: external_draft_sidecar_missing
    if hit.cache is not None and hasattr(hit.cache, "close"):
        hit.cache.close()


def test_racing_provider_protects_nothing_and_logs_once_at_debug(caplog):
    import logging

    apc = APCv2(max_size=4, layout_name="test-kv-v1")
    key = APCKey("race")
    apc.store(key, [1, 2, 3], [_state(3)])

    def racing():
        raise RuntimeError("dictionary changed size during iteration")

    apc.set_live_alias_provider(racing)
    with caplog.at_level(logging.DEBUG, logger="mlx2.runtime.apc_v2"):
        assert apc.unleased_resident_nbytes() > 0
        assert apc.unleased_resident_nbytes() > 0
    failures = [r for r in caplog.records if "live-alias provider failed" in r.message]
    assert len(failures) == 1 and failures[0].levelno == logging.DEBUG
    assert failures[0].exc_info is None


def test_decode_admission_charges_the_first_write_copy_of_a_protected_boundary():
    """The lane's first write copies every buffer the protected entry shares.

    The decode round's 4x cache charge (external_speculative ``_admit``)
    already holds that one full copy; nothing extra is added for the alias.
    """
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.models import hy_v4_flash as H

    indexed = H.HY4FlashLayerCache(H.INDEXED, ratio=4)
    indexed.append(mx.zeros((1, 1024, 24)))
    H._buf_append(indexed, "tk", mx.zeros((1, 1024, 8)))
    H._buf_append(indexed, "gk", mx.zeros((1, 256, 8), dtype=mx.float32))
    copy_bytes = indexed.nbytes  # what the first write copies
    target = SimpleNamespace(num_key_value_heads=0, head_dim=0, num_hidden_layers=0,
                             hidden_size=0, intermediate_size=0, vocab_size=0)
    draft = SimpleNamespace(num_hidden_layers=0, intermediate_size=0, hidden_size=0,
                            num_key_value_heads=0, head_dim=0, feature_width=0)
    batch = SimpleNamespace(
        memory_headroom=lambda: 1 << 40, _target_args=target,
        draft=SimpleNamespace(config=draft), layers=(), scheduler_stats={},
        continuation_policy=None, continuation_pool_ast_bridge=False,
    )
    lane = SimpleNamespace(cache=[indexed], draft_cache=[], remaining=[0],
                           tail=mx.zeros((1, 0)))
    assert ExternalDraftBatchGenerator._admit(batch, [lane], 1)
    assert batch.scheduler_stats["reservation_bytes"] == 4 * copy_bytes >= copy_bytes


def _reclaiming_generator(apc, lanes):
    """The external executor's admission reclaim, bound to a real APCv2."""
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    gen = ExternalDraftBatchGenerator.__new__(ExternalDraftBatchGenerator)
    gen._prelaunched = {}
    gen._parked_inflight = None
    gen.lanes = lanes
    gen.scheduler_stats = {}
    gen.reclaim_memory = None
    gen.settle_memory = None
    gen._settle_once = None
    gen._settle_pending = False
    gen._allocator_reclaimed = True
    gen._reclaims_left = 2
    gen.evict_checkpoint = apc.evict_oldest_unleased  # serving's callback evicts here
    return gen


def test_executor_decode_reclaim_path_follows_the_written_position():
    """``_fit_cohort -> _reclaim_for_admission -> evict_checkpoint``.

    This is the path behind ``scheduler.memory_pressure_evictions``.  While
    the lane has not written past its boundary the reclaim finds nothing to
    evict; once it has (with decode-first off, the default, one poll runs
    the prefill phase that ends the prompt and then the lane's first decode
    round, before serving publishes the boundary), the boundary is a copy
    and is evicted, with its cause and caller in ``recent_drops``.
    """
    key = APCKey("reclaim")
    prompt = list(range(1, 9))
    gen, lane = _external_lane(5, prompt)
    job = _job(5, (key, tuple(prompt)), completion_tokens=1)
    apc = APCv2(max_size=4, layout_name="test-kv-v1")
    apc.store(key, prompt, [_state(8)], sidecar=_sidecar(8))
    apc.set_live_alias_provider(lambda: live_boundary_aliases([job], gen))
    executor = _reclaiming_generator(apc, gen.lanes)
    assert not executor._reclaim_for_admission()
    assert apc.apc_stats["pressure_live_alias_skips"] == 1
    assert "memory_pressure_evictions" not in executor.scheduler_stats
    lane.history.extend([99, 20])  # the first decode round wrote
    executor._reclaims_left = 2
    assert executor._reclaim_for_admission()
    assert executor.scheduler_stats["memory_pressure_evictions"] == 1
    drop = apc.apc_stats["recent_drops"][-1]
    assert drop["reason"].startswith("pressure") and drop["sidecar"] and drop["cow"]
    assert drop["tokens"] == 8
    assert any("_reclaim_for_admission" in caller for caller in drop["callers"])


def test_default_round_writes_the_lane_before_serving_sees_its_prompt_end():
    """Decode-first off: prefill phase, then decode phase, in one poll."""
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    gen = ExternalDraftBatchGenerator.__new__(ExternalDraftBatchGenerator)
    order = []
    gen.lanes = {}
    gen._memory_waiting_uids = set()
    gen._decode_fit_waiting_uids = set()
    gen._atomic_prefill_waiting_uids = set()
    gen._schedule_cursor = 0
    gen.round_timing = False
    gen._refuse_incomplete_atomic_cohorts = lambda: None
    gen._prefill_phase = lambda candidates: order.append("prefill") or ["prompt_end"]
    gen._decode_phase = lambda ordered: order.append("decode")
    gen._drain_ready = lambda prompts: order.append("drain") or []
    gen._reclaim_after_emission = lambda count: None
    round_ = gen._scheduler_round("off")
    try:
        next(round_)
    except StopIteration as stop:
        prompts, _ = stop.value
    assert order == ["prefill", "decode", "drain"] and prompts == ["prompt_end"]
