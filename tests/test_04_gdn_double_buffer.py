"""External-round snapshots without deepcopy (descriptor COW), CPU only."""
import copy
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)
from test_external_dflash2_cpu import generator, tiny

import mlx2.runtime.external_speculative as module
from mlx2.runtime.cow_cache import (
    external_round_cow_enabled,
    snapshot_committed_cache,
)
from mlx2.runtime.external_speculative import HostDraftRow, RoundDecision


def _cow(monkeypatch, enabled):
    monkeypatch.setenv("MLX_LM_EXTERNAL_ROUND_COW", "1" if enabled else "0")


def _planes(lane):
    """Host copy of every array a lane's caches and tail hold."""
    return (
        [[np.asarray(a) for a in c.state] for c in lane.cache],
        [[np.asarray(a) for a in c.state] for c in lane.draft_cache if c.offset],
        [int(c.offset) for c in lane.cache],
        [int(c.offset) for c in lane.draft_cache],
        np.asarray(lane.tail),
    )


def _assert_planes_equal(now, old):
    for a, b in zip(now[0] + now[1], old[0] + old[1]):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
    assert now[2:4] == old[2:4]
    np.testing.assert_array_equal(now[4], old[4])


def _drain(b, limit=200):
    out = {}
    finals = {}
    for _ in range(limit):
        if not b.lanes:
            break
        _, responses = b.next()
        for r in responses:
            out.setdefault(r.uid, []).append(r.token)
            if r.finish_reason:
                finals[r.uid] = r
    return out, finals


def test_knob_defaults_off_and_opts_in(monkeypatch):
    monkeypatch.delenv("MLX_LM_EXTERNAL_ROUND_COW", raising=False)
    assert not external_round_cow_enabled()
    monkeypatch.setenv("MLX_LM_EXTERNAL_ROUND_COW", "1")
    assert external_round_cow_enabled()


def test_default_mode_adds_no_counters(monkeypatch):
    monkeypatch.delenv("MLX_LM_EXTERNAL_ROUND_COW", raising=False)
    m, d = tiny()
    b = generator(m, d)
    b.insert([[1, 2, 3]], max_tokens=[4])
    _drain(b)
    assert not any("cow" in key for key in b.scheduler_stats)


@pytest.mark.parametrize("cow", [True, False])
def test_failure_after_commit_restores_lane_and_caches_bit_equal(monkeypatch, cow):
    _cow(monkeypatch, cow)
    m, d = tiny()
    b = generator(m, d)
    b.insert([[1, 2, 3, 4], [5, 6, 7]], max_tokens=[6, 6],
             sampling_configs=[{"sampling_temp": 0.8}, {}])
    cohort = list(b.lanes.values())
    for lane in cohort:
        while lane.anchor is None:
            b._prefill(lane)
    # Advance one real round so the draft plane and tail are non-empty.
    b._round(cohort)
    for lane in cohort:
        lane.ready.clear()
    before = [_planes(lane) for lane in cohort]
    host = [
        (list(l.history), l.anchor, l.generated, l.rng.snapshot(), l.proposed,
         l.accepted, l.external_rounds)
        for l in cohort
    ]
    restores = b.scheduler_stats["recovery_checkpoint_restores"]
    real_commit = b._commit

    def fail_after_commit(*args, **kwargs):
        real_commit(*args, **kwargs)
        # Commit really advanced the live caches before the failure.
        assert all(len(l.history) > len(h[0]) for l, h in zip(cohort, host))
        raise RuntimeError("injected post-commit failure")

    monkeypatch.setattr(b, "_commit", fail_after_commit)
    with pytest.raises(RuntimeError, match="injected post-commit"):
        b._round(cohort)
    assert not b._open
    assert b.scheduler_stats["recovery_checkpoint_restores"] == restores + 2
    assert b.scheduler_stats.get("external_cow_snapshots", 0) > 0 if cow else (
        "external_cow_snapshots" not in b.scheduler_stats
    )
    for lane, old, h in zip(cohort, before, host):
        assert (list(lane.history), lane.anchor, lane.generated, lane.rng.snapshot(),
                lane.proposed, lane.accepted, lane.external_rounds) == h
        assert not lane.ready
        _assert_planes_equal(_planes(lane), old)
    # The restored lanes continue normally.
    monkeypatch.setattr(b, "_commit", real_commit)
    _out, finals = _drain(b)
    assert set(finals) == {lane.uid for lane in cohort}


def test_restore_leaves_checkpoint_pristine_for_a_second_restore(monkeypatch):
    _cow(monkeypatch, True)
    m, d = tiny()
    b = generator(m, d)
    b.insert([[1, 2, 3]], max_tokens=[5])
    lane = b.lanes[0]
    while lane.anchor is None:
        b._prefill(lane)
    b._round([lane])
    lane.ready.clear()
    before = _planes(lane)
    (snapshot,) = b._snapshot_round([lane])
    assert snapshot.mode == "descriptor_cow"
    b._round([lane])  # mutate live state past the checkpoint
    b._restore_round([lane], [snapshot])
    _assert_planes_equal(_planes(lane), before)
    # Mutating the restored lane must not reach back into the checkpoint.
    lane.ready.clear()
    b._round([lane])
    b._restore_round([lane], [snapshot])
    _assert_planes_equal(_planes(lane), before)


@pytest.mark.parametrize("sampled", [False, True])
def test_cow_and_deepcopy_modes_are_token_rng_and_cache_identical(monkeypatch, sampled):
    def run(cow):
        _cow(monkeypatch, cow)
        m, d = tiny()
        b = generator(m, d)
        config = {"sampling_temp": 0.9, "top_p": 0.95} if sampled else {}
        b.insert([[1, 2, 3, 4, 5], [6, 7]], max_tokens=[9, 7],
                 sampling_configs=[config, config])
        boundaries = []
        out, finals = {}, {}
        for _ in range(200):
            if not b.lanes:
                break
            prompts, responses = b.next()
            for p in prompts:
                if p.end_of_prompt:
                    boundary = b.pop_prompt_boundary(p.uid)
                    boundaries.append(
                        [np.asarray(a) for c in boundary["target_cache"] for a in c.state]
                    )
            for r in responses:
                out.setdefault(r.uid, []).append(
                    (r.token, r.speculative_receipt["round_accepted"],
                     r.speculative_receipt["round_proposed"])
                )
                if r.finish_reason:
                    finals[r.uid] = (
                        [np.asarray(a) for c in r.prompt_cache for a in c.state],
                        r.cache_sidecar.rng_draws,
                        r.all_tokens,
                    )
        stats = {k: v for k, v in b.scheduler_stats.items() if "cow" not in k}
        return out, finals, boundaries, stats, b.scheduler_stats

    cow_out, cow_finals, cow_bounds, cow_stats, raw = run(True)
    ref_out, ref_finals, ref_bounds, ref_stats, _ = run(False)
    assert cow_out == ref_out and cow_stats == ref_stats
    assert raw["external_cow_snapshots"] > 0 and "external_cow_fallbacks" not in raw
    for uid in ref_finals:
        assert cow_finals[uid][1:] == ref_finals[uid][1:]
        for a, b in zip(cow_finals[uid][0], ref_finals[uid][0]):
            np.testing.assert_array_equal(a, b)
    for x, y in zip(cow_bounds, ref_bounds):
        for a, b in zip(x, y):
            np.testing.assert_array_equal(a, b)


def test_neither_snapshot_mode_duplicates_cache_bytes(monkeypatch):
    """mx.array deep copies share the immutable buffer; so does descriptor COW."""
    m, d = tiny()
    b = generator(m, d)
    b.insert([list(range(1, 30))], max_tokens=[4])
    lane = b.lanes[0]
    while lane.anchor is None:
        b._prefill(lane)
    mx.eval([c.state for c in lane.cache], lane.tail)
    cache_bytes = sum(int(c.nbytes) for c in lane.cache)
    assert cache_bytes > 0

    def allocated(enabled):
        _cow(monkeypatch, enabled)
        mx.clear_cache()
        start = mx.get_active_memory()
        snapshots = b._snapshot_round([lane])
        # Materialize whatever the snapshot holds.
        frozen = snapshots[0].slot._checkpoint._state
        mx.eval([a for a in _arrays(frozen)])
        return mx.get_active_memory() - start, snapshots

    cow_bytes, keep_cow = allocated(True)
    copy_bytes, keep_copy = allocated(False)
    assert keep_cow[0].mode == "descriptor_cow" and keep_copy[0].mode == "deepcopy"
    assert cow_bytes < cache_bytes // 8
    assert copy_bytes < cache_bytes // 8


def _arrays(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _arrays(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _arrays(item, seen)
    elif hasattr(value, "__dict__") and not callable(value):
        for item in vars(value).values():
            yield from _arrays(item, seen)


def test_unfreezable_graph_falls_back_to_deepcopy_and_still_restores(monkeypatch):
    _cow(monkeypatch, True)
    from mlx2.runtime.cow_cache import COWCacheUnsupported

    def refuse(*args, **kwargs):
        raise COWCacheUnsupported("injected live transaction")

    m, d = tiny()
    b = generator(m, d)
    b.insert([[1, 2, 3]], max_tokens=[5])
    lane = b.lanes[0]
    while lane.anchor is None:
        b._prefill(lane)
    before = _planes(lane)
    history = list(lane.history)
    monkeypatch.setattr(module, "snapshot_prompt_cache_descriptors", refuse)
    (snapshot,) = b._snapshot_round([lane])
    assert snapshot.mode == "deepcopy"
    assert b.scheduler_stats["external_cow_fallbacks"] == 1
    b._round([lane])
    b._restore_round([lane], [snapshot])
    assert lane.history == history
    _assert_planes_equal(_planes(lane), before)


def test_live_speculation_transaction_is_never_frozen_by_descriptor():
    from mlx2.runtime.models.cache import KVCache

    cache = KVCache()
    cache.update_and_fetch(mx.ones((1, 1, 3, 4)), mx.ones((1, 1, 3, 4)))
    frozen, _sidecar, mode = snapshot_committed_cache([cache], enabled=True)
    assert mode == "descriptor_cow"
    cache.speculating = True
    frozen, _sidecar, mode = snapshot_committed_cache([cache], enabled=True)
    assert mode == "deepcopy_fallback"
    np.testing.assert_array_equal(np.asarray(frozen[0].keys), np.asarray(cache.keys))
    _frozen, _sidecar, mode = snapshot_committed_cache([cache], enabled=False)
    assert mode == "deepcopy_disabled"


def test_propose_verify_phase_seam_accepts_one_row_compact_blocks(monkeypatch):
    """A P3-shaped one-row block (mx tokens + dense_laws) verifies like the host row."""
    _cow(monkeypatch, True)

    def run(compact):
        m, d = tiny()
        b = generator(m, d)
        b.insert([[1, 2, 3, 4]], max_tokens=[6])
        lane = b.lanes[0]
        while lane.anchor is None:
            b._prefill(lane)
        real = b._propose

        def propose(cohort):
            blocks = real(cohort)
            assert all(isinstance(x, HostDraftRow) for x in blocks)
            if not compact:
                return blocks
            out = []
            for block in blocks:
                laws = list(block.laws)
                out.append(SimpleNamespace(
                    tokens=mx.array([block.tokens + [0]], dtype=mx.int32),
                    lengths=(len(block.tokens),),
                    width=block.width,
                    dense_laws=lambda vocab, laws=laws: [laws],
                ))
            return out

        monkeypatch.setattr(b, "_propose", propose)
        decisions = []
        real_verify = b._verify

        def verify(cohort, blocks, logits):
            result = real_verify(cohort, blocks, logits)
            assert all(isinstance(x, RoundDecision) for x in result)
            decisions.extend(result)
            return result

        monkeypatch.setattr(b, "_verify", verify)
        b._round([lane])
        return [(x.accepted, x.emitted) for x in decisions], list(lane.history), \
            [(r.token, r.speculative_receipt["round_proposed"]) for r in lane.ready]

    assert run(False) == run(True)


def test_zero_count_round_has_no_blocks_and_keeps_draft_paired(monkeypatch):
    _cow(monkeypatch, True)
    m, d = tiny()
    b = generator(m, d)
    b.insert([[1, 2, 3]], max_tokens=[1])
    lane = b.lanes[0]
    while lane.anchor is None:
        b._prefill(lane)
    assert b._propose([lane]) == [None]
    assert all(int(c.offset) == len(lane.history) for c in lane.draft_cache if c.offset)


def test_prompt_lookup_boundaries_and_finishes_use_descriptor_cow(monkeypatch):
    import test_pld_batched_verify as pld

    model = pld._north()

    def run(cow):
        _cow(monkeypatch, cow)
        tokens, finals, stats = pld._run(model, batched=True, tokens=10)
        caches = [[np.asarray(a) for c in f.prompt_cache for a in c.state] for f in finals]
        return tokens, caches, stats

    cow_tokens, cow_caches, cow_stats = run(True)
    ref_tokens, ref_caches, ref_stats = run(False)
    assert cow_tokens == ref_tokens
    for x, y in zip(cow_caches, ref_caches):
        for a, b in zip(x, y):
            np.testing.assert_array_equal(a, b)
    # One prompt boundary and one finish per request.
    assert cow_stats["pld_cow_snapshots"] == 2 * len(pld.PROMPTS)
    assert "pld_cow_fallbacks" not in cow_stats
    assert "pld_cow_snapshots" not in ref_stats


def test_prometheus_maps_snapshot_counters():
    from mlx2 import prometheus

    for key in ("external_cow_snapshots", "external_cow_fallbacks",
                "pld_cow_snapshots", "pld_cow_fallbacks"):
        assert key in prometheus._SCHEDULER_EVENTS
    assert prometheus._scheduler_mechanism("external_cow_snapshots") == "external_speculative"
    assert prometheus._scheduler_mechanism("pld_cow_fallbacks") == "prompt_lookup"


def test_deepcopy_mode_round_snapshot_matches_historical_semantics(monkeypatch):
    _cow(monkeypatch, False)
    m, d = tiny()
    b = generator(m, d)
    b.insert([[1, 2]], max_tokens=[3])
    lane = b.lanes[0]
    while lane.anchor is None:
        b._prefill(lane)
    (snapshot,) = b._snapshot_round([lane])
    state = snapshot.slot._checkpoint._state
    assert set(state) == set(vars(lane))
    assert state["cache"] is not lane.cache
    assert copy.deepcopy is snapshot.slot._checkpoint._restore
