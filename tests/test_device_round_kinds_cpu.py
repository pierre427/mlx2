"""device_round.fixed_shape: one round kind for every pipelined round (slice 1a-5), CPU only.

Design: docs/research/2026-11-25-gpu-resident-spec-rounds-design.md (1a-5 row). The 2026-11-27
GPU check (P6b, 11 layouts in 16 Raon rounds) left no width-dependent kernel; the remaining
layouts were round kinds: the last rounds of a request, which had no round in flight ahead of
them (host-list presence path, a data-dependent scatter), and its budget-limited rounds, which
the pipeline left to the serial round (another decision tail). Pins, on the CPU device:
  * ``DeviceHistory.first`` (no round ahead) is the host presence processor, bitwise;
  * under ``fixed_shape`` every pipelined round after the first builds the same graph (the same
    primitive sequence; an ``async_eval``'d array counts as scheduled, as the GPU encodes it),
    whether or not a round is in flight ahead of it and whatever its budget;
  * budget-limited rounds stay pipelined (no ``short_depth`` skips) and count their padding.
Bitwise tokens, caches and receipts against the serial round are pinned by
tests/test_device_round_fixed_shape_commit_cpu.py (budget ends, presence windows).
"""

import io
import re

import mlx.core as mx
import numpy as np
import pytest
from test_device_commit_raon_cpu import NUM_DRAFT, PROMPT
from test_device_pipeline_raon_cpu import PIPE, _decode_presence

from mlx2.runtime import speculative_device_pipeline as pipe
from mlx2.runtime.speculative_device_pipeline import SKIPS
from mlx2.runtime.speculative_device_presence import DeviceHistory

PIPE_PROC_FIX = {**PIPE, "processors": True, "fixed_shape": True}


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("seed", range(20))
def test_first_history_is_the_host_presence_processor(seed):
    from mlx2.runtime.sample_utils import make_presence_penalty

    rng = np.random.default_rng(seed)
    vocab, menu = 40, NUM_DRAFT
    host = [int(t) for t in rng.integers(0, 10, size=int(rng.integers(0, 25)))]
    anchor = int(rng.integers(0, 10))
    context = int(rng.choice([0, 1, 3, 20]))
    start = rng.choice([None, 0, len(host) // 2, len(host) + 3])
    processor = make_presence_penalty(1.5, context, generation_start=start)
    history = DeviceHistory.first(mx, host, anchor, menu)
    drafts = mx.array(rng.integers(0, 10, size=menu).astype(np.int32))
    values = mx.array(rng.normal(size=(menu + 1, vocab)).astype(np.float32))
    count = int(rng.integers(0, menu + 1))
    rows = history.apply_rows([processor], drafts, count, values)
    for j in range(menu + 1):
        d = min(j, count)
        seq = host + [anchor] + np.array(drafts).tolist()[:d]
        want = processor(mx.array(seq, mx.int32), values[j][None])
        got = history.apply([processor], drafts[:d] if d else None, values[j][None])
        assert mx.array_equal(want, got).item(), (seed, j)
        assert mx.array_equal(want[0], rows[j]).item(), (seed, j)


def _primitives(arrays):
    """The primitive names of the graph, in export order. Shape-blind: two rounds that differ
    only in an array's size compare equal (the presence window with ``context`` 0 still grows
    every round; that needs device-resident presence state, not covered here)."""
    out = io.StringIO()
    mx.export_to_dot(out, *arrays)
    return tuple(re.findall(r'label ="([^"]+)"', out.getvalue()))


def _round_graphs(monkeypatch, steps):
    """The primitive sequence each pipelined dispatch schedules, in dispatch order."""
    graphs, inside = [], []
    real_async, real_dispatch = mx.async_eval, pipe.dispatch

    def async_eval(*arrays):
        flat = [a for a in arrays if isinstance(a, mx.array)]
        if inside and flat:
            inside[-1].append(_primitives(flat))  # every scheduling call of the dispatch
        real_async(*arrays)
        mx.eval(*flat)  # scheduled work leaves later graphs, as on the GPU

    def dispatch(*args, **kwargs):
        inside.append([])
        try:
            return real_dispatch(*args, **kwargs)
        finally:
            graphs.append((tuple(inside.pop()) or None, kwargs.get("previous") is not None))

    import test_device_commit_raon_cpu as route

    real_route = route.raon_route

    def evaluated_route(*args, **kwargs):
        # The tiny models initialise their weights lazily: evaluate them first, so
        # the first round's graph holds the round, not the random initialisers.
        target, draft = real_route(*args, **kwargs)
        mx.eval(target.parameters(), draft.parameters())
        return target, draft

    with pytest.MonkeyPatch.context() as patch:  # undoes only its own patches
        patch.setattr(route, "raon_route", evaluated_route)
        patch.setattr(mx, "async_eval", async_eval)
        patch.setattr(pipe, "dispatch", dispatch)
        result = _decode_presence(PIPE_PROC_FIX, patch, steps=steps, oracle=False)
    return graphs, result[3]


@pytest.mark.parametrize("steps", [40, 23])
def test_every_pipelined_round_builds_one_graph(monkeypatch, steps):
    graphs, stats = _round_graphs(monkeypatch, steps)
    assert all(graph is not None for graph, _ in graphs)
    # The first round of the request reads the entry (KV capacity growth is open: in-place state).
    later = graphs[1:]
    assert any(following for _, following in later), "no round dispatched behind another"
    assert any(not following for _, following in later), "no round with nothing ahead of it"
    assert len({graph for graph, _ in later}) == 1
    assert stats[SKIPS].get("short_depth", 0) == 0
    assert stats["external_device_round_fixed_shape_padded_drafts"] > 0
    assert stats["external_device_round_fixed_shape_rounds"] == stats["external_device_round_rounds"]


def test_without_fixed_shape_budget_ends_stay_serial(monkeypatch):
    stats = _decode_presence({**PIPE, "processors": True}, monkeypatch, steps=23, oracle=False)[3]
    assert stats[SKIPS].get("short_depth", 0) > 0


def test_padded_pipelined_drafts_are_counted_on_the_lane(monkeypatch):
    got, _state, final, stats = _decode_presence(PIPE_PROC_FIX, monkeypatch, steps=23,
                                                 oracle=False)
    lane = final.speculative_receipt["device_round"]["lane"]
    assert stats["external_device_round_fixed_shape_padded_drafts"] > 0
    assert lane["fixed_shape_padded_drafts"] == stats["external_device_round_fixed_shape_padded_drafts"]
    assert lane["pipeline_rounds"] == lane["rounds"] == lane["fixed_shape_rounds"]
    assert len(got) == 23


@pytest.mark.parametrize("steps", [13, 23])
@pytest.mark.parametrize("context,start", [(20, None), (0, len(PROMPT)), (3, len(PROMPT))])
def test_pipelined_presence_budget_ends_equal_the_serial_round(monkeypatch, steps, context, start):
    """Budget-limited presence rounds, now pipelined, against the serial fixed-shape round."""
    from test_device_pipeline_raon_cpu import DC4, _assert_like_dc4

    serial = _decode_presence({**DC4, "processors": True, "fixed_shape": True}, monkeypatch,
                              context=context, start=start, steps=steps, oracle=False)
    piped = _decode_presence(PIPE_PROC_FIX, monkeypatch, context=context, start=start,
                             steps=steps, oracle=False)
    stats = _assert_like_dc4(serial, piped)
    assert stats[SKIPS].get("short_depth", 0) == 0
    assert stats["external_device_round_fixed_shape_padded_drafts"] > 0


def test_negative_presence_window_keeps_the_host_history():
    """A window ``presence_contracts`` refuses never gets a device history (it would raise)."""
    from mlx2.runtime.sample_utils import make_logits_processors
    from mlx2.runtime.speculative_device_presence import presence_contracts

    processors = make_logits_processors(presence_penalty=1.5, presence_context_size=-2,
                                        penalty_generation_start=0)
    assert presence_contracts(processors) is None


# -- fixed-shape presence state (``DeviceHistory(store=...)``) -----------------------------------

def _host_rounds(rng, rounds, menu):
    """A growing history, as booked rounds extend it (1..menu+1 tokens each)."""
    history = [int(t) for t in rng.integers(0, 10, size=int(rng.integers(0, 12)))]
    for _ in range(rounds):
        history = history + [int(t) for t in rng.integers(0, 10, size=int(rng.integers(1, menu + 2)))]
        yield list(history)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("context,start", [(0, None), (0, 6), (0, 40), (3, None), (20, 5), (1, 0)])
def test_stored_presence_equals_the_host_processor_every_round(seed, context, start):
    from mlx2.runtime.sample_utils import make_presence_penalty

    rng = np.random.default_rng(seed)
    vocab, menu = 40, NUM_DRAFT
    processor = make_presence_penalty(1.5, context, generation_start=start)
    store, shapes = {}, set()
    for host in _host_rounds(rng, 12, menu):
        if rng.random() < 0.15:
            host = host[: max(0, len(host) - 7)]  # a restore: the covered prefix is gone
        anchor = int(rng.integers(0, 10))
        prev = mx.array(rng.integers(0, 10, size=menu).astype(np.int32))
        kept = int(rng.integers(-1, menu + 1))
        last = mx.array([anchor if kept < 0 else int(rng.integers(0, 10))], mx.int32)
        stored = DeviceHistory(mx, host, anchor, prev, mx.array([kept], mx.int32), last, store=store)
        plain = DeviceHistory(mx, host, anchor, prev, mx.array([kept], mx.int32), last)
        drafts = mx.array(rng.integers(0, 10, size=menu).astype(np.int32))
        values = mx.array(rng.normal(size=(menu + 1, vocab)).astype(np.float32))
        count = int(rng.integers(0, menu + 1))
        got = stored.apply_rows([processor], drafts, count, values)
        want = plain.apply_rows([processor], drafts, count, values)
        assert mx.array_equal(got, want).item()
        for j in range(menu + 1):
            d = min(j, count)
            tail = drafts[:d] if d else None
            assert mx.array_equal(stored.apply([processor], tail, values[j][None]),
                                  plain.apply([processor], tail, values[j][None])).item()
        tokens, positions, valid, seen = stored._host_part(processor.presence_window, vocab)
        shapes.add((tokens.shape, positions.shape, valid.shape, None if seen is None else seen.shape))
    assert len(shapes) == 1, shapes  # one host-part shape whatever the history length


def test_stored_presence_adds_one_fixed_scatter_per_round():
    """A round that books at most menu + 1 tokens advances the seen map by one padded chunk:
    the same update graph (one scatter) every round, after a one-time build."""
    rng = np.random.default_rng(3)
    menu, vocab = NUM_DRAFT, 40
    store, graphs = {}, []
    for host in _host_rounds(rng, 10, menu):
        seen = DeviceHistory.first(mx, host, 1, menu, store=store)._seen(4, vocab)
        graphs.append(_primitives([seen]))
        mx.eval(seen, store[("seen", 4, vocab)][1])  # booked: the next round starts from data
    assert len(set(graphs[1:])) == 1
    assert graphs[1].count("Scatter Sum") == 1


def test_pipelined_presence_host_parts_keep_one_shape(monkeypatch):
    """Served shape (``context`` 0 from the prompt end) and a window: the host part every
    round builds has one shape under fixed_shape; without it, the whole-history part grows."""
    from mlx2.runtime import speculative_device_presence as presence

    def run(policy, context, start):
        shapes = []
        real = presence.DeviceHistory._host_part

        def host_part(self, contract, vocab):
            out = real(self, contract, vocab)
            shapes.append(tuple(None if a is None else tuple(a.shape) for a in out))
            return out

        monkeypatch.setattr(presence.DeviceHistory, "_host_part", host_part)
        try:
            stats = _decode_presence(policy, monkeypatch, context=context, start=start, steps=40,
                                     oracle=False)[3]
        finally:
            monkeypatch.setattr(presence.DeviceHistory, "_host_part", real)
        return shapes, stats

    for context, start in ((0, len(PROMPT)), (20, None)):
        shapes, stats = run(PIPE_PROC_FIX, context, start)
        assert len(shapes) > 10 and len(set(shapes)) == 1, (context, set(shapes))
        assert stats["external_device_round_fixed_shape_rounds"] == stats["external_device_round_rounds"]
    shapes, _ = run({**PIPE, "processors": True}, 0, len(PROMPT))
    assert len(set(shapes)) > 1  # the unstored whole-history part grows every round


@pytest.mark.parametrize("seed", range(16))
def test_stored_presence_equals_the_host_processor_on_the_sequence(seed):
    """Against ``make_presence_penalty`` itself on ``H + [anchor] + kept drafts + [last] + tail``
    (``first``: ``H + [anchor] + tail``), with two processors, restores that re-extend the
    history with different tokens, and one store serving two vocabularies."""
    from mlx2.runtime.sample_utils import make_presence_penalty

    rng = np.random.default_rng(100 + seed)
    menu = NUM_DRAFT
    processors = [make_presence_penalty(1.5, 0, generation_start=int(rng.integers(0, 8))),
                  make_presence_penalty(0.5, int(rng.choice([2, 7])), generation_start=None)]
    store = {}
    host = [int(t) for t in rng.integers(0, 10, size=6)]
    for _ in range(14):
        if rng.random() < 0.25:
            cut = int(rng.integers(0, len(host) + 1))  # restore, then re-extend differently
            host = host[:cut] + [int(t) for t in rng.integers(0, 10, size=int(rng.integers(1, 9)))]
        else:
            host = host + [int(t) for t in rng.integers(0, 10, size=int(rng.integers(1, menu + 2)))]
        anchor = int(rng.integers(0, 10))
        first = rng.random() < 0.3
        if first:
            history = DeviceHistory.first(mx, host, anchor, menu, store=store)
            committed = []
        else:
            prev = rng.integers(0, 10, size=menu).astype(np.int32)
            kept, last = int(rng.integers(0, menu + 1)), int(rng.integers(0, 10))
            history = DeviceHistory(mx, host, anchor, mx.array(prev), mx.array([kept], mx.int32),
                                    mx.array([last], mx.int32), store=store)
            committed = prev.tolist()[:kept] + [last]
        for vocab in (40, 32):
            tail = rng.integers(0, 10, size=int(rng.integers(0, menu))).astype(np.int32)
            logits = rng.normal(size=(1, vocab)).astype(np.float32)
            seq = mx.array(host + [anchor] + committed + tail.tolist(), mx.int32)
            want = mx.array(logits)  # the host processor subtracts in place: its own copy
            for processor in processors:
                want = processor(seq, want)
            got = history.apply(processors, mx.array(tail) if len(tail) else None, mx.array(logits))
            assert mx.array_equal(want, got).item(), (seed, vocab)


def test_stored_presence_clips_ids_beyond_the_vocabulary_like_the_exact_window():
    from mlx2.runtime.sample_utils import make_presence_penalty

    processor = make_presence_penalty(1.5, 0, generation_start=0)
    host, vocab, menu = [3, 45, 7, 60, 2], 40, NUM_DRAFT
    zeros = mx.zeros((menu,), mx.int32)
    args = (mx, host, 1, zeros, mx.array([0], mx.int32), mx.array([5], mx.int32))
    logits = mx.zeros((1, vocab), mx.float32)
    stored = DeviceHistory(*args, store={}).apply([processor], None, logits)
    plain = DeviceHistory(*args).apply([processor], None, logits)
    assert mx.array_equal(stored, plain).item()
    assert stored[0, vocab - 1].item() == -1.5


def test_served_presence_takes_the_incremental_path(monkeypatch):
    """The served shape (``context`` 0 from the prompt end): one rebuild per vocabulary key at
    entry, then one single-chunk update per round; never a gap wider than one round."""
    stats = _decode_presence(PIPE_PROC_FIX, monkeypatch, context=0, start=len(PROMPT), steps=40,
                             oracle=False)[3]
    assert 1 <= stats["external_device_round_presence_seen_rebuilds"] <= 2
    assert stats["external_device_round_presence_seen_multi_chunk"] == 0
    assert stats["external_device_round_pipeline_rounds"] > 10


# -- in-place K/V: the entry reserve (``DeviceOffsetKVCache.reserve``) ---------------------------

@pytest.fixture
def small_kv_step(monkeypatch):
    """Grow device-mode K/V every 8 rows, so a 40-token request would grow it often."""
    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.models.device_offset_kv import DeviceOffsetKVCache

    monkeypatch.setattr(KVCache, "step", 8)  # the prefilled host cache too
    monkeypatch.setattr(DeviceOffsetKVCache, "step", 8)


def test_entry_reserve_leaves_no_device_mode_growth(monkeypatch, small_kv_step):
    from mlx2.runtime import speculative_device_commit as commit
    from test_device_pipeline_raon_cpu import _assert_like_dc4

    reserved = _decode_presence(PIPE_PROC_FIX, monkeypatch, steps=40, oracle=False)
    monkeypatch.setattr(commit, "KV_RESERVE_CAP", 0)
    grown = _decode_presence(PIPE_PROC_FIX, monkeypatch, steps=40, oracle=False)
    _assert_like_dc4(grown, reserved)  # capacity changes allocation only: bitwise
    assert reserved[3].get("external_device_round_kv_capacity_growths", 0) == 0
    assert grown[3]["external_device_round_kv_capacity_growths"] > 0


def test_reserved_rounds_build_one_graph_where_growth_splits_them(monkeypatch, small_kv_step):
    from mlx2.runtime import speculative_device_commit as commit

    graphs, _ = _round_graphs(monkeypatch, 40)
    assert len({graph for graph, _ in graphs[1:]}) == 1
    monkeypatch.setattr(commit, "KV_RESERVE_CAP", 0)
    graphs, _ = _round_graphs(monkeypatch, 40)
    assert len({graph for graph, _ in graphs[1:]}) > 1, "growth should be a separate round kind"


def test_reserve_grows_once_to_the_step(monkeypatch):
    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.models.device_offset_kv import DeviceOffsetKVCache

    host = KVCache()
    keys = mx.random.normal((1, 2, 5, 4))
    host.update_and_fetch(keys, mx.random.normal((1, 2, 5, 3)))
    cache = DeviceOffsetKVCache.adopt(host)
    have = int(cache.keys.shape[2])
    assert not cache.reserve_rows(0, 64) and not cache.reserve_rows(have - 5, 64)
    assert cache.reserve_rows(have, 64)
    grown = int(cache.keys.shape[2])
    assert grown >= 5 + have and grown % cache.step == 0
    assert int(cache.values.shape[2]) == grown and cache.bound == 5
    assert mx.array_equal(cache.keys[..., :5, :], keys).item()  # committed rows kept
    empty = DeviceOffsetKVCache().enter_device_mode()
    assert not empty.reserve_rows(16, 64) and empty.keys is None
    with pytest.raises(ValueError):
        cache.reserve_rows(4, 0)


def test_reserved_host_form_exposes_only_committed_rows():
    """After leave, the reserve's spare rows are capacity, not state (APCv2, capsules)."""
    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.models.device_offset_kv import DeviceOffsetKVCache

    host = KVCache()
    host.update_and_fetch(mx.zeros((1, 2, 5, 4)), mx.zeros((1, 2, 5, 3)))
    cache = DeviceOffsetKVCache.adopt(host)
    cache.reserve_rows(600, 512)
    back = cache.to_host_kv()
    assert int(back.keys.shape[2]) > 600
    keys, values = back.state
    assert int(keys.shape[2]) == 5 and int(values.shape[2]) == 5
    plain = DeviceOffsetKVCache.adopt(host).to_host_kv()
    assert "_fixed_capacity_step" not in plain.__dict__


def test_growth_past_the_reserve_takes_a_whole_step(monkeypatch, small_kv_step):
    """A request longer than the cap grows, counted, one reserve step at a time."""
    from mlx2.runtime import speculative_device_commit as commit
    from test_device_pipeline_raon_cpu import _assert_like_dc4

    monkeypatch.setattr(commit, "KV_RESERVE_CAP", 16)
    capped = _decode_presence(PIPE_PROC_FIX, monkeypatch, steps=40, oracle=False)
    monkeypatch.setattr(commit, "KV_RESERVE_CAP", 4096)
    full = _decode_presence(PIPE_PROC_FIX, monkeypatch, steps=40, oracle=False)
    _assert_like_dc4(full, capped)
    growths = capped[3]["external_device_round_kv_capacity_growths"]
    # 40 tokens past a 16-row reserve in 16-row steps, per K/V layer: a few, not one per 8.
    assert 0 < growths <= 4 * 2


def test_growth_counter_is_absent_without_fixed_shape(monkeypatch, small_kv_step):
    stats = _decode_presence({**PIPE, "processors": True}, monkeypatch, steps=40, oracle=False)[3]
    assert "external_device_round_kv_capacity_growths" not in stats
