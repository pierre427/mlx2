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

    monkeypatch.setattr(mx, "async_eval", async_eval)
    monkeypatch.setattr(pipe, "dispatch", dispatch)
    result = _decode_presence(PIPE_PROC_FIX, monkeypatch, steps=steps, oracle=False)
    monkeypatch.undo()
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
