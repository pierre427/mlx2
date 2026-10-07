"""Decode-fairness stall bound end to end (sweep 2026-10-06, SS-2/SS-3).

A fake clock charges each prefill forward its true cost (prompt rows x
padded width, or processed tokens for a self-MTP slice / preparation) and
each decode step a fixed cost, so the scheduler measures exactly what the
kernels would pay.  Before the cost model the bound was row-blind and kept
the running maximum rate; the self-MTP one-call preparation had no bound at
all.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_decode_first_publish import tiny_model
from test_mtp_cohort_formation import _admission, _generator

from mlx2.runtime import generate as G
from mlx2.runtime import hybrid_speculative as H


@pytest.fixture(scope="module")
def model():
    return tiny_model()


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.cost_per_row_token = 1e-4
        self.decode_cost = 0.020
        self.prefill_rounds = []  # (rows, width, seconds)

    def __call__(self):
        return self.t


def _install_clock(monkeypatch, clock):
    monkeypatch.setattr(
        G, "time", SimpleNamespace(perf_counter=clock, monotonic=clock, time=clock)
    )


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    _install_clock(monkeypatch, clock)
    real_prompt = G.PromptProcessingBatch.prompt
    real_next = G.GenerationBatch.next

    def prompt(self, tokens, *, forward_fn=None):
        rows = sum(1 for t in tokens if len(t))
        width = max((len(t) for t in tokens), default=0)
        out = real_prompt(self, tokens, forward_fn=forward_fn)
        seconds = len(tokens) * width * clock.cost_per_row_token
        clock.t += seconds
        if width > 1:
            clock.prefill_rounds.append((rows, width, seconds))
        return out

    def next_(self):
        out = real_next(self)
        clock.t += clock.decode_cost
        return out

    monkeypatch.setattr(G.PromptProcessingBatch, "prompt", prompt)
    monkeypatch.setattr(G.GenerationBatch, "next", next_)
    return clock


def _ordinary(model, **kw):
    return G.BatchGenerator(
        model,
        completion_batch_size=8,
        prefill_batch_size=8,
        prefill_step_size=4096,
        decode_time_fairness={
            "enabled": True, "fair_share": 0.5, "stall_target_ms": 100.0,
        },
        **kw,
    )


def test_multirow_round_respects_stall_target(model, clock):
    """Five prompt rows share one forward: the bound is per row now."""
    gen = _ordinary(model)
    try:
        gen.insert([[5, 6, 7, 8]], max_tokens=[400])
        for _ in range(3):
            gen.next()
        gen.insert([[(i % 50) + 1 for i in range(6000)]], max_tokens=[2])
        for _ in range(12):
            gen.next()
        clock.prefill_rounds.clear()
        gen.insert(
            [[(i * 3 + k) % 50 + 1 for i in range(6000)] for k in range(4)],
            max_tokens=[2] * 4,
        )
        for _ in range(12):
            gen.next()
        multi = [r for r in clock.prefill_rounds if r[0] > 1]
        assert multi, clock.prefill_rounds
        assert max(r[2] for r in multi) <= 0.150, multi
    finally:
        gen.close()


def test_slowdown_is_tracked_within_three_slices(model, clock):
    gen = _ordinary(model)
    try:
        gen.insert([[5, 6, 7, 8]], max_tokens=[4000])
        for _ in range(3):
            gen.next()
        gen.insert([[(i % 50) + 1 for i in range(30000)]], max_tokens=[2])
        for _ in range(6):
            gen.next()
        clock.cost_per_row_token *= 4
        clock.prefill_rounds.clear()
        for _ in range(120):
            gen.next()
        slices = [r[2] for r in clock.prefill_rounds]
        assert len(slices) >= 4, slices
        # The first slice after the shift is unavoidable; by the third the
        # bound has followed it.
        assert max(slices[2:]) <= 0.150, slices
        assert gen.scheduler_stats["decode_fairness_cost_regime_shifts_up"] >= 1
    finally:
        gen.close()


def test_serving_construction_enforces_its_recorded_target(model, clock):
    """adaptive_prefill=True, prefill_batch_size=2, 500 ms (serving shape).

    Before: after a 16x slowdown the adaptive EWMA settled at 1024 ms slices
    while route identity recorded a 500 ms stall target.
    """
    clock.cost_per_row_token = 2.5e-4
    gen = G.BatchGenerator(
        model, completion_batch_size=8, prefill_batch_size=2,
        prefill_batch_window=1, prefill_step_size=2048, adaptive_prefill=True,
        decode_time_fairness={
            "enabled": True, "fair_share": 0.5, "stall_target_ms": 500.0,
        },
    )
    try:
        gen.insert([[5, 6, 7, 8]], max_tokens=[3000])
        for _ in range(3):
            gen.next()
        gen.insert([[(i % 50) + 1 for i in range(60000)]], max_tokens=[2])
        for _ in range(20):
            gen.next()
        clock.cost_per_row_token *= 16
        clock.prefill_rounds.clear()
        for _ in range(200):
            gen.next()
        assert max(r[2] for r in clock.prefill_rounds[-3:]) <= 0.75
    finally:
        gen.close()


def test_running_max_estimator_stays_reachable(model, clock):
    gen = G.BatchGenerator(
        model, completion_batch_size=8, prefill_batch_size=8,
        prefill_step_size=4096,
        decode_time_fairness={
            "enabled": True, "stall_target_ms": 100.0,
            "estimator": "running_max",
        },
    )
    try:
        assert gen.decode_time_fairness.estimator == "running_max"
    finally:
        gen.close()


# -- self-MTP preparation is a stall-bounded prefill (SS-3) ----------------


def _mtp_clock(monkeypatch, per_token=0.002):
    now = [100.0]
    clock = lambda: now[0]  # noqa: E731
    monkeypatch.setattr(
        G, "time", SimpleNamespace(perf_counter=clock, monotonic=clock, time=clock)
    )
    stalls = []
    real_prepare = H.prepare_self_mtp_lane
    real_advance = H.advance_self_mtp_prefill
    real_next = G.MTPGenerationBatch.next

    def prepare(prompt, *args, **kwargs):
        out = real_prepare(prompt, *args, **kwargs)
        cost = per_token * int(prompt.shape[0])
        now[0] += cost
        stalls[-1] += cost
        return out

    def advance(prompt, *args, max_tokens, **kwargs):
        out = real_advance(prompt, *args, max_tokens=max_tokens, **kwargs)
        cost = per_token * int(out[3])
        now[0] += cost
        stalls[-1] += cost
        return out

    def next_(self):
        out = real_next(self)
        now[0] += 0.020
        return out

    monkeypatch.setattr(H, "prepare_self_mtp_lane", prepare)
    monkeypatch.setattr(H, "advance_self_mtp_prefill", advance)
    monkeypatch.setattr(G.MTPGenerationBatch, "next", next_)
    return stalls


def test_mtp_one_chunk_burst_respects_stall_target(model, monkeypatch):
    """Six 60-token arrivals beside a decoding lane (120 ms each).

    Before: one round prepared all six back to back (720 ms) with no debt.
    """
    stalls = _mtp_clock(monkeypatch)
    m = tiny_model()
    gen = _generator(
        m, mtp_admission=_admission(),
        decode_time_fairness={"enabled": True, "stall_target_ms": 100.0},
    )
    try:
        gen.insert([[3, 4, 5, 6, 7]], max_tokens=[300])
        for _ in range(4):
            stalls.append(0.0)
            gen.next()
        gen.insert(
            [[(i * 5 + k) % 50 + 1 for i in range(60)] for k in range(6)],
            max_tokens=[5] * 6,
        )
        stalls.clear()
        for _ in range(40):
            stalls.append(0.0)
            gen.next()
        f = gen.decode_time_fairness
        # The oldest row always runs (120 ms > 100 ms target); never two.
        assert max(stalls) <= 0.150, stalls
        assert f.counters["prefill_chunks"] >= 6
        assert gen.scheduler_stats["mtp_stall_budget_deferred_rows"] > 0
        assert not gen._unprocessed_sequences
    finally:
        gen.close()


def test_mtp_long_prompt_tail_is_not_prepared_whole_beside_decode(
    model, monkeypatch
):
    """GPU 2026-10-06 (Flash-Next, step 8192): a 16K prompt beside a B1
    decoding lane stalled it 9.5 s.  Slices were bounded until the residual
    fit the uncontended step, then the whole remaining 8K was prepared in
    one call.  Scaled: step 1024, a 1500-token prompt, 1 ms per token.
    """
    stalls = _mtp_clock(monkeypatch, per_token=0.001)
    m = tiny_model()
    config = {
        "num_draft": 2, "persistent": True, "segment_aware_live_tip": True,
        "segment_aware_cohort_size": 8,
        "segment_aware_async_qsa_promotion": False,
    }
    gen = G.BatchGenerator(
        m, completion_batch_size=8, prefill_step_size=1024,
        prefill_batch_size=2, prefill_batch_window=1, adaptive_prefill=True,
        self_mtp=config, mtp_admission=_admission(),
        decode_time_fairness={"enabled": True, "stall_target_ms": 100.0},
    )
    try:
        gen.insert([[3, 4, 5, 6, 7]], max_tokens=[2000])
        for _ in range(4):
            stalls.append(0.0)
            gen.next()
        gen.insert([[(i * 7) % 50 + 1 for i in range(1500)]], max_tokens=[5])
        stalls.clear()
        for _ in range(300):
            stalls.append(0.0)
            gen.next()
            if not gen._unprocessed_sequences:
                break
        assert not gen._unprocessed_sequences
        assert max(stalls) <= 0.150, max(stalls)
    finally:
        gen.close()
