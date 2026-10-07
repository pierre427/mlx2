"""A self-MTP one-call burst is several forwards, not one (Codex review of
the 2026-10-06 scheduling sweep, P2).

``_stall_budget_rows`` summed the burst's residual rows against one stall
bound at the first candidate's depth, as if the burst were one forward.  Each
``prepare_self_mtp_lane`` call pays its own fixed cost and runs at its own KV
depth, so several short prompts overran the target.  The existing regression
used a zero-fixed-cost clock, so it passed for the wrong reason.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_decode_first_publish import tiny_model
from test_mtp_cohort_formation import _admission, _generator

from mlx2.runtime import generate as G
from mlx2.runtime import hybrid_speculative as H
from mlx2.runtime.adaptive_policy import DecodeTimeFairness


def _mtp_clock(monkeypatch, *, fixed, per_token):
    now = [100.0]
    clock = lambda: now[0]  # noqa: E731
    monkeypatch.setattr(
        G, "time", SimpleNamespace(perf_counter=clock, monotonic=clock, time=clock)
    )
    stalls = []
    real_prepare = H.prepare_self_mtp_lane
    real_advance = H.advance_self_mtp_prefill
    real_next = G.MTPGenerationBatch.next
    real_plain_next = G.GenerationBatch.next

    def prepare(prompt, *args, **kwargs):
        out = real_prepare(prompt, *args, **kwargs)
        cost = fixed + per_token * int(prompt.shape[0])
        now[0] += cost
        stalls[-1] += cost
        return out

    def advance(prompt, *args, max_tokens, **kwargs):
        out = real_advance(prompt, *args, max_tokens=max_tokens, **kwargs)
        cost = fixed + per_token * int(out[3])
        now[0] += cost
        stalls[-1] += cost
        return out

    def next_(self):
        out = real_next(self)
        now[0] += 0.020
        return out

    def plain_next(self):
        # Plain-fallback lanes decode too, so they repay debt like MTP ones.
        out = real_plain_next(self)
        now[0] += 0.020
        return out

    monkeypatch.setattr(H, "prepare_self_mtp_lane", prepare)
    monkeypatch.setattr(H, "advance_self_mtp_prefill", advance)
    monkeypatch.setattr(G.MTPGenerationBatch, "next", next_)
    monkeypatch.setattr(G.GenerationBatch, "next", plain_next)
    return stalls


def test_short_prompt_burst_pays_one_fixed_cost_per_preparation(monkeypatch):
    """Eight 4-token arrivals; each preparation costs 40 ms + 0.2 ms/token.

    Before: the row bound (about 13 rows) admitted three per round
    (~125 ms against the 100 ms target).
    """
    stalls = _mtp_clock(monkeypatch, fixed=0.040, per_token=0.0002)
    gen = _generator(
        tiny_model(), mtp_admission=_admission(),
        decode_time_fairness={"enabled": True, "stall_target_ms": 100.0},
    )
    try:
        gen.insert([[3, 4, 5, 6, 7]], max_tokens=[400])
        for _ in range(4):
            stalls.append(0.0)
            gen.next()
        for k in range(2):
            # Calibrate the "mtp" key on a few short preparations first.
            gen.insert([[k + 1, k + 2, k + 3, k + 4]], max_tokens=[2])
            for _ in range(3):
                stalls.append(0.0)
                gen.next()
        gen.insert(
            [[(k * 3 + i) % 40 + 1 for i in range(4)] for k in range(8)],
            max_tokens=[2] * 8,
        )
        stalls.clear()
        for _ in range(200):
            stalls.append(0.0)
            gen.next()
            if not gen._unprocessed_sequences:
                break
        assert not gen._unprocessed_sequences
        assert max(stalls) <= 0.100, stalls
        assert gen.scheduler_stats["mtp_stall_budget_deferred_rows"] > 0
    finally:
        gen.close()


def test_burst_keep_charges_each_forward_at_its_own_depth():
    policy = DecodeTimeFairness(enabled=True, stall_target_ms=100.0)
    for _ in range(4):
        policy.observe_decode(0.010)
    for _ in range(4):
        policy.observe_prefill(64, 0.020, contended=True, depth=0, kind="mtp")
        policy.observe_prefill(
            64, 0.080, contended=True, depth=40000, kind="mtp"
        )
    shallow = [(32, 0)] * 6
    deep = [(32, 0)] + [(32, 40000)] * 5
    assert policy.burst_keep(shallow, kind="mtp") > policy.burst_keep(
        deep, kind="mtp"
    )
    # Each forward pays the fixed cost: many tiny forwards do not all fit.
    assert policy.burst_keep([(1, 0)] * 50, kind="mtp") < 50
    # The first forward always runs.
    assert policy.burst_keep([(100000, 40000)], kind="mtp") == 1
