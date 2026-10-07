"""Co-arrived self-MTP prompts form one cohort (GPU 2026-10-07, Flash-Next).

Eight ~700-token prompts arrived together at an idle Flash-Next MTP server
(``num_draft`` 2, step 8192).  On one repetition the first boundary prepared
only two of them (the other six were queued at that boundary); the six were
then "arrivals beside a decoding lane" and the stall bound rationed them in
waves: TTFT 1.25 s (2), 8.9 s (3), 12.2 s (3), served aggregate 86.7 tok/s
against 143-162 when all eight were prepared together.  The two decoding
lanes were their own co-arrivals: there was no earlier neighbour to protect.

A prompt that arrives while a lane is already decoding stays stall-bounded.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_decode_first_publish import tiny_model
from test_mtp_cohort_formation import SELF_MTP, _admission, _widths

from mlx2.runtime import generate as G
from mlx2.runtime import hybrid_speculative as H
from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy


@pytest.fixture(scope="module")
def model():
    return tiny_model()


class _FirstBoundaryQueues:
    """Memory admission that queues all but ``keep`` fresh rows once.

    Stands in for the GPU run's first boundary, where freed pages had not yet
    returned and the controller admitted two of eight joining rows.
    """

    def __init__(self, keep=2):
        self._inner = _admission()
        self._keep = keep
        self.fired = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, rows):
        decisions = dict(self._inner(rows))
        fresh = [row[0] for row in rows if not row[3]]
        if not self.fired and len(fresh) > self._keep:
            self.fired = True
            for uid in fresh[self._keep:]:
                decisions[uid] = "queue"
        return decisions


def _clock(monkeypatch, per_token=0.002):
    """Each preparation/slice costs ``per_token`` per token; decode 20 ms."""
    now = [100.0]
    clock = lambda: now[0]  # noqa: E731
    monkeypatch.setattr(
        G, "time", SimpleNamespace(perf_counter=clock, monotonic=clock, time=clock)
    )
    rounds = []
    real_prepare = H.prepare_self_mtp_lane
    real_advance = H.advance_self_mtp_prefill
    real_next = G.MTPGenerationBatch.next
    real_plain_next = G.GenerationBatch.next

    def prepare(prompt, *args, **kwargs):
        out = real_prepare(prompt, *args, **kwargs)
        now[0] += per_token * int(prompt.shape[0])
        rounds[-1]["prepared"] += 1
        return out

    def advance(prompt, *args, max_tokens, **kwargs):
        out = real_advance(prompt, *args, max_tokens=max_tokens, **kwargs)
        cost = per_token * int(out[3])
        now[0] += cost
        rounds[-1]["stall"] += cost
        return out

    def next_(self):
        out = real_next(self)
        now[0] += 0.020
        return out

    def plain_next(self):
        out = real_plain_next(self)
        now[0] += 0.020
        return out

    monkeypatch.setattr(H, "prepare_self_mtp_lane", prepare)
    monkeypatch.setattr(H, "advance_self_mtp_prefill", advance)
    monkeypatch.setattr(G.MTPGenerationBatch, "next", next_)
    monkeypatch.setattr(G.GenerationBatch, "next", plain_next)
    return rounds


def _fn_generator(model, admission, **kwargs):
    # Flash-Next serving shape, scaled: one-chunk prompts, stall target 100 ms.
    return G.BatchGenerator(
        model,
        completion_batch_size=8,
        prefill_step_size=1024,
        prefill_batch_size=2,
        prefill_batch_window=1,
        adaptive_prefill=True,
        self_mtp=dict(SELF_MTP, prefill_step_size=1024),
        mtp_admission=admission,
        mtp_ordinary_handoff=MTPOrdinaryHandoffPolicy.from_value(
            {"enabled": True, "max_mtp_width": 3}
        ),
        decode_time_fairness={"enabled": True, "stall_target_ms": 100.0},
        prefill_scheduling={"order": "srpt", "one_slice_contention": True},
        decode_first={"enabled": True},
        **kwargs,
    )


def _drain(gen, rounds, limit=60):
    for _ in range(limit):
        rounds.append({"prepared": 0, "stall": 0.0})
        gen.next()
        if not gen._unprocessed_sequences:
            return
    raise AssertionError("queued prompts never prepared")


def test_coarrivals_split_by_the_first_boundary_prepare_together(model, monkeypatch):
    rounds = _clock(monkeypatch)
    admission = _FirstBoundaryQueues(keep=2)
    gen = _fn_generator(model, admission)
    try:
        prompts = [[(5 * k + i) % 50 + 1 for i in range(60)] for k in range(8)]
        gen.insert(prompts, max_tokens=[40] * 8)
        _drain(gen, rounds)
        assert admission.fired
        prepared = [r["prepared"] for r in rounds if r["prepared"]]
        # Two at the split boundary, then the other six in one round: none
        # of them waits behind a neighbour it arrived with.
        assert prepared == [2, 6], prepared
        stats = gen.scheduler_stats
        assert stats.get("mtp_stall_budget_deferred_rows", 0) == 0
        assert stats["mtp_coarrival_exempt_rows"] == 6
    finally:
        gen.close()


def test_arrivals_beside_a_decoding_lane_stay_stall_bounded(model, monkeypatch):
    """The same six prompts, inserted after the first lanes decoded."""
    rounds = _clock(monkeypatch)
    gen = _fn_generator(model, _admission())
    try:
        gen.insert(
            [[(5 * k + i) % 50 + 1 for i in range(60)] for k in range(2)],
            max_tokens=[400] * 2,
        )
        for _ in range(3):
            rounds.append({"prepared": 0, "stall": 0.0})
            gen.next()
        gen.insert(
            [[(5 * k + i) % 50 + 1 for i in range(60)] for k in range(2, 8)],
            max_tokens=[40] * 6,
        )
        rounds.clear()
        _drain(gen, rounds)
        assert max(r["prepared"] for r in rounds) < 6, rounds
        assert gen.scheduler_stats["mtp_stall_budget_deferred_rows"] > 0
        assert gen.scheduler_stats.get("mtp_coarrival_exempt_rows", 0) == 0
    finally:
        gen.close()


def test_outlier_coarrival_stays_bounded_beside_its_released_sibling(
    model, monkeypatch
):
    """150 tokens beside 1024 at step 256: the short lane is released early
    (the hold's outlier guard), and the long sibling's slices beside it are
    stall-bounded, not exempt."""
    rounds = _clock(monkeypatch, per_token=0.001)
    gen = G.BatchGenerator(
        model,
        completion_batch_size=8,
        prefill_step_size=256,
        prefill_batch_size=2,
        prefill_batch_window=1,
        adaptive_prefill=True,
        self_mtp=dict(SELF_MTP, prefill_step_size=256),
        mtp_admission=_admission(),
        decode_time_fairness={"enabled": True, "stall_target_ms": 100.0},
    )
    try:
        short = [(3 * j) % 50 + 1 for j in range(150)]
        long_ = [(5 * j) % 50 + 1 for j in range(1024)]
        gen.insert([short, long_], max_tokens=[3000, 4])
        decoding = []
        for _ in range(600):
            rounds.append({"prepared": 0, "stall": 0.0})
            gen.next()
            decoding.append(bool(_widths(gen)[0]))
            if not gen._unprocessed_sequences:
                break
        assert not gen._unprocessed_sequences
        beside = [r["stall"] for r, live in zip(rounds[1:], decoding) if live]
        assert beside and max(beside) <= 0.150, beside
        assert gen.scheduler_stats.get("mtp_coarrival_exempt_rows", 0) == 0
    finally:
        gen.close()

