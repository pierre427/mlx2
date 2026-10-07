"""Warm continuations cut prefill where a cold prefill does (sweep 2026-10-06, APC-3).

Tiny Qwen3.8 GDN hybrid served at step 16.  Request A publishes an interior
checkpoint at the end of a shared 60-token preamble; request B (a different
user turn) resumes from it.  With the served ``auto`` policy B is a
"continuation" (min_uncached_fraction 0.5): it used to plan no cuts at all,
so it sliced 76, 83 where a cold B slices 76, 80, 81, 83 and its logprobs
differed from cold in the last bits.  Serving now passes the cold plan's
cuts above the cached offset as cut-only positions.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from route_harness import make_adapter, tiny_qwen38_mtp  # noqa: E402

from mlx2 import memory, serving  # noqa: E402
from mlx2.runtime import os_memory  # noqa: E402
from mlx2.runtime.generate import BatchGenerator  # noqa: E402

POLICY = {
    "count": 4, "min_stride": 16, "placement": "auto",
    "min_uncached_fraction": 0.5,
}


@pytest.fixture
def served(monkeypatch):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    cuts = []
    real = BatchGenerator._record_prefill_chunk

    def record(self, uid, width):
        cuts.append((int(uid), int(width)))
        return real(self, uid, width)

    monkeypatch.setattr(BatchGenerator, "_record_prefill_chunk", record)
    model, vocab = tiny_qwen38_mtp()
    mark = vocab - 1

    class Adapter(make_adapter(model, vocab)):
        def apc_turn_marker_ids(self):
            return (mark,)

        def cache_budget(self, *, mtp):
            from mlx2.adapters.qwen38_memory import Qwen38CacheBudget

            return Qwen38CacheBudget.from_config(dict(vars(model.args)), mtp=mtp)

    pre = [(7 * i + 3) % (vocab - 2) + 1 for i in range(60)]
    ua = [(5 * i + 1) % (vocab - 2) + 1 for i in range(20)]
    ub = [(11 * i + 2) % (vocab - 2) + 1 for i in range(20)]
    prompts = {
        "A": pre + [mark] + ua + [mark, 3, 4],
        "B": pre + [mark] + ub + [mark, 3, 4],
    }
    return Adapter, prompts, cuts


def _engine(adapter, mtp, policy=POLICY):
    engine = serving.ServingEngine(
        "tiny", adapter_factory=adapter, qualification_mode=True, mtp=mtp,
        max_lanes=1, prefill_step=16,
        execution_policy={"apc_interior_checkpoints": policy},
    )
    assert engine.ready.wait(120), engine.error
    return engine


def _run(engine, tokens):
    job = engine.submit(
        {"tokens": list(tokens), "max_tokens": 6, "temperature": 0, "logprobs": True}
    )
    logprobs = []
    while True:
        event = job.events.get(timeout=120)
        assert "error" not in event, event
        if "logprob" in event:
            logprobs.append(event["logprob"])
        if "finish_reason" in event:
            return logprobs, job


def _last_request_cuts(cuts, start):
    last = cuts[-1][0]
    position, out = start, []
    for uid, width in cuts:
        if uid == last:
            position += width
            out.append(position)
    return out


def _arm(adapter, prompts, cuts, *, warm, mtp, policy=POLICY):
    engine = _engine(adapter, mtp, policy)
    try:
        if warm:
            _run(engine, prompts["A"])
        cuts.clear()
        logprobs, job = _run(engine, prompts["B"])
        cached = int(job.cached_tokens or 0)
        return _last_request_cuts(cuts, cached), logprobs, job, cached
    finally:
        engine.close()


@pytest.mark.parametrize(
    "skip_floor", [0.5, 0.0], ids=["continuation_skip", "auto_replan"]
)
@pytest.mark.parametrize("mtp", [False, True], ids=["ordinary", "self_mtp"])
def test_warm_continuation_cuts_and_logprobs_match_cold(served, mtp, skip_floor):
    """``continuation_skip``: the warm request planned nothing.  ``auto_replan``:
    the warm plan respaced its tail above the cached offset and cut where
    cold never does; warm captures are now restricted to cold cuts."""
    adapter, prompts, cuts = served
    policy = dict(POLICY, min_uncached_fraction=skip_floor)
    (cold_cuts, cold_lp, cold_job, _) = _arm(
        adapter, prompts, cuts, warm=False, mtp=mtp, policy=policy
    )
    (warm_cuts, warm_lp, warm_job, cached) = _arm(
        adapter, prompts, cuts, warm=True, mtp=mtp, policy=policy
    )
    assert cached == 60
    assert cached in cold_cuts
    assert warm_cuts == [c for c in cold_cuts if c > cached], (cold_cuts, warm_cuts)
    assert warm_lp == cold_lp
    assert cold_job.slice_aligned is True
    assert warm_job.slice_aligned is True


def test_cold_prefill_cuts_at_the_10_02_geometry():
    """P=5393 with turn markers at 0, 5301, 5386 (the 2026-10-02 27B case)."""
    from mlx2.runtime.interior_placement import cold_prefill_cuts
    from mlx2.serving import APC_INTERIOR_AUTO_POLICY

    mark = 999
    tokens = [(7 * i + 3) % 500 + 1 for i in range(5393)]
    for position in (0, 5301, 5386):
        tokens[position] = mark
    cuts = cold_prefill_cuts(
        tokens, policy=dict(APC_INTERIOR_AUTO_POLICY), marker_ids=(mark,)
    )
    # 5120 sits below the 5301 turn point: since 2026-10-07 a picked point
    # above a tail point no longer suppresses it (warm hit depth fix).
    assert cuts == (4096, 5120, 5301, 5386)
    # The cached offset of the warm follow-up (5301) is a cold cut, so the
    # warm request cut at 5386 only slices exactly as cold.
    assert [c for c in cuts if c > 5301] == [5386]
    suffix = tokens[-3:]
    with_turn = cold_prefill_cuts(
        tokens, policy={"count": 0, "min_stride": 1}, generation_suffixes=(suffix,)
    )
    assert with_turn == (5390,)
