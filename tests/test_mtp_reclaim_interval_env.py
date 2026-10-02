"""The self-MTP round honours MLX2_ALLOCATOR_RECLAIM_STEP_INTERVAL (sweep
2026-10-02 S3).  ``_next`` and ``_next_mixed`` read the env override;
``_next_mtp`` used the 512-round constant, so ``0`` (documented: disables
the clear) and any A/B interval were ignored on the MTP route."""

import traceback

import mlx.core as mx
import pytest
from test_batched_mtp import _tiny_qwen4_model

from mlx2.runtime import generate as G
from mlx2.runtime.sample_utils import LaneRNG


def _mtp_clears(monkeypatch):
    hits = []
    real = mx.clear_cache

    def counting():
        if traceback.extract_stack()[-2].name == "_next_mtp":
            hits.append(1)
        real()

    monkeypatch.setattr(G.mx, "clear_cache", counting)
    monkeypatch.setattr(G, "ALLOCATOR_RECLAIM_MTP_TOKEN_INTERVAL", 10**9)
    mx.random.seed(3)
    model = _tiny_qwen4_model()
    gen = G.BatchGenerator(
        model, completion_batch_size=2, prefill_step_size=64,
        self_mtp={"num_draft": 2, "persistent": True},
    )
    gen.insert(
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        max_tokens=[24, 24],
        lane_rngs=[LaneRNG(1), LaneRNG(2)],
        self_mtp_configs=[{"sampling_temp": 0.0}] * 2,
    )
    done, rounds = set(), 0
    try:
        for _ in range(400):
            _, responses = gen.next()
            rounds += 1
            done |= {r.uid for r in responses if r.finish_reason}
            if len(done) == 2:
                break
    finally:
        gen.close()
    return len(hits), gen._steps_counter


def test_interval_zero_disables_the_mtp_step_clear(monkeypatch):
    monkeypatch.setattr(G, "ALLOCATOR_RECLAIM_STEP_INTERVAL", 2)  # would fire
    monkeypatch.setenv("MLX2_ALLOCATOR_RECLAIM_STEP_INTERVAL", "0")
    clears, steps = _mtp_clears(monkeypatch)
    assert steps >= 4
    assert clears == 0


@pytest.mark.parametrize("interval", [2, 3])
def test_env_interval_sets_the_mtp_step_cadence(monkeypatch, interval):
    monkeypatch.setattr(G, "ALLOCATOR_RECLAIM_STEP_INTERVAL", 10**9)
    monkeypatch.setenv("MLX2_ALLOCATOR_RECLAIM_STEP_INTERVAL", str(interval))
    clears, steps = _mtp_clears(monkeypatch)
    assert steps >= 2 * interval
    assert clears == steps // interval
