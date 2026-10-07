"""Opt-in QSA stage-one selection timing (MLX_QWEN4_QSA_STAGE1_SELECT_TIMING).

The served GVR A/B needs a P99 selection time; the default path must stay
untimed and its status unchanged.
"""

from __future__ import annotations

import mlx.core as mx

from mlx2.runtime.models import qwen4_qsa_stage1 as stage1


def _arrays():
    with mx.stream(mx.cpu):
        scores = mx.zeros((2, 8), dtype=mx.float32)
        positions = mx.array([7, 7], dtype=mx.int32)
    return scores, positions


def test_default_is_untimed_and_status_unchanged(monkeypatch):
    monkeypatch.setattr(stage1, "_SELECT_TIMING", False)
    monkeypatch.setattr(stage1, "_SELECT_TIMING_STATS", {})
    calls = []

    def fake(scores, positions, *, topk, compress_ratio):
        calls.append(topk)
        return mx.zeros((2, topk), dtype=mx.uint32, stream=mx.cpu)

    monkeypatch.setattr(stage1, "_select_scores_untimed", fake)
    scores, positions = _arrays()
    stage1._select_scores(scores, positions, topk=4, compress_ratio=4)
    assert calls == [4]
    assert "select_timing" not in stage1.qsa_stage1_candidate_status()


def test_timing_histograms_per_producer_and_resets(monkeypatch):
    monkeypatch.setattr(stage1, "_SELECT_TIMING", True)
    monkeypatch.setattr(stage1, "_SELECT_TIMING_STATS", {})
    monkeypatch.setattr(stage1, "_DIRECT_SELECTOR", "gvr")
    monkeypatch.setattr(stage1, "_DIRECT_SELECTOR_MIN_BLOCKS", 1)
    monkeypatch.setattr(
        stage1,
        "_select_scores_untimed",
        lambda scores, positions, *, topk, compress_ratio: mx.zeros(
            (2, topk), dtype=mx.uint32, stream=mx.cpu
        ),
    )
    scores, positions = _arrays()
    for _ in range(3):
        stage1._select_scores(scores, positions, topk=4, compress_ratio=4)
    timing = stage1.qsa_stage1_candidate_status()["select_timing"]
    assert timing["enabled"] is True
    gvr = timing["producers"]["gvr_exact"]
    assert gvr["count"] == 3
    assert sum(gvr["buckets"].values()) == 3
    assert all(key.startswith("le_") for key in gvr["buckets"])
    assert gvr["max_us"] <= gvr["total_us"]
    stage1.qsa_stage1_candidate_status(reset=True)
    assert stage1.qsa_stage1_candidate_status()["select_timing"]["producers"] == {}


def test_record_bins_by_upper_edge(monkeypatch):
    monkeypatch.setattr(stage1, "_SELECT_TIMING_STATS", {})
    stage1._record_select_time("radix_exact", 0.000_011)  # 11 us
    stage1._record_select_time("radix_exact", 100.0)  # beyond the last edge
    buckets = stage1._SELECT_TIMING_STATS["radix_exact"]["buckets"]
    first = next(e for e in stage1.SELECT_TIMING_EDGES_US if e >= 11)
    assert buckets[first] == 1
    assert buckets[stage1.SELECT_TIMING_EDGES_US[-1] * 2] == 1
