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


def _fake_select(scores, positions, *, topk, compress_ratio):
    rows = int(scores.shape[0])
    return mx.broadcast_to(mx.arange(topk, dtype=mx.uint32)[None], (rows, topk))


def test_exact_band_refine_is_timed_apart_from_primary(monkeypatch):
    # The exact-band route dispatches the selector twice per stage-one call:
    # over all pooled blocks (block_topk + 32 candidates), then over those
    # candidates (block_topk).  Both resolve to the same producer name, so
    # one histogram held an even mixture and its P50 was the refine dispatch
    # (sweep 2026-10-08, flashnext-kernels#3).
    monkeypatch.setattr(stage1, "qsa_stage1_kernel_available", lambda: True)
    monkeypatch.setattr(stage1, "nax_kernel_available", lambda: True)
    monkeypatch.setattr(stage1, "_KEYS_STATIONARY", False)
    monkeypatch.setattr(stage1, "_ONEPASS_TOPK", False)
    monkeypatch.setattr(stage1, "_DIRECT_SELECTOR", "off")
    monkeypatch.setattr(stage1, "_SELECT_TIMING", True)
    monkeypatch.setattr(stage1, "_SELECT_TIMING_STATS", {})
    monkeypatch.setattr(
        stage1,
        "_mpp_scores",
        lambda q, pooled: mx.zeros(
            (int(q.shape[1]), int(pooled.shape[1])), dtype=mx.float32
        ),
    )
    widths = []

    def select(scores, positions, *, topk, compress_ratio):
        widths.append(int(scores.shape[1]))
        return _fake_select(scores, positions, topk=topk, compress_ratio=compress_ratio)

    monkeypatch.setattr(stage1, "_select_scores_untimed", select)

    (blocks, topk, ratio, length) = (64, 8, 4, 2)
    q = mx.zeros((1, length, 4, 128), dtype=mx.float16)
    pooled = mx.zeros((1, blocks, 128), dtype=mx.float16)
    positions = mx.full((1, length), blocks * ratio - 1, dtype=mx.int32)

    route = stage1.qsa_stage1_route(q, pooled, block_topk=topk)
    assert route["score_producer"] == "mpp_exact_band"
    assert route["selector"] == route["refine_selector"] == "radix_exact"

    calls = 3
    for _ in range(calls):
        stage1.qsa_stage1_select(
            q, pooled, positions, block_topk=topk, compress_ratio=ratio
        )
    assert widths == [blocks, topk + 32] * calls
    producers = stage1.qsa_stage1_candidate_status(reset=True)["select_timing"][
        "producers"
    ]
    # Both dispatches are still timed ...
    assert sum(p["count"] for p in producers.values()) == 2 * calls
    # ... but the primary selector's histogram holds only primary dispatches.
    assert producers[route["selector"]]["count"] == calls
    assert producers[route["refine_selector"] + "_refine"]["count"] == calls
