from __future__ import annotations

import inspect
import struct
from types import SimpleNamespace

from mlx2.runtime.models import qwen4_exp
from mlx2.runtime.models import qwen4_qsa_selector as direct_selector
from mlx2.runtime.models import qwen4_qsa_stage1 as stage1


def _tensor(shape, dtype):
    return SimpleNamespace(shape=shape, dtype=dtype)


def _ordered_float(value):
    bits = struct.unpack("I", struct.pack("f", value))[0]
    return ~bits & 0xFFFFFFFF if bits & 0x80000000 else bits ^ 0x80000000


def _onepass_algorithm(values, topk, *, candidate_cap):
    keys = [_ordered_float(value) for value in values]
    rank = topk - 1
    prefix = 0
    mask = 0

    histogram = [0] * 4096
    for key in keys:
        histogram[key >> 20] += 1
    for bin_id in range(4095, -1, -1):
        if rank < histogram[bin_id]:
            bucket_size = histogram[bin_id]
            prefix = bin_id << 20
            mask = 0xFFF00000
            break
        rank -= histogram[bin_id]

    candidates = [key for key in keys if key & mask == prefix]
    fits = bucket_size <= candidate_cap
    for shift in (10, 0):
        histogram = [0] * 1024
        source = candidates if fits else keys
        for key in source:
            if key & mask == prefix:
                histogram[(key >> shift) & 1023] += 1
        for bin_id in range(1023, -1, -1):
            if rank < histogram[bin_id]:
                prefix |= bin_id << shift
                mask |= 1023 << shift
                break
            rank -= histogram[bin_id]

    greater = [index for index, key in enumerate(keys) if key > prefix]
    tied = [index for index, key in enumerate(keys) if key == prefix]
    return sorted(greater + list(reversed(tied))[: topk - len(greater)])


def test_keys_stationary_candidate_is_default_off_and_geometry_gated(monkeypatch):
    q = _tensor((1, 64, 4, 128), stage1.mx.bfloat16)
    pooled = _tensor((1, 4096, 128), stage1.mx.bfloat16)
    monkeypatch.setattr(stage1, "nax_kernel_available", lambda: True)
    monkeypatch.setattr(stage1, "_KEYS_STATIONARY", False)

    assert stage1.qsa_stage1_score_producer(q, pooled) == "mpp_exact_band"

    monkeypatch.setattr(stage1, "_KEYS_STATIONARY", True)
    monkeypatch.setattr(stage1, "_KEYS_STATIONARY_MIN_QUERY", 64)
    assert (
        stage1.qsa_stage1_score_producer(q, pooled) == "mpp_keys_stationary_exact_band"
    )

    short_q = _tensor((1, 63, 4, 128), stage1.mx.bfloat16)
    assert stage1.qsa_stage1_score_producer(short_q, pooled) == "mpp_exact_band"

    batched_q = _tensor((2, 64, 4, 128), stage1.mx.bfloat16)
    batched_pooled = _tensor((2, 4096, 128), stage1.mx.bfloat16)
    assert stage1.qsa_stage1_score_producer(batched_q, batched_pooled) == "mlx"


def test_keys_stationary_source_keeps_keys_outside_query_loop():
    source = stage1._MPP_KEYS_STATIONARY_SOURCE
    load = source.index("threadgroup InT pooled_tile")
    query_loop = source.index("for (uint query_tile")

    assert load < query_loop
    assert "const uint key_tile = threadgroup_position_in_grid.x" in source
    assert "query_tile / key_tiles" not in source
    assert "scores[size_t(query) * blocks + block]" in source


def test_onepass_selector_is_default_off_and_size_gated(monkeypatch):
    monkeypatch.setattr(stage1, "_DIRECT_SELECTOR", "off")
    monkeypatch.setattr(stage1, "_ONEPASS_TOPK", False)
    assert (
        stage1.qsa_stage1_selector_producer(blocks=65536, block_topk=544)
        == "radix_exact"
    )

    monkeypatch.setattr(stage1, "_ONEPASS_TOPK", True)
    monkeypatch.setattr(stage1, "_ONEPASS_TOPK_MIN_BLOCKS", 2048)
    assert (
        stage1.qsa_stage1_selector_producer(blocks=65536, block_topk=544)
        == "onepass_exact"
    )
    assert (
        stage1.qsa_stage1_selector_producer(blocks=2047, block_topk=512)
        == "radix_exact"
    )
    assert (
        stage1.qsa_stage1_selector_producer(blocks=65536, block_topk=1025)
        == "radix_exact"
    )


def test_direct_selector_is_default_off_mode_gated_and_precedes_onepass(
    monkeypatch,
):
    monkeypatch.setattr(stage1, "_DIRECT_SELECTOR", "off")
    monkeypatch.setattr(stage1, "_ONEPASS_TOPK", False)
    assert (
        stage1.qsa_stage1_selector_producer(blocks=65536, block_topk=544)
        == "radix_exact"
    )

    monkeypatch.setattr(stage1, "_DIRECT_SELECTOR", "direct8")
    monkeypatch.setattr(stage1, "_DIRECT_SELECTOR_MIN_BLOCKS", 1)
    assert (
        stage1.qsa_stage1_selector_producer(blocks=544, block_topk=512)
        == "direct8_exact"
    )

    monkeypatch.setattr(stage1, "_DIRECT_SELECTOR", "direct4")
    monkeypatch.setattr(stage1, "_ONEPASS_TOPK", True)
    assert (
        stage1.qsa_stage1_selector_producer(blocks=65536, block_topk=544)
        == "direct4_exact"
    )

    monkeypatch.setattr(stage1, "_DIRECT_SELECTOR_MIN_BLOCKS", 2048)
    assert (
        stage1.qsa_stage1_selector_producer(blocks=544, block_topk=512) == "radix_exact"
    )
    assert (
        stage1.qsa_stage1_selector_producer(blocks=65536, block_topk=1025)
        == "radix_exact"
    )


def test_stage1_route_reports_primary_and_exact_band_refinement(monkeypatch):
    q = _tensor((1, 64, 4, 128), stage1.mx.bfloat16)
    pooled = _tensor((1, 65536, 128), stage1.mx.bfloat16)
    monkeypatch.setattr(stage1, "nax_kernel_available", lambda: True)
    monkeypatch.setattr(stage1, "_KEYS_STATIONARY", True)
    monkeypatch.setattr(stage1, "_DIRECT_SELECTOR", "off")
    monkeypatch.setattr(stage1, "_ONEPASS_TOPK", True)
    monkeypatch.setattr(stage1, "_ONEPASS_TOPK_MIN_BLOCKS", 2048)

    assert stage1.qsa_stage1_route(q, pooled, block_topk=512) == {
        "score_producer": "mpp_keys_stationary_exact_band",
        "selector": "onepass_exact",
        "refine_selector": "radix_exact",
    }


def test_onepass_source_preserves_reference_tie_and_canonicalization_rules():
    source = inspect.getsource(stage1._onepass_select_kernel)

    assert "HISTOGRAM_BINS = 4096" in source
    assert "CANDIDATE_CAP = 1024" in source
    assert "chunk_end = valid_count" in source
    assert "chunk_end > WIDTH ? chunk_end - WIDTH" in source
    assert "qsa_id_before" in source
    assert "Oversized buckets fail closed to row rescans" in source


def test_direct_selector_source_preserves_reference_tie_and_output_rules():
    direct8 = inspect.getsource(direct_selector._direct8_kernel)
    direct4 = inspect.getsource(direct_selector._direct4_kernel)

    assert "for (uint pass = 0u; pass < 8u" in direct8
    assert "qsa_direct_composite_key" in direct8
    assert "for (uint pass = 0u; pass < 4u" in direct4
    assert "valid_count - 1u - reverse" in direct4
    assert "metal::simd_prefix_exclusive_sum" in direct4
    assert "larger block IDs win" in direct4
    assert "qsa_direct_id_before" in direct_selector._SORT_AND_STORE


def test_direct_selector_dispatch_is_reached_and_counted(monkeypatch):
    scores = SimpleNamespace(shape=(3, 4096))
    positions = object()
    calls = []
    sentinel = object()

    def direct4(scores_arg, positions_arg, *, topk, compress_ratio):
        calls.append((scores_arg, positions_arg, topk, compress_ratio))
        return sentinel

    monkeypatch.setattr(stage1, "_DIRECT_SELECTOR", "direct4")
    monkeypatch.setattr(stage1, "_DIRECT_SELECTOR_MIN_BLOCKS", 1)
    monkeypatch.setattr(stage1, "select_scores_direct4", direct4)
    stage1.qsa_stage1_candidate_status(reset=True)

    assert (
        stage1._select_scores(scores, positions, topk=512, compress_ratio=128)
        is sentinel
    )
    assert calls == [(scores, positions, 512, 128)]
    assert stage1.qsa_stage1_candidate_status()["runtime_counts"] == {
        "direct4_topk_dispatches": 1
    }


def test_onepass_algorithm_matches_composite_key_reference_with_overflow():
    cases = [
        ([3.0, 1.0, 2.0, 3.0, -1.0, 3.0, 2.0], 4),
        ([0.0] * 40, 9),
        ([float(index % 7 - 3) for index in range(80)], 17),
    ]
    for values, topk in cases:
        expected = sorted(
            sorted(
                range(len(values)),
                key=lambda index: (_ordered_float(values[index]), index),
                reverse=True,
            )[:topk]
        )
        assert _onepass_algorithm(values, topk, candidate_cap=4) == expected


def test_candidate_dispatch_is_isolated_from_reference_mpp_scorer():
    source = inspect.getsource(stage1.qsa_stage1_select)

    assert 'producer == "mpp_keys_stationary_exact_band"' in source
    assert "_mpp_keys_stationary_scores(q, pooled)" in source
    assert "else _mpp_scores(q, pooled)" in source


def test_candidate_dispatch_counts_are_bounded_and_resettable():
    stage1.qsa_stage1_candidate_status(reset=True)
    stage1._record_candidate_dispatch("keys_stationary_dispatches")
    stage1._record_candidate_dispatch("onepass_topk_dispatches")
    stage1._record_candidate_dispatch("direct8_topk_dispatches")
    stage1._record_candidate_dispatch("direct4_topk_dispatches")

    assert stage1.qsa_stage1_candidate_status()["runtime_counts"] == {
        "keys_stationary_dispatches": 1,
        "onepass_topk_dispatches": 1,
        "direct8_topk_dispatches": 1,
        "direct4_topk_dispatches": 1,
    }
    stage1.qsa_stage1_candidate_status(reset=True)
    assert stage1.qsa_stage1_candidate_status()["runtime_counts"] == {}


def test_stage1_status_reports_and_resets_bounded_receipts(monkeypatch):
    qwen4_exp.qsa_stage1_status(reset=True)
    monkeypatch.setattr(stage1, "_KEYS_STATIONARY", True)
    qwen4_exp._record_qsa_stage1(
        engaged=True,
        reason="engaged_mpp_keys_stationary_exact_band",
        batch=1,
        query_width=64,
        blocks=4096,
    )

    status = qwen4_exp.qsa_stage1_status()
    assert status["candidates"] == {
        "keys_stationary_configured": True,
        "keys_stationary_min_query": stage1._KEYS_STATIONARY_MIN_QUERY,
        "keys_stationary_qualification": "not_qualified_benefit",
        "keys_stationary_selected": False,
        "onepass_topk_configured": stage1._ONEPASS_TOPK,
        "onepass_topk_min_blocks": stage1._ONEPASS_TOPK_MIN_BLOCKS,
        "onepass_topk_qualification": "rejected_performance",
        "onepass_topk_selected": False,
        "direct_selector_configured": stage1._DIRECT_SELECTOR,
        "direct_selector_min_blocks": stage1._DIRECT_SELECTOR_MIN_BLOCKS,
        "direct8_qualification": "qualified_default_off",
        "direct4_qualification": "qualified_default_off",
        "gvr_qualification": "unqualified_research_candidate",
        "gvr_count_paths": stage1._GVR_COUNT_PATHS,
        "direct_selector_selected": stage1._DIRECT_SELECTOR != "off",
        "direct_selector_observed_used": False,
        "qualification_receipt": (
            "qualification/runs/qsa-stage1-pr91-20260928/qualification.json"
        ),
        "direct_selector_qualification_receipt": (
            "qualification/runs/qsa-stage1-direct-selector-20261004/"
            "qualification-direct8-accepted-final.json"
        ),
        "direct_selector_model_receipt": (
            "qualification/runs/qsa-stage1-direct-selector-20261004/"
            "model-ab-65k-direct8.json"
        ),
        "default_producer": "mpp_exact_band",
        "default_selector": "radix_exact",
        "runtime_counts": {},
    }
    assert status["counts"] == {
        "engaged_mpp_keys_stationary_exact_band": 1,
    }
    assert status["attempts"] == 1
    assert status["engagements"] == 1
    assert status["fallbacks"] == 0
    assert status["last_receipt"] == {
        "engaged": True,
        "reason": "engaged_mpp_keys_stationary_exact_band",
        "batch": 1,
        "query_width": 64,
        "blocks": 4096,
        "score_producer": None,
        "selector": None,
        "refine_selector": None,
    }

    qwen4_exp.qsa_stage1_status(reset=True)
    reset = qwen4_exp.qsa_stage1_status()
    assert reset["counts"] == {}
    assert reset["attempts"] == 0
    assert reset["last_receipt"] is None
