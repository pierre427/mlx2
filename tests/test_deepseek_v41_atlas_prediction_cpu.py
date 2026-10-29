"""Optional exact-byte hints rank misses before the bounded candidate limit."""

from collections import Counter

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_expert_stream import PackedExpertStream
from mlx2.runtime.expert_stream_atlas import ExpertStreamAtlas
from mlx2.runtime.gguf.reader import GGUFTensor
from mlx2.runtime.models.deepseek_v41_engram import file_stamp

mx.set_default_device(mx.cpu)


def make_stream(tmp_path):
    rng = np.random.default_rng(4143)
    tensors, arrays, payload = {}, {}, bytearray()
    for kind in ("gate", "up", "down"):
        name = "blk.0.ffn_" + kind + "_exps.weight"
        raw = rng.integers(0, 256, (8, 256*50), np.uint8)
        tensors[name] = GGUFTensor(name, (256, 256, 8), 19, len(payload), raw.nbytes)
        arrays[kind] = raw
        payload.extend(raw.tobytes())
    path = tmp_path / "experts.bin"
    path.write_bytes(payload)
    atlas = ExpertStreamAtlas("synthetic-exact-bytes", layers=1, experts=8, top_k=2)
    stream = PackedExpertStream(path, tensors=tensors, data_offset=0,
                               expected_stat=file_stamp(path.stat()),
                               ceiling_bytes=6*3*256*50, top_k=2, atlas=atlas)
    return stream, arrays, atlas.keys([7], [-1, -1, -1])


def test_atlas_excludes_before_limit_and_preserves_default_order():
    atlas = ExpertStreamAtlas("synthetic", layers=1, experts=8, top_k=2)
    keys = atlas.keys([7], [-1, -1, -1])
    routes = np.array([[0, 1], [2, 3], [4, 5], [6, 7]], np.uint32)
    original = routes.copy()
    atlas.observe(0, keys*4, routes)
    diagnostics = {}
    assert atlas.predict(0, keys, diagnostics=diagnostics) == (0, 1, 2, 3, 4, 5)
    assert diagnostics["truncated_total"] == 2
    assert atlas.predict(0, keys, excluded=set(range(6)), diagnostics=diagnostics) == (6, 7)
    assert diagnostics == dict(candidates_total=8, eligible_total=2,
                               excluded_experts=tuple(range(6)), selected_total=2,
                               truncated_total=0)
    assert atlas.predict(0, keys) == (0, 1, 2, 3, 4, 5)
    np.testing.assert_array_equal(routes, original)


def test_coupling_attempt_without_learned_candidates_is_reported_empty():
    atlas = ExpertStreamAtlas("synthetic", layers=2, experts=8, top_k=2)
    diagnostics = {}
    assert atlas.predict(1, (), previous=np.array([[0, 1]], np.uint32),
                         diagnostics=diagnostics) == ()
    assert atlas.stats["coupling_predictions"] == 1
    assert diagnostics == dict(candidates_total=0, eligible_total=0,
                               excluded_experts=(), selected_total=0, truncated_total=0)


@pytest.mark.parametrize("excluded", [None, [-1], [8], [True], ["1"], list(range(9))])
def test_invalid_exclusions_are_rejected(excluded):
    atlas = ExpertStreamAtlas("synthetic", layers=1, experts=8, top_k=2)
    with pytest.raises(ValueError, match="exclusions"):
        atlas.predict(0, (), excluded=excluded)


def test_lower_ranked_misses_are_prefetched_and_preserve_exact_demand_rows(tmp_path):
    stream, arrays, keys = make_stream(tmp_path)
    try:
        stream.gather("blk.0", mx.array([[0, 1], [2, 3], [4, 5]], mx.uint32), row_buffers=True)
        stream.atlas.tokens[0][keys[0]] = Counter({i: 8-i for i in range(8)})
        stream.begin_context(keys)
        assert set(stream._hints) == {("blk.0", 6), ("blk.0", 7)}
        assert stream.stats["hint_skip_resident"] == 6
        assert stream.stats["hint_prediction_candidates"] == 8
        assert stream.stats["hint_prediction_selected"] == 2
        remap, planes = stream.gather("blk.0", mx.array([[7, 6]], mx.uint32),
                                     row_buffers=True, host_ids=True)
        np.testing.assert_array_equal(remap, [[1, 0]])
        for kind, plane in zip(("gate", "up", "down"), planes):
            for expert, row in zip((6, 7), plane):
                assert np.array(row.data).tobytes() == arrays[kind][expert].tobytes()
        assert stream.stats["hint_issued"] == stream.stats["hint_used"] == 2
        assert stream.stats["hint_useful_bytes"] == 2*stream.row_bytes
        assert stream.stats["bytes_read"] == 6*stream.row_bytes
    finally:
        stream.close()


def test_prediction_counters_distinguish_empty_resident_owned_and_budget(tmp_path):
    stream, _, keys = make_stream(tmp_path)
    try:
        stream.begin_context(keys)
        assert stream.stats["hint_prediction_empty"] == 1
        stream.end_context()
        stream.gather("blk.0", mx.array([[0, 1]], mx.uint32), row_buffers=True)
        stream.atlas.tokens[0][keys[0]] = Counter({0: 2, 1: 1})
        stream.begin_context(keys)
        assert stream.stats["hint_prediction_all_resident"] == 1
        assert stream.stats["hint_skip_resident"] == 2
        stream.end_context()
        stream.caches["blk.0"].clear()
        stream.begin_context(keys)
        assert stream.stats["hint_issued"] == 2
        for future in stream._hints.values():
            future.result(timeout=5)
        # Both completed and pending owned hints exclude duplicate reads.
        stream._predict(0)
        assert stream.stats["hint_prediction_all_inflight"] == 1
        assert stream.stats["hint_skip_inflight"] == 2
        assert stream.stats["hint_prediction_all_excluded"] == 2
        stream._discard_hints()
        stream.hint_budget = stream.row_bytes
        stream._predict(0)
        assert stream.stats["hint_issued"] == 3
        assert stream.stats["hint_skip_budget"] == 1
        assert stream.stats["hint_budget_exhausted"] == 1
        assert stream.hint_owned_bytes() <= stream.hint_budget
    finally:
        stream.close()


def test_submission_failure_is_counted_and_exact_demand_stays_available(tmp_path, monkeypatch):
    stream, arrays, keys = make_stream(tmp_path)
    try:
        stream.atlas.tokens[0][keys[0]] = Counter({0: 2, 1: 1})
        def fail(*args, **kwargs):
            raise OSError("synthetic hint submission failure")
        monkeypatch.setattr(stream.hint_pool, "submit", fail)
        stream.begin_context(keys)
        assert stream.stats["hint_submit_failed"] == stream.stats["hint_failed"] == 1
        stream._predict(0)
        assert stream.stats["hint_skip_disabled"] == 1
        remap, planes = stream.gather("blk.0", mx.array([[1, 0]], mx.uint32),
                                     row_buffers=True, host_ids=True)
        np.testing.assert_array_equal(remap, [[1, 0]])
        for kind, plane in zip(("gate", "up", "down"), planes):
            for expert, row in zip((0, 1), plane):
                assert np.array(row.data).tobytes() == arrays[kind][expert].tobytes()
        assert stream.stats["hint_issued"] == 0
        assert stream.stats["bytes_read"] == 2*stream.row_bytes
    finally:
        stream.close()
