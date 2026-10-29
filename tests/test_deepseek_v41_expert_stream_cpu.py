"""Packed-row byte identity, remapping, bounded ownership and auto admission."""


import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_expert_stream import PackedExpertStream, read_rows
from mlx2.runtime.gguf.reader import GGUFTensor
from mlx2.runtime.models.deepseek_v41_engram import file_stamp
from mlx2.runtime.streamed_load import auto_expert_streaming_decision

mx.set_default_device(mx.cpu)


def fixture(tmp_path):
    rng = np.random.default_rng(4142)
    tensors, arrays, data = {}, {}, bytearray()
    for kind in ("gate", "up", "down"):
        name = "blk.0.ffn_"+kind+"_exps.weight"
        raw = rng.integers(0, 256, (4, 256*50), np.uint8)
        tensors[name] = GGUFTensor(name, (256, 256, 4), 19, len(data), raw.nbytes)
        data.extend(raw.tobytes())
        arrays[kind] = raw
    path = tmp_path / "experts.bin"
    path.write_bytes(data)
    stream = PackedExpertStream(path, tensors=tensors, data_offset=0, expected_stat=file_stamp(path.stat()),
                               ceiling_bytes=2*3*256*50, top_k=2)
    return stream, arrays, path


def test_native_packed_rows_preserve_topk_order_and_evict_exactly(tmp_path):
    stream, arrays, _ = fixture(tmp_path)
    try:
        for values in ([[3, 1]], [[2, 3]], [[1, 3]], [[0, 2], [3, 1]]):
            ids = mx.array(values, mx.uint32)
            remap, matrices = stream.gather("blk.0", ids)
            unique = np.unique(values)
            np.testing.assert_array_equal(unique[np.array(remap)], values)
            for kind, matrix in zip(("gate", "up", "down"), matrices):
                np.testing.assert_array_equal(np.array(matrix.data).reshape(len(unique), -1), arrays[kind][unique])
            assert stream.live_bytes() <= stream.ceiling
            assert stream.unfilled_reserve_bytes()+stream.live_bytes() == stream.reserved_bytes()
        assert stream.stats["evictions"] > 0 and stream.stats["cache_hits"] > 0
        with pytest.raises(ValueError, match="outside"):
            stream.gather("blk.0", mx.array([[4]], mx.uint32))
    finally:
        stream.close()
    with pytest.raises(ValueError, match="closed"):
        stream.gather("blk.0", mx.array([[0]], mx.uint32))


def test_source_mutation_refuses_before_publishing_rows(tmp_path):
    stream, _, path = fixture(tmp_path)
    try:
        with path.open("ab") as f:
            f.write(b"x")
        with pytest.raises(ValueError, match="changed"):
            stream.gather("blk.0", mx.array([[0]], mx.uint32))
    finally:
        stream.close()


def test_selected_hits_are_protected_before_loading_misses(tmp_path):
    stream, _, _ = fixture(tmp_path)
    try:
        for ids in ([[1, 3]], [[2, 1]], [[1, 2]]):
            stream.gather("blk.0", mx.array(ids, mx.uint32), row_buffers=True)
        assert stream.stats["cache_misses"] == 3
        assert stream.stats["evictions"] == 1
        assert set(stream.caches["blk.0"]) == {1, 2}
    finally:
        stream.close()


def test_cold_rows_upload_in_one_bounded_batch(tmp_path, monkeypatch):
    stream, _, _ = fixture(tmp_path)
    original, calls = mx.eval, []

    def evaluate(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(mx, "eval", evaluate)
    try:
        stream.gather("blk.0", mx.array([[0,1],[2,3]], mx.uint32), row_buffers=True)
        assert len(calls) == 1
        assert stream.stats["upload_batches"] == 1 and stream.stats["uploaded_experts"] == 4
    finally:
        stream.close()


@pytest.mark.parametrize("short", [False,True])
def test_expert_read_keeps_exact_bytes_through_short_reads_and_eof(tmp_path, monkeypatch, short):
    import os
    stream, _, path = fixture(tmp_path)
    original = os.pread
    if short:
        monkeypatch.setattr(os,"pread",lambda fd,n,off: original(fd,min(7,n),off))
    try:
        spans = ((0,37),(37,43),(80,51))
        actual = read_rows(stream.fd, [(3,spans)])
        raw = path.read_bytes()
        assert actual == [(3,tuple(raw[off:off+n] for off,n in spans))]
        with pytest.raises(OSError,match="short"):
            read_rows(stream.fd, [(3,((path.stat().st_size-1,5),))])
    finally:
        stream.close()


def test_direct_row_buffers_avoid_copies_and_survive_cache_eviction(tmp_path, monkeypatch):
    stream, arrays, _ = fixture(tmp_path)

    def forbid_copy(*args, **kwargs):
        raise AssertionError("direct expert rows must not concatenate weights")

    monkeypatch.setattr(mx, "concatenate", forbid_copy)
    try:
        ids, first = stream.gather("blk.0", mx.array([[3, 1]], mx.uint32), row_buffers=True)
        np.testing.assert_array_equal(np.array(ids), [[1, 0]])
        _, again = stream.gather("blk.0", mx.array([[1, 3]], mx.uint32), row_buffers=True)
        assert all(a.data is b.data for old, new in zip(first, again) for a, b in zip(old, new))
        stream.gather("blk.0", mx.array([[0, 2]], mx.uint32), row_buffers=True)
        assert stream.stats["evictions"] == 2
        for kind, plane in zip(("gate", "up", "down"), first):
            for expert, row in zip((1, 3), plane):
                np.testing.assert_array_equal(np.array(row.data), arrays[kind][expert])
        host_ids, wide = stream.gather("blk.0", mx.array([[0, 1], [2, 3]], mx.uint32),
                                      row_buffers=True, host_ids=True)
        assert host_ids.dtype == np.uint32 and host_ids.shape == (2, 2)
        for kind, plane in zip(("gate", "up", "down"), wide):
            for expert, row in enumerate(plane):
                np.testing.assert_array_equal(np.array(row.data), arrays[kind][expert])
        with pytest.raises(ValueError, match="host expert IDs"):
            stream.gather("blk.0", mx.array([[0, 1]], mx.uint32), host_ids=True)
    finally:
        stream.close()


def test_auto_admission_uses_adapter_measurement_for_native_payload(tmp_path):
    class Native:
        weight_streaming_modes = frozenset({"moe_experts"})
        resident_weight_bytes = staticmethod(lambda path: 100)

    result = auto_expert_streaming_decision(Native, tmp_path, admission_limit_bytes=50, explicit=False, headroom_bytes=1)
    assert result["selected"] and result["weight_bytes"] == 100 and result["fits"] is False
    Native.resident_weight_bytes = staticmethod(lambda path: True)
    result = auto_expert_streaming_decision(Native, tmp_path, admission_limit_bytes=50, explicit=False, headroom_bytes=1)
    assert not result["selected"] and result["reason"] == "artifact_unmeasured"


def test_streamed_copy_contract_preserves_other_route_refusals():
    from mlx2.adapters.deepseek_v41 import DeepSeekV41Adapter
    from mlx2.serving import ServingEngine

    engine = object.__new__(ServingEngine)
    engine.adapter_factory = DeepSeekV41Adapter
    engine.lane_matmul = "off"
    engine.sp_qmm_enabled = False
    from types import SimpleNamespace

    engine.int8_prefill_policy = SimpleNamespace(enabled=False)
    engine.verify_bitexact_policy = SimpleNamespace(enabled=False)
    engine.multi_lora_policy = engine.lora_root = None
    engine._refuse_unevidenced_weight_streaming({"draft_model": "prompt_lookup"})
    with pytest.raises(ValueError, match="external draft"):
        engine._refuse_unevidenced_weight_streaming({"draft_model": "foreign_learned"})
    engine.sp_qmm_enabled = True
    with pytest.raises(ValueError, match="sp_qmm"):
        engine._refuse_unevidenced_weight_streaming({"draft_model": "prompt_lookup"})


def test_unresolved_streamed_external_contract_is_rechecked_after_resolution():
    from types import SimpleNamespace

    from mlx2.serving import ServingEngine

    engine = object.__new__(ServingEngine)
    engine.lane_matmul = "off"
    engine.sp_qmm_enabled = False
    engine.int8_prefill_policy = SimpleNamespace(enabled=False)
    engine.verify_bitexact_policy = SimpleNamespace(enabled=False)
    engine.multi_lora_policy = engine.lora_root = None
    policy = {"draft_model": "foreign_learned"}
    engine._refuse_unevidenced_weight_streaming(policy, defer_external=True)
    engine.adapter_factory = type("Undeclared", (), {})
    with pytest.raises(ValueError, match="external draft"):
        engine._refuse_unevidenced_weight_streaming(policy)
