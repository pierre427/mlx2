"""Exact compact PLE, predictive hints, owned reads and row plans (CPU only)."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.expert_stream_atlas import ExpertStreamAtlas
from mlx2.runtime.models.deepseek_v41_engram import EngramRows, decode_engram_bf16, decode_engram_rows, file_stamp
from mlx2.runtime.gguf.v41_row_exact import plan
from mlx2.adapters.deepseek_v41_expert_stream import read_coalesced_rows, read_rows
from mlx2.adapters.deepseek_v41_expert_stream import PackedExpertStream
from test_deepseek_v41_expert_stream_cpu import fixture

mx.set_default_device(mx.cpu)


def test_optional_atlas_unreadable_and_unsavable_does_not_break_demand(tmp_path, monkeypatch):
    path = tmp_path/"atlas.json"
    atlas = ExpertStreamAtlas("source", layers=1, experts=4, top_k=2, path=path)
    def denied(*args, **kwargs):
        raise PermissionError("unreadable optional hint store")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "exists", denied)
        atlas.load()
    assert atlas.stats["ignored"] == 1 and not atlas.predict(0, ())
    with monkeypatch.context() as patch:
        patch.setattr(Path, "mkdir", denied)
        assert atlas.save() is False
    assert atlas.stats["save_failed"] == 1
    assert atlas.save() is True


def test_cancelled_read_releases_ticket_and_allows_next_exact_gather(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    stream, arrays, _ = fixture(tmp_path)
    try:
        ticket = stream.prepare("blk.0", mx.array([[1,3]], mx.uint32))
        with ThreadPoolExecutor(1) as worker:
            with pytest.raises(ValueError, match="foreign"):
                worker.submit(stream.resolve, ticket, row_buffers=True).result()
        assert stream._ticket is ticket and not ticket.closed
        stream.cancel(ticket)
        stream.cancel(ticket)
        assert ticket.closed and not ticket.futures and not ticket.found and stream._ticket is None
        assert stream.live_bytes() == 0
        remap, planes = stream.gather("blk.0", mx.array([[3,1]], mx.uint32), row_buffers=True, host_ids=True)
        np.testing.assert_array_equal(remap, [[1,0]])
        for kind, plane in zip(("gate","up","down"), planes):
            for expert, matrix in zip((1,3), plane):
                assert np.array(matrix.data).tobytes() == arrays[kind][expert].tobytes()
    finally:
        stream.close()


def test_shared_failure_cancels_owned_expert_reads(tmp_path):
    from types import SimpleNamespace
    from mlx2.runtime.models.deepseek_v41 import Block
    stream, _, _ = fixture(tmp_path)
    block = object.__new__(Block)
    block.c = SimpleNamespace(num_experts_per_tok=2, hidden_size=256)
    block.w = SimpleNamespace(expert_stream=stream, expert_io_policy="overlap")
    block.prefix = "blk.0."
    block.attention_half = lambda *args: (None, mx.zeros((1,1,256)), None, None, None)
    block.router = lambda x: (mx.array([[[1,3]]],mx.uint32), mx.ones((1,1,2)))
    def fail(x):
        raise RuntimeError("shared projection failed")
    block.shared_expert = fail
    try:
        with pytest.raises(RuntimeError, match="shared projection"):
            block(None, None, None, None, None)
        assert stream._ticket is None and stream.live_bytes() == 0
        stream.gather("blk.0", mx.array([[1,3]], mx.uint32), row_buffers=True)
    finally:
        stream.close()


@pytest.mark.parametrize("bad", [{"workers":0},{"workers":True},{"workers":-1},{"ceiling_bytes":1.5},{"top_k":0}])
def test_stream_rejects_invalid_bounds_before_open(bad):
    values=dict(tensors={}, data_offset=0, expected_stat={}, ceiling_bytes=100, workers=8, top_k=6)
    values.update(bad)
    with pytest.raises(ValueError,match="bounds"):
        PackedExpertStream("does-not-exist", **values)


def test_compact_format_matches_reference_bytes_and_rejects_nonfinite():
    # Every code across all scales where every finite E4M3 value stays finite.
    raw = np.zeros((246, 264), np.uint8)
    raw[:, :256] = np.arange(256, dtype=np.uint8)
    raw[:, (127,255)] = 0
    raw[:, 256:] = np.arange(246, dtype=np.uint8)[:, None]
    reference = decode_engram_rows(raw)
    actual = (decode_engram_bf16(raw).astype(np.uint32) << 16).view(np.float32)
    assert actual.tobytes() == reference.tobytes()
    for code, scale in ((127,127),(255,127),(0,255),(126,254)):
        row = np.full((1,264), code, np.uint8);row[:,256:]=scale
        with pytest.raises(ValueError,match="nonfinite"):decode_engram_rows(row)
        with pytest.raises(ValueError,match="nonfinite"):decode_engram_bf16(row)


@pytest.mark.parametrize("compact", [False,True])
def test_engram_hit_protection_does_not_reinsert_evicted_hits(tmp_path, compact):
    raw = np.zeros((12,264),np.uint8);raw[:,:256]=np.arange(12)[:,None];raw[:,256:]=127
    path=tmp_path/"ple";path.write_bytes(raw.tobytes())
    reader=EngramRows(path,offset=0,rows=12,expected_stat=file_stamp(path.stat()),
                      cache_bytes=(512 if compact else 1024)*8,cache_format="bf16_lut" if compact else "fp32")
    try:
        ids=np.arange(12,dtype=np.uint32)
        expected=decode_engram_rows(raw)
        for _ in range(3): assert reader.read(ids).tobytes()==expected.tobytes()
        assert reader.stats["cache_misses"]==20 and reader.stats["evictions"]==12
        assert reader.diagnostics()["cached_bytes"]==(512 if compact else 1024)*8
    finally:reader.close()


def test_atlas_is_source_bound_bounded_and_never_changes_route_ids(tmp_path):
    path=tmp_path/"atlas.json"
    atlas=ExpertStreamAtlas("source:recipe",layers=2,experts=4,top_k=2,path=path,contexts=2)
    keys=atlas.keys([1,2,3],[-1,-1,-1]);ids=np.array([[0,2],[1,3],[2,3]],np.uint32)
    original=ids.copy()
    atlas.observe(0,keys,ids);atlas.observe(1,keys,ids[:,::-1],previous=ids)
    assert len(atlas.tokens[0])==2
    assert set(atlas.predict(0,keys[-1:],limit=2))=={2,3}
    assert atlas.predict(1,(),previous=np.array([[0,2]]))
    assert ids.tobytes()==original.tobytes()
    atlas.save()
    loaded=ExpertStreamAtlas("source:recipe",layers=2,experts=4,top_k=2,path=path,contexts=2)
    assert loaded.stats["loaded"]==1 and loaded.predict(0,keys[-1:])==atlas.predict(0,keys[-1:])
    foreign=ExpertStreamAtlas("other",layers=2,experts=4,top_k=2,path=path,contexts=2)
    assert foreign.stats["ignored"]==1 and not foreign.predict(0,keys)
    data=json.loads(path.read_text());data["tokens"][0][0][1]=[[999,1]];path.write_text(json.dumps(data))
    malformed=ExpertStreamAtlas("source:recipe",layers=2,experts=4,top_k=2,path=path,contexts=2)
    assert malformed.stats["ignored"]==1


def test_coalesced_exact_ranges_and_read_ticket_lifetime(tmp_path,monkeypatch):
    stream,arrays,path=fixture(tmp_path)
    try:
        entries=stream._entries("blk.0",[0,1,2,3])
        actual=read_coalesced_rows(stream.fd,entries)
        assert actual==read_rows(stream.fd,entries)
        ticket=stream.prepare("blk.0",mx.array([[3,1]],mx.uint32))
        assert ticket.futures and stream.live_bytes()==0
        with pytest.raises(ValueError,match="already owned"):stream.prepare("blk.0",mx.array([[0]],mx.uint32))
        ids,planes=stream.resolve(ticket,row_buffers=True,host_ids=True)
        np.testing.assert_array_equal(ids,[[1,0]])
        for kind,plane in zip(("gate","up","down"),planes):
            for expert,row in zip((1,3),plane):assert np.array(row.data).tobytes()==arrays[kind][expert].tobytes()
        with pytest.raises(ValueError,match="closed"):stream.resolve(ticket)
    finally:stream.close()


@pytest.mark.parametrize("reuse",[1,2,4])
def test_row_exact_groups_recover_original_slot_order(reuse):
    rng=np.random.default_rng(417)
    for tokens in (1,2,6,10,15,1024):
        ids=rng.integers(0,384,(tokens,6),np.uint32)
        inverse,groups=plan(ids,384,reuse)
        order=np.concatenate([g[2] for g in groups])
        np.testing.assert_array_equal(order[inverse],np.arange(ids.size))
        for begin,end,order,tiles in groups:
            assert 1<=end-begin<=6
            covered=[]
            for first,expert,count in tiles:
                assert 1<=count<=reuse
                selected=order[first:first+count]
                assert np.all(ids.ravel()[selected]==begin+expert)
                covered.extend(range(int(first),int(first+count)))
            assert covered==list(range(len(order)))


def test_hint_consumption_and_failed_prediction_fall_back_to_exact_demand(tmp_path):
    from concurrent.futures import Future
    old,arrays,path=fixture(tmp_path)
    tensors={t.name:t for t in old.tables["blk.0"]};old.close()
    atlas=ExpertStreamAtlas("synthetic",layers=1,experts=4,top_k=2)
    key=atlas.keys([7],[-1,-1,-1]);routes=np.array([[1,3]],np.uint32)
    stream=PackedExpertStream(path,tensors=tensors,data_offset=0,expected_stat=file_stamp(path.stat()),
                              ceiling_bytes=2*3*256*50,top_k=2,atlas=atlas,coalesced=True)
    try:
        for failure in (False,True):
            atlas.observe(0,key,routes);stream.caches["blk.0"].clear()
            stream.begin_context(key)
            for future in stream._hints.values():future.result(timeout=10)
            if failure:
                failed=Future();failed.set_exception(OSError("synthetic failed hint"))
                stream._hints[("blk.0",1)]=failed
            _,planes=stream.gather("blk.0",mx.array(routes),row_buffers=True)
            for kind,plane in zip(("gate","up","down"),planes):
                for expert,row in zip((1,3),plane):assert np.array(row.data).tobytes()==arrays[kind][expert].tobytes()
            stream.end_context()
        assert stream.stats["hint_used"]==3 and stream.stats["hint_failed"]==1
        assert stream.stats["bytes_read"]==stream.row_bytes
    finally:stream.close()


def test_joint_residency_lends_only_at_safe_boundaries(tmp_path):
    stream,_,_=fixture(tmp_path)
    raw=np.zeros((4,264),np.uint8);raw[:,256:]=127
    path=tmp_path/"ple";path.write_bytes(raw.tobytes())
    ple=EngramRows(path,offset=0,rows=4,expected_stat=file_stamp(path.stat()),cache_bytes=2048,cache_format="bf16_lut")
    try:
        stream.enable_elastic({1:ple});sync=[]
        stream.bind_execution_synchronizer(lambda:sync.append(True))
        ple.read(np.array([0,1,2,3],np.uint32))
        stream.elastic.set_claim("apcv2_workspace",1024)
        assert ple.cache_rows==2 and sync==[True]
        assert stream.elastic.receipt()["claims"]=={"apcv2_workspace":1024}
        ticket=stream.prepare("blk.0",mx.array([[0,1]],mx.uint32))
        with pytest.raises(ValueError,match="acquired"):stream._resize_cache_budget(stream.ceiling)
        stream.resolve(ticket,row_buffers=True)
        ple.prefetch(np.array([0],np.uint32))
        with pytest.raises(ValueError,match="owned"):ple.resize_cache_budget(0)
        ple.read(np.array([0],np.uint32))
        stream.elastic.set_claim("apcv2_workspace",0)
        assert ple.cache_rows==4
    finally:stream.close();ple.close()
