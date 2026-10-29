"""Synthetic CPU ownership boundaries; no model load, GPU, or qualification."""

import os
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_expert_stream import PackedExpertStream, read_rows
from mlx2.runtime.expert_stream_atlas import ExpertStreamAtlas
from mlx2.runtime.io_ownership import CloseAction, ReadPool, ReadTimeout, release_worker
from mlx2.runtime.models.deepseek_v41_engram import EngramRows, file_stamp
from test_deepseek_v41_expert_stream_cpu import fixture

mx.set_default_device(mx.cpu)


def ple(tmp_path, **kwargs):
    raw=np.zeros((8,264),np.uint8)
    raw[:,:256]=np.arange(8)[:,None]+56
    raw[:,256:]=127
    path=tmp_path/"ple.bin";path.write_bytes(raw.tobytes())
    return EngramRows(path,offset=0,rows=8,expected_stat=file_stamp(path.stat()),**kwargs),path


def test_expert_park_activate_keeps_source_binding_and_refuses_acquired_ticket(tmp_path):
    stream,arrays,path=fixture(tmp_path)
    try:
        reservation=stream.resume_reservation_bytes()
        stream.gather("blk.0",mx.array([[0,1]],mx.uint32),row_buffers=True)
        for _ in range(2):
            ticket=stream.prepare("blk.0",mx.array([[0,1]],mx.uint32))
            with pytest.raises(ValueError,match="acquired"):stream.park()
            stream.cancel(ticket)
            receipt=stream.park();stream.park()
            assert receipt["parked"] and stream.live_bytes()==0 and stream.reserved_bytes()==0
            assert stream.resume_reservation_bytes()==reservation
            os.fstat(stream.fd)
            with pytest.raises(ValueError,match="parked"):stream.gather("blk.0",mx.array([[0,1]],mx.uint32))
            stream.activate();stream.activate()
            _,planes=stream.gather("blk.0",mx.array([[0,1]],mx.uint32),row_buffers=True)
            for kind,plane in zip(("gate","up","down"),planes):
                for expert,p in enumerate(plane):assert np.array(p.data).tobytes()==arrays[kind][expert].tobytes()
        assert stream.stats["parks"]==stream.stats["activations"]==2
        stream.park()
        replacement=path.with_name("replacement");replacement.write_bytes(path.read_bytes());replacement.replace(path)
        with pytest.raises(ValueError,match="source changed"):stream.activate()
        assert stream.receipt()["parked"]
    finally:stream.close()


def test_ple_park_activate_refuses_owned_window_and_revalidates_path(tmp_path):
    table,path=ple(tmp_path,cache_bytes=2048,cache_format="bf16_lut")
    ids=np.array([0,1,2,3],np.uint32)
    try:
        expected=table.read(ids)
        for _ in range(2):
            table.prefetch(ids)
            with pytest.raises(ValueError,match="owned"):table.park()
            table.read(ids)
            table.park();table.park()
            assert table.diagnostics()["cached_bytes"]==0 and table.reserved_bytes()==0
            assert table.resume_reservation_bytes()==2048
            os.fstat(table.fd)
            with pytest.raises(ValueError,match="parked"):table.read(ids)
            table.activate();table.activate()
            assert table.read(ids).tobytes()==expected.tobytes()
        assert table.stats["parks"]==table.stats["activations"]==2
        table.park()
        with path.open("ab") as target:target.write(b"changed")
        with pytest.raises(ValueError,match="source changed"):table.activate()
        assert table.diagnostics()["parked"]
    finally:table.close()


def test_required_read_failure_cannot_be_bypassed_by_cached_hits_or_activation(tmp_path):
    stream,_,_=fixture(tmp_path);table,_=ple(tmp_path)
    try:
        stream.gather("blk.0",mx.array([[0,1]],mx.uint32),row_buffers=True)
        table.read(np.array([0,1],np.uint32))
        stream.pool.fail(ReadTimeout("injected required read failure"))
        table.pool.fail(ReadTimeout("injected required read failure"))
        with pytest.raises(ReadTimeout,match="reload required"):stream.gather("blk.0",mx.array([[0,1]],mx.uint32))
        with pytest.raises(ReadTimeout,match="reload required"):table.read(np.array([0,1],np.uint32))
        stream.park();table.park()
        with pytest.raises(ReadTimeout,match="reload required"):stream.activate()
        with pytest.raises(ReadTimeout,match="reload required"):table.activate()
    finally:stream.close();table.close()


def test_ticket_hints_still_count_against_prefetch_budget(tmp_path):
    old,_,path=fixture(tmp_path)
    tensors={t.name:t for t in old.tables["blk.0"]};old.close()
    tensors.update({t.name.replace("blk.0","blk.1"):replace(t,name=t.name.replace("blk.0","blk.1")) for t in tuple(tensors.values())})
    atlas=ExpertStreamAtlas("synthetic",layers=2,experts=4,top_k=2)
    keys=atlas.keys([7],[-1,-1,-1])
    atlas.observe(1,keys,np.array([[1,2]],np.uint32),previous=np.array([[0,1]],np.uint32))
    stream=PackedExpertStream(path,tensors=tensors,data_offset=0,expected_stat=file_stamp(path.stat()),
                              ceiling_bytes=4*3*256*50,top_k=2,atlas=atlas)
    try:
        stream.hint_budget=2*stream.row_bytes
        for prefix,expert in (("blk.0",0),("blk.1",3)):
            future=stream.hint_pool.submit(read_rows,stream.hint_fd,stream._entries(prefix,[expert]))
            future.result(timeout=5);stream._hints[(prefix,expert)]=future
        stream._keys=keys
        ticket=stream.prepare("blk.0",mx.array([[0,1]],mx.uint32))
        assert len(ticket.hints)==1 and len(stream._hints)==1
        assert stream.hint_owned_bytes()==stream.hint_budget
        assert stream.stats["hint_issued"]==0
        stream.resolve(ticket,row_buffers=True)
        assert stream.hint_owned_bytes()==stream.row_bytes
    finally:stream.close()


def test_park_timeout_keeps_pending_hint_and_descriptor_owned(tmp_path):
    old,_,path=fixture(tmp_path);tensors={t.name:t for t in old.tables["blk.0"]};old.close()
    atlas=ExpertStreamAtlas("synthetic",layers=1,experts=4,top_k=2)
    stream=PackedExpertStream(path,tensors=tensors,data_offset=0,expected_stat=file_stamp(path.stat()),
                              ceiling_bytes=2*3*256*50,top_k=2,atlas=atlas)
    started,release=threading.Event(),threading.Event()
    def read():
        started.set();release.wait(3)
        return read_rows(stream.hint_fd,stream._entries("blk.0",[0]))
    future=stream.hint_pool.submit(read);stream._hints[("blk.0",0)]=future
    try:
        assert started.wait(1)
        with pytest.raises(ReadTimeout,match="did not drain"):stream.park(timeout=0.01)
        assert not stream.receipt()["parked"] and stream.hint_owned_bytes()==stream.row_bytes
        os.fstat(stream.hint_fd)
        assert stream.hint_pool.counters()["weight_read_inflight"]==1
    finally:
        release.set()
        try:future.result(timeout=3)
        except ReadTimeout:pass
        stream.close()


def test_failed_required_read_cannot_be_hidden_by_parking(tmp_path):
    stream,_,_=fixture(tmp_path);table,_=ple(tmp_path)
    release=threading.Event();started=[threading.Event(),threading.Event()]
    def blocked(event):
        event.set();release.wait(3)
        return b"revoked"
    futures=[owner.pool.submit(blocked,event) for owner,event in zip((stream,table),started)]
    try:
        assert all(event.wait(1) for event in started)
        for owner in (stream,table):
            owner.pool.fail(ReadTimeout("injected timeout"))
            with pytest.raises(ValueError,match="remain owned"):owner.park()
            assert not owner._parked
            os.fstat(owner.fd)
    finally:
        release.set()
        for future in futures:
            try:future.result(timeout=3)
            except ReadTimeout:pass
        stream.close();table.close()


def test_cancelled_ticket_retains_running_optional_hint_reservation(tmp_path,monkeypatch):
    old,_,path=fixture(tmp_path);tensors={t.name:t for t in old.tables["blk.0"]};old.close()
    atlas=ExpertStreamAtlas("synthetic",layers=1,experts=4,top_k=2)
    stream=PackedExpertStream(path,tensors=tensors,data_offset=0,expected_stat=file_stamp(path.stat()),
                              ceiling_bytes=2*3*256*50,top_k=2,atlas=atlas)
    release,started=threading.Event(),threading.Event()
    def blocked():
        started.set();release.wait(3)
        return read_rows(stream.hint_fd,stream._entries("blk.0",[0]))
    future=stream.hint_pool.submit(blocked)
    original_result=future.result
    try:
        assert started.wait(1)
        stream._hints[("blk.0",0)]=future
        ticket=stream.prepare("blk.0",mx.array([[0]],mx.uint32))
        monkeypatch.setattr(future,"result",lambda **kwargs:(_ for _ in ()).throw(FutureTimeout()))
        stream.cancel(ticket)
        assert ticket.closed and stream._ticket is None
        assert stream.hint_owned_bytes()==stream.row_bytes
        assert stream.hint_pool.failure is not None
        with pytest.raises(ReadTimeout,match="did not drain"):stream.park(timeout=.01)
        assert not stream._parked
    finally:
        release.set()
        try:original_result(timeout=3)
        except ReadTimeout:pass
        stream.close()


def test_atlas_rejects_invalid_layer_keys_and_previous_before_mutating():
    atlas=ExpertStreamAtlas("synthetic",layers=2,experts=4,top_k=2)
    keys=atlas.keys([7],[-1,-1,-1]);routes=np.array([[1,2]],np.uint32)
    for layer,used_keys,previous in ((-1,keys,None),(2,keys,None),(1,("bad",),None),
                                    (1,keys,np.array([[999,0]],np.uint32)),(1,keys,np.array([[0,1],[1,2]],np.uint32))):
        with pytest.raises(ValueError):atlas.observe(layer,used_keys,routes,previous=previous)
    assert atlas.stats["observations"]==0 and all(not layer for layer in atlas.tokens+atlas.couples)
    with pytest.raises(ValueError):atlas.predict(0,keys,limit=-1)


def test_dense_pager_is_accounted_once_and_participates_in_elastic_budget(tmp_path):
    class Dense:
        minimum_cache_bytes=10
        maximum_cache_bytes=100
        cache_budget_bytes=80
        def reserved_bytes(self):return self.resume_reservation_bytes()
        def resume_reservation_bytes(self):return 20+self.cache_budget_bytes
        def resident_bytes(self):return 10+min(40,self.cache_budget_bytes)
        def resize_cache_budget(self,budget):self.cache_budget_bytes=budget;return budget
    stream,_,_=fixture(tmp_path);dense=Dense()
    try:
        baseline=stream.reserved_bytes()
        stream.bind_dense_provider(dense)
        assert stream.reserved_bytes()==baseline+100 and stream.live_bytes()==50
        assert stream.unfilled_reserve_bytes()==baseline+50
        stream.enable_elastic({})
        assert stream.elastic.resources["dense"].budget_bytes==80
        stream.bind_execution_synchronizer(lambda:None)
        stream.elastic.set_claim("apcv2_restore",75)
        assert dense.cache_budget_bytes==25
        assert stream.reserved_bytes()==baseline+45 and stream.live_bytes()==35
    finally:stream.close()


def test_fixed_ple_accounting_survives_elastic_binding_without_double_count(tmp_path):
    stream,_,_=fixture(tmp_path);table,_=ple(tmp_path,cache_bytes=2048,cache_format="bf16_lut")
    try:
        baseline=stream.reserved_bytes()
        with pytest.raises(ValueError,match="distinct ownership"):stream.bind_ple_providers({0:table,1:table})
        stream.bind_ple_providers({1:table});stream.bind_ple_providers({1:table})
        table.read(np.array([0,1],np.uint32))
        assert stream.reserved_bytes()==baseline+2048
        assert stream.resume_reservation_bytes()==baseline+2048
        assert stream.live_bytes()==1024 and stream.unfilled_reserve_bytes()==baseline+1024
        with pytest.raises(ValueError,match="already bound"):stream.bind_ple_providers({})
        stream.enable_elastic({1:table})
        assert stream.reserved_bytes()==baseline+2048 and stream.live_bytes()==1024
        stream.park()
        assert stream.live_bytes()==1024 and stream.reserved_bytes()==2048
        table.park()
        assert stream.live_bytes()==stream.reserved_bytes()==0
        assert stream.resume_reservation_bytes()==baseline+2048
    finally:stream.close();table.close()


def test_read_pool_retirement_requires_registered_successful_finalizer():
    pool=ReadPool(1,"retirement-empty",owner=object())
    assert not pool.retirement_receipt()["ownership_drained"]
    assert pool.close()
    assert not pool.retirement_receipt()["ownership_drained"]
    calls=[]
    assert pool.close(CloseAction(lambda:calls.append("closed")))
    proof=pool.retirement_receipt()
    assert proof["ownership_drained"] and proof["finalizer_succeeded"] and calls==["closed"]


def test_read_pool_retirement_retains_timed_out_worker_and_late_finalizer():
    pool=ReadPool(1,"retirement-worker",owner=object())
    started,release,closed=threading.Event(),threading.Event(),threading.Event()
    def blocked():started.set();release.wait(3);return b"late allocation"
    future=pool.submit(blocked)
    try:
        assert started.wait(1)
        assert not pool.close(CloseAction(closed.set))
        proof=pool.retirement_receipt()
        assert not proof["ownership_drained"] and proof["pending_reads"]==1 and proof["quarantined"]
        release.set()
        with pytest.raises(ReadTimeout):future.result(timeout=3)
        assert closed.wait(3)
        pool._finalizer.result(timeout=3)
        assert pool.retirement_receipt()["ownership_drained"]
    finally:release.set()


def test_read_pool_retirement_retains_failed_completed_finalizer():
    pool=ReadPool(1,"retirement-failed-finalizer",owner=object())
    def failed():raise OSError("injected uncertain descriptor close")
    try:
        assert not pool.close(CloseAction(failed))
        proof=pool.retirement_receipt()
        assert proof["closed"] and proof["finalizer_complete"] and proof["pending_reads"]==0
        assert proof["quarantined"] and not proof["finalizer_succeeded"] and not proof["ownership_drained"]
    finally:
        # Synthetic finalizer owns no descriptor; do not leak its intentional
        # uncertainty into the process-global quarantine of later test cases.
        release_worker(pool)


def test_read_pool_retirement_waits_for_late_successful_descriptor_close():
    pool=ReadPool(1,"retirement-late-close",owner=object())
    entered,release=threading.Event(),threading.Event()
    def blocked_close():entered.set();release.wait(3)
    try:
        assert not pool.close(CloseAction(blocked_close),timeout=.01)
        assert entered.is_set()
        proof=pool.retirement_receipt()
        assert proof["pending_reads"]==0 and not proof["finalizer_complete"] and not proof["ownership_drained"]
        release.set();pool._finalizer.result(timeout=3)
        assert pool.retirement_receipt()["ownership_drained"]
    finally:release.set()


@pytest.mark.parametrize("bad",[{"workers":True},{"max_rows":1.5},{"cache_bytes":-1},{"timeout":False},{"timeout":float("nan")}])
def test_ple_invalid_bounds_fail_before_file_open(bad):
    with pytest.raises(ValueError,match="bounds"):
        EngramRows("does-not-exist",offset=0,rows=8,expected_stat={},**bad)
