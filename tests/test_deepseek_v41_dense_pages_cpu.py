"""Synthetic immutable-page/native arithmetic/lifetime gates; CPU only."""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import os

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.immutable_weight_pages import ImmutablePage, ImmutableWeightPages, file_stamp, read_page
from mlx2.adapters.deepseek_v41_weights import DeepSeekV41Weights
from mlx2.runtime.gguf.weights import GGUFWeights
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from test_gguf_mlx_storage_cpu import fixture as gguf_fixture
from test_deepseek_v41_cpu import Weights, config

mx.set_default_device(mx.cpu)


@pytest.mark.parametrize("partial", [False, True])
def test_page_preadv_reads_exact_file_bytes_into_one_bounded_destination(tmp_path, monkeypatch, partial):
    if not hasattr(os, "preadv"):
        pytest.skip("host does not expose positional vector reads")
    path = tmp_path / "direct-page"
    content = bytes(range(256)) * 3
    path.write_bytes(content)
    original, calls = os.preadv, []
    def read_into(fd, buffers, offset):
        assert len(buffers) == 1
        view = buffers[0]
        assert isinstance(view, memoryview) and not view.readonly and view.c_contiguous
        assert 0 < view.nbytes <= 32
        calls.append((offset, view.nbytes, view.obj))
        return original(fd, [view[:7] if partial else view], offset)
    def forbidden(*args):
        raise AssertionError("available preadv must not allocate intermediate pread bytes")
    monkeypatch.setattr(os, "preadv", read_into)
    monkeypatch.setattr(os, "pread", forbidden)
    with path.open("rb") as source:
        actual = read_page(source.fileno(), 13, 117, 32)
    assert actual.dtype == np.uint8 and actual.flags.owndata
    assert actual.tobytes() == content[13:130]
    assert all(owner is actual for _, _, owner in calls)
    assert [offset for offset, _, _ in calls] == list(range(13, 130, 7 if partial else 32))


def test_page_preadv_eof_after_partial_file_read_is_not_published(tmp_path, monkeypatch):
    if not hasattr(os, "preadv"):
        pytest.skip("host does not expose positional vector reads")
    path = tmp_path / "short-page"
    path.write_bytes(b"short")
    def forbidden(*args):
        raise AssertionError("EOF must not switch to pread")
    monkeypatch.setattr(os, "pread", forbidden)
    with path.open("rb") as source, pytest.raises(OSError, match="short"):
        read_page(source.fileno(), 0, 12, 8)


def test_page_preadv_failure_after_partial_read_keeps_original_error(tmp_path, monkeypatch):
    if not hasattr(os, "preadv"):
        pytest.skip("host does not expose positional vector reads")
    path = tmp_path / "failed-page"
    path.write_bytes(bytes(range(64)))
    original, calls = os.preadv, []
    failure = OSError("injected positional read failure")
    def read_into(fd, buffers, offset):
        calls.append(offset)
        if len(calls) == 2:
            raise failure
        return original(fd, [buffers[0][:7]], offset)
    def forbidden(*args):
        raise AssertionError("read failure must not switch to pread")
    monkeypatch.setattr(os, "preadv", read_into)
    monkeypatch.setattr(os, "pread", forbidden)
    with path.open("rb") as source, pytest.raises(OSError) as caught:
        read_page(source.fileno(), 4, 25, 16)
    assert caught.value is failure and calls == [4, 11]


def test_page_falls_back_only_when_preadv_is_absent(tmp_path, monkeypatch):
    path = tmp_path / "fallback-page"
    content = bytes(range(64))
    path.write_bytes(content)
    original, calls = os.pread, []
    def read(fd, size, offset):
        calls.append((offset, size))
        assert 0 < size <= 16
        return original(fd, min(size, 5), offset)
    monkeypatch.delattr(os, "preadv", raising=False)
    monkeypatch.setattr(os, "pread", read)
    with path.open("rb") as source:
        actual = read_page(source.fileno(), 3, 29, 16)
        assert actual.tobytes() == content[3:32]
        with pytest.raises(OSError, match="short"):
            read_page(source.fileno(), 60, 8, 16)
    assert calls[:6] == [(3, 16), (8, 16), (13, 16), (18, 14), (23, 9), (28, 4)]


def store(tmp_path, *, cache=512):
    path=tmp_path/'pages';path.write_bytes(bytes(range(256))*4)
    pages={str(i):ImmutablePage(i*256,256,256,'raw256') for i in range(4)}
    owners={};fences=[]
    def materialize(name,raw):
        owners[name]=raw.tobytes();return owners[name]
    p=ImmutableWeightPages(path,pages,binding='synthetic@pages-v1',expected_stat=file_stamp(path.stat()),
        cache_bytes=cache,materialize=materialize,evict=lambda name:owners.pop(name),
        synchronize=lambda:fences.append(True),chunk_bytes=64)
    return p,owners,fences,path


def test_leases_protect_working_set_and_exact_bytes_survive_eviction(tmp_path):
    p,owners,fences,_=store(tmp_path)
    try:
        with p.window():
            assert p.acquire('0')==bytes(range(256))
            p.acquire('1')
            with pytest.raises(ValueError,match='owned'):p.resize(256)
            with pytest.raises(ValueError,match='owned'):p.park()
        with p.window():
            p.acquire('0');p.acquire('2')
            assert set(owners)=={'0','2'}
        assert len(fences)==2 and p.receipt()['leased_pages']==0
        p.resize(256)
        assert p.resident_bound_bytes==256
        with p.window():assert p.acquire('1')==bytes(range(256))
        assert p.stats['evictions']==3 and p.stats['read_bytes']==1024
    finally:p.close()
    assert not owners


def test_provider_refuses_cross_thread_and_unsized_active_set(tmp_path):
    p,_,_,_=store(tmp_path)
    try:
        with p.window():p.acquire('0')
        def foreign():
            with p.window():pass
        with ThreadPoolExecutor(1) as worker:
            with pytest.raises(ValueError,match='another evaluation'):worker.submit(foreign).result()
        with pytest.raises(ValueError,match='admitted budget'):
            with p.window():
                p.acquire('0');p.acquire('1');p.acquire('2')
        assert not p._active and not p._leased and p.receipt()['error']
        with pytest.raises(ValueError,match='failed'):
            with p.window():pass
    finally:p.close()


def test_source_replacement_and_park_activation_are_exact(tmp_path):
    p,owners,_,path=store(tmp_path)
    try:
        with p.window():p.acquire('3')
        p.park();assert p.receipt()['parked'] and not owners
        with pytest.raises(ValueError,match='parked'):
            with p.window():pass
        p.activate()
        with p.window():assert p.acquire('3')==bytes(range(256))
        p.park();path.write_bytes(path.read_bytes()+b'!')
        with pytest.raises(ValueError,match='changed'):p.activate()
        assert p.receipt()['parked']
    finally:p.close()


def test_failed_page_read_never_publishes_payload(tmp_path,monkeypatch):
    import mlx2.runtime.immutable_weight_pages as module
    p,owners,_,_=store(tmp_path)
    def broken(*args):raise OSError('injected read failure')
    monkeypatch.setattr(module,'read_page',broken)
    try:
        with pytest.raises(OSError,match='injected'):
            with p.window():p.acquire('0')
        assert not owners and p.resident_bound_bytes==0 and p.receipt()['error']
        p.park()
        with pytest.raises(ValueError,match='failed'):p.activate()
    finally:p.close()


def weights(tmp_path, *, paging=True):
    path=tmp_path/'weights.gguf';gguf_fixture(path)
    admitted=file_stamp(path.stat());admitted.pop('ctime_ns')  # Actual V4.1 artifact seal schema.
    w=DeepSeekV41Weights(path,expected_stat=admitted)
    w.expert_stream=object()  # Synthetic source has no experts; admission contract only.
    if paging:w.enable_dense_paging(168,binding='synthetic-native@paging-v1',synchronize=lambda:None)
    return w,path


def test_native_linear_transpose_law_and_alias_eviction(tmp_path):
    w,path=weights(tmp_path);resident=GGUFWeights(path)
    x=mx.array([[0.25,-1.5,2.25,-0.75]],mx.float32)
    try:
        for name in ('linear','bf16','linear','bf16'):
            control=resident.linear(name)(x);mx.eval(control)
            with w.dense_window():
                actual=w.linear(name)(x);w.evaluate_dense_window(actual)
            assert np.array(actual).tobytes()==np.array(control).tobytes()
            assert w.resident_bytes()<=w.dense_pages.cache_bytes
        with w.dense_window():
            packed=w.packed('q2');w.evaluate_dense_window(packed.data)
        assert not w._native_linear_fp32 and not w._dense
        assert np.array(packed.data).tobytes()==bytes(range(168))
        assert w.dense_pages.stats['evictions']>0
    finally:w.close();resident.close()


@pytest.mark.parametrize('paging',[False,True])
def test_weight_park_releases_owners_and_reloads_exact_native_payload(tmp_path,paging):
    w,_=weights(tmp_path,paging=paging)
    try:
        with w.dense_window():
            out=w.linear('linear')(mx.ones((1,4),mx.float32));w.evaluate_dense_window(out)
            before=np.array(out)
        assert w.resident_bytes()>0
        reserve=w.resume_reservation_bytes()
        w.park();assert not w._packed and not w._dense and not w._native_linear_fp32
        assert w.resident_bytes()==0 and w.reserved_bytes()==0 and w.resume_reservation_bytes()==reserve
        with pytest.raises(ValueError,match='parked'):w.array('linear')
        w.activate()
        with w.dense_window():
            out=w.linear('linear')(mx.ones((1,4),mx.float32));w.evaluate_dense_window(out)
            assert np.array(out).tobytes()==before.tobytes()
    finally:w.close()


def test_page_budget_and_preload_cannot_silently_expand_working_set(tmp_path):
    w,_=weights(tmp_path)
    try:
        assert w.minimum_cache_bytes==168
        assert w.dense_paging_plan(168)['staging_bound_bytes']==168+(16<<20)
        with pytest.raises(ValueError,match='stage'):w.dense_paging_plan(167)
        with pytest.raises(ValueError,match='evaluation window'):w.packed('q8')
        assert not w._packed and not w._dense
    finally:w.close()


def test_failed_completion_fence_keeps_page_owners_until_successful_cleanup(tmp_path):
    p,owners,_,_=store(tmp_path)
    def fail():raise RuntimeError('completion fence failed')
    p.synchronize=fail
    with pytest.raises(RuntimeError,match='completion fence'):
        with p.window():p.acquire('0')
    assert p._active and p._leased=={'0'} and '0' in owners
    with pytest.raises(RuntimeError,match='completion fence'):p.close()
    assert p._active and '0' in owners and not p._closed
    p.synchronize=lambda:None
    p.close()
    assert not owners and p._closed and not p._active


def test_body_failure_is_fenced_before_releasing_leases(tmp_path):
    p,owners,fences,_=store(tmp_path)
    try:
        with pytest.raises(RuntimeError,match='body failed'):
            with p.window():
                p.acquire('0')
                raise RuntimeError('body failed')
        assert fences==[True] and not p._active and not p._leased
        assert '0' in owners and p._error
    finally:p.close()


def test_source_change_after_materialization_retains_cleanup_ownership(tmp_path):
    p,owners,_,path=store(tmp_path)
    original=p.materialize
    def replaced(name,raw):
        value=original(name,raw)
        path.write_bytes(path.read_bytes()+b'!')
        return value
    p.materialize=replaced
    try:
        with pytest.raises(ValueError,match='source changed'):
            with p.window():p.acquire('0')
        assert '0' in owners and p.resident_bound_bytes==256 and p._error
        assert not p._active and not p._leased
    finally:p.close()
    assert not owners


def test_timed_out_host_read_keeps_staging_reservation_until_drained(tmp_path,monkeypatch):
    import threading
    import time
    import mlx2.runtime.immutable_weight_pages as module
    from mlx2.runtime.io_ownership import ReadPool,ReadTimeout,storage_owner
    p,_,_,path=store(tmp_path)
    release=threading.Event();started=threading.Event()
    original=module.read_page
    def blocked(*args):
        started.set()
        assert release.wait(3)
        return original(*args)
    monkeypatch.setattr(module,'read_page',blocked)
    p.pool.close()
    p.pool=ReadPool(1,'synthetic-page-timeout',timeout=.03,max_pending=1,
                    owner=storage_owner('immutable-weights',path))
    try:
        with pytest.raises(ReadTimeout):
            with p.window():p.acquire('0')
        assert started.is_set() and not p._active
        assert p.receipt()['read_pool']['weight_read_inflight']==1
        with pytest.raises(ValueError,match='still owns staging'):p.park()
        assert not p._parked
        release.set()
        deadline=time.monotonic()+2
        while p.pool.counters()['weight_read_inflight'] and time.monotonic()<deadline:time.sleep(.005)
        assert p.pool.counters()['weight_read_inflight']==0
        p.park();assert p._parked
    finally:
        release.set();p.close()


def test_model_stage_boundaries_and_rebuilt_anchors_preserve_state():
    c=config();w=Weights(c);model=DeepSeekV41Model(c,w,identity='synthetic@dense-boundaries')
    events=[]
    @contextmanager
    def window():
        events.append('begin')
        yield
        assert events[-1]=='evaluated'
        events.append('end')
    w.dense_window=window
    def evaluated(*values):events.append('evaluated')
    w.evaluate_dense_window=evaluated
    cache=model.make_cache();model(mx.array([[1,2,3]],mx.uint32),cache)
    assert events.count('begin')==c.num_hidden_layers+2
    saved=[s.snapshot() for s in cache]
    expected=model(mx.array([[4]],mx.uint32),cache);mx.eval(expected)
    for s,snapshot in zip(cache,saved):s.restore(snapshot)
    model.release_weight_references();assert not model.layers and model.norm_weight is None
    assert not model.parameters()['hc'] and not model.parameters()['rotary']
    with pytest.raises(ValueError,match='parked'):model(mx.array([[4]],mx.uint32),cache)
    w.tensors={};w.preload_native_linear_fp32=lambda:None
    model.restore_weight_references()
    actual=model(mx.array([[4]],mx.uint32),cache);mx.eval(actual)
    assert np.array(actual).tobytes()==np.array(expected).tobytes()
    model._weight_executions=1
    with pytest.raises(ValueError,match='active'):model.release_weight_references()
    model._weight_executions=0
