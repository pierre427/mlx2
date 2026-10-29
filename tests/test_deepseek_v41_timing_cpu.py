"""Nested CPU attribution must subtract children just once."""

import importlib
from pathlib import Path


def test_nested_calling_thread_cpu_is_additive(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts/research"))
    module=importlib.import_module("analyze_deepseek_v41_10token")
    class Array:
        pass
    class Backend:
        array=Array
        @staticmethod
        def eval(*args):pass
    clock=iter((0.0,1.0,2.0,3.0))
    monkeypatch.setattr(module.time,"thread_time",lambda:next(clock))
    profiler=module.CPUWallProfiler(Backend)
    profiler.call("parent","parent",lambda:profiler.call("child","child",lambda:42,(),{}),(),{})
    groups=profiler.summary()["categories"]
    assert groups["parent"]["exclusive_calling_thread_cpu_seconds"]==2.0
    assert groups["child"]["exclusive_calling_thread_cpu_seconds"]==1.0


def test_dense_page_profile_separates_io_materialization_and_fences(tmp_path,monkeypatch):
    import os
    import threading
    import numpy as np
    import mlx.core as mx
    from mlx2.runtime.immutable_weight_pages import ImmutablePage,ImmutableWeightPages,file_stamp
    mx.set_default_device(mx.cpu)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts/research"))
    module=importlib.import_module("analyze_deepseek_v41_10token")
    source=tmp_path/"weights.bin";source.write_bytes(bytes(range(32)))
    owner=threading.get_ident();calls=[]
    def materialize(name,raw):
        calls.append(threading.get_ident());value=mx.array(raw);mx.eval(value);return value
    pages=ImmutableWeightPages(source,{"a":ImmutablePage(0,16,16,"u8"),"b":ImmutablePage(16,16,16,"u8")},
        binding="synthetic",expected_stat=file_stamp(os.stat(source)),cache_bytes=16,
        materialize=materialize,evict=lambda name:None,synchronize=mx.synchronize)
    try:
        with pages.window():
            reference=np.array(pages.acquire("a"))
        pages.park();pages.activate()
        profiler=module.CPUWallProfiler(mx)
        with monkeypatch.context() as patch:
            module.core.patch_dense_pages(pages,profiler,patch.setattr)
            with pages.window():actual=np.array(pages.acquire("a"))
            with pages.window():pages.acquire("b")
        assert actual.tobytes()==reference.tobytes()
        categories=profiler.summary()["categories"]
        assert categories["dense_page_read"]["calls"]==2
        assert categories["dense_page_materialize"]["calls"]==2
        assert categories["dense_page_fence"]["calls"]==2
        assert categories["dense_page_evict"]["calls"]==1
        assert all(thread==owner for thread in calls)
        count=len(profiler.events)
        with pages.window():pages.acquire("a")
        assert len(profiler.events)==count
    finally:pages.close()
