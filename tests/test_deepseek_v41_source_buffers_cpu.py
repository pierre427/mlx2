"""Synthetic exact storage/ownership oracles; no model or GPU execution."""
from dataclasses import replace
import gc
import hashlib
import json
import os
import weakref

import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_source_buffers as buffers
from mlx2.runtime.immutable_weight_pages import file_stamp


def original_file(tmp_path):
    layout = buffers.canonical_layout("original")
    path = tmp_path / "expert.safetensors"
    rows = {}
    with path.open("wb") as stream:
        stream.write(b"header00")
        for i, t in enumerate(reversed(layout.tensors)):
            part, field = t.name.split(".")
            source = {"gate": "w1", "up": "w3", "down": "w2"}[part]
            key = f"layers.0.ffn.experts.0.{source}.{'weight' if field == 'qweight' else 'scale'}"
            rows[key] = dict(file=path.name, offset=stream.tell(), size=t.nbytes,
                             dtype="I8" if field == "qweight" else "F8_E8M0",
                             shape=(t.shape[0], t.shape[1]*4) if field == "qweight" else t.shape)
            block = bytes([i+1]) * (1 << 16)
            left = t.nbytes
            while left:
                count = min(left, len(block))
                stream.write(block[:count])
                left -= count
    return dict(root=tmp_path, tensors=rows, seals={path.name: file_stamp(path.stat())}), path


def q1_host(*, header=904, reverse=False):
    canonical = buffers.canonical_layout("q1")
    ordered = tuple(reversed(canonical.tensors)) if reverse else canonical.tensors
    cursor, rows = 0, []
    for t in ordered:
        rows.append(replace(t, offset=cursor))
        cursor += t.nbytes
    layout = buffers.ExpertLayout("q1", tuple(rows))
    raw = np.empty(header + layout.nbytes, np.uint8)
    raw[:header] = 91
    for i, t in enumerate(layout.tensors):
        raw[header+t.offset:header+t.offset+t.nbytes] = i+1
    host = buffers.HostExpert(raw[header:], layout)
    return raw, host.views()


@pytest.mark.parametrize("kind,size,count", [("original", 18_800_640, 6), ("q1", 6_635_520, 9)])
def test_exact_layout(kind, size, count):
    layout = buffers.canonical_layout(kind)
    assert layout.nbytes == size and len(layout.tensors) == count
    assert sum(t.nbytes for t in layout.tensors) == size
    for t in layout.tensors:
        assert t.offset % np.dtype(t.dtype).itemsize == 0


@pytest.mark.parametrize("change", [dict(offset=1), dict(offset=-4), dict(offset=True),
                                      dict(dtype=">u4"), dict(shape=(2304, 641)), dict(name="other")])
def test_bad_layout_refused(change):
    layout = buffers.canonical_layout("original")
    with pytest.raises(ValueError):
        buffers.ExpertLayout("original", (replace(layout.tensors[0], **change),) + layout.tensors[1:])


def test_direct_original_read_uses_one_destination_owner_and_partial_reads(tmp_path, monkeypatch):
    artifact, _ = original_file(tmp_path)
    owners, calls = set(), []
    allocations, empty = [], np.empty
    def allocate(*args, **kwargs):
        a = empty(*args, **kwargs)
        allocations.append(a.nbytes)
        return a
    monkeypatch.setattr(np, "empty", allocate)
    def reader(fd, offset, destination):
        owners.add(id(destination.obj))
        calls.append(len(destination))
        return buffers._read_into(fd, offset, destination[:min(len(destination), 65537)])
    host = buffers.read_original_packed(artifact, 0, 0, read_into=reader)
    assert owners == {id(host.data.base)}
    assert allocations == [buffers.ORIGINAL_BYTES]
    assert max(calls) <= 1 << 20
    assert host.host_allocation_nbytes == buffers.ORIGINAL_BYTES
    for i, t in enumerate(reversed(host.layout.tensors)):
        assert np.all(host.data[t.offset:t.offset+t.nbytes] == i+1)
    assert not host.data.flags.writeable


@pytest.mark.parametrize("count", [0, -1, True, None, 1 << 30])
def test_failed_read_never_returns_partial_expert(tmp_path, count):
    artifact, _ = original_file(tmp_path)
    with pytest.raises(OSError):
        buffers.read_original_packed(artifact, 0, 0, read_into=lambda *args: count)


def test_changed_source_refused_after_read(tmp_path):
    artifact, path = original_file(tmp_path)
    changed = False
    def reader(fd, offset, destination):
        nonlocal changed
        count = buffers._read_into(fd, offset, destination)
        if not changed:
            st = path.stat()
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns+10_000_000))
            changed = True
        return count
    with pytest.raises(ValueError, match="changed"):
        buffers.read_original_packed(artifact, 0, 0, read_into=reader)


def test_overlap_rejected_before_reader(tmp_path):
    artifact, _ = original_file(tmp_path)
    rows = list(artifact["tensors"].values())
    rows[1]["offset"] = rows[0]["offset"]
    with pytest.raises(ValueError, match="overlap"):
        buffers.read_original_packed(artifact, 0, 0, read_into=lambda *args: pytest.fail("read invoked"))


def test_q1_borrow_preserves_file_order_owner_and_header_accounting():
    raw, views = q1_host(reverse=True)
    host = buffers.borrow_q1_packed(views)
    assert host.host_allocation_nbytes == buffers.Q1_BYTES+904
    assert host.data.nbytes == buffers.Q1_BYTES
    assert host.data.ctypes.data == raw.ctypes.data+904
    for name, a in host.views().items():
        assert np.shares_memory(a, views[name]) and np.array_equal(a.view(np.uint8), views[name].view(np.uint8))


@pytest.mark.parametrize("defect", ["copy", "overlap", "dtype", "shape", "missing"])
def test_q1_bad_storage_refused(defect):
    raw, views = q1_host()
    name = "gate.qweight"
    if defect == "copy":
        views[name] = views[name].copy()
    elif defect == "overlap":
        views["up.qweight"] = views[name]
    elif defect == "dtype":
        views[name] = views[name].view(">u4")
    elif defect == "shape":
        views[name] = views[name].reshape(-1)
    else:
        del views[name]
    with pytest.raises(ValueError):
        buffers.borrow_q1_packed(views)


def test_q1_misaligned_payload_refused():
    with pytest.raises(ValueError, match="alignment"):
        q1_host(header=905)


def test_unbuffered_readinto_fallback_uses_existing_destination(tmp_path, monkeypatch):
    path = tmp_path / "bytes"
    path.write_bytes(b"abcdefgh")
    monkeypatch.delattr(os, "preadv", raising=False)
    slab = np.zeros(4, np.uint8)
    fd = os.open(path, os.O_RDONLY)
    try:
        assert buffers._read_into(fd, 2, memoryview(slab)) == 4
        assert slab.tobytes() == b"cdef"
        assert os.fstat(fd).st_size == 8  # readinto must not close provider fd
    finally:
        os.close(fd)


@pytest.mark.parametrize("defect", ["gap", "trailing", "independent_owner"])
def test_q1_cannot_hide_extra_storage_or_distinct_owner(defect):
    raw, views = q1_host()
    if defect == "independent_owner":
        other = np.empty(views["gate.qweight"].nbytes, np.uint8)
        views["gate.qweight"] = other.view("<u4").reshape(views["gate.qweight"].shape)
    else:
        extra = np.empty(raw.nbytes+4, np.uint8)
        layout = buffers.canonical_layout("q1")
        views = {t.name: extra[904+t.offset+(4 if defect == "gap" and i else 0):
                                   904+t.offset+(4 if defect == "gap" and i else 0)+t.nbytes]
                 .view(t.dtype).reshape(t.shape) for i, t in enumerate(layout.tensors)}
    with pytest.raises(ValueError):
        buffers.borrow_q1_packed(views)


def test_real_native_reader_output_is_borrowable(tmp_path):
    from mlx2.adapters.deepseek_v41_mlx_weights import _stamp, read_native_file
    raw, views = q1_host(reverse=True)
    host = buffers.borrow_q1_packed(views)
    header = {t.name: dict(dtype="U32" if t.dtype == "<u4" else "F16",
                          shape=list(t.shape), data_offsets=[t.offset, t.offset+t.nbytes])
              for t in host.layout.tensors}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path = tmp_path / "q1.safetensors"
    with path.open("wb") as out:
        out.write(len(encoded).to_bytes(8, "little")); out.write(encoded); out.write(memoryview(host.data))
    record = dict(size=path.stat().st_size, stat=_stamp(path.stat()), header_bytes=8+len(encoded),
                  tensors=header, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    loaded = buffers.borrow_q1_packed(read_native_file(tmp_path, path.name, record))
    assert loaded.host_allocation_nbytes == record["size"]
    assert np.array_equal(loaded.data, host.data)


@pytest.mark.parametrize("kind", ["original", "q1"])
def test_cpu_materialization_one_backing_exact_views_no_stale_host(kind, monkeypatch):
    import mlx.core as mx
    if kind == "q1":
        raw, views = q1_host(reverse=True)
        host = buffers.borrow_q1_packed(views)
        del views
    else:
        raw = np.full(buffers.ORIGINAL_BYTES, 13, np.uint8)
        host = buffers.HostExpert(raw, buffers.canonical_layout(kind))
    reference = weakref.ref(raw)
    calls, array = [], mx.array
    def construct(value, *args, **kwargs):
        calls.append((value.nbytes, value.dtype))
        return array(value, *args, **kwargs)
    monkeypatch.setattr(mx, "array", construct)
    packed = buffers.materialize(host, stream=mx.cpu)
    with mx.stream(mx.cpu):
        mx.eval(packed.backing, *packed.values())
        backing = np.asarray(packed.backing)
        assert packed.allocation_nbytes == sum(a.nbytes for a in packed.values()) == host.layout.nbytes
        assert len(calls) == 1 and calls[0] == (host.layout.nbytes, np.dtype("uint8"))
        for t in host.layout.tensors:
            a = np.asarray(packed[t.name])
            assert np.shares_memory(a, backing)
            assert a.ctypes.data-backing.ctypes.data == t.offset
            assert a.shape == t.shape and a.dtype.str == t.dtype
            assert np.array_equal(a.view(np.uint8).reshape(-1), host.data[t.offset:t.offset+t.nbytes])
        with pytest.raises(TypeError):
            packed["extra"] = packed.backing
    del host, raw
    gc.collect()
    assert reference() is None


def lfu_fixture(tmp_path, monkeypatch, *, packed=True, mixed=False):
    from types import SimpleNamespace
    from mlx2.adapters import deepseek_v41_source_lfu as lfu
    from mlx2.adapters import deepseek_v41_source_weights as weights
    artifact, path = original_file(tmp_path)
    for e in (1, 2):
        artifact['tensors'].update({k.replace('experts.0.', f'experts.{e}.'): dict(v)
                                   for k, v in list(artifact['tensors'].items()) if 'experts.0.' in k})
    artifact['args'] = SimpleNamespace(num_hidden_layers=1, n_routed_experts=3, num_experts_per_tok=1)
    class Pool:
        failure = None
        closed = False
        def __init__(self, *args, **kwargs): self.calls = []
        def call(self, function, *args):
            self.calls.append(function.__name__)
            return function(*args)
        def close(self): self.closed = True
        def counters(self): return {'weight_read_inflight': 0}
    monkeypatch.setattr(weights, 'ReadPool', Pool)
    budget = 2*buffers.ORIGINAL_BYTES + (buffers.Q1_BYTES if mixed else 0)
    bounds = dict(expert_cache_bytes=budget, minimum_expert_cache_bytes=buffers.ORIGINAL_BYTES,
                  expert_table_bytes=3*buffers.ORIGINAL_BYTES, manifest_binding_sha256='synthetic-no-model',
                  expert_active_bound_bytes=budget, expert_staging_bound_bytes=0)
    store = lfu.LFUSourceExpertStore(artifact, bounds, {'mode': 'original_lfu', 'budget_bytes': budget},
                                   packed_buffers=packed)
    store._synchronize = lambda: None  # All work in these tests uses mx.cpu.
    if mixed:
        store.q1_root = tmp_path
        store.hot = frozenset({(0, 0)})
        store.policy.update(mode='hot_original_q1', hot_bytes=buffers.ORIGINAL_BYTES)
        store.q1_records = {(0, e): {'file': path.name, 'size': buffers.Q1_BYTES+904} for e in (1, 2)}
        monkeypatch.setattr(lfu, '_verify_file', lambda *args: path)
        monkeypatch.setattr(lfu, 'q1_read', lambda *args: q1_host(reverse=True)[1])
    return store


@pytest.mark.parametrize('mixed', [False, True])
def test_lfu_packed_miss_hit_lease_evict_and_close(tmp_path, monkeypatch, mixed):
    import mlx.core as mx
    store = lfu_fixture(tmp_path, monkeypatch, mixed=mixed)
    with mx.stream(mx.cpu):
        with store.window():
            first = store.acquire('blk.0', 0)
            assert store.acquire('blk.0', 0) is first
            assert isinstance(first, buffers.PackedExpert)
            if mixed:
                q1 = store.acquire('blk.0', 1)
                assert q1.layout.kind == 'q1'
        with store.window():
            assert store.acquire('blk.0', 0) is first
        with store.window():
            last = store.acquire('blk.0', 2)
        assert store.stats['cache_hits'] == 1
        assert store.stats['evictions'] == 1
        assert (0, 0) in store._cache if mixed else (0, 0) not in store._cache
        receipt = store.receipt()
        assert receipt['packed_live_backing_payload_bytes'] == store._cache_bytes == store.live_bytes()
        assert receipt['packed_live_backing_allocations'] == (2 if mixed else 1)
        assert receipt['pending_host_source_bytes'] == 0
        assert store.stats['packed_backing_allocations'] == (3 if mixed else 2)
        assert store.stats['packed_typed_views'] == (24 if mixed else 12)
        assert store.stats['packed_peak_retained_host_bytes'] == buffers.ORIGINAL_BYTES
        assert store._cache_bytes+buffers.ORIGINAL_BYTES <= store.cache_budget_bytes
        store.close()
        assert not store._cache and not store._leased and not store._pending_host and not store._pending_uploads
        assert store.pool_bytes == {'original': 0, 'q1': 0}
        assert store.receipt()['packed_live_backing_allocations'] == 0


@pytest.mark.parametrize('mixed', [False, True])
def test_legacy_loader_remains_dict_per_tensor_upload(tmp_path, monkeypatch, mixed):
    import mlx.core as mx
    from mlx2.adapters import deepseek_v41_source_lfu as lfu
    store = lfu_fixture(tmp_path, monkeypatch, packed=False, mixed=mixed)
    monkeypatch.setattr(lfu, 'materialize', lambda *args: pytest.fail('packed materialization on legacy path'))
    monkeypatch.setattr(lfu, 'read_original_packed', lambda *args: pytest.fail('packed read on legacy path'))
    with mx.stream(mx.cpu), store.window():
        expert = store.acquire('blk.0', 1 if mixed else 0)
        assert type(expert) is dict and len(expert) == (9 if mixed else 6)
    assert 'packed_buffers' not in store.receipt()
    assert not any(k.startswith('packed_') for k in store.stats)
    store.close()


@pytest.mark.parametrize('mixed', [False, True])
@pytest.mark.parametrize('fence_failure', [False, True])
@pytest.mark.parametrize('failure', [RuntimeError, KeyboardInterrupt])
def test_pending_packed_owners_survive_eval_until_fence(tmp_path, monkeypatch, mixed, fence_failure, failure):
    import mlx.core as mx
    from mlx2.adapters.deepseek_v41_mlx_weights import _UPLOAD_QUARANTINE
    store = lfu_fixture(tmp_path, monkeypatch, mixed=mixed)
    key = (0, 1 if mixed else 0)
    def fail_eval(*leaves):
        host, packed = store._pending_host[key], store._pending_uploads[key]
        assert isinstance(host, buffers.HostExpert) and isinstance(packed, buffers.PackedExpert)
        assert leaves[0] is packed.backing and len(leaves) == len(packed)+1
        assert store.receipt()['pending_host_source_bytes'] == (buffers.Q1_BYTES+904 if mixed else buffers.ORIGINAL_BYTES)
        raise failure('injected evaluation failure')
    monkeypatch.setattr(mx, 'eval', fail_eval)
    if fence_failure:
        def fail_fence(): raise RuntimeError('injected fence failure')
        store._synchronize = fail_fence
    with mx.stream(mx.cpu), pytest.raises((RuntimeError, KeyboardInterrupt)):
        with store.window(): store.acquire('blk.0', key[1])
    assert bool(store._pending_host) == fence_failure
    assert bool(store._pending_uploads) == fence_failure
    if fence_failure:
        assert _UPLOAD_QUARANTINE[id(store)] is store
        with pytest.raises(RuntimeError, match='fence'):
            store.close()
        assert store._pending_host and store._pending_uploads
    store._synchronize = lambda: None
    store.close()
    assert not store._pending_host and not store._pending_uploads and id(store) not in _UPLOAD_QUARANTINE


def test_packed_active_expert_cannot_be_evicted(tmp_path, monkeypatch):
    import mlx.core as mx
    store = lfu_fixture(tmp_path, monkeypatch)
    with mx.stream(mx.cpu):
        with pytest.raises(MemoryError, match='protected/active'):
            with store.window():
                retained = store.acquire('blk.0', 0)
                store.acquire('blk.0', 1)
        assert store._cache[(0, 0)] is retained
        assert store.stats['evictions'] == 0
        assert store._cache_bytes == buffers.ORIGINAL_BYTES
        store.close()


def test_packed_flag_requires_boolean_before_constructing_reader():
    from mlx2.adapters.deepseek_v41_source_lfu import LFUSourceExpertStore
    with pytest.raises(ValueError, match='boolean'):
        LFUSourceExpertStore(None, None, None, packed_buffers=1)


def test_packed_receipt_never_scans_cache_including_inherited_live_bytes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    store = lfu_fixture(tmp_path, monkeypatch)
    class NoScanCache(dict):
        def __iter__(self): pytest.fail('full cache iteration')
        def values(self): pytest.fail('full cache values scan')
        def items(self): pytest.fail('full cache items scan')
    # Metadata only: neither tensors nor a model are allocated for this test.
    objects = {(0,e): SimpleNamespace(backing=object(), allocation_nbytes=buffers.ORIGINAL_BYTES)
               for e in range(5448)}
    cached = objects[(0,0)]
    store._cache = NoScanCache(objects)
    store._cache_bytes = len(objects)*buffers.ORIGINAL_BYTES
    store._leased = {(0,0):cached}
    store._pending_uploads = {(0,0):cached}
    store.ple_tables = {'synthetic':SimpleNamespace(_cache={0:None,1:None},row_bytes=12,reserved_bytes=lambda:24)}
    receipt = store.receipt()
    assert receipt['packed_live_backing_allocations'] == 5448
    assert receipt['packed_live_backing_payload_bytes'] == store._cache_bytes
    assert receipt['live_bytes'] == store.resident_bytes() == store._cache_bytes+24
    store.close()


def test_packed_pending_count_keeps_distinct_owner_at_same_key_and_deduplicates(tmp_path, monkeypatch):
    from types import SimpleNamespace
    store = lfu_fixture(tmp_path, monkeypatch)
    cached = SimpleNamespace(backing=object(), allocation_nbytes=buffers.ORIGINAL_BYTES)
    failed = SimpleNamespace(backing=object(), allocation_nbytes=buffers.Q1_BYTES)
    leased = SimpleNamespace(backing=object(), allocation_nbytes=buffers.ORIGINAL_BYTES)
    store._cache = {(0,0):cached};store._cache_bytes=buffers.ORIGINAL_BYTES
    store._leased = {(0,0):cached,(0,1):leased}
    store._pending_uploads = {(0,0):failed,(0,1):leased,(0,2):{}}
    receipt=store.receipt()
    assert receipt['packed_live_backing_allocations']==3
    assert receipt['packed_live_backing_payload_bytes']==2*buffers.ORIGINAL_BYTES+buffers.Q1_BYTES
    assert receipt['live_bytes']==receipt['packed_live_backing_payload_bytes']
    store.close()
