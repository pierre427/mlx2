"""Synthetic packed-byte CPU tests; no checkpoint or GPU inference."""
from collections import OrderedDict
from dataclasses import replace
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_source_bank as bank
from mlx2.adapters.deepseek_v41_source_buffers import (
    ExpertLayout, PackedExpert, canonical_layout,
)


class CPU:
    uint8, uint32, float16 = np.uint8, np.uint32, np.float16
    def __init__(self): self.evals, self.cats = 0, 0
    def eval(self, *values): self.evals += 1
    def array(self, value): return np.array(value)
    def concatenate(self, values, axis):
        self.cats += 1
        return np.concatenate(values, axis=axis)


def expert(kind="q1", reverse=False, seed=7):
    layout = canonical_layout(kind)
    if reverse:
        rows, cursor = [], 0
        for t in reversed(layout.tensors):
            rows.append(replace(t, offset=cursor))
            cursor += t.nbytes
        layout = ExpertLayout(kind, tuple(rows))
    data = np.arange(layout.nbytes, dtype=np.uint8) + np.uint8(seed)
    tensors = {t.name: data[t.offset:t.offset+t.nbytes].view(t.dtype).reshape(t.shape)
               for t in layout.tensors}
    return PackedExpert(data, layout, tensors)


def store(*values):
    cache = OrderedDict(values)
    result = SimpleNamespace(_lock=threading.RLock(), _active=False, _leased={},
        _pending_uploads={}, _pending_host={}, _completion_collector=None,
        _cache=cache, _cache_bytes=sum(v.allocation_nbytes for v in cache.values()),
        cache_budget_bytes=75_000_000_000, binding="synthetic-source",
        q1_binding="synthetic-q1", hot_binding="synthetic-map", fences=0,
        _ready=lambda: None, pool=SimpleNamespace(counters=lambda: {}),
        pool_bytes={k: sum(v.allocation_nbytes for v in cache.values() if v.layout.kind == k)
                    for k in ("original", "q1")})
    result.kind = lambda key: cache[key].layout.kind
    def fence(): result.fences += 1
    result._fence = fence
    return result


def test_lossless_original_q1_and_native_layout_groups():
    old = {(0, 7): expert("original"), (1, 5): expert(),
           (1, 9): expert(reverse=True), (39, 383): expert(seed=91)}
    s, mx = store(*old.items()), CPU()
    p = bank.plan_bank(s, tuple(old))
    assert len(p.bank_nbytes) == 3
    assert p.payload_bytes == s._cache_bytes
    assert p.peak_bound_bytes == 2*p.payload_bytes + max(p.bank_nbytes) + 3*p.metadata_bytes
    b = bank.build_bank(mx, s, tuple(old), ownercheck=lambda: None)
    assert mx.cats == 3 and s.fences == 1
    assert b.lut.shape == (40, 384, 4) and b.lut.dtype == np.uint32
    assert b.lut[0, 0, 0] == 0
    for key, value in old.items():
        row = b.lookup_cpu(*key)
        assert tuple(b.lut[key]) == (bank.KIND_CODES[value.layout.kind], row.bank_index,
                                     row.offset_words, 1)
        for name, a in value.items():
            np.testing.assert_array_equal(s._cache[key][name], a)
            assert np.shares_memory(s._cache[key][name], b.banks[row.bank_index])
        assert s._cache[key].layout == value.layout
    assert all(a.shape == (bank.BANK_ANCHOR_WORDS,) and np.shares_memory(a, full)
               for a, full in zip(b.anchors, b.banks))
    assert s._resident_bank_pinned_keys == frozenset(old)
    assert b.receipt()["steady_coefficient_copies"] == 1
    assert "pointer" not in str(b.plan.receipt())
    with pytest.raises(ValueError, match="route miss"):
        b.lookup_cpu(0, 0)


def test_bank_split_preserves_span_and_budget():
    a, c = expert(), expert(seed=8)
    s = store(((0, 1), a), ((0, 2), c))
    p = bank.plan_bank(s, tuple(s._cache), max_bank_bytes=a.allocation_nbytes)
    assert p.bank_nbytes == (a.allocation_nbytes, a.allocation_nbytes)
    assert [r.offset_words for r in p.slices] == [0, 0]
    with pytest.raises(ValueError, match="peak exceeds"):
        bank.plan_bank(s, tuple(s._cache), budget_bytes=p.peak_bound_bytes-1)
    with pytest.raises(ValueError, match="single Source expert"):
        bank.plan_bank(s, tuple(s._cache), max_bank_bytes=4)


@pytest.mark.parametrize("keys", [(), ((0, 0), (0, 0)), ((40, 0),), ((0, True),), ([0, 0],)])
def test_invalid_closed_set(keys):
    s = store(((0, 0), expert()))
    with pytest.raises(ValueError): bank.plan_bank(s, keys)


def test_missing_preload_and_wrong_generation_refuse_without_backend_work():
    s, mx = store(((0, 0), expert())), CPU()
    with pytest.raises(ValueError, match="ownership"):
        bank.build_bank(mx, s, ((0, 0),), ownercheck=None)
    with pytest.raises(ValueError, match="preload"):
        bank.build_bank(mx, s, ((0, 1),), ownercheck=lambda: None)
    assert mx.cats == mx.evals == 0
    b = bank.build_bank(mx, s, ((0, 0),), ownercheck=lambda: None)
    with pytest.raises(ValueError, match="existing"):
        bank.build_bank(mx, s, ((0, 0),), ownercheck=lambda: None)
    b.close()
    assert not s._cache and not s._resident_bank_pinned_keys and s._cache_bytes == 0
    s._cache[(0, 0)] = expert()
    s._cache_bytes = s._cache[(0, 0)].allocation_nbytes
    with pytest.raises(ValueError, match="increase"):
        bank.build_bank(mx, s, ((0, 0),), ownercheck=lambda: None)


def test_lease_has_no_eval_or_fence_and_close_requires_completion():
    s, mx = store(((0, 0), expert())), CPU()
    b = bank.build_bank(mx, s, ((0, 0),), ownercheck=lambda: None)
    before = (mx.evals, s.fences)
    with b.lease() as held:
        assert held is b
        b.assert_current(contents=False)
        with pytest.raises(ValueError, match="live forward"):
            b.close()
    assert before == (mx.evals, s.fences)
    s._active = True
    with pytest.raises(ValueError, match="idle"):
        b.close()
    s._active = False
    b.close()
    assert s.fences == before[1]+1
    with pytest.raises(ValueError, match="closed"):
        b.assert_current()


def test_failure_keeps_original_cache_and_quarantines_failed_completion():
    original = expert()
    s, mx = store(((0, 0), original)), CPU()
    def fail(*args, **kwargs): raise RuntimeError("allocation failed")
    mx.concatenate = fail
    with pytest.raises(RuntimeError, match="allocation failed"):
        bank.build_bank(mx, s, ((0, 0),), ownercheck=lambda: None)
    assert s._cache[(0, 0)] is original and not getattr(s, "_resident_bank", None)
    calls = 0
    def fence():
        nonlocal calls
        calls += 1
        if calls > 1: raise RuntimeError("failed wait")
    s._fence = fence
    start = len(bank._QUARANTINE)
    try:
        with pytest.raises(RuntimeError, match="allocation failed"):
            bank.build_bank(mx, s, ((0, 0),), ownercheck=lambda: None)
        assert len(bank._QUARANTINE) == start+1 and "failed wait" in s._error
        assert bank._QUARANTINE[-1][1][(0, 0)] is original
    finally:
        del bank._QUARANTINE[start:]


def test_input_disconnected_typed_view_refuses_atomic_publication():
    a = expert()
    a._tensors["gate.scales"] = a["gate.scales"].copy()
    s, mx = store(((0, 0), a)), CPU()
    with pytest.raises(ValueError, match="does not alias"):
        bank.build_bank(mx, s, ((0, 0),), ownercheck=lambda: None)
    assert s._cache[(0, 0)] is a and not getattr(s, "_resident_bank", None)


def test_stale_cache_and_profile_are_distinct_from_closed_route_miss():
    s, mx = store(((0, 0), expert())), CPU()
    b = bank.build_bank(mx, s, ((0, 0),), ownercheck=lambda: None)
    old = s._cache[(0, 0)]
    s._cache[(0, 0)] = expert()
    assert b.assert_current(contents=False) is b
    with pytest.raises(ValueError, match="ownership changed"): b.assert_current()
    s._cache[(0, 0)] = old
    s.hot_binding = "changed"
    with pytest.raises(ValueError, match="profile"): b.assert_current(contents=False)


def test_over_two_gib_bank_slices_words_before_byte_view(monkeypatch):
    """Metadata-only backend reproduces MLX signed32 dimension refusal.

    No multi-GiB allocation or GPU work: the full bank is virtual. Independent
    byte-offset checks cover every expert and typed field including final tail.
    """
    class Virtual:
        def __init__(self, shape, dtype, root=None, offset=0):
            self.shape, self.dtype = tuple(shape), np.dtype(dtype)
            self.nbytes = int(np.prod(shape)) * self.dtype.itemsize
            self.root, self.offset = (object() if root is None else root), offset
        def __getitem__(self, row):
            start, stop, step = row.indices(self.shape[0])
            assert step == 1
            return Virtual((max(0, stop-start),), self.dtype, self.root,
                           self.offset + start*self.dtype.itemsize)
        def view(self, dtype):
            size = self.nbytes // np.dtype(dtype).itemsize
            if size >= 2**31:
                raise OverflowError("full-bank byte dimension exceeds MLX signed32")
            return Virtual((size,), dtype, self.root, self.offset)
        def reshape(self, shape):
            assert int(np.prod(shape))*self.dtype.itemsize == self.nbytes
            return Virtual(shape, self.dtype, self.root, self.offset)
    class MetaCPU(CPU):
        def concatenate(self, values, axis):
            assert axis == 0 and all(v.dtype == np.uint32 for v in values)
            self.cats += 1
            return Virtual((sum(v.shape[0] for v in values),), np.uint32)
    checked = []
    def alias(full, view, *, byte_offset=0, byte_length=None):
        assert full.root is view.root
        assert view.offset == full.offset+byte_offset
        assert view.nbytes == byte_length
        assert 0 <= byte_offset <= full.nbytes-view.nbytes
        checked.append((byte_offset, view.nbytes))
    monkeypatch.setattr(bank, "_alias", alias)
    layout = canonical_layout("original")
    values = []
    for i in range(115):
        data = Virtual((layout.nbytes,), np.uint8)
        tensors = {t.name: data[t.offset:t.offset+t.nbytes].view(t.dtype).reshape(t.shape)
                   for t in layout.tensors}
        values.append(((0, i), PackedExpert(data, layout, tensors)))
    s, mx = store(*values), MetaCPU()
    p = bank.plan_bank(s, tuple(s._cache))
    assert 2**31 < p.payload_bytes < 4*2**30
    assert len(p.bank_nbytes) == 1
    b = bank.build_bank(mx, s, tuple(s._cache), ownercheck=lambda: None)
    final = b.lookup_cpu(0, 114)
    assert final.offset_words*4+layout.nbytes == p.payload_bytes
    assert s._cache[(0, 114)].backing.nbytes == layout.nbytes
    assert (final.offset_words*4, layout.nbytes) in checked
    assert b.anchors[0].shape == (bank.BANK_ANCHOR_WORDS,)


def test_superset_replacement_keeps_whole_old_backing_until_fenced_publish():
    s,mx=store(((0,1),expert()),((0,2),expert(seed=19))),CPU()
    old=bank.build_bank(mx,s,((0,1),),ownercheck=lambda:True)
    old_view=s._cache[(0,1)];old_bytes=s._cache_bytes
    with old.replacement(((0,1),(0,2))):
        assert s._resident_bank is None and s._cache[(0,1)]is old_view
        assert old._resources is not None and s._cache_bytes==old_bytes
        assert (0,1)in s._resident_bank_pinned_keys
        fresh=bank.build_bank(mx,s,((0,1),(0,2)),ownercheck=lambda:True,generation=2)
    assert old._closed and old._resources is None and s._resident_bank is fresh
    assert s._cache_bytes==old_bytes and set(s._cache)=={(0,1),(0,2)}
    np.testing.assert_array_equal(s._cache[(0,1)].backing,old_view.backing)


def test_completed_replacement_allocation_failure_restores_original_bank_and_cache():
    s,mx=store(((0,1),expert())),CPU()
    old=bank.build_bank(mx,s,((0,1),),ownercheck=lambda:True);view=s._cache[(0,1)]
    with pytest.raises(MemoryError):
        with old.replacement(((0,1),)):
            mx.concatenate=lambda *a,**k:(_ for _ in ()).throw(MemoryError('actual copy allocation'))
            bank.build_bank(mx,s,((0,1),),ownercheck=lambda:True,generation=2)
    assert s._resident_bank is old and not old._closed and s._cache[(0,1)]is view
    assert s._resident_bank_pinned_keys==frozenset({(0,1)})
    old.assert_current()


def test_replacement_refuses_live_lease_and_cannot_drop_backing_siblings():
    s,mx=store(((0,1),expert()),((0,2),expert(seed=9))),CPU()
    old=bank.build_bank(mx,s,((0,1),(0,2)),ownercheck=lambda:True)
    with old.lease():
        with pytest.raises(ValueError,match='leased'):
            with old.replacement(((0,1),(0,2))):pass
    with pytest.raises(ValueError,match='preserve every old'):
        with old.replacement(((0,1),)):pass
