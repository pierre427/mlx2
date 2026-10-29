# SPDX-License-Identifier: MIT
"""Lossless, closed-set Source resident banks; preparation is never token work.

Provenance: provenance/deepseek-v41-source-resident-bank.json. Import/planning
are CPU-only. ``build_bank`` requires caller-owned execution and an idle store.
Banks keep original packed bytes and exact physical ExpertLayout unchanged.
"""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

import numpy as np

from .deepseek_v41_source_buffers import ExpertLayout, PackedExpert

LUT_SHAPE = (40, 384, 4)
LUT_COLUMNS = ("valid_kind", "bank_index", "offset_words", "generation")
KIND_CODES = {"original": 1, "q1": 2}
MAX_BANKS = 24
MAX_BANK_BYTES = 4 << 30
# MLX's custom Metal signature uses constant pointers for arrays with <8
# elements; sixteen uint32 words stays above that threshold with a 64-byte alias.
# Local source: mlx/backend/common/metal_kernel.cpp:19,101,309, revision
# 4ea1bcdb8369ae2f247cca56cf11dbe668ae2299. Installed0.32.4 GPU gate remains
# necessary: the four-byte r2 alias compiled as a constant pointer.
BANK_ANCHOR_WORDS = 16
_QUARANTINE = []


def _live(check):
    if not callable(check):
        raise ValueError("Source bank preparation requires caller ownership/live callback")
    if check() is False:
        raise ValueError("Source bank caller ownership is no longer live")


def _serial(layout):
    return (layout.kind, tuple((t.name, t.offset, t.dtype, t.shape) for t in layout.tensors))


def _binding(store):
    return tuple((name, getattr(store, name, None)) for name in ("binding", "q1_binding", "hot_binding"))


def _idle(store):
    store._ready()
    if (store._active or store._leased or store._pending_uploads or store._pending_host
            or getattr(store, "_completion_collector", None) is not None
            or store.pool.counters().get("weight_read_inflight", 0)):
        raise ValueError("Source bank preparation requires idle, unleased, completed preload")


@dataclass(frozen=True)
class BankSlice:
    key: tuple
    bank_index: int
    offset_words: int
    layout: ExpertLayout


@dataclass(frozen=True)
class SourceBankPlan:
    slices: tuple
    bank_layouts: tuple
    bank_nbytes: tuple
    binding: tuple
    generation: int
    old_cache_bytes: int
    payload_bytes: int
    metadata_bytes: int
    staging_bytes: int
    scratch_bound_bytes: int
    peak_bound_bytes: int
    budget_bytes: int
    fingerprint: str
    construction: str = "atomic"

    @property
    def keys(self):
        return frozenset(s.key for s in self.slices)

    def receipt(self):
        return dict(schema="mlx2.source-resident-bank-plan.v1", generation=self.generation,
                    fingerprint=self.fingerprint, binding=dict(self.binding),
                    experts=len(self.slices), banks=len(self.bank_nbytes),
                    bank_nbytes=list(self.bank_nbytes), payload_bytes=self.payload_bytes,
                    old_cache_bytes=self.old_cache_bytes, metadata_bytes=self.metadata_bytes,
                    staging_bytes=self.staging_bytes, scratch_bound_bytes=self.scratch_bound_bytes,
                    peak_bound_bytes=self.peak_bound_bytes, expert_budget_ceiling_bytes=self.budget_bytes,
                    precision="unchanged original packed coefficients and physical layouts",
                    construction=self.construction,
                    closed_set=True, build_is_token_work=False, qualified=False)


def plan_bank(store, keys, *, generation=1, budget_bytes=75_000_000_000,
              max_bank_bytes=MAX_BANK_BYTES, staging_bytes=0, construction="atomic"):
    """Describe already-resident keys only; never acquire/read/allocate coefficients.

    Atomic peak bounds retain all current cache payload, all new banks, one
    bank-sized concatenation scratch allowance, LUT copies and staging. The
    incremental builder instead retains one new bank and one scratch bank;
    it retires completed old cohorts before moving to the next allocation.
    Cache capacity is a ceiling, not a request to fill the complete checkpoint.
    """
    if (construction not in ("atomic", "incremental")
            or type(generation) is not int or not 1 <= generation <= 0xffffffff
            or type(max_bank_bytes) is not int or not 4 <= max_bank_bytes <= MAX_BANK_BYTES
            or max_bank_bytes % 4 or type(budget_bytes) is not int or budget_bytes <= 0
            or type(staging_bytes) is not int or staging_bytes < 0):
        raise ValueError("invalid Source bank generation/byte bounds")
    raw = tuple(keys)
    if (not raw or any(type(k) is not tuple or len(k) != 2
                      or any(type(v) is not int for v in k)
                      or not 0 <= k[0] < 40 or not 0 <= k[1] < 384 for k in raw)
            or len(set(raw)) != len(raw)):
        raise ValueError("Source bank requires distinct layer/expert keys")
    if budget_bytes > store.cache_budget_bytes:
        raise ValueError("Source bank budget exceeds configured expert maximum")
    groups = {}
    for key in sorted(raw):
        expert = store._cache.get(key)
        if not isinstance(expert, PackedExpert) or not isinstance(expert.layout, ExpertLayout):
            raise ValueError("closed Source bank miss or non-packed expert; preload first")
        layout = expert.layout
        if (tuple(expert.backing.shape) != (layout.nbytes,)
                or int(expert.backing.nbytes) != layout.nbytes
                or layout.kind != store.kind(key)):
            raise ValueError("Source expert backing/format disagrees with immutable profile")
        if layout.nbytes > max_bank_bytes:
            raise ValueError("single Source expert exceeds bank allocation bound")
        groups.setdefault(_serial(layout), []).append((key, layout))
    slices, layouts, sizes = [], [], []
    for group in groups.values():
        bank_index, used = -1, 0
        for key, layout in group:
            if bank_index < 0 or used + layout.nbytes > max_bank_bytes:
                bank_index, used = len(sizes), 0
                layouts.append(layout)
                sizes.append(0)
            slices.append(BankSlice(key, bank_index, used // 4, layout))
            used += layout.nbytes
            sizes[bank_index] = used
    if len(sizes) > MAX_BANKS:
        raise ValueError("Source bank finite Metal input binding limit exceeded")
    old = sum(int(getattr(v, "allocation_nbytes", sum(t.nbytes for t in v.values())))
              for v in store._cache.values())
    if old != store._cache_bytes:
        raise ValueError("Source cache payload accounting disagrees with resident objects")
    payload, metadata = sum(sizes), int(np.prod(LUT_SHAPE)) * 4
    scratch = max(sizes)
    peak = old + (payload if construction == "atomic" else scratch) + 3 * metadata + staging_bytes + scratch
    if peak > budget_bytes:
        raise ValueError("Source bank old+new+scratch+metadata peak exceeds expert ceiling")
    binding = _binding(store)
    identity = dict(binding=binding, generation=generation,
                    slices=[(s.key, s.bank_index, s.offset_words, _serial(s.layout)) for s in slices],
                    banks=sizes, budget_bytes=budget_bytes, peak_bound_bytes=peak, construction=construction)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return SourceBankPlan(tuple(slices), tuple(layouts), tuple(sizes), binding, generation,
                          old, payload, metadata, staging_bytes, scratch, peak, budget_bytes, digest, construction)


def _alias(full, view, *, byte_offset=0, byte_length=None):
    """Transient CPU alias proof only; no pointer is stored, uploaded or emitted."""
    a, b = memoryview(full), memoryview(view)
    if not a.c_contiguous or not b.c_contiguous:
        raise ValueError("Source bank input alias is not contiguous")
    x, y = np.frombuffer(a, dtype=np.uint8), np.frombuffer(b, dtype=np.uint8)
    if (y.ctypes.data != x.ctypes.data + byte_offset
            or y.nbytes != (byte_length if byte_length is not None else y.nbytes)
            or byte_offset < 0 or byte_offset + y.nbytes > x.nbytes):
        raise ValueError("Source bank view does not alias its admitted full backing")


class PublishedSourceBank:
    """Factory-published immutable geometry; resources remain pinned by each lease.

    Call assert_current(contents=True) at forward entry, O(1) contents=False
    between layer jobs. Closing evicts the bank's typed views after one idle
    fence so per-expert LFU accounting never claims partially retained banks.
    """
    __slots__ = ("_store", "_plan", "_resources", "_rebound", "_ownercheck",
                 "_leases", "_closed", "_by_key", "_keys")

    def __init__(self, store, plan, resources, rebound, ownercheck):
        self._store, self._plan, self._resources = store, plan, resources
        self._rebound, self._ownercheck = rebound, ownercheck
        self._leases, self._closed = 0, False
        self._by_key = MappingProxyType({s.key: s for s in plan.slices})
        self._keys = plan.keys

    @property
    def plan(self): return self._plan

    @property
    def banks(self): return self._get()[0]
    @property
    def anchors(self): return self._get()[1]
    @property
    def lut(self): return self._get()[2]
    @property
    def bank_layouts(self): return self.plan.bank_layouts
    @property
    def bank_kinds(self): return tuple(x.kind for x in self.bank_layouts)
    @property
    def generation(self): return self.plan.generation
    @property
    def keys(self): return self._keys
    @property
    def binding(self): return self.plan.binding
    @property
    def fingerprint(self): return self.plan.fingerprint

    def _get(self):
        if self._closed or self._resources is None:
            raise ValueError("Source resident bank is closed")
        return self._resources

    def assert_current(self, *, contents=True):
        self._get()
        store = self._store
        if (getattr(store, "_resident_bank", None) is not self
                or getattr(store, "_resident_bank_generation", None) != self.generation
                or _binding(store) != self.binding):
            raise ValueError("stale Source resident bank generation/profile")
        if contents and any(store._cache.get(k) is not v for k, v in self._rebound.items()):
            raise ValueError("Source resident bank cache ownership changed")
        return self

    def lookup_cpu(self, layer, expert):
        """Debug/planning only; GPU consumers use LUT and never read router IDs."""
        self.assert_current(contents=False)
        try: return self._by_key[(layer, expert)]
        except KeyError: raise ValueError("closed Source resident bank route miss") from None

    @contextmanager
    def lease(self, *, ownercheck=None):
        check = self._ownercheck if ownercheck is None else ownercheck
        with self._store._lock:
            _live(check)
            self.assert_current()
            self._leases += 1
        try:
            yield self
        finally:
            with self._store._lock:
                self._leases -= 1

    def receipt(self):
        return dict(self.plan.receipt(), published=not self._closed,
                    active_leases=self._leases, LUT_columns=list(LUT_COLUMNS),
                    steady_coefficient_copies=1, pointer_addresses_recorded=False)

    @contextmanager
    def replacement(self,keys):
        """Exclusive idle replacement; old views remain fully pinned until copied.

        Only a superset may replace this bank. No old logical eviction or free
        credit occurs while its shared backing remains alive. Atomic builder
        failure restores the original published bank after a proven fence.
        """
        store=self._store;requested=frozenset(keys)
        with store._lock:
            _idle(store);_live(self._ownercheck);self.assert_current()
            if self._leases or getattr(store,'_resident_bank_replacing',None)is not None:
                raise ValueError('Cannot replace a leased or already replacing Source bank')
            if not self.plan.keys.issubset(requested):raise ValueError('Source bank replacement must preserve every old cohort key')
            store._fence();old_pins=store._resident_bank_pinned_keys
            store._resident_bank_replacing=self;store._resident_bank=None
            try:
                yield
                replacement=getattr(store,'_resident_bank',None)
                if (replacement is None or replacement is self or replacement.generation<=self.generation
                        or not self.plan.keys.issubset(replacement.plan.keys)):
                    raise ValueError('Source bank replacement did not publish a newer complete superset')
                replacement.assert_current();store._fence()
            except BaseException:
                # Only the atomic builder is supported: on failure no cache
                # alias may have changed. Unknown completion retains owners.
                try:
                    store._fence()
                    if getattr(store,'_error',None)or any(store._cache.get(k)is not v for k,v in self._rebound.items()):
                        raise RuntimeError('Source bank replacement cannot restore changed or poisoned backing')
                    if getattr(store,'_resident_bank',None)is not None:
                        raise RuntimeError('Source bank replacement failed after foreign publication')
                    store._resident_bank=self;store._resident_bank_pinned_keys=old_pins
                except BaseException as cleanup:
                    store._error='Source bank replacement retirement failed: '+str(cleanup)
                    _QUARANTINE.append((self,store))
                    raise
                raise
            else:
                self._closed=True;self._resources=None;self._rebound={}
            finally:
                if not getattr(store,'_error',None):store._resident_bank_replacing=None

    def close(self):
        store = self._store
        with store._lock:
            if self._closed: return
            if self._leases: raise ValueError("Source resident bank has live forward leases")
            _live(self._ownercheck)
            _idle(store)
            store._fence()
            self.assert_current()
            for key, value in self._rebound.items():
                del store._cache[key]
                store._cache_bytes -= value.allocation_nbytes
                if hasattr(store, "pool_bytes"):
                    store.pool_bytes[value.layout.kind] -= value.allocation_nbytes
            store._resident_bank = None
            store._resident_bank_pinned_keys = frozenset()
            self._closed = True
            self._resources, self._rebound = None, {}


def build_bank(bankmx, store, keys, *, ownercheck, generation=1,
               budget_bytes=75_000_000_000, max_bank_bytes=MAX_BANK_BYTES,
               staging_bytes=0):
    """Prepare under caller-owned execution, then publish atomically.

    No model load, expert acquisition, shader construction or coefficient
    conversion occurs here. Caller must preload the explicit closed key set.
    """
    _live(ownercheck)
    resources = []
    with store._lock:
        _idle(store)
        if getattr(store, "_resident_bank", None) is not None:
            raise ValueError("close existing Source resident bank before replacement")
        if generation <= getattr(store, "_resident_bank_generation", 0):
            raise ValueError("Source resident bank generation must increase")
        store._fence()
        plan = plan_bank(store, keys, generation=generation, budget_bytes=budget_bytes,
                         max_bank_bytes=max_bank_bytes, staging_bytes=staging_bytes)
        originals = {s.key: store._cache[s.key] for s in plan.slices}
        for value in originals.values():
            if value.backing.dtype != bankmx.uint8:
                raise ValueError("Source bank requires uint8 packed backing")
        try:
            banks, anchors, rebound = [], [], {}
            for index, size in enumerate(plan.bank_nbytes):
                _live(ownercheck)
                rows = [s for s in plan.slices if s.bank_index == index]
                bankmx.eval(*(v for s in rows for v in
                              (originals[s.key].backing, *originals[s.key].values())))
                for s in rows:
                    original = originals[s.key]
                    if set(original) != {t.name for t in s.layout.tensors}:
                        raise ValueError("Source packed input tensor inventory mismatch")
                    for t in s.layout.tensors:
                        value = original[t.name]
                        dtype = {"<u4": bankmx.uint32, "|u1": bankmx.uint8,
                                 "<f2": bankmx.float16}[t.dtype]
                        if tuple(value.shape) != t.shape or value.dtype != dtype:
                            raise ValueError("Source packed input typed geometry mismatch")
                        _alias(original.backing, value, byte_offset=t.offset,
                               byte_length=t.nbytes)
                inputs = [originals[s.key].backing.view(bankmx.uint32) for s in rows]
                resources.extend(inputs)
                full = bankmx.concatenate(inputs, axis=0)
                resources.append(full)
                if tuple(full.shape) != (size // 4,) or full.dtype != bankmx.uint32:
                    raise ValueError("Source bank backend returned wrong bank geometry")
                anchor = full[:BANK_ANCHOR_WORDS]
                resources.append(anchor)
                local = []
                for s in rows:
                    layout = s.layout
                    # MLX dimensions are signed32. A <=4GiB uint32 bank has a
                    # valid word dimension, but viewing the full bank as uint8
                    # can overflow it. Slice a single payload in WORDS first;
                    # its <=18.8MB byte view always has an admissible dimension.
                    words = full[s.offset_words:s.offset_words + layout.nbytes // 4]
                    resources.append(words)
                    backing = words.view(bankmx.uint8)
                    if tuple(backing.shape) != (layout.nbytes,):
                        raise ValueError("Source bank expert byte-view geometry mismatch")
                    tensors = {t.name: backing[t.offset:t.offset+t.nbytes].view(
                        {"<u4": bankmx.uint32, "|u1": bankmx.uint8, "<f2": bankmx.float16}[t.dtype]
                        ).reshape(t.shape) for t in layout.tensors}
                    expert = PackedExpert(backing, layout, tensors)
                    resources.extend((backing, *tensors.values()))
                    rebound[s.key] = expert
                    local.append((s, expert))
                bankmx.eval(full, anchor, *(v for _, e in local for v in (e.backing, *e.values())))
                _alias(full, anchor, byte_length=4 * BANK_ANCHOR_WORDS)
                for s, expert in local:
                    _alias(full, expert.backing, byte_offset=s.offset_words*4,
                           byte_length=s.layout.nbytes)
                    for t in s.layout.tensors:
                        _alias(full, expert[t.name], byte_offset=s.offset_words*4+t.offset,
                               byte_length=t.nbytes)
                banks.append(full)
                anchors.append(anchor)
            lut_cpu = np.zeros(LUT_SHAPE, np.uint32)
            resources.append(lut_cpu)
            for s in plan.slices:
                lut_cpu[s.key] = (KIND_CODES[s.layout.kind], s.bank_index, s.offset_words, generation)
            lut = bankmx.array(lut_cpu)
            resources.append(lut)
            bankmx.eval(lut)
            if tuple(lut.shape) != LUT_SHAPE or lut.dtype != bankmx.uint32:
                raise ValueError("Source bank backend returned wrong LUT geometry")
            _live(ownercheck)
            if _binding(store) != plan.binding or any(store._cache.get(k) is not v for k, v in originals.items()):
                raise ValueError("Source cache/profile changed during resident bank preparation")
            published = PublishedSourceBank(store, plan, (tuple(banks), tuple(anchors), lut), rebound, ownercheck)
            replacement = OrderedDict((k, rebound.get(k, v)) for k, v in store._cache.items())
            store._cache = replacement
            store._resident_bank_generation = generation
            store._resident_bank_pinned_keys = plan.keys
            store._resident_bank = published
            return published
        except BaseException as exc:
            try:
                store._fence()
            except BaseException as cleanup:
                store._error = "Source bank preparation completion failed: " + str(cleanup)
                _QUARANTINE.append((store, originals, tuple(resources)))
            raise exc
