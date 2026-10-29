# SPDX-License-Identifier: MIT
"""Bounded output-dependent epochs over immutable Source expert objects.

The ordinary provider remains synchronous. This coordinator is selected only
by an explicit Source policy, retains leases through completion, and is never
an authority to reuse a pending upload or publish incomplete cache state.
Provenance: provenance/deepseek-v41-source-round2.json.
"""
from collections import Counter, deque
from contextlib import contextmanager, ExitStack
import time


class BoundedSourceCompletion:
    def __init__(self, stream, *, maximum=2, backend=None):
        if type(maximum) is not int or not 1 <= maximum <= 2:
            raise ValueError('Source completion admits one or two pending epochs')
        self.stream, self.maximum, self.backend = stream, maximum, backend
        self.active = False
        self._scope_keys = None
        self._epochs = deque()
        self._references = Counter()
        self._values = None
        self._stack = None
        self.stats = dict(submitted_epochs=0,retired_epochs=0,max_pending_epochs=0,
                          maximum_leased_objects=0,lease_releases=0,executions=0,
                          final_drains=0,error_drains=0,frontier_submit_ns=0,
                          marker_build_ns=0,marker_submit_ns=0,retirement_wait_ns=0,
                          final_context_exit_ns=0)

    def _mx(self):
        if self.backend is None:
            import mlx.core as mx
            return mx
        return self.backend

    def note_acquire(self,key):
        if not self.active or self._scope_keys is None:
            raise ValueError('Source completion acquisition outside a scope')
        self._scope_keys.add(key)

    def evaluate(self,*values):
        if not self.active or self._scope_keys is None:
            raise ValueError('Source asynchronous evaluation outside a scope')
        mx=self._mx()
        def leaves(value):
            if isinstance(value,mx.array):yield value
            elif isinstance(value,dict):
                for v in value.values():yield from leaves(v)
            elif isinstance(value,(tuple,list)):
                for v in value:yield from leaves(v)
        arrays=tuple(leaves(values))
        if not arrays:raise ValueError('Source completion scope has no live output')
        started=time.perf_counter_ns()
        mx.async_eval(*arrays)
        self.stats['frontier_submit_ns']+=time.perf_counter_ns()-started
        # More than one evaluation may submit independent lease consumers.
        # Keep every root, not just the last call or its first array.
        retained={id(v):v for v in (self._values or ())}
        retained.update((id(v),v) for v in arrays)
        self._values=tuple(retained.values())

    def _retire_one(self):
        marker,keys=self._epochs[0]
        # Do not remove the epoch or references until completion succeeds.
        started=time.perf_counter_ns()
        self._mx().eval(marker)
        self.stats['retirement_wait_ns']+=time.perf_counter_ns()-started
        self._epochs.popleft()
        for key in keys:
            self._references[key]-=1
            if not self._references[key]:
                del self._references[key]
                with self.stream._lock:
                    self.stream._leased.pop(key,None)
                self.stats['lease_releases']+=1
        self.stats['retired_epochs']+=1

    @contextmanager
    def execution(self):
        if self.active or self.stream._active or self._epochs:
            raise ValueError('Source completion execution is already active or undrained')
        self.active=True
        self.stats['executions']+=1
        self.stream._completion_collector=self.note_acquire
        try:
            try:
                with ExitStack() as self._stack:
                    yield
                    if self._scope_keys is not None:
                        raise ValueError('Source completion execution exited inside a scope')
                    while self._epochs:self._retire_one()
                    exit_started=time.perf_counter_ns()
                self.stats['final_context_exit_ns']+=time.perf_counter_ns()-exit_started
                self.stats['final_drains']+=1
            except BaseException:
                self.stats['error_drains']+=1
                raise
        finally:
            # The provider's window has performed its final/error global fence.
            # Failed fences keep provider ownership and epoch references alive.
            if not self.stream._active:
                self._epochs.clear();self._references.clear()
            self.stream._completion_collector=None
            self._scope_keys=self._values=None
            self._stack=None
            self.active=False

    @contextmanager
    def scope(self):
        if not self.active or self._scope_keys is not None:
            raise ValueError('Source completion scope must be nonnested within execution')
        # Validation or CPU preparation may fail before the first weight scope.
        # Open provider ownership lazily so those input errors do not poison an
        # otherwise unused healthy weight provider.
        if not self.stream._active:
            self._stack.enter_context(self.stream.window())
        while len(self._epochs)>=self.maximum:self._retire_one()
        self._scope_keys=set();self._values=None
        try:
            yield
            if self._values is None:
                raise ValueError('Source completion scope has no evaluated output')
            mx=self._mx()
            started=time.perf_counter_ns()
            # One scalar witness per submitted root covers independent branches
            # without flattening/materializing an entire strided cache view.
            witnesses=[]
            for value in self._values:
                if value.size:
                    first=value[(0,)*value.ndim] if value.ndim else value
                    witnesses.append(first.astype(mx.float32).reshape(1))
            if not witnesses:
                raise ValueError('Source completion scope has only empty outputs')
            joined=witnesses[0] if len(witnesses)==1 else mx.concatenate(witnesses)
            marker=mx.asarray(joined,copy=True)
            self.stats['marker_build_ns']+=time.perf_counter_ns()-started
            started=time.perf_counter_ns()
            mx.async_eval(marker)
            self.stats['marker_submit_ns']+=time.perf_counter_ns()-started
            keys=frozenset(self._scope_keys)
            self._references.update(keys)
            self._epochs.append((marker,keys))
            self.stats['submitted_epochs']+=1
            self.stats['max_pending_epochs']=max(self.stats['max_pending_epochs'],len(self._epochs))
            self.stats['maximum_leased_objects']=max(self.stats['maximum_leased_objects'],len(self.stream._leased))
        finally:
            self._scope_keys=self._values=None

    def receipt(self):
        return dict(policy='bounded_2',maximum_pending_epochs=self.maximum,
                    pending_epochs=len(self._epochs),active=self.active,
                    reference_counted_experts=len(self._references),stats=dict(self.stats),
                    completion='output-dependent immutable marker plus final/error provider fence',
                    qualification='unqualified')
