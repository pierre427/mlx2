# SPDX-License-Identifier: MIT
"""Opt-in layer-major prefill expert read-ahead for the Source LFU provider.

``deepseek_source_prefill_readahead: layer_ahead_v1`` (requires
``deepseek_source_prefill_order: layer_major_v1`` and a bounded LFU policy).
Inside a layer-major wave every layer runs several sub-chunks back to back.
When the second sub-chunk of layer ``L`` begins, layer ``L``'s demand reads
are complete (they happen in its first sub-chunk), so the provider starts
host reads of the non-resident experts of layers ``L+1 .. L+DEPTH``, hottest
first by the LFU token counts, while the remaining sub-chunks of ``L``
compute. A wide wave routes almost every expert of a layer (4,096 rows x 6 of
384), so the whole non-resident layer is read ("fetch-all" for big passes, as
the streaming-llama weight fetcher does), bounded by ``MAX_BYTES`` held host
bytes, which are charged to the provider's adaptive staging reserve and so
come out of the configured expert maximum.

Reads are host-only: the job is the identical ``_expert_read_job(key)`` the
demand path runs, consumed by ``acquire`` in place of its synchronous read.
Nothing is uploaded, cached, leased or counted by LFU here; uploads, eviction,
split pools, kernels and numerics are the ordinary ones. Unconsumed reads are
retired at the next layer or at the end of the forward and reported as waste.
A failed read never poisons: it disables read-ahead and the ordinary
synchronous read supplies the bytes. Chunk-major prefill and decode never
issue reads (a layer is never revisited consecutively), so decode is
unaffected (cf. the recorded 3-7% warm-decode cost of per-call overlap_v1).
Provenance: provenance/deepseek-v41-source-prefill-readahead.json.
"""
from concurrent.futures import TimeoutError as FutureTimeout
import time

KEY = 'deepseek_source_prefill_readahead'
REFERENCE = 'reference'
POLICY = 'layer_ahead_v1'
DEPTH = 2
MAX_BYTES = 3 << 30
WORKERS = 2
MAX_PENDING = 512
COUNTERS = ('readahead_issue_events', 'readahead_reads_issued', 'readahead_bytes_issued',
            'readahead_used', 'readahead_used_bytes', 'readahead_ready_bytes', 'readahead_wasted',
            'readahead_wasted_bytes', 'readahead_failed', 'readahead_skipped_budget',
            'readahead_skipped_memory', 'readahead_wait_ns', 'readahead_peak_held_bytes')
_REFUSED = (('deepseek_source_expert_io', 'reference'),)


def validate_policy(policy):
    selected = policy.get(KEY, REFERENCE)
    if selected not in (REFERENCE, POLICY):
        raise ValueError('invalid Source policy: ' + KEY)
    if selected == REFERENCE:
        return selected
    if policy.get('deepseek_source_prefill_order', 'reference') != 'layer_major_v1':
        raise ValueError('Source prefill read-ahead requires deepseek_source_prefill_order=layer_major_v1')
    expert = policy.get('deepseek_source_experts')
    if expert is None:
        raise ValueError('Source prefill read-ahead requires a bounded deepseek_source_experts policy')
    if expert.get('residency') == 'dynamic_precision_v1':
        raise ValueError('Source prefill read-ahead cannot combine with dynamic_precision_v1')
    for name, reference in _REFUSED:
        if policy.get(name, reference) != reference:
            raise ValueError('Source prefill read-ahead cannot combine with ' + name)
    return selected


class LayerReadAhead:
    """Bounded host read-ahead owned by one LFU store; caller holds its lock."""

    def __init__(self, store, *, depth=DEPTH, max_bytes=MAX_BYTES, pool=None):
        if type(depth) is not int or not 1 <= depth <= 4 or type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('Source read-ahead requires depth 1..4 and positive bytes')
        self.store, self.depth, self.max_bytes = store, depth, max_bytes
        if pool is None:
            from ..runtime.io_ownership import ReadPool, storage_owner
            pool = ReadPool(WORKERS, 'v41-source-prefill-readahead', timeout=60, max_pending=MAX_PENDING,
                            owner=storage_owner('v41-source-prefill-readahead', store.root))
        self.pool = pool
        self.reads = {}          # key -> (future, bytes, function)
        self.retired = []        # (future, bytes) whose syscall may still run
        self.disabled = False
        self.last_prefix = None
        self.issued_through = -1
        for name in COUNTERS:
            store.stats.setdefault(name, 0)

    # ---- accounting
    def held_bytes(self):
        self.retired[:] = [(f, n) for f, n in self.retired if not f.done()]
        return sum(entry[1] for entry in self.reads.values()) + sum(n for _, n in self.retired)

    def _size(self, key):
        store = self.store
        if store.kind(key) == 'q1':
            return int(store.q1_records[key]['size'])
        return int(store.minimum_cache_bytes)

    def _retire(self, key):
        future, size, _ = self.reads.pop(key)
        stats = self.store.stats
        stats['readahead_wasted'] += 1
        stats['readahead_wasted_bytes'] += size
        if future.cancel():
            return
        if not future.done():
            self.retired.append((future, size))
            return
        try:
            future.result()
        except Exception:
            stats['readahead_failed'] += 1

    def retire_below(self, layer):
        for key in [k for k in self.reads if k[0] < layer]:
            self._retire(key)

    def end_wave(self):
        for key in list(self.reads):
            self._retire(key)
        self.last_prefix = None
        self.issued_through = -1

    # ---- hooks
    def observe_call(self, prefix, rows):
        """A repeated layer prefix is a layer-major sub-chunk: read ahead."""
        layer = int(prefix.split('.')[1])
        previous, self.last_prefix = self.last_prefix, prefix
        if previous is not None and int(previous.split('.')[1]) > layer:
            self.end_wave()
            self.last_prefix = prefix
        self.retire_below(layer)
        if previous != prefix or rows <= 1 or self.disabled:
            return 0
        return self.issue(layer)

    def _memory_tight(self):
        last = getattr(self.store, '_adaptive_last', {}) or {}
        if last.get('availability_known') is not True:
            return False
        headroom = max(getattr(self.store, '_adaptive_headroom', 0), getattr(self.store, '_adaptive_protected_bytes', 0))
        return last.get('measured_available_bytes', 0) < headroom + self.max_bytes

    def issue(self, layer):
        store, stats = self.store, self.store.stats
        stats['readahead_issue_events'] += 1
        if self._memory_tight():
            stats['readahead_skipped_memory'] += 1
            return 0
        from ..runtime.io_ownership import ReadQueueFull
        transient = getattr(store, '_prefill_transient', {})
        issued = 0
        held = self.held_bytes()
        for target in range(max(layer + 1, self.issued_through + 1), layer + self.depth + 1):
            prefix = f'blk.{target}'
            if prefix not in store.tables:
                break
            keys = [k for k in store.tables[prefix]
                    if k not in store._cache and k not in store._leased and k not in self.reads
                    and k not in transient and k not in store._pending_uploads]
            keys.sort(key=lambda k: (-store.index.counts[k], -store.index.last.get(k, 0), k[1]))
            complete = True
            for key in keys:
                size = self._size(key)
                if held + size > self.max_bytes:
                    stats['readahead_skipped_budget'] += 1
                    complete = False
                    break
                function, *args = store._expert_read_job(key)
                try:
                    future = self.pool.submit(function, *args)
                except ReadQueueFull:
                    complete = False
                    break
                self.reads[key] = (future, size, function)
                held += size
                issued += 1
                stats['readahead_reads_issued'] += 1
                stats['readahead_bytes_issued'] += size
            stats['readahead_peak_held_bytes'] = max(stats['readahead_peak_held_bytes'], held)
            if not complete:
                break
            self.issued_through = target
        return issued

    def take(self, key, function):
        """Return the early bytes of this exact read job, or None to read now."""
        entry = self.reads.pop(key, None)
        if entry is None:
            return None
        future, size, issued_function = entry
        stats = self.store.stats
        if issued_function is not function:
            # The read job changed (it cannot under exact precision); discard.
            self.reads[key] = entry
            self._retire(key)
            return None
        ready = future.done()
        started = time.perf_counter_ns()
        try:
            raw = future.result(timeout=60)
        except FutureTimeout as error:
            from ..runtime.io_ownership import ReadTimeout
            failure = ReadTimeout('Source prefill read-ahead exceeded its deadline')
            self.pool.fail(failure)
            self.disabled = True
            raise failure from error
        except Exception:
            stats['readahead_failed'] += 1
            self.disabled = True
            return None
        finally:
            stats['readahead_wait_ns'] += time.perf_counter_ns() - started
        stats['readahead_used'] += 1
        stats['readahead_used_bytes'] += size
        if ready:
            stats['readahead_ready_bytes'] += size
        return raw

    def drain(self, timeout=60.0):
        self.end_wave()
        deadline = time.monotonic() + timeout
        for future, _ in tuple(self.retired):
            try:
                future.result(timeout=max(0.0, deadline - time.monotonic()))
            except Exception:
                pass
        self.retired.clear()

    def close(self):
        try:
            self.drain(timeout=5.0)
        finally:
            self.pool.close()

    def receipt(self):
        stats = self.store.stats
        return dict(policy=POLICY, depth=self.depth, max_held_bytes=self.max_bytes, workers=WORKERS,
                    disabled=self.disabled, pending_reads=len(self.reads), held_bytes=self.held_bytes(),
                    read_failure=getattr(self.pool, 'failure', None),
                    counters={k: stats.get(k, 0) for k in COUNTERS},
                    trigger='second same-layer sub-chunk of a layer-major wave',
                    contract='host reads of the identical read job only; uploads, eviction, LFU and kernels unchanged',
                    qualification='unqualified')
