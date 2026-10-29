"""Prefill shield (protect_decode_v1) and layer-major read-ahead (layer_ahead_v1).

CPU synthetic stores only (scaled expert sizes, MLX pinned to the CPU device):
no model load, no Metal work. Reuses the residency test's synthetic store.
"""
from concurrent.futures import Future

import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_source_lfu as lfu
from mlx2.adapters import deepseek_v41_source_prefill_residency as pr
from mlx2.adapters import deepseek_v41_source_prefill_readahead as ra
from mlx2.adapters.deepseek_v41_source import source_policy
from tests.test_deepseek_v41_source_residency_cpu import (  # noqa: F401 - fixture
    CHURN, HOT, O, Q, assert_exact, base, forward, make, raw_for, small, use)

DECODE = [(1, 0), (1, 1), (1, 2), (1, 3)]    # Q1-kind decode working set
PREFILL = [k for k in CHURN if k not in DECODE]  # 20 Q1 objects; the Q1 pool holds 16


# ---- Policy surface -----------------------------------------------------

def test_prefill_residency_policy_is_opt_in_and_fails_closed():
    assert 'prefill_residency' not in lfu.parse_policy(base())
    assert lfu.parse_policy(base(prefill_residency='protect_decode_v1'))['prefill_residency'] == 'protect_decode_v1'
    assert lfu.parse_policy(base(residency='shared_v1', prefill_residency='protect_decode_v1'))
    for bad in ('lru', 'terminal_hot', True, None):
        with pytest.raises(ValueError, match='prefill_residency'):
            lfu.parse_policy(base(prefill_residency=bad))
    with pytest.raises(ValueError, match='dynamic'):
        lfu.parse_policy(base(residency='dynamic_precision_v1', prefill_residency='protect_decode_v1'))
    with pytest.raises(ValueError):
        lfu.parse_policy({'mode': 'original_lfu', 'budget_bytes': 8 << 30, 'prefill_residency': 'protect_decode_v1'})
    with pytest.raises(ValueError, match='adaptive'):
        lfu.parse_policy(base(prefill_residency='protect_decode_v1'), adaptive=False)


def scenario(store):
    forward(store, *DECODE)                       # decode-resident set
    reads_before = len(store.reads)
    store.observe_moe_call('blk.1', 128)          # a wide prefill call begins
    for layer, expert in PREFILL:                 # prefill touches > the Q1 pool
        with store.window():
            # Many prompt rows select each expert: prefill counts dominate LFU.
            store.observe_routes(f'blk.{layer}', np.full((8, 1), expert, np.uint32))
            store.acquire(f'blk.{layer}', expert)
        store.end_context()
    prefill_reads = len(store.reads) - reads_before
    store.observe_moe_call('blk.1', 1)            # first decode call
    before = len(store.reads)
    forward(store, *DECODE)
    return prefill_reads, len(store.reads) - before


def test_protect_keeps_decode_set_and_evict_control_rereads(small):
    control = make(fraction=0.5, slots=0)
    protected = make(fraction=0.5, slots=0, extra=dict(prefill_residency='protect_decode_v1'))
    _, control_decode_reads = scenario(control)
    prefill_reads, decode_reads = scenario(protected)
    assert control_decode_reads > 0                # today's eviction makes decode re-read
    assert decode_reads == 0                       # shielded set survived prefill
    assert prefill_reads == len(PREFILL)           # every prefill miss read exactly once
    stats = protected.stats
    assert stats['prefill_shield_snapshots'] == 1 and stats['prefill_shield_releases'] == 1
    assert stats['prefill_shield_snapshot_objects'] == len(DECODE)
    # Prefill evicted its own older objects; the shield never had to yield.
    assert stats['sync_evictions_q1'] > 0 and stats['prefill_shield_yields'] == 0
    assert protected._prefill_shield is None and not protected._prefill_transient
    targets = protected.pool_targets()
    assert protected.pool_bytes['q1'] <= targets['q1']
    receipt = protected.residency_receipt()['prefill_residency']
    assert receipt['policy'] == 'protect_decode_v1' and receipt['qualification'] == 'unqualified'
    assert control.residency_receipt()['prefill_residency'] == {'policy': 'evict_v1', 'qualification': 'unqualified'}
    assert_exact(protected)


def test_transient_bytes_are_identical_reads_and_reused_within_a_layer(small):
    s = make(fraction=0.5, slots=0, extra=dict(prefill_residency='protect_decode_v1'))
    forward(s, *[(1, e) for e in range(16)])      # fill the Q1 pool: all shielded
    s.observe_moe_call('blk.0', 128)
    key = (0, 9)
    forward(s, key)
    assert key not in s._cache and key in s._prefill_transient
    assert s.reads[-1] == (key, 'read_tensors' if s.kind(key) == 'original' else 'q1_read')
    reads = len(s.reads)
    forward(s, key)                               # next same-layer sub-chunk
    assert len(s.reads) == reads and s.stats['prefill_transient_hits'] == 1
    forward(s, (1, 15))                           # another layer clears transients
    assert not s._prefill_transient
    forward(s, key)
    assert len(s.reads) == reads + 1              # re-read, never published
    assert s.live_bytes() >= s._cache_bytes
    assert_exact(s)


def test_transient_cap_yields_shield_and_hard_maximum_holds(small, monkeypatch):
    monkeypatch.setattr(pr, 'TRANSIENT_MAX_BYTES', Q)
    s = make(fraction=0.5, slots=0, extra=dict(prefill_residency='protect_decode_v1'))
    forward(s, *[(1, e) for e in range(16)])
    s.observe_moe_call('blk.0', 128)
    use(s, (0, 8), (0, 9), (0, 10))               # one window: transients exceed the cap
    s.end_context()
    assert s.stats['prefill_shield_yields'] >= 1 and s.stats['prefill_shield_yield_bytes'] >= Q
    assert s._cache_bytes + s._prefill_transient_bytes <= s.maximum_cache_bytes - O
    assert_exact(s)


def test_pressure_eviction_is_never_shielded(small):
    s = make(fraction=0.5, slots=0, extra=dict(prefill_residency='protect_decode_v1'))
    forward(s, *DECODE)
    s.observe_moe_call('blk.1', 128)
    s.available[0] = 0                            # live memory collapses
    s._shield_on = True                           # even inside a shielded miss
    s.adaptive_residency_update()
    s._shield_on = False
    assert sum(k in s._cache for k in DECODE) < len(DECODE)


def test_narrow_verify_rows_release_and_default_never_snapshots(small):
    s = make(fraction=0.5, slots=0, extra=dict(prefill_residency='protect_decode_v1'))
    forward(s, *DECODE)
    s.observe_moe_call('blk.0', pr.PROTECT_MIN_ROWS - 1)
    assert s._prefill_shield is None
    s.observe_moe_call('blk.0', pr.PROTECT_MIN_ROWS)
    assert s._prefill_shield == frozenset(DECODE)
    s.observe_moe_call('blk.0', 6)                # DSpark verify width
    assert s._prefill_shield is None
    d = make(fraction=0.5, slots=0)
    forward(d, *DECODE)
    d.observe_moe_call('blk.0', 4096)
    assert d._prefill_shield is None and 'prefill_shield_snapshots' not in d.stats
    with pytest.raises(ValueError):
        d.observe_moe_call('blk.9', 4)


# ---- Layer-major read-ahead ----------------------------------------------

def q1_read(key):
    return raw_for(key, 'q1_read')


def read_tensors(key):
    return raw_for(key, 'read_tensors')


class SyncPool:
    failure = None

    def __init__(self):
        self.calls = []

    def submit(self, function, *args):
        self.calls.append(args[0])
        future = Future()
        future.set_result(function(*args))
        return future

    def call(self, function, *args):
        self.calls.append(('sync',) + args)
        return function(*args)

    def fail(self, reason):
        self.failure = str(reason)

    def close(self):
        pass

    def counters(self):
        return {'weight_read_inflight': 0}


def readahead_store(**options):
    s = make(fraction=0.5, slots=0)
    del s._read_expert                            # use the class read path
    s.pool = SyncPool()
    s._expert_read_job = lambda key: ((q1_read if s.kind(key) == 'q1' else read_tensors), key)
    s.enable_prefill_readahead(pool=SyncPool())
    for name, value in options.items():
        setattr(s._prefill_readahead, name, value)
    return s


def sub_chunk(s, layer, experts, rows=2048):
    s.observe_moe_call(f'blk.{layer}', rows)
    use(s, *[(layer, e) for e in experts])


def test_readahead_consumes_identical_jobs_for_the_next_layer(small):
    s = readahead_store()
    head = s._prefill_readahead
    sub_chunk(s, 0, range(4))                     # demand reads of layer 0
    assert head.reads == {}                       # first sub-chunk: nothing ahead
    sub_chunk(s, 0, range(4))                     # second sub-chunk: read ahead
    issued = dict(head.reads)
    assert issued and all(k[0] == 1 for k in issued)
    sync_before = len(s.pool.calls)
    sub_chunk(s, 1, range(6))
    assert len(s.pool.calls) == sync_before       # every layer-1 miss came early
    assert s.stats['readahead_used'] == 6
    s.end_context()                               # unconsumed reads are waste
    assert head.reads == {} and s.stats['readahead_wasted'] == len(issued) - 6
    receipt = s.receipt()['prefill_readahead']
    assert receipt['policy'] == 'layer_ahead_v1' and receipt['counters']['readahead_used'] == 6
    assert_exact(s)


def test_readahead_byte_bound_order_and_staging_charge(small):
    s = readahead_store(max_bytes=3 * Q)
    s.index.counts[(1, 7)] = 50
    s.index.counts[(1, 3)] = 40
    sub_chunk(s, 0, [0])
    sub_chunk(s, 0, [0])
    head = s._prefill_readahead
    assert list(head.reads)[:2] == [(1, 7), (1, 3)]    # hottest first
    assert head.held_bytes() <= 3 * Q and s.stats['readahead_skipped_budget'] >= 1
    assert s.readahead_host_reserve_bytes() == 3 * Q
    assert s._adaptive_staging >= O + 3 * Q


def test_readahead_failure_falls_back_and_never_poisons(small):
    s = readahead_store()
    sub_chunk(s, 0, [0])
    sub_chunk(s, 0, [0])
    head = s._prefill_readahead
    bad = Future()
    bad.set_exception(OSError('read failed'))
    key = (1, 0)
    head.reads[key] = (bad, Q, head.reads[key][2])
    sub_chunk(s, 1, [0])
    assert head.disabled and s.stats['readahead_failed'] == 1
    assert s.pool.calls[-1] == ('sync', key)      # ordinary synchronous read supplied it
    assert s._error is None


def test_readahead_policy_requires_layer_major_and_lfu():
    experts = {'mode': 'original_lfu', 'budget_bytes': 75_000_000_000}
    ok = source_policy({'deepseek_source_prefill_readahead': 'layer_ahead_v1',
                        'deepseek_source_prefill_order': 'layer_major_v1', 'deepseek_source_experts': experts})
    assert ok['deepseek_source_prefill_readahead'] == 'layer_ahead_v1'
    assert 'deepseek_source_prefill_readahead' not in source_policy({})
    with pytest.raises(ValueError, match='layer_major'):
        source_policy({'deepseek_source_prefill_readahead': 'layer_ahead_v1', 'deepseek_source_experts': experts})
    with pytest.raises(ValueError, match='deepseek_source_experts'):
        source_policy({'deepseek_source_prefill_readahead': 'layer_ahead_v1',
                       'deepseek_source_prefill_order': 'layer_major_v1'})
    with pytest.raises(ValueError, match='expert_io'):
        source_policy({'deepseek_source_prefill_readahead': 'layer_ahead_v1',
                       'deepseek_source_prefill_order': 'layer_major_v1', 'deepseek_source_experts': experts,
                       'deepseek_source_expert_io': 'overlap_v1'})
    with pytest.raises(ValueError):
        source_policy({'deepseek_source_prefill_readahead': 'all_layers'})


def test_readahead_observer_is_inert_for_chunk_major_and_decode():
    head = ra.LayerReadAhead.__new__(ra.LayerReadAhead)
    head.last_prefix, head.disabled, head.reads, head.issued_through = None, False, {}, -1
    issued = []
    head.issue = lambda layer: issued.append(layer) or 0
    head.end_wave = lambda: None
    for layer in range(4):                        # chunk-major: never repeats a layer
        head.observe_call(f'blk.{layer}', 1024)
    head.observe_call('blk.3', 1)                 # decode repeats rows=1 only
    head.observe_call('blk.3', 1)
    assert issued == []
    head.observe_call('blk.0', 2048)
    head.observe_call('blk.0', 2048)
    assert issued == [0]
    assert np.all(np.array(issued) >= 0)
