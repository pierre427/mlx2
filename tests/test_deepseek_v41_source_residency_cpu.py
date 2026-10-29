"""Split pools, pre-eviction, live maximum, recalibrated maps and dynamic precision.

CPU synthetic stores only: no model load, no model-path GPU run. Expert sizes
are scaled down (original 3 KiB, Q1 1 KiB) for the store-level tests; the
accounting is size-agnostic and the real constants are checked separately.
"""
from collections import OrderedDict, defaultdict
from concurrent.futures import Future
import hashlib
import json
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_source_lfu as lfu
from mlx2.adapters import deepseek_v41_source_residency as residency
from mlx2.adapters import deepseek_v41_source_defaults as defaults
from mlx2.adapters import deepseek_v41_source_hotmap as hotmap
from mlx2.adapters.deepseek_v41_source import source_policy
from mlx2.runtime.streamed_load import WeightStreamRequest

O, Q = 3072, 1024          # scaled original / Q1 expert payload bytes
LAYERS, EXPERTS = 2, 16


@pytest.fixture
def small(monkeypatch):
    import mlx.core as mx
    monkeypatch.setattr(lfu, 'ORIGINAL_BYTES', O)
    monkeypatch.setattr(lfu, 'Q1_BYTES', Q)
    monkeypatch.setattr(lfu, 'checked_file', lambda *a, **k: None)
    monkeypatch.setattr(lfu, '_verify_file', lambda *a, **k: None)
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def raw_for(key, function):
    if function == 'q1_read':
        return {'q': np.zeros(Q, np.uint8)}
    result = {}
    for part in ('w1', 'w2', 'w3'):
        name = f'layers.{key[0]}.ffn.experts.{key[1]}.{part}.'
        result[name+'weight'] = np.zeros(O//3-64, np.uint8)
        result[name+'scale'] = np.zeros(64, np.uint8)
    return result


def make(*, mode='split_v1', fraction=0.5, slots=2, maximum=None, available=None, hot=None, extra=None):
    s = lfu.LFUSourceExpertStore.__new__(lfu.LFUSourceExpertStore)
    maximum = maximum or (12*O)
    s.cache_budget_bytes = s.maximum_cache_bytes = maximum
    s.minimum_cache_bytes = O
    s.bounds = {'expert_staging_bound_bytes': O, 'expert_active_bound_bytes': 1 << 40}
    from mlx2.adapters.deepseek_v41_mlx_weights import NativeExpertPlan
    s.plan = NativeExpertPlan(ceiling_bytes=maximum, expert_bytes=O, top_k=2)
    s.atlas_path = None; s._atlas_saved_at = float('inf'); s.preheat_receipt = {'requested': False, 'performed': False}
    s._cache = OrderedDict(); s._leased = {}; s._pending_uploads = {}; s._pending_host = {}; s._cache_bytes = 0
    s._lock = threading.RLock(); s.ple_tables = {}; s._parked = s._active = s._closed = False
    s._error = None; s._owner = None; s._synchronize = lambda: None
    s.pool = SimpleNamespace(failure=None, counters=lambda: {'weight_read_inflight': 0}, close=lambda: None)
    s.q1_root = '/q1'; s.root = '/source'
    s.hot = frozenset(hot if hot is not None else ((0, e) for e in range(8)))
    s.index = lfu.LFUIndex(); s.packed_buffers = False
    s.tables = {f'blk.{l}': tuple((l, e) for e in range(EXPERTS)) for l in range(LAYERS)}
    s.files = {k: {} for keys in s.tables.values() for k in keys}
    s.artifact = {'tensors': defaultdict(lambda: {'file': 'shard'}), 'seals': defaultdict(dict)}
    s.q1_records = defaultdict(lambda: {'file': 'q1', 'size': Q})
    policy = dict(mode='hot_original_q1', budget_bytes=maximum, hot_bytes=8*O, q1_root='/q1', hotset_path='/hot')
    if mode != 'shared_v1':
        policy.update(residency=mode, high_fraction=fraction, pre_evict_slots=slots)
    policy.update(extra or {})
    s.policy = policy
    s.pool_bytes = {'original': 0, 'q1': 0}; s.peak_payload_bytes = 0
    s.stats = dict(cache_hits=0, cache_misses=0, bytes_read=0, evictions=0, uploaded_experts=0, upload_batches=0,
                   demand_wait_ns=0, lookups=0, windows=0, parks=0, activations=0, original_hits=0, original_misses=0,
                   q1_hits=0, q1_misses=0, original_bytes_read=0, q1_bytes_read=0, q1_token_selections=0,
                   original_token_selections=0)
    s.binding = 'c'*64; s.hot_binding = 'a'*64; s.q1_binding = 'b'*64; s.hot_map_provenance = None
    s._init_residency()
    box = [available if available is not None else 1 << 40]
    s.available = box
    s.configure_adaptive_residency(available_bytes=lambda: box[0], working_set_experts=1)
    s.reads = []

    def read(key, function, *args):
        s.reads.append((key, function.__name__))
        return raw_for(key, function.__name__)
    s._read_expert = read
    return s


def use(s, *keys):
    with s.window():
        for layer, expert in keys:
            s.observe_routes(f'blk.{layer}', np.array([[expert]], np.uint32))
            s.acquire(f'blk.{layer}', expert)


def forward(s, *keys):
    use(s, *keys)
    s.end_context()


def assert_exact(s):
    size = lambda arrays: sum(v.nbytes for v in arrays.values())
    assert s._cache_bytes == sum(size(v) for v in s._cache.values())
    for kind in ('original', 'q1'):
        assert s.pool_bytes[kind] == sum(size(v) for k, v in s._cache.items() if s.kind(k) == kind)
    for key, arrays in s._cache.items():
        assert size(arrays) == (O if s.kind(key) == 'original' else Q)


HOT = [(0, e) for e in range(8)]
COLD = [(1, e) for e in range(16)]
CHURN = COLD+[(0, e) for e in range(8, 16)]   # 24 Q1-kind experts > the Q1 pool


# ---- Policy surface -----------------------------------------------------

def base(**extra):
    return {**dict(mode='hot_original_q1', budget_bytes=75_000_000_000, hot_bytes=60_000_000_000,
                   q1_root='/q1', hotset_path='/hot.json'), **extra}


def test_hot_q1_default_selects_split_pools_and_explicit_shared_is_preserved():
    value = base()
    parsed = lfu.parse_policy(value)
    assert parsed == {**value, 'residency': 'split_v1'}
    assert residency.settings(parsed) == dict(residency='split_v1', cache_maximum='explicit',
                                              high_fraction='auto', pre_evict_slots=0)
    shared = base(residency='shared_v1')
    assert lfu.parse_policy(shared) == shared
    assert residency.settings(shared) == dict(residency='shared_v1', cache_maximum='explicit',
                                              high_fraction=None, pre_evict_slots=0)
    assert lfu.parse_policy(value, adaptive=False).get('residency') is None
    for extra in (dict(residency='split_v1'), dict(residency='split_v1', high_fraction=0.7, pre_evict_slots=0),
                  dict(residency='dynamic_precision_v1', promote_slots=4, promote_decay=0.9),
                  dict(cache_maximum='live_v1', live_reserve_bytes=8 << 30, budget_bytes=110_000_000_000)):
        value = base(**extra)
        expected = {'residency': 'split_v1', **value}
        assert lfu.parse_policy(value) == expected
        assert source_policy({'deepseek_source_experts': value})['deepseek_source_experts'] == expected


@pytest.mark.parametrize('extra,match', [
    (dict(residency='shared_v1', high_fraction=0.5), 'require split_v1'),
    (dict(residency='shared_v1', pre_evict_slots=2), 'require split_v1'),
    (dict(residency='split_v1', promote_slots=2), 'dynamic_precision_v1'),
    (dict(residency='split_v1', high_fraction=1.0), 'high_fraction'),
    (dict(residency='split_v1', high_fraction=0.01), 'high_fraction'),
    (dict(residency='split_v1', high_fraction=1), 'high_fraction'),
    (dict(residency='split_v1', high_fraction=True), 'high_fraction'),
    (dict(residency='split_v1', pre_evict_slots=17), 'pre_evict_slots'),
    (dict(residency='split_v1', pre_evict_slots=True), 'pre_evict_slots'),
    (dict(residency='split_v2'), 'residency'),
    (dict(live_reserve_bytes=8 << 30), 'live_reserve_bytes'),
    (dict(budget_bytes=100_000_000_000), '90GB'),
    (dict(cache_maximum='live_v1', budget_bytes=130_000_000_000), '90GB'),
    (dict(residency='dynamic_precision_v1', promote_decay=1.0), 'promote_decay'),
])
def test_invalid_residency_settings_fail_closed(extra, match):
    with pytest.raises(ValueError, match=match):
        lfu.parse_policy(base(**extra))


def test_split_needs_q1_profile_and_adaptive_residency():
    with pytest.raises(ValueError):
        lfu.parse_policy({'mode': 'original_lfu', 'budget_bytes': 8 << 30, 'residency': 'split_v1'})
    assert lfu.parse_policy({'mode': 'original_lfu', 'budget_bytes': 100_000_000_000, 'cache_maximum': 'live_v1'})
    with pytest.raises(ValueError, match='adaptive'):
        lfu.parse_policy(base(budget_bytes=75_000_000_000, hot_bytes=60_000_000_000, residency='split_v1'), adaptive=False)


def test_auto_high_fraction_is_hot_map_bytes_over_maximum():
    payload = 75_000_000_000-lfu.ORIGINAL_BYTES
    value, source = residency.resolve_high_fraction('auto', hot_objects=3191, original_bytes=lfu.ORIGINAL_BYTES,
                                                    payload_maximum=payload)
    assert source == 'auto_hot_map_bytes_over_maximum' and abs(value-0.8001) < 1e-3
    assert residency.resolve_high_fraction('auto', hot_objects=10**6, original_bytes=1, payload_maximum=1)[0] == 0.95
    assert residency.resolve_high_fraction(0.6, hot_objects=1, original_bytes=1, payload_maximum=1) == (0.6, 'explicit')


# ---- Exact split pools ---------------------------------------------------

def test_pool_isolation_q1_churn_never_evicts_hot_originals(small):
    s = make(fraction=0.5, slots=0)         # payload 11*O: high 16896 (5 O), q1 16896 (16 Q)
    targets = s.pool_targets()
    forward(s, *HOT[:5])
    originals = {k: s._cache[k] for k in HOT[:5]}
    for key in CHURN:
        forward(s, key)
    assert all(s._cache.get(k) is v for k, v in originals.items())
    assert s.pool_bytes['q1'] <= targets['q1'] and s.stats['sync_evictions_original'] == 0
    q1_resident = {k: s._cache[k] for k in CHURN if k in s._cache}
    assert q1_resident and s.stats['sync_evictions_q1'] > 0
    forward(s, *HOT[5:8])                   # originals now evict only originals
    assert s.pool_bytes['original'] <= targets['original']
    assert all(s._cache.get(k) is v for k, v in q1_resident.items())
    assert s.stats['sync_evictions_original'] == 3
    assert_exact(s)


def test_shared_mode_still_lets_q1_evict_originals(small):
    s = make(mode='shared_v1', maximum=6*O)
    forward(s, *HOT[:4])
    for key in COLD[:12]:
        forward(s, key)
    assert sum(k in s._cache for k in HOT[:4]) < 4
    assert s.residency_receipt()['pools'] == 'shared_total_adaptive' and 'sync_evictions' not in s.stats


@pytest.mark.parametrize('fraction', [0.25, 0.5, 0.75])
def test_fractions_follow_live_target_and_never_violate_leases(small, fraction):
    s = make(fraction=fraction, slots=0)
    for key in HOT:                         # one forward each: own-pool LFU eviction
        forward(s, key)
    for key in COLD:
        forward(s, key)
    full = s.pool_targets()
    assert full['original'] == int(sum(full.values())*fraction)
    assert s.pool_bytes['original'] <= full['original'] and s.pool_bytes['q1'] <= full['q1']
    # Live memory drops: both targets shrink in proportion and trim by kind.
    s.available[0] = 0
    with s.window():
        leased = HOT[0]
        s.acquire('blk.0', leased[1])
        held = s._cache[leased]
        small_targets = s.pool_targets()
        assert sum(small_targets.values()) < sum(full.values())
        assert small_targets['original'] == int(sum(small_targets.values())*fraction)
        assert s._cache.get(leased) is held   # an active lease survives the shrink
    for kind in ('original', 'q1'):
        assert s.pool_bytes[kind] <= max(small_targets[kind], O if kind == 'original' else 0)
    # Memory returns: targets grow back to the configured maximum.
    s.available[0] = 1 << 40
    s.adaptive_residency_update()
    assert s.pool_targets() == full
    assert_exact(s)


def test_pinned_bank_objects_are_never_trimmed(small):
    s = make(fraction=0.5, slots=4)
    forward(s, *HOT[:5])
    s._resident_bank_pinned_keys = frozenset(HOT[:5])
    s.available[0] = 0
    s.adaptive_residency_update()
    forward(s, COLD[0])
    assert all(k in s._cache for k in HOT[:5])
    assert_exact(s)


# ---- Pre-eviction --------------------------------------------------------

def test_pre_eviction_is_bounded_and_spares_touched_pinned_and_hinted(small):
    s = make(fraction=0.5, slots=0)
    for key in CHURN[:16]:                  # fill the Q1 pool to its reservation
        forward(s, key)
    assert s.pool_bytes['q1'] == 16*Q and s.stats['pre_evicted_objects'] == 0
    s._pre_evict_slots = 2; s._miss_ewma['q1'] = 5.0   # a burst of misses is expected
    before = s.stats['pre_evicted_objects']
    recent = CHURN[15]
    pinned = next(k for k in CHURN if k in s._cache and k != recent)
    hinted = next(k for k in CHURN if k in s._cache and k not in (recent, pinned))
    s._resident_bank_pinned_keys = frozenset({pinned})
    s._overlap = {'hints': OrderedDict({hinted: None}), 'ticket': None}
    s._cycle_touched = {recent}
    s._cycle_pending = True
    s._run_cycle_boundary()
    evicted = s.stats['pre_evicted_objects']-before
    assert evicted == 2                     # min(slots, expected shortfall)
    assert s.stats['pre_evicted_bytes'] >= evicted*Q
    assert all(k in s._cache for k in (recent, pinned, hinted))
    assert s._pre_evict_credit['q1'] > 0
    del s._overlap
    sync = s.stats['sync_evictions']
    use(s, CHURN[20])
    assert s.stats['sync_evictions_avoided'] >= 1 and s.stats['sync_evictions'] == sync
    assert_exact(s)


def test_pre_eviction_disabled_and_unpressured_pools_do_nothing(small):
    s = make(fraction=0.5, slots=0)
    for key in COLD:
        forward(s, key)
    assert s.stats['pre_evicted_objects'] == 0
    t = make(fraction=0.5, slots=4)
    forward(t, COLD[0])                     # nearly empty pools: no pressure
    assert t.stats['pre_evicted_objects'] == 0 and t.stats['pre_evict_pressure_events'] == 0


def test_memory_pressure_pre_evicts_at_most_slots_per_pool(small):
    s = make(fraction=0.5, slots=1)
    forward(s, *HOT[:4], *COLD[:8])
    s._adaptive_headroom = 1 << 50          # live memory is below the headroom target
    s.available[0] = 1 << 40
    before = s.stats['pre_evicted_objects']
    s._cycle_pending = True
    s._run_cycle_boundary()
    assert s.stats['pre_evict_memory_pressure_events'] >= 1
    assert s.stats['pre_evicted_objects']-before <= 2


def test_boundary_waits_for_an_open_completion_window(small):
    s = make(fraction=0.5, slots=2)
    with s.window():
        s.acquire('blk.1', 0)
        s.end_context()                     # forward done, completion window still open
        assert s._cycle_pending and s.stats['residency_cycles'] == 0
    assert not s._cycle_pending and s.stats['residency_cycles'] == 1


def test_failed_upload_poisons_store_and_keeps_accounting_exact(small, monkeypatch):
    s = make(fraction=0.5, slots=2)
    forward(s, *HOT[:3], *COLD[:6])
    snapshot = (s._cache_bytes, dict(s.pool_bytes))
    import mlx.core as mx
    real = mx.eval
    monkeypatch.setattr(mx, 'eval', lambda *a: (_ for _ in ()).throw(RuntimeError('injected upload failure')))
    with pytest.raises(RuntimeError, match='injected'):
        use(s, COLD[10])
    monkeypatch.setattr(mx, 'eval', real)
    assert s._error is not None
    assert (s._cache_bytes, s.pool_bytes) == snapshot
    with pytest.raises(ValueError, match='failed'):
        use(s, COLD[11])
    s.end_context()                         # a poisoned store runs no boundary work
    assert_exact(s)


# ---- Live-memory maximum -------------------------------------------------

PLAN = dict(dense_payload_bytes=20 << 30, dense_prepared_bound_bytes=4 << 30,
            dense_staging_bound_bytes=1 << 30, metadata_bound_bytes=256 << 20)


@pytest.mark.parametrize('available,expected,reason', [
    (200 << 30, 110_000_000_000, 'clamped_to_maximum'),
    (100 << 30, (100 << 30)-(25 << 30)-(256 << 20)-(16 << 30), 'derived_from_live_memory'),
    (10 << 30, 2*lfu.ORIGINAL_BYTES, 'live_memory_below_progress_floor'),
    (None, 75_000_000_000, 'availability_unknown_fallback'),
    (-1, 75_000_000_000, 'availability_unknown_fallback'),
])
def test_live_maximum_derivation(available, expected, reason):
    value, receipt = residency.derive_live_maximum(110_000_000_000, PLAN, reserve_bytes=16 << 30,
        available_bytes=lambda: available, minimum=2*lfu.ORIGINAL_BYTES)
    assert value == expected and receipt['reason'] == reason and receipt['derived_maximum_bytes'] == value
    assert receipt['availability_known'] is (type(available) is int and available >= 0)


def test_live_maximum_fails_safe_when_probe_raises():
    def broken():
        raise OSError('no probe')
    value, receipt = residency.derive_live_maximum(60_000_000_000, PLAN, reserve_bytes=1 << 30,
        available_bytes=broken, minimum=1)
    assert value == 60_000_000_000 and not receipt['availability_known']


def test_source_admission_derives_maximum_but_keeps_explicit_path(monkeypatch):
    monkeypatch.setattr(lfu, 'weight_plan', lambda artifact, budget: dict(PLAN, minimum_expert_cache_bytes=lfu.ORIGINAL_BYTES,
        total_reservation_bytes=budget, expert_cache_bytes=budget))
    request = WeightStreamRequest(mode='moe_experts', budget_bytes=110_000_000_000, admission_limit_bytes=None)
    live = base(cache_maximum='live_v1', budget_bytes=110_000_000_000)
    result = lfu.source_admission({}, request, live, available_bytes=lambda: 128 << 30)
    assert result['expert_cache_bytes'] == (128 << 30)-(25 << 30)-(256 << 20)-(16 << 30)
    assert result['expert_cache_maximum']['policy'] == 'live_v1'
    explicit = base()
    request = WeightStreamRequest(mode='moe_experts', budget_bytes=75_000_000_000, admission_limit_bytes=None)
    result = lfu.source_admission({}, request, explicit, available_bytes=lambda: pytest.fail('explicit path probed'))
    assert result['expert_cache_bytes'] == 75_000_000_000 and 'expert_cache_maximum' not in result


# ---- Hot-map recalibration ----------------------------------------------

def guarded_policy(tmp_path, monkeypatch, hot_path):
    q1 = tmp_path/'q1'; q1.mkdir(exist_ok=True)
    manifest = {'source': defaults.SOURCE_FINGERPRINT, 'dense_modified': False,
                'quantization': 'source-q1-affine-g64-lloyd4-f16-v1', 'experts': []}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    monkeypatch.setattr(defaults, 'Q1_FINGERPRINT', digest)
    (q1/'manifest.json').write_text(json.dumps({**manifest, 'fingerprint': digest}))
    return {'deepseek_source_experts': {'mode': 'hot_original_q1', 'hot_bytes': defaults.HOT_BYTES,
            'budget_bytes': 75_000_000_000, 'q1_root': str(q1), 'hotset_path': str(hot_path)}}


def atlas(tmp_path, source=defaults.SOURCE_FINGERPRINT):
    path = tmp_path/'atlas.json'
    rows = [{'layer': l, 'expert': e, 'token_selections': (l*384+e) % 97+1, 'last_access': 1}
            for l in range(40) for e in range(384)]
    path.write_text(json.dumps({'schema': 'mlx2.expert-frequency-atlas.v1', 'source': source, 'clock': 5, 'counts': rows}))
    return path


def test_guard_accepts_schema_valid_recalibrated_map(tmp_path, monkeypatch):
    out = tmp_path/'recal.json'
    built = hotmap.build_recalibrated(atlas(tmp_path), out, hot_bytes=defaults.HOT_BYTES)
    assert built['original_objects'] == defaults.HOT_BYTES//hotmap.ORIGINAL_EXPERT_BYTES
    policy = guarded_policy(tmp_path, monkeypatch, out)
    assert defaults.default_route_policy_applies('ordinary', policy) is True
    receipt = defaults.hot_map_provenance(out.read_bytes(), json.loads(out.read_text()))
    assert receipt['kind'] == 'recalibrated' and receipt['sha256'] == built['hot_map_sha256']
    assert receipt['recalibration']['atlas_sha256'] == hashlib.sha256((tmp_path/'atlas.json').read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        hotmap.build_recalibrated(atlas(tmp_path), out, hot_bytes=defaults.HOT_BYTES)
    # Exact split residency keeps the selected scope; dynamic precision does not.
    split = json.loads(json.dumps(policy)); split['deepseek_source_experts'].update(residency='split_v1')
    assert defaults.default_route_policy_applies('ordinary', split) is True
    dynamic = json.loads(json.dumps(policy)); dynamic['deepseek_source_experts'].update(residency='dynamic_precision_v1')
    assert defaults.default_route_policy_applies('ordinary', dynamic) is False


@pytest.mark.parametrize('mutate', [
    lambda m: m.update(source='f'*64),
    lambda m: m.update(hot_budget_bytes=defaults.HOT_BYTES-1),
    lambda m: m.update(schema='mlx2.source-expert-hot-map.v0'),
    lambda m: m.pop('recalibration'),
    lambda m: m['recalibration'].update(atlas_sha256='nothex'),
    lambda m: m['experts'].append(list(m['experts'][0])),
    lambda m: m['experts'].append([40, 0]),
    lambda m: m.update(original_expert_bytes=1),
])
def test_guard_refuses_invalid_recalibrated_maps(tmp_path, monkeypatch, mutate):
    out = tmp_path/'recal.json'
    hotmap.build_recalibrated(atlas(tmp_path), out, hot_bytes=defaults.HOT_BYTES)
    value = json.loads(out.read_text()); mutate(value); out.write_text(json.dumps(value))
    assert defaults.default_route_policy_applies('ordinary', guarded_policy(tmp_path, monkeypatch, out)) is False


def test_measured_map_provenance_stays_distinct(tmp_path, monkeypatch):
    hot = tmp_path/'measured.json'
    hot.write_text(json.dumps({'source': defaults.SOURCE_FINGERPRINT, 'hot_budget_bytes': defaults.HOT_BYTES}))
    monkeypatch.setattr(defaults, 'HOT_MAP_SHA256', hashlib.sha256(hot.read_bytes()).hexdigest())
    assert defaults.default_route_policy_applies('ordinary', guarded_policy(tmp_path, monkeypatch, hot)) is True
    assert defaults.hot_map_provenance(hot.read_bytes(), json.loads(hot.read_text()))['kind'] == 'measured'
    hot.write_text(hot.read_text()+' ')
    assert defaults.hot_map_provenance(hot.read_bytes(), json.loads(hot.read_text()))['kind'] == 'unrecognized'


# ---- Dynamic precision ---------------------------------------------------

class ImmediateReader:
    """A promotion reader whose original-expert reads complete at once."""
    def __init__(self, fail=False):
        self.calls = []; self.fail = fail
    def submit(self, function, artifact, names):
        parts = list(names)[0].split('.')
        key = (int(parts[1]), int(parts[4]))
        self.calls.append((function.__name__, key)); future = Future()
        if self.fail:
            future.set_exception(OSError('injected promotion read'))
        else:
            future.set_result(raw_for(key, 'read_tensors'))
        return future
    def close(self):
        pass


def dynamic(**extra):
    s = make(mode='dynamic_precision_v1', fraction=0.5, slots=0,
             extra=dict(promote_slots=4, promote_min_score=2.0, promote_hysteresis=1.5, promote_decay=0.9, **extra))
    s._promotion_pool = ImmediateReader()
    return s


def test_dynamic_serves_q1_stand_in_on_miss_even_for_hot_map_experts(small):
    s = dynamic()
    forward(s, HOT[0])
    assert s.kind(HOT[0]) == 'q1' and s.reads[-1] == (HOT[0], 'q1_read')
    assert s.pool_bytes == {'original': 0, 'q1': Q}


def test_dynamic_promotion_takes_effect_only_at_a_cycle_boundary(small):
    s = dynamic()
    key = COLD[3]
    with s.window():                        # one forward: three selections, then more
        for _ in range(3):
            s.observe_routes('blk.1', np.array([[key[1]]], np.uint32))
            s.acquire('blk.1', key[1])
        assert s.kind(key) == 'q1'
    assert s.kind(key) == 'q1' and not s._promotions
    s.end_context()                         # boundary 1: background read issued
    assert key in s._promotions and s.kind(key) == 'q1'
    with s.window():
        s.observe_routes('blk.1', np.array([[key[1]]], np.uint32))
        assert s.acquire('blk.1', key[1]) is s._cache[key] and s.kind(key) == 'q1'   # still Q1 in-forward
    s.end_context()                         # boundary 2: install MXFP4, drop Q1 copy
    assert s.kind(key) == 'original' and key in s._dyn_high
    assert s.pool_bytes == {'original': O, 'q1': 0}
    assert s.stats['promotions'] == 1 and s.stats['promotion_bytes'] == O
    row = s.residency_receipt()['dynamic']['history'][-1]
    assert row['promotions'] == 1 and row['q1_selections'] == 1
    forward(s, key)
    assert s.residency_receipt()['dynamic']['history'][-1]['mxfp4_selections'] == 1
    assert_exact(s)


def test_dynamic_high_pool_is_frozen_mid_forward_and_demoted_at_boundary(small):
    s = dynamic()
    key = COLD[0]
    for _ in range(3):
        forward(s, key, key)
    assert key in s._dyn_high
    s.available[0] = 0                      # live target collapses mid-forward
    with s.window():
        s.acquire('blk.1', 1)
        assert key in s._dyn_high and key in s._cache and s.kind(key) == 'original'
    s.end_context()                         # pressure demotion happens here only
    assert key not in s._dyn_high and s.stats['demotions'] >= 1 and s.kind(key) == 'q1'
    assert_exact(s)


def test_dynamic_promotion_read_failure_disables_but_never_poisons(small):
    s = dynamic()
    s._promotion_pool = ImmediateReader(fail=True)
    for _ in range(3):
        forward(s, COLD[2], COLD[2])
    assert s._promotions_disabled and s.stats['promotion_read_failures'] == 1 and s._error is None
    forward(s, COLD[2])


def test_dynamic_promotion_upload_failure_poisons(small, monkeypatch):
    s = dynamic()
    forward(s, COLD[2], COLD[2], COLD[2])
    assert COLD[2] in s._promotions and COLD[2] not in s._dyn_high
    use(s, COLD[2])                         # a resident Q1 hit: no upload
    import mlx.core as mx
    monkeypatch.setattr(mx, 'eval', lambda *a: (_ for _ in ()).throw(RuntimeError('injected promotion upload')))
    with pytest.raises(RuntimeError, match='injected'):
        s.end_context()
    assert s._error is not None


def test_dynamic_identity_and_apc_revision_are_salted(small):
    exact, split, dyn = make(mode='shared_v1'), make(), dynamic()
    assert exact.apc_revision() == split.apc_revision()
    assert dyn.apc_revision() == exact.apc_revision()+':residency=dynamic_precision_v1'
    assert exact.identity_hot_binding() == exact.hot_binding
    salted = dyn.identity_hot_binding()
    assert salted != dyn.hot_binding and len(salted) == 64
    assert dyn.receipt.__qualname__.startswith('LFUSourceExpertStore')


@pytest.mark.parametrize('extra', [
    {'deepseek_physical_batch': 'b2'},
    {'draft_model': 'embedded_dspark'},
    {'draft_model': 'prompt_lookup', 'deepseek_verify_commit': 'local_frontier'},
    {'deepseek_source_prefill_partition': 'canonical_m1_v1'},
    {'deepseek_source_prefill_partition': 'position_grid_v1'},
    {'deepseek_source_prefill_order': 'layer_major_v1'},
    {'deepseek_ced': 'dependency_exact'},
    {'deepseek_source_resident_gather': 'resident_bank_v1', 'deepseek_source_expert_math': 'routed_fused_fp32'},
])
def test_dynamic_refuses_fixed_precision_contracts(extra):
    value = {'deepseek_source_experts': base(residency='dynamic_precision_v1'), **extra}
    with pytest.raises(ValueError):
        source_policy(value)
    split = {'deepseek_source_experts': base(residency='split_v1'), **extra}
    try:
        source_policy(split)
    except ValueError as error:
        assert 'dynamic_precision_v1' not in str(error)


def test_dynamic_admits_ordinary_b1_with_overlap():
    value = source_policy({'deepseek_source_experts': base(residency='dynamic_precision_v1'),
                           'deepseek_source_expert_io': 'overlap_v1'})
    assert value['deepseek_source_experts']['residency'] == 'dynamic_precision_v1'


def test_dynamic_disables_apcv2_publication_and_reuse(monkeypatch):
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter
    from mlx2.adapters.deepseek_v41_mlx import DeepSeekV41MLXAdapter
    adapter = DeepSeekV41SourceAdapter.__new__(DeepSeekV41SourceAdapter)
    adapter.kernel_policy = {'deepseek_source_experts': base(residency='dynamic_precision_v1')}
    monkeypatch.setattr(DeepSeekV41MLXAdapter, 'execution_config', lambda self, **k: {})
    config = adapter.execution_config(max_lanes=1, prefill_step=128)
    assert config['apcv2_reuse_disabled_reason'] == residency.DYNAMIC_APC_REASON
    assert adapter.apc_publication_refusal([], []) == residency.DYNAMIC_APC_REASON
    adapter.kernel_policy = {'deepseek_source_experts': base(residency='split_v1')}
    assert adapter.execution_config(max_lanes=1, prefill_step=128)['apcv2_reuse_disabled_reason'] == 'adaptive_prefill_partitions_not_qualified'
    assert adapter.apc_publication_refusal([], []) is None


def test_receipts_report_pools_and_precision_mix(small):
    s = dynamic()
    forward(s, COLD[1], HOT[1])
    receipt = s.receipt()
    assert receipt['expert_residency']['mode'] == 'dynamic_precision_v1'
    assert receipt['numeric_profile'] == 'source_dynamic_precision_mxfp4_q1g64_v1'
    assert receipt['pool_capacity_policy'] == 'split_kind_pools_adaptive'
    assert receipt['expert_residency']['dynamic']['apcv2'].startswith('disabled')
    split = make()
    forward(split, COLD[1])
    value = split.receipt()['expert_residency']
    assert value['high_fraction'] == 0.5 and value['last_cycle']['misses'] == {'original': 0, 'q1': 1}
