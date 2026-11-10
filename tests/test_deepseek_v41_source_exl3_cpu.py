"""DS4.1 Source routed experts from an EXL3 bank (deepseek_source_experts.exl3).

CPU only (MLX pinned to the CPU device). Synthetic EXL3 tensors (random trellis
words are valid codes) are decoded by the independent host decoder
(exl3_reference.effective_weight) for the NumPy oracle; the runtime path runs
through ``_moe_routes`` of SourceV41Weights with a stub provider.
"""
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_source_exl3 as ex
from mlx2.adapters import deepseek_v41_source_lfu as lfu
from mlx2.runtime.models.exl3_reference import effective_weight
from mlx2.runtime.trellis.upstream import CODEBOOKS

SOURCE = 'c' * 64
GB = 10 ** 9
H, I = 256, 128          # small hidden / intermediate for the routed-math oracle
WORDS = 48


@pytest.fixture(scope='module')
def mx():
    import mlx.core as mx
    previous = mx.default_device(); mx.set_default_device(mx.cpu)
    yield mx
    mx.set_default_device(previous)


# ---- Synthetic experts -----------------------------------------------------------

def exl3_host(rng, i, o, words=WORDS):
    trellis = rng.integers(0, 1 << 16, (i // 16, o // 16, words), dtype=np.uint32).astype(np.uint16)
    suh = (rng.choice([-1.0, 1.0], i) * rng.uniform(0.5, 1.5, i)).astype(np.float16) * np.float16(0.05)
    svh = (rng.choice([-1.0, 1.0], o) * rng.uniform(0.5, 1.5, o)).astype(np.float16) * np.float16(0.05)
    return trellis, suh, svh


def small_exl3(mx, seed, law='reference'):
    rng = np.random.default_rng(seed)
    arrays, dense = {}, {}
    for kind, (i, o) in (('gate', (H, I)), ('up', (H, I)), ('down', (I, H))):
        t, suh, svh = exl3_host(rng, i, o)
        arrays[kind + '.trellis'] = mx.array(t[None]); arrays[kind + '.suh'] = mx.array(suh[None])
        arrays[kind + '.svh'] = mx.array(svh[None])
        dense[kind] = effective_weight(t, suh, svh, 'mcg')            # [in, out] float64
    expert = ex.Exl3Expert(arrays); expert.words, expert.law = WORDS, law
    return expert, dense


def small_original(mx, seed):
    rng = np.random.default_rng(seed)
    arrays, dense = {}, {}
    for kind, (n, k) in (('gate', (I, H)), ('up', (I, H)), ('down', (H, I))):
        w = mx.array((rng.standard_normal((n, k)) * 0.05).astype(np.float32))
        q, s = mx.quantize(w, mode='mxfp4')[:2]
        arrays[kind + '.qweight'], arrays[kind + '.scales'] = q, s
        dense[kind] = np.array(mx.dequantize(q, s, mode='mxfp4').astype(mx.float32), np.float64).T   # [in, out]
    return arrays, dense


def oracle(mx, x, ids, scores, dense):
    """NumPy float64 projections; the Source middle/rounding/QAT ops; expert-ID combine."""
    from mlx2.runtime.models.deepseek_v4 import limited_swiglu
    from mlx2.runtime.models.deepseek_v41_quant import quantize
    rnd = lambda v: np.array(mx.array(np.asarray(v, np.float32)).astype(mx.bfloat16).astype(mx.float32))
    xq = np.array(quantize(mx.array(x), 'fp8').astype(mx.float32), np.float64)
    out = np.zeros((len(ids), H), np.float32)
    for r, row in enumerate(ids):
        for slot in np.argsort(row, kind='stable'):
            W = dense[int(row[slot])]
            g, u = rnd(xq[r:r + 1] @ W['gate']), rnd(xq[r:r + 1] @ W['up'])
            mid = rnd(np.array(limited_swiglu(mx.array(g), mx.array(u))) * scores[r, slot])
            dq = np.array(quantize(mx.array(mid), 'fp8').astype(mx.float32), np.float64)
            out[r] = out[r] + rnd(dq @ W['down'])[0]
    return out


def rule_for(records_keys, **settings):
    records = {k: {'file': 'f', 'sha256': '0' * 64, 'size': 1} for k in records_keys}
    base = dict(root='/bank', scope='all', pool_bytes=0, layers=[0, 1023], math='reference')
    base.update(settings)
    return ex.Exl3Rule(base, root=Path('/bank'), records=records, fingerprint='f' * 64, words=WORDS, meta={},
                       layers=40, source_binding=SOURCE)


def weights_case(mx, bank, rule, *, policy='reference', batch='reference'):
    from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
    acquired = []
    store = SimpleNamespace(_active=True, _lock=threading.RLock(), _leased={}, stats={'lookups': 0},
                            packed_buffers=False)
    store._ready = lambda: None

    def acquire(prefix, e):
        acquired.append(e)
        key = (0, e)
        if rule.serves(key):
            rule.record_served(key)
        else:
            rule.record_fallback(key)
        return bank[e]
    store.acquire = acquire
    store.exl3_route = lambda prefix, values: frozenset(rule.route(0, np.asarray(values).reshape(-1)))
    w = object.__new__(SourceV41Weights); w._ready = lambda: None
    w.artifact = {'args': SimpleNamespace(hidden_size=H, num_experts_per_tok=6)}
    w.expert_math_policy = policy; w.scheduled_expert_calls = {}; w.expert_stream = store
    w.source_prefill_batch_policy = batch
    return SimpleNamespace(w=w, store=store, acquired=acquired)


def build(mx, exl3_ids, original_ids):
    bank, dense = {}, {}
    for e in exl3_ids:
        bank[e], dense[e] = small_exl3(mx, 100 + e)
    for e in original_ids:
        bank[e], dense[e] = small_original(mx, 200 + e)
    return bank, dense


def inputs(rows, seed=1):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((rows, H)) * 2).astype(np.float32)
    scores = rng.uniform(0.05, 0.4, (rows, 6)).astype(np.float32)
    return x, scores


def close(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert np.isfinite(a).all() and np.abs(b).max() > 0
    assert np.abs(a - b).max() <= 1e-2 * np.abs(b).max(), np.abs(a - b).max() / np.abs(b).max()
    assert np.linalg.norm(a - b) <= 2e-3 * np.linalg.norm(b)


# ---- 1. Projection law ------------------------------------------------------------

def test_project_equals_effective_weight(mx):
    expert, dense = small_exl3(mx, 3)
    x = np.random.default_rng(4).standard_normal((5, H)).astype(np.float32)
    for kind in ('gate', 'up'):
        y = np.array(ex.project(expert, kind, mx.array(x)))
        assert y.dtype == np.float32
        ref = x.astype(np.float64) @ dense[kind]
        assert np.allclose(y, ref, rtol=1e-4, atol=1e-5 * np.abs(ref).max())
    with pytest.raises(ValueError, match='geometry'):
        ex.project(expert, 'down', mx.array(x))                       # down takes I-wide rows
    with pytest.raises(ValueError, match='geometry'):
        ex.project({'gate.suh': None}, 'gate', mx.array(x))           # not an EXL3 object


# ---- 2. Routed math through _moe_routes -------------------------------------------------

@pytest.mark.parametrize('rows', [1, 5])
def test_all_exl3_rows_match_oracle(mx, rows):
    ids = np.array([[9, 1, 7, 2, 6, 0]] * rows, np.uint32)
    if rows > 1:
        ids[1:] = np.random.default_rng(2).permutation(np.tile(np.arange(10, dtype=np.uint32), (rows - 1, 1)).T).T[:, :6]
    bank, dense = build(mx, np.unique(ids).tolist(), [])
    rule = rule_for([(0, int(e)) for e in np.unique(ids)])
    case = weights_case(mx, bank, rule)
    x, scores = inputs(rows)
    out = case.w.moe(mx.array(x), mx.array(ids), mx.array(scores), 'blk.0'); mx.eval(out)
    close(np.array(out), oracle(mx, x, ids, scores, dense))
    calls = case.w.scheduled_expert_calls
    assert calls['official_source_exl3_experts'] == len(np.unique(ids))
    assert calls['official_source_exl3_rows'] == ids.size
    assert sum(rule.slots['exl3']) == ids.size and sum(rule.slots['original']) == 0
    assert rule.stats['fallback_acquires'] == 0 and not rule.missing


def test_mixed_exl3_and_original_rows_and_fallback_accounting(mx):
    ids = np.array([[9, 1, 7, 2, 6, 0], [3, 9, 4, 1, 8, 5], [0, 2, 4, 6, 8, 3]], np.uint32)
    exl3_ids, original_ids = [0, 1, 2, 3, 9], [4, 5, 6, 7, 8]
    bank, dense = build(mx, exl3_ids, original_ids)
    rule = rule_for([(0, e) for e in exl3_ids])
    case = weights_case(mx, bank, rule)
    x, scores = inputs(3, seed=5)
    out = case.w.moe(mx.array(x), mx.array(ids), mx.array(scores), 'blk.0'); mx.eval(out)
    close(np.array(out), oracle(mx, x, ids, scores, dense))
    assert case.w.scheduled_expert_calls['official_source_exl3_experts'] == 5
    n_exl3 = int(np.isin(ids, exl3_ids).sum())
    assert rule.slots['exl3'][0] == n_exl3 and rule.slots['original'][0] == ids.size - n_exl3
    assert rule.stats['mixed_calls'] == 1 and rule.stats['exl3_calls'] == 1
    assert rule.served[0] == 5 and rule.fallback[0] == 5
    assert rule.missing_experts() == {'0': original_ids}
    receipt = rule.receipt()
    assert receipt['missing_count'] == 5 and receipt['approximate'] and receipt['qualification'] == 'unqualified'
    assert receipt['per_layer_fallback_acquires'][0] == 5 and receipt['exl3_slot_fraction'] == n_exl3 / ids.size


def test_duplicate_zero_weight_slot(mx):
    # A zero-weight duplicate of a kept expert (route-drop shape) contributes exactly zero.
    ids = np.array([[4, 1, 4, 2, 6, 0]], np.uint32)
    bank, dense = build(mx, [0, 1, 2, 4], [6])
    rule = rule_for([(0, e) for e in (0, 1, 2, 4)])
    case = weights_case(mx, bank, rule)
    x, scores = inputs(1, seed=8)
    scores[0, 2] = 0.0
    out = case.w.moe(mx.array(x), mx.array(ids), mx.array(scores), 'blk.0'); mx.eval(out)
    close(np.array(out), oracle(mx, x, ids, scores, dense))
    dropped = np.array([[4, 1, 2, 6, 0]], np.uint32)
    assert np.allclose(oracle(mx, x, ids, scores, dense), oracle(mx, x, dropped, scores[:, [0, 1, 3, 4, 5]], dense))
    assert case.acquired.count(4) == 1 and rule.slots['exl3'][0] == 5


def test_batched_prefill_kernels_are_bypassed_only_for_exl3_calls(mx, monkeypatch):
    from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
    hits = []
    monkeypatch.setattr(SourceV41Weights, '_prefill_batched_gate_up',
                        lambda self, *a, **k: hits.append(1) or pytest.fail('batched prefill reached'))
    ids = np.array([[9, 1, 7, 2, 6, 0], [3, 9, 4, 1, 8, 5]], np.uint32)
    bank, dense = build(mx, [1], [0, 2, 3, 4, 5, 6, 7, 8, 9])
    rule = rule_for([(0, 1)])
    case = weights_case(mx, bank, rule, batch='gate_up')
    x, scores = inputs(2, seed=6)
    out = case.w.moe(mx.array(x), mx.array(ids), mx.array(scores), 'blk.0'); mx.eval(out)
    close(np.array(out), oracle(mx, x, ids, scores, dense))
    assert case.w.scheduled_expert_calls['official_source_exl3_fused_paths_bypassed'] == 1 and not hits
    # A call with no bank expert keeps the ordinary batched route.
    monkeypatch.setattr(SourceV41Weights, '_prefill_batched_gate_up', lambda self, *a, **k: hits.append(1) or mx.zeros((2, H)))
    case.w.moe(mx.array(x), mx.array(np.array([[0, 2, 3, 4, 5, 6]] * 2, np.uint32)), mx.array(scores), 'blk.0')
    assert hits == [1]


def test_fused_middle_policy_uses_the_fused_middle_and_skips_down_combine(mx, monkeypatch):
    from mlx2.runtime.gguf import source_middle_fusion, source_down_fusion
    from mlx2.adapters import deepseek_v41_source_math as assembly
    from mlx2.runtime.models import deepseek_v41_quant as qat
    seen = []

    def law(gate, up, scores, slots):
        seen.append(gate.shape)
        return gate * mx.sigmoid(up) * scores[slots][:, None]
    monkeypatch.setattr(source_middle_fusion, 'middle', law)
    monkeypatch.setattr(source_down_fusion, 'down_combine', lambda *a: pytest.fail('fused down reached'))
    monkeypatch.setattr(qat, 'fused_quantize', lambda v, kind: v)
    ids = np.array([[9, 1, 7, 2, 6, 0]], np.uint32)
    bank, dense = build(mx, np.unique(ids).tolist(), [])
    rule = rule_for([(0, int(e)) for e in np.unique(ids)])
    case = weights_case(mx, bank, rule, policy='routed_fused_fp32')
    case.store.packed_buffers = True
    x, scores = inputs(1, seed=3)
    out = case.w.moe(mx.array(x), mx.array(ids), mx.array(scores), 'blk.0'); mx.eval(out)
    assert seen == [(1, I)] * 6
    assert case.w.scheduled_expert_calls['official_source_exl3_fused_paths_bypassed'] == 1
    # Oracle with the same patched middle: dense EXL3 projections, expert-ID order.
    rnd = lambda v: np.array(mx.array(np.asarray(v, np.float32)).astype(mx.bfloat16).astype(mx.float32))
    expect = np.zeros((1, H), np.float32)
    for slot in np.argsort(ids[0], kind='stable'):
        W = dense[int(ids[0, slot])]
        g, u = x.astype(np.float64) @ W['gate'], x.astype(np.float64) @ W['up']
        mid = np.array(law(mx.array(g.astype(np.float32)), mx.array(u.astype(np.float32)),
                           mx.array(scores.reshape(-1)), mx.array(np.array([slot], np.uint32))))
        expect = expect + rnd(mid.astype(np.float64) @ W['down'])
    close(np.array(out), expect)


# ---- 3. Bank: manifest, read, store acquire, pool, live bytes ------------------------------

def write_bank(root, keys, *, seed=0, words=WORDS, extra=()):
    from safetensors.numpy import save_file
    rng = np.random.default_rng(seed)
    files = {}
    for layer, expert in keys:
        tensors = {}
        for kind, part in ex.PARTS.items():
            i, o = ex.GEOMETRY[kind]
            t, suh, svh = exl3_host(rng, i, o, words)
            base = f'layers.{layer}.ffn.experts.{expert}.{part}'
            tensors[base + '.trellis'] = t.view(np.int16)
            tensors[base + '.suh'], tensors[base + '.svh'] = suh, svh
            tensors[base + '.mcg'] = np.array([CODEBOOKS['mcg']], np.uint32).view(np.int32)
        rel = f'w{words}/layers.{layer}.ffn.experts.{expert}.safetensors'
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(root / rel))
        files[rel] = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    listed = {k: v for k, v in files.items() if k not in extra}
    manifest = {'source': 'DeepSeek-V4.1-Flash-1017-Source@' + SOURCE, 'format': ex.MANIFEST_FORMAT,
                'bits_per_weight': words / 16, 'selection': 'test', 'files': listed}
    manifest['fingerprint'] = hashlib.sha256(json.dumps(listed, sort_keys=True).encode()).hexdigest()
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    return manifest


@pytest.fixture(scope='module')
def bank_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('exl3bank')
    unlisted = 'w48/layers.1.ffn.experts.5.safetensors'
    manifest = write_bank(root, [(0, 3), (0, 7), (1, 5)], extra=(unlisted,))
    return SimpleNamespace(root=root, manifest=manifest)


def test_manifest_admits_only_listed_files(bank_root):
    root, records, fp, words, meta = ex.load_manifest(bank_root.root, SOURCE)
    assert set(records) == {(0, 3), (0, 7)} and words == 48 and fp == bank_root.manifest['fingerprint']
    assert ex.load_manifest(bank_root.root, SOURCE, expected_fingerprint=fp)[2] == fp
    with pytest.raises(ValueError, match='policy pin'):
        ex.load_manifest(bank_root.root, SOURCE, expected_fingerprint='a' * 64)
    with pytest.raises(ValueError, match='Source checkpoint'):
        ex.load_manifest(bank_root.root, 'd' * 64)


@pytest.mark.parametrize('change,match', [
    (dict(format='exl3 (other)'), 'format'), (dict(fingerprint='0' * 64), 'fingerprint'),
    (dict(bits_per_weight=2.0), 'expert path'), (dict(bits_per_weight=3.03), 'whole tile words'),
    (dict(files={}), 'no files'),
])
def test_manifest_fails_closed(tmp_path, bank_root, change, match):
    import shutil
    shutil.copytree(bank_root.root, tmp_path / 'b')
    manifest = dict(bank_root.manifest, **change)
    if 'files' in change or 'bits_per_weight' in change and 'fingerprint' not in change:
        manifest['fingerprint'] = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
    (tmp_path / 'b/manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        ex.load_manifest(tmp_path / 'b', SOURCE)


def test_read_checks_geometry_and_hash(tmp_path, bank_root):
    root, records, *_ = ex.load_manifest(bank_root.root, SOURCE)
    raw = ex.exl3_read(root, records[(0, 3)], (0, 3), 48)
    assert raw['gate.trellis'].shape == (1, 320, 144, 48) and raw['gate.trellis'].dtype == np.uint16
    assert raw['down.suh'].shape == (1, 2304) and raw['down.svh'].shape == (1, 5120)
    with pytest.raises(ValueError, match='inventory'):
        ex.exl3_read(root, records[(0, 3)], (0, 7), 48)              # file names another expert
    with pytest.raises(ValueError, match='geometry'):
        ex.exl3_read(root, records[(0, 3)], (0, 3), 32)
    with pytest.raises(ValueError, match='admitted hash'):
        from mlx2.adapters import deepseek_v41_mlx_weights as native
        native.clear_bank_verification()
        ex.exl3_read(root, dict(records[(0, 3)], sha256='0' * 64), (0, 3), 48)


def stub_store(rule, *, packed=False):
    store = SimpleNamespace(_leased={}, _pending_uploads={}, _pending_host={}, _lock=threading.RLock(),
                            stats={'bytes_read': 0, 'uploaded_experts': 0}, packed_buffers=packed,
                            exl3=rule, _completion_collector=None)
    store._read_expert = lambda key, function, *args: function(*args)
    return store


def test_store_acquire_pool_and_live_bytes(mx, bank_root):
    root, records, fp, words, meta = ex.load_manifest(bank_root.root, SOURCE)
    size = ex.payload_bytes(48)
    assert size == 13315584
    settings = dict(root=str(root), scope='all', pool_bytes=size, layers=[0, 39], math='reference')
    rule = ex.Exl3Rule(settings, root=root, records=records, fingerprint=fp, words=words, meta=meta, layers=40,
                       source_binding=SOURCE)
    store = stub_store(rule)
    acquire = lambda key: lfu.LFUSourceExpertStore._exl3_acquire(store, rule, key)
    assert acquire((1, 5)) is None and rule.missing_experts() == {'1': [5]}   # unlisted file -> fallback
    a = acquire((0, 3))
    assert ex.is_exl3(a) and a.words == 48 and sum(v.nbytes for v in a.values()) == size
    assert store._leased[(0, 3)] is a and acquire((0, 3)) is a                # leased: same object
    assert rule.stats['exl3_reads'] == 1 and rule.stats['exl3_bytes_read'] == records[(0, 3)]['size']
    assert rule.pool.live == size
    store._leased.clear()
    assert acquire((0, 3)) is a and rule.pool.stats['hits'] == 1             # pool hit after the window
    store._leased.clear()
    b = acquire((0, 7))                                                       # evicts (0, 3): capacity one
    assert rule.pool.stats['evictions'] == 1 and list(rule.pool.objects) == [(0, 7)]
    store._leased.clear()
    assert lfu.LFUSourceExpertStore._exl3_unleased_bytes(store) == size      # pool object, not leased
    store._leased[(0, 7)] = b
    assert lfu.LFUSourceExpertStore._exl3_unleased_bytes(store) == 0
    assert lfu.LFUSourceExpertStore._exl3_unleased_bytes(stub_store(rule, packed=True)) == size
    store._leased[(0, 3)] = {'gate.qweight': None}
    with pytest.raises(ValueError, match='leased as an original'):
        acquire((0, 3))
    route = lfu.LFUSourceExpertStore.exl3_route(SimpleNamespace(exl3=rule, _lock=threading.RLock(),
        tables={'blk.0': tuple((0, e) for e in range(384))}), 'blk.0', np.array([[3, 7, 1, 2, 4, 5]], np.uint32))
    assert route == {3, 7} and rule.slots['exl3'][0] == 2 and rule.slots['original'][0] == 4


def test_out_of_range_bank_experts_are_not_missing():
    rule = rule_for([(0, 1), (5, 2)], layers=[0, 3])
    assert not rule.serves((5, 2))
    rule.record_fallback((5, 2)); rule.record_fallback((5, 3))
    assert rule.missing_experts() == {'5': [3]} and rule.stats['out_of_range_acquires'] == 1


def test_pool_scales_with_adaptive_fraction(mx):
    pool = ex.Exl3Pool(300)
    obj = lambda: {'a': mx.zeros((25,), mx.float32)}                         # 100 bytes each
    objects = {k: obj() for k in range(3)}
    for k, v in objects.items():
        assert pool.insert(k, v, 100, {})
    assert pool.scale(0.5, {0: objects[0]}) == 200 and pool.target == 150     # (0) leased: kept
    assert list(pool.objects) == [0] and pool.stats['pressure_evictions'] == 2
    assert not pool.insert(3, obj(), 100, {0: objects[0]})                    # target leaves no room
    pool.scale(2.0, {})
    assert pool.target == 300 and pool.insert(4, obj(), 100, {})


def test_missing_file_is_converter_input(tmp_path):
    rule = rule_for([(0, 1)], missing_path=str(tmp_path / 'missing.json'))
    for key in ((2, 9), (0, 4), (2, 3), (0, 4)):
        rule.record_fallback(key)
    assert rule.write_missing() and not rule.write_missing()                 # rewritten only on new keys
    assert json.loads((tmp_path / 'missing.json').read_text()) == {'0': [4], '2': [3, 9]}
    assert rule.receipt()['missing_top'][0] == [0, 4, 2]
    rule.record_fallback((5, 1))
    assert rule.write_missing() and json.loads((tmp_path / 'missing.json').read_text())['5'] == [1]


# ---- 4. Policy, refusals, identity ----------------------------------------------------------

def policy(**exl3):
    value = {'root': '/bank'}
    value.update(exl3)
    return {'mode': 'original_lfu', 'budget_bytes': 90 * GB, 'exl3': value}


def test_policy_schema():
    parsed = lfu.parse_policy(policy(pool_bytes=40 * GB, layers=[0, 20], math='kernel'))['exl3']
    assert parsed == dict(root='/bank', scope='all', pool_bytes=40 * GB, layers=[0, 20], math='kernel')
    assert lfu.parse_policy(policy())['exl3']['pool_bytes'] == 0
    for bad, match in ((dict(scope='decode'), 'scope'), (dict(root='rel'), 'absolute'), (dict(extra=1), 'keys'),
                       (dict(pool_bytes=-1), 'pool_bytes'), (dict(pool_bytes=90 * GB), 'room'),
                       (dict(math='fast'), 'math'), (dict(layers=[3, 1]), 'layers'),
                       (dict(fingerprint='X'), 'fingerprint'), (dict(missing_path='m.json'), 'absolute')):
        with pytest.raises(ValueError, match=match):
            lfu.parse_policy(policy(**bad))
    for extra in ({'atlas_path': '/a.json'}, {'q1_root': '/q1', 'q1_on_miss': {'max_weight': 0.1}}):
        with pytest.raises(ValueError, match='exl3'):
            lfu.parse_policy({**policy(), **extra})
    hybrid = {'mode': 'hot_original_q1', 'budget_bytes': 90 * GB, 'hot_bytes': 60 * GB, 'q1_root': '/q1',
              'hotset_path': '/h.json', 'exl3': {'root': '/bank'}}
    with pytest.raises(ValueError, match='original_lfu'):
        lfu.parse_policy(hybrid)


def test_route_refusals():
    from mlx2.adapters.deepseek_v41_source import source_policy
    assert source_policy({'deepseek_source_experts': policy()})['deepseek_source_experts']['exl3']['scope'] == 'all'
    for key, value in (('deepseek_source_expert_io', 'overlap_v1'), ('deepseek_physical_batch', 'b2'),
                       ('deepseek_source_resident_gather', 'bank_v1')):
        with pytest.raises(ValueError):
            source_policy({'deepseek_source_experts': policy(), key: value})
    with pytest.raises(ValueError, match='cannot combine with: deepseek_source_expert_io'):
        ex.validate_combination({'deepseek_source_experts': {'exl3': {}}, 'deepseek_source_expert_io': 'overlap_v1'})
    with pytest.raises(ValueError, match='draft_model'):
        ex.validate_combination({'deepseek_source_experts': {'exl3': {}}, 'draft_model': '/d'})
    ex.validate_combination({'deepseek_source_experts': {'mode': 'original_lfu'}, 'draft_model': '/d'})   # off: inert


def test_identity_salt_and_contract():
    a = rule_for([(0, 1)])
    assert a.salt == rule_for([(0, 1)], pool_bytes=5 * GB, missing_path='/m.json').salt   # residency only
    assert a.salt != rule_for([(0, 1)], math='kernel').salt
    other = ex.Exl3Rule(a.settings, root=Path('/bank'), records=a.records, fingerprint='e' * 64, words=WORDS,
                        meta={}, layers=40, source_binding=SOURCE)
    assert other.salt != a.salt
    p = {'deepseek_source_experts': {'exl3': {}}}
    assert ex.apc_refusal_reason(p) == ex.APC_REASON and ex.apc_refusal_reason({}) is None
    adapter = SimpleNamespace(kernel_policy=p, weight_stream=SimpleNamespace(exl3=a), numerics_revision='n',
                              profile='p', numeric_profile='q')
    assert ex.bind_identity(adapter) == a.salt
    assert a.salt in adapter.numerics_revision and adapter.numeric_profile.endswith('_approximate_unqualified')
    with pytest.raises(ValueError, match='did not bind'):
        ex.bind_identity(SimpleNamespace(kernel_policy=p, weight_stream=SimpleNamespace(exl3=None)))
    assert ex.diagnostics(adapter)['apcv2_publication'].startswith('refused')


def test_model_class_stamps_rows_that_served_exl3(tmp_path):
    rule = rule_for([(0, 1)], missing_path=str(tmp_path / 'm.json'))

    class Base:
        def _forward(self, inputs, cache=None, **options):
            for key in inputs:
                (rule.record_served if rule.serves(key) else rule.record_fallback)(key)
            return 'ok'
    p = {'deepseek_source_experts': {'exl3': {}}}
    assert ex.model_class(Base, {}) is Base
    model = ex.model_class(Base, p)(); model.source_exl3 = rule
    rows = [SimpleNamespace(), SimpleNamespace()]
    assert model._forward([(0, 2)], rows) == 'ok'
    assert all(getattr(r, 'ast_private_profile', None) is None for r in rows)   # fallback only: unstamped
    assert json.loads((tmp_path / 'm.json').read_text()) == {'0': [2]}
    model._forward([(0, 1)], rows)
    assert {r.ast_private_profile for r in rows} == {ex.private_profile(rule.salt)}
    assert rule.stats['stamped_forwards'] == 1


# ---- 5. The LFU store's acquire() glue (synthetic residency store) ---------------------------

def test_lfu_acquire_serves_bank_experts_and_falls_back(monkeypatch):
    import mlx.core as mx
    from tests.test_deepseek_v41_source_residency_cpu import make, O
    monkeypatch.setattr(lfu, 'ORIGINAL_BYTES', O)
    monkeypatch.setattr(lfu, 'checked_file', lambda *a, **k: None)
    monkeypatch.setattr(lfu, '_verify_file', lambda *a, **k: None)
    previous = mx.default_device(); mx.set_default_device(mx.cpu)
    try:
        small = sum(a.nbytes for a in small_exl3(mx, 1)[0].values())
        s = make(mode='shared_v1', maximum=12 * O + small)
        s.q1_root = None; s.q1_records = {}; s.q1_binding = None; s.hot = frozenset(); s.hot_binding = None
        s.payload_ceiling = s.cache_budget_bytes - O
        s.policy = {'mode': 'original_lfu', 'budget_bytes': s.cache_budget_bytes}
        monkeypatch.setattr(ex, 'payload_bytes', lambda words: small)
        rule = rule_for([(0, 1), (1, 2)], pool_bytes=small)
        s.exl3 = rule
        total = s.cache_budget_bytes
        s._carve_exl3_pool()
        assert s.expert_cache_total_bytes == total and s.maximum_cache_bytes == total - small
        original_read = s._read_expert

        def read(key, function, *args):
            if function is ex.exl3_read:
                s.reads.append((key, 'exl3_read'))
                expert = small_exl3(mx, 7 + key[1])[0]
                return {k: np.array(v) for k, v in expert.items()}
            return original_read(key, function, *args)
        s._read_expert = read
        with s.window():
            got = {k: s.acquire(f'blk.{k[0]}', k[1]) for k in ((0, 1), (0, 3), (1, 2))}
        assert ex.is_exl3(got[(0, 1)]) and ex.is_exl3(got[(1, 2)]) and not ex.is_exl3(got[(0, 3)])
        assert [r[1] for r in s.reads] == ['exl3_read', 'read_tensors', 'exl3_read']
        assert (0, 3) in s._cache and (0, 1) not in s._cache                  # EXL3 lives in its own pool
        assert rule.missing_experts() == {'0': [3]} and rule.served == [1, 1] + [0] * 38
        assert s.stats['exl3_bytes_read'] == 2 and s.stats['original_misses'] == 1
        assert ':exl3_bank_v1=' + rule.salt in s.apc_revision()
        assert not s._leased                                                  # window released leases
        # Capacity one: (1, 2) could not evict the still-leased (0, 1) and was served uncached.
        assert list(rule.pool.objects) == [(0, 1)] and rule.pool.stats['uncached'] == 1
        assert s._exl3_unleased_bytes() == small
        with s.window():
            assert s.acquire('blk.0', 1) is got[(0, 1)]                       # pool hit, no read
        assert len(s.reads) == 3 and rule.pool.stats['hits'] == 1
        # Adaptive residency sizes the pool with the original share (reduced-tier law).
        last = s.adaptive_residency_update(measured_available_bytes=1 << 40)
        assert last['exl3_pool']['capacity_bytes'] == small and last['exl3_pool_fraction'] == 1.0
        tight = s.adaptive_residency_update(measured_available_bytes=0)
        assert tight['exl3_pool_fraction'] < 1.0 and rule.pool.target < small and rule.pool.live == 0
        s.park()
        assert rule.pool.live == 0
    finally:
        mx.set_default_device(previous)


def test_measured_default_bundle_does_not_apply_but_its_keys_compose():
    from mlx2.adapters import deepseek_v41_source_defaults as defaults
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter, source_policy
    assert not defaults.default_route_policy_applies('ordinary', {'deepseek_source_experts': policy()})
    # An operator may still name the bundle explicitly; EXL3 calls bypass its fused kernels.
    bundle = DeepSeekV41SourceAdapter.default_route_execution_policy['ordinary']
    explicit = source_policy({**bundle, 'deepseek_source_experts': policy()})
    assert explicit['deepseek_source_expert_math'] == 'routed_fused_fp32'
