"""Q2 bank format g32h (affine 2-bit g32 over W' = W H, block Hadamard 256).

CPU only (MLX pinned to the CPU device). Objects are built by the actual bank
converter (scripts/research/ds41_expert_q2/convert_g32h.py: rotate/quantize/
convert/write_manifest) from synthetic MXFP4 sources, then read through the
runtime loader (manifest -> q2_read -> _upload_q2 -> Q2Expert) and served by
the runtime Q2 branch of ``_moe_routes``. Also: format policy/resolver
threading, identity salt, and the hash-once sealed bank read.
"""
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_mlx_weights as native
from mlx2.adapters import deepseek_v41_source_lfu as lfu
from mlx2.adapters import deepseek_v41_source_q1_on_miss as q1m
from mlx2.adapters import deepseek_v41_source_tiers as tiers

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts/research/ds41_expert_q2'
REAL_BANK = Path('~/mlx-models/DeepSeek-V4.1-Flash-1017-Source-Expert-Q2-g32h')
G32H = tiers.Q2_FORMATS['g32h']
GB = 10 ** 9


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def cpu():
    import mlx.core as mx
    previous = mx.default_device(); mx.set_default_device(mx.cpu)
    yield mx
    mx.set_default_device(previous)


@pytest.fixture(scope='module')
def g32h(cpu):
    """convert_g32h, with ``from convert import decode_mxfp4`` bound to the Q2 convert.py."""
    saved = sys.modules.get('convert')
    sys.modules['convert'] = _load('convert', SCRIPTS / 'convert.py')
    try:
        yield _load('ds41_q2_convert_g32h', SCRIPTS / 'convert_g32h.py')
    finally:
        if saved is None:
            sys.modules.pop('convert', None)
        else:
            sys.modules['convert'] = saved


def hadamard(mx, n=256):
    return np.array(mx.hadamard_transform(mx.eye(n, dtype=mx.float32)), np.float32)


def block_rotate_np(x, h):
    k = x.shape[-1]
    return (x.reshape(*x.shape[:-1], k // 256, 256) @ h.T).reshape(x.shape)


def dequant(mx, q, s, b, group):
    return np.array(mx.dequantize(mx.array(q), mx.array(s).astype(mx.float32), mx.array(b).astype(mx.float32),
                                  group_size=group, bits=2), np.float32)


# ---- 1. Numeric oracle on a small synthetic W -------------------------------

def test_block_hadamard_is_orthonormal_and_matches_sylvester(cpu):
    h = hadamard(cpu)
    assert np.allclose(h @ h.T, np.eye(256), atol=1e-5)
    syl = np.array([[1.0]])
    while syl.shape[0] < 256:
        syl = np.block([[syl, syl], [syl, -syl]])
    assert np.allclose(h, syl / 16.0, atol=1e-6)


def test_runtime_rotation_and_q2_matmul_oracle(cpu, g32h):
    mx = cpu
    rng = np.random.default_rng(11)
    w = (rng.standard_normal((96, 512)) * 0.02).astype(np.float32)
    w[:, 7] += 0.3                                             # an outlier column: what the rotation is for
    rotated = g32h.rotate(w)
    q, s, b, deq = g32h.quantize(rotated)
    assert q.shape == (96, 32) and s.shape == b.shape == (96, 16) and s.dtype == np.float16
    h = hadamard(mx)
    assert np.allclose(block_rotate_np(w, h), rotated, atol=1e-5)          # W' = W H (row-wise H w)
    x = rng.standard_normal((3, 512)).astype(np.float32)
    xr = q1m.rotate_activation(mx.array(x), 'block_hadamard_256')
    assert np.allclose(np.array(xr), block_rotate_np(x, h), atol=1e-5)
    y = np.array(mx.quantized_matmul(xr, mx.array(q), mx.array(s), mx.array(b), transpose=True,
                                     group_size=32, bits=2))
    assert np.allclose(y, np.array(xr) @ deq.T, rtol=1e-4, atol=1e-5)    # == dequant(W') @ rot(x)
    exact = x @ w.T
    rel = np.linalg.norm(y - exact) / np.linalg.norm(exact)
    plain = np.array(mx.quantized_matmul(mx.array(x), mx.array(q), mx.array(s), mx.array(b), transpose=True,
                                         group_size=32, bits=2))
    assert rel < 0.5                                                       # approximates W @ x
    assert np.linalg.norm(plain - exact) / np.linalg.norm(exact) > 0.9    # unrotated input is wrong
    assert q1m.rotate_activation(mx.array(x), None) is not None
    with pytest.raises(ValueError, match='rotation'):
        q1m.rotate_activation(mx.array(x[:, :300]), 'block_hadamard_256')
    with pytest.raises(ValueError, match='rotation'):
        q1m.rotate_activation(mx.array(x), 'block_hadamard_128')


# ---- 2. Converter -> manifest -> loader -> Q2Expert ---------------------------

def synthetic_raw(layer, expert, seed):
    rng = np.random.default_rng(seed + 1000 * layer + expert)
    raw = {}
    for part, (n, k) in (('w1', (2304, 5120)), ('w3', (2304, 5120)), ('w2', (5120, 2304))):
        prefix = f'layers.{layer}.ffn.experts.{expert}.{part}.'
        raw[prefix + 'weight'] = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
        raw[prefix + 'scale'] = rng.integers(118, 124, (n, k // 32), dtype=np.uint8)
    return raw


@pytest.fixture(scope='module')
def bank(g32h, tmp_path_factory):
    root = tmp_path_factory.mktemp('q2g32h')
    source = 'c' * 64
    g32h.ARTIFACT = {'identity': {'fingerprint': source},
                     'tensors': defaultdict(lambda: {'file': 'shard.safetensors', 'offset': 0})}
    g32h.DEST = root
    g32h.read_tensors = lambda artifact, names: synthetic_raw(int(names[0].split('.')[1]),
                                                              int(names[0].split('.')[4]), 5)
    records = [g32h.convert((0, 0))]
    fingerprint = g32h.write_manifest(root, source, 'test', records, 1.0)
    return SimpleNamespace(root=root, source=source, records=records, fingerprint=fingerprint)


def upload(arrays, bank_obj):
    store = SimpleNamespace(_pending_host={}, _pending_uploads={})
    return lfu.LFUSourceExpertStore._upload_q2(store, (0, 0), arrays, bank_obj)


def test_converter_bank_loads_through_the_runtime_loader(cpu, g32h, bank):
    mx = cpu
    assert bank.records[0]['payload_bytes'] == G32H['payload_bytes'] == 13271040
    root, loaded, binding = q1m.load_q2_manifest(bank.root, bank.source, expected_experts=1, q2_format='g32h')
    assert binding == bank.fingerprint and loaded[(0, 0)]['q2_format'] == 'g32h'
    assert q1m.load_q2_manifest(bank.root, bank.source, expected_experts=1)[1][(0, 0)]['q2_format'] == 'g32h'
    with pytest.raises(ValueError, match='policy names q2_format g64'):
        q1m.load_q2_manifest(bank.root, bank.source, expected_experts=1, q2_format='g64')
    host = q1m.q2_read(root, loaded[(0, 0)])
    assert host['gate.q2weight'].shape == (2304, 320) and host['gate.q2scales'].shape == (2304, 160)
    assert host['down.q2weight'].shape == (5120, 144) and host['down.q2biases'].shape == (5120, 72)
    q2bank = q1m.Bank('q2', root, loaded, binding, G32H['payload_bytes'], 'g32h')
    expert = upload(host, q2bank)
    assert q1m.is_q2(expert) and expert.group_size == 32 and expert.rotation == 'block_hadamard_256'
    with pytest.raises(ValueError, match='admitted bytes'):
        upload(host, q1m.Bank('q2', root, loaded, binding, tiers.Q2_BYTES, 'g32h'))
    # The served projection approximates the decoded Source weight and equals the dequantized oracle.
    decode = sys.modules['convert'].decode_mxfp4
    raw = synthetic_raw(0, 0, 5)
    rng = np.random.default_rng(3)
    h = hadamard(mx)
    for kind, part in (('gate', 'w1'), ('down', 'w2')):
        w = decode(raw[f'layers.0.ffn.experts.0.{part}.weight'], raw[f'layers.0.ffn.experts.0.{part}.scale'])
        x = rng.standard_normal((2, w.shape[1])).astype(np.float32)
        xr = q1m.rotate_activation(mx.array(x), expert.rotation)
        y = np.array(mx.quantized_matmul(xr, expert[kind + '.q2weight'], expert[kind + '.q2scales'],
                                         expert[kind + '.q2biases'], transpose=True, group_size=32, bits=2))
        deq = dequant(mx, host[kind + '.q2weight'], host[kind + '.q2scales'], host[kind + '.q2biases'], 32)
        assert np.allclose(y, block_rotate_np(x, h) @ deq.T, rtol=1e-3, atol=1e-3)
        exact = x @ w.T
        assert np.linalg.norm(y - exact) / np.linalg.norm(exact) < 0.45


def test_manifest_formats_are_exact(cpu, bank, tmp_path):
    shutil.copy(bank.root / bank.records[0]['file'], tmp_path / bank.records[0]['file'])
    manifest = json.loads((bank.root / 'manifest.json').read_text()); manifest.pop('fingerprint')

    def write(**changes):
        value = dict(manifest, **changes)
        value = {k: v for k, v in value.items() if v is not None}
        value['fingerprint'] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        (tmp_path / 'manifest.json').write_text(json.dumps(value))

    for bad in (dict(rotation=None), dict(group_size=64), dict(quantization='source-q2-mlx-affine-g32-v9'),
                dict(weight_bits=3), dict(rotation='block_hadamard_128')):
        write(**bad)
        with pytest.raises(ValueError, match='not admitted'):
            q1m.load_q2_manifest(tmp_path, bank.source, expected_experts=1)
    # A g64 manifest naming this (g32h-sized) inventory is refused by payload size.
    write(quantization=tiers.Q2_FORMATS['g64']['revision'], group_size=64, rotation=None)
    with pytest.raises(ValueError, match='inventory'):
        q1m.load_q2_manifest(tmp_path, bank.source, expected_experts=1, q2_format='g64')


# ---- 3. Runtime Q2 branch of _moe_routes -------------------------------------

def test_moe_routes_g32h_branch_equals_dequantized_rotated_oracle(cpu, bank, monkeypatch):
    mx = cpu
    from tests.test_deepseek_v41_source_batch_provider_cpu import _case
    from mlx2.runtime.gguf import source_middle_fusion
    root, loaded, binding = q1m.load_q2_manifest(bank.root, bank.source, expected_experts=1, q2_format='g32h')
    host = q1m.q2_read(root, loaded[(0, 0)])
    expert = upload(host, q1m.Bank('q2', root, loaded, binding, G32H['payload_bytes'], 'g32h'))
    case = _case(monkeypatch, ids=[[9, 1, 7, 2, 6, 0]])
    case.x = mx.array(np.random.default_rng(9).standard_normal((1, 5120)).astype(np.float32))
    for e in list(case.bank):
        case.bank[e] = expert
    case.store.q2_tier_active = lambda prefix: True
    seen = []

    def law(gate, up, scores, slots):                    # one fixed middle for both arms
        seen.append(gate.shape)
        return gate * mx.sigmoid(up) * scores[slots][:, None]
    monkeypatch.setattr(source_middle_fusion, 'middle', law)
    runtime = case.w.moe(case.x, case.ids, case.scores, 'blk.0'); mx.eval(runtime)
    assert case.w.scheduled_expert_calls['official_source_q2_tier_experts'] == 6
    assert case.w.scheduled_expert_calls['official_source_q2_rotated_experts'] == 6
    assert seen == [(1, 2304)] * 6 and case.calls['down'] == [] and case.calls['batch'] == []

    # Oracle arm: dense dequantized W' and an explicit block-Hadamard matrix.
    h = mx.array(hadamard(mx))
    real_qmm = mx.quantized_matmul
    rotations = []

    def dense_qmm(x, w, scales=None, biases=None, transpose=True, group_size=None, bits=None, **kw):
        if bits != 2:
            return real_qmm(x, w, scales, biases, transpose=transpose, group_size=group_size, bits=bits, **kw)
        assert group_size == 32 and transpose
        d = mx.dequantize(w, scales.astype(mx.float32), biases.astype(mx.float32), group_size=32, bits=2)
        return x @ d.T

    def dense_rotate(x, rotation):
        assert rotation == 'block_hadamard_256'
        rotations.append(x.shape[-1])
        k = x.shape[-1]
        return (x.reshape(*x.shape[:-1], k // 256, 256) @ h.T).reshape(x.shape)
    monkeypatch.setattr(mx, 'quantized_matmul', dense_qmm)
    monkeypatch.setattr(q1m, 'rotate_activation', dense_rotate)
    oracle = case.w.moe(case.x, case.ids, case.scores, 'blk.0'); mx.eval(oracle)
    assert sorted(set(rotations)) == [2304, 5120] and len(rotations) == 18   # gate, up (5120) and down (2304)
    a, b = np.array(runtime), np.array(oracle)
    assert np.isfinite(a).all() and np.abs(b).max() > 0
    assert np.allclose(a, b, rtol=2e-2, atol=2e-2 * np.abs(b).max())


def test_moe_routes_g64_objects_keep_the_unrotated_law(cpu, monkeypatch):
    mx = cpu
    from tests.test_deepseek_v41_source_batch_provider_cpu import _case
    from mlx2.runtime.gguf import source_middle_fusion
    calls = []
    real = q1m.rotate_activation
    monkeypatch.setattr(q1m, 'rotate_activation', lambda x, r: calls.append(r) or real(x, r))
    case = _case(monkeypatch, ids=[[9, 1, 7, 2, 6, 0]])
    w = mx.zeros((2304, 5120), mx.float16); d = mx.zeros((5120, 2304), mx.float16)
    arrays = {}
    for kind, value in (('gate', w), ('up', w), ('down', d)):
        q, sc, b = mx.quantize(value, group_size=64, bits=2)
        arrays.update({kind + '.q2weight': q, kind + '.q2scales': sc, kind + '.q2biases': b})
    for e in list(case.bank):
        case.bank[e] = q1m.Q2Expert(arrays, q2_format='g64')
    case.store.q2_tier_active = lambda prefix: True
    monkeypatch.setattr(source_middle_fusion, 'middle', lambda gate, up, scores, slots: gate)
    mx.eval(case.w.moe(case.x, case.ids, case.scores, 'blk.0'))
    assert set(calls) == {None} and 'official_source_q2_rotated_experts' not in case.w.scheduled_expert_calls


# ---- 4. Real bank expert (when the bank is on disk) ----------------------------

@pytest.mark.skipif(not (REAL_BANK / 'l00-e000.json').exists(), reason='g32h bank not on this machine')
def test_real_bank_expert_runs_through_loader_and_q2_path(cpu, tmp_path):
    mx = cpu
    record = json.loads((REAL_BANK / 'l00-e000.json').read_text())
    shutil.copy(REAL_BANK / record['file'], tmp_path / record['file'])
    manifest = {'schema': 'mlx2.source-expert-q2.v1', 'source': record['source'], 'source_revision': 'test',
                'quantization': G32H['revision'], 'group_size': 32, 'weight_bits': 2,
                'rotation': 'block_hadamard_256', 'dense_modified': False, 'experts': [record]}
    manifest['fingerprint'] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    root, loaded, binding = q1m.load_q2_manifest(tmp_path, record['source'], expected_experts=1, q2_format='g32h')
    host = q1m.q2_read(root, loaded[(0, 0)])
    expert = upload(host, q1m.Bank('q2', root, loaded, binding, G32H['payload_bytes'], 'g32h'))
    x = np.random.default_rng(1).standard_normal((1, 5120)).astype(np.float32)
    xr = q1m.rotate_activation(mx.array(x), expert.rotation)
    y = np.array(mx.quantized_matmul(xr, expert['gate.q2weight'], expert['gate.q2scales'], expert['gate.q2biases'],
                                     transpose=True, group_size=expert.group_size, bits=2))
    deq = dequant(mx, host['gate.q2weight'], host['gate.q2scales'], host['gate.q2biases'], 32)
    assert np.isfinite(y).all() and np.allclose(y, np.array(xr) @ deq.T, rtol=1e-3, atol=1e-3)


# ---- 5. Policy, budgets, layout identity, salt -----------------------------------

def tiers_policy(**tier):
    value = {'usage_map': '/u.json', 'q2_root': '/q2', 'q2_bytes': 15 * GB, 'q1_bytes': 10 * GB}
    value.update(tier)
    return {'mode': 'original_lfu', 'budget_bytes': 90 * GB, 'q1_root': '/q1', 'tiers': value}


def test_policy_names_the_format_and_budgets_use_its_size():
    assert lfu.parse_policy(tiers_policy())['tiers']['q2_format'] == 'g64'          # back-compat default
    parsed = lfu.parse_policy(tiers_policy(q2_format='g32h'))['tiers']
    assert parsed['q2_format'] == 'g32h'
    for bad in ('g32', 'G32H', 1, None):
        with pytest.raises(ValueError, match='q2_format'):
            lfu.parse_policy(tiers_policy(q2_format=bad))
    no_q2 = tiers_policy(q2_format='g32h'); del no_q2['tiers']['q2_root']; del no_q2['tiers']['q2_bytes']
    with pytest.raises(ValueError, match='require tiers.q2_root'):
        lfu.parse_policy(no_q2)
    # q2_experts overrides are sized by the configured format.
    over = [{'layers': 3, 'q2_experts': 10}]
    g64 = q1m.resolve_layout(dict(parsed, q2_format='g64', overrides=over), 'tiers', budget_bytes=90 * GB)
    g32 = q1m.resolve_layout(dict(parsed, overrides=over), 'tiers', budget_bytes=90 * GB)
    assert g64['budgets'][3]['q2'] == 10 * 11059200 and g32['budgets'][3]['q2'] == 10 * 13271040
    assert g32['expert_bytes']['q2'] == 13271040 and g64['identity'] != g32['identity']
    usage = [[384 - e for e in range(384)] for _ in range(40)]
    a = q1m.resolve_layout(dict(parsed, q2_format='g64'), 'tiers', budget_bytes=90 * GB, usage=usage)
    b = q1m.resolve_layout(parsed, 'tiers', budget_bytes=90 * GB, usage=usage)
    per_layer = 15 * GB // 40
    assert a['counts'][0]['q2'] == per_layer // 11059200 and b['counts'][0]['q2'] == per_layer // 13271040
    # g64 layouts keep their identity (no expert_bytes in the body at the default sizes).
    plain = tiers.resolve(budget_bytes=90 * GB, totals={'q2': 15 * GB, 'q1': 10 * GB}, usage=usage)
    assert plain['identity'] == a['identity']


def test_rule_binds_format_into_salt_and_receipt():
    settings = lfu.parse_policy(tiers_policy(q2_format='g32h'))['tiers']
    usage = [[384 - e for e in range(384)] for _ in range(40)]
    records = defaultdict(lambda: {'file': 'q', 'size': 1})

    def rule(fmt, settings=settings, binding='f' * 64):
        bank = q1m.Bank('q2', '/q2', records, binding, tiers.Q2_FORMATS[fmt]['payload_bytes'], fmt)
        return q1m.Q1OnMiss(settings, root='/q1', records=records, bank_binding='d' * 64, layers=40, mode='tiers',
                            budget_bytes=90 * GB, q2_bank=bank, usage=usage, usage_binding='e' * 64)
    r = rule('g32h')
    receipt = r.receipt()
    assert receipt['bank_formats']['q2'] == dict(format='g32h', revision=G32H['revision'], group_size=32,
                                                 rotation='block_hadamard_256', payload_bytes=13271040,
                                                 fingerprint='f' * 64)
    with pytest.raises(ValueError, match='format disagree'):
        rule('g64')                                                   # bank format != policy format
    g64 = rule('g64', settings=dict(settings, q2_format='g64'))
    assert len({r.salt, g64.salt, rule('g32h', binding='a' * 64).salt}) == 3


# ---- 6. Hash once, then stat seal ---------------------------------------------------

def sealed_file(tmp_path, name='x.safetensors'):
    from safetensors.numpy import save_file
    path = tmp_path / name
    save_file({'a': np.arange(64, dtype=np.float32), 'b': np.ones((4, 4), np.float16)}, str(path))
    record = {'file': name, 'size': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'stat': native._stamp(path.stat())}
    return path, record


def test_bank_reads_hash_once_then_stat_seal(tmp_path, monkeypatch):
    native.clear_bank_verification()
    path, record = sealed_file(tmp_path)
    opens = []
    real_open = os.open
    monkeypatch.setattr(os, 'open', lambda *a, **k: opens.append(a[0]) or real_open(*a, **k))
    before = dict(native.BANK_READ_STATS)
    first = native.read_sealed_bank_file(tmp_path, record['file'], record)
    second = native.read_sealed_bank_file(tmp_path, record['file'], record)
    assert np.array_equal(first['a'], np.arange(64, dtype=np.float32)) and second['b'].shape == (4, 4)
    assert native.BANK_READ_STATS['content_hashes'] - before['content_hashes'] == 1
    assert native.BANK_READ_STATS['seal_verified_reads'] - before['seal_verified_reads'] == 1
    assert len(opens) == 2                                            # one open per read
    os.utime(path, ns=(1, 1))                                         # any stat change fails closed
    with pytest.raises(ValueError, match='changed after admission'):
        native.read_sealed_bank_file(tmp_path, record['file'], record)


def test_bank_hash_mismatch_is_never_cached(tmp_path):
    native.clear_bank_verification()
    path, record = sealed_file(tmp_path)
    bad = dict(record, sha256='0' * 64)
    for _ in range(2):
        with pytest.raises(ValueError, match='admitted hash'):
            native.read_sealed_bank_file(tmp_path, record['file'], bad)
    # A rewrite (new stat seal) re-hashes: the cache is keyed by the seal.
    native.read_sealed_bank_file(tmp_path, record['file'], record)
    path.write_bytes(path.read_bytes())
    fresh = dict(record, stat=native._stamp(path.stat()))
    before = native.BANK_READ_STATS['content_hashes']
    native.read_sealed_bank_file(tmp_path, record['file'], fresh)
    assert native.BANK_READ_STATS['content_hashes'] == before + 1


def test_q2_read_opens_each_file_once(cpu, bank, monkeypatch):
    native.clear_bank_verification()
    root, loaded, _ = q1m.load_q2_manifest(bank.root, bank.source, expected_experts=1, q2_format='g32h')
    opens = []
    real_open = os.open
    monkeypatch.setattr(os, 'open', lambda *a, **k: opens.append(a[0]) or real_open(*a, **k))
    monkeypatch.setattr(Path, 'open', lambda *a, **k: pytest.fail('second open of a Q2 file'))
    before = native.BANK_READ_STATS['content_hashes']
    q1m.q2_read(root, loaded[(0, 0)]); q1m.q2_read(root, loaded[(0, 0)])
    assert len(opens) == 2 and native.BANK_READ_STATS['content_hashes'] == before + 1
