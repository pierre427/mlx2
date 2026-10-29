# SPDX-License-Identifier: MIT
"""CPU contracts for resident MXFP8 WoA storage; GPU parity is the _metal file."""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_source import source_policy
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
from mlx2.runtime.gguf import source_dense_mxfp8 as m8

WOA = 'blk.0.attn_output_a.weight'


def stub(shape, dtype):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype)


def test_plan_reads_codes_and_scales_only():
    p = m8.plan(1)
    assert p['code_storage_bytes'] == 8*1024*4096 and p['scale_storage_bytes'] == 8*1024*128
    assert p['bf16_expansion_bytes'] == 0 and p['grid'] == (64*128, 1, 8) and p['threadgroup'] == (128, 1, 1)
    for rows in (0, 2, 99, True):
        with pytest.raises(ValueError):
            m8.plan(rows)


@pytest.mark.parametrize('change', ['x_dtype', 'x_rows', 'x_groups', 'q_dtype', 'q_shape', 's_dtype', 's_shape'])
def test_geometry_refusals_return_none_before_device(change, monkeypatch):
    from mlx2.runtime.gguf import source_dense
    monkeypatch.setattr(source_dense, 'device_supported', lambda: pytest.fail('device probed'))
    x, q, s = stub((1, 1, 8, 4096), mx.float32), stub((8192, 1024), mx.uint32), stub((8192, 128), mx.uint8)
    x = {'x_dtype': stub((1, 1, 8, 4096), mx.bfloat16), 'x_rows': stub((1, 2, 8, 4096), mx.float32),
         'x_groups': stub((1, 1, 4, 4096), mx.float32)}.get(change, x)
    q = {'q_dtype': stub((8192, 1024), mx.uint8), 'q_shape': stub((8192, 4096), mx.uint32)}.get(change, q)
    s = {'s_dtype': stub((8192, 128), mx.uint32), 's_shape': stub((256, 128), mx.uint8)}.get(change, s)
    assert m8.supports(x, q, s) is False
    assert m8.maybe_project(x, q, s) is None


def test_device_refusal_and_structural_accept(monkeypatch):
    from mlx2.runtime.gguf import source_dense
    x, q, s = stub((1, 1, 8, 4096), mx.float32), stub((8192, 1024), mx.uint32), stub((8192, 128), mx.uint8)
    assert m8.supports(x, q, s, check_device=False)
    monkeypatch.setattr(source_dense, 'device_supported', lambda: False)
    assert m8.maybe_project(x, q, s) is None
    with pytest.raises(ValueError, match='M5 Max'):
        m8.project(x, q, s)


def test_kernel_source_mirrors_pinned_gemv_law():
    spec = m8.kernel_spec()
    src = spec['source']
    assert 'result[tm]+=inter[tn]*v_coeff[tn]' in src
    assert 'for(ushort sn=(SN/2);sn>=1;sn>>=1)' in src and 'simd_shuffle_down' in src
    assert 'BLOCK_N=128' in src and 'TM=4, TN=4' in src
    assert 'scale*mlx2_source_mxfp8::e4m3' in src
    assert 'bfloat' not in src and 'threadgroup_barrier' not in src
    assert 'converted * 256.0' in spec['header'] and '0x400000u' in spec['header']


def test_scale_envelope_is_enforced():
    assert m8.validate_scale_bytes(np.array([[10, 246]], np.uint8)) == (10, 246)
    for bad in ([[9, 120]], [[120, 247]], [[0]], [[255]]):
        with pytest.raises(ValueError, match='exact BF16 envelope'):
            m8.validate_scale_bytes(np.array(bad, np.uint8))
    for bad in (np.array([], np.uint8), np.array([[120]], np.int8)):
        with pytest.raises(ValueError):
            m8.validate_scale_bytes(bad)


def test_cpu_decode_law_matches_official_bf16_expansion():
    """Host model of the kernel's decode: every finite E4M3 byte x admitted E8M0 scale."""
    with mx.stream(mx.cpu):
        codes = np.arange(256, dtype=np.uint8)
        codes = codes[(codes & 0x7F) != 0x7F]
        codes = np.resize(codes, 256).reshape(1, 256)
        for scale in (m8.SCALE_MIN, 60, 120, 127, 200, m8.SCALE_MAX):
            q = mx.array(codes.view('<u4'))
            s = mx.full((1, 256//32), scale, mx.uint8)
            bf = m8.expand_bf16(q, s).astype(mx.float32)
            f32 = mx.dequantize(q, scales=s, mode='mxfp8', dtype=mx.float32)
            mx.eval(bf, f32)
            # The BF16 store is exact: the F32 product the kernel uses is unchanged.
            np.testing.assert_array_equal(np.asarray(bf).view(np.uint32), np.asarray(f32).view(np.uint32))


# ---- policy surface --------------------------------------------------------

def test_policy_is_explicit_and_requires_direct_woa_consumer():
    assert 'deepseek_source_woa_storage' not in source_policy()
    for loads in ('woa_f32', 'direct_f32'):
        policy = source_policy({'deepseek_source_woa_storage': 'mxfp8_gemv_v1', 'deepseek_source_bf16_loads': loads})
        assert policy['deepseek_source_woa_storage'] == 'mxfp8_gemv_v1'
    for loads in (None, 'reference'):
        value = {'deepseek_source_woa_storage': 'mxfp8_gemv_v1'}
        if loads:
            value['deepseek_source_bf16_loads'] = loads
        with pytest.raises(ValueError, match='requires deepseek_source_bf16_loads'):
            source_policy(value)


@pytest.mark.parametrize('extra', [
    {'deepseek_source_resident_gather': 'resident_bank_v1', 'deepseek_source_expert_math': 'routed_fused_fp32'},
    {'deepseek_source_dspark_prepare': 'oracle_v1', 'deepseek_source_dspark_target_forward': 'fully_batched_m1_v1',
     'deepseek_source_dspark_target_expert_law': 'source_ast_m1_rows_v1', 'deepseek_source_expert_math': 'routed_fused_fp32'},
])
def test_policy_refuses_consumers_without_full_bf16_coefficient(extra):
    with pytest.raises(ValueError, match='MXFP8 WoA storage cannot combine'):
        source_policy({'deepseek_source_woa_storage': 'mxfp8_gemv_v1', 'deepseek_source_bf16_loads': 'woa_f32', **extra})


def test_selected_defaults_accept_the_option():
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter
    defaults = dict(DeepSeekV41SourceAdapter.default_route_execution_policy['ordinary'])
    assert defaults['deepseek_source_woa_storage'] == 'mxfp8_gemv_v1'
    assert defaults['deepseek_source_bf16_loads'] == 'direct_f32'
    for loads in ('woa_f32', 'direct_f32'):
        merged = {**defaults, 'deepseek_source_bf16_loads': loads, 'deepseek_source_woa_storage': 'mxfp8_gemv_v1'}
        assert source_policy(merged)['deepseek_source_woa_storage'] == 'mxfp8_gemv_v1'


@pytest.mark.parametrize('kwargs,match', [
    ({'bf16_loads_policy': 'reference'}, 'direct WoA consumer'),
    ({'bf16_loads_policy': 'woa_f32', 'resident_gather_policy': 'resident_bank_v1',
      'expert_math_policy': 'routed_fused_fp32'}, 'resident bank'),
    ({'bf16_loads_policy': 'woa_f32'}, 'canonical Source WoA geometry'),
    ({'bf16_loads_policy': 'woa_f32', 'woa_storage_policy': 'bf16'}, 'invalid Source WoA storage'),
])
def test_weights_constructor_fails_closed_before_artifact_use(kwargs, match):
    kwargs = {'woa_storage_policy': 'mxfp8_gemv_v1', **kwargs}
    with pytest.raises(ValueError, match=match):
        SourceV41Weights({'args': SimpleNamespace()}, expert_cache_bytes=1, **kwargs)


# ---- weights integration (CPU stream; device kernel stubbed) ---------------

def weights(storage='mxfp8_gemv_v1', loads='woa_f32'):
    w = SourceV41Weights.__new__(SourceV41Weights)
    w.source_bf16_loads_policy = loads
    w.source_woa_storage_policy = storage
    w.source_woa_pp_policy = 'reference'
    w.scheduled_dense_calls = {}
    w.scheduled_grouped_calls = {}
    w._ensure_dense = lambda: None
    w._ready = lambda: None
    w.source_resident_bank_session = None
    return w


def packed(groups=1, n=4, k=64, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 256, size=(groups*n, k), dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x30
    q = mx.array(codes.view('<u4'))
    s = mx.array(rng.integers(115, 125, size=(groups*n, k//32), dtype=np.uint8))
    return q, s


def test_decode_row_uses_original_codes(monkeypatch):
    seen = []
    sentinel = object()
    monkeypatch.setattr(m8, 'maybe_project', lambda x, q, s: seen.append((x.dtype, q, s)) or sentinel)
    with mx.stream(mx.cpu):
        w = weights()
        q, s = packed()
        w._dense = {WOA+'.q': q, WOA+'.s': s}
        x = mx.ones((1, 1, 1, 64), mx.bfloat16)
        assert w.grouped(x, WOA, groups=1) is sentinel
        assert seen == [(mx.float32, q, s)]
        assert w.scheduled_grouped_calls == {'source_mxfp8_grouped_gemv_f32': 1}


@pytest.mark.parametrize('rows', [1, 2, 7, 9])
def test_non_gemv_rows_equal_the_bf16_expansion_profile(rows, monkeypatch):
    from mlx2.runtime.gguf import source_dense, source_dense_legacy
    monkeypatch.setattr(m8, 'maybe_project', lambda *a: None)  # e.g. CPU/device refusal
    monkeypatch.setattr(source_dense, 'maybe_project', lambda *a, **k: None)
    monkeypatch.setattr(source_dense_legacy, 'supports', lambda *a: False)
    with mx.stream(mx.cpu):
        q, s = packed(groups=2, n=4, k=64, seed=rows)
        x = mx.random.normal((1, rows, 2, 64), key=mx.random.key(rows)).astype(mx.float32)
        stored = weights()
        stored._dense = {WOA+'.q': q, WOA+'.s': s}
        expanded = weights(storage='reference')
        expanded._dense = {WOA: mx.dequantize(q, scales=s, mode='mxfp8', dtype=mx.bfloat16)}
        a, b = stored.grouped(x, WOA, groups=2), expanded.grouped(x, WOA, groups=2)
        mx.eval(a, b)
        np.testing.assert_array_equal(np.asarray(a).view(np.uint32), np.asarray(b).view(np.uint32))
        assert stored.scheduled_grouped_calls['source_mxfp8_grouped_bf16_expansion'] == 1


def test_bf16_coefficient_consumers_fail_closed():
    w = weights()
    q, s = packed()
    w._dense = {WOA+'.q': q, WOA+'.s': s}
    with pytest.raises(ValueError, match='no resident BF16 coefficient'):
        w.array(WOA)
    w.source_resident_bank_session = SimpleNamespace(active=True, operand_policy='coefficient_views_v1')
    with mx.stream(mx.cpu), pytest.raises(ValueError, match='no coefficient view'):
        w.grouped(mx.ones((1, 1, 1, 64), mx.float32), WOA, groups=1)


class _Pool:
    def __init__(self, raw):
        self.raw = raw

    def call(self, fn, artifact, names):
        return {name: self.raw[name] for name in names}


def _artifact(woa_scale=120):
    rng = np.random.default_rng(5)
    codes = rng.integers(0, 0x7E, size=(64, 64), dtype=np.uint8)
    head = rng.standard_normal((8, 64)).astype(np.float32)
    head_bf16 = np.asarray(mx.array(head).astype(mx.bfloat16).view(mx.uint16))
    raw = {'wo_a.weight': codes, 'wo_a.scale': np.full((2, 2), woa_scale, np.uint8), 'head.weight': head_bf16}
    tensors = {'wo_a.weight': {'dtype': 'F8_E4M3', 'shape': (64, 64)},
               'wo_a.scale': {'dtype': 'F8_E8M0', 'shape': (2, 2)},
               'head.weight': {'dtype': 'BF16', 'shape': (8, 64)}}
    return {'aliases': {WOA: 'wo_a.weight', 'output.weight': 'head.weight'}, 'tensors': tensors}, raw


@pytest.mark.parametrize('storage,loads', [('reference', 'woa_f32'), ('mxfp8_gemv_v1', 'woa_f32'),
                                           ('reference', 'direct_f32'), ('mxfp8_gemv_v1', 'direct_f32')])
def test_load_keeps_only_the_selected_representations(storage, loads):
    with mx.stream(mx.cpu):
        artifact, raw = _artifact()
        w = weights(storage=storage, loads=loads)
        w.artifact, w._read_pool = artifact, _Pool(raw)
        w._dense, w._quant, w._pending_dense, w._pending_dense_host = {}, {}, {}, None
        w._load_dense()
        assert (WOA in w._dense) is (storage == 'reference')
        assert (WOA+'.q' in w._dense) is (storage != 'reference') and WOA not in w._quant
        assert w._dense['output.weight'].dtype == (mx.float32 if loads == 'woa_f32' else mx.bfloat16)
        if storage != 'reference':
            assert w._dense[WOA+'.q'].dtype == mx.uint32 and w._dense[WOA+'.s'].shape == (64, 2)


def test_load_refuses_scales_outside_exact_envelope():
    with mx.stream(mx.cpu):
        artifact, raw = _artifact(woa_scale=5)
        w = weights()
        w.artifact, w._read_pool = artifact, _Pool(raw)
        w._dense, w._quant, w._pending_dense, w._pending_dense_host = {}, {}, {}, None
        with pytest.raises(ValueError, match='exact BF16 envelope'):
            w._load_dense()
        assert not w._dense
