"""Source mHC ``mega_v1`` policy, wiring and kernel text; CPU only.

No GPU/model claim. The Metal bit gate lives in
``test_deepseek_v41_source_hc_mega_metal.py``.
"""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_mlx import execution_policy
from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter, source_policy
from mlx2.runtime.models import deepseek_v41 as v41
from mlx2.runtime.models import deepseek_v41_source_hc_mega as mega
from test_deepseek_v41_cpu import Weights, config

KEY = 'deepseek_source_hc_mega'
COEFF = {'deepseek_source_hc_coefficients': 'fused_fp32'}


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast, 'metal_kernel',
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError('GPU forbidden')))
    yield
    mx.set_default_device(old)


def test_policy_is_opt_in_and_fails_closed():
    assert KEY not in source_policy()
    assert DeepSeekV41SourceAdapter.default_route_execution_policy['ordinary'][KEY] == 'mega_v1'
    assert KEY in DeepSeekV41SourceAdapter.adapter_owned_default_policy_keys
    assert source_policy({KEY: 'mega_v1', **COEFF})[KEY] == 'mega_v1'
    with pytest.raises(ValueError):
        execution_policy({KEY: 'mega_v1'})
    with pytest.raises(ValueError, match='Source HC mega requires'):
        source_policy({KEY: 'mega_v1'})
    with pytest.raises(ValueError, match='Source HC mega requires'):
        source_policy({KEY: 'mega_v1', **COEFF, 'deepseek_source_hc_boundary': 'expand_collapse_seq'})
    with pytest.raises(ValueError):
        source_policy({KEY: 'mega_v1', **COEFF, 'deepseek_hc': 'fused_fp32'})
    for value in (None, True, 1, [], {}, 'bad', 'mega_v2'):
        with pytest.raises(ValueError):
            source_policy({KEY: value, **COEFF})


def test_selected_default_profile_accepts_the_option():
    merged = {**DeepSeekV41SourceAdapter.default_route_execution_policy['ordinary'], KEY: 'mega_v1'}
    owned = DeepSeekV41SourceAdapter.adapter_owned_default_policy_keys
    assert source_policy({k: v for k, v in merged.items() if k in owned})[KEY] == 'mega_v1'


def test_model_constructor_validates_and_salts_identity():
    c = config(activation_dtype='bfloat16')
    base = v41.DeepSeekV41Model(c, Weights(c), identity='mega@cpu', hc_coeff_policy='source_fused_fp32')
    chosen = v41.DeepSeekV41Model(c, Weights(c), identity='mega@cpu', hc_coeff_policy='source_fused_fp32',
                                  source_hc_mega_policy='mega_v1')
    assert base.source_hc_mega_policy == 'reference' and ':source_hc_mega=' not in base.identity
    assert ':source_hc_mega=mega_v1:' + mega.implementation_identity() in chosen.identity
    assert chosen.identity != base.identity
    for kwargs, cfg in (
            (dict(), c),
            (dict(hc_coeff_policy='source_fused_fp32', source_hc_boundary_policy='expand_collapse_seq'), c),
            (dict(hc_coeff_policy='source_fused_fp32'), config(activation_dtype='float32')),
            (dict(hc_policy='fused_fp32'), c)):
        with pytest.raises(ValueError):
            v41.DeepSeekV41Model(cfg, Weights(cfg), identity='mega@cpu', source_hc_mega_policy='mega_v1', **kwargs)
    with pytest.raises(ValueError, match='unknown Source HC mega'):
        v41.DeepSeekV41Model(c, Weights(c), identity='mega@cpu', source_hc_mega_policy='bad')


def test_implementation_identity_is_stable_hex():
    first = mega.implementation_identity()
    assert first == mega.implementation_identity() and len(first) == 64
    int(first, 16)


def test_kernel_text_keeps_stock_laws_and_parallel_sinkhorn():
    pre = mega.compilation_source('pre')
    assert '#pragma clang fp contract(off)' in pre['header']
    assert 'source_hc_sigmoid' in pre['header'] and 'source_hc_sum4' in pre['header']
    source = pre['source']
    for anchor in ('total=sq+total;', 'metal::precise::rsqrt(denominator)', 'flat_total*constants[0]',
                   'rms_total/float(DIM)', 'static_cast<float>(weight[d])*normalized',
                   'metal::fast::exp(logit-maximum)', 'simd_shuffle(c,ushort(12+s))'):
        assert anchor in source
    assert 'SINKHORN_BODY' not in source and 'thread_position_in_grid' not in source
    assert pre['inputs'] == ['h', 'raw', 'prev_pre', 'weight', 'scale', 'base', 'constants']
    for shared in (False, True):
        store = mega.compilation_source('store', shared=shared)
        assert ('mega_bf16(branch[i]+shared[i])' in store['source']) is shared
        assert 'contribution+mixed[offset]' in store['source']
    with pytest.raises(ValueError):
        mega.compilation_source('other')


def test_changed_selected_sinkhorn_body_fails_closed(monkeypatch):
    from mlx2.runtime.models import deepseek_v41_source_hc as source_hc
    monkeypatch.setattr(source_hc, 'SOURCE', source_hc.SOURCE.replace('values[s]*inverse+epsilon', 'values[s]*inverse'))
    with pytest.raises(RuntimeError, match='Sinkhorn body changed'):
        mega.compilation_source('pre')


def operands():
    h = mx.ones((1, 1, 4, 5120), mx.float32)
    pre = mx.ones((1, 1, 4), mx.float32)
    hc = (mx.ones((24, 20480), mx.float32), mx.ones((3,), mx.float32), mx.ones((24,), mx.float32))
    return h, pre, hc, mx.ones((5120,), mx.bfloat16)


@pytest.mark.parametrize('change', ['rows', 'h_dtype', 'pre_shape', 'fn_shape', 'weight_dtype'])
def test_unsupported_geometry_declines_before_device(monkeypatch, change):
    h, pre, (fn, scale, base), weight = operands()
    if change == 'rows':
        h = mx.ones((1, 2, 4, 5120), mx.float32)
    elif change == 'h_dtype':
        h = h.astype(mx.bfloat16)
    elif change == 'pre_shape':
        pre = mx.ones((1, 1, 3), mx.float32)
    elif change == 'fn_shape':
        fn = mx.ones((24, 20479), mx.float32)
    else:
        weight = weight.astype(mx.float16)
    monkeypatch.setattr(mega, '_device', lambda: (_ for _ in ()).throw(AssertionError('device touched')))
    assert not mega.can_pre(h, pre, fn, scale, base, weight)
    with pytest.raises(ValueError):
        mega.pre(h, pre, (fn, scale, base), weight, iterations=20, hc_eps=1e-6, rms_eps=1e-20)
    branch = mx.ones((1, 2, 5120), mx.float32)
    assert not mega.can_store(branch, h, pre, mx.ones((1, 1, 4, 4), mx.float32))


def block_fixture(policy):
    h, pre, hc, weight = operands()
    calls = []
    m = SimpleNamespace(source_hc_mega_policy=policy, scheduled_hc_calls={},
                        collapse=lambda *a: calls.append('collapse') or h[:, :, 0],
                        norm=lambda *a: calls.append('norm') or h[:, :, 0],
                        expand=lambda *a: calls.append('expand') or h)
    c = SimpleNamespace(hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-20)
    b = SimpleNamespace(model=m, c=c, hc={'attn': hc, 'ffn': hc}, attn_norm=weight, ffn_norm=weight,
                        mixes=lambda x, kind: calls.append('mixes') or (pre, pre, mx.ones((1, 1, 4, 4))),
                        attn=lambda x, *a: calls.append('attn') or x)
    b._mega = lambda kind, x, p: v41.Block._mega(b, kind, x, p)
    b._mega_store = lambda *a: v41.Block._mega_store(b, *a)
    return b, h, pre, calls


def test_attention_half_uses_two_fused_boundaries(monkeypatch):
    block, h, pre, calls = block_fixture('mega_v1')
    seen = []
    x_attn, x_ffn, updated = h[:, :, 1] + 1, h[:, :, 2] + 2, h + 3
    coefficients = {'attn': (pre + 1, pre + 2, mx.zeros((1, 1, 4, 4))), 'ffn': (pre + 4, pre + 5, mx.ones((1, 1, 4, 4)))}

    def fake_pre(x, p, hc, weight, **options):
        kind = 'attn' if not seen else 'ffn'
        seen.append((kind, x, p, weight, options))
        return (*coefficients[kind], x_attn if kind == 'attn' else x_ffn)

    monkeypatch.setattr(mega, 'self_check', lambda **kw: True)
    monkeypatch.setattr(mega, 'pre', fake_pre)
    monkeypatch.setattr(mega, 'store', lambda *a: seen.append(('store', a)) or updated)
    result = v41.Block.attention_half(block, h, pre, None, None, {})
    assert calls == ['attn']
    assert seen[0][1] is h and seen[0][2] is pre
    assert seen[1][0] == 'store' and seen[1][1][0] is x_attn and seen[1][1][1] is h
    assert seen[1][1][2] is coefficients['attn'][1] and seen[1][1][3] is coefficients['attn'][2]
    assert seen[2][0] == 'ffn' and seen[2][1] is updated and seen[2][2] is coefficients['attn'][0]
    assert seen[2][4] == dict(iterations=20, hc_eps=1e-6, rms_eps=1e-20)
    assert result[0] is updated and result[1] is x_ffn
    assert all(a is b for a, b in zip(result[2:], coefficients['ffn']))
    assert block.model.scheduled_hc_calls == {'source_hc_mega_pre': 2, 'source_hc_mega_store': 1}


def test_reference_policy_and_unsupported_rows_stay_ordinary(monkeypatch):
    monkeypatch.setattr(mega, 'pre', lambda *a, **k: (_ for _ in ()).throw(AssertionError('mega called')))
    monkeypatch.setattr(mega, 'store', lambda *a, **k: (_ for _ in ()).throw(AssertionError('mega called')))
    block, h, pre, calls = block_fixture('reference')
    v41.Block.attention_half(block, h, pre, None, None, {})
    assert calls == ['mixes', 'collapse', 'norm', 'attn', 'expand', 'mixes', 'collapse', 'norm']
    assert block.model.scheduled_hc_calls == {}
    block, _, pre, calls = block_fixture('mega_v1')
    wide = mx.ones((1, 2, 4, 5120), mx.float32)
    v41.Block.attention_half(block, wide, mx.ones((1, 2, 4), mx.float32), None, None, {})
    assert calls == ['mixes', 'collapse', 'norm', 'attn', 'expand', 'mixes', 'collapse', 'norm']
    assert block.model.scheduled_hc_calls == {'source_hc_mega_pre_reference': 1, 'source_hc_mega_store_reference': 1}


def test_finish_uses_shared_store_only_when_selected(monkeypatch):
    block, h, pre, _ = block_fixture('mega_v1')
    marker = h + 9
    captured = []
    monkeypatch.setattr(mega, 'store', lambda *a: captured.append(a) or marker)
    routed = shared = mx.ones((1, 1, 5120), mx.float32)
    comb = mx.ones((1, 1, 4, 4), mx.float32)
    assert v41.Block.finish(block, h, routed, shared, pre, comb) is marker
    assert captured[0][4] is shared
    assert block.model.scheduled_hc_calls == {'source_hc_mega_store_shared': 1}
