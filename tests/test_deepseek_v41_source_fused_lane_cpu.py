"""Source shared fused middle and router BF16 GEMV: policy, wiring, geometry.

CPU only. No GPU, Metal dispatch or model load; bit equality of either kernel
with the ordinary path is a GPU claim that these tests do not make.
"""
from collections import Counter
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_source_shared_middle as shmid
from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter, source_policy
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
from mlx2.runtime.gguf import source_router_gemv as router
from mlx2.runtime.models import deepseek_v41 as v41

SHARED = 'deepseek_source_shared_middle'
ROUTER = 'deepseek_source_router'
QAT = {'deepseek_source_dense_qat': 'fused_fp32'}


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast, 'metal_kernel',
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError('GPU forbidden')))
    router._CHECKED['state'] = None
    router._CHECKED['names'] = set()
    yield
    router._CHECKED['state'] = None
    router._CHECKED['names'] = set()
    mx.set_default_device(old)


# --- policy -----------------------------------------------------------------

def test_both_options_are_opt_in_and_defaults_unchanged():
    defaults = DeepSeekV41SourceAdapter.default_route_execution_policy['ordinary']
    for key in (SHARED, ROUTER):
        assert key not in source_policy()
        assert key not in defaults
        assert key in DeepSeekV41SourceAdapter.adapter_owned_default_policy_keys
    assert source_policy({SHARED: 'fused_middle_v1', **QAT})[SHARED] == 'fused_middle_v1'
    assert source_policy({ROUTER: 'bf16_coeff_gemv_v1'})[ROUTER] == 'bf16_coeff_gemv_v1'


def test_shared_middle_requires_fused_dense_qat_and_rejects_unknown_values():
    with pytest.raises(ValueError, match='requires deepseek_source_dense_qat=fused_fp32'):
        source_policy({SHARED: 'fused_middle_v1'})
    for key in (SHARED, ROUTER):
        for value in (None, True, 1, [], {}, 'v2'):
            with pytest.raises(ValueError):
                source_policy({key: value, **QAT})


def test_selected_default_bundle_accepts_both_options_without_dropping_defaults():
    from mlx2.adapters.deepseek_v41_source_defaults import default_route_policy_filter
    defaults = DeepSeekV41SourceAdapter.default_route_execution_policy['ordinary']
    operator = {SHARED: 'fused_middle_v1', ROUTER: 'bf16_coeff_gemv_v1'}
    assert default_route_policy_filter('ordinary', defaults, operator) == defaults
    merged = source_policy({**defaults, **operator})
    assert merged[SHARED] == 'fused_middle_v1' and merged[ROUTER] == 'bf16_coeff_gemv_v1'


# --- shared fused middle -------------------------------------------------------

def weights(shared='fused_middle_v1', qat='fused_fp32'):
    w = SourceV41Weights.__new__(SourceV41Weights)
    w.source_dense_qat_policy = qat
    w.source_shared_middle_policy = shared
    w._quant = {'g': ('gq', 'gs'), 'u': ('uq', 'us'), 'd': ('dq', 'ds')}
    w.scheduled_dense_calls = {}
    w._ready = lambda: None
    w.artifact = {'args': SimpleNamespace(activation_dtype='bfloat16')}
    return w


def test_shared_fused_path_wiring_and_counters(monkeypatch):
    from mlx2.runtime.models import deepseek_v41_quant as q
    monkeypatch.setattr(q, 'fused_quantize', lambda *a: (_ for _ in ()).throw(AssertionError('down QAT must be folded')))
    calls = []
    gate = mx.arange(64, dtype=mx.float32).reshape(1, 1, 64)
    up = gate + 1
    w = weights()
    w.gate_up = lambda x, a, b, **kw: calls.append(('gate_up', a, b, kw)) or (gate, up)

    def middle(g, u, scores, slots):
        calls.append(('middle', g.shape, u.shape, scores.tolist(), slots.tolist(), slots.dtype))
        return g + u

    def qmm(value, packed, *, scales, mode):
        calls.append(('qmm', packed, scales, mode))
        return value[:, :8] * 2
    monkeypatch.setattr(mx, 'quantized_matmul', qmm)
    x = mx.ones((1, 1, 16), mx.float32)
    out = shmid.shared_expert(w, x, 'g', 'u', 'd', middle=middle)
    assert out.shape == (1, 1, 8)
    np.testing.assert_array_equal(np.asarray(out).ravel(), (2*np.arange(8)+1)*2)
    assert calls[0] == ('gate_up', 'g', 'u', {})
    assert calls[1] == ('middle', (1, 64), (1, 64), [1.0], [0], mx.uint32)
    assert calls[2] == ('qmm', 'dq', 'ds', 'mxfp8')
    assert w.scheduled_dense_calls == {'source_shared_middle_fused': 1, 'official_mxfp8_act32': 1,
                                       'source_dense_qat_calls_avoided': 1}


@pytest.mark.parametrize('case', ('policy', 'qat', 'missing', 'dtype', 'width', 'f32_activations', 'batched', 'device'))
def test_shared_fused_path_declines_before_dispatch(monkeypatch, case):
    monkeypatch.setattr(mx, 'quantized_matmul', lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    w = weights(shared='reference' if case == 'policy' else 'fused_middle_v1',
                qat='reference' if case == 'qat' else 'fused_fp32')
    if case == 'missing':
        w._quant.pop('d')
    if case == 'f32_activations':
        w.artifact = {'args': SimpleNamespace(activation_dtype='float32')}
    width = 48 if case == 'width' else 64
    dtype = mx.bfloat16 if case == 'dtype' else mx.float32
    w.gate_up = lambda *a, **k: (mx.zeros((1, 1, width), dtype), mx.zeros((1, 1, width), dtype))
    middle = lambda *a: (_ for _ in ()).throw(AssertionError('middle must not run'))
    x = mx.ones((2, 1, 16)) if case == 'batched' else mx.ones((1, 1, 16))
    # CPU default device: without an injected kernel the Metal epilogue declines.
    kwargs = {} if case == 'device' else {'middle': middle}
    assert shmid.shared_expert(w, x, 'g', 'u', 'd', **kwargs) is None
    if case in ('dtype', 'width', 'device'):
        assert w.scheduled_dense_calls['source_shared_middle_fallback'] == 1


def test_weights_hook_is_inert_under_reference():
    w = weights(shared='reference')
    assert w.shared_expert_fused(mx.ones((1, 1, 16)), 'g', 'u', 'd') is None


def _block(fused):
    rounds = []
    model = SimpleNamespace(round=lambda v: rounds.append(v) or v + 0.5,
                            linear=lambda *a, **k: (_ for _ in ()).throw(AssertionError('ordinary down')))
    w = SimpleNamespace(shared_policy='reference', shared_expert_fused=fused,
                        gate_up=lambda *a, **k: (_ for _ in ()).throw(AssertionError('ordinary gate/up')))
    block = v41.Block.__new__(v41.Block)
    block.model, block.w, block.prefix = model, w, 'blk.3.'
    block.c = SimpleNamespace(swiglu_limit=10.0)
    return block, rounds


def test_model_hook_applies_the_ordinary_bf16_store_and_passes_names():
    seen = []
    raw = mx.array([[[1.0, 2.0]]])
    block, rounds = _block(lambda x, a, b, c, **kw: seen.append((a, b, c, kw)) or raw)
    out = block.shared_expert(mx.ones((1, 1, 4)))
    assert seen == [('blk.3.ffn_gate_shexp.weight', 'blk.3.ffn_up_shexp.weight', 'blk.3.ffn_down_shexp.weight', {})]
    assert rounds == [raw]
    np.testing.assert_array_equal(np.asarray(out), [[[1.5, 2.5]]])


def test_model_hook_falls_back_to_ordinary_when_declined():
    block, _ = _block(lambda *a, **k: None)
    with pytest.raises(AssertionError, match='ordinary gate/up'):
        block.shared_expert(mx.ones((1, 1, 4)))


def test_unit_score_is_identity_for_every_f32_class():
    bits = np.random.default_rng(7).integers(0, 2**32, 1 << 16, dtype=np.uint64).astype(np.uint32)
    values = np.concatenate([bits.view(np.float32), np.array([0.0, -0.0, np.inf, -np.inf, 1e-45, -1e-45], np.float32)])
    finite = values[np.isfinite(values) | np.isinf(values)]
    np.testing.assert_array_equal((finite * np.float32(1.0)).view(np.uint32), finite.view(np.uint32))


def test_fused_middle_epilogue_matches_fused_qat_text():
    """The middle kernel's FP8 epilogue is the fused_quantize KIND==0 law."""
    import inspect
    from mlx2.runtime.gguf import source_middle_fusion
    from mlx2.runtime.models import deepseek_v41_quant as q
    text = inspect.getsource(source_middle_fusion._kernel)
    for fragment in ('max(maximum,1e-4f)*inverse[0]', '((bits>>23)&255u)+uint((bits&0x7fffffu)!=0u)',
                     'as_type<float>(exponent<<23)', 'v41_e4m3_round(clamp(normalized,-448.0f,448.0f))*scale',
                     'header=FUSED_HEADER'):
        assert fragment in text
    assert '#pragma clang fp contract(off)' in q.FUSED_HEADER
    assert 'maximum*inverse[0]' in q.FUSED_SOURCE and 'KIND==0 ? 1e-4f' in q.FUSED_SOURCE
    assert 'scale=as_type<float>(exponent<<23)' in q.FUSED_SOURCE


# --- router BF16 GEMV -------------------------------------------------------------

def test_router_plan_row_and_k_partition_is_complete():
    p = router.plan(1)
    bm, bn, sm, sn, tm, tn = p['gemv_geometry']
    assert (bn, sm, sn, tm, tn) == (1, 1, 32, 4, 4)
    owners = Counter()
    for group in range(p['grid'][0]//p['threadgroup'][0]):
        for sg in range(bm):
            for row in range(tm):
                owners[group*bm*tm+sg*tm+row] += 1
    assert owners == Counter(range(router.N))
    ks = Counter(base+lane*tn+i for lane in range(sn) for base in range(0, router.K, sn*tn) for i in range(tn))
    assert ks == Counter(range(router.K))
    assert p['global_f32_coefficient_copy_bytes'] == 0
    assert p['avoided_f32_copy_bytes_per_call'] == 384*5120*4
    with pytest.raises(ValueError):
        router.plan(2)


def a(shape, dtype):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype)


@pytest.mark.parametrize('x,w,ok', [
    (a((1, 1, 5120), mx.float32), a((384, 5120), mx.bfloat16), True),
    (a((1, 2, 5120), mx.float32), a((384, 5120), mx.bfloat16), False),
    (a((1, 1, 5120), mx.bfloat16), a((384, 5120), mx.bfloat16), False),
    (a((1, 1, 5120), mx.float32), a((384, 5120), mx.float32), False),
    (a((1, 1, 5120), mx.float32), a((256, 5120), mx.bfloat16), False),
    (a((1, 5120), mx.float32), a((384, 5120), mx.bfloat16), False),
])
def test_router_structural_gate(x, w, ok):
    assert router.supports(x, w, check_device=False) is ok


def test_router_kernel_text_uses_the_pinned_gemv_helper_with_bf16_coefficients():
    try:
        spec = router.kernel_spec()
    except RuntimeError as error:  # installed MLX differs from the pinned audit
        pytest.skip(str(error))
    assert 'GEMVKernel<float,4,1,1,32,4,4,false>' in spec['source']
    assert 'const device bfloat* mat' in spec['header']
    assert 'N=384, K=5120' in spec['source']


def router_weights(monkeypatch, state, *, supported=True):
    w = SourceV41Weights.__new__(SourceV41Weights)
    w.source_router_policy = 'bf16_coeff_gemv_v1'
    w.source_bf16_loads_policy = 'direct_f32'
    w.scheduled_dense_calls = {}
    w._quant = {}
    w._ensure_dense = lambda: None
    w._ready = lambda: None
    w.array = lambda name: mx.ones((3, 2), mx.bfloat16)
    router._CHECKED['state'] = state
    router._CHECKED['names'] = {'blk.0.ffn_gate_inp.weight'} if state else set()
    monkeypatch.setattr(router, 'supports', lambda x, weight: supported)
    monkeypatch.setattr(router, 'project', lambda x, weight: 'candidate')
    monkeypatch.setattr(router, 'first_use_check', lambda x, weight, name=None: 'stock-checked')
    return w


@pytest.mark.parametrize('state,supported,expected,counter', [
    (None, True, 'stock-checked', 'source_router_gemv_first_use_checks'),
    (True, True, 'candidate', 'source_router_gemv'),
    (False, True, None, 'source_router_gemv_check_refused'),
    (True, False, None, 'source_router_gemv_fallback'),
])
def test_router_dispatch_states(monkeypatch, state, supported, expected, counter):
    w = router_weights(monkeypatch, state, supported=supported)
    x = mx.ones((1, 1, 2), mx.float32)
    value = w.linear('blk.0.ffn_gate_inp.weight')(x)
    if expected is None:  # ordinary stock product
        mx.eval(value)
        np.testing.assert_array_equal(np.asarray(value), [[[2, 2, 2]]])
    else:
        assert value == expected
    assert w.scheduled_dense_calls == {counter: 1}


def test_router_policy_reference_never_touches_candidate(monkeypatch):
    w = router_weights(monkeypatch, True)
    w.source_router_policy = 'reference'
    monkeypatch.setattr(router, 'supports', lambda *a: (_ for _ in ()).throw(AssertionError))
    value = w.linear('blk.0.ffn_gate_inp.weight')(mx.ones((1, 1, 2), mx.float32))
    mx.eval(value)
    assert w.scheduled_dense_calls == {}


def test_identities_are_stable_strings():
    assert len(shmid.implementation_identity()) == 64
    try:
        assert len(router.implementation_identity()) == 64
    except RuntimeError as error:
        pytest.skip(str(error))


def test_router_check_is_per_layer_and_errors_refuse(monkeypatch):
    x = mx.ones((1, 1, 2), mx.float32)
    w = mx.ones((3, 2), mx.float32)
    monkeypatch.setattr(router, 'project', lambda x, weight: x @ weight.T)
    router.first_use_check(x, w, 'blk.0.ffn_gate_inp.weight')
    assert router.checked('blk.0.ffn_gate_inp.weight') is True
    assert router.checked('blk.1.ffn_gate_inp.weight') is None  # each layer gets its own check
    monkeypatch.setattr(router, 'project', lambda x, weight: (_ for _ in ()).throw(RuntimeError('compile')))
    value = router.first_use_check(x, w, 'blk.1.ffn_gate_inp.weight')  # error -> stock, refused
    np.testing.assert_array_equal(np.asarray(value), [[[2, 2, 2]]])
    assert router.checked('blk.0.ffn_gate_inp.weight') is False
