"""27B policy and fused routing regressions; no Metal or artifact loads."""
import mlx.core as mx
import pytest
from mlx2.runtime.models import qwen38_27b
from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.models.qwen3_5 import TextModelArgs, GatedDeltaNet as Reference


def layer():
    args = TextModelArgs(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
        linear_num_key_heads=16, linear_num_value_heads=48,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4)
    obj = qwen38_27b.GatedDeltaNet(args)
    obj.set_dtype(mx.bfloat16)
    obj.eval()
    return obj


def operands(obj, rows=1, width=1):
    return [mx.zeros((rows, width, n), mx.bfloat16)
            for n in (obj.conv_dim, obj.value_dim, obj.num_v_heads, obj.num_v_heads)]


def cache(obj, rows=1):
    c = ArraysCache(size=2)
    c[0] = mx.zeros((rows, 3, obj.conv_dim), mx.bfloat16)
    c[1] = mx.zeros((rows, 48, 128, 128), mx.float32)
    return c


def test_default_off_is_reference_and_does_not_count():
    obj = layer()
    assert obj.fused_gdn_enabled is False
    c = cache(obj)
    old = list(c.cache)
    assert obj._try_fused_decode(*operands(obj), None, c) is None
    assert all(a is b for a, b in zip(c.cache, old))
    assert not obj.fused_gdn_counters['fallbacks']
    assert type(obj.norm) is type(Reference(obj_config()).norm)


def obj_config():
    return TextModelArgs(hidden_size=16, linear_num_key_heads=16,
        linear_num_value_heads=48, linear_key_head_dim=128, linear_value_head_dim=128)


@pytest.mark.parametrize('rows', [1, 2, 4, 8, 16])
def test_cpu_admission_reaches_runtime_and_counts(rows):
    obj = layer()
    obj.set_fused_gdn_enabled(True)
    c = cache(obj, rows)
    old = list(c.cache)
    assert obj._try_fused_decode(*operands(obj, rows), None, c) is None
    assert obj.fused_gdn_counters['reasons'] == {'Metal runtime unavailable': 1}
    assert all(a is b for a, b in zip(c.cache, old))


@pytest.mark.parametrize('width', [3, 9, 17])
def test_verify_arithmetic_refusal_preserves_rollback_cache(width):
    obj = layer()
    obj.set_fused_gdn_enabled(True)
    c = cache(obj)
    c.start_speculation()
    assert obj._try_fused_decode(*operands(obj, width=width), None, c) is None
    assert 'verify arithmetic incompatible' in obj.fused_gdn_counters['last_fallback']


def test_strict_policy_default_and_type():
    from mlx2.adapters.qwen38_27b import fused_gdn_policy
    assert fused_gdn_policy({}) is False
    assert fused_gdn_policy({'fused_gdn': True}) is True
    for bad in (1, 'fused', None):
        with pytest.raises(ValueError, match='fused_gdn must be boolean'):
            fused_gdn_policy({'fused_gdn': bad})


@pytest.mark.parametrize('rows', [1, 2, 4, 8, 16])
def test_successful_routing_uses_qwen35_numerics_and_commits_once(monkeypatch, rows):
    from mlx2.runtime.models import qwen38_fused_gdn as route
    obj = layer()
    obj.set_fused_gdn_enabled(True)
    c = cache(obj, rows)
    qkv, z, b, a = operands(obj, rows)
    conv, state = c[0] + 1, c[1] + 1
    calls = []

    def build(*args, **kwargs):
        assert kwargs['architecture'] == 'qwen38'
        assert kwargs['num_key_heads'] == 16
        assert kwargs['num_value_heads'] == 48
        calls.append(kwargs)
        return z, conv, state

    monkeypatch.setattr(route.kernels, 'fused_gdn_runtime_supported', lambda: True)
    monkeypatch.setattr(route.kernels, 'served_silu_refusal', lambda: None)
    monkeypatch.setattr(route.kernels, 'probe_qwen4_fused_gdn_decode', lambda *a, **kw: 32)
    kernel_name = 'qwen4_fused_gdn_decode' if rows == 1 else 'qwen4_fused_gdn_batch_decode'
    monkeypatch.setattr(route.kernels, kernel_name, build)
    output = obj._try_fused_decode(qkv, z, b, a, None, c)
    assert output.shape == (rows, 1, 16)
    assert len(calls) == 1 and c[0] is conv and c[1] is state
    assert obj.fused_gdn_counters['decode_calls' if rows == 1 else 'batch_decode_calls'] == 1
    assert obj.fused_gdn_counters['batch_decode_rows'] == (rows if rows > 1 else 0)
    assert obj.fused_gdn_counters['fallbacks'] == 0


@pytest.mark.parametrize('reason', ['training', 'distributed sharding', 'unsupported gated norm', 'unsupported geometry'])
def test_named_refusals_happen_before_runtime_probe(monkeypatch, reason):
    from mlx2.runtime.models import qwen38_fused_gdn as route
    obj = layer()
    c = cache(obj)
    args = operands(obj)
    if reason == 'training':
        obj.train()
    elif reason == 'distributed sharding':
        obj.sharding_group = object()
    elif reason == 'unsupported gated norm':
        import mlx.nn as nn
        obj.norm = nn.RMSNorm(128)
    else:
        obj.num_v_heads = 32
    obj.set_fused_gdn_enabled(True)
    monkeypatch.setattr(route.kernels, 'fused_gdn_runtime_supported',
                        lambda: pytest.fail('refusal must precede Metal access'))
    assert obj._try_fused_decode(*args, None, c) is None
    assert obj.fused_gdn_counters['last_fallback'].startswith(reason)
    assert obj.fused_gdn_counters['fallbacks'] == 1


def test_off_forward_is_bit_identical_and_is_the_kill_switch():
    from mlx.utils import tree_flatten
    obj = layer()
    ref = Reference(obj_config())
    ref.load_weights(tree_flatten(obj.parameters()))
    ref.eval()
    x = mx.zeros((1, 1, 16), mx.bfloat16)
    left, right = cache(obj), cache(obj)
    got, expected = obj(x, cache=left), ref(x, cache=right)
    mx.eval(got, expected, left[0], left[1], right[0], right[1])
    assert mx.array_equal(got, expected).item()
    assert mx.array_equal(left[0], right[0]).item()
    assert mx.array_equal(left[1], right[1]).item()
    obj.set_fused_gdn_enabled(True)
    obj.set_fused_gdn_enabled(False)
    assert obj._try_fused_decode(*operands(obj), None, left) is None
    assert obj.fused_gdn_counters['fallbacks'] == 0


def test_prefill_and_one_token_speculation_are_counted_reference_fallbacks():
    from mlx2.runtime.models.qwen38_fused_gdn import PREFILL_REFUSAL
    obj = layer()
    obj.set_fused_gdn_enabled(True)
    c = cache(obj)
    assert obj._try_fused_decode(*operands(obj, width=32), None, c) is None
    assert obj.fused_gdn_counters['last_fallback'] == PREFILL_REFUSAL
    c.start_speculation()
    assert obj._try_fused_decode(*operands(obj), None, c) is None
    assert obj.fused_gdn_counters['last_fallback'] == 'speculative rollback'
