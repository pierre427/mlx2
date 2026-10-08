"""27B policy and fused routing regressions; no Metal or artifact loads."""
import mlx.core as mx
import mlx.nn as nn
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


def test_qwen35_adapter_binds_supported_fused_geometry(monkeypatch):
    from mlx2.adapters.qwen35_4b import Qwen354BAdapter
    from mlx2.adapters.qwen35_9b import Qwen359BAdapter
    from mlx2.adapters.qwen36_27b import Qwen3627BAdapter
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    from mlx2.runtime.models import qwen38_fused_gdn as route

    assert Qwen354BAdapter.fused_gdn_architecture == 'qwen35'
    assert Qwen359BAdapter.fused_gdn_architecture == 'qwen35'
    assert Qwen3627BAdapter.fused_gdn_architecture == 'qwen38'
    assert Qwen3827BAdapter.fused_gdn_architecture == 'qwen38'
    assert "default_fused_gdn" not in vars(Qwen354BAdapter)
    assert "default_fused_gdn" not in vars(Qwen359BAdapter)
    assert "default_fused_gdn" not in vars(Qwen3627BAdapter)
    assert Qwen3827BAdapter.default_fused_gdn is True
    args = TextModelArgs(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
        linear_num_key_heads=16, linear_num_value_heads=32,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4)
    obj = qwen38_27b.GatedDeltaNet(args)
    obj.set_dtype(mx.bfloat16)
    obj.eval()
    class Model:
        def named_modules(self):
            yield 'linear_attn', obj
    c = ArraysCache(size=2)
    c[0] = mx.zeros((1, 3, obj.conv_dim), mx.bfloat16)
    c[1] = mx.zeros((1, 32, 128, 128), mx.float32)
    monkeypatch.setattr(route.kernels, 'fused_gdn_runtime_supported',
                        lambda: False)
    obj.set_fused_gdn_enabled(True)
    assert obj._try_fused_decode(*operands(obj), None, c) is None
    assert obj.fused_gdn_counters['last_fallback'].startswith('unsupported geometry')
    route.configure(Model(), True, architecture=Qwen354BAdapter.fused_gdn_architecture)
    assert obj.fused_gdn_enabled is True
    assert obj.fused_gdn_architecture == 'qwen35'
    assert obj._try_fused_decode(*operands(obj), None, c) is None
    assert obj.fused_gdn_counters['last_fallback'] == 'Metal runtime unavailable'


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


def test_verify_corrected_path_reaches_runtime_without_mutating_cache(monkeypatch):
    from mlx2.runtime.models import qwen38_fused_gdn as route
    obj = layer()
    obj.set_fused_gdn_enabled(True)
    c = cache(obj)
    c.start_speculation()
    old = list(c.cache)
    monkeypatch.setattr(route.kernels, 'fused_gdn_runtime_supported', lambda: False)
    assert obj._try_fused_decode(*operands(obj, width=3), None, c) is None
    assert obj.fused_gdn_counters['last_fallback'] == 'Metal runtime unavailable'
    assert all(a is b for a, b in zip(c.cache, old))


def test_successful_verify_records_exact_rollback_and_commits_once(monkeypatch):
    from mlx2.runtime.models import qwen38_fused_gdn as route
    obj = layer()
    obj.set_fused_gdn_enabled(True)
    c = cache(obj)
    c.start_speculation()
    qkv, z, b, a = operands(obj, width=3)
    conv, state = c[0] + 1, c[1] + 1
    state_snaps = mx.stack([c[1] + 2, c[1] + 3], axis=1)
    conv_snaps = mx.stack([c[0] + 2, c[0] + 3], axis=1)
    recorded = []

    monkeypatch.setattr(route.kernels, 'fused_gdn_runtime_supported', lambda: True)
    monkeypatch.setattr(route.kernels, 'served_silu_refusal', lambda: None)
    monkeypatch.setattr(route.verify_kernels, 'probe_qwen4_fused_gdn_verify',
                        lambda *a, **kw: 32)
    monkeypatch.setattr(route.verify_kernels, 'qwen4_fused_gdn_verify',
                        lambda *a, **kw: (z, conv, state, state_snaps, conv_snaps))
    monkeypatch.setattr(c, 'record_rollback',
                        lambda steps, fn, snapshot: recorded.append((steps, fn, snapshot)))
    output = obj._try_fused_decode(qkv, z, b, a, None, c)
    assert output.shape == (1, 3, 16)
    assert c[0] is conv and c[1] is state and recorded[0][0] == 3
    restored = recorded[0][1](2)
    assert mx.array_equal(restored[0], conv_snaps[:, 1]).item()
    assert mx.array_equal(restored[1], state_snaps[:, 1]).item()
    assert obj.fused_gdn_counters['verify_calls'] == 1
    assert obj.fused_gdn_counters['verify_tokens'] == 3
    assert obj.fused_gdn_counters['verify_rollbacks'] == 1


def test_successful_bounded_prefill_uses_corrected_catchup(monkeypatch):
    from mlx2.runtime.models import qwen38_fused_gdn as route
    obj = layer()
    obj.set_fused_gdn_enabled(True)
    c = cache(obj)
    qkv, z, b, a = operands(obj, width=3)
    conv, state = c[0] + 1, c[1] + 1
    monkeypatch.setattr(route.kernels, 'fused_gdn_runtime_supported', lambda: True)
    monkeypatch.setattr(route.kernels, 'served_silu_refusal', lambda: None)
    monkeypatch.setattr(route.verify_kernels, 'probe_qwen4_fused_gdn_catchup',
                        lambda *a, **kw: 32)

    def build(*args, **kwargs):
        assert kwargs['architecture'] == 'qwen38'
        assert kwargs['num_value_heads'] == 48
        return z, conv, state

    monkeypatch.setattr(route.verify_kernels, 'qwen4_fused_gdn_catchup', build)
    output = obj._try_fused_decode(qkv, z, b, a, None, c)
    assert output.shape == (1, 3, 16)
    assert c[0] is conv and c[1] is state
    assert obj.fused_gdn_counters['prefill_calls'] == 1
    assert obj.fused_gdn_counters['prefill_tokens'] == 3


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
    obj = layer()
    obj.set_fused_gdn_enabled(True)
    c = cache(obj)
    assert obj._try_fused_decode(*operands(obj, width=32), None, c) is None
    assert 'verify width 32 above' in obj.fused_gdn_counters['last_fallback']
    c.start_speculation()
    assert obj._try_fused_decode(*operands(obj), None, c) is None
    assert obj.fused_gdn_counters['last_fallback'] == 'speculative rollback'


def test_fused_gdn_diagnostics_export_tree_kernel_engagement(monkeypatch):
    """Tree verify bypasses the step kernel; its launches must still show.

    The DFlash2 tree route runs every target GDN layer through the owned tree
    kernel (lane_multi._gdn), never through GatedDeltaNet.__call__, so the
    step-kernel counters stay zero on that route.
    """
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    from mlx2.runtime.models import qwen38_tree_gdn as tree

    adapter = object.__new__(Qwen3827BAdapter)
    adapter.model = nn.Module()
    adapter.fused_gdn_architecture = 'qwen38'
    monkeypatch.setitem(tree._COUNTERS, 'tree_calls', 7)
    monkeypatch.setitem(tree._COUNTERS, 'tree_rows', 91)
    report = adapter._fused_gdn_diagnostics()
    assert report['tree_calls'] == 7
    assert report['tree_rows'] == 91
    assert report['decode_calls'] == report['verify_calls'] == 0
