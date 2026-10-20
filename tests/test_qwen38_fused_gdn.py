"""27B policy and fused routing regressions; no Metal or artifact loads."""
import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime.models import qwen38_27b
from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.models.qwen3_5 import GatedDeltaNet as Reference
from mlx2.runtime.models.qwen3_5 import TextModelArgs
from mlx2.runtime.models.qwen_gdn_contracts import (
    QWEN36_GDN_CONTRACT,
    QWEN38_GDN_CONTRACT,
)
from mlx2.runtime.segmented_batch_cache import SegmentedBatchArraysCache


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
    assert obj.fused_gdn_contract is QWEN36_GDN_CONTRACT
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


def ragged_cache(obj, spans, *, segmented=False):
    merged = cache(obj, len(spans))
    if segmented:
        rows = []
        for lane in range(len(spans)):
            row = ArraysCache(size=2)
            row[0] = merged[0][lane:lane + 1]
            row[1] = merged[1][lane:lane + 1]
            row.start_speculation()
            rows.append(row)
        result = SegmentedBatchArraysCache(rows)
    else:
        result = merged
        result.start_speculation()
    result.prepare(lengths=list(spans))
    return result


def install_fake_batch_verify(monkeypatch, obj, final_delta=100):
    from mlx2.runtime.models import qwen38_fused_gdn as route

    calls = []

    def build(
        qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias,
        recurrent_state, norm_weight, norm_eps, row_steps, **kwargs,
    ):
        del qkv, b, a, conv_weight, A_log, dt_bias, norm_weight, norm_eps
        assert kwargs['contract'] is QWEN38_GDN_CONTRACT
        assert kwargs['num_value_heads'] == obj.num_v_heads
        calls.append(tuple(row_steps))
        steps = int(z.shape[1])
        states = mx.stack(
            [recurrent_state + 10 * (position + 1)
             for position in range(steps - 1)],
            axis=1,
        )
        convs = mx.stack(
            [conv_state + position + 1 for position in range(steps - 1)],
            axis=1,
        )
        for lane, valid in enumerate(row_steps):
            if valid < steps:
                states[lane, valid - 1:] = mx.nan
                convs[lane, valid - 1:] = mx.nan
        return (
            mx.zeros(z.shape, z.dtype),
            conv_state + final_delta,
            recurrent_state + final_delta,
            states,
            convs,
        )

    monkeypatch.setattr(route.kernels, 'fused_gdn_runtime_supported', lambda: True)
    monkeypatch.setattr(route.kernels, 'served_silu_refusal', lambda: None)
    monkeypatch.setattr(
        route.verify_kernels, 'probe_qwen4_fused_gdn_batch_verify',
        lambda *args, **kwargs: 32,
    )
    monkeypatch.setattr(
        route.verify_kernels, 'qwen4_fused_gdn_batch_verify', build,
    )
    return calls


def authorized_batch_verify(obj, cache, rows, steps):
    from mlx2.runtime.models.qwen_hybrid_row_exact import (
        _authorize_gdn_batch_verify,
    )

    return _authorize_gdn_batch_verify(obj, cache, rows, steps)


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


def test_row_exact_canary_refusal_does_not_publish_counters(monkeypatch):
    from mlx2.runtime.models import qwen38_fused_gdn as route

    obj = layer()
    obj.set_fused_gdn_enabled(True)
    c = cache(obj)
    c.start_speculation()
    obj.fused_gdn_counters['decode_calls'] = 7
    obj.fused_gdn_counters['fallbacks'] = 2
    obj.fused_gdn_counters['last_fallback'] = 'prior serving refusal'
    obj.fused_gdn_counters['reasons']['prior serving refusal'] = 2
    before = {
        **obj.fused_gdn_counters,
        'reasons': dict(obj.fused_gdn_counters['reasons']),
    }
    monkeypatch.setattr(route.kernels, 'fused_gdn_runtime_supported', lambda: False)
    assert obj.row_exact_probe_fused_decode(
        *operands(obj, width=3), None, c
    ) is None
    assert obj.fused_gdn_counters == before


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


def test_row_exact_canary_success_and_rollback_do_not_publish_counters(monkeypatch):
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
    obj.fused_gdn_counters['verify_calls'] = 5
    obj.fused_gdn_counters['verify_tokens'] = 15
    obj.fused_gdn_counters['verify_rollbacks'] = 3
    before = {
        **obj.fused_gdn_counters,
        'reasons': dict(obj.fused_gdn_counters['reasons']),
    }

    monkeypatch.setattr(route.kernels, 'fused_gdn_runtime_supported', lambda: True)
    monkeypatch.setattr(route.kernels, 'served_silu_refusal', lambda: None)
    monkeypatch.setattr(
        route.verify_kernels, 'probe_qwen4_fused_gdn_verify',
        lambda *a, **kw: 32,
    )
    monkeypatch.setattr(
        route.verify_kernels, 'qwen4_fused_gdn_verify',
        lambda *a, **kw: (z, conv, state, state_snaps, conv_snaps),
    )
    monkeypatch.setattr(
        c, 'record_rollback',
        lambda steps, fn, snapshot: recorded.append((steps, fn, snapshot)),
    )
    output = obj.row_exact_probe_fused_decode(qkv, z, b, a, None, c)
    assert output.shape == (1, 3, 16)
    assert recorded[0][0] == 3
    restored = recorded[0][1](2)
    assert mx.array_equal(restored[0], conv_snaps[:, 1]).item()
    assert mx.array_equal(restored[1], state_snaps[:, 1]).item()
    assert obj.fused_gdn_counters == before


@pytest.mark.parametrize('segmented', [False, True])
def test_batched_verify_ragged_rows_restore_partial_full_and_zero(
    monkeypatch, segmented
):
    obj = layer()
    obj.set_fused_gdn_enabled(True)
    spans = [4, 3, 2, 1]
    accepted = [2, 3, 0, 1]
    c = ragged_cache(obj, spans, segmented=segmented)
    initial_conv, initial_state = c[0], c[1]
    calls = install_fake_batch_verify(monkeypatch, obj)
    qkv, z, b, a = operands(obj, rows=4, width=4)

    with authorized_batch_verify(obj, c, 4, 4):
        output = obj._try_fused_decode(qkv, z, b, a, c.make_mask(4), c)
    mx.eval(output)
    assert output.shape == (4, 4, 16)
    assert calls == [(4, 3, 2, 1)]
    assert obj.row_exact_fused_verify(4, 4) == 'batch_verify'
    assert obj.row_exact_fused_verify_calls() == 1
    counts = obj.fused_gdn_counters
    assert counts['batch_verify_calls'] == 1
    assert counts['batch_verify_rows'] == 4
    assert counts['batch_verify_tokens'] == 10
    assert counts['batch_verify_ragged_calls'] == 1

    c.finalize()
    c.trim_ragged([span - keep for span, keep in zip(spans, accepted)])
    mx.eval(c[0], c[1])
    for lane, (span, keep) in enumerate(zip(spans, accepted)):
        if keep == 0:
            conv_delta = state_delta = 0
        elif keep == span:
            conv_delta = state_delta = 100
        else:
            conv_delta, state_delta = keep, 10 * keep
        assert mx.array_equal(
            c[0][lane], initial_conv[lane] + conv_delta
        ).item(), lane
        assert mx.array_equal(
            c[1][lane], initial_state[lane] + state_delta
        ).item(), lane
    assert counts['batch_verify_rollbacks'] >= 1


def test_batched_verify_refusal_and_canary_leave_serving_state_isolated(
    monkeypatch,
):
    from mlx2.runtime.models import qwen38_fused_gdn as route

    obj = layer()
    obj.set_fused_gdn_enabled(True)
    spans = [4, 3, 2, 1]
    qkv, z, b, a = operands(obj, rows=4, width=4)

    refused = ragged_cache(obj, spans)
    old = list(refused.cache)
    before = {
        **obj.fused_gdn_counters,
        'reasons': dict(obj.fused_gdn_counters['reasons']),
    }
    monkeypatch.setattr(route.kernels, 'fused_gdn_runtime_supported', lambda: False)
    with authorized_batch_verify(obj, refused, 4, 4):
        assert obj.row_exact_probe_fused_decode(
            qkv, z, b, a, refused.make_mask(4), refused
        ) is None
    assert all(left is right for left, right in zip(refused.cache, old))
    assert not refused._rollbacks
    assert obj.fused_gdn_counters == before

    canary = ragged_cache(obj, spans)
    calls = install_fake_batch_verify(monkeypatch, obj)
    with authorized_batch_verify(obj, canary, 4, 4):
        output = obj.row_exact_probe_fused_decode(
            qkv, z, b, a, canary.make_mask(4), canary
        )
    mx.eval(output)
    assert calls == [(4, 3, 2, 1)]
    assert canary._rollbacks
    canary.finalize()
    canary.trim_ragged([2, 0, 2, 0])
    mx.eval(canary[0], canary[1])
    assert obj.fused_gdn_counters == before


def test_batched_verify_dispatch_failure_precedes_state_publication(monkeypatch):
    from mlx2.runtime.models import qwen38_fused_gdn as route

    obj = layer()
    obj.set_fused_gdn_enabled(True)
    c = ragged_cache(obj, [4, 3, 2, 1])
    old = list(c.cache)
    monkeypatch.setattr(route.kernels, 'fused_gdn_runtime_supported', lambda: True)
    monkeypatch.setattr(route.kernels, 'served_silu_refusal', lambda: None)
    monkeypatch.setattr(
        route.verify_kernels, 'probe_qwen4_fused_gdn_batch_verify',
        lambda *args, **kwargs: 32,
    )

    def fail(*args, **kwargs):
        raise RuntimeError('injected')

    monkeypatch.setattr(
        route.verify_kernels, 'qwen4_fused_gdn_batch_verify', fail,
    )
    with authorized_batch_verify(obj, c, 4, 4):
        assert obj._try_fused_decode(
            *operands(obj, rows=4, width=4), c.make_mask(4), c
        ) is None
    assert obj.fused_gdn_counters['last_fallback'] == (
        'Metal kernel dispatch failed: RuntimeError'
    )
    assert all(left is right for left, right in zip(c.cache, old))
    assert not c._rollbacks


def test_batched_verify_compact_mode_declines_before_dispatch(monkeypatch):
    from mlx2.runtime.models import qwen38_fused_gdn as route

    obj = layer()
    obj.set_fused_gdn_enabled(True)

    class CompactOperations:
        provider_revision = 'test'

        def contract(self, cache):
            del cache

        def authorize_compact_gdn_replay(self, cache, contract, steps):
            raise AssertionError((cache, contract, steps))

    obj.set_fused_gdn_compact_replay_enabled(
        True, CompactOperations()
    )
    c = ragged_cache(obj, [4, 3, 2, 1])
    old = list(c.cache)
    monkeypatch.setattr(
        route.verify_kernels, 'admit_qwen4_fused_gdn_batch_verify',
        lambda **kwargs: pytest.fail('compact refusal reached admission'),
    )
    assert obj.row_exact_fused_verify(4, 4) is None
    with authorized_batch_verify(obj, c, 4, 4):
        assert obj._try_fused_decode(
            *operands(obj, rows=4, width=4), c.make_mask(4), c
        ) is None
    assert obj.fused_gdn_counters['last_fallback'] == (
        'batched compact replay unavailable'
    )
    assert all(left is right for left, right in zip(c.cache, old))
    assert not c._rollbacks


def test_batched_verify_never_engages_outside_row_exact_authorization(monkeypatch):
    from mlx2.runtime.models import qwen38_fused_gdn as route

    obj = layer()
    obj.set_fused_gdn_enabled(True)
    c = ragged_cache(obj, [4, 3, 2, 1])
    old = list(c.cache)
    monkeypatch.setattr(
        route.verify_kernels, 'admit_qwen4_fused_gdn_batch_verify',
        lambda **kwargs: pytest.fail('ordinary B>1 reached AST batch admission'),
    )
    assert obj._try_fused_decode(
        *operands(obj, rows=4, width=4), c.make_mask(4), c
    ) is None
    assert all(left is right for left, right in zip(c.cache, old))
    assert not c._rollbacks
    assert obj.fused_gdn_counters['batch_verify_calls'] == 0
    assert obj.fused_gdn_counters['fallbacks'] == 0


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
        assert kwargs['contract'] is QWEN38_GDN_CONTRACT
        assert 'architecture' not in kwargs
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
        assert kwargs['contract'] is QWEN38_GDN_CONTRACT
        assert 'architecture' not in kwargs
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


def test_tree_gdn_pre_refuses_one_tap_and_mismatched_plans(monkeypatch):
    """One tap leaves an empty conv state (0, C): the pre kernel's state-row
    clamp min(row, TAPS - 2) would be -1.  A plan built for another tap count
    indexes windows[w * TAPS + tap] out of step.  Both refuse before launch."""
    from mlx2.runtime.models import qwen38_tree_gdn as S

    def dispatched(*args, **kwargs):
        raise AssertionError("refused geometry reached a Metal launch")

    monkeypatch.setattr(S, "_kernel", dispatched)
    nk, nv, dk = 1, 1, 32
    C = 2 * nk * dk + nv * dk
    rows = 2
    qkv = mx.zeros((1, rows, C), dtype=mx.bfloat16)
    zba = mx.zeros((1, rows, nv * dk + 2 * nv), dtype=mx.bfloat16)
    a_log = mx.zeros((nv,), dtype=mx.float32)
    dt_bias = mx.zeros((nv,), dtype=mx.float32)
    one_tap = mx.zeros((C, 1), dtype=mx.bfloat16)
    empty_state = mx.zeros((1, 0, C), dtype=mx.bfloat16)
    plan0 = S.ConvPlan([[-1, 0]], n_keep=0)
    with pytest.raises(ValueError, match="taps"):
        S.gdn_pre(qkv, [empty_state], one_tap, plan0, zba, a_log, dt_bias,
                  nk=nk, nv=nv, dk=dk, dv=dk)
    four_taps = mx.zeros((C, 4), dtype=mx.bfloat16)
    state3 = mx.zeros((1, 3, C), dtype=mx.bfloat16)
    plan1 = S.ConvPlan([[-1, 0]], n_keep=1)
    with pytest.raises(ValueError, match="taps"):
        S.gdn_pre(qkv, [state3], four_taps, plan1, zba, a_log, dt_bias,
                  nk=nk, nv=nv, dk=dk, dv=dk)
    ab = mx.zeros((1, rows, nv), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="taps"):
        S.gdn_pre_split(qkv, empty_state, one_tap, mx.zeros((rows * 1,), dtype=mx.int32),
                        ab, ab, a_log, dt_bias, nk=nk, nv=nv, dk=dk, dv=dk)
    with pytest.raises(ValueError, match="taps"):
        S.gdn_pre_split(qkv, state3, four_taps, mx.zeros((rows * 2,), dtype=mx.int32),
                        ab, ab, a_log, dt_bias, nk=nk, nv=nv, dk=dk, dv=dk)
