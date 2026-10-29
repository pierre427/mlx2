"""Source plan integration over synthetic CPU models, never a GPU/checkpoint."""
import numpy as np
import mlx.core as mx
import pytest

from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter, source_policy
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from mlx2.runtime.models.deepseek_v41_short_attention import JOINT_PROFILE, NATIVE_PROFILE, implementation_identity
from test_deepseek_v41_cpu import config, Weights


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast, 'metal_kernel', lambda *a, **k: (_ for _ in ()).throw(AssertionError('GPU forbidden')))


def make_model(profile='reference', *, actual_attention=False, weights=None):
    options = dict(index_topk=512, candidate_topk_blocks=2048)
    if actual_attention:
        options.update(num_attention_heads=64, head_dim=512, o_groups=8)
    c = config(**options)
    return DeepSeekV41Model(c, Weights(c) if weights is None else weights,
        identity='synthetic-plan-attention@seed567', index_policy='finite_all_retained',
        selected_kv_policy='decode_reuse', source_attention_math_policy=profile)


def test_new_options_are_explicit_and_reference_defaults_survive():
    plain = source_policy()
    assert 'deepseek_source_submission' not in plain
    assert 'deepseek_source_row_packet' not in plain
    assert 'deepseek_source_attention_math' not in plain
    assert source_policy({'deepseek_source_submission': 'shared_first_v1'})['deepseek_source_submission'] == 'shared_first_v1'
    for profile in (JOINT_PROFILE, NATIVE_PROFILE):
        selected = source_policy({'deepseek_source_attention_math': profile, 'deepseek_index_policy': 'finite_all_retained'})
        assert selected['deepseek_source_attention_math'] == profile
        with pytest.raises(ValueError, match='finite_all_retained'):
            source_policy({'deepseek_source_attention_math': profile})
    with pytest.raises(ValueError, match='authoritative'):
        source_policy({'deepseek_source_row_packet': 'row_packet_v1'})
    packet = source_policy({'deepseek_source_row_packet': 'row_packet_v1',
        'deepseek_source_frame_reuse': 'prefill_v1', 'deepseek_source_prefill_batch': 'gate_up',
        'deepseek_source_expert_math': 'routed_fused_fp32'})
    assert packet['deepseek_source_row_packet'] == 'row_packet_v1'


def test_short_profile_implementation_hash_is_in_model_apc_identity():
    models = [make_model(profile) for profile in ('reference', JOINT_PROFILE, NATIVE_PROFILE)]
    assert len({m.identity for m in models}) == 3
    for model in models[1:]:
        assert implementation_identity() in model.identity
        with pytest.raises(ValueError, match='identity'):
            model(mx.array([[1]], mx.int32), models[0].make_cache())


def test_submission_and_row_packet_adapter_binders_are_profile_bound():
    base = DeepSeekV41SourceAdapter.__new__(DeepSeekV41SourceAdapter)
    base.kernel_policy = {}; base.profile = 'source-test'; base.numerics_revision = 'test-revision'
    base._bind_submission_identity(); base._bind_row_packet_identity()
    assert (base.profile, base.numerics_revision) == ('source-test', 'test-revision')
    for key, value, method, marker in (
        ('deepseek_source_submission', 'shared_first_v1', '_bind_submission_identity', 'source_submission='),
        ('deepseek_source_row_packet', 'row_packet_v1', '_bind_row_packet_identity', 'source_row_packet=')):
        obj = DeepSeekV41SourceAdapter.__new__(DeepSeekV41SourceAdapter)
        obj.kernel_policy = {key: value}; obj.profile = 'source-test'; obj.numerics_revision = 'test-revision'
        getattr(obj, method)()
        assert marker in obj.numerics_revision and ':implementation=' in obj.numerics_revision
        assert obj.profile != 'source-test'


@pytest.mark.parametrize('profile', [JOINT_PROFILE, NATIVE_PROFILE])
def test_tiny_geometry_fallback_preserves_complete_forward_and_state(profile):
    ref, test = make_model(), make_model(profile)
    rc, tc = ref.make_cache(), test.make_cache()
    for tokens in ([[1, 2, 3, 4, 5]], [[6]]):
        x = mx.array(tokens, mx.int32)
        expected, actual = ref(x, rc), test(x, tc)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert test.scheduled_kv_calls['source_short_attention_reference_geometry'] == 10
    for a, b in zip(rc, tc):
        for av, bv in zip(a.state, b.state):
            if av is not None: np.testing.assert_array_equal(np.asarray(av), np.asarray(bv))


def test_actual_attention_geometry_joint_decode_and_snapshot_replay():
    ref, test = make_model(actual_attention=True), make_model(JOINT_PROFILE, actual_attention=True)
    rc, tc = ref.make_cache(), test.make_cache()
    prompt = mx.array([[1, 2, 3, 4, 5]], mx.int32)
    np.testing.assert_array_equal(np.asarray(ref(prompt, rc)), np.asarray(test(prompt, tc)))
    snapshots = [state.snapshot() for state in tc]
    token = mx.array([[6]], mx.int32)
    actual = test(token, tc); expected = ref(token, rc)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-5, atol=3e-6)
    assert test.scheduled_kv_calls['source_short_attention_joint'] == 5
    assert test.scheduled_kv_calls['decode_gather_build'] > 0
    for state, snapshot in zip(tc, snapshots): state.restore(snapshot)
    replay = test(token, tc)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(replay))


def test_native_actual_geometry_cpu_refusal_uses_reference_without_compilation():
    ref, test = make_model(actual_attention=True), make_model(NATIVE_PROFILE, actual_attention=True)
    rc, tc = ref.make_cache(), test.make_cache()
    for x in (mx.array([[1, 2, 3]], mx.int32), mx.array([[4]], mx.int32)):
        np.testing.assert_array_equal(np.asarray(test(x, tc)), np.asarray(ref(x, rc)))
    assert test.scheduled_kv_calls['source_short_attention_reference_device'] == 5
    assert 'source_short_attention_native' not in test.scheduled_kv_calls


def test_shared_first_changes_submission_order_and_preserves_whole_layer_math(monkeypatch):
    c = config(index_topk=512, candidate_topk_blocks=2048)
    ref_weights, test_weights = Weights(c), Weights(c)
    test_weights.source_submission_policy = 'shared_first_v1'
    test_weights.scheduled_dense_calls = {}
    events = []
    original_moe, original_gu = test_weights.moe, test_weights.gate_up
    def moe(*args, **kwargs):
        events.append('routed'); return original_moe(*args, **kwargs)
    def gate_up(*args, **kwargs):
        events.append('shared'); return original_gu(*args, **kwargs)
    monkeypatch.setattr(test_weights, 'moe', moe)
    monkeypatch.setattr(test_weights, 'gate_up', gate_up)
    ref = DeepSeekV41Model(c, ref_weights, identity='synthetic-shared-first@cpu')
    test = DeepSeekV41Model(c, test_weights, identity='synthetic-shared-first@cpu')
    rc, tc = ref.make_cache(), test.make_cache()
    for token in (1, 2):
        x = mx.array([[token]], mx.int32)
        np.testing.assert_array_equal(np.asarray(test(x, tc)), np.asarray(ref(x, rc)))
    assert events == ['shared', 'routed']*10
    assert test_weights.scheduled_dense_calls == {'source_shared_first_submissions': 10}
