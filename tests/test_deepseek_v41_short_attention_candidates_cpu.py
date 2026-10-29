"""CPU-only short-attention math/source contracts; no GPU/checkpoint is run."""
import numpy as np
import pytest
import mlx.core as mx

from mlx2.runtime.models import deepseek_v41_short_attention as candidate


@pytest.fixture(autouse=True)
def forbid_gpu(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast, 'metal_kernel', lambda *a, **k: (_ for _ in ()).throw(AssertionError('GPU forbidden')))


def fixture(local_rows=7, selected_rows=3, *, seed=7):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(1, 64, 1, 512)).astype(np.float32)*0.2
    local = rng.normal(size=(1, 1, local_rows, 512)).astype(np.float32)
    pooled = rng.normal(size=(1, max(1, selected_rows+2), 512)).astype(np.float32)
    indices = np.arange(selected_rows, dtype=np.uint32).reshape(1, 1, -1)
    mask = np.ones((1, 1, 1, local_rows), bool)
    valid = np.ones((1, 1, selected_rows), bool)
    sinks = rng.normal(size=64).astype(np.float32)
    return tuple(mx.array(v) for v in (q, local, mask, pooled, indices, valid, sinks))


def oracle(args):
    q, local, mask, pooled, indices, valid, sinks = (np.asarray(a) if a is not None else None for a in args)
    result = np.zeros(q.shape, np.float64)
    for h in range(64):
        vectors = [local[0, 0, i].astype(np.float64) for i in range(local.shape[2]) if mask[0, 0, 0, i]]
        if pooled is not None:
            vectors += [pooled[0, indices[0, 0, i]].astype(np.float64) for i in range(indices.shape[-1]) if valid[0, 0, i]]
        if vectors:
            vectors = np.stack(vectors)
            logits = vectors @ (q[0, h, 0].astype(np.float64)*512**-0.5)
            top = max(np.max(logits), float(sinks[h]))
            mass = np.exp(logits-top)
            result[0, h, 0] = (mass @ vectors)/(mass.sum()+np.exp(float(sinks[h])-top))
    return result.astype(np.float32)


@pytest.mark.parametrize('local_rows,selected_rows', [(1, 0), (7, 3), (127, 1), (128, 512)])
def test_joint_full_geometry_matches_independent_sink_oracle(local_rows, selected_rows):
    args = fixture(local_rows, selected_rows)
    actual = candidate.joint_attention(*args, numerical_profile=candidate.JOINT_PROFILE)
    np.testing.assert_allclose(np.asarray(actual), oracle(args), rtol=3e-5, atol=3e-6)


def test_masked_nan_padding_and_invalid_unused_indices_do_not_poison():
    q, local, mask, pooled, indices, valid, sinks = fixture(7, 3)
    local_np, pooled_np = np.array(local), np.array(pooled)
    local_np[0, 0, 0] = np.nan; pooled_np[0, 0] = np.nan
    mask_np, valid_np, ids_np = np.array(mask), np.array(valid), np.array(indices)
    mask_np[..., 0] = False; valid_np[..., 0] = False; ids_np[..., 0] = np.iinfo(np.uint32).max
    args = (q, mx.array(local_np), mx.array(mask_np), mx.array(pooled_np), mx.array(ids_np), mx.array(valid_np), sinks)
    actual = candidate.joint_attention(*args, numerical_profile=candidate.JOINT_PROFILE)
    np.testing.assert_allclose(np.asarray(actual), oracle(args), rtol=3e-5, atol=3e-6)
    assert np.isfinite(np.asarray(actual)).all()


def test_sink_only_and_extreme_sink_are_finite_zero():
    args = list(fixture(1, 0)); args[2] = mx.zeros(args[2].shape, mx.bool_)
    args[6] = mx.full((64,), -1000., mx.float32)
    actual = candidate.joint_attention(*args, numerical_profile=candidate.JOINT_PROFILE)
    np.testing.assert_array_equal(np.asarray(actual), np.zeros((1, 64, 1, 512), np.float32))
    args = list(fixture(1, 1)); args[6] = mx.full((64,), 1000., mx.float32)
    np.testing.assert_array_equal(np.asarray(candidate.joint_attention(*args, numerical_profile=candidate.JOINT_PROFILE)), np.zeros((1, 64, 1, 512), np.float32))


def test_no_pooled_source_and_reused_gather():
    args = fixture(7, 3)
    q, local, mask, pooled, indices, valid, sinks = args
    rows = pooled[0, indices[0]][None]
    fresh = candidate.joint_attention(*args, numerical_profile=candidate.JOINT_PROFILE)
    reused = candidate.joint_attention(*args, gathered=rows, numerical_profile=candidate.JOINT_PROFILE)
    np.testing.assert_array_equal(np.asarray(fresh), np.asarray(reused))
    empty = (q, local, mask, None, None, None, sinks)
    np.testing.assert_allclose(np.asarray(candidate.joint_attention(*empty, numerical_profile=candidate.JOINT_PROFILE)), oracle(empty), rtol=3e-5, atol=3e-6)
    with pytest.raises(ValueError, match='without pooled'):
        candidate.joint_attention(*empty, gathered=rows, numerical_profile=candidate.JOINT_PROFILE)


@pytest.mark.parametrize('mutation', ['query_width', 'query_dtype', 'mask_dtype', 'mask_shape', 'local_rows', 'sink', 'selected_dtype', 'selected_shape'])
def test_unsupported_geometry_refuses_and_dispatch_observes_fallback(mutation):
    args = list(fixture())
    if mutation == 'query_width': args[0] = mx.zeros((1, 64, 2, 512), mx.float32)
    if mutation == 'query_dtype': args[0] = args[0].astype(mx.bfloat16)
    if mutation == 'mask_dtype': args[2] = args[2].astype(mx.float32)
    if mutation == 'mask_shape': args[2] = mx.ones((1, 64, 1, 7), mx.bool_)
    if mutation == 'local_rows': args[1] = mx.zeros((1, 1, 129, 512), mx.float32)
    if mutation == 'sink': args[6] = mx.zeros((63,), mx.float32)
    if mutation == 'selected_dtype': args[4] = args[4].astype(mx.float32)
    if mutation == 'selected_shape': args[5] = mx.ones((1, 1, 4), mx.bool_)
    with pytest.raises(ValueError): candidate.joint_attention(*args, numerical_profile=candidate.JOINT_PROFILE)
    fallback = object(); counters = {}
    assert candidate.dispatch(*args, policy=candidate.JOINT_PROFILE, reference=lambda: fallback, counters=counters) is fallback
    assert counters == {'source_short_attention_reference_geometry': 1}


def test_explicit_profile_and_native_cpu_refusal_before_compilation():
    args = fixture()
    with pytest.raises(ValueError, match='numerical profile'): candidate.joint_attention(*args, numerical_profile='reference')
    with pytest.raises(ValueError, match='owned Metal'): candidate.native_attention(*args, numerical_profile=candidate.NATIVE_PROFILE)
    fallback = object(); counters = {}
    assert candidate.dispatch(*args, policy=candidate.NATIVE_PROFILE, reference=lambda: fallback, counters=counters) is fallback
    assert counters == {'source_short_attention_reference_device': 1}
    counters = {}
    candidate.dispatch(*args, policy=candidate.JOINT_PROFILE, reference=lambda: fallback, counters=counters)
    assert counters == {'source_short_attention_joint': 1}


@pytest.mark.parametrize('head_group', [1, 2, 4])
def test_native_source_dynamic_lengths_and_tail_guards_without_compilation(head_group):
    s = candidate.kernel_spec(head_group=head_group)
    assert s['grid'][0]//256*head_group == 64
    assert s['dynamic_lengths'] and s['threadgroup_scratch_bytes'] <= 11 << 10
    assert 'lengths[0]' in s['source'] and 'lengths[1]' in s['source']
    assert 'if(live)' in s['source'] and 'if(keep[key])' in s['source']
    assert 'for(uint key=0;key<N;++key)' in s['source']
    assert 'HEAD_GROUP' not in s['source']
    assert s['profile'] == candidate.NATIVE_PROFILE


@pytest.mark.parametrize('bad', [0, 3, 8, True, 1.0])
def test_kernel_shape_family_is_bounded(bad):
    with pytest.raises(ValueError): candidate.kernel_spec(head_group=bad)
