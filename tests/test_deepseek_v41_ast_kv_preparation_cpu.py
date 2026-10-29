"""Synthetic source/selection reuse and invalidation; no model or GPU claim."""
from types import SimpleNamespace
import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_ast_kv_preparation import prepare_selected_class


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast, 'metal_kernel', lambda **kw: pytest.fail('GPU forbidden'))


def setup():
    logical = tuple((0, 1, 2 + n) for n in range(16))
    owner = SimpleNamespace(logical_rows=logical,
        pool_table=mx.arange(18 * 4, dtype=mx.float32).reshape(1, 18, 4), current=True)
    owner._stamp = lambda: ('synthetic-owner@f00d', 3)
    def current():
        if not owner.current:
            raise ValueError('prefix changed')
    owner.assert_current = current
    shared = [dict(indices=mx.array([[[0, 2]]], mx.int32),
                   selected_mask=mx.array([[[True, True]]]), _index_generation=1)
              for _ in range(16)]
    live = lambda: True
    return owner, shared, live


def call(owner, shared, live, nodes=(1, 6, 11), profile='synthetic-exact-v1'):
    return prepare_selected_class(mx, owner, shared, nodes, check_live=live, profile=profile)


def test_borrowing_layers_reuse_exact_chronological_gather_and_validity():
    owner, shared, live = setup()
    value, mask, first = call(owner, shared, live)
    again, same_mask, second = call(owner, shared, live)
    assert value is again and mask is same_mask
    assert not first['reused'] and second['reused']
    assert first['built_payload_bytes'] == value.nbytes
    assert second['built_payload_bytes'] == 0
    expected = np.array([[np.arange(72).reshape(18, 4)[i] for i in (0, 2 + n)] for n in (1, 6, 11)], np.float32)
    np.testing.assert_array_equal(value, expected)
    np.testing.assert_array_equal(mask, np.ones((3, 2), bool))


@pytest.mark.parametrize('change', ('selection', 'validity', 'generation', 'owner_array', 'map', 'epoch', 'profile'))
def test_any_changed_source_or_selection_authority_prepares_again(change):
    owner, shared, live = setup()
    old, _, _ = call(owner, shared, live)
    profile = 'synthetic-exact-v1'
    if change == 'selection':shared[1]['indices'] = mx.array([[[1, 2]]], mx.int32)
    if change == 'validity':shared[1]['selected_mask'] = mx.array([[[True, False]]])
    if change == 'generation':shared[1]['_index_generation'] += 1
    if change == 'owner_array':owner.pool_table = mx.asarray(owner.pool_table, copy=True)
    if change == 'map':owner.logical_rows = tuple(tuple(list(row)) for row in owner.logical_rows)
    if change == 'epoch':live = lambda: True
    if change == 'profile':profile = 'synthetic-exact-v2'
    new, _, receipt = call(owner, shared, live, profile=profile)
    assert new is not old and not receipt['reused']


def test_lost_transaction_or_changed_prefix_refuses_even_a_cache_hit():
    owner, shared, live = setup()
    call(owner, shared, live)
    owner.current = False
    with pytest.raises(ValueError, match='prefix changed'):call(owner, shared, live)
    owner.current = True
    with pytest.raises(RuntimeError, match='no longer live'):call(owner, shared, lambda: False)


def test_distinct_count_classes_and_bad_ownership_are_not_reused():
    owner, shared, live = setup()
    call(owner, shared, live)
    _, _, receipt = call(owner, shared, live, nodes=(2, 7, 12))
    assert not receipt['reused']
    shared[6]['indices'] = mx.array([[[0]]], mx.int32)
    shared[6]['selected_mask'] = mx.array([[[True]]])
    with pytest.raises(ValueError, match='count class'):call(owner, shared, live)
    with pytest.raises(ValueError, match='distinct canonical'):call(owner, shared, live, nodes=(1, 1))
    with pytest.raises(ValueError, match='capability'):call(owner, shared, None)


def test_full_synthetic_tree_preserves_all_frontiers_and_reuses_borrowed_source(monkeypatch):
    from test_deepseek_v41_cpu import config, Weights
    from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
    from mlx2.adapters.deepseek_v41_ast_batched_target import execute_batched_tree
    from mlx2.adapters import deepseek_v41_ast_batched_dense as dense_cpu
    from mlx2.adapters import deepseek_v41_ast_batched_dense_exact as dense_exact
    # This exercises the real attention/source/cache traversal with tiny CPU
    # equations, not the artifact-specific H5120 BF16/MXFP8 dense kernels.
    for name in ('linear_tile', 'paired_linear_tile', 'grouped_tile',
                 'shared_expert_tile', 'router_tile', 'head_tile', 'engram_tile'):
        monkeypatch.setattr(dense_exact, name, getattr(dense_cpu, name))
    monkeypatch.setattr(dense_exact, 'mixes_tile', lambda block, h, kind: block.mixes(h, kind))
    c = config()
    model = DeepSeekV41Model(c, Weights(c), identity='synthetic-selected-preparation@f00d')
    cache = model.make_cache()
    model._forward_impl(mx.array([[1, 2, 3, 4]], mx.int32), cache)
    paths = ((4, 5, 6, 7, 8), (9, 10, 11, 12, 13), (14, 15, 16, 17, 18))
    kwargs = dict(cpu_reference=True, expert_law='ordinary_m16_v1', forward_law='fully_batched_m1_v1')
    model.source_ast_kv_preparation_policy = 'reference'
    reference = execute_batched_tree(model, cache, 3, paths, (2, 3, 4), **kwargs)
    expected_logits = np.array(reference.logits)
    expected_features = np.array(reference.features)
    expected = [[(tuple(row.meta_state), tuple(None if v is None else np.array(v) for v in row.state))
                 for row in frontier] for frontier in reference.frontiers]
    reference.transaction.abort()
    model.source_ast_kv_preparation_policy = 'descendant_reuse_v1'
    candidate = execute_batched_tree(model, cache, 3, paths, (2, 3, 4), **kwargs)
    try:
        np.testing.assert_array_equal(candidate.logits, expected_logits)
        np.testing.assert_array_equal(candidate.features, expected_features)
        for frontier, controls in zip(candidate.frontiers, expected):
            for row, (metadata, values) in zip(frontier, controls):
                assert tuple(row.meta_state) == metadata
                for actual, control in zip(row.state, values):
                    if control is None:assert actual is None
                    else:np.testing.assert_array_equal(actual, control)
        assert candidate.receipt['selected_kv_preparation']['hits'] > 0
        assert candidate.receipt['selected_kv_preparation']['builds'] > 0
    finally:
        candidate.transaction.abort()
