"""Synthetic equations/state only, never actual Source/GPU qualification."""
from contextlib import contextmanager
import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from test_deepseek_v41_cpu import config, Weights, hash_metadata
from mlx2.runtime.models.deepseek_v41_engram import EngramHasher
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from mlx2.adapters.deepseek_v41_ast_target import (
    execute_tree, PARENTS, _clone, SourceASTTargetTreeExecutor,
)

PATHS = ((4, 5, 6, 7, 8), (9, 10, 11, 12, 13), (14, 15, 16, 17, 18))


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda *a, **k: (_ for _ in ()).throw(AssertionError("GPU forbidden")))


def fixture():
    c = config()
    weights = Weights(c)
    model = DeepSeekV41Model(c, weights, identity="synthetic-ds41-ast@seed567")
    model.eval_interval = 0
    cache = model.make_cache()
    model(mx.array([[1, 2]], mx.int32), cache)
    return model, weights, cache


def same_cache(left, right):
    for a, b in zip(left, right):
        assert a.meta_state == b.meta_state
        assert not a.pending and not b.pending
        for x, y in zip(a.state, b.state):
            assert (x is None) == (y is None)
            if x is not None:
                np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=2e-6, rtol=2e-5)


def test_one16row_tile_matches_every_ordinary_ancestry_and_all_state_planes():
    model, weights, cache = fixture()
    base = [_clone(row) for row in cache]
    shapes, scopes = [], []
    original = weights.moe
    def moe(x, ids, scores, prefix):
        shapes.append((x.shape, ids.shape))
        return original(x, ids, scores, prefix)
    weights.moe = moe
    @contextmanager
    def scope(rows):
        scopes.append(rows)
        yield
    weights.execution_scope_for_rows = scope
    result = execute_tree(model, cache, 3, PATHS, (2, 3, 4), cpu_reference=True)
    assert scopes == [16]
    assert shapes == [((16, 8), (16, 2))] * 5
    assert result.logits.shape == (1, 16, 19)
    assert result.features.shape == (1, 16, 24)
    assert result.receipt["physical_target_invocations"] == 1
    assert result.receipt["serial_whole_model_calls"] == 0
    same_cache(cache, base)
    weights.moe = original
    tokens = (3,) + tuple(v for path in PATHS for v in path)
    ordinary = []
    for node, parent in enumerate(PARENTS):
        local = [_clone(row) for row in (base if parent < 0 else ordinary[parent])]
        logits, features = model._forward_impl(mx.array([[tokens[node]]], mx.int32), local,
                                          logits_to_keep=0, capture_layers=(2, 3, 4))
        np.testing.assert_allclose(np.asarray(result.logits[:, node:node+1]), np.asarray(logits), atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(np.asarray(result.features[:, node:node+1]), np.asarray(features), atol=2e-6, rtol=2e-5)
        same_cache(result.frontiers[node], local)
        ordinary.append(local)
    committed = result.transaction.commit_paths([(0, 6, 7, 8)])[0]
    same_cache(committed, ordinary[8])
    assert all(row.offset == 6 for row in committed)
    same_cache(cache, base)
    assert result.transaction.closed


def test_siblings_do_not_affect_other_branches_and_abort_keeps_live_cache():
    model, _, cache = fixture()
    before = [_clone(row) for row in cache]
    first = execute_tree(model, cache, 3, PATHS, (4,), cpu_reference=True)
    saved = np.asarray(first.logits[:, :6]).copy()
    first.transaction.abort()
    changed = (PATHS[0], (18, 18, 18, 18, 18), (17, 17, 17, 17, 17))
    second = execute_tree(model, cache, 3, changed, (4,), cpu_reference=True)
    np.testing.assert_array_equal(saved, np.asarray(second.logits[:, :6]))
    second.transaction.abort()
    same_cache(cache, before)


def test_transaction_rejects_cross_sibling_and_changed_live_frontier():
    model, _, cache = fixture()
    result = execute_tree(model, cache, 3, PATHS, (4,), cpu_reference=True)
    for bad in ((0, 1, 7), (1,), (0, True), (0, 16)):
        with pytest.raises(ValueError, match="root-connected"):
            result.transaction.prepared_cache(bad)
    cache[0].keys = mx.asarray(cache[0].keys, copy=True)
    with pytest.raises(ValueError, match="frontier changed"):
        result.transaction.commit_paths([(0,)])
    result.transaction.abort()


def test_provider_contract_exact_shape_and_unadmitted_backend():
    model, _, cache = fixture()
    provider = SourceASTTargetTreeExecutor(model, cpu_reference=True)
    tokens = [3] + [v for path in PATHS for v in path]
    logits, features, transaction = provider.forward(model, tokens, PARENTS, cache, (4,))
    assert provider.dispatch_metrics(1, transaction, cached=True).width == 1
    assert provider.ast_execution_backend(None) is None
    assert not provider.receipt()["qualified"]
    transaction.abort()
    with pytest.raises(ValueError, match="ancestry"):
        provider.forward(model, tokens, tuple([-1] + [0]*15), cache, (4,))


def test_refusals_do_not_leave_active_target_or_execution_count():
    model, weights, cache = fixture()
    with pytest.raises(ValueError, match="H5120"):
        execute_tree(model, cache, 3, PATHS)
    weights.source_resident_bank_session = SimpleNamespace(active=True)
    with pytest.raises(ValueError, match="active B1"):
        execute_tree(model, cache, 3, PATHS, (4,), cpu_reference=True)
    weights.source_resident_bank_session.active = False
    def refused(rows):
        raise ValueError("test scope refusal")
    weights.execution_scope_for_rows = refused
    with pytest.raises(ValueError, match="scope refusal"):
        execute_tree(model, cache, 3, PATHS, (4,), cpu_reference=True)
    assert model._weight_executions == 0
    assert model._mlx2_ds41_ast_active is None


def test_engram_hash_tail_and_attention_input_features_follow_only_ancestry():
    c = config(engram_layer_ids=(1, 3), engram_num_embeddings=(48, 72),
               engram_compressed_vocab_size=5)
    weights = Weights(c)
    rng = np.random.default_rng(1989)
    for layer in c.engram_layer_ids:
        weights.data[f"blk.{layer}.engram_q_norm.weight"] = mx.ones((4, 8))
        weights.data[f"blk.{layer}.engram_k_norm.weight"] = mx.ones((4, 8))
        weights.data[f"blk.{layer}.engram_kv.weight"] = mx.array(rng.normal(0, .0004, (40, 6144)).astype(np.float32))
    metadata = hash_metadata()
    metadata["deepseek41.engram.token_map"] = [i % 5 for i in range(19)]
    class Rows:
        def __init__(self):
            self.pending = None
            self.windows = []
        def prefetch(self, ids):
            ids = np.asarray(ids)
            if self.pending is not None and not np.array_equal(ids, self.pending):
                raise ValueError("Engram prefetch window is already owned")
            self.pending = ids.copy()
        def read(self, ids):
            self.prefetch(ids)
            self.windows.append(self.pending.copy())
            values, self.pending = self.pending, None
            return np.broadcast_to(values[..., None]/100, (*values.shape, 256)).astype(np.float32).copy()
    model = DeepSeekV41Model(c, weights, identity="synthetic-ds41-ast-engram@seed1989",
                             hasher=EngramHasher(metadata), engram_tables={i: Rows() for i in c.engram_layer_ids})
    model.eval_interval = 0
    cache = model.make_cache()
    model(mx.array([[1, 2]], mx.int32), cache)
    result = execute_tree(model, cache, 3, PATHS, (1, 3, 4), cpu_reference=True)
    for table in model.engram_tables.values():
        assert table.pending is None
        assert len(table.windows) == 2  # Initial prefix + one grouped tile.
        assert table.windows[-1].shape[0] == 16
    # Ordinary oracle on each complete branch, including source pools and
    # partial ratio2 groups on both sides of the compression boundary.
    for rank, path in enumerate(PATHS):
        local = [_clone(row) for row in cache]
        outputs, taps = [], []
        for token in (3,) + path:
            out, tap = model._forward_impl(mx.array([[token]], mx.int32), local,
                                      logits_to_keep=0, capture_layers=(1, 3, 4))
            outputs.append(out)
            taps.append(tap)
        rows = [0] + list(range(1 + rank*5, 6 + rank*5))
        np.testing.assert_allclose(np.asarray(result.logits[:, rows]), np.asarray(mx.concatenate(outputs, 1)), atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(np.asarray(result.features[:, rows]), np.asarray(mx.concatenate(taps, 1)), atol=2e-6, rtol=2e-5)
        same_cache(result.frontiers[rows[-1]], local)
    result.transaction.abort()
    # Failure before the first Engram layer must not leave prefetched windows
    # for later layers owned by a transaction which has already aborted.
    weights.moe = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("before Engram"))
    before = {layer: len(table.windows) for layer, table in model.engram_tables.items()}
    with pytest.raises(RuntimeError, match="before Engram"):
        execute_tree(model, cache, 3, PATHS, (1, 3, 4), cpu_reference=True)
    assert all(table.pending is None and len(table.windows) == before[layer]
               for layer, table in model.engram_tables.items())


def test_failed_grouped_expert_call_leaves_original_frontier_unchanged():
    model, weights, cache = fixture()
    before = [_clone(row) for row in cache]
    def fail(*args, **kwargs):
        raise RuntimeError("injected grouped expert failure")
    weights.moe = fail
    with pytest.raises(RuntimeError, match="injected"):
        execute_tree(model, cache, 3, PATHS, (4,), cpu_reference=True)
    same_cache(cache, before)
    assert model._weight_executions == 0
    assert model._mlx2_ds41_ast_active is None


def test_public_forward_refuses_open_tree_and_resumes_after_abort():
    model, _, cache = fixture()
    result = execute_tree(model, cache, 3, PATHS, (4,), cpu_reference=True)
    for call in (model._forward, model._forward_ordinary):
        with pytest.raises(ValueError, match="active or poisoned AST"):
            call(mx.array([[3]], mx.int32), cache)
    result.transaction.abort()
    assert model(mx.array([[3]], mx.int32), cache).shape == (1, 1, 19)


def test_failed_completion_poison_keeps_all_partial_owners(monkeypatch):
    model, weights, cache = fixture()
    before = [_clone(row) for row in cache]
    def fail(*args, **kwargs):
        raise RuntimeError("injected grouped expert failure")
    weights.moe = fail
    def bad_fence(*args, **kwargs):
        raise RuntimeError("injected completion failure")
    monkeypatch.setattr(mx, "eval", bad_fence)
    with pytest.raises(RuntimeError, match="grouped expert"):
        execute_tree(model, cache, 3, PATHS, (4,), cpu_reference=True)
    assert model._mlx2_ds41_ast_poisoned
    roots, base, frontiers, txn = model._mlx2_ds41_ast_failed_roots
    assert roots and len(base) == 5 and len(frontiers) == 16
    assert all(row[0] is not None for row in frontiers)
    assert txn.closed
    with pytest.raises(ValueError, match="active or poisoned AST"):
        model(mx.array([[3]], mx.int32), cache)
    same_cache(cache, before)


def test_selected_path_copy_completion_failure_retains_owners_and_poison(monkeypatch):
    model, _, cache = fixture()
    result = execute_tree(model, cache, 3, PATHS, (4,), cpu_reference=True)
    def fail(*args):
        raise RuntimeError("selected copy completion failure")
    monkeypatch.setattr(mx, "eval", fail)
    with pytest.raises(RuntimeError, match="selected copy"):
        result.transaction.commit_paths([(0, 1)])
    assert result.transaction.closed and model._mlx2_ds41_ast_poisoned
    assert model._mlx2_ds41_ast_failed_roots[0][0].offset == 4
    assert all(row.offset == 2 for row in cache)


def test_tolerance_drift_cannot_be_admitted_as_exact_ast_oracle(tmp_path):
    model, _, _ = fixture()
    provider = SourceASTTargetTreeExecutor(model, cpu_reference=True)
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(dict(schema="mlx2.deepseek-v41-source-ast-oracle.v1",
        binding={}, input_widths=[16], checks={key: True for key in
        ("logits", "features", "prepared_state", "continuation", "isolation", "commit_abort")},
        receipt_sha256="f"*64, numerical_classification="tolerance_valid")))
    with pytest.raises(ValueError, match="exact numerical oracle"):
        provider.with_ast_oracle(path, artifact="synthetic@test", capture_layers=(4,), target_binding="cpu")
    assert provider.ast_execution_backend(None) is None


def test_explicit_m1_law_uses_one_grouped_handler_per_layer(monkeypatch):
    from mlx2.adapters import deepseek_v41_ast_moe as ast_moe
    model, weights, cache = fixture()
    calls = []
    def grouped(owner, x, ids, scores, prefix):
        assert owner is weights
        calls.append((x.shape, ids.shape, prefix))
        return owner.moe(x, ids, scores, prefix)
    monkeypatch.setattr(ast_moe, "moe", grouped)
    result = execute_tree(model, cache, 3, PATHS, (4,), cpu_reference=True,
                          expert_law="source_ast_m1_rows_v1")
    assert len(calls) == 5 and all(row[:2] == ((16, 8), (16, 2)) for row in calls)
    assert result.receipt["expert_law"] == "source_ast_m1_rows_v1"
    assert result.receipt["numerical_profile"] == "source_ast_b1_attention_m1_expert_occurrences_v1"
    a = SourceASTTargetTreeExecutor(model, cpu_reference=True)
    b = SourceASTTargetTreeExecutor(model, cpu_reference=True, expert_law="source_ast_m1_rows_v1")
    assert a.source_revision_token != b.source_revision_token
    result.transaction.abort()


def test_provider_breadth_first_order_preserves_outputs_features_and_commit_indices():
    model, _, cache = fixture()
    provider = SourceASTTargetTreeExecutor(model, cpu_reference=True)
    tokens = [3] + [v for path in PATHS for v in path]
    expected_logits, expected_features, fixed = provider.forward(model, tokens, PARENTS, cache, (2, 3, 4))
    expected_cache = fixed.prepared_cache((0, 6, 7, 8))
    fixed.abort()
    bfs = (0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15)
    inverse = {old: new for new, old in enumerate(bfs)}
    parents = tuple(-1 if PARENTS[old] < 0 else inverse[PARENTS[old]] for old in bfs)
    logits, features, transaction = provider.forward_many(model, [[tokens[i] for i in bfs]],
        [parents], [cache], (2, 3, 4), cached=True, configured_limit=1)
    np.testing.assert_array_equal(np.asarray(logits), np.asarray(expected_logits[:, list(bfs)]))
    np.testing.assert_array_equal(np.asarray(features), np.asarray(expected_features[:, list(bfs)]))
    assert provider.dispatch_metrics(1, transaction, cached=True).provider_protocol == provider.protocol
    same_cache(transaction.prepared_cache((0, 2, 5, 8)), expected_cache)
    with pytest.raises(ValueError, match="root-connected caller"):
        transaction.prepared_cache((0, 1, 5))
    actual = transaction.commit_paths([(0, 2, 5, 8)])[0]
    same_cache(actual, expected_cache)
    assert transaction.closed and model._mlx2_ds41_ast_active is None
    assert all(row.offset == 2 for row in cache)


@pytest.mark.parametrize("parents", [
    (-1,) + (0,)*15,
    (-1, 0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 0, 12, 13, 14),
    (-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, True),
])
def test_provider_refuses_topologies_outside_exact_three_five_node_chains(parents):
    model, _, cache = fixture()
    provider = SourceASTTargetTreeExecutor(model, cpu_reference=True)
    with pytest.raises(ValueError, match="ancestry|chains"):
        provider.forward(model, [3]*16, parents, cache, (4,))
    assert getattr(model, "_mlx2_ds41_ast_active", None) is None


def test_hybrid_oracle_requires_both_declared_numerical_arm_checks(tmp_path):
    import json
    from types import SimpleNamespace
    from mlx2.adapters.deepseek_v41_ast_target import SourceASTTargetTreeExecutor
    provider=SourceASTTargetTreeExecutor(SimpleNamespace(identity='synthetic'),cpu_reference=True,
        expert_law='source_ast_hybrid_rows_v1',forward_law='fully_batched_m1_v1')
    ordinary_checks=('logits','features','prepared_state','continuation','isolation','commit_abort')
    path=tmp_path/'oracle.json';path.write_text(json.dumps(dict(schema='mlx2.deepseek-v41-source-ast-oracle.v1',
        binding={},input_widths=[16],checks=dict.fromkeys(ordinary_checks,True),receipt_sha256='a'*64,
        numerical_classification='raw_bits_exact')))
    with pytest.raises(ValueError,match='incomplete'):
        provider.with_ast_oracle(path,artifact='artifact',capture_layers=(37,38,39),target_binding={})
    assert provider.ast_oracle is None
