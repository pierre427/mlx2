"""Rows per request in batched AST tree rounds (Kapitsa #207 port); CPU only."""

import math
import sys
from types import SimpleNamespace

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime import qwen38_tensorfold
from mlx2.runtime.external_speculative import TreeDraftRow
from mlx2.server import apply_ast_invocation

from test_dflash_pair_select import drain, generator, tiny
from test_dflash_tree_bridge import _fake_model, fake_tensorfold  # noqa: F401 - fixture


def _ragged_reference(batch, monkeypatch, seen):
    """CPU oracle: each lane's exact reference tree, zero-padded to one width."""

    def target(cohort, inputs, parents):
        seen.append(tuple(len(row) for row in inputs))
        results = [
            batch._reference_tree_forward(lane, row, parent_row)
            for lane, row, parent_row in zip(cohort, inputs, parents)
        ]
        width = max(len(row) for row in inputs)

        def pad(array):
            return mx.pad(array, [(0, 0), (0, width - array.shape[1]), (0, 0)])

        from mlx2.runtime.target_tree_executor import TargetTreeDispatchMetrics

        batch._apply_target_tree_dispatch_metrics(
            cohort,
            TargetTreeDispatchMetrics(
                width=len(cohort), physical_packed=False, executor_cached=False,
                cached=False, provider_protocol="cpu-reference-oracle",
            ),
        )
        return (
            mx.concatenate([pad(result[0]) for result in results]),
            mx.concatenate([pad(result[1]) for result in results]),
            qwen38_tensorfold.TensorfoldCohortTransaction([result[2] for result in results]),
        )

    monkeypatch.setattr(batch, "_target_tree_forward_many", target)


def _batch(model, draft, monkeypatch, **kwargs):
    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", "/unused-by-cpu-oracle")
    return generator(
        model, draft, completion_batch_size=4, ready_drain="all",
        dynamic_singleton_tree=True, dynamic_tree_max_width=4,
        tree_node_budget_by_lanes={1: 15, 2: 7, 3: 4, 4: 3}, **kwargs,
    )


def _block(path_ps, parents=None):
    count = len(path_ps)
    parents = parents if parents is not None else [-1] + list(range(count - 1))
    return TreeDraftRow(list(range(10, 10 + count)), parents, 2, info={
        "depths": [1] * count, "log_q": [-0.1] * count,
        "path_log_p": [math.log(p) for p in path_ps],
    })


def test_rows_go_to_the_most_likely_nodes_as_per_lane_prefixes(monkeypatch):
    model, draft = tiny(vocab=64, top_k=16, block_size=16)
    batch = _batch(model, draft, monkeypatch, ast_tree_calibration="order",
                   ast_batch_row_sharing=True)
    # B2 keeps the equal split's 16 rows: 2 anchors + 14 nodes.
    assert batch._shared_cohort_rows(2) == 16
    assert batch._tree_node_budget(2) == 14
    assert batch._tree_node_budget(4) == 12
    confident = _block([0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3])
    doubtful = _block([0.5, 0.2, 0.1, 0.05] + [0.01] * 10)
    shared = batch._share_cohort_rows([object(), object()], [confident, doubtful])
    kept = [len(block.tokens) for block in shared]
    assert sum(kept) == 14
    assert kept[0] > kept[1] >= 1
    for original, cut in zip((confident, doubtful), shared):
        assert cut.tokens == original.tokens[: len(cut.tokens)]
        assert cut.parents == original.parents[: len(cut.parents)]
    histogram = batch.scheduler_stats["external_tree_row_sharing_lane_nodes"]
    assert sum(histogram.values()) == 2
    # The tile row bound the lane-exact window prices is the cohort total.
    assert batch.verify_rows_by_width(4)["4"] == 16


@pytest.mark.parametrize("width", [2, 4])
def test_shared_rows_keep_exact_greedy_tokens(monkeypatch, width):
    prompts = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]][:width]
    model, draft = tiny(vocab=64, top_k=16, block_size=16)
    reference = generator(model, draft, completion_batch_size=4, ready_drain="all")
    reference.insert(prompts, max_tokens=[24] * width,
                     sampling_configs=[{"sampling_temp": 0.0}] * width)
    expected = drain(reference)[0]

    batch = _batch(model, draft, monkeypatch, ast_tree_calibration="order",
                   ast_batch_row_sharing=True)
    seen = []
    _ragged_reference(batch, monkeypatch, seen)
    batch.insert(prompts, max_tokens=[24] * width,
                 sampling_configs=[{"sampling_temp": 0.0}] * width)
    actual, _receipts = drain(batch)
    assert actual == expected
    stats = batch.scheduler_stats
    assert stats["external_tree_row_sharing_rounds"] > 0
    assert all(type(key) is int for key in stats["external_tree_verified_nodes"])
    assert sum(stats["external_tree_verified_nodes"].values()) == stats["external_tree_rounds"]
    assert all(type(key) is int for key in stats["external_tree_row_sharing_lane_nodes"])
    multi = [rows for rows in seen if len(rows) > 1]
    assert multi and all(sum(rows) <= batch._shared_cohort_rows(len(rows)) for rows in multi)


def test_row_sharing_fails_closed(monkeypatch):
    model, draft = tiny(vocab=64, top_k=16, block_size=16)
    with pytest.raises(ValueError, match="requires ast_tree_calibration"):
        _batch(model, draft, monkeypatch, ast_batch_row_sharing=True)
    with pytest.raises(ValueError, match="must be a boolean"):
        _batch(model, draft, monkeypatch, ast_tree_calibration="order",
               ast_batch_row_sharing="yes")
    from mlx2.adapters.qwen38_target_tree_executor import Qwen38TargetTreeExecutor

    monkeypatch.setattr(Qwen38TargetTreeExecutor, "supports_ragged_cohort", False)
    with pytest.raises(ValueError, match="ragged cohorts"):
        _batch(model, draft, monkeypatch, ast_tree_calibration="order",
               ast_batch_row_sharing=True)


def test_packed_provider_returns_ragged_lanes_padded(fake_tensorfold):  # noqa: F811
    roots, _calls, _ = fake_tensorfold

    class Entry:
        def __init__(self, offset):
            self.offset = offset

    caches = [[Entry(5)], [Entry(9)]]
    with pytest.raises(ValueError, match="uniform token/parent widths"):
        qwen38_tensorfold.forward_many(
            _fake_model(), [[1, 2, 3], [4]], [[-1, 0, 0], [-1]], caches, (0, 3),
            roots[0], cached=True,
        )
    logits, features, transaction = qwen38_tensorfold.forward_many(
        _fake_model(), [[1, 2, 3], [4]], [[-1, 0, 0], [-1]], caches, (0, 3),
        roots[0], cached=True, ragged=True,
    )
    assert logits.shape == (2, 3, 4) and features.shape == (2, 3, 4)
    transaction.commit_paths([[0, 2], [0]])
    assert caches[0][-1] == ("commit", [0, 2], 3, 5, [-1, 0, 0])
    assert caches[1][-1] == ("commit", [0], 1, 9, [-1])
    assert [item.width for item in transaction.transactions] == [3, 1]
    lane_multi = sys.modules["tensorfold.kernels.qwen.dense.v1.lane_multi"]
    assert lane_multi.MULTI_CALLS == 1


def test_row_sharing_is_in_the_route_receipt_and_policy_checks(monkeypatch):
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    from mlx2.runtime import external_speculative

    adapter = object.__new__(Qwen3827BAdapter)
    adapter.model = object()
    adapter.draft_model = object()
    adapter.identity = {"fingerprint": "test"}
    monkeypatch.setattr(Qwen3827BAdapter, "_external_num_draft", lambda self: 15)
    monkeypatch.setattr(Qwen3827BAdapter, "_initialize_external_feedback", lambda self: None)
    monkeypatch.setattr(Qwen3827BAdapter, "_external_execution_config",
                        lambda self, **kwargs: {"backend": "external_draft"})
    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", "/pinned/source")
    calls = []
    monkeypatch.setattr(external_speculative, "ExternalDraftBatchGenerator",
                        lambda *args, **kwargs: calls.append(kwargs) or object())
    adapter.external_policy = {
        "batch_size_route": "tree15_b1_b4_chain_b5plus_v1",
        "ast_tree_calibration": "order", "ast_batch_row_sharing": True,
    }
    config = adapter.execution_config(max_lanes=4, prefill_step=16)
    assert config["ast_batch_row_sharing"]["enabled"] is True
    adapter.create_external_batch()
    assert calls[-1]["ast_batch_row_sharing"] is True


def test_cli_row_sharing_needs_calibration():
    def args(**overrides):
        values = dict(model="/m/target", ast=True, ast_drafter="/m/dflash", ast_topology=None,
                      ast_graft_pld=False, draft_model=None, external_draft=False,
                      native_mtp=False, prompt_lookup=False)
        values.update(overrides)
        return SimpleNamespace(**values)

    resolution = SimpleNamespace(
        adapter_type=type("Adapter", (), {"ast_tree_tiles": True}),
        artifact={"has_mtp": True}, default_route="native_mtp",
    )
    with pytest.raises(ValueError, match="needs --ast-tree-calibration"):
        apply_ast_invocation(args(ast_batch_row_sharing=True), {}, resolution)
    namespace = args(ast_batch_row_sharing=True, ast_tree_calibration="order")
    policy = apply_ast_invocation(namespace, {}, resolution)
    assert policy["ast_batch_row_sharing"] is True
    assert namespace.ast_route["batch_row_sharing"] is True
