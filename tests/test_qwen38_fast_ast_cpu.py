"""Fast-law admission and ordinary rollback mechanics; no MPP/GPU evidence."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.qwen38_fast_ast import Qwen38FastAST
from mlx2.adapters.qwen38_target_tree_executor import (
    AST_FAST_TREE_ORACLE_CHECKS,
    AST_FAST_TREE_ORACLE_SCHEMA,
    Qwen38TargetTreeExecutor,
)
from mlx2.runtime import row_exact_verify as REV
from mlx2.runtime.ast_planner import ExecutionProfile
from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _fake_installed(monkeypatch):
    from mlx2.runtime.lane import installer

    group = object()
    prepared = SimpleNamespace(backend="mpp", format="q4g64")
    modules = [
        SimpleNamespace(
            _lane_min_rows=1,
            _lane_max_rows=32,
            _lane_group=group,
            weight=mx.zeros((2, 2)),
        )
        for _ in range(2)
    ]
    model = SimpleNamespace(named_modules=lambda: list(zip(("a", "b"), modules)))
    monkeypatch.setattr(installer, "installed_lane", lambda _module: prepared)
    route = Qwen38FastAST(model)
    receipt = {
        "backend": "mpp",
        "available": True,
        "covered": {"q4g64": 2},
        "groups": {"q4g64x2": 1},
        "law_id": "lane-matmul-v1+grouped",
        "min_rows": {"q4": 1, "q5": 1},
        "max_rows": 32,
    }
    return model, route, receipt


def test_fast_lane_requires_actual_exact_grouped_identity(monkeypatch):
    from mlx2.runtime.lane import installer

    model, route, receipt = _fake_installed(monkeypatch)
    with pytest.raises(ValueError, match="absent"):
        route.require_bound()
    # Exact mode itself sets thresholds to1 in apply_policy.
    route.validate_policy(
        {"mode": "exact", "grouping": True, "max_rows": 32, "min_rows": {"q4": 8}}
    )
    for change in (
        {"mode": "auto"},
        {"grouping": False},
        {"max_rows": 8},
    ):
        with pytest.raises(ValueError, match="grouped exact"):
            route.validate_policy(
                {
                    "mode": "exact",
                    "grouping": True,
                    "max_rows": 32,
                    "min_rows": {"q4": 8},
                    **change,
                }
            )
    for change in (
        {"backend": "simd"},
        {"available": False},
        {"groups": {}},
        {"min_rows": {"q4": 8}},
        {"max_rows": 8},
    ):
        with pytest.raises(ValueError, match="installed grouped"):
            route.bind({**receipt, **change})
    first = route.bind(receipt)
    first["groups"].clear()
    assert route.identity()["groups"] == [["a", "b"]]
    with REV.window(REV.Window(4)), pytest.raises(ValueError, match="changed"):
        route.require_bound()
    monkeypatch.setitem(installer.__dict__, "GROUPING", [False])
    with pytest.raises(ValueError, match="changed"):
        route.require_bound()
    assert model.progressive_external_exact_max_lanes == 1


@pytest.mark.parametrize("accepted", [0, 2, 4])
def test_ordinary_same_law_fallback_state_and_continuation(monkeypatch, accepted):
    from test_qwen38_dflash2_cpu import tiny_target

    model = tiny_target()
    route = Qwen38FastAST(model)
    # CPU tests exercise state/dispatch only, not claim an installed MPP kernel.
    monkeypatch.setattr(route, "require_bound", lambda: None)
    base = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3]]), cache=base))
    tokens = [7, 8, 9, 10]
    ordinary = copy.deepcopy(base)
    logits, features = [], []
    for token in tokens:
        out, taps = model.forward_with_taps(mx.array([[token]]), ordinary, (1, 6))
        logits.append(out)
        features.append(taps)
    owner = HybridVerifyRows([copy.deepcopy(base)])
    transaction = owner.begin([4])
    result = route.execute(mx.array([tokens]), transaction.caches, (1, 6))
    assert REV.current() is None
    np.testing.assert_array_equal(
        np.asarray(result.logits), np.asarray(mx.concatenate(logits, axis=1))
    )
    np.testing.assert_array_equal(
        np.asarray(result.features), np.asarray(mx.concatenate(features, axis=1))
    )
    if accepted:
        current = transaction.commit([accepted])[0]
    else:
        transaction.abort()
        current = owner.rows[0]
    reference = copy.deepcopy(base)
    for token in tokens[:accepted]:
        model.forward_with_taps(mx.array([[token]]), reference, (1, 6))
    out = model(mx.array([[11]]), cache=current)
    expected = model(mx.array([[11]]), cache=reference)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))


def test_fast_tree_forward_never_enters_rev_and_refuses_foreign_geometry(monkeypatch):
    from mlx2.runtime import qwen38_tensorfold

    _model, route, receipt = _fake_installed(monkeypatch)
    route.bind(receipt)
    provider = Qwen38TargetTreeExecutor((), "cpu-source", ast_fast_tree=True)
    seen = []

    def forward(model, inputs, parents, *args, **kwargs):
        assert REV.current() is None
        seen.append(tuple(parents))
        return "logits", "features", SimpleNamespace()

    monkeypatch.setattr(qwen38_tensorfold, "forward", forward)
    provider.forward(_model, [1, 2, 3, 4], [-1, 0, 0, 2], [], (1,), cached=True)
    assert seen == [(-1, 0, 0, 2)]
    for parents in ([-1, -1], [-1, 1], [-1, True]):
        with pytest.raises(ValueError, match="shared-root"):
            provider.forward(_model, [1, 2], parents, [], (1,), cached=True)
    with pytest.raises(ValueError, match="B1-only"):
        provider.forward_many(
            _model, [[1]], [[-1]], [[]], (1,), cached=True, configured_limit=1
        )
    with REV.window(REV.Window(4)), pytest.raises(ValueError, match="changed"):
        provider.forward(_model, [1, 2], [-1, 0], [], (1,), cached=True)
    assert len(seen) == 1


def test_ordinary_fallback_failure_rewinds_existing_transaction(monkeypatch):
    from test_qwen38_dflash2_cpu import tiny_target

    model = tiny_target()
    route = Qwen38FastAST(model)
    monkeypatch.setattr(route, "require_bound", lambda: None)
    base = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3]]), cache=base))
    owner = HybridVerifyRows([copy.deepcopy(base)])
    transaction = owner.begin([4])
    ordinary = model.forward_with_taps
    calls = []

    def fail(tokens, cache, layers):
        result = ordinary(tokens, cache, layers)
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected partial target failure")
        return result

    monkeypatch.setattr(model, "forward_with_taps", fail)
    with pytest.raises(RuntimeError, match="injected partial"):
        route.execute(mx.array([[7, 8, 9, 10]]), transaction.caches, (1, 6))
    transaction.abort()
    actual = model(mx.array([[11]]), cache=owner.rows[0])
    reference = model(mx.array([[11]]), cache=copy.deepcopy(base))
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))


def test_adapter_selects_distinct_protocol_without_rev():
    from test_qwen38_dflash2_cpu import tiny_target
    from mlx2.adapters.qwen38_27b import (
        Qwen3827BAdapter,
        validate_target_verify_row_exact,
    )

    adapter = Qwen3827BAdapter.__new__(Qwen3827BAdapter)
    adapter.model = tiny_target()
    adapter.external_policy = {
        "ast_target_tree_projection_law": "lane_mpp_exact",
        "target_verify_row_exact": False,
        "progressive_verification_tile": "auto",
        "progressive_verification_fidelity": "exact",
    }
    assert validate_target_verify_row_exact(adapter.external_policy) is False
    adapter._select_target_verify_row_exact()
    assert adapter.target_verify_row_exact is None
    assert adapter.fast_ast_target.model is adapter.model
    assert getattr(adapter.model, "_mlx2_qwen_hybrid_row_exact", None) is None
    assert adapter.model.progressive_external_exact_terminal_input_widths[-1] == 16
    with pytest.raises(ValueError, match="without"):
        validate_target_verify_row_exact(
            {**adapter.external_policy, "target_verify_row_exact": True}
        )
    with pytest.raises(ValueError, match="installed grouped"):
        adapter.bind_serving_lane_matmul(None)


def test_fast_oracle_separates_tree_law_from_rev_chain(monkeypatch, tmp_path):
    from mlx2.runtime import qwen38_tensorfold

    model, route, receipt = _fake_installed(monkeypatch)
    route.bind(receipt)
    provider = Qwen38TargetTreeExecutor((), "cpu-source", ast_fast_tree=True)
    context = {
        "projection_law": "lane_mpp_exact",
        "topology_law": "parent_ordered_rooted_tree_v1",
        "lane_numerical_identity": route.identity(),
    }
    monkeypatch.setattr(
        Qwen38TargetTreeExecutor, "_ast_context", lambda self, **kwargs: context
    )
    profile = ExecutionProfile(
        binding="cpu",
        device="cpu",
        build="test",
        max_lanes=1,
        tile_widths=tuple(range(1, 7)),
        terminal_widths=tuple(range(2, 8)),
        packed_min_width=1,
        row_costs=tuple(range(1, 17)),
        launch_cost=0,
        context_row_cost=0,
        draft_cost=1,
    )
    kwargs = dict(model=model, target_binding={}, artifact="cpu", capture_layers=(1,))
    document = {
        "schema": AST_FAST_TREE_ORACLE_SCHEMA,
        "bindings": [provider.ast_oracle_binding(profile, **kwargs)],
        "input_widths": list(range(1, 17)),
        "checks": dict.fromkeys(AST_FAST_TREE_ORACLE_CHECKS, True),
    }
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(document))
    admitted = provider.with_ast_oracle(path, **kwargs)
    assert admitted.ast_tree_tile_rows == 16
    assert admitted.ast_execution_backend(profile).input_widths == tuple(range(1, 17))
    assert admitted.receipt()["ast_target_tree"]["projection_law"] == "lane_mpp_exact"
    seen = []

    def forward(_model, inputs, parents, *args, **kwargs):
        assert REV.current() is None
        seen.append(tuple(parents))
        return "logits", "features", SimpleNamespace()

    monkeypatch.setattr(qwen38_tensorfold, "forward", forward)
    admitted.forward(model, [1, 2, 3, 4], [-1, 0, 0, 2], [], (1,), cached=True)
    assert seen == [(-1, 0, 0, 2)]
    chain = replace(admitted, ast_fast_tree=False, ast_chain=True)
    with pytest.raises(ValueError, match="linear chain"):
        chain.forward(model, [1, 2, 3, 4], [-1, 0, 0, 2], [], (1,), cached=True)
    assert len(seen) == 1
    with pytest.raises(ValueError, match="model binding"):
        admitted.forward(
            SimpleNamespace(_mlx2_qwen38_fast_ast=route),
            [1],
            [-1],
            [],
            (1,),
            cached=True,
        )
    changed = copy.deepcopy(document)
    changed["bindings"][0]["lane_numerical_identity"]["lane_law_id"] = "foreign"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="live profile"):
        provider.with_ast_oracle(path, **kwargs).ast_execution_backend(profile)
    document["checks"].pop("branch_paths")
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="incomplete"):
        provider.with_ast_oracle(path, **kwargs)
    assert replace(provider, ast_fast_tree=False).ast_tree_tile_rows == 0
