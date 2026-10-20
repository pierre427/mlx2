"""Strict packed-cohort authority and dispatch; no real artifact/GPU claim."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.adapters.qwen38_target_tree_executor import (
    AST_FAST_COHORT_ORACLE_SCHEMA,
    AST_FAST_TREE_ORACLE_CHECKS,
    AST_FAST_TREE_ORACLE_SCHEMA,
    Qwen38TargetTreeExecutor,
    ast_fast_cohort_domain,
)
from mlx2.runtime.ast_planner import ExecutionProfile
from test_qwen38_fast_ast_cpu import _fake_installed


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _fixture(monkeypatch, tmp_path):
    model, route, receipt = _fake_installed(monkeypatch)
    route.bind(receipt)
    provider = Qwen38TargetTreeExecutor((), "cpu-source", ast_fast_tree=True)
    context = {"source": "cpu-source", "projection_law": "lane_mpp_exact"}
    monkeypatch.setattr(
        Qwen38TargetTreeExecutor, "_ast_context", lambda self, **kwargs: context
    )
    profile = ExecutionProfile(
        binding="cpu",
        device="cpu",
        build="test",
        max_lanes=1,
        tile_widths=tuple(range(1, 16)),
        terminal_widths=tuple(range(2, 17)),
        packed_min_width=1,
        row_costs=tuple(range(1, 17)),
        launch_cost=0,
        context_row_cost=0,
        draft_cost=1,
    )
    kwargs = dict(model=model, target_binding={}, artifact="cpu", capture_layers=(1,))
    document = {
        "schema": AST_FAST_COHORT_ORACLE_SCHEMA,
        "bindings": [provider.ast_oracle_binding(profile, **kwargs)],
        "input_widths": list(range(1, 17)),
        "checks": dict.fromkeys(AST_FAST_TREE_ORACLE_CHECKS, True),
        "cohort_domain": ast_fast_cohort_domain(),
    }
    path = tmp_path / "cohort.json"

    def load(value=document):
        path.write_text(json.dumps(value))
        return provider.with_ast_oracle(path, **kwargs)

    return model, route, provider, profile, document, load


def test_v2_cohort_domain_is_distinct_from_old_b1_evidence(monkeypatch, tmp_path):
    _model, _route, _provider, profile, document, load = _fixture(monkeypatch, tmp_path)
    admitted = load()
    assert admitted.ast_tree_max_sessions == 4
    assert admitted.ast_tree_cohort_input_widths(2) == tuple(range(1, 9))
    assert admitted.ast_tree_cohort_input_widths(3) == tuple(range(1, 6))
    assert admitted.ast_tree_cohort_input_widths(4) == tuple(range(1, 5))
    assert admitted.ast_tree_cohort_input_widths(True) == ()
    assert admitted.ast_execution_backend(profile).maximum_sessions == 1
    assert not admitted.ast_cohort_probe
    legacy = copy.deepcopy(document)
    legacy["schema"] = AST_FAST_TREE_ORACLE_SCHEMA
    legacy.pop("cohort_domain")
    assert load(legacy).ast_tree_max_sessions == 1


@pytest.mark.parametrize(
    "fault", ["missing", "false", "integer", "width", "extra", "b1"]
)
def test_cohort_evidence_refuses_incomplete_or_broadened_domain(
    monkeypatch, tmp_path, fault
):
    _model, _route, _provider, _profile, document, load = _fixture(
        monkeypatch, tmp_path
    )
    value = copy.deepcopy(document)
    if fault == "missing":
        value.pop("cohort_domain")
    elif fault in ("false", "integer"):
        value["cohort_domain"]["checks"]["mixed_path_commit"] = (
            False if fault == "false" else 1
        )
    elif fault == "width":
        value["cohort_domain"]["input_widths_by_sessions"]["4"].append(5)
    elif fault == "extra":
        value["cohort_domain"]["unbound"] = True
    else:
        value["input_widths"].remove(16)
    with pytest.raises(ValueError, match="cohort"):
        load(value)


def test_fast_packed_dispatch_requires_live_owner_law_and_exact_domain(
    monkeypatch, tmp_path
):
    from mlx2.runtime import qwen38_tensorfold, row_exact_verify as REV

    model, route, provider, profile, document, load = _fixture(monkeypatch, tmp_path)
    admitted = load()
    admitted.ast_execution_backend(profile)
    calls = []

    def forward_many(_model, inputs, parents, caches, layers, **kwargs):
        assert REV.current() is None
        assert kwargs == {"cached": True, "validated_token": "cpu-source"}
        calls.append((inputs, parents, caches))
        return "logits", "features", SimpleNamespace(physical_packed=True)

    monkeypatch.setattr(qwen38_tensorfold, "forward_many", forward_many)
    inputs = [[1, 2, 3, 4], [5, 6, 7, 8]]
    parents = [[-1, 0, 1, 2], [-1, 0, 0, 1]]
    caches = [[object()], [object()]]
    result = admitted.forward_many(
        model, inputs, parents, caches, (1,), cached=True, configured_limit=4
    )
    assert result[2].provider_revision_token == "cpu-source"
    assert result[2].physical_packed and len(calls) == 1
    for changes in (
        {"caches": [caches[0], caches[0]]},
        {"caches": [caches[0], list(caches[0])]},
        {"inputs": [[1] * 9, [2] * 9], "parents": [list(range(-1, 8))] * 2},
        {"inputs": [[1], [2, 3]], "parents": [[-1], [-1, 0]]},
        {"parents": [[-1, 0, 1, 2], [-1, 1, 0, 1]]},
        {"layers": (2,)},
        {"cached": False},
        {"model": SimpleNamespace(_mlx2_qwen38_fast_ast=route)},
    ):
        args = dict(
            model=model,
            inputs=inputs,
            parents=parents,
            caches=caches,
            layers=(1,),
            cached=True,
            configured_limit=4,
        )
        with pytest.raises(ValueError):
            admitted.forward_many(**{**args, **changes})
    assert len(calls) == 1
    with pytest.raises(ValueError, match="B1-only"):
        provider.forward_many(
            model, inputs, parents, caches, (1,), cached=True, configured_limit=4
        )
    probe = replace(provider, ast_cohort_probe=True)
    probe.forward_many(
        model, inputs, parents, caches, (1,), cached=True, configured_limit=4
    )
    assert probe.ast_tree_max_sessions == 1
    assert probe.ast_tree_cohort_input_widths(2) == ()
    foreign = copy.deepcopy(document)
    foreign["bindings"][0]["source"] = "foreign"
    with pytest.raises(ValueError, match="live profile"):
        load(foreign).ast_execution_backend(profile)
