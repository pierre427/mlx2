"""Offline receipt admission contracts, not TensorFold numerical qualification."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.adapters.external_draft_policy import (
    normalize_progressive_external_policy,
    progressive_external_identity,
)
from mlx2.adapters.qwen38_target_tree_executor import (
    AST_CHAIN_ORACLE_CHECKS,
    AST_CHAIN_ORACLE_SCHEMA,
    Qwen38TargetTreeExecutor,
)
from mlx2.runtime.ast_planner import ExecutionProfile


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _policy():
    return {
        "num_draft": 7,
        "progressive_verification_tile": "auto",
        "progressive_verification_fidelity": "exact",
        "progressive_alternative_branches": 1,
        "continuation_pool": {
            "sources": ["external"],
            "limit": 1,
            "ast_planner": True,
        },
    }


def _fixture(monkeypatch, tmp_path):
    context = {
        "source_revision": "a" * 40,
        "source_hashes": {"source.py": "b" * 64},
        "artifact": "cpu-model-artifact",
        "executable_model_manifest": "c" * 64,
        "target_protocol_binding": {"fidelity": "exact"},
        "capture_layers": [0, 1],
        "provider_protocol": "qwen38_tensorfold_v1",
        "provider_revision_token": "provider-source",
        "fidelity": "exact",
        "layout": "independent_private_b1_chain",
        "projection_law": "row_exact_valid_rows_window_v1",
    }
    monkeypatch.setattr(
        Qwen38TargetTreeExecutor,
        "_ast_context",
        lambda _self, **_kwargs: copy.deepcopy(context),
    )
    provider = Qwen38TargetTreeExecutor((), "provider-source", ast_chain=True)
    profile = ExecutionProfile(
        binding="cpu-profile",
        device="cpu",
        build="cpu-test",
        max_lanes=4,
        tile_widths=(1, 2, 3, 4, 5, 6),
        terminal_widths=(2, 3, 4, 5, 6, 7),
        packed_min_width=1,
        row_costs=tuple(float(n) for n in range(1, 29)),
        launch_cost=0,
        context_row_cost=0,
        draft_cost=1,
    )
    kwargs = dict(
        model=object(),
        target_binding={"fidelity": "exact"},
        artifact="cpu-model-artifact",
        capture_layers=(0, 1),
    )
    binding = provider.ast_oracle_binding(profile, **kwargs)
    document = {
        "schema": AST_CHAIN_ORACLE_SCHEMA,
        "bindings": [binding],
        "input_widths": list(range(1, 8)),
        "checks": {key: True for key in AST_CHAIN_ORACLE_CHECKS},
    }
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(document))
    return provider, profile, kwargs, document, path


def test_exact_primary_control_and_candidate_share_numerical_identity():
    control = normalize_progressive_external_policy(_policy(), num_draft=7)
    selected = normalize_progressive_external_policy(
        {**_policy(), "ast_target_tree_oracle": "/offline/oracle.json"}, num_draft=7
    )
    assert progressive_external_identity(control) == progressive_external_identity(
        selected
    )
    for change in (
        {"progressive_verification_fidelity": "near_tie_bounded"},
        {"progressive_alternative_branches": 2},
        {"ast_physical_target": True},
        {"ast_target_tree_oracle": False},
        {
            "continuation_pool": {
                "sources": ["external"],
                "limit": 2,
                "ast_planner": True,
            }
        },
    ):
        with pytest.raises(ValueError):
            normalize_progressive_external_policy({**selected, **change}, num_draft=7)


def test_loader_admits_only_probed_input_widths_without_hot_canaries(
    monkeypatch, tmp_path
):
    provider, profile, kwargs, document, path = _fixture(monkeypatch, tmp_path)
    document["input_widths"] = [1, 3, 4, 5, 6]
    path.write_text(json.dumps(document))
    admitted = provider.with_ast_oracle(path, **kwargs)
    backend = admitted.ast_execution_backend(profile)
    assert backend.input_widths == (1, 3, 4, 5, 6)
    assert backend.oracle.equivalent is True
    assert backend.cost(2) == float("inf")
    from mlx2.runtime import qwen38_tensorfold

    observed = []
    monkeypatch.setattr(
        qwen38_tensorfold,
        "forward",
        lambda *_args, **_kwargs: observed.append(True) or (1, 2, SimpleNamespace()),
    )
    admitted.forward(None, [7], [-1], [], (0, 1), cached=False)
    assert observed == [True]
    # f567bfd7a generalised the refusal to "qualified geometry" when the
    # same admission check began to cover AST fast trees.
    with pytest.raises(ValueError, match="outside its qualified geometry"):
        admitted.forward(None, [7, 8], [-1, 0], [], (0, 1), cached=False)
    with pytest.raises(ValueError, match="linear chain"):
        admitted.forward(None, [7, 8, 9], [-1, 0, 0], [], (0, 1), cached=False)
    with pytest.raises(ValueError, match="outside its qualified geometry"):
        admitted.forward(None, [7], [-1], [], (0,), cached=False)
    assert observed == [True]


@pytest.mark.parametrize(
    "field",
    [
        "source_revision",
        "artifact",
        "executable_model_manifest",
        "capture_layers",
        "provider_revision_token",
        "source_hashes",
        "projection_law",
    ],
)
def test_stale_or_foreign_offline_binding_refused(monkeypatch, tmp_path, field):
    provider, profile, kwargs, document, path = _fixture(monkeypatch, tmp_path)
    document["bindings"][0][field] = "foreign"
    path.write_text(json.dumps(document))
    candidate = provider.with_ast_oracle(path, **kwargs)
    with pytest.raises(ValueError, match="live profile"):
        candidate.ast_execution_backend(profile)


def test_live_profile_build_device_and_failed_or_malformed_evidence_refused(
    monkeypatch, tmp_path
):
    provider, profile, kwargs, document, path = _fixture(monkeypatch, tmp_path)
    candidate = provider.with_ast_oracle(path, **kwargs)
    for other in (
        replace(profile, binding="other"),
        replace(profile, build="other"),
        replace(profile, device="other"),
    ):
        with pytest.raises(ValueError, match="live profile"):
            candidate.ast_execution_backend(other)
    bad = copy.deepcopy(document)
    bad["checks"]["continuation"] = False
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="did not pass"):
        provider.with_ast_oracle(path, **kwargs)
    bad = copy.deepcopy(document)
    bad["input_widths"] = [True]
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="widths"):
        provider.with_ast_oracle(path, **kwargs)
    bad = copy.deepcopy(document)
    bad["unknown"] = True
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="incomplete"):
        provider.with_ast_oracle(path, **kwargs)


def test_ast_chain_projection_scope_is_explicit_and_clears_inherited_geometry(
    monkeypatch,
):
    from mlx2.runtime import qwen38_tensorfold, row_exact_verify as REV

    provider = Qwen38TargetTreeExecutor((), "test", ast_chain=True)
    outer = REV.Window(3)
    observed = []

    def forward(*args, **kwargs):
        window = REV.current()
        observed.append(window)
        assert window.rows == 2 and window is not outer
        assert REV.compact_rows(mx.zeros((2, 2, 4)), stage="projection") is None
        window.note("projection", "row_exact_qmv", 2)
        return 1, 2, SimpleNamespace()

    monkeypatch.setattr(qwen38_tensorfold, "forward", forward)
    with REV.valid_rows((2, 1), 2), REV.window(outer):
        result = provider.forward(None, [7, 8], [-1, 0], [], (0,), cached=False)
        assert REV.current() is outer
        assert REV.compact_rows(mx.zeros((2, 2, 4)), stage="outer") is not None
    assert REV.current() is None
    assert result[2].ast_projection_window == {
        "rows": 2,
        "stages": {"projection": {"row_exact_qmv": 2}},
        "failures": {},
    }
    assert observed[0].exact

    # Historical tree execution neither opens this law nor constrains parents.
    monkeypatch.setattr(
        qwen38_tensorfold,
        "forward",
        lambda *_args, **_kwargs: (REV.current(), 2, SimpleNamespace()),
    )
    legacy = replace(provider, ast_chain=False)
    assert legacy.forward(None, [7, 8], [-1, -1], [], (0,), cached=False)[0] is None
    with pytest.raises(ValueError, match="linear chain"):
        provider.forward(None, [7, 8], [-1, -1], [], (0,), cached=False)
    with pytest.raises(ValueError, match="B1-only"):
        provider.forward_many(
            None, [[7]], [[-1]], [[]], (0,), cached=False, configured_limit=1
        )


def test_projection_failure_aborts_returned_private_transaction(monkeypatch):
    from mlx2.runtime import qwen38_tensorfold, row_exact_verify as REV

    private = [0]
    transaction = SimpleNamespace(abort=lambda: private.__setitem__(0, 0))

    def forward(*args, **kwargs):
        private[0] = 2
        REV.current().fail("unsupported_projection")
        return 1, 2, transaction

    monkeypatch.setattr(qwen38_tensorfold, "forward", forward)
    provider = Qwen38TargetTreeExecutor((), "test", ast_chain=True)
    with pytest.raises(RuntimeError, match="unsupported_projection"):
        provider.forward(None, [7, 8], [-1, 0], private, (0,), cached=False)
    assert private == [0] and REV.current() is None


def test_paired_timing_prior_uses_common_units_and_declines_nonpositive_fit(
    monkeypatch, tmp_path
):
    provider, profile, kwargs, document, path = _fixture(monkeypatch, tmp_path)
    profile = replace(profile, launch_cost=2, context_row_cost=0.1)
    # A numerical oracle without timing evidence has no invented weight-pass win.
    baseline = provider.with_ast_oracle(path, **kwargs).ast_execution_backend(profile)
    assert baseline.cost(4, 10) == profile.cost((4,), (10,))
    assert baseline.cost_authority == "ordinary_exact_projection_prior_v1"
    document["target_timings"] = [
        {
            "input_width": 4,
            "context_length": 10,
            "ordinary_seconds": 1.0,
            "candidate_seconds": 0.8,
        },
        {
            "input_width": 3,
            "context_length": 10,
            "ordinary_seconds": 1.0,
            "candidate_seconds": 0.1,
        },
    ]
    path.write_text(json.dumps(document))
    backend = provider.with_ast_oracle(path, **kwargs).ast_execution_backend(profile)
    assert backend.cost(4, 10) == pytest.approx(profile.cost((4,), (10,)) * 0.8)
    assert backend.cost(3, 10) == profile.cost((3,), (10,))
    assert backend.cost(2, 10) == profile.cost((2,), (10,))
    assert (
        backend.cost_authority
        == "offline_paired_target_successor_timing_at_probe_context_v1"
    )
    assert backend.timing_boundary == "target_successor"


def test_invalid_or_unprobed_timing_evidence_refused(monkeypatch, tmp_path):
    provider, profile, kwargs, document, path = _fixture(monkeypatch, tmp_path)
    valid = {
        "input_width": 4,
        "context_length": 10,
        "ordinary_seconds": 1.0,
        "candidate_seconds": 0.8,
    }
    for samples in (
        [{**valid, "candidate_seconds": float("nan")}],
        [{**valid, "candidate_seconds": True}],
        [{**valid, "ordinary_seconds": 0}],
        [{**valid, "input_width": 8}],
        [{**valid, "context_length": -1}],
        [valid, valid],
        [None],
    ):
        document["target_timings"] = samples
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="timing evidence"):
            provider.with_ast_oracle(path, **kwargs)
