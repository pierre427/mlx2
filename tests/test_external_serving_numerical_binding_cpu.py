"""Merged serving laws bind learning and guard selected math before writes."""

import copy
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_map
from test_continuation_session_binding_cpu import bind
from test_standard_xpress_serving_cpu import drain, reference, tiny

from mlx2.runtime.apc_numerics import execution_numerics_identity
from mlx2.runtime.proposal_pool import PoolRound, ProposalPath
from mlx2.runtime.recurrent_state_codec import RecurrentStateCodecPolicy
from mlx2.serving import (
    ServingEngine,
    apc_semantic_namespace,
    row_exact_target_mutation_guard,
)


@pytest.fixture(autouse=True)
def cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def namespace(**laws):
    return apc_semantic_namespace("external-learning-v1", **laws)


@pytest.mark.parametrize("pool", [False, True])
def test_default_and_stock_namespace_preserve_every_existing_route_pin(pool):
    adapter = bind(pool=pool)
    old, identity, layout = (
        adapter.draft_model,
        copy.deepcopy(adapter.identity),
        adapter.layout,
    )
    default = namespace(
        lane_matmul_receipt={"law_id": "stock", "covered": {}},
        recurrent_state_codec=RecurrentStateCodecPolicy(),
    )
    assert default == "external-learning-v1"
    assert not adapter.bind_external_serving_namespace(default)
    assert (
        adapter.draft_model is old
        and adapter.identity == identity
        and adapter.layout == layout
    )


@pytest.mark.parametrize(
    "laws",
    [
        {"execution_numerics": execution_numerics_identity({"MLX_ENABLE_TF32": "1"})},
        {"execution_numerics": execution_numerics_identity({}, verify_bitexact=True)},
        {"execution_numerics": execution_numerics_identity({}, sp_qmm=True)},
        {"recurrent_state_codec": RecurrentStateCodecPolicy(enabled=True)},
    ],
)
def test_nondefault_effective_laws_split_model_session_route_keep_artifacts(laws):
    adapter = bind()
    old = adapter.draft_model
    registry = old.proposal_pool.ranking_registry
    records = old.source_records.copy()
    identity, layout = adapter.identity.copy(), adapter.layout
    physical_namespace = namespace(**laws)
    assert adapter.bind_external_serving_namespace(physical_namespace)
    new = adapter.draft_model
    assert new.backend is old.backend and new.providers == old.providers
    assert new.proposal_pool.ranking_registry is registry
    assert new.session.target_revision != old.session.target_revision
    assert new.session.session_revision != old.session.session_revision
    assert new.session.tokenizer_revision == old.session.tokenizer_revision
    assert registry.model_key(new.session) != registry.model_key(old.session)
    for name, source in new.source_records.items():
        assert source.revision == records[name].revision
        assert source.target_revision == new.session.target_revision
        assert source.session_revision == new.session.session_revision
        assert source.tokenizer_revision == records[name].tokenizer_revision
        assert registry.source_key(source) != registry.source_key(records[name])
    assert (
        adapter.identity["fingerprint"] != identity["fingerprint"]
        and adapter.layout != layout
    )
    assert adapter.descriptor.cache_layout == adapter.layout
    assert not adapter.bind_external_serving_namespace(physical_namespace)


def test_binding_refuses_after_execution_or_consumed_pool_without_partial_change():
    for executed in (False, True):
        adapter = bind()
        if executed:
            batch = adapter.create_external_batch()
        else:
            drafter = adapter.draft_model
            round = PoolRound(drafter.session.session_revision, "d" * 64, "request", 0)
            source = drafter.source_records["external"]
            selection = drafter.proposal_pool.select(
                [
                    ProposalPath(
                        "path",
                        "external",
                        source.revision,
                        source.session_revision,
                        round.context_revision,
                        (1, 2),
                    )
                ],
                round,
            )
            drafter.proposal_pool.commit_feedback(selection, (1, 2))
        old, identity = adapter.draft_model, adapter.identity.copy()
        with pytest.raises(ValueError, match="before execution|unconsumed"):
            adapter.bind_external_serving_namespace(
                namespace(execution_numerics={"version": 1, "sp_qmm": True})
            )
        assert adapter.draft_model is old and adapter.identity == identity
        if executed:
            batch.close()


def test_feedback_manager_receives_new_target_and_route_before_initialization(
    monkeypatch,
):
    adapter = bind(pool=False)
    adapter.external_policy["lilicorr_feedback"] = {"directory": "unused"}
    captured = []

    def manager(drafter, policy, **kwargs):
        captured.append(kwargs)
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(
        "mlx2.runtime.lilicorr_feedback.LiLiCorrFeedbackManager", manager
    )
    adapter.bind_external_serving_namespace(
        namespace(execution_numerics={"version": 1, "sp_qmm": True})
    )
    adapter._initialize_external_feedback()
    assert captured == [
        {
            "target_revision": adapter._external_target_revision,
            "draft_revision": adapter._external_draft_revision,
            "binding": adapter.identity["fingerprint"],
        }
    ]


def test_rebound_pool_runs_and_publishes_revision_bound_paired_sidecar():
    adapter = bind()
    adapter.bind_external_serving_namespace(
        namespace(execution_numerics={"version": 1, "sp_qmm": True})
    )
    batch = adapter.create_external_batch(prefill_step_size=64)
    try:
        uid = batch.insert([[1, 2, 3]], max_tokens=[4])[0]
        tokens, finishes = drain(batch)
        assert tokens[uid] == reference(adapter.model, [1, 2, 3], 4)
        finish = finishes[uid]
        finish.cache_sidecar.validate(
            adapter.identity["fingerprint"], len(finish.all_tokens)
        )
        ranking = finish.speculative_receipt["continuation_pool"]["ranking"][
            "ranking_registry"
        ]
        assert ranking["target_revision"] == adapter._external_target_revision
        assert ranking["model_counts"] and ranking["session_counts"]
    finally:
        batch.close()


def selected_target():
    model, _ = tiny()
    model.update(tree_map(lambda value: value.astype(mx.bfloat16), model.parameters()))
    model.configure_target_verify_row_exact(True)
    return SimpleNamespace(model=model)


def test_lane_guard_is_before_mutation_and_preserves_stock_device_skip_noops(
    monkeypatch,
):
    adapter = selected_target()
    classes = [(name, type(module)) for name, module in adapter.model.named_modules()]
    policy = {"mode": "crossover", "min_rows": {"bf16": 8}, "max_rows": 64, "skip": []}
    monkeypatch.setattr("mlx2.runtime.lane.available", lambda: False)
    row_exact_target_mutation_guard(adapter, lane_policy=policy)
    monkeypatch.setattr("mlx2.runtime.lane.available", lambda: True)
    for no_op in (
        {**policy, "mode": "off"},
        {**policy, "skip": ["*"]},
        {**policy, "min_rows": {"bf16": 128}},
    ):
        row_exact_target_mutation_guard(adapter, lane_policy=no_op)
    with pytest.raises(ValueError, match="native projections"):
        row_exact_target_mutation_guard(adapter, lane_policy=policy)
    assert classes == [
        (name, type(module)) for name, module in adapter.model.named_modules()
    ]
    # A storage codec on a nonrecurrent target is harmless to native math.
    row_exact_target_mutation_guard(adapter)


def test_lora_guard_is_reached_before_actual_service_install(monkeypatch):
    engine = ServingEngine.__new__(ServingEngine)
    engine.adapter = selected_target()
    engine.model_path = Path("tiny")
    engine.lora_session = engine.multi_lora = None
    engine.status = lambda: {"model": "tiny"}
    engine._resolve_lora_path = lambda _path: Path("unused")
    engine._exclusive_adapter_operation = lambda _name, operation: operation(
        engine.adapter
    )
    installed = []
    monkeypatch.setattr(
        "mlx2.runtime.lora.install_lora", lambda *a, **kw: installed.append(True)
    )
    with pytest.raises(ValueError, match="dynamic LoRA"):
        engine.load_lora_adapter("adapter", "unused")
    assert not installed


def test_replacement_projection_rejected_before_first_cache_write():
    adapter = selected_target()

    class Replacement(nn.Linear):
        pass

    adapter.model.layers[-1].mlp.down_proj.__class__ = Replacement
    cache = adapter.model.make_cache()
    with pytest.raises(ValueError, match="native unquantized Linear"):
        adapter.model.forward_with_taps(mx.array([[1, 2]]), cache, [0, 2])
    assert all(c.offset == 0 and c.keys is None and c.values is None for c in cache)
