"""Adapter-selected math pins cache and learning without artifact migration."""

import copy
from types import SimpleNamespace

import pytest
from test_continuation_session_binding_cpu import bind
from test_parallel_draft_apcv2_batching_cpu import pair
from test_standard_xpress_serving_cpu import drain, reference

from mlx2.adapters.external_draft_policy import ExternalDraftAdapterMixin
from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
from mlx2.contracts import Capability, ModelDescriptor
from mlx2.runtime.apc_v2 import APCv2
from mlx2.runtime.models.qwen3_5 import TextModelArgs
from mlx2.runtime.models.qwen38_fused_gdn import GatedDeltaNet
from mlx2.serving import adapter_numerical_contract, apc_semantic_namespace


def selected_contract(enabled):
    layer = GatedDeltaNet(
        TextModelArgs(
            hidden_size=16,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
        )
    )
    layer.set_fused_gdn_enabled(enabled)
    adapter = SimpleNamespace(
        fused_gdn=enabled,
        model=SimpleNamespace(named_modules=lambda: [("layer", layer)]),
    )
    adapter.execution_numerics_contract = lambda: (
        Qwen3827BAdapter.execution_numerics_contract(adapter)
    )
    return adapter_numerical_contract(adapter)


def namespace(semantic, enabled):
    return apc_semantic_namespace(
        semantic, adapter_execution_numerics=selected_contract(enabled)
    )


@pytest.mark.parametrize("pool", [False, True])
def test_disabled_contract_preserves_route_cache_source_and_session_identity(pool):
    adapter = bind(pool=pool)
    old = adapter.draft_model
    identity, layout = copy.deepcopy(adapter.identity), adapter.layout
    assert selected_contract(False) is None
    assert adapter_numerical_contract(SimpleNamespace()) is None
    assert namespace("external-learning-v1", False) == "external-learning-v1"
    assert namespace("apc-semantic", False) == "apc-semantic"
    assert not adapter.bind_external_serving_namespace(
        namespace("external-learning-v1", False)
    )
    assert adapter.draft_model is old
    assert adapter.identity == identity and adapter.layout == layout


def test_selected_law_splits_pool_source_session_and_apc_without_changing_artifacts():
    ordinary, selected = bind(), bind()
    before = selected.draft_model
    artifacts = selected.identity.copy()
    assert selected.bind_external_serving_namespace(
        namespace("external-learning-v1", True)
    )
    after = selected.draft_model
    assert after.session.target_revision != before.session.target_revision
    assert after.session.session_revision != before.session.session_revision
    assert after.session.tokenizer_revision == before.session.tokenizer_revision
    assert after.backend is before.backend and after.providers == before.providers
    assert after.proposal_pool.ranking_registry is before.proposal_pool.ranking_registry
    for name, source in after.source_records.items():
        old = before.source_records[name]
        assert source.revision == old.revision
        assert source.target_revision != old.target_revision
        assert source.session_revision != old.session_revision
        assert source.tokenizer_revision == old.tokenizer_revision
    assert selected.identity["target_fingerprint"] == artifacts["target_fingerprint"]
    assert selected.identity["draft_fingerprint"] == artifacts["draft_fingerprint"]
    assert selected._external_draft_revision == ordinary._external_draft_revision
    # Exact same artifact parameters: the semantic law alone splits APC keys.
    keys = [
        APCv2.key(
            "target",
            revision="revision",
            adapter="adapter",
            tokenizer_fingerprint="tokenizer",
            cache_layout_fingerprint="layout",
            semantic_fingerprint=namespace("semantic", enabled),
        )
        for enabled in (False, True)
    ]
    assert keys[0] != keys[1]
    assert selected.identity["fingerprint"] != ordinary.identity["fingerprint"]
    assert not selected.bind_external_serving_namespace(
        namespace("external-learning-v1", True)
    )


def test_selected_law_sidecar_and_source_ranking_follow_execution_pin():
    adapter = bind()
    adapter.bind_external_serving_namespace(namespace("external-learning-v1", True))
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


def direct_adapter(enabled):
    model, drafter = pair("xpress")
    adapter = ExternalDraftAdapterMixin()
    adapter.model, adapter.layout = model, "target-layout"
    adapter.identity = {"fingerprint": "a" * 64}
    adapter.external_policy = {"num_draft": 2, "continuation_pool": {}}
    adapter.execution_numerics_contract = lambda: selected_contract(enabled)
    adapter._bind_external_drafter(
        {"fingerprint": "b" * 64, "args": drafter.config},
        lambda _record, _target: drafter,
        ModelDescriptor(
            model_type="qwen3",
            family="qwen3",
            variant="tiny",
            cache_layout="tiny",
            capabilities=frozenset({Capability.TEXT}),
            state_planes=frozenset(),
        ),
    )
    return adapter


def test_direct_adapter_pins_selected_math_before_source_creation_and_preserves_off():
    disabled, selected = direct_adapter(False), direct_adapter(True)
    assert disabled.identity == bind().identity
    first, second = disabled.draft_model, selected.draft_model
    assert first.session.target_revision != second.session.target_revision
    assert first.session.session_revision != second.session.session_revision
    assert (
        first.session.tokenizer_revision
        == second.session.tokenizer_revision
        == "a" * 64
    )
    assert (
        selected.identity["target_execution_fingerprint"]
        == second.session.target_revision
    )
    assert selected.identity["target_fingerprint"] == "a" * 64
    assert selected.identity["draft_fingerprint"] == "b" * 64
    a, b = first.source_records["external"], second.source_records["external"]
    assert a.revision == b.revision == "b" * 64
    assert a.target_revision != b.target_revision
    registry = second.proposal_pool.ranking_registry
    assert registry.model_key(first.session) != registry.model_key(second.session)
    assert registry.source_key(a) != registry.source_key(b)
    batch = selected.create_external_batch(prefill_step_size=64)
    try:
        uid = batch.insert([[1, 2, 3]], max_tokens=[4])[0]
        tokens, finishes = drain(batch)
        assert tokens[uid] == reference(selected.model, [1, 2, 3], 4)
        finish = finishes[uid]
        finish.cache_sidecar.validate(
            selected.identity["fingerprint"], len(finish.all_tokens)
        )
        assert (
            finish.speculative_receipt["continuation_pool"]["ranking"][
                "ranking_registry"
            ]["target_revision"]
            == second.session.target_revision
        )
    finally:
        batch.close()


def test_contract_is_frozen_and_mapping_order_has_canonical_namespace():
    source = {"algorithm": "candidate-v1", "scope": {"verify": False, "decode": True}}
    adapter = SimpleNamespace(execution_numerics_contract=lambda: source)
    snapshot = adapter_numerical_contract(adapter)
    first = apc_semantic_namespace("semantic", adapter_execution_numerics=snapshot)
    source["scope"]["decode"] = False
    assert snapshot["scope"]["decode"] is True
    assert first == apc_semantic_namespace(
        "semantic",
        adapter_execution_numerics={
            "scope": {"decode": True, "verify": False},
            "algorithm": "candidate-v1",
        },
    )


def test_direct_teacher_feedback_receives_effective_target_and_raw_draft_pins(
    monkeypatch,
):
    adapter = direct_adapter(True)
    adapter.external_policy["lilicorr_feedback"] = {"directory": "unused"}
    captured = []

    def manager(drafter, policy, **pins):
        captured.append(pins)
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(
        "mlx2.runtime.lilicorr_feedback.LiLiCorrFeedbackManager", manager
    )
    adapter._initialize_external_feedback()
    assert captured == [
        {
            "target_revision": adapter.draft_model.session.target_revision,
            "draft_revision": "b" * 64,
            "binding": adapter.identity["fingerprint"],
        }
    ]


@pytest.mark.parametrize("bad", [{}, [], True, {"cost": float("nan")}])
def test_invalid_adapter_law_refuses_before_binding_or_publication(bad):
    adapter = bind()
    identity, old = adapter.identity.copy(), adapter.draft_model
    adapter.execution_numerics_contract = lambda: bad
    with pytest.raises(ValueError):
        adapter_numerical_contract(adapter)
    assert adapter.identity == identity and adapter.draft_model is old


def test_noncallable_contract_refuses():
    with pytest.raises(ValueError, match="callable"):
        adapter_numerical_contract(
            SimpleNamespace(execution_numerics_contract={"law": "x"})
        )


@pytest.mark.parametrize("policy_enabled, live_enabled", [(False, True), (True, False)])
def test_live_model_policy_divergence_refuses_before_publication(
    policy_enabled, live_enabled
):
    layer = GatedDeltaNet(
        TextModelArgs(
            hidden_size=16,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
        )
    )
    layer.set_fused_gdn_enabled(live_enabled)
    adapter = SimpleNamespace(
        fused_gdn=policy_enabled,
        model=SimpleNamespace(named_modules=lambda: [("layer", layer)]),
    )
    with pytest.raises(ValueError, match="live target layers"):
        Qwen3827BAdapter.execution_numerics_contract(adapter)
