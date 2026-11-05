"""CPU contracts for opt-in SC-MXFP4 serving, identity and cache publication.

No model checkpoint runs here; synthetic host-contract fixtures exercise the
working source tree. Real-model/GPU evidence belongs to owned probe receipts.
"""
from types import SimpleNamespace as NS

import pytest

from mlx2.runtime.microscaling import installer as sc
from mlx2.serving import (
    ServingEngine, apc_semantic_namespace, bind_microscaling_for_serving,
    microscaling_decode_multiplier, microscaling_prefill_cache_allowed,
    microscaling_prefill_status, validate_microscaling_composition,
    weight_streaming_refusals,
)


def test_cli_default_off_and_engine_mapping():
    from mlx2.server import build_parser, serving_engine_kwargs
    args = build_parser().parse_args(["--model", "fixture"])
    assert args.microscaling_prefill == "off"
    kwargs = serving_engine_kwargs(args, None, native_mtp=False,
        approximate_kv=None, max_request_bytes=64 << 20)
    assert kwargs["microscaling_prefill"] == "off"
    args = build_parser().parse_args(["--model", "fixture",
        "--microscaling-prefill", "experts"])
    kwargs = serving_engine_kwargs(args, None, native_mtp=False,
        approximate_kv=None, max_request_bytes=64 << 20)
    assert kwargs["microscaling_prefill"] == "experts"


def test_unqualified_route_can_load_with_cache_publication_disabled():
    engine = ServingEngine("unused", _validate_only=True, mtp=False,
        int8_prefill="off", microscaling_prefill="mlp")
    assert engine.qualification_state == "unqualified"
    assert engine.microscaling_prefill_policy.enabled
    assert engine.apc_reuse_disabled_reason == "unqualified_microscaling_prefill"
    assert engine.microscaling_prefill_handle is None


def test_explicit_sc_policy_steps_auto_int8_aside_with_reason():
    engine = ServingEngine("unused", _validate_only=True, mtp=False,
        int8_prefill="auto", microscaling_prefill="experts")
    assert not engine.int8_prefill_policy.enabled
    assert engine.int8_prefill_auto["resolved"] == "off"
    assert "microscaling" in engine.int8_prefill_auto["reason"]


@pytest.mark.parametrize("mode,record,allowed", [(False, None, False),
    (True, None, True), (False, "matched-later.json", True)])
def test_cache_permission_is_separate_from_route_load(mode, record, allowed):
    assert microscaling_prefill_cache_allowed(
        qualification_mode=mode, qualification=record) is allowed


@pytest.mark.parametrize("extra", [
    {"int8_prefill": "mlp", "qualification_mode": True},
    {"lane_matmul": "exact"},
    {"execution_policy": {"tensorfold_prefill": True}},
    {"execution_policy": {"invariant_prefill": True}},
])
def test_constructor_rejects_competing_projection_owners(extra):
    with pytest.raises(ValueError, match="microscaling"):
        ServingEngine("unused", _validate_only=True, mtp=False,
            microscaling_prefill="mlp", **extra)


@pytest.mark.parametrize("options", [{"sp_qmm": True}, {"qmm_row_pad": True},
    {"verify_bitexact": True}, {"approximate_kv": True}, {"spomin": True},
    {"lora_root": "adapters"}, {"execution_policy": {"varlen_sparse_moe": True}}])
def test_conflicts_refuse_before_install(options):
    with pytest.raises(ValueError, match="microscaling"):
        validate_microscaling_composition(sc.MicroscalingPolicy.from_value("mlp"), **options)
    validate_microscaling_composition(sc.MicroscalingPolicy(), **options)


def test_expanded_decode_bound_uses_adapter_or_nested_topk():
    assert microscaling_decode_multiplier(NS(config={"text_config":
        {"num_experts_per_tok": 10}})) == 10
    assert microscaling_decode_multiplier(NS(config={})) == 1
    assert microscaling_decode_multiplier(NS(config={},
        microscaling_decode_row_multiplier=lambda: 8)) == 8
    with pytest.raises(ValueError, match="positive integer"):
        microscaling_decode_multiplier(NS(config={"num_experts_per_token": True}))


def _handle(artifact="artifact-a", path="model.layers.0.mlp.up_proj"):
    handle = sc.MicroscalingHandle("mlp", artifact_fingerprint=artifact)
    handle.active = True
    handle._bound = {1: NS(spec=NS(path=path, kind="dense", experts=1, n=128, k=256),
        metadata_fingerprint="metadata-sha", metadata_bytes=256)}
    return handle


def test_apc_identity_binds_active_handle_artifact_mask_and_is_identity_off():
    baseline = apc_semantic_namespace("semantic")
    assert baseline == apc_semantic_namespace("semantic", microscaling_prefill_handle=None)
    a = apc_semantic_namespace("semantic", microscaling_prefill_handle=_handle())
    assert a != baseline
    assert a != apc_semantic_namespace("semantic", microscaling_prefill_handle=_handle("other"))
    assert a != apc_semantic_namespace("semantic", microscaling_prefill_handle=_handle(path="other_projection"))
    inactive = _handle(); inactive.active = False
    with pytest.raises(sc.MicroscalingError, match="active"):
        apc_semantic_namespace("semantic", microscaling_prefill_handle=inactive)


def test_status_and_response_receipts_report_engagement_and_cache_restriction():
    handle = _handle()
    handle.counts["engaged_calls"] = 3
    handle.counts["dense_calls"] = 3
    handle.counts["engaged_rows"] = 1536
    engine = NS(microscaling_prefill_handle=handle, microscaling_prefill_policy=handle.policy,
        qualification_state="unqualified", apc_reuse_disabled_reason="unqualified_microscaling_prefill")
    status = microscaling_prefill_status(engine)
    receipt = microscaling_prefill_status(engine, receipt=True)
    for value in (status, receipt):
        assert value["engaged"] and value["counts"]["engaged_calls"] == 3
        assert value["qualified"] is False
        assert value["qualification"] == "unqualified"
        assert value["apcv2_reuse_disabled_reason"] == "unqualified_microscaling_prefill"
    assert receipt["revision"] == handle.revision


def test_workspace_admission_charges_each_active_provider_once():
    engine = object.__new__(ServingEngine)
    engine.int8_prefill_handle = NS(unfilled_reserve_bytes=lambda: 100)
    engine.microscaling_prefill_handle = NS(unfilled_reserve_bytes=lambda: 200)
    assert engine.projection_workspace_reserve_gib() == 300 / (1 << 30)
    engine.microscaling_prefill_handle = NS(unfilled_reserve_bytes=lambda: -1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        engine.projection_workspace_reserve_gib()


def test_binding_uses_expanded_bound_and_static_qualification_settings(monkeypatch):
    handle = _handle()
    captured = {}
    def apply(adapter, policy, *, max_decode_rows):
        captured["rows"] = max_decode_rows
        return handle
    monkeypatch.setattr(sc, "apply_for_adapter", apply)
    adapter = NS(config={"num_experts_per_tok": 10})
    actual, settings = bind_microscaling_for_serving(adapter, "mlp", max_lanes=4,
        config={}, speculation="ordinary", memory_headroom=lambda: 1 << 30)
    assert actual is handle and captured["rows"] == 80
    assert settings["revision"] == handle.revision
    assert settings["max_decode_rows"] == 8 and settings["max_decode_assignment_rows"] == 80
    assert settings["projection_paths"] == ["model.layers.0.mlp.up_proj"]
    assert settings["workspace_max_prefill_rows"] == 512 * 4
    assert settings["workspace_reserve_bytes"] > 128 * 256
    assert "counts" not in settings and "qualified" not in settings


def test_binding_rolls_install_back_on_insufficient_headroom(monkeypatch):
    handle = _handle(); removed = []
    monkeypatch.setattr(sc, "apply_for_adapter", lambda *a, **kw: handle)
    monkeypatch.setattr(sc, "remove", lambda h: removed.append(h))
    with pytest.raises(sc.MicroscalingError, match="workspace"):
        bind_microscaling_for_serving(NS(config={}), "mlp", max_lanes=1,
            config={}, speculation="ordinary", memory_headroom=lambda: 0)
    assert removed == [handle]


def test_weight_streaming_refuses_microscaling():
    reasons = weight_streaming_refusals(lane_matmul="off", sp_qmm=False,
        int8_prefill=False, verify_bitexact=False, multi_lora=False,
        lora_root=None, execution_policy=None, microscaling_prefill=True)
    assert reasons == ["microscaling_prefill"]
