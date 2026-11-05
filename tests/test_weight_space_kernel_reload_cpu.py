"""Model-free compatibility contracts for the merged opt-in execution modes.

Checkpoint: synthetic fixture@fixture-artifact. No checkpoint inference or GPU.
"""

import json
import threading
from types import SimpleNamespace as NS

import pytest
from test_kernel_reload_lifecycle_cpu import (
    FakeReloader,
    engine_fixture,
    finish_reload,
    start_reload,
)
from test_kernel_reload_lifecycle_cpu import (
    release_fixture_owners as isolated_owners,
)
from test_prometheus import FakeEngine, assert_valid_prometheus_text
from test_tenant_auth import TenantEngine, _call
from test_tenant_auth import served as served_fixture
from test_weight_space_serving_cpu import artifact
from test_weight_space_serving_cpu import engine as compiled_engine_fixture

from mlx2.experimental.weight_space.live import LiveWeightSpaceService
from mlx2.experimental.weight_space.self_repair import RepairPolicy, RepairSession
from mlx2.kernel_reload import KernelReloader, KernelReloadError
from mlx2.kernel_reload_serving import KernelWeightSpaceConflict
from mlx2.prometheus import render_engine_metrics
from mlx2.runtime.weight_space_metrics import WeightSpaceMetrics
from mlx2.serving import AdmissionClosed

release_fixture_owners = isolated_owners
compiled_engine = compiled_engine_fixture
served = served_fixture


def bind_feature(engine, feature):
    if feature == "global_weight_program":
        engine.lora_session = NS(weight_program={"program_id": "fixture"})
    elif feature == "registered_weight_program":
        engine.multi_lora = NS(registry={"candidate": NS(weight_program={"program_id": "fixture"})})
    elif feature == "weight_program_default":
        engine._weight_program_deployment = {"default_adapter": "candidate"}
    else:
        setattr(engine, "weight_space_live" if feature == "live_weight_space" else feature, object())


@pytest.mark.parametrize("feature", [
    "live_weight_space", "global_weight_program", "registered_weight_program",
    "weight_program_default", "weight_overlays", "layer_excision", "expert_masks",
])
def test_reload_refuses_bound_feature_before_prepare_or_cache_mutation(feature):
    engine = engine_fixture()
    bind_feature(engine, feature)
    model, weight = engine.adapter.model, engine.adapter.model.weight
    with pytest.raises(KernelWeightSpaceConflict) as caught:
        engine.reload_kernels(modules=["fixture.kernel"])
    assert caught.value.receipt["conflicts"] == [feature]
    assert caught.value.receipt["model_changed"] is False
    assert engine.kernel_reload_status()["weight_space_conflicts"] == [feature]
    assert engine._kernel_reloader.prepared == engine._kernel_reloader.applied == []
    assert engine._kernel_reload_request is None and engine._kernel_reload_epoch == 0
    assert engine._kernel_reload_phase == "idle" and engine._service_state == "serving"
    assert engine.apc.entries == 3 and engine.apc.calls == []
    assert engine.model_revision == 4 and engine.adapter.model is model
    assert model.weight is weight
    assert engine.counts["kernel_reload_rejected_weight_space_conflict"] == 1
    assert engine.operation_receipts[-1] == caught.value.receipt


def test_reload_keeps_existing_legacy_lora_behavior():
    engine = engine_fixture()
    engine.lora_session = NS(weight_program=None)
    engine.multi_lora = NS(registry={"legacy": NS(weight_program=None)})
    caller, result = start_reload(engine, modules=["fixture.kernel"])
    finish_reload(engine, caller, result)
    assert "error" not in result and engine._kernel_reload_epoch == 1


@pytest.mark.parametrize("state", ["epoch", "identity", "pending", "preparing", "purging", "failed"])
@pytest.mark.parametrize("operation", [
    "_validate_weight_program_route", "configure_weight_overlays",
    "configure_layer_excision", "configure_expert_masks", "live_create", "live_cached",
])
def test_weight_space_refuses_changed_pending_or_failed_execution(state, operation):
    engine = engine_fixture()
    if operation == "live_cached":
        service = LiveWeightSpaceService(engine, {"policy": RepairPolicy()})
        service._service = object()
    if state == "epoch":
        engine._kernel_reload_epoch = 1
    elif state == "identity":
        engine._kernel_reload_identity = "changed"
    elif state == "pending":
        engine._kernel_reload_request = {}
    elif state == "failed":
        engine._service_state = "kernel_reload_failed"
    else:
        engine._kernel_reload_phase = state
    with pytest.raises(KernelWeightSpaceConflict):
        if operation == "live_create":
            LiveWeightSpaceService(engine, {"policy": RepairPolicy()})
        elif operation == "live_cached":
            service._bound()
        else:
            getattr(engine, operation)()
    assert engine.model_revision == 4 and engine.apc.entries == 3
    assert engine._kernel_reloader.applied == []
    for name in ("weight_overlays", "layer_excision", "expert_masks"):
        assert getattr(engine, name, None) is None


def test_prepare_callback_binding_is_rechecked_before_drain():
    engine = engine_fixture()
    original = engine._kernel_reloader.prepare

    def prepare(**kwargs):
        plan = original(**kwargs)
        engine.weight_overlays = object()  # Simulate a trusted callback changing ownership.
        return plan

    engine._kernel_reloader.prepare = prepare
    with pytest.raises(KernelWeightSpaceConflict):
        engine.reload_kernels(modules=["fixture.kernel"])
    assert len(engine._kernel_reloader.prepared) == 1
    assert engine._kernel_reload_request is None and engine._service_state == "serving"
    assert engine._kernel_reloader.applied == [] and engine.apc.calls == []


def test_preparation_serializes_compiled_registration_until_admission_closes():
    entered, proceed, attempted, finished = (threading.Event() for _ in range(4))

    class BlockingReloader(FakeReloader):
        def prepare(self, **kwargs):
            entered.set()
            assert proceed.wait(2)
            return super().prepare(**kwargs)

    engine = engine_fixture(reloader=BlockingReloader())
    results = {}

    def reload():
        try:
            results["reload"] = engine.reload_kernels(modules=["fixture.kernel"])
        except BaseException as error:  # noqa: BLE001 - capture thread evidence
            results["reload_error"] = error

    def register():
        attempted.set()
        try:
            with engine._lora_control_section():
                engine._check_weight_space_kernel_compatibility("weight_program")
                results["mutated"] = True
        except (AdmissionClosed, KernelWeightSpaceConflict) as error:
            results["register_error"] = error
        finally:
            finished.set()

    caller = threading.Thread(target=reload, daemon=True)
    registration = threading.Thread(target=register, daemon=True)
    caller.start()
    assert entered.wait(2)
    registration.start()
    assert attempted.wait(2) and not finished.wait(0.02)
    proceed.set()
    assert finished.wait(2)
    assert "mutated" not in results and "register_error" in results
    finish_reload(engine, caller, results)
    registration.join(2)
    assert "reload_error" not in results and engine._kernel_reload_epoch == 1


@pytest.mark.parametrize("attribute", [
    "weight_program", "weight_overlay_view", "layer_excision_view", "expert_mask_view",
])
@pytest.mark.parametrize("base_receipt", ["candidate_validation", {"route": "ordinary", "qualified": True}])
def test_terminal_receipt_composes_feature_then_execution_metadata(attribute, base_receipt):
    engine = engine_fixture()
    engine._kernel_reload_identity = "changed"
    job = NS(**{attribute: {"selected": True}})
    receipt = engine._kernel_route_receipt(engine._weight_space_route_receipt(job, base_receipt))
    assert receipt["qualification"] == "unqualified" and receipt["qualified"] is False
    assert receipt["weight_space_operation"] == attribute.removesuffix("_view")
    assert receipt["kernel_reload"]["execution_identity"] == "changed"
    assert receipt["route"] in {"candidate_validation", "ordinary"}
    assert engine._kernel_route_receipt("candidate_validation")["route"] == "candidate_validation"


def test_both_metrics_and_explicit_conflict_reason_survive_merge():
    engine = FakeEngine()
    engine.weight_space_metrics = WeightSpaceMetrics()
    engine.weight_space_metrics.record("runtime_compatibility", "rejected")
    engine.counts["kernel_reload_rejected_weight_space_conflict"] = 1
    output = render_engine_metrics(engine)
    assert_valid_prometheus_text(output)
    assert 'mlx2_kernel_reload_rejections_total{reason="weight_space_conflict"} 1' in output
    assert 'mlx2_weight_space_operations_total{operation="runtime_compatibility",outcome="rejected"} 1' in output
    assert "mlx2_kernel_reload_duration_seconds_sum" in output
    assert "mlx2_weight_space_operation_duration_seconds_bucket" in output


@pytest.mark.parametrize("module", [
    "mlx2.experimental.weight_space.live", "mlx2.experimental.weight_space.self_repair",
    "mlx2.runtime.weight_space_metrics", "mlx2.runtime.weight_program_observer",
    "mlx2.runtime.weight_overlays", "mlx2.runtime.layer_excision", "mlx2.runtime.expert_masks",
])
def test_host_weight_space_modules_cannot_be_loaded_as_tensor_kernels(tmp_path, module):
    with pytest.raises(KernelReloadError, match="Not a reloadable tensor module"):
        KernelReloader(tmp_path).prepare(modules=[module])


def test_actual_compiled_registration_after_reload_does_not_wrap_model(compiled_engine, tmp_path):
    path = tmp_path / "candidate"
    engine = compiled_engine
    key = "model.layers.3.self_attn.q_proj"
    module = dict(engine.adapter.model.named_modules())[key]
    artifact(path, key=key, dims=(module.weight.shape[1], module.weight.shape[0]))
    engine._kernel_reload_identity = "changed-runtime"
    original = engine.adapter.model
    registry = dict(engine.multi_lora.registry)
    with pytest.raises(KernelWeightSpaceConflict):
        engine.load_lora_adapter("candidate", path)
    assert engine.adapter.model is original and engine.multi_lora.registry == registry
    assert engine.weight_program_deployment_state()["default_adapter"] is None


def test_live_service_is_published_initialized_and_cannot_be_replaced():
    engine = engine_fixture()
    service = LiveWeightSpaceService(engine, {"policy": RepairPolicy()})
    assert engine.weight_space_live is service
    assert service.engine is engine and service._lock is not None and service._trials == {}
    with pytest.raises(ValueError, match="already configured"):
        LiveWeightSpaceService(engine, {"policy": RepairPolicy()})
    assert engine.weight_space_live is service


def test_merged_admin_endpoints_return_direct_incompatibility_receipts(served):
    error = KernelWeightSpaceConflict("kernel_reload", ["live_weight_space"])

    def refuse(*args, **kwargs):
        raise error

    engine = TenantEngine()
    engine.kernel_reload_status = lambda: {"enabled": True, "weight_space_conflicts": ["live_weight_space"]}
    engine.reload_kernels = refuse
    service = NS(capabilities=lambda: {"implemented": True}, call=refuse)
    host = served(None, engine, admin_token="private-admin", weight_space_service=service)
    headers = {"Authorization": "Bearer private-admin"}
    for path in ("/v1/admin/kernels", "/v1/admin/weight-space/capabilities"):
        assert _call(host.base, "GET", path, headers=headers)[0] == 200
        assert _call(host.base, "GET", path)[0] == 401
    for path, body in (("/v1/admin/kernels/reload", {"modules": ["fixture.kernel"]}),
                       ("/v1/admin/weight-space/call", {"tool": "repair_capabilities", "arguments": {}})):
        status, _, raw = _call(host.base, "POST", path, body, headers)
        payload = json.loads(raw)
        assert status == 409 and payload["error"]["code"] == error.code
        assert payload["mlx2"] == error.receipt


def test_live_publication_does_not_wait_for_draining_lifecycle_owner():
    engine = engine_fixture()
    request = {"messages": [{"role": "user", "content": "active request"}], "session_id": "session"}
    session = RepairSession("session", request)
    service = LiveWeightSpaceService(engine, {
        "policy": RepairPolicy(), "raw": {}, "tenant_id": "tenant", "session": session,
        "validators": {},
    })
    finished = threading.Event()
    engine._admissions_preparing = 1

    def publish():
        with engine.submission_lock:
            service.observe_submission(NS(tenant_id="tenant", request=request, lora_name=None))
        finished.set()

    publication = threading.Thread(target=publish, daemon=True)
    try:
        with engine._weight_space_lifecycle_lock():
            publication.start()
            assert finished.wait(2), "publication waited for the lifecycle owner draining it"
    finally:
        publication.join(2)
    assert service._service is None and service.live_session_observed
    assert session.revision == 1 and session.transcript_origin == "live_submission"


def test_cached_live_binding_does_not_wait_for_trial_promotion_lifecycle():
    engine = engine_fixture()
    service = LiveWeightSpaceService(engine, {"policy": RepairPolicy()})
    bound = service._service = object()
    finished = threading.Event()
    result = {}

    def read_bound():
        result["service"] = service._bound()
        finished.set()

    caller = threading.Thread(target=read_bound, daemon=True)
    try:
        with engine._weight_space_lifecycle_lock():
            caller.start()
            assert finished.wait(2), "cached tool dispatch acquired a trial's lifecycle lock"
    finally:
        caller.join(2)
    assert result["service"] is bound
