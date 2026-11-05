"""No-model live service configuration, admission capture and actual HTTP auth."""

import copy
import json
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from test_tenant_auth import TenantEngine, _call
from test_tenant_auth import served as served_fixture

from mlx2.experimental.weight_space.live import (
    LIVE_SCHEMA,
    LiveWeightSpaceService,
    load_live_config,
    tool_schema_digest,
)
from mlx2.experimental.weight_space.self_repair import RepairPolicy
from mlx2.experimental.weight_space.tool_service import CONFIG_SCHEMA
from mlx2.server import build_parser, handler_for

served = served_fixture


def write_json(path, value):
    path.write_text(json.dumps(value))
    os.chmod(path, 0o600)


@pytest.fixture
def live(tmp_path):
    # load_live_config needs the optional weight-space-research extra.
    pytest.importorskip("jsonschema")
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    np.savez(root / "capture.npz", weight=np.ones((3, 4), np.float32),
             direction=np.array([1, 0, 0], np.float32))
    os.chmod(root / "capture.npz", 0o600)
    checkpoint, source = "fixture@fixture-fingerprint", "fixture-source"
    assets = {"schema": CONFIG_SCHEMA, "session_id": "conversation", "programs": {}, "captures": [{
        "capture_id": "linear-capture", "checkpoint": checkpoint, "source_revision": source,
        "data_scope": "shared", "arrays": "capture.npz",
        "targets": {"linear": {"weight": "weight", "direction": "direction", "columns": [0, 1]}},
    }]}
    write_json(root / "assets.json", assets)
    schema = {"type": "object", "properties": {"x": {"type": "integer", "minimum": 0}},
              "required": ["x"], "additionalProperties": False}
    request = {"messages": [{"role": "user", "content": "call the tool"}],
               "session_id": "conversation", "tools": [{"type": "function", "function": {
                   "name": "positive", "parameters": schema}}]}
    case = {"case_id": "repair", "messages": request["messages"],
            "expected_tool": "positive", "expected_arguments": {"x": 1}}
    config = {"schema": LIVE_SCHEMA, "checkpoint": checkpoint, "source_revision": source,
              "assets_config": "assets.json", "attempts_dir": "attempts",
              "session": {"session_id": "conversation", "tenant_id": "tenant-a", "request": request},
              "suites": [{"tool_name": "positive", "tool_schema_digest": tool_schema_digest("positive", schema),
                           "argument_schema": schema, "repair": [case],
                           "regressions": [{**case, "case_id": "preserve", "expected_arguments": {"x": 2}}]}]}
    write_json(root / "live.json", config)
    loaded = load_live_config(root / "live.json")
    engine = TenantEngine()
    engine.ready = threading.Event()
    engine.ready.set()
    engine.error = None
    engine.model_path = "fixture"
    engine.adapter = SimpleNamespace(identity={"fingerprint": "fixture-fingerprint"})
    engine.multi_lora = SimpleNamespace(delta_precision="fp32")
    engine.lora_root = root
    engine.snapshot = {"runtime": {"source_sha256": "runtime-source"}}
    return root, config, engine, LiveWeightSpaceService(engine, loaded)


def test_live_config_and_server_flag_are_off_by_default(live):
    _, _, engine, service = live
    assert build_parser().parse_args(["--model", "fixture"]).weight_space_config is None
    capabilities = service.capabilities()
    assert capabilities["runtime_trials_enabled"] is False
    assert capabilities["automatic_promotion"] is False
    assert capabilities["session_source"] == "operator_seed"
    result = service.call({"tool": "compile_analytical_repair", "arguments": {"capture_id": "linear-capture"}})
    assert result["compiled"] is True and result["deployed"] is False
    assert not engine.submitted
    with pytest.raises(ValueError, match="admin-token-file"):
        handler_for(engine, weight_space_service=service)


def test_live_overlay_mode_binds_real_cpu_manager_without_lora_or_repair(live, served):
    import mlx.core as mx
    from test_weight_overlays_runtime_cpu import Tiny

    from mlx2.experimental.weight_space.overlays import TensorTarget
    from mlx2.runtime.weight_overlays import WeightOverlayBinding, WeightOverlayManager

    root, raw, engine, _ = live
    # This fake host is reconfigured before any service bind or submission.
    engine.weight_space_live = None
    raw["experiments"] = {"mode": "quantization_overlay", "budget_bytes": 8192,
                          "max_materialized_bytes": 8192}
    raw["suites"] = []
    write_json(root / "live.json", raw)
    model = Tiny()
    mx.eval(model.parameters())
    target = TensorTarget(0, None, "linear")
    engine.multi_lora, engine.lora_root = None, None

    def configure(**kwargs):
        engine.weight_overlays = WeightOverlayManager(model,
            {target: WeightOverlayBinding(model, "linear")}, checkpoint=raw["checkpoint"], **kwargs)
        return engine.weight_overlays

    engine.configure_weight_overlays = configure
    service = LiveWeightSpaceService(engine, load_live_config(root / "live.json"))
    capabilities = service.capabilities()
    assert capabilities["experiment_mode"] == "quantization_overlay"
    assert capabilities["experiments"]["quantization_overlays_registered"] is True
    assert capabilities["runtime_trials_enabled"] is False
    assert service._bound()._controller is None
    app = served(None, engine=engine, admin_token="operator-token", weight_space_service=service)
    auth = {"Authorization": "Bearer operator-token"}
    status, _, raw_result = _call(app.base, "POST", "/v1/admin/weight-space/call", {
        "tool": "create_quantization_overlay", "arguments": {
            "name": "q3", "target_id": "L0.dense.linear", "bits": 3,
            "format": "affine_uniform", "group_size": 17}}, auth)
    assert status == 200, raw_result
    result = json.loads(raw_result)
    assert result["request_selection"] == {"weight_overlay": "q3"}
    assert result["default_changed"] is False
    assert not engine.submitted
    assert engine.weight_overlays.bank.snapshot()["active_requests"] == 0
    raw["policy"] = {"enabled": True}
    write_json(root / "live.json", raw)
    with pytest.raises(ValueError, match="cannot compose"):
        load_live_config(root / "live.json")


def test_live_excision_mode_registers_loaded_geometry(live):
    from test_layer_excision_runtime_cpu import model_adapter

    from mlx2.runtime.layer_excision import LayerExcisionManager

    root, raw, engine, _ = live
    # This fake host is reconfigured before any service bind or submission.
    engine.weight_space_live = None
    raw["experiments"], raw["suites"] = {"mode": "layer_excision"}, []
    write_json(root / "live.json", raw)
    model, adapter = model_adapter()
    engine.multi_lora, engine.lora_root = None, None
    engine.configure_layer_excision = lambda **kwargs: LayerExcisionManager(
        model, adapter.layer_excision_binding(), checkpoint=raw["checkpoint"], **kwargs)
    service = LiveWeightSpaceService(engine, load_live_config(root / "live.json"))
    assert service.capabilities()["experiments"]["excision_inventory_ids"] == ["loaded_model"]
    receipt = service.call({"tool": "compile_layer_excision", "arguments": {
        "inventory_id": "loaded_model", "name": "remove_one", "removed_layers": [1]}})
    assert receipt["runtime_registered"] is True
    assert receipt["request_selection"] == {"layer_excision": "remove_one"}
    assert not engine.submitted


def test_live_analysis_mode_never_requires_runtime_experiment_provider(live):
    root, raw, engine, _ = live
    # This fake host is reconfigured before any service bind or submission.
    engine.weight_space_live = None
    raw["experiments"], raw["suites"] = {"mode": "analysis"}, []
    write_json(root / "live.json", raw)
    engine.multi_lora, engine.lora_root = None, None
    service = LiveWeightSpaceService(engine, load_live_config(root / "live.json"))
    assert service.capabilities()["experiment_mode"] == "analysis"
    with pytest.raises(RuntimeError, match="provider"):
        service.call({"tool": "inspect_quantization_targets", "arguments": {}})


def test_live_expert_mask_mode_registers_only_host_bound_native_plan(live):
    from test_expert_masks_runtime_cpu import SyntheticSource, binding, plan

    from mlx2.runtime.expert_masks import ExpertMaskManager

    root, raw, engine, _ = live
    # This fake host is reconfigured before any service bind or submission.
    engine.weight_space_live = None
    raw["experiments"], raw["suites"] = {"mode": "expert_mask"}, []
    write_json(root / "live.json", raw)
    model = SyntheticSource()
    engine.multi_lora, engine.lora_root = None, None
    manager = ExpertMaskManager(model, binding(model), checkpoint=raw["checkpoint"])
    engine.configure_expert_masks = lambda: manager
    service = LiveWeightSpaceService(engine, load_live_config(root / "live.json"))
    description = service.call({"tool": "inspect_expert_mask_runtime", "arguments": {}})
    assert description["native_expert_store_unchanged"] is True
    service._bound().register_expert_source("calibrated", plan=plan(manager, tuple(range(4))),
        importance={0: [0.1, 0.9, 0.2, 0.8]})
    result = service.call({"tool": "compile_expert_mask", "arguments": {
        "source_id": "calibrated", "keep_counts": {"0": 2}}})
    name = result["request_selection"]["expert_mask"]
    assert name in manager.registry
    assert manager.registry[name].layers[0].enabled == (1, 3)
    assert not engine.submitted


def test_live_http_reuses_admin_auth_and_rejects_caller_verdicts(live, served):
    _, _, engine, service = live
    app = served(None, engine=engine, admin_token="operator-token", weight_space_service=service)
    path = "/v1/admin/weight-space/capabilities"
    assert _call(app.base, "GET", path)[0] == 401
    assert _call(app.base, "GET", path, headers={"Authorization": "Bearer wrong"})[0] == 401
    auth = {"Authorization": "Bearer operator-token"}
    status, _, raw = _call(app.base, "GET", path, headers=auth)
    assert status == 200 and json.loads(raw)["oracle_kind"] == "local_json_schema"
    call = "/v1/admin/weight-space/call"
    status, _, raw = _call(app.base, "POST", call,
        {"tool": "compile_analytical_repair", "arguments": {"capture_id": "linear-capture"}}, auth)
    assert status == 200 and json.loads(raw)["compiled"] is True
    for body in (
        {"tool": "promote_weight_program", "arguments": {}},
        {"tool": "trial_candidate", "arguments": {"attempt_id": "candidate", "passed": True}},
        {"tool": "compile_analytical_repair", "arguments": {"capture_id": "linear-capture", "path": "/tmp/x"}},
    ):
        assert _call(app.base, "POST", call, body, auth)[0] == 400
    assert not engine.submitted


def test_live_capture_tracks_only_bound_session_and_skips_trials(live):
    _, config, _, service = live
    session = service.config["session"]
    request = copy.deepcopy(config["session"]["request"])
    request["messages"].append({"role": "user", "content": "new turn"})
    def observed(tenant="tenant-a", **extra):
        service.observe_submission(SimpleNamespace(tenant_id=tenant, request={**request, **extra}, lora_name="candidate"))
    observed("other")
    observed(_mlx2_weight_program_replay="trusted")
    assert session.revision == 0
    observed()
    assert session.revision == 1 and session.active_adapter == "candidate"
    assert session.transcript_origin == "live_submission"
    assert session.request["messages"][-1]["content"] == "new turn"
    assert service.capabilities()["live_session_observed"] is True


def test_validated_session_override_binds_ordinary_requests_and_exact_content(live):
    from mlx2.experimental.weight_space.self_repair import _digest, _request_context

    _, config, _, service = live
    session = service.config["session"]
    request = copy.deepcopy(config["session"]["request"])
    session.routing_override = {
        "name": "validated-candidate", "fingerprint": "content-fingerprint",
        "previous_model": None, "messages": copy.deepcopy(request["messages"]),
        "allowed_tools": ["positive"],
        "request_context_digest": _digest(_request_context(request)), "expires_at": time.time() + 60,
    }
    original = copy.deepcopy(session.routing_override)
    foreign = copy.deepcopy(request)
    assert service.resolve_session_model(foreign, tenant_id="other") is None
    assert "_mlx2_weight_session_expected_fingerprint" not in foreign
    assert session.routing_override == original
    assert service.resolve_session_model(request, tenant_id="tenant-a") == "validated-candidate"
    assert request["_mlx2_weight_session_expected_fingerprint"] == "content-fingerprint"
    request["messages"].extend([
        {"role": "assistant", "content": None, "tool_calls": [{"id": "task-call", "type": "function",
         "function": {"name": "positive", "arguments": '{"x":1}'}}]},
        {"role": "tool", "tool_call_id": "task-call", "content": "success"},
    ])
    assert service.resolve_session_model(request, tenant_id="tenant-a") == "validated-candidate"
    request["messages"].append({"role": "user", "content": "new task"})
    assert service.resolve_session_model(request, tenant_id="tenant-a") is None
    assert session.routing_override is None


@pytest.mark.parametrize("change", ["schema", "adapter", "expiry"])
def test_session_override_ends_for_changed_schema_explicit_adapter_or_expiry(live, change):
    from mlx2.experimental.weight_space.self_repair import _digest, _request_context

    _, config, _, service = live
    session = service.config["session"]
    request = copy.deepcopy(config["session"]["request"])
    session.routing_override = {
        "name": "candidate", "fingerprint": "fp", "previous_model": None,
        "messages": copy.deepcopy(request["messages"]),
        "request_context_digest": _digest(_request_context(request)), "expires_at": time.time() + 60,
    }
    if change == "schema":
        request["tools"][0]["function"]["parameters"]["required"] = []
    elif change == "adapter":
        request["model"] = "explicit-other-adapter"
    else:
        session.routing_override["expires_at"] = time.time() - 1
    assert service.resolve_session_model(request, tenant_id="tenant-a") == request.get("model")
    assert session.routing_override is None


def test_live_config_accepts_only_host_bound_exact_repair_aliases(live):
    root, raw, engine, _ = live
    # This fake host is reconfigured before any service bind or submission.
    engine.weight_space_live = None
    raw["repair_tool_aliases"] = {"hermes__trial_candidate": "trial_candidate"}
    write_json(root / "live.json", raw)
    config = load_live_config(root / "live.json")
    service = LiveWeightSpaceService(engine, config)
    assert service.capabilities()["repair_tool_aliases"] == raw["repair_tool_aliases"]
    for aliases in ({"hermes__trial_candidate": "arbitrary_python"},
                    {"trial_candidate": "rollback_deployment"}, {"bad name": "trial_candidate"}, []):
        raw["repair_tool_aliases"] = aliases
        write_json(root / "live.json", raw)
        with pytest.raises(ValueError, match="repair_tool_aliases"):
            load_live_config(root / "live.json")


def test_live_tool_schema_drift_does_not_reuse_the_seed(live):
    _, config, _, service = live
    request = copy.deepcopy(config["session"]["request"])
    request["tools"][0]["function"]["parameters"]["required"] = []
    service.observe_submission(SimpleNamespace(tenant_id="tenant-a", request=request, lora_name=None))
    with pytest.raises(ValueError, match="tool schema differs"):
        service.capabilities()


def test_live_binding_refuses_wrong_checkpoint_without_loading(live):
    _, _, engine, service = live
    engine.adapter.identity["fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="checkpoint differs"):
        service.capabilities()
    assert not engine.submitted


def test_live_config_refuses_unowned_files_refs_and_auto_execution_defaults(live):
    root, config, _, _ = live
    os.chmod(root / "live.json", 0o644)
    with pytest.raises(ValueError, match="owner-private"):
        load_live_config(root / "live.json")
    config["policy"] = {"automatic_promotion": True}
    write_json(root / "live.json", config)
    with pytest.raises(ValueError, match="explicit enabled execution"):
        load_live_config(root / "live.json")
    config.pop("policy")
    config["suites"][0]["argument_schema"] = {"$ref": "https://example.test/schema"}
    write_json(root / "live.json", config)
    with pytest.raises(ValueError, match="cannot resolve"):
        load_live_config(root / "live.json")


def test_live_trial_is_nonblocking_pollable_and_submit_once(live):
    _, _, _, service = live
    entered, release = threading.Event(), threading.Event()
    calls = []
    service.config["policy"] = RepairPolicy(enabled=True, execution_enabled=True)

    def trial(attempt_id):
        calls.append(attempt_id)
        entered.set()
        assert release.wait(2)
        return {"id": attempt_id, "session_id": "conversation", "state": "validated"}

    durable = {"state": "candidate"}

    def trial(attempt_id, wrapped=trial):
        result = wrapped(attempt_id)
        durable["state"] = result["state"]
        return result

    service._service = SimpleNamespace(
        get_attempt=lambda attempt_id: {"id": attempt_id, "session_id": "conversation", **durable},
        capabilities=lambda: {"mode": "fixture"}, trial_candidate=trial,
    )
    request = {"tool": "trial_candidate", "arguments": {"attempt_id": "candidate"}}
    try:
        receipt = service.call(request)
        assert receipt["pending"] is True and entered.wait(1)
        assert service.call(request) == receipt
        poll = {"tool": "get_repair_attempt", "arguments": {"attempt_id": "candidate"}}
        assert service.call(poll)["pending"] is True
        assert service.capabilities()["pending_trial_ids"] == ["candidate"]
        with pytest.raises(RuntimeError, match="trial is running"):
            service.call({"tool": "compile_analytical_repair", "arguments": {"capture_id": "linear-capture"}})
    finally:
        release.set()
    deadline = time.monotonic() + 2
    while service.call(poll)["pending"] and time.monotonic() < deadline:
        time.sleep(0.005)
    result = service.call(poll)
    assert result["state"] == "validated" and result["pending"] is False
    assert service.call(request) == result and calls == ["candidate"]


def _poll_until_done(service, attempt_id):
    poll = {"tool": "get_repair_attempt", "arguments": {"attempt_id": attempt_id}}
    deadline = time.monotonic() + 2
    while service.call(poll).get("pending") and time.monotonic() < deadline:
        time.sleep(0.005)
    return service.call(poll)


def test_live_poll_reports_controller_state_after_failed_manual_promotion(live):
    _, _, _, service = live
    service.config["policy"] = RepairPolicy(enabled=True, execution_enabled=True)
    durable = {"state": "candidate"}

    def trial(attempt_id):
        durable["state"] = "validated"
        return {"id": attempt_id, "session_id": "conversation", "state": "validated"}

    def promote(attempt_id):
        durable["state"] = "expired"  # what SelfRepairController journals
        raise ValueError("validation expired; a fresh trial is required")

    service._service = SimpleNamespace(
        get_attempt=lambda attempt_id: {"id": attempt_id, "session_id": "conversation", **durable},
        capabilities=dict, trial_candidate=trial, promote_validated_candidate=promote)
    service.call({"tool": "trial_candidate", "arguments": {"attempt_id": "a1"}})
    assert _poll_until_done(service, "a1")["state"] == "validated"
    with pytest.raises(ValueError, match="expired"):
        service.call({"tool": "promote_validated_candidate", "arguments": {"attempt_id": "a1"}})
    assert _poll_until_done(service, "a1")["state"] == "expired"
    # The cache refreshed after the failed mutation is also truthful.
    assert service._trials["a1"]["state"] == "expired"


def test_live_trial_worker_error_keeps_controller_recovery_state(live):
    _, _, _, service = live
    service.config["policy"] = RepairPolicy(enabled=True, execution_enabled=True,
                                            automatic_promotion=True)
    durable = {"state": "candidate"}

    def trial(attempt_id):  # deployment CAS applied, then its journal write failed
        durable.update(state="promotion_recovery_required", candidate={"name": attempt_id})
        raise OSError("disk full")

    service._service = SimpleNamespace(
        get_attempt=lambda attempt_id: {"id": attempt_id, "session_id": "conversation", **durable},
        capabilities=dict, trial_candidate=trial)
    service.call({"tool": "trial_candidate", "arguments": {"attempt_id": "a2"}})
    polled = _poll_until_done(service, "a2")
    assert polled["state"] == "promotion_recovery_required"
    assert polled["candidate"] == {"name": "a2"}
    assert polled["trial_worker_error"] == {"error_type": "OSError", "error": "disk full"}
    assert polled["pending"] is False


def test_live_trial_worker_error_without_controller_receipt_stays_pollable(live):
    _, _, _, service = live
    service.config["policy"] = RepairPolicy(enabled=True, execution_enabled=True)
    reads = []

    def get_attempt(attempt_id):
        reads.append(attempt_id)
        if len(reads) > 1:
            raise RuntimeError("controller unavailable")
        return {"id": attempt_id, "session_id": "conversation", "state": "candidate"}

    def trial(attempt_id):
        raise RuntimeError("worker crashed")

    service._service = SimpleNamespace(get_attempt=get_attempt, capabilities=dict,
                                       trial_candidate=trial)
    service.call({"tool": "trial_candidate", "arguments": {"attempt_id": "a3"}})
    for _ in range(2):  # every poll, not only the first, answers from the fallback
        polled = _poll_until_done(service, "a3")
        assert polled["state"] == "trial_failed" and polled["pending"] is False
        assert polled["trial_worker_error"]["error"] == "worker crashed"
        assert "controller unavailable" in polled["controller_receipt_error"]


def test_remote_proxy_bank_snapshot_omits_null_name(live, tmp_path):
    from jsonschema import Draft202012Validator

    from mlx2.experimental.weight_space.live import TOOLS
    from mlx2.experimental.weight_space.mcp_server import RemoteWeightSpaceToolService

    token = tmp_path / "admin.token"
    token.write_text("token")
    token.chmod(0o600)
    proxy = RemoteWeightSpaceToolService("http://127.0.0.1:8000", token)
    sent = []
    proxy._request = lambda endpoint, payload=None: sent.append(payload) or {}
    proxy.inspect_quantization_overlay()
    proxy.inspect_quantization_overlay("q3")
    assert [body["arguments"] for body in sent] == [{}, {"name": "q3"}]
    schema = TOOLS["inspect_quantization_overlay"]["inputSchema"]
    assert all(Draft202012Validator(schema).is_valid(body["arguments"]) for body in sent)


def test_live_experiments_null_is_absent_for_capabilities(live):
    root, raw, engine, _ = live
    engine.weight_space_live = None
    raw["experiments"] = None
    write_json(root / "live.json", raw)
    service = LiveWeightSpaceService(engine, load_live_config(root / "live.json"))
    assert service.capabilities()["experiment_mode"] is None


def test_live_one_malformed_turn_does_not_disable_service_forever(live):
    _, config, _, service = live
    bad = copy.deepcopy(config["session"]["request"])
    bad.pop("tools")
    service.observe_submission(SimpleNamespace(tenant_id="tenant-a", request=bad, lora_name=None))
    with pytest.raises(ValueError, match="tool schema differs"):
        service.capabilities()
    good = copy.deepcopy(config["session"]["request"])
    good["messages"].append({"role": "user", "content": "next"})
    service.observe_submission(SimpleNamespace(tenant_id="tenant-a", request=good, lora_name=None))
    assert service.config["session"].request["messages"][-1]["content"] == "next"
    assert service.capabilities()["session_source"] == "live_submission"


def test_live_experiment_bad_assets_install_no_engine_manager(live):
    root, raw, engine, _ = live
    engine.weight_space_live = None
    raw["experiments"] = {"mode": "layer_excision", "assets": "missing-assets.json"}
    raw["suites"] = []
    write_json(root / "live.json", raw)
    installed = []
    engine.configure_layer_excision = lambda **_: installed.append(1)
    service = LiveWeightSpaceService(engine, load_live_config(root / "live.json"))
    with pytest.raises(OSError):
        service.capabilities()
    assert installed == []
    raw["experiments"]["assets"] = {"path": "x"}
    write_json(root / "live.json", raw)
    with pytest.raises(ValueError, match="owner-relative"):
        load_live_config(root / "live.json")


def test_live_experiment_bind_failure_after_install_keeps_original_cause(live):
    root, raw, engine, _ = live
    engine.weight_space_live = None
    raw["experiments"], raw["suites"] = {"mode": "layer_excision"}, []
    write_json(root / "live.json", raw)
    installed = []

    def configure_layer_excision(**_):
        if installed:
            raise ValueError("a weight surgery manager is already configured")
        installed.append(1)
        # One layer cannot hold an interior excision: inventory registration fails.
        return SimpleNamespace(binding=SimpleNamespace(layers=[0], hidden_size=4),
                               register=lambda *args: None)

    engine.configure_layer_excision = configure_layer_excision
    service = LiveWeightSpaceService(engine, load_live_config(root / "live.json"))
    with pytest.raises(ValueError) as first:
        service.capabilities()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="restart the server") as later:
            service.capabilities()
        assert str(first.value) in str(later.value)
        assert "already configured" not in str(later.value)
    assert installed == [1]


def test_main_names_the_weight_space_extra_when_it_is_missing(tmp_path, monkeypatch, capsys):
    import sys

    from mlx2 import server
    from mlx2.experimental.weight_space import live as live_module

    token = tmp_path / "admin.token"
    token.write_text("operator-token")
    token.chmod(0o600)

    def missing(_path):
        raise ModuleNotFoundError("No module named 'jsonschema'", name="jsonschema")

    monkeypatch.setattr(live_module, "load_live_config", missing)
    monkeypatch.setattr(sys, "argv", ["mlx2.server", "--model", "unused", "--admin-token-file",
                                      str(token), "--weight-space-config", str(tmp_path / "live.json")])
    with pytest.raises(SystemExit) as caught:
        server.main()
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert "weight-space-research" in err and "jsonschema" in err


def test_live_experiment_reserved_inventory_collision_installs_no_manager(live):
    root, raw, engine, _ = live
    engine.weight_space_live = None
    write_json(root / "experiment-assets.json", {
        "schema": "mlx2.weight-experiment-assets.v1",
        "excision_inventories": [{"inventory_id": "loaded_model", "layer_count": 8, "hidden_size": 4}]})
    raw["experiments"] = {"mode": "layer_excision", "assets": "experiment-assets.json"}
    raw["suites"] = []
    write_json(root / "live.json", raw)
    installed = []
    engine.configure_layer_excision = lambda **_: installed.append(1) or SimpleNamespace(
        binding=SimpleNamespace(layers=[0] * 8, hidden_size=4), register=lambda *args: None)
    service = LiveWeightSpaceService(engine, load_live_config(root / "live.json"))
    for _ in range(2):  # an ordinary, retryable refusal, not the restart diagnostic
        with pytest.raises(ValueError, match="reserved loaded_model"):
            service.capabilities()
    assert installed == []
