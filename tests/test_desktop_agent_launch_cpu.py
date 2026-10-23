"""Synthetic agent/model-instance launches; no model, HTTP or GPU work."""
import json
import threading
from types import SimpleNamespace

import pytest

from mlx2 import desktop_agent_launch as module


@pytest.fixture
def launch(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    path = root / "synthetic-q4"
    path.mkdir()
    fingerprint = {"value": "weights-v1"}
    def checked(raw, identity):
        if str(raw) != str(path) or identity != "current-folder":
            raise ValueError("stale library entry")
        return path
    inspector = lambda _path: SimpleNamespace(artifact_fingerprint=fingerprint["value"], descriptor=SimpleNamespace(metadata={"execution_kind": "autoregressive"}))
    controller = SimpleNamespace(home=tmp_path, preferences={"profiles": [], "agent_delegation_enabled": True},
        lock=threading.RLock(), user_profiles=SimpleNamespace(active="alice"), primary=SimpleNamespace(library=SimpleNamespace(checked_path=checked)),
        engines={}, event=lambda *args, **kwargs: None, stopping=threading.Event())
    controller.profile = lambda identity: next((row for row in controller.preferences["profiles"] if row["id"] == identity), None)
    controller.configure_preferences = lambda changes: controller.preferences.update(changes)
    calls = []
    agent_state = {"value": "completed"}
    def agents(request):
        calls.append(request)
        if request["command"] in {"agent_fork", "agent_delegate"}:
            return {"job_id": "synthetic-agent", "parent_id": request["parent_id"]}
        return {"id": request.get("job_id"), "status": agent_state["value"], "synthesis": "Synthetic child report"}
    controller.agents = SimpleNamespace(dispatch=agents)
    ready = threading.Event()
    ready.set()
    created = threading.Event()
    request_calls = []
    def start(identity):
        profile = controller.profile(identity)
        child = SimpleNamespace(pid=111, poll=lambda: None)
        engine = SimpleNamespace(settings={**profile}, child=child, preparation_thread=object(), stopped=0)
        engine.worker = {"state": "Loading", "url": "http://127.0.0.1:19999", "checkpoint": path.name + "@" + fingerprint["value"]}
        engine.snapshot = lambda: engine.worker.update(state="Ready" if ready.is_set() else "Loading")
        def stop():
            engine.stopped += 1
            engine.child = None
            engine.worker["state"] = "Stopped"
        engine.stop = stop
        def request(method, route, body=None):
            request_calls.append((method, route, body))
            if method == "POST":
                return {"id": body["lora_name"], "fingerprint": "adapter-v1"}
            return {"artifact": fingerprint["value"], "runtime": {"source_sha256": "actual-runner-source"},
                    "multi_lora": {"enabled": True, "registered": [] if not any(call[0] == "POST" for call in request_calls) else ["adapted"]}}
        engine.suspension = SimpleNamespace(request=request)
        controller.engines[identity] = engine
        created.set()
    controller.start_profile = start
    monkeypatch.setattr(module, "source_identity", lambda: {"source_revision": "synthetic-source"})
    manager = module.AgentLaunch(controller, inspector=inspector, poll_seconds=0.001)
    yield SimpleNamespace(manager=manager, controller=controller, path=path, fingerprint=fingerprint,
        ready=ready, created=created, calls=calls, agent_state=agent_state, request_calls=request_calls)
    manager.close()


def run(launch, **changes):
    receipt = launch.manager.dispatch({"command": "agent_launch", "path": str(launch.path), "identity": "current-folder",
        "goal": "Inspect a synthetic candidate", **changes}, principal="chat:alice")
    row = launch.manager.jobs[receipt["job_id"]]
    row["thread"].join(2)
    return row


def test_new_instance_binds_exact_quant_artifact_and_never_shares_or_autostarts(launch):
    row = run(launch, lanes=3, context=4096)
    assert row["state"] == "completed"
    profile = row["created_profile"]
    assert not profile["sticky_port"] and not profile["share"] and not profile["autostart"]
    assert profile["lanes"] == 3 and profile["context"] == 4096
    assert launch.calls[0]["target"] == {"url": "http://127.0.0.1:19999", "pid": 111}
    assert row["checkpoint"] == "synthetic-q4@weights-v1"
    assert row["engine"].stopped == 1
    assert launch.controller.profile(row["profile_id"]) is None
    assert launch.manager.snapshot(principal="chat:other")["jobs"] == []
    assert len(launch.manager.snapshot(principal="chat:alice")["jobs"]) == 1


def test_seeded_fork_retains_parent_provenance_but_never_serializes_history(launch):
    history = [{"role": "user", "content": "PRIVATE HISTORY SENTINEL"}]
    row = run(launch, _history=history, parent_id="parent-branch")
    assert row["state"] == "completed"
    assert launch.calls[0]["command"] == "agent_fork"
    assert launch.calls[0]["_history"] == history
    assert launch.calls[0]["parent_id"] == "parent-branch"
    assert "PRIVATE HISTORY SENTINEL" not in (launch.manager.home / (row["id"] + ".json")).read_text()
    assert row["seed_receipt"]["messages"] == 1


def user_instance(launch, **options):
    # Reuse tests bind an independently operator-owned running profile.
    first = run(launch, **options)
    profile = {**first["created_profile"], "id": "user-owned"}
    launch.controller.preferences["profiles"].append(profile)
    launch.controller.start_profile("user-owned")
    first["profile_id"] = "user-owned"
    first["engine"] = launch.controller.engines["user-owned"]
    first["engine"].snapshot()
    return first


def test_reuse_exact_ready_instance_never_creates_or_stops_another_engine(launch):
    first = user_instance(launch)
    launch.calls.clear()
    second = run(launch, instance="reuse", profile_id=first["profile_id"])
    assert second["state"] == "completed"
    assert len(launch.controller.preferences["profiles"]) == 1
    assert second["engine"] is first["engine"]
    assert first["engine"].stopped == 0


def test_failed_child_stops_only_its_newly_owned_instance(launch):
    launch.agent_state['value'] = 'failed'
    row = run(launch)
    assert row['state'] == 'failed' and row['engine'].stopped == 1


def test_failed_child_on_reused_engine_does_not_stop_user_engine(launch):
    first = user_instance(launch)
    launch.agent_state['value'] = 'failed'
    second = run(launch, instance='reuse', profile_id=first['profile_id'])
    assert second['state'] == 'failed'
    assert first['engine'].stopped == 0


def test_cancel_load_stops_only_new_owned_instance(launch):
    launch.ready.clear()
    receipt = launch.manager.dispatch({"command": "agent_launch", "path": str(launch.path), "identity": "current-folder", "goal": "bounded"})
    assert launch.created.wait(1)
    launch.manager.dispatch({"command": "agent_launch_cancel", "job_id": receipt["job_id"]})
    row = launch.manager.jobs[receipt["job_id"]]
    row["thread"].join(2)
    assert row["state"] == "cancelled"
    assert row["engine"].stopped == 1
    assert launch.calls == []


def test_profile_switch_cancels_without_exposing_old_launch_receipts(launch):
    launch.ready.clear()
    receipt = launch.manager.dispatch({"command": "agent_launch", "path": str(launch.path), "identity": "current-folder", "goal": "bounded"})
    assert launch.created.wait(1)
    launch.controller.user_profiles.active = "bob"
    row = launch.manager.jobs[receipt["job_id"]]
    row["thread"].join(2)
    assert row["state"] == "cancelled"
    assert launch.manager.snapshot()["jobs"] == []
    with pytest.raises(ValueError, match="Unknown"):
        launch.manager.dispatch({"command": "agent_launch_status", "job_id": row["id"]})


def test_explicit_lora_uses_ordinary_route_and_named_verified_adapter(launch, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    row = run(launch, lora_path=str(adapter), lora_name="adapted", max_lora_rank=32)
    assert row["state"] == "completed"
    profile = row["created_profile"]
    assert profile["route"] == "ordinary" and profile["lora_root"] == str(adapter.resolve())
    assert profile["max_loras"] == 1 and profile["max_lora_rank"] == 32
    assert row["lora_receipt"]["fingerprint"] == "adapter-v1"
    assert launch.calls[0]["target"]["model"] == "adapted"
    assert launch.request_calls[1][1] == "/v1/load_lora_adapter"


@pytest.mark.parametrize("changes", [{"lanes": True}, {"lanes": 5}, {"context": -1}, {"instance": "shell"},
    {"goal": ""}, {"identity": "stale"}, {"lora_name": "unbound"}, {"max_tokens": 1}, {"unknown": "x"}])
def test_invalid_inputs_never_start_a_profile_or_delegate(launch, changes):
    with pytest.raises((ValueError, TypeError)):
        run(launch, **changes)
    assert launch.controller.engines == {}
    assert launch.calls == []


def test_reload_receipts_marks_active_jobs_interrupted_without_replay(launch):
    path = launch.manager.home / "old.json"
    path.write_text(json.dumps({"id": "old", "state": "loading", "user_profile": "alice", "owner": "chat:alice"}))
    restored = module.AgentLaunch(launch.controller, inspector=lambda _path: pytest.fail("No startup inspection"))
    assert restored.snapshot(principal="chat:alice")["jobs"][0]["state"] == "interrupted"
    assert launch.controller.engines == {}
    restored.close()


def test_same_checkpoint_new_instances_have_distinct_ownership(launch):
    first = user_instance(launch)
    second = run(launch)
    assert first["checkpoint"] == second["checkpoint"]
    assert first["profile_id"] != second["profile_id"]
    assert first["engine"] is not second["engine"]
    assert first["child"] is not second["child"]


def test_artifact_change_while_loading_refuses_agent_and_retires_owned_instance(launch):
    launch.ready.clear()
    receipt = launch.manager.dispatch({"command": "agent_launch", "path": str(launch.path), "identity": "current-folder", "goal": "bounded"})
    assert launch.created.wait(1)
    launch.fingerprint["value"] = "different-quant-artifact"
    launch.ready.set()
    row = launch.manager.jobs[receipt["job_id"]]
    row["thread"].join(2)
    assert row["state"] == "failed" and "artifact changed" in row["error"]
    assert row["engine"].stopped == 1
    assert launch.calls == []


def test_replaced_child_is_never_stopped_by_stale_job_cleanup(launch):
    launch.ready.clear()
    receipt = launch.manager.dispatch({"command": "agent_launch", "path": str(launch.path), "identity": "current-folder", "goal": "bounded"})
    assert launch.created.wait(1)
    row = launch.manager.jobs[receipt["job_id"]]
    # Wait for launch ownership capture before replacing its child.
    assert threading.Event().wait(0.02) is False
    engine = launch.controller.engines[row["profile_id"]]
    original = engine.child
    row["child"] = original
    engine.child = SimpleNamespace(pid=222, poll=lambda: None)
    row["cancel"].set()
    row["thread"].join(2)
    assert row["state"] == "cancelled"
    assert engine.stopped == 0
    assert engine.child.pid == 222


def test_model_principal_cannot_read_or_cancel_another_principal_job(launch):
    row = run(launch)
    for command in ("agent_launch_status", "agent_launch_cancel"):
        with pytest.raises(ValueError, match="Unknown"):
            launch.manager.dispatch({"command": command, "job_id": row["id"]}, principal="chat:other")
    assert not row["cancel"].is_set()


def test_four_independent_launches_run_concurrently_without_lost_profiles(launch):
    launch.agent_state["value"] = "running"
    receipts = []
    for number in range(4):
        receipts.append(launch.manager.dispatch({"command": "agent_launch", "path": str(launch.path),
            "identity": "current-folder", "goal": f"Independent goal {number}"}))
    deadline = module.time.monotonic() + 2
    while any(launch.manager.jobs[row["job_id"]]["state"] != "running" for row in receipts) and module.time.monotonic() < deadline:
        threading.Event().wait(0.001)
    assert all(launch.manager.jobs[row["job_id"]]["state"] == "running" for row in receipts)
    assert len(launch.controller.preferences["profiles"]) == 4
    assert len({profile["id"] for profile in launch.controller.preferences["profiles"]}) == 4
    with pytest.raises(ValueError, match="four"):
        launch.manager.dispatch({"command": "agent_launch", "path": str(launch.path), "identity": "current-folder", "goal": "fifth"})
    launch.agent_state["value"] = "completed"
    for receipt in receipts:
        row = launch.manager.jobs[receipt["job_id"]]
        row["thread"].join(2)
        assert row["state"] == "completed"
        assert row["agent_result"]["synthesis"] == "Synthetic child report"


def test_fresh_launch_uses_only_goal_seed_and_status_contains_final_child_result(launch):
    row = run(launch, goal="Fresh isolated goal")
    fork = launch.calls[0]
    assert fork["command"] == "agent_fork"
    assert fork["_history"] == [{"role": "user", "content": "Fresh isolated goal"}]
    assert row["seed_kind"] == "goal_only"
    status = launch.manager.dispatch({"command": "agent_launch_status", "job_id": row["id"]}, principal="chat:alice")
    assert status["agent_result"]["synthesis"] == "Synthetic child report"
    assert "history" not in status
    assert "_history" not in status["agent_result"]


def test_final_child_result_is_bounded_and_private_parent_context_is_not_duplicated():
    result = module.AgentLaunch._bounded_agent_result({"id": "child", "status": "completed",
        "synthesis": "x" * 1_000_000, "_history": [{"content": "PRIVATE PARENT"}], "target": {"api_key": "PRIVATE KEY"}})
    assert result["truncated"] and result["total_bytes"] > 256 * 1024
    assert len(result["preview"].encode()) <= 64 * 1024
    assert "PRIVATE PARENT" not in json.dumps(result)
    assert "PRIVATE KEY" not in json.dumps(result)
    assert module.AgentLaunch._bounded_agent_result({"status": "failed", "error": "child failure", "_history": ["PRIVATE"]}) == {
        "status": "failed", "error": "child failure"}


@pytest.mark.parametrize('proof', ['exact', 'wrong', 'missing'])
def test_reusing_registered_lora_requires_exact_source_fingerprint(launch, tmp_path, proof):
    from mlx2.runtime.apc_suspension import lora_file_receipt
    adapter = tmp_path / 'adapter-reuse'
    adapter.mkdir()
    (adapter / 'adapter_config.json').write_text(json.dumps({'lora_parameters': {'rank': 8, 'keys': ['self_attn.q_proj']}}))
    (adapter / 'adapters.safetensors').write_bytes(b'synthetic non-loaded weights')
    expected = lora_file_receipt(adapter)['fingerprint']
    first = user_instance(launch, lora_path=str(adapter), lora_name='adapted')
    assert first['state'] == 'completed'
    def request(method, route, body=None):
        assert method == 'GET', 'Registered adapter must not be loaded again'
        state = {'enabled': True, 'registered': ['adapted']}
        if proof != 'missing':
            state['registered_fingerprints'] = {'adapted': expected if proof == 'exact' else 'foreign'}
        return {'artifact': 'weights-v1', 'runtime': {'source_sha256': 'actual-runner-source'}, 'multi_lora': state}
    first['engine'].suspension.request = request
    second = run(launch, instance='reuse', profile_id=first['profile_id'], lora_path=str(adapter), lora_name='adapted')
    assert second['state'] == ('completed' if proof == 'exact' else 'failed')
    assert first['engine'].stopped == 0
    if proof == 'exact':
        assert second['lora_receipt']['status'] == 'reused'


def test_successful_launches_retire_owned_profiles_before_profile_cap(launch):
    for _ in range(34):
        row = run(launch)
        assert row["state"] == "completed" and row["engine"].stopped == 1
        assert launch.controller.preferences["profiles"] == []
