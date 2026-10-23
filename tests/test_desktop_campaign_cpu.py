"""Synthetic CPU-only queue/identity/ownership contracts, no checkpoint loads."""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2.desktop_campaign import CampaignQueue, comparison_rows, validate_profile
from mlx2.desktop_campaign_worker import verify_claim
from mlx2.desktop_records import RecordStore


def binding(checkpoint="small@abc", **changes):
    return {"checkpoint": checkpoint, "source_revision": "commit", "source_sha256": "snapshot",
            "runtime": {"mlx_native_sha256": "native", "source_scope": "adapter-one"},
            "host": "host", "profile": {"route": "ordinary", "context": 1024}, **changes}


def measured(checkpoint="small@abc", **changes):
    return {"binding": binding(checkpoint, **changes), "level": 3, "passed": True,
            "samples": [{"context_tokens": 1024, "width": 1, "prefill_tps": 99, "decode_tps": 10}]}


def test_compare_models_same_shape_only_and_newest_failure_invalidates():
    first = measured()
    second = measured("large@def", runtime={"mlx_native_sha256": "native", "source_scope": "adapter-two"})
    rows = comparison_rows([first, second])
    assert len(rows) == 2
    assert rows[0]["group"] == rows[1]["group"]
    changed = measured("other@ghi", profile={"route": "ordinary", "context": 4096})
    assert comparison_rows([changed, first])[0]["group"] != rows[0]["group"]
    failed = {**first, "created": time.time(), "passed": False}
    assert comparison_rows([failed, first]) == []


@pytest.mark.parametrize("value", [{"context": True}, {"lanes": 0}, {"lanes": 5, "inflight": 4},
                                    {"route": "shell"}, {"features": ["--anything"]}, {"unknown": 1}])
def test_profile_is_typed(value):
    with pytest.raises(ValueError):
        validate_profile(value)


def test_full_preflight_requires_explicit_approval(tmp_path):
    control = SimpleNamespace(home=tmp_path, store=RecordStore(tmp_path), engines={})
    queue = CampaignQueue(control)
    with pytest.raises(ValueError, match="explicit"):
        queue.enqueue({"paths": ["any"], "levels": [2], "approve_preflight": "yes"})
    assert not queue.running
    control.store.close()


def test_construction_does_not_resume_queued_jobs(tmp_path):
    store = RecordStore(tmp_path)
    identity = store.enqueue("qualification", {})
    other = store.enqueue("download", {})
    control = SimpleNamespace(home=tmp_path, store=store)
    queue = CampaignQueue(control)
    assert not queue.running
    states = {job["id"]: job["state"] for job in store.jobs()}
    assert states[identity] == "interrupted"
    assert states[other] == "queued"
    store.close()


def test_serial_queue_deduplicates_and_sorts_smallest(tmp_path, monkeypatch):
    rows = []
    for name, size in (("large", 200), ("small", 10)):
        path = tmp_path / name
        path.mkdir()
        rows.append({"path": str(path), "identity": name, "allocated_bytes": size})
    library = SimpleNamespace(rows=rows, checked_path=lambda raw, identity: Path(raw))
    store = RecordStore(tmp_path)
    control = SimpleNamespace(home=tmp_path, store=store, engines={}, primary=SimpleNamespace(library=library))
    queue = CampaignQueue(control)
    observed = []
    def unit(identity):
        job = next(j for j in store.jobs() if j["id"] == identity)
        observed.append(job["spec"]["path"])
        store.update(identity, "completed", "Fake CPU-only completion")
    monkeypatch.setattr(queue, "_unit", unit)
    result = queue.enqueue({"paths": [rows[0]["path"], rows[1]["path"], rows[0]["path"]], "levels": [1]})
    queue.thread.join(2)
    assert len(result["job_ids"]) == 2
    assert observed == [rows[1]["path"], rows[0]["path"]]
    store.close()


def test_lease_generation_expiry_and_owner_fail_closed():
    claim = {"claimed": True, "task_id": "task", "lease_generation": 2,
             "owner_agent_id": "desktop", "lease_expires_at": 200}
    assert verify_claim(claim, task="task", agent="desktop", now=100) == 2
    for change in ({"lease_generation": True}, {"lease_generation": 3}, {"lease_expires_at": 110},
                   {"lease_expires_at": float("nan")}, {"owner_agent_id": "foreign"}, {"claimed": False}):
        with pytest.raises(RuntimeError):
            verify_claim({**claim, **change}, task="task", agent="desktop", generation=2, now=100)


def test_badge_never_combines_source_or_profile_and_newest_failure_wins(tmp_path):
    store = RecordStore(tmp_path)
    for level in (1, 2):
        store.evidence({"binding": binding(), "level": level, "passed": True})
    store.evidence(measured(source_revision="other commit"))
    assert not store.summary("small@abc")["qualified"]
    store.evidence(measured())
    assert store.summary("small@abc")["qualified"]
    store.evidence({"binding": binding(), "level": 2, "passed": False})
    assert not store.summary("small@abc")["qualified"]
    store.close()


def test_ordinary_profile_cannot_claim_full_native_route_coverage(tmp_path):
    store = RecordStore(tmp_path)
    store.evidence({"binding": binding(), "level": 1, "passed": True})
    store.evidence({"binding": binding(), "level": 2, "passed": True,
                    "coverage": {"required_routes": ["ordinary", "native_mtp"], "tested_routes": ["ordinary"]}})
    store.evidence(measured())
    assert not store.summary("small@abc")["qualified"]
    assert "coverage pending" in store.summary("small@abc")["label"]
    store.close()


def test_human_coherence_review_uses_retained_exact_binding(tmp_path):
    store = RecordStore(tmp_path)
    control = SimpleNamespace(home=tmp_path, store=store)
    queue = CampaignQueue(control)
    identity = store.enqueue("qualification", {})
    result = {"review_pending": True, "binding": binding(), "ordinary_output": {"text": "Hello."}, "evidence_path": str(tmp_path)}
    store.update(identity, "awaiting_review", "Review", result)
    queue.review({"job_id": identity, "coherent": True})
    row = store.records()[0]
    assert row["binding"] == result["binding"]
    assert row["level"] == 1 and row["passed"]
    assert row["qualification_claim"] is False
    with pytest.raises(ValueError, match="unreviewed"):
        queue.review({"job_id": identity, "coherent": True})
    store.close()


class FakeRadio:
    def __init__(self):
        self.calls = []

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        responses = {
            "plan_session": {"session_id": "session", "task_ids": ["task"]},
            "start_workflow": {"workflow_id": "workflow"},
            "register_worker": {"worker_id": "worker"},
            "claim_task": {"claimed": True, "task_id": "task", "lease_generation": 1,
                           "owner_agent_id": "mlx2-desktop-job", "lease_expires_at": time.time() + 180},
            "release_task": {"released": True},
            "complete_worker": {"status": "completed"},
        }
        return responses[name]


def test_real_desktop_identity_and_release_on_body_failure(tmp_path):
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    radio = FakeRadio()
    with pytest.raises(RuntimeError, match="synthetic body failure"), DesktopCPGLease(tmp_path, "job", radio=radio) as lease:
        assert lease.check()["agent"] == "mlx2_desktop_campaign"
        assert lease.owner["cpg_session"] == "session"
        raise RuntimeError("synthetic body failure")
    assert radio.calls[-2][0] == "release_task"
    assert radio.calls[-2][1]["status"] == "failed"
    assert radio.calls[-1][0] == "complete_worker"
    assert not lease.thread.is_alive()
    with pytest.raises(RuntimeError, match="no longer live"):
        lease.check()


def test_lease_renewal_failure_never_silently_continues(tmp_path):
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    lease = DesktopCPGLease(tmp_path, "job", radio=FakeRadio())
    lease.owner["expires"] = time.time() + 100
    lease.error = RuntimeError("radio unavailable")
    with pytest.raises(RuntimeError, match="renewal failed"):
        lease.check()


def test_cancel_retires_nested_independent_process_groups(tmp_path):
    import os
    import subprocess
    import sys

    import psutil

    from mlx2.desktop_campaign_worker import stop
    pid_file = tmp_path / "child.pid"
    program = (
        "import pathlib,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True); "
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid));time.sleep(60)"
    )
    process = subprocess.Popen([sys.executable, "-c", program, str(pid_file)], start_new_session=True)
    try:
        deadline = time.monotonic() + 3
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_file.exists()
        inner = psutil.Process(int(pid_file.read_text()))
        assert os.getpgid(inner.pid) != os.getpgid(process.pid)
        stop(process, timeout=3)
        assert process.poll() is not None
        assert not inner.is_running() or inner.status() == psutil.STATUS_ZOMBIE
    finally:
        if process.poll() is None:
            stop(process, timeout=1)


def test_badges_reject_historical_source_host_artifact_runtime_and_shape(tmp_path):
    from mlx2.desktop_badges import BadgeResolver
    host = {"hostname": "host", "platform": "test"}
    build = {"source_sha256": "current", "mlx_native_sha256": "native"}
    source = {"source_revision": "commit", "source_sha256": "snapshot"}
    resolver = BadgeResolver(tmp_path, source_identity=source, build_identity=build,
                             runtime_resolver=lambda old: build, host=host)
    row = {"path": "/model", "checkpoint": "small@abc", "default_route": "ordinary"}
    bound = binding(host=host, runtime=build, model_path="/model",
                    profile={"route": "ordinary", "settings": {"max_context": 1024, "max_lanes": 1}})
    records = [{"binding": bound, "level": level, "passed": True} for level in (1, 2, 3)]
    assert resolver.summarize(row, records)["qualified"]
    for changes in ({"source_revision": "old"}, {"source_sha256": "old"}, {"checkpoint": "small@changed"},
                    {"host": {"hostname": "another", "platform": "test"}},
                    {"runtime": {"source_sha256": "old", "mlx_native_sha256": "native"}}):
        stale = [{**record, "binding": {**bound, **changes}} for record in records]
        summary = resolver.summarize(row, stale)
        assert not summary["qualified"]
        assert summary["historical_records"] == 3
    profiles = [{"path": "/model", "route": "ordinary", "context": 4096, "lanes": 1}]
    assert not resolver.summarize(row, records, profiles)["qualified"]
    profiles[0]["context"] = 1024
    assert resolver.summarize(row, records, profiles)["qualified"]


def test_build_identity_can_be_read_without_importing_mlx():
    import os
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    program = '''import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "mlx" or name.startswith("mlx."):
        raise RuntimeError("Unexpected MLX import during CPU identity computation")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from mlx2.serving import runtime_identity
identity = runtime_identity()
assert identity["source_sha256"] and identity["mlx_native_sha256"]
'''
    result = subprocess.run([sys.executable, "-c", program], env={**os.environ, "PYTHONPATH": str(root / "src")},
                            capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("route", [None, [], {}, True, 1])
def test_invalid_route_types_are_actionable_validation_errors(route):
    with pytest.raises(ValueError, match="route"):
        validate_profile({"route": route})


def test_empty_or_unknown_selection_with_idle_unloaded_engine_never_starts(tmp_path):
    store = RecordStore(tmp_path)
    engine = SimpleNamespace(worker=None, child=None, preparing=lambda: False)
    control = SimpleNamespace(home=tmp_path, store=store, engines={"default": engine},
                              primary=SimpleNamespace(library=SimpleNamespace(rows=[])))
    queue = CampaignQueue(control)
    for paths, message in (([], "Select one"), ([str(tmp_path / "gone")], "Refresh")):
        with pytest.raises(ValueError, match=message):
            queue.enqueue({"paths": paths, "levels": [1]})
    assert store.jobs() == []
    assert queue.thread is None
    assert queue.status()["comparisons"] == []
    assert "No qualification" in queue.logs()["text"]
    store.close()


def test_preparing_engine_without_worker_receipt_blocks_queue_before_acquisition(tmp_path):
    store = RecordStore(tmp_path)
    engine = SimpleNamespace(worker=None, child=None, preparing=lambda: True)
    queue = CampaignQueue(SimpleNamespace(home=tmp_path, store=store, engines={"default": engine}))
    with pytest.raises(ValueError, match="Stop managed"):
        queue.enqueue({"paths": ["unavailable"], "levels": [1]})
    assert queue.thread is None
    assert store.jobs() == []
    store.close()


@pytest.mark.parametrize("url", [None, 42, {}, [], "https://127.0.0.1:8766/mcp", "http://example.com:8766/mcp"])
def test_invalid_cpg_settings_never_persist_or_launch(tmp_path, url):
    store = RecordStore(tmp_path)
    queue = CampaignQueue(SimpleNamespace(home=tmp_path, store=store))
    with pytest.raises(ValueError):
        queue.dispatch({"command": "campaign_configure", "settings": {"mcp_url": url}})
    assert not queue.settings_path.exists()
    assert queue.thread is None
    store.close()
