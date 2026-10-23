"""Synthetic bytes/HF metadata only; no external network, models or GPU."""
import hashlib
import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2 import desktop_disk as disk
from mlx2 import desktop_downloads as downloads
from mlx2.desktop_records import RecordStore


class Response(io.BytesIO):
    def __init__(self, payload, status=200, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(payload))}


@pytest.fixture
def environment(tmp_path, monkeypatch):
    import huggingface_hub
    root = tmp_path / "models"
    root.mkdir()
    preferences = {"download_parallelism": 5, "download_bytes_per_second": 0, "hf_token_file": ""}
    store = RecordStore(tmp_path)
    control = SimpleNamespace(home=tmp_path, preferences=preferences, store=store, disk=disk.DiskBudget(),
        primary=SimpleNamespace(library=SimpleNamespace(root=root, refresh=lambda: None)),
        configure_preferences=lambda changes: preferences.update(changes))
    payloads = {"config.json": b'{"model_type":"synthetic"}', "weights.bin": b"synthetic weights"}
    calls, revisions = [], []
    commit = "a" * 40
    def info(repository, revision, **kw):
        revisions.append(revision)
        return SimpleNamespace(sha=commit, siblings=[SimpleNamespace(rfilename=name, size=len(data), lfs={"sha256": hashlib.sha256(data).hexdigest()}) for name, data in payloads.items()])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kw: SimpleNamespace(model_info=info))
    monkeypatch.setattr(huggingface_hub, "hf_hub_url", lambda repo, name, revision: "https://huggingface.co/" + repo + "/resolve/" + revision + "/" + name)
    def open_request(request, **kw):
        calls.append(request)
        name = request.full_url.split("/resolve/", 1)[1].split("/", 1)[1]
        data = payloads[name]
        if request.get_header("Range"):
            offset = int(request.get_header("Range").split("=")[1].split("-")[0])
            return Response(data[offset:], 206, {"Content-Range": f"bytes {offset}-{len(data)-1}/{len(data)}", "Content-Length": str(len(data)-offset)})
        return Response(data)
    monkeypatch.setattr(downloads.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=open_request))
    monkeypatch.setattr(disk.shutil, "disk_usage", lambda p: SimpleNamespace(free=1 << 40))
    manager = downloads.Downloads(control)
    yield SimpleNamespace(manager=manager, controller=control, root=root, payloads=payloads, calls=calls, revisions=revisions, info=info, opener=open_request)
    manager.close()
    store.close()


def finish(env, identity):
    for thread in list(env.manager.threads):
        thread.join(3)
        assert not thread.is_alive()
    return next(row for row in env.controller.store.jobs() if row["id"] == identity)


def test_default_parallelism_exact_pin_and_actual_bytes(environment):
    env = environment
    assert env.manager.pool._max_workers == 5
    identity = env.manager.start("owner/model")["job_id"]
    job = finish(env, identity)
    assert job["state"] == "completed", job
    assert env.revisions == ["main"]
    for name, data in env.payloads.items():
        assert (env.root / "model" / name).read_bytes() == data
    receipt = json.loads((env.root / "model/.mlx2-acquisition.json").read_text())
    assert receipt["resolved_revision"] == "a" * 40
    assert all("/resolve/" + "a" * 40 + "/" in request.full_url for request in env.calls)
    assert not env.controller.disk.reservations


def test_metadata_path_traversal_and_duplicates_refused_before_writes(environment, monkeypatch):
    import huggingface_hub
    env = environment
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kw: SimpleNamespace(model_info=lambda *a, **k: SimpleNamespace(sha="a"*40,
        siblings=[SimpleNamespace(rfilename="../outside", size=3)])))
    job = finish(env, env.manager.start("owner/model")["job_id"])
    assert job["state"] == "failed" and "filename" in job["progress"]
    assert not env.calls and not (env.root.parent / "outside").exists()
    assert not env.controller.disk.reservations


@pytest.mark.parametrize("name", ["/absolute", "../escape", "a/../escape", "a//b", "a/./b", "a\\b", ".mlx2-acquisition.json", "x.mlx2-partial"])
def test_unsafe_repository_names(name):
    with pytest.raises(ValueError):
        downloads.Downloads._name(name)


def test_state_and_partial_symlink_refusal(environment, tmp_path):
    env = environment
    outside = tmp_path / "outside"
    outside.mkdir()
    (env.root / ".download-state").symlink_to(outside, target_is_directory=True)
    job = finish(env, env.manager.start("owner/model")["job_id"])
    assert job["state"] == "failed" and not list(outside.iterdir())
    (env.root / ".download-state").unlink()
    staging = env.root / "staging"
    staging.mkdir()
    victim = outside / "personal"
    victim.write_bytes(b"keep")
    (staging / "config.json.mlx2-partial").symlink_to(victim)
    sibling = env.info("x", "main").siblings[0]
    with pytest.raises(ValueError, match="symlink"):
        env.manager._file("owner/model", "a"*40, sibling, staging, None, threading.Event(), "fake")
    assert victim.read_bytes() == b"keep"


def test_digest_mismatch_never_publishes(environment, monkeypatch):
    env = environment
    monkeypatch.setattr(downloads.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=lambda request, **kw: Response(b"x" * len(env.payloads[request.full_url.split("/resolve/", 1)[1].split("/", 1)[1]]))))
    job = finish(env, env.manager.start("owner/model")["job_id"])
    assert job["state"] == "failed" and "digest" in job["progress"]
    assert not (env.root / "model").exists() and not env.controller.disk.reservations


def test_pinned_resume_partial_range_and_finished_file_reuse(environment):
    env = environment
    identity = env.controller.store.enqueue("download", {"repository": "owner/model", "requested_revision": "main", "adapter_after": False, "root": str(env.root)})
    staging = env.root / ".download-state" / ("hf-" + identity)
    staging.mkdir(parents=True)
    data = env.payloads["config.json"]
    (staging / "config.json").write_bytes(data)
    (staging / "weights.bin.mlx2-partial").write_bytes(env.payloads["weights.bin"][:5])
    marker = {"repository": "owner/model", "job_id": identity, "resolved_revision": "a"*40,
        "expected_files": [{"path": n, "bytes": len(d)} for n, d in env.payloads.items()],
        "files": [{"path": "config.json", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}]}
    (staging / ".mlx2-acquisition.json").write_text(json.dumps(marker))
    env.controller.store.update(identity, "interrupted", "synthetic interruption")
    env.manager.dispatch({"command": "download_resume", "job_id": identity})
    job = finish(env, identity)
    assert job["state"] == "completed", job
    assert env.revisions == ["a"*40]
    assert len(env.calls) == 1 and env.calls[0].get_header("Range") == "bytes=5-"
    assert (env.root / "model/weights.bin").read_bytes() == env.payloads["weights.bin"]


@pytest.mark.parametrize("with_digest", [True, False])
def test_complete_partial_resumes_without_an_unsatisfiable_range(environment, monkeypatch, with_digest):
    env = environment
    staging = env.root / 'staging'
    staging.mkdir()
    payload = env.payloads['weights.bin']
    (staging / 'weights.bin.mlx2-partial').write_bytes(payload)
    sibling = env.info('owner/model', 'main').siblings[1]
    if not with_digest:
        sibling.lfs = None
    env.manager.progress['complete-partial'] = {'files_done': 0, 'downloaded_bytes': len(payload)}
    env.controller.disk.reserve('complete-partial', env.root, 0)

    def open_request(request, **kwargs):
        # HTTP servers reject bytes=<file-size>- with 416.
        assert request.get_header('Range') is None
        return env.opener(request, **kwargs)

    monkeypatch.setattr(downloads.urllib.request, 'build_opener', lambda *args: SimpleNamespace(open=open_request))
    result = env.manager._file('owner/model', 'a' * 40, sibling, staging, None, threading.Event(), 'complete-partial')
    assert result['sha256'] == hashlib.sha256(payload).hexdigest()
    assert (staging / 'weights.bin').read_bytes() == payload
    assert len(env.calls) == (0 if with_digest else 1)
    assert env.manager.progress['complete-partial'] == {'files_done': 1, 'downloaded_bytes': len(payload)}
    env.controller.disk.release('complete-partial')


def test_complete_partial_digest_mismatch_is_not_published(environment):
    env = environment
    staging = env.root / 'staging'
    staging.mkdir()
    payload = env.payloads['weights.bin']
    (staging / 'weights.bin.mlx2-partial').write_bytes(b'x' * len(payload))
    sibling = env.info('owner/model', 'main').siblings[1]
    env.manager.progress['bad-partial'] = {'files_done': 0, 'downloaded_bytes': len(payload)}
    with pytest.raises(downloads.DownloadError, match='digest'):
        env.manager._file('owner/model', 'a' * 40, sibling, staging, None, threading.Event(), 'bad-partial')
    assert not (staging / 'weights.bin').exists()
    assert not env.calls


def test_oversized_body_refused_before_unreserved_write(environment, monkeypatch):
    env = environment
    monkeypatch.setattr(downloads.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=lambda request, **kw: Response(b"x"*100000, headers={})))
    job = finish(env, env.manager.start("owner/model")["job_id"])
    assert job["state"] == "failed" and not (env.root / "model").exists()
    staging = env.root / ".download-state" / ("hf-" + job["id"])
    assert all(path.stat().st_size <= max(map(len, env.payloads.values())) for path in staging.glob("*.mlx2-partial"))


def test_crosshost_redirect_strips_authorization_and_refuses_http():
    request = downloads.urllib.request.Request("https://huggingface.co/a", headers={"Authorization": "Bearer private"})
    handler = downloads._SafeHubRedirect()
    redirected = handler.redirect_request(request, None, 302, "Found", {}, "https://cdn.example/model")
    assert redirected.get_header("Authorization") is None
    with pytest.raises(ValueError):
        handler.redirect_request(request, None, 302, "Found", {}, "http://cdn.example/model")


def test_secret_file_mode_and_no_token_in_job(environment):
    env = environment
    env.manager.dispatch({"command": "hf_save_token", "token": "hf_synthetic_secret"})
    token = Path(env.controller.preferences["hf_token_file"])
    assert token.stat().st_mode & 0o777 == 0o600
    identity = env.manager.start("owner/model")["job_id"]
    job = finish(env, identity)
    assert job["state"] == "completed"
    assert "hf_synthetic_secret" not in json.dumps(job)
    assert all(r.get_header("Authorization") == "Bearer hf_synthetic_secret" for r in env.calls)


def test_global_rate_is_aggregate_across_workers(monkeypatch):
    monkeypatch.setattr(downloads.time, "monotonic", lambda: 0)
    throttle = downloads.GlobalThrottle(100)
    delays = []
    cancel = SimpleNamespace(wait=lambda delay: delays.append(delay) or False)
    threads = [threading.Thread(target=throttle.consume, args=(100, cancel)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(delays) == [1, 2]


def test_disk_shared_device_reservations_used_and_external_pressure(tmp_path, monkeypatch):
    free = [100]
    monkeypatch.setattr(disk.shutil, "disk_usage", lambda path: SimpleNamespace(free=free[0]))
    budget = disk.DiskBudget()
    budget.reserve("one", tmp_path, 60, reserve=20)
    with pytest.raises(ValueError, match="Insufficient"):
        budget.reserve("two", tmp_path, 30, reserve=0)
    budget.used("one", 20)
    free[0] = 80
    budget.check("one", 10)
    budget.used("one", 10)
    assert budget.reservations["one"]["remaining"] == 50
    free[0] = 60
    with pytest.raises(ValueError):
        budget.check("one")
    with pytest.raises(ValueError):
        budget.consumed("one", -1)
    with pytest.raises(ValueError):
        budget.consumed("one", 1000)
    budget.release("one")
    assert not budget.reservations


def test_cancel_retires_workers_before_terminal_state_and_release(environment, monkeypatch):
    env = environment
    entered, released, retired = threading.Event(), threading.Event(), threading.Event()
    class BlockingResponse(Response):
        def read(self, size=-1):
            entered.set()
            released.wait(3)
            retired.set()
            raise OSError("synthetic connection interrupted")
        def close(self):
            released.set()
            super().close()
    monkeypatch.setattr(downloads.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=lambda request, **kw: BlockingResponse(env.payloads[request.full_url.split("/resolve/", 1)[1].split("/", 1)[1]])))
    original = env.controller.store.update
    def update(identity, state, progress, result=None):
        if state in {"cancelled", "failed", "completed"}:
            assert retired.is_set()
            assert identity not in env.controller.disk.reservations
            assert identity not in env.manager.responses
        return original(identity, state, progress, result)
    monkeypatch.setattr(env.controller.store, "update", update)
    identity = env.manager.start("owner/model")["job_id"]
    assert entered.wait(2)
    assert identity in env.controller.disk.reservations
    env.manager.dispatch({"command": "download_cancel", "job_id": identity})
    assert finish(env, identity)["state"] == "cancelled"


def test_bad_resume_range_refused_without_touching_partial(environment, monkeypatch):
    env = environment
    staging = env.root / "staging"
    staging.mkdir()
    partial = staging / "weights.bin.mlx2-partial"
    partial.write_bytes(b"synth")
    sibling = env.info("owner/model", "main").siblings[1]
    env.controller.disk.reserve("range", env.root, sibling.size - 5)
    env.manager.progress["range"] = {"files_done": 0, "downloaded_bytes": 5}
    monkeypatch.setattr(downloads.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=lambda *a, **k: Response(b"x", 206, {"Content-Range": "bytes 5-100/999"})))
    with pytest.raises(ValueError, match="range"):
        env.manager._file("owner/model", "a"*40, sibling, staging, None, threading.Event(), "range")
    assert partial.read_bytes() == b"synth"
    env.controller.disk.release("range")


def test_ignored_range_full_restart_counts_only_net_growth(environment, monkeypatch):
    env = environment
    staging = env.root / "staging"
    staging.mkdir()
    partial = staging / "weights.bin.mlx2-partial"
    partial.write_bytes(env.payloads["weights.bin"][:5])
    sibling = env.info("owner/model", "main").siblings[1]
    env.controller.disk.reserve("range", env.root, sibling.size - 5)
    env.manager.progress["range"] = {"files_done": 0, "downloaded_bytes": 5}
    monkeypatch.setattr(downloads.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=lambda *a, **k: Response(env.payloads["weights.bin"])))
    env.manager._file("owner/model", "a"*40, sibling, staging, None, threading.Event(), "range")
    assert env.controller.disk.reservations["range"]["remaining"] == 0
    assert env.manager.progress["range"]["downloaded_bytes"] == sibling.size
    env.controller.disk.release("range")


def test_exclusive_publish_never_overwrites_existing_directory(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir(); destination.mkdir()
    (source / "downloaded").write_bytes(b"new")
    with pytest.raises(OSError):
        downloads.Downloads._publish(source, destination)
    assert source.is_dir() and not list(destination.iterdir())


def test_provider_error_cannot_leak_token(environment, monkeypatch):
    import huggingface_hub
    def fail(*a, **kw):
        raise ValueError("Authorization: Bearer hf_private_signed_url")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kw: SimpleNamespace(model_info=fail))
    job = finish(environment, environment.manager.start("owner/model")["job_id"])
    assert job["state"] == "failed"
    assert "hf_private" not in json.dumps(job)


@pytest.mark.parametrize("value", ["http://huggingface.co/a/b", "https://huggingface.co:444/a/b", "https://huggingface.co/a/b?token=private", "https://user:pass@huggingface.co/a/b", "../model"])
def test_hf_target_rejects_untrusted_origins(value):
    with pytest.raises(ValueError):
        downloads.parse_hf_target(value)


def test_closed_download_manager_refuses_before_queuing_record(environment):
    env = environment
    env.manager.close()
    with pytest.raises(downloads.DownloadError, match="not queued"):
        env.manager.start("owner/model")
    assert env.controller.store.jobs() == []
    assert env.calls == []


def test_pasted_hf_target_allows_surrounding_whitespace_without_changing_revision():
    assert downloads.parse_hf_target("  https://huggingface.co/owner/model/tree/pinned  \n") == ("owner/model", "pinned")


@pytest.mark.parametrize("failure", ["refresh", "helper"])
def test_committed_download_remains_complete_when_followup_fails(environment, failure):
    env = environment
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic follow-up failure")
    env.controller.event = lambda *args, **kwargs: None
    env.controller.helpers = SimpleNamespace(dispatch=fail)
    if failure == "refresh":
        env.controller.primary.library.refresh = fail
    identity = env.manager.start("owner/model", adapter_after=failure == "helper")["job_id"]
    job = finish(env, identity)
    assert job["state"] == "completed"
    assert job["result"]["model_path"] == str(env.root / "model")
    assert (env.root / "model/config.json").is_file()
    assert identity not in env.controller.disk.reservations


def test_reinterrupted_resume_keeps_previously_verified_receipts(environment, monkeypatch):
    import concurrent.futures
    env = environment
    spec = {"repository": "owner/model", "requested_revision": "main", "adapter_after": False, "root": str(env.root), "resume": True}
    identity = env.controller.store.enqueue("download", spec)
    staging = env.root / ".download-state" / ("hf-" + identity)
    staging.mkdir(parents=True)
    data = env.payloads["config.json"]
    (staging / "config.json").write_bytes(data)
    verified = {"path": "config.json", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    marker = {"repository": "owner/model", "job_id": identity, "resolved_revision": "a"*40,
        "expected_files": [{"path": n, "bytes": len(d)} for n, d in env.payloads.items()], "files": [verified]}
    receipt = staging / ".mlx2-acquisition.json"
    receipt.write_text(json.dumps(marker))
    def not_started(*args):
        future = concurrent.futures.Future()
        future.set_exception(InterruptedError("cancelled before verification task ran"))
        return future
    monkeypatch.setattr(env.manager.pool, "submit", not_started)
    env.manager.progress[identity] = {"downloaded_bytes": 0, "files_done": 0}
    env.manager._run(identity, spec, threading.Event())
    assert json.loads(receipt.read_text())["files"] == [verified]
    assert env.controller.store.jobs()[0]["state"] == "cancelled"
    assert not env.controller.disk.reservations
