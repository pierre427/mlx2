"""Shared media protocol/lifecycle tests. No service sockets or model loads."""
from __future__ import annotations

import io
import json
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mlx2.desktop_media_direct import (
    DirectServices,
    SharedMediaService,
    VideoJobs,
    serve,
    video_catalog,
)


def test_video_inventory_is_truthful_and_never_loads_missing_artifacts(tmp_path):
    settings = {"video_source": str(tmp_path / "source"), "video_model": str(tmp_path / "model"), "video_runtime": str(tmp_path / "runtime")}
    value = video_catalog(settings)
    assert value["implemented"] is True
    assert value["qualified"] is False
    assert value["available"] is False
    assert len(value["unavailable_reasons"]) == 3
    assert value["lora"]["load"] is True
    assert value["lora"]["training"] is False
    assert value["voice_training"]["implemented"] is False


def isolated_jobs(tmp_path):
    jobs = VideoJobs.__new__(VideoJobs)
    jobs.home = tmp_path
    jobs.settings = {"model_root": str(tmp_path)}
    jobs.root = tmp_path / "jobs"
    jobs.root.mkdir()
    jobs.lock = threading.RLock()
    jobs.queue = queue.Queue()
    jobs.cancel = {}
    jobs.stop = threading.Event()
    return jobs


def test_video_submission_durable_unique_and_cancel_before_ownership(tmp_path, monkeypatch):
    jobs = isolated_jobs(tmp_path)
    monkeypatch.setattr("mlx2.desktop_media_direct.video_catalog", lambda _: {"available": True, "checkpoint": "toy@cpu"})
    one = jobs.submit({"prompt": "A leaf in the wind", "width": 256, "height": 256, "frames": 9})
    two = jobs.submit({"prompt": "A leaf in the wind", "width": 256, "height": 256, "frames": 9})
    assert one["job_id"] != two["job_id"]
    assert len(jobs.jobs()) == 2
    assert "parameters" not in one
    assert jobs.cancel_job(one["job_id"])["status"] == "cancelled"
    assert jobs.cancel[one["job_id"]].is_set()
    with pytest.raises(ValueError, match="multiples"):
        jobs.submit({"prompt": "leaf", "width": 257})
    with pytest.raises(ValueError, match="Unsupported"):
        jobs.submit({"prompt": "leaf", "command": "shell"})
    with pytest.raises(ValueError):
        jobs.path("../../private-file")


def test_atomic_admission_drain_blocks_generation_and_can_resume():
    service = SharedMediaService.__new__(SharedMediaService)
    service.capability = "video"
    service.admission = threading.RLock()
    service.admitting = True
    service.video = SimpleNamespace(jobs=list, submit=Mock(return_value={"job_id": "one"}))
    assert service.call("service_drain", {}) == {"admission_closed": True, "active_jobs": 0}
    with pytest.raises(ValueError, match="admission"):
        service.call("generate_video", {"prompt": "a leaf"})
    service.video.submit.assert_not_called()
    service.call("service_resume", {})
    assert service.call("generate_video", {"prompt": "a leaf"}) == {"job_id": "one"}


def test_shared_mcp_has_real_schema_and_no_unexposed_training_claims():
    service = SharedMediaService.__new__(SharedMediaService)
    service.capability = "video"
    schemas = {row["name"]: row["inputSchema"] for row in service.tools()}
    assert {"generate_video", "get_video_job", "service_drain", "service_resume"} <= schemas.keys()
    assert "lora_path" in schemas["generate_video"]["properties"]
    assert not any("train" in name for name in schemas)
    service.capability = "image"
    image = {row["name"]: row["inputSchema"] for row in service.tools()}
    assert "reference_images" in image["edit_image"]["properties"]
    assert "backend" in image["generate_image"]["properties"]


def test_compatible_existing_service_is_reused_without_starting_process(tmp_path, monkeypatch):
    host = SimpleNamespace(home=tmp_path, preferences={"media_services": []}, primary=SimpleNamespace(library=SimpleNamespace(root=tmp_path)), configure_preferences=Mock())
    host.media = SimpleNamespace(client=lambda _: SimpleNamespace(list_tools=lambda: [{"name": "generate_image"}, {"name": "get_image_job"}]))
    spawn = Mock(side_effect=AssertionError("Existing canonical service must be reused"))
    monkeypatch.setattr("mlx2.desktop_media_direct.subprocess.Popen", spawn)
    result = DirectServices(host).start("image")
    assert result["state"] == "reused"
    spawn.assert_not_called()
    host.configure_preferences.assert_called_once()


def test_thin_mcp_authentication_and_dispatch_without_socket_or_runtime(tmp_path, monkeypatch):
    token = tmp_path / "token"
    token.write_text("private-test-token")
    token.chmod(0o600)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"token_file": str(token), "capability": "video", "home": str(tmp_path), "port": 8778}))
    responses = []
    backend = SimpleNamespace(admitting=True, tools=list, call=Mock(return_value={"job_id": "video_cpu"}), close=Mock())
    monkeypatch.setattr("mlx2.desktop_media_direct.SharedMediaService", lambda *_: backend)
    monkeypatch.setattr("mlx2.desktop_media_direct.signal.signal", lambda *_: None)
    class FakeServer:
        def __init__(self, address, handler):
            assert address == ("127.0.0.1", 8778)
            self.handler = handler
        def serve_forever(self, **_):
            for authorization in ["wrong", "Bearer private-test-token"]:
                message = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "generate_video", "arguments": {"prompt": "leaf"}}}).encode()
                handler = self.handler.__new__(self.handler)
                handler.headers = {"Authorization": authorization, "Content-Length": str(len(message))}
                handler.path = "/mcp"
                handler.rfile, handler.wfile = io.BytesIO(message), io.BytesIO()
                handler.send_response = lambda code: responses.append(code)
                handler.send_header = lambda *_: None
                handler.end_headers = lambda: None
                handler.do_POST()
                responses.append(json.loads(handler.wfile.getvalue()))
        def server_close(self):
            pass
    monkeypatch.setattr("mlx2.desktop_media_direct.ThreadingHTTPServer", FakeServer)
    serve(settings)
    assert responses[0] == 401
    assert responses[2] == 200
    assert responses[3]["result"]["structuredContent"] == {"job_id": "video_cpu"}
    backend.call.assert_called_once_with("generate_video", {"prompt": "leaf"})
    backend.close.assert_called_once()


def test_video_close_terminalizes_queued_jobs_and_refuses_new_admission(tmp_path, monkeypatch):
    jobs = isolated_jobs(tmp_path)
    jobs.worker = Mock()
    monkeypatch.setattr("mlx2.desktop_media_direct.video_catalog", lambda _: {"available": True, "checkpoint": "toy@cpu"})
    one = jobs.submit({"prompt": "a leaf"})
    jobs.close()
    assert jobs.read(one["id"])["status"] == "cancelled"
    assert jobs.cancel[one["id"]].is_set()
    with pytest.raises(ValueError, match="closing"):
        jobs.submit({"prompt": "another leaf"})
