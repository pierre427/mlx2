from collections import Counter
from http.server import ThreadingHTTPServer
import json
import os
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mlx2.server import (
    SignalShutdownController,
    authorize_admin,
    handler_for,
    load_admin_token,
    validate_quiesce_body,
    validate_resume_body,
)
from mlx2.serving import AdmissionClosed, Job, SuspendUnavailable


class AdminEngine:
    model_path = "fixture"

    def __init__(self):
        self.lock = threading.Lock()
        self.counts = Counter()
        self.state = "serving"
        self.calls = []

    def service_state(self):
        return {
            "state": self.state,
            "since": 1.0,
            "timestamps": {self.state: 1.0},
            "last_transition": {"result": {"status": "test"}},
        }

    def quiesce(self, **options):
        self.calls.append(("quiesce", options))
        if self.state == "serving":
            self.state = "draining"
        return self.service_state()

    def resume(self, *, prefetch_sessions=()):
        self.calls.append(("resume", tuple(prefetch_sessions)))
        self.state = "serving"
        return self.service_state()

    def status(self):
        return {
            "healthy": True,
            "error": None,
            "model": "fixture",
            "quiesce": self.service_state(),
            "counts": dict(self.counts),
        }

    def batching_status(self):
        return {"gauges": {"queue_depth": 0}}


class RejectEngine(AdminEngine):
    def __init__(self, state="suspended"):
        super().__init__()
        self.state = state

    def acquire_admission(self, endpoint_class="generation"):
        raise AdmissionClosed(self.state, endpoint_class)

    def count_tokens(self, request):
        return 1


@pytest.fixture
def admin_server():
    engine = AdminEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield engine, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def _post(base, path, body=b"", headers=None):
    return urlopen(
        Request(
            base + path,
            data=body,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
    )


def test_admin_quiesce_state_health_and_resume_are_idempotent(admin_server):
    engine, base = admin_server
    with _post(base, "/v1/admin/quiesce", b"{}") as response:
        assert response.status == 202
        assert json.load(response)["state"] == "draining"
    with _post(base, "/v1/admin/quiesce", b'{"suspend":false}') as response:
        assert response.status == 202
        assert json.load(response)["state"] == "draining"
    with pytest.raises(HTTPError) as caught:
        urlopen(base + "/health")
    assert caught.value.code == 503
    assert json.load(caught.value) == {"status": "draining"}
    with urlopen(base + "/v1/admin/state") as response:
        assert json.load(response)["state"] == "draining"
    body = json.dumps(
        {"prefetch_sessions": [{"tenant": "t", "session_id": "s-1"}]}
    ).encode()
    with _post(base, "/v1/admin/resume", body) as response:
        assert response.status == 202
        assert json.load(response)["state"] == "serving"
    assert engine.calls[-1] == ("resume", (("t", "s-1"),))


def test_admin_auth_helpers_and_token_file(tmp_path):
    assert authorize_admin("127.0.0.1", None) is None
    assert authorize_admin("127.255.255.255", None) is None
    assert authorize_admin("::1", None) is None
    assert authorize_admin("192.168.1.2", None)[0] == 403
    assert authorize_admin("::ffff:127.0.0.1", None)[0] == 403
    assert authorize_admin("127.0.0.1", "Bearer secret", "secret") is None
    assert authorize_admin("127.0.0.1", "Bearer wrong", "secret")[0] == 401
    assert authorize_admin("127.0.0.1", "Bearer sécret", "sécret") is None
    assert authorize_admin("127.0.0.1", "Bearer ☃", "sécret")[0] == 401

    path = tmp_path / "admin.token"
    path.write_text("secret\n")
    os.chmod(path, 0o600)
    assert load_admin_token(path) == "secret"
    os.chmod(path, 0o640)
    with pytest.raises(ValueError, match="0600"):
        load_admin_token(path)


def test_admin_token_is_required_on_loopback(tmp_path):
    engine = AdminEngine()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), handler_for(engine, admin_token="secret")
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(HTTPError) as caught:
            urlopen(base + "/v1/admin/state")
        assert caught.value.code == 401
        with urlopen(
            Request(
                base + "/v1/admin/state",
                headers={"Authorization": "Bearer secret"},
            )
        ) as response:
            assert json.load(response)["state"] == "serving"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_suspend_without_disk_tier_is_http_409():
    class NoDiskEngine(AdminEngine):
        def quiesce(self, **options):
            raise SuspendUnavailable("cache suspension requires a disk tier")

    engine = NoDiskEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with pytest.raises(HTTPError) as caught:
            _post(base, "/v1/admin/quiesce", b"{}")
        assert caught.value.code == 409
        assert "disk tier" in json.load(caught.value)["error"]["message"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "value",
    [
        {"unknown": 1},
        {"drain_timeout_seconds": True},
        {"drain_timeout_seconds": 0},
        {"drain_timeout_seconds": 3601},
        {"suspend": 1},
    ],
)
def test_quiesce_body_validation_is_exact(value):
    with pytest.raises(ValueError):
        validate_quiesce_body(value)


def test_resume_body_validation_is_exact_and_bounded():
    assert validate_resume_body({}) == ()
    assert validate_resume_body(
        {"prefetch_sessions": [{"tenant": "t", "session_id": "session:1"}]}
    ) == (("t", "session:1"),)
    with pytest.raises(ValueError):
        validate_resume_body({"prefetch_sessions": [{"tenant": "t", "session_id": "!"}]})
    with pytest.raises(ValueError):
        validate_resume_body({"prefetch_sessions": [{}]})


@pytest.mark.parametrize(
    ("path", "body", "expected_type"),
    [
        (
            "/v1/chat/completions",
            {"model": "fixture", "messages": [{"role": "user", "content": "hi"}]},
            "server_error",
        ),
        (
            "/v1/messages",
            {"model": "fixture", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
            "api_error",
        ),
    ],
)
def test_nonserving_admission_uses_endpoint_envelope_and_retry_after(
    path, body, expected_type
):
    engine = RejectEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with pytest.raises(HTTPError) as caught:
            _post(base, path, json.dumps(body).encode())
        assert caught.value.code == 503
        assert caught.value.headers["Retry-After"] == "1"
        payload = json.load(caught.value)
        assert payload["error"]["type"] == expected_type
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_accepted_embedding_batch_row_uses_batch_admission_during_drain():
    class BatchEmbeddingEngine(AdminEngine):
        def embed(
            self,
            inputs,
            *,
            dimensions=None,
            admission_class="embeddings",
            admitted=False,
        ):
            if self.state != "serving" and not (
                admission_class == "batch" and admitted
            ):
                raise AdmissionClosed(self.state, admission_class)
            return [[1.0] for _ in inputs], len(inputs)

    engine = BatchEmbeddingEngine()
    handler_for(engine)
    engine.state = "draining"
    status, payload = engine.api_resources["batches"].executor(
        "/v1/embeddings",
        {"model": "fixture", "input": "accepted"},
        "default",
    )
    assert status == 200
    assert payload["usage"] == {"prompt_tokens": 1, "total_tokens": 1}


def test_stream_started_before_quiesce_finishes_while_new_request_is_rejected():
    class StreamingEngine(AdminEngine):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.finish = threading.Event()
            self.leases = set()

        def acquire_admission(self, endpoint_class="generation"):
            if self.state != "serving":
                raise AdmissionClosed(self.state, endpoint_class)
            token = object()
            self.leases.add(token)
            return token

        def release_admission(self, token):
            self.leases.discard(token)

        def submit(self, request, *, tenant_id="default", admitted=False):
            assert admitted
            job = Job(request, tenant_id=tenant_id)
            job.prompt_tokens = 1
            job.completion_tokens = 1
            self.started.set()

            def produce():
                job.events.put({"text": "hello"})
                self.finish.wait(2)
                job.events.put({"finish_reason": "stop", "receipt": {"cache": "apcv2"}})

            threading.Thread(target=produce, daemon=True).start()
            return job

    engine = StreamingEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    first = {}

    def run_stream():
        body = {
            "model": "fixture",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        with _post(base, "/v1/chat/completions", json.dumps(body).encode()) as response:
            first["status"] = response.status
            first["body"] = response.read().decode()

    client = threading.Thread(target=run_stream)
    try:
        client.start()
        assert engine.started.wait(1)
        with _post(base, "/v1/admin/quiesce", b'{"suspend":false}') as response:
            assert json.load(response)["state"] == "draining"
        with pytest.raises(HTTPError) as caught:
            _post(
                base,
                "/v1/chat/completions",
                json.dumps(
                    {
                        "model": "fixture",
                        "messages": [{"role": "user", "content": "new"}],
                    }
                ).encode(),
            )
        assert caught.value.code == 503
        assert caught.value.headers["Retry-After"] == "1"
        engine.finish.set()
        client.join(2)
        assert first["status"] == 200
        assert "hello" in first["body"]
        assert "[DONE]" in first["body"]
    finally:
        engine.finish.set()
        server.shutdown()
        server.server_close()
        thread.join()
        if client.ident is not None:
            client.join(2)


def test_signal_controller_default_and_second_signal_escape_hatch():
    class Server:
        def __init__(self):
            self.stopped = threading.Event()

        def shutdown(self):
            self.stopped.set()

    class Engine:
        def __init__(self):
            self.calls = []
            self.release = threading.Event()

        def quiesce(self, **kwargs):
            self.calls.append(kwargs)

        def wait_for_quiesce(self, _timeout):
            self.release.wait(2)

    immediate_server, immediate_engine = Server(), Engine()
    SignalShutdownController(immediate_server, immediate_engine)()
    assert immediate_server.stopped.wait(1)
    assert immediate_engine.calls == []

    graceful_server, graceful_engine = Server(), Engine()
    SignalShutdownController(graceful_server, graceful_engine, 3.0)()
    deadline = time.monotonic() + 1
    while not graceful_engine.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert graceful_engine.calls == [
        {"drain_timeout_seconds": 3.0, "suspend": False}
    ]
    assert not graceful_server.stopped.is_set()
    graceful_engine.release.set()
    assert graceful_server.stopped.wait(1)

    server, engine = Server(), Engine()
    controller = SignalShutdownController(server, engine, 3.0)
    assert not hasattr(controller, "_lock")
    controller()
    deadline = time.monotonic() + 1
    while not engine.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert engine.calls == [{"drain_timeout_seconds": 3.0, "suspend": False}]
    controller()
    assert server.stopped.wait(1)
    engine.release.set()
