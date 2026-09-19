"""HTTP validation and tenant routing for APCv2 session hints."""

import json
import threading
from collections import Counter
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mlx2.runtime.apc_v2 import APCSessionNotFound, APCSessionUnavailable
from mlx2.server import handler_for, validate_request
from mlx2.serving import Job, ServingEngine, persistent_runtime_revision


class SessionEngine:
    model_path = "fixture"
    apc_sessions_enabled = True

    def __init__(self):
        self.values = {("tenant-a", "conversation-1"): {"state": "resident"}}
        self.jobs = []
        self.counts = Counter()

    def status(self):
        return {"healthy": True, "error": None, "model": "fixture"}

    def batching_status(self):
        return {}

    def _value(self, tenant, session):
        try:
            return {
                "tenant": tenant,
                "session_id": session,
                **self.values[(tenant, session)],
            }
        except KeyError as error:
            raise APCSessionNotFound(session) from error

    def apc_session_state(self, tenant, session):
        return self._value(tenant, session)

    def apc_sessions(self, tenant, *, limit, cursor):
        data = [
            self._value(scope, session)
            for scope, session in sorted(self.values)
            if scope == tenant
        ]
        return {"data": data[cursor : cursor + limit], "next_cursor": None}

    def apc_session_park(self, tenant, session, *, ttl_seconds):
        value = self._value(tenant, session)
        self.values[(tenant, session)] = {
            "state": "disk",
            "ttl_seconds": ttl_seconds,
        }
        return {**value, **self.values[(tenant, session)]}

    def apc_session_resume(self, tenant, session, *, ttl_seconds=None):
        value = self._value(tenant, session)
        self.values[(tenant, session)] = {
            "state": "resident",
            "ttl_seconds": ttl_seconds,
        }
        return {**value, **self.values[(tenant, session)]}

    def apc_session_delete(self, tenant, session):
        self._value(tenant, session)
        del self.values[(tenant, session)]
        return {"tenant": tenant, "session_id": session, "removed_entries": 1}

    def submit(self, request, *, tenant_id="default"):
        job = Job(request)
        job.tenant_id = tenant_id
        job.prompt_tokens = 1
        job.completion_tokens = 1
        job.events.put({"text": "ok"})
        job.events.put({"finish_reason": "stop", "receipt": {"cache": "apcv2"}})
        self.jobs.append(job)
        return job


@pytest.fixture
def endpoint():
    engine = SessionEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield engine, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def _request(base, method, path, body=None, tenant="tenant-a", headers=None):
    request_headers = {"X-Tenant-ID": tenant, **(headers or {})}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        request_headers["Content-Type"] = "application/json"
    return urlopen(
        Request(base + path, method=method, data=data, headers=request_headers)
    )


def test_session_control_routes_and_tenant_isolation(endpoint):
    _engine, base = endpoint
    with _request(base, "GET", "/v1/apc/sessions/conversation-1") as response:
        assert json.load(response)["state"] == "resident"
    with _request(
        base,
        "POST",
        "/v1/apc/sessions/conversation-1/park",
        {"ttl_seconds": 60},
    ) as response:
        assert json.load(response)["state"] == "disk"
    with _request(
        base, "POST", "/v1/apc/sessions/conversation-1/resume", {}
    ) as response:
        assert response.status == 202
        assert json.load(response)["state"] == "resident"
    with _request(base, "GET", "/v1/apc/sessions?limit=1&cursor=0") as response:
        assert len(json.load(response)["data"]) == 1
    with pytest.raises(HTTPError) as denied:
        _request(
            base,
            "POST",
            "/v1/apc/sessions/conversation-1/park",
            {"ttl_seconds": 60},
            tenant="tenant-b",
        )
    assert denied.value.code == 404
    with _request(base, "DELETE", "/v1/apc/sessions/conversation-1") as response:
        assert json.load(response)["removed_entries"] == 1


def test_request_session_field_header_validation_and_translation(endpoint):
    engine, base = endpoint
    with _request(
        base,
        "POST",
        "/v1/chat/completions",
        {
            "model": "fixture",
            "messages": [{"role": "user", "content": "hi"}],
            "session_id": "conversation-2",
        },
        headers={"X-mlx2-Session-ID": "conversation-2"},
    ) as response:
        assert response.status == 200
    assert engine.jobs[-1].request["session_id"] == "conversation-2"

    with _request(
        base,
        "POST",
        "/v1/responses",
        {"model": "fixture", "input": "hi", "session_id": "responses-1"},
    ) as response:
        assert response.status == 200
    assert engine.jobs[-1].request["session_id"] == "responses-1"

    with pytest.raises(HTTPError) as conflict:
        _request(
            base,
            "POST",
            "/v1/chat/completions",
            {
                "model": "fixture",
                "messages": [{"role": "user", "content": "hi"}],
                "session_id": "body-value",
            },
            headers={"X-mlx2-Session-ID": "header-value"},
        )
    assert conflict.value.code == 400


@pytest.mark.parametrize("value", ["", " space", "slash/value", "x" * 129, 7])
def test_invalid_session_ids_fail_exact_validation(value):
    with pytest.raises(ValueError, match="session_id"):
        validate_request(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "session_id": value,
            }
        )


def test_session_routes_are_hidden_without_disk_tier():
    engine = SessionEngine()
    engine.apc_sessions_enabled = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(HTTPError) as error:
            _request(
                f"http://127.0.0.1:{server.server_port}",
                "GET",
                "/v1/apc/sessions/conversation-1",
            )
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_session_internal_errors_are_generic_and_closed_resume_is_503(endpoint):
    engine, base = endpoint

    def internal_failure(_tenant, _session):
        raise OSError("/private/secret/apc-persist/payload")

    engine.apc_session_state = internal_failure
    with pytest.raises(HTTPError) as failure:
        _request(base, "GET", "/v1/apc/sessions/conversation-1")
    assert failure.value.code == 500
    assert b"private/secret" not in failure.value.read()

    def closed_resume(_tenant, _session, *, ttl_seconds=None):
        raise APCSessionUnavailable("APCv2 session service is closed")

    engine.apc_session_resume = closed_resume
    with pytest.raises(HTTPError) as failure:
        _request(base, "POST", "/v1/apc/sessions/conversation-1/resume", {})
    assert failure.value.code == 503
    assert b"session service is closed" in failure.value.read()


def test_session_control_scope_is_tenant_owned_even_when_cache_is_shared():
    class SharedEngine:
        tenant_scoped_cache = False

    assert ServingEngine._session_scope(SharedEngine(), "tenant-a") == "tenant-a"
    assert ServingEngine._session_scope(SharedEngine(), "tenant-b") == "tenant-b"


def test_persistent_runtime_revision_binds_native_and_mlx_identity():
    identity = {
        "source_sha256": "source",
        "mlx_native_sha256": "native-a",
        "mlx": "1.0",
        "python": "3.12",
        "macos": "15",
        "transformers": "1",
        "dependencies": {"numpy": "2"},
    }
    baseline = persistent_runtime_revision(identity)
    assert baseline != persistent_runtime_revision(
        {**identity, "mlx_native_sha256": "native-b"}
    )
    assert baseline != persistent_runtime_revision({**identity, "mlx": "2.0"})
    assert baseline != persistent_runtime_revision(
        {**identity, "dependencies": {"numpy": "3"}}
    )
