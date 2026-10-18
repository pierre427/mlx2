import base64
import json
import threading
from collections import Counter
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import pytest

from mlx2.runtime.prefill_handoff import (
    OPERATION,
    OPERATION_REVISION,
    RECORD_SCHEMA,
    PrefillHandoffError,
    make_plan,
    plan_digest,
    token_digest,
)
from mlx2.server import (
    PrefillHandoffHandles,
    authorize_admin,
    build_parser,
    validate_prefill_handoff_config,
    validate_request,
)
from mlx2.serving import Job


class _Engine:
    model_path = "fixture"

    def __init__(self, *, qualification_mode=False):
        self.qualification_mode = qualification_mode
        self.lock = threading.Lock()
        self.counts = Counter()

    def status(self):
        return {"healthy": True, "error": None, "model": "fixture"}

    def service_state(self):
        return {"state": "serving"}

    def prefill_handoff_identity(self):
        return {
            "artifact_fingerprint": "artifact",
            "tokenizer_digest": "tokenizer",
            "layout_fingerprint": "layout",
            "runtime_digest": "receiver",
            "execution_digest": "execution",
        }

    def prefill_handoff_plan(self, request):
        plan = make_plan(3, 2)
        return {
            "token_ids": [1, 2, 3],
            "prompt_tokens": 3,
            "plan": plan,
            "plan_digest": plan_digest(plan),
            "token_digest": token_digest([1, 2]),
            **self.prefill_handoff_identity(),
        }


@contextmanager
def _server(engine, **options):
    from mlx2.server import handler_for

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine, **options))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _request(port, path, *, body=b"", token=None, headers=None):
    connection = HTTPConnection("127.0.0.1", port, timeout=10)
    request_headers = dict(headers or {})
    if token is not None:
        request_headers["Authorization"] = f"Bearer {token}"
    connection.request("POST", path, body=body, headers=request_headers)
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    return response.status, json.loads(raw)


def test_flag_parsing_and_defaults():
    parser = build_parser()
    args = parser.parse_args(["--model", "model"])
    assert args.prefill_handoff == "off"
    assert args.prefill_handoff_allow_provisional is False
    args = parser.parse_args(["--model", "model", "--prefill-handoff", "both"])
    assert args.prefill_handoff == "both"
    with pytest.raises(ValueError, match="admin-token-file"):
        validate_prefill_handoff_config("import", None)
    validate_prefill_handoff_config("import", "secret")


def test_public_private_field_boundary():
    body = validate_request(
        {"model": "m", "messages": [{"role": "user", "content": "x"}],
         "prefill_handoff": "a" * 32}
    )
    assert body["prefill_handoff"] == "a" * 32
    with pytest.raises(ValueError, match="unsupported request fields"):
        validate_request({"messages": [], "_mlx2_prefill_handoff": {}})


def test_admin_authorization_required_for_handle_use():
    assert authorize_admin("127.0.0.1", None, "secret")[0] == 401
    assert authorize_admin("127.0.0.1", "Bearer secret", "secret") is None
    assert authorize_admin("192.0.2.1", "Bearer secret", "secret")[0] == 403


def test_handles_are_single_use_bounded_and_expire():
    now = [10.0]
    table = PrefillHandoffHandles(maximum=2, ttl=5, clock=lambda: now[0])
    first = table.put({"id": 1})
    assert table.take(first) == {"id": 1}
    with pytest.raises(LookupError):
        table.take(first)
    old = table.put({"id": 2})
    table.put({"id": 3})
    table.put({"id": 4})
    with pytest.raises(LookupError):
        table.take(old)
    expiring = table.put({"id": 5})
    now[0] = 16.0
    with pytest.raises(LookupError):
        table.take(expiring)


def test_chat_handle_requires_admin_authentication():
    body = json.dumps({
        "model": "fixture",
        "messages": [{"role": "user", "content": "hello"}],
        "prefill_handoff": "a" * 32,
    }).encode()
    with _server(_Engine(), admin_token="secret", prefill_handoff="import") as port:
        status, _ = _request(
            port, "/v1/chat/completions", body=body,
            headers={"Content-Type": "application/json"},
        )
    assert status == 401


def test_models_advertise_only_when_enabled():
    def get(port):
        connection = HTTPConnection("127.0.0.1", port, timeout=10)
        connection.request("GET", "/v1/models")
        response = connection.getresponse()
        value = json.loads(response.read())
        connection.close()
        return value["data"][0]

    with _server(_Engine()) as port:
        assert "prefill_handoff" not in get(port)
    with _server(_Engine(), prefill_handoff="both") as port:
        capability = get(port)["prefill_handoff"]
        assert capability["export"] is True and capability["import"] is True
        assert capability["execution_digest"] == "execution"


def test_loading_model_omits_capability_and_admin_operations_are_not_ready():
    class _LoadingEngine(_Engine):
        def prefill_handoff_identity(self):
            raise PrefillHandoffError("not_ready", "execution identity is not ready")

    engine = _LoadingEngine()
    with _server(
        engine, admin_token="secret", prefill_handoff="both"
    ) as port:
        connection = HTTPConnection("127.0.0.1", port, timeout=10)
        connection.request("GET", "/v1/models")
        response = connection.getresponse()
        model = json.loads(response.read())["data"][0]
        connection.close()
        assert response.status == 200
        assert "prefill_handoff" not in model

        for action in ("plan", "export", "import"):
            status, value = _request(
                port,
                f"/v1/admin/prefill-handoff/{action}",
                body=b"{}",
                token="secret",
                headers={"Content-Type": "application/json"},
            )
            assert status == 503
            assert value["error"]["code"] == "not_ready"


def test_final_streaming_chunk_carries_prefill_handoff_receipt():
    class _StreamingEngine(_Engine):
        def submit(self, request, *, tenant_id="default", admitted=False):
            job = Job(request, tenant_id=tenant_id)
            job.prompt_tokens = 3
            job.completion_tokens = 1
            job.events.put({"text": "hello"})
            job.events.put({
                "finish_reason": "stop",
                "receipt": {
                    "prefill_handoff": {
                        "operation": "prefill-handoff-approx-v1",
                        "observed_used": True,
                    }
                },
            })
            return job

    body = json.dumps({
        "model": "fixture",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }).encode()
    with _server(_StreamingEngine()) as port:
        connection = HTTPConnection("127.0.0.1", port, timeout=10)
        connection.request(
            "POST", "/v1/chat/completions", body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        chunks = [
            json.loads(line.removeprefix("data: "))
            for line in response.read().decode().splitlines()
            if line.startswith("data: {")
        ]
        connection.close()
    final = next(chunk for chunk in chunks if chunk["choices"][0]["finish_reason"])
    assert final["mlx2"]["prefill_handoff"]["observed_used"] is True


@pytest.mark.parametrize(("qualification_mode", "status"), [(False, 409), (True, 200)])
def test_import_record_gate(monkeypatch, tmp_path, qualification_mode, status):
    import mlx2.runtime.prefill_handoff as handoff

    plan = make_plan(3, 2)
    manifest = {
        "producer": {"runtime_digest": "producer"},
        "plan": plan,
        "plan_digest": plan_digest(plan),
        "token_digest": token_digest([1, 2]),
        "token_count": 2,
    }
    monkeypatch.setattr(handoff, "validate_handoff", lambda blob, expected: manifest)
    expected = {
        "artifact_fingerprint": "artifact",
        "tokenizer_digest": "tokenizer",
        "layout_fingerprint": "layout",
    }
    encoded = base64.urlsafe_b64encode(json.dumps(expected).encode()).decode().rstrip("=")
    with _server(
        _Engine(qualification_mode=qualification_mode),
        admin_token="secret",
        prefill_handoff="import",
        prefill_handoff_records=tmp_path,
    ) as port:
        got, value = _request(
            port,
            "/v1/admin/prefill-handoff/import",
            body=b"blob",
            token="secret",
            headers={"X-Mlx2-Handoff-Expected": encoded},
        )
    assert got == status
    if qualification_mode:
        assert value["receipt"]["qualified"] is False
        assert value["receipt"]["reason"] == "candidate_validation"
    else:
        assert value["error"]["code"] == "no_qualification_record"


def _record(*, provisional=False, max_prompt_tokens=8):
    gates = ("G1", "G2", "G3", "G4") if provisional else (
        "G1", "G2", "G3", "G4", "G5"
    )
    return {
        "schema": RECORD_SCHEMA,
        "record_id": "record-1",
        "operation": OPERATION,
        "operation_revision": OPERATION_REVISION,
        "verdict": "qualified",
        "provisional": provisional,
        "artifact_fingerprint": "artifact",
        "producer": {"runtime_digest": "producer"},
        "receiver": {
            "runtime_digest": "receiver",
            "execution_digest": "execution",
        },
        "plan_chunks": [2],
        "max_prompt_tokens": max_prompt_tokens,
        "gates": {gate: {"pass": True} for gate in gates},
    }


def test_plan_returns_authoritative_record_match(tmp_path):
    (tmp_path / "record.json").write_text(json.dumps(_record()))
    body = json.dumps({
        "model": "fixture",
        "messages": [{"role": "user", "content": "hello"}],
        "producer_runtime_digest": "producer",
    }).encode()
    with _server(
        _Engine(),
        admin_token="secret",
        prefill_handoff="import",
        prefill_handoff_records=tmp_path,
    ) as port:
        status, value = _request(
            port,
            "/v1/admin/prefill-handoff/plan",
            body=body,
            token="secret",
            headers={"Content-Type": "application/json"},
        )
    assert status == 200
    assert value["record"] == {
        "record_id": "record-1",
        "record_state": "final",
        "max_prompt_tokens": 8,
    }


def test_plan_preserves_matching_record_ceiling_for_context_refusal(tmp_path):
    (tmp_path / "record.json").write_text(
        json.dumps(_record(max_prompt_tokens=2))
    )
    body = json.dumps({
        "model": "fixture",
        "messages": [{"role": "user", "content": "hello"}],
        "producer_runtime_digest": "producer",
    }).encode()
    with _server(
        _Engine(),
        admin_token="secret",
        prefill_handoff="import",
        prefill_handoff_records=tmp_path,
    ) as port:
        status, value = _request(
            port,
            "/v1/admin/prefill-handoff/plan",
            body=body,
            token="secret",
            headers={"Content-Type": "application/json"},
        )
    assert status == 200
    assert value["plan"]["prompt_tokens"] == 3
    assert value["record"]["max_prompt_tokens"] == 2


@pytest.mark.parametrize(
    ("allow_provisional", "status"),
    [(False, 409), (True, 200)],
)
def test_provisional_record_requires_explicit_flag(
    monkeypatch, tmp_path, allow_provisional, status
):
    import mlx2.runtime.prefill_handoff as handoff

    plan = make_plan(3, 2)
    manifest = {
        "producer": {"runtime_digest": "producer"},
        "plan": plan,
        "plan_digest": plan_digest(plan),
        "token_digest": token_digest([1, 2]),
        "token_count": 2,
    }
    monkeypatch.setattr(handoff, "validate_handoff", lambda blob, expected: manifest)
    (tmp_path / "provisional.json").write_text(json.dumps(_record(provisional=True)))
    expected = {
        "artifact_fingerprint": "artifact",
        "tokenizer_digest": "tokenizer",
        "layout_fingerprint": "layout",
    }
    encoded = base64.urlsafe_b64encode(json.dumps(expected).encode()).decode().rstrip("=")
    with _server(
        _Engine(),
        admin_token="secret",
        prefill_handoff="import",
        prefill_handoff_records=tmp_path,
        prefill_handoff_allow_provisional=allow_provisional,
    ) as port:
        got, value = _request(
            port,
            "/v1/admin/prefill-handoff/import",
            body=b"blob",
            token="secret",
            headers={"X-Mlx2-Handoff-Expected": encoded},
        )
    assert got == status
    if allow_provisional:
        assert value["receipt"]["record_id"] == "record-1"
        assert value["receipt"]["record_state"] == "provisional"
        assert value["receipt"]["qualified"] is False
    else:
        assert value["error"]["code"] == "no_qualification_record"
