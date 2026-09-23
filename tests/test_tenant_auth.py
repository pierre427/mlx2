"""Authenticated tenant identity: verification, fail-closed HTTP, consumers."""

import json
import os
import re
import threading
from collections import Counter
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mlx2 import tenant_auth
from mlx2.api_resources import FileStore, ResponseStore
from mlx2.reasoning_signatures import ReasoningSigner
from mlx2.runtime.apc_v2 import APCSessionNotFound
from mlx2.server import build_parser, build_tenant_authenticator, handler_for
from mlx2.serving import Job
from mlx2.tenant_auth import (
    TenantAuthError,
    TenantAuthenticator,
    hash_api_key,
    load_keys_file,
    mint_token,
)

SERVER_SOURCE = Path(__file__).resolve().parents[1] / "src/mlx2/server.py"
SECRET = b"s" * 40
KEY_A = "mlx2k_" + "a" * 43
KEY_B = "mlx2k_" + "b" * 43
KEY_ADMIN_LORA = "mlx2k_" + "c" * 43
KEY_OFF = "mlx2k_" + "d" * 43


def _write_keys(tmp_path, entries=None):
    entries = entries or [
        {"key_id": "a1", "tenant": "tenant-a", "sha256": hash_api_key(KEY_A)},
        {"key_id": "b1", "tenant": "tenant-b", "sha256": hash_api_key(KEY_B)},
        {
            "key_id": "ops",
            "tenant": "ops",
            "sha256": hash_api_key(KEY_ADMIN_LORA),
            "scopes": ["inference", "adapters"],
            "adapters": ["ops-lora"],
        },
        {
            "key_id": "old",
            "tenant": "tenant-a",
            "sha256": hash_api_key(KEY_OFF),
            "disabled": True,
        },
    ]
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({"version": 1, "keys": entries}))
    os.chmod(path, 0o600)
    return path


def _auth(tmp_path, **kwargs):
    return TenantAuthenticator(
        keys=load_keys_file(_write_keys(tmp_path)), token_secret=SECRET, **kwargs
    )


# -- unit ---------------------------------------------------------------------


def test_api_keys_resolve_tenant_and_fail_closed(tmp_path):
    auth = _auth(tmp_path)
    assert auth.authenticate({"Authorization": f"Bearer {KEY_A}"}).tenant == "tenant-a"
    principal = auth.authenticate({"x-api-key": KEY_B})
    assert (principal.tenant, principal.method, principal.key_id) == (
        "tenant-b",
        "api_key",
        "b1",
    )
    assert auth.authenticate(
        {"Authorization": f"Bearer {KEY_ADMIN_LORA}"}, required_scope="adapters"
    ).extra == {"adapters": ("ops-lora",)}
    cases = [
        ({}, "missing"),
        ({"Authorization": "Basic abc"}, "malformed"),
        ({"Authorization": "Bearer "}, "malformed"),
        ({"x-api-key": " "}, "malformed"),
        ({"Authorization": "Bearer " + "x" * 5000}, "malformed"),
        ({"Authorization": f"Bearer {KEY_A}", "x-api-key": KEY_B}, "ambiguous"),
        ({"Authorization": "Bearer mlx2k_nope"}, "unknown_key"),
        ({"x-api-key": KEY_OFF}, "disabled"),
        ({"x-api-key": KEY_A, "X-Tenant-ID": "tenant-b"}, "tenant_mismatch"),
    ]
    for headers, reason in cases:
        with pytest.raises(TenantAuthError) as caught:
            auth.authenticate(headers)
        assert caught.value.reason == reason
    with pytest.raises(TenantAuthError) as caught:
        auth.authenticate({"x-api-key": KEY_A}, required_scope="adapters")
    assert (caught.value.reason, caught.value.status) == ("scope", 403)
    # Same credential in both headers is fine (some SDKs send both).
    assert auth.authenticate({"Authorization": f"Bearer {KEY_A}", "x-api-key": KEY_A})
    status = auth.status()
    assert status["verified"]["api_key"] == 4
    assert status["failures"]["tenant_mismatch"] == 1
    assert status["failures"]["malformed"] == 4
    assert "tenant-a" not in json.dumps(status)


def test_tokens_verify_mac_before_parsing_and_enforce_expiry(tmp_path):
    now = 1_800_000_000
    auth = TenantAuthenticator(token_secret=SECRET, clock=lambda: now)
    token = mint_token(SECRET, "tenant-t", ttl_seconds=3600, now=now - 10)
    principal = auth.authenticate({"Authorization": f"Bearer {token}"})
    assert (principal.tenant, principal.method) == ("tenant-t", "token")

    prefix, payload, mac = token.split(".")
    other = mint_token(SECRET, "tenant-u", ttl_seconds=3600, now=now)
    forged = ".".join([prefix, other.split(".")[1], mac])  # payload swap
    bad = {
        "bad_signature": [
            mint_token(b"t" * 40, "tenant-t", ttl_seconds=60, now=now),
            forged,
            f"{prefix}.{payload}.{mac[:-2]}AA",
        ],
        "malformed": [f"{prefix}.{payload}", f"{prefix}.{payload}.!!", f"{prefix}.a.b.c"],
        "expired": [
            mint_token(SECRET, "tenant-t", ttl_seconds=60, now=now - 120),
            mint_token(SECRET, "tenant-t", ttl_seconds=30 * 86400, now=now),
            mint_token(SECRET, "tenant-t", ttl_seconds=60, now=now + 3600),
        ],
    }
    for reason, tokens in bad.items():
        for candidate in tokens:
            with pytest.raises(TenantAuthError) as caught:
                auth.authenticate({"x-api-key": candidate})
            assert caught.value.reason == reason, candidate
    # An API key is not accepted when only tokens are configured.
    with pytest.raises(TenantAuthError) as caught:
        auth.authenticate({"x-api-key": KEY_A})
    assert caught.value.reason == "unknown_key"


def test_keys_file_validation_is_strict(tmp_path):
    good = {"key_id": "k", "tenant": "t", "sha256": "0" * 64}
    for entries, message in [
        ([{**good, "tenant": "bad tenant"}], "tenant ids"),
        ([good, {**good, "sha256": "1" * 64}], "duplicate tenant key_id"),
        ([good, {**good, "key_id": "k2"}], "duplicates another"),
        ([{**good, "surprise": 1}], "unknown fields"),
        ([{**good, "scopes": ["root"]}], "unknown scopes"),
        ([{**good, "sha256": "ABC"}], "sha256"),
    ]:
        with pytest.raises(ValueError, match=message):
            load_keys_file(_write_keys(tmp_path, entries))
    path = _write_keys(tmp_path, [good])
    os.chmod(path, 0o666)
    with pytest.raises(ValueError, match="world-writable"):
        load_keys_file(path)
    secret = tmp_path / "secret"
    secret.write_bytes(SECRET)
    os.chmod(secret, 0o644)
    with pytest.raises(ValueError, match="0600"):
        tenant_auth.load_token_secret(path=secret)
    os.chmod(secret, 0o600)
    assert tenant_auth.load_token_secret(path=secret) == SECRET
    secret.write_bytes(b"short")
    with pytest.raises(ValueError, match="32 bytes"):
        tenant_auth.load_token_secret(path=secret)


def test_new_key_cli_prints_matching_digest(capsys):
    assert tenant_auth._main(["new-key", "--tenant", "team", "--key-id", "k"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["api_key"].startswith("mlx2k_")
    assert printed["keys_file_entry"]["sha256"] == hash_api_key(printed["api_key"])


# -- CLI wiring ------------------------------------------------------------------


def _args(*extra):
    return build_parser().parse_args(["--model", "m", *extra])


def test_cli_auth_off_by_default_and_requires_cache_isolation(tmp_path, monkeypatch):
    assert build_tenant_authenticator(_args()) is None
    keys = str(_write_keys(tmp_path))
    with pytest.raises(ValueError, match="--tenant-scoped-cache"):
        build_tenant_authenticator(_args("--tenant-auth-keys-file", keys))
    auth = build_tenant_authenticator(
        _args("--tenant-auth-keys-file", keys, "--tenant-scoped-cache")
    )
    assert auth.status()["cache_isolation"] == "tenant"
    shared = build_tenant_authenticator(
        _args(
            "--tenant-auth-keys-file",
            keys,
            "--tenant-auth-allow-shared-cache",
            "--tenant-header-policy",
            "ignore",
        )
    )
    assert (shared.status()["cache_isolation"], shared.header_policy) == (
        "shared",
        "ignore",
    )
    monkeypatch.setenv("MLX2_TEST_TENANT_SECRET", SECRET.decode())
    token_auth = build_tenant_authenticator(
        _args(
            "--tenant-auth-token-secret-env",
            "MLX2_TEST_TENANT_SECRET",
            "--tenant-scoped-cache",
        )
    )
    assert token_auth.methods == ["token"]
    assert token_auth.uses_secret(SECRET)
    assert not token_auth.uses_secret(b"x" * 40)
    with pytest.raises(ValueError, match="requires tenant auth"):
        build_tenant_authenticator(_args("--tenant-auth-allow-shared-cache"))


def test_no_raw_tenant_header_reads_outside_the_resolver():
    reads = re.findall(r'headers\.get\("X-Tenant-ID"', SERVER_SOURCE.read_text())
    # Exactly one: the auth-off branch of Handler._authenticate_tenant.
    assert len(reads) == 1


# -- HTTP ---------------------------------------------------------------------


class TenantEngine:
    model_path = "fixture"
    apc_sessions_enabled = True

    def __init__(self):
        self.counts = Counter()
        self.reasoning_signer = ReasoningSigner(b"reasoning-secret-distinct")
        self.submitted = []
        self.requests = []
        self.submitted_many = []
        self.session_calls = []
        self.lora_calls = []

    def status(self):
        return {"healthy": True, "error": None, "model": "fixture"}

    def batching_status(self):
        return {}

    def _job(self, request, tenant_id):
        job = Job(request)
        job.tenant_id = tenant_id
        job.prompt_tokens = 1
        job.completion_tokens = 1
        job.events.put({"delta": {"reasoning_content": "private thought"}})
        job.events.put({"delta": {"content": "ok"}})
        job.events.put({"finish_reason": "stop", "receipt": {"cache": "apcv2"}})
        return job

    def submit(self, request, *, tenant_id="default", **_kwargs):
        self.submitted.append(tenant_id)
        self.requests.append(request)
        return self._job(request, tenant_id)

    def admit_parallel_samples(self, count):
        return {"samples": count}

    def submit_many(self, requests, *, tenant_id="default", **_kwargs):
        self.submitted_many.append(tenant_id)
        return [self._job(request, tenant_id) for request in requests]

    def _session(self, op, tenant, session_id=None, **_kwargs):
        self.session_calls.append((op, tenant, session_id))
        if session_id is not None and tenant != "tenant-a":
            raise APCSessionNotFound(session_id)
        return {"tenant": tenant, "session_id": session_id, "state": "resident"}

    def apc_session_state(self, tenant, session_id):
        return self._session("state", tenant, session_id)

    def apc_sessions(self, tenant, *, limit, cursor):
        self.session_calls.append(("list", tenant, None))
        return {"data": [], "next_cursor": None}

    def apc_session_park(self, tenant, session_id, *, ttl_seconds):
        return self._session("park", tenant, session_id)

    def apc_session_resume(self, tenant, session_id, *, ttl_seconds=None):
        return self._session("resume", tenant, session_id)

    def apc_session_delete(self, tenant, session_id):
        return self._session("delete", tenant, session_id)

    def load_lora_adapter(self, name, path, *, base_model_name=None):
        self.lora_calls.append(name)
        return {"loaded": True}


@pytest.fixture
def served(tmp_path):
    servers = []

    def start(authenticator, engine=None, **kwargs):
        engine = engine or TenantEngine()
        response_store = ResponseStore()
        file_store = FileStore()
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            handler_for(
                engine,
                response_store=response_store,
                file_store=file_store,
                tenant_authenticator=authenticator,
                **kwargs,
            ),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return SimpleNamespace(
            engine=engine,
            responses=response_store,
            files=file_store,
            base=f"http://127.0.0.1:{server.server_port}",
        )

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join()


def _call(base, method, path, body=None, headers=None):
    data = None
    request_headers = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode()
        request_headers["Content-Type"] = "application/json"
    try:
        with urlopen(
            Request(base + path, method=method, data=data, headers=request_headers)
        ) as response:
            raw = response.read()
            return response.status, dict(response.headers), raw
    except HTTPError as error:
        return error.code, dict(error.headers), error.read()


def _json(raw):
    return json.loads(raw)


CHAT = {"model": "fixture", "messages": [{"role": "user", "content": "hi"}]}


def test_every_tenant_consumer_uses_the_authenticated_tenant(served, tmp_path):
    auth = _auth(tmp_path, header_policy="ignore")
    app = served(auth)
    a = {"Authorization": f"Bearer {KEY_A}", "X-Tenant-ID": "tenant-b"}  # spoof
    b = {"x-api-key": KEY_B}

    # Generation: chat, completions, n>1, Anthropic messages.
    status, headers, _ = _call(app.base, "POST", "/v1/chat/completions", CHAT, a)
    assert status == 200 and headers["X-MLX2-Tenant-Auth"] == "api_key"
    assert _call(app.base, "POST", "/v1/chat/completions", {**CHAT, "n": 2}, a)[0] == 200
    assert _call(
        app.base,
        "POST",
        "/v1/completions",
        {"model": "fixture", "prompt": "hi", "max_tokens": 1},
        a,
    )[0] == 200
    status, _, raw = _call(
        app.base,
        "POST",
        "/v1/messages",
        {"model": "fixture", "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}]},
        a,
    )
    assert status == 200, raw
    # Responses: create, retrieve, cross-tenant retrieve, delete.
    status, _, raw = _call(
        app.base,
        "POST",
        "/v1/responses",
        {"model": "fixture", "input": "hi", "include": ["reasoning.encrypted_content"]},
        a,
    )
    assert status == 200, raw
    created = _json(raw)
    assert app.responses.get("tenant-a", created["id"])["id"] == created["id"]
    assert _call(app.base, "GET", f"/v1/responses/{created['id']}", headers=a)[0] == 200
    assert _call(app.base, "GET", f"/v1/responses/{created['id']}", headers=b)[0] == 404
    status, _, raw = _call(
        app.base,
        "POST",
        "/v1/responses",
        {"model": "fixture", "input": "again", "previous_response_id": created["id"]},
        b,
    )
    assert status == 404, raw  # B cannot chain onto A's stored response
    assert _call(app.base, "DELETE", f"/v1/responses/{created['id']}", headers=b)[0] == 404
    assert _call(app.base, "DELETE", f"/v1/responses/{created['id']}", headers=a)[0] == 200

    # APC sessions: every op is scoped by the credential's tenant.
    for method, path, body in [
        ("GET", "/v1/apc/sessions/conv-1", None),
        ("GET", "/v1/apc/sessions", None),
        ("POST", "/v1/apc/sessions/conv-1/park", {"ttl_seconds": 60}),
        ("POST", "/v1/apc/sessions/conv-1/resume", {}),
        ("DELETE", "/v1/apc/sessions/conv-1", None),
    ]:
        assert _call(app.base, method, path, body, a)[0] in {200, 202}
        if path != "/v1/apc/sessions":
            # Session hijack attempt from B: indistinguishable from "absent".
            assert _call(app.base, method, path, body, b)[0] == 404

    # Files and batches listing are tenant-scoped too.
    app.files.create(
        "tenant-a",
        filename="a.txt",
        purpose="user_data",
        content_type="text/plain",
        content=b"a",
    )
    assert len(_json(_call(app.base, "GET", "/v1/files", headers=a)[2])["data"]) == 1
    assert _json(_call(app.base, "GET", "/v1/files", headers=b)[2])["data"] == []
    assert _call(app.base, "GET", "/v1/batches", headers=a)[0] == 200

    engine = app.engine
    assert set(engine.submitted) == {"tenant-a"} and len(engine.submitted) == 4
    assert engine.submitted_many == ["tenant-a"]
    a_calls = [call for call in engine.session_calls if call[1] == "tenant-a"]
    b_calls = [call for call in engine.session_calls if call[1] == "tenant-b"]
    assert len(a_calls) == 5 and len(b_calls) == 4
    status = auth.status()
    assert status["verified"]["api_key"] > 0  # mechanism counter moved
    assert status["tenant_header_ignored"] > 0
    assert sum(status["failures"].values()) == 0


def test_spoofed_header_is_refused_under_must_match(served, tmp_path):
    auth = _auth(tmp_path)
    app = served(auth)
    status, headers, raw = _call(
        app.base,
        "POST",
        "/v1/chat/completions",
        CHAT,
        {"x-api-key": KEY_B, "X-Tenant-ID": "tenant-a"},
    )
    assert status == 403 and "X-MLX2-Tenant-Auth" not in headers
    assert KEY_B not in raw.decode() and "tenant-a" not in raw.decode()
    status, _, raw = _call(
        app.base,
        "POST",
        "/v1/messages",
        {"model": "fixture", "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}]},
        {"x-api-key": KEY_B, "X-Tenant-ID": "tenant-a"},
    )
    assert status == 403 and _json(raw)["type"] == "error"  # Anthropic envelope
    assert app.engine.submitted == []
    # A matching header is accepted.
    assert _call(
        app.base,
        "POST",
        "/v1/chat/completions",
        CHAT,
        {"x-api-key": KEY_A, "X-Tenant-ID": "tenant-a"},
    )[0] == 200
    assert auth.status()["tenant_header_matched"] == 1
    assert auth.status()["failures"]["tenant_mismatch"] == 2


def test_unauthenticated_requests_fail_closed_everywhere(served, tmp_path):
    auth = _auth(tmp_path)
    app = served(auth, admin_token="admin-secret")
    guarded = [
        ("GET", "/v1/models"),
        ("GET", "/v1/status"),
        ("GET", "/v1/status/batching"),
        ("GET", "/v1/files"),
        ("GET", "/v1/batches"),
        ("GET", "/v1/responses/resp_x"),
        ("GET", "/v1/apc/sessions"),
        ("GET", "/no/such/path"),  # authenticate before 404
        ("DELETE", "/v1/files/file_x"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/responses"),
        ("POST", "/v1/messages"),
        ("POST", "/v1/embeddings"),
        ("POST", "/no/such/path"),
    ]
    for method, path in guarded:
        body = CHAT if method == "POST" else None
        status, headers, raw = _call(app.base, method, path, body)
        assert status == 401, (method, path)
        assert headers["WWW-Authenticate"] == "Bearer"
        spoofed = _call(app.base, method, path, body, {"X-Tenant-ID": "tenant-a"})
        assert spoofed[0] == 401, (method, path)
    assert app.engine.submitted == [] and app.engine.session_calls == []
    # Open allowlist: health, loopback metrics.
    assert _call(app.base, "GET", "/health")[0] == 200
    status, _, metrics = _call(app.base, "GET", "/metrics")
    assert status == 200
    text = metrics.decode()
    assert 'mlx2_tenant_auth_failures_total{reason="missing"} ' in text
    assert "mlx2_tenant_auth_enabled 1" in text
    for secret in (KEY_A, KEY_B, hash_api_key(KEY_A), "tenant-a", "tenant-b"):
        assert secret not in text
    # Admin keeps its own credential; a tenant key is not an admin token.
    assert _call(
        app.base, "GET", "/v1/admin/state", headers={"Authorization": f"Bearer {KEY_A}"}
    )[0] == 401
    # Authenticated status exposes the auth block without tenant names.
    status, _, raw = _call(app.base, "GET", "/v1/status", headers={"x-api-key": KEY_A})
    block = _json(raw)["tenant_auth"]
    assert block["enabled"] is True and block["failures"]["missing"] >= len(guarded)
    assert "tenant-a" not in json.dumps(block)


def test_lora_lifecycle_requires_adapters_scope(served, tmp_path):
    auth = _auth(tmp_path)
    app = served(auth)
    body = {"lora_name": "x", "lora_path": "/nonexistent"}
    status, _, _ = _call(
        app.base, "POST", "/v1/load_lora_adapter", body, {"x-api-key": KEY_A}
    )
    assert status == 403
    assert auth.status()["failures"]["scope"] == 1
    assert app.engine.lora_calls == []
    status, _, _ = _call(
        app.base, "POST", "/v1/load_lora_adapter", body, {"x-api-key": KEY_ADMIN_LORA}
    )
    assert status == 200
    assert app.engine.lora_calls == ["x"]


def test_token_credentials_carry_tenant_and_receipt(served, tmp_path):
    auth = TenantAuthenticator(token_secret=SECRET)
    app = served(auth)
    token = mint_token(SECRET, "tenant-t", ttl_seconds=600)
    status, headers, _ = _call(
        app.base, "POST", "/v1/chat/completions", CHAT, {"Authorization": f"Bearer {token}"}
    )
    assert status == 200 and headers["X-MLX2-Tenant-Auth"] == "token"
    assert app.engine.submitted == ["tenant-t"]
    forged = mint_token(b"f" * 40, "tenant-t", ttl_seconds=600)
    assert _call(
        app.base, "POST", "/v1/chat/completions", CHAT, {"x-api-key": forged}
    )[0] == 401
    assert auth.status()["verified"]["token"] == 1
    assert auth.status()["failures"]["bad_signature"] == 1


def test_reasoning_replay_across_tenants_is_rejected(served, tmp_path):
    auth = _auth(tmp_path, header_policy="ignore")
    app = served(auth)
    signer = app.engine.reasoning_signer
    token = signer.sign_responses(model="fixture", tenant="tenant-a", text="secret plan")
    replay = {
        "model": "fixture",
        "input": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "secret plan"}],
                "encrypted_content": token,
            },
            {"role": "user", "content": "continue"},
        ],
    }
    # B relabels itself as A through the header: ignored, verification uses B.
    status, _, raw = _call(
        app.base,
        "POST",
        "/v1/responses",
        replay,
        {"x-api-key": KEY_B, "X-Tenant-ID": "tenant-a"},
    )
    assert status == 200, raw
    assert app.engine.counts["reasoning_signature_rejections"] == 1
    assert not _carries_reasoning(app.engine.requests[-1], "secret plan")
    # The legitimate owner replays cleanly and the reasoning is restored.
    status, _, raw = _call(app.base, "POST", "/v1/responses", replay, {"x-api-key": KEY_A})
    assert status == 200, raw
    assert app.engine.counts["reasoning_signature_rejections"] == 1
    assert _carries_reasoning(app.engine.requests[-1], "secret plan")


def _carries_reasoning(request, text):
    return any(
        message.get("reasoning_content") == text for message in request["messages"]
    )


def test_auth_off_preserves_header_routing(served):
    app = served(None)
    status, headers, _ = _call(
        app.base, "POST", "/v1/chat/completions", CHAT, {"X-Tenant-ID": "tenant-z"}
    )
    assert status == 200 and "X-MLX2-Tenant-Auth" not in headers
    assert _call(app.base, "POST", "/v1/chat/completions", CHAT)[0] == 200
    assert app.engine.submitted == ["tenant-z", "default"]
    status, _, raw = _call(app.base, "GET", "/v1/status")
    assert _json(raw)["tenant_auth"] == {"enabled": False}


def test_status_receipts_are_scoped_to_the_authenticated_tenant(served, tmp_path):
    from mlx2.serving import ServingEngine

    class ReceiptEngine(TenantEngine):
        # The real engine's receipt log and filter, fed by submissions.
        recent_receipts = ServingEngine.recent_receipts

        def __init__(self):
            super().__init__()
            self.receipt_log = []

        def submit(self, request, *, tenant_id="default", **kwargs):
            job = super().submit(request, tenant_id=tenant_id, **kwargs)
            self.receipt_log.append(
                (tenant_id, {"request_id": job.id, "session_id": request.get("session_id")})
            )
            return job

        def status(self):
            return {**super().status(), "recent_receipts": self.recent_receipts()}

    app = served(_auth(tmp_path, header_policy="ignore"), engine=ReceiptEngine())
    a = {"Authorization": f"Bearer {KEY_A}"}
    b = {"Authorization": f"Bearer {KEY_B}"}
    private = {**CHAT, "session_id": "alice-private-thread"}
    assert _call(app.base, "POST", "/v1/chat/completions", private, a)[0] == 200
    assert _call(app.base, "POST", "/v1/chat/completions", CHAT, b)[0] == 200

    status, _, raw = _call(app.base, "GET", "/v1/status", headers=b)
    assert status == 200
    receipts = _json(raw)["recent_receipts"]
    assert [receipt["session_id"] for receipt in receipts] == [None]
    status, _, raw = _call(app.base, "GET", "/v1/status", headers=a)
    assert [r["session_id"] for r in _json(raw)["recent_receipts"]] == [
        "alice-private-thread"
    ]


def test_status_without_a_tenant_receipt_filter_fails_closed(served, tmp_path):
    class LeakyEngine(TenantEngine):
        def status(self):
            return {**super().status(), "recent_receipts": [{"session_id": "x"}]}

    app = served(_auth(tmp_path, header_policy="ignore"), engine=LeakyEngine())
    status, _, raw = _call(
        app.base, "GET", "/v1/status", headers={"Authorization": f"Bearer {KEY_B}"}
    )
    assert status == 200 and _json(raw)["recent_receipts"] == []
    # Without tenant auth there is no tenant boundary to enforce.
    open_app = served(None, engine=LeakyEngine())
    status, _, raw = _call(open_app.base, "GET", "/v1/status")
    assert _json(raw)["recent_receipts"] == [{"session_id": "x"}]


def test_batching_status_is_scoped_to_the_authenticated_tenant(served, tmp_path):
    # /v1/status/batching listed every tenant's request ids, tenant ids,
    # events and per-tenant token rates to any authenticated tenant.
    from mlx2.batch_metrics import BatchRuntimeMetrics
    from mlx2.serving import ServingEngine

    class BatchingEngine(TenantEngine):
        # The real engine's batching view over real runtime metrics.
        batching_status = ServingEngine.batching_status

        def __init__(self):
            super().__init__()
            ticks = iter(range(10_000))
            self.batch_metrics = BatchRuntimeMetrics(clock=lambda: float(next(ticks)))
            self.lock = threading.Lock()
            self.snapshot = {}
            self.queued_jobs = 0

        def submit(self, request, *, tenant_id="default", **kwargs):
            job = super().submit(request, tenant_id=tenant_id, **kwargs)
            self.batch_metrics.admitted(job.id, tenant_id, 0)
            self.batch_metrics.dequeued(job.id, 0)
            self.batch_metrics.token(job.id)
            self.batch_metrics.terminal(job.id, "completed", "stop")
            return job

    a = {"Authorization": f"Bearer {KEY_A}"}
    b = {"Authorization": f"Bearer {KEY_B}"}

    def batching(app, headers=None):
        status, _, raw = _call(app.base, "GET", "/v1/status/batching", headers=headers)
        assert status == 200
        return _json(raw), raw.decode()

    app = served(_auth(tmp_path, header_policy="ignore"), engine=BatchingEngine())
    assert _call(app.base, "POST", "/v1/chat/completions", CHAT, a)[0] == 200
    assert _call(app.base, "POST", "/v1/chat/completions", CHAT, b)[0] == 200
    app.engine.batch_metrics.admitted("alice-inflight", "tenant-a", 0)
    alice_done, bob_done = (
        row["request_id"]
        for row in app.engine.batching_status()["completed_requests"]
    )

    view, raw = batching(app, b)
    assert [row["request_id"] for row in view["completed_requests"]] == [bob_done]
    assert view["active_requests"] == []
    assert view["events"] and {e["request_id"] for e in view["events"]} == {bob_done}
    assert list(view["fairness"]["tenant_token_rates"]) == ["tenant-b"]
    for private in ("tenant-a", alice_done, "alice-inflight"):
        assert private not in raw
    # Aggregates stay whole: two tenants' rates, two requests in flight or not.
    assert view["fairness"]["jain_tenant_token_rate"] is not None
    assert view["gauges"]["inflight_requests"] == 1
    assert view["counters"]["admitted"] == 3

    view, _ = batching(app, a)
    assert [row["request_id"] for row in view["completed_requests"]] == [alice_done]
    assert view["active_requests"] == ["alice-inflight"]
    assert list(view["fairness"]["tenant_token_rates"]) == ["tenant-a"]

    # An engine that cannot scope its snapshot shows no per-request rows.
    class LeakyEngine(TenantEngine):
        def batching_status(self):
            return {
                "active_requests": ["x"],
                "completed_requests": [{"request_id": "x", "tenant_id": "tenant-a"}],
                "events": [{"kind": "admitted", "request_id": "x"}],
                "fairness": {"jain_tenant_token_rate": 1.0, "tenant_token_rates": {"tenant-a": 1.0}},
                "counters": {"admitted": 1},
            }

    app = served(_auth(tmp_path, header_policy="ignore"), engine=LeakyEngine())
    view, _ = batching(app, b)
    assert view["active_requests"] == view["completed_requests"] == view["events"] == []
    assert view["fairness"] == {"jain_tenant_token_rate": 1.0, "tenant_token_rates": {}}
    assert view["counters"] == {"admitted": 1}
    # Without tenant auth there is no tenant boundary: the view is unchanged.
    open_app = served(None, engine=LeakyEngine())
    view, _ = batching(open_app)
    assert view == LeakyEngine().batching_status()


def test_serving_engine_recent_receipts_filter_by_owner():
    from mlx2.serving import ServingEngine

    engine = object.__new__(ServingEngine)
    engine.receipt_log = [("tenant-a", {"request_id": 1}), ("tenant-b", {"request_id": 2})]
    assert engine.recent_receipts() == [{"request_id": 1}, {"request_id": 2}]
    assert engine.recent_receipts(tenant_id="tenant-b") == [{"request_id": 2}]
    assert engine.recent_receipts(tenant_id="tenant-c") == []


def test_gate_api_key_and_tenant_auth_are_refused_together(tmp_path):
    from mlx2.http_security import policy_for_bind

    keys = str(_write_keys(tmp_path))
    # Both read Authorization/x-api-key, so together they refuse every
    # request that is not both the gate key and a tenant key at once.
    for gate in (["--api-key-env", "MLX2_API_KEY"], ["--api-key-file", "k"]):
        with pytest.raises(ValueError, match="cannot be combined"):
            build_tenant_authenticator(
                _args("--tenant-auth-keys-file", keys, "--tenant-scoped-cache", *gate)
            )
    with pytest.raises(ValueError, match="cannot be combined"):
        handler_for(
            TenantEngine(),
            http_security=policy_for_bind("127.0.0.1", api_key="gate-key-123"),
            tenant_authenticator=_auth(tmp_path),
        )
    # Each alone is still accepted.
    handler_for(
        TenantEngine(),
        http_security=policy_for_bind("127.0.0.1"),
        tenant_authenticator=_auth(tmp_path),
    )
    assert build_tenant_authenticator(_args("--api-key-env", "MLX2_API_KEY")) is None
