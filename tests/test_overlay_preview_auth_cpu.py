"""POST /v1/overlays/preview authentication over live HTTP (CPU only).

Morning job 2026-11-06 coverage gap, plus Pierre's L1/L2 decisions:
the presented bearer (not the body) decides operator vs tenant before the
body is read, and an operator preview runs as the reserved ``_operator``
tenant rather than "default".
"""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from test_server_quiesce import AdminEngine

import mlx2.server as server_module
from mlx2.server import handler_for
from mlx2.tenant_auth import TenantAuthenticator, TenantPrincipal, hash_api_key
from mlx2.usage import OPERATOR_TENANT

TENANT_KEY = "mlx2k_alice_fixture"
ADMIN = "private-admin"
NAMED = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4,
         "mlx2_overlay": {"id": "ovl-1"}}
AD_HOC = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4,
          "mlx2_overlay": {"parameters": {"strength": 1.0}}}


@pytest.fixture
def serve(monkeypatch):
    calls = []

    def fake_preview(engine, body, *, tenant_id, ad_hoc_allowed=False, **_):
        calls.append({"tenant": tenant_id, "ad_hoc_allowed": ad_hoc_allowed,
                      "selection": body.get("mlx2_overlay")})
        return {"text": "ok", "tokens": 1}

    monkeypatch.setattr(server_module, "overlay_preview", fake_preview)
    servers = []

    def start(*, admin_token=ADMIN, tenant_auth=True):
        engine = AdminEngine()
        engine.ablation_overlays = object()
        authenticator = TenantAuthenticator(
            keys={hash_api_key(TENANT_KEY): (TenantPrincipal("alice", "api_key"), False)}
        ) if tenant_auth else None
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(
            engine, admin_token=admin_token, tenant_authenticator=authenticator))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        base = "http://127.0.0.1:" + str(server.server_port)

        def post(body, token=None, raw=None, headers=()):
            data = raw if raw is not None else json.dumps(body).encode()
            request = Request(base + "/v1/overlays/preview", data=data, method="POST",
                              headers={"Content-Type": "application/json", **dict(headers),
                                       **({"Authorization": "Bearer " + token} if token else {})})
            try:
                with urlopen(request, timeout=10) as response:
                    return response.status, json.loads(response.read())
            except HTTPError as error:
                return error.code, None

        return post

    yield start, calls
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_named_preview_under_tenant_auth_needs_a_tenant_key(serve):
    start, calls = serve
    post = start()
    assert post(NAMED)[0] == 401
    assert post(NAMED, token="mlx2k_wrong")[0] == 401
    assert calls == []
    status, payload = post(NAMED, token=TENANT_KEY)
    assert (status, payload["text"]) == (200, "ok")
    assert calls == [{"tenant": "alice", "ad_hoc_allowed": False, "selection": {"id": "ovl-1"}}]


def test_unauthenticated_preview_is_refused_before_the_body_is_parsed(serve):
    # L1: malformed JSON from an unauthenticated client is 401, not 400.
    start, calls = serve
    post = start()
    assert post(None, raw=b"{not json")[0] == 401
    assert post(None, raw=b"{not json", token=TENANT_KEY)[0] == 400
    assert calls == []


def test_ad_hoc_admin_preview_works_with_tenant_auth_on(serve):
    start, calls = serve
    post = start()
    status, _ = post(AD_HOC, token=ADMIN)
    assert status == 200
    # L2: operator previews never run as a real tenant (or as "default").
    assert calls == [{"tenant": OPERATOR_TENANT, "ad_hoc_allowed": True,
                      "selection": AD_HOC["mlx2_overlay"]}]
    assert post(NAMED, token=ADMIN)[0] == 200
    assert calls[-1]["tenant"] == OPERATOR_TENANT


def test_ad_hoc_preview_with_a_tenant_key_is_refused(serve):
    start, calls = serve
    post = start()
    assert post(AD_HOC, token=TENANT_KEY)[0] == 401  # loopback: bad admin bearer
    assert post(AD_HOC)[0] == 401
    assert calls == []


def test_non_loopback_admin_bearer_is_not_an_operator(serve, monkeypatch):
    start, calls = serve
    post = start()
    monkeypatch.setattr(server_module, "is_loopback_address", lambda address: False)
    assert post(AD_HOC, token=ADMIN)[0] == 401  # falls through to tenant auth
    assert post(NAMED, token=ADMIN)[0] == 401
    assert calls == []


def test_reserved_tenant_header_is_refused_without_tenant_auth(serve):
    start, calls = serve
    post = start(tenant_auth=False)
    for reserved in ("_operator", "_overflow"):
        assert post(NAMED, headers={"X-Tenant-ID": reserved})[0] == 400
    assert calls == []


def test_admin_usage_body_accepts_the_operator_book_only():
    from mlx2.server import validate_usage_admin_body

    assert validate_usage_admin_body({"tenant": OPERATOR_TENANT})[0] == OPERATOR_TENANT
    with pytest.raises(ValueError):
        validate_usage_admin_body({"tenant": "_other"})


def test_operator_tenant_cannot_be_claimed_by_a_tenant():
    from mlx2.tenant_auth import validate_tenant
    from mlx2.usage import OVERFLOW_TENANT

    assert OPERATOR_TENANT != OVERFLOW_TENANT
    with pytest.raises(ValueError):
        validate_tenant(OPERATOR_TENANT)


def test_without_tenant_auth_behaviour_is_unchanged_for_named(serve):
    start, calls = serve
    post = start(tenant_auth=False)
    assert post(NAMED, headers={"X-Tenant-ID": "bob"})[0] == 200
    assert calls[-1]["tenant"] == "bob"
    assert post(AD_HOC)[0] == 401  # admin token configured, none presented
    status, _ = post(AD_HOC, token=ADMIN)
    assert status == 200 and calls[-1]["tenant"] == OPERATOR_TENANT


def test_ad_hoc_preview_without_an_admin_token_file_is_403(serve):
    start, calls = serve
    post = start(admin_token=None, tenant_auth=False)
    assert post(AD_HOC)[0] == 403
    assert post(NAMED)[0] == 200
    assert calls[-1]["tenant"] == "default"


def test_pcore_probe_refuses_a_relative_probe_python():
    # Morning job 2026-11-06 item 6: the LaunchAgent arm execs MLX2_PROBE_PYTHON,
    # so a relative path must be refused, not resolved against some cwd.
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_pcore_share_probe.py"
    env = {**os.environ, "MLX2_PROBE_PYTHON": ".venv/bin/python"}
    result = subprocess.run([sys.executable, str(script), "--help"], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "must be an absolute path" in result.stderr
    env["MLX2_PROBE_PYTHON"] = sys.executable
    assert subprocess.run([sys.executable, str(script), "--help"], env=env,
                          capture_output=True, timeout=60).returncode == 0
