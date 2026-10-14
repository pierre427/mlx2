"""POST /v1/admin/apc/clear: admin-keyed, and the engine drops every prefix cache."""

from collections import Counter

import mlx.core as mx
import pytest

from mlx2.http_security import policy_for_bind
from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.models.cache import KVCache
from mlx2.server import validate_apc_clear_body
from mlx2.serving import HostPromptCache, ServingEngine
from test_http_security import _request, _server
from test_server_quiesce import AdminEngine

mx.set_default_device(mx.cpu)


class ClearingEngine(AdminEngine):
    def clear_prefix_caches(self, *, reason="admin"):
        self.calls.append(("clear", reason))
        return {"reason": reason, "apc": {"entries": 3}}


@pytest.fixture
def clearing(monkeypatch):
    import test_http_security

    monkeypatch.setattr(test_http_security, "AdminEngine", ClearingEngine)


def test_clear_refuses_without_a_configured_admin_key(clearing):
    with _server(policy_for_bind("127.0.0.1")) as port:
        status, _, body = _request(port, "POST", "/v1/admin/apc/clear", body=b"{}")
    assert status == 403 and "--admin-token-file" in body["error"]["message"]


def test_clear_needs_the_admin_bearer_and_reports(clearing):
    with _server(policy_for_bind("127.0.0.1"), admin_token="admin") as port:
        assert _request(port, "POST", "/v1/admin/apc/clear", body=b"{}")[0] == 401
        assert _request(
            port, "POST", "/v1/admin/apc/clear",
            headers={"Authorization": "Bearer wrong"}, body=b"{}",
        )[0] == 401
        status, _, body = _request(
            port, "POST", "/v1/admin/apc/clear",
            headers={"Authorization": "Bearer admin", "Content-Type": "application/json"},
            body=b'{"reason": "ladder-cell 16384 w4"}',
        )
        assert status == 200 and body == {"reason": "ladder-cell 16384 w4", "apc": {"entries": 3}}
        status, _, body = _request(
            port, "POST", "/v1/admin/apc/clear",
            headers={"Authorization": "Bearer admin"}, body=b'{"scope": "all"}',
        )
        assert status == 400 and "scope" in body["error"]["message"]


def test_clear_on_an_engine_without_prefix_caches_is_a_conflict():
    with _server(policy_for_bind("127.0.0.1"), admin_token="admin") as port:
        status, _, _ = _request(
            port, "POST", "/v1/admin/apc/clear",
            headers={"Authorization": "Bearer admin"}, body=b"",
        )
    assert status == 409


def test_clear_body_validation():
    assert validate_apc_clear_body(None) == "admin"
    assert validate_apc_clear_body({}) == "admin"
    assert validate_apc_clear_body({"reason": "x"}) == "x"
    for bad in ([], {"reason": ""}, {"reason": "y" * 65}, {"reason": 3}, {"other": 1}):
        with pytest.raises(ValueError):
            validate_apc_clear_body(bad)


def test_engine_clear_drops_apc_and_host_prompt_cache_but_keeps_the_model_revision():
    engine = object.__new__(ServingEngine)
    engine.apc = APCv2(max_size=4, layout_name="admin-clear-v1")
    engine.host_prompt_cache = HostPromptCache()
    engine.counts = Counter()
    engine.model_revision = 7
    cache = KVCache()
    values = mx.arange(3, dtype=mx.float32).reshape(1, 1, 3, 1)
    cache.update_and_fetch(values, values)
    engine.apc.store(APCKey("m"), [1, 2, 3], [cache])
    engine.host_prompt_cache.put({"prompt": "hi"}, [1, 2, 3])
    assert engine.apc.lookup(APCKey("m"), [1, 2, 3, 4]).cached_tokens == 3
    report = engine.clear_prefix_caches(reason="ladder")
    assert report["reason"] == "ladder" and report["host_prompt_cache_entries"] == 1
    assert engine.apc.lookup(APCKey("m"), [1, 2, 3, 4]).cached_tokens == 0
    assert engine.host_prompt_cache.status()["entries"] == 0
    assert engine.counts["apc_admin_clears"] == 1 and engine.model_revision == 7
