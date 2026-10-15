"""HTTP route census and public replay contracts using CPU-only fake engines."""

import ast
import base64
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

# Reuse the real-handler fixture and authenticated two-tenant setup already
# shared by the tenant consumer tests; these tests add the expanded census.
from test_tenant_auth import KEY_A, _auth, _call
from test_tenant_auth import served as fixture_served

from mlx2.logprobs import distribution_contract
from mlx2.serving import ServingEngine

served = fixture_served


def test_authenticated_feature_scope_overrides_both_caller_and_adapter_fields():
    observed = []

    def prepare(request, *, file_loader=None):
        observed.append(request["_mlx2_feature_cache_tenant"])
        return {**request, "_mlx2_feature_cache_tenant": "adapter-forged"}

    engine = ServingEngine.__new__(ServingEngine)
    engine.model_path = "fixture"
    engine.qualification_mode = False
    engine.counts = Counter()
    engine.apc_reuse_disabled_reason = None
    engine.adapter = SimpleNamespace(prepare_multimodal_request=prepare)
    job = engine._prepare_job(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "unused-fixture"}},
                    ],
                }
            ],
            "_mlx2_feature_cache_tenant": "caller-forged",
        },
        tenant_id="trusted-tenant",
    )
    assert observed == ["trusted-tenant"]
    assert job.tenant_id == "trusted-tenant"
    assert job.request["_mlx2_feature_cache_tenant"] == "trusted-tenant"


def test_every_declared_http_path_authenticates_before_dispatch(served, tmp_path):
    app = served(_auth(tmp_path), admin_token="operator-secret")
    source = Path(__file__).resolve().parents[1] / "src/mlx2/server.py"
    # A newly added literal route enters this census automatically, even when
    # its actual executor is unavailable on this CPU fixture. Prefix routes
    # are probed with a resource id; private admin paths have their own gate.
    declared = {
        node.value
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith(("/v1/", "/tokenize", "/apply-template"))
        and not any(char in node.value for char in (" ", "{", "?"))
    }
    declared |= {
        "/v1/files/file_missing/content",
        "/v1/responses/resp_missing/input_items",
        "/v1/batches/batch_missing/cancel",
        "/v1/apc/sessions/session_missing/resume",
        "/v1/unknown-future-route",
    }
    assert len(declared) >= 30
    for path in sorted(declared):
        path += "missing" if path.endswith("/") else ""
        for method in ("GET", "POST", "DELETE"):
            status, _headers, _raw = _call(
                app.base,
                method,
                path,
                {} if method == "POST" else None,
                {"X-Tenant-ID": "tenant-a"},
            )
            assert status == 401, (method, path, status)
    assert app.engine.submitted == []
    assert app.engine.session_calls == []
    assert app.engine.lora_calls == []


def test_status_declares_readable_reasoning_replay_and_keeps_v1_compatibility(
    served, tmp_path
):
    app = served(_auth(tmp_path))
    code, _headers, raw = _call(
        app.base,
        "GET",
        "/v1/status",
        headers={"x-api-key": KEY_A},
    )
    assert code == 200
    contract = json.loads(raw)["reasoning_replay"]
    assert contract["confidentiality"] is False
    assert contract["authentication"] == "HMAC-SHA256"
    assert contract["scope"] == ["model", "tenant"]
    assert contract["key_persistence"] == "configured"
    assert "secret" not in json.dumps(contract)
    signer = app.engine.reasoning_signer
    token = signer.sign_responses(model="fixture", tenant="tenant-a", text="private é")
    assert token.startswith("mlx2.reasoning.v1.")
    payload = token.split(".")[4]
    assert (
        base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)).decode()
        == "private é"
    )
    assert (
        signer.verify_responses(token, model="fixture", tenant="tenant-a")
        == "private é"
    )
    assert signer.verify_responses(token, model="fixture", tenant="tenant-b") is None


def test_logprob_contract_distinguishes_greedy_verification_from_sampling_law():
    contract = distribution_contract()
    assert contract["ordinary"] == "post_processor_pre_sampler"
    for route in ("native_mtp", "external_draft"):
        assert contract[route]["greedy"] == contract["ordinary"]
        assert contract[route]["sampled"] == "post_temperature_and_sampling_filters"
    assert contract["route_can_change_within_request"] is True
    assert contract["draft_or_residual_probabilities"] is False
    # Mutable receipts cannot change the next request's contract.
    contract["native_mtp"]["sampled"] = "corrupted"
    assert distribution_contract()["native_mtp"]["sampled"] != "corrupted"


@pytest.mark.parametrize("operation", ["park", "delete"])
def test_http_session_mutation_reports_real_metadata_timeout_as_indeterminate(
    served, tmp_path, monkeypatch, operation
):
    import threading

    from test_peer_survey_persistence_cpu import _cache, _fixture, _persistent

    from mlx2.runtime import apc_v2

    app = served(_auth(tmp_path))
    key, raw, _draft, _hidden, _sidecar = _fixture()
    apc = _persistent(tmp_path / "apc", key)
    tag = ("tenant-a", "session")
    gate = threading.Event()
    original = apc._fsync_directory

    def delayed_fsync(path):
        assert gate.wait(5)
        return original(path)

    apc.store(key, list(range(8)), [_cache(raw, raw)], session_tag=tag)
    # Materialize on the owning CPU/model thread. HTTP park updates the
    # existing immutable disk snapshot and exercises real manifest I/O.
    apc.park_session(*tag, ttl_seconds=60)
    app.engine.apc_session_park = apc.park_session
    app.engine.apc_session_delete = apc.delete_session
    monkeypatch.setattr(apc, "_fsync_directory", delayed_fsync)
    monkeypatch.setattr(apc_v2, "_MANIFEST_WRITE_SECONDS", 0.03)
    try:
        code, _headers, raw = _call(
            app.base, "POST" if operation == "park" else "DELETE",
            "/v1/apc/sessions/session/park" if operation == "park" else "/v1/apc/sessions/session",
            {"ttl_seconds": 300} if operation == "park" else None,
            {"x-api-key": KEY_A},
        )
        assert code == 503
        error = json.loads(raw)["error"]
        assert error["code"] == "apc_persistence_indeterminate"
        assert "indeterminate" in error["message"]
        assert "reload" in error["message"]
    finally:
        gate.set()
        apc.close()
