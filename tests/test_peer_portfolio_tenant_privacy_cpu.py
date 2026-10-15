"""Request telemetry excludes other tenants while internal critics stay shared."""

import pytest
from test_drafter_metrics import receipt as metric_receipt
from test_proposal_pool import D, E, path, pool, round_, session, source
from test_tenant_auth import (
    CHAT,
    KEY_A,
    KEY_B,
    TenantEngine,
    _auth,
    _call,
)
from test_tenant_auth import (
    served as tenant_server,
)

from mlx2.drafter_metrics import DrafterRuntimeMetrics
from mlx2.http_security import policy_for_bind
from mlx2.prometheus import render_engine_metrics
from mlx2.usage import UsageLedger

served = tenant_server


def test_scoped_receipts_hide_shared_health_counts_and_revisions():
    proposals = pool()
    before_b = proposals.receipt(E)
    predicted_before = proposals.ranking_registry.predict(session(), source(), E, 0, None)
    selected_a = proposals.select([path((1,))], round_(request="a", scope=D))
    proposals.commit_feedback(selected_a, [2])

    # Another scope's committed labels may inform the intentional shared critic,
    # but its detailed activity must not appear in this caller's public receipt.
    assert proposals.ranking_registry.predict(session(), source(), E, 0, None) != predicted_before
    assert proposals.source_health("alpha")["rounds"] == 1
    assert proposals.receipt(E) == before_b
    public_a = proposals.receipt(D)
    assert public_a["telemetry_scope"] == "session"
    assert public_a["ranking_registry"]["session_counts"][0]["rejected"] == 1
    assert not {"source_health", "feedback_revision", "ranking_revision"} & public_a.keys()
    assert not {
        "global_counts", "model_counts", "revision", "retained_models",
        "retained_sessions", "evictions",
    } & public_a["ranking_registry"].keys()

    selected_b = proposals.select([path((1,))], round_(request="b", scope=E))
    proposals.commit_feedback(selected_b, [1])
    assert proposals.receipt(D) == public_a
    assert proposals.receipt(E)["ranking_registry"]["session_counts"][0]["accepted"] == 1
    operator = proposals.receipt()
    assert operator["telemetry_scope"] == "operator"
    assert operator["source_health"]["alpha"]["rounds"] == 2
    assert operator["feedback_revision"] == operator["ranking_revision"] == 2
    assert operator["ranking_registry"]["global_counts"][0]["accepted"] == 1


def test_default_scope_is_public_and_empty_scope_is_rejected():
    proposals = pool(session_scope_hash=D)
    assert proposals.receipt() == proposals.receipt(D)
    assert "source_health" not in proposals.receipt()
    with pytest.raises(ValueError, match="session_scope_hash"):
        proposals.receipt("")


class MetricEngine(TenantEngine):
    def __init__(self):
        super().__init__()
        self.drafter_metrics = DrafterRuntimeMetrics()
        self.export_scopes = []

    def submit(self, request, *, tenant_id="default", **kwargs):
        job = super().submit(request, tenant_id=tenant_id, **kwargs)
        mechanism = "dspark" if tenant_id == "tenant-b" else "lilicorr"
        self.drafter_metrics.observe_receipt(
            tenant_id if getattr(self, "tenant_metrics_scoped", True) else "default",
            metric_receipt(mechanism, proposed=5, accepted=2, partial=1, full=0),
        )
        return job

    def prometheus_metrics(self, *, tenant_id=None):
        self.export_scopes.append(tenant_id)
        return render_engine_metrics(self, tenant_id=tenant_id)


@pytest.mark.parametrize("tenant_header", ["__unverified_no_metrics__", "tenant-a"])
def test_unauthenticated_scrape_cannot_select_any_request_metric_partition(
    served, tenant_header, tmp_path,
):
    engine = MetricEngine()
    app = served(_auth(tmp_path), engine=engine, admin_token="operator-fixture")
    status, _, _ = _call(
        app.base, "POST", "/v1/chat/completions", CHAT,
        {"x-api-key": KEY_A},
    )
    assert status == 200 and engine.submitted == ["tenant-a"]
    # Even previously retained legacy state must not turn an invented empty
    # partition into access for a caller exempted from tenant authentication.
    engine.drafter_metrics.observe_receipt(
        "__unverified_no_metrics__",
        metric_receipt("dspark", proposed=5, accepted=2, partial=1, full=0),
    )
    for headers in ({}, {"X-Tenant-ID": tenant_header}):
        status, _, raw = _call(app.base, "GET", "/metrics", headers=headers)
        assert status == 200 and b"mlx2_build_info" in raw
        assert b"mlx2_drafter_" not in raw
    # The unverified exempt path must bypass a custom exporter entirely.
    assert engine.export_scopes == []
    status, _, raw = _call(
        app.base, "GET", "/metrics",
        headers={"Authorization": "Bearer operator-fixture"},
    )
    assert status == 200 and b'drafter="lilicorr"' in raw
    assert engine.export_scopes == [None]


def test_legacy_aggregate_scrape_requires_http_gate_but_does_not_gain_usage_authority(
    served, tmp_path, monkeypatch,
):
    # Simulate a remote collector: passing the legacy HTTP gate authorizes its
    # aggregate scrape, but does not make it a loopback/admin usage reader.
    monkeypatch.setattr("mlx2.server.is_loopback_address", lambda _address: False)
    engine = MetricEngine()
    gate_key = "synthetic-metrics-gate-key"
    ledger = UsageLedger(tmp_path / "usage", writer=False)
    try:
        ledger.record("private-usage-tenant", prompt_tokens=7)
        app = served(
            None, engine=engine, admin_token="operator-fixture", usage_ledger=ledger,
            http_security=policy_for_bind("127.0.0.1", api_key=gate_key),
        )
        assert engine.tenant_metrics_scoped is False
        admitted = {"x-api-key": gate_key, "X-Tenant-ID": "__unverified_no_metrics__"}
        status, _, _ = _call(app.base, "POST", "/v1/chat/completions", CHAT, admitted)
        assert status == 200
        assert engine.drafter_metrics.prometheus_snapshot("default")["tokens"]
        assert not engine.drafter_metrics.prometheus_snapshot("__unverified_no_metrics__")["tokens"]
        for headers in ({}, {"x-api-key": "wrong", "X-Tenant-ID": "default"}):
            status, _, raw = _call(app.base, "GET", "/metrics", headers=headers)
            assert status == 401 and b"mlx2_drafter_" not in raw
        status, _, raw = _call(app.base, "GET", "/metrics", headers=admitted)
        assert status == 200 and b'drafter="lilicorr"' in raw
        assert b"mlx2_num_requests_running" in raw
        assert b"private-usage-tenant" not in raw and b"mlx2_usage_" not in raw
        assert engine.export_scopes == [None]
    finally:
        ledger.close()


def test_authenticated_metrics_keep_own_counts_and_operator_keeps_aggregate(
    served, tmp_path, monkeypatch,
):
    # Use real loopback HTTP transport but classify it as a remote caller: the
    # intentional loopback /metrics exemption skips tenant authentication.
    monkeypatch.setattr("mlx2.server.is_loopback_address", lambda _address: False)
    engine = MetricEngine()
    app = served(_auth(tmp_path), engine=engine, admin_token="operator-fixture")
    for key in (KEY_A, KEY_B):
        status, _, _ = _call(
            app.base, "POST", "/v1/chat/completions", CHAT, {"x-api-key": key},
        )
        assert status == 200
    for key, own, other in ((KEY_A, "lilicorr", "dspark"), (KEY_B, "dspark", "lilicorr")):
        status, _, raw = _call(app.base, "GET", "/metrics", headers={"x-api-key": key})
        assert status == 200
        assert f'drafter="{own}"'.encode() in raw
        assert f'drafter="{other}"'.encode() not in raw
    status, _, raw = _call(
        app.base, "GET", "/metrics",
        headers={"x-api-key": KEY_A, "X-Tenant-ID": "tenant-b"},
    )
    assert status == 403 and b"mlx2_drafter_" not in raw
    status, _, raw = _call(app.base, "GET", "/metrics")
    assert status == 401 and b"mlx2_drafter_" not in raw
    assert engine.export_scopes == ["tenant-a", "tenant-b"]


def test_renderer_can_omit_all_request_metrics_without_a_magic_tenant():
    engine = MetricEngine()
    engine.submit(CHAT, tenant_id="__unverified_no_metrics__")
    for tenant in (None, "__unverified_no_metrics__"):
        text = render_engine_metrics(engine, tenant_id=tenant, include_request_metrics=False)
        assert "mlx2_build_info" in text
        assert "mlx2_drafter_" not in text
        assert "mlx2_num_requests_running" not in text
