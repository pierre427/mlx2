"""Drafter telemetry cannot interrupt the real serving terminal owner."""

import pytest
from route_harness import patch_host
from test_exact_decode_checkpoint_cpu import _engine, _model
from test_suspend_replay import collect

from mlx2.drafter_metrics import DrafterRuntimeMetrics


@pytest.mark.parametrize("tenant_scoped", [False, True])
def test_drafter_receipt_failure_preserves_terminal_and_worker(monkeypatch, tenant_scoped):
    patch_host(monkeypatch)
    engine = _engine(_model())
    engine.tenant_metrics_scoped = tenant_scoped
    observed = []
    original = DrafterRuntimeMetrics.observe_receipt

    def observe(metrics, tenant, receipt):
        observed.append((tenant, receipt))
        if len(observed) == 1:
            # Counter conversion is now hardened upstream. Exercise a failure
            # of the observer itself through the real terminal call boundary.
            raise RuntimeError("injected drafter observer failure")
        return original(metrics, tenant, receipt)

    monkeypatch.setattr(DrafterRuntimeMetrics, "observe_receipt", observe)
    try:
        assert engine.ready.wait(10), engine.error
        for tenant, start in (("tenant-a", 1), ("tenant-b", 20)):
            job = engine.submit({
                "tokens": list(range(start, start + 18)),
                "max_tokens": 3,
                "temperature": 0,
            }, tenant_id=tenant)
            tokens, receipt = collect(job)
            assert len(tokens) == 3 and receipt["completion_tokens"] == 3
            assert job.id not in engine.jobs
            assert job.exact_decode_checkpoint is None
            assert not engine._terminal_cleanup_pending
        assert len(observed) == 2
        expected_tenants = ["tenant-a", "tenant-b"] if tenant_scoped else ["default"] * 2
        assert [tenant for tenant, _receipt in observed] == expected_tenants
        assert engine.counts["drafter_metric_failures"] == 1
        assert engine.counts["completed"] == 2
        assert len(engine.recent_receipts()) == 2
        assert engine.error is None
    finally:
        engine.close()


def test_invalid_counter_conversion_is_nonfatal_and_preserves_valid_fields():
    metrics = DrafterRuntimeMetrics()
    metrics.observe_receipt("tenant", {"speculation": {
        "continuation_pool": {"provider_stats": {"resident": {
            "mechanism": "lilicorr",
            "enabled_rounds": "invalid counter",
            "proposed_tokens": None,
            "accepted_tokens": 2,
            "full_rounds": 1,
            "proposal_depths": "not-a-map",
        }}},
    }})
    for tenant in (None, "tenant"):
        snapshot = metrics.prometheus_snapshot(tenant)
        assert snapshot["events"][("lilicorr", "enabled")] == 0
        assert snapshot["events"][("lilicorr", "full")] == 1
        assert snapshot["tokens"][("lilicorr", "proposed")] == 0
        assert snapshot["tokens"][("lilicorr", "accepted")] == 2
        assert snapshot["depths"] == {}
