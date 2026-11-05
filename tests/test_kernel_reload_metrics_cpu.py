"""Host-only reload telemetry, including actual loopback /metrics scrapes."""

from urllib.request import urlopen

import pytest
from test_kernel_reload_http_cpu import KernelEngine, request, serving
from test_prometheus import FakeEngine, assert_valid_prometheus_text

from mlx2.kernel_reload_serving import KernelReloadMixin
from mlx2.prometheus import render_engine_metrics


def scrape(port):
    with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as response:
        assert response.status == 200
        return response.read().decode()


def test_all_reload_work_reaches_metrics_with_bounded_labels():
    engine = FakeEngine()
    status = {
        "enabled": True, "epoch": 3, "pending": True, "phase": "purging",
        "source_root": "/secret/path", "execution_identity": "private-source-hash",
        "last_reload": {"modules": ["secret.module"]},
    }
    engine.kernel_reload_status = lambda: status
    engine.counts.update({
        "kernel_reload_requests": 9, "kernel_reload_completed": 3,
        "kernel_reload_failed": 2, "kernel_reload_drain_timeout": 1,
        "kernel_reload_rejected": 3, "kernel_reload_rejected_prepare": 2,
        "kernel_reload_rejected_disabled": 1, "kernel_reload_failed_apply": 1,
        "kernel_reload_failed_purge": 1, "kernel_reload_weights_checked": 4,
        "kernel_reload_weights_preserved": 3, "kernel_reload_weights_failed": 1,
        "kernel_reload_apc_entries_purged": 12, "kernel_reload_apc_bytes_purged": 4096,
        "kernel_reload_apc_sidecars_purged": 8, "kernel_reload_apc_parked_purged": 2,
        "kernel_reload_apc_disk_bytes_purged": 8192,
        "kernel_reload_host_prompt_entries_purged": 5,
        "kernel_reload_mlx_cache_bytes_released": 16384,
        "kernel_reload_rejected_private-reason": 99,
    })
    for index, kind in enumerate(("modules", "functions", "classes", "caches", "settings", "setter_calls", "layer_targets"), 1):
        engine.counts[f"kernel_reload_{kind}"] = index
    for index, stage in enumerate(("prepare", "drain", "purge", "apply", "total"), 1):
        engine.counts[f"kernel_reload_{stage}_seconds_sum"] = index / 2
        engine.counts[f"kernel_reload_{stage}_seconds_count"] = index
    before = dict(engine.counts)
    with serving(engine) as port:
        first = scrape(port)
        second = scrape(port)
    assert first == second
    assert engine.counts == before
    assert_valid_prometheus_text(first)
    for sample in (
        "mlx2_kernel_reload_enabled 1", "mlx2_kernel_reload_epoch 3",
        "mlx2_kernel_reload_in_progress 1", 'mlx2_kernel_reload_state{state="purging"} 1',
        "mlx2_kernel_reload_requests_total 9",
        'mlx2_kernel_reload_outcomes_total{outcome="completed"} 3',
        'mlx2_kernel_reload_outcomes_total{outcome="failed"} 2',
        'mlx2_kernel_reload_outcomes_total{outcome="drain_timeout"} 1',
        'mlx2_kernel_reload_outcomes_total{outcome="rejected"} 3',
        'mlx2_kernel_reload_rejections_total{reason="prepare"} 2',
        'mlx2_kernel_reload_rejections_total{reason="disabled"} 1',
        'mlx2_kernel_reload_failures_total{stage="apply"} 1',
        'mlx2_kernel_reload_failures_total{stage="purge"} 1',
        "mlx2_kernel_reload_weight_checks_total 4",
        "mlx2_kernel_reload_weights_preserved_total 3",
        "mlx2_kernel_reload_weight_check_failures_total 1",
        'mlx2_kernel_reload_apcv2_purged_total{kind="entries"} 12',
        'mlx2_kernel_reload_apcv2_purged_total{kind="sidecars"} 8',
        'mlx2_kernel_reload_apcv2_purged_total{kind="parked_sessions"} 2',
        "mlx2_kernel_reload_apcv2_purged_bytes_total 4096",
        "mlx2_kernel_reload_apcv2_disk_purged_bytes_total 8192",
        "mlx2_kernel_reload_host_prompt_purged_total 5",
        "mlx2_kernel_reload_buffer_released_bytes_total 16384",
    ):
        assert sample in first
    for index, kind in enumerate(("modules", "functions", "classes", "caches", "settings", "setter_calls", "layer_targets"), 1):
        assert f'mlx2_kernel_reload_applied_total{{kind="{kind}"}} {index}' in first
    for index, stage in enumerate(("prepare", "drain", "purge", "apply", "total"), 1):
        assert f'mlx2_kernel_reload_duration_seconds_sum{{stage="{stage}"}} {index / 2:g}' in first
        assert f'mlx2_kernel_reload_duration_seconds_count{{stage="{stage}"}} {index}' in first
    for private in ("/secret/path", "private-source-hash", "secret.module", "private-reason"):
        assert private not in first


def test_http_envelope_rejections_and_engine_disabled_are_counted_once():
    class Disabled(KernelReloadMixin, KernelEngine):
        def __init__(self):
            super().__init__()
            self._init_kernel_reload(None)

    engine = Disabled()
    with serving(engine) as port:
        assert request(port, body={})[0] == 400
        assert request(port, body={"modules": ["fixture.kernel"]})[0] == 409
        assert request(port, body={}, token="wrong")[0] == 401
        text = scrape(port)
    assert "mlx2_kernel_reload_enabled 0" in text
    assert "mlx2_kernel_reload_requests_total 2" in text
    assert 'mlx2_kernel_reload_outcomes_total{outcome="rejected"} 2' in text
    assert 'mlx2_kernel_reload_rejections_total{reason="invalid_request"} 1' in text
    assert 'mlx2_kernel_reload_rejections_total{reason="disabled"} 1' in text
    assert 'mlx2_kernel_reload_duration_seconds_count{stage="total"} 2' in text


@pytest.mark.parametrize("phase,active", [("preparing", 1), ("draining", 1), ("purging", 1), ("applying", 1), ("failed", 0), ("private-phase", 0)])
def test_reload_phase_is_live_bounded_and_operator_scoped(phase, active):
    engine = FakeEngine()
    engine.snapshot["kernel_reload"] = {"phase": "idle"}
    engine.kernel_reload_status = lambda: {"enabled": True, "phase": phase}
    text = render_engine_metrics(engine)
    assert f"mlx2_kernel_reload_in_progress {active}" in text
    bounded = "unknown" if phase == "private-phase" else phase
    assert f'mlx2_kernel_reload_state{{state="{bounded}"}} 1' in text
    assert "private-phase" not in text
    assert "mlx2_kernel_reload_" not in render_engine_metrics(engine, tenant_id="tenant")
    assert "mlx2_kernel_reload_" not in render_engine_metrics(engine, include_request_metrics=False)
