"""Tenant isolation and partial-acceptance metrics for multi-drafter serving."""

from mlx2.drafter_metrics import DrafterRuntimeMetrics
from mlx2.prometheus import render_engine_metrics


def receipt(mechanism, *, proposed, accepted, partial, full):
    return {
        "speculation": {
            "continuation_pool": {
                "provider_stats": {
                    "resident": {
                        "mechanism": mechanism,
                        "enabled_rounds": 1,
                        "admitted_rounds": 1,
                        "selected_rounds": 1,
                        "contributed_rounds": 1,
                        "candidate_paths": 2,
                        "proposed_tokens": proposed,
                        "accepted_tokens": accepted,
                        "zero_rounds": int(accepted == 0),
                        "partial_rounds": partial,
                        "full_rounds": full,
                        "proposal_depths": {str(proposed): 1},
                        "accepted_depths": {str(accepted): 1},
                        "first_rejected_positions": (
                            {str(accepted): 1} if partial else {}
                        ),
                        "selected_ranks": {"1": 1},
                        "score_sum": 1.5,
                        "score_observations": 1,
                        "parked_rounds": 0,
                        "fallback_rounds": 0,
                    }
                },
                "winning_combinations": {mechanism: 1},
                "last_admission": {
                    "admitted": ["resident"],
                    "parked": [],
                    "lane_slots_used": 1,
                    "lane_slots_available": 5,
                    "tile_rows_used": 3,
                    "tile_rows_available": 75,
                },
            }
        }
    }


class Engine:
    def __init__(self, metrics):
        self.drafter_metrics = metrics

    @staticmethod
    def status():
        return {
            "state": "ready",
            "model": "fixture",
            "profile": "fixture",
            "qualification": "unqualified",
        }


def test_tenant_scrape_contains_only_own_partial_and_full_drafts():
    metrics = DrafterRuntimeMetrics()
    metrics.observe_receipt(
        "tenant-a", receipt("lilicorr", proposed=5, accepted=2, partial=1, full=0)
    )
    metrics.observe_receipt(
        "tenant-b", receipt("dspark", proposed=4, accepted=4, partial=0, full=1)
    )

    tenant = render_engine_metrics(Engine(metrics), tenant_id="tenant-a")
    assert 'drafter="lilicorr",outcome="partial"' in tenant
    assert 'drafter="lilicorr",kind="accepted"} 2' in tenant
    assert 'drafter="dspark"' not in tenant
    assert "tenant-a" not in tenant and "tenant-b" not in tenant
    assert "mlx2_drafter_metric_tenant_evictions_total" not in tenant
    assert "mlx2_num_requests_running" not in tenant

    operator = render_engine_metrics(Engine(metrics))
    assert 'drafter="lilicorr"' in operator
    assert 'drafter="dspark"' in operator
    assert "mlx2_drafter_metric_tenant_evictions_total" in operator


def test_operator_totals_remain_monotonic_after_tenant_eviction():
    metrics = DrafterRuntimeMetrics(max_tenants=1)
    metrics.observe_receipt(
        "tenant-a", receipt("lilicorr", proposed=5, accepted=2, partial=1, full=0)
    )
    before = metrics.prometheus_snapshot()["tokens"].copy()
    metrics.observe_receipt(
        "tenant-b", receipt("dspark", proposed=4, accepted=4, partial=0, full=1)
    )
    after = metrics.prometheus_snapshot()["tokens"]

    assert after[("lilicorr", "proposed")] == before[("lilicorr", "proposed")]
    assert after[("dspark", "proposed")] == 4
    assert metrics.prometheus_snapshot("tenant-a")["tokens"] == {}
    assert metrics.prometheus_snapshot()["evictions"] == 1


def test_malformed_provider_stats_are_ignored_without_partial_failure():
    metrics = DrafterRuntimeMetrics()
    malformed = receipt("lilicorr", proposed=1, accepted=1, partial=0, full=1)
    provider = malformed["speculation"]["continuation_pool"]["provider_stats"][
        "resident"
    ]
    provider["proposed_tokens"] = None
    provider["proposal_depths"] = "not-a-map"
    metrics.observe_receipt("tenant", malformed)
    assert metrics.prometheus_snapshot("tenant")["events"][("lilicorr", "full")] == 1


def test_latch_state_tracks_latest_receipt_instead_of_historical_peak():
    metrics = DrafterRuntimeMetrics()
    active = receipt("lilicorr", proposed=2, accepted=1, partial=1, full=0)
    metrics.observe_receipt("tenant", active)
    assert metrics.prometheus_snapshot("tenant")["states"] == {"lilicorr": "active"}

    parked = receipt("lilicorr", proposed=0, accepted=0, partial=0, full=0)
    admission = parked["speculation"]["continuation_pool"]["last_admission"]
    admission["admitted"] = []
    admission["parked"] = [{"source_id": "resident"}]
    metrics.observe_receipt("tenant", parked)
    assert metrics.prometheus_snapshot("tenant")["states"] == {"lilicorr": "parked"}

    idle = receipt("lilicorr", proposed=0, accepted=0, partial=0, full=0)
    admission = idle["speculation"]["continuation_pool"]["last_admission"]
    admission["admitted"] = []
    metrics.observe_receipt("tenant", idle)
    assert metrics.prometheus_snapshot("tenant")["states"] == {"lilicorr": "idle"}
