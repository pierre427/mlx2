"""Pure parser, counter, and renderer tests for mlx2-top."""

from __future__ import annotations

import plistlib
import time

from mlx2.inference_monitor import (
    Metrics,
    ServerEndpoint,
    ServerReading,
    build_reading,
    histogram_quantile,
    parse_lsof_listeners,
)
from mlx2.system_monitor import (
    CPUTicks,
    GPUReading,
    HostReading,
    cpu_busy_percent,
    parse_ioreg_gpus,
    parse_powermetrics_plist,
)
from mlx2.top import format_bytes, render


def test_cpu_busy_is_delta_based_and_includes_nice():
    before = (CPUTicks(10, 20, 70, 0), CPUTicks(5, 5, 90, 0))
    after = (CPUTicks(20, 30, 140, 10), CPUTicks(5, 5, 100, 0))
    assert cpu_busy_percent(before, after) == (30.0, 0.0)
    assert cpu_busy_percent(before, after[:1]) == ()


def test_ioreg_parser_uses_per_device_performance_statistics():
    payload = plistlib.dumps(
        [
            {
                "IORegistryEntryName": "AGXAcceleratorG17X",
                "PerformanceStatistics": {
                    "Device Utilization %": 42,
                    "Renderer Utilization %": 41,
                    "Tiler Utilization %": 3,
                    "In use system memory": 1234,
                    "Alloc system memory": 5678,
                },
            },
            {"IORegistryEntryName": "not-a-device", "busy": 1},
        ]
    )
    assert parse_ioreg_gpus(payload) == (
        GPUReading(
            name="AGXAcceleratorG17X",
            busy_percent=42.0,
            renderer_percent=41.0,
            tiler_percent=3.0,
            memory_bytes=1234,
            allocated_bytes=5678,
        ),
    )


def test_powermetrics_parser_keeps_gpu_and_thermal_fields():
    parsed = parse_powermetrics_plist(
        {
            "gpu": {"idle_ratio": 0.25, "freq_hz": 1_200_000_000},
            "processor": {"gpu_power": 2500},
            "thermal_pressure": "Nominal",
        }
    )
    assert parsed.gpu_busy_percent == 75.0
    assert parsed.gpu_frequency_mhz == 1200.0
    assert parsed.gpu_power_mw == 2500.0
    assert parsed.thermal_pressure == "Nominal"


def _host_reading() -> HostReading:
    return HostReading(
        sampled_at=time.time(),
        interval_seconds=1.0,
        cpu_busy_percent=(10.0, 20.0),
        load_average=(1.0, 2.0, 3.0),
        memory=None,
        swap=None,
        gpus=(),
        thermal_state="nominal",
        thermal_pressure=None,
    )


def test_renderer_drops_accelerators_and_keeps_host_sections():
    output = render(_host_reading(), servers=None)
    assert "Accelerators" not in output
    assert "ANE" not in output and "NAX" not in output
    assert "LLM" not in output
    assert "CPU 00" in output
    assert "Load 1.00 2.00 3.00" in output


_EXPOSITION_BEFORE = """\
# HELP mlx2_build_info build
# TYPE mlx2_build_info gauge
mlx2_build_info{revision="d3a8c102d8f571c0",version="0.1.0"} 1
mlx2_model_info{model_name="Demo-4bit",profile="demo-mtp2",qualification="candidate"} 1
mlx2_server_state{state="draining"} 0
mlx2_server_state{state="serving"} 1
mlx2_ready 1
mlx2_process_uptime_seconds 100
mlx2_generation_tokens_total 1000
mlx2_prompt_tokens_total 4000
mlx2_cached_prompt_tokens_total 1000
mlx2_requests_total{finished_reason="stop",outcome="completed"} 9
mlx2_requests_total{finished_reason="none",outcome="failed"} 0
mlx2_num_requests_running 2
mlx2_num_requests_waiting 1
mlx2_active_lanes 2
mlx2_peak_active_lanes 4
mlx2_batch_size_sum 100
mlx2_batch_size_count 100
mlx2_time_to_first_token_seconds_bucket{le="0.1"} 5
mlx2_time_to_first_token_seconds_bucket{le="0.2"} 9
mlx2_time_to_first_token_seconds_bucket{le="+Inf"} 10
mlx2_time_to_first_token_seconds_sum 1.2
mlx2_time_to_first_token_seconds_count 10
mlx2_segmented_mtp_events_total{event="accepted_all"} 6
mlx2_segmented_mtp_events_total{event="accepted_partial"} 3
mlx2_segmented_mtp_events_total{event="accepted_zero"} 1
mlx2_prefix_cache_lookups_total 10
mlx2_prefix_cache_hits_total 4
mlx2_prefix_cache_query_tokens_total 4000
mlx2_fused_gdn_events_total{event="verify_calls"} 50
mlx2_fail_closed_total{capability="tool_calls",reason="parallel_bound"} 0
mlx2_http_requests_total{method="POST",route="chat_completions",status_class="2xx"} 9
mlx2_http_requests_total{method="POST",route="chat_completions",status_class="4xx"} 2
"""


def _after() -> str:
    return (
        _EXPOSITION_BEFORE.replace("generation_tokens_total 1000", "generation_tokens_total 1500")
        .replace("prompt_tokens_total 4000", "prompt_tokens_total 6000")
        .replace("batch_size_sum 100", "batch_size_sum 130")
        .replace("batch_size_count 100", "batch_size_count 110")
        .replace('le="0.1"} 5', 'le="0.1"} 6')
        .replace('le="0.2"} 9', 'le="0.2"} 11')
        .replace('le="+Inf"} 10', 'le="+Inf"} 12')
        .replace("seconds_count 10", "seconds_count 12")
        .replace('event="accepted_all"} 6', 'event="accepted_all"} 10')
        .replace('event="verify_calls"} 50', 'event="verify_calls"} 250')
        .replace("process_uptime_seconds 100", "process_uptime_seconds 102")
    )


def test_prometheus_parser_sums_matching_series_and_reads_labels():
    metrics = Metrics.parse(_EXPOSITION_BEFORE)
    assert metrics.value("requests_total") == 9
    assert metrics.value("requests_total", outcome="completed") == 9
    assert metrics.value("missing_total") is None
    assert metrics.labels("model_info")["model_name"] == "Demo-4bit"
    assert metrics.buckets("time_to_first_token_seconds")[-1] == (float("inf"), 10)


def test_histogram_quantile_interpolates_inside_the_bucket():
    buckets = [(0.1, 5.0), (0.2, 9.0), (float("inf"), 10.0)]
    assert histogram_quantile(0.5, buckets) == 0.1
    assert round(histogram_quantile(0.7, buckets), 6) == 0.15
    assert histogram_quantile(0.99, buckets) == 0.2
    assert histogram_quantile(0.5, [(0.1, 0.0)]) is None


def test_lsof_listeners_become_loopback_urls_with_model_labels():
    output = "p4015\nf6\nn127.0.0.1:8297\np6004\nn*:8296\n"
    endpoints = parse_lsof_listeners(
        output,
        {
            4015: "python -m mlx2.server --model /models/A --port 8297",
            6004: "python -m mlx2.server --model /models/B --port 8296",
        },
    )
    assert [endpoint.url for endpoint in endpoints] == [
        "http://127.0.0.1:8296",
        "http://127.0.0.1:8297",
    ]
    assert endpoints[1].pid == 4015 and endpoints[1].model_path == "/models/A"


def test_reading_turns_counters_into_rates_and_window_statistics():
    endpoint = ServerEndpoint(url="http://127.0.0.1:8297", pid=4015)
    before = Metrics.parse(_EXPOSITION_BEFORE)
    after = Metrics.parse(_after())
    status = {
        "model": "Demo-4bit",
        "max_lanes": 4,
        "settings": {"speculation": "self_mtp", "adapter_policy": {"num_draft": 2}},
        "recent_receipts": [
            {
                "request_id": "chatcmpl-abcdef0123",
                "prompt_tokens": 100,
                "cached_tokens": 64,
                "completion_tokens": 11,
                "ttft_seconds": 0.5,
                "elapsed_seconds": 1.5,
                "route": "native_mtp",
                "mtp": {
                    "route": "segmented_self_mtp",
                    "stats": {
                        "draft_proposed": 10,
                        "draft_accepted": 8,
                        "total_emitted": 11,
                        "cycles": 5,
                    },
                },
            }
        ],
    }
    batching = {
        "latency_ms": {"ttft": {"count": 12, "p50": 90.0, "p95": 180.0, "p99": 195.0}},
        "batch_composition": {"1": 3, "2": 1},
        "active_requests": ["chatcmpl-live"],
        "events": [
            {"kind": "admitted", "request_id": "chatcmpl-live", "at": 98.0},
            {"kind": "dequeued", "request_id": "chatcmpl-live", "at": 98.5},
            {"kind": "lane_attached", "request_id": "chatcmpl-live", "at": 98.5},
        ],
        "completed_requests": [{"request_id": "chatcmpl-abcdef0123", "status": "completed"}],
    }
    reading = build_reading(
        endpoint,
        after,
        now=100.0,
        previous=(98.0, before),
        window=(98.0, before),
        batching=batching,
        status=status,
    )
    assert reading.reachable and reading.state == "serving" and reading.ready is True
    assert reading.generation_tps == 250.0
    assert reading.prompt_tps == 1000.0
    assert reading.batch_width_window == 3.0
    assert reading.outcomes == (("completed:stop", 9),)
    ttft = dict(reading.latency)["TTFT"]
    assert ttft.exact and ttft.p50 == 0.09 and ttft.count == 12
    assert ttft.window_count == 2 and round(ttft.window_p95, 6) == 0.19
    assert reading.speculation.cycles == 14 and reading.speculation.window_accepted_all == 4
    assert reading.speculation.receipt_accepted == 8 and reading.speculation.num_draft == 2
    assert reading.mechanisms[0].name == "fused_gdn_events:verify_calls"
    assert reading.mechanisms[0].per_second == 100.0
    assert reading.http_errors == (("chat_completions 4xx", 2),)
    assert reading.failures == ()
    live = reading.active_requests[0]
    assert live.phase == "prefill" and live.age_seconds == 2.0 and live.queue_ms == 500.0
    recent = reading.recent_requests[0]
    assert recent.status == "completed" and recent.draft_acceptance == 0.8
    assert recent.decode_tokens_per_second == 10.0 and recent.tokens_per_cycle == 2.2

    output = render(_host_reading(), servers=(reading,))
    assert "── LLM  Demo-4bit" in output
    assert "gen 250.0 now" in output
    assert "verify cycles 14" in output
    assert "fused_gdn_events:verify_calls" in output
    assert "abcdef01" in output and "prefill" in output
    assert "Accelerators" not in output


def test_renderer_reports_unreachable_and_missing_servers():
    down = ServerReading(
        endpoint=ServerEndpoint(url="http://127.0.0.1:8285"),
        reachable=False,
        error="Connection refused",
    )
    assert "unreachable — Connection refused" in render(_host_reading(), servers=(down,))
    assert "no running mlx2 server found" in render(_host_reading(), servers=())


def test_format_bytes_uses_binary_units_and_rate_suffix():
    assert format_bytes(1536) == "1.5 KiB"
    assert format_bytes(2 << 30) == "2.0 GiB"
    assert format_bytes(3 << 20, rate=True) == "3.0 MiB/s"
