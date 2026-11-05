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
            "gpu": {"idle_ratio": 0.25, "freq_hz": 1200.0},
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


def test_external_draft_receipts_and_progressive_tiles_reach_top():
    endpoint = ServerEndpoint(url="http://127.0.0.1:8297", pid=4015)
    status = {
        "model": "Nemotron-Super-4bit",
        "settings": {
            "speculation": "external_draft",
            "adapter_policy": {"num_draft": 7},
        },
        "scheduler": {
            "external_progressive_verify_tile": 4,
            "external_progressive_verify_rounds": 38,
            "external_progressive_verify_launches": 60,
            "external_progressive_verify_target_rows": 151,
            "external_progressive_verify_padding_rows": 9,
            "external_progressive_verify_ragged_launches": 4,
            "external_progressive_verify_rejections": 16,
            "external_progressive_verify_full_tiles": 22,
        },
        "recent_receipts": [
            {
                "request_id": "chatcmpl-nemotron",
                "prompt_tokens": 80,
                "cached_tokens": 0,
                "completion_tokens": 128,
                "ttft_seconds": 0.4,
                "elapsed_seconds": 4.4,
                "route": "external_draft",
                "mtp": None,
                "speculation": {
                    "kind": "external_draft_v1",
                    "external_rounds": 38,
                    "accepted": 106,
                    "proposed": 153,
                    "progressive_verification": {
                        "verification_tile": 4,
                        "rounds": 38,
                        "target_launches": 60,
                        "target_rows": 151,
                        "padding_rows": 9,
                        "ragged_launches": 4,
                        "rejections": 16,
                        "full_nonfinal_tiles": 22,
                        "observed_used": True,
                    },
                },
            }
        ],
    }
    reading = build_reading(
        endpoint,
        Metrics.parse(_EXPOSITION_BEFORE),
        now=100.0,
        status=status,
    )
    spec = reading.speculation
    assert spec.mode == "external_draft" and spec.num_draft == 7
    assert spec.receipt_kind == "external_draft"
    assert spec.receipt_requests == 1 and spec.receipt_cycles == 38
    assert (spec.receipt_accepted, spec.receipt_proposed) == (106, 153)
    assert spec.progressive_tile == 4 and spec.progressive_rounds == 38
    assert spec.progressive_launches == 60 and spec.progressive_target_rows == 151
    assert spec.progressive_padding_rows == 9
    assert spec.progressive_ragged_launches == 4
    assert spec.progressive_rejections == 16 and spec.progressive_full_tiles == 22
    recent = reading.recent_requests[0]
    assert recent.route == "external_draft"
    assert recent.draft_acceptance == 106 / 153
    # External receipts do not publish a definitive total-emitted count; the
    # monitor must not invent one from accepted proposals plus rounds.
    assert recent.tokens_per_cycle is None
    output = render(_host_reading(), servers=(reading,))
    assert "external_draft draft 7" in output
    assert "draft accept 69.3% (106/153)" in output
    assert "9 padding" in output and "4 ragged" in output
    assert "progressive tile 4" in output
    assert "60 launches / 38 rounds" in output
    assert "151 rows" in output and "22 continued" in output


def test_external_draft_receipt_without_proposals_renders_safely():
    reading = build_reading(
        ServerEndpoint(url="http://127.0.0.1:8297"),
        Metrics.parse(_EXPOSITION_BEFORE),
        now=100.0,
        status={
            "settings": {"speculation": "external_draft"},
            "recent_receipts": [
                {
                    "request_id": "chatcmpl-ordinary-fallback",
                    "route": "external_draft",
                    "speculation": {
                        "kind": "external_draft_v1",
                        "external_rounds": 0,
                        "accepted": 0,
                        "proposed": 0,
                    },
                }
            ],
        },
    )
    output = render(_host_reading(), servers=(reading,))
    assert "last 1 receipts: no draft proposals" in output
    assert reading.recent_requests[0].draft_acceptance is None


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


def _soc_sampler(table=(300, 600, 900)):
    from mlx2.apple_telemetry import EnergyReading
    from mlx2.system_monitor import SocPowerSampler

    class Energy:
        def read(self):
            return EnergyReading(
                seconds=1.0,
                watts={"GPU Energy": 12.5, "DRAM": 2.0, "CPU Energy": 4.0, "ANE": 0.0},
                gpu_states=[("OFF", 50), ("P1", 25), ("P2", 0), ("P3", 25)],
            )

    class Temperature:
        def die_summary(self):
            return {"die_max_c": 70.0, "die_mean_c": 60.5, "battery_c": None}

    return SocPowerSampler(
        energy_factory=Energy,
        temperature_factory=Temperature,
        frequency_table=lambda: list(table),
    )


def test_soc_power_sampler_reads_unprivileged_power_dvfs_and_temperature():
    reading = _soc_sampler().sample()
    assert reading.gpu_power_mw == 12500.0 and reading.dram_power_mw == 2000.0
    assert reading.cpu_power_mw == 4000.0 and reading.ane_power_mw == 0.0
    assert reading.gpu_active_percent == 50.0
    assert reading.gpu_mean_pstate == 2.0
    assert reading.gpu_frequency_mhz == 600.0
    assert (reading.die_max_c, reading.die_mean_c) == (70.0, 60.5)
    # A table that does not map onto the P-states yields no clock.
    assert _soc_sampler(table=(300, 600)).sample().gpu_frequency_mhz is None


def test_soc_power_sampler_degrades_when_ioreport_is_missing():
    from mlx2.system_monitor import SocPowerSampler

    def refuse():
        raise RuntimeError("IOReport is unavailable on this host")

    sampler = SocPowerSampler(energy_factory=refuse)
    assert sampler.sample() is None
    assert sampler.error == "IOReport unavailable: IOReport is unavailable on this host"


def test_renderer_shows_ioreport_power_and_die_temperatures():
    from dataclasses import replace

    soc = _soc_sampler().sample()
    gpu = GPUReading(name="AGX", busy_percent=50.0, power_mw=soc.gpu_power_mw,
                     frequency_mhz=soc.gpu_frequency_mhz)
    output = render(replace(_host_reading(), gpus=(gpu,), soc_power=soc), servers=None)
    assert "12500 mW" in output and "600 MHz" in output
    assert "Power (IOReport, unprivileged)" in output
    assert "GPU 12.50 W" in output and "DRAM 2.00 W" in output
    assert "active 50.0%" in output and "mean P2.0" in output
    assert "die max 70.0 °C" in output and "mean 60.5 °C" in output
    assert "power/frequency unavailable" not in output


def test_renderer_names_why_power_is_missing():
    from dataclasses import replace

    gpu = GPUReading(name="AGX", busy_percent=50.0)
    output = render(
        replace(_host_reading(), gpus=(gpu,), soc_power_error="IOReport unavailable: x"),
        servers=None,
    )
    assert "power/frequency unavailable — IOReport unavailable: x" in output
    assert "die temperatures unavailable — IOReport unavailable: x" in output
    assert "Power (IOReport" not in output


def test_top_defaults_to_ioreport_and_keeps_powermetrics_optional():
    from mlx2.top import build_parser

    args = build_parser().parse_args([])
    assert args.ioreport is True and args.powermetrics is False
    args = build_parser().parse_args(["--powermetrics", "--no-ioreport"])
    assert args.ioreport is False and args.powermetrics is True


def test_soc_power_sampler_and_host_collector_close_the_native_samplers():
    from mlx2.apple_telemetry import EnergyReading
    from mlx2.system_monitor import HostCollector, SocPowerSampler

    closed = []

    class Energy:
        def read(self):
            return EnergyReading(seconds=1.0, watts={}, gpu_states=[])

        def close(self):
            closed.append("energy")

    class Temperature:
        def die_summary(self):
            return {}

        def close(self):
            closed.append("temperature")

    sampler = SocPowerSampler(energy_factory=Energy, temperature_factory=Temperature)
    collector = HostCollector(ioreport=False)
    collector.soc_power = sampler
    collector.close()
    collector.close()
    assert sorted(closed) == ["energy", "temperature"]
    assert sampler.sample() is None


def test_powermetrics_gpu_freq_hz_is_already_mhz():
    # /usr/bin/powermetrics prints the plist "gpu" freq_hz from the same double
    # as "GPU HW active frequency: %0.0f MHz" (unscaled); only the CPU cluster
    # freq_hz is multiplied by 1e6.
    parsed = parse_powermetrics_plist(
        {
            "gpu": {
                "freq_hz": 1398.0,
                "idle_ratio": 0.2,
                "dvfm_states": [{"freq": 1398, "used_ratio": 0.8}],
            },
            "processor": {"gpu_power": 31000.0},
        }
    )
    assert parsed.gpu_frequency_mhz == 1398.0


def test_powermetrics_gpu_clock_does_not_replace_ioreport_with_a_scaled_value():
    from types import SimpleNamespace

    from mlx2.system_monitor import HostCollector, SocPowerReading

    collector = HostCollector(ioreport=False)
    try:
        collector._gpus = lambda: (GPUReading(name="AGX", busy_percent=80.0),)
        collector.soc_power = SimpleNamespace(
            sample=lambda: SocPowerReading(gpu_frequency_mhz=1400.0),
            error=None,
            close=lambda: None,
        )
        collector.powermetrics = SimpleNamespace(
            latest=parse_powermetrics_plist(
                {"gpu": {"freq_hz": 1398.0, "idle_ratio": 0.2}}
            ),
            error=None,
            close=lambda: None,
        )
        reading = collector.sample()
    finally:
        collector.powermetrics = None
        collector.soc_power = None
        collector.close()
    # mlx2-top renders this as f"{frequency_mhz:.0f} MHz"; a 1e6-scaled value
    # printed as "0 MHz" and hid IOReport's correct clock.
    assert reading.gpus[0].frequency_mhz == 1398.0


def test_loading_engine_names_its_model_in_top():
    import threading
    from types import SimpleNamespace

    from mlx2.prometheus import render_engine_metrics

    # The constructor's loading snapshot: identity only, no profile yet.
    engine = SimpleNamespace(
        lock=threading.Lock(),
        snapshot={"state": "loading", "model": "Demo-4bit", "max_lanes": 4},
        counts={},
        queued_jobs=0,
        model_path="/models/Demo-4bit",
    )
    metrics = Metrics.parse(render_engine_metrics(engine))
    reading = build_reading(
        ServerEndpoint(url="http://127.0.0.1:8297"),
        metrics,
        now=100.0,
        status=dict(engine.snapshot),
    )
    assert reading.model == "Demo-4bit"
    assert reading.profile is None and reading.qualification is None
    assert reading.ready is False and reading.max_lanes == 4
    output = render(_host_reading(), servers=(reading,))
    assert "LLM  Demo-4bit" in output
    route = next(line for line in output.splitlines() if line.startswith("Route"))
    assert "loading" in route and "unavailable" not in route


def test_top_live_sessions_use_request_metrics_even_with_remote_clocks():
    from mlx2.batch_metrics import BatchRuntimeMetrics

    now = [100.0]
    metrics = BatchRuntimeMetrics(history_size=1, clock=lambda: now[0])
    metrics.admitted("chatcmpl-live", "alice", 0)
    metrics.dequeued("chatcmpl-live", 0)
    metrics.prompt("chatcmpl-live", 32768, 16384)
    metrics.lane_attached("chatcmpl-live", 1, "ordinary")
    now[0] = 102.0
    metrics.token("chatcmpl-live")
    now[0] = 103.0
    metrics.token("chatcmpl-live")
    now[0] = 104.0
    # Erase live's events without altering its authoritative state.
    metrics.admitted("chatcmpl-wait", "alice", 1)
    reading = build_reading(
        ServerEndpoint(url="http://remote:8297"),
        Metrics.parse(_EXPOSITION_BEFORE),
        now=9000.0,
        batching=metrics.snapshot(),
        clock_comparable=False,
    )
    live, waiting = reading.active_requests
    assert live.phase == "decode" and live.age_seconds == 4.0
    assert live.context_tokens == 32770 and live.prompt_tokens == 32768
    assert live.cached_tokens == 16384 and live.completion_tokens == 2
    assert live.prefill_tokens_per_second == 8192.0
    assert live.decode_tokens_per_second == 0.5
    assert waiting.context_tokens is None and waiting.decode_tokens_per_second is None
    output = render(_host_reading(), servers=(reading,), width=103)
    assert "Sessions  2 active" in output
    assert "32,770" in output and "avg pp" in output and "avg dec" in output
    assert "tenant alice" in output and "TTFT 2.00s" in output
    assert "Latency " not in output and "per-stream decode" not in output
    assert "In flight" not in output
    idle = render(_host_reading(), servers=(ServerReading(reading.endpoint, True),))
    assert "Sessions  none observed" in idle


def test_session_observations_stay_visible_idle_and_across_turns():
    from mlx2.batch_metrics import BatchRuntimeMetrics

    now = [100.0]
    metrics = BatchRuntimeMetrics(history_size=2, clock=lambda: now[0])
    endpoint = ServerEndpoint("http://127.0.0.1:8297")
    exposition = Metrics.parse(_EXPOSITION_BEFORE)
    metrics.admitted("turn1", "alice", 0, session_id="chat")
    metrics.dequeued("turn1", 0)
    metrics.prompt("turn1", 100, 40)
    metrics.lane_attached("turn1", 1, "ordinary")
    now[0] = 102.0
    metrics.token("turn1")
    now[0] = 103.0
    metrics.token("turn1")
    now[0] = 104.0
    metrics.terminal("turn1", "completed")
    finished = build_reading(endpoint, exposition, now=104.0, batching=metrics.snapshot())
    row, = finished.observed_sessions
    assert not row.active and row.phase == "completed" and row.session_id == "chat"
    assert row.context_tokens == 102 and row.prefill_tokens_per_second == 30.0
    assert row.decode_tokens_per_second == 0.5
    output = render(_host_reading(), servers=(finished,))
    assert "Sessions  0 active · 1 retained" in output and "last observed" in output

    # Poll gaps and history eviction retain the observed numbers, not zeroes.
    idle = build_reading(endpoint, exposition, now=500.0, batching={},
                         retained_sessions=finished.observed_sessions)
    assert idle.observed_sessions == finished.observed_sessions
    metrics.admitted("turn2", "alice", 0, session_id="chat")
    queued = build_reading(endpoint, exposition, now=500.0, batching=metrics.snapshot(),
                           retained_sessions=idle.observed_sessions)
    row, = queued.observed_sessions
    assert row.active and row.request_id == "turn2" and row.context_tokens == 102
    assert row.completion_tokens == 0  # an observed zero replaces the old count
    assert row.prefill_tokens_per_second == 30.0 and row.decode_tokens_per_second == 0.5
    assert "context_tokens" in row.last_observed_fields
    assert "last observed:" in render(_host_reading(), servers=(queued,))
    # A new prompt replaces context; throughput remains until newly measured.
    metrics.prompt("turn2", 120, 0)
    resumed = build_reading(endpoint, exposition, now=500.0, batching=metrics.snapshot(),
                            retained_sessions=queued.observed_sessions)
    row, = resumed.observed_sessions
    assert row.context_tokens == 120 and "context_tokens" not in row.last_observed_fields
    assert row.decode_tokens_per_second == 0.5
    # Same session string under another tenant is a distinct observation.
    metrics.admitted("foreign", "bob", 0, session_id="chat")
    scoped = build_reading(endpoint, exposition, now=500.0, batching=metrics.snapshot(),
                           retained_sessions=resumed.observed_sessions)
    assert len(scoped.observed_sessions) == 2
    assert next(r for r in scoped.observed_sessions if r.tenant_id == "bob").context_tokens is None


def test_retained_sessions_are_bounded_and_newest_turn_wins():
    completed = [{
        "request_id": f"turn-{i}", "tenant_id": "alice", "session_id": f"chat-{i}",
        "status": "completed", "context_tokens": i,
    } for i in range(520)]
    completed.extend([
        {"request_id": "old", "tenant_id": "alice", "session_id": "shared", "context_tokens": 10},
        {"request_id": "new", "tenant_id": "alice", "session_id": "shared", "context_tokens": 20},
    ])
    reading = build_reading(
        ServerEndpoint("http://127.0.0.1:8297"), Metrics.parse(_EXPOSITION_BEFORE), now=1.0,
        batching={"completed_requests": completed},
        status={"recent_receipts": [{"request_id": "new", "prompt_tokens": 20, "completion_tokens": 0}]},
    )
    assert len(reading.observed_sessions) == 512
    shared = [row for row in reading.observed_sessions if row.session_id == "shared"]
    assert len(shared) == 1 and shared[0].request_id == "new" and shared[0].context_tokens == 20
