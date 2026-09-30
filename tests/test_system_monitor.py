"""Pure parser, counter, and renderer tests for mlx2-top."""

from __future__ import annotations

import plistlib
import time

from mlx2.system_monitor import (
    CPUTicks,
    GPUReading,
    HostReading,
    NAXReading,
    cpu_busy_percent,
    parse_ioreg_gpus,
    parse_nax_status,
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


def test_powermetrics_parser_keeps_gpu_ane_and_thermal_distinct():
    parsed = parse_powermetrics_plist(
        {
            "gpu": {"idle_ratio": 0.25, "freq_hz": 1_200_000_000},
            "ane": [
                {
                    "ane-id": 2,
                    "idle_ratio": 0.8,
                    "freq_hz": 800_000_000,
                    "ane_power": 320,
                }
            ],
            "processor": {"gpu_power": 2500, "ane_power": 300},
            "thermal_pressure": "Nominal",
        }
    )
    assert parsed.gpu_busy_percent == 75.0
    assert parsed.gpu_frequency_mhz == 1200.0
    assert parsed.gpu_power_mw == 2500.0
    assert parsed.ane[0].index == 2
    assert round(parsed.ane[0].busy_percent, 6) == 20.0
    assert parsed.ane[0].frequency_mhz == 800.0
    assert parsed.thermal_pressure == "Nominal"


def test_nax_status_is_receipt_counts_not_a_fake_percentage():
    parsed = parse_nax_status(
        {
            "int8_prefill": {
                "active": True,
                "counts": {"engaged_calls": 9, "engaged_rows": 2048},
            },
            "execution": {
                "indexed_qsa": {"counts": {"nax_engaged": 4}}
            },
        },
        "http://127.0.0.1:8285",
    )
    assert parsed.reachable is True
    assert parsed.active is True
    assert parsed.int8_calls_total == 9
    assert parsed.int8_rows_total == 2048
    assert parsed.qsa_engagements_total == 4


def test_renderer_calls_out_unavailable_nax_hardware_counter():
    reading = HostReading(
        sampled_at=time.time(),
        interval_seconds=1.0,
        cpu_busy_percent=(10.0, 20.0),
        load_average=(1.0, 2.0, 3.0),
        memory=None,
        swap=None,
        gpus=(),
        ane=(),
        nax=NAXReading(
            service="http://127.0.0.1:8285",
            reachable=True,
            active=True,
            int8_calls_total=10,
            int8_rows_total=512,
            int8_calls_delta=1,
            int8_rows_delta=64,
        ),
        thermal_state="nominal",
        thermal_pressure=None,
    )
    output = render(reading)
    assert "NAX HW    utilization unavailable" in output
    assert "Δ calls 1 / rows 64" in output
    assert "CPU 00" in output
    assert "Load 1.00 2.00 3.00" in output


def test_format_bytes_uses_binary_units_and_rate_suffix():
    assert format_bytes(1536) == "1.5 KiB"
    assert format_bytes(2 << 30) == "2.0 GiB"
    assert format_bytes(3 << 20, rate=True) == "3.0 MiB/s"
