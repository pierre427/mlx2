"""A top-like Apple-Silicon monitor for long-running mlx2 tests."""

from __future__ import annotations

import argparse
import os
import select
import shutil
import sys
import termios
import time
import tty
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path

from .clients import DEFAULT_URL
from .inference_monitor import (
    CompletedRequest,
    InferenceCollector,
    LatencyStat,
    ServerReading,
)
from .system_monitor import HostCollector, HostReading

GIB = 1 << 30
MIB = 1 << 20


def format_bytes(value: float | None, *, rate: bool = False) -> str:
    if value is None:
        return "n/a"
    amount = float(value)
    units = ((1 << 40, "TiB"), (GIB, "GiB"), (MIB, "MiB"), (1 << 10, "KiB"))
    for scale, label in units:
        if abs(amount) >= scale:
            precision = 1 if abs(amount) < 10 * scale else 0
            return f"{amount / scale:.{precision}f} {label}{'/s' if rate else ''}"
    return f"{amount:.0f} B{'/s' if rate else ''}"


def _bar(percent: float | None, width: int = 10) -> str:
    if percent is None:
        return "[" + "·" * width + "]"
    count = max(0, min(width, round(percent * width / 100.0)))
    return "[" + "█" * count + "░" * (width - count) + "]"


def _percent(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:5.1f}%"


def _join_present(parts: list[str | None], separator: str = "  ") -> str:
    return separator.join(part for part in parts if part)


def _seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 0.001:
        return f"{value * 1_000_000:.0f}µs"
    if value < 1.0:
        return f"{value * 1000:.1f}ms" if value < 0.1 else f"{value * 1000:.0f}ms"
    if value < 120:
        return f"{value:.2f}s"
    return _duration(value)


def _duration(value: float | None) -> str:
    if value is None:
        return "n/a"
    seconds = int(value)
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


def _count(value: float | None) -> str:
    if value is None:
        return "n/a"
    amount = float(value)
    for scale, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(amount) >= 10 * scale:
            return f"{amount / scale:.0f}{suffix}" if abs(amount) >= 100 * scale else f"{amount / scale:.1f}{suffix}"
    return f"{amount:,.0f}"


def _tps(value: float | None) -> str:
    if value is None:
        return "  —"
    return f"{value:,.1f}" if value < 1000 else _count(value)


def _ratio(part: float | None, whole: float | None) -> str:
    if not whole or part is None:
        return "n/a"
    return f"{100.0 * part / whole:.0f}%"


def _share(part: float, whole: float) -> str:
    share = 100.0 * part / whole
    return "<1%" if 0 < share < 0.5 else f"{share:.0f}%"


def _short_id(request_id: str) -> str:
    return request_id.rpartition("-")[2][:8] or request_id[:8]


def _latency_row(name: str, stat: LatencyStat) -> str:
    marker = " " if stat.exact else "~"
    window = (
        f"{_seconds(stat.window_p50):>8} {_seconds(stat.window_p95):>8} {stat.window_count:>6}"
        if stat.window_count
        else f"{'—':>8} {'—':>8} {0:>6}"
    )
    return (
        f"  {name:<8}{marker}{_seconds(stat.p50):>8} {_seconds(stat.p95):>8} "
        f"{_seconds(stat.p99):>8} {_seconds(stat.average):>8} {stat.count:>7}  │ {window}"
    )


def _request_row(request: CompletedRequest) -> str:
    def number(value: int | None) -> str:
        return "—" if value is None else _count(value)

    decode = request.decode_tokens_per_second
    accept = request.draft_acceptance
    return (
        f"  {_short_id(request.request_id):<9}{(request.status or '—')[:9]:<10}"
        f"{number(request.prompt_tokens):>7}{number(request.cached_tokens):>8}"
        f"{number(request.completion_tokens):>7}{_seconds(request.ttft_seconds):>9}"
        f"{_seconds(request.elapsed_seconds):>9}{_tps(decode):>9}"
        f"{'—' if accept is None else f'{100 * accept:.0f}%':>8}"
        f"{'—' if request.tokens_per_cycle is None else f'{request.tokens_per_cycle:.2f}':>8}"
        f"  {'think ' if request.thinking else ''}{request.route or ''}"
    )


def render_server(server: ServerReading, *, width: int = 100) -> list[str]:
    """Render one inference server's panel."""
    endpoint = server.endpoint
    where = endpoint.url.removeprefix("http://")
    title = f"── LLM  {server.model or endpoint.model_path or 'unknown model'}  ·  {where}"
    if endpoint.pid is not None:
        title += f"  pid {endpoint.pid}"
    lines = [(title + " ").ljust(min(width, 120), "─")]
    if not server.reachable:
        lines.append(f"  unreachable — {server.error or 'no response'}")
        return lines

    readiness = (
        server.state or ("ready" if server.ready else "not ready" if server.ready is not None else None)
    )
    lines.append(
        "Route     "
        + _join_present(
            [
                server.profile,
                server.route,
                server.qualification,
                readiness,
                f"up {_duration(server.uptime_seconds)}" if server.uptime_seconds is not None else None,
            ],
            " · ",
        )
    )
    selected = len(server.selected_capabilities)
    lines.append(
        "Build     "
        + _join_present(
            [
                server.revision,
                f"mlx {server.mlx_version}" if server.mlx_version else None,
                f"ctx {server.max_context:,}" if server.max_context else None,
                f"lanes {server.max_lanes}" if server.max_lanes else None,
                f"inflight {server.max_inflight}" if server.max_inflight else None,
                f"qualified caps {len(server.qualified_capabilities)}/{selected}" if selected else None,
                f"status stale ({server.status_error})" if server.status_error else None,
            ],
            " · ",
        )
    )

    lane_percent = (
        None
        if not server.max_lanes or server.active_lanes is None
        else 100.0 * server.active_lanes / server.max_lanes
    )
    lines.append(
        "Load      "
        + _join_present(
            [
                f"running {server.running if server.running is not None else 'n/a'}",
                f"waiting {server.waiting if server.waiting is not None else 'n/a'}",
                (f"lanes {_bar(lane_percent, 8)} {server.active_lanes if server.active_lanes is not None else '?'}"
                f"/{server.max_lanes or '?'} (peak {server.peak_active_lanes if server.peak_active_lanes is not None else '?'})"),
                f"batch {server.batch_width_window:.2f} 30s" if server.batch_width_window is not None else None,
                f"{server.batch_width_lifetime:.2f} life" if server.batch_width_lifetime is not None else None,
                f"mem-wait {server.memory_waiting}" if server.memory_waiting else None,
                f"{server.scheduler_cycles_per_second:.0f} cyc/s"
                if server.scheduler_cycles_per_second is not None
                else None,
            ],
            " · ",
        )
    )
    if server.batch_composition:
        total = sum(count for _, count in server.batch_composition) or 1
        lines.append(
            "Batch mix "
            + " · ".join(
                f"w{width_} {_share(count, total)}" for width_, count in server.batch_composition
            )
            + f"  ({_count(total)} cycles)"
        )

    lines.append(
        "Tokens/s  "
        + _join_present(
            [
                f"gen {_tps(server.generation_tps).strip()} now / {_tps(server.generation_tps_window).strip()} 30s",
                f"prompt {_tps(server.prompt_tps).strip()} now / {_tps(server.prompt_tps_window).strip()} 30s",
                f"cached {_tps(server.cached_prompt_tps_window).strip()} 30s",
                f"{server.requests_per_second_window:.2f} req/s"
                if server.requests_per_second_window is not None
                else None,
            ],
            "  ·  ",
        )
    )
    lines.append(
        "Totals    "
        + _join_present(
            [
                f"generated {_count(server.generation_tokens_total)}",
                f"prompt {_count(server.prompt_tokens_total)}",
                (f"cached {_count(server.cached_prompt_tokens_total)} "
                f"({_ratio(server.cached_prompt_tokens_total, server.prompt_tokens_total)})"),
            ],
            " · ",
        )
    )
    if server.outcomes:
        lines.append(
            "Requests  "
            + f"{sum(count for _, count in server.outcomes):,} done: "
            + " · ".join(f"{name} {count:,}" for name, count in server.outcomes)
        )
    if server.tenant_token_rates:
        tenants = " · ".join(f"{name} {rate:.1f} tok/s" for name, rate in server.tenant_token_rates[:4])
        jain = f"  Jain {server.jain_fairness:.2f}" if server.jain_fairness is not None else ""
        lines.append(f"Tenants   {tenants}{jain}")

    lines.append(
        f"Latency   {'':<1}{'p50':>8} {'p95':>8} {'p99':>8} {'avg':>8} {'n':>7}  │ "
        f"{'30s p50':>8} {'p95':>8} {'n':>6}"
    )
    for name, stat in server.latency:
        if stat.count or stat.window_count:
            lines.append(_latency_row(name, stat))
    itl = dict(server.latency).get("ITL")
    if itl is not None and itl.p50:
        lines.append(f"          per-stream decode ≈ {1.0 / itl.p50:.1f} tok/s at ITL p50  (~ = bucket estimate)")

    spec = server.speculation
    if spec.cycles or spec.receipt_requests or spec.mode:
        lines.append(
            "Spec      "
            + _join_present(
                [
                    f"{spec.mode or 'speculative'}"
                    + (f" draft {spec.num_draft}" if spec.num_draft else ""),
                    f"verify cycles {_count(spec.cycles)}: all {_ratio(spec.accepted_all, spec.cycles)}"
                    f" · partial {_ratio(spec.accepted_partial, spec.cycles)}"
                    f" · zero {_ratio(spec.accepted_zero, spec.cycles)}"
                    if spec.cycles
                    else None,
                    f"30s all-accept {_ratio(spec.window_accepted_all, spec.window_cycles)}"
                    if spec.window_cycles
                    else None,
                ],
                "  ·  ",
            )
        )
        if spec.receipt_requests:
            lines.append(
                "          "
                + _join_present(
                    [
                        (f"last {spec.receipt_requests} receipts: draft accept "
                        f"{_ratio(spec.receipt_accepted, spec.receipt_proposed)}"
                        f" ({_count(spec.receipt_accepted)}/{_count(spec.receipt_proposed)})"),
                        f"{spec.receipt_emitted / spec.receipt_cycles:.2f} tok/cycle"
                        if spec.receipt_cycles
                        else None,
                        f"copy-draft {_ratio(spec.copy_accepted, spec.copy_proposed)}"
                        f" ({_count(spec.copy_accepted)}/{_count(spec.copy_proposed)})"
                        if spec.copy_proposed
                        else None,
                    ],
                    "  ·  ",
                )
            )

    if server.cache_lookups is not None:
        usage = (
            None
            if not server.cache_capacity_bytes or server.cache_resident_bytes is None
            else 100.0 * server.cache_resident_bytes / server.cache_capacity_bytes
        )
        lines.append(
            "APCv2     "
            + _join_present(
                [
                    (f"hit {_ratio(server.cache_hits, server.cache_lookups)}"
                    f" ({_count(server.cache_hits)}/{_count(server.cache_lookups)})"),
                    f"30s {_ratio(server.cache_window_hits, server.cache_window_lookups)}"
                    if server.cache_window_lookups
                    else None,
                    f"token reuse {_ratio(server.cached_prompt_tokens_total, server.cache_query_tokens)}",
                    (f"resident {_bar(usage, 8)} {format_bytes(server.cache_resident_bytes)}"
                    f" / {format_bytes(server.cache_capacity_bytes)}"),
                    f"{_count(server.cache_entries)} entries",
                ],
                " · ",
            )
        )
        host_cache = server.host_prompt_cache or {}
        lines.append(
            "          "
            + _join_present(
                [
                    f"disk {_count(server.cache_disk_entries)} entries {format_bytes(server.cache_disk_bytes)}"
                    if server.cache_disk_entries
                    else None,
                    f"interior {_count(server.cache_interior_hits)}",
                    f"rolling {_count(server.cache_rolling_hits)}",
                    f"junction {_count(server.cache_junction_hits)}",
                    f"evictions {_count(server.cache_evictions_window)} 30s"
                    if server.cache_evictions_window
                    else None,
                    f"host prompt cache {_count(host_cache.get('hits'))} hits / "
                    f"{_count(host_cache.get('entries'))} entries"
                    if host_cache
                    else None,
                ],
                " · ",
            )
        )

    lines.append(
        "Memory    "
        + _join_present(
            [
                f"metal {format_bytes(server.metal_active_bytes)} (peak {format_bytes(server.metal_peak_bytes)})",
                f"footprint {format_bytes(server.footprint_bytes)}",
                f"headroom {format_bytes(server.headroom_bytes)}",
                f"weights {format_bytes(server.weights_bytes)} wired" if server.weights_bytes else None,
                f"limit {format_bytes(server.wired_limit_bytes)}" if server.wired_limit_bytes else None,
                f"host pressure {int(server.host_pressure_level)}"
                if server.host_pressure_level
                else None,
            ],
            " · ",
        )
    )
    lines.append(
        "HTTP      "
        + _join_present(
            [
                f"{server.http_requests_per_second:.1f} req/s 30s"
                if server.http_requests_per_second is not None
                else None,
                "errors " + " · ".join(f"{name} {count:,}" for name, count in server.http_errors[:4])
                if server.http_errors
                else "no 4xx/5xx",
            ],
            "  ·  ",
        )
    )
    lines.append(
        "Failures  "
        + (
            " · ".join(f"{name} {count:,}" for name, count in server.failures[:5])
            if server.failures
            else "none (fail-closed, rejected admissions, export errors)"
        )
    )

    if server.mechanisms:
        lines.append("Hot paths                                                  Δ/s     total")
        for item in server.mechanisms[:8]:
            lines.append(
                f"  {item.name[:52]:<52} {_count(item.per_second) if item.per_second else '·':>9} "
                f"{_count(item.total):>9}"
            )

    if server.active_requests:
        lines.append("In flight  id       phase    age      queue     TTFT     mechanism")
        for request in server.active_requests[:8]:
            lines.append(
                f"  {_short_id(request.request_id):<9}{request.phase:<9}"
                f"{_seconds(request.age_seconds):>7}"
                f"{'—' if request.queue_ms is None else _seconds(request.queue_ms / 1000):>9}"
                f"{'—' if request.ttft_ms is None else _seconds(request.ttft_ms / 1000):>9}"
                f"  {request.mechanism or ''}"
            )
    else:
        lines.append("In flight  none")

    if server.recent_requests:
        age = (
            f" (status {server.status_age_seconds:.0f}s old)"
            if server.status_age_seconds and server.status_age_seconds >= 1.5
            else ""
        )
        lines.append(
            f"Recent{age}"
        )
        lines.append(
            f"  {'id':<9}{'status':<10}{'prompt':>7}{'cached':>8}{'out':>7}{'TTFT':>9}"
            f"{'total':>9}{'dec t/s':>9}{'accept':>8}{'tok/cyc':>8}  route"
        )
        lines.extend(_request_row(request) for request in server.recent_requests)
    return lines


def render(
    reading: HostReading,
    *,
    servers: tuple[ServerReading, ...] | None = (),
    width: int = 100,
) -> str:
    """Render one self-contained snapshot without terminal control codes.

    ``servers`` is ``None`` when the inference panel is disabled.
    """
    width = max(60, width)
    timestamp = datetime.fromtimestamp(
        reading.sampled_at, UTC
    ).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    load = reading.load_average
    cpu = reading.cpu_busy_percent
    average = sum(cpu) / len(cpu) if cpu else None
    lines = [
        f"mlx2-top  {timestamp}  sample {reading.interval_seconds:.2f}s",
        _join_present(
            [
                f"Load {load[0]:.2f} {load[1]:.2f} {load[2]:.2f}",
                f"CPU {_percent(average).strip()} busy",
                f"Thermal {reading.thermal_state}",
                (
                    f"pressure {reading.thermal_pressure.lower()}"
                    if reading.thermal_pressure
                    else None
                ),
            ],
            "  ·  ",
        ),
    ]

    if servers is not None:
        if servers:
            for server in servers:
                lines.append("")
                lines.extend(render_server(server, width=width))
        else:
            lines.extend(
                [
                    "",
                    ("── LLM  no running mlx2 server found "
                    "(looked for mlx2.server listeners; pass --url to add one)"),
                ]
            )

    lines.extend(["", "CPU busy per logical CPU"])

    if cpu:
        cell_width = 24
        columns = max(1, min(4, width // cell_width))
        cells = [
            f"CPU {index:02d} {_bar(value, 8)} {value:5.1f}%"
            for index, value in enumerate(cpu)
        ]
        for start in range(0, len(cells), columns):
            lines.append("  ".join(cell.ljust(cell_width) for cell in cells[start : start + columns]).rstrip())
    else:
        lines.append("  unavailable (Mach host_processor_info failed)")

    lines.extend(["", "GPU busy per device"])
    if reading.gpus:
        for index, gpu in enumerate(reading.gpus):
            extras = _join_present(
                [
                    f"render {_percent(gpu.renderer_percent).strip()}"
                    if gpu.renderer_percent is not None
                    else None,
                    f"tiler {_percent(gpu.tiler_percent).strip()}"
                    if gpu.tiler_percent is not None
                    else None,
                    f"{gpu.frequency_mhz:.0f} MHz"
                    if gpu.frequency_mhz is not None
                    else None,
                    f"{gpu.power_mw:.0f} mW" if gpu.power_mw is not None else None,
                ],
                "  ·  ",
            )
            lines.append(
                f"GPU {index:02d} {_bar(gpu.busy_percent)} {_percent(gpu.busy_percent)}  "
                f"{gpu.name}  [{gpu.source}]"
            )
            if extras:
                lines.append(f"       {extras}")
            if gpu.memory_bytes is not None or gpu.allocated_bytes is not None:
                lines.append(
                    "       driver memory "
                    f"in-use {format_bytes(gpu.memory_bytes)}  ·  "
                    f"allocated {format_bytes(gpu.allocated_bytes)}"
                )
    else:
        lines.append("  unavailable (no AGXAccelerator utilization counters)")
    if reading.gpus and reading.gpus[0].power_mw is None:
        reason = reading.soc_power_error or reading.powermetrics_error
        if reason:
            lines.append(f"       power/frequency unavailable — {reason}")

    soc = reading.soc_power
    if soc is not None:

        def watts(value: float | None) -> str | None:
            return None if value is None else f"{value / 1000.0:.2f} W"

        lines.extend(["", "Power (IOReport, unprivileged)"])
        lines.append(
            "SoC       "
            + _join_present(
                [
                    f"GPU {watts(soc.gpu_power_mw)}" if soc.gpu_power_mw is not None else None,
                    f"DRAM {watts(soc.dram_power_mw)}" if soc.dram_power_mw is not None else None,
                    f"CPU {watts(soc.cpu_power_mw)}" if soc.cpu_power_mw is not None else None,
                    f"ANE {watts(soc.ane_power_mw)}" if soc.ane_power_mw is not None else None,
                ],
                "  ·  ",
            )
        )
        dvfs = _join_present(
            [
                f"active {_percent(soc.gpu_active_percent).strip()}"
                if soc.gpu_active_percent is not None
                else None,
                f"mean P{soc.gpu_mean_pstate:.1f}" if soc.gpu_mean_pstate is not None else None,
                f"{soc.gpu_frequency_mhz:.0f} MHz" if soc.gpu_frequency_mhz is not None else None,
            ],
            "  ·  ",
        )
        if dvfs:
            lines.append(f"GPU DVFS  {dvfs}")

    lines.extend(["", "Memory"])
    memory = reading.memory
    if memory is None:
        lines.append("  unavailable (Mach HOST_VM_INFO64 failed)")
    else:
        lines.append(
            f"Physical  used {format_bytes(memory.used_bytes)} / "
            f"{format_bytes(memory.total_bytes)}  ·  "
            f"available {format_bytes(memory.available_bytes)}  ·  "
            f"pressure {memory.pressure}"
        )
        lines.append(
            f"Pages     active {format_bytes(memory.active_bytes)}  ·  "
            f"inactive {format_bytes(memory.inactive_bytes)}  ·  "
            f"free {format_bytes(memory.free_bytes)}  ·  "
            f"speculative {format_bytes(memory.speculative_bytes)}"
        )
        lines.append(
            f"Pinned    wired {format_bytes(memory.wired_bytes)}  ·  "
            f"compressed {format_bytes(memory.compressed_bytes)} "
            f"(represents {format_bytes(memory.uncompressed_in_compressor_bytes)})"
        )
        lines.append(
            f"Reclaim   file-backed {format_bytes(memory.file_backed_bytes)}  ·  "
            f"purgeable {format_bytes(memory.purgeable_bytes)}  ·  "
            f"anonymous {format_bytes(memory.anonymous_bytes)}"
        )
        lines.append(
            f"VM rates  page-in {memory.pageins_per_second:,.0f}/s  ·  "
            f"page-out {memory.pageouts_per_second:,.0f}/s  ·  "
            f"compress {memory.compressed_per_second:,.0f}/s  ·  "
            f"decompress {memory.decompressed_per_second:,.0f}/s"
        )

    swap = reading.swap
    if swap is None:
        lines.append("Swap      unavailable")
    else:
        percent = 0.0 if swap.total_bytes == 0 else 100 * swap.used_bytes / swap.total_bytes
        lines.append(
            f"Swap      {_bar(percent)} {percent:5.1f}%  "
            f"used {format_bytes(swap.used_bytes)} / {format_bytes(swap.total_bytes)}  ·  "
            f"free {format_bytes(swap.free_bytes)}  ·  "
            f"{'encrypted' if swap.encrypted else 'not encrypted'}"
        )
        lines.append(
            f"Swap I/O  in {format_bytes(swap.swapins_per_second, rate=True)}  ·  "
            f"out {format_bytes(swap.swapouts_per_second, rate=True)}  ·  "
            f"lifetime in {format_bytes(swap.swapins_bytes)}  ·  "
            f"out {format_bytes(swap.swapouts_bytes)}"
        )

    lines.extend(
        [
            "",
            "Thermals",
            f"State     {reading.thermal_state} (NSProcessInfo)",
            (
                f"Pressure  {reading.thermal_pressure} (powermetrics)"
                if reading.thermal_pressure
                else "Pressure  unavailable without a privileged powermetrics sample"
            ),
            (
                f"Sensors   die max {soc.die_max_c:.1f} °C  ·  mean {soc.die_mean_c:.1f} °C (HID)"
                if soc is not None
                and soc.die_max_c is not None
                and soc.die_mean_c is not None
                else "Sensors   die temperatures unavailable"
                + (f" — {reading.soc_power_error}" if reading.soc_power_error else "")
            ),
        ]
    )
    for warning in reading.warnings:
        lines.append(f"WARNING   {warning}")
    return "\n".join(lines)


class _Terminal(AbstractContextManager):
    """Full-screen redraw with a scrollable viewport when interactive."""

    def __init__(self):
        self.interactive = sys.stdin.isatty() and sys.stdout.isatty()
        self._settings = None
        self._text = ""
        self.offset = 0

    def __enter__(self):
        if self.interactive:
            descriptor = sys.stdin.fileno()
            self._settings = termios.tcgetattr(descriptor)
            tty.setcbreak(descriptor)
            sys.stdout.write("\x1b[?25l")
            sys.stdout.flush()
        return self

    def wait(self, seconds: float) -> bool:
        """Wait for the next sample, redrawing on scroll keys.

        Returns false when q was pressed.
        """
        if not self.interactive:
            time.sleep(seconds)
            return True
        deadline = time.monotonic() + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            readable, _, _ = select.select([sys.stdin], [], [], remaining)
            if not readable:
                return True
            key = sys.stdin.read(1)
            if key.lower() == "q":
                return False
            page = max(1, shutil.get_terminal_size((100, 30)).lines - 3)
            step = {"j": 1, "k": -1, " ": page, "b": -page, "g": -10**9, "G": 10**9}.get(key)
            if step is not None:
                self.offset = max(0, self.offset + step)
                self._draw()

    def display(self, text: str) -> None:
        self._text = text
        if self.interactive:
            self._draw()
        else:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()

    def _draw(self) -> None:
        size = shutil.get_terminal_size((100, 30))
        lines = [line[: size.columns] for line in self._text.split("\n")]
        rows = max(1, size.lines - 2)
        self.offset = min(self.offset, max(0, len(lines) - rows))
        view = lines[self.offset : self.offset + rows]
        more = len(lines) - self.offset - len(view)
        footer = "q quit  ·  j/k scroll  ·  space/b page  ·  g/G top/bottom"
        if self.offset or more:
            footer += f"  ·  lines {self.offset + 1}-{self.offset + len(view)} of {len(lines)}"
        sys.stdout.write("\x1b[H\x1b[2J" + "\n".join(view) + "\n\n" + footer[: size.columns])
        sys.stdout.flush()

    def __exit__(self, exc_type, exc, traceback):
        if self.interactive:
            if self._settings is not None:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._settings)
            sys.stdout.write("\x1b[?25h\n")
            sys.stdout.flush()
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx2-top",
        description="Top-like Apple-Silicon and mlx2 inference telemetry for LLM testing.",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1.0,
        help="sample interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "-1",
        "--once",
        action="store_true",
        help="print one sample and exit",
    )
    parser.add_argument(
        "--ioreport",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="read SoC power, GPU DVFS residency and die temperatures without "
        "privilege from IOReport and the HID sensors (default: on)",
    )
    parser.add_argument(
        "--powermetrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also run privileged powermetrics (needs cached sudo) for GPU "
        "busy, frequency and thermal pressure; it overrides IOReport's GPU "
        "power and frequency when it reports them (default: off)",
    )
    parser.add_argument(
        "--mlx2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the inference panel for running mlx2 servers",
    )
    parser.add_argument(
        "--discover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="find listening mlx2.server processes on this host (default: on)",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="mlx2 server URL to monitor; repeatable.  Without --url or discovered "
        f"servers, MLX2_URL or {DEFAULT_URL} is tried",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=5.0,
        help="seconds between full /v1/status reads for receipts and identity "
        "(default: 5; the payload is large)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=8,
        help="recent requests to list per server (default: 8)",
    )
    parser.add_argument(
        "--api-key-file",
        help="file holding the mlx2 API key (default: MLX2_API_KEY)",
    )
    return parser


def _api_key(args, environment: dict[str, str]) -> str | None:
    if args.api_key_file:
        try:
            return Path(args.api_key_file).read_text(encoding="utf-8").strip()
        except OSError as error:
            raise ValueError(f"cannot read API key file: {error}") from error
    return environment.get("MLX2_API_KEY")


def main(argv=None, *, environment=None) -> int:
    environment = dict(os.environ if environment is None else environment)
    args = build_parser().parse_args(argv)
    if not 0.1 <= args.interval <= 60.0:
        print("error: --interval must be between 0.1 and 60 seconds", file=sys.stderr)
        return 2
    if args.status_interval < 0.5:
        print("error: --status-interval must be at least 0.5 seconds", file=sys.stderr)
        return 2
    try:
        api_key = _api_key(args, environment)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    once = bool(args.once or not sys.stdout.isatty())
    collector = HostCollector(
        interval_seconds=args.interval,
        powermetrics=args.powermetrics,
        ioreport=args.ioreport,
    )
    inference = None
    if args.mlx2:
        inference = InferenceCollector(
            urls=tuple(args.url),
            fallback_urls=(environment.get("MLX2_URL") or DEFAULT_URL,),
            discover=args.discover,
            api_key=api_key,
            status_every_seconds=args.status_interval,
            recent_requests=max(0, args.requests),
        )
        inference.sample()  # prime counters so the first frame has rates
    try:
        with _Terminal() as terminal:
            while terminal.wait(args.interval):
                reading = collector.sample()
                servers = inference.sample() if inference is not None else None
                terminal.display(
                    render(
                        reading,
                        servers=servers,
                        width=shutil.get_terminal_size((100, 30)).columns,
                    )
                )
                if once:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        collector.close()
        if inference is not None:
            inference.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
