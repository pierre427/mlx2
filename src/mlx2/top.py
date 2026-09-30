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


def render(reading: HostReading, *, width: int = 100) -> str:
    """Render one self-contained snapshot without terminal control codes."""
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
        "",
        "CPU busy per logical CPU",
    ]

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

    lines.extend(["", "Accelerators"])
    if reading.ane:
        for ane in reading.ane:
            detail = _join_present(
                [
                    f"{ane.frequency_mhz:.0f} MHz"
                    if ane.frequency_mhz is not None
                    else None,
                    f"{ane.power_mw:.0f} mW" if ane.power_mw is not None else None,
                ],
                "  ·  ",
            )
            lines.append(
                f"ANE {ane.index:02d} {_bar(ane.busy_percent)} {_percent(ane.busy_percent)}"
                + (f"  {detail}" if detail else "")
            )
    else:
        reason = reading.powermetrics_error or "powermetrics has not sampled yet"
        lines.append(f"ANE       unavailable — {reason}")

    # NAX is an M5 GPU matrix facility, not the separate Apple Neural Engine.
    # Keep the absent hardware percentage separate from mlx2 route evidence.
    lines.append("NAX HW    utilization unavailable — macOS publishes no NAX busy counter")
    nax = reading.nax
    if nax.reachable:
        lines.append(
            "NAX mlx2  "
            f"int8 {'active' if nax.active else 'inactive'}  ·  "
            f"Δ calls {nax.int8_calls_delta:,} / rows {nax.int8_rows_delta:,}  ·  "
            f"Δ QSA engagements {nax.qsa_engagements_delta:,}"
        )
        lines.append(
            "           totals "
            f"calls {nax.int8_calls_total:,} / rows {nax.int8_rows_total:,} / "
            f"QSA {nax.qsa_engagements_total:,}  [{nax.service}]"
        )
    else:
        lines.append(f"NAX mlx2  no route receipts — {nax.error or 'service unavailable'}")

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
            "Sensors   die temperatures unavailable through the public interfaces used here",
        ]
    )
    for warning in reading.warnings:
        lines.append(f"WARNING   {warning}")
    return "\n".join(lines)


class _Terminal(AbstractContextManager):
    def __init__(self):
        self.interactive = sys.stdin.isatty() and sys.stdout.isatty()
        self._settings = None

    def __enter__(self):
        if self.interactive:
            descriptor = sys.stdin.fileno()
            self._settings = termios.tcgetattr(descriptor)
            tty.setcbreak(descriptor)
            sys.stdout.write("\x1b[?25l")
            sys.stdout.flush()
        return self

    def wait(self, seconds: float) -> bool:
        """Wait for the next sample; return false when q was pressed."""
        if not self.interactive:
            time.sleep(seconds)
            return True
        readable, _, _ = select.select([sys.stdin], [], [], seconds)
        if not readable:
            return True
        return sys.stdin.read(1).lower() != "q"

    def display(self, text: str) -> None:
        if self.interactive:
            sys.stdout.write("\x1b[H\x1b[2J" + text + "\n\nq quit  ·  Ctrl-C quit\n")
        else:
            sys.stdout.write(text + "\n")
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
        description="Top-like Apple-Silicon telemetry for LLM testing.",
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
        "--powermetrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use privileged GPU/ANE/thermal sampling when sudo is cached",
    )
    parser.add_argument(
        "--mlx2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="poll local mlx2 route counters for NAX engagement",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("MLX2_URL") or DEFAULT_URL,
        help=f"mlx2 service URL (default: MLX2_URL or {DEFAULT_URL})",
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
    try:
        api_key = _api_key(args, environment)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    once = bool(args.once or not sys.stdout.isatty())
    collector = HostCollector(
        interval_seconds=args.interval,
        powermetrics=args.powermetrics,
        mlx2_url=args.url if args.mlx2 else None,
        api_key=api_key,
    )
    try:
        with _Terminal() as terminal:
            while terminal.wait(args.interval):
                reading = collector.sample()
                terminal.display(
                    render(reading, width=shutil.get_terminal_size((100, 30)).columns)
                )
                if once:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        collector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
