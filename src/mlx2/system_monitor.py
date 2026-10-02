"""Read-only macOS host telemetry for :mod:`mlx2.top`.

The collectors deliberately distinguish public host counters from optional
privileged ``powermetrics`` samples.  Inference-server telemetry lives in
:mod:`mlx2.inference_monitor`.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import plistlib
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .runtime.os_memory import estimate_host_available_bytes

_CPU_STATE_MAX = 4
_PROCESSOR_CPU_LOAD_INFO = 2
_HOST_VM_INFO64 = 4
_HOST_VM_INFO64_REV1_COUNT = 38
_KERN_SUCCESS = 0


class _VMStatistics64(ctypes.Structure):
    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


class _XSWUsage(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_uint64),
        ("available", ctypes.c_uint64),
        ("used", ctypes.c_uint64),
        ("page_size", ctypes.c_uint32),
        ("encrypted", ctypes.c_bool),
    ]


@dataclass(frozen=True)
class CPUTicks:
    user: int
    system: int
    idle: int
    nice: int


@dataclass(frozen=True)
class MemoryReading:
    total_bytes: int
    available_bytes: int
    free_bytes: int
    active_bytes: int
    inactive_bytes: int
    speculative_bytes: int
    wired_bytes: int
    compressed_bytes: int
    uncompressed_in_compressor_bytes: int
    file_backed_bytes: int
    anonymous_bytes: int
    purgeable_bytes: int
    pressure: str
    pageins_per_second: float
    pageouts_per_second: float
    compressed_per_second: float
    decompressed_per_second: float

    @property
    def used_bytes(self) -> int:
        return max(0, self.total_bytes - self.available_bytes)


@dataclass(frozen=True)
class SwapReading:
    total_bytes: int
    used_bytes: int
    free_bytes: int
    encrypted: bool
    swapins_bytes: int
    swapouts_bytes: int
    swapins_per_second: float
    swapouts_per_second: float


@dataclass(frozen=True)
class GPUReading:
    name: str
    busy_percent: float | None
    renderer_percent: float | None = None
    tiler_percent: float | None = None
    memory_bytes: int | None = None
    allocated_bytes: int | None = None
    frequency_mhz: float | None = None
    power_mw: float | None = None
    source: str = "ioreg"


@dataclass(frozen=True)
class PowermetricsReading:
    gpu_busy_percent: float | None = None
    gpu_frequency_mhz: float | None = None
    gpu_power_mw: float | None = None
    thermal_pressure: str | None = None


@dataclass(frozen=True)
class HostReading:
    sampled_at: float
    interval_seconds: float
    cpu_busy_percent: tuple[float, ...]
    load_average: tuple[float, float, float]
    memory: MemoryReading | None
    swap: SwapReading | None
    gpus: tuple[GPUReading, ...]
    thermal_state: str
    thermal_pressure: str | None
    powermetrics_error: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class _VMRaw:
    page_size: int
    total_bytes: int
    free: int
    active: int
    inactive: int
    wired: int
    speculative: int
    compressor: int
    uncompressed_in_compressor: int
    file_backed: int
    anonymous: int
    purgeable: int
    pageins: int
    pageouts: int
    compressions: int
    decompressions: int
    swapins: int
    swapouts: int


def cpu_busy_percent(
    previous: tuple[CPUTicks, ...], current: tuple[CPUTicks, ...]
) -> tuple[float, ...]:
    """Return busy percentages from two per-CPU Mach tick readings."""
    if len(previous) != len(current):
        return ()
    result = []
    for before, after in zip(previous, current):
        user = max(0, after.user - before.user)
        system = max(0, after.system - before.system)
        idle = max(0, after.idle - before.idle)
        nice = max(0, after.nice - before.nice)
        total = user + system + idle + nice
        result.append(0.0 if total == 0 else 100.0 * (user + system + nice) / total)
    return tuple(result)


def _rate(after: int, before: int, seconds: float, multiplier: int = 1) -> float:
    if seconds <= 0 or after < before:
        return 0.0
    return (after - before) * multiplier / seconds


def _ratio_to_busy(value: Any) -> float | None:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return None
    if ratio > 1.0:
        ratio /= 100.0
    return max(0.0, min(100.0, (1.0 - ratio) * 100.0))


def parse_ioreg_gpus(payload: bytes) -> tuple[GPUReading, ...]:
    """Parse ``ioreg -a`` output without treating IOService busy as GPU use."""
    try:
        entries = plistlib.loads(payload)
    except Exception:  # noqa: BLE001 - an optional probe must fail closed
        return ()
    result = []
    for entry in entries if isinstance(entries, list) else ():
        stats = entry.get("PerformanceStatistics") or {}
        if "Device Utilization %" not in stats:
            continue
        result.append(
            GPUReading(
                name=str(
                    entry.get("IORegistryEntryName")
                    or entry.get("IOClass")
                    or f"GPU {len(result)}"
                ),
                busy_percent=float(stats["Device Utilization %"]),
                renderer_percent=_optional_float(stats.get("Renderer Utilization %")),
                tiler_percent=_optional_float(stats.get("Tiler Utilization %")),
                memory_bytes=_optional_int(stats.get("In use system memory")),
                allocated_bytes=_optional_int(stats.get("Alloc system memory")),
            )
        )
    return tuple(result)


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_powermetrics_plist(document: Any) -> PowermetricsReading:
    """Parse the stable fields emitted by ``powermetrics --format plist``."""
    if not isinstance(document, dict):
        return PowermetricsReading()
    gpu = document.get("gpu") or {}
    processor = document.get("processor") or {}
    gpu_power = _optional_float(processor.get("gpu_power"))
    if gpu_power is None:
        gpu_power = _optional_float(gpu.get("gpu_power"))
    frequency = _optional_float(gpu.get("freq_hz"))
    return PowermetricsReading(
        gpu_busy_percent=_ratio_to_busy(gpu.get("idle_ratio")),
        gpu_frequency_mhz=None if frequency is None else frequency / 1_000_000.0,
        gpu_power_mw=gpu_power,
        thermal_pressure=(
            str(document["thermal_pressure"])
            if document.get("thermal_pressure") is not None
            else None
        ),
    )


class DarwinMach:
    """Small ctypes wrapper around read-only Mach and sysctl host counters."""

    def __init__(self):
        self.library = None
        if platform.system() != "Darwin":
            return
        try:
            lib = ctypes.CDLL(ctypes.util.find_library("System") or "libSystem.dylib")
            lib.mach_host_self.argtypes = []
            lib.mach_host_self.restype = ctypes.c_uint32
            lib.host_processor_info.argtypes = [
                ctypes.c_uint32,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)),
                ctypes.POINTER(ctypes.c_uint32),
            ]
            lib.host_processor_info.restype = ctypes.c_int
            lib.host_page_size.argtypes = [
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_size_t),
            ]
            lib.host_page_size.restype = ctypes.c_int
            lib.host_statistics64.argtypes = [
                ctypes.c_uint32,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint32),
            ]
            lib.host_statistics64.restype = ctypes.c_int
            lib.mach_port_deallocate.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
            lib.mach_port_deallocate.restype = ctypes.c_int
            lib.vm_deallocate.argtypes = [ctypes.c_uint32, ctypes.c_uint64, ctypes.c_size_t]
            lib.vm_deallocate.restype = ctypes.c_int
            lib.sysctlbyname.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            lib.sysctlbyname.restype = ctypes.c_int
            ctypes.c_uint32.in_dll(lib, "mach_task_self_")
            self.library = lib
        except Exception:  # noqa: BLE001
            self.library = None

    @property
    def available(self) -> bool:
        return self.library is not None

    def _task_self(self) -> int:
        return ctypes.c_uint32.in_dll(self.library, "mach_task_self_").value

    def _sysctl_scalar(self, name: str, ctype):
        if self.library is None:
            return None
        value = ctype()
        size = ctypes.c_size_t(ctypes.sizeof(value))
        if self.library.sysctlbyname(
            name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0
        ) != 0:
            return None
        return value

    def cpu_ticks(self) -> tuple[CPUTicks, ...]:
        if self.library is None:
            return ()
        host = self.library.mach_host_self()
        cpu_count = ctypes.c_uint32(0)
        info_count = ctypes.c_uint32(0)
        info = ctypes.POINTER(ctypes.c_uint32)()
        try:
            result = self.library.host_processor_info(
                host,
                _PROCESSOR_CPU_LOAD_INFO,
                ctypes.byref(cpu_count),
                ctypes.byref(info),
                ctypes.byref(info_count),
            )
            if result != _KERN_SUCCESS or not info:
                return ()
            values = tuple(info[index] for index in range(info_count.value))
            if len(values) < cpu_count.value * _CPU_STATE_MAX:
                return ()
            return tuple(
                CPUTicks(*values[index : index + _CPU_STATE_MAX])
                for index in range(0, cpu_count.value * _CPU_STATE_MAX, _CPU_STATE_MAX)
            )
        finally:
            if info:
                self.library.vm_deallocate(
                    self._task_self(),
                    ctypes.cast(info, ctypes.c_void_p).value,
                    info_count.value * ctypes.sizeof(ctypes.c_uint32),
                )
            self.library.mach_port_deallocate(self._task_self(), host)

    def vm(self) -> _VMRaw | None:
        if self.library is None:
            return None
        total = self._sysctl_scalar("hw.memsize", ctypes.c_uint64)
        if total is None:
            return None
        host = self.library.mach_host_self()
        try:
            page_size = ctypes.c_size_t(0)
            stats = _VMStatistics64()
            count = ctypes.c_uint32(_HOST_VM_INFO64_REV1_COUNT)
            result = self.library.host_page_size(host, ctypes.byref(page_size))
            if result == _KERN_SUCCESS:
                result = self.library.host_statistics64(
                    host, _HOST_VM_INFO64, ctypes.byref(stats), ctypes.byref(count)
                )
            if result != _KERN_SUCCESS or count.value < _HOST_VM_INFO64_REV1_COUNT:
                return None
            return _VMRaw(
                page_size=int(page_size.value),
                total_bytes=int(total.value),
                free=int(stats.free_count),
                active=int(stats.active_count),
                inactive=int(stats.inactive_count),
                wired=int(stats.wire_count),
                speculative=int(stats.speculative_count),
                compressor=int(stats.compressor_page_count),
                uncompressed_in_compressor=int(
                    stats.total_uncompressed_pages_in_compressor
                ),
                file_backed=int(stats.external_page_count),
                anonymous=int(stats.internal_page_count),
                purgeable=int(stats.purgeable_count),
                pageins=int(stats.pageins),
                pageouts=int(stats.pageouts),
                compressions=int(stats.compressions),
                decompressions=int(stats.decompressions),
                swapins=int(stats.swapins),
                swapouts=int(stats.swapouts),
            )
        finally:
            self.library.mach_port_deallocate(self._task_self(), host)

    def swap(self) -> _XSWUsage | None:
        return self._sysctl_scalar("vm.swapusage", _XSWUsage)

    def pressure(self) -> str:
        value = self._sysctl_scalar(
            "kern.memorystatus_vm_pressure_level", ctypes.c_int
        )
        if value is None:
            return "unknown"
        return {1: "normal", 2: "warn", 4: "critical"}.get(
            int(value.value), f"level {int(value.value)}"
        )


def thermal_state() -> str:
    """Return the public ``NSProcessInfo.thermalState`` without PyObjC."""
    if platform.system() != "Darwin":
        return "unavailable"
    try:
        ctypes.CDLL(ctypes.util.find_library("Foundation"))
        objc = ctypes.CDLL(ctypes.util.find_library("objc"))
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        object_message = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        )(address)
        integer_message = ctypes.CFUNCTYPE(
            ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p
        )(address)
        process_info = object_message(
            objc.objc_getClass(b"NSProcessInfo"),
            objc.sel_registerName(b"processInfo"),
        )
        state = integer_message(process_info, objc.sel_registerName(b"thermalState"))
        return {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}.get(
            int(state), f"unknown ({state})"
        )
    except Exception:  # noqa: BLE001
        return "unavailable"


class PowermetricsSampler:
    """Continuous privilege-gated powermetrics plist reader."""

    def __init__(self, interval_seconds: float):
        self.interval_seconds = max(0.1, float(interval_seconds))
        self._lock = threading.Lock()
        self._latest: PowermetricsReading | None = None
        self._error: str | None = None
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    @property
    def latest(self) -> PowermetricsReading | None:
        with self._lock:
            return self._latest

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def start(self) -> None:
        if platform.system() != "Darwin" or not Path("/usr/bin/powermetrics").exists():
            self._error = "powermetrics is unavailable on this host"
            return
        command = [] if os.geteuid() == 0 else ["/usr/bin/sudo", "-n"]
        command += [
            "/usr/bin/powermetrics",
            "--sample-rate",
            str(max(100, int(self.interval_seconds * 1000))),
            "--sample-count",
            "-1",
            "--samplers",
            "gpu_power,thermal",
            "--format",
            "plist",
            "--buffer-size",
            "1",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            self._error = f"powermetrics failed to start: {error}"
            return
        self._thread = threading.Thread(
            target=self._read_loop, name="mlx2-top-powermetrics", daemon=True
        )
        self._thread.start()

    def _read_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        pending = b""
        while True:
            chunk = process.stdout.read1(65536)
            if not chunk:
                break
            pending += chunk
            while b"\0" in pending:
                packet, pending = pending.split(b"\0", 1)
                self._consume(packet)
        self._consume(pending)
        process.wait()
        if process.returncode and self.latest is None:
            detail = ""
            if process.stderr is not None:
                detail = process.stderr.read(4096).decode("utf-8", "replace").strip()
            with self._lock:
                self._error = (
                    "powermetrics needs cached sudo credentials; run `sudo -v` first"
                    if "password is required" in detail
                    else detail or f"powermetrics exited {process.returncode}"
                )

    def _consume(self, packet: bytes) -> None:
        packet = packet.strip()
        if not packet:
            return
        try:
            parsed = parse_powermetrics_plist(plistlib.loads(packet))
        except Exception:  # noqa: BLE001
            return
        with self._lock:
            self._latest = parsed
            self._error = None

    def close(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if self._thread is not None:
            self._thread.join(timeout=2)


class HostCollector:
    """Stateful sampler that turns cumulative host counters into rates."""

    def __init__(
        self,
        *,
        interval_seconds: float = 1.0,
        powermetrics: bool = True,
    ):
        self.interval_seconds = float(interval_seconds)
        self.mach = DarwinMach()
        self._last_time = time.monotonic()
        self._last_cpu = self.mach.cpu_ticks()
        self._last_vm = self.mach.vm()
        self.powermetrics = (
            PowermetricsSampler(interval_seconds) if powermetrics else None
        )
        if self.powermetrics is not None:
            self.powermetrics.start()

    def close(self) -> None:
        if self.powermetrics is not None:
            self.powermetrics.close()

    def _gpus(self) -> tuple[GPUReading, ...]:
        if platform.system() != "Darwin":
            return ()
        try:
            result = subprocess.run(
                ["/usr/sbin/ioreg", "-r", "-a", "-d", "1", "-c", "AGXAccelerator"],
                check=False,
                capture_output=True,
                timeout=max(0.25, min(1.0, self.interval_seconds)),
            )
        except (OSError, subprocess.TimeoutExpired):
            return ()
        return parse_ioreg_gpus(result.stdout) if result.returncode == 0 else ()

    def _memory(self, vm: _VMRaw | None, seconds: float) -> MemoryReading | None:
        if vm is None:
            return None
        before = self._last_vm or vm
        page = vm.page_size
        available = estimate_host_available_bytes(
            page_size=page,
            physical_bytes=vm.total_bytes,
            active=vm.active,
            inactive=vm.inactive,
            speculative=vm.speculative,
            wired=vm.wired,
            compressor=vm.compressor,
            file_backed=vm.file_backed,
            purgeable=vm.purgeable,
        )
        return MemoryReading(
            total_bytes=vm.total_bytes,
            available_bytes=available,
            free_bytes=vm.free * page,
            active_bytes=vm.active * page,
            inactive_bytes=vm.inactive * page,
            speculative_bytes=vm.speculative * page,
            wired_bytes=vm.wired * page,
            compressed_bytes=vm.compressor * page,
            uncompressed_in_compressor_bytes=vm.uncompressed_in_compressor * page,
            file_backed_bytes=vm.file_backed * page,
            anonymous_bytes=vm.anonymous * page,
            purgeable_bytes=vm.purgeable * page,
            pressure=self.mach.pressure(),
            pageins_per_second=_rate(vm.pageins, before.pageins, seconds),
            pageouts_per_second=_rate(vm.pageouts, before.pageouts, seconds),
            compressed_per_second=_rate(vm.compressions, before.compressions, seconds),
            decompressed_per_second=_rate(
                vm.decompressions, before.decompressions, seconds
            ),
        )

    def _swap(self, vm: _VMRaw | None, seconds: float) -> SwapReading | None:
        usage = self.mach.swap()
        if usage is None or vm is None:
            return None
        before = self._last_vm or vm
        return SwapReading(
            total_bytes=int(usage.total),
            used_bytes=int(usage.used),
            free_bytes=int(usage.available),
            encrypted=bool(usage.encrypted),
            swapins_bytes=vm.swapins * vm.page_size,
            swapouts_bytes=vm.swapouts * vm.page_size,
            swapins_per_second=_rate(
                vm.swapins, before.swapins, seconds, vm.page_size
            ),
            swapouts_per_second=_rate(
                vm.swapouts, before.swapouts, seconds, vm.page_size
            ),
        )

    def sample(self) -> HostReading:
        now = time.monotonic()
        seconds = max(1e-6, now - self._last_time)
        cpu = self.mach.cpu_ticks()
        vm = self.mach.vm()
        busy = cpu_busy_percent(self._last_cpu, cpu)
        memory = self._memory(vm, seconds)
        swap = self._swap(vm, seconds)
        gpus = self._gpus()
        power = self.powermetrics.latest if self.powermetrics is not None else None
        if power is not None and gpus:
            first = gpus[0]
            gpus = (
                GPUReading(
                    **{
                        **first.__dict__,
                        "busy_percent": power.gpu_busy_percent
                        if power.gpu_busy_percent is not None
                        else first.busy_percent,
                        "frequency_mhz": power.gpu_frequency_mhz,
                        "power_mw": power.gpu_power_mw,
                        "source": "powermetrics"
                        if power.gpu_busy_percent is not None
                        else first.source,
                    }
                ),
                *gpus[1:],
            )
        self._last_time = now
        self._last_cpu = cpu
        self._last_vm = vm
        return HostReading(
            sampled_at=time.time(),
            interval_seconds=seconds,
            cpu_busy_percent=busy,
            load_average=tuple(float(value) for value in os.getloadavg()),
            memory=memory,
            swap=swap,
            gpus=gpus,
            thermal_state=thermal_state(),
            thermal_pressure=power.thermal_pressure if power is not None else None,
            powermetrics_error=(
                self.powermetrics.error
                if self.powermetrics is not None
                else "powermetrics disabled"
            ),
            warnings=() if self.mach.available else ("Mach host counters unavailable",),
        )


__all__ = [
    "CPUTicks",
    "GPUReading",
    "HostCollector",
    "HostReading",
    "MemoryReading",
    "PowermetricsReading",
    "SwapReading",
    "cpu_busy_percent",
    "parse_ioreg_gpus",
    "parse_powermetrics_plist",
    "thermal_state",
]
