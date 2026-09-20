# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; original notices in provenance/NOTICE.
"""Best-effort process and host memory readings from macOS.

MLX allocator counters do not include every page charged to the process.
Darwin's physical-footprint counter provides an independent safety floor for
admission decisions while failing closed to ``None`` on unsupported systems.

Host-wide signals (``host_memory_snapshot`` and ``MemoryPressureMonitor``) read
Mach VM statistics and the kernel's memorystatus pressure level through
read-only syscalls.  The available-memory estimate follows Splash's
``estimateHostAvailableMemory`` (runtime/engine/MemoryGovernor.cpp, Apache-2.0,
see provenance/splash-09-host-memory-signals.json).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Optional

_RUSAGE_INFO_V4 = 4
_RUSAGE_INFO_V4_SIZE = 296


class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
    ]


def _load_libproc():
    if platform.system() != "Darwin":
        return None
    if ctypes.sizeof(_RUsageInfoV4) != _RUSAGE_INFO_V4_SIZE:
        return None
    try:
        library = ctypes.CDLL(ctypes.util.find_library("proc") or "libproc.dylib")
        library.proc_pid_rusage.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.proc_pid_rusage.restype = ctypes.c_int
        return library
    except Exception:  # noqa: BLE001 - an optional safety probe must not fail import
        return None


_LIBPROC = _load_libproc()


def physical_footprint_bytes(pid: int | None = None) -> int | None:
    """Return Darwin's physical-footprint counter, or ``None`` on failure."""
    if _LIBPROC is None:
        return None
    try:
        info = _RUsageInfoV4()
        pointer = ctypes.cast(ctypes.byref(info), ctypes.POINTER(ctypes.c_void_p))
        result = _LIBPROC.proc_pid_rusage(
            int(os.getpid() if pid is None else pid), _RUSAGE_INFO_V4, pointer
        )
        if result != 0:
            return None
        return int(info.ri_phys_footprint)
    except Exception:  # noqa: BLE001 - admission must survive probe failure
        return None


# Prefix of ``struct vm_statistics64`` through its rev1 fields.  Newer kernels
# append fields; asking for exactly HOST_VM_INFO64_REV1_COUNT keeps this layout
# valid on every macOS that serves mlx2 without depending on the SDK revision.
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


_HOST_VM_INFO64 = 4
_HOST_VM_INFO64_REV1_COUNT = 38
_KERN_SUCCESS = 0


def _load_libsystem():
    if platform.system() != "Darwin":
        return None
    if ctypes.sizeof(_VMStatistics64) != 4 * _HOST_VM_INFO64_REV1_COUNT:
        return None
    try:
        library = ctypes.CDLL(ctypes.util.find_library("System") or "libSystem.dylib")
        library.mach_host_self.argtypes = []
        library.mach_host_self.restype = ctypes.c_uint32
        library.host_page_size.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        library.host_page_size.restype = ctypes.c_int
        library.host_statistics64.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        library.host_statistics64.restype = ctypes.c_int
        library.mach_port_deallocate.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        library.mach_port_deallocate.restype = ctypes.c_int
        library.sysctlbyname.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        library.sysctlbyname.restype = ctypes.c_int
        # mach_task_self() is a macro over this exported global.
        ctypes.c_uint32.in_dll(library, "mach_task_self_")
        return library
    except Exception:  # noqa: BLE001 - an optional safety probe must not fail import
        return None


_LIBSYSTEM = _load_libsystem()


def _sysctl_value(name: str, ctype):
    """Read one fixed-size sysctl scalar, or ``None`` on any failure."""
    if _LIBSYSTEM is None:
        return None
    try:
        value = ctype()
        size = ctypes.c_size_t(ctypes.sizeof(value))
        result = _LIBSYSTEM.sysctlbyname(
            name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0
        )
        if result != 0 or size.value != ctypes.sizeof(value):
            return None
        return int(value.value)
    except Exception:  # noqa: BLE001
        return None


_PHYSICAL_MEMORY_BYTES: Optional[int] = None


def _physical_memory_bytes() -> Optional[int]:
    # Physical capacity is immutable; read hw.memsize once per process.
    global _PHYSICAL_MEMORY_BYTES
    if _PHYSICAL_MEMORY_BYTES is None:
        value = _sysctl_value("hw.memsize", ctypes.c_uint64)
        _PHYSICAL_MEMORY_BYTES = value if value else None
    return _PHYSICAL_MEMORY_BYTES


def estimate_host_available_bytes(
    *,
    page_size: int,
    physical_bytes: int,
    active: int,
    inactive: int,
    speculative: int,
    wired: int,
    compressor: int,
    file_backed: int,
    purgeable: int,
) -> int:
    """Splash's host estimate: capacity minus pages that are not reclaimable.

    ``hw.memsize - (active + inactive + speculative + wired + compressor
    - file_backed - purgeable) * page``.  File-backed (external) and purgeable
    pages are counted as available because the kernel can drop them without
    writing anything.  Returns 0 whenever a term is inconsistent.
    """
    if page_size <= 0 or physical_bytes <= 0:
        return 0
    used = active + inactive + speculative + wired + compressor
    if file_backed > used:
        return 0
    used -= file_backed
    if purgeable > used:
        return 0
    used -= purgeable
    used_bytes = used * page_size
    return physical_bytes - used_bytes if used_bytes < physical_bytes else 0


@dataclass(frozen=True)
class HostMemorySnapshot:
    """One ``host_statistics64(HOST_VM_INFO64)`` reading, in pages."""

    page_size: int
    physical_bytes: int
    free: int
    active: int
    inactive: int
    speculative: int
    wired: int
    compressor: int
    file_backed: int
    purgeable: int

    @property
    def available_bytes(self) -> int:
        """Splash estimate; counts file-backed and purgeable pages as free."""
        return estimate_host_available_bytes(
            page_size=self.page_size,
            physical_bytes=self.physical_bytes,
            active=self.active,
            inactive=self.inactive,
            speculative=self.speculative,
            wired=self.wired,
            compressor=self.compressor,
            file_backed=self.file_backed,
            purgeable=self.purgeable,
        )

    @property
    def reclaimable_bytes(self) -> int:
        """``vm_stat`` free + inactive + speculative, in bytes.

        Mach's ``free_count`` already includes speculative pages (``vm_stat``
        prints ``free - speculative`` as "Pages free"), so the historical
        vm_stat sum is exactly ``free + inactive``.
        """
        return (self.free + self.inactive) * self.page_size


def host_memory_snapshot() -> HostMemorySnapshot | None:
    """Return current Mach VM statistics, or ``None`` off Darwin / on failure."""
    if _LIBSYSTEM is None:
        return None
    physical = _physical_memory_bytes()
    if physical is None:
        return None
    try:
        host = _LIBSYSTEM.mach_host_self()
        try:
            page_size = ctypes.c_size_t(0)
            stats = _VMStatistics64()
            count = ctypes.c_uint32(_HOST_VM_INFO64_REV1_COUNT)
            result = _LIBSYSTEM.host_page_size(host, ctypes.byref(page_size))
            if result == _KERN_SUCCESS:
                result = _LIBSYSTEM.host_statistics64(
                    host, _HOST_VM_INFO64, ctypes.byref(stats), ctypes.byref(count)
                )
        finally:
            # mach_host_self() adds a send-right reference on every call.
            _LIBSYSTEM.mach_port_deallocate(
                ctypes.c_uint32.in_dll(_LIBSYSTEM, "mach_task_self_").value, host
            )
        if result != _KERN_SUCCESS or not page_size.value:
            return None
        if count.value < _HOST_VM_INFO64_REV1_COUNT:
            return None
        return HostMemorySnapshot(
            page_size=int(page_size.value),
            physical_bytes=physical,
            free=int(stats.free_count),
            active=int(stats.active_count),
            inactive=int(stats.inactive_count),
            speculative=int(stats.speculative_count),
            wired=int(stats.wire_count),
            compressor=int(stats.compressor_page_count),
            file_backed=int(stats.external_page_count),
            purgeable=int(stats.purgeable_count),
        )
    except Exception:  # noqa: BLE001 - admission must survive probe failure
        return None


class PressureLevel(IntEnum):
    NORMAL = 0
    WARN = 1
    CRITICAL = 2


def pressure_level_from_kernel(value: int) -> PressureLevel:
    """Map ``kern.memorystatus_vm_pressure_level`` (1/2/4) onto three levels."""
    if value >= 4:
        return PressureLevel.CRITICAL
    if value >= 2:
        return PressureLevel.WARN
    return PressureLevel.NORMAL


def host_pressure_level() -> PressureLevel | None:
    """Poll the kernel memorystatus pressure level, or ``None`` if unavailable."""
    value = _sysctl_value("kern.memorystatus_vm_pressure_level", ctypes.c_int)
    return None if value is None else pressure_level_from_kernel(value)


class MemoryPressureMonitor:
    """Hysteresis over the polled kernel pressure level.

    A rise is reported on the first reading that shows it.  A fall is reported
    only after every reading for ``fall_after_seconds`` stayed below the
    current level, and lands on the highest level seen in that window, so a
    consumer that sheds work at WARN cannot oscillate on a flapping signal.
    Unavailable readings leave the reported level unchanged.
    """

    def __init__(
        self,
        fall_after_seconds: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        *,
        reader: Optional[Callable[[], Optional[PressureLevel]]] = None,
    ):
        if not fall_after_seconds >= 0:
            raise ValueError("fall_after_seconds must be non-negative")
        self.fall_after_seconds = float(fall_after_seconds)
        self._clock = clock
        # Resolved per call so the module-level probe stays patchable.
        self._reader = reader or (lambda: host_pressure_level())
        self._lock = threading.Lock()
        self._level = PressureLevel.NORMAL
        self._fall_since: Optional[float] = None
        self._fall_peak = PressureLevel.NORMAL
        self.raw: Optional[PressureLevel] = None

    def level(self) -> PressureLevel:
        observed = self._reader()
        with self._lock:
            self.raw = observed
            if observed is None:
                return self._level
            observed = PressureLevel(observed)
            if observed >= self._level:
                self._level = observed
                self._fall_since = None
                return self._level
            now = self._clock()
            if self._fall_since is None:
                self._fall_since = now
                self._fall_peak = observed
            else:
                self._fall_peak = max(self._fall_peak, observed)
            if now - self._fall_since >= self.fall_after_seconds:
                self._level = self._fall_peak
                # A further fall needs its own full stable window.
                self._fall_since = None
            return self._level


__all__ = [
    "HostMemorySnapshot",
    "MemoryPressureMonitor",
    "PressureLevel",
    "estimate_host_available_bytes",
    "host_memory_snapshot",
    "host_pressure_level",
    "physical_footprint_bytes",
    "pressure_level_from_kernel",
]
