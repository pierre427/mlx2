# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; original notices in provenance/NOTICE.
"""Best-effort process memory readings from macOS.

MLX allocator counters do not include every page charged to the process.
Darwin's physical-footprint counter provides an independent safety floor for
admission decisions while failing closed to ``None`` on unsupported systems.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform

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


__all__ = ["physical_footprint_bytes"]
