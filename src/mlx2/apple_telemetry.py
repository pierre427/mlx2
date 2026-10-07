"""Unprivileged Apple-silicon power, GPU DVFS and die-temperature sampling.

``powermetrics`` needs root.  The same counters are readable without
privilege: IOReport's "Energy Model" group (energy per subsystem, so power
over an interval), its "GPU Stats / GPU Performance States" channel (time
resident in each GPU DVFS state, i.e. the clock), and the HID temperature
sensor services (die temperatures).  Everything here is host-side and never
touches MLX; a failure on another platform or OS revision degrades to an
empty reading rather than raising.

Each ``EnergySampler`` owns its own IOReport subscription, so independent
callers (a 1 Hz background series and per-work-segment reads) do not
disturb each other's deltas.  Every Create/Copy result is checked (a NULL
raises ``RuntimeError`` after releasing what was already owned) and the
samplers release their CoreFoundation references in an idempotent
``close()``.
"""

from __future__ import annotations

import ctypes as C
import platform
import plistlib
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, field

_vp = C.c_void_p
#: GPU residency states counted as idle by ``gpu_active_fraction``.
GPU_IDLE_STATES = ("OFF", "IDLE", "DOWN")
_UTF8 = 0x08000100
_LIBS = None
#: (kCFTypeDictionaryKeyCallBacks, kCFTypeDictionaryValueCallBacks) addresses,
#: so a created dictionary retains its keys and values.
_DICT_CALLBACKS = (None, None)


def _load():
    global _LIBS, _DICT_CALLBACKS
    if _LIBS is not None:
        return _LIBS
    if platform.system() != "Darwin":
        _LIBS = False
        return _LIBS
    try:
        cf = C.cdll.LoadLibrary("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        iok = C.cdll.LoadLibrary("/System/Library/Frameworks/IOKit.framework/IOKit")
        ior = C.cdll.LoadLibrary("/usr/lib/libIOReport.dylib")
    except OSError:
        _LIBS = False
        return _LIBS
    for lib, specs in (
        (cf, [
            ("CFStringCreateWithCString", _vp, [_vp, C.c_char_p, C.c_uint32]),
            ("CFStringGetCString", C.c_bool, [_vp, C.c_char_p, C.c_long, C.c_uint32]),
            ("CFDictionaryGetValue", _vp, [_vp, _vp]),
            ("CFArrayGetCount", C.c_long, [_vp]),
            ("CFArrayGetValueAtIndex", _vp, [_vp, C.c_long]),
            ("CFRelease", None, [_vp]),
            ("CFDictionaryCreateMutableCopy", _vp, [_vp, C.c_long, _vp]),
            ("CFNumberCreate", _vp, [_vp, C.c_int, _vp]),
            ("CFDictionaryCreate", _vp, [_vp, _vp, _vp, C.c_long, _vp, _vp]),
        ]),
        (ior, [
            ("IOReportCopyChannelsInGroup", _vp, [_vp, _vp, C.c_uint64, C.c_uint64, C.c_uint64]),
            ("IOReportMergeChannels", None, [_vp, _vp, _vp]),
            ("IOReportCreateSubscription", _vp, [_vp, _vp, C.POINTER(_vp), C.c_uint64, _vp]),
            ("IOReportCreateSamples", _vp, [_vp, _vp, _vp]),
            ("IOReportCreateSamplesDelta", _vp, [_vp, _vp, _vp]),
            ("IOReportChannelGetGroup", _vp, [_vp]),
            ("IOReportChannelGetChannelName", _vp, [_vp]),
            ("IOReportChannelGetUnitLabel", _vp, [_vp]),
            ("IOReportSimpleGetIntegerValue", C.c_int64, [_vp, C.c_int32]),
            ("IOReportStateGetCount", C.c_int32, [_vp]),
            ("IOReportStateGetNameForIndex", _vp, [_vp, C.c_int32]),
            ("IOReportStateGetResidency", C.c_int64, [_vp, C.c_int32]),
        ]),
        (iok, [
            ("IOHIDEventSystemClientCreate", _vp, [_vp]),
            ("IOHIDEventSystemClientSetMatching", C.c_int, [_vp, _vp]),
            ("IOHIDEventSystemClientCopyServices", _vp, [_vp]),
            ("IOHIDServiceClientCopyProperty", _vp, [_vp, _vp]),
            ("IOHIDServiceClientCopyEvent", _vp, [_vp, C.c_int64, C.c_int32, C.c_int64]),
            ("IOHIDEventGetFloatValue", C.c_double, [_vp, C.c_int32]),
        ]),
    ):
        for name, res, args in specs:
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = res, args
    _DICT_CALLBACKS = tuple(
        C.addressof(C.c_char.in_dll(cf, name))
        for name in ("kCFTypeDictionaryKeyCallBacks", "kCFTypeDictionaryValueCallBacks")
    )
    _LIBS = (cf, iok, ior)
    return _LIBS


def available() -> bool:
    return bool(_load())


def _owned(ref, what: str):
    """``ref`` from a Create/Copy call, or ``RuntimeError`` if it is NULL."""
    if not ref:
        raise RuntimeError(f"{what} returned NULL")
    return ref


def _release(*refs) -> None:
    cf = _LIBS[0]
    for ref in refs:
        if ref:
            cf.CFRelease(ref)


def _cfs(text: str):
    cf = _LIBS[0]
    return _owned(
        cf.CFStringCreateWithCString(None, text.encode(), _UTF8),
        f"CFStringCreateWithCString({text!r})",
    )


def _str(ref) -> str | None:
    if not ref:
        return None
    buf = C.create_string_buffer(256)
    return buf.value.decode() if _LIBS[0].CFStringGetCString(ref, buf, 256, _UTF8) else None


def gpu_frequency_table_mhz() -> list[int]:
    """Non-zero GPU DVFS frequencies from the pmgr ``voltage-states9`` table."""

    try:
        raw = subprocess.run(
            ["ioreg", "-a", "-rd1", "-c", "AppleARMIODevice", "-n", "pmgr"],
            capture_output=True, timeout=10, check=True,
        ).stdout
        node = plistlib.loads(raw)
        node = node[0] if isinstance(node, list) else node
        blob = node.get("voltage-states9")
        pairs = [struct.unpack("<II", blob[i:i + 8]) for i in range(0, len(blob), 8)]
        return [f // 1_000_000 for f, _ in pairs if f]
    except Exception:
        return []


@dataclass
class EnergyReading:
    seconds: float
    watts: dict = field(default_factory=dict)       # channel -> mean W
    gpu_states: list = field(default_factory=list)  # [(name, residency)]

    @property
    def gpu_watts(self) -> float | None:
        return self.watts.get("GPU Energy")

    @property
    def dram_watts(self) -> float | None:
        return self.watts.get("DRAM")

    def gpu_active_fraction(self) -> float | None:
        total = sum(r for _, r in self.gpu_states)
        if not total:
            return None
        idle = sum(r for n, r in self.gpu_states if n in GPU_IDLE_STATES)
        return 1.0 - idle / total

    def gpu_mean_state(self) -> float | None:
        """Residency-weighted mean P-state index over active time (P1 = 1)."""

        active = [(int(n[1:]), r) for n, r in self.gpu_states
                  if n.startswith("P") and n[1:].isdigit() and r]
        total = sum(r for _, r in active)
        return sum(i * r for i, r in active) / total if total else None

    def as_dict(self) -> dict:
        return {
            "seconds": self.seconds,
            "gpu_w": self.gpu_watts,
            "dram_w": self.dram_watts,
            "cpu_w": self.watts.get("CPU Energy"),
            "gpu_active": self.gpu_active_fraction(),
            "gpu_mean_pstate": self.gpu_mean_state(),
            "gpu_states": [[n, r] for n, r in self.gpu_states if r],
        }


class EnergySampler:
    """Energy (power) and GPU P-state residency between successive reads.

    ``read`` and ``close`` are serialized, so a sampler may be shared across
    threads; a read after ``close`` raises ``RuntimeError``.
    """

    def __init__(self):
        if not _load():
            raise RuntimeError("IOReport is unavailable on this host")
        cf, _, ior = _LIBS
        self._lock = threading.Lock()
        self._sub = self._key = self._prev = None
        self._subbed = _vp()
        temporaries = []
        try:
            def created(ref, what):
                temporaries.append(_owned(ref, what))
                return ref

            energy = created(ior.IOReportCopyChannelsInGroup(
                created(_cfs("Energy Model"), "group"), None, 0, 0, 0),
                "IOReportCopyChannelsInGroup(Energy Model)")
            gpu = created(ior.IOReportCopyChannelsInGroup(
                created(_cfs("GPU Stats"), "group"),
                created(_cfs("GPU Performance States"), "subgroup"), 0, 0, 0),
                "IOReportCopyChannelsInGroup(GPU Stats)")
            ior.IOReportMergeChannels(energy, gpu, None)
            desired = created(cf.CFDictionaryCreateMutableCopy(None, 0, energy),
                              "CFDictionaryCreateMutableCopy")
            # The subscription keeps neither ``desired`` nor ``_subbed``;
            # ``_subbed`` is ours (Create rule) and outlives it in ``close``.
            self._sub = _owned(ior.IOReportCreateSubscription(
                None, desired, C.byref(self._subbed), 0, None),
                "IOReportCreateSubscription")
            _owned(self._subbed.value, "IOReportCreateSubscription channels")
            self._key = _cfs("IOReportChannels")
            self._prev = _owned(ior.IOReportCreateSamples(self._sub, self._subbed, None),
                                "IOReportCreateSamples")
        except BaseException:
            self.close()
            raise
        finally:
            _release(*temporaries)
        self._t = time.monotonic()

    def read(self) -> EnergyReading:
        cf, _, ior = _LIBS
        with self._lock:
            if self._prev is None:
                raise RuntimeError("EnergySampler is closed")
            cur = _owned(ior.IOReportCreateSamples(self._sub, self._subbed, None),
                         "IOReportCreateSamples")
            now = time.monotonic()
            delta = ior.IOReportCreateSamplesDelta(self._prev, cur, None)
            dt = max(now - self._t, 1e-6)
            cf.CFRelease(self._prev)
            # A failed delta still advances the baseline: the next read
            # covers only its own interval.
            self._prev, self._t = cur, now
            _owned(delta, "IOReportCreateSamplesDelta")
            try:
                return self._parse(delta, dt)
            finally:
                cf.CFRelease(delta)

    def _parse(self, delta, dt) -> EnergyReading:
        cf, _, ior = _LIBS
        reading = EnergyReading(seconds=dt)
        chans = cf.CFDictionaryGetValue(delta, self._key)
        for i in range(cf.CFArrayGetCount(chans) if chans else 0):
            ch = cf.CFArrayGetValueAtIndex(chans, i)
            group = _str(ior.IOReportChannelGetGroup(ch))
            name = _str(ior.IOReportChannelGetChannelName(ch))
            if group == "Energy Model":
                scale = {"mJ": 1e-3, "uJ": 1e-6, "nJ": 1e-9}.get(
                    _str(ior.IOReportChannelGetUnitLabel(ch)))
                if scale and name in ("GPU Energy", "CPU Energy", "DRAM", "ANE", "GPU SRAM"):
                    reading.watts[name] = ior.IOReportSimpleGetIntegerValue(ch, 0) * scale / dt
            elif group == "GPU Stats" and name == "GPUPH":
                reading.gpu_states = [
                    (_str(ior.IOReportStateGetNameForIndex(ch, k)),
                     ior.IOReportStateGetResidency(ch, k))
                    for k in range(ior.IOReportStateGetCount(ch))
                ]
        return reading

    def close(self) -> None:
        """Release the subscription and its samples; safe to call twice."""
        with self._lock:
            refs = (self._prev, self._sub, self._subbed.value, self._key)
            self._prev = self._sub = self._key = None
            self._subbed = _vp()
        _release(*refs)


class TemperatureSampler:
    """Die and package temperatures from the HID sensor services (deg C)."""

    def __init__(self):
        if not _load():
            raise RuntimeError("IOKit HID is unavailable on this host")
        cf, iok, _ = _LIBS
        self._lock = threading.Lock()
        self._client = self._product = None
        temporaries = []
        try:
            def created(ref, what):
                temporaries.append(_owned(ref, what))
                return ref

            self._client = _owned(iok.IOHIDEventSystemClientCreate(None),
                                  "IOHIDEventSystemClientCreate")
            page, usage = C.c_int32(0xFF00), C.c_int32(5)
            keys = (_vp * 2)(created(_cfs("PrimaryUsagePage"), "key"),
                             created(_cfs("PrimaryUsage"), "key"))
            vals = (_vp * 2)(created(cf.CFNumberCreate(None, 3, C.byref(page)), "CFNumberCreate"),
                             created(cf.CFNumberCreate(None, 3, C.byref(usage)), "CFNumberCreate"))
            # The kCFType callbacks retain keys and values, so all of these
            # are temporaries once the client has taken the matching.
            matching = created(cf.CFDictionaryCreate(
                None, keys, vals, 2, *_DICT_CALLBACKS), "CFDictionaryCreate")
            iok.IOHIDEventSystemClientSetMatching(self._client, matching)
            self._product = _cfs("Product")
        except BaseException:
            self.close()
            raise
        finally:
            _release(*temporaries)

    def read(self, match: tuple[str, ...] | None = None) -> dict:
        """Sensor name -> deg C; ``match`` keeps names containing any entry.

        Filtering by name first skips the per-sensor event copy, the bulk of
        a read's cost.
        """
        cf, iok, _ = _LIBS
        with self._lock:
            if self._client is None:
                raise RuntimeError("TemperatureSampler is closed")
            services = iok.IOHIDEventSystemClientCopyServices(self._client)
            out = {}
            for i in range(cf.CFArrayGetCount(services) if services else 0):
                svc = cf.CFArrayGetValueAtIndex(services, i)
                prop = iok.IOHIDServiceClientCopyProperty(svc, self._product)
                name = _str(prop)
                if prop:
                    cf.CFRelease(prop)  # a Copy call returns an owned reference
                if not name or (match is not None and not any(m in name for m in match)):
                    continue
                event = iok.IOHIDServiceClientCopyEvent(svc, 15, 0, 0)
                if name and event:
                    value = iok.IOHIDEventGetFloatValue(event, 15 << 16)
                    cf.CFRelease(event)
                    if -50.0 < value < 150.0:
                        out[name] = value
            if services:
                cf.CFRelease(services)
            return out

    def close(self) -> None:
        """Release the HID client; safe to call twice."""
        with self._lock:
            refs = (self._product, self._client)
            self._product = self._client = None
        _release(*refs)

    def die_summary(self) -> dict:
        temps = self.read(match=("tdie", "gas gauge battery"))
        die = [v for k, v in temps.items() if "tdie" in k]
        return {
            "die_max_c": max(die) if die else None,
            "die_mean_c": sum(die) / len(die) if die else None,
            "battery_c": temps.get("gas gauge battery"),
        }
