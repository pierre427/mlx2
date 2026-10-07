"""Native ownership of the IOReport / HID samplers (CPU, no GPU work).

A fake CoreFoundation/IOReport/IOKit records every owned reference, so each
Create/Copy is pinned to exactly one release and every NULL result is pinned
to a ``RuntimeError`` that unwinds what was already owned.  The real-host
checks run only where IOReport exists; ``MLX2_RUN_SLOW_TESTS=1`` lengthens
the leak check to 2000 reads.
"""

import ctypes
import os
import threading

import pytest

import mlx2.apple_telemetry as apple


class FakeNative:
    """One object standing in for the CF, IOKit and IOReport libraries."""

    def __init__(self, fail=None):
        self.fail = fail  # (function name, 1-based call number) to return NULL
        self.calls = {}
        self.live = {}  # owned handle -> creating function
        self.text = {}
        self.callbacks = None
        self._next = 0x1000

    def _new(self, name, text=None):
        self.calls[name] = self.calls.get(name, 0) + 1
        if self.fail == (name, self.calls[name]):
            return None
        self._next += 16
        self.live[self._next] = name
        if text is not None:
            self.text[self._next] = text
        return self._next

    # CoreFoundation
    def CFRelease(self, ref):
        assert ref in self.live, f"release of an unowned reference {ref:#x}"
        del self.live[ref]

    def CFStringCreateWithCString(self, _alloc, text, _encoding):
        return self._new("CFStringCreateWithCString", text.decode())

    def CFStringGetCString(self, ref, buf, _size, _encoding):
        if ref not in self.text:
            return False
        buf.value = self.text[ref].encode()
        return True

    def CFDictionaryGetValue(self, _dictionary, _key):
        return None  # no channels: the parse loop is covered elsewhere

    def CFArrayGetCount(self, _array):
        return 1

    def CFArrayGetValueAtIndex(self, _array, _index):
        return 0xBEEF  # a borrowed service reference

    def CFDictionaryCreateMutableCopy(self, *_args):
        return self._new("CFDictionaryCreateMutableCopy")

    def CFNumberCreate(self, *_args):
        return self._new("CFNumberCreate")

    def CFDictionaryCreate(self, _alloc, _keys, _values, _count, key_cb, value_cb):
        self.callbacks = (key_cb, value_cb)
        return self._new("CFDictionaryCreate")

    # IOReport
    def IOReportCopyChannelsInGroup(self, *_args):
        return self._new("IOReportCopyChannelsInGroup")

    def IOReportMergeChannels(self, *_args):
        return None

    def IOReportCreateSubscription(self, _alloc, _desired, subbed, _id, _extra):
        sub = self._new("IOReportCreateSubscription")
        if sub is not None:
            subbed._obj.value = self._new("subscribed channels")
        return sub

    def IOReportCreateSamples(self, *_args):
        return self._new("IOReportCreateSamples")

    def IOReportCreateSamplesDelta(self, *_args):
        return self._new("IOReportCreateSamplesDelta")

    # IOKit HID
    def IOHIDEventSystemClientCreate(self, _alloc):
        return self._new("IOHIDEventSystemClientCreate")

    def IOHIDEventSystemClientSetMatching(self, _client, _matching):
        return 0

    def IOHIDEventSystemClientCopyServices(self, _client):
        return self._new("IOHIDEventSystemClientCopyServices")

    def IOHIDServiceClientCopyProperty(self, _service, _key):
        return self._new("IOHIDServiceClientCopyProperty", "PMU tdie1")

    def IOHIDServiceClientCopyEvent(self, *_args):
        return self._new("IOHIDServiceClientCopyEvent")

    def IOHIDEventGetFloatValue(self, _event, _field):
        return 55.0


@pytest.fixture
def native(monkeypatch):
    def install(fail=None):
        fake = FakeNative(fail)
        monkeypatch.setattr(apple, "_LIBS", (fake, fake, fake))
        monkeypatch.setattr(apple, "_DICT_CALLBACKS", (0xCB1, 0xCB2))
        return fake

    return install


def test_every_owned_reference_is_released_once(native):
    fake = native()
    energy = apple.EnergySampler()
    temperature = apple.TemperatureSampler()
    for _ in range(5):
        assert energy.read().seconds > 0
        assert temperature.die_summary()["die_max_c"] == 55.0
    # The matching dictionary retains its keys and values (kCFType callbacks).
    assert fake.callbacks == (0xCB1, 0xCB2)
    # Only what the samplers hold across reads is still owned.
    assert sorted(fake.live.values()) == sorted([
        "IOReportCreateSubscription", "subscribed channels",
        "CFStringCreateWithCString", "IOReportCreateSamples",
        "IOHIDEventSystemClientCreate", "CFStringCreateWithCString",
    ])
    for sampler in (energy, temperature):
        sampler.close()
        sampler.close()  # idempotent
    assert fake.live == {}
    with pytest.raises(RuntimeError, match="closed"):
        energy.read()
    with pytest.raises(RuntimeError, match="closed"):
        temperature.read()


@pytest.mark.parametrize("fail", [
    ("CFStringCreateWithCString", 1),
    ("IOReportCopyChannelsInGroup", 1),
    ("CFStringCreateWithCString", 3),
    ("IOReportCopyChannelsInGroup", 2),
    ("CFDictionaryCreateMutableCopy", 1),
    ("IOReportCreateSubscription", 1),
    ("subscribed channels", 1),
    ("CFStringCreateWithCString", 4),
    ("IOReportCreateSamples", 1),
])
def test_energy_sampler_null_results_raise_and_unwind(native, fail):
    fake = native(fail)
    with pytest.raises(RuntimeError, match="returned NULL"):
        apple.EnergySampler()
    assert fake.live == {}


@pytest.mark.parametrize("fail", [
    ("IOHIDEventSystemClientCreate", 1),
    ("CFStringCreateWithCString", 2),
    ("CFNumberCreate", 2),
    ("CFDictionaryCreate", 1),
    ("CFStringCreateWithCString", 3),
])
def test_temperature_sampler_null_results_raise_and_unwind(native, fail):
    fake = native(fail)
    with pytest.raises(RuntimeError, match="returned NULL"):
        apple.TemperatureSampler()
    assert fake.live == {}


@pytest.mark.parametrize("name", ["IOReportCreateSamples", "IOReportCreateSamplesDelta"])
def test_a_null_sample_raises_without_leaking_and_the_sampler_recovers(native, name):
    fake = native()
    energy = apple.EnergySampler()
    fake.fail = (name, fake.calls.get(name, 0) + 1)
    with pytest.raises(RuntimeError, match="returned NULL"):
        energy.read()
    assert energy.read().seconds > 0
    energy.close()
    assert fake.live == {}


def test_concurrent_reads_and_close_never_double_release(native):
    fake = native()
    energy = apple.EnergySampler()
    errors = []

    def reader():
        for _ in range(200):
            try:
                energy.read()
            except RuntimeError as error:  # closed underneath us: expected
                assert "closed" in str(error)
                return
            except Exception as error:  # noqa: BLE001 - recorded for the assert
                errors.append(error)
                return

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    energy.close()
    for thread in threads:
        thread.join()
    assert errors == [] and fake.live == {}


# -- the real host ---------------------------------------------------------


def _resident_bytes() -> int:
    class Info(ctypes.Structure):  # mach_task_basic_info
        _fields_ = [
            ("virtual_size", ctypes.c_uint64), ("resident_size", ctypes.c_uint64),
            ("resident_size_max", ctypes.c_uint64), ("user_time", ctypes.c_uint64),
            ("system_time", ctypes.c_uint64), ("policy", ctypes.c_int),
            ("suspend_count", ctypes.c_int),
        ]

    libc = ctypes.CDLL(None)
    libc.mach_task_self.restype = ctypes.c_uint32
    info = Info()
    count = ctypes.c_uint32(ctypes.sizeof(Info) // 4)
    assert libc.task_info(libc.mach_task_self(), 20, ctypes.byref(info), ctypes.byref(count)) == 0
    return info.resident_size


@pytest.fixture
def real_host():
    if not apple.available():
        pytest.skip("IOReport is not available on this host")


def test_real_samplers_long_run_memory_stays_bounded(real_host):
    slow = bool(os.environ.get("MLX2_RUN_SLOW_TESTS"))
    reads, temperature_reads, constructions = (2000, 200, 100) if slow else (200, 20, 30)
    for _ in range(3):  # warm the frameworks' own caches
        energy, temperature = apple.EnergySampler(), apple.TemperatureSampler()
        energy.read()
        temperature.die_summary()
        energy.close()
        temperature.close()
    before = _resident_bytes()
    # Before the fix each construction leaked about 300 KiB of native state.
    for _ in range(constructions):
        apple.EnergySampler().close()
        apple.TemperatureSampler().close()
    energy, temperature = apple.EnergySampler(), apple.TemperatureSampler()
    try:
        for index in range(reads):
            reading = energy.read()
            if index < temperature_reads:
                temperature.die_summary()
        assert reading.gpu_watts is not None and reading.dram_watts is not None
    finally:
        energy.close()
        temperature.close()
    growth = _resident_bytes() - before
    assert growth < 4 << 20, f"resident memory grew {growth / 2**20:.1f} MiB"
