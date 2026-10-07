"""Opt-in power, GPU DVFS and die-temperature telemetry for a serving engine.

A daemon thread reads :mod:`mlx2.apple_telemetry` every ``interval_seconds``:
IOReport energy per subsystem (so power over the interval), GPU P-state
residency and HID die temperatures, all without privilege.  It never touches
MLX or Metal, so it cannot stall or reorder device work; a scrape or status
read copies the last snapshot under a lock.  Memory is bounded: a rolling
window holding the newest ``window_seconds`` of interval records (a record
straddling its start is trimmed in proportion; at most twice the nominal
record count), one small entry per active request, and closed label sets.  Off a Darwin host, or when IOReport cannot subscribe, the
telemetry reports ``available: false`` with the reason and starts no thread.

Energy attribution (an estimate, not a measurement per request).  Each
interval's GPU + DRAM joules are split across requests by their share of the
output tokens delivered in that interval.  An interval that delivered no
output token is split across the requests that were prefilling (attached to
a lane, no output token yet), weighted by their uncached prompt tokens, a
proxy for prefill work since per-interval prefill progress is not tracked; an
interval with neither is idle and attributed to no request.  A request that
finishes mid-interval cannot wait for the interval to close, so its receipt
charges the tokens it delivered since the last closed interval at the
rolling-window joules per output token.  Within a mixed interval prefilling
requests receive nothing: their prefill energy lands on the requests that
emitted tokens.  CPU and ANE energy are reported but not attributed.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass

from .apple_telemetry import GPU_IDLE_STATES

log = logging.getLogger(__name__)

#: (label, IOReport "Energy Model" channel); the label is the metric domain.
DOMAINS = (("gpu", "GPU Energy"), ("dram", "DRAM"), ("cpu", "CPU Energy"), ("ane", "ANE"))
#: Domains whose energy is attributed to requests.
ATTRIBUTED_DOMAINS = ("gpu", "dram")
#: Closed GPU P-state label set; other residency names are not exported.  It
#: holds every idle state the active ratio excludes, so both describe the
#: same residency population.
GPU_STATES = (*GPU_IDLE_STATES, *(f"P{index}" for index in range(1, 16)))
RECEIPT_SCHEMA = "mlx2.energy-estimate.v1"
ATTRIBUTION_LAW = "interval_output_token_share"


@dataclass(frozen=True)
class PowerTelemetryPolicy:
    enabled: bool = False
    interval_seconds: float = 1.0
    window_seconds: float = 60.0

    @classmethod
    def from_value(cls, value) -> PowerTelemetryPolicy:
        """Parse ``None``/``False`` (off), ``True`` or an object."""
        if value is None or value is False:
            return cls()
        if value is True:
            return cls(enabled=True)
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError("power_telemetry must be a boolean or an object")
        unknown = set(value) - {"enabled", "interval_seconds", "window_seconds"}
        if unknown:
            raise ValueError(
                "unknown power_telemetry fields: " + ", ".join(sorted(unknown))
            )
        enabled = value.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("power_telemetry enabled must be boolean")
        interval = _seconds(value.get("interval_seconds", 1.0), "interval_seconds")
        window = _seconds(value.get("window_seconds", 60.0), "window_seconds")
        if not 0.1 <= interval <= 60:
            raise ValueError("power_telemetry interval_seconds must be 0.1 to 60")
        if not interval <= window <= 3600:
            raise ValueError(
                "power_telemetry window_seconds must be at least interval_seconds "
                "and at most 3600"
            )
        return cls(enabled=enabled, interval_seconds=interval, window_seconds=window)

    def as_dict(self) -> dict:
        return asdict(self)


def _seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"power_telemetry {name} must be a number of seconds")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"power_telemetry {name} must be finite")
    return value


def _default_energy_sampler():
    from .apple_telemetry import EnergySampler

    return EnergySampler()


def _default_temperature_sampler():
    from .apple_telemetry import TemperatureSampler

    return TemperatureSampler()


def _default_frequency_table():
    from .apple_telemetry import gpu_frequency_table_mhz

    return gpu_frequency_table_mhz()


def governed_watts(reading) -> tuple[float | None, str | None]:
    """GPU + DRAM mean watts: ``(watts, None)`` or ``(None, why)``.

    Both attributed domains must be present, finite and nonnegative; a
    missing channel is no measurement, not a 0 W one.
    """
    total = 0.0
    for label, channel in DOMAINS:
        if label not in ATTRIBUTED_DOMAINS:
            continue
        value = reading.watts.get(channel)
        if value is None:
            return None, f"no {channel!r} channel"
        value = float(value)
        if not math.isfinite(value) or value < 0:
            return None, f"{channel!r} reading {value!r} is not a finite nonnegative power"
        total += value
    return total, None


def _p_index(name) -> int | None:
    if isinstance(name, str) and name[:1] == "P" and name[1:].isdigit():
        return int(name[1:])
    return None


def frequency_mapping(table, gpu_states) -> tuple[tuple[int, ...] | None, str | None]:
    """Map pmgr GPU frequencies onto ``P1..Pn``: ``(table, None)`` or ``((), why)``.

    The mapping is positional, so it is accepted only when the non-zero
    frequencies ascend and number exactly as many as the P-states.  ``None``
    means no P-state channel was seen yet; try again on a later reading.
    ``table`` may be a callable (it shells out), called only once P-states
    exist; a failing call counts as an unavailable table.
    """
    p_states = sorted(
        index for index in (_p_index(name) for name, _ in gpu_states)
        if index is not None
    )
    if not p_states:
        return None, "no GPU P-state residency channel"
    if callable(table):
        try:
            table = table()
        except Exception:  # noqa: BLE001 - the mapping is optional
            table = ()
    table = [int(mhz) for mhz in table or ()]
    if not table:
        return (), "GPU frequency table unavailable"
    if (
        len(table) != len(p_states)
        or table != sorted(table)
        or p_states != list(range(1, len(p_states) + 1))
    ):
        return (), (
            f"{len(table)} table frequencies for {len(p_states)} P-states; "
            "no clean mapping"
        )
    return tuple(table), None


def gpu_state_means(gpu_states, table=()) -> tuple[float | None, float | None]:
    """Residency-weighted mean active P-state index and clock (MHz, if mapped)."""
    active = [
        (_p_index(name), residency) for name, residency in gpu_states
        if _p_index(name) is not None and residency
    ]
    total = sum(residency for _, residency in active)
    if not total:
        return None, None
    mean_pstate = sum(index * residency for index, residency in active) / total
    mean_mhz = (
        sum(table[index - 1] * residency for index, residency in active) / total
        if table else None
    )
    return mean_pstate, mean_mhz


class _Request:
    __slots__ = ("intervals", "joules", "seen_tokens")

    def __init__(self):
        self.seen_tokens = 0
        self.joules = 0.0
        self.intervals = 0


class PowerTelemetry:
    """Background power sampler with per-request energy attribution.

    ``token_source()`` returns ``(delivered_output_tokens_total, {request_id:
    (output_tokens, uncached_prompt_tokens, prefilling)})`` atomically; see
    :meth:`mlx2.batch_metrics.BatchRuntimeMetrics.token_progress`.  The
    sampler factories are injectable for tests.
    """

    def __init__(
        self,
        policy,
        token_source,
        *,
        energy_sampler_factory=None,
        temperature_sampler_factory=None,
        frequency_table=None,
    ):
        self.policy = PowerTelemetryPolicy.from_value(policy)
        self._token_source = token_source
        energy_sampler_factory = energy_sampler_factory or _default_energy_sampler
        temperature_sampler_factory = (
            temperature_sampler_factory or _default_temperature_sampler
        )
        self._frequency_table_fn = frequency_table or _default_frequency_table
        self._lock = threading.Lock()
        # ``_lifecycle`` guards thread start/stop; ``_sampling`` serializes
        # sampler reads with each other and with closing the samplers.
        self._lifecycle = threading.Lock()
        self._sampling = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._energy = None
        self._temperature = None
        self.reason = None
        self.temperature_reason = None
        try:
            self._energy = energy_sampler_factory()
        except Exception as error:  # noqa: BLE001 - degrade, never fail serving
            self.reason = f"{type(error).__name__}: {error}"
        if self._energy is not None:
            try:
                self._temperature = temperature_sampler_factory()
            except Exception as error:  # noqa: BLE001 - temperatures are optional
                self.temperature_reason = f"{type(error).__name__}: {error}"
        self.available = self._energy is not None
        # Frequency mapping is resolved on the first sample, off the caller.
        self._frequency_table = None
        self._frequency_reason = "not sampled yet"
        self._samples = 0
        self._errors = 0
        self._last_error = None
        self._last_cost = None
        self._last = None
        self._energy_total = {}
        self._residency_total = {}
        self._attributed = 0.0
        self._finished_estimated = 0.0
        self._idle = 0.0
        # (seconds, gpu_dram_joules, output_tokens) records covering at most
        # window_seconds; a late read's long interval must not stretch it.
        self._window = deque()
        self._window_seconds = 0.0
        self._window_records = 2 * max(
            1, math.ceil(self.policy.window_seconds / self.policy.interval_seconds)
        )
        self._requests: dict[str, _Request] = {}
        # Requests whose receipt already took their energy; skipped until they
        # leave the token source so a late sample cannot re-open them.
        self._closed: set[str] = set()
        self._delivered = 0
        # Called with (seconds, gpu_dram_watts) after each closed interval,
        # outside the lock (the power governor's measurement input), and
        # only when both governed domains were measured.
        self._listeners = []
        self._signal_reason = None
        self._signal_gaps = 0
        if self.available:
            self._delivered, active = token_source()
            for request_id, (tokens, _uncached, _prefilling) in active.items():
                self._requests.setdefault(request_id, _Request()).seen_tokens = tokens

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> PowerTelemetry:
        with self._lifecycle:
            if self.available and self._thread is None and not self._stop.is_set():
                self._thread = threading.Thread(
                    target=self._run, name="mlx2-power-telemetry", daemon=True
                )
                self._thread.start()
        return self

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def close(self) -> None:
        """Stop sampling and release the samplers; safe to call twice."""
        with self._lifecycle:
            self._stop.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=self.policy.interval_seconds + 2)
        with self._sampling:  # waits out a read still in flight
            samplers = (self._temperature, self._energy)
            self._temperature = self._energy = None
        for sampler in samplers:
            close = getattr(sampler, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # closing must not fail shutdown
                    log.exception("power telemetry sampler close failed")

    def _run(self) -> None:
        while not self._stop.wait(self.policy.interval_seconds):
            try:
                self.sample_once()
            except Exception as error:  # keep sampling; counted in status
                with self._lock:
                    self._errors += 1
                    first = self._last_error is None
                    self._last_error = f"{type(error).__name__}: {error}"
                if first:
                    log.exception("power telemetry sample failed")

    # -- sampling ------------------------------------------------------------

    def sample_once(self) -> None:
        """Close one interval: read the samplers and attribute its energy."""
        if not self.available:
            return
        with self._sampling:
            if self._energy is not None:  # None once closed
                self._sample_locked()

    def _sample_locked(self) -> None:
        started = time.perf_counter()
        reading = self._energy.read()
        delivered, active = self._token_source()
        temperatures = None
        if self._temperature is not None:
            try:
                temperatures = self._temperature.die_summary()
            except Exception as error:  # noqa: BLE001 - temperatures are optional
                self.temperature_reason = f"{type(error).__name__}: {error}"
        if self._frequency_table is None:
            self._resolve_frequency_table(reading)
        self._observe(reading, temperatures, delivered, active,
                      time.perf_counter() - started)
        watts, why = governed_watts(reading)
        with self._lock:
            self._signal_reason = why
            if why is not None:
                self._signal_gaps += 1
        if watts is not None:
            seconds = float(reading.seconds)
            for listener in tuple(self._listeners):
                listener(seconds, watts)

    def add_listener(self, listener) -> None:
        """Receive ``(seconds, gpu_dram_watts)`` for every closed interval."""
        self._listeners.append(listener)

    def _resolve_frequency_table(self, reading) -> None:
        self._frequency_table, self._frequency_reason = frequency_mapping(
            self._frequency_table_fn, reading.gpu_states
        )

    def _observe(self, reading, temperatures, delivered, active, cost) -> None:
        seconds = float(reading.seconds)
        watts = {
            label: float(reading.watts[channel])
            for label, channel in DOMAINS
            if reading.watts.get(channel) is not None
        }
        joules = {label: value * seconds for label, value in watts.items()}
        attributable = sum(joules.get(label, 0.0) for label in ATTRIBUTED_DOMAINS)
        states = [(name, residency) for name, residency in reading.gpu_states if name]
        residency_total = sum(residency for _, residency in states)
        mean_pstate, mean_mhz = gpu_state_means(states, self._frequency_table or ())
        with self._lock:
            output_tokens = max(0, int(delivered) - self._delivered)
            self._delivered = int(delivered)
            live = {
                request_id: progress for request_id, progress in active.items()
                if request_id not in self._closed
            }
            shares = {}
            if output_tokens:
                basis = "output_tokens"
                for request_id, (tokens, _uncached, _prefilling) in live.items():
                    entry = self._requests.get(request_id)
                    emitted = tokens - (entry.seen_tokens if entry else 0)
                    if emitted > 0:
                        shares[request_id] = emitted / output_tokens
            else:
                weights = {
                    request_id: uncached
                    for request_id, (_tokens, uncached, prefilling) in live.items()
                    if prefilling and uncached > 0
                }
                total_weight = sum(weights.values())
                basis = "prompt_tokens" if total_weight else "idle"
                shares = {
                    request_id: weight / total_weight
                    for request_id, weight in weights.items()
                }
            share_total = sum(shares.values())
            if share_total > 1.0:  # defensive: never attribute more than measured
                shares = {key: value / share_total for key, value in shares.items()}
                share_total = 1.0
            for request_id, (tokens, _uncached, _prefilling) in live.items():
                entry = self._requests.setdefault(request_id, _Request())
                entry.seen_tokens = tokens
                share = shares.get(request_id)
                if share:
                    entry.joules += attributable * share
                    entry.intervals += 1
            for request_id in [key for key in self._requests if key not in live]:
                del self._requests[request_id]
            self._closed &= set(active)
            attributed = attributable * share_total
            self._attributed += attributed
            if basis == "idle":
                self._idle += attributable
            else:
                # Output share left over belongs to requests that finished
                # inside this interval; their receipts already estimated it.
                self._finished_estimated += attributable - attributed
            for label, value in joules.items():
                self._energy_total[label] = self._energy_total.get(label, 0.0) + value
            if residency_total:
                for name, residency in states:
                    if name in GPU_STATES:
                        self._residency_total[name] = (
                            self._residency_total.get(name, 0.0)
                            + residency / residency_total * seconds
                        )
            self._append_window_locked(seconds, attributable, output_tokens)
            self._samples += 1
            self._last_cost = cost
            self._last = {
                "seconds": seconds,
                "watts": watts,
                "gpu_active_ratio": reading.gpu_active_fraction(),
                "gpu_mean_pstate": mean_pstate,
                "gpu_frequency_mean_mhz": mean_mhz,
                "die_temperature_c": (
                    {
                        "max": temperatures.get("die_max_c"),
                        "mean": temperatures.get("die_mean_c"),
                    }
                    if temperatures else None
                ),
                "output_tokens": output_tokens,
                "attribution_basis": basis,
            }

    def _append_window_locked(self, seconds, joules, tokens) -> None:
        window = self._window
        window.append((seconds, joules, tokens))
        self._window_seconds += seconds
        if len(window) > self._window_records:
            self._window_seconds -= window.popleft()[0]
        limit = self.policy.window_seconds
        while window:
            excess = self._window_seconds - limit
            if excess <= limit * 1e-9:
                break
            oldest = window[0]
            if oldest[0] <= excess:
                window.popleft()
                self._window_seconds -= oldest[0]
            else:
                # Keep the newest part of the oldest record, in proportion.
                keep = (oldest[0] - excess) / oldest[0]
                window[0] = (oldest[0] - excess, oldest[1] * keep, oldest[2] * keep)
                self._window_seconds = limit
                break
        if not window:
            self._window_seconds = 0.0

    # -- views ---------------------------------------------------------------

    def _window_locked(self) -> dict:
        seconds = sum(record[0] for record in self._window)
        joules = sum(record[1] for record in self._window)
        tokens = sum(record[2] for record in self._window)
        return {
            "seconds": seconds,
            "intervals": len(self._window),
            "gpu_dram_joules": joules,
            "output_tokens": tokens,
            "tokens_per_second": tokens / seconds if seconds else None,
            "joules_per_output_token": joules / tokens if tokens else None,
        }

    def request_energy(self, request_id, completion_tokens) -> dict:
        """Close ``request_id``'s energy account for its terminal receipt."""
        if not self.available:
            return {"schema": RECEIPT_SCHEMA, "available": False, "reason": self.reason}
        with self._lock:
            entry = self._requests.pop(request_id, None)
            self._closed.add(request_id)
            measured = entry.joules if entry is not None else 0.0
            seen = entry.seen_tokens if entry is not None else 0
            pending = max(0, int(completion_tokens or 0) - seen)
            rate = self._window_locked()["joules_per_output_token"]
        tail = pending * rate if rate is not None else (0.0 if not pending else None)
        total = measured + (tail or 0.0)
        return {
            "schema": RECEIPT_SCHEMA,
            "available": True,
            "estimate": True,
            "attribution": ATTRIBUTION_LAW,
            "domains": list(ATTRIBUTED_DOMAINS),
            "joules": total,
            "measured_interval_joules": measured,
            "measured_intervals": entry.intervals if entry is not None else 0,
            "open_interval_tokens": pending,
            # None: no rolling rate yet, so the tail tokens are uncharged.
            "open_interval_joules": tail,
            "joules_per_output_token": (
                total / completion_tokens if completion_tokens else None
            ),
        }

    def status(self) -> dict:
        base = {
            "enabled": True,
            "available": self.available,
            "reason": self.reason,
            "source": "ioreport",
            "privileged": False,
            "interval_seconds": self.policy.interval_seconds,
            "window_seconds": self.policy.window_seconds,
        }
        if not self.available:
            return base
        with self._lock:
            return {
                **base,
                "running": self.running,
                "samples": self._samples,
                "errors": self._errors,
                "last_error": self._last_error,
                "last_sample_cost_seconds": self._last_cost,
                "last": dict(self._last) if self._last is not None else None,
                "energy_joules_total": dict(self._energy_total),
                "gpu_pstate_residency_seconds_total": dict(self._residency_total),
                "gpu_frequency": {
                    "mapped": bool(self._frequency_table),
                    "table_mhz": list(self._frequency_table or ()),
                    "reason": self._frequency_reason,
                },
                "temperature_reason": self.temperature_reason,
                "governor_signal": {
                    "complete": self._samples > 0 and self._signal_reason is None,
                    "reason": self._signal_reason,
                    "incomplete_intervals": self._signal_gaps,
                },
                "window": self._window_locked(),
                "attribution": {
                    "law": ATTRIBUTION_LAW,
                    "estimate": True,
                    "domains": list(ATTRIBUTED_DOMAINS),
                    "attributed_joules": self._attributed,
                    "finished_in_interval_joules": self._finished_estimated,
                    "idle_joules": self._idle,
                    "tracked_requests": len(self._requests),
                },
            }

