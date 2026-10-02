"""Live process and host headroom for admission, including non-Metal pages."""


def available_execution_bytes(
    *, available, recommended, active, cached, footprint, host_quota_credit=0
):
    """``min(host available + credit, advisory - in use)``; 0 when unmeasurable.

    ``host_quota_credit`` (``host_term_reserve_credit_bytes``) is the part of
    the admission service reserve that the host-available term has already
    collected by measuring the rest of the host; see that function.
    """
    if available is None or footprint is None or recommended <= 0:
        return 0
    return max(
        0,
        min(
            available + max(0, int(host_quota_credit)),
            recommended - max(active + cached, footprint),
        ),
    )


def host_term_reserve_credit_bytes(physical_bytes, recommended_bytes):
    """Service reserve the host-available term must not charge a second time.

    Admission subtracts one hard reserve (service + driver [+ stream]) from
    ``min(host available, advisory - in use)``.  The service share is the
    host's *non-lane quota*: 25% of RAM on the calibration host, of which the
    advisory withholds 16 GiB for macOS and every other process and the
    service reserve adds 16 more (``SelfMTPLaneAdmissionController``
    docstring: "the rule charges the quota once").  The advisory term needs
    that charge: it knows nothing about the rest of the host.  The host term
    does not: host available is physical RAM minus *measured* use by this
    process **and by everything else**, so the rest of the host's actual
    pages have already been taken out of it.  Charging the full 16 GiB again
    there withheld ``others - 16`` GiB beyond the quota whenever the rest of
    the host used more than the advisory's 16 -- on this shared M5 Max, other
    sessions held 19-27 GiB, so a Flash-Next native-MTP B2 warm cohort that
    needed 3.6 GiB above the reserve saw 3.3 usable (HTTP 429) with 34 GiB of
    advisory residue left (2026-10-01 diagnosis of qualify-709bedf8).

    The credit is the service reserve above its floor
    (``MIN_SERVICE_RESERVE_GIB``), so the host term still keeps the floor
    plus the driver allowance (plus any streamed-weight ceiling) free after
    every admitted lane: 3 + 4 = 7 GiB of truly available host memory on the
    calibration host.  It is exact, not a bonus: ``min(A + credit, R - F)``
    equals charging the host term only the quota the rest of the host has not
    already used, clamped to the floor.  Where the rest of the host uses no
    more than the advisory's share the advisory term binds and nothing moves.
    Hosts whose service reserve is already at its floor (the 36 GiB M3 Pro)
    get no credit.  Unknown readings give no credit.
    """
    from .runtime.memory_policy import SelfMTPLaneAdmissionController as C

    if not physical_bytes or not recommended_bytes or physical_bytes <= 0 or recommended_bytes <= 0:
        return 0
    gib = float(1 << 30)
    (service, _driver) = C.host_scaled_reserves(physical_bytes / gib, recommended_bytes / gib)
    return int(max(0.0, service - C.MIN_SERVICE_RESERVE_GIB) * gib)


def host_available_bytes(host_signals=False):
    """Host-wide available memory for the admission ``available`` term.

    Default: psutil's figure.  With the server-owned ``host_memory_signals``
    policy the Mach-statistics estimate replaces it, which also counts
    file-backed and purgeable pages; psutil remains the fallback when the
    probe is unavailable.
    """
    if host_signals:
        from .runtime.os_memory import host_memory_snapshot

        snapshot = host_memory_snapshot()
        if snapshot is not None:
            return snapshot.available_bytes
    import psutil

    return psutil.virtual_memory().available


def metal_advisory_gib():
    """``max_recommended_working_set_size`` in GiB, or ``None`` if unreadable.

    This is the per-process ceiling ``available_execution_bytes`` applies
    above, and it is also how much of the host macOS has *already* withheld
    from us: 112.0 GiB of a 128 GiB host (87.5%), 28.08 GiB of a 36 GiB M3
    Pro (78.0%), both measured.  The admission reserve needs it because the
    withheld share is not a constant fraction -- see
    ``SelfMTPLaneAdmissionController.host_scaled_reserves``.  Probe once at
    server start and inject the result; it never changes.
    """
    try:
        import mlx.core as mx

        advisory = int(
            mx.device_info().get("max_recommended_working_set_size", 0) or 0
        )
    except Exception:  # noqa: BLE001 - a sizing probe must not break startup
        return None
    if advisory <= 0:
        return None
    return advisory / float(1 << 30)


def host_memory_gib():
    """Physical unified memory in GiB, or ``None`` when it cannot be measured.

    Reported alongside ``metal_advisory_gib`` above: the reserve is derived
    from both, because what it must add is the part of the host's non-lane
    quota that Metal's advisory has not already taken.
    Probe once at server start and inject the result -- it never changes.
    """
    total = 0
    try:
        import mlx.core as mx

        total = int(mx.device_info().get("memory_size", 0) or 0)
    except Exception:  # noqa: BLE001 - a sizing probe must not break startup
        total = 0
    if total <= 0:
        try:
            import psutil

            total = int(psutil.virtual_memory().total)
        except Exception:  # noqa: BLE001
            return None
    if total <= 0:
        return None
    return total / float(1 << 30)


def execution_headroom(host_signals=False):
    import mlx.core as mx
    from .runtime.os_memory import physical_footprint_bytes

    info = mx.device_info()
    recommended = info.get("max_recommended_working_set_size", 0)
    return available_execution_bytes(
        available=host_available_bytes(host_signals),
        recommended=recommended,
        active=mx.get_active_memory(),
        cached=mx.get_cache_memory(),
        footprint=physical_footprint_bytes(),
        host_quota_credit=host_term_reserve_credit_bytes(
            info.get("memory_size", 0), recommended
        ),
    )


def mlx_footprint_reading():
    """``(active, cached, footprint)`` bytes, or ``None`` when unmeasurable."""
    try:
        import mlx.core as mx
        from .runtime.os_memory import physical_footprint_bytes

        footprint = physical_footprint_bytes()
        if footprint is None:
            return None
        return (int(mx.get_active_memory()), int(mx.get_cache_memory()), int(footprint))
    except Exception:  # noqa: BLE001 - admission must survive probe failure
        return None


SETTLE_BUDGET_SECONDS = 1.0


def bounded_settle(settle, seconds=SETTLE_BUDGET_SECONDS, clock=None):
    """Share one settle deadline across every wait of one admission attempt.

    ``settle(timeout)`` is called with what is left of ``seconds``, counted
    from the first call; once it is spent the wrapper returns ``None``
    without waiting, so a reclaim / eviction / depth-fallback sequence blocks
    its thread for at most ``seconds`` in total and leaves the rest of the
    work to later scheduler rounds.  Wrapping is idempotent: an already
    bounded callable keeps its own (outer) deadline.
    """
    if settle is None or getattr(settle, "settle_budget", None) is not None:
        return settle
    import time

    clock = clock or time.monotonic
    deadline = []

    def bounded():
        now = clock()
        if not deadline:
            deadline.append(now + seconds)
        remaining = deadline[0] - now
        if remaining <= 0:
            return None
        return settle(timeout=remaining)

    bounded.settle_budget = seconds
    return bounded


class FootprintSettler:
    """Wait out Metal's deferred release of buffers MLX has just freed.

    ``mx.clear_cache()`` returns in microseconds, but macOS takes the freed
    ``MTLBuffer`` pages out of the process's physical footprint later: 8 GiB
    released in 256 MiB-1 GiB buffers stayed in the footprint, flat, for
    100-200 ms and was gone after 70-250 ms (M5 Max, 2026-10-02).  Admission
    charges ``max(active + cached, footprint)``, so a measurement taken in that
    window charges memory MLX no longer holds.  Every prefill chunk ends with a
    ``clear_cache``, so a 4 x 32K Flash-Next cohort's chunk re-check read 5-9
    GiB of such pages and refused (HTTP 429) a cohort it had admitted cold.

    Admission calls ``settle`` only when a reading taken after its
    synchronized reclaim still falls short, so a request that fits never
    waits.  It never credits bytes: it waits, bounded, and the caller
    re-measures.  It stops when the footprint's excess over MLX's
    ``active + cached`` (the *overhang*) is back within ``tolerance`` of the
    quiet-state overhang (``baseline``), when the footprint has not fallen
    for ``quiet`` seconds, or at the timeout.

    ``baseline`` comes from *verified* quiet readings of the last ``window``
    seconds: the reading at load, an idle serving-loop reading that matches
    the previous idle one (nothing was still being returned), and the final
    reading of a settle that had to wait.  Any other real reading in the
    window may still lower it -- a reading is only non-MLX residency plus
    pending frees, so a lower one is never optimistic -- but cannot stand in
    for a verified one: during a long prefill every loop reading lands right
    after a chunk's clear, and a baseline built from those alone would
    certify the very pages it should wait for.  With no verified reading in
    the window there is no baseline and only the quiet and timeout exits
    apply, so a stale high value cannot certify a reading taken right after
    a clear.  A drop in non-MLX residency lowers it at the next reading; a
    rise is picked up once the older, lower verified readings leave the
    window.  One lock serializes callers on the scheduler and HTTP threads.
    """

    def __init__(
        self,
        *,
        read=mlx_footprint_reading,
        clock=None,
        sleep=None,
        tolerance_bytes=512 << 20,
        drop_bytes=16 << 20,
        quiet=0.3,
        timeout=SETTLE_BUDGET_SECONDS,
        interval=0.01,
        window=10.0,
    ):
        from collections import deque
        import threading
        import time

        self._read = read
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self.tolerance_bytes = int(tolerance_bytes)
        self.drop_bytes = int(drop_bytes)
        self.quiet = float(quiet)
        self.timeout = float(timeout)
        self.interval = float(interval)
        self.window = float(window)
        self._samples = deque(maxlen=4096)  # (time, overhang, verified)
        self._last_idle = None
        self._lock = threading.Lock()

    @staticmethod
    def _overhang(reading):
        active, cached, footprint = reading
        return footprint - (active + cached)

    def _note(self, reading, now, verified):
        self._samples.append((now, max(0, self._overhang(reading)), bool(verified)))

    def _baseline(self, now):
        while self._samples and now - self._samples[0][0] > self.window:
            self._samples.popleft()
        if not any(verified for (_t, _o, verified) in self._samples):
            return None
        return min(o for (_t, o, _v) in self._samples)

    @property
    def baseline(self):
        return self._baseline(self._clock())

    def observe(self):
        """Record a known-quiet reading (the server calls this after load)."""
        reading = self._read()
        now = self._clock()
        if reading is not None:
            self._note(reading, now, verified=True)
        return self._baseline(now)

    def refresh(self, idle=False):
        """Record a serving-loop reading taken outside rejection.

        Never blocks: a settle in progress skips it.  An idle reading whose
        overhang matches the previous idle reading's is verified quiet.
        """
        if not self._lock.acquire(blocking=False):
            return None
        try:
            reading = self._read()
            now = self._clock()
            if reading is None:
                self._last_idle = None
                return self._baseline(now)
            overhang = self._overhang(reading)
            verified = (
                idle
                and self._last_idle is not None
                and abs(overhang - self._last_idle) <= self.drop_bytes
            )
            self._last_idle = overhang if idle else None
            self._note(reading, now, verified)
            return self._baseline(now)
        finally:
            self._lock.release()

    def settle(self, timeout=None):
        """Wait (bounded) until recently freed pages leave the footprint."""
        budget = self.timeout if timeout is None else min(self.timeout, float(timeout))
        if budget <= 0:
            return {"waited": 0.0, "released_bytes": 0, "exit": "budget"}
        start = self._clock()
        if not self._lock.acquire(timeout=budget):
            return {"waited": self._clock() - start, "released_bytes": 0, "exit": "busy"}
        try:
            return self._settle(start, budget)
        finally:
            self._lock.release()

    def _settle(self, start, budget):
        reading = self._read()
        if reading is None:
            return {"waited": 0.0, "released_bytes": 0, "exit": "unmeasurable"}
        now = self._clock()
        # Certify only against readings taken before this one: the first
        # reading after a clear may itself be inflated.
        baseline = self._baseline(now)
        first = lowest = reading[2]
        last_drop = now
        slept = False
        exit_reason = "settled"
        while baseline is None or self._overhang(reading) > baseline + self.tolerance_bytes:
            if now - start >= budget:
                exit_reason = "timeout"
                break
            if now - last_drop >= self.quiet:
                exit_reason = "quiet"
                break
            self._sleep(self.interval)
            slept = True
            fresh = self._read()
            now = self._clock()
            if fresh is None:
                exit_reason = "unmeasurable"
                break
            reading = fresh
            if reading[2] < lowest - self.drop_bytes:
                lowest = reading[2]
                last_drop = now
        # A settle that waited ended on a stable or baseline-level reading;
        # one certified at once proves nothing new.
        self._note(
            reading, now,
            verified=exit_reason == "quiet"
            or (exit_reason == "settled" and slept),
        )
        return {
            "waited": now - start,
            "released_bytes": max(0, first - reading[2]),
            "exit": exit_reason,
        }
