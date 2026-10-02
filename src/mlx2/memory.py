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
    re-measures.  It
    stops when the footprint's excess over MLX's ``active + cached`` (the
    *overhang*) is back within ``tolerance`` of the quiet-state overhang
    (``baseline``: Python heap, mapped tables and other non-MLX pages), when the
    footprint has not fallen for ``quiet`` seconds, or at ``timeout``.  The
    baseline is learned from settled readings; before one exists only the
    quiet and timeout exits apply.
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
        timeout=1.0,
        interval=0.01,
    ):
        import time

        self._read = read
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self.tolerance_bytes = int(tolerance_bytes)
        self.drop_bytes = int(drop_bytes)
        self.quiet = float(quiet)
        self.timeout = float(timeout)
        self.interval = float(interval)
        self.baseline = None

    @staticmethod
    def _overhang(reading):
        active, cached, footprint = reading
        return footprint - (active + cached)

    def observe(self):
        """Record a quiet-state overhang (for example once the model is loaded)."""
        reading = self._read()
        if reading is not None:
            self.baseline = max(0, self._overhang(reading))
        return self.baseline

    def _settled(self, reading):
        return (
            self.baseline is not None
            and self._overhang(reading) <= self.baseline + self.tolerance_bytes
        )

    def settle(self):
        """Wait (bounded) until recently freed pages leave the footprint."""
        reading = self._read()
        if reading is None:
            return {"waited": 0.0, "released_bytes": 0, "exit": "unmeasurable"}
        start = now = self._clock()
        first = lowest = reading[2]
        last_drop = start
        exit_reason = "settled"
        while not self._settled(reading):
            if now - start >= self.timeout:
                exit_reason = "timeout"
                break
            if now - last_drop >= self.quiet:
                exit_reason = "quiet"
                break
            self._sleep(self.interval)
            now = self._clock()
            fresh = self._read()
            if fresh is None:
                exit_reason = "unmeasurable"
                break
            reading = fresh
            if reading[2] < lowest - self.drop_bytes:
                lowest = reading[2]
                last_drop = now
        overhang = max(0, self._overhang(reading))
        if exit_reason == "quiet":
            # Nothing more is being returned: this is the quiet-state overhang,
            # even if non-MLX pages have grown since the last one.
            self.baseline = overhang
        elif exit_reason == "settled":
            self.baseline = min(self.baseline, overhang)
        return {
            "waited": now - start,
            "released_bytes": max(0, first - reading[2]),
            "exit": exit_reason,
        }
