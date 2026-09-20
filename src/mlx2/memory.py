"""Live process and host headroom for admission, including non-Metal pages."""


def available_execution_bytes(*, available, recommended, active, cached, footprint):
    if available is None or footprint is None or recommended <= 0:
        return 0
    return max(0, min(available, recommended - max(active + cached, footprint)))


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

    return available_execution_bytes(
        available=host_available_bytes(host_signals),
        recommended=mx.device_info().get("max_recommended_working_set_size", 0),
        active=mx.get_active_memory(),
        cached=mx.get_cache_memory(),
        footprint=physical_footprint_bytes(),
    )
