"""Live process and host headroom for admission, including non-Metal pages."""


def available_execution_bytes(*, available, recommended, active, cached, footprint):
    if available is None or footprint is None or recommended <= 0:
        return 0
    return max(0, min(available, recommended - max(active + cached, footprint)))


def execution_headroom():
    import mlx.core as mx
    import psutil
    from .runtime.os_memory import physical_footprint_bytes

    return available_execution_bytes(
        available=psutil.virtual_memory().available,
        recommended=mx.device_info().get("max_recommended_working_set_size", 0),
        active=mx.get_active_memory(),
        cached=mx.get_cache_memory(),
        footprint=physical_footprint_bytes(),
    )
