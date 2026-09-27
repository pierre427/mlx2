"""Keep the serving process's weights GPU-resident for its whole lifetime.

MLX (this venv: 0.32.2.dev20260919+39400a0d4, ``mlx/backend/metal/
resident.cpp``) tracks every Metal allocation in size-capped
``MTLResidencySet``s with a standing ``requestResidency()``, but admits
allocations into them only up to the wired limit set by
``mx.set_wired_limit`` -- 0 by default, so nothing is wired unless asked.

mlx2 used to raise that limit only inside ``BatchGenerator`` (the ordinary and
native-MTP routes).  The prompt-lookup and external-draft generators never
did, so on those routes the weights were ordinary pageable memory.  Raising
it once, right after the adapter loads and before any cache exists, covers
every route and admits the weights first: when the limit grows, MLX adds the
allocations that exist at that moment, and at that point they are the weights.
On a small host (36 GB M3 Pro, recommendedMaxWorkingSetSize 28 GiB) later
KV/prefix-cache buffers fill whatever is left of the budget; the weights keep
their place.

Design input (idea only, no code): Inco Splash, Apache-2.0, f786bed, PR #148
("Keep model weights resident while serving").
"""

from __future__ import annotations

import mlx.core as mx


def wire_serving_weights() -> dict:
    """Set the process wired limit to the device's recommended working set.

    Idempotent and process-lifetime: it is never restored, because the
    process exists to serve this model.  Returns a receipt for /v1/status.
    """
    try:
        available = mx.metal.is_available()
    except Exception:  # noqa: BLE001 - a non-Metal build has nothing to wire
        available = False
    info = mx.device_info() if available else {}
    limit = info.get("max_recommended_working_set_size")
    if not available or not limit:
        return {"wired": False, "reason": "no Metal working-set size"}
    previous = mx.set_wired_limit(int(limit))
    return {
        "wired": True,
        "wired_limit_bytes": int(limit),
        "previous_wired_limit_bytes": int(previous),
        "active_bytes_at_wire": int(mx.get_active_memory()),
        "mechanism": "mlx set_wired_limit -> MTLResidencySet requestResidency",
    }
