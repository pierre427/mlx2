"""Host-scaled APCv2 resident prefix-cache default, and its post-load clamp.

Pure arithmetic: nothing here imports MLX, so the server's argument parser can
size its ``--cache-bytes`` default on a host without touching the GPU.
"""

from __future__ import annotations

import os

GIB = 1 << 30

# The historical rule, kept as the floor the post-load clamp never goes below:
# physical/8 inside [1 GiB, 16 GiB].  16 GiB is the qualified cache geometry
# on the 128 GiB host.  A flat 16 GiB is a hazard on the 36 GiB M3: its Metal
# advisory is 28.08 GiB, so a 19 GiB Qwen3.6 artifact plus a full 16 GiB
# prefix cache is past the advisory before one lane is costed.  One eighth of
# physical memory matches the 12.5% service share the host-scaled reserves use
# (runtime/memory_policy.py): 4.5 GiB at 36 GiB.
LEGACY_MAX_DEFAULT_CACHE_BYTES = 16 * GIB
MIN_DEFAULT_CACHE_BYTES = 1 * GIB

# The current curve.  Up to a 64 GiB host it is the legacy physical/8 rule,
# unchanged: those hosts cannot hold a mid-size model, its lanes and a larger
# cache under the advisory.  Above 64 GiB every further GiB of RAM gives the
# cache 5/8 GiB (``LARGE_HOST_EXCESS_SHARE``), so the curve is continuous at
# 64 GiB (8 GiB) and reaches 48 GiB at 128 GiB.  48 GiB is the measured need
# on the 128 GiB host: one Qwen3.8-27B APC entry is 3.7 GiB at 25K tokens and
# 4.7 GiB at 32K, so 16 GiB held three to four and evicted interior
# checkpoints as soon as they were stored (shared-prefix requests then fully
# re-prefilled 25-32K tokens in 38-51 s), while 48 GiB hit every one at
# 0.31-0.36 s (qualification/runs/sp-replay-27b-20260925 on r3-sp-small).
# The cap of 96 GiB is reached at ~205 GiB and holds the cache to 37.5% of a
# 256 GiB host and 19% of a 512 GiB host, where the models are larger too.
#
#   host GiB    16   36   64   96  128  192  256  512
#   cache GiB    2  4.5    8   28   48   88   96   96
#
# The default is sized before the model loads, so it cannot know the model's
# own footprint; ``clamp_host_default_cache_bytes`` bounds it after the load.
LARGE_HOST_THRESHOLD_BYTES = 64 * GIB
LARGE_HOST_EXCESS_SHARE = (5, 8)
MAX_DEFAULT_CACHE_BYTES = 96 * GIB

# Post-load clamp: the lanes the cache must leave room for, each costed at
# the route's depth floor with this much context.
CACHE_CLAMP_LANES = 4
CACHE_CLAMP_REFERENCE_CONTEXT = 8192


def physical_memory_bytes() -> int | None:
    """Physical RAM without importing MLX (the parser must stay GPU-free)."""
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def legacy_default_cache_bytes(physical=None) -> int:
    """The pre-2026-09-26 default: physical/8 in [1 GiB, 16 GiB]."""
    physical = physical_memory_bytes() if physical is None else physical
    if not physical or physical <= 0:
        return LEGACY_MAX_DEFAULT_CACHE_BYTES
    return max(
        MIN_DEFAULT_CACHE_BYTES,
        min(LEGACY_MAX_DEFAULT_CACHE_BYTES, int(physical) // 8),
    )


def default_cache_bytes(physical=None) -> int:
    """Host default for ``--cache-bytes``; see the curve above.

    An unmeasurable host keeps the legacy 128 GiB geometry (16 GiB) rather
    than the large-host curve it cannot be shown to have.
    """
    physical = physical_memory_bytes() if physical is None else physical
    if not physical or physical <= 0:
        return LEGACY_MAX_DEFAULT_CACHE_BYTES
    physical = int(physical)
    if physical <= LARGE_HOST_THRESHOLD_BYTES:
        cache = physical // 8
    else:
        (numerator, denominator) = LARGE_HOST_EXCESS_SHARE
        cache = LARGE_HOST_THRESHOLD_BYTES // 8 + (
            (physical - LARGE_HOST_THRESHOLD_BYTES) * numerator // denominator
        )
    return max(MIN_DEFAULT_CACHE_BYTES, min(MAX_DEFAULT_CACHE_BYTES, cache))


def clamp_host_default_cache_bytes(
    requested_bytes,
    *,
    floor_bytes,
    admission_limit_bytes,
    resident_bytes,
    stream_reserve_bytes,
    lane_need_bytes,
    lanes,
):
    """Bound a host-default cache by what the loaded model leaves free.

    APCv2 insertion is bounded only by its own ``max_bytes``; nothing checks
    headroom when a checkpoint is stored, and the MLX memory limit is a
    guideline that never evicts it.  Admission does evict unleased entries
    before it defers a lane, but leased entries and checkpoints published
    while lanes run are not reclaimable at that seam, so a cache larger than
    the room beside the model can push the process past the advisory
    (swap or a Metal out-of-memory) instead of being evicted.

    The budget is the MLX admission limit (advisory minus service and driver
    reserves) minus the resident model, the streamed-weight ceiling, and
    ``lanes`` lanes at their minimum admission need.  The result never falls
    below ``floor_bytes`` (the legacy default, which is already the operated
    geometry) and never rises above the request.  An unmeasurable headroom
    falls back to the floor.  Returns ``(effective_bytes, detail)``.
    """
    requested = int(requested_bytes)
    floor = min(requested, int(floor_bytes))
    detail = {
        "requested_bytes": requested,
        "floor_bytes": floor,
        "admission_limit_bytes": (
            None if admission_limit_bytes is None else int(admission_limit_bytes)
        ),
        "resident_bytes": None if resident_bytes is None else int(resident_bytes),
        "stream_reserve_bytes": int(stream_reserve_bytes),
        "lanes": int(lanes),
        "lane_need_bytes": None if lane_need_bytes is None else int(lane_need_bytes),
        "budget_bytes": None,
    }
    if admission_limit_bytes is None or resident_bytes is None or lane_need_bytes is None:
        detail["reason"] = "headroom_unmeasured"
        return (floor, detail)
    budget = (
        int(admission_limit_bytes)
        - int(resident_bytes)
        - int(stream_reserve_bytes)
        - int(lanes) * int(lane_need_bytes)
    )
    detail["budget_bytes"] = budget
    effective = min(requested, max(floor, budget))
    if effective >= requested:
        detail["reason"] = "fits"
    elif effective == floor:
        detail["reason"] = "legacy_floor"
    else:
        detail["reason"] = "headroom"
    return (effective, detail)
