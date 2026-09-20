# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from __future__ import annotations
from typing import Dict

COUNTER_NAMES = (
    "ple_tail_prefetch_requests",
    "ple_tail_prefetch_tables",
    "ple_tail_prefetch_declined",
    "ple_tail_prefetch_failures",
    "ple_prefetch_submitted",
    "ple_prefetch_rows",
    "ple_dq_hits",
    "ple_dq_misses",
    "device_sampled_drafts",
    "hedge_built",
    "hedge_hit",
    "hedge_miss",
    "hedge_consumed",
    "hedge_skipped",
    "hedge_discarded",
    "eager_async_evals",
    "eager_dispatch_forwards",
    "eager_dispatch_row_declines",
    "qsa_pooled_key_cache_hits",
    "qsa_pooled_key_cache_misses",
    "qsa_scatter_chosen_calls",
)
_COUNTERS: Dict[str, float] = {name: 0 for name in COUNTER_NAMES}


def bump(name: str, amount=1) -> None:
    """Increment a mechanism counter (GIL-serialized; used from the hot loop
    and the PLE prefetch pool, where a lost increment costs nothing)."""
    _COUNTERS[name] += amount


def counters() -> Dict[str, float]:
    return dict(_COUNTERS)


def reset_counters() -> None:
    for name in COUNTER_NAMES:
        _COUNTERS[name] = 0
