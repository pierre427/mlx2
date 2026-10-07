"""APCv2 retention-policy parsing, kept free of model-module imports.

``ServingEngine`` validates the execution policy before an adapter pins its
import-time environment; importing ``apc_v2`` there would load
``runtime.models`` early and trip every adapter's import-order guard.
"""

from __future__ import annotations

import math
from typing import Optional


APC_RETENTION_VALUE_DEFAULTS = {
    "policy": "value",
    # Decay of an entry's hit rate: a hit this long ago counts half.
    "half_life_seconds": 600.0,
    # Prefill depth at which attention work equals the linear (projection,
    # MLP) work, so recomputing ``d`` tokens costs d + d^2 / (2 * this).
    "attention_tokens": 8192,
}


def apc_retention_policy(value) -> Optional[dict]:
    """Validate the opt-in APCv2 retention policy; None keeps LRU.

    ``"lru"`` (or None) is the historical rank-then-recency order.
    ``"value"`` or ``{"policy": "value", ...}`` evicts the entry with the
    least prefill cost saved per resident byte first (see
    ``APCv2._entry_retention_value_locked``).
    """
    if value is None or value == "lru":
        return None
    if value == "value":
        value = {"policy": "value"}
    if not isinstance(value, dict) or value.get("policy") != "value":
        raise ValueError('apc_retention_policy must be "lru", "value" or a value-policy object')
    unknown = set(value) - set(APC_RETENTION_VALUE_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown APC retention policy settings: {sorted(unknown)}")
    policy = {**APC_RETENTION_VALUE_DEFAULTS, **value}
    half_life = policy["half_life_seconds"]
    if (
        isinstance(half_life, bool)
        or not isinstance(half_life, (int, float))
        or not math.isfinite(half_life)
        or half_life <= 0
    ):
        raise ValueError("APC retention half_life_seconds must be finite and positive")
    attention = policy["attention_tokens"]
    if isinstance(attention, bool) or not isinstance(attention, int) or attention < 1:
        raise ValueError("APC retention attention_tokens must be a positive integer")
    policy["half_life_seconds"] = float(half_life)
    return policy
