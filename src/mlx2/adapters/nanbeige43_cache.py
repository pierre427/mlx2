"""Nanbeige-owned bounded capacity and explicit approximate KV declarations.

Serving applies the declaration through its revision-bound approximate-state
operation. This module never secretly replaces an exact serving cache with Q8.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

SCHEMA = "mlx2.nanbeige43-cache-candidates.v1"


def approximate_operations():
    from ..runtime.approximate_kv import KVQuantizationDescriptor
    return {"kv_q8": KVQuantizationDescriptor(8, 8, group_size=64)}


def install_reservation(model, tokens: int, *, maximum: int = 8192):
    """Bounded opt-in reservation for newly owned B1 exact KV caches."""
    if type(maximum) is not int or maximum < 1:
        raise ValueError("KV reservation maximum must be a positive integer")
    if type(tokens) is not int or not 1 <= tokens <= maximum:
        raise ValueError("KV reservation must fit its explicitly admitted maximum")
    if getattr(model, "_mlx2_kv_reservation", None) is not None:
        raise ValueError("KV reservation is already installed")
    ordinary_factory = model.make_cache

    def make_cache():
        from ..runtime.models.cache import KVCache
        caches = ordinary_factory()
        if any(type(cache) is not KVCache for cache in caches):
            raise ValueError("Nanbeige reservation requires exact plain KV cache owners")
        for cache in caches:
            cache.reserve(tokens)
        return caches

    def prepare_decode_cache(caches):
        from ..runtime.models.cache import BatchKVCache, KVCache
        if any(type(cache) not in (KVCache, BatchKVCache) for cache in caches):
            raise ValueError("Nanbeige reservation requires exact KV cache owners")
        for cache in caches:
            # A plan, not permission to overwrite any restored spare capacity.
            # Batch merge strips APC spare storage before the first append.
            cache._reserve_tokens = max(tokens, getattr(cache, "_reserve_tokens", 0))

    model.make_cache = make_cache
    model._mlx2_prepare_decode_cache = prepare_decode_cache
    args = model.args
    # Activations in this artifact are BF16: geometry estimate, not resident bytes.
    bytes_per_token = int(args.num_hidden_layers) * int(args.num_key_value_heads) * int(args.head_dim) * 2 * 2
    result = {"schema": SCHEMA, "tokens": tokens, "maximum": maximum,
              "planned_bf16_kv_bytes": tokens * bytes_per_token,
              "committed_state_changed": False, "qualified": False,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    model._mlx2_kv_reservation = result
    return result


def candidate_q8_cache(model, *, allow_approximate: bool = False):
    """Direct-probe factory; production publication uses approximate_operations.

    Ordinary dense/quantized attention may consume the packed representation.
    Generic exact AST transactions intentionally refuse this cache class.
    """
    if allow_approximate is not True:
        raise ValueError("packed Q8 KV is an explicit approximate candidate")
    from ..runtime.models.cache import QuantizedKVCache
    return [QuantizedKVCache(group_size=64, bits=8) for _ in model.layers]
