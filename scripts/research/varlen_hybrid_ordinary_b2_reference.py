"""Import-safe ordinary B2 hybrid reference for an explicit research gate.

This reuses the runtime's existing cache merge and the already configured
ordinary model. It does not install a route, replace model math, or select a
native cache. MLX and generation imports occur only when the caller invokes it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _mx_module():
    import mlx.core as mx
    return mx


def _as_host_ints(values: Any) -> tuple[int, ...]:
    return tuple(int(value.item()) for value in values)


@dataclass
class OrdinaryHybridB2Reference:
    """One stock B2 cache, merged once from two completed ordinary B1 prompts."""

    model: Any
    merged_cache: list[Any]
    full_attention_layers: tuple[int, ...]
    recurrent_layers: tuple[int, ...]
    initial_offsets: tuple[int, int]
    mx: Any
    steps: int = 0

    @classmethod
    def from_row_caches(cls, model: Any, row_caches: tuple[tuple[Any, ...], ...],
                        *, full_attention_layers: tuple[int, ...],
                        recurrent_layers: tuple[int, ...],
                        expected_offsets: tuple[int, int], mx: Any = None):
        if (type(row_caches) is not tuple or len(row_caches) != 2 or
                any(type(row) is not tuple for row in row_caches) or
                len(row_caches[0]) == 0 or len(row_caches[0]) != len(row_caches[1])):
            raise ValueError("two equal-width ordinary cache rows are required")
        count = len(row_caches[0])
        if (type(full_attention_layers) is not tuple or
                type(recurrent_layers) is not tuple or
                tuple(sorted(full_attention_layers + recurrent_layers)) != tuple(range(count)) or
                len(set(full_attention_layers + recurrent_layers)) != count):
            raise ValueError("full-attention and recurrent layer map is incomplete")
        if (type(expected_offsets) is not tuple or len(expected_offsets) != 2 or
                any(type(n) is not int or n < 1 for n in expected_offsets)):
            raise ValueError("two positive prompt offsets are required")
        mx = mx or _mx_module()
        for row, expected in zip(row_caches, expected_offsets):
            for index in full_attention_layers:
                cache = row[index]
                if (type(getattr(cache, "offset", None)) is not int or
                        cache.offset != expected or cache.keys is None or
                        cache.values is None):
                    raise ValueError("ordinary full-attention row is not at prompt boundary")
            for index in recurrent_layers:
                cache = row[index]
                if (len(getattr(cache, "cache", ())) != 2 or
                        any(value is None or value.shape[0] != 1 for value in cache.cache) or
                        bool(getattr(cache, "speculating", False))):
                    raise ValueError("ordinary recurrent row is not an initialized B1 boundary")
        from mlx2.runtime.generate import _merge_caches
        merged = _merge_caches([list(row) for row in row_caches])
        if len(merged) != count:
            raise RuntimeError("ordinary cache merge dropped a model layer")
        maximum = max(expected_offsets)
        for index in full_attention_layers:
            cache = merged[index]
            if (_as_host_ints(cache.offset) != expected_offsets or
                    _as_host_ints(cache.left_padding) !=
                    tuple(maximum - n for n in expected_offsets) or
                    cache.keys.shape[0] != 2 or cache.values.shape[0] != 2):
                raise RuntimeError("stock ragged KV merge lost logical offsets or padding")
        for index in recurrent_layers:
            cache = merged[index]
            if (len(getattr(cache, "cache", ())) != 2 or
                    any(value is None or value.shape[0] != 2 for value in cache.cache) or
                    bool(getattr(cache, "speculating", False))):
                raise RuntimeError("stock recurrent merge lost B2 state rows")
        return cls(model, merged, full_attention_layers, recurrent_layers,
                   expected_offsets, mx)

    def forward_one(self, token_ids: tuple[int, int]):
        """One ordinary B2 ready step; cache stays merged for the next step."""
        if (type(token_ids) is not tuple or len(token_ids) != 2 or
                any(type(token) is not int or token < 0 for token in token_ids)):
            raise ValueError("two nonnegative token IDs are required")
        x = self.mx.array(((token_ids[0],), (token_ids[1],)))
        hidden = self.model.model(x, cache=self.merged_cache)
        logits = self.model.logits(hidden)[:, -1, :]
        roots = tuple(value for cache in self.merged_cache
                      for value in cache.state if value is not None)
        self.mx.eval(logits, *roots)
        if not bool(self.mx.all(self.mx.isfinite(logits)).item()):
            raise ValueError("ordinary B2 reference produced nonfinite logits")
        expected = tuple(offset + self.steps + 1 for offset in self.initial_offsets)
        if any(_as_host_ints(self.merged_cache[index].offset) != expected
               for index in self.full_attention_layers):
            raise RuntimeError("ordinary B2 reference full-attention offset drifted")
        self.steps += 1
        return logits


def compare_recurrent_slots(private_lanes: tuple[tuple[Any, ...], ...],
                            reference: OrdinaryHybridB2Reference, *,
                            atol: float, rtol: float = 0.0,
                            relative_floor: float = 1e-6) -> dict[str, Any]:
    """Compare every GDN/conv leaf against the same ordinary B2 step.

    The pass condition is elementwise |private-reference| <= atol +
    rtol*|reference|. Relative error uses an explicit floor for near-zero
    values and is diagnostic, not the acceptance rule.
    """
    if (type(private_lanes) is not tuple or len(private_lanes) != 2 or
            any(type(lane) is not tuple or len(lane) != len(reference.recurrent_layers)
                for lane in private_lanes)):
        raise ValueError("two recurrent lanes in model layer order are required")
    if any(not math.isfinite(value) or value < 0 for value in (atol, rtol)) or not (
            math.isfinite(relative_floor) and relative_floor > 0):
        raise ValueError("finite nonnegative tolerances and positive relative floor required")
    mx = reference.mx
    slots = []
    for ordinal, layer_index in enumerate(reference.recurrent_layers):
        ref_cache = reference.merged_cache[layer_index]
        for lane_index in range(2):
            private = private_lanes[lane_index][ordinal]
            if len(getattr(private, "cache", ())) != 2:
                raise ValueError("private recurrent cache is incomplete")
            for slot_index in range(2):
                actual = private.cache[slot_index]
                batched = ref_cache.cache[slot_index]
                if actual is None or batched is None:
                    raise ValueError("recurrent slot is uninitialized")
                expected = batched[lane_index:lane_index + 1]
                if actual.shape != expected.shape or actual.dtype != expected.dtype:
                    raise ValueError("recurrent slot shape/dtype differs from ordinary B2")
                actual32 = actual.astype(mx.float32)
                expected32 = expected.astype(mx.float32)
                if not bool(mx.all(mx.isfinite(actual32) & mx.isfinite(expected32)).item()):
                    raise ValueError("recurrent slot contains nonfinite values")
                delta = mx.abs(actual32 - expected32)
                max_abs = float(mx.max(delta).item())
                max_rel = float(mx.max(delta / mx.maximum(mx.abs(expected32), relative_floor)).item())
                passed = bool(mx.all(delta <= atol + rtol * mx.abs(expected32)).item())
                slots.append({"lane": lane_index, "layer": layer_index,
                              "slot": slot_index, "max_abs": max_abs,
                              "max_rel": max_rel, "passed": passed})
    return {"passed": all(slot["passed"] for slot in slots),
            "max_abs": max(slot["max_abs"] for slot in slots),
            "max_rel": max(slot["max_rel"] for slot in slots),
            "slots": slots, "atol": atol, "rtol": rtol,
            "relative_floor": relative_floor,
            "reference": "stock-merged-ordinary-B2"}


__all__ = ["OrdinaryHybridB2Reference", "compare_recurrent_slots"]
