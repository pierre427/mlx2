"""Actual full-attention boundary probe for the default-off hybrid Q1 gate.

The ordinary capture wraps the loaded model's real SDPA entry point. The
candidate capture receives the actual native result from an opt-in callback.
Only the separately labelled replay calls stock SDPA a second time. No MLX
or model runtime is imported until a caller invokes this research module.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any


LAYERS = (7, 27)


def _mx():
    import mlx.core as mx
    return mx


def _maximum_difference(left, right, mx) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError("FA probe tensor geometry differs")
    left32, right32 = left.astype(mx.float32), right.astype(mx.float32)
    mx.eval(left32, right32)
    if not bool(mx.all(mx.isfinite(left32) & mx.isfinite(right32)).item()):
        raise ValueError("FA probe tensor is nonfinite")
    return float(mx.max(mx.abs(left32 - right32)).item())


class FABoundaryProbe:
    """Root actual candidate tensors and intercept actual ordinary SDPA calls."""

    def __init__(self, *, target_layers: tuple[int, ...] = LAYERS):
        if (type(target_layers) is not tuple or target_layers != LAYERS):
            raise ValueError("this probe is bounded to full-attention layers 7 and 27")
        self.candidate: dict[int, dict[str, Any]] = {}
        self.ordinary: dict[int, dict[str, Any]] = {}
        self._stock_sdpa = None
        self._ordinary_calls = 0

    def candidate_callback(self, event: dict[str, Any]) -> None:
        """Pass directly as candidate._fa_boundary_probe after source hookup."""
        layer = event.get("layer_index")
        if layer not in LAYERS:
            return
        required = {"layer_index", "fa_index", "offsets", "queries", "keys",
                    "values", "native_attention", "hidden_dtype"}
        if (type(event) is not dict or set(event) != required or
                layer in self.candidate or type(event["offsets"]) not in (tuple, list) or
                len(event["offsets"]) != 2):
            raise ValueError("hybrid FA callback is incomplete or duplicated")
        self.candidate[layer] = dict(event)

    @contextmanager
    def capture_ordinary(self, attention_module: Any):
        """Wrap the already configured model module during ONE stock B2 step.

        If invariant SDPA bypasses this symbol, zero/missing captures fail the
        final audit; no substitute attention output is invented.
        """
        original = attention_module.scaled_dot_product_attention
        if self._stock_sdpa is not None:
            raise RuntimeError("ordinary FA capture is already active")
        self._stock_sdpa = original
        self._ordinary_calls = 0

        def observe(queries, keys, values, *, cache, scale, mask, **kwargs):
            ordinal = self._ordinary_calls
            self._ordinary_calls += 1
            layer = 4 * ordinal + 3
            result = original(queries, keys, values, cache=cache,
                              scale=scale, mask=mask, **kwargs)
            if layer in LAYERS:
                self.ordinary[layer] = {
                    "queries": queries, "keys": keys, "values": values,
                    "attention": result, "mask": mask, "cache": cache,
                    "scale": scale, "layer_index": layer,
                    "logical_offsets": cache.offset,
                    "left_padding": cache.left_padding,
                }
            return result

        attention_module.scaled_dot_product_attention = observe
        try:
            yield
        finally:
            attention_module.scaled_dot_product_attention = original
            if self._ordinary_calls != 16 or set(self.ordinary) != set(LAYERS):
                raise RuntimeError("actual ordinary FA entry points were not captured")

    def compare(self, branches: tuple[Any, Any], *,
                attention_module: Any, mx: Any = None) -> dict[str, Any]:
        """Inspect terminal private KV and distinguish actual output from replay."""
        if (type(branches) is not tuple or len(branches) != 2 or
                set(self.candidate) != set(LAYERS) or
                set(self.ordinary) != set(LAYERS) or self._stock_sdpa is None):
            raise ValueError("both complete actual FA captures and branches required")
        mx = mx or _mx()
        details = []
        for layer in LAYERS:
            candidate = self.candidate[layer]
            ordinary = self.ordinary[layer]
            ordinal = candidate["fa_index"]
            if ordinal != layer // 4 or any(ordinal >= len(branch.layers) for branch in branches):
                raise ValueError("candidate FA layer ordinal differs")
            native_rows = tuple(_export_logical_native_kv(branch.layers[ordinal], mx)
                                for branch in branches)
            native_keys = tuple(row[0] for row in native_rows)
            native_values = tuple(row[1] for row in native_rows)
            offsets = tuple(row.shape[1] for row in native_keys)
            before = tuple(int(value) for value in candidate["offsets"])
            if offsets != tuple(value + 1 for value in before):
                raise RuntimeError("private FA table does not cover callback token")
            pad = tuple(int(value.item()) for value in ordinary["left_padding"])
            ordinary_offsets = tuple(int(value.item()) for value in ordinary["logical_offsets"])
            if ordinary_offsets != offsets or pad != tuple(max(offsets) - n for n in offsets):
                raise RuntimeError("ordinary FA mask/cache offsets differ from private state")
            mask = ordinary["mask"]
            if mask is None or mask.dtype != mx.bool_ or tuple(mask.shape) != (2, 1, 1, max(offsets)):
                raise ValueError("ordinary ragged FA mask geometry is unsupported")
            mask_counts = tuple(int(mx.sum(mask[row]).item()) for row in range(2))
            if mask_counts != offsets:
                raise RuntimeError("ordinary FA mask does not expose exact logical KV")
            query = candidate["queries"][:, :, None, :]
            ordinary_query = ordinary["queries"]
            new_key = candidate["keys"]
            new_value = candidate["values"]
            new_key_error = _maximum_difference(new_key, ordinary["keys"][:, :, -1, :], mx)
            new_value_error = _maximum_difference(new_value, ordinary["values"][:, :, -1, :], mx)
            logical_key_error = max(_maximum_difference(
                native_keys[row], ordinary["keys"][row, :, pad[row]:pad[row] + offsets[row], :], mx)
                for row in range(2))
            logical_value_error = max(_maximum_difference(
                native_values[row], ordinary["values"][row, :, pad[row]:pad[row] + offsets[row], :], mx)
                for row in range(2))
            # Replay is explicitly diagnostic. The actual ordinary tensor is
            # captured above from the model forward, before gate and o_proj.
            replay_keys = mx.stack(tuple(mx.pad(native_keys[row],
                    ((0, 0), (pad[row], 0), (0, 0))) for row in range(2)))
            replay_values = mx.stack(tuple(mx.pad(native_values[row],
                    ((0, 0), (pad[row], 0), (0, 0))) for row in range(2)))
            replay = self._stock_sdpa(query, replay_keys, replay_values,
                                      cache=None, scale=ordinary["scale"], mask=mask)
            native = candidate["native_attention"]
            if native.ndim == 3:
                native = native[:, :, None, :]
            details.append({
                "layer": layer, "candidate_pre_offsets": before,
                "ordinary_post_offsets": ordinary_offsets,
                "mask_valid_tokens": mask_counts,
                "query_max_abs": _maximum_difference(query, ordinary_query, mx),
                "new_key_max_abs": new_key_error,
                "new_value_max_abs": new_value_error,
                "logical_key_max_abs": logical_key_error,
                "logical_value_max_abs": logical_value_error,
                "native_vs_actual_ordinary_attention_max_abs":
                    _maximum_difference(native, ordinary["attention"], mx),
                "native_vs_replayed_stock_sdpa_max_abs":
                    _maximum_difference(native, replay, mx),
                "replay_vs_actual_ordinary_attention_max_abs":
                    _maximum_difference(replay, ordinary["attention"], mx),
            })
        return {"schema": "mlx2.hybrid-fa-boundary-probe.v1",
                "actual_ordinary_calls": self._ordinary_calls,
                "candidate_layers": tuple(sorted(self.candidate)),
                "ordinary_layers": tuple(sorted(self.ordinary)),
                "details": details, "qualified": False}


def _export_logical_native_kv(owner: Any, mx: Any):
    """Copy accepted page bytes while the private owner and arena are pinned."""
    if owner._pending or owner._failed or owner.writer.poisoned or owner.sequence.retained_start != 0:
        raise RuntimeError("FA diagnostic needs a completed full-prefix private owner")
    handles = owner.accepted_handles()
    profile = owner.profile
    marker = mx.array([1], dtype=mx.uint8)
    dtype = mx.bfloat16 if profile.dtype == "bfloat16" else mx.float16
    pages = []
    for handle in handles:
        raw_key, raw_value = owner.writer.backend.diagnostic_read(
            marker, handle.page_id * profile.page_bytes, profile.page_bytes,
            permit_diagnostic=True)
        mx.eval(raw_key, raw_value)
        pages.append((raw_key.view(dtype).reshape(profile.kv_heads, 64, profile.head_dim),
                      raw_value.view(dtype).reshape(profile.kv_heads, 64, profile.head_dim)))
    if not pages or len(pages) != (owner.offset + 63) // 64:
        raise RuntimeError("FA diagnostic page count differs from logical state")
    keys = mx.concatenate(tuple(page[0][:, :min(64, owner.offset - index * 64), :]
                                for index, page in enumerate(pages)), axis=1)
    values = mx.concatenate(tuple(page[1][:, :min(64, owner.offset - index * 64), :]
                                  for index, page in enumerate(pages)), axis=1)
    return keys, values


__all__ = ["FABoundaryProbe", "LAYERS"]
