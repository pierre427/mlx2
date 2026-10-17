"""Live-row compaction around Nemotron Mamba projections.

The hybrid cache keeps a rectangular right-padded slab so attention and
recurrent state retain their ordinary layout.  Projection arithmetic is
row-independent, however, so an adapter-selected Mamba block may gather only
the live rows, call its existing (possibly quantized) projection unchanged,
and scatter zeroes back into the padding positions.

This module does not define scheduling or cache semantics.  The scheduler
publishes per-lane lengths and the Nemotron adapter opts in explicitly.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


def adaptive_row_exact_selected(boundary: str, shape, lengths) -> bool:
    """Measured Super projection selector for a right-padded slab.

    The first source-bound component sweep found no input-projection win and
    found an output-projection win only at the B4/high-padding geometry.  Keep
    this model-specific cost decision out of the scheduler and default off.
    """

    if boundary != "out_proj" or len(shape) < 2:
        return False
    batch, width = map(int, shape[:2])
    padding = batch * width - sum(int(value) for value in lengths)
    return batch == 4 and width == 4 and padding >= 5


@lru_cache(maxsize=64)
def _live_indices(lengths: tuple[int, ...], width: int):
    import numpy as np

    rows = []
    for batch, length in enumerate(lengths):
        rows.extend(batch * width + column for column in range(length))
    return mx.array(np.asarray(rows, dtype=np.int32))


def project_live_rows(
    module,
    values,
    lengths,
    counters,
    *,
    boundary: str,
    arithmetic: str = "stock",
):
    """Project live right-padded rows and restore rectangular geometry.

    Returns ``None`` when the geometry is not a padded multi-row slab.  The
    caller then executes the ordinary projection.  A selected geometry either
    completes exactly or raises; it never silently substitutes other math.
    """

    if values.ndim != 3:
        counters["geometry_declines"] += 1
        return None
    batch, width, features = map(int, values.shape)
    if not isinstance(lengths, (list, tuple)) or len(lengths) != batch:
        counters["length_declines"] += 1
        return None
    lengths = tuple(int(value) for value in lengths)
    if any(value < 0 or value > width for value in lengths):
        raise ValueError("Nemotron projection row lengths exceed the slab")
    live = sum(lengths)
    rectangular = batch * width
    padding = rectangular - live
    counters["scopes"] += 1
    counters["logical_rows"] += live
    counters["rectangular_rows"] += rectangular
    counters["padding_rows"] += padding
    if batch < 2 or padding == 0:
        counters["unpadded_declines"] += 1
        return None
    if live == 0:
        raise ValueError("Nemotron projection slab cannot contain only padding")
    indices = _live_indices(lengths, width)
    packed = values.reshape(rectangular, features)[indices].reshape(1, live, features)
    if arithmetic == "stock":
        projected = module(packed)
        route = "stock"
    elif arithmetic == "row_exact":
        from .row_exact_qmv import quantized_linear

        projected, route = quantized_linear(module, packed)
    else:
        raise ValueError("Nemotron projection arithmetic must be stock or row_exact")
    output_features = int(projected.shape[-1])
    restored = mx.zeros((rectangular, output_features), dtype=projected.dtype)
    restored[indices] = projected.reshape(live, output_features)
    counters[f"{boundary}_compaction_calls"] += 1
    counters[f"{boundary}_live_rows"] += live
    counters[f"{boundary}_padding_rows_skipped"] += padding
    counters[f"{boundary}_{route}_calls"] += 1
    return restored.reshape(batch, width, output_features)


__all__ = ["adaptive_row_exact_selected", "project_live_rows"]
