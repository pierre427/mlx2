# SPDX-License-Identifier: Apache-2.0
"""Block recipes and the layerwise executor for exact verify strips.

A model adapter does not write its own verify loop.  It declares, per block,
how a ``[1, R, H]`` strip runs with each row's one-token decode arithmetic:

* ``PER_ROW`` - call the block once per row (always exact; the fallback);
* ``SEGMENT`` - the block's own mixer consumes a short row segment exactly
  (for example a recurrent mixer with a row-exact segment kernel and a
  rollback tape); the adapter vouches for it;
* ``residual(norm, mixer)`` - ``x + mixer(norm(x))`` where the norm is
  row-independent and ``mixer`` is an exact strip function built from
  ``row_exact.ops`` (a mixture-of-experts block, say).

``run_layerwise`` walks the layers with those recipes, hands each block its
cache, collects taps, and notes every route on the open window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import mlx.core as mx

from .. import row_exact_verify as REV
from . import ops

BlockRun = Callable[[Any, mx.array, Any], mx.array]


def per_row_block(layer, hidden: mx.array, cache) -> mx.array:
    return ops.per_row(lambda row: layer(row, mask=None, cache=cache), hidden)


def segment_block(layer, hidden: mx.array, cache) -> mx.array:
    return layer(hidden, mask=None, cache=cache)


@dataclass(frozen=True)
class Recipe:
    """How one block runs a strip; ``name`` is the receipt route."""

    name: str
    run: BlockRun


PER_ROW = Recipe("per_row", per_row_block)
SEGMENT = Recipe("segment", segment_block)


def residual(
    norm_key: tuple,
    mixer: Callable[[Any, mx.array], mx.array],
    *,
    name: str = "residual_exact",
) -> Recipe:
    """``x + mixer(layer, norm(x))``: a row-independent norm and an exact mixer."""

    def run(layer, hidden: mx.array, cache) -> mx.array:
        normalized = ops.row_independent(
            layer.norm, hidden, stage=f"{name}_norm", key=norm_key
        )
        return hidden + mixer(layer, normalized)

    return Recipe(name, run)


def run_layerwise(
    layers: Sequence[Any],
    recipe_for: Callable[[Any], Recipe],
    hidden: mx.array,
    caches: Sequence[Optional[Any]],
    capture_layers: Sequence[int] = (),
) -> tuple[mx.array, list]:
    """Run every layer on the strip with its recipe; return hidden and taps.

    ``caches[i]`` is layer ``i``'s cache (``None`` for cacheless blocks).
    """
    taps = []
    capture = set(capture_layers)
    window = REV.current()
    for index, (layer, cache) in enumerate(zip(layers, caches, strict=True)):
        recipe = recipe_for(layer)
        if window is not None:
            window.note("blocks", recipe.name)
        hidden = recipe.run(layer, hidden, cache)
        if index in capture:
            taps.append(hidden)
    return hidden, taps
