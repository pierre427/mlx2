# SPDX-License-Identifier: Apache-2.0
"""Per-model parameters of the row-exact verify technology, and the one
mask/position law and per-row attention loop every row-exact route shares.

A row-exact verify route computes every verify row with the arithmetic of the
model's ordinary one-token decode at the same position.  What differs between
models is not the loop but a handful of parameters:

``mask_law``
    The mask a verify row's one-token attention forward receives.
    ``"ordinary_b1_explicit"``: the explicit all-valid mask the ordinary B=1
    ``BatchKVCache`` route hands a one-token forward
    (``create_causal_mask(1, offset, left_padding=[0])``), built from the row
    cache's offset (see ``array_offset`` / ``host_offset``); a row whose
    offset cannot be read on the host has no mask and the route fails
    closed.  ``"cache_make_mask"``: whatever the row cache's own
    ``make_mask`` gives (``models.base.create_attention_mask``), as the
    model's decode step does.
``array_offset``
    ``ordinary_b1_explicit`` only: how a cache whose ``offset`` is an
    ``mx.array`` (a B=1 ``BatchKVCache``) is read.  ``"physical_idx"``: a
    one-element offset is replaced by the cache's host ``_idx`` (Flash-Next);
    ``"refuse"``: no mask, the route fails closed (Qwen hybrids).
``host_offset``
    ``ordinary_b1_explicit`` only: which host offsets count.  ``"int"``:
    ``isinstance(offset, int)`` and the value is passed through unchanged
    (Flash-Next); ``"integral"``: any ``numbers.Integral``, passed as
    ``int(offset)`` (Qwen hybrids).
``max_positions``
    Context limit a layerwise route checks before writing any cache row
    (``max_position_embeddings``).
``route_note``
    When the shared layerwise executor notes a block's recipe on the open
    window: ``"before"`` the block runs (Nemotron-H), ``"after"`` (Maxwell),
    or ``"none"``.

Original mlx2 code.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral
from typing import Any, Callable, Optional

import mlx.core as mx

from ..technologies.dispatch import spec_from_config

MASK_LAWS = ("ordinary_b1_explicit", "cache_make_mask")
ARRAY_OFFSETS = ("physical_idx", "refuse")
HOST_OFFSETS = ("int", "integral")
ROUTE_NOTES = ("before", "after", "none")
# Spec field -> checkpoint config key(s).
CONFIG_FIELDS = {"max_positions": "max_position_embeddings"}


@dataclass(frozen=True)
class RowExactSpec:
    mask_law: str = "ordinary_b1_explicit"
    array_offset: str = "refuse"
    host_offset: str = "integral"
    max_positions: Optional[int] = None
    route_note: str = "before"

    def __post_init__(self):
        for name, allowed in (
            ("mask_law", MASK_LAWS),
            ("array_offset", ARRAY_OFFSETS),
            ("host_offset", HOST_OFFSETS),
            ("route_note", ROUTE_NOTES),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"RowExactSpec.{name} must be one of {allowed}")

    @classmethod
    def from_config(cls, config: Any, **overrides) -> "RowExactSpec":
        """Only the context limit is in a checkpoint config; the mask law is
        route evidence and comes from ``overrides``.  (Lane and row limits
        stay the routes' own guards: ``MAX_EVIDENCED_LANES``,
        ``MAX_PHYSICAL_LANES`` / ``MAX_TOKEN_ROWS``.)"""
        return spec_from_config(cls, config, CONFIG_FIELDS, **overrides)

    def with_config(self, config: Any) -> "RowExactSpec":
        """This spec with its config-derived fields read from ``config``."""
        keep = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in CONFIG_FIELDS
        }
        return type(self).from_config(config, **keep)

    @property
    def refuses_missing_mask(self) -> bool:
        """Whether a row without a mask means the route cannot run it (the
        ``refuse_missing_mask`` the hybrid serial attention passes)."""
        return self.mask_law == "ordinary_b1_explicit"


def ordinary_b1_mask(cache, spec: RowExactSpec):
    """The ordinary B=1 route's explicit all-valid one-token mask, or ``None``
    when ``spec`` cannot read the cache's offset on the host."""
    from ..models.base import create_causal_mask

    offset = getattr(cache, "offset", None)
    if isinstance(offset, mx.array):
        if spec.array_offset == "refuse" or offset.size != 1:
            return None
        offset = getattr(cache, "_idx", None)
    if spec.host_offset == "int":
        if not isinstance(offset, int):
            return None
    else:
        if not isinstance(offset, Integral):
            return None
        offset = int(offset)
    return create_causal_mask(1, offset=offset, left_padding=mx.array([0]))


def row_mask(h, cache, spec: RowExactSpec):
    """The mask one verify row ``h`` (``[B, 1, H]``) attends with."""
    if spec.mask_law == "cache_make_mask":
        from ..models.base import create_attention_mask

        return create_attention_mask(h, cache)
    return ordinary_b1_mask(cache, spec)


def serial_attention_rows(
    attend: Callable,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cache,
    mask: Callable[[slice], Any],
    *,
    start: int,
    length: int,
    lane: Optional[int] = None,
    refuse_missing_mask: bool = False,
):
    """One one-token ``attend(q, k, v, 1, 1, mask, cache)`` per row.

    Rows ``start .. start + length`` of the projected ``q``/``k``/``v``
    (of lane ``lane`` when given) append to ``cache`` one at a time, each
    with ``mask(row)``.  Returns the per-row outputs, or ``None`` at the
    first missing mask when ``refuse_missing_mask`` (rows before it have
    already been written, as in the loops this replaces).
    """
    outputs = []
    for position in range(length):
        row = slice(start + position, start + position + 1)
        row_mask_value = mask(row)
        if row_mask_value is None and refuse_missing_mask:
            return None
        if lane is None:
            qr, kr, vr = q[:, row], k[:, row], v[:, row]
        else:
            lanes = slice(lane, lane + 1)
            qr, kr, vr = q[lanes, row], k[lanes, row], v[lanes, row]
        outputs.append(attend(qr, kr, vr, 1, 1, row_mask_value, cache))
    return outputs


def merge_window_routes(counts: dict, record) -> None:
    """Add a closed window's stage routes and failures to a route's counters
    (``counts["stages"]`` / ``counts["failures"]``)."""
    for stage, routes in record.stages.items():
        target = counts["stages"].setdefault(stage, {})
        for route, n in routes.items():
            target[route] = target.get(route, 0) + n
    for reason, n in record.failures.items():
        counts["failures"][reason] = counts["failures"].get(reason, 0) + n


__all__ = [
    "ARRAY_OFFSETS",
    "HOST_OFFSETS",
    "MASK_LAWS",
    "ROUTE_NOTES",
    "RowExactSpec",
    "merge_window_routes",
    "ordinary_b1_mask",
    "row_mask",
    "serial_attention_rows",
]
