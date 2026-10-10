"""Model-agnostic orchestration for bounded recurrent-depth forwards.

The runtime owns pass counting and cache separation. Model adapters own the
hidden-state seam because embedding, trunk, and output-head contracts differ
between model families. This module is experimental and is not a serving
route or qualification claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RecurrentDepthConfig:
    """A bounded, weight-tied forward schedule."""

    passes: int = 2
    input_mode: str = "direct_hidden"

    def __post_init__(self) -> None:
        if isinstance(self.passes, bool) or not isinstance(self.passes, int):
            raise TypeError("recurrent-depth passes must be an integer")
        if not 1 <= self.passes <= 8:
            raise ValueError("recurrent-depth passes must be in 1..8")
        if self.input_mode != "direct_hidden":
            raise ValueError("unsupported recurrent-depth input mode")


@dataclass(frozen=True, slots=True)
class RecurrentDepthOutput:
    logits: Any
    hidden: Any
    receipt: dict[str, Any]


def recurrent_depth_forward(
    adapter: Any,
    inputs: Any,
    caches: Sequence[Any],
    *,
    config: RecurrentDepthConfig | None = None,
    first_pass_kwargs: Mapping[str, Any] | None = None,
) -> RecurrentDepthOutput:
    """Run one token slab through the same adapter trunk ``passes`` times.

    Pass zero consumes ordinary token embeddings and may receive request-scoped
    conditioning. Every later pass consumes the preceding pass's final hidden
    states directly. Each pass must have its own cache stack; sharing a stack
    would mix distinct recurrent-depth states and is rejected.
    """

    config = RecurrentDepthConfig() if config is None else config
    if len(caches) != config.passes:
        raise ValueError("recurrent-depth cache count must equal pass count")
    if len({id(cache) for cache in caches}) != len(caches):
        raise ValueError("recurrent-depth passes require distinct cache stacks")
    forward_hidden = getattr(adapter, "recurrent_depth_hidden", None)
    project_logits = getattr(adapter, "recurrent_depth_logits", None)
    if not callable(forward_hidden) or not callable(project_logits):
        raise TypeError("adapter does not expose recurrent-depth hidden-state seams")
    kwargs = dict(first_pass_kwargs or {})
    if set(kwargs) & {"cache", "input_embeddings"}:
        raise ValueError(
            "recurrent-depth conditioning may not override cache or input embeddings"
        )

    hidden = None
    for pass_index, cache in enumerate(caches):
        hidden = forward_hidden(
            inputs,
            cache=cache,
            input_embeddings=hidden,
            **(kwargs if pass_index == 0 else {}),
        )
    return RecurrentDepthOutput(
        logits=project_logits(hidden),
        hidden=hidden,
        receipt={
            "schema": "mlx2.recurrent-depth-forward.v1",
            "passes": config.passes,
            "input_mode": config.input_mode,
            "cache_stacks": len(caches),
            "conditioned_passes": 1 if kwargs else 0,
        },
    )
