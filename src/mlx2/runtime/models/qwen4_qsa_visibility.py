"""Opt-in plain-causal QSA geometry inspired by llama.cpp #30965.

Original implementation; provenance/llamacpp-30965-qsa-visibility.json.
Arbitrary mask arrays retain the ordinary reference path.
"""

from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from threading import Lock

import mlx.core as mx

_COUNTS = Counter()
_LOCK = Lock()


def record(reason):
    with _LOCK:
        _COUNTS[reason] += 1


def status(*, reset=False):
    with _LOCK:
        result = dict(_COUNTS)
        if reset:
            _COUNTS.clear()
    return {"qualification": "unqualified", "counts": result}


@dataclass(frozen=True)
class QSAVisibility:
    """Producer-proven causal diagonal with optional per-row left padding.

    This proof is never inferred from an arbitrary array's shape/dtype or
    from mask absence. Compact consumers carry the same coordinates; dense
    consumers share one lazily materialized reference mask.
    """

    length: int
    physical_width: int
    batch: int
    left_padding: mx.array | None = None
    register: bool = True

    def __post_init__(self):
        if self.length < 1 or self.physical_width < self.length or self.batch < 1:
            raise ValueError("invalid QSA visibility geometry")
        if self.left_padding is not None and self.left_padding.shape != (self.batch,):
            raise ValueError("QSA visibility needs one padding value per row")

    @property
    def shape(self):
        return (
            self.batch if self.left_padding is not None else 1,
            1,
            self.length,
            self.physical_width,
        )

    @cached_property
    def _dense(self):
        from .base import create_causal_mask
        from .qsdpa_verify_metal import lift_causal_mask, register_causal_mask

        record("dense_materializations")
        mask = create_causal_mask(
            self.length,
            offset=self.physical_width - self.length,
            left_padding=self.left_padding,
        )
        if self.register:
            register_causal_mask(mask, self.left_padding)
        return lift_causal_mask(mask)

    def materialize(self):
        return self._dense
