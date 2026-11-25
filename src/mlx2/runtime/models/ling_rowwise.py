"""Ling per-row verify laws: diagnostic attribution opt-ins (default off).

A speculative verify forward scores 2-16 rows at once; ordinary decode scores one
row per forward.  Matmul, attention and scan kernels may reduce in a different
order at L > 1 than at L == 1, so a verify row's logits can drift from the decode
step's by a few bf16 steps and flip near ties (Ling-3.2-air attribution
2026-11-25, qualification/runs/ling32-air-20261123/attr-d8702e440).

Each component law replays the rows of a verify forward one at a time through
the same one-row arithmetic ordinary decode uses, inside the one L-row forward
(so the speculative cache transaction is unchanged):

``dense`` every target ``nn.Linear`` / ``nn.QuantizedLinear`` / ``nn.RMSNorm``, the
          final norm and the head (row-local reductions keep the one-row shape);
``moe``   the whole routed + shared expert block (``SparseMoeBlock``);
``kda``   the KDA short convolutions and gated-delta scan (forward and rollback);
``mla``   MLA RoPE and latent attention (each row over its own causal prefix).

All four together (``verify_rowwise``) is the exact control: the verify rows then
equal the L == 1 decode sequence bit for bit.  These laws are slow by design and
only engage inside a forward whose caches are speculating and whose length is
2-16; prompt chunks and one-row decode are untouched.
"""

from __future__ import annotations

import contextvars
from collections import Counter
from contextlib import contextmanager

import mlx.core as mx
from mlx import nn

COMPONENTS = ("dense", "moe", "kda", "mla")
MAX_ROWS = 16
STATS = Counter()
_ACTIVE = contextvars.ContextVar("ling_rowwise_active", default=False)


def active() -> bool:
    return _ACTIVE.get()


def _speculating(cache) -> bool:
    if cache is None:
        return False
    return any(bool(getattr(item, "speculating", False)) for item in cache if item is not None)


@contextmanager
def scope(inputs, cache):
    """Mark a target forward as a verify forward (2-16 rows, speculating caches)."""
    if _ACTIVE.get():
        yield True
        return
    length = inputs.shape[1] if getattr(inputs, "ndim", 0) >= 2 else 0
    on = 1 < length <= MAX_ROWS and _speculating(cache)
    token = _ACTIVE.set(on)
    try:
        if on:
            STATS["verify_forwards"] += 1
        yield on
    finally:
        _ACTIVE.reset(token)


def rows(fn, x, axis=-2):
    """``fn`` applied to each length-one slice of ``x`` along ``axis``, concatenated."""
    axis = axis % x.ndim
    pieces = []
    for index in range(x.shape[axis]):
        key = [slice(None)] * x.ndim
        key[axis] = slice(index, index + 1)
        pieces.append(fn(x[tuple(key)]))
    return mx.concatenate(pieces, axis=axis)


_LINEAR_CLASSES = {}


def _rowwise_class(cls):
    sub = _LINEAR_CLASSES.get(cls)
    if sub is None:
        base = cls.__call__

        def __call__(self, x):
            # Rows are the sequence axis: 1 for [B, L, ...], else -2.
            axis = 1 if x.ndim >= 3 else -2
            if _ACTIVE.get() and x.ndim >= 2 and x.shape[axis] > 1:
                STATS["dense_rows"] += x.shape[axis]
                return rows(lambda row: base(self, row), x, axis=axis)
            return base(self, x)

        sub = type("Rowwise" + cls.__name__, (cls,), {"__call__": __call__, "ling_rowwise": True})
        _LINEAR_CLASSES[cls] = sub
    return sub


def bind_dense(modules) -> int:
    """Swap each linear / RMSNorm in ``modules`` to its per-row class (parameters unchanged)."""
    count = 0
    for module in modules:
        if getattr(module, "ling_rowwise", False):
            continue
        if isinstance(module, (nn.Linear, nn.QuantizedLinear, nn.RMSNorm)):
            module.__class__ = _rowwise_class(type(module))
            count += 1
    return count


def status() -> dict:
    return dict(STATS)
