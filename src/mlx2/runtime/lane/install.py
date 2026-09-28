"""Route a model's linear projections through the lane matmul.

Installation changes the arithmetic of every covered projection for every
call of 1-``max_rows`` rows, one-token decode included.  That is what makes a
verify row equal to the same row decoded alone.  It is a different numerical
law from stock MLX, so a route that installs it must bind ``LAW_ID`` into its
cache identity and qualify the model under it.  Longer calls (chunked
prefill) keep stock kernels.  Unsupported projections keep stock kernels and
are reported.
"""

from __future__ import annotations

from collections import Counter

from mlx import nn

from .matmul import MAX_ROWS, LaneUnsupported, available, lane_matmul, prepare

LAW_ID = "lane-matmul-v1"
_PREPARED: dict[int, object] = {}
# Process-wide switch for paired A/B measurement; a route never flips it.
ENABLED = [True]


def _rows(x) -> int:
    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    return rows


class _LaneMixin:
    _lane_max_rows = MAX_ROWS

    def __call__(self, x):
        lw = _PREPARED.get(id(self))
        if lw is not None and ENABLED[0] and _rows(x) <= self._lane_max_rows and available():
            try:
                return lane_matmul(x, lw)
            except LaneUnsupported:
                pass
        return self._lane_stock_call(x)


class LaneQuantizedLinear(_LaneMixin, nn.QuantizedLinear):
    def _lane_stock_call(self, x):
        return nn.QuantizedLinear.__call__(self, x)


class LaneLinear(_LaneMixin, nn.Linear):
    def _lane_stock_call(self, x):
        return nn.Linear.__call__(self, x)


_SWAP = {nn.QuantizedLinear: LaneQuantizedLinear, nn.Linear: LaneLinear}
_RESTORE = {new: old for old, new in _SWAP.items()}


def install(model, *, max_rows: int = 32, unquantized: bool = True,
            skip=lambda name, module: False) -> dict:
    """Swap every supported projection to its lane class; returns a receipt.

    ``skip(name, module)`` lets an adapter keep a projection on stock kernels
    (for example a huge vocabulary head it measures separately).  Idempotent.
    """
    if not 1 <= max_rows <= MAX_ROWS:
        raise ValueError(f"max_rows must be 1-{MAX_ROWS}")
    covered: Counter = Counter()
    refused: Counter = Counter()
    for name, module in model.named_modules():
        kind = type(module)
        if kind in _RESTORE:
            covered["already"] += 1
            continue
        if kind not in _SWAP or (kind is nn.Linear and not unquantized):
            continue
        if skip(name, module):
            refused["skipped"] += 1
            continue
        try:
            lw = prepare(module)
        except LaneUnsupported as exc:
            refused[str(exc)] += 1
            continue
        _PREPARED[id(module)] = lw
        module.__class__ = _SWAP[kind]
        object.__setattr__(module, "_lane_max_rows", int(max_rows))
        covered[lw.format] += 1
    return {"law_id": LAW_ID, "max_rows": max_rows, "covered": dict(covered),
            "refused": dict(refused), "available": available()}


def uninstall(model) -> int:
    """Restore stock classes; returns how many projections were restored."""
    restored = 0
    for _name, module in model.named_modules():
        old = _RESTORE.get(type(module))
        if old is not None:
            module.__class__ = old
            _PREPARED.pop(id(module), None)
            restored += 1
    return restored


def set_enabled(on: bool) -> None:
    """Process-wide on/off for paired A/B measurement (stock kernels when off)."""
    ENABLED[0] = bool(on)


def installed(module) -> bool:
    return type(module) in _RESTORE and id(module) in _PREPARED
