"""Route a model's linear projections through the lane matmul.

Covered projections use the lane matmul for calls of ``min_rows`` to
``max_rows`` rows and stock MLX otherwise (chunked prefill above, and by
default one-token decode and short verifies below).

* ``min_rows=4`` (default, "crossover"): at 1-3 rows stock MLX is cheaper and
  is kept, so one-token decode is unchanged; from 4 rows the lane matmul is
  cheaper and flat.  Verify rows are then not bitwise equal to one-token
  decode, the same contract as stock multi-row verification.
* ``min_rows=1`` ("exact"): every call of 1-``max_rows`` rows uses the lane
  arithmetic, so a verify row equals the same row decoded alone.

Either way the installed arithmetic is a numerical law distinct from stock
MLX: a route that installs it binds ``law_id(min_rows)`` into its cache
identity and qualifies the model under it.  Unsupported projections keep
stock kernels and are reported.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import mlx.core as mx
from mlx import nn

from .matmul import (
    MAX_ROWS,
    UNQUANTIZED_BITS,
    LaneUnsupported,
    LaneWeights,
    available,
    lane_matmul,
    prepare,
    split_k,
)

LAW_ID = "lane-matmul-v1"
_PREPARED: dict[int, object] = {}

# Sibling projections that read the same input, by attribute name under one
# parent module.  Common across model families; an adapter may pass its own.
DEFAULT_GROUPS = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
    ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
    ("wq", "wk", "wv"),
    ("w1", "w3"),
    ("query", "key", "value"),
)


@dataclass
class _Group:
    """Same-format siblings stacked along N; one launch computes all of them."""

    lw: LaneWeights
    columns: dict = field(default_factory=dict)   # id(module) -> (start, stop)
    last: tuple | None = None                     # (x, stacked output) of the latest call


_GROUP_OF: dict[int, _Group] = {}
# Process-wide switches for paired A/B measurement; a route never flips them.
ENABLED = [True]
GROUPING = [True]

# Host-side call counters for covered projections (status and /metrics).
# Row buckets keep label cardinality bounded.
ROW_BUCKETS = ((1, 3), (4, 7), (8, 15), (16, 32), (33, MAX_ROWS))
STATS: Counter = Counter()


def _bucket(rows: int) -> str:
    for low, high in ROW_BUCKETS:
        if low <= rows <= high:
            return f"{low}-{high}"
    return f">{MAX_ROWS}"


def stats() -> dict:
    """Copy of the call counters (see ``STATS`` keys)."""
    return dict(STATS)


def _rows(x) -> int:
    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    return rows


def law_id(min_rows: int) -> str:
    """Numerical-law identity for a given crossover (1 = exact)."""
    return LAW_ID if min_rows == 1 else f"{LAW_ID}+stock-below-{min_rows}"


class _LaneMixin:
    _lane_min_rows = 1
    _lane_max_rows = MAX_ROWS

    def __call__(self, x):
        lw = _PREPARED.get(id(self))
        rows = _rows(x)
        if lw is None or not ENABLED[0] or not available():
            STATS["stock_disabled"] += 1
        elif rows < self._lane_min_rows:
            STATS["stock_below_min_rows"] += 1
            STATS[f"rows_{_bucket(rows)}"] += 1
        elif rows > self._lane_max_rows:
            STATS["stock_above_max_rows"] += 1
        else:
            group = _GROUP_OF.get(id(self)) if GROUPING[0] else None
            try:
                if group is None:
                    y = lane_matmul(x, lw)
                    STATS["lane_launches"] += 1
                else:
                    # The first sibling to see this input computes the whole
                    # group; the others take their columns of the same result.
                    if group.last is None or group.last[0] is not x:
                        group.last = (x, lane_matmul(x, group.lw))
                        STATS["lane_launches"] += 1
                        STATS["group_launches"] += 1
                    else:
                        STATS["group_reuses"] += 1
                    start, stop = group.columns[id(self)]
                    y = group.last[1][..., start:stop]
                    if lw.bias is not None:
                        y = y + lw.bias
                STATS["lane_calls"] += 1
                STATS["lane_rows"] += rows
                STATS[f"rows_{_bucket(rows)}"] += 1
                return y
            except LaneUnsupported:
                STATS["stock_unsupported"] += 1
        return self._lane_stock_call(x)


class LaneQuantizedLinear(_LaneMixin, nn.QuantizedLinear):
    def _lane_stock_call(self, x):
        return nn.QuantizedLinear.__call__(self, x)


class LaneLinear(_LaneMixin, nn.Linear):
    def _lane_stock_call(self, x):
        return nn.Linear.__call__(self, x)


_SWAP = {nn.QuantizedLinear: LaneQuantizedLinear, nn.Linear: LaneLinear}
_RESTORE = {new: old for old, new in _SWAP.items()}


def _format_key(lw: LaneWeights) -> tuple:
    return (lw.bits, lw.group_size, lw.k, lw.weight.dtype,
            None if lw.scale_bias is None else lw.scale_bias.dtype)


def _stack(members) -> _Group:
    """Stack same-format siblings; each module's arrays become views of the stack."""
    first = _PREPARED[id(members[0])]
    quantized = first.bits != UNQUANTIZED_BITS
    names = ("weight", "scales", "biases") if quantized else ("weight",)
    stacked = {name: mx.concatenate([m[name] for m in members], axis=0) for name in names}
    mx.eval(*stacked.values())
    columns, start = {}, 0
    for m in members:
        stop = start + int(m["weight"].shape[0])
        for name in names:
            setattr(m, name, stacked[name][start:stop])     # zero-copy row views
        columns[id(m)] = (start, stop)
        start = stop
    mx.eval([m[name] for m in members for name in names])
    n = start
    pairs = (mx.contiguous(mx.stack([stacked["scales"].T, stacked["biases"].T], axis=-1))
             if quantized else None)
    lw = LaneWeights(first.bits, first.group_size, n, first.k,
                     split_k(n, first.k, first.group_size, first.bits),
                     stacked["weight"], pairs, None)
    for m in members:
        _PREPARED[id(m)] = prepare(m)       # individual views, for unmatched inputs
    return _Group(lw, columns)


def _group_siblings(model, groups) -> Counter:
    formed: Counter = Counter()
    for _name, parent in model.named_modules():
        for names in groups:
            present = [getattr(parent, n, None) for n in names]
            present = [m for m in present if m is not None and id(m) in _PREPARED
                       and id(m) not in _GROUP_OF]
            by_format: dict[tuple, list] = {}
            for m in present:
                by_format.setdefault(_format_key(_PREPARED[id(m)]), []).append(m)
            for members in by_format.values():
                if len(members) < 2:
                    continue
                group = _stack(members)
                for m in members:
                    _GROUP_OF[id(m)] = group
                formed[f"{group.lw.format}x{len(members)}"] += 1
    return formed


def install(model, *, min_rows: int = 4, max_rows: int = 32, unquantized: bool = True,
            groups=DEFAULT_GROUPS, skip=lambda name, module: False,
            min_rows_by_format: dict | None = None) -> dict:
    """Swap every supported projection to its lane class; returns a receipt.

    ``groups``: tuples of sibling attribute names that read the same input;
    same-format siblings are stacked and run as one launch (``()`` disables).
    ``skip(name, module)`` lets an adapter keep a projection on stock kernels
    (for example a huge vocabulary head it measures separately).  Idempotent
    for the class swap; a repeat call updates the row window.
    """
    if not 1 <= min_rows <= max_rows <= MAX_ROWS:
        raise ValueError(f"need 1 <= min_rows <= max_rows <= {MAX_ROWS}")
    from .policy import format_class

    def threshold(module):
        """Per-projection min rows; None keeps the projection on stock."""
        if min_rows_by_format is None:
            return min_rows
        value = min_rows_by_format.get(format_class(module))
        return None if value is None or value > max_rows else int(value)
    covered: Counter = Counter()
    refused: Counter = Counter()
    for name, module in model.named_modules():
        kind = type(module)
        if kind in _RESTORE:
            rows = threshold(module)
            object.__setattr__(module, "_lane_min_rows", rows if rows is not None else max_rows + 1)
            object.__setattr__(module, "_lane_max_rows", int(max_rows))
            covered["already"] += 1
            continue
        if kind not in _SWAP or (kind is nn.Linear and not unquantized):
            continue
        if skip(name, module):
            refused["skipped"] += 1
            continue
        rows = threshold(module)
        if rows is None:
            refused["no threshold for this format"] += 1
            continue
        try:
            lw = prepare(module)
        except LaneUnsupported as exc:
            refused[str(exc)] += 1
            continue
        _PREPARED[id(module)] = lw
        module.__class__ = _SWAP[kind]
        object.__setattr__(module, "_lane_min_rows", int(rows))
        object.__setattr__(module, "_lane_max_rows", int(max_rows))
        covered[lw.format] += 1
    formed = _group_siblings(model, groups) if groups else Counter()
    if min_rows_by_format is None:
        law = law_id(min_rows)
    else:
        spec = ",".join(f"{k}:{v}" for k, v in sorted(min_rows_by_format.items()))
        law = f"{LAW_ID}+stock-below[{spec}]"
    return {"law_id": law + ("+grouped" if groups else ""),
            "min_rows": min_rows if min_rows_by_format is None else dict(min_rows_by_format),
            "max_rows": max_rows, "covered": dict(covered),
            "groups": dict(formed), "refused": dict(refused), "available": available()}


def uninstall(model) -> int:
    """Restore stock classes; returns how many projections were restored."""
    restored = 0
    for _name, module in model.named_modules():
        old = _RESTORE.get(type(module))
        if old is not None:
            module.__class__ = old
            _PREPARED.pop(id(module), None)
            _GROUP_OF.pop(id(module), None)
            restored += 1
    return restored


def set_enabled(on: bool, *, grouping: bool | None = None) -> None:
    """Process-wide on/off for paired A/B measurement (stock kernels when off)."""
    ENABLED[0] = bool(on)
    if grouping is not None:
        GROUPING[0] = bool(grouping)


def installed(module) -> bool:
    return type(module) in _RESTORE and id(module) in _PREPARED


def apply_policy(model, policy: dict) -> dict | None:
    """Install according to a resolved ``policy.resolve`` result (None when off)."""
    from .policy import skipped

    if policy["mode"] == "off":
        return None
    if not available():
        # Do not rewrite weights into grouped views on a device that can
        # never execute the lane kernel.  Stock fallback then retains its
        # original modules and allocations.
        return {
            "law_id": "stock", "covered": {}, "groups": {},
            "refused": {"device_unsupported": sum(policy["detected"]["formats"].values())},
            "available": False,
            "policy": {k: policy[k] for k in (
                "mode", "min_rows", "max_rows", "grouping", "skip",
                "detected", "family", "sources")},
        }
    by_format = ({fmt: 1 for fmt in policy["min_rows"]} if policy["mode"] == "exact"
                 else dict(policy["min_rows"]))
    receipt = install(
        model, min_rows=1, max_rows=int(policy["max_rows"]),
        groups=DEFAULT_GROUPS if policy["grouping"] else (),
        skip=lambda name, module: skipped(policy, name),
        min_rows_by_format=by_format,
    )
    return {**receipt, "policy": {k: policy[k] for k in (
        "mode", "min_rows", "max_rows", "grouping", "skip", "detected", "family", "sources")}}
