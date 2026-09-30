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

import hashlib
from collections import Counter
from dataclasses import dataclass, field

import mlx.core as mx
from mlx import nn

from . import simd
from .matmul import (
    MAX_ROWS,
    UNQUANTIZED_BITS,
    LaneUnsupported,
    LaneWeights,
    available,
    backend,
    lane_matmul,
    prepare,
    split_k,
)

LAW_ID = "lane-matmul-v1"
# One numerical law per backend: the M5 tensor-unit kernels and the M1-M4
# simdgroup kernels compute different bits for the same row.
LAW_IDS = {"mpp": LAW_ID, "simd": "lane-simd-v1"}


# Lane state lives on each module (outside its parameter tree), never in
# process-wide maps keyed by id(module): CPython reuses a freed module's id at
# once, so a model dropped without uninstall() left its prepared weights and
# groups behind for the next model's modules, which then computed with the
# old model's stacked weights (and kept them alive).
def _prepared(module):
    return module.__dict__.get("_lane_prepared")


def _group(module):
    return module.__dict__.get("_lane_group")

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


@dataclass(frozen=True)
class ProjectionGroup:
    """Adapter-declared projections that read one input tensor object.

    ``parent`` is the module class whose ``__call__`` passes the identical
    tensor to every member (the adapter vouches for that call site, and a
    static test pins it); ``members`` are paths under it, a digit indexing a
    list (``"lmk_q_proj.0"``).  The group forms only on modules of exactly
    that class and only when every member is covered, ungrouped and of one
    weight format; otherwise none of it is stacked and the members fall
    through to ``DEFAULT_GROUPS``.
    """

    name: str
    parent: type
    members: tuple[str, ...]

    def __post_init__(self):
        if not self.name or len(self.members) < 2 or len(set(self.members)) != len(self.members):
            raise ValueError(f"projection group {self.name!r} needs 2+ distinct members")

    @property
    def spec(self) -> str:
        parent = f"{self.parent.__module__}.{self.parent.__qualname__}"
        return f"{self.name}={parent}:{','.join(self.members)}"


def _declared_digest(declared) -> str:
    """Law suffix for the declared groups that formed: stacking changes N,
    and with it split-K, for every member (docs: lane matmul law)."""
    specs = sorted({group.spec for group in declared})
    digest = hashlib.sha256("\n".join(specs).encode()).hexdigest()[:12]
    return f"+declared[{','.join(sorted({g.name for g in declared}))}@{digest}]"


@dataclass
class _Group:
    """Same-format siblings stacked along N; one launch computes all of them."""

    lw: LaneWeights
    last: tuple | None = None                     # (x, stacked output) of the latest call
    declared: str | None = None                   # adapter group name, None for defaults
    size: int = 0
    served: set = field(default_factory=set)      # members served by the latest launch
    stack: dict = field(default_factory=dict)     # the MLX arrays the members' weights view
    call: dict = field(default_factory=dict)      # the members' stock quantization arguments
    # Below the crossover the members' stock calls run as one stacked stock
    # launch when a probe proved it bitwise equal (see _probe_stock_stack).
    stock_stacked: bool = False
    stock_last: tuple | None = None               # (x, stacked stock output) of the latest call
    taken: int = 0                                # members served from ``last``
    stock_taken: int = 0                          # members served from ``stock_last``

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


def law_id(min_rows: int, backend_name: str | None = None) -> str:
    """Numerical-law identity for a given crossover (1 = exact) and backend."""
    base = LAW_IDS[backend_name or backend() or "mpp"]
    return base if min_rows == 1 else f"{base}+stock-below-{min_rows}"


class _LaneMixin:
    _lane_min_rows = 1
    _lane_max_rows = MAX_ROWS

    def __call__(self, x):
        lw = _prepared(self)
        rows = _rows(x)
        if lw is None or not ENABLED[0] or not available():
            STATS["stock_disabled"] += 1
        elif rows < self._lane_min_rows:
            STATS["stock_below_min_rows"] += 1
            STATS[f"rows_{_bucket(rows)}"] += 1
            group = _group(self)
            if group is not None and group.stock_stacked and GROUPING[0]:
                STATS["stock_stacked_calls"] += 1
                return _stock_stacked(self, group, x)
        elif rows > self._lane_max_rows:
            STATS["stock_above_max_rows"] += 1
        else:
            group = _group(self) if GROUPING[0] else None
            try:
                if group is None:
                    y = lane_matmul(x, lw)
                    STATS["lane_launches"] += 1
                else:
                    # The first sibling to see this input computes the whole
                    # group; the others take their columns of the same result.
                    start, stop = self.__dict__["_lane_columns"]
                    declared = group.declared
                    if group.last is None or group.last[0] is not x:
                        if declared is not None and group.last is not None \
                                and len(group.served) < group.size:
                            # The previous launch fed only part of the group:
                            # its members did not all see one tensor object.
                            STATS[f"declared_partial:{declared}"] += 1
                        group.last = (x, lane_matmul(x, group.lw))
                        group.taken = 0
                        STATS["lane_launches"] += 1
                        STATS["group_launches"] += 1
                        if declared is not None:
                            group.served = {start}
                            STATS[f"declared_launches:{declared}"] += 1
                    else:
                        STATS["group_reuses"] += 1
                        if declared is not None:
                            group.served.add(start)
                            STATS[f"declared_reuses:{declared}"] += 1
                    y = group.last[1][..., start:stop]
                    group.taken += 1
                    if group.taken >= group.size:
                        # Every member took its columns: drop the input and
                        # stacked output so no layer holds them between steps.
                        group.last = None
                    if lw.bias is not None:
                        y = y + lw.bias
                STATS["lane_calls"] += 1
                STATS["lane_rows"] += rows
                STATS[f"rows_{_bucket(rows)}"] += 1
                return y
            except LaneUnsupported:
                STATS["stock_unsupported"] += 1
        return self._lane_stock_call(x)


def _stock_matmul(group: _Group, x):
    stack, call = group.stack, group.call
    if "scales" not in stack:
        return x @ stack["weight"].T
    return mx.quantized_matmul(x, stack["weight"], stack["scales"], stack["biases"],
                               transpose=True, group_size=call["group_size"],
                               bits=call["bits"], mode=call["mode"])


def _stock_stacked(module, group: _Group, x):
    """This member's columns of one stock launch over the group's stack."""
    if group.stock_last is None or group.stock_last[0] is not x:
        group.stock_last = (x, _stock_matmul(group, x))
        group.stock_taken = 0
    start, stop = module.__dict__["_lane_columns"]
    y = group.stock_last[1][..., start:stop]
    group.stock_taken += 1
    if group.stock_taken >= group.size:
        group.stock_last = None
    bias = module.get("bias")
    return y if bias is None else y + bias


def _probe_stock_stack(members, group: _Group, rows_below: int, seen: dict) -> bool:
    """Is one stacked stock launch bitwise equal to the members' own stock calls?

    Checked on this GPU, in bf16 and fp16, for every row count that takes
    the stock path below the crossover, once per member-shape tuple (kernel
    choice depends only on shapes and dtypes).
    """
    stack = group.stack
    key = (tuple(tuple(m["weight"].shape) for m in members), tuple(sorted(group.call.items())),
           str(stack["weight"].dtype), str(stack["scales"].dtype if "scales" in stack else None),
           rows_below)
    if key in seen:
        return seen[key]
    same = True
    for dtype in (mx.bfloat16, mx.float16):
        for rows in range(1, rows_below + 1):
            x = (mx.random.normal((rows, group.lw.k), key=mx.random.key(rows)) * 0.5).astype(dtype)
            apart = mx.concatenate([m._lane_stock_call(x) for m in members], axis=-1)
            if not bool(mx.array_equal(apart, _stock_matmul(group, x)).item()):
                same = False
                break
        if not same:
            break
    seen[key] = same
    return same


def _enable_stock_stacks(model) -> dict:
    """Probe each group's stacked stock launch for the rows below its crossover."""
    if not available():
        return {}
    members: dict[int, list] = {}
    groups: dict[int, _Group] = {}
    for _name, module in model.named_modules():
        group = _group(module)
        if group is not None:
            groups[id(group)] = group
            members.setdefault(id(group), []).append(module)
    seen: dict = {}
    proven = unproven = 0
    for gid, group in groups.items():
        ordered = sorted(members[gid], key=lambda m: m.__dict__["_lane_columns"][0])
        rows_below = min(int(m._lane_min_rows) for m in ordered) - 1
        rows_below = min(rows_below, int(ordered[0]._lane_max_rows))
        if rows_below < 1:
            continue
        group.stock_stacked = _probe_stock_stack(ordered, group, rows_below, seen)
        proven += group.stock_stacked
        unproven += not group.stock_stacked
    return {"groups": proven, "unproven": unproven} if proven or unproven else {}


class LaneQuantizedLinear(_LaneMixin, nn.QuantizedLinear):
    def _lane_stock_call(self, x):
        return nn.QuantizedLinear.__call__(self, x)


class LaneLinear(_LaneMixin, nn.Linear):
    def _lane_stock_call(self, x):
        return nn.Linear.__call__(self, x)


_SWAP = {nn.QuantizedLinear: LaneQuantizedLinear, nn.Linear: LaneLinear}
_RESTORE = {new: old for old, new in _SWAP.items()}


def _format_key(lw: LaneWeights) -> tuple:
    return (lw.backend, lw.bits, lw.group_size, lw.k, lw.weight.dtype, lw.scales_dtype)


def _stack(members) -> _Group:
    """Stack same-format siblings; each module's arrays become views of the stack."""
    first = _prepared(members[0])
    quantized = first.bits != UNQUANTIZED_BITS
    names = ("weight", "scales", "biases") if quantized else ("weight",)
    stacked = {name: mx.concatenate([m[name] for m in members], axis=0) for name in names}
    mx.eval(*stacked.values())
    start = 0
    for m in members:
        stop = start + int(m["weight"].shape[0])
        for name in names:
            setattr(m, name, stacked[name][start:stop])     # zero-copy row views
        object.__setattr__(m, "_lane_columns", (start, stop))
        start = stop
    mx.eval([m[name] for m in members for name in names])
    n = start
    if first.backend == "simd":
        # The simd kernels read MLX's scale/bias layout: the stack is enough.
        lw = LaneWeights(first.bits, first.group_size, n, first.k, simd.splits(n, first.k),
                         stacked["weight"], None, None, "simd", stacked["scales"], stacked["biases"])
    else:
        pairs = (mx.contiguous(mx.stack([stacked["scales"].T, stacked["biases"].T], axis=-1))
                 if quantized else None)
        lw = LaneWeights(first.bits, first.group_size, n, first.k,
                         split_k(n, first.k, first.group_size, first.bits),
                         stacked["weight"], pairs, None)
    for m in members:
        object.__setattr__(m, "_lane_prepared", prepare(m, first.backend))  # individual views
    call = ({"bits": int(members[0].bits), "group_size": int(members[0].group_size),
             "mode": getattr(members[0], "mode", "affine")} if quantized else {})
    return _Group(lw, stack=stacked, call=call)


def _member(parent, path: str):
    module = parent
    for part in path.split("."):
        if part.isdigit():
            if not isinstance(module, (list, tuple)) or int(part) >= len(module):
                return None
            module = module[int(part)]
        else:
            module = getattr(module, part, None)
        if module is None:
            return None
    return module


def _declared_refusal(members) -> str | None:
    if any(m is None for m in members):
        return "member missing"
    if any(not isinstance(m, nn.Module) or _prepared(m) is None for m in members):
        return "member not covered"
    if any(_group(m) is not None for m in members):
        return "member already grouped"
    if len({id(m) for m in members}) != len(members):
        return "member repeated"
    if len({_format_key(_prepared(m)) for m in members}) != 1:
        return "mixed formats"
    return None


def _dissolve(model) -> None:
    """Drop every group; members keep their (view) weights and preparation."""
    for _name, module in model.named_modules():
        for name in ("_lane_group", "_lane_columns"):
            module.__dict__.pop(name, None)


def _group_declared(model, declared) -> tuple[Counter, dict, dict]:
    """Stack adapter-declared groups whole or not at all (before defaults)."""
    formed: Counter = Counter()
    by_name: dict[str, Counter] = {}
    refused: dict[str, Counter] = {}
    for _name, parent in model.named_modules():
        for spec in declared:
            if type(parent) is not spec.parent:
                continue
            members = [_member(parent, path) for path in spec.members]
            reason = _declared_refusal(members)
            if reason is not None:
                refused.setdefault(spec.name, Counter())[reason] += 1
                continue
            group = _stack(members)
            group.declared, group.size = spec.name, len(members)
            for m in members:
                object.__setattr__(m, "_lane_group", group)
            formed[f"{group.lw.format}x{len(members)}"] += 1
            by_name.setdefault(spec.name, Counter())[group.lw.format] += 1
    return (formed, {k: dict(v) for k, v in by_name.items()},
            {k: dict(v) for k, v in refused.items()})


def _group_siblings(model, groups) -> Counter:
    formed: Counter = Counter()
    for _name, parent in model.named_modules():
        for names in groups:
            present = [getattr(parent, n, None) for n in names]
            present = [m for m in present if m is not None and _prepared(m) is not None
                       and _group(m) is None]
            by_format: dict[tuple, list] = {}
            for m in present:
                by_format.setdefault(_format_key(_prepared(m)), []).append(m)
            for members in by_format.values():
                if len(members) < 2:
                    continue
                group = _stack(members)
                group.size = len(members)
                for m in members:
                    object.__setattr__(m, "_lane_group", group)
                formed[f"{group.lw.format}x{len(members)}"] += 1
    return formed


DEFAULT_MAX_ROWS = 32


def apc_lane_fingerprint(base, receipt):
    """APCv2 namespace for prefix state computed under an installed lane law.

    The lane arithmetic is a numerical law distinct from stock MLX, so KV and
    recurrent state produced under it (in memory, idle disk and a persist
    dir) must not be served to a route running stock kernels or another law.
    """
    if not receipt or not receipt.get("covered"):
        return base
    return (base, "lane-matmul", receipt["law_id"])


def install(model, *, min_rows: int = 4, max_rows: int = DEFAULT_MAX_ROWS, unquantized: bool = True,
            groups=DEFAULT_GROUPS, skip=lambda name, module: False,
            min_rows_by_format: dict | None = None, declared=()) -> dict:
    """Swap every supported projection to its lane class; returns a receipt.

    ``groups``: tuples of sibling attribute names that read the same input;
    same-format siblings are stacked and run as one launch (``()`` disables).
    ``declared``: adapter ``ProjectionGroup``s, formed before ``groups`` and
    only when grouping is on; any that form add their digest to the law.
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
    current = backend() or "mpp"
    if any(lw is not None and lw.backend != current
           for lw in (_prepared(m) for _n, m in model.named_modules())):
        # Installed under the other backend: its prepared weights and groups
        # would keep running that law under this receipt.  Start over.
        uninstall(model)
    covered: Counter = Counter()
    refused: Counter = Counter()
    for name, module in model.named_modules():
        kind = type(module)
        if kind in _RESTORE:
            rows = threshold(module)
            if rows is None:
                # Covered by an earlier install, not by this one: back to
                # stock, so the receipt describes what runs.
                _restore(module)
                refused["no threshold for this format"] += 1
                continue
            object.__setattr__(module, "_lane_min_rows", rows)
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
        object.__setattr__(module, "_lane_prepared", lw)
        module.__class__ = _SWAP[kind]
        object.__setattr__(module, "_lane_min_rows", int(rows))
        object.__setattr__(module, "_lane_max_rows", int(max_rows))
        covered[lw.format] += 1
    declared = tuple(declared or ())
    if any(not isinstance(spec, ProjectionGroup) for spec in declared):
        raise TypeError("declared projection groups must be ProjectionGroup instances")
    if len({spec.name for spec in declared}) != len(declared):
        raise ValueError("declared projection group names must be unique")
    if declared and not groups:
        raise ValueError("declared projection groups require grouping")
    declared_formed, declared_refused = {}, {}
    if declared or any(getattr(_group(m), "declared", None) is not None
                       for _n, m in model.named_modules()):
        # Declared groups change which siblings stack: re-form every group
        # from scratch so none from an earlier install outlives its spec.
        _dissolve(model)
    if groups:
        formed, declared_formed, declared_refused = _group_declared(model, declared)
        formed.update(_group_siblings(model, groups))
    else:
        # A repeat install without grouping dissolves earlier groups; they
        # kept running the stacked launch (a different split-K) while the
        # receipt said ungrouped.
        formed = Counter()
        _dissolve(model)
    twins = _check_simd_twins(model)
    stacked_stock = _enable_stock_stacks(model)
    base = LAW_IDS[current]
    if min_rows_by_format is None:
        law = law_id(min_rows)
    else:
        spec = ",".join(f"{k}:{v}" for k, v in sorted(min_rows_by_format.items()))
        law = f"{base}+stock-below[{spec}]"
    if max_rows != DEFAULT_MAX_ROWS:
        # Calls up to max_rows take the lane arithmetic: a wider window is a
        # different law (33-64-row verify or prefill tails change).
        law += f"+rows-le-{max_rows}"
    live = {group.declared for _name, module in model.named_modules()
            if (group := _group(module)) is not None and group.declared is not None}
    grouped = any(_group(module) is not None for _name, module in model.named_modules())
    if live:
        law += _declared_digest([spec for spec in declared if spec.name in live])
    if twins.get("affine"):
        # A 5/6/8-bit shape whose twins differ here runs affine_rows: other bits.
        shapes = ",".join(twins["affine"])
        law += f"+simd-affine[{hashlib.sha256(shapes.encode()).hexdigest()[:12]}]"
    receipt = {"law_id": law + ("+grouped" if grouped else ""),
               "min_rows": min_rows if min_rows_by_format is None else dict(min_rows_by_format),
               "max_rows": max_rows, "covered": dict(covered),
               "groups": dict(formed), "refused": dict(refused), "available": available(),
               "backend": backend()}
    if twins:
        # simd only: per-shape scalar/matrix twin checks on this GPU.
        receipt["simd_twins"] = twins
    if stacked_stock:
        # Groups whose sub-crossover stock calls run as one proven-equal launch.
        receipt["stock_stacked"] = stacked_stock
    if declared:
        # Only present when declared groups were passed, so receipts of
        # every other install are byte-identical to before.
        receipt["declared_groups"] = {
            spec.name: {"members": list(spec.members),
                        "parent": f"{spec.parent.__module__}.{spec.parent.__qualname__}",
                        "formed": declared_formed.get(spec.name, {}),
                        "refused": declared_refused.get(spec.name, {})}
            for spec in declared}
    return receipt


def _check_simd_twins(model) -> dict:
    """Run the simd twin check once per weight shape actually launched.

    A shape whose scalar twin differs from the matrix kernel on this GPU is
    routed so that every row count keeps one arithmetic (see ``simd.check``).
    Both a group's stack and its members are checked: ``set_enabled(grouping=
    False)`` launches members alone.
    """
    seen: dict = {}
    for _name, module in model.named_modules():
        group = _group(module)
        for lw in (_prepared(module), group.lw if group is not None else None):
            if lw is None or lw.backend != "simd":
                continue
            key = (lw.n, lw.k, lw.group_size, lw.bits, str(lw.scales_dtype))
            if key not in seen:
                simd.check(lw.weight, lw.scales, lw.biases, lw.group_size, lw.bits)
                # Read the route back: a shape rerouted by an earlier install
                # stays rerouted, and must be reported (and in the law) again.
                seen[key] = simd.rerouted(lw.n, lw.k, lw.group_size, lw.bits)
    if not seen:
        return {}
    by_kind = {kind: sorted(f"{n}x{k}q{bits}g{gs}" for (n, k, gs, bits, _d), how in seen.items()
                            if how == kind) for kind in ("affine", "mma")}
    return {"shapes": len(seen), "rerouted": sum(1 for how in seen.values() if how),
            **{kind: shapes for kind, shapes in by_kind.items() if shapes}}


def _restore(module) -> None:
    module.__class__ = _RESTORE[type(module)]
    for name in ("_lane_prepared", "_lane_group", "_lane_columns", "_lane_min_rows",
                 "_lane_max_rows"):
        module.__dict__.pop(name, None)


def uninstall(model) -> int:
    """Restore stock classes; returns how many projections were restored."""
    restored = 0
    for _name, module in model.named_modules():
        if type(module) in _RESTORE:
            _restore(module)
            restored += 1
    return restored


def set_enabled(on: bool, *, grouping: bool | None = None) -> None:
    """Process-wide on/off for paired A/B measurement (stock kernels when off)."""
    ENABLED[0] = bool(on)
    if grouping is not None:
        GROUPING[0] = bool(grouping)


def installed(module) -> bool:
    return type(module) in _RESTORE and _prepared(module) is not None


def apply_policy(model, policy: dict, declared=()) -> dict | None:
    """Install according to a resolved ``policy.resolve`` result (None when off).

    ``declared``: the adapter's ``ProjectionGroup``s; stacked only when the
    policy selects ``declared_groups`` (default off), and always reported.
    """
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
                "mode", "min_rows", "max_rows", "grouping", "declared_groups", "skip",
                "detected", "family", "sources")},
        }
    by_format = ({fmt: 1 for fmt in policy["min_rows"]} if policy["mode"] == "exact"
                 else dict(policy["min_rows"]))
    declared = tuple(declared or ())
    selected = bool(policy.get("declared_groups")) and bool(policy["grouping"])
    if selected and not declared:
        raise ValueError("lane policy selects declared_groups but the adapter declares none")
    receipt = install(
        model, min_rows=1, max_rows=int(policy["max_rows"]),
        groups=DEFAULT_GROUPS if policy["grouping"] else (),
        skip=lambda name, module: skipped(policy, name),
        min_rows_by_format=by_format,
        declared=declared if selected else (),
    )
    if selected and not any(entry["formed"] for entry in receipt["declared_groups"].values()):
        # Asked for and not delivered: restore stock rather than serve a
        # route whose policy says grouped while nothing was stacked.
        uninstall(model)
        raise ValueError("lane policy selects declared_groups but none formed: "
                         f"{ {k: v['refused'] for k, v in receipt['declared_groups'].items()} }")
    if declared and not selected:
        # Offered but not selected: report it without touching the law.
        receipt["declared_groups"] = {
            spec.name: {"members": list(spec.members),
                        "parent": f"{spec.parent.__module__}.{spec.parent.__qualname__}",
                        "formed": {}, "refused": {"not selected": 1}}
            for spec in declared}
    return {**receipt, "policy": {k: policy[k] for k in (
        "mode", "min_rows", "max_rows", "grouping", "declared_groups", "skip", "detected",
        "family", "sources")}}
