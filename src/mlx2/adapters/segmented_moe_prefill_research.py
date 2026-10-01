# SPDX-License-Identifier: Apache-2.0
"""Segmented sorted MoE prefill: isolated research candidate.

RESEARCH ONLY, default-off, UNQUALIFIED, UNSELECTED, never observed-used.
Nothing in the scheduler, cache, serving or model code imports this module,
no served route can select it, and the ordinary reference (``SwitchGLU`` /
``FusedGateUpSwitchGLU`` with ``_gather_sort`` and ``mx.gather_qmm``) is
untouched.

Mechanism (oMLX ``omlx/patches/m5_gather_qmm_nax.py`` @ d6b2b92, Apache-2.0;
see provenance/segmented-moe-prefill-research.json): a one-threadgroup
pre-pass cuts each expert's run of sorted rows into single-expert tiles of at
most BM rows, and a NAX tensor-unit matmul computes one BM x 64 output tile
per threadgroup. Two kernels are retained:

* row-mapped fused gate/up + SwiGLU: reads the token rows ``[T, 1, D]``
  through the sorted row -> token map (``order // k``) instead of the
  replicated ``x[order // k]`` copy, takes the fused ``[gate; up]`` table
  ``[E, 2I, D]`` as the model holds it (weight rows paired in the tile
  loader), rounds each fp32 projection to the activation dtype and applies
  ``Multiply()(Multiply()(g, Sigmoid()(g)), u)`` with MLX's own elementwise
  functors (the op sequence of the compiled ``nn.silu(gate) * up``);
  output ``[M, 1, I]`` in sorted row order;
* segmented sorted down projection on the ``[M, 1, I]`` sorted rows,
  output ``[M, 1, D]`` in sorted row order. The ordinary
  ``down_proj(hidden, idx, sorted_indices=True)`` remains usable on the
  gate/up output instead.

Retained arithmetic: affine dequantization ``scale * q + bias`` in fp32
rounded once to bf16, 16x32x16 tensor ops in mlx's K order, fp32
accumulation, 32-bit row offsets, row-map element offsets ``map * K``.
Upstream reports bit identity with mlx's sorted kernel and with the unfused
path; NOTHING here establishes that on this tree. Host tests do not
establish Metal numerical identity.

Contract (refused, never approximated): affine 4-bit group 64, bfloat16,
fused ``[gate; up]`` table, no training, no expert bias, no folded shared
expert, hidden and intermediate multiples of 64, at most 2048 experts, every
expert's sorted rows in ONE contiguous run, row map inside the token rows,
32-bit offsets, and the MLX row-block admission ``rows >= 16`` and
``rows // experts >= 4`` (default 35B E256 k8: T >= 128; Flash-Next E512
k10: T >= 205). That guard overrides oMLX's ``M >= 8``: smaller calls such
as MTP verify and decode stay on ``gather_qmv`` (whose summation order they
depend on). The planner is a port of oMLX's, measured upstream on M5 Ultra
and NOT measured here.

Import is host-only: this module never imports MLX at module scope. The
explicit research entry points need ``research_opt_in=True`` (an
acknowledgement that the call may build and dispatch Metal kernels; it is
NOT GPU ownership, which the caller must hold separately). Every refusal
happens before MLX is imported or a kernel is constructed; genuine backend
exceptions propagate unchanged. There are no runtime canaries, timers,
synchronisations or readbacks; engagement statistics are bounded host
counters of SUCCESSFUL lazy dispatch chains (scan + matmul both returned),
not of every issued scan and not of GPU execution.

Signed 32-bit bounds: besides the int32 uniforms, shapes and grid, the
retained kernel text forms signed int intermediates (``K * kBits``, the db
row offset ``r * K + col``, the seg row offsets ``tm * K`` / ``tm * N``,
NAX fragment offsets, the scan loop ``g0 += 4 * 1024``). Admission bounds
D, I <= INT32_MAX // 32 and padded rows <= INT32_MAX - 4096, and every
chosen plan (pinned plans included) is checked against
``signed_int_intermediates``. These are host bounds on generic shapes, not
a native qualification of general shapes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Tuple

SCHEMA = "mlx2.segmented-moe-prefill-research.v1"
# Immutable: nothing in this module may flip these at runtime.
STATE = MappingProxyType({
    "default": "off",
    "implemented": "research-candidate",
    "qualified": False,
    "selected": False,
    "observed_used": False,
    "route": None,
    "native_qualification": "external, pending",
})
SOURCE = MappingProxyType({
    "repository": "https://github.com/jundot/omlx",
    "path": "omlx/patches/m5_gather_qmm_nax.py",
    "revision": "d6b2b92b11ebcbf4a4c1f002b0d590a3117212a8",
    "blob": "48236b4d15155e51e9d235d45c86eb218d19b203",
    "sha256": "9b16a7b325760f758b40151c21d0a603d59308777252e04698023710d321fb66",
    "license": "Apache-2.0",
})
ORDINARY_REFERENCE = "ordinary-reference"

# Kernel geometry fixed by the retained Metal source.
BN = 64
WN = 2
TILE_ROWS = (64, 96, 128)
K_STEPS = (64, 128)
MAX_EXPERTS = 2048
GX = 32
MAX_PAD_BYTES = 8192
MAX_THREADGROUP_MEMORY = 32768

# Contract: the two named targets are affine q4 g64 bf16.
DTYPE = "bfloat16"
MODE = "affine"
BITS = 4
GROUP_SIZE = 64
LAYOUT = "fused_gate_up"
DIM_ALIGN = 64

# MLX's row-block criterion (B >= 16 and B / E >= 4); below it MLX runs
# gather_qmv, whose summation order verify/decode depend on.
MIN_ROWS = 16
MIN_ROWS_PER_EXPERT = 4

INT32_MAX = 2**31 - 1
UINT32_LIMIT = 2**32
# Conservative dim floor (root review): keeps K * kBits and 31 * K + col in int32.
MAX_DIM = INT32_MAX // 32
# Scan pre-pass: 1024 threads walk 4 rows per step, so g0 reaches rows - 1 + 4096.
SCAN_STRIDE = 4 * 1024
MAX_ROWS = INT32_MAX - SCAN_STRIDE
# The local seam pads > 32768 sorted rows to a multiple of 64 for MLX's
# int16 row-offset defect; the candidate uses 32-bit offsets and needs none.
SEAM_PAD_THRESHOLD = 32768
SEAM_PAD_MULTIPLE = 64

NAMED_GEOMETRIES = MappingProxyType({
    "default35b": MappingProxyType({"experts": 256, "top_k": 8, "hidden": 2048, "intermediate": 512}),
    "flash_next": MappingProxyType({"experts": 512, "top_k": 10, "hidden": 2560, "intermediate": 640}),
})

REFUSAL_REASONS = (
    "opt_in_missing", "unsealed", "type", "shape", "dtype", "quantization",
    "layout", "training", "bias", "shared_fold", "too_small", "overflow",
    "indices", "row_map", "routes", "plan",
)


class SegmentedMoERefused(ValueError):
    """Outside the candidate's contract; the ordinary reference applies."""

    def __init__(self, reason: str, message: str):
        if reason not in REFUSAL_REASONS:
            raise ValueError(f"unknown refusal reason {reason!r}")
        super().__init__(f"{reason}: {message}")
        self.reason = reason


class SegmentedMoEUnavailable(RuntimeError):
    """The native research backend is not available on this host."""


def _refuse(reason: str, message: str):
    raise SegmentedMoERefused(reason, message)


def _is_int(value) -> bool:
    return type(value) is int


# -------------------------------------------------------------------- planner

SCHED_SEG = 0
SCHED_DB = 1
_SCHED_NAMES = {SCHED_SEG: "seg", SCHED_DB: "db"}

PLAN_PROVENANCE = (
    "Port of oMLX _plan at d6b2b92 (thresholds measured upstream on M5 Ultra at "
    "Qwen3.8 / GLM-5.3 / MiMo-V2.6 expert shapes). NOT measured or tuned in mlx2; "
    "no environment override."
)


class Plan(NamedTuple):
    """Schedule, tile rows, K step, tile-on-x group (0: mlx layout), extra tg bytes."""

    sched: int
    bm: int
    bk: int
    gx: int
    pad: int

    def describe(self) -> str:
        s = f"{_SCHED_NAMES[self.sched]} {self.bm}x{BN} bk{self.bk}"
        if self.gx:
            s += f" gx{self.gx}"
        if self.pad:
            s += f" pad{self.pad}"
        return s


def plan_kernel(rows: int, experts: int, K: int, N: int) -> Plan:
    """oMLX's measured planner (mean rows per expert and K); see PLAN_PROVENANCE."""
    if K % 64 or N % 64:
        return Plan(SCHED_SEG, 64, 64, 0, 0)
    per_expert = rows / max(1, experts)
    if per_expert < 36 or (K < 1024 and per_expert < 120):
        return Plan(SCHED_DB, 64, 64, 0, 0)
    if K < 1024:
        return Plan(SCHED_SEG, 96, 128, GX, 0)
    if per_expert < 48:
        return Plan(SCHED_DB, 64, 64, GX, 0)
    if per_expert < 96:
        return Plan(SCHED_DB, 96, 64, GX, 0)
    return Plan(SCHED_SEG, 128, 128, GX, MAX_PAD_BYTES)


def effective_plan(plan: Plan, K: int, N: int, *, paired: bool) -> Plan:
    """db runs aligned 64-deep K steps only (and full N tiles when unpaired)."""
    if plan.sched == SCHED_DB and (K % 64 or plan.bk != 64 or (not paired and N % 64)):
        return plan._replace(sched=SCHED_SEG)
    return plan


def threadgroup_memory_bytes(plan: Plan) -> int:
    """Ws: (db ? 2 : 1) * BN * (BK + 8) bf16 values + PAD bytes."""
    return (2 if plan.sched == SCHED_DB else 1) * BN * (plan.bk + 8) * 2 + plan.pad


def validate_plan(plan) -> Plan:
    """An explicitly pinned plan (research only; never read from the environment)."""
    if not isinstance(plan, Plan):
        _refuse("plan", "a pinned plan must be a Plan")
    if not all(_is_int(v) for v in plan):
        _refuse("plan", "plan fields must be ints")
    if (plan.sched not in _SCHED_NAMES or plan.bm not in TILE_ROWS or plan.bk not in K_STEPS
            or plan.gx not in (0, GX) or not 0 <= plan.pad <= MAX_PAD_BYTES):
        _refuse("plan", f"unsupported plan {tuple(plan)}")
    if threadgroup_memory_bytes(plan) > MAX_THREADGROUP_MEMORY:
        _refuse("plan", "threadgroup memory bound exceeded")
    return plan


def max_tiles(rows: int, experts: int, bm: int) -> int:
    """Upper bound on single-expert tiles: one partial tile per present expert."""
    return (rows + bm - 1) // bm + min(experts, rows)


def dispatch_geometry(N: int, tiles: int, plan: Plan) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """(grid, threadgroup) in threads, as oMLX _launch lays them out."""
    n_cols = (N + BN - 1) // BN
    if plan.gx:
        tg = (plan.gx, ((tiles + plan.gx - 1) // plan.gx) * n_cols)
    else:
        tg = (n_cols, tiles)
    return (tg[0] * 32, tg[1] * WN, plan.bm // 32), (32, WN, plan.bm // 32)


# ------------------------------------------------------------------ admission

@dataclass(frozen=True)
class MoEPrefillRequest:
    """Shape metadata of one sorted MoE prefill call (no tensors)."""

    tokens: int
    top_k: int
    experts: int
    hidden: int
    intermediate: int
    dtype: str = DTYPE
    mode: str = MODE
    bits: int = BITS
    group_size: int = GROUP_SIZE
    gate_up_layout: str = LAYOUT
    training: bool = False
    has_bias: bool = False
    shared_folded: bool = False


@dataclass(frozen=True)
class ProjectionPlan:
    K: int
    N: int
    out_cols: int
    paired: bool
    planned: Plan
    plan: Plan
    max_tiles: int
    align_k: bool
    align_n: bool
    grid: Tuple[int, int, int]
    threadgroup: Tuple[int, int, int]


_SEAL = object()


@dataclass(frozen=True)
class SegmentedMoEAdmission:
    request: MoEPrefillRequest
    rows: int
    rows_per_expert: int
    min_tokens: int
    gate_up: ProjectionPlan
    down: ProjectionPlan
    named_geometry: Optional[str]
    route: str = ORDINARY_REFERENCE
    plan_provenance: str = PLAN_PROVENANCE
    state: Mapping[str, Any] = field(default_factory=lambda: STATE, compare=False)
    _seal: object = field(default=None, repr=False, compare=False)


def min_admitted_tokens(top_k: int, experts: int) -> int:
    """Smallest T with T * k >= 16 and (T * k) // E >= 4."""
    need = max(MIN_ROWS, MIN_ROWS_PER_EXPERT * experts)
    return -(-need // top_k)


def signed_int_intermediates(K: int, N: int, plan: Plan, *, paired: bool) -> Mapping[str, int]:
    """Largest signed-int intermediates of the retained kernel text for one plan.

    Pointer additions are separate operations; only int products/sums are
    listed. Paired (gate/up) kernels address activation rows through the uint
    row map (bounded by T * D < 2**32 at admission), so they have no int row
    offset into x. ``tm`` is a short row-simdgroup offset (0..bm - 32).
    """
    tm_max = plan.bm - 32
    ld_out = N // 2 if paired else N            # store_act ldy = half_n; plain store ld = N
    out = {
        "k_times_bits": K * BITS,                # K_w = K * Q::kBits / 8; TileLoader K * kBits / 8
        "n_round_up": N + BN - 1,                # tile_of n_cols = (N + kBN - 1) / kBN
        "nax_fragment_offset": 24 * max(ld_out, 0 if paired else K),  # NAX load/store r * ld, r <= 24
    }
    if plan.sched == SCHED_SEG:
        out["seg_row_offset_out"] = tm_max * ld_out          # y + tm * ldy / y + tm * N
        if not paired:
            out["seg_row_offset_x"] = tm_max * K             # x + tm * K
            out["seg_substep_offset_x"] = 16 * K             # xn + mm * 16 * K
    elif not paired:
        out["db_row_offset_x"] = 31 * K + 28                 # x_off = r * K + sc.x
    return MappingProxyType(out)


def projection_plan(rows: int, experts: int, K: int, N: int, *, paired: bool,
                    pinned: Optional[Plan] = None) -> ProjectionPlan:
    """Plan + dispatch geometry; int32 uniforms, shapes, grid, scan loop and the
    signed intermediates of the CHOSEN plan (pinned or planned) are checked."""
    planned = plan_kernel(rows, experts, K, N) if pinned is None else validate_plan(pinned)
    plan = effective_plan(planned, K, N, paired=paired)
    tiles = max_tiles(rows, experts, plan.bm)
    grid, threadgroup = dispatch_geometry(N, tiles, plan)
    if max(rows, K, N, 4 * tiles) > INT32_MAX or max(grid) > INT32_MAX:
        _refuse("overflow", "rows, K, N, the tile buffer or the dispatch grid exceed int32")
    if rows > MAX_ROWS:
        _refuse("overflow", f"scan loop g0 += {SCAN_STRIDE} needs rows <= INT32_MAX - {SCAN_STRIDE}")
    over = {k: v for k, v in signed_int_intermediates(K, N, plan, paired=paired).items() if v > INT32_MAX}
    if over:
        _refuse("overflow", f"signed int intermediates exceed int32 for {plan.describe()}: {sorted(over)}")
    return ProjectionPlan(K=K, N=N, out_cols=N // 2 if paired else N, paired=paired,
                          planned=planned, plan=plan, max_tiles=tiles,
                          align_k=K % plan.bk == 0, align_n=N % BN == 0,
                          grid=grid, threadgroup=threadgroup)


def admit(request: MoEPrefillRequest) -> SegmentedMoEAdmission:
    """Pure metadata admission: no tensors, no MLX, no GPU, no side effects."""
    if not isinstance(request, MoEPrefillRequest):
        _refuse("type", "request must be a MoEPrefillRequest")
    r = request
    for name in ("training", "has_bias", "shared_folded"):
        if type(getattr(r, name)) is not bool:
            _refuse("type", f"{name} must be a bool")
    if r.training:
        _refuse("training", "training mode is not supported (inference prefill only)")
    if r.has_bias:
        _refuse("bias", "expert projection bias is not supported")
    if r.shared_folded:
        _refuse("shared_fold", "a shared expert folded into the expert table is not supported")
    if r.gate_up_layout != LAYOUT:
        _refuse("layout", f"only the fused [gate; up] table ({LAYOUT}) is supported")
    if r.dtype != DTYPE:
        _refuse("dtype", f"activations must be {DTYPE}")
    if r.mode != MODE or r.bits != BITS or r.group_size != GROUP_SIZE:
        _refuse("quantization", f"only {MODE} {BITS}-bit group {GROUP_SIZE} is supported")
    dims = ("tokens", "top_k", "experts", "hidden", "intermediate")
    if not all(_is_int(getattr(r, d)) and getattr(r, d) > 0 for d in dims):
        _refuse("shape", "tokens, top_k, experts, hidden, intermediate must be positive ints")
    T, k, E, D, I = (getattr(r, d) for d in dims)
    if E > MAX_EXPERTS or k > E:
        _refuse("shape", f"need 1 <= top_k <= experts <= {MAX_EXPERTS}")
    if D % DIM_ALIGN or I % DIM_ALIGN:
        _refuse("shape", f"hidden and intermediate must be multiples of {DIM_ALIGN}")
    if D > MAX_DIM or I > MAX_DIM:
        _refuse("overflow", f"hidden and intermediate must be <= INT32_MAX // 32 = {MAX_DIM}")
    rows = T * k
    if rows < MIN_ROWS or rows // E < MIN_ROWS_PER_EXPERT:
        _refuse("too_small", f"{rows} rows over {E} experts is below the row-block admission "
                f"(rows >= {MIN_ROWS}, rows // experts >= {MIN_ROWS_PER_EXPERT}; "
                f"T >= {min_admitted_tokens(k, E)} here); ordinary gather_qmv applies")
    # Room for the seam's padding, the scan loop stride, the uint4 tile buffer and int32 params.
    padded = rows + SEAM_PAD_MULTIPLE - 1
    if padded > MAX_ROWS:
        _refuse("overflow", f"sorted rows (+ seam pad) exceed INT32_MAX - {SCAN_STRIDE} (scan loop, shapes, params)")
    if 4 * max_tiles(padded, E, TILE_ROWS[0]) > INT32_MAX:
        _refuse("overflow", "rows exceed the int32 tile buffer")
    if T * D >= UINT32_LIMIT:
        _refuse("overflow", "token rows exceed 32-bit row-map element offsets")
    named = None
    for label, g in NAMED_GEOMETRIES.items():
        if (E, k, D, I) == (g["experts"], g["top_k"], g["hidden"], g["intermediate"]):
            named = label
    return SegmentedMoEAdmission(
        request=r, rows=rows, rows_per_expert=rows // E, min_tokens=min_admitted_tokens(k, E),
        gate_up=projection_plan(rows, E, D, 2 * I, paired=True),
        down=projection_plan(rows, E, I, D, paired=False),
        named_geometry=named, _seal=_SEAL)


@dataclass(frozen=True)
class RouteDecision:
    route: str
    research_candidate_admissible: bool
    reason: str
    qualified: bool = False
    selected: bool = False


def decide_route(request: MoEPrefillRequest, *, evidence: Any = None) -> RouteDecision:
    """Served-route decision: always the ordinary reference in this scope.

    ``evidence`` (a smoke result, host report, receipt, ...) is recorded in the
    reason and never promotes qualification or selection.
    """
    try:
        admit(request)
        admissible, why = True, "candidate admissible by metadata; native qualification external/pending"
    except SegmentedMoERefused as exc:
        admissible, why = False, str(exc)
    if evidence is not None:
        why += "; supplied evidence ignored: no qualification or selection is accepted in this scope"
    return RouteDecision(route=ORDINARY_REFERENCE, research_candidate_admissible=admissible, reason=why)


# ------------------------------------------------------- host route controls

@dataclass(frozen=True)
class SortedRoutes:
    """Host-validated sorted routes: what the candidate's tensors must encode."""

    tokens: int
    top_k: int
    experts: int
    sorted_experts: Tuple[int, ...]      # length assignments + pad
    row_map: Tuple[int, ...]             # sorted row -> token row
    order: Optional[Tuple[int, ...]]     # sorted row -> flat assignment (length assignments)
    inv_order: Optional[Tuple[int, ...]] # flat assignment -> sorted row
    pad: int
    runs: Tuple[Tuple[int, int, int], ...]  # (expert, start, stop), row order, non-empty
    expert_ids: Optional[Tuple[int, ...]] = None  # router ids [T * k], token-major, when known
    _seal: object = field(default=None, repr=False, compare=False)

    @property
    def rows(self) -> int:
        return len(self.sorted_experts)

    @property
    def assignments(self) -> int:
        return self.rows - self.pad


def _int_tuple(values, reason: str, name: str) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _refuse(reason, f"{name} must be a host sequence of ints")
    out = tuple(values)
    if not all(_is_int(v) for v in out):
        _refuse(reason, f"{name} must hold Python ints (no bools, floats or tensors)")
    return out


def seam_pad(assignments: int) -> int:
    """Rows the local ``_gather_sort`` appends (> 32768 and not a multiple of 64)."""
    if assignments > SEAM_PAD_THRESHOLD and assignments % SEAM_PAD_MULTIPLE:
        return SEAM_PAD_MULTIPLE - assignments % SEAM_PAD_MULTIPLE
    return 0


def sort_routes_host(expert_ids, *, top_k: int, experts: int, pad_policy: str = "none") -> SortedRoutes:
    """Host mirror of ``sort_routes``: order = argsort(ids), row_map = order // k.

    The host sort is stable; ``mx.argsort`` need not be, which only permutes
    rows inside an expert run (row_map and inv_order stay consistent).
    ``pad_policy="seam"`` mirrors the local ``_gather_sort`` padding.
    """
    if pad_policy not in ("none", "seam"):
        _refuse("routes", "pad_policy must be 'none' or 'seam'")
    if not (_is_int(top_k) and _is_int(experts) and 0 < top_k <= experts <= MAX_EXPERTS):
        _refuse("shape", "need ints 1 <= top_k <= experts <= 2048")
    ids = _int_tuple(expert_ids, "indices", "expert_ids")
    if not ids or len(ids) % top_k:
        _refuse("indices", "expert_ids must be a non-empty token-major [T, k] flattening")
    if any(not 0 <= e < experts for e in ids):
        _refuse("indices", f"expert ids must lie in [0, {experts})")
    n = len(ids)
    order = tuple(sorted(range(n), key=lambda i: (ids[i], i)))
    inv = [0] * n
    for pos, a in enumerate(order):
        inv[a] = pos
    sorted_experts = [ids[a] for a in order]
    row_map = [a // top_k for a in order]
    pad = seam_pad(n) if pad_policy == "seam" else 0
    sorted_experts += [sorted_experts[-1]] * pad
    row_map += [row_map[-1]] * pad
    return validate_sorted_routes(sorted_experts, row_map, tokens=n // top_k, top_k=top_k,
                                  experts=experts, order=order, inv_order=tuple(inv), pad=pad,
                                  expert_ids=ids)


def validate_sorted_routes(sorted_experts, row_map, *, tokens: int, top_k: int, experts: int,
                           order=None, inv_order=None, pad: int = 0, expert_ids=None) -> SortedRoutes:
    """Bounds, single-run expert ownership, row-map bounds and permutation checks."""
    if not all(_is_int(v) for v in (tokens, top_k, experts, pad)) or tokens <= 0 or pad < 0:
        _refuse("shape", "tokens, top_k, experts, pad must be ints (tokens > 0, pad >= 0)")
    if not 0 < top_k <= experts <= MAX_EXPERTS:
        _refuse("shape", "need 1 <= top_k <= experts <= 2048")
    se = _int_tuple(sorted_experts, "indices", "sorted_experts")
    rm = _int_tuple(row_map, "row_map", "row_map")
    n = tokens * top_k
    if len(se) != n + pad or len(rm) != n + pad:
        _refuse("routes", f"sorted_experts and row_map need {n} + pad {pad} rows")
    if any(not 0 <= e < experts for e in se):
        _refuse("indices", f"sorted expert ids must lie in [0, {experts})")
    if any(not 0 <= t < tokens for t in rm):
        _refuse("row_map", f"row map entries must lie in [0, {tokens})")
    runs, seen = [], set()
    for i, e in enumerate(se):
        if i and se[i - 1] == e:
            continue
        if e in seen:
            _refuse("indices", f"expert {e} owns more than one run of sorted rows")
        seen.add(e)
        runs.append([e, i, i])
    for run in runs:
        stop = run[1]
        while stop < len(se) and se[stop] == run[0]:
            stop += 1
        run[2] = stop
    if pad and any(se[i] != se[n - 1] or rm[i] != rm[n - 1] for i in range(n, n + pad)):
        _refuse("routes", "padding rows must repeat the last real row")
    if (order is None) != (inv_order is None):
        _refuse("routes", "order and inv_order come together")
    if order is not None:
        order = _int_tuple(order, "routes", "order")
        inv_order = _int_tuple(inv_order, "routes", "inv_order")
        if sorted(order) != list(range(n)) or len(inv_order) != n:
            _refuse("routes", "order must be a permutation of the assignments")
        if any(inv_order[a] != pos for pos, a in enumerate(order)):
            _refuse("routes", "inv_order must invert order")
        if any(rm[pos] != a // top_k for pos, a in enumerate(order)):
            _refuse("row_map", "row_map must equal order // top_k")
    if expert_ids is not None:
        if order is None:
            _refuse("routes", "expert_ids need the order that sorted them")
        expert_ids = _int_tuple(expert_ids, "indices", "expert_ids")
        if len(expert_ids) != n or any(se[pos] != expert_ids[a] for pos, a in enumerate(order)):
            _refuse("indices", "sorted_experts must equal expert_ids[order]")
    counts = [0] * tokens
    for t in rm[:n]:
        counts[t] += 1
    if any(c != top_k for c in counts):
        _refuse("row_map", f"every token row must be mapped exactly top_k={top_k} times")
    return SortedRoutes(tokens=tokens, top_k=top_k, experts=experts, sorted_experts=se, row_map=rm,
                        order=order, inv_order=inv_order, pad=pad, expert_ids=expert_ids,
                        runs=tuple(tuple(r) for r in runs), _seal=_SEAL)


def tile_table_host(sorted_experts, *, experts: int, bm: int) -> Tuple[Tuple[int, int, int], ...]:
    """Host mirror of the tile pre-pass: (row_start, expert, rows), expert-major.

    Run bounds come from neighbour comparison exactly as the kernel finds
    them, so an expert split over two runs leaves rows uncovered (the reason
    single-run ownership is validated before any dispatch).
    """
    se = _int_tuple(sorted_experts, "indices", "sorted_experts")
    if bm not in TILE_ROWS or not _is_int(experts) or not 0 < experts <= MAX_EXPERTS:
        _refuse("plan", "bm must be a tile height and experts in 1..2048")
    M = len(se)
    start, end = [0] * experts, [0] * experts
    for j, e in enumerate(se):
        if not 0 <= e < experts:
            continue
        if j == 0 or se[j - 1] != e:
            start[e] = j
        if j == M - 1 or se[j + 1] != e:
            end[e] = j + 1
    cap = max_tiles(M, experts, bm)
    tiles = []
    for e in range(experts):
        cnt = max(0, end[e] - start[e])
        for r in range(start[e], start[e] + cnt, bm):
            if len(tiles) < cap:
                tiles.append((r, e, min(bm, start[e] + cnt - r)))
    return tuple(tiles)


def restore_token_order_host(sorted_rows: Sequence[Any], routes: SortedRoutes) -> list:
    """Inverse restoration: flat assignment order, padding rows dropped."""
    if not isinstance(routes, SortedRoutes) or routes._seal is not _SEAL or routes.inv_order is None:
        _refuse("unsealed", "routes must come from sort_routes_host/validate_sorted_routes with an order")
    if len(sorted_rows) != routes.rows:
        _refuse("routes", f"need {routes.rows} sorted rows")
    return [sorted_rows[p] for p in routes.inv_order]


# ------------------------------------------------------- engagement counters

COUNTER_CAP = 2**63 - 1
DISPATCH_KINDS = ("gate_up_mapped_swiglu", "down_segmented")


class EngagementStats:
    """Bounded host counters (fixed keys, saturating) of the research entry.

    ``*.successful_chains.<kind>`` counts lazy dispatch chains (tile scan +
    matmul) that BOTH returned into the graph: not every issued scan (a scan
    issued before a raising matmul is not counted) and NOT GPU executions.
    ``backend_raised`` counts calls whose backend construction, capability
    check or dispatch raised (the exception still propagates). A future
    nonzero gate must pair a fresh successful chain with an evaluated,
    checked output. Substituted backends are counted apart, never as native.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._epoch = 0
        self._counts = {"calls": 0, "backend_raised": 0}
        self._counts.update({f"refused.{r}": 0 for r in REFUSAL_REASONS})
        self._counts.update({f"native.successful_chains.{k}": 0 for k in DISPATCH_KINDS})
        self._counts.update({f"substituted.successful_chains.{k}": 0 for k in DISPATCH_KINDS})

    def _bump(self, key: str) -> None:
        with self._lock:
            self._counts[key] = min(COUNTER_CAP, self._counts[key] + 1)
            self._epoch = min(COUNTER_CAP, self._epoch + 1)

    def snapshot(self) -> Mapping[str, int]:
        with self._lock:
            return MappingProxyType(dict(self._counts, epoch=self._epoch))

    def fresh_since(self, before: Mapping[str, int]) -> Mapping[str, int]:
        now = self.snapshot()
        return MappingProxyType({k: now[k] - before.get(k, 0) for k in now if k != "epoch"})


ENGAGEMENT = EngagementStats()


# ------------------------------------------------------ explicit research entry

def _dtype_name(dtype) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def _meta(array, name: str) -> Tuple[Tuple[int, ...], str]:
    shape, dtype = getattr(array, "shape", None), getattr(array, "dtype", None)
    if shape is None or dtype is None:
        _refuse("type", f"{name} must be an array with shape and dtype")
    return tuple(int(d) for d in shape), _dtype_name(dtype)


def _expect(array, name: str, shape: Tuple[int, ...], dtype: str) -> None:
    got_shape, got_dtype = _meta(array, name)
    if got_shape != shape:
        _refuse("shape", f"{name} has shape {got_shape}, need {shape}")
    if got_dtype != dtype:
        _refuse("dtype" if dtype == DTYPE else "quantization", f"{name} has dtype {got_dtype}, need {dtype}")


def _check_table(weight, scales, biases, *, experts: int, N: int, K: int, label: str) -> None:
    if biases is None:
        _refuse("quantization", f"{label} needs affine biases")
    _expect(weight, f"{label} weight", (experts, N, K * BITS // 32), "uint32")
    _expect(scales, f"{label} scales", (experts, N, K // GROUP_SIZE), DTYPE)
    _expect(biases, f"{label} biases", (experts, N, K // GROUP_SIZE), DTYPE)


def _gate(admission, routes, research_opt_in) -> None:
    """Opt-in, then seal AND content: sealed objects are re-admitted and revalidated,
    so a dataclasses.replace() copy (which keeps the seal) cannot pass altered."""
    if research_opt_in is not True:
        _refuse("opt_in_missing", "ordinary reference only: the research candidate needs "
                "research_opt_in=True (acknowledgement, not GPU ownership)")
    if not isinstance(admission, SegmentedMoEAdmission) or admission._seal is not _SEAL:
        _refuse("unsealed", "admission must come from admit()")
    if admit(admission.request) != admission or admission.state is not STATE:
        _refuse("unsealed", "admission content differs from admit(request)")
    if not isinstance(routes, SortedRoutes) or routes._seal is not _SEAL:
        _refuse("unsealed", "routes must come from sort_routes_host/validate_sorted_routes")
    if routes.expert_ids is None or routes.order is None:
        _refuse("routes", "the research entry needs routes sorted from router ids (sort_routes_host)")
    fresh = validate_sorted_routes(routes.sorted_experts, routes.row_map, tokens=routes.tokens,
                                   top_k=routes.top_k, experts=routes.experts, order=routes.order,
                                   inv_order=routes.inv_order, pad=routes.pad,
                                   expert_ids=routes.expert_ids)
    if fresh != routes:
        _refuse("unsealed", "routes content differs from their revalidation")
    r = admission.request
    if (routes.tokens, routes.top_k, routes.experts) != (r.tokens, r.top_k, r.experts):
        _refuse("routes", "routes do not match the admitted request")
    if routes.assignments != admission.rows:
        _refuse("routes", "route assignments do not match the admitted rows")


def _run(kind: str, admission, routes, research_opt_in, prepare, dispatch, backend):
    stats = ENGAGEMENT
    stats._bump("calls")
    try:
        _gate(admission, routes, research_opt_in)
        prepared = prepare()
    except SegmentedMoERefused as exc:
        stats._bump(f"refused.{exc.reason}")
        raise
    constructed = backend is None
    try:
        if constructed:
            backend = _native_backend()  # first MLX import, only past every refusal
        backend.require_capability()
        out = dispatch(backend, prepared)
    except BaseException:
        stats._bump("backend_raised")
        raise                            # genuine backend errors propagate unchanged
    # Native only for a backend this entry constructed AND of the class captured at
    # import (no instance/subclass passthrough, no substituted factory or class).
    prefix = "native" if constructed and _is_native(backend) else "substituted"
    stats._bump(f"{prefix}.successful_chains.{kind}")
    return ResearchOutput(output=out, kind=kind, plan=prepared.plan,
                          rows=routes.rows, native=prefix == "native")


@dataclass(frozen=True)
class ResearchOutput:
    output: Any
    kind: str
    plan: ProjectionPlan
    rows: int
    native: bool
    route: str = "research-explicit-unserved"
    state: Mapping[str, Any] = field(default_factory=lambda: STATE, compare=False)


@dataclass(frozen=True)
class _Prepared:
    plan: ProjectionPlan
    args: Tuple[Any, ...]


def research_mapped_gate_up_swiglu(x_tokens, routes: SortedRoutes, weight, scales, biases, *,
                                   admission: SegmentedMoEAdmission, research_opt_in: bool = False,
                                   plan: Optional[Plan] = None, backend=None) -> ResearchOutput:
    """``silu(gate) * up`` of the fused ``[gate; up]`` table for ``x_tokens[row_map]``.

    ``x_tokens`` ``[T, 1, D]`` bf16; ``weight``/``scales``/``biases`` the fused
    table ``[E, 2I, *]``. Returns ``[M, 1, I]`` in sorted row order (M =
    assignments + padding). Research only, unqualified, unserved.
    """
    def prepare():
        r = admission.request
        _expect(x_tokens, "x_tokens", (r.tokens, 1, r.hidden), DTYPE)
        _check_table(weight, scales, biases, experts=r.experts, N=2 * r.intermediate,
                     K=r.hidden, label="gate_up")
        pp = projection_plan(routes.rows, r.experts, r.hidden, 2 * r.intermediate,
                             paired=True, pinned=plan)
        return _Prepared(pp, (x_tokens, weight, scales, biases))

    def dispatch(be, p):
        x, w, s, b = p.args
        return be.mapped_gate_up_swiglu(x, be.upload_u32(routes.row_map), be.upload_u32(routes.sorted_experts),
                                        w, s, b, plan=p.plan, rows=routes.rows, experts=routes.experts)

    return _run("gate_up_mapped_swiglu", admission, routes, research_opt_in, prepare, dispatch, backend)


def research_segmented_down(hidden_sorted, routes: SortedRoutes, weight, scales, biases, *,
                            admission: SegmentedMoEAdmission, research_opt_in: bool = False,
                            plan: Optional[Plan] = None, backend=None) -> ResearchOutput:
    """Segmented sorted down projection of ``[M, 1, I]`` sorted rows -> ``[M, 1, D]``."""
    def prepare():
        r = admission.request
        _expect(hidden_sorted, "hidden_sorted", (routes.rows, 1, r.intermediate), DTYPE)
        _check_table(weight, scales, biases, experts=r.experts, N=r.hidden, K=r.intermediate,
                     label="down")
        pp = projection_plan(routes.rows, r.experts, r.intermediate, r.hidden,
                             paired=False, pinned=plan)
        return _Prepared(pp, (hidden_sorted, weight, scales, biases))

    def dispatch(be, p):
        x, w, s, b = p.args
        return be.segmented_down(x, be.upload_u32(routes.sorted_experts), w, s, b,
                                 plan=p.plan, rows=routes.rows, experts=routes.experts)

    return _run("down_segmented", admission, routes, research_opt_in, prepare, dispatch, backend)


# ------------------------------------------------------------ native backend
#
# Everything below runs only from an admitted, opted-in research call.
# Kernel text adapted from oMLX omlx/patches/m5_gather_qmm_nax.py @ d6b2b92
# (Apache-2.0, the oMLX authors). The NAX tile primitives and elementwise
# functors are read at build time from the INSTALLED mlx package headers
# (MIT, Copyright Apple Inc.); none are vendored in this repository.

_MLX_UTILS_HEADERS = (
    "mlx/backend/metal/kernels/utils.h",
    "mlx/backend/metal/kernels/bf16.h",
    "mlx/backend/metal/kernels/bf16_math.h",
    "mlx/backend/metal/kernels/complex.h",
    "mlx/backend/metal/kernels/defines.h",
    "mlx/backend/metal/kernels/logging.h",
)
_MLX_MM_HEADERS = ("mlx/backend/metal/kernels/steel/gemm/nax.h",)
_MLX_OPS_HEADERS = (
    "mlx/backend/metal/kernels/unary_ops.h",
    "mlx/backend/metal/kernels/binary_ops.h",
)


def read_mlx_headers(root: Path, paths: Tuple[str, ...]) -> str:
    """Flatten installed mlx kernel headers (oMLX ``_read_mlx_headers``).

    ``mx.fast.metal_kernel`` prepends mlx's ``utils.h`` preamble, so it (and
    what it includes) is skipped; quoted mlx includes are inlined once and
    ``#pragma once`` dropped. Missing headers raise (never a silent decline).
    """
    root = Path(root)
    if not root.is_dir():
        raise SegmentedMoEUnavailable(f"mlx kernel headers not found under {root}")
    seen = {root / p for p in _MLX_UTILS_HEADERS}

    def expand(rel: str) -> str:
        path = root / rel
        if path in seen:
            return ""
        seen.add(path)
        lines = []
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith('#include "mlx/') and stripped.endswith('"'):
                lines.append(expand(stripped[len('#include "'):-1]))
            elif stripped != "#pragma once":
                lines.append(line)
        return "\n".join(lines)

    try:
        return "\n".join(expand(p) for p in paths)
    except OSError as exc:
        raise SegmentedMoEUnavailable(f"mlx kernel header unreadable: {exc}") from exc


_KERNEL_LOCK = threading.Lock()
_KERNELS: dict = {}


class _MLXBackend:
    """Lazy MLX backend; constructed only by an admitted research call."""

    def __init__(self):
        import mlx.core as mx  # noqa: PLC0415 - deliberately lazy

        self.mx = mx

    def require_capability(self) -> None:
        mx = self.mx
        if not mx.metal.is_available():
            raise SegmentedMoEUnavailable("Metal is not available")
        arch = str(mx.device_info().get("architecture", ""))
        if not arch.startswith("applegpu_g17"):
            raise SegmentedMoEUnavailable(f"NAX tensor units need an M5-class GPU, found {arch!r}")

    def upload_u32(self, values):
        return self.mx.array(list(values), dtype=self.mx.uint32)

    def _kernel(self, kind: str):
        with _KERNEL_LOCK:
            kernel = _KERNELS.get(kind)
            if kernel is not None:
                return kernel
            mx = self.mx
            if kind == "scan":
                kernel = mx.fast.metal_kernel(
                    name="mlx2_research_gqmm_tile_scan", input_names=["idx", "params"],
                    output_names=["tiles", "tile_count"], header=_SCAN_HEADER, source=_SCAN_SOURCE)
            else:
                root = Path(mx.__file__).parent / "include"
                mapped = kind == "gate_up"
                headers = _MLX_MM_HEADERS + (_MLX_OPS_HEADERS if mapped else ())
                header = read_mlx_headers(root, headers) + _MM_HEADER + (_ACT_HEADER if mapped else "")
                inputs = ["x", "w", "scales", "biases", "tiles", "tile_count", "params"]
                if mapped:
                    inputs += ["lim", "rmap"]
                kernel = mx.fast.metal_kernel(
                    name="mlx2_research_gqmm_affine_swiglu_map" if mapped else "mlx2_research_gqmm_affine",
                    input_names=inputs, output_names=["y"], header=header,
                    source=_AFFINE_ACT_MAP_SOURCE if mapped else _AFFINE_SOURCE)
            _KERNELS[kind] = kernel
            return kernel

    def _tiles(self, idx, plan: ProjectionPlan, rows: int, experts: int):
        mx = self.mx
        return self._kernel("scan")(
            inputs=[idx, mx.array([rows, experts, plan.max_tiles], dtype=mx.int32)],
            template=[("BM", plan.plan.bm), ("MAXE", MAX_EXPERTS)],
            grid=(1024, 1, 1), threadgroup=(1024, 1, 1),
            output_shapes=[(plan.max_tiles * 4,), (1,)], output_dtypes=[mx.uint32, mx.uint32])

    def _grid(self, plan: ProjectionPlan):
        return dict(grid=plan.grid, threadgroup=plan.threadgroup)

    def _template(self, plan: ProjectionPlan, x):
        p = plan.plan
        return [("T", x.dtype), ("GS", GROUP_SIZE), ("BITS", BITS), ("SCHED", int(p.sched))]

    def _tail(self, plan: ProjectionPlan):
        p = plan.plan
        return [("ALIGN_K", plan.align_k), ("BM", p.bm), ("BK", p.bk), ("GX", p.gx), ("PAD", p.pad)]

    def mapped_gate_up_swiglu(self, x, row_map, idx, w, scales, biases, *, plan, rows, experts):
        mx = self.mx
        tiles, tile_count = self._tiles(idx, plan, rows, experts)
        return self._kernel("gate_up")(
            inputs=[x, w, scales, biases, tiles, tile_count,
                    mx.array([plan.N, plan.K], dtype=mx.int32),
                    mx.array(0.0, dtype=x.dtype).reshape(1), row_map],
            template=self._template(plan, x) + [("EPI", 1)] + self._tail(plan),
            output_shapes=[(rows, 1, plan.out_cols)], output_dtypes=[x.dtype], **self._grid(plan))[0]

    def segmented_down(self, x, idx, w, scales, biases, *, plan, rows, experts):
        mx = self.mx
        tiles, tile_count = self._tiles(idx, plan, rows, experts)
        return self._kernel("down")(
            inputs=[x, w, scales, biases, tiles, tile_count, mx.array([plan.N, plan.K], dtype=mx.int32)],
            template=self._template(plan, x) + [("ALIGN_N", plan.align_n)] + self._tail(plan),
            output_shapes=[(rows, 1, plan.out_cols)], output_dtypes=[x.dtype], **self._grid(plan))[0]


# The native class is bound here, at import, as default arguments: rebinding the
# module name _MLXBackend or substituting _native_backend cannot make a CPU fake
# count as native. Evidence-association hygiene, not protection against arbitrary code.
def _native_backend(_cls=_MLXBackend):
    return _cls()


def _is_native(backend, _cls=_MLXBackend) -> bool:
    return type(backend) is _cls


# --------------------------------------------------------------- kernel text
# Adapted from oMLX omlx/patches/m5_gather_qmm_nax.py @ d6b2b92 (Apache-2.0).
# Modifications (see provenance): MXFP4 loader removed; clamped (EPI == 2)
# epilogue removed; only the affine plain and affine row-mapped SwiGLU
# sources are kept. Every other line is the upstream text.

_SCAN_HEADER = r"""
using namespace metal;

// Cuts the sorted rows into (row_start, expert, rows, 0) tiles of at most
// BM rows of one expert, expert-major (the tile order of mlx's segmented
// gather_qmm). One threadgroup: the run bounds of every expert are found in
// parallel over the rows, then a threadgroup scan of the per-expert tile
// counts gives each expert's first tile. At most max_tiles tiles are
// written (a guard for unsorted input, which the contract excludes).
template <int BM>
METAL_FUNC void omlx_gqmm_tile_scan(
    const device uint32_t* idx,
    const constant int* params,
    device uint32_t* tiles,
    device uint32_t* tile_count,
    threadgroup uint32_t* run_start,
    threadgroup uint32_t* run_end,
    threadgroup uint32_t* simd_tot,
    const uint lid,
    const uint tg_size,
    const uint sg,
    const uint lane) {
  const int M = params[0];
  const int E = params[1];
  const uint32_t max_tiles = uint32_t(params[2]);
  for (int e = int(lid); e < E; e += int(tg_size)) {
    run_start[e] = 0;
    run_end[e] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Each thread walks 4 consecutive rows per step, with their neighbours
  // (0xffffffff past either end, never a valid expert).
  for (int g0 = 4 * int(lid); g0 < M; g0 += 4 * int(tg_size)) {
    const int cnt = min(4, M - g0);
    uint32_t v[6];
    v[0] = g0 > 0 ? idx[g0 - 1] : 0xffffffffu;
    for (int j = 0; j < 4; j++) {
      v[j + 1] = j < cnt ? idx[g0 + j] : 0xffffffffu;
    }
    v[5] = g0 + 4 < M ? idx[g0 + 4] : 0xffffffffu;
    for (int j = 0; j < cnt; j++) {
      const uint32_t e = v[j + 1];
      if (e < uint32_t(E)) {
        if (v[j] != e) {
          run_start[e] = uint32_t(g0 + j);
        }
        if (v[j + 2] != e) {
          run_end[e] = uint32_t(g0 + j + 1);
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n_simd = (tg_size + 31) / 32;
  uint32_t running = 0;
  for (int base = 0; base < E; base += int(tg_size)) {
    const int e = base + int(lid);
    uint32_t start = 0;
    uint32_t cnt = 0;
    if (e < E) {
      start = run_start[e];
      const uint32_t end = run_end[e];
      cnt = end > start ? end - start : 0;
    }
    const uint32_t nt = (cnt + BM - 1) / BM;
    const uint32_t local = simd_prefix_exclusive_sum(nt);
    const uint32_t stot = simd_sum(nt);
    if (lane == 0) {
      simd_tot[sg] = stot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint32_t prefix = 0;
    uint32_t total = 0;
    for (uint s = 0; s < n_simd; s++) {
      const uint32_t v = simd_tot[s];
      prefix += (s < sg) ? v : 0;
      total += v;
    }
    const uint32_t off = running + prefix + local;
    for (uint32_t j = 0; j < nt && off + j < max_tiles; j++) {
      const uint32_t r = start + j * BM;
      *((device uint4*)tiles + off + j) =
          uint4(r, uint32_t(e), min(uint32_t(BM), start + cnt - r), 0);
    }
    running += total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) {
    tile_count[0] = min(running, max_tiles);
  }
}
"""

_SCAN_SOURCE = r"""
    threadgroup uint32_t run_start[MAXE];
    threadgroup uint32_t run_end[MAXE];
    threadgroup uint32_t simd_tot[32];
    omlx_gqmm_tile_scan<BM>(
        idx, params, tiles, tile_count, run_start, run_end, simd_tot,
        thread_index_in_threadgroup, threads_per_threadgroup.x,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

_MM_HEADER = r"""
using namespace metal;
using namespace mlx::steel;

namespace omlx_gqmm {

STEEL_CONST int kBN = 64;
STEEL_CONST int kWN = 2;
STEEL_CONST short kSM = 32;
STEEL_CONST short kSN = kBN / kWN;
STEEL_CONST short kSK = 32;
STEEL_CONST short kTM = kSM / 16;
STEEL_CONST short kTN = kSN / 16;
STEEL_CONST short kTK = kSK / 16;

// Tile geometry: BM rows in BM / 32 row simdgroups times kWN column
// simdgroups, K steps BK deep. kLT loader threads dequantize the kBN x BK
// weight tile, each kVPT consecutive values of one weight row: every
// thread when they split the tile evenly, else the largest power of two
// below the thread count (96-row tiles: 128 of 192).
template <int BM, int BK>
struct Geo {
  STEEL_CONST int kBM = BM;
  STEEL_CONST int kBK = BK;
  STEEL_CONST int kWM = BM / kSM;
  STEEL_CONST int kThreads = kWM * kWN * 32;
  STEEL_CONST int kLT = (kThreads & (kThreads - 1)) == 0
      ? kThreads
      : (kThreads > 256 ? 256 : (kThreads > 128 ? 128 : 64));
  STEEL_CONST int kVPT = kBN * BK / kLT;
  STEEL_CONST int kTPR = BK / kVPT;
  static_assert(BM % kSM == 0 && BK % kSK == 0, "tile geometry");
  static_assert(kTPR >= 1 && kTPR * kVPT == BK, "loader split");
};

// Affine: w = scale * q + bias computed in fp32 and rounded once to T, as
// mlx's dequantize() does (scale * q is exact in fp32).
template <typename T, int GS, int BITS>
struct AffineQ {
  using WT = T;
  STEEL_CONST int kBits = BITS;
  STEEL_CONST int kGroup = GS;
  const device T* scales;
  const device T* biases;

  struct P {
    float s;
    float b;
  };

  METAL_FUNC void advance(const size_t n) thread {
    scales += n;
    biases += n;
  }
  METAL_FUNC P params(const int g) const thread {
    return P{float(scales[g]), float(biases[g])};
  }
  METAL_FUNC static WT dq(thread const P& p, const uint32_t q) {
    return static_cast<WT>(p.s * float(q) + p.b);
  }
};

// Gate/up pairing (activation epilogue): weight rows [gate; up] of one
// expert, half_n rows each. Row r of a paired kBN x BK weight tile loads
// weight row pair_row(r) past the tile's first gate row: 16-row blocks
// alternate the gate and the up rows of the same 16 output columns, so the
// two 16-column fragments of every simdgroup's 32-column block accumulate
// gate and up of the same output columns in the same lanes.
METAL_FUNC int pair_row(const int r, const int half_n) {
  return ((r >> 4) & 1) * half_n + ((r >> 5) << 4) + (r & 15);
}

// Activation epilogue of a paired simdgroup block (defined with the
// activation kernels only; see _ACT_HEADER).
template <typename T, int EPI, typename DTile>
METAL_FUNC void store_act(
    thread const DTile& D,
    device T* y,
    const int ld,
    const int rows,
    const T limit);

// Weight-tile loader: loader thread lid owns row lid / kTPR of the
// kBN x BK tile and the kVPT values from column (lid % kTPR) * kVPT, in
// kNG chunks that each lie in one quantization group. fetch() reads the
// packed words and group parameters of one K step, store() dequantizes
// them into threadgroup memory (row stride BKP). The *_tail variants
// cover a K tail of k_valid (a multiple of 32) columns and never touch a
// word or group at or past it. PAIR maps tile rows through pair_row().
template <typename Q, typename G, bool PAIR = false>
struct TileLoader {
  using WT = typename Q::WT;
  using P = typename Q::P;
  STEEL_CONST int kBits = Q::kBits;
  STEEL_CONST int kVPT = G::kVPT;
  STEEL_CONST int kWords = kVPT * kBits / 32;
  STEEL_CONST int kPer = 32 / kBits;
  STEEL_CONST uint32_t kMask = (1u << kBits) - 1u;
  STEEL_CONST int kGV = kVPT < Q::kGroup ? kVPT : Q::kGroup;
  STEEL_CONST int kNG = kVPT / kGV;
  STEEL_CONST int kWPG = kGV * kBits / 32;
  STEEL_CONST int kBKP = G::kBK + 16 / sizeof(WT);
  static_assert(kWords * 32 == kVPT * kBits, "whole words per thread");
  static_assert(kWPG >= 1 && kNG * kWPG == kWords, "group split");

  const device uint32_t* src;
  Q q;
  const short row;
  const short col;
  uint32_t raw[kWords];
  P p[kNG];

  METAL_FUNC TileLoader(
      const device uint8_t* w_tile,
      const int K,
      thread const Q& q_,
      const uint lid,
      const int half_n = 0) thread
      : q(q_),
        row(short(lid / G::kTPR)),
        col(short((lid % G::kTPR) * kVPT)) {
    const size_t w_off = PAIR ? size_t(pair_row(row, half_n)) : size_t(row);
    src = (const device uint32_t*)(w_tile + w_off * (K * kBits / 8) +
                                   col * kBits / 8);
    q.advance(w_off * (K / Q::kGroup));
  }

  METAL_FUNC void fetch(const int kb) thread {
    const device uint32_t* ptr = src + kb * (G::kBK * kBits / 32);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      raw[i] = ptr[i];
    }
    STEEL_PRAGMA_UNROLL
    for (short g = 0; g < kNG; g++) {
      p[g] = q.params((kb * G::kBK + col + g * kGV) / Q::kGroup);
    }
  }

  METAL_FUNC void fetch_tail(const int kb, const int k_valid) thread {
    const device uint32_t* ptr = src + kb * (G::kBK * kBits / 32);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      if (col + i * kPer < k_valid) {
        raw[i] = ptr[i];
      }
    }
    STEEL_PRAGMA_UNROLL
    for (short g = 0; g < kNG; g++) {
      if (col + g * kGV < k_valid) {
        p[g] = q.params((kb * G::kBK + col + g * kGV) / Q::kGroup);
      }
    }
  }

  METAL_FUNC void store_words(threadgroup WT* Ws, const int k_valid) const
      thread {
    threadgroup WT* dst = Ws + row * kBKP + col;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      if (col + i * kPer < k_valid) {
        vec<WT, kPer> v;
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kPer; j++) {
          v[j] = Q::dq(p[i / kWPG], (raw[i] >> (kBits * j)) & kMask);
        }
        *(threadgroup vec<WT, kPer>*)(dst + i * kPer) = v;
      }
    }
  }

  METAL_FUNC void store(threadgroup WT* Ws) const thread {
    store_words(Ws, G::kBK);
  }

  METAL_FUNC void zero(threadgroup WT* Ws) const thread {
    threadgroup WT* dst = Ws + row * kBKP + col;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kVPT; i++) {
      dst[i] = WT(0);
    }
  }
};

// One 32-deep sub-step of a simdgroup's 32 x 32 block: full row blocks run
// tile_matmad_nax; partial ones skip the 16-row fragments without rows
// (the tensor ops of the others are the ones tile_matmad_nax issues).
template <typename T, typename WT, int BKP, bool FULL>
METAL_FUNC void sub_step(
    thread NAXTile<float, kTM, kTN>& Dtile,
    const device T* xn,
    const threadgroup WT* ws,
    const int K,
    const short sgp_sm) {
  NAXTile<WT, kTN, kTK> Btile;
  if constexpr (FULL) {
    NAXTile<T, kTM, kTK> Atile;

    volatile int compiler_barrier;

    Atile.load(xn, K);
    Btile.template load<WT, BKP, 1>(ws);

    tile_matmad_nax(
        Dtile,
        Atile,
        metal::bool_constant<false>{},
        Btile,
        metal::bool_constant<true>{});

    (void)compiler_barrier;
  } else {
    Btile.template load<WT, BKP, 1>(ws);
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < kTM; mm++) {
      if (mm * 16 < sgp_sm) {
        NAXTile<T, 1, kTK> Arow;
        Arow.load_safe(xn + mm * 16 * K, K, short2(kSK, sgp_sm - mm * 16));
        STEEL_PRAGMA_UNROLL
        for (short nn = 0; nn < kTN; nn += 2) {
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < kTK; kk++) {
            BaseNAXFrag::mma(
                Dtile.frag_at(mm, nn),
                Dtile.frag_at(mm, nn + 1),
                Arow.frag_at(0, kk),
                metal::bool_constant<false>{},
                Btile.frag_at(nn, kk),
                Btile.frag_at(nn + 1, kk),
                metal::bool_constant<true>{});
          }
        }
      }
    }
  }
}

// Row-mapped activations (MAP): sorted row r of the product is token row
// rmap[r] of x, read in place instead of from a replicated copy. a_off[i][h]
// is this lane's element offset of activation row i * 16 + h * 8 + sc.y of
// its simdgroup block (rmap[row] * K + sc.x; rows past the block point at
// its last row), so lane values are exactly those NAXTile::load reads from
// the copy.
template <typename T, short R>
METAL_FUNC void load_a_map(
    thread NAXTile<T, R, kTK>& A,
    const device T* xk,
    const thread uint (&a_off)[kTM][2],
    const short i0) {
  STEEL_PRAGMA_UNROLL
  for (short r = 0; r < R; r++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const device T* xp = xk + a_off[i0 + r][h];
      STEEL_PRAGMA_UNROLL
      for (short kk = 0; kk < kTK; kk++) {
        const vec<T, 4> v = *(const device vec<T, 4>*)(xp + kk * 16);
        STEEL_PRAGMA_UNROLL
        for (short c = 0; c < 4; c++) {
          A.frag_at(r, kk)[h * 4 + c] = v[c];
        }
      }
    }
  }
}

// sub_step with row-mapped activations (xk: the token rows advanced to this
// sub-step's K offset): the same fragment values (rows past sgp_sm of a
// partial block zero as load_safe makes them) and the same tensor ops in the
// same order.
template <typename T, typename WT, int BKP, bool FULL>
METAL_FUNC void sub_step_map(
    thread NAXTile<float, kTM, kTN>& Dtile,
    const device T* xk,
    const thread uint (&a_off)[kTM][2],
    const threadgroup WT* ws,
    const short sgp_sm) {
  NAXTile<WT, kTN, kTK> Btile;
  if constexpr (FULL) {
    NAXTile<T, kTM, kTK> Atile;

    volatile int compiler_barrier;

    load_a_map<T, kTM>(Atile, xk, a_off, 0);
    Btile.template load<WT, BKP, 1>(ws);

    tile_matmad_nax(
        Dtile,
        Atile,
        metal::bool_constant<false>{},
        Btile,
        metal::bool_constant<true>{});

    (void)compiler_barrier;
  } else {
    const short2 sc = BaseNAXFrag::get_coord();
    Btile.template load<WT, BKP, 1>(ws);
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < kTM; mm++) {
      if (mm * 16 < sgp_sm) {
        NAXTile<T, 1, kTK> Arow;
        load_a_map<T, 1>(Arow, xk, a_off, mm);
        STEEL_PRAGMA_UNROLL
        for (short h = 0; h < 2; h++) {
          if (mm * 16 + h * 8 + sc.y >= sgp_sm) {
            STEEL_PRAGMA_UNROLL
            for (short kk = 0; kk < kTK; kk++) {
              STEEL_PRAGMA_UNROLL
              for (short c = 0; c < 4; c++) {
                Arow.frag_at(0, kk)[h * 4 + c] = T(0);
              }
            }
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short nn = 0; nn < kTN; nn += 2) {
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < kTK; kk++) {
            BaseNAXFrag::mma(
                Dtile.frag_at(mm, nn),
                Dtile.frag_at(mm, nn + 1),
                Arow.frag_at(0, kk),
                metal::bool_constant<false>{},
                Btile.frag_at(nn, kk),
                Btile.frag_at(nn + 1, kk),
                metal::bool_constant<true>{});
          }
        }
      }
    }
  }
}

// Element offsets of this lane's activation rows (see load_a_map) for the
// simdgroup block starting at tile row m0 of a tile of tile_rows rows at
// sorted row row_start.
METAL_FUNC void map_rows(
    thread uint (&a_off)[kTM][2],
    const device uint32_t* rmap,
    const int row_start,
    const int tile_rows,
    const int m0,
    const int K) {
  const short2 sc = BaseNAXFrag::get_coord();
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kTM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = min(m0 + i * 16 + h * 8 + int(sc.y), tile_rows - 1);
      a_off[i][h] = rmap[row_start + r] * uint(K) + uint(sc.x);
    }
  }
}

// seg: mlx's segmented sorted gather kernel (affine_gather_qmm_rhs_seg_nax /
// fp_gather_qmm_rhs_seg_nax): one single-expert BM x kBN tile per
// threadgroup, the weight tile of each BK-deep K step dequantized into
// threadgroup memory between two barriers. K tail (K % BK, a multiple of
// 32): only its sub-steps run. N tail: weight rows past N are zero, stores
// are bounded.
//
// EPI > 0 (activation epilogue; N = 2 * half_n [gate; up] rows, aligned):
// the tile of fused columns [y_col, y_col + kBN) computes gate and up of
// output columns [y_col / 2, y_col / 2 + kBN / 2) through the paired row
// map and writes act(gate, up) to the [M, half_n] output.
template <
    typename T,
    typename Q,
    typename G,
    bool ALIGN_N,
    bool ALIGN_K,
    int EPI = 0,
    bool MAP = false>
METAL_FUNC void gather_seg(
    const device T* x,
    const device uint32_t* rmap,
    const device uint8_t* w,
    thread Q& q,
    const uint4 desc,
    const int y_col,
    device T* y,
    const int N,
    const int K,
    threadgroup typename Q::WT* Ws,
    const uint sgid,
    const uint lane,
    const T limit = T(0)) {
  using WT = typename Q::WT;
  constexpr bool kPair = EPI != 0;
  static_assert(!kPair || ALIGN_N, "paired gate/up tiles are full");
  constexpr int BKP = G::kBK + 16 / sizeof(WT);
  const int row_start = int(desc.x);
  const uint32_t expert = desc.y;
  const int rows = int(desc.z);

  const int K_w = K * Q::kBits / 8;
  const int K_g = K / Q::kGroup;
  const int K_it = K / G::kBK;
  const short tgp_bn = ALIGN_N ? short(kBN) : short(min(kBN, N - y_col));
  const int k_remain = K - K_it * G::kBK;
  // First weight row of the tile past the expert's rows and the output
  // row stride (paired: the tile's first output column, half_n columns).
  const int half_n = N / 2;
  const int w_col = kPair ? y_col / 2 : y_col;
  const int ldy = kPair ? half_n : N;

  const size_t w_row = size_t(expert) * N + w_col;
  q.advance(w_row * K_g);
  TileLoader<Q, G, kPair> loader(
      w + w_row * K_w, K, q, sgid * 32 + lane, half_n);
  const bool loads = G::kLT == G::kThreads || sgid * 32 + lane < uint(G::kLT);
  const bool row_live = ALIGN_N || loader.row < tgp_bn;

  if constexpr (!MAP) {
    x += size_t(row_start) * K;
  }
  y += size_t(row_start) * ldy + w_col;

  const short tm = kSM * short(sgid / kWN);
  const short tn = kSN * short(sgid % kWN);
  const short sgp_sm = short(min(int(kSM), max(0, rows - int(tm))));
  const short sgp_sn =
      ALIGN_N ? kSN : short(min(int(kSN), max(0, N - (y_col + tn))));
  const bool sg_active = sgp_sm > 0;
  uint a_off[kTM][2];
  if constexpr (MAP) {
    map_rows(a_off, rmap, row_start, rows, int(tm), K);
  }

  NAXTile<float, kTM, kTN> Dtile;
  Dtile.clear();
  // MAP: xn walks K over the token rows; a_off selects each lane's rows.
  const device T* xn = MAP ? x : x + tm * K;
  const threadgroup WT* ws = Ws + tn * BKP;

  dispatch_bool(sgp_sm == kSM, [&](auto kAlignedM) {
    for (int k = 0; k < K_it; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (loads) {
        if (row_live) {
          loader.fetch(k);
          loader.store(Ws);
        } else {
          loader.zero(Ws);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < G::kBK; kk1 += kSK) {
        if (sg_active) {
          if constexpr (MAP) {
            sub_step_map<T, WT, BKP, kAlignedM.value>(
                Dtile, xn + kk1, a_off, ws + kk1, sgp_sm);
          } else {
            sub_step<T, WT, BKP, kAlignedM.value>(
                Dtile, xn + kk1, ws + kk1, K, sgp_sm);
          }
        }
      }
      xn += G::kBK;
    }

    if (!ALIGN_K) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (loads) {
        if (row_live) {
          loader.fetch_tail(K_it, k_remain);
          loader.store_words(Ws, k_remain);
        } else {
          loader.zero(Ws);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < k_remain; kk1 += kSK) {
        if (sg_active) {
          if constexpr (MAP) {
            sub_step_map<T, WT, BKP, kAlignedM.value>(
                Dtile, xn + kk1, a_off, ws + kk1, sgp_sm);
          } else {
            sub_step<T, WT, BKP, kAlignedM.value>(
                Dtile, xn + kk1, ws + kk1, K, sgp_sm);
          }
        }
      }
    }

    if constexpr (kPair) {
      if (sg_active) {
        store_act<T, EPI>(
            Dtile, y + tm * ldy + tn / 2, ldy, int(sgp_sm), limit);
      }
    } else if (kAlignedM.value && sgp_sn == kSN) {
      Dtile.store(y + tm * N + tn, N);
    } else if (sg_active) {
      Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
    }
  });
}

// db: the same tiles and arithmetic with double-buffered 64-deep weight
// tiles: the packed words of step k + 1 are fetched before the tensor ops
// of step k and dequantized into the other buffer after them, so each K
// step has a single barrier. Activation fragments are read straight from
// device memory (rows past the tile are clamped to its last row and never
// stored) and 16-row fragments without rows of the tile are skipped.
// Requires K % 64 == 0 and N % 64 == 0. EPI > 0: the activation epilogue
// of gather_seg.
template <typename T, typename Q, typename G, int EPI = 0, bool MAP = false>
METAL_FUNC void gather_db(
    const device T* x,
    const device uint32_t* rmap,
    const device uint8_t* w,
    thread Q& q,
    const uint4 desc,
    const int y_col,
    device T* y,
    const int N,
    const int K,
    threadgroup typename Q::WT* Ws,
    const uint sgid,
    const uint lane,
    const T limit = T(0)) {
  using WT = typename Q::WT;
  static_assert(G::kBK == 64, "db runs 64-deep K steps");
  constexpr bool kPair = EPI != 0;
  constexpr int BKP = G::kBK + 16 / sizeof(WT);
  constexpr int kTile = kBN * BKP;
  const int row_start = int(desc.x);
  const uint32_t expert = desc.y;
  const int tile_rows = int(desc.z);

  const int K_w = K * Q::kBits / 8;
  const int K_g = K / Q::kGroup;
  const int K_it = K / G::kBK;
  const int half_n = N / 2;
  const int w_col = kPair ? y_col / 2 : y_col;

  const size_t w_row = size_t(expert) * N + w_col;
  q.advance(w_row * K_g);
  TileLoader<Q, G, kPair> loader(
      w + w_row * K_w, K, q, sgid * 32 + lane, half_n);
  const bool loads = G::kLT == G::kThreads || sgid * 32 + lane < uint(G::kLT);

  const int m0 = kSM * int(sgid / kWN);
  const int rows = min(int(kSM), tile_rows - m0);
  // MAP: x holds the token rows; offsets address them through rmap.
  const device T* xs = MAP
      ? x
      : x + size_t(row_start + max(0, min(m0, tile_rows - 1))) * K;

  const short2 sc = BaseNAXFrag::get_coord();
  metal::conditional_t<MAP, uint, int> x_off[kTM][2];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kTM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = min(int(i * 16 + sc.y + h * 8), max(rows, 1) - 1);
      if constexpr (MAP) {
        x_off[i][h] = rmap[row_start + min(m0 + r, tile_rows - 1)] * uint(K) +
            uint(sc.x);
      } else {
        x_off[i][h] = r * K + sc.x;
      }
    }
  }
  const short m_frags = rows > 0 ? short((rows + 15) / 16) : short(0);
  const threadgroup WT* wsg = Ws + (sgid % kWN) * kSN * BKP;

  NAXTile<float, kTM, kTN> D;
  D.clear();

  if (loads) {
    loader.fetch(0);
    loader.store(Ws);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int kb = 0; kb < K_it; kb++) {
    const bool more = kb + 1 < K_it;
    if (more && loads) {
      loader.fetch(kb + 1);
    }
    const threadgroup WT* wb = wsg + (kb & 1) * kTile;
    STEEL_PRAGMA_UNROLL
    for (short kk1 = 0; kk1 < G::kBK; kk1 += kSK) {
      NAXTile<WT, kTN, 2> Btile;
      Btile.template load<WT, BKP, 1>(wb + kk1);
      const int k = kb * G::kBK + kk1;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kTM; i++) {
        if (i < m_frags) {
          NAXTile<T, 1, 2> Atile;
          STEEL_PRAGMA_UNROLL
          for (short h = 0; h < 2; h++) {
            const device T* xp = xs + x_off[i][h] + k;
            const vec<T, 4> a0 = *(const device vec<T, 4>*)(xp);
            const vec<T, 4> a1 = *(const device vec<T, 4>*)(xp + 16);
            STEEL_PRAGMA_UNROLL
            for (short c = 0; c < 4; c++) {
              Atile.frag_at(0, 0)[h * 4 + c] = a0[c];
              Atile.frag_at(0, 1)[h * 4 + c] = a1[c];
            }
          }
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < 2; kk++) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < kTN; j += 2) {
              BaseNAXFrag::mma(
                  D.frag_at(i, j),
                  D.frag_at(i, j + 1),
                  Atile.frag_at(0, kk),
                  metal::bool_constant<false>{},
                  Btile.frag_at(j, kk),
                  Btile.frag_at(j + 1, kk),
                  metal::bool_constant<true>{});
            }
          }
        }
      }
    }
    if (more && loads) {
      loader.store(Ws + ((kb + 1) & 1) * kTile);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if constexpr (kPair) {
    if (rows > 0) {
      store_act<T, EPI>(
          D,
          y + size_t(row_start + m0) * half_n + w_col + (kSN / 2) * (sgid % kWN),
          half_n,
          rows,
          limit);
    }
  } else {
    device T* yb = y + size_t(row_start + m0) * N + y_col + kSN * (sgid % kWN);
    if (rows >= kSM) {
      D.store(yb, N);
    } else if (rows > 0) {
      D.store_safe(yb, N, short2(kSN, short(rows)));
    }
  }
}

// The row tile and output column of this threadgroup. GX == 0: grid
// (columns, tiles) as mlx lays it out. GX > 0: tile t on grid x t % GX and
// (t / GX, column) on y, so all threadgroups of a row tile share one x
// coordinate and a tile's columns run GX threadgroups apart.
template <int GX>
METAL_FUNC bool tile_of(
    const device uint32_t* tiles,
    const uint tile_count,
    const uint3 tid,
    const int N,
    thread uint4& desc,
    thread int& y_col) {
  uint t;
  uint c;
  if constexpr (GX > 0) {
    const uint n_cols = uint((N + kBN - 1) / kBN);
    t = (tid.y / n_cols) * GX + tid.x;
    c = tid.y % n_cols;
  } else {
    t = tid.y;
    c = tid.x;
  }
  if (t >= tile_count) {
    return false;
  }
  desc = *((const device uint4*)tiles + t);
  y_col = int(c) * kBN;
  return true;
}

} // namespace omlx_gqmm
"""

_AFFINE_SOURCE = r"""
    using Q = omlx_gqmm::AffineQ<T, GS, BITS>;
    using G = omlx_gqmm::Geo<BM, BK>;
    using WT = typename Q::WT;
    constexpr int BKP = BK + 16 / sizeof(WT);
    threadgroup WT Ws[(SCHED == 1 ? 2 : 1) * omlx_gqmm::kBN * BKP +
                      PAD / sizeof(WT)];
    uint4 desc;
    int y_col;
    if (!omlx_gqmm::tile_of<GX>(
            tiles, tile_count[0], threadgroup_position_in_grid, params[0],
            desc, y_col)) {
        return;
    }
    Q q{scales, biases};
    if constexpr (SCHED == 1) {
        omlx_gqmm::gather_db<T, Q, G>(
            x, tiles, (const device uint8_t*)w, q, desc, y_col, y, params[0],
            params[1], Ws, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup);
    } else {
        omlx_gqmm::gather_seg<T, Q, G, ALIGN_N, ALIGN_K>(
            x, tiles, (const device uint8_t*)w, q, desc, y_col, y, params[0],
            params[1], Ws, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup);
    }
"""

_ACT_HEADER = r"""
namespace omlx_gqmm {

// The unfused path's activation on the two rounded projections, op for op
// and every intermediate in T like MLX's compiled kernel of
// nn.silu(gate) * up (Sigmoid, Multiply, Multiply). mlx2: only EPI == 1;
// the clamped (EPI == 2) form is not retained and limit is unused.
template <typename T, int EPI>
METAL_FUNC T act(T g, T u, const T limit) {
  static_assert(EPI == 1, "mlx2 retains only silu(gate) * up");
  return Multiply()(Multiply()(g, Sigmoid()(g)), u);
}

// D.frag_at(i, 0) holds gate and D.frag_at(i, 1) up of the same 16 output
// columns (pair_row), so each lane holds both projections of its (row,
// column) elements. Each is rounded to T as the plain store rounds it,
// then act() writes the [rows, 16] block of the [M, ld] output (y points
// at its first element).
template <typename T, int EPI, typename DTile>
METAL_FUNC void store_act(
    thread const DTile& D,
    device T* y,
    const int ld,
    const int rows,
    const T limit) {
  const short2 sc = BaseNAXFrag::get_coord();
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < DTile::kTileRows; i++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < BaseNAXFrag::kElemRows; h++) {
      const int r = i * BaseNAXFrag::kFragRows +
          h * BaseNAXFrag::kElemRowsJump + sc.y;
      if (r < rows) {
        vec<T, BaseNAXFrag::kElemCols> v;
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < BaseNAXFrag::kElemCols; j++) {
          const short e = h * BaseNAXFrag::kElemCols + j;
          v[j] = act<T, EPI>(
              static_cast<T>(D.frag_at(i, 0)[e]),
              static_cast<T>(D.frag_at(i, 1)[e]),
              limit);
        }
        *(device vec<T, BaseNAXFrag::kElemCols>*)(y + size_t(r) * ld + sc.x) =
            v;
      }
    }
  }
}

} // namespace omlx_gqmm
"""

_AFFINE_ACT_MAP_SOURCE = r"""
    using Q = omlx_gqmm::AffineQ<T, GS, BITS>;
    using G = omlx_gqmm::Geo<BM, BK>;
    using WT = typename Q::WT;
    constexpr int BKP = BK + 16 / sizeof(WT);
    threadgroup WT Ws[(SCHED == 1 ? 2 : 1) * omlx_gqmm::kBN * BKP +
                      PAD / sizeof(WT)];
    uint4 desc;
    int y_col;
    if (!omlx_gqmm::tile_of<GX>(
            tiles, tile_count[0], threadgroup_position_in_grid, params[0],
            desc, y_col)) {
        return;
    }
    Q q{scales, biases};
    const T limit = lim[0];
    if constexpr (SCHED == 1) {
        omlx_gqmm::gather_db<T, Q, G, EPI, true>(
            x, rmap, (const device uint8_t*)w, q, desc, y_col, y, params[0],
            params[1], Ws, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup, limit);
    } else {
        omlx_gqmm::gather_seg<T, Q, G, true, ALIGN_K, EPI, true>(
            x, rmap, (const device uint8_t*)w, q, desc, y_col, y, params[0],
            params[1], Ws, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup, limit);
    }
"""
