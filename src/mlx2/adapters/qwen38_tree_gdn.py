"""Sparse parent-restart tree GDN recurrence candidate (Qwen3.8 scalar-gate GDN).

Isolated, default-off, UNQUALIFIED and UNSELECTED. Nothing in the scheduler,
runtime routes or model forward imports this module. Provenance:
provenance/qwen38-tree-gdn-sparse-restart.json (design input ddalcu/mlx-serve
#645 bf6caf5d treeTable/K1TR/K1R; ordinary M1 arithmetic as in the unmerged
local prototype bc0bf635).

A draft tree's rows are walked parents first. A row continues from the
previous row's state when its parent is row t - 1, restarts from the round's
entry state when its parent is -1 (forests included), and otherwise restarts
from a retained state. Only parents that some later row restarts from are
retained, in compact slots. The forward returns readouts plus frozen host
snapshots of the entry state and per-row prework; it never exports per-node
states. The state after an accepted root-to-node path is rebuilt by replaying
that path from the entry.

What exists here:
  * ``plan_tree_restarts``: strict host plan (frozen, digest-bound);
  * ``sparse_tree_forward`` / ``replay_accepted_path``: CPU host executors
    using the ordinary fp32 scalar-gate M1 step with a fixed lane layout and
    ascending-butterfly reduction (the order MLX's own packed GDN kernel
    documents as matching ``simd_sum`` on current Apple GPUs). This is a host
    model, not a statement about Metal bits;
  * ``diagnostic_full_node_reference``: an all-node oracle for tests only;
  * Metal kernel SOURCE definitions plus dispatch entry points that refuse
    unless explicitly opted in on the GPU. They are pending a Metal gate and
    have never been run as part of this candidate.

No register, peak-memory or throughput gain is claimed or inferred.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import threading
import weakref
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple

import mlx.core as mx
import numpy as np

PLAN_SCHEMA = "mlx2.qwen38-tree-gdn-restart-plan.v1"
MAX_WIDTH = 32
HEAD_DIM = 128  # bounded D128 first (Dk = Dv)
MAX_HEADS = 128
LANES = 32
_ACTIVATION_DTYPES = (mx.bfloat16, mx.float32)


class TreeGDNPlanError(ValueError):
    """The parents do not describe a supported parent-first tree."""


class TreeGDNGeometryError(ValueError):
    """Inputs are outside the bounded scalar-gate geometry."""


class TreeGDNUnavailable(RuntimeError):
    """The Metal candidate is not admitted here (CPU, not opted in, or ungated)."""


# ---------------------------------------------------------------- counters

_LOCK = threading.Lock()
_COUNTERS = {"host_plans": 0, "host_sparse_forwards": 0, "host_replays": 0,
             "host_diagnostic_full_node": 0, "device_launches_issued": 0,
             "device_engagement_confirmed": 0}


def _bump(name: str) -> None:
    with _LOCK:
        _COUNTERS[name] += 1


def status(*, reset: bool = False) -> dict:
    """Counters.

    ``device_launches_issued`` counts lazily ISSUED Metal launches (a graph
    node, not an execution). ``device_engaged`` is true only after
    ``confirm_device_engagement`` read a kernel-written engagement flag back
    on the GPU; CPU runs, mocks and lazy issuance never make it true.
    """
    with _LOCK:
        out = dict(_COUNTERS)
        if reset:
            for key in _COUNTERS:
                _COUNTERS[key] = 0
    out["device_engaged"] = out["device_engagement_confirmed"] > 0
    return out


def _require_host_cpu() -> None:
    """Host executors run only with the default CPU device (no silent GPU work)."""
    if mx.default_device() != mx.cpu:
        raise TreeGDNUnavailable("CPU host executors need mx.default_device() == mx.cpu")


# ---------------------------------------------------------------- host plan

def _digest(parents: Tuple[int, ...]) -> str:
    return hashlib.sha256(f"{PLAN_SCHEMA}:{','.join(map(str, parents))}".encode()).hexdigest()


@dataclass(frozen=True)
class TreeRestartPlan:
    """Frozen restart plan; only ``plan_tree_restarts`` builds a valid one.

    ``restart[t]`` is ``"entry"`` (start of a tree in the forest),
    ``"carry"`` (parent is row t - 1) or ``"slot"``; ``slot_of[t]`` is the
    compact slot row t's state is retained in, or -1. ``slots`` lists the
    retained rows in slot order.
    """

    parents: Tuple[int, ...]
    restart: Tuple[str, ...]
    slot_of: Tuple[int, ...]
    restart_slot: Tuple[int, ...]
    slots: Tuple[int, ...]
    has_forest: bool
    digest: str

    @property
    def width(self) -> int:
        return len(self.parents)

    @property
    def n_slots(self) -> int:
        return len(self.slots)


def plan_tree_restarts(parents: Sequence[int]) -> TreeRestartPlan:
    """Validate ``parents`` and derive the compact restart table (a host proposal).

    Refused, never coerced: non-list/tuple input, bool, float, numpy scalars
    or any other non-``int`` entry, width outside 1..32, row 0 not a root, a
    parent below -1 or not strictly earlier than its row.
    """
    plan = _derive_plan(parents)
    _bump("host_plans")
    return plan


def _derive_plan(parents: Sequence[int]) -> TreeRestartPlan:
    """Pure derivation: no counters, no global state."""
    if not isinstance(parents, (list, tuple)):
        raise TreeGDNPlanError("parents must be a list or tuple of int")
    parents = tuple(parents)
    if not 1 <= len(parents) <= MAX_WIDTH:
        raise TreeGDNPlanError(f"tree width must be 1..{MAX_WIDTH}")
    for row, parent in enumerate(parents):
        if type(parent) is not int:
            raise TreeGDNPlanError(f"row {row}: parent {parent!r} is not a plain int")
        if parent != -1 and not 0 <= parent < row:
            raise TreeGDNPlanError(f"row {row}: parent {parent} must be -1 or in 0..{row - 1}")
    restart, restart_parent = [], []
    retained = set()
    for row, parent in enumerate(parents):
        if parent == -1:
            restart.append("entry")
        elif parent == row - 1:
            restart.append("carry")
        else:
            restart.append("slot")
            retained.add(parent)
        restart_parent.append(parent)
    slots = tuple(sorted(retained))
    index = {row: i for i, row in enumerate(slots)}
    slot_of = tuple(index.get(row, -1) for row in range(len(parents)))
    restart_slot = tuple(index[p] if kind == "slot" else -1 for kind, p in zip(restart, restart_parent))
    return TreeRestartPlan(parents=parents, restart=tuple(restart), slot_of=slot_of, restart_slot=restart_slot,
                           slots=slots, has_forest=any(p == -1 for p in parents[1:]), digest=_digest(parents))


def verified_plan(plan: TreeRestartPlan) -> TreeRestartPlan:
    """Re-derive ``plan`` from its parents; refuse anything that differs."""
    if type(plan) is not TreeRestartPlan:
        raise TreeGDNPlanError("not a TreeRestartPlan")
    fresh = _derive_plan(plan.parents)  # pure: counts nothing, restores nothing
    if fresh != plan:
        raise TreeGDNPlanError("plan does not match the table derived from its parents")
    return fresh


def accepted_path_rows(plan: TreeRestartPlan, path: Sequence[int]) -> Tuple[int, ...]:
    """Validate a root-to-node path (oldest first) through ``plan``."""
    if not isinstance(path, (list, tuple)) or not path:
        raise TreeGDNPlanError("accepted path must be a non-empty list or tuple")
    rows = tuple(path)
    for i, row in enumerate(rows):
        if type(row) is not int or not 0 <= row < plan.width:
            raise TreeGDNPlanError(f"path entry {row!r} is not a row of this tree")
        expected = -1 if i == 0 else rows[i - 1]
        if plan.parents[row] != expected:
            raise TreeGDNPlanError(f"path is not a root-to-node chain at row {row}")
    return rows


# ---------------------------------------------------------------- geometry

@dataclass(frozen=True)
class TreeGeometry:
    width: int
    hk: int
    hv: int
    dk: int
    dv: int
    activation: Any


def check_geometry(q, k, v, g, beta, state, plan: TreeRestartPlan) -> TreeGeometry:
    """Bounded scalar-gate geometry: B1, D128, fp32 gates/state, shared q/k/v dtype."""
    arrays = (q, k, v, g, beta, state)
    if not all(isinstance(x, mx.array) for x in arrays):
        raise TreeGDNGeometryError("q, k, v, g, beta and state must be mx.array")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise TreeGDNGeometryError("q and k must be [1, W, Hk, Dk] and v [1, W, Hv, Dv]")
    batch, width, hk, dk = map(int, q.shape)
    if batch != 1 or int(v.shape[0]) != 1:
        raise TreeGDNGeometryError("only B1 is supported")
    if width != plan.width or int(v.shape[1]) != width:
        raise TreeGDNGeometryError("row count differs from the plan")
    hv, dv = int(v.shape[2]), int(v.shape[3])
    if dk != HEAD_DIM or dv != HEAD_DIM:
        raise TreeGDNGeometryError(f"Dk and Dv must both be {HEAD_DIM}")
    if not 1 <= hk <= hv <= MAX_HEADS or hv % hk:
        raise TreeGDNGeometryError("head geometry must satisfy 1 <= Hk <= Hv <= 128 and Hv % Hk == 0")
    if q.dtype not in _ACTIVATION_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TreeGDNGeometryError("q, k and v must share bf16 or fp32")
    if g.shape != (1, width, hv) or beta.shape != g.shape:
        raise TreeGDNGeometryError("only scalar gates [1, W, Hv] are supported")
    if g.dtype != mx.float32 or beta.dtype != mx.float32:
        raise TreeGDNGeometryError("g and beta must be fp32")
    if state.shape != (1, hv, dv, dk) or state.dtype != mx.float32:
        raise TreeGDNGeometryError("state must be fp32 [1, Hv, Dv, Dk] (no per-token state cast)")
    return TreeGeometry(width, hk, hv, dk, dv, q.dtype)


# ---------------------------------------------------------------- host arithmetic

def _host(x) -> np.ndarray:
    out = np.array(x.astype(mx.float32), dtype=np.float32, copy=True)
    out.setflags(write=False)
    return out


def _butterfly(x: np.ndarray) -> np.ndarray:
    """Ascending xor butterfly over the trailing 32-lane axis; returns lane 0."""
    lanes = np.arange(LANES)
    for mask in (1, 2, 4, 8, 16):
        x = x + x[..., lanes ^ mask]
    return x[..., 0]


def _lane_sum(products: np.ndarray) -> np.ndarray:
    """Each lane accumulates its N contiguous products in order, then reduce."""
    acc = np.zeros(products.shape[:-1], dtype=np.float32)
    for i in range(products.shape[-1]):
        acc = acc + products[..., i]
    return _butterfly(acc)


def _step(state: np.ndarray, q: np.ndarray, k: np.ndarray, v: np.ndarray, g: np.ndarray, beta: np.ndarray,
          group: int):
    """One ordinary fp32 scalar-gate M1 step for every (Hv, Dv) row.

    state [Hv, Dv, Dk]; q/k [Hk, Dk]; v [Hv, Dv]; g/beta [Hv]. Lane ``l`` owns
    Dk elements ``N*l .. N*l+N-1`` (N = Dk / 32), as in the ordinary kernel.
    """
    hv, dv, dk = state.shape
    n = dk // LANES
    kh = np.repeat(k, group, axis=0).reshape(hv, 1, LANES, n)
    qh = np.repeat(q, group, axis=0).reshape(hv, 1, LANES, n)
    s = state.reshape(hv, dv, LANES, n) * g.reshape(hv, 1, 1, 1)
    kv = _lane_sum(s * kh)
    delta = (v - kv) * beta.reshape(hv, 1)
    s = s + kh * delta[..., None, None]
    out = _lane_sum(s * qh)
    return s.reshape(hv, dv, dk), out


@dataclass(frozen=True)
class _Prework:
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    g: np.ndarray
    beta: np.ndarray
    entry: np.ndarray
    digest: str

    @staticmethod
    def _hash(parts) -> str:
        h = hashlib.sha256()
        for part in parts:
            h.update(f"{part.dtype.str}{part.shape}".encode())  # shape and dtype, not bytes alone
            h.update(part.tobytes())
        return h.hexdigest()

    @staticmethod
    def capture(q, k, v, g, beta, state) -> "_Prework":
        parts = [_host(x)[0] for x in (q, k, v, g, beta, state)]
        return _Prework(*parts, digest=_Prework._hash(parts))

    def check(self) -> None:
        if self._hash((self.q, self.k, self.v, self.g, self.beta, self.entry)) != self.digest:
            raise TreeGDNPlanError("retained prework changed after the forward")

    def row(self, row: int, state: np.ndarray, group: int):
        return _step(state, self.q[row], self.k[row], self.v[row], self.g[row], self.beta[row], group)


RECORD_SCHEMA = "mlx2.qwen38-tree-gdn-forward-record.v1"
_BINDING_KEY = os.urandom(32)       # process-private; a record cannot be re-signed outside
_ISSUED: "weakref.WeakSet[TreeForwardResult]" = weakref.WeakSet()


def _record_binding(plan: TreeRestartPlan, geometry: TreeGeometry, prework: _Prework) -> str:
    message = "|".join((RECORD_SCHEMA, plan.digest, repr(plan.parents), str(geometry.width), str(geometry.hk),
                        str(geometry.hv), str(geometry.dk), str(geometry.dv), str(geometry.activation),
                        prework.digest))
    return hmac.new(_BINDING_KEY, message.encode(), hashlib.sha256).hexdigest()


@dataclass(frozen=True, eq=False)
class TreeForwardResult:
    """Readouts plus what an accepted-path replay needs; no per-node states.

    Bound at issue: only the exact object ``sparse_tree_forward`` returned is
    replayable (copies, ``dataclasses.replace`` and retargeted records are
    refused), and its plan, geometry, dtypes, schema and prework digest must
    still match a keyed binding taken at issue.
    """

    plan: TreeRestartPlan
    geometry: TreeGeometry
    y: mx.array                       # [1, W, Hv, Dv], activation dtype
    receipt: Mapping[str, Any]
    _prework: _Prework = field(repr=False)
    _binding: str = field(repr=False)


def _check_record(result) -> None:
    if type(result) is not TreeForwardResult or result not in _ISSUED:
        raise TreeGDNPlanError("replay needs the exact record issued by sparse_tree_forward")
    binding = getattr(result, "_binding", None)
    if not isinstance(binding, str) or type(result._prework) is not _Prework:
        raise TreeGDNPlanError("forward record has no binding")
    result._prework.check()
    expected = _record_binding(result.plan, result.geometry, result._prework)
    if not hmac.compare_digest(binding, expected):
        raise TreeGDNPlanError("forward record plan, geometry or prework no longer matches its binding")


def _readout(rows: list, geometry: TreeGeometry) -> mx.array:
    y = np.stack(rows)[None]  # [1, W, Hv, Dv] float32
    return mx.array(y).astype(geometry.activation)


def sparse_tree_forward(q, k, v, g, beta, state, parents: Sequence[int]) -> TreeForwardResult:
    """CPU host executor of the sparse restart plan (candidate arithmetic).

    Holds the current state and one state per compact slot during traversal.
    The host record always snapshots the entry for later path replay; extra
    entry register storage in the Metal source is needed only for forests.
    Inputs are read, never mutated.
    """
    _require_host_cpu()
    plan = plan_tree_restarts(parents)
    geometry = check_geometry(q, k, v, g, beta, state, plan)
    work = _Prework.capture(q, k, v, g, beta, state)
    group = geometry.hv // geometry.hk
    kept = [None] * plan.n_slots
    current, rows, live_peak = work.entry, [], 0
    for row in range(plan.width):
        kind = plan.restart[row]
        if kind == "entry":
            current = work.entry
        elif kind == "slot":
            current = kept[plan.restart_slot[row]]
        current, out = work.row(row, current, group)
        rows.append(out)
        if plan.slot_of[row] >= 0:
            kept[plan.slot_of[row]] = current
        live_peak = max(live_peak, sum(s is not None for s in kept))
    _bump("host_sparse_forwards")
    receipt = MappingProxyType({
        "executor": "cpu_host_reference", "plan_digest": plan.digest, "width": plan.width,
        "retained_slots": plan.n_slots, "retained_slot_rows": list(plan.slots),
        "entry_retained_for_forest": plan.has_forest, "live_slot_peak": live_peak,
        "per_node_states_exported": False, "host_proposal": True,
        "device_launches_issued": 0, "device_engaged": False,
        "note": ("host model only (ascending-butterfly reduction, not a Metal simd_sum bit claim); "
                 "no register, memory or throughput claim"),
    })
    result = TreeForwardResult(plan, geometry, _readout(rows, geometry), receipt, work,
                               _record_binding(plan, geometry, work))
    _ISSUED.add(result)
    return result


def replay_accepted_path(result: TreeForwardResult, path: Sequence[int]) -> mx.array:
    """fp32 state [1, Hv, Dv, Dk] after the accepted root-to-node ``path``.

    Replays the path from the forward's retained entry with its retained
    prework; the path may branch away from row order, but it must be a chain
    of parent links from a root (row-order prefixes that are not such a
    chain, and non-ancestor sequences, are refused). Touches no cache.
    """
    _require_host_cpu()
    _check_record(result)
    plan = verified_plan(result.plan)
    rows = accepted_path_rows(plan, path)
    group = result.geometry.hv // result.geometry.hk
    current = result._prework.entry
    for row in rows:
        current, _ = result._prework.row(row, current, group)
    _bump("host_replays")
    return mx.array(current[None])


def diagnostic_full_node_reference(q, k, v, g, beta, state, parents: Sequence[int]):
    """Test oracle only: readouts and EVERY node's state (what the candidate avoids)."""
    _require_host_cpu()
    plan = plan_tree_restarts(parents)
    geometry = check_geometry(q, k, v, g, beta, state, plan)
    work = _Prework.capture(q, k, v, g, beta, state)
    group = geometry.hv // geometry.hk
    states, rows = [], []
    for row, parent in enumerate(plan.parents):
        prior = work.entry if parent == -1 else states[parent]
        new, out = work.row(row, prior, group)
        states.append(new)
        rows.append(out)
    _bump("host_diagnostic_full_node")
    return _readout(rows, geometry), [mx.array(s[None]) for s in states]


def ordinary_path_reference(q, k, v, g, beta, state, path: Sequence[int]) -> mx.array:
    """Ordinary per-row chain recurrence along ``path`` rows from ``state``."""
    _require_host_cpu()
    work = _Prework.capture(q, k, v, g, beta, state)
    group = int(v.shape[2]) // int(q.shape[2])
    current = work.entry
    for row in path:
        current, _ = work.row(row, current, group)
    return mx.array(current[None])


# ---------------------------------------------------------------- Metal (pending gate)
#
# Low-level, UNBOUND experiments: the Metal replay is not bound to a forward
# record, cache revision or commit and cannot qualify a cache commit. A future
# GPU record must be an MLX-owned immutable snapshot (no silent full host
# copy), and must fail closed on a missing replay or capability. Each kernel
# writes an ``engaged`` flag; only ``confirm_device_engagement`` (GPU readback)
# may count engagement. Never built or dispatched by the CPU tests.

# Forward: readouts only. Same per-row arithmetic as the prototype's research
# kernel; restart from compact slots (KSLOTS = max(1, retained)), entry kept
# only for forests. ``table`` = parents[W] ++ restart_slot[W] ++ slot_of[W].
SPARSE_FORWARD_SOURCE = r"""
    const int hv = thread_position_in_grid.z;
    const int hk = hv / (Hv / Hk);
    const int dv = thread_position_in_grid.y;
    const int dk = thread_position_in_threadgroup.x;
    constexpr int N = Dk / 32;
    const auto source = state_in + (hv * Dv + dv) * Dk;
    float s[N];
    float entry[FOREST ? N : 1];
    float kept[KSLOTS][N];
    for (int i = 0; i < N; ++i) s[i] = static_cast<float>(source[N * dk + i]);
    if (FOREST) for (int i = 0; i < N; ++i) entry[i] = s[i];
    for (int row = 0; row < W; ++row) {
        const int parent = table[row];
        if (row > 0 && parent < 0) {
            for (int i = 0; i < N; ++i) s[i] = entry[i];
        } else if (row > 0 && parent != row - 1) {
            const int slot = table[W + row];
            for (int i = 0; i < N; ++i) s[i] = kept[slot][i];
        }
        const auto qr = q + (row * Hk + hk) * Dk;
        const auto kr = k + (row * Hk + hk) * Dk;
        const auto vr = v + (row * Hv + hv) * Dv;
        const float decay = static_cast<float>(g[row * Hv + hv]);
        const float rate = static_cast<float>(beta[row * Hv + hv]);
        float kv = 0.0f;
        for (int i = 0; i < N; ++i) {
            const int j = N * dk + i;
            s[i] = s[i] * decay;
            kv += s[i] * kr[j];
        }
        kv = simd_sum(kv);
        const float delta = (vr[dv] - kv) * rate;
        float value = 0.0f;
        for (int i = 0; i < N; ++i) {
            const int j = N * dk + i;
            s[i] = s[i] + kr[j] * delta;
            value += s[i] * qr[j];
        }
        value = simd_sum(value);
        if (thread_index_in_simdgroup == 0)
            y[(row * Hv + hv) * Dv + dv] = static_cast<InT>(value);
        const int own = table[2 * W + row];
        if (own >= 0) for (int i = 0; i < N; ++i) kept[own][i] = s[i];
    }
    if (hv == 0 && dv == 0 && dk == 0) engaged[0] = 1u;
"""

# Replay: the state after ``path`` (oldest first) from the entry; fp32 state.
REPLAY_SOURCE = r"""
    const int hv = thread_position_in_grid.z;
    const int hk = hv / (Hv / Hk);
    const int dv = thread_position_in_grid.y;
    const int dk = thread_position_in_threadgroup.x;
    constexpr int N = Dk / 32;
    const auto source = state_in + (hv * Dv + dv) * Dk;
    float s[N];
    for (int i = 0; i < N; ++i) s[i] = static_cast<float>(source[N * dk + i]);
    for (int p = 0; p < P; ++p) {
        const int row = path[p];
        const auto kr = k + (row * Hk + hk) * Dk;
        const auto vr = v + (row * Hv + hv) * Dv;
        const float decay = static_cast<float>(g[row * Hv + hv]);
        const float rate = static_cast<float>(beta[row * Hv + hv]);
        float kv = 0.0f;
        for (int i = 0; i < N; ++i) {
            const int j = N * dk + i;
            s[i] = s[i] * decay;
            kv += s[i] * kr[j];
        }
        kv = simd_sum(kv);
        const float delta = (vr[dv] - kv) * rate;
        for (int i = 0; i < N; ++i) s[i] = s[i] + kr[N * dk + i] * delta;
    }
    for (int i = 0; i < N; ++i) state_out[(hv * Dv + dv) * Dk + N * dk + i] = s[i];
    if (hv == 0 && dv == 0 && dk == 0) engaged[0] = 1u;
"""

_KERNELS: dict = {}


def _metal_kernel(name: str):
    """Lazily construct one isolated kernel object (never at import)."""
    if name not in _KERNELS:
        if name == "forward":
            _KERNELS[name] = mx.fast.metal_kernel(
                name="mlx2_qwen38_tree_gdn_sparse_forward_v1",
                input_names=["q", "k", "v", "g", "beta", "state_in", "table"],
                output_names=["y", "engaged"], source=SPARSE_FORWARD_SOURCE)
        else:
            _KERNELS[name] = mx.fast.metal_kernel(
                name="mlx2_qwen38_tree_gdn_path_replay_v1",
                input_names=["k", "v", "g", "beta", "state_in", "path"],
                output_names=["state_out", "engaged"], source=REPLAY_SOURCE)
    return _KERNELS[name]


def _admit_metal(allow_unqualified_metal: bool) -> None:
    if allow_unqualified_metal is not True:
        raise TreeGDNUnavailable("Metal tree GDN is unqualified; pass allow_unqualified_metal=True to test it")
    if mx.default_device() != mx.gpu:
        raise TreeGDNUnavailable("Metal tree GDN needs the default GPU device")
    if not mx.metal.is_available():
        raise TreeGDNUnavailable("Metal is unavailable")


# id(engaged) -> engaged, held WEAKLY: telemetry never owns a launch's graph;
# a flag the caller drops simply leaves (and was never engagement).
_PENDING_FLAGS: "weakref.WeakValueDictionary[int, mx.array]" = weakref.WeakValueDictionary()


def confirm_device_engagement(engaged) -> bool:
    """Count engagement only for a flag this module issued, read back as 1 on the GPU.

    CPU default device, foreign arrays and unwritten (0) flags are never
    engagement. A lazily issued launch is not engagement until this runs.
    """
    if mx.default_device() != mx.gpu or _PENDING_FLAGS.get(id(engaged)) is not engaged:
        return False
    _PENDING_FLAGS.pop(id(engaged), None)
    if engaged.dtype != mx.uint32 or engaged.shape != (1,) or int(engaged.item()) != 1:
        return False
    _bump("device_engagement_confirmed")
    return True


def _issue(name: str, call):
    """Issue one lazy launch; only kernels from this module's builder cache can
    later confirm engagement (a substituted or mocked kernel never can)."""
    kernel = _metal_kernel(name)
    out, engaged = call(kernel)
    if _KERNELS.get(name) is kernel:
        _PENDING_FLAGS[id(engaged)] = engaged
    _bump("device_launches_issued")
    return out, engaged


def metal_kernel_table(plan: TreeRestartPlan) -> Tuple[int, ...]:
    """``parents ++ restart_slot ++ slot_of`` from a verified plan (host ints)."""
    plan = verified_plan(plan)
    return plan.parents + plan.restart_slot + plan.slot_of


def metal_sparse_tree_forward(q, k, v, g, beta, state, parents: Sequence[int], *,
                              allow_unqualified_metal: bool = False):
    """UNQUALIFIED, unbound Metal forward: ``(y, engaged)``; pending a native gate."""
    _admit_metal(allow_unqualified_metal)
    plan = plan_tree_restarts(parents)
    geometry = check_geometry(q, k, v, g, beta, state, plan)
    return _issue("forward", lambda kernel: kernel(
        inputs=[mx.contiguous(x) for x in (q, k, v, g, beta, state)]
               + [mx.array(metal_kernel_table(plan), dtype=mx.int32)],
        template=[("InT", geometry.activation), ("Dk", geometry.dk), ("Dv", geometry.dv),
                  ("Hk", geometry.hk), ("Hv", geometry.hv), ("W", plan.width),
                  ("KSLOTS", max(1, plan.n_slots)), ("FOREST", int(plan.has_forest))],
        grid=(32, geometry.dv, geometry.hv), threadgroup=(32, 4, 1),
        output_shapes=[(1, plan.width, geometry.hv, geometry.dv), (1,)],
        output_dtypes=[geometry.activation, mx.uint32], init_value=0))


def metal_replay_accepted_path(k, v, g, beta, state, parents: Sequence[int], path: Sequence[int], *,
                               allow_unqualified_metal: bool = False):
    """UNQUALIFIED, unbound Metal path replay: ``(state, engaged)``; not a cache commit."""
    _admit_metal(allow_unqualified_metal)
    plan = plan_tree_restarts(parents)
    check_geometry(k, k, v, g, beta, state, plan)
    rows = accepted_path_rows(plan, path)
    hk, hv, dk, dv = int(k.shape[2]), int(v.shape[2]), int(k.shape[3]), int(v.shape[3])
    return _issue("replay", lambda kernel: kernel(
        inputs=[mx.contiguous(x) for x in (k, v, g, beta, state)] + [mx.array(rows, dtype=mx.int32)],
        template=[("Dk", dk), ("Dv", dv), ("Hk", hk), ("Hv", hv), ("P", len(rows))],
        grid=(32, dv, hv), threadgroup=(32, 4, 1),
        output_shapes=[(1, hv, dv, dk), (1,)], output_dtypes=[mx.float32, mx.uint32], init_value=0))
