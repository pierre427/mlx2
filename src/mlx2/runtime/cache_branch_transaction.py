# SPDX-License-Identifier: Apache-2.0
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from __future__ import annotations
import os
import threading
import uuid
import weakref
import dataclasses
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Protocol
from .cache_planes import CachePlaneKind


class CacheBranchTransactionError(RuntimeError):
    """A branch, generation, layout, or boundary contract was violated."""


class BranchStatus(str, Enum):
    ACTIVE = "active"
    PROMOTED = "promoted"
    REJECTED = "rejected"


class ExactCompactionAdapter(Protocol):
    """Capability-attested adapter that never mutates source payloads."""

    source_payloads_read_only: bool
    destinations_are_fresh: bool

    def materialize(
        self, base: "PlaneBase", deltas: tuple["PlaneDelta", ...]
    ) -> "PlaneBase": ...


def cache_delta_promotion_enabled(value: bool | None = None) -> bool:
    if value is not None:
        return bool(value)
    return os.environ.get("MLX_LM_CACHE_DELTA_PROMOTION", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class PlaneBase:
    kind: CachePlaneKind
    payload: Any
    start: int
    length: int
    layout_id: str
    generation: int
    logical_bytes: int = 0
    payload_is_immutable: bool = False
    ring_capacity: int | None = None

    def __post_init__(self) -> None:
        if min(self.start, self.length, self.generation, self.logical_bytes) < 0:
            raise ValueError(
                "base coordinates, generation, and bytes must be non-negative"
            )
        if not self.layout_id:
            raise ValueError("base layout_id is required")
        if not self.payload_is_immutable:
            raise ValueError("base payload must be declared immutable")
        if self.ring_capacity is not None and self.ring_capacity < 1:
            raise ValueError("ring capacity must be positive")
        if self.kind == CachePlaneKind.ATTENTION_RING:
            if self.ring_capacity is None:
                raise ValueError("attention_ring base requires ring_capacity")
        elif self.ring_capacity is not None:
            raise ValueError("ring_capacity is only valid for attention_ring")

    @property
    def stop(self) -> int:
        return self.start + self.length


@dataclass(frozen=True)
class PlaneDelta:
    """One plane's immutable result at a shared logical boundary."""

    kind: CachePlaneKind
    payload: Any
    start: int
    length: int
    layout_id: str
    generation: int
    logical_bytes: int = 0
    payload_is_immutable: bool = False
    ring_capacity: int | None = None
    physical_start: int | None = None
    physical_stop: int | None = None
    wrap_epoch: int | None = None

    def __post_init__(self) -> None:
        if min(self.start, self.length, self.generation, self.logical_bytes) < 0:
            raise ValueError(
                "delta coordinates, generation, and bytes must be non-negative"
            )
        if self.length < 1:
            raise ValueError("an aligned delta must advance at least one token")
        if not self.layout_id:
            raise ValueError("delta layout_id is required")
        if not self.payload_is_immutable:
            raise ValueError("delta payload must be declared immutable")
        ring_fields = (self.physical_start, self.physical_stop, self.wrap_epoch)
        if self.kind == CachePlaneKind.ATTENTION_RING:
            if self.ring_capacity is None:
                raise ValueError("attention_ring delta requires ring_capacity")
        elif self.ring_capacity is not None:
            raise ValueError("ring_capacity is only valid for attention_ring")
        if self.ring_capacity is None:
            if any((value is not None for value in ring_fields)):
                raise ValueError("ring coordinates require ring_capacity")
            return
        if self.ring_capacity < 1:
            raise ValueError("ring capacity must be positive")
        if any((value is None for value in ring_fields)):
            raise ValueError("ring deltas require physical start/stop and wrap epoch")
        if self.physical_start != self.start % self.ring_capacity:
            raise ValueError("ring physical_start disagrees with logical start")
        if self.physical_stop != self.stop % self.ring_capacity:
            raise ValueError("ring physical_stop disagrees with logical stop")
        if self.wrap_epoch != self.stop // self.ring_capacity:
            raise ValueError("ring wrap_epoch disagrees with logical stop")

    @property
    def stop(self) -> int:
        return self.start + self.length


@dataclass(frozen=True)
class AlignedDeltaCheckpoint:
    checkpoint_id: str
    position: int
    deltas: tuple[PlaneDelta, ...]
    previous: "AlignedDeltaCheckpoint | None"
    depth: int
    cumulative_bytes: int


@dataclass(frozen=True)
class DeltaChainView:
    lineage_id: str
    generation: int
    bases: tuple[PlaneBase, ...]
    tip: AlignedDeltaCheckpoint


@dataclass(frozen=True)
class PromotionReceipt:
    lineage_id: str
    source_generation: int
    successor_generation: int
    accepted_position: int
    accepted_delta_nodes: int
    abandoned_delta_nodes: int
    abandoned_bytes: int
    pointer_swaps: int = 1
    copied_bytes: int = 0


@dataclass(frozen=True)
class RejectionReceipt:
    lineage_id: str
    generation: int
    abandoned_delta_nodes: int
    abandoned_bytes: int
    pointer_swaps: int = 0
    copied_bytes: int = 0


@dataclass(frozen=True)
class CompactionPlan:
    plan_id: str
    lineage_id: str
    expected_generation: int
    expected_tip_id: str
    expected_arena_version: int
    position: int
    bases: tuple[PlaneBase, ...]
    plane_deltas: tuple[tuple[CachePlaneKind, tuple[PlaneDelta, ...]], ...]


@dataclass(frozen=True)
class CompactionReceipt:
    lineage_id: str
    source_generation: int
    successor_generation: int
    position: int
    delta_nodes_compacted: int
    source_delta_bytes: int
    replacement_bytes: int


class CacheDeltaBranch:
    """Request-private append-only delta chain from one lineage tip."""

    def __init__(
        self,
        lineage: "CacheDeltaLineage",
        *,
        owner_id: str,
        generation: int,
        origin: AlignedDeltaCheckpoint,
    ) -> None:
        self.branch_id = uuid.uuid4().hex
        self.owner_id = owner_id
        self.generation = int(generation)
        self._lineage = lineage
        self._origin = origin
        self._tip = origin
        self._status = BranchStatus.ACTIVE

    @property
    def status(self) -> BranchStatus:
        with self._lineage._lock:
            return self._status

    @property
    def position(self) -> int:
        with self._lineage._lock:
            if self._status != BranchStatus.ACTIVE:
                raise CacheBranchTransactionError(
                    f"branch is already {self._status.value}"
                )
            return self._tip.position

    @property
    def delta_depth(self) -> int:
        with self._lineage._lock:
            if self._status != BranchStatus.ACTIVE:
                raise CacheBranchTransactionError(
                    f"branch is already {self._status.value}"
                )
            return self._tip.depth - self._origin.depth

    def append_checkpoint(
        self, deltas: Mapping[CachePlaneKind, PlaneDelta] | Iterable[PlaneDelta]
    ) -> AlignedDeltaCheckpoint:
        return self._lineage._append(self, deltas)

    def promote(self, accepted_position: int) -> PromotionReceipt:
        return self._lineage.promote(self, accepted_position)

    def reject(self) -> RejectionReceipt:
        return self._lineage.reject(self)

    def close(self) -> RejectionReceipt | None:
        return self._lineage._close_branch(self)

    def __enter__(self) -> "CacheDeltaBranch":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class CacheDeltaLineage:
    """One CAS tip shared by competing aligned speculative branches."""

    def __init__(self, bases: Iterable[PlaneBase]) -> None:
        bases = tuple(sorted(bases, key=lambda item: item.kind.value))
        if not bases:
            raise CacheBranchTransactionError("a cache-delta lineage needs planes")
        if len({base.kind for base in bases}) != len(bases):
            raise CacheBranchTransactionError("duplicate base cache plane")
        positions = {base.stop for base in bases}
        generations = {base.generation for base in bases}
        if len(positions) != 1:
            raise CacheBranchTransactionError("base cache planes are not aligned")
        if len(generations) != 1:
            raise CacheBranchTransactionError("base cache generations disagree")
        self.lineage_id = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._bases = bases
        self._layouts = {base.kind: base.layout_id for base in bases}
        self._ring_capacities = {base.kind: base.ring_capacity for base in bases}
        self._generation = generations.pop()
        self._closed = False
        position = positions.pop()
        self._tip = AlignedDeltaCheckpoint(uuid.uuid4().hex, position, (), None, 0, 0)
        self._arena: dict[str, AlignedDeltaCheckpoint] = {
            self._tip.checkpoint_id: self._tip
        }
        self._checkpoint_index: dict[tuple[str, int], AlignedDeltaCheckpoint] = {}
        self._arena_delta_bytes = 0
        self._arena_version = 0
        self._issued_compactions: dict[str, tuple[Any, ...]] = {}
        self._branches: weakref.WeakSet[CacheDeltaBranch] = weakref.WeakSet()
        self._counters = {
            "forks": 0,
            "delta_checkpoints": 0,
            "delta_segments": 0,
            "delta_bytes": 0,
            "promotions": 0,
            "promotion_pointer_swaps": 0,
            "promotion_copied_bytes": 0,
            "rejections": 0,
            "abandoned_delta_nodes": 0,
            "abandoned_bytes": 0,
            "compaction_plans": 0,
            "compaction_cancellations": 0,
            "compaction_retired_empty_branches": 0,
            "compactions": 0,
            "compaction_failures": 0,
            "compacted_delta_nodes": 0,
            "compacted_source_bytes": 0,
            "compacted_replacement_bytes": 0,
            "rebases": 0,
            "rebased_delta_nodes": 0,
        }

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def position(self) -> int:
        with self._lock:
            self._assert_open_locked()
            return self._tip.position

    def _assert_active_current(self, branch: CacheDeltaBranch) -> None:
        self._assert_open_locked()
        if branch._lineage is not self:
            raise CacheBranchTransactionError("branch belongs to another lineage")
        if branch._status != BranchStatus.ACTIVE:
            raise CacheBranchTransactionError(
                f"branch is already {branch._status.value}"
            )
        if branch.generation != self._generation or branch._origin is not self._tip:
            raise CacheBranchTransactionError("branch generation is stale")

    def _assert_open_locked(self) -> None:
        if self._closed:
            raise CacheBranchTransactionError("cache-delta lineage is closed")

    def fork(self, *, owner_id: str) -> CacheDeltaBranch:
        if not owner_id:
            raise CacheBranchTransactionError("branch owner_id is required")
        with self._lock:
            self._assert_open_locked()
            branch = CacheDeltaBranch(
                self, owner_id=owner_id, generation=self._generation, origin=self._tip
            )
            self._branches.add(branch)
            self._counters["forks"] += 1
            return branch

    def _normalize_deltas(
        self, deltas: Mapping[CachePlaneKind, PlaneDelta] | Iterable[PlaneDelta]
    ) -> tuple[PlaneDelta, ...]:
        if isinstance(deltas, Mapping):
            values = tuple(deltas.values())
            if any((kind != delta.kind for (kind, delta) in deltas.items())):
                raise CacheBranchTransactionError("delta map key and kind disagree")
        else:
            values = tuple(deltas)
        values = tuple(sorted(values, key=lambda item: item.kind.value))
        if len({delta.kind for delta in values}) != len(values):
            raise CacheBranchTransactionError("duplicate delta cache plane")
        if {delta.kind for delta in values} != set(self._layouts):
            raise CacheBranchTransactionError(
                "an aligned checkpoint must cover every lineage plane exactly once"
            )
        return values

    def _append(
        self,
        branch: CacheDeltaBranch,
        deltas: Mapping[CachePlaneKind, PlaneDelta] | Iterable[PlaneDelta],
    ) -> AlignedDeltaCheckpoint:
        with self._lock:
            self._assert_active_current(branch)
            values = self._normalize_deltas(deltas)
            starts = {delta.start for delta in values}
            stops = {delta.stop for delta in values}
            generations = {delta.generation for delta in values}
            if starts != {branch._tip.position} or len(stops) != 1:
                raise CacheBranchTransactionError(
                    "delta cache planes do not share one contiguous boundary"
                )
            if generations != {branch.generation}:
                raise CacheBranchTransactionError("delta generation is stale")
            for delta in values:
                if delta.layout_id != self._layouts[delta.kind]:
                    raise CacheBranchTransactionError(
                        f"{delta.kind.value} delta layout changed"
                    )
                if delta.ring_capacity != self._ring_capacities[delta.kind]:
                    raise CacheBranchTransactionError(
                        f"{delta.kind.value} ring capacity changed"
                    )
            checkpoint = AlignedDeltaCheckpoint(
                checkpoint_id=uuid.uuid4().hex,
                position=next(iter(stops)),
                deltas=values,
                previous=branch._tip,
                depth=branch._tip.depth + 1,
                cumulative_bytes=branch._tip.cumulative_bytes
                + sum((delta.logical_bytes for delta in values)),
            )
            branch._tip = checkpoint
            self._arena[checkpoint.checkpoint_id] = checkpoint
            self._checkpoint_index[branch.branch_id, checkpoint.position] = checkpoint
            self._arena_version += 1
            self._arena_delta_bytes += sum((delta.logical_bytes for delta in values))
            self._counters["delta_checkpoints"] += 1
            self._counters["delta_segments"] += len(values)
            self._counters["delta_bytes"] += sum(
                (delta.logical_bytes for delta in values)
            )
            return checkpoint

    def promote(
        self, branch: CacheDeltaBranch, accepted_position: int
    ) -> PromotionReceipt:
        with self._lock:
            self._assert_active_current(branch)
            accepted_position = int(accepted_position)
            if accepted_position == branch._origin.position:
                raise CacheBranchTransactionError(
                    "zero acceptance must use cheap branch rejection"
                )
            checkpoint = self._checkpoint_index.get(
                (branch.branch_id, accepted_position)
            )
            if checkpoint is None:
                raise CacheBranchTransactionError(
                    "accepted position is not an aligned delta checkpoint"
                )
            source = self._generation
            accepted_nodes = checkpoint.depth - branch._origin.depth
            abandoned_nodes = branch._tip.depth - checkpoint.depth
            abandoned_bytes = branch._tip.cumulative_bytes - checkpoint.cumulative_bytes
            successor = source + 1
            receipt = PromotionReceipt(
                lineage_id=self.lineage_id,
                source_generation=source,
                successor_generation=successor,
                accepted_position=checkpoint.position,
                accepted_delta_nodes=accepted_nodes,
                abandoned_delta_nodes=abandoned_nodes,
                abandoned_bytes=abandoned_bytes,
            )
            new_counters = dict(self._counters)
            new_counters["promotions"] += 1
            new_counters["promotion_pointer_swaps"] += 1
            new_counters["abandoned_delta_nodes"] += abandoned_nodes
            new_counters["abandoned_bytes"] += abandoned_bytes
            abandoned_tip = branch._tip
            self._tip = checkpoint
            self._generation = successor
            branch._status = BranchStatus.PROMOTED
            branch._tip = None
            branch._origin = None
            self._branches.discard(branch)
            self._counters = new_counters
            self._retire_branch_nodes_locked(branch, abandoned_tip, checkpoint)
            return receipt

    def reject(self, branch: CacheDeltaBranch) -> RejectionReceipt:
        with self._lock:
            if branch._lineage is not self:
                raise CacheBranchTransactionError("branch belongs to another lineage")
            if branch._status != BranchStatus.ACTIVE:
                raise CacheBranchTransactionError(
                    f"branch is already {branch._status.value}"
                )
            return self._reject_locked(branch)

    def _reject_locked(self, branch: CacheDeltaBranch) -> RejectionReceipt:
        """Retire an active branch while the lineage lock is held."""
        nodes = branch._tip.depth - branch._origin.depth
        abandoned_bytes = branch._tip.cumulative_bytes - branch._origin.cumulative_bytes
        receipt = RejectionReceipt(
            lineage_id=self.lineage_id,
            generation=self._generation,
            abandoned_delta_nodes=nodes,
            abandoned_bytes=abandoned_bytes,
        )
        new_counters = dict(self._counters)
        new_counters["rejections"] += 1
        new_counters["abandoned_delta_nodes"] += nodes
        new_counters["abandoned_bytes"] += abandoned_bytes
        abandoned_tip = branch._tip
        origin = branch._origin
        branch._status = BranchStatus.REJECTED
        branch._tip = None
        branch._origin = None
        self._branches.discard(branch)
        self._counters = new_counters
        self._retire_branch_nodes_locked(branch, abandoned_tip, origin)
        return receipt

    def _retire_branch_nodes_locked(
        self,
        branch: CacheDeltaBranch,
        tip: AlignedDeltaCheckpoint,
        keep: AlignedDeltaCheckpoint,
    ) -> None:
        """Drop a finished branch's unreachable nodes from the arena and index.

        Nodes after ``keep`` were appended by this branch alone, so once it is
        promoted or rejected nothing can reach them; its index entries are only
        consulted to promote this branch.
        """
        node = tip
        while node is not None and node is not keep:
            if self._arena.pop(node.checkpoint_id, None) is not None:
                self._arena_delta_bytes -= sum(
                    (delta.logical_bytes for delta in node.deltas)
                )
            node = node.previous
        for key in [key for key in self._checkpoint_index if key[0] == branch.branch_id]:
            del self._checkpoint_index[key]

    def _close_branch(self, branch: CacheDeltaBranch) -> RejectionReceipt | None:
        with self._lock:
            if branch._lineage is not self:
                raise CacheBranchTransactionError("branch belongs to another lineage")
            if branch._status != BranchStatus.ACTIVE:
                return None
            return self._reject_locked(branch)

    def current_view(self) -> DeltaChainView:
        with self._lock:
            self._assert_open_locked()
            return DeltaChainView(
                self.lineage_id, self._generation, self._bases, self._tip
            )

    def rebase(self, replacements: Mapping[CachePlaneKind, PlaneBase]) -> int:
        """Replace the bases with caller-attested full state at the current tip.

        For a lineage whose checkpoints each record every plane's complete state
        (not an increment), the tip alone describes the state and the chain
        behind it is dead weight. The generation is unchanged: this is a
        representation change at one boundary, not a new state, so a promotion
        still advances the generation exactly once. Returns the number of delta
        nodes folded away.
        """
        with self._lock:
            self._assert_open_locked()
            if any(
                (
                    branch._status == BranchStatus.ACTIVE
                    and branch._tip is not branch._origin
                    for branch in tuple(self._branches)
                )
            ):
                raise CacheBranchTransactionError(
                    "rebase requires active branches to have no deltas"
                )
            if set(replacements) != set(self._layouts):
                raise CacheBranchTransactionError(
                    "rebase must replace every lineage plane"
                )
            ordered = []
            for previous in self._bases:
                replacement = replacements[previous.kind]
                if (
                    replacement.kind != previous.kind
                    or replacement.start != 0
                    or replacement.stop != self._tip.position
                    or replacement.layout_id != previous.layout_id
                    or replacement.generation != self._generation
                    or replacement.ring_capacity != previous.ring_capacity
                ):
                    raise CacheBranchTransactionError(
                        f"invalid rebased {previous.kind.value} plane"
                    )
                ordered.append(replacement)
            folded = self._tip.depth
            new_tip = AlignedDeltaCheckpoint(
                uuid.uuid4().hex, self._tip.position, (), None, 0, 0
            )
            # A branch still open on the old tip holds no deltas; the new tip's
            # identity makes it stale exactly as a promotion would.
            if self._issued_compactions:
                self._counters["compaction_cancellations"] += len(
                    self._issued_compactions
                )
                self._issued_compactions.clear()
            self._bases = tuple(ordered)
            self._tip = new_tip
            self._arena = {new_tip.checkpoint_id: new_tip}
            self._checkpoint_index = {}
            self._arena_delta_bytes = 0
            self._arena_version += 1
            self._counters["rebases"] += 1
            self._counters["rebased_delta_nodes"] += folded
            return folded

    def needs_compaction(self, *, max_delta_depth: int, max_delta_bytes: int) -> bool:
        if max_delta_depth < 0 or max_delta_bytes < 0:
            raise ValueError("compaction thresholds must be non-negative")
        with self._lock:
            self._assert_open_locked()
            return (
                max(self._tip.depth, len(self._arena) - 1) >= max_delta_depth
                or self._arena_delta_bytes >= max_delta_bytes
            )

    def prepare_compaction(self) -> CompactionPlan:
        with self._lock:
            self._assert_open_locked()
            if any(
                (
                    branch._status == BranchStatus.ACTIVE
                    and branch._tip is not branch._origin
                    for branch in tuple(self._branches)
                )
            ):
                raise CacheBranchTransactionError(
                    "compaction requires active branches to have no deltas"
                )
            if self._issued_compactions:
                issued = next(iter(self._issued_compactions.values()))
                if (
                    issued[1] is self._tip
                    and issued[2] is self._bases
                    and (issued[5] == self._arena_version)
                ):
                    raise CacheBranchTransactionError(
                        "a compaction plan is already outstanding"
                    )
                cancelled = len(self._issued_compactions)
                self._issued_compactions.clear()
                self._counters["compaction_cancellations"] += cancelled
            per_plane = self._canonical_plane_deltas_locked()
            plan = CompactionPlan(
                plan_id=uuid.uuid4().hex,
                lineage_id=self.lineage_id,
                expected_generation=self._generation,
                expected_tip_id=self._tip.checkpoint_id,
                expected_arena_version=self._arena_version,
                position=self._tip.position,
                bases=self._bases,
                plane_deltas=per_plane,
            )
            source_ids = set()
            for base in self._bases:
                source_ids.update(_payload_object_ids(base.payload))
            for _, deltas in plan.plane_deltas:
                for delta in deltas:
                    source_ids.update(_payload_object_ids(delta.payload))
            self._issued_compactions[plan.plan_id] = (
                plan,
                self._tip,
                self._bases,
                plan.plane_deltas,
                frozenset(source_ids),
                self._arena_version,
            )
            self._counters["compaction_plans"] += 1
            return plan

    def _canonical_plane_deltas_locked(
        self,
    ) -> tuple[tuple[CachePlaneKind, tuple[PlaneDelta, ...]], ...]:
        chain = []
        node = self._tip
        while node.previous is not None:
            chain.append(node)
            node = node.previous
        chain.reverse()
        return tuple(
            (
                (
                    base.kind,
                    tuple(
                        (
                            delta
                            for checkpoint in chain
                            for delta in checkpoint.deltas
                            if delta.kind == base.kind
                        )
                    ),
                )
                for base in self._bases
            )
        )

    def cancel_compaction(self, plan: CompactionPlan) -> bool:
        """Cancel one exact issued plan without accepting a forged copy."""
        with self._lock:
            issued = self._issued_compactions.get(plan.plan_id)
            if issued is None or issued[0] is not plan:
                return False
            self._issued_compactions.pop(plan.plan_id)
            self._counters["compaction_cancellations"] += 1
            return True

    def commit_compaction(
        self, plan: CompactionPlan, replacements: Mapping[CachePlaneKind, PlaneBase]
    ) -> CompactionReceipt:
        with self._lock:
            self._assert_open_locked()

            def refuse(message: str):
                self._counters["compaction_failures"] += 1
                raise CacheBranchTransactionError(message)

            issued = self._issued_compactions.get(plan.plan_id)
            if issued is None or issued[0] is not plan:
                refuse("compaction plan was not issued or was already consumed")
            (
                _,
                issued_tip,
                issued_bases,
                issued_plane_deltas,
                source_ids,
                issued_arena_version,
            ) = issued
            if (
                plan.lineage_id != self.lineage_id
                or plan.expected_generation != self._generation
                or plan.expected_tip_id != self._tip.checkpoint_id
                or (plan.expected_arena_version != self._arena_version)
                or (issued_tip is not self._tip)
                or (issued_bases is not self._bases)
                or (issued_plane_deltas != self._canonical_plane_deltas_locked())
                or (issued_arena_version != self._arena_version)
                or (plan.position != issued_tip.position)
            ):
                self._issued_compactions.pop(plan.plan_id)
                refuse("compaction plan is stale")
            self._issued_compactions.pop(plan.plan_id)
            if set(replacements) != set(self._layouts):
                refuse("compaction must replace every lineage plane")
            successor = self._generation + 1
            ordered = []
            for previous in issued_bases:
                replacement = replacements[previous.kind]
                if replacement.kind != previous.kind:
                    refuse("compaction map key and plane kind disagree")
                if (
                    replacement.start != 0
                    or replacement.stop != issued_tip.position
                    or replacement.layout_id != previous.layout_id
                    or (replacement.generation != successor)
                    or (replacement.ring_capacity != previous.ring_capacity)
                ):
                    refuse(f"invalid compacted {previous.kind.value} plane")
                if source_ids.intersection(_payload_object_ids(replacement.payload)):
                    refuse(f"compacted {previous.kind.value} payload aliases source")
                ordered.append(replacement)
            source_generation = self._generation
            source_depth = self._tip.depth
            source_bytes = self._tip.cumulative_bytes
            replacement_bytes = sum((base.logical_bytes for base in ordered))
            new_bases = tuple(ordered)
            new_tip = AlignedDeltaCheckpoint(
                uuid.uuid4().hex, issued_tip.position, (), None, 0, 0
            )
            new_arena = {new_tip.checkpoint_id: new_tip}
            new_index = {}
            new_counters = dict(self._counters)
            new_counters["compactions"] += 1
            new_counters["compacted_delta_nodes"] += source_depth
            new_counters["compacted_source_bytes"] += source_bytes
            new_counters["compacted_replacement_bytes"] += replacement_bytes
            empty_branches = tuple(
                (
                    branch
                    for branch in self._branches
                    if branch._status == BranchStatus.ACTIVE
                    and branch._tip is branch._origin
                )
            )
            new_counters["compaction_retired_empty_branches"] += len(empty_branches)
            receipt = CompactionReceipt(
                lineage_id=self.lineage_id,
                source_generation=source_generation,
                successor_generation=successor,
                position=issued_tip.position,
                delta_nodes_compacted=source_depth,
                source_delta_bytes=source_bytes,
                replacement_bytes=replacement_bytes,
            )
            old_arena = self._arena
            old_index = self._checkpoint_index
            for branch in empty_branches:
                branch._status = BranchStatus.REJECTED
                branch._tip = None
                branch._origin = None
                self._branches.discard(branch)
            self._bases = new_bases
            self._generation = successor
            self._tip = new_tip
            self._arena = new_arena
            self._checkpoint_index = new_index
            self._arena_delta_bytes = 0
            self._arena_version = 0
            self._counters = new_counters
        old_index.clear()
        old_arena.clear()
        return receipt

    def dispose(self) -> None:
        """Invalidate all handles and release retained arenas off-lock."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for branch in tuple(self._branches):
                if branch._status == BranchStatus.ACTIVE:
                    branch._status = BranchStatus.REJECTED
                    branch._tip = None
                    branch._origin = None
            self._branches.clear()
            old_arena = self._arena
            old_index = self._checkpoint_index
            old_bases = self._bases
            old_plans = self._issued_compactions
            self._arena = {}
            self._checkpoint_index = {}
            self._bases = ()
            self._issued_compactions = {}
            self._tip = None
            self._arena_delta_bytes = 0
        old_index.clear()
        old_arena.clear()
        old_plans.clear()
        del old_bases

    close = dispose

    def __enter__(self) -> "CacheDeltaLineage":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.dispose()

    def compact(self, adapter: ExactCompactionAdapter) -> CompactionReceipt:
        """Two-phase compaction through a fresh-destination adapter.

        Capability checks happen before the adapter sees a source. The adapter
        contract forbids in-place writes, and commit additionally rejects any
        replacement payload graph that aliases a source payload graph.
        """
        materialize = getattr(adapter, "materialize", None)
        if (
            not bool(getattr(adapter, "source_payloads_read_only", False))
            or not bool(getattr(adapter, "destinations_are_fresh", False))
            or (not callable(materialize))
        ):
            raise CacheBranchTransactionError(
                "compaction adapter must attest read-only sources and fresh destinations"
            )
        plan = self.prepare_compaction()
        deltas = dict(plan.plane_deltas)
        try:
            replacements = {
                base.kind: materialize(base, deltas[base.kind]) for base in plan.bases
            }
        except BaseException:
            self.cancel_compaction(plan)
            with self._lock:
                self._counters["compaction_failures"] += 1
            raise
        return self.commit_compaction(plan, replacements)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._counters)
            result.update(
                {
                    "lineage_id": self.lineage_id,
                    "generation": self._generation,
                    "position": None if self._tip is None else self._tip.position,
                    "delta_depth": 0 if self._tip is None else self._tip.depth,
                    "live_delta_bytes": 0
                    if self._tip is None
                    else self._tip.cumulative_bytes,
                    "arena_delta_bytes": self._arena_delta_bytes,
                    "arena_nodes": len(self._arena),
                    "arena_version": self._arena_version,
                    "active_compaction_plans": len(self._issued_compactions),
                    "planes": tuple((base.kind.value for base in self._bases)),
                    "closed": self._closed,
                }
            )
            return result


def create_cache_delta_lineage(
    bases: Iterable[PlaneBase], *, enabled: bool | None = None
) -> CacheDeltaLineage | None:
    if not cache_delta_promotion_enabled(enabled):
        return None
    return CacheDeltaLineage(bases)


def _payload_object_ids(value: Any, seen: set[int] | None = None) -> set[int]:
    """Common Python payload-graph identity proof with no device readback.

    Opaque extension objects (including MLX arrays) are atomic leaves whose
    own identity is checked. Python containers, dataclasses, instance dicts,
    deques, and slots are recursively inspected.
    """
    if isinstance(value, (type(None), bool, int, float, complex, str, bytes)):
        return set()
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return set()
    seen.add(identity)
    result = {identity}
    if isinstance(value, Mapping):
        for key, item in value.items():
            result.update(_payload_object_ids(key, seen))
            result.update(_payload_object_ids(item, seen))
    elif isinstance(value, (list, tuple, set, frozenset, deque)):
        for item in value:
            result.update(_payload_object_ids(item, seen))
    if dataclasses.is_dataclass(value) and (not isinstance(value, type)):
        for field in dataclasses.fields(value):
            result.update(_payload_object_ids(getattr(value, field.name), seen))
    instance_dict = getattr(value, "__dict__", None)
    if isinstance(instance_dict, Mapping):
        result.update(_payload_object_ids(instance_dict, seen))
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot in {"__dict__", "__weakref__"}:
                continue
            lookup_slot = slot
            if slot.startswith("__") and (not slot.endswith("__")):
                class_name = cls.__name__.lstrip("_")
                if class_name:
                    lookup_slot = f"_{class_name}{slot}"
            try:
                item = getattr(value, lookup_slot)
            except (AttributeError, TypeError):
                continue
            result.update(_payload_object_ids(item, seen))
    return result
