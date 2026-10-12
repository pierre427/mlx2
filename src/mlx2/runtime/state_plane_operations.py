"""Model-supplied contracts for movable inference state.

The scheduler may move state between checkpoint, private-lane and batched
owners, but it must not infer the meaning of a model's cache classes.  A model
provider declares the operations it supports and the revision/layout/fidelity
of each plane.  Consumers validate these declarations before allocating or
publishing a successor and fail closed when a declaration is absent or drifts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

STATE_PLANE_OPERATIONS = frozenset(
    {"checkpoint", "private_successor", "rollback", "promotion"}
)


class StatePlaneContractError(RuntimeError):
    """A cache does not declare a compatible state movement contract."""


class StatePlaneOperationRefused(RuntimeError):
    """A provider cannot perform an exact state movement operation."""


@dataclass(frozen=True)
class StatePlaneBinding:
    """Exact loaded-model and cache/numerics identity for movable state.

    ``provider_revision`` describes code.  The remaining keys bind state to
    the particular loaded artifact and its numerical/cache contract, so two
    model instances using the same provider revision are not interchangeable.
    """

    provider_revision: str
    artifact_key: tuple[Any, ...]
    cache_key: tuple[Any, ...]
    numerics_key: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.provider_revision:
            raise ValueError("state-plane binding requires a provider revision")
        if not self.artifact_key or not self.cache_key or not self.numerics_key:
            raise ValueError("state-plane binding keys must be non-empty")


@dataclass(frozen=True)
class SharedPrefixPlan:
    """Opaque, provider-owned plan for side-effect-free prefix sharing."""

    binding: StatePlaneBinding
    payload: Any
    admitted: bool
    telemetry: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class SharedPrefixMaterialization:
    """Replacement target rows built without mutating their source rows."""

    target_rows: tuple[tuple[Any, ...], ...]
    telemetry: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class StatePlaneContract:
    revision: str
    layout: str
    fidelity: str
    operations: frozenset[str]

    def __post_init__(self) -> None:
        if not self.revision or not self.layout:
            raise ValueError("state-plane revision and layout must be non-empty")
        if self.fidelity not in {"exact", "approximate"}:
            raise ValueError("state-plane fidelity must be exact or approximate")
        unknown = set(self.operations) - STATE_PLANE_OPERATIONS
        if unknown:
            raise ValueError(f"unknown state-plane operations: {sorted(unknown)!r}")


@runtime_checkable
class StatePlaneOperations(Protocol):
    """Operations and declarations supplied by a model implementation."""

    provider_revision: str

    def contract(self, cache: Any) -> StatePlaneContract | None: ...


@runtime_checkable
class CompactGdnReplayOperations(StatePlaneOperations, Protocol):
    """Model-owned admission for exact compact recurrent-state replay."""

    def authorize_compact_gdn_replay(
        self, cache: Any, gdn_contract: Any, steps: int
    ) -> Any: ...


@runtime_checkable
class SegmentedViewOperations(StatePlaneOperations, Protocol):
    """Provider-owned construction of a non-owning segmented compute view."""

    def build_segmented_view(
        self,
        row_caches: Iterable[Any],
        *,
        note: Callable[[str, int], None] | None = None,
        shared_prefix: bool = False,
    ) -> Any: ...

    def validate_segmented_view(self, view: Any, row_caches: Iterable[Any]) -> None: ...


@runtime_checkable
class SegmentedPromotionOperations(SegmentedViewOperations, Protocol):
    """Exact model operations used by segmented-to-physical promotion."""

    def reserve_promotion_layer(
        self,
        cache: Any,
        tail: int,
        *,
        rows: int | None = None,
        shared_prefix: bool = False,
        timing: dict[str, int] | None = None,
    ) -> StatePlaneReservation: ...

    def validate_staged_layer(
        self, source: Any, candidate: Any, base: int | None
    ) -> None: ...

    def materialize_promoted_layer(
        self,
        source: Any,
        candidate: Any,
        base: int | None,
        *,
        advance: int,
        target: bool,
    ) -> StatePlaneMaterialization: ...


@runtime_checkable
class SegmentedAttachOperations(StatePlaneOperations, Protocol):
    """Model-owned operations used by shared-prefix cohort attachment."""

    def bind_loaded_state(
        self, model: Any, row_caches: Iterable[Any]
    ) -> StatePlaneBinding: ...

    def plan_shared_prefix(
        self,
        row_caches: Iterable[Any],
        *,
        binding: StatePlaneBinding,
        prefix_id: str,
        remaining_tokens: int,
        note: Callable[[str, int], None] | None = None,
    ) -> SharedPrefixPlan: ...

    def materialize_shared_prefix(
        self, plan: SharedPrefixPlan
    ) -> SharedPrefixMaterialization: ...

    def is_shared_prefix_cache(self, cache: Any) -> bool: ...

    def materialize_private_prefix(self, cache: Any) -> Any: ...


@dataclass(frozen=True)
class StatePlaneReservation:
    cache: Any
    base: int | None
    reserved_bytes: int = 0
    evaluation_values: tuple[Any, ...] = ()


@dataclass(frozen=True)
class StatePlaneMaterialization:
    cache: Any
    patched_bytes: int = 0
    recurrent_arrays_reused: int = 0
    evaluation_values: tuple[Any, ...] = ()


def require_state_plane_contracts(
    provider: StatePlaneOperations,
    caches: Iterable[Any],
    *,
    operations: Iterable[str],
    fidelity: str = "exact",
) -> tuple[StatePlaneContract, ...]:
    """Return validated contracts for ``caches`` or fail closed.

    Contracts are deliberately checked per layer: hybrid models can have
    different attention and recurrent layouts under one provider revision.
    """

    if not isinstance(provider, StatePlaneOperations):
        raise StatePlaneContractError("state-plane provider protocol is missing")
    required = frozenset(operations)
    unknown = required - STATE_PLANE_OPERATIONS
    if unknown:
        raise ValueError(
            f"unknown required state-plane operations: {sorted(unknown)!r}"
        )
    if fidelity not in {"exact", "approximate"}:
        raise ValueError("required state-plane fidelity must be exact or approximate")
    contracts = []
    for cache in caches:
        contract = provider.contract(cache)
        if not isinstance(contract, StatePlaneContract):
            raise StatePlaneContractError(
                f"{type(cache).__name__} has no declared state-plane contract"
            )
        if contract.fidelity != fidelity:
            raise StatePlaneContractError(
                f"{type(cache).__name__} declares {contract.fidelity} state; "
                f"{fidelity} state is required"
            )
        missing = required - contract.operations
        if missing:
            raise StatePlaneContractError(
                f"{type(cache).__name__} lacks state-plane operations "
                f"{sorted(missing)!r}"
            )
        contracts.append(contract)
    return tuple(contracts)
