"""Model-neutral target-tree executor contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class TargetTreeDispatchMetrics:
    """Validated facts from one completed physical target-tree dispatch.

    Providers describe the dispatch.  The shared scheduler applies the facts
    transactionally so provider callbacks can never strand an open cache
    transaction after a successful model forward.
    """

    width: int
    physical_packed: bool
    executor_cached: bool
    cached: bool
    provider_protocol: str


@runtime_checkable
class TargetTreeTransaction(Protocol):
    """Open target-cache transaction returned by a target-tree provider."""

    closed: bool
    provider_protocol: str
    provider_revision_token: str

    def abort(self) -> None: ...


@runtime_checkable
class TargetTreeExecutorProvider(Protocol):
    """Revision-bound adapter provider consumed by external draft scheduling."""

    source_identity: tuple[tuple[str, str], ...]
    source_revision_token: str
    protocol: str
    maximum_cohort_width: int

    def validate_cohort(self, width: int, configured_limit: int) -> None: ...

    def forward(
        self, model: Any, inputs: Any, parents: Any, cache: Any, layers: Any,
        *, cached: bool,
    ) -> Any: ...

    def forward_many(
        self, model: Any, inputs: Any, parents: Any, caches: Any, layers: Any,
        *, cached: bool, configured_limit: int,
    ) -> Any: ...

    def dispatch_metrics(
        self, width: int, transaction: Any, *, cached: bool,
    ) -> TargetTreeDispatchMetrics: ...

    def validate_revision(self) -> None: ...

    def receipt(self) -> dict: ...


@runtime_checkable
class ASTTargetTreeExecutorProvider(TargetTreeExecutorProvider, Protocol):
    """Provider that may publish a numerically admitted AST backend.

    ``ast_execution_backend`` receives the already source/device/build-bound
    primary profile. Returning ``None`` is a refusal. A returned backend must
    carry a matching affirmative numerical oracle; ``ExecutionProfile``
    validates that binding before the planner can see it.
    """

    def ast_execution_backend(self, profile: Any) -> Any: ...
