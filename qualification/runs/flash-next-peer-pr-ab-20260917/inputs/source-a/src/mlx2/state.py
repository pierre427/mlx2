"""Revision-bound request state transactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .contracts import FIDELITY_RANK, Capability, Fidelity, QualifiedProfile, StatePlane


class StateOperation(str, Enum):
    APPEND = "append"
    FORK = "fork"
    ROLLBACK = "rollback"
    RESTORE = "restore"
    SPILL = "spill"
    COMPACT = "compact"


@dataclass(frozen=True, slots=True)
class StateIdentity:
    request_id: str
    revision: int

    def __post_init__(self) -> None:
        if not self.request_id or self.revision < 0:
            raise ValueError("request_id must be set and revision must be non-negative")


@dataclass(frozen=True, slots=True)
class StateManifest:
    identity: StateIdentity
    planes: Mapping[StatePlane, Any] = field(default_factory=dict)
    fidelity: Fidelity = Fidelity.EXACT

    def __post_init__(self) -> None:
        object.__setattr__(self, "planes", MappingProxyType(dict(self.planes)))


class StateConflict(RuntimeError):
    """A transaction no longer matches the published state revision."""


class InvalidStatePublication(ValueError):
    """A transaction tried to publish a fidelity its operation cannot produce."""


class RequestStateTransaction:
    def __init__(
        self,
        current: StateManifest,
        operation: StateOperation,
        *,
        profile: QualifiedProfile | None = None,
    ) -> None:
        self.current = current
        self.operation = operation
        self.profile = profile
        self._prepared: StateManifest | None = None
        self._closed = False

    def prepare(
        self,
        planes: Mapping[StatePlane, Any],
        *,
        fidelity: Fidelity = Fidelity.EXACT,
    ) -> StateManifest:
        if self._closed:
            raise RuntimeError("transaction is closed")
        if fidelity is not Fidelity.EXACT:
            qualified_compaction = (
                self.operation is StateOperation.COMPACT
                and self.profile is not None
                and Capability.COMPACTION in self.profile.capabilities
            )
            if not qualified_compaction:
                raise InvalidStatePublication(
                    "non-exact state requires a qualified compaction profile"
                )
            if FIDELITY_RANK[fidelity] > FIDELITY_RANK[self.profile.fidelity]:
                raise InvalidStatePublication(
                    "state fidelity exceeds the qualified compaction profile"
                )
        self._prepared = StateManifest(
            identity=StateIdentity(
                request_id=self.current.identity.request_id,
                revision=self.current.identity.revision + 1,
            ),
            planes=planes,
            fidelity=fidelity,
        )
        return self._prepared

    def commit(self, published: StateManifest) -> StateManifest:
        if self._closed or self._prepared is None:
            raise RuntimeError("transaction is not prepared")
        if published.identity != self.current.identity:
            raise StateConflict("published state changed after transaction began")
        self._closed = True
        return self._prepared

    def abort(self) -> None:
        self._prepared = None
        self._closed = True
