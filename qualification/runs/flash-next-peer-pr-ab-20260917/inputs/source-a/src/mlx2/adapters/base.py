"""Backend-neutral request and execution adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ..contracts import ModelDescriptor
from ..state import StateManifest


@dataclass(frozen=True, slots=True)
class RequestContext:
    request_id: str
    descriptor_key: str
    prompt_tokens: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id or not self.descriptor_key:
            raise ValueError("request_id and descriptor_key are required")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class SequenceState:
    sequence_id: str
    manifest: StateManifest


@dataclass(frozen=True, slots=True)
class TokenStep:
    sequence_id: str
    token_id: int
    state: StateManifest
    finished: bool = False


@runtime_checkable
class ExecutionAdapter(Protocol):
    """The only interface through which the scheduler executes model math."""

    @property
    def descriptor(self) -> ModelDescriptor: ...

    def prefill_batch(
        self, requests: Sequence[RequestContext]
    ) -> tuple[SequenceState, ...]: ...

    def decode_batch(
        self, sequences: Sequence[SequenceState]
    ) -> tuple[TokenStep, ...]: ...
