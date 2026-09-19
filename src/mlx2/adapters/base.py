"""Backend-neutral request and execution adapter boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

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


@dataclass(frozen=True, slots=True)
class AudioOutput:
    """Revision-bound binary audio returned by an adapter-owned synthesizer."""

    data: bytes
    media_type: str
    sample_rate: int | None = None
    channels: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("audio output data must be nonempty bytes")
        if not isinstance(self.media_type, str) or not self.media_type.startswith("audio/"):
            raise ValueError("audio output media_type must be an audio MIME type")
        if self.sample_rate is not None and self.sample_rate <= 0:
            raise ValueError("audio output sample_rate must be positive")
        if self.channels is not None and self.channels <= 0:
            raise ValueError("audio output channels must be positive")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


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


def decoder_input_representations(
    adapter,
    texts: Sequence[str],
    *,
    dimensions: int | None = None,
    array_module=None,
) -> tuple[list[list[float]], int]:
    """Mean-pool normalized decoder input embeddings.

    This is an explicit representation contract, not a claim that a generative
    model is retrieval-trained.  Keeping the operation adapter-owned makes the
    semantics stable while model-specific embedding lookup stays out of HTTP
    and scheduling code.
    """
    if array_module is None:
        import mlx.core as array_module

    model = getattr(adapter, "model", None)
    embedding = getattr(getattr(model, "model", None), "embed_tokens", None)
    tokenizer = getattr(adapter, "tokenizer", None)
    if not callable(embedding) or tokenizer is None:
        raise NotImplementedError(
            "adapter has no decoder input-embedding representation"
        )
    vectors = []
    prompt_tokens = 0
    for text in texts:
        tokens = list(tokenizer.encode(text, add_special_tokens=False))
        if not tokens:
            raise ValueError("embedding input tokenized to an empty sequence")
        prompt_tokens += len(tokens)
        hidden = embedding(array_module.array(tokens))
        pooled = array_module.mean(hidden.astype(array_module.float32), axis=0)
        if dimensions is not None:
            if dimensions > int(pooled.shape[-1]):
                raise ValueError(
                    "embedding dimensions exceed the model hidden dimension"
                )
            pooled = pooled[:dimensions]
        norm = array_module.sqrt(array_module.sum(pooled * pooled))
        if hasattr(array_module, "eval"):
            array_module.eval(norm)
        norm_value = float(norm.item() if hasattr(norm, "item") else norm)
        if not norm_value:
            raise RuntimeError("embedding representation has zero norm")
        pooled = pooled / norm
        if hasattr(array_module, "eval"):
            array_module.eval(pooled)
        vectors.append([float(value) for value in pooled.tolist()])
    return vectors, prompt_tokens


def approximate_kv_operations(adapter) -> Mapping[str, Any]:
    """Approximate KV operations an adapter declares.  Default: none.

    An adapter opts in by defining ``approximate_kv_operations()`` returning
    ``{operation name: descriptor}``.  It may do so only when every attention
    plane its model allocates supports ``to_quantized`` and its attention goes
    through the quantized-SDPA-capable helper
    (``runtime.models.base.scaled_dot_product_attention``).  An adapter that
    says nothing is unsupported, and a server that selects an operation on it
    fails closed before it becomes ready.
    """
    declared = getattr(adapter, "approximate_kv_operations", None)
    if declared is None:
        return {}
    operations = declared() if callable(declared) else declared
    if not isinstance(operations, Mapping):
        raise ValueError("approximate_kv_operations must return a mapping")
    return dict(operations)
