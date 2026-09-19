"""KV-cache quantization published through the approximate-state seam.

This is the first live consumer of :mod:`approximate_state`.  A model adapter
declares which operations its attention and cache classes support (a name ->
:class:`KVQuantizationDescriptor` mapping); the serving engine selects one by
policy, binds it to the adapter identity, and applies it to a request-private
lane cache before the lane is inserted into the ordinary batch generator.

The tensor math itself is the cache classes' own ``to_quantized`` plus the
quantized SDPA helper; nothing here touches arrays except through
``maybe_quantize_kv_cache``.  Recurrent/linear state planes have no
``to_quantized`` and stay exact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .approximate_state import ApproximateStateError

_POLICY_KEYS = frozenset({"operation", "enabled", "start_tokens", "evidence"})
_SUPPORTED_BITS = (2, 3, 4, 5, 6, 8)
_SUPPORTED_GROUPS = (32, 64, 128)


def _plain_int(value, label, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class KVQuantizationDescriptor:
    """Adapter-declared parameters of one KV quantization operation."""

    key_bits: int
    value_bits: int
    group_size: int = 64
    rotate: bool = False
    start: int = 0

    def __post_init__(self):
        for label in ("key_bits", "value_bits"):
            if _plain_int(getattr(self, label), label) not in _SUPPORTED_BITS:
                raise ValueError(f"unsupported {label}: {getattr(self, label)}")
        if _plain_int(self.group_size, "group_size") not in _SUPPORTED_GROUPS:
            raise ValueError(f"unsupported group_size: {self.group_size}")
        if not isinstance(self.rotate, bool):
            raise ValueError("rotate must be a boolean")
        if _plain_int(self.start, "start") != 0:
            # Continuous batching merges lanes plane by plane; a lane that
            # turned approximate mid-decode could not stay in its batch.
            raise ValueError("KV quantization operations quantize from token 0")

    def as_dict(self) -> dict:
        return {
            "key_bits": self.key_bits,
            "value_bits": self.value_bits,
            "group_size": self.group_size,
            "rotate": self.rotate,
            "start": self.start,
        }


def standard_kv_quantization_operations(*, group_size: int = 64) -> dict:
    """The two named operations an eligible adapter may declare."""
    return {
        "kv_q8": KVQuantizationDescriptor(8, 8, group_size=group_size),
        "kv_k8v4": KVQuantizationDescriptor(8, 4, group_size=group_size),
    }


@dataclass(frozen=True)
class ServingApproximateKVPolicy:
    """Server-side selection.  Default off; never inferred from a bare flag."""

    enabled: bool = False
    operation: str | None = None
    start_tokens: int = 0
    evidence: tuple[str, ...] = ()

    @classmethod
    def from_value(cls, value) -> "ServingApproximateKVPolicy":
        if value is None or value is False:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(
                "approximate KV policy must be a mapping naming an operation"
            )
        unknown = set(value) - _POLICY_KEYS
        if unknown:
            raise ValueError(
                f"unknown approximate KV policy keys: {sorted(unknown)}"
            )
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("approximate KV enabled must be a boolean")
        operation = value.get("operation")
        if not isinstance(operation, str) or not operation:
            raise ValueError("approximate KV policy requires an operation name")
        start_tokens = _plain_int(
            value.get("start_tokens", 0), "approximate KV start_tokens"
        )
        evidence = value.get("evidence", ())
        if isinstance(evidence, (str, bytes)) or not all(
            isinstance(item, str) and item for item in evidence
        ):
            raise ValueError("approximate KV evidence must be a list of strings")
        return cls(enabled, operation, start_tokens, tuple(evidence))

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "operation": self.operation,
            "start_tokens": self.start_tokens,
            "evidence": list(self.evidence),
        }


def operation_revision(adapter_fingerprint, name, descriptor) -> str:
    """Bind an operation to the adapter identity and its exact parameters."""
    payload = json.dumps(
        {
            "schema": "mlx2.approximate-kv-operation.v1",
            "adapter": adapter_fingerprint,
            "operation": str(name),
            "descriptor": descriptor.as_dict(),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _leaves(planes):
    if not isinstance(planes, (list, tuple)):
        # An opaque cache object is its own (single) leaf.
        planes = (planes,)
    for plane in planes:
        children = getattr(plane, "caches", None)
        if isinstance(children, (tuple, list)):
            yield from _leaves(children)
        else:
            yield plane


def _is_quantized(leaf) -> bool:
    return hasattr(leaf, "key_bits") or hasattr(leaf, "bits")


def quantized_plane_count(planes) -> int:
    return sum(1 for leaf in _leaves(planes or ()) if _is_quantized(leaf))


def prompt_cache_is_approximate(planes) -> bool:
    """Structural check used to keep approximate state out of exact stores."""
    return any(_is_quantized(leaf) for leaf in _leaves(planes or ()))


@dataclass(frozen=True)
class LaneKVState:
    """Revision-bound view of one lane's cache planes."""

    revision: str
    planes: tuple
    quantized_planes: int = 0


def stage_lane_state(state: LaneKVState) -> LaneKVState:
    """Private staging without copying device arrays.

    Quantization replaces plane *objects* in the staged tuple and only reads
    the source arrays, so a fresh container is sufficient isolation: the
    source branch keeps its exact planes untouched.
    """
    return LaneKVState(state.revision, tuple(state.planes), state.quantized_planes)


class KVQuantizationOperation:
    """``ApproximateStateAdapter`` for one adapter-declared descriptor."""

    def __init__(self, name, descriptor, *, adapter_fingerprint):
        if not isinstance(descriptor, KVQuantizationDescriptor):
            raise ValueError(
                f"approximate KV operation {name!r} must be declared with a "
                "KVQuantizationDescriptor"
            )
        self.name = str(name)
        self.descriptor = descriptor
        self.revision = operation_revision(adapter_fingerprint, name, descriptor)

    def apply(self, state: LaneKVState) -> LaneKVState:
        from .generate import maybe_quantize_kv_cache

        planes = list(state.planes)
        try:
            maybe_quantize_kv_cache(
                planes,
                self.descriptor.start,
                self.descriptor.group_size,
                None,
                kv_key_bits=self.descriptor.key_bits,
                kv_value_bits=self.descriptor.value_bits,
                kv_rotate=self.descriptor.rotate,
            )
        except ValueError as error:
            raise ApproximateStateError(str(error)) from error
        quantized = quantized_plane_count(planes)
        if not quantized:
            raise ApproximateStateError(
                f"approximate KV operation {self.name!r} found no quantizable "
                "attention plane"
            )
        for leaf in _leaves(planes):
            if hasattr(leaf, "to_quantized") and not _is_quantized(leaf):
                raise ApproximateStateError(
                    f"approximate KV operation {self.name!r} left "
                    f"{type(leaf).__name__} exact"
                )
            if _is_quantized(leaf) and not hasattr(leaf, "merge"):
                raise ApproximateStateError(
                    f"{type(leaf).__name__} cannot join a continuous batch"
                )
        return LaneKVState(f"{state.revision}:{self.name}", tuple(planes), quantized)


def declared_operations(adapter, *, adapter_fingerprint) -> dict:
    """Bind every operation the adapter declares.  Default: none."""
    from ..adapters.base import approximate_kv_operations

    return {
        name: KVQuantizationOperation(
            name, descriptor, adapter_fingerprint=adapter_fingerprint
        )
        for name, descriptor in approximate_kv_operations(adapter).items()
    }
