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
from typing import Mapping

from .approximate_state import ApproximateStateError

_POLICY_KEYS = frozenset(
    {"operation", "enabled", "start_tokens", "evidence", "compose_mtp"}
)
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
class XingLatentKV8Descriptor:
    """Candidate-only Xing MLA format; positional key remains dense BF16.

    This descriptor is deliberately separate from the generic key/value
    quantizer: a Xing ``KVCache`` stores the latent in ``keys`` and the RoPE
    key in ``values``.  Generic KV8 would quantize both and change RoPE math.
    An adapter must explicitly declare this operation after model-path and
    APCv2 qualification before serving can select it.
    """

    cache_layout: str = "xing-mla-latent-kv8-rope-bf16-v1"
    latent_bits: int = 8
    latent_group_size: int = 64
    positional_dtype: str = "bfloat16"

    def as_dict(self) -> dict:
        return {
            "cache_layout": self.cache_layout,
            "latent_bits": self.latent_bits,
            "latent_group_size": self.latent_group_size,
            "positional_dtype": self.positional_dtype,
        }


class XingLatentKV8Operation:
    """Revision-bound, private exact-to-approximate Xing cache transform."""

    def __init__(self, *, adapter_fingerprint):
        self.name = "xing_latent_kv8"
        self.descriptor = XingLatentKV8Descriptor()
        self.revision = operation_revision(adapter_fingerprint, self.name, self.descriptor)

    def apply(self, state: "LaneKVState") -> "LaneKVState":
        from .models.cache import KVCache
        from .models.xing_latent_kv8 import XingLatentKV8Cache

        if not state.planes or any(type(plane) is not KVCache for plane in state.planes):
            raise ApproximateStateError(
                "Xing latent KV8 requires homogeneous exact KVCache source planes"
            )
        planes = tuple(XingLatentKV8Cache.from_exact(plane) for plane in state.planes)
        return LaneKVState(
            f"{state.revision}:{self.name}:{self.descriptor.cache_layout}",
            planes,
            len(planes),
        )


@dataclass(frozen=True)
class ServingApproximateKVPolicy:
    """Server-side selection.  Default off; never inferred from a bare flag."""

    enabled: bool = False
    operation: str | None = None
    start_tokens: int = 0
    evidence: tuple[str, ...] = ()
    # Self-MTP composition: quantize the lane's *target* planes only and keep
    # the MTP head's draft cache exact.  Default off; the MTP refusal stands
    # unless this is set explicitly.
    compose_mtp: bool = False

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
        compose_mtp = value.get("compose_mtp", False)
        if not isinstance(compose_mtp, bool):
            raise ValueError("approximate KV compose_mtp must be a boolean")
        return cls(enabled, operation, start_tokens, tuple(evidence), compose_mtp)

    def as_dict(self) -> dict:
        payload = {
            "enabled": self.enabled,
            "operation": self.operation,
            "start_tokens": self.start_tokens,
            "evidence": list(self.evidence),
        }
        # Present only when selected, so existing qualification records (whose
        # settings hash predates the key) keep matching the ordinary route.
        if self.compose_mtp:
            payload["compose_mtp"] = True
        return payload


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


def source_state_revision(adapter_fingerprint, cache_layout) -> str:
    """Bind the exact state an operation reads to the adapter and layout."""
    payload = json.dumps(
        {
            "schema": "mlx2.approximate-kv-source.v1",
            "adapter": adapter_fingerprint,
            "cache_layout": cache_layout,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def lane_source_revision(planes, *, fresh_revision: str, warm: bool) -> str:
    """Source revision named by the planes' own provenance.

    A warm APCv2 branch records the key it was stored under, which names the
    adapter and cache layout that produced it; warm planes without one fail
    closed.  Fresh planes are the serving adapter's own empty state.
    """
    key = getattr(getattr(planes, "cow_metadata", None), "key", None)
    if key is None:
        if warm:
            raise ApproximateStateError("approximate KV source state has no provenance")
        return fresh_revision
    return source_state_revision(
        getattr(key, "adapter", None), getattr(key, "cache_layout_fingerprint", None)
    )


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


class SourceBoundKVQuantization:
    """One operation bound to the exact source state it may read.

    ``ApproximateKVController`` requires the operation's revision to equal the
    revision of the state it transforms, so binding it to the serving
    adapter's source revision refuses planes produced by anything else.
    """

    def __init__(self, operation: KVQuantizationOperation, source_revision: str):
        self.operation = operation
        self.name = operation.name
        self.descriptor = operation.descriptor
        self.revision = str(source_revision)

    def apply(self, state: LaneKVState) -> LaneKVState:
        return self.operation.apply(state)


def declared_operations(adapter, *, adapter_fingerprint) -> dict:
    """Bind every operation the adapter declares.  Default: none."""
    from ..adapters.base import approximate_kv_operations

    operations = {}
    for name, descriptor in approximate_kv_operations(adapter).items():
        if isinstance(descriptor, XingLatentKV8Descriptor):
            operation = XingLatentKV8Operation(adapter_fingerprint=adapter_fingerprint)
            if name != operation.name or descriptor != operation.descriptor:
                raise ValueError("invalid Xing latent KV8 operation declaration")
        else:
            operation = KVQuantizationOperation(
                name, descriptor, adapter_fingerprint=adapter_fingerprint
            )
        operations[name] = operation
    return operations
