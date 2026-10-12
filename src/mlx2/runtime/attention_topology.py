"""Model-neutral, revision-bound attention-state topology contracts.

The scheduler must not infer cache planes from model names or cache classes.
Adapters describe their ordered target and auxiliary layers here; APCv2 and
state-promotion capabilities can then bind and validate that description.
Ordinary execution does not require this contract, so adapters without one
remain loadable until a selected capability explicitly asks for it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

FULL_KV = "full_kv"
SLIDING_KV = "sliding_kv"
RECURRENT = "recurrent"
INDEXED_KV = "indexed_kv"
LAYER_KINDS = frozenset({FULL_KV, SLIDING_KV, RECURRENT, INDEXED_KV})


class AttentionTopologyError(ValueError):
    """A selected state capability lacks an exact topology contract."""


def _nonempty(value, field):
    if not isinstance(value, str) or not value:
        raise AttentionTopologyError(f"attention topology {field} is required")
    return value


def _layers(values: Sequence[str], field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise AttentionTopologyError(f"attention topology {field} must be a sequence")
    result = tuple(values)
    unknown = set(result) - LAYER_KINDS
    if unknown:
        raise AttentionTopologyError(
            f"attention topology {field} has unknown layer kind: {min(unknown)}"
        )
    return result


@dataclass(frozen=True, slots=True)
class AttentionTopology:
    """Ordered attention/recurrent planes owned by one immutable model state.

    ``state_revision`` binds model weights and execution numerics;
    ``cache_layout`` binds the physical cache ABI.  Layout names for indexed
    and recurrent planes are adapter-owned because sharing a layer kind does
    not imply sharing its state representation.
    """

    state_revision: str
    cache_layout: str
    layers: tuple[str, ...]
    auxiliary_layers: tuple[str, ...] = ()
    sliding_window: int | None = None
    indexed_layout: str | None = None
    recurrent_layout: str | None = None
    schema: str = "mlx2.attention-topology.v1"

    def __post_init__(self):
        _nonempty(self.state_revision, "state revision")
        _nonempty(self.cache_layout, "cache layout")
        object.__setattr__(self, "layers", _layers(self.layers, "layers"))
        object.__setattr__(
            self,
            "auxiliary_layers",
            _layers(self.auxiliary_layers, "auxiliary layers"),
        )
        if not self.layers:
            raise AttentionTopologyError("attention topology must declare target layers")
        if SLIDING_KV in self.layers + self.auxiliary_layers:
            if type(self.sliding_window) is not int or self.sliding_window < 1:
                raise AttentionTopologyError(
                    "sliding attention requires a positive window"
                )
        elif self.sliding_window is not None:
            raise AttentionTopologyError(
                "sliding window is invalid without sliding attention"
            )
        if INDEXED_KV in self.layers + self.auxiliary_layers:
            _nonempty(self.indexed_layout, "indexed layout")
        elif self.indexed_layout is not None:
            raise AttentionTopologyError(
                "indexed layout is invalid without indexed attention"
            )
        if RECURRENT in self.layers + self.auxiliary_layers:
            _nonempty(self.recurrent_layout, "recurrent layout")
        elif self.recurrent_layout is not None:
            raise AttentionTopologyError(
                "recurrent layout is invalid without recurrent layers"
            )

    def as_dict(self) -> dict:
        return {
            "schema": self.schema,
            "state_revision": self.state_revision,
            "cache_layout": self.cache_layout,
            "layers": list(self.layers),
            "auxiliary_layers": list(self.auxiliary_layers),
            "sliding_window": self.sliding_window,
            "indexed_layout": self.indexed_layout,
            "recurrent_layout": self.recurrent_layout,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def binding(self) -> dict:
        """Small immutable-state identity suitable for APCv2/promotion keys."""
        return {
            "schema": self.schema,
            "state_revision": self.state_revision,
            "cache_layout": self.cache_layout,
            "topology_fingerprint": self.fingerprint,
        }

    def validate_binding(self, binding, *, capability: str) -> dict:
        """Fail closed for a selected capability and return a detached copy."""
        capability = _nonempty(capability, "capability")
        if not isinstance(binding, Mapping):
            raise AttentionTopologyError(
                f"{capability} requires an attention topology binding"
            )
        expected = self.binding()
        if set(binding) != set(expected) or any(
            binding.get(key) != value for key, value in expected.items()
        ):
            raise AttentionTopologyError(
                f"{capability} attention topology revision or layout mismatch"
            )
        return dict(expected)


def require_adapter_topology(adapter, *, capability: str) -> AttentionTopology:
    """Read an adapter-owned descriptor only when a capability selects it."""
    provider = getattr(adapter, "attention_topology", None)
    if not callable(provider):
        raise AttentionTopologyError(
            f"{capability} requires an adapter attention topology provider"
        )
    topology = provider()
    if not isinstance(topology, AttentionTopology):
        raise AttentionTopologyError(
            f"{capability} adapter returned an invalid attention topology"
        )
    return topology


def apcv2_cache_layout_fingerprint(adapter):
    """Compose an APCv2 namespace for adapters that declare this contract.

    Undeclared adapters retain their legacy namespace during the incremental
    migration.  Once an adapter declares a provider, APCv2 is a selected state
    capability: malformed, stale, or mismatched descriptors fail closed.
    """
    legacy = getattr(adapter, "layout", None)
    provider = getattr(adapter, "attention_topology", None)
    if not callable(provider):
        return legacy
    topology = require_adapter_topology(adapter, capability="APCv2 exact state reuse")
    identity = getattr(adapter, "identity", None)
    revision = identity.get("fingerprint") if isinstance(identity, Mapping) else None
    if topology.cache_layout != legacy or topology.state_revision != revision:
        raise AttentionTopologyError(
            "APCv2 exact state reuse attention topology revision or layout mismatch"
        )
    return (legacy, topology.fingerprint)


__all__ = [
    "FULL_KV",
    "INDEXED_KV",
    "RECURRENT",
    "SLIDING_KV",
    "AttentionTopology",
    "AttentionTopologyError",
    "apcv2_cache_layout_fingerprint",
    "require_adapter_topology",
]
