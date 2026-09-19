"""Stable descriptors shared by routing, state, cache, and adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class Capability(str, Enum):
    TEXT = "text"
    VISION = "vision"
    VIDEO = "video"
    AUDIO = "audio"
    OUTPUT_AUDIO = "output_audio"
    CONTINUOUS_BATCH = "continuous_batch"
    PREFIX_REUSE = "prefix_reuse"
    APC_V2 = "apc_v2"
    MTP = "mtp"
    EXTERNAL_DRAFT = "external_draft"
    PROMPT_LOOKUP = "prompt_lookup"
    STREAMING = "streaming"
    TOOLS = "tools"
    REASONING = "reasoning"
    LAYERED_CACHE = "layered_cache"
    SEGMENTED_MTP = "segmented_mtp"
    GRAMMAR = "grammar"
    COMPACTION = "compaction"
    ACTIVE_RECOMPUTE = "active_recompute"


class StatePlane(str, Enum):
    ATTENTION_KV = "attention_kv"
    RECURRENT = "recurrent"
    SPARSE_INDEX = "sparse_index"
    DRAFT = "draft"
    RNG = "rng"
    GRAMMAR = "grammar"
    TRANSCRIPT = "transcript"


class Fidelity(str, Enum):
    EXACT = "exact"
    NUMERICALLY_BOUNDED = "numerically_bounded"
    APPROXIMATE = "approximate"


FIDELITY_RANK = {
    Fidelity.APPROXIMATE: 0,
    Fidelity.NUMERICALLY_BOUNDED: 1,
    Fidelity.EXACT: 2,
}


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    model_type: str
    family: str
    variant: str
    state_planes: frozenset[StatePlane]
    capabilities: frozenset[Capability]
    cache_layout: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_type or not self.family or not self.variant:
            raise ValueError("model_type, family, and variant are required")
        if Capability.APC_V2 in self.capabilities and not self.cache_layout:
            raise ValueError("APCv2 requires an explicit cache layout")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def key(self) -> str:
        return f"{self.model_type}:{self.variant}"


@dataclass(frozen=True, slots=True)
class QualifiedProfile:
    name: str
    capabilities: frozenset[Capability]
    fidelity: Fidelity
    evidence: tuple[str, ...]
    implementation: str

    def __post_init__(self) -> None:
        if not self.name or not self.implementation:
            raise ValueError("profile name and implementation are required")
        if not self.evidence:
            raise ValueError("a selectable profile requires qualification evidence")


@dataclass(frozen=True, slots=True)
class RouteRequest:
    model_type: str
    variant: str
    required: frozenset[Capability]
    minimum_fidelity: Fidelity = Fidelity.EXACT

    @property
    def model_key(self) -> str:
        return f"{self.model_type}:{self.variant}"
