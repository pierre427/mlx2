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


@dataclass
class PagedCandidateReceipt:
    """Default-off candidate counters, separate from serving qualification.

    Only a successfully completed ``AttentionUse`` with a caller-observed read
    execution can contribute rows/pages. A future native caller must supply
    real dispatch and terminal evidence; this host contract does not execute
    a Metal read.
    """

    profile: str = "dense_vector_v1"
    actual_paged_rows: int = 0
    actual_paged_pages: int = 0
    completed_reads: int = 0
    refusals: list[str] = field(default_factory=list)

    def record_refusal(self, reason: str) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("refusal reason is required")
        self.refusals.append(reason)

    def record_completed_attention(self, use: Any, *, kernel_executed: bool) -> None:
        from .runtime.paged_kv_cache import AttentionUse

        if type(kernel_executed) is not bool or not kernel_executed:
            raise ValueError("actual paged work requires executed read evidence")
        if type(use) is not AttentionUse or not use.completed_successfully or use.state != "closed":
            raise ValueError("actual paged work requires a successful terminal proof")
        if use.receipt_recorded:
            raise ValueError("attention use was already recorded")
        plan = use.plan
        if plan.profile != self.profile:
            raise ValueError("paged profile does not match receipt")
        # Count logical page-table entries actually addressed by query rows.
        pages = 0
        for span in plan.spans:
            first = span.visible_bounds(0)[0] // plan.page_size
            last = (span.visible_bounds(span.row_count - 1)[1] - 1) // plan.page_size
            pages += last - first + 1
        self.actual_paged_rows += plan.total_rows
        self.actual_paged_pages += pages
        self.completed_reads += 1
        use.receipt_recorded = True

    def record_admission(self, decision: Any) -> None:
        from .runtime.batch_admission import PagedAdmissionDecision

        if type(decision) is not PagedAdmissionDecision:
            raise TypeError("a paged admission decision is required")
        if not decision.accepted:
            self.record_refusal(decision.reason)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "implemented": True,
            "qualified": False,
            "selected": False,
            "observed_used": self.completed_reads > 0,
            "actual_paged_rows": self.actual_paged_rows,
            "actual_paged_pages": self.actual_paged_pages,
            "completed_reads": self.completed_reads,
            "refusals": list(self.refusals),
        }
