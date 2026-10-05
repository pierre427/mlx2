"""Authoritative geometry and backend contract for ragged verification.

The scheduler, target adapter, cache owner, and route receipt must agree on one
description of a verification round.  A draft depth ``K`` consumes ``K + 1``
target rows: the current token plus the candidate rows needed to authorize the
accepted prefix and correction/bonus token.

This module is deliberately tensor-runtime independent.  It validates host
metadata before any cache mutation or model forward and describes how a
backend will consume the logical rows.  Numerical execution remains owned by
the model adapter and its ordinary reference path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Iterable, Mapping


class RaggedVerifyError(ValueError):
    """The requested verification geometry cannot be represented safely."""


class RaggedBackendMode(str, Enum):
    NATIVE_RAGGED = "native_ragged"
    PADDED = "padded"
    FLATTENED = "flattened"


@dataclass(frozen=True)
class RaggedVerifyCapabilities:
    """Backend capabilities that are safe for the current model/cache law."""

    native_ragged: bool = False
    padded: bool = False
    flattened: bool = False
    max_lanes: int | None = None
    max_query_len: int | None = None
    max_total_rows: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_lanes", "max_query_len", "max_total_rows"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Integral) or value < 1
            ):
                raise RaggedVerifyError(f"{name} must be a positive integer")
        if not (self.native_ragged or self.padded or self.flattened):
            raise RaggedVerifyError("a ragged verifier backend must expose one safe mode")


@dataclass(frozen=True)
class RaggedBackendPlan:
    mode: RaggedBackendMode
    logical_rows: int
    physical_rows: int
    padding_rows: int
    max_query_len: int

    def receipt(self) -> dict:
        return {
            "backend_mode": self.mode.value,
            "logical_rows": self.logical_rows,
            "physical_rows": self.physical_rows,
            "padding_rows": self.padding_rows,
            "max_query_len": self.max_query_len,
        }


def _integers(values: Iterable[int], *, label: str, minimum: int) -> tuple[int, ...]:
    result = tuple(values)
    if not result or any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < minimum
        for value in result
    ):
        raise RaggedVerifyError(
            f"{label} requires one integer >= {minimum} per live lane"
        )
    return tuple(int(value) for value in result)


@dataclass(frozen=True)
class RaggedVerifyLayout:
    """One immutable logical verification layout for the current membership."""

    lane_uids: tuple[int, ...]
    draft_depths: tuple[int, ...]
    query_lengths: tuple[int, ...]
    cu_query_lengths: tuple[int, ...]
    max_query_len: int
    right_padding: tuple[int, ...]
    token_to_lane: tuple[int, ...]

    @classmethod
    def from_draft_depths(
        cls, lane_uids: Iterable[int], draft_depths: Iterable[int]
    ) -> "RaggedVerifyLayout":
        uids = _integers(lane_uids, label="lane_uids", minimum=0)
        depths = _integers(draft_depths, label="draft_depths", minimum=0)
        if len(uids) != len(depths):
            raise RaggedVerifyError("lane_uids and draft_depths must have equal length")
        if len(set(uids)) != len(uids):
            raise RaggedVerifyError("ragged verification requires unique lane_uids")
        lengths = tuple(depth + 1 for depth in depths)
        width = max(lengths)
        offsets = [0]
        mapping = []
        for row, length in enumerate(lengths):
            offsets.append(offsets[-1] + length)
            mapping.extend([row] * length)
        return cls(
            lane_uids=uids,
            draft_depths=depths,
            query_lengths=lengths,
            cu_query_lengths=tuple(offsets),
            max_query_len=width,
            right_padding=tuple(width - length for length in lengths),
            token_to_lane=tuple(mapping),
        )

    @property
    def lane_count(self) -> int:
        return len(self.lane_uids)

    @property
    def logical_rows(self) -> int:
        return self.cu_query_lengths[-1]

    @property
    def padded_rows(self) -> int:
        return self.lane_count * self.max_query_len

    @property
    def is_ragged(self) -> bool:
        return len(set(self.query_lengths)) > 1

    def lane_slice(self, lane_index: int) -> slice:
        if (
            isinstance(lane_index, bool)
            or not isinstance(lane_index, Integral)
            or not 0 <= lane_index < self.lane_count
        ):
            raise RaggedVerifyError("lane index is outside the current layout")
        return slice(
            self.cu_query_lengths[lane_index],
            self.cu_query_lengths[lane_index + 1],
        )

    def validate_acceptance(self, accepted_lengths: Iterable[int]) -> tuple[int, ...]:
        accepted = tuple(accepted_lengths)
        if len(accepted) != self.lane_count or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in accepted
        ):
            raise RaggedVerifyError("accepted_lengths must cover every live lane")
        accepted = tuple(int(value) for value in accepted)
        if any(
            value < 0 or value > depth
            for value, depth in zip(accepted, self.draft_depths, strict=True)
        ):
            raise RaggedVerifyError("accepted prefix exceeds its proposed draft depth")
        return accepted

    def select_backend(
        self, capabilities: RaggedVerifyCapabilities
    ) -> RaggedBackendPlan:
        limits = {
            "lane count": (self.lane_count, capabilities.max_lanes),
            "query length": (self.max_query_len, capabilities.max_query_len),
            "logical rows": (self.logical_rows, capabilities.max_total_rows),
        }
        for label, (actual, maximum) in limits.items():
            if maximum is not None and actual > maximum:
                raise RaggedVerifyError(
                    f"ragged verify {label} {actual} exceeds backend limit {maximum}"
                )
        if capabilities.native_ragged:
            mode = RaggedBackendMode.NATIVE_RAGGED
            physical = self.logical_rows
        elif capabilities.padded:
            mode = RaggedBackendMode.PADDED
            physical = self.padded_rows
        elif capabilities.flattened:
            mode = RaggedBackendMode.FLATTENED
            physical = self.logical_rows
        else:  # pragma: no cover - constructor already makes this unreachable.
            raise RaggedVerifyError("no backend mode can consume the layout")
        return RaggedBackendPlan(
            mode=mode,
            logical_rows=self.logical_rows,
            physical_rows=physical,
            padding_rows=physical - self.logical_rows,
            max_query_len=self.max_query_len,
        )

    def receipt(self, plan: RaggedBackendPlan) -> dict:
        if plan.logical_rows != self.logical_rows or plan.max_query_len != self.max_query_len:
            raise RaggedVerifyError("backend plan does not belong to this layout")
        return {
            "schema": "mlx2.ragged-verify-layout.v1",
            "lane_uids": list(self.lane_uids),
            "draft_depths": list(self.draft_depths),
            "query_lengths": list(self.query_lengths),
            "cu_query_lengths": list(self.cu_query_lengths),
            "right_padding": list(self.right_padding),
            "ragged": self.is_ragged,
            **plan.receipt(),
        }


def padded_capabilities(**limits: int | None) -> RaggedVerifyCapabilities:
    """Current exact MLX segmented-cache ABI: right-padded query rows."""

    return RaggedVerifyCapabilities(padded=True, **limits)


def capabilities_from_mapping(values: Mapping[str, object]) -> RaggedVerifyCapabilities:
    """Strict adapter boundary for capability manifests and source-bound profiles."""

    allowed = {
        "native_ragged",
        "padded",
        "flattened",
        "max_lanes",
        "max_query_len",
        "max_total_rows",
    }
    unknown = set(values) - allowed
    if unknown:
        raise RaggedVerifyError(f"unknown ragged backend capabilities: {sorted(unknown)}")
    for name in ("native_ragged", "padded", "flattened"):
        if name in values and type(values[name]) is not bool:
            raise RaggedVerifyError(f"{name} must be boolean")
    return RaggedVerifyCapabilities(**values)


__all__ = [
    "RaggedBackendMode",
    "RaggedBackendPlan",
    "RaggedVerifyCapabilities",
    "RaggedVerifyError",
    "RaggedVerifyLayout",
    "capabilities_from_mapping",
    "padded_capabilities",
]
