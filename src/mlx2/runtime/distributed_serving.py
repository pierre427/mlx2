"""Fail-closed contracts for adapter-owned two-rank model execution.

This module deliberately contains no scheduler or model-name policy.  It binds
one serving request to an exact cluster, artifact, route and state revision,
then validates the ordered messages exchanged by a transport implementation.
The first transport may be MLX ring, but the ownership rules are independent
of that choice.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

SCHEMA = "mlx2.distributed-serving.v1"


def _text(name: str, value: object, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ValueError(f"{name} must be nonempty text of at most {limit} characters")
    return value


def _digest(name: str, value: object) -> str:
    value = _text(name, value, limit=64)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class ClusterIdentity:
    cluster_id: str
    source_revision: str
    artifact_sha256: str
    tokenizer_sha256: str
    route: str
    world_size: int = 2

    def __post_init__(self) -> None:
        _text("cluster_id", self.cluster_id)
        _text("source_revision", self.source_revision)
        _digest("artifact_sha256", self.artifact_sha256)
        _digest("tokenizer_sha256", self.tokenizer_sha256)
        _text("route", self.route)
        if type(self.world_size) is not int or self.world_size != 2:
            raise ValueError("the initial distributed serving contract requires two ranks")

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical(self.as_dict())).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "cluster_id": self.cluster_id,
            "source_revision": self.source_revision,
            "artifact_sha256": self.artifact_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "route": self.route,
            "world_size": self.world_size,
        }


@dataclass(frozen=True)
class RequestIdentity:
    request_id: str
    tenant_sha256: str
    cluster_sha256: str
    state_revision: int
    max_tokens: int

    def __post_init__(self) -> None:
        _text("request_id", self.request_id)
        _digest("tenant_sha256", self.tenant_sha256)
        _digest("cluster_sha256", self.cluster_sha256)
        if type(self.state_revision) is not int or self.state_revision < 0:
            raise ValueError("state_revision must be a nonnegative integer")
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 1_000_000:
            raise ValueError("max_tokens must be an integer in [1, 1000000]")

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical(self.as_dict())).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "tenant_sha256": self.tenant_sha256,
            "cluster_sha256": self.cluster_sha256,
            "state_revision": self.state_revision,
            "max_tokens": self.max_tokens,
        }


class Phase(str, Enum):
    ADMIT = "admit"
    ACK = "ack"
    STEP = "step"
    CANCEL = "cancel"
    COMMIT = "commit"
    ABORT = "abort"


@dataclass(frozen=True)
class RankMessage:
    phase: Phase
    request_sha256: str
    sequence: int
    sender_rank: int
    state_revision: int
    payload_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.phase, Phase):
            raise TypeError("phase must be a Phase")
        _digest("request_sha256", self.request_sha256)
        _digest("payload_sha256", self.payload_sha256)
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("sequence must be a nonnegative integer")
        if type(self.sender_rank) is not int or self.sender_rank not in (0, 1):
            raise ValueError("sender_rank must be 0 or 1")
        if type(self.state_revision) is not int or self.state_revision < 0:
            raise ValueError("state_revision must be a nonnegative integer")


class RequestTransaction:
    """Validate rank messages and publish state only after both ranks agree."""

    def __init__(self, identity: RequestIdentity) -> None:
        self.identity = identity
        self._phase: Phase | None = None
        self._next_sequence = 0
        self._step = 0
        self._terminal = False
        self._rank_ack: set[int] = set()
        self._last_step_payload: dict[int, str] = {}

    @property
    def terminal(self) -> bool:
        return self._terminal

    def _protocol_error(self, reason: str) -> None:
        """Poison the transaction before exposing a protocol failure."""
        self._terminal = True
        self._phase = Phase.ABORT
        self._last_step_payload.clear()
        raise RuntimeError(reason)

    def apply(self, message: RankMessage) -> dict[str, object] | None:
        if self._terminal:
            raise RuntimeError("distributed request is already terminal")
        if message.request_sha256 != self.identity.digest:
            self._protocol_error("message request identity differs from admission")
        if message.state_revision != self.identity.state_revision + self._step:
            self._protocol_error(
                "message state revision differs from the owned revision"
            )
        if message.sequence != self._next_sequence:
            self._protocol_error(
                "out-of-order, duplicated or missing distributed message"
            )
        self._next_sequence += 1

        if message.phase is Phase.ADMIT:
            if self._phase is not None or message.sender_rank != 0:
                self._protocol_error("only rank 0 may begin admission")
            self._phase = Phase.ADMIT
            return None
        if message.phase is Phase.ACK:
            if self._phase not in (Phase.ADMIT, Phase.ACK):
                self._protocol_error(
                    "rank acknowledgement is not currently admissible"
                )
            if message.sender_rank in self._rank_ack:
                self._protocol_error("rank acknowledged admission twice")
            self._rank_ack.add(message.sender_rank)
            self._phase = Phase.ACK
            return None
        if message.phase is Phase.STEP:
            if self._rank_ack != {0, 1} or self._phase not in (Phase.ACK, Phase.STEP):
                self._protocol_error(
                    "a model step requires admission from both ranks"
                )
            if self._step >= self.identity.max_tokens:
                self._terminal = True
                self._phase = Phase.ABORT
                raise RuntimeError("distributed request exceeded its admitted token limit")
            if message.sender_rank in self._last_step_payload:
                self._protocol_error("rank submitted the same model step twice")
            self._last_step_payload[message.sender_rank] = message.payload_sha256
            self._phase = Phase.STEP
            if set(self._last_step_payload) != {0, 1}:
                return None
            if len(set(self._last_step_payload.values())) != 1:
                self._terminal = True
                self._phase = Phase.ABORT
                raise RuntimeError("rank outputs disagree; transaction aborted")
            payload = self._last_step_payload[0]
            self._last_step_payload.clear()
            self._step += 1
            return {
                "schema": SCHEMA,
                "event": "step_committed",
                "request_sha256": self.identity.digest,
                "state_revision": self.identity.state_revision + self._step,
                "payload_sha256": payload,
            }
        if message.phase in (Phase.CANCEL, Phase.ABORT):
            # Either rank may report a fault or observe cancellation. No state
            # becomes publishable after this point.
            self._terminal = True
            self._phase = message.phase
            self._last_step_payload.clear()
            return {
                "schema": SCHEMA,
                "event": message.phase.value,
                "request_sha256": self.identity.digest,
                "published_state_revision": None,
            }
        if message.phase is Phase.COMMIT:
            if message.sender_rank != 0 or self._last_step_payload:
                self._protocol_error(
                    "only rank 0 may commit at a completed step boundary"
                )
            if self._rank_ack != {0, 1}:
                self._protocol_error("commit requires both admitted ranks")
            self._terminal = True
            self._phase = Phase.COMMIT
            return {
                "schema": SCHEMA,
                "event": "request_committed",
                "request_sha256": self.identity.digest,
                "published_state_revision": self.identity.state_revision + self._step,
                "steps": self._step,
            }
        raise AssertionError("unreachable distributed phase")


def admission_receipt(
    cluster: ClusterIdentity,
    request: RequestIdentity,
    *,
    rank_capabilities: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    """Create a receipt only when both ranks advertise one exact route."""

    if request.cluster_sha256 != cluster.digest:
        raise RuntimeError("request names a different distributed cluster")
    if set(rank_capabilities) != {0, 1}:
        raise RuntimeError("both rank capability receipts are required")
    normalized: dict[int, dict[str, object]] = {}
    for rank, raw in rank_capabilities.items():
        if not isinstance(raw, Mapping):
            raise TypeError("rank capability receipt must be an object")
        if raw.get("cluster_sha256") != cluster.digest:
            raise RuntimeError(f"rank {rank} has a foreign cluster identity")
        if raw.get("route") != cluster.route:
            raise RuntimeError(f"rank {rank} does not implement the requested route")
        capacity = raw.get("available_bytes")
        reserved = raw.get("reserved_bytes")
        if (
            type(capacity) is not int
            or type(reserved) is not int
            or capacity < 0
            or reserved < 0
            or reserved > capacity
        ):
            raise ValueError(f"rank {rank} has invalid memory admission fields")
        normalized[rank] = {
            "cluster_sha256": cluster.digest,
            "route": cluster.route,
            "available_bytes": capacity,
            "reserved_bytes": reserved,
        }
    return {
        "schema": SCHEMA,
        "admitted": True,
        "qualified": False,
        "cluster_sha256": cluster.digest,
        "request_sha256": request.digest,
        "state_revision": request.state_revision,
        "rank_capabilities": normalized,
    }
