"""CPU-only policy model for an external, exact-model mlx2 supervisor.

This is a research preflight, not a server or a production route.  Fake children
exercise admission, ownership and failure ordering without importing MLX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Callable, Mapping


class Refused(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    artifact: str
    runtime: Mapping[str, object]
    settings: Mapping[str, object]
    route_receipt: str
    resident_bytes: int
    apc_dir: str
    capabilities: frozenset[str]


@dataclass
class ChildRecord:
    child: object
    generation: int
    state: str
    active_leases: set[int]
    last_used: float

    @property
    def pins(self) -> int:
        return len(self.active_leases)


@dataclass(frozen=True)
class RequestLease:
    model_id: str
    generation: int
    lease_id: int
    child: object
    route_receipt: str
    route: str
    runtime: Mapping[str, object]
    artifact: str
    apc_dir: str

    def receipt(self) -> dict[str, object]:
        return {
            "requested_model": self.model_id,
            "served_model": self.model_id,
            "child_generation": self.generation,
            "child_route_receipt": self.route_receipt,
            "child_route": self.route,
            "child_runtime": dict(self.runtime),
            "child_artifact": self.artifact,
            "apc_dir": self.apc_dir,
        }


@dataclass
class SupervisorPreflight:
    catalog: Mapping[str, ModelSpec]
    budget_bytes: int
    spawn: Callable[[ModelSpec], object]
    clock: Callable[[], float]
    api_key: str
    idle_ttl_seconds: float = 180.0
    evict_to_fit: bool = False
    _lock: RLock = field(default_factory=RLock, init=False)
    _children: dict[str, ChildRecord] = field(default_factory=dict, init=False)
    _generations: dict[str, int] = field(default_factory=dict, init=False)
    _crashes: dict[str, list[float]] = field(default_factory=dict, init=False)
    _oom_failed: set[str] = field(default_factory=set, init=False)
    _draining: set[str] = field(default_factory=set, init=False)
    _next_lease_id: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.budget_bytes <= 0 or not self.api_key:
            raise ValueError("positive budget and nonempty key required")
        if set(self.catalog) != {spec.model_id for spec in self.catalog.values()}:
            raise ValueError("catalog keys must be exact model IDs")
        if len({spec.apc_dir for spec in self.catalog.values()}) != len(self.catalog):
            raise ValueError("each child needs a private APCv2 directory")
        if any(spec.resident_bytes <= 0 for spec in self.catalog.values()):
            raise ValueError("resident estimates must be positive")

    def _validate_child(self, spec: ModelSpec, child: object) -> None:
        status = child.status()
        expected = {
            "model": spec.model_id,
            "artifact": spec.artifact,
            "runtime": dict(spec.runtime),
            "settings": dict(spec.settings),
            "route_receipt": spec.route_receipt,
            "qualification": "qualified",
        }
        if not status.get("healthy") or any(status.get(k) != v for k, v in expected.items()):
            raise Refused("child_identity_or_qualification_mismatch")
        qualified = set(status.get("qualified_capabilities") or ())
        selected = set(status.get("selected_capabilities") or ())
        if not spec.capabilities <= qualified & selected:
            raise Refused("child_capability_mismatch")

    def _evict(self, model_id: str) -> None:
        record = self._children[model_id]
        if record.pins:
            raise Refused("child_pinned")
        record.state = "draining"
        record.child.close()
        del self._children[model_id]

    def _load(self, spec: ModelSpec) -> ChildRecord:
        if spec.model_id in self._oom_failed:
            raise Refused("child_oom_failed")
        now = self.clock()
        recent = [t for t in self._crashes.get(spec.model_id, ()) if now - t < 120]
        if len(recent) >= 3:
            raise Refused("child_restart_exhausted")
        if recent and now - recent[-1] < min(30, 2 ** (len(recent) - 1)):
            raise Refused("child_restart_backoff")
        if spec.resident_bytes > self.budget_bytes:
            raise Refused("model_exceeds_budget")
        # A draining child still owns its weights until it actually exits.
        live = dict(self._children)
        used = sum(self.catalog[name].resident_bytes for name in live)
        need = used + spec.resident_bytes - self.budget_bytes
        victims = []
        if need > 0:
            if not self.evict_to_fit:
                raise Refused("insufficient_memory")
            for name, row in sorted(live.items(), key=lambda item: item[1].last_used):
                if row.pins == 0:
                    victims.append(name)
                    need -= self.catalog[name].resident_bytes
                    if need <= 0:
                        break
            if need > 0:
                raise Refused("all_eviction_candidates_pinned")
        for name in victims:
            self._evict(name)
        child = self.spawn(spec)
        try:
            self._validate_child(spec, child)
        except BaseException:
            child.close()
            raise
        generation = self._generations.get(spec.model_id, 0) + 1
        self._generations[spec.model_id] = generation
        record = ChildRecord(child, generation, "ready", set(), now)
        self._children[spec.model_id] = record
        return record

    def admit(
        self,
        model_id: str,
        *,
        credential: str,
        body_reader: Callable[[int], bytes] | None = None,
        max_body_bytes: int = 2 << 20,
        required_capabilities: frozenset[str] = frozenset(),
    ) -> RequestLease:
        # Authentication and model selection precede body I/O or child spawn.
        if credential != self.api_key:
            raise Refused("unauthorized")
        spec = self.catalog.get(model_id) if isinstance(model_id, str) else None
        if spec is None:
            raise Refused("unknown_model")
        if not required_capabilities <= spec.capabilities:
            raise Refused("unsupported_capability")
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        if body_reader is not None:
            # A real proxy must cap the read itself, not allocate an unlimited
            # body and only check its length afterward.
            if len(body_reader(max_body_bytes + 1)) > max_body_bytes:
                raise Refused("request_too_large")
        with self._lock:
            if model_id in self._draining:
                raise Refused("child_draining_or_failed")
            record = self._children.get(model_id)
            if record is not None and record.state != "ready":
                raise Refused("child_draining_or_failed")
            if record is None:
                record = self._load(spec)
            else:
                self._validate_child(spec, record.child)
            self._next_lease_id += 1
            lease_id = self._next_lease_id
            record.active_leases.add(lease_id)
            record.last_used = self.clock()
            return RequestLease(
                model_id, record.generation, lease_id, record.child, spec.route_receipt,
                str(spec.settings["route"]),
                spec.runtime, spec.artifact, spec.apc_dir,
            )

    def validate_response(
        self, lease: RequestLease, *, response_model: str, child_receipt: Mapping[str, object]
    ) -> None:
        """Check the observed child receipt, not just its startup status."""
        with self._lock:
            record = self._children.get(lease.model_id)
            if record is None or record.generation != lease.generation:
                raise Refused("stale_child_generation")
            if (
                response_model != lease.model_id
                or child_receipt.get("route_receipt") != lease.route_receipt
                or child_receipt.get("route") != lease.route
                or child_receipt.get("qualification") != "qualified"
            ):
                raise Refused("child_receipt_mismatch")

    def finish(self, lease: RequestLease) -> bool:
        """Release only the exact live generation; stale streams cannot unpin it."""
        with self._lock:
            record = self._children.get(lease.model_id)
            if record is None or record.generation != lease.generation:
                return False
            if lease.lease_id not in record.active_leases:
                raise ValueError("lease already finished")
            record.active_leases.remove(lease.lease_id)
            record.last_used = self.clock()
            return True

    def drain(self, model_id: str) -> bool:
        with self._lock:
            if model_id not in self.catalog:
                raise Refused("unknown_model")
            self._draining.add(model_id)
            record = self._children.get(model_id)
            if record is None:
                return True
            record.state = "draining"
            return record.pins == 0

    def resume(self, model_id: str) -> None:
        with self._lock:
            if model_id not in self.catalog:
                raise Refused("unknown_model")
            if model_id in self._children:
                raise Refused("child_must_unload_before_resume")
            self._draining.discard(model_id)

    def unload(self, model_id: str) -> None:
        with self._lock:
            if model_id in self._children:
                self._evict(model_id)

    def sweep_idle(self) -> tuple[str, ...]:
        with self._lock:
            now = self.clock()
            names = tuple(
                name for name, row in self._children.items()
                if row.pins == 0 and now - row.last_used >= self.idle_ttl_seconds
            )
            for name in names:
                self._evict(name)
            return names

    def crash(self, model_id: str, *, oom: bool = False) -> None:
        """Invalidate old leases. Their streams fail; no cross-child replay."""
        with self._lock:
            record = self._children.pop(model_id, None)
            if record is None:
                return
            if oom:
                self._oom_failed.add(model_id)
            else:
                self._crashes.setdefault(model_id, []).append(self.clock())
            record.state = "failed"
            record.active_leases.clear()
            record.child.close()
