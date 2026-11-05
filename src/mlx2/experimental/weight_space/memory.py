"""CPU calibration and revision-bound, reversible memory programs.

This fits supplied activation/residual pairs; it does not discover directions
from prose, run a model, or establish recall quality. Semantic capsules remain
authoritative. A compiled program is a derived approximation, and mounting it
only selects an immutable artifact: it does not change serving weights or prove
engagement. Contexts must come from the trusted host, never public model input.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from mlx2.runtime.hyper_directory import DirectoryContext, Scope
from mlx2.runtime.semantic_capsules import canonical_json
from mlx2.runtime.semantic_memory import SemanticMemory
from mlx2.runtime.weight_space_metrics import set_weight_space_gauge, track_weight_space

from .programs import WeightProgram, create_program

MEMORY_PROGRAM_SCHEMA = "mlx2.weight-memory-program.v1"
REVOKED_SCHEMA = "mlx2.weight-memory-revoked.v1"
# Handles owned by the authoritative store. A derived program must never
# overwrite them: SemanticMemory.load() resolves its graph through this name.
RESERVED_HANDLES = frozenset({"semantic-memory"})
MAX_CAPTURE_ELEMENTS = 2_097_152
MAX_CAPTURE_ROWS = 512
MAX_TARGETS = 32


def _text(value, label):
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError(f"{label} must be bounded nonempty text")
    return value


def _digest(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _ids(values, label):
    result = tuple(values)
    if not result or len(result) > MAX_CAPTURE_ROWS:
        raise ValueError(f"{label} must contain 1..{MAX_CAPTURE_ROWS} identifiers")
    for value in result:
        _text(value, label)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate identifiers")
    return result


def _matrix(value, label):
    array = np.asarray(value)
    if array.ndim != 2 or min(array.shape) < 1 or array.size > MAX_CAPTURE_ELEMENTS:
        raise ValueError(f"{label} exceeds bounded matrix geometry")
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite real values")
    with np.errstate(over="ignore"):
        array = np.asarray(array, dtype=np.float32, order="C")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} exceeds float32 range")
    # Bytes-backed arrays cannot be made writeable by a caller.
    return np.frombuffer(array.tobytes(), dtype=np.float32).reshape(array.shape)


@dataclass(frozen=True)
class CapturedActivations:
    target: str
    sample_ids: tuple[str, ...]
    inputs: np.ndarray
    target_deltas: np.ndarray

    def __post_init__(self):
        _text(self.target, "target")
        object.__setattr__(self, "sample_ids", _ids(self.sample_ids, "sample_ids"))
        for name in ("inputs", "target_deltas"):
            object.__setattr__(self, name, _matrix(getattr(self, name), name))
        if self.inputs.shape[0] != len(self.sample_ids) or self.target_deltas.shape[0] != len(self.sample_ids):
            raise ValueError("captured rows must match sample identifiers")


@dataclass(frozen=True)
class CalibrationSplit:
    calibration_ids: tuple[str, ...]
    heldout_ids: tuple[str, ...]

    def __post_init__(self):
        for name in ("calibration_ids", "heldout_ids"):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        if set(self.calibration_ids) & set(self.heldout_ids):
            raise ValueError("calibration and heldout identifiers must be disjoint")


@dataclass(frozen=True)
class MemorySource:
    """Claimed source binding; the live store revalidates it at publication."""

    semantic_digest: str
    session_revision: int
    context_model: str
    tenant: str
    session: str
    checkpoint: str
    tokenizer_binding: str
    runtime_binding: str
    edge_ids: tuple[str, ...]

    def __post_init__(self):
        _digest(self.semantic_digest, "semantic_digest")
        if type(self.session_revision) is not int or self.session_revision < 1:
            raise ValueError("source session_revision must be positive")
        for name in ("context_model", "tenant", "session", "checkpoint", "tokenizer_binding", "runtime_binding"):
            _text(getattr(self, name), name)
        if "@" not in self.checkpoint or not all(self.checkpoint.split("@", 1)):
            raise ValueError("checkpoint must identify artifact@fingerprint")
        edges = tuple(self.edge_ids)
        if not edges or len(edges) > 32 or len(set(edges)) != len(edges):
            raise ValueError("source requires 1..32 unique semantic edge IDs")
        for edge in edges:
            _digest(edge, "edge_id")
        object.__setattr__(self, "edge_ids", tuple(sorted(edges)))

    def to_dict(self):
        return {name: list(value) if name == "edge_ids" else value for name, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("memory source fields differ")
        return cls(**dict(value))


@dataclass(frozen=True)
class MemoryCalibration:
    program: WeightProgram
    report: dict


def _capture_digest(captures, selected_ids):
    digest = hashlib.sha256(canonical_json({"sample_ids": list(selected_ids)}))
    for capture in sorted(captures, key=lambda item: item.target):
        lookup = {sample: index for index, sample in enumerate(capture.sample_ids)}
        rows = [lookup[sample] for sample in selected_ids]
        digest.update(capture.target.encode())
        for values in (capture.inputs[rows], capture.target_deltas[rows]):
            digest.update(canonical_json({"shape": values.shape, "dtype": "float32"}))
            digest.update(values.tobytes())
    return digest.hexdigest()


def _fit(inputs, residuals, rank, ridge):
    # Ridge regression via a thin SVD avoids allocating a width-by-width map.
    u, singular, vh = np.linalg.svd(inputs.astype(np.float64), full_matrices=False)
    # ridge=0 is an unregularized pseudo-inverse: use the lstsq/pinv rank
    # cutoff, or a rounding-noise singular value swamps the fit. The ridge>0
    # path keeps every direction (ridge already bounds its weight).
    cutoff = (np.finfo(np.float64).eps * max(inputs.shape) * (singular[0] if singular.size else 0.0)
              if ridge == 0 else -1.0)
    weights = np.divide(singular, singular * singular + ridge,
                        out=np.zeros_like(singular), where=(singular > cutoff) & ((singular * singular + ridge) > 0))
    left = vh.T * weights
    right = u.T @ residuals.astype(np.float64)
    q_left, r_left = np.linalg.qr(left, mode="reduced")
    q_right, r_right = np.linalg.qr(right.T, mode="reduced")
    small_u, small_s, small_vh = np.linalg.svd(r_left @ r_right.T, full_matrices=False)
    kept = min(rank, len(small_s))
    a = (q_left @ small_u[:, :kept]) * np.sqrt(small_s[:kept])
    b = np.sqrt(small_s[:kept])[:, None] * (small_vh[:kept] @ q_right.T)
    return a.astype(np.float32), b.astype(np.float32)


def _score(inputs, residuals, a, b):
    predicted = (inputs.astype(np.float64) @ a.astype(np.float64)) @ b.astype(np.float64)
    truth = residuals.astype(np.float64)
    baseline = float(np.mean(truth * truth))
    error = float(np.mean((predicted - truth) ** 2))
    return {"rows": len(inputs), "baseline_mse": baseline, "residual_mse": error,
            "relative_improvement": (baseline - error) / baseline if baseline else (0.0 if error == 0 else -1.0)}


@track_weight_space("memory_calibrate")
def calibrate_memory_program(
    captures: Sequence[CapturedActivations], *, split: CalibrationSplit,
    source: MemorySource, source_revision: str, rank: int = 8,
    ridge: float = 1e-4, strength: float = 1.0,
) -> MemoryCalibration:
    """Fit and select one target using calibration rows only.

    Heldout rows are scored once after selection. Their error is activation
    reconstruction evidence, never a task-quality or factual-recall verdict.
    ``source_revision`` is the compiler/runtime revision, not the session CAS.
    """
    captures = tuple(captures)
    if not captures or len(captures) > MAX_TARGETS or any(not isinstance(c, CapturedActivations) for c in captures):
        raise ValueError("expected 1..32 bounded activation captures")
    if len({c.target for c in captures}) != len(captures):
        raise ValueError("candidate targets must be unique")
    if sum(c.inputs.size + c.target_deltas.size for c in captures) > MAX_CAPTURE_ELEMENTS * 4:
        raise ValueError("candidate captures exceed the aggregate element budget")
    if not isinstance(split, CalibrationSplit) or not isinstance(source, MemorySource):
        raise TypeError("explicit calibration split and memory source required")
    if type(rank) is not int or not 1 <= rank <= 1024:
        raise ValueError("rank must be in 1..1024")
    if isinstance(ridge, bool) or not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    if isinstance(strength, bool) or not math.isfinite(strength) or abs(strength) > 16:
        raise ValueError("strength must be finite with absolute value at most 16")
    expected = set(split.calibration_ids) | set(split.heldout_ids)
    if any(set(c.sample_ids) != expected for c in captures):
        raise ValueError("each candidate must cover exactly the declared split")
    fitted = []
    for capture in captures:
        lookup = {sample: index for index, sample in enumerate(capture.sample_ids)}
        rows = [lookup[sample] for sample in split.calibration_ids]
        a, b = _fit(capture.inputs[rows], capture.target_deltas[rows], rank, ridge)
        b = b * np.float32(strength)
        score = _score(capture.inputs[rows], capture.target_deltas[rows], a, b)
        fitted.append((capture, a, b, score))
    selected, a, b, _ = min(fitted, key=lambda row: (-row[3]["relative_improvement"], row[0].target))
    lookup = {sample: index for index, sample in enumerate(selected.sample_ids)}
    heldout = [lookup[sample] for sample in split.heldout_ids]
    report = {
        "schema": "mlx2.weight-memory-calibration.v1", "construction": "fitted_approximation",
        "selected_target": selected.target, "selection_split": "calibration_only",
        "calibration_digest": _capture_digest(captures, split.calibration_ids),
        "heldout_digest": _capture_digest(captures, split.heldout_ids),
        "calibration_ids": list(split.calibration_ids), "heldout_ids": list(split.heldout_ids),
        "candidates": [{"target": c.target, **score} for c, _, _, score in fitted],
        "heldout": _score(selected.inputs[heldout], selected.target_deltas[heldout], a, b),
        "requested_rank": rank, "retained_rank": int(a.shape[1]), "ridge": float(ridge), "strength": float(strength),
        "evidence_scope": "supplied_activation_reconstruction_only", "task_quality_evaluated": False,
        "qualified": False, "default_selected": False, "observed_used": False,
    }
    program = create_program(
        factors={selected.target: (a, b)}, checkpoint=source.checkpoint,
        source_revision=source_revision, construction="calibrated_memory_linear_map",
        provenance={"memory_source": source.to_dict(), "calibration": report,
                    "data_scope": "session", "contains_private_memory": True,
                    "promotion_scope": "session"}, approximate=True,
    )
    set_weight_space_gauge("memory_calibration_error_mse", report["heldout"]["residual_mse"])
    return MemoryCalibration(program, report)


def _session(context):
    if not isinstance(context, DirectoryContext) or context.request is not None:
        raise ValueError("memory programs require a trusted session context")
    context.key_for(Scope.SESSION)
    return context


def _derived_handle(handle):
    if not isinstance(handle, str) or not handle or handle in RESERVED_HANDLES:
        raise ValueError("memory program handle is empty or reserved for authoritative state")
    return handle


def _source_edges(graph, edge_ids):
    indexed = {}
    for raw in graph["edges"]:
        edge = SemanticMemory._normalize_edge(raw)
        # Normalization synchronizes a legacy field from status. Do not let
        # that promote a caller-supplied active-but-uncommitted source.
        edge["source_was_committed"] = raw.get("authority") == "committed"
        if edge["edge_id"] in indexed:
            raise ValueError("duplicate semantic source edge")
        indexed[edge["edge_id"]] = edge
    for edge_id in edge_ids:
        edge = indexed.get(edge_id)
        if edge is None or edge["status"] != "active" or not edge["source_was_committed"]:
            raise ValueError("memory source is missing, conflicting, retracted, or uncommitted")
        conflict_key = edge.get("conflict_key_digest")
        if conflict_key and any(
            other["edge_id"] != edge_id and other.get("conflict_key_digest") == conflict_key
            and other["subject"] == edge["subject"] and other["relation"] == edge["relation"]
            and other["object"] != edge["object"] and other["status"] in {"active", "contested"}
            for other in indexed.values()
        ):
            raise ValueError("memory source has an unresolved conflicting fact")


@dataclass(frozen=True)
class MemoryMount:
    program: WeightProgram
    receipt: dict


class MemoryProgramStore:
    """Publish/mount derived patches without changing authoritative facts.

    All methods require a host-authorized context. An offline MemorySource JSON
    is only a claim; snapshot, publish and mount recheck the live graph. A mount
    is a frozen selection; call validate_mount immediately before application.
    Changing the program requires fresh model state, never reused base KV.
    """

    def __init__(self, semantic_memory: SemanticMemory):
        self.memory = semantic_memory

    def snapshot(self, context, *, expected_revision, expected_semantic_digest, edge_ids):
        context = _session(context)
        with self.memory.directory.transaction():
            graph, digest, revision = self.memory.load(context)
            if digest != expected_semantic_digest or revision != expected_revision:
                raise ValueError("semantic source digest or session revision changed")
            source = MemorySource(
                digest, revision, context.model, context.tenant, context.session,
                self.memory.bindings["model_binding"], self.memory.bindings["tokenizer_binding"],
                self.memory.bindings["runtime_binding"], tuple(edge_ids),
            )
            _source_edges(graph, source.edge_ids)
            return source

    @track_weight_space("memory_publish")
    def publish(self, context, program, *, expected_revision, expires_at_ns, now_ns=None, handle="weight-memory"):
        context = _session(context)
        handle = _derived_handle(handle)
        now = time.time_ns() if now_ns is None else now_ns
        if type(now) is not int or type(expires_at_ns) is not int or now < 0 or expires_at_ns <= now:
            raise ValueError("memory program requires a future integer expiry")
        if not isinstance(program, WeightProgram):
            raise TypeError("memory publication requires a weight program")
        program = WeightProgram.from_dict(program.to_dict())
        provenance = program.provenance
        if (not program.approximate or provenance.get("data_scope") != "session"
                or provenance.get("contains_private_memory") is not True
                or provenance.get("promotion_scope") != "session"):
            raise ValueError("memory program must retain its approximate, private session scope")
        source = MemorySource.from_dict(provenance.get("memory_source"))
        with self.memory.directory.transaction():
            live = self.snapshot(context, expected_revision=expected_revision,
                                 expected_semantic_digest=source.semantic_digest, edge_ids=source.edge_ids)
            if live != source or program.checkpoint != source.checkpoint:
                raise ValueError("program memory source, scope, or checkpoint differs")
            capsule = self.memory.capsules.put(
                kind="neural_state", parents=(source.semantic_digest,),
                data={"schema": MEMORY_PROGRAM_SCHEMA, "source": source.to_dict(),
                      "directory_revision": expected_revision + 1, "expires_at_ns": expires_at_ns,
                      "handle": handle, "program": program.to_dict(), "program_id": program.program_id},
                provenance={"operation": "derived-memory-weight-program", "source_revision": program.source_revision,
                            "qualified": False, "authoritative": False}, **self.memory.bindings,
            )
            layer = self.memory.directory.update(Scope.SESSION, context,
                                                expected_revision=expected_revision, handles={handle: capsule.digest})
            return {"capsule_digest": capsule.digest, "program_id": program.program_id,
                    "directory_revision": layer["revision"], "source_semantic_digest": source.semantic_digest,
                    "published": True, "selected": False, "engaged": False, "qualified": False}

    @track_weight_space("memory_mount")
    def mount(self, context, *, checkpoint, expected_directory_revision, expected_capsule_digest,
              now_ns=None, handle="weight-memory"):
        context = _session(context)
        handle = _derived_handle(handle)
        now = time.time_ns() if now_ns is None else now_ns
        if type(now) is not int or now < 0:
            raise ValueError("mount clock must be nonnegative integer nanoseconds")
        with self.memory.directory.transaction():
            graph, semantic_digest, revision = self.memory.load(context)
            resolved = self.memory.directory.resolve(context)
            if revision != expected_directory_revision:
                raise ValueError("memory mount session revision changed")
            digest = resolved.handles.get(handle)
            if digest != expected_capsule_digest:
                raise ValueError("memory handle changed or was revoked")
            capsule = self.memory.capsules.get(digest)
            data = capsule["data"]
            if capsule["kind"] != "neural_state" or data.get("schema") != MEMORY_PROGRAM_SCHEMA:
                raise ValueError("memory program is revoked or has the wrong capsule kind")
            source = MemorySource.from_dict(data["source"])
            expected_bindings = {"model": source.checkpoint, "tokenizer": source.tokenizer_binding,
                                 "runtime": source.runtime_binding}
            if capsule["bindings"] != expected_bindings or expected_bindings != {
                "model": self.memory.bindings["model_binding"], "tokenizer": self.memory.bindings["tokenizer_binding"],
                "runtime": self.memory.bindings["runtime_binding"],
            }:
                raise ValueError("memory runtime bindings differ")
            if (source.context_model, source.tenant, source.session) != (context.model, context.tenant, context.session):
                raise ValueError("memory tenant or session scope differs")
            if source.semantic_digest != semantic_digest or source.checkpoint != checkpoint:
                raise ValueError("memory source is stale or checkpoint differs")
            if data["directory_revision"] != revision or data["handle"] != handle or source.session_revision + 1 != revision:
                raise ValueError("memory publication revision differs")
            if type(data["expires_at_ns"]) is not int or now >= data["expires_at_ns"]:
                raise ValueError("memory program expired")
            if capsule["parents"] != [source.semantic_digest]:
                raise ValueError("memory parent differs from authoritative source")
            _source_edges(graph, source.edge_ids)
            program = WeightProgram.from_dict(data["program"])
            if (program.program_id != data["program_id"] or program.checkpoint != checkpoint
                    or program.provenance.get("memory_source") != source.to_dict()
                    or not program.approximate or program.provenance.get("data_scope") != "session"
                    or program.provenance.get("contains_private_memory") is not True
                    or program.provenance.get("promotion_scope") != "session"):
                raise ValueError("memory program content or provenance differs")
            return MemoryMount(program, {
                "capsule_digest": digest, "program_id": program.program_id, "directory_revision": revision,
                "source_semantic_digest": semantic_digest, "expires_at_ns": data["expires_at_ns"], "handle": handle,
                "selected": True, "engaged": False, "observed_used": False, "qualified": False,
                "fidelity": "approximate", "requires_fresh_cache": True,
            })

    def validate_mount(self, context, mount, *, now_ns=None):
        refreshed = self.mount(context, checkpoint=mount.program.checkpoint,
                               expected_directory_revision=mount.receipt["directory_revision"],
                               expected_capsule_digest=mount.receipt["capsule_digest"],
                               handle=mount.receipt["handle"], now_ns=now_ns)
        if refreshed.program.program_id != mount.program.program_id:
            raise ValueError("mounted memory program changed")
        return refreshed

    def revoke(self, context, *, expected_revision, expected_capsule_digest, handle="weight-memory"):
        context = _session(context)
        handle = _derived_handle(handle)
        with self.memory.directory.transaction():
            _, _, revision = self.memory.load(context)
            current = self.memory.directory.resolve(context).handles.get(handle)
            if revision != expected_revision or current != expected_capsule_digest:
                raise ValueError("revocation revision or memory handle changed")
            # Only a live derived program in this exact scope can be revoked;
            # never tombstone an unrelated capsule or an absent handle.
            capsule = self.memory.capsules.get(current) if current is not None else None
            data = capsule["data"] if capsule is not None else {}
            if (capsule is None or capsule["kind"] != "neural_state"
                    or data.get("schema") != MEMORY_PROGRAM_SCHEMA or data.get("handle") != handle):
                raise ValueError("revocation target is not a derived memory program")
            source = MemorySource.from_dict(data.get("source"))
            if (source.context_model, source.tenant, source.session) != (context.model, context.tenant, context.session):
                raise ValueError("revocation target tenant or session scope differs")
            tombstone = self.memory.capsules.put(
                kind="neural_state", data={"schema": REVOKED_SCHEMA, "revoked": current},
                parents=(), provenance={"operation": "revoke-derived-memory-program"}, **self.memory.bindings,
            )
            layer = self.memory.directory.update(Scope.SESSION, context,
                                                expected_revision=expected_revision, handles={handle: tombstone.digest})
            return {"revoked": True, "directory_revision": layer["revision"], "capsule_digest": tombstone.digest}
