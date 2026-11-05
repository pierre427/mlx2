"""Host-bound tools for CPU proposals and trusted self-repair controllers.

Remote callers select registered identifiers. They cannot provide paths, a
controller, a validation verdict, a tenant, or deployment authority. Reporting
a failure is untrusted evidence, never proof that a repair is necessary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .asset_arrays import bounded_npz_arrays
from .programs import (
    WeightProgram,
    analytical_directional_repair,
    create_program,
    inspect_program,
    load_program,
)
from .tool_experiments import ExperimentTools

CONFIG_SCHEMA = "mlx2.weight-space-tool-service.v1"
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_RESOURCES = 128
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_FAILURE_FIELDS = frozenset({"tool_name", "tool_schema_digest", "failed_arguments", "corrected_arguments",
                             "error_kind", "observed_error"})


def _json(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode()) > MAX_JSON_BYTES:
        raise ValueError("tool metadata exceeds its byte bound")
    return encoded


def _identifier(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("expected a bounded registered identifier")
    return value


def _array(value, dimensions):
    raw = np.asarray(value)
    if raw.ndim != dimensions or raw.size < 1 or raw.size > 16 * 1024 * 1024 or raw.dtype.kind not in "fiu":
        raise ValueError("capture tensor has invalid or excessive geometry")
    with np.errstate(over="ignore"):
        array = np.asarray(raw, dtype="<f4", order="C")
    if not np.isfinite(array).all():
        raise ValueError("capture tensors must be finite float32 values")
    return np.frombuffer(array.tobytes(), dtype="<f4").reshape(array.shape)


@dataclass(frozen=True)
class RegisteredCapture:
    """Host-provided weights and unit directions; never inferred from prose."""

    capture_id: str
    checkpoint: str
    source_revision: str
    weights: Mapping
    directions: Mapping
    selected_columns: Mapping
    data_scope: str = "session"
    activations: Mapping | None = None
    reference_activations: Mapping | None = None
    contrast_activations: Mapping | None = None
    target_metadata: Mapping | None = None
    reference_inputs: Mapping | None = None
    contrast_inputs: Mapping | None = None

    def __post_init__(self):
        _identifier(self.capture_id)
        if self.data_scope not in {"shared", "session", "tenant"}:
            raise ValueError("capture data_scope must be shared, session, or tenant")
        if (not self.weights or len(self.weights) > 32 or set(self.weights) != set(self.directions)
                or set(self.weights) != set(self.selected_columns)):
            raise ValueError("capture weights, directions, and columns must cover 1..32 identical targets")
        weights = {key: _array(value, 2) for key, value in self.weights.items()}
        directions = {key: _array(value, 1) for key, value in self.directions.items()}
        columns = {key: tuple(value) for key, value in self.selected_columns.items()}
        activations = {key: _array(value, 2) for key, value in (self.activations or {}).items()}
        if (set(activations) - set(weights)
                or any(value.shape[1] != weights[key].shape[1] for key, value in activations.items())):
            raise ValueError("capture input activations must match registered projection widths")
        groups = {name: {key: _array(value, 2) for key, value in (getattr(self, name) or {}).items()}
                  for name in ("reference_activations", "contrast_activations", "reference_inputs", "contrast_inputs")}
        metadata = dict(self.target_metadata or {})
        if set(metadata) != set(groups["reference_activations"]) or set(metadata) != set(groups["contrast_activations"]):
            raise ValueError("contrastive captures require matching reference, contrast, and target metadata")
        if set(metadata) - set(weights):
            raise ValueError("contrastive target is not registered")
        for target, spec in metadata.items():
            if (not isinstance(spec, dict) or set(spec) != {"layer", "component"}
                    or type(spec["layer"]) is not int or not 0 <= spec["layer"] <= 10000
                    or spec["component"] not in {"mlp_output", "attention_output"}):
                raise ValueError("abliteration requires explicit output component and layer metadata")
            for name in ("reference_activations", "contrast_activations"):
                values = groups[name][target]
                if values.shape[1] != weights[target].shape[0] or len(values) > 512:
                    raise ValueError("contrastive output capture geometry differs from its target")
        if set(groups["reference_inputs"]) != set(groups["contrast_inputs"]):
            raise ValueError("retention needs both input capture groups")
        if set(groups["reference_inputs"]) - set(metadata):
            raise ValueError("retention input target lacks contrastive metadata")
        for name in ("reference_inputs", "contrast_inputs"):
            for target, values in groups[name].items():
                if values.shape[1] != weights[target].shape[1] or len(values) > 512:
                    raise ValueError("retention input geometry differs from its target")
        all_arrays = (*weights.values(), *activations.values(), *(v for group in groups.values() for v in group.values()))
        if sum(value.nbytes for value in all_arrays) > MAX_FILE_BYTES:
            raise ValueError("capture weights exceed aggregate byte budget")
        # Validate all checkpoint, target, direction and column contracts once
        # at host registration, without reading or loading any model.
        analytical_directional_repair(weights, directions, columns, strength=0,
                                      checkpoint=self.checkpoint, source_revision=self.source_revision,
                                      provenance={"operation": "validate-host-capture"})
        object.__setattr__(self, "weights", MappingProxyType(weights))
        object.__setattr__(self, "directions", MappingProxyType(directions))
        object.__setattr__(self, "selected_columns", MappingProxyType(columns))
        object.__setattr__(self, "activations", MappingProxyType(activations))
        for name, group in groups.items():
            object.__setattr__(self, name, MappingProxyType(group))
        object.__setattr__(self, "target_metadata", MappingProxyType({key: MappingProxyType(dict(value)) for key, value in metadata.items()}))

    @property
    def digest(self):
        metadata = {"id": self.capture_id, "checkpoint": self.checkpoint,
                    "source_revision": self.source_revision, "data_scope": self.data_scope,
                    "columns": dict(self.selected_columns),
                    "target_metadata": {key: dict(value) for key, value in self.target_metadata.items()}}
        digest = hashlib.sha256(_json(metadata).encode())
        for target in sorted(self.weights):
            digest.update(target.encode())
            for array in (self.weights[target], self.directions[target]):
                digest.update(_json({"shape": array.shape, "dtype": "float32"}).encode())
                digest.update(array.tobytes())
            if target in self.activations:
                array = self.activations[target]
                digest.update(_json({"kind": "activations", "shape": array.shape}).encode())
                digest.update(array.tobytes())
            for name in ("reference_activations", "contrast_activations", "reference_inputs", "contrast_inputs"):
                group = getattr(self, name)
                if target in group:
                    array = group[target]
                    digest.update(_json({"kind": name, "shape": array.shape}).encode())
                    digest.update(array.tobytes())
        return digest.hexdigest()


class WeightSpaceToolService(ExperimentTools):
    """One host-authorized session and optional trusted runtime controller.

    The default is offline CPU mode. Setting ``allow_runtime`` is a host-only
    decision; callers cannot switch it through a tool. A supplied controller
    owns validators, preservation suites, durable attempts and deployment CAS.
    """

    def __init__(self, *, session_id="offline", controller=None, allow_runtime=False,
                 captures=(), programs=None, maximum_factor_rank=32, proposal_data_scope="session",
                 maximum_registered_bytes=256 * 1024 * 1024,
                 maximum_analysis_workspace_bytes=512 * 1024 * 1024):
        self.session_id = _identifier(session_id)
        if not isinstance(allow_runtime, bool) or (allow_runtime and controller is None):
            raise ValueError("runtime mode requires a host-registered controller")
        self._controller = controller
        self._allow_runtime = allow_runtime
        if type(maximum_factor_rank) is not int or not 1 <= maximum_factor_rank <= 128:
            raise ValueError("host factor rank budget must be in 1..128")
        self._maximum_factor_rank = maximum_factor_rank
        if proposal_data_scope not in {"shared", "session", "tenant"}:
            raise ValueError("host proposal_data_scope must be shared, session, or tenant")
        self._proposal_data_scope = proposal_data_scope
        self._captures = {}
        self._programs = {}
        self._reports = {}
        self._lock = threading.RLock()
        if type(maximum_registered_bytes) is not int or not 1 <= maximum_registered_bytes <= 1024**3:
            raise ValueError("host registered tensor byte budget must be positive and at most 1GiB")
        self._maximum_registered_bytes, self._registered_bytes = maximum_registered_bytes, 0
        if type(maximum_analysis_workspace_bytes) is not int or not 1 <= maximum_analysis_workspace_bytes <= 4 * 1024**3:
            raise ValueError("host analysis workspace admission budget must be positive and at most 4GiB")
        self._maximum_analysis_workspace_bytes = maximum_analysis_workspace_bytes
        self._init_experiments()
        for capture in captures:
            self.register_capture(capture)
        for name, program in (programs or {}).items():
            self.register_program(name, program)

    def _check_registry_bytes(self, additional):
        if type(additional) is not int or additional < 0 or self._registered_bytes + additional > self._maximum_registered_bytes:
            raise ValueError("aggregate registered tensor byte budget exceeded")

    def _check_analysis_workspace(self, estimate):
        if type(estimate) is not int or estimate < 0 or estimate > self._maximum_analysis_workspace_bytes:
            raise ValueError("CPU analysis exceeds conservative workspace admission budget")

    def register_capture(self, capture):
        """Host registration API, intentionally absent from MCP tools."""
        if not isinstance(capture, RegisteredCapture):
            raise TypeError("expected a validated host capture")
        with self._lock:
            if capture.capture_id in self._captures or len(self._captures) >= MAX_RESOURCES:
                raise ValueError("capture identity exists or registry is full")
            size = sum(array.nbytes for group in (capture.weights, capture.directions, capture.activations,
                                                  capture.reference_activations, capture.contrast_activations,
                                                  capture.reference_inputs, capture.contrast_inputs)
                       for array in group.values())
            self._check_registry_bytes(size)
            self._captures[capture.capture_id] = capture
            self._registered_bytes += size

    def register_program(self, program_id, program):
        """Host registration API; validate content even for direct dataclasses."""
        _identifier(program_id)
        if not isinstance(program, WeightProgram):
            raise TypeError("expected a weight program")
        with self._lock:
            existing = self._programs.get(program_id)
            if existing is not None and existing.program_id != program.program_id:
                raise ValueError("registered program identity cannot be replaced")
            if existing is not None:
                return
            if existing is None and len(self._programs) >= MAX_RESOURCES:
                raise ValueError("program registry is full")
            size = sum((factor.a.size + factor.b.size) * 4 for factor in program.factors)
            self._check_registry_bytes(size)
            program = WeightProgram.from_dict(program.to_dict())
            self._programs[program_id] = program
            self._registered_bytes += size

    def capabilities(self):
        with self._lock:
            mode = "trusted_runtime" if self._allow_runtime else (
                "host_experiments" if any(manager is not None for manager in
                                           (self._overlay_manager, self._excision_manager, self._expert_mask_manager)) else "offline_cpu")
            return {"schema": CONFIG_SCHEMA, "mode": mode,
                    "session_id": self.session_id, "capture_ids": sorted(self._captures),
                    "program_ids": sorted(self._programs), "runtime_callbacks_registered": self._controller is not None,
                    "runtime_trials_enabled": self._allow_runtime, "caller_validation_flags_accepted": False,
                    "failure_reports_are_untrusted": True,
                    "maximum_proposed_factor_rank": self._maximum_factor_rank,
                    "model_proposal_data_scope": self._proposal_data_scope,
                    "maximum_tensor_window": [32, 32],
                    "registered_tensor_bytes": self._registered_bytes,
                    "maximum_registered_tensor_bytes": self._maximum_registered_bytes,
                    "maximum_analysis_workspace_bytes": self._maximum_analysis_workspace_bytes,
                    "analysis_workspace_is_admission_estimate_not_rss": True,
                    "tensor_budget_excludes": ["host_stream_controller_budgets", "overlay_bank_budget", "temporary_workspaces"],
                    "experiments": self.experiment_capabilities(),
                    "promotion_policy_owner": "host_controller", "offline_programs_persisted": False}

    def registered_assets(self):
        """Host-only immutable asset handoff; not registered as a model tool."""
        with self._lock:
            return tuple(self._captures.values()), dict(self._programs)

    def inspect_program(self, program_id):
        with self._lock:
            program = self._program(program_id)
            return {"registered_id": program_id, **inspect_program(program)}

    def inspect_capture(self, capture_id):
        with self._lock:
            capture = self._capture(capture_id)
            return {"capture_id": capture.capture_id, "capture_digest": capture.digest,
                    "checkpoint": capture.checkpoint, "source_revision": capture.source_revision,
                    "data_scope": capture.data_scope, "evidence_scope": "host_supplied_tensor_geometry_only",
                    "targets": [{"target": target, "weight_shape": list(weight.shape),
                                 "direction_norm": float(np.linalg.norm(capture.directions[target].astype(np.float64))),
                                 "selected_columns": list(capture.selected_columns[target]),
                                 "input_activation_shape": list(capture.activations[target].shape) if target in capture.activations else None,
                                 "abliteration_target": dict(capture.target_metadata[target]) if target in capture.target_metadata else None,
                                 "weight_frobenius_norm": float(np.linalg.norm(weight.astype(np.float64)))}
                                for target, weight in capture.weights.items()],
                    "model_loaded": False, "quality_evaluated": False}

    def inspect_tensor_window(self, capture_id, target, *, tensor="weights", row_start=0, column_start=0,
                              rows=8, columns=8):
        """Read at most 32x32 values from explicitly registered research data."""
        for value, maximum in ((row_start, None), (column_start, None), (rows, 32), (columns, 32)):
            if type(value) is not int or value < (0 if maximum is None else 1) or (maximum is not None and value > maximum):
                raise ValueError("tensor window offsets/counts are out of bounds")
        with self._lock:
            capture = self._capture(capture_id)
            if target not in capture.weights:
                raise ValueError("target is not registered for this capture")
            if tensor == "weights":
                values = capture.weights[target]
            elif tensor == "direction":
                values = capture.directions[target][None, :]
            elif tensor == "activations" and target in capture.activations:
                values = capture.activations[target]
            elif tensor in {"reference_activations", "contrast_activations", "reference_inputs", "contrast_inputs"} and target in getattr(capture, tensor):
                values = getattr(capture, tensor)[target]
            else:
                raise ValueError("requested tensor is not registered")
            if row_start >= values.shape[0] or column_start >= values.shape[1]:
                raise ValueError("tensor window starts outside the registered tensor")
            window = values[row_start:row_start + rows, column_start:column_start + columns]
            return {"capture_id": capture_id, "capture_digest": capture.digest, "target": target,
                    "tensor": tensor, "full_shape": list(values.shape), "row_start": row_start,
                    "column_start": column_start, "window_shape": list(window.shape), "values": window.tolist(),
                    "source": "host_registered_capture", "data_scope": capture.data_scope}

    @staticmethod
    def _capture_provenance(capture, operation):
        private = capture.data_scope != "shared"
        return {"operation": operation, "capture_id": capture.capture_id, "capture_digest": capture.digest,
                "data_scope": capture.data_scope, "contains_private_memory": private,
                "promotion_scope": "session" if private else "deployment"}

    def _proposal_provenance(self, capture, operation):
        provenance = self._capture_provenance(capture, operation)
        # A shared weight matrix does not make a model-authored edit public.
        # The host must bind the proposing session to shared-authority data.
        if self._proposal_data_scope != "shared":
            provenance.update(data_scope=self._proposal_data_scope, contains_private_memory=True,
                              promotion_scope="session")
        provenance["proposal_session_id"] = self.session_id
        provenance["proposal_scope_authority"] = "host_configuration"
        return provenance

    def _compiled(self, capture, program):
        self.register_program(program.program_id, program)
        return {"registered_id": program.program_id, "program_id": program.program_id,
                "capture_digest": capture.digest, "compiled": True, "qualified": False,
                "selected": False, "observed_used": False, "deployed": False,
                "evidence_scope": "factor_construction_only"}

    def propose_directional_program(self, capture_id, target, direction, selected_columns, *, strength=1.0):
        """Compile a caller-proposed unit direction against host-bound weights."""
        if not isinstance(direction, list) or len(direction) > 8192:
            raise ValueError("proposed direction must be a bounded numeric list")
        if not isinstance(selected_columns, list) or not 1 <= len(selected_columns) <= 8192:
            raise ValueError("proposed columns must be a bounded index list")
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or abs(strength) > 16:
            raise ValueError("repair strength must be finite with absolute value at most 16")
        with self._lock:
            capture = self._capture(capture_id)
            if target not in capture.weights:
                raise ValueError("proposed target is not registered")
            self._check_analysis_workspace(capture.weights[target].size * 64 + 128 * 1024)
            program = analytical_directional_repair(
                {target: capture.weights[target]}, {target: direction}, {target: selected_columns},
                strength=strength, checkpoint=capture.checkpoint, source_revision=capture.source_revision,
                provenance=self._proposal_provenance(capture, "model_proposed_direction_unvalidated"),
            )
            return self._compiled(capture, program)

    def propose_factor_program(self, capture_id, target, a, b):
        """Accept bounded low-rank factors as an unvalidated model proposal."""
        if not isinstance(a, list) or not isinstance(b, list):
            raise TypeError("factor proposals must be nested numeric lists")
        if sum(len(row) if isinstance(row, list) else 262145 for row in a + b) > 262144:
            raise ValueError("factor proposal exceeds the aggregate element bound")
        with self._lock:
            left, right = _array(a, 2), _array(b, 2)
            capture = self._capture(capture_id)
            if target not in capture.weights:
                raise ValueError("proposed target is not registered")
            weight = capture.weights[target]
            if (left.shape[0] != weight.shape[1] or right.shape[1] != weight.shape[0]
                    or left.shape[1] != right.shape[0] or left.shape[1] > self._maximum_factor_rank):
                raise ValueError("factor geometry differs from target or host rank budget")
            program = create_program(
                {target: (left, right)}, checkpoint=capture.checkpoint, source_revision=capture.source_revision,
                construction="model_proposed_low_rank", approximate=True,
                provenance=self._proposal_provenance(capture, "model_proposed_factors_unvalidated"),
            )
            return self._compiled(capture, program)

    def compile_abliteration(self, capture_id, *, peak_layer, radius, peak_strength=1.0, edge_strength=0.0,
                            orthogonalize_reference=False, winsorization_quantile=1.0,
                            row_normalization="none", full_rank=None):
        """Build an explicit contrastive projection edit from host captures.

        Retention measurements, when input captures exist, concern only these
        linear projections. They do not establish whole-model preservation.
        """
        from .heretic import (
            LayerProfile,
            compile_abliteration,
            estimate_direction,
            retention_metrics,
        )

        if full_rank is not None and (type(full_rank) is not int or not 1 <= full_rank <= self._maximum_factor_rank):
            raise ValueError("full normalization rank exceeds the host factor rank budget")
        with self._lock:
            capture = self._capture(capture_id)
            if not capture.target_metadata:
                raise ValueError("capture has no registered contrastive output targets")
            self._check_analysis_workspace(sum(weight.size for weight in capture.weights.values()) * 64
                                           + sum(array.size for group in (capture.reference_activations, capture.contrast_activations,
                                                                          capture.reference_inputs, capture.contrast_inputs)
                                                 for array in group.values()) * 80 + 128 * 1024)
            profile = LayerProfile(peak_layer, radius, peak_strength, edge_strength)
            directions = {target: estimate_direction(
                capture.reference_activations[target], capture.contrast_activations[target],
                orthogonalize_reference=orthogonalize_reference, winsorization_quantile=winsorization_quantile,
            ) for target in capture.target_metadata}
            program = compile_abliteration(
                {target: capture.weights[target] for target in capture.target_metadata}, directions,
                {target: dict(spec) for target, spec in capture.target_metadata.items()},
                profiles={spec["component"]: profile for spec in capture.target_metadata.values()},
                checkpoint=capture.checkpoint, source_revision=capture.source_revision,
                provenance=self._capture_provenance(capture, "registered_contrastive_abliteration"),
                row_normalization=row_normalization, full_rank=full_rank,
            )
            retention = {}
            for factor in program.factors:
                target = factor.target
                if target not in capture.reference_inputs:
                    retention[target] = {"measured": False, "reason": "no_registered_projection_input_pairs",
                                         "task_retention_verified": False}
                    continue
                inputs = [capture.reference_inputs[target].astype(np.float64), capture.contrast_inputs[target].astype(np.float64)]
                outputs = [values @ capture.weights[target].astype(np.float64).T for values in inputs]
                patched = [output + values @ factor.a.astype(np.float64) @ factor.b.astype(np.float64)
                           for values, output in zip(inputs, outputs)]
                retention[target] = {**retention_metrics(*outputs, *patched), "measured": True,
                                     "evidence_scope": "registered_linear_projection_inputs_only"}
            result = self._compiled(capture, program)
            return {**result, "retention_metrics": retention,
                    "direction_estimates": {target: estimate.report for target, estimate in directions.items()},
                    "behavioral_quality_evaluated": False, "ara_optimizer_executed": False}

    def report_tool_failure(self, failure):
        if not isinstance(failure, dict) or set(failure) != _FAILURE_FIELDS:
            raise ValueError("failure fields differ; caller verdicts and session overrides are forbidden")
        for key in ("tool_name", "tool_schema_digest", "error_kind", "observed_error"):
            if not isinstance(failure[key], str) or not failure[key] or len(failure[key]) > 8192:
                raise ValueError("failure labels and error must be bounded nonempty text")
        if not isinstance(failure["failed_arguments"], dict) or not isinstance(failure["corrected_arguments"], dict):
            raise TypeError("failed and corrected arguments must be objects")
        copied = json.loads(_json(failure))
        with self._lock:
            if self._controller is not None and self._allow_runtime:
                return self._controller.report_failure(copied, session_id=self.session_id)
            if len(self._reports) >= MAX_RESOURCES:
                raise ValueError("offline failure report budget exhausted")
            attempt_id = "reported-" + uuid.uuid4().hex
            receipt = {"id": attempt_id, "state": "reported_unverified", "failure": copied,
                       "session_id": self.session_id, "verified": False, "runtime_trial": False,
                       "deployment_changed": False}
            self._reports[attempt_id] = receipt
            return json.loads(_json(receipt))

    def compile_analytical_repair(self, capture_id, *, strength):
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or abs(strength) > 16:
            raise ValueError("repair strength must be finite with absolute value at most 16")
        with self._lock:
            capture = self._capture(capture_id)
            self._check_analysis_workspace(sum(weight.size for weight in capture.weights.values()) * 64 + 128 * 1024)
            program = analytical_directional_repair(
                capture.weights, capture.directions, capture.selected_columns, strength=strength,
                checkpoint=capture.checkpoint, source_revision=capture.source_revision,
                provenance=self._capture_provenance(capture, "host_registered_directional_capture"),
            )
            return self._compiled(capture, program)

    def propose_repair(self, attempt_id, program_id):
        with self._lock:
            self._owned_attempt(attempt_id)
            if len(self._programs) >= MAX_RESOURCES:
                raise ValueError("program registry cannot reserve the canonical candidate")
            result = self._runtime().propose(attempt_id, self._program(program_id))
            # The host controller binds failure/parent provenance and may
            # compose a prior deployment, producing a new program identity.
            # This path is supplied by that trusted controller, never a tool.
            canonical = load_program(result["artifact_path"])
            if canonical.program_id != result["program_id"]:
                raise ValueError("controller candidate content differs from its receipt")
            try:
                self.register_program(canonical.program_id, canonical)
            except ValueError as exc:
                # The controller already journaled the candidate; raising
                # would hide it and a retry is refused as "already has a
                # candidate". Trials load the artifact, not this registry.
                return {**result, "program_registered": False,
                        "program_registration_error": str(exc)[:1024]}
            return result

    def trial_candidate(self, attempt_id):
        with self._lock:
            self._owned_attempt(attempt_id)
            return self._runtime().trial(attempt_id)

    def get_attempt(self, attempt_id):
        with self._lock:
            _identifier(attempt_id)
            if attempt_id in self._reports:
                return json.loads(_json(self._reports[attempt_id]))
            return self._owned_attempt(attempt_id)

    def promote_validated_candidate(self, attempt_id):
        with self._lock:
            self._owned_attempt(attempt_id)
            return self._runtime().promote(attempt_id)

    def rollback_deployment(self, attempt_id, *, reason):
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 2048:
            raise ValueError("rollback requires a bounded reason")
        with self._lock:
            self._owned_attempt(attempt_id)
            return self._runtime().rollback(attempt_id, reason=reason)

    def _owned_attempt(self, attempt_id):
        attempt = self._runtime().get_attempt(_identifier(attempt_id))
        if not isinstance(attempt, dict) or attempt.get("session_id") != self.session_id:
            raise ValueError("repair attempt belongs to another host session")
        return attempt

    def _runtime(self):
        if not self._allow_runtime or self._controller is None:
            raise ValueError("offline CPU mode has no trusted runtime trial or deployment controller")
        return self._controller

    def _program(self, program_id):
        _identifier(program_id)
        if program_id not in self._programs:
            raise ValueError("program is not host-registered")
        return self._programs[program_id]

    def _capture(self, capture_id):
        _identifier(capture_id)
        if capture_id not in self._captures:
            raise ValueError("capture is not host-registered")
        return self._captures[capture_id]


def _private_file(root, relative, *, maximum=MAX_FILE_BYTES):
    """Read an operator asset under a fixed root without following symlinks."""
    if not isinstance(relative, str):
        raise TypeError("operator asset path must be text")
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(part in {"..", "."} for part in path.parts):
        raise ValueError("operator assets must use relative paths inside the configured root")
    current = root
    for part in path.parts:
        current = current / part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("operator assets must be owner-private and contain no symlinks")
    fd = os.open(current, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("operator asset is not a bounded owner-private regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            value = stream.read(maximum + 1)
        if len(value) > maximum:
            raise ValueError("operator asset exceeds byte limit")
        return value
    finally:
        os.close(fd)


def service_from_operator_config(root, config):
    """Load owner-private JSON/NPZ allowlists; cannot enable runtime callbacks."""
    supplied = Path(root).absolute()
    if supplied.is_symlink():
        raise ValueError("operator root cannot be a symlink")
    root = supplied.resolve()
    info = root.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("operator root must be an owner-private directory")
    payload = json.loads(_private_file(root, config, maximum=MAX_JSON_BYTES))
    required = {"schema", "session_id", "captures", "programs"}
    if (not isinstance(payload, dict) or not required <= set(payload)
            or set(payload) - required - {"experiments"} or payload["schema"] != CONFIG_SCHEMA):
        raise ValueError("operator configuration fields or schema differ")
    if not isinstance(payload["captures"], list) or not isinstance(payload["programs"], dict) or len(payload["captures"]) > MAX_RESOURCES or len(payload["programs"]) > MAX_RESOURCES:
        raise ValueError("operator resource registry exceeds bounds")
    service = WeightSpaceToolService(session_id=payload["session_id"])
    for item in payload["captures"]:
        if not isinstance(item, dict) or set(item) != {"capture_id", "checkpoint", "source_revision", "data_scope", "arrays", "targets"}:
            raise ValueError("operator capture fields differ")
        if not isinstance(item["targets"], dict) or not 1 <= len(item["targets"]) <= 32:
            raise ValueError("operator capture requires 1..32 targets")
        raw = _private_file(root, item["arrays"])
        arrays = bounded_npz_arrays(raw, maximum_bytes=MAX_FILE_BYTES)
        weights, directions, columns, activations = {}, {}, {}, {}
        reference, contrast, metadata, reference_inputs, contrast_inputs = {}, {}, {}, {}, {}
        for target, refs in item["targets"].items():
            allowed = {"weight", "direction", "columns", "activations", "reference", "contrast",
                       "layer", "component", "reference_inputs", "contrast_inputs"}
            if not isinstance(refs, dict) or "weight" not in refs or set(refs) - allowed:
                raise ValueError("operator target tensor references differ")
            weights[target] = arrays[refs["weight"]]
            if "reference" in refs or "contrast" in refs:
                from .heretic import estimate_direction

                if not {"reference", "contrast", "layer", "component"} <= set(refs):
                    raise ValueError("contrastive target metadata is incomplete")
                reference[target], contrast[target] = arrays[refs["reference"]], arrays[refs["contrast"]]
                metadata[target] = {"layer": refs["layer"], "component": refs["component"]}
                directions[target] = estimate_direction(reference[target], contrast[target]).direction
            if "direction" in refs:
                directions[target] = arrays[refs["direction"]]
            if target not in directions:
                raise ValueError("capture target needs a direction or contrastive captures")
            columns[target] = refs.get("columns", list(range(weights[target].shape[1])))
            if "activations" in refs:
                activations[target] = arrays[refs["activations"]]
            for name, group in (("reference_inputs", reference_inputs), ("contrast_inputs", contrast_inputs)):
                if name in refs:
                    group[target] = arrays[refs[name]]
        service.register_capture(RegisteredCapture(item["capture_id"], item["checkpoint"], item["source_revision"],
                                           weights, directions, columns, item["data_scope"], activations,
                                           reference, contrast, metadata, reference_inputs, contrast_inputs))
    for name, path in payload["programs"].items():
        program = WeightProgram.from_dict(json.loads(_private_file(root, path, maximum=MAX_FILE_BYTES)))
        service.register_program(name, program)
    if "experiments" in payload:
        from .experiment_config import configure_experiment_assets

        settings = payload["experiments"]
        if (not isinstance(settings, dict) or set(settings) != {"assets", "checkpoint", "source_revision"}
                or not isinstance(settings["checkpoint"], str) or "@" not in settings["checkpoint"]
                or not isinstance(settings["source_revision"], str) or not settings["source_revision"]):
            raise ValueError("offline experiment settings require exact source and assets binding")
        configure_experiment_assets(service, root, settings, checkpoint=settings["checkpoint"],
                                    source_revision=settings["source_revision"])
    return service
