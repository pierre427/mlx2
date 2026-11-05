"""Owner-configured bridge from a running mlx2 server to repair tools.

Configuration is data only. It binds one tenant/session, captured assets and
local JSON Schema oracles; callers cannot provide paths, policies or verdicts.
Construction never loads a model or runs inference. Trials stay opt-in.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import threading
from contextlib import nullcontext
from pathlib import Path

from ...runtime.weight_space_metrics import set_weight_space_gauge, track_weight_space
from .self_repair import (
    RepairPolicy,
    RepairSession,
    RepairSuite,
    SelfRepairController,
    ServingRepairBackend,
    ToolCase,
)
from .tool_service import (
    MAX_JSON_BYTES,
    WeightSpaceToolService,
    _private_file,
    service_from_operator_config,
)

LIVE_SCHEMA = "mlx2.weight-space-live.v1"
_STRING = {"type": "string", "minLength": 1, "maxLength": 8192}
_ID = {"type": "string", "minLength": 1, "maxLength": 128}
_NUMBER = {"type": "number"}
_INT = {"type": "integer"}
_MATRIX = {"type": "array", "minItems": 1, "maxItems": 8192,
           "items": {"type": "array", "minItems": 1, "maxItems": 8192, "items": _NUMBER}}


def _spec(method, required=(), **properties):
    return {"method": method, "inputSchema": {"type": "object", "properties": properties,
            "required": list(required), "additionalProperties": False}}


TOOLS = {
    "repair_capabilities": _spec("capabilities"),
    "inspect_weight_program": _spec("inspect_program", ("program_id",), program_id=_ID),
    "inspect_repair_capture": _spec("inspect_capture", ("capture_id",), capture_id=_ID),
    "inspect_tensor_window": _spec("inspect_tensor_window", ("capture_id", "target"),
        capture_id=_ID, target=_STRING, tensor={"enum": ["weights", "direction", "activations",
            "reference_activations", "contrast_activations", "reference_inputs", "contrast_inputs"]},
        row_start=_INT, column_start=_INT, rows=_INT, columns=_INT),
    "report_tool_failure": _spec("report_tool_failure", ("failure",), failure={
        "type": "object", "properties": {
            "tool_name": _ID, "tool_schema_digest": _STRING, "failed_arguments": {"type": "object"},
            "corrected_arguments": {"type": "object"}, "observed_error": _STRING,
            "error_kind": {"enum": ["arguments", "tool_selection", "protocol"]}},
        "required": ["tool_name", "tool_schema_digest", "failed_arguments", "corrected_arguments",
                     "observed_error", "error_kind"], "additionalProperties": False}),
    "compile_analytical_repair": _spec("compile_analytical_repair", ("capture_id",),
        capture_id=_ID, strength=_NUMBER),
    "propose_directional_program": _spec("propose_directional_program",
        ("capture_id", "target", "direction", "selected_columns"),
        capture_id=_ID, target=_STRING, strength=_NUMBER,
        direction={"type": "array", "minItems": 1, "maxItems": 8192, "items": _NUMBER},
        selected_columns={"type": "array", "minItems": 1, "maxItems": 8192, "items": _INT}),
    "propose_factor_program": _spec("propose_factor_program", ("capture_id", "target", "a", "b"),
        capture_id=_ID, target=_STRING, a=_MATRIX, b=_MATRIX),
    "compile_abliteration": _spec("compile_abliteration", ("capture_id", "peak_layer", "radius"),
        capture_id=_ID, peak_layer=_NUMBER, radius=_NUMBER, peak_strength=_NUMBER, edge_strength=_NUMBER,
        orthogonalize_reference={"type": "boolean"}, winsorization_quantile=_NUMBER,
        row_normalization={"enum": ["none", "pre", "full"]}, full_rank={"type": ["integer", "null"]}),
    "propose_repair": _spec("propose_repair", ("attempt_id", "program_id"), attempt_id=_ID, program_id=_ID),
    "trial_candidate": _spec("trial_candidate", ("attempt_id",), attempt_id=_ID),
    "get_repair_attempt": _spec("get_attempt", ("attempt_id",), attempt_id=_ID),
    "promote_validated_candidate": _spec("promote_validated_candidate", ("attempt_id",), attempt_id=_ID),
    "rollback_deployment": _spec("rollback_deployment", ("attempt_id", "reason"), attempt_id=_ID, reason=_STRING),
    "inspect_quantization_targets": _spec("inspect_quantization_targets", offset=_INT, limit=_INT),
    "create_quantization_overlay": _spec("create_quantization_overlay", ("name", "target_id"),
        name=_ID, target_id=_ID, bits={"type": "integer", "minimum": 1, "maximum": 15},
        format={"enum": ["affine_uniform", "fp4_e2m1_f32_scale"]}, group_size=_INT),
    "inspect_quantization_overlay": _spec("inspect_quantization_overlay", name=_ID),
    "compose_quantization_overlays": _spec("compose_quantization_overlays", ("name", "names"),
        name=_ID, names={"type": "array", "items": _ID, "minItems": 1, "maxItems": 32}),
    "delete_quantization_overlay": _spec("delete_quantization_overlay", ("name",), name=_ID),
    "inspect_expert_source": _spec("inspect_expert_source", ("source_id",), source_id=_ID),
    "inspect_expert_mask_runtime": _spec("inspect_expert_mask_runtime"),
    "compile_expert_mask": _spec("compile_expert_mask", ("source_id", "keep_counts"),
        source_id=_ID, keep_counts={"type": "object", "additionalProperties": _INT, "maxProperties": 256}),
    "plan_expert_residency": _spec("plan_expert_residency", (
        "source_id", "plan_id", "memory_budget_bytes", "nonexpert_reserved_bytes", "chunk_bytes"),
        source_id=_ID, plan_id=_ID, memory_budget_bytes=_INT, nonexpert_reserved_bytes=_INT,
        chunk_bytes=_INT, overlay_bytes=_INT, maximum_resident_bytes={"type": ["integer", "null"]}),
    "prefetch_experts": _spec("prefetch_experts", ("source_id", "experts"), source_id=_ID,
        experts={"type": "array", "minItems": 1, "maxItems": 1024, "items": {
            "type": "object", "properties": {"layer": _INT, "expert": _INT, "projection": _ID},
            "required": ["layer", "expert"], "additionalProperties": False}}),
    "advance_expert_stream": _spec("advance_expert_stream", ("source_id",), source_id=_ID, max_chunks=_INT),
    "compile_layer_excision": _spec("compile_layer_excision", ("inventory_id", "name", "removed_layers"),
        inventory_id=_ID, name=_ID,
        removed_layers={"type": "array", "items": _INT, "minItems": 1, "maxItems": 10000}),
    "inspect_layer_excision": _spec("inspect_layer_excision", ("name",), name=_ID),
    "fit_excision_healing": _spec("fit_excision_healing", ("name", "capture_id", "output_name"),
        name=_ID, capture_id=_ID, output_name=_ID, rank=_INT, ridge=_NUMBER),
    "inspect_memory_calibration": _spec("inspect_memory_calibration", ("capture_id",), capture_id=_ID),
    "calibrate_memory": _spec("calibrate_memory", ("capture_id",),
        capture_id=_ID, rank=_INT, ridge=_NUMBER, strength=_NUMBER),
    "publish_memory_program": _spec("publish_memory_program", ("capture_id", "program_id"),
        capture_id=_ID, program_id=_ID, ttl_seconds=_NUMBER),
    "mount_memory_program": _spec("mount_memory_program", ("capture_id",), capture_id=_ID),
    "revoke_memory_program": _spec("revoke_memory_program", ("capture_id",), capture_id=_ID),
    "evaluate_quantization_quality": _spec("evaluate_quantization_quality", ("capture_id",), capture_id=_ID),
}


def _json(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(text.encode()) > MAX_JSON_BYTES:
        raise ValueError("weight-space configuration exceeds its JSON bound")
    return text


def tool_schema_digest(name, schema):
    return hashlib.sha256(_json({"name": name, "parameters": schema}).encode()).hexdigest()


def _no_refs(value):
    if isinstance(value, dict):
        if any(key in value for key in ("$ref", "$dynamicRef", "$recursiveRef")):
            raise ValueError("live tool oracles cannot resolve schema references")
        for item in value.values():
            _no_refs(item)
    elif isinstance(value, list):
        for item in value:
            _no_refs(item)


def _directory(root, relative):
    if not isinstance(relative, str):
        raise TypeError("attempts_dir must be a relative directory")
    part = Path(relative)
    if part.is_absolute() or not part.parts or any(p in {".", ".."} for p in part.parts):
        raise ValueError("attempts_dir must remain within the owner configuration root")
    current = root
    for name in part.parts:
        current = current / name
        if current.exists() or current.is_symlink():
            info = current.lstat()
            if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) & 0o077):
                raise ValueError("attempts directory must be owner-private without symlinks")
    return current


def load_live_config(path):
    """Validate owner-private JSON and capture assets before opening a server."""
    from jsonschema import Draft202012Validator, FormatChecker

    supplied = Path(path).expanduser().absolute()
    root = supplied.parent
    info = root.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077):
        raise ValueError("weight-space config parent must be an owner-private directory")
    value = json.loads(_private_file(root, supplied.name, maximum=MAX_JSON_BYTES))
    _json(value)
    required = {"schema", "checkpoint", "source_revision", "assets_config", "attempts_dir", "session", "suites"}
    optional = {"policy", "proposal_data_scope", "runtime_source_sha256", "experiments", "repair_tool_aliases"}
    if (not isinstance(value, dict) or not required <= set(value) or set(value) - required - optional
            or value["schema"] != LIVE_SCHEMA):
        raise ValueError("live weight-space config fields or schema differ")
    for key in ("checkpoint", "source_revision"):
        if not isinstance(value[key], str) or not value[key] or "*" in value[key]:
            raise ValueError("live configuration requires exact checkpoint and source revision")
    if "@" not in value["checkpoint"]:
        raise ValueError("live checkpoint must be artifact@fingerprint")
    policy_raw = value.get("policy", {})
    if not isinstance(policy_raw, dict) or set(policy_raw) - {
        "enabled", "execution_enabled", "automatic_promotion", "promotion_scope", "max_attempts_per_failure",
        "require_observed_failure",
    }:
        raise ValueError("unknown live repair policy setting")
    policy = RepairPolicy(**{"require_observed_failure": True, **policy_raw})
    if policy.automatic_promotion and not (policy.enabled and policy.execution_enabled):
        raise ValueError("automatic promotion requires explicit enabled execution")
    experiments = value.get("experiments")
    if experiments is not None:
        if (not isinstance(experiments, dict) or "mode" not in experiments
                or set(experiments) - {"mode", "budget_bytes", "max_materialized_bytes", "assets", "construction_budget_bytes"}
                or experiments["mode"] not in {"quantization_overlay", "layer_excision", "expert_mask", "analysis"}):
            raise ValueError("unknown live experiment mode or setting")
        for key in ("budget_bytes", "max_materialized_bytes", "construction_budget_bytes"):
            size = experiments.get(key, 64 * 1024**2)
            if type(size) is not int or not 1 <= size <= 64 * 1024**3:
                raise ValueError("experiment byte budgets must be explicit bounded positive integers")
        if "assets" in experiments and (not isinstance(experiments["assets"], str) or not experiments["assets"]):
            raise ValueError("experiment assets must name one owner-relative JSON file")
        if policy.enabled or policy.execution_enabled or policy.automatic_promotion:
            raise ValueError("experiment mode cannot compose with automatic LoRA repair")
    scope = value.get("proposal_data_scope", "session")
    if scope not in {"session", "shared"}:
        raise ValueError("proposal_data_scope must be session or explicitly shared")
    assets = service_from_operator_config(root, value["assets_config"])
    captures, programs = assets.registered_assets()
    if any((item.checkpoint, item.source_revision) != (value["checkpoint"], value["source_revision"])
           for item in (*captures, *programs.values())):
        raise ValueError("registered assets differ from the live checkpoint/source")
    session = value["session"]
    if (not isinstance(session, dict) or set(session) != {"session_id", "tenant_id", "request"}
            or session["session_id"] != assets.session_id
            or not isinstance(session["tenant_id"], str) or not session["tenant_id"]
            or not isinstance(session["request"], dict)):
        raise ValueError("live session must match the registered assets and one trusted tenant")
    repair_session = RepairSession(session["session_id"], copy.deepcopy(session["request"]),
                                   repair_tool_aliases=value.get("repair_tool_aliases", {}))
    repair_session.snapshot()
    validators, suites = {}, []
    minimum_suites = 0 if experiments is not None else 1
    if not isinstance(value["suites"], list) or not minimum_suites <= len(value["suites"]) <= 32:
        raise ValueError("live repair needs 1..32 bounded host tool suites")
    for item in value["suites"]:
        fields = {"tool_name", "tool_schema_digest", "argument_schema", "repair", "regressions"}
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("live tool suite fields differ")
        name, schema = item["tool_name"], item["argument_schema"]
        if not isinstance(name, str) or not name or name in validators or not isinstance(schema, dict):
            raise ValueError("each live tool needs one named argument schema")
        _no_refs(schema)
        Draft202012Validator.check_schema(schema)
        if item["tool_schema_digest"] != tool_schema_digest(name, schema):
            raise ValueError("tool_schema_digest must bind its name and exact argument schema")
        validators[name] = Draft202012Validator(schema, format_checker=FormatChecker())
        groups = []
        for group in ("repair", "regressions"):
            if not isinstance(item[group], list) or not 1 <= len(item[group]) <= 64:
                raise ValueError("repair/preservation cases must each contain 1..64 cases")
            cases = []
            for case in item[group]:
                if not isinstance(case, dict) or set(case) != {
                    "case_id", "messages", "expected_tool", "expected_arguments",
                }:
                    raise ValueError("host tool case fields differ")
                cases.append(ToolCase.create(**case))
            groups.append(tuple(cases))
        suites.append(RepairSuite(name, item["tool_schema_digest"], *groups))
    for suite in suites:
        for case in suite.repair + suite.regressions:
            if case.expected_tool not in validators or not validators[case.expected_tool].is_valid(
                json.loads(case.expected_arguments_json)
            ):
                raise ValueError("host expected tool case fails its argument oracle")
    return {
        "raw": value, "root": root, "attempts_root": _directory(root, value["attempts_dir"]),
        "policy": policy, "session": repair_session, "tenant_id": session["tenant_id"],
        "captures": captures, "programs": programs, "validators": validators,
        "suites": tuple(suites), "proposal_data_scope": scope,
    }


class LiveWeightSpaceService:
    """Lazy binding to one already-loaded engine; no model load or inference."""

    def __init__(self, engine, config):
        self.engine, self.config = engine, config
        self._service = None
        self._error = None
        # An engine experiment manager cannot be uninstalled; keep the cause
        # of a bind that failed after installing one, not "already configured".
        self._bind_error = None
        self._lock = threading.RLock()
        self.live_session_observed = False
        self._trials = {}
        self._last_capabilities = None
        # Publish only a fully initialized callback and serialize that binding
        # with kernel development. Existing controllers cannot be replaced.
        with getattr(engine, "_weight_space_control_section", nullcontext)():
            if getattr(engine, "weight_space_live", None) is not None:
                raise ValueError("a live weight-space service is already configured")
            for name, field in (("live_enabled", "enabled"),
                                ("live_execution_enabled", "execution_enabled"),
                                ("live_automatic_promotion", "automatic_promotion")):
                set_weight_space_gauge(name, int(getattr(config["policy"], field)))
            engine.weight_space_live = self

    def _check_request_tools(self, request):
        if not isinstance(request.get("messages"), list) or not request["messages"]:
            raise ValueError("active repair session requires the complete messages transcript")
        declared = {item.get("function", {}).get("name"): item.get("function", {}).get("parameters")
                    for item in request.get("tools", ()) if isinstance(item, dict)}
        for name, validator in self.config["validators"].items():
            if declared.get(name) != validator.schema:
                raise ValueError("active session tool schema differs from the host repair suite")

    def _bound(self):
        # Construction already installed the permanent host marker atomically:
        # kernel reload cannot start while this service exists. Do not acquire
        # lifecycle here: an asynchronous trial may promote under that lock
        # while its caller holds the service lock. Provider setup guards itself.
        check = getattr(self.engine, "_check_weight_space_kernel_compatibility", None)
        if callable(check):
            check()
        with self._lock:
            if self._error is not None:
                raise ValueError(self._error)
            if self._bind_error is not None:
                raise RuntimeError(self._bind_error)
            if self._service is not None:
                return self._service
            engine, config = self.engine, self.config
            if not engine.ready.is_set() or engine.error:
                raise RuntimeError("engine is not ready for weight-space tools")
            checkpoint = f"{Path(engine.model_path).name}@{engine.adapter.identity['fingerprint']}"
            if checkpoint != config["raw"]["checkpoint"]:
                raise ValueError("live engine checkpoint differs from the weight-space configuration")
            expected_runtime = config["raw"].get("runtime_source_sha256")
            actual_runtime = engine.snapshot.get("build_runtime", engine.snapshot.get("runtime", {})).get("source_sha256")
            if expected_runtime is not None and expected_runtime != actual_runtime:
                raise ValueError("live engine runtime source differs from the weight-space configuration")
            experiments = config["raw"].get("experiments")
            if experiments is not None:
                service = WeightSpaceToolService(
                    session_id=config["session"].session_id, captures=config["captures"],
                    programs=config["programs"], proposal_data_scope=config["proposal_data_scope"],
                )
                budget = experiments.get("budget_bytes", 64 * 1024**2)
                materialized = experiments.get("max_materialized_bytes", 64 * 1024**2)
                # Owner assets are CPU-only and need no engine manager: load
                # them first so a bad asset never leaves a manager installed.
                if "assets" in experiments:
                    from .experiment_config import configure_experiment_assets

                    configure_experiment_assets(service, config["root"], experiments,
                        checkpoint=checkpoint, source_revision=config["raw"]["source_revision"])
                # Validate every manager-independent input before installing;
                # an installed engine manager cannot be removed again.
                bank = None
                if experiments["mode"] == "quantization_overlay":
                    from .overlays import InMemoryOverlayBank

                    bank = InMemoryOverlayBank(checkpoint=checkpoint,
                        source_revision=config["raw"]["source_revision"], budget_bytes=budget,
                        construction_budget_bytes=experiments.get("construction_budget_bytes", 256 * 1024**2))
                elif experiments["mode"] == "layer_excision" and (
                        "loaded_model" in service._excision_inventories
                        or len(service._excision_inventories) >= 128):
                    raise ValueError("experiment assets must leave room for, and not declare, "
                                     "the reserved loaded_model inventory")
                manager = None
                try:
                    if experiments["mode"] == "quantization_overlay":
                        manager = engine.configure_weight_overlays(max_materialized_bytes=materialized)
                        bank.source_validator = manager.source_validator
                        manager.bank = bank
                        service.attach_experiment_hosts(overlay_manager=manager)
                    elif experiments["mode"] == "layer_excision":
                        manager = engine.configure_layer_excision(max_healing_bytes=materialized)
                        service.attach_experiment_hosts(excision_manager=manager)
                        service.register_excision_inventory("loaded_model", checkpoint=checkpoint,
                            source_revision=config["raw"]["source_revision"],
                            layer_count=len(manager.binding.layers), hidden_size=manager.binding.hidden_size)
                    elif experiments["mode"] == "expert_mask":
                        manager = engine.configure_expert_masks()
                        service.attach_experiment_hosts(expert_mask_manager=manager)
                except Exception as error:
                    # Only manager-dependent checks (binding geometry, manager
                    # capabilities) can still fail here.
                    if manager is not None:
                        self._bind_error = (
                            "weight-space experiment binding failed after installing its engine "
                            f"manager; restart the server: {type(error).__name__}: {error}")[:2048]
                    raise
                self._service = service
                return service
            if engine.multi_lora is None or engine.multi_lora.delta_precision != "fp32":
                raise ValueError("live repair tools require ordinary concurrent LoRA with fp32 deltas")
            if engine.lora_root is None or not config["attempts_root"].resolve().is_relative_to(engine.lora_root):
                raise ValueError("repair attempts_dir must lie within the engine LoRA allowlist")
            self._check_request_tools(config["session"].request)
            current = config["root"]
            for component in config["attempts_root"].relative_to(current).parts:
                current = current / component
                current.mkdir(exist_ok=True, mode=0o700)
            _directory(config["root"], config["raw"]["attempts_dir"])
            controller = SelfRepairController(
                checkpoint=checkpoint, source_revision=config["raw"]["source_revision"],
                root=config["attempts_root"], policy=config["policy"],
                backend=ServingRepairBackend(engine, tenant_id=config["tenant_id"]),
                verify_tool=lambda name, arguments: name in config["validators"]
                    and isinstance(arguments, dict) and config["validators"][name].is_valid(arguments),
                suites=config["suites"], sessions=(config["session"],),
            )
            self._service = WeightSpaceToolService(
                session_id=config["session"].session_id, controller=controller, allow_runtime=True,
                captures=config["captures"], programs=config["programs"],
                proposal_data_scope=config["proposal_data_scope"],
            )
            return self._service

    def capabilities(self):
        service = self._bound()
        policy = self.config["policy"]
        with self._lock:
            pending = [key for key, value in self._trials.items() if value["pending"]]
            if not pending:
                self._last_capabilities = service.capabilities()
            current = copy.deepcopy(self._last_capabilities or {})
        return {
            **current, "live_checkpoint": self.config["raw"]["checkpoint"],
            "source_revision": self.config["raw"]["source_revision"],
            "runtime_source_sha256": self.engine.snapshot.get("build_runtime", self.engine.snapshot.get("runtime", {})).get("source_sha256"),
            "oracle_kind": "local_json_schema", "live_session_observed": self.live_session_observed,
            "session_source": "live_submission" if self.live_session_observed else "operator_seed",
            "runtime_trials_enabled": policy.enabled and policy.execution_enabled,
            "automatic_promotion": policy.automatic_promotion,
            "require_observed_failure": policy.require_observed_failure,
            "repair_tool_aliases": dict(self.config["session"].repair_tool_aliases),
            "deployment_persistence": "running_engine_only",
            "experiment_mode": (self.config["raw"].get("experiments") or {}).get("mode"),
            "pending_trial_ids": pending,
            "trial_execution": "asynchronous_submit_once_poll_get_repair_attempt",
            "tools": [{"name": name, "inputSchema": copy.deepcopy(spec["inputSchema"])}
                      for name, spec in TOOLS.items()],
            "tool_schema_digests": {suite.tool_name: suite.tool_schema_digest for suite in self.config["suites"]},
        }

    @track_weight_space("tool_dispatch")
    def call(self, body):
        from jsonschema import Draft202012Validator

        _json(body)
        if not isinstance(body, dict) or set(body) != {"tool", "arguments"} or body.get("tool") not in TOOLS:
            raise ValueError("weight-space call requires one enumerated tool and arguments")
        spec = TOOLS[body["tool"]]
        if not Draft202012Validator(spec["inputSchema"]).is_valid(body["arguments"]):
            raise ValueError("weight-space arguments differ from the tool schema")
        if body["tool"] == "repair_capabilities":
            return self.capabilities()
        arguments = copy.deepcopy(body["arguments"])
        if body["tool"] == "compile_analytical_repair":
            arguments.setdefault("strength", 1.0)
        service = self._bound()
        if body["tool"] == "trial_candidate":
            return self._submit_trial(service, arguments["attempt_id"])
        with self._lock:
            pending = any(value["pending"] for value in self._trials.values())
            cached = self._trials.get(arguments.get("attempt_id"))
            # A running trial holds the controller lock, so answer from the
            # cache only then. Afterwards the controller's receipt is
            # authoritative: promotion recovery, expiry and failed promotion
            # are written there, never into this cache.
            if body["tool"] == "get_repair_attempt" and cached is not None and pending:
                return copy.deepcopy(cached)
            if pending:
                raise RuntimeError("a repair trial is running; poll get_repair_attempt before another operation")
        try:
            result = getattr(service, spec["method"])(**arguments)
        except Exception as error:
            if cached is not None and body["tool"] == "get_repair_attempt":
                # No controller receipt (e.g. the worker's fallback): the
                # cached receipt stays pollable and says why it is cached.
                with self._lock:
                    return {**copy.deepcopy(self._trials[arguments["attempt_id"]]),
                            "controller_receipt_error": (type(error).__name__ + ": " + str(error))[:1024]}
            # A failed mutation may still have advanced the durable attempt.
            if cached is not None:
                self._refresh_trial(service, arguments["attempt_id"])
            raise
        with self._lock:
            if isinstance(result, dict) and result.get("id") in self._trials:
                recorded = self._record_trial(result)
                if body["tool"] == "get_repair_attempt":
                    return recorded
        return result

    def _record_trial(self, result):
        """Cache a controller receipt, keeping any background worker error."""
        with self._lock:
            entry = {**result, "pending": False}
            worker_error = self._trials.get(result["id"], {}).get("trial_worker_error")
            if worker_error is not None:
                entry["trial_worker_error"] = worker_error
            self._trials[result["id"]] = entry
            return copy.deepcopy(entry)

    def _refresh_trial(self, service, attempt_id):
        try:
            current = service.get_attempt(attempt_id)
        except Exception:  # noqa: BLE001 - keep the caller's original error
            return
        if isinstance(current, dict) and current.get("id") == attempt_id:
            self._record_trial(current)

    def _submit_trial(self, service, attempt_id):
        with self._lock:
            if attempt_id in self._trials:
                return copy.deepcopy(self._trials[attempt_id])
            if any(value["pending"] for value in self._trials.values()):
                raise RuntimeError("a repair trial is already running")
            if not (self.config["policy"].enabled and self.config["policy"].execution_enabled):
                raise ValueError("live execution is disabled by the owner policy")
            # Validate attempt/session ownership before launching, never by
            # accepting a caller-supplied path or success claim.
            owned = service.get_attempt(attempt_id)
            if owned.get("session_id") != self.config["session"].session_id:
                raise ValueError("repair attempt belongs to another host session")
            if owned.get("state") != "candidate":
                raise ValueError("only a compiled candidate can start a repair trial")
            if len(self._trials) >= 128:
                raise ValueError("live trial registry is full")
            self._last_capabilities = service.capabilities()
            pending = {"id": attempt_id, "session_id": self.config["session"].session_id,
                       "state": "trial_submitted", "pending": True}
            self._trials[attempt_id] = pending

            def run():
                try:
                    result = service.trial_candidate(attempt_id)
                    _json(result)
                    if not isinstance(result, dict) or result.get("id") != attempt_id:
                        raise ValueError("controller returned an invalid trial receipt")
                except Exception as error:  # noqa: BLE001 - a failed background worker must remain pollable
                    worker_error = {"error_type": type(error).__name__, "error": str(error)[:1024]}
                    # The controller journals post-CAS states such as
                    # promotion_recovery_required (candidate possibly still
                    # deployed); report that receipt, not a synthetic one.
                    try:
                        result = service.get_attempt(attempt_id)
                        _json(result)
                        if not isinstance(result, dict) or result.get("id") != attempt_id:
                            raise ValueError("controller returned an invalid attempt receipt")
                    except Exception:  # noqa: BLE001 - fall back to a transport-only receipt
                        result = {"id": attempt_id, "session_id": self.config["session"].session_id,
                                  "state": "trial_failed", **worker_error}
                    result = {**result, "trial_worker_error": worker_error}
                with self._lock:
                    self._trials[attempt_id] = {**result, "pending": False}

            threading.Thread(target=run, name="mlx2-weight-space-trial", daemon=True).start()
            return copy.deepcopy(pending)

    def resolve_session_model(self, request, *, tenant_id):
        """Bind a validated active task before ordinary LoRA resolution.

        This reads only the session lock, never a controller lock held by an
        asynchronous trial. Explicit other adapters and new tasks end the
        session override; deployment-default routing remains the engine's job.
        """
        import time

        from .self_repair import _digest, _repair_continuation, _request_context

        request.pop("_mlx2_weight_session_expected_fingerprint", None)
        model = request.get("model")
        if (self.config.get("raw", {}).get("experiments") or tenant_id != self.config["tenant_id"]
                or request.get("session_id") != self.config["session"].session_id):
            return model
        session = self.config["session"]
        with session.lock:
            override = session.routing_override
            if override is None:
                return model
            public = {key: value for key, value in request.items() if not key.startswith("_mlx2_")}
            permitted_models = {None, Path(self.engine.model_path).name,
                                override["previous_model"], override["name"]}
            if (self._error is not None or model not in permitted_models
                    or (override["expires_at"] is not None and time.time() >= override["expires_at"])
                    or _digest(_request_context(public)) != override["request_context_digest"]
                    or _digest(dict(session.repair_tool_aliases)) != override.get(
                        "repair_tool_aliases_digest", _digest({}))
                    or not _repair_continuation(override["messages"], public.get("messages", []),
                                               allowed_tools=override.get("allowed_tools", ()))):
                session.routing_override = None
                return model
            # Serving compares this host-written fingerprint after its lookup;
            # a same-name unload/re-register may not substitute other content.
            request["_mlx2_weight_session_expected_fingerprint"] = override["fingerprint"]
            return override["name"]

    def observe_submission(self, job):
        """Capture only the configured authenticated session, never trial rows."""
        if (self.config.get("raw", {}).get("experiments") is not None
                or job.tenant_id != self.config["tenant_id"]
                or job.request.get("session_id") != self.config["session"].session_id
                or job.request.get("_mlx2_weight_program_replay") is not None):
            return
        try:
            # Publication owns an admission reservation/submission lock. Never
            # bind here: controller binding takes the lifecycle lock, whose
            # owner may be draining that very reservation for a model edit.
            request = {key: copy.deepcopy(value) for key, value in job.request.items()
                       if not key.startswith("_mlx2_")}
            self._check_request_tools(request)
            _json(request)
            session = self.config["session"]
            with session.lock:
                session.request = request
                session.active_adapter = job.lora_name
                session.transcript_origin = "live_submission"
                session.revision += 1
            self.live_session_observed = True
            # A later well-formed turn supersedes one malformed turn.
            self._error = None
        except (ValueError, RuntimeError) as error:
            # Misconfigured repair tools must not break ordinary serving, and
            # must not keep using an old seed as if it were the current chat.
            self._error = str(error)
