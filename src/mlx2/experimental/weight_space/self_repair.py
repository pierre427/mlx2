"""Opt-in tool-error repair trials with host-owned oracles and deployment CAS.

The model proposes a failure/correction and immutable weight program. It cannot
provide test verdicts, change the test suite, or grant itself deployment scope.
Live inference is a host-injected backend; this module never imports MLX.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import threading
import time
import uuid
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from mlx2.runtime.weight_space_metrics import (
    record_weight_space,
    set_active_weight_candidates,
    track_weight_space,
)

from .programs import (
    WeightProgram,
    compose_programs,
    create_program,
    export_program,
    inspect_program,
    load_program,
)


def _json(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode()) > 1024 * 1024:
        raise ValueError("repair record exceeds 1 MiB")
    return encoded


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _text(value, name, limit=4096):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"invalid {name}")
    return value


def _private(value):
    if isinstance(value, dict):
        if (value.get("contains_private_memory") is True
                or value.get("promotion_scope") == "session"
                or value.get("data_scope") in {"session", "tenant"}
                or any(key in value for key in ("memory_source", "source_semantic_digest"))):
            return True
        return any(_private(item) for item in value.values())
    return isinstance(value, list) and any(_private(item) for item in value)


# These are the public MCP function names, not caller-supplied permissions.
_REPAIR_TOOLS = frozenset({
    "repair_capabilities", "inspect_weight_program", "inspect_repair_capture",
    "inspect_tensor_window", "report_tool_failure", "compile_analytical_repair",
    "propose_directional_program", "propose_factor_program", "compile_abliteration",
    "propose_repair", "trial_candidate", "get_repair_attempt",
    "promote_validated_candidate", "rollback_deployment",
})


def validate_repair_tool_aliases(aliases):
    """Host-only exact aliases; a name suffix never grants orchestration scope."""
    if not isinstance(aliases, Mapping) or len(aliases) > 64:
        raise ValueError("repair_tool_aliases must be a host mapping of at most 64 names")
    result = {}
    for alias, canonical in aliases.items():
        if (not isinstance(alias, str) or not 1 <= len(alias) <= 256
                or not all(character.isascii() and (character.isalnum() or character in "_-.:")
                           for character in alias)
                or not isinstance(canonical, str) or canonical not in _REPAIR_TOOLS
                or (alias in _REPAIR_TOOLS and alias != canonical)):
            raise ValueError("repair_tool_aliases require exact names and canonical repair tools")
        result[alias] = canonical
    return result


def _repair_tool_names(snapshot):
    return _REPAIR_TOOLS | frozenset(snapshot.get("repair_tool_aliases", {}))


def _request_context(request):
    # The effective selected adapter is bound separately. A public model alias
    # can differ while resolving to that exact same adapter.
    return {key: value for key, value in request.items() if key not in {"messages", "model"}}


def _validation_request(request, *, messages=None):
    result = copy.deepcopy(request)
    if messages is not None:
        result["messages"] = copy.deepcopy(list(messages))
    # Paired functional checks use an explicit greedy profile. Trial prompts
    # must not become user-chat submissions or inherit stochastic user sampling.
    result.pop("session_id", None)
    result.update(temperature=0.0, seed=0)
    return result


def _calls(message):
    calls = message.get("tool_calls", [])
    return calls if isinstance(calls, list) else []


def _task_tool_names(request, aliases=()):
    names = set(_REPAIR_TOOLS) | set(aliases)
    for tool in request.get("tools", []):
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
            name = tool["function"].get("name")
            if isinstance(name, str):
                names.add(name)
    return frozenset(names)


def _repair_continuation(original, current, *, allowed_tools=_REPAIR_TOOLS):
    """Permit append-only repair orchestration, never a new task or rewritten past."""
    if current[:len(original)] != original:
        return False
    pending, seen = {}, set()
    for message in original:
        if not isinstance(message, dict):
            return False
        if message.get("role") == "assistant":
            for call in _calls(message):
                if (isinstance(call, dict) and isinstance(call.get("id"), str)
                        and isinstance(call.get("function"), dict)):
                    name = call["function"].get("name")
                    pending[call["id"]] = name
                    seen.add(call["id"])
        elif message.get("role") == "tool" and isinstance(message.get("tool_call_id"), str):
            pending.pop(message["tool_call_id"], None)
    for message in current[len(original):]:
        if not isinstance(message, dict):
            return False
        if message.get("role") == "assistant":
            if "function_call" in message:
                return False  # Legacy function dispatch is not this repair protocol.
            if message.get("tool_calls") is not None and not isinstance(message["tool_calls"], list):
                return False
            for call in _calls(message):
                if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                    return False
                call_id, name = call.get("id"), call["function"].get("name")
                if (not isinstance(call_id, str) or not call_id or call_id in seen
                        or not isinstance(name, str) or name not in allowed_tools
                        or call.get("type", "function") != "function"):
                    return False
                seen.add(call_id)
                pending[call_id] = name
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if (not isinstance(call_id, str) or call_id not in pending
                    or message.get("name", pending[call_id]) != pending[call_id]):
                return False
            del pending[call_id]
        else:
            return False
    return True


def _failure_replay(snapshot, failure, *, require_observed):
    """Bind the submitted failure to its generation prompt, before repair chatter."""
    request = copy.deepcopy(snapshot["request"])
    messages = request["messages"]
    origin = snapshot.get("transcript_origin", "operator_seed")
    if require_observed and origin != "live_submission":
        raise ValueError("reported failure requires a live submitted transcript, not an operator seed")
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in reversed(_calls(message)):
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                continue
            function = call["function"]
            if function.get("name") != failure["tool_name"]:
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (ValueError, TypeError):
                    continue
            if not isinstance(arguments, dict) or _json(arguments) != _json(failure["failed_arguments"]):
                continue
            if not index or not _repair_continuation(messages[:index + 1], messages,
                                                     allowed_tools=_repair_tool_names(snapshot)):
                raise ValueError("reported failure belongs to a changed task or has no replay prompt")
            request["messages"] = messages[:index]
            return request, {"kind": "observed_tool_call" if origin == "live_submission" else "operator_seed",
                             "transcript_origin": origin, "message_index": index,
                             "tool_call_id": call.get("id"),
                             "source_message_digest": _digest(message),
                             "replay_request_digest": _digest(request)}
    if require_observed:
        raise ValueError("reported failure has no matching observed tool call in the active transcript")
    return request, {"kind": "operator_seed", "transcript_origin": origin,
                     "replay_request_digest": _digest(request)}


@dataclass(frozen=True)
class RepairPolicy:
    """Host configuration, never accepted from a model tool call."""

    enabled: bool = False
    execution_enabled: bool = False
    automatic_promotion: bool = False
    promotion_scope: str = "deployment"
    max_attempts_per_failure: int = 3
    max_attempts_per_session: int = 32
    validation_ttl_seconds: float = 300.0
    require_observed_failure: bool = False

    def __post_init__(self):
        if any(type(v) is not bool for v in
               (self.enabled, self.execution_enabled, self.automatic_promotion,
                self.require_observed_failure)):
            raise ValueError("repair policy switches must be booleans")
        if self.promotion_scope != "deployment":
            raise ValueError("this controller promotes exact-checkpoint deployment defaults")
        if (type(self.max_attempts_per_failure) is not int
                or not 1 <= self.max_attempts_per_failure <= 10):
            raise ValueError("repair attempt budget must be 1..10")
        if (type(self.max_attempts_per_session) is not int
                or not 1 <= self.max_attempts_per_session <= 128):
            raise ValueError("session repair budget must be 1..128")
        if (type(self.validation_ttl_seconds) not in (float, int)
                or not math.isfinite(self.validation_ttl_seconds)
                or not 0 < self.validation_ttl_seconds <= 3600):
            raise ValueError("validation TTL must be positive and at most one hour")


@dataclass(frozen=True)
class ToolCase:
    case_id: str
    messages: tuple
    expected_tool: str
    expected_arguments_json: str

    @classmethod
    def create(cls, case_id, messages, expected_tool, expected_arguments):
        _text(case_id, "case id", 128)
        _text(expected_tool, "tool name", 128)
        if not isinstance(expected_arguments, dict):
            raise TypeError("expected arguments must be an object")
        if not isinstance(messages, (list, tuple)) or not messages:
            raise ValueError("a tool case needs a replayable transcript")
        encoded = _json(list(messages))
        return cls(case_id, tuple(json.loads(encoded)), expected_tool, _json(expected_arguments))

    def to_dict(self):
        return {"case_id": self.case_id, "messages": list(self.messages),
                "expected_tool": self.expected_tool,
                "expected_arguments": json.loads(self.expected_arguments_json)}


@dataclass(frozen=True)
class RepairSuite:
    tool_name: str
    tool_schema_digest: str
    repair: tuple[ToolCase, ...]
    regressions: tuple[ToolCase, ...]

    def __post_init__(self):
        _text(self.tool_name, "tool name", 128)
        _text(self.tool_schema_digest, "tool schema digest", 256)
        cases = self.repair + self.regressions
        if not self.repair or not self.regressions or len(cases) > 128:
            raise ValueError("bounded repair and preservation cases are both required")
        ids = [case.case_id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("repair and regression case ids must be disjoint")

    @property
    def digest(self):
        return _digest({"tool": self.tool_name, "schema": self.tool_schema_digest,
                        "repair": [c.to_dict() for c in self.repair],
                        "regressions": [c.to_dict() for c in self.regressions]})


@dataclass
class RepairSession:
    """Host-owned active conversation; tools may name it but cannot replace it."""

    session_id: str
    request: dict
    active_adapter: str | None = None
    revision: int = 0
    transcript_origin: str = "operator_seed"
    repair_tool_aliases: Mapping = field(default_factory=dict, repr=False)
    routing_override: dict | None = field(default=None, repr=False)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self):
        self.repair_tool_aliases = MappingProxyType(validate_repair_tool_aliases(self.repair_tool_aliases))

    def snapshot(self):
        with self.lock:
            if not isinstance(self.request.get("messages"), list) or not self.request["messages"]:
                raise ValueError("repair requires the complete replayable conversation")
            if self.transcript_origin not in {"operator_seed", "live_submission"}:
                raise ValueError("unknown host transcript origin")
            return {"request": json.loads(_json(self.request)),
                    "active_adapter": self.active_adapter, "revision": self.revision,
                    "transcript_origin": self.transcript_origin,
                    "repair_tool_aliases": validate_repair_tool_aliases(self.repair_tool_aliases)}


class ServingRepairBackend:
    """Bridge to an existing loaded engine, with an operator-owned Job reader.

    ``observe_job(job)`` must collect the real model's emitted tool call and
    return {tool_name, arguments, protocol_valid, program_id, fingerprint,
    checkpoint, replayed, engaged}. No inference happens during construction.
    """

    def __init__(self, engine, observe_job=None, *, tenant_id="default"):
        self.engine = engine
        self.tenant_id = _text(tenant_id, "tenant id", 128)
        if observe_job is None:
            from mlx2.runtime.weight_program_observer import observe_weight_program_job

            observe_job = observe_weight_program_job
        self.observe_job = observe_job

    def state(self):
        return self.engine.weight_program_deployment_state()

    def deployed_program(self, previous):
        manager = self.engine.multi_lora
        with manager.lock:
            registered = manager.registry[previous["default_adapter"]]
            if registered.fingerprint != previous["fingerprint"]:
                raise ValueError("deployed adapter changed before composition")
            program = load_program(registered.path)
        if program.program_id != previous["program_id"]:
            raise ValueError("deployed program identity changed")
        return program

    def register(self, name, path, program):
        # Reserved before the load, so no client can name the candidate in
        # between; only replay, the session override or promotion select it.
        # Lifecycle ownership spans check, reserve, load, bind and cleanup,
        # so no operator unload/reload of the name can interleave.
        reserve = getattr(self.engine, "reserve_private_weight_program", None)
        section = getattr(self.engine, "_lora_control_section", None)
        with section() if callable(section) else nullcontext():
            # An operator adapter already holding the name keeps serving: no
            # reservation (which would hide it) is taken for a load that fails.
            manager = getattr(self.engine, "multi_lora", None)
            if manager is not None and name in getattr(manager, "registry", {}):
                raise ValueError("a LoRA adapter with this candidate name is already loaded")
            token = reserve(name) if callable(reserve) else None
            try:
                result = self.engine.load_lora_adapter(name, str(path))
            except BaseException:
                if token is not None:
                    self.engine.release_private_weight_program(name, token=token)
                raise
            try:
                fingerprint = result.get("fingerprint")
                if not fingerprint:
                    fingerprint = self.engine.multi_lora.registry[name].fingerprint
                if token is not None:
                    self.engine.bind_private_weight_program(name, token, fingerprint)
            except BaseException:
                # Loaded but unbound: unload it.  If that fails too the
                # reservation stays, so the candidate remains hidden.
                unloaded = True
                try:
                    self.engine.unload_lora_adapter(name)
                except (ValueError, RuntimeError, OSError):
                    unloaded = False
                if unloaded and token is not None:
                    self.engine.release_private_weight_program(name, token=token)
                raise
        return {"name": name, "fingerprint": fingerprint, "program_id": program.program_id}

    def unregister(self, candidate):
        """Unload a terminal candidate's tensors; its artifact stays on disk."""
        manager = self.engine.multi_lora
        # Hold lifecycle ownership across the identity check and the unload:
        # an admin unload/re-register of the same name in between would
        # otherwise make this cleanup delete different content.
        release = getattr(self.engine, "release_private_weight_program", None)
        owns = getattr(self.engine, "owns_private_weight_program", None)
        with self.engine._lora_control_section():
            if callable(owns) and not owns(candidate["name"], candidate["fingerprint"]):
                return  # Unloaded meanwhile; the name may now be an operator's.
            with manager.lock:
                registered = manager.registry.get(candidate["name"])
                if registered is None or registered.fingerprint != candidate["fingerprint"]:
                    # Already released, or a same-name load is not ours (the
                    # reservation is fingerprint-bound, so it is not hidden).
                    if callable(release):
                        release(candidate["name"], fingerprint=candidate["fingerprint"])
                    return
            # Refuses (ValueError) while live requests pin it or it is the default.
            self.engine.unload_lora_adapter(candidate["name"])
            if callable(release):
                release(candidate["name"], fingerprint=candidate["fingerprint"])

    def run(self, request, candidate):
        job = self.engine.submit_weight_program_replay(
            copy.deepcopy(request), name=candidate["name"],
            expected_fingerprint=candidate["fingerprint"],
            tenant_id=self.tenant_id,
        )
        return self.observe_job(job)

    def baseline(self, request, previous):
        request = copy.deepcopy(request)
        if previous.get("default_adapter") is None:
            request["model"] = Path(self.engine.model_path).name + ":base"
        else:
            request["model"] = previous["default_adapter"]
        # A baseline uses its own exact adapter identity, never candidate state.
        return self.observe_job(self.engine.submit(request, tenant_id=self.tenant_id))

    def promote(self, candidate, generation, receipt):
        result = self.engine.promote_weight_program(
            candidate["name"], expected_fingerprint=candidate["fingerprint"],
            expected_generation=generation, validation_receipt=receipt,
        )
        return {**result, "base_model_alias": Path(self.engine.model_path).name + ":base"}

    def promotion_control_section(self):
        """Acquire lifecycle ownership before the controller locks its session."""
        return self.engine._lora_control_section()

    def rollback(self, generation, previous):
        result = self.engine.rollback_weight_program_default(
            expected_generation=generation, previous_state=previous,
        )
        return {**result, "base_model_alias": Path(self.engine.model_path).name + ":base"}


class SelfRepairController:
    """Failure -> host verification -> candidate -> replay -> test -> promotion.

    ``verify_tool(name, arguments)`` is a deterministic, side-effect-free
    schema/sandbox oracle returning a strict bool. It must NOT trust model prose.
    Test suites and sessions are registered by the operator, not MCP inputs.
    Failed candidates remain inspectable; defaults change only after all gates.
    """

    def __init__(self, *, checkpoint, source_revision, root, policy, backend,
                 verify_tool, suites, sessions):
        _text(checkpoint, "checkpoint")
        if "@" not in checkpoint or "*" in checkpoint:
            raise ValueError("exact checkpoint required")
        _text(source_revision, "source revision", 256)
        if not isinstance(policy, RepairPolicy):
            raise TypeError("host RepairPolicy required")
        self.checkpoint = checkpoint
        self.source_revision = source_revision
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink():
            raise ValueError("repair root cannot be a symlink")
        self.policy = policy
        self.backend = backend
        self.verify_tool = verify_tool
        self.suites = {(s.tool_name, s.tool_schema_digest): s for s in suites}
        self.sessions = {s.session_id: s for s in sessions}
        self.attempts = {}
        self.failure_counts = {}
        self.session_counts = {}
        self._unreleased = set()
        self.lock = threading.RLock()
        self._recover()

    def _recover(self):
        """Recover budgets/history; never resume an interrupted live trial."""
        for folder in self.root.glob("repair-*"):
            path = folder / "receipt.json"
            if folder.is_symlink() or path.is_symlink() or not path.is_file():
                raise ValueError("invalid repair history entry")
            if path.stat().st_size > 1024 * 1024:
                raise ValueError("oversized repair history")
            attempt = json.loads(path.read_text())
            if (attempt.get("id") != folder.name or attempt.get("checkpoint") != self.checkpoint
                    or attempt.get("source_revision") != self.source_revision):
                raise ValueError("repair history belongs to a different controller identity")
            if attempt.get("artifact_path", str(folder / "candidate")) != str(folder / "candidate"):
                raise ValueError("repair history contains an unexpected artifact path")
            key, session_id = attempt["failure_digest"], attempt["session_id"]
            self.failure_counts[key] = self.failure_counts.get(key, 0) + 1
            self.session_counts[session_id] = self.session_counts.get(session_id, 0) + 1
            if attempt["state"] in {"reported", "candidate", "trial", "validated"}:
                attempt.update(state="interrupted", recovery="new verification and trial required")
            if attempt["state"] in {"promoting", "promotion_recovery_required"}:
                current = self.backend.state()
                if (current.get("program_id") == attempt["candidate"]["program_id"]
                        and current.get("fingerprint") == attempt["candidate"]["fingerprint"]
                        and current["generation"] == attempt["previous_deployment"]["generation"] + 1):
                    attempt.update(state="promoted", promoted_deployment=current,
                                   recovery="reconciled write-ahead promotion with live deployment")
                else:
                    attempt.update(state="interrupted", recovery="promotion was not observed; no automatic retry")
            if (attempt["state"] == "promoted"
                    and self.backend.state() != attempt.get("promoted_deployment")):
                attempt.update(state="historical", recovery="deployment differs; no automatic restoration")
            self.attempts[attempt["id"]] = attempt
            self._save(attempt)

    def _enabled(self, execution=False):
        if not self.policy.enabled:
            raise ValueError("self repair is disabled")
        if execution and not self.policy.execution_enabled:
            raise ValueError("live execution is disabled; CPU compilation only")

    def _save(self, attempt):
        set_active_weight_candidates(self, sum(row["state"] in {"candidate", "trial", "validated"}
                                               for row in self.attempts.values()),
                                     recovery_required=sum(row["state"] == "promotion_recovery_required"
                                                           for row in self.attempts.values()))
        folder = self.root / attempt["id"]
        folder.mkdir(mode=0o700, exist_ok=True)
        # Model input never determines paths or IDs. Atomic durable phase journal.
        pending = folder / "receipt.json.tmp"
        with pending.open("w") as stream:
            os.chmod(pending, 0o600)
            stream.write(_json(attempt) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(folder / "receipt.json")

    def _release_candidate(self, attempt):
        """Unload a terminal, undeployed candidate from the serving registry.

        Called for rejected, failed, expired, promotion_failed and rolled_back
        attempts. Otherwise the candidate stays listed and selectable by name,
        and its tensors stay resident for the life of the process. A job still
        pinning it, or a deployment default still naming it, defers the release
        to the next trial; the artifact and receipt are kept.
        """
        release = getattr(self.backend, "unregister", None)
        candidate = attempt.get("candidate")
        if release is None or candidate is None:
            return
        try:
            if self.backend.state().get("default_adapter") == candidate["name"]:
                raise ValueError("candidate is the current deployment default")
            release(candidate)
        except Exception as exc:  # noqa: BLE001 - release is cleanup, never a verdict
            attempt.update(candidate_released=False,
                           candidate_release_error=(type(exc).__name__ + ": " + str(exc))[:1024])
            self._unreleased.add(attempt["id"])
        else:
            attempt["candidate_released"] = True
            attempt.pop("candidate_release_error", None)
            self._unreleased.discard(attempt["id"])

    def _retry_releases(self):
        for attempt_id in sorted(self._unreleased):
            attempt = self.attempts[attempt_id]
            self._release_candidate(attempt)
            if attempt["candidate_released"]:
                self._save(attempt)

    def _get(self, attempt_id):
        if not isinstance(attempt_id, str) or attempt_id not in self.attempts:
            raise ValueError("unknown repair attempt")
        return self.attempts[attempt_id]

    def get_attempt(self, attempt_id):
        with self.lock:
            attempt = self._get(attempt_id)
            # The transcript and failed tool arguments stay in the host registry.
            return copy.deepcopy({key: value for key, value in attempt.items()
                                  if key not in {"failure", "session_snapshot", "switched_session",
                                                 "replay_request"}})

    def _require_task_epoch(self, attempt, *, switched=False):
        current = self.sessions[attempt["session_id"]].snapshot()
        source = attempt["session_snapshot"]
        adapter = attempt["switched_session"]["active_adapter"] if switched else source["active_adapter"]
        allowed_tools = (_task_tool_names(source["request"], source["repair_tool_aliases"])
                         if switched else _repair_tool_names(source))
        if (current["active_adapter"] != adapter
                or current["transcript_origin"] != source["transcript_origin"]
                or _digest(current["repair_tool_aliases"]) != attempt["task_epoch"]["repair_tool_aliases_digest"]
                or _digest(_request_context(current["request"])) != attempt["task_epoch"]["request_context_digest"]
                or not _repair_continuation(source["request"]["messages"], current["request"]["messages"],
                                            allowed_tools=allowed_tools)):
            raise ValueError("session task changed; start a repair from its current transcript")
        return current

    @track_weight_space("failure_report")
    def report_failure(self, failure, *, session_id):
        with self.lock:
            self._enabled()
            fields = {"tool_name", "tool_schema_digest", "failed_arguments",
                      "corrected_arguments", "error_kind", "observed_error"}
            if not isinstance(failure, dict) or set(failure) != fields:
                raise ValueError("failure fields differ from the repair schema")
            for key in ("tool_name", "tool_schema_digest", "observed_error"):
                _text(failure[key], key)
            if failure["error_kind"] not in {"arguments", "tool_selection", "protocol"}:
                raise ValueError("only reproducible model tool errors are repairable")
            if any(not isinstance(failure[k], dict)
                   for k in ("failed_arguments", "corrected_arguments")):
                raise ValueError("tool arguments must be JSON objects")
            if (failure["tool_name"], failure["tool_schema_digest"]) not in self.suites:
                raise ValueError("no host-owned repair and regression suite for this tool revision")
            if session_id not in self.sessions:
                raise ValueError("unknown host session")
            session_snapshot = self.sessions[session_id].snapshot()
            deployment = copy.deepcopy(self.backend.state())
            if session_snapshot["active_adapter"] != deployment.get("default_adapter"):
                raise ValueError("session uses a different adapter; preserve it in a separate scoped repair")
            failure = json.loads(_json(failure))
            replay_request, failure_source = _failure_replay(
                session_snapshot, failure, require_observed=self.policy.require_observed_failure)
            key = _digest({"checkpoint": self.checkpoint,
                           "failure": {k: v for k, v in failure.items() if k != "observed_error"}})
            if self.failure_counts.get(key, 0) >= self.policy.max_attempts_per_failure:
                raise ValueError("repair attempt budget exhausted for this failure")
            if self.session_counts.get(session_id, 0) >= self.policy.max_attempts_per_session:
                raise ValueError("repair attempt budget exhausted for this session")
            self.failure_counts[key] = self.failure_counts.get(key, 0) + 1
            self.session_counts[session_id] = self.session_counts.get(session_id, 0) + 1
            attempt = {"id": "repair-" + uuid.uuid4().hex,
                       "state": "reported", "failure": failure, "failure_digest": key,
                       "session_id": session_id,
                       "session_snapshot": session_snapshot,
                       "replay_request": replay_request, "failure_source": failure_source,
                       "task_epoch": {"source_snapshot_digest": _digest(session_snapshot),
                                      "repair_tool_aliases_digest": _digest(session_snapshot["repair_tool_aliases"]),
                                      "request_context_digest": _digest(_request_context(session_snapshot["request"])),
                                      "reported_messages_digest": _digest(session_snapshot["request"]["messages"])},
                       "checkpoint": self.checkpoint, "source_revision": self.source_revision,
                       "previous_deployment": deployment,
                       "qualified": False, "observed_used": False,
                       "promotion_scope": "deployment"}
            self.attempts[attempt["id"]] = attempt
            self._save(attempt)
            return self.get_attempt(attempt["id"])

    @track_weight_space("repair_propose")
    def propose(self, attempt_id, program):
        with self.lock:
            self._enabled()
            attempt = self._get(attempt_id)
            if attempt["state"] != "reported":
                raise ValueError("attempt already has a candidate or is terminal")
            self._require_task_epoch(attempt)
            # Validate immutable program even when a host Python caller supplied it.
            program = WeightProgram.from_dict(program.to_dict())
            if program.checkpoint != self.checkpoint or program.source_revision != self.source_revision:
                raise ValueError("candidate checkpoint/source revision mismatch")
            if _private(program.provenance):
                raise ValueError("private/session memory cannot become a deployment default")
            if not any(row["delta_frobenius_norm"] > 0
                       for row in inspect_program(program)["factor_statistics"]):
                raise ValueError("a zero-effect program cannot repair a failure")
            failure = attempt["failure"]
            if self.verify_tool(failure["tool_name"], failure["failed_arguments"]) is not False:
                raise ValueError("host oracle did not reproduce the claimed failure")
            if self.verify_tool(failure["tool_name"], failure["corrected_arguments"]) is not True:
                raise ValueError("host oracle did not verify the correction")
            # Tools propose an incremental delta. The controller, rather than
            # caller-authored lineage metadata, preserves the deployed program.
            parent = attempt["previous_deployment"].get("program_id")
            if parent is not None:
                parent_program = None
                for previous_attempt in self.attempts.values():
                    if previous_attempt.get("program_id") == parent:
                        parent_program = load_program(previous_attempt["artifact_path"])
                        break
                if parent_program is None:
                    parent_program = self.backend.deployed_program(attempt["previous_deployment"])
                if parent_program.program_id != parent:
                    raise ValueError("deployed program artifact changed")
                if parent in program.provenance.get("parents", []):
                    raise ValueError("propose an incremental repair; controller composes the deployed adapter")
                program = compose_programs([parent_program, program],
                                           provenance={"operation": "preserve_deployed_repair"})
            program = create_program(
                program.factor_map, checkpoint=program.checkpoint,
                source_revision=program.source_revision, construction=program.construction,
                approximate=program.approximate, numerical_policy=program.numerical_policy,
                provenance={"parent_provenance": program.provenance,
                            "parent_program_ids": [parent] if parent else [],
                            "repair_attempt_id": attempt_id,
                            "failure_digest": attempt["failure_digest"],
                            "failure_source": attempt["failure_source"],
                            "task_epoch": attempt["task_epoch"],
                            "promotion_scope": "deployment"},
            )
            path = self.root / attempt_id / "candidate"
            export_program(program, path)
            attempt.update(state="candidate", program_id=program.program_id,
                           artifact_path=str(path), correction_verified=True)
            self._save(attempt)
            return self.get_attempt(attempt_id)

    def _check_observation(self, observation, case, candidate=None, baseline=None):
        # No input contains a model-authored score or pass flag.
        if not isinstance(observation, dict):
            raise TypeError("backend did not return an observation")
        if observation.get("checkpoint") != self.checkpoint:
            raise ValueError("observed checkpoint mismatch")
        if baseline is not None and any(observation.get(key) != baseline.get(key)
                                        for key in ("program_id", "fingerprint")):
            raise ValueError("baseline execution identity mismatch")
        if candidate is not None:
            for key in ("program_id", "fingerprint"):
                if observation.get(key) != candidate[key]:
                    raise ValueError("candidate execution identity mismatch")
            if observation.get("replayed") is not True or observation.get("engaged") is not True:
                raise ValueError("candidate did not execute a fresh replay with a nonzero adapter")
        arguments = observation.get("arguments")
        good = (observation.get("protocol_valid") is True
                and observation.get("tool_name") == case.expected_tool
                and isinstance(arguments, dict)
                and _json(arguments) == case.expected_arguments_json)
        passed = good and self.verify_tool(case.expected_tool, arguments) is True
        return {"case_id": case.case_id, "passed": bool(passed),
                "observation_digest": _digest(observation)}

    @track_weight_space("repair_trial")
    def trial(self, attempt_id):
        with self.lock:
            self._enabled(execution=True)
            self._retry_releases()
            attempt = self._get(attempt_id)
            if attempt["state"] != "candidate":
                raise ValueError("trial requires a compiled candidate")
            session = self.sessions[attempt["session_id"]]
            self._require_task_epoch(attempt)
            if self.backend.state() != attempt["previous_deployment"]:
                raise ValueError("deployment changed; rebase and retest the candidate")
            program = load_program(attempt["artifact_path"])
            if program.program_id != attempt["program_id"]:
                raise ValueError("candidate artifact changed")
            suite = self.suites[(attempt["failure"]["tool_name"],
                                 attempt["failure"]["tool_schema_digest"])]
            candidate = self.backend.register(attempt_id, attempt["artifact_path"], program)
            if candidate.get("program_id") != program.program_id:
                raise ValueError("registered program identity mismatch")
            attempt.update(state="trial", candidate=candidate, tool_suite_digest=suite.digest)
            self._save(attempt)
            try:
                rows = []
                for kind, cases in (("repair", suite.repair), ("regression", suite.regressions)):
                    for case in cases:
                        request = _validation_request(attempt["session_snapshot"]["request"],
                                                      messages=case.messages)
                        # Baseline runs the same prompts against the prior deployed program.
                        baseline = self._check_observation(
                            self.backend.baseline(request, attempt["previous_deployment"]), case,
                            baseline=attempt["previous_deployment"])
                        patched = self._check_observation(self.backend.run(request, candidate), case, candidate)
                        rows.append({"kind": kind, "base": baseline, "candidate": patched})
                repair_rows = [r for r in rows if r["kind"] == "repair"]
                repair_passed = (all(r["candidate"]["passed"] for r in repair_rows)
                                 and any(not r["base"]["passed"] for r in repair_rows))
                regressions_passed = all(r["candidate"]["passed"] for r in rows
                                         if r["kind"] == "regression")
                live_case = ToolCase.create("active-session-replay",
                    attempt["replay_request"]["messages"],
                    attempt["failure"]["tool_name"], attempt["failure"]["corrected_arguments"])
                live_request = _validation_request(attempt["replay_request"])
                live_base = self._check_observation(
                    self.backend.baseline(copy.deepcopy(live_request), attempt["previous_deployment"]),
                    live_case, baseline=attempt["previous_deployment"])
                live = self._check_observation(self.backend.run(live_request, candidate), live_case, candidate)
                replay_passed = not live_base["passed"] and live["passed"]
                passed = repair_passed and regressions_passed and replay_passed
                receipt = {"passed": passed, "repair_passed": repair_passed,
                           "regressions_passed": regressions_passed, "replay_passed": replay_passed,
                           "validation_sampling": {"temperature": 0.0, "seed": 0},
                           "active_request_digest": _digest(live_request),
                           "checkpoint": self.checkpoint, "source_revision": self.source_revision,
                           "program_id": candidate["program_id"], "fingerprint": candidate["fingerprint"],
                           "tool_suite_digest": suite.digest,
                           "failure_source": attempt["failure_source"],
                           "task_epoch": attempt["task_epoch"],
                           "evidence_digest": _digest({"rows": rows, "live_base": live_base, "live": live}),
                           "rows": rows, "live_base": live_base, "live": live, "qualified": False}
                attempt["validation"] = receipt
                if not passed:
                    attempt["state"] = "rejected"
                    record_weight_space("repair_reject", "rejected")
                    self._release_candidate(attempt)
                    self._save(attempt)
                    return self.get_attempt(attempt_id)
                with session.lock:
                    self._require_task_epoch(attempt)
                    session.active_adapter = candidate["name"]
                    session.request["model"] = candidate["name"]
                    session.routing_override = {
                        "name": candidate["name"], "fingerprint": candidate["fingerprint"],
                        "previous_model": attempt["session_snapshot"]["request"].get("model"),
                        "request_context_digest": attempt["task_epoch"]["request_context_digest"],
                        "messages": copy.deepcopy(attempt["session_snapshot"]["request"]["messages"]),
                        "allowed_tools": sorted(_task_tool_names(attempt["session_snapshot"]["request"],
                                                               attempt["session_snapshot"]["repair_tool_aliases"])),
                        "repair_tool_aliases_digest": attempt["task_epoch"]["repair_tool_aliases_digest"],
                        "expires_at": time.time() + self.policy.validation_ttl_seconds,
                    }
                    session.revision += 1
                    attempt["switched_session"] = session.snapshot()
                attempt.update(state="validated", observed_used=True,
                               validated_at=time.time(),
                               validation_expires_at=time.time() + self.policy.validation_ttl_seconds)
                self._save(attempt)
                if self.policy.automatic_promotion:
                    return self.promote(attempt_id)
                return self.get_attempt(attempt_id)
            except BaseException as exc:
                # Promotion owns its post-CAS recovery state. Overwriting it
                # with "failed" would hide a possibly still-live deployment.
                if attempt["state"] not in {"promotion_recovery_required", "promoted"}:
                    self._restore_session(attempt)
                if attempt["state"] not in {"promotion_recovery_required", "promotion_failed",
                                            "promoted", "rolled_back", "expired"}:
                    attempt["state"] = "failed"
                if attempt["state"] in {"failed", "expired", "promotion_failed", "rolled_back"}:
                    self._release_candidate(attempt)
                attempt["error"] = type(exc).__name__ + ": " + str(exc)
                self._save(attempt)
                raise

    def _restore_session(self, attempt):
        session = self.sessions[attempt["session_id"]]
        switched = attempt.get("switched_session")
        if switched is None:
            return
        with session.lock:
            previous = attempt["session_snapshot"]
            if session.active_adapter != switched["active_adapter"]:
                return  # Another explicit switch superseded this controller.
            # Restore only routing; never erase messages appended by the user.
            if "model" in previous["request"]:
                session.request["model"] = previous["request"]["model"]
            else:
                session.request.pop("model", None)
            session.active_adapter = previous["active_adapter"]
            session.routing_override = None
            session.revision += 1

    def _require_unexpired_validation(self, attempt):
        if time.time() < attempt["validation_expires_at"]:
            return
        self._restore_session(attempt)
        attempt["state"] = "expired"
        record_weight_space("repair_expire", "expired")
        self._release_candidate(attempt)
        self._save(attempt)
        raise ValueError("validation expired; a fresh trial is required")

    @track_weight_space("repair_promote")
    def promote(self, attempt_id):
        with self.lock:
            self._enabled(execution=True)
            attempt = self._get(attempt_id)
            if attempt["state"] != "validated" or attempt["validation"]["passed"] is not True:
                raise ValueError("only this controller's validated trial can be promoted")
            self._require_unexpired_validation(attempt)
            program = load_program(attempt["artifact_path"])
            if program.program_id != attempt["program_id"] or _private(program.provenance):
                raise ValueError("candidate changed or is private")
            # Publication captures the session while admissions drain. Never
            # hold that session lock while waiting for an adapter lifecycle
            # owner which may itself be waiting for publication to complete.
            control = getattr(self.backend, "promotion_control_section", nullcontext)
            with control(), self.sessions[attempt["session_id"]].lock:
                self._require_unexpired_validation(attempt)
                self._require_task_epoch(attempt, switched=True)
                attempt["state"] = "promoting"
                self._save(attempt)  # Write intent before the live deployment CAS.
                try:
                    result = self.backend.promote(attempt["candidate"],
                        attempt["previous_deployment"]["generation"], attempt["validation"])
                except BaseException:
                    self._restore_session(attempt)
                    attempt["state"] = "promotion_failed"
                    self._release_candidate(attempt)
                    self._save(attempt)
                    raise
                attempt.update(state="promoted", promoted_deployment=result)
                override = self.sessions[attempt["session_id"]].routing_override
                if override is not None and override["fingerprint"] == attempt["candidate"]["fingerprint"]:
                    override["expires_at"] = None
                try:
                    self._save(attempt)
                except BaseException:
                    # If durable publication fails, revert only our generation.
                    # The write-ahead journal supports recovery after process death.
                    try:
                        rollback = self.backend.rollback(result["generation"], attempt["previous_deployment"])
                    except BaseException:
                        attempt["state"] = "promotion_recovery_required"
                        raise
                    self._restore_session(attempt)
                    attempt.update(state="rolled_back", rollback_deployment=rollback,
                                   rollback_reason="promotion journal write failed")
                    self._release_candidate(attempt)
                    self._save(attempt)
                    raise
                return self.get_attempt(attempt_id)

    @track_weight_space("repair_rollback")
    def rollback(self, attempt_id, reason):
        with self.lock:
            self._enabled(execution=True)
            _text(reason, "rollback reason", 1024)
            attempt = self._get(attempt_id)
            if attempt["state"] not in {"promoted", "validated"}:
                raise ValueError("attempt has no active switch to roll back")
            if attempt["state"] == "promoted":
                attempt["rollback_deployment"] = self.backend.rollback(
                    attempt["promoted_deployment"]["generation"], attempt["previous_deployment"])
            self._restore_session(attempt)
            attempt.update(state="rolled_back", rollback_reason=reason)
            self._release_candidate(attempt)
            self._save(attempt)
            return self.get_attempt(attempt_id)
