"""CPU transaction/oracle tests; no real model or GPU is loaded."""

import copy
import json
import threading
from contextlib import contextmanager

import numpy as np
import pytest

from mlx2.experimental.weight_space.programs import create_program, load_program
from mlx2.experimental.weight_space.self_repair import (
    RepairPolicy,
    RepairSession,
    RepairSuite,
    SelfRepairController,
    ToolCase,
)

CHECKPOINT = "synthetic-tool-model@" + "a" * 64
REVISION = "e448f070b84642bfc5d11dc04719e40336326a68"


class Backend:
    def __init__(self):
        self.current = {"generation": 0, "default_adapter": None,
                        "fingerprint": None, "program_id": None}
        self.runs = []
        self.fail_case = None
        self.break_identity = False
        self.raise_case = None
        self.during_run = None

    def state(self):
        return copy.deepcopy(self.current)

    def register(self, name, path, program):
        return {"name": name, "program_id": program.program_id,
                "fingerprint": "fp-" + program.program_id}

    def _observation(self, request, candidate=None):
        case_id = request["messages"][-1]["content"]
        if self.raise_case == case_id:
            raise RuntimeError("generation cancelled")
        if self.during_run:
            callback, self.during_run = self.during_run, None
            callback()
        args = {"path": "ok"}
        if (candidate is None and case_id != "regression") or self.fail_case == case_id:
            args = {"filename": "wrong"}
        return {"checkpoint": CHECKPOINT, "tool_name": "read", "arguments": args,
                "protocol_valid": True, "replayed": True, "engaged": True,
                "program_id": candidate["program_id"] if candidate else None,
                "fingerprint": ("wrong" if self.break_identity else candidate["fingerprint"])
                if candidate else None}

    def run(self, request, candidate):
        self.runs.append(copy.deepcopy(request))
        return self._observation(request, candidate)

    def baseline(self, request, previous):
        result = self._observation(request)
        result.update(program_id=previous["program_id"], fingerprint=previous["fingerprint"])
        return result

    def promote(self, candidate, generation, receipt):
        if generation != self.current["generation"]:
            raise ValueError("generation changed")
        assert receipt["passed"] is True
        self.current = {"generation": generation + 1, "default_adapter": candidate["name"],
                        "fingerprint": candidate["fingerprint"], "program_id": candidate["program_id"]}
        return self.state()

    def rollback(self, generation, previous):
        if generation != self.current["generation"]:
            raise ValueError("generation changed")
        self.current = {**previous, "generation": generation + 1}
        return self.state()


def case(name):
    return ToolCase.create(name, [{"role": "user", "content": name}], "read", {"path": "ok"})


def setup(tmp_path, *, enabled=True, execution=True, auto=True):
    backend = Backend()
    session = RepairSession("chat", {"messages": [{"role": "user", "content": "active"}]})
    suite = RepairSuite("read", "schema-v1", (case("heldout"),), (case("regression"),))
    controller = SelfRepairController(
        checkpoint=CHECKPOINT, source_revision=REVISION, root=tmp_path,
        policy=RepairPolicy(enabled, execution, auto), backend=backend,
        verify_tool=lambda name, args: name == "read" and args == {"path": "ok"},
        suites=[suite], sessions=[session],
    )
    return controller, backend, session


def failure():
    return {"tool_name": "read", "tool_schema_digest": "schema-v1",
            "failed_arguments": {"filename": "wrong"}, "corrected_arguments": {"path": "ok"},
            "error_kind": "arguments", "observed_error": "required field path"}


def program(**overrides):
    return create_program({"linear": (np.ones((2, 1)), np.ones((1, 2)))},
        checkpoint=overrides.get("checkpoint", CHECKPOINT), source_revision=REVISION,
        construction="synthetic_fixture", provenance=overrides.get("provenance", {"fixture": "cpu"}))


def proposed(controller):
    attempt = controller.report_failure(failure(), session_id="chat")
    return controller.propose(attempt["id"], program())


def test_verified_trial_switches_active_chat_and_promotes_whole_deployment(tmp_path):
    controller, backend, session = setup(tmp_path)
    candidate = proposed(controller)
    assert backend.current["default_adapter"] is None
    result = controller.trial(candidate["id"])
    assert result["state"] == "promoted"
    assert result["validation"]["repair_passed"]
    assert result["validation"]["regressions_passed"]
    assert backend.current["generation"] == 1
    assert session.active_adapter == result["candidate"]["name"]
    assert backend.runs[-1]["messages"] == [{"role": "user", "content": "active"}]
    assert not result["qualified"]
    assert "failure" not in result and "session_snapshot" not in result
    saved = json.loads((tmp_path / result["id"] / "receipt.json").read_text())
    assert saved["state"] == "promoted"
    rollback = controller.rollback(result["id"], "later preservation regression")
    assert rollback["state"] == "rolled_back"
    assert backend.current["default_adapter"] is None
    assert backend.current["generation"] == 2
    assert session.active_adapter is None


@pytest.mark.parametrize("bad_case", ["heldout", "regression", "active"])
def test_failure_at_any_quality_gate_never_switches_default(tmp_path, bad_case):
    controller, backend, session = setup(tmp_path)
    candidate = proposed(controller)
    backend.fail_case = bad_case
    result = controller.trial(candidate["id"])
    assert result["state"] == "rejected"
    assert backend.current["generation"] == 0
    assert session.active_adapter is None


def test_model_report_and_success_claim_cannot_bypass_oracles(tmp_path):
    controller, backend, _ = setup(tmp_path)
    report = failure()
    report["passed"] = True
    with pytest.raises(ValueError, match="schema"):
        controller.report_failure(report, session_id="chat")
    report = failure()
    report["failed_arguments"] = {"path": "ok"}
    attempt = controller.report_failure(report, session_id="chat")
    with pytest.raises(ValueError, match="reproduce"):
        controller.propose(attempt["id"], program())
    with pytest.raises(ValueError, match="validated"):
        controller.promote(attempt["id"])
    assert backend.current["generation"] == 0


def test_disabled_live_execution_still_allows_cpu_compilation(tmp_path):
    controller, backend, _ = setup(tmp_path, execution=False)
    candidate = proposed(controller)
    with pytest.raises(ValueError, match="live execution is disabled"):
        controller.trial(candidate["id"])
    assert not backend.runs


def test_default_policy_is_off(tmp_path):
    controller, _, _ = setup(tmp_path, enabled=False)
    with pytest.raises(ValueError, match="disabled"):
        controller.report_failure(failure(), session_id="chat")


@pytest.mark.parametrize("change", ["session", "deployment", "artifact"])
def test_stale_candidate_cannot_run(tmp_path, change):
    controller, backend, session = setup(tmp_path)
    candidate = proposed(controller)
    if change == "session":
        session.request["messages"].append({"role": "user", "content": "new"})
    elif change == "deployment":
        backend.current["generation"] += 1
    else:
        path = tmp_path / candidate["id"] / "candidate" / "adapters.safetensors"
        path.write_bytes(b"corrupt")
    with pytest.raises((ValueError, RuntimeError)):
        controller.trial(candidate["id"])
    assert not backend.runs


def test_false_engagement_or_wrong_identity_is_a_contract_failure(tmp_path):
    controller, backend, session = setup(tmp_path)
    candidate = proposed(controller)
    backend.break_identity = True
    with pytest.raises(ValueError, match="execution identity"):
        controller.trial(candidate["id"])
    assert controller.get_attempt(candidate["id"])["state"] == "failed"
    assert session.active_adapter is None
    assert backend.current["generation"] == 0


def test_cancellation_preserves_deployment_and_chat(tmp_path):
    controller, backend, session = setup(tmp_path)
    candidate = proposed(controller)
    backend.raise_case = "active"
    with pytest.raises(RuntimeError, match="cancelled"):
        controller.trial(candidate["id"])
    assert session.active_adapter is None
    assert backend.current["generation"] == 0


def test_deployment_cas_prevents_promoting_outdated_validation(tmp_path):
    controller, backend, session = setup(tmp_path, auto=False)
    candidate = proposed(controller)
    result = controller.trial(candidate["id"])
    assert result["state"] == "validated"
    backend.current["generation"] += 1
    with pytest.raises(ValueError, match="generation"):
        controller.promote(candidate["id"])
    assert session.active_adapter is None


@pytest.mark.parametrize("provenance", [
    {"contains_private_memory": True},
    {"parent": {"promotion_scope": "session"}},
    {"parents": [{"data_scope": "session"}]},
])
def test_private_memory_cannot_be_laundered_into_deployment_patch(tmp_path, provenance):
    controller, _, _ = setup(tmp_path)
    attempt = controller.report_failure(failure(), session_id="chat")
    with pytest.raises(ValueError, match="private/session"):
        controller.propose(attempt["id"], program(provenance=provenance))


def test_attempt_budget_is_bounded(tmp_path):
    controller, _, _ = setup(tmp_path)
    for _ in range(3):
        controller.report_failure(failure(), session_id="chat")
    with pytest.raises(ValueError, match="budget exhausted"):
        controller.report_failure(failure(), session_id="chat")


def test_mutating_returned_receipt_cannot_change_promotion_verdict(tmp_path):
    controller, backend, _ = setup(tmp_path)
    candidate = proposed(controller)
    backend.fail_case = "heldout"
    result = controller.trial(candidate["id"])
    result["state"] = "validated"
    result["validation"]["passed"] = True
    with pytest.raises(ValueError, match="validated"):
        controller.promote(candidate["id"])


def test_error_prose_cannot_bypass_per_failure_budget(tmp_path):
    controller, _, _ = setup(tmp_path)
    for i in range(3):
        report = failure()
        report["observed_error"] += str(i)
        controller.report_failure(report, session_id="chat")
    with pytest.raises(ValueError, match="budget exhausted"):
        controller.report_failure(failure(), session_id="chat")


def test_active_task_adapter_is_never_silently_replaced(tmp_path):
    controller, _, session = setup(tmp_path)
    session.active_adapter = "trained-task-adapter"
    with pytest.raises(ValueError, match="different adapter"):
        controller.report_failure(failure(), session_id="chat")


def test_second_repair_composes_actual_prior_factors(tmp_path):
    controller, backend, _ = setup(tmp_path)
    first = proposed(controller)
    controller.trial(first["id"])
    previous = load_program(first["artifact_path"])
    second = proposed(controller)
    combined = load_program(second["artifact_path"])
    np.testing.assert_allclose(combined.factor_map["linear"][0] @ combined.factor_map["linear"][1],
                               previous.factor_map["linear"][0] @ previous.factor_map["linear"][1]
                               + np.full((2, 2), 1.0))
    assert combined.rank == previous.rank + 1
    assert backend.current["generation"] == 1  # proposed is not promoted


def test_validation_expiry_restores_route_without_erasing_new_messages(tmp_path, monkeypatch):
    controller, backend, session = setup(tmp_path, auto=False)
    candidate = proposed(controller)
    validated = controller.trial(candidate["id"])
    session.request["messages"].append({"role": "user", "content": "new message"})
    monkeypatch.setattr("mlx2.experimental.weight_space.self_repair.time.time",
                        lambda: validated["validation_expires_at"] + 1)
    with pytest.raises(ValueError, match="expired"):
        controller.promote(candidate["id"])
    assert session.active_adapter is None
    assert session.request["messages"][-1]["content"] == "new message"
    assert backend.current["generation"] == 0


def test_restart_recovers_budgets_but_never_restarts_live_trial(tmp_path):
    controller, backend, session = setup(tmp_path)
    candidate = proposed(controller)
    recovered = SelfRepairController(checkpoint=CHECKPOINT, source_revision=REVISION,
        root=tmp_path, policy=controller.policy, backend=backend, verify_tool=controller.verify_tool,
        suites=list(controller.suites.values()), sessions=[session])
    assert recovered.get_attempt(candidate["id"])["state"] == "interrupted"
    with pytest.raises(ValueError, match="compiled candidate"):
        recovered.trial(candidate["id"])
    assert recovered.failure_counts == controller.failure_counts


def test_baseline_must_execute_the_previous_exact_program(tmp_path):
    controller, backend, session = setup(tmp_path)
    candidate = proposed(controller)
    original = backend.baseline
    backend.baseline = lambda request, previous: {
        **original(request, previous), "fingerprint": "another-adapter"}
    with pytest.raises(ValueError, match="baseline execution identity"):
        controller.trial(candidate["id"])
    assert backend.current["generation"] == 0
    assert session.active_adapter is None


def test_failed_promotion_journal_rolls_back_live_default(tmp_path, monkeypatch):
    controller, backend, session = setup(tmp_path)
    candidate = proposed(controller)
    original = controller._save

    def fail_promoted(attempt):
        if attempt["state"] == "promoted":
            raise OSError("disk full")
        original(attempt)

    monkeypatch.setattr(controller, "_save", fail_promoted)
    with pytest.raises(OSError, match="disk full"):
        controller.trial(candidate["id"])
    assert backend.current["default_adapter"] is None
    assert backend.current["generation"] == 2
    assert session.active_adapter is None
    assert controller.get_attempt(candidate["id"])["state"] == "rolled_back"


def test_failed_post_promotion_rollback_preserves_recovery_required_state(tmp_path, monkeypatch):
    controller, backend, session = setup(tmp_path)
    candidate = proposed(controller)
    original = controller._save

    def fail_promoted(attempt):
        if attempt["state"] == "promoted":
            raise OSError("disk full")
        original(attempt)

    def failed_rollback(*_):
        raise RuntimeError("rollback unavailable")

    monkeypatch.setattr(controller, "_save", fail_promoted)
    monkeypatch.setattr(backend, "rollback", failed_rollback)
    with pytest.raises(RuntimeError, match="rollback unavailable"):
        controller.trial(candidate["id"])
    assert controller.get_attempt(candidate["id"])["state"] == "promotion_recovery_required"
    assert backend.current["default_adapter"] == session.active_adapter
    saved = json.loads((tmp_path / candidate["id"] / "receipt.json").read_text())
    assert saved["state"] == "promotion_recovery_required"
    recovered = SelfRepairController(checkpoint=CHECKPOINT, source_revision=REVISION,
        root=tmp_path, policy=controller.policy, backend=backend, verify_tool=controller.verify_tool,
        suites=list(controller.suites.values()), sessions=[session])
    assert recovered.get_attempt(candidate["id"])["state"] == "promoted"


def call_message(name, arguments, call_id):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def append_round(session, name, arguments, call_id):
    session.request["messages"].extend([
        call_message(name, arguments, call_id),
        {"role": "tool", "tool_call_id": call_id, "content": "{}"},
    ])
    session.revision += 1  # The live observer captures each model submission.


def observed_setup(tmp_path, *, auto=True):
    controller, backend, session = setup(tmp_path, auto=auto)
    controller.policy = RepairPolicy(True, True, auto, require_observed_failure=True)
    session.transcript_origin = "live_submission"
    session.request["tools"] = [{"type": "function", "function": {"name": "read",
                                 "parameters": {"type": "object", "required": ["path"]}}}]
    session.request["messages"].extend([
        call_message("read", failure()["failed_arguments"], "failed-read"),
        {"role": "tool", "tool_call_id": "failed-read", "content": "required field path"},
    ])
    return controller, backend, session


def test_hermes_rounds_keep_task_epoch_and_replay_before_failed_call(tmp_path):
    controller, backend, session = observed_setup(tmp_path)
    attempt = controller.report_failure(failure(), session_id="chat")
    append_round(session, "report_tool_failure", {"failure": failure()}, "report")
    append_round(session, "inspect_repair_capture", {"capture_id": "host-capture"}, "inspect")
    append_round(session, "compile_analytical_repair", {"capture_id": "host-capture"}, "compile")
    candidate = controller.propose(attempt["id"], program())
    append_round(session, "propose_repair", {"attempt_id": attempt["id"]}, "propose")
    append_round(session, "trial_candidate", {"attempt_id": attempt["id"]}, "trial")
    # Async MCP polling may continue while the actual replay runs.
    backend.during_run = lambda: append_round(
        session, "get_repair_attempt", {"attempt_id": attempt["id"]}, "poll")
    result = controller.trial(candidate["id"])
    assert result["state"] == "promoted"
    assert result["failure_source"]["kind"] == "observed_tool_call"
    assert result["failure_source"]["tool_call_id"] == "failed-read"
    assert backend.runs[-1]["messages"] == [{"role": "user", "content": "active"}]
    assert session.request["messages"][-1]["tool_call_id"] == "poll"
    assert session.active_adapter == result["candidate"]["name"]
    assert "replay_request" not in result
    assert load_program(result["artifact_path"]).provenance["task_epoch"] == result["task_epoch"]


def test_observed_failure_requires_matching_call_and_rejects_old_task(tmp_path):
    controller, _, session = observed_setup(tmp_path)
    bad = failure()
    bad["failed_arguments"] = {"filename": "different"}
    with pytest.raises(ValueError, match="matching observed"):
        controller.report_failure(bad, session_id="chat")
    session.request["messages"].append({"role": "user", "content": "another task"})
    with pytest.raises(ValueError, match="changed task"):
        controller.report_failure(failure(), session_id="chat")


def test_failed_call_replay_keeps_the_entire_preceding_transcript(tmp_path):
    controller, backend, session = observed_setup(tmp_path)
    preceding = [{"role": "system", "content": "Use read tools."},
                 {"role": "user", "content": "Earlier context."},
                 {"role": "assistant", "content": "Earlier answer."}]
    session.request["messages"][:0] = preceding
    expected = copy.deepcopy(session.request["messages"][:-2])
    candidate = proposed(controller)
    result = controller.trial(candidate["id"])
    assert result["state"] == "promoted"
    assert backend.runs[-1]["messages"] == expected
    assert result["failure_source"]["message_index"] == len(expected)


@pytest.mark.parametrize("change", ["user", "system", "developer", "schema", "past", "tool", "adapter"])
def test_task_epoch_rejects_task_schema_history_and_route_changes(tmp_path, change):
    controller, backend, session = observed_setup(tmp_path)
    candidate = proposed(controller)
    if change in {"user", "system", "developer"}:
        session.request["messages"].append({"role": change, "content": "new instructions"})
    elif change == "schema":
        session.request["tools"][0]["function"]["parameters"]["required"] = ["filename"]
    elif change == "past":
        session.request["messages"][0]["content"] = "rewritten"
    elif change == "tool":
        append_round(session, "unrelated_mutation", {}, "unrelated")
    else:
        session.active_adapter = "other-adapter"
    with pytest.raises(ValueError, match="session task changed"):
        controller.trial(candidate["id"])
    assert not backend.runs


def test_task_change_during_trial_prevents_switch_without_erasing_messages(tmp_path):
    controller, backend, session = observed_setup(tmp_path)
    candidate = proposed(controller)
    backend.during_run = lambda: session.request["messages"].append(
        {"role": "user", "content": "new task during trial"})
    with pytest.raises(ValueError, match="session task changed"):
        controller.trial(candidate["id"])
    assert session.active_adapter is None
    assert backend.current["default_adapter"] is None
    assert session.request["messages"][-1]["content"] == "new task during trial"


def test_manual_promotion_allows_repair_polling_rounds(tmp_path):
    controller, backend, session = observed_setup(tmp_path, auto=False)
    candidate = proposed(controller)
    result = controller.trial(candidate["id"])
    assert result["state"] == "validated"
    assert session.routing_override["name"] == result["candidate"]["name"]
    assert session.routing_override["fingerprint"] == result["candidate"]["fingerprint"]
    assert session.routing_override["expires_at"] is not None
    append_round(session, "read", {"path": "ok"}, "corrected-task-call")
    append_round(session, "get_repair_attempt", {"attempt_id": candidate["id"]}, "poll")
    append_round(session, "promote_validated_candidate", {"attempt_id": candidate["id"]}, "promote")
    assert controller.promote(candidate["id"])["state"] == "promoted"
    assert backend.current["generation"] == 1
    assert session.routing_override["expires_at"] is None
    assert session.request["messages"][-1]["tool_call_id"] == "promote"


def test_operator_seed_is_explicit_and_observation_policy_is_boolean(tmp_path):
    controller, _, _ = setup(tmp_path)
    result = proposed(controller)
    assert result["failure_source"]["kind"] == "operator_seed"
    with pytest.raises(ValueError, match="booleans"):
        RepairPolicy(require_observed_failure=1)


def test_seeded_matching_tool_call_cannot_claim_live_observed_failure(tmp_path):
    controller, _, session = observed_setup(tmp_path)
    session.transcript_origin = "operator_seed"
    with pytest.raises(ValueError, match="operator seed"):
        controller.report_failure(failure(), session_id="chat")
    controller.policy = RepairPolicy(True, True, True)
    result = proposed(controller)
    assert result["failure_source"]["kind"] == "operator_seed"
    assert result["failure_source"]["message_index"] == 1


def test_trial_uses_matched_greedy_sampling_without_changing_user_settings(tmp_path):
    controller, backend, session = observed_setup(tmp_path)
    session.request.update(temperature=1.3, seed=994, session_id="chat")
    candidate = proposed(controller)
    original = backend.baseline
    baseline_requests = []

    def baseline(request, previous):
        baseline_requests.append(copy.deepcopy(request))
        return original(request, previous)

    backend.baseline = baseline
    result = controller.trial(candidate["id"])
    assert result["state"] == "promoted"
    assert result["validation"]["validation_sampling"] == {"temperature": 0.0, "seed": 0}
    for request in backend.runs + baseline_requests:
        assert request["temperature"] == 0.0 and request["seed"] == 0
        assert "session_id" not in request
    assert session.request["temperature"] == 1.3 and session.request["seed"] == 994
    assert result["validation"]["live_base"]["passed"] is False


def test_unreproduced_active_failure_cannot_promote_even_when_heldout_improves(tmp_path):
    controller, backend, _ = observed_setup(tmp_path)
    candidate = proposed(controller)
    original = backend.baseline

    def baseline(request, previous):
        observation = original(request, previous)
        if request["messages"][-1]["content"] == "active":
            observation["arguments"] = {"path": "ok"}
        return observation

    backend.baseline = baseline
    result = controller.trial(candidate["id"])
    assert result["state"] == "rejected"
    assert result["validation"]["live_base"]["passed"] is True
    assert result["validation"]["replay_passed"] is False
    assert backend.current["default_adapter"] is None


def test_host_aliases_allow_real_prefixed_repair_rounds_and_bind_source_digest(tmp_path):
    controller, backend, original = observed_setup(tmp_path)
    aliases = {"hermes__report_tool_failure": "report_tool_failure",
               "hermes__compile_analytical_repair": "compile_analytical_repair",
               "hermes__propose_repair": "propose_repair", "hermes__trial_candidate": "trial_candidate"}
    session = RepairSession("chat", copy.deepcopy(original.request), transcript_origin="live_submission",
                            repair_tool_aliases=aliases)
    controller.sessions["chat"] = session
    aliases["untrusted__trial_candidate"] = "trial_candidate"
    assert "untrusted__trial_candidate" not in session.repair_tool_aliases
    with pytest.raises(TypeError):
        session.repair_tool_aliases["new"] = "trial_candidate"
    append_round(session, "hermes__report_tool_failure", {"failure": failure()}, "report")
    attempt = controller.report_failure(failure(), session_id="chat")
    append_round(session, "hermes__compile_analytical_repair", {"capture_id": "host-capture"}, "compile")
    candidate = controller.propose(attempt["id"], program())
    append_round(session, "hermes__propose_repair", {"attempt_id": attempt["id"]}, "propose")
    append_round(session, "hermes__trial_candidate", {"attempt_id": attempt["id"]}, "trial")
    result = controller.trial(candidate["id"])
    assert result["state"] == "promoted"
    assert len(result["task_epoch"]["repair_tool_aliases_digest"]) == 64
    assert backend.runs[-1]["messages"] == [{"role": "user", "content": "active"}]


@pytest.mark.parametrize("change", ["unknown_alias", "changed_alias_map"])
def test_untrusted_prefix_or_changed_host_alias_map_invalidates_attempt(tmp_path, change):
    controller, backend, session = observed_setup(tmp_path)
    candidate = proposed(controller)
    if change == "unknown_alias":
        append_round(session, "untrusted__trial_candidate", {"attempt_id": candidate["id"]}, "unknown")
    else:
        session.repair_tool_aliases = {"new__trial_candidate": "trial_candidate"}
    with pytest.raises(ValueError, match="session task changed"):
        controller.trial(candidate["id"])
    assert not backend.runs


@pytest.mark.parametrize("expire_while_waiting", [False, True])
def test_promotion_waits_for_lifecycle_without_holding_publication_session_lock(
        tmp_path, monkeypatch, expire_while_waiting):
    controller, backend, session = observed_setup(tmp_path, auto=False)
    candidate = proposed(controller)
    validated = controller.trial(candidate["id"])
    lifecycle = threading.RLock()
    entering_control = threading.Event()
    now = [validated["validated_at"] + 0.01]
    monkeypatch.setattr("mlx2.experimental.weight_space.self_repair.time.time", lambda: now[0])

    @contextmanager
    def control():
        entering_control.set()
        with lifecycle:
            yield

    backend.promotion_control_section = control
    original = backend.promote

    def promote(*args):
        with lifecycle:
            return original(*args)

    backend.promote = promote
    outcome = {}

    def invoke():
        try:
            outcome["result"] = controller.promote(candidate["id"])
        except Exception as error:  # noqa: BLE001 - capture exact thread outcome for the assertion below.
            outcome["error"] = error

    thread = threading.Thread(target=invoke, daemon=True)
    try:
        # Simulate an adapter owner draining an admission whose publication
        # callback must still capture this session before it can finish.
        with lifecycle:
            thread.start()
            assert entering_control.wait(1)
            captured = session.lock.acquire(timeout=0.5)
            assert captured, "promotion held session lock while waiting for the draining lifecycle owner"
            try:
                append_round(session, "get_repair_attempt", {"attempt_id": candidate["id"]}, "publication")
                if expire_while_waiting:
                    now[0] = validated["validation_expires_at"] + 1
            finally:
                session.lock.release()
    finally:
        thread.join(timeout=2)
    assert not thread.is_alive()
    if expire_while_waiting:
        assert isinstance(outcome.get("error"), ValueError)
        assert "expired" in str(outcome["error"])
        assert backend.current["generation"] == 0
        assert controller.get_attempt(candidate["id"])["state"] == "expired"
    else:
        assert "error" not in outcome
        assert outcome["result"]["state"] == "promoted"
        assert backend.current["generation"] == 1


class ReleasingBackend(Backend):
    """Tracks the serving registry the way ServingRepairBackend exposes it."""

    def __init__(self):
        super().__init__()
        self.registry, self.released, self.pinned = set(), [], set()

    def register(self, name, path, program):
        self.registry.add(name)
        return super().register(name, path, program)

    def unregister(self, candidate):
        if candidate["name"] in self.pinned:
            raise ValueError(f"LoRA adapter {candidate['name']!r} is in use by live requests")
        self.registry.discard(candidate["name"])
        self.released.append(candidate["name"])


def releasing_setup(tmp_path, **kwargs):
    controller, _, session = setup(tmp_path, **kwargs)
    backend = controller.backend = ReleasingBackend()
    return controller, backend, session


def test_rejected_and_failed_candidates_leave_the_serving_registry(tmp_path):
    controller, backend, _ = releasing_setup(tmp_path / "rejected")
    candidate = proposed(controller)
    backend.fail_case = "heldout"
    result = controller.trial(candidate["id"])
    assert result["state"] == "rejected" and result["candidate_released"] is True
    assert candidate["id"] not in backend.registry
    assert (tmp_path / "rejected" / candidate["id"] / "candidate").is_dir()  # artifact kept
    saved = json.loads((tmp_path / "rejected" / candidate["id"] / "receipt.json").read_text())
    assert saved["candidate_released"] is True

    controller, backend, _ = releasing_setup(tmp_path / "failed")
    candidate = proposed(controller)
    backend.raise_case = "active"
    with pytest.raises(RuntimeError, match="cancelled"):
        controller.trial(candidate["id"])
    assert controller.get_attempt(candidate["id"])["state"] == "failed"
    assert candidate["id"] not in backend.registry


def test_expired_candidate_leaves_the_serving_registry(tmp_path, monkeypatch):
    controller, backend, _ = releasing_setup(tmp_path, auto=False)
    candidate = proposed(controller)
    validated = controller.trial(candidate["id"])
    assert candidate["id"] in backend.registry  # validated: still selectable for the chat
    monkeypatch.setattr("mlx2.experimental.weight_space.self_repair.time.time",
                        lambda: validated["validation_expires_at"] + 1)
    with pytest.raises(ValueError, match="expired"):
        controller.promote(candidate["id"])
    assert controller.get_attempt(candidate["id"])["candidate_released"] is True
    assert candidate["id"] not in backend.registry


def test_pinned_rejected_candidate_release_is_retried_by_the_next_trial(tmp_path):
    controller, backend, _ = releasing_setup(tmp_path)
    first = proposed(controller)
    backend.fail_case = "heldout"
    backend.pinned.add(first["id"])
    result = controller.trial(first["id"])
    assert result["state"] == "rejected" and result["candidate_released"] is False
    assert "in use" in result["candidate_release_error"]
    assert first["id"] in backend.registry
    backend.pinned.clear()
    second = proposed(controller)
    controller.trial(second["id"])
    released = controller.get_attempt(first["id"])
    assert released["candidate_released"] is True and "candidate_release_error" not in released
    assert first["id"] not in backend.registry


def test_serving_backend_unregister_matches_exact_candidate_identity():
    from types import SimpleNamespace

    from mlx2.experimental.weight_space.self_repair import ServingRepairBackend

    unloaded = []
    registry = {"repair-a": SimpleNamespace(fingerprint="fp-a")}
    engine = SimpleNamespace(
        multi_lora=SimpleNamespace(lock=threading.Lock(), registry=registry),
        _lora_control_section=lambda: threading.RLock(),
        unload_lora_adapter=lambda name: unloaded.append(name))
    backend = ServingRepairBackend(engine, observe_job=lambda job: job)
    backend.unregister({"name": "repair-a", "fingerprint": "other"})  # same name, other content
    backend.unregister({"name": "repair-missing", "fingerprint": "fp"})
    assert unloaded == []
    backend.unregister({"name": "repair-a", "fingerprint": "fp-a"})
    assert unloaded == ["repair-a"]


def test_serving_backend_unregister_holds_lifecycle_across_check_and_unload():
    from types import SimpleNamespace

    from mlx2.experimental.weight_space.self_repair import ServingRepairBackend

    lifecycle = threading.RLock()
    registry = {"repair-a": SimpleNamespace(fingerprint="fp-a")}
    unloaded, swapped, admins = [], threading.Event(), []

    def admin_reload():  # an operator unload + same-name re-register of other content
        with lifecycle:
            registry["repair-a"] = SimpleNamespace(fingerprint="fp-b")
            swapped.set()

    class CheckLock:
        """The admin reload is attempted right after the identity check."""

        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            self.lock.acquire()

        def __exit__(self, *exc):
            self.lock.release()
            if not admins:
                admins.append(threading.Thread(target=admin_reload))
                admins[0].start()
                swapped.wait(0.2)

    def unload(name):
        with lifecycle:
            unloaded.append(registry.pop(name).fingerprint)

    engine = SimpleNamespace(multi_lora=SimpleNamespace(lock=CheckLock(), registry=registry),
                             _lora_control_section=lambda: lifecycle, unload_lora_adapter=unload)
    ServingRepairBackend(engine, observe_job=lambda job: job).unregister(
        {"name": "repair-a", "fingerprint": "fp-a"})
    admins[0].join(2)
    assert unloaded == ["fp-a"]  # never the operator's replacement content
    assert registry["repair-a"].fingerprint == "fp-b"


def test_promotion_failure_releases_undeployed_candidate(tmp_path):
    controller, backend, _ = releasing_setup(tmp_path / "manual", auto=False)
    candidate = proposed(controller)
    controller.trial(candidate["id"])
    backend.current["generation"] += 1  # deployment CAS refuses
    with pytest.raises(ValueError, match="generation"):
        controller.promote(candidate["id"])
    result = controller.get_attempt(candidate["id"])
    assert result["state"] == "promotion_failed" and result["candidate_released"] is True
    assert candidate["id"] not in backend.registry

    controller, backend, _ = releasing_setup(tmp_path / "automatic")
    candidate = proposed(controller)

    def refused(*_):
        raise ValueError("generation changed")

    backend.promote = refused
    with pytest.raises(ValueError, match="generation"):
        controller.trial(candidate["id"])
    assert controller.get_attempt(candidate["id"])["state"] == "promotion_failed"
    assert candidate["id"] not in backend.registry


def test_promotion_failure_never_unloads_a_candidate_that_is_deployed(tmp_path):
    controller, backend, _ = releasing_setup(tmp_path, auto=False)
    candidate = proposed(controller)
    controller.trial(candidate["id"])
    applied = backend.promote

    def applied_then_raised(*args):
        applied(*args)
        raise RuntimeError("lost the promotion reply")

    backend.promote = applied_then_raised
    with pytest.raises(RuntimeError, match="lost"):
        controller.promote(candidate["id"])
    result = controller.get_attempt(candidate["id"])
    assert result["state"] == "promotion_failed" and result["candidate_released"] is False
    assert "deployment default" in result["candidate_release_error"]
    assert candidate["id"] in backend.registry and not backend.released


def test_rollbacks_release_the_discarded_candidate(tmp_path, monkeypatch):
    controller, backend, _ = releasing_setup(tmp_path / "validated", auto=False)
    candidate = proposed(controller)
    controller.trial(candidate["id"])
    result = controller.rollback(candidate["id"], "operator discarded the validated trial")
    assert result["state"] == "rolled_back" and result["candidate_released"] is True
    assert candidate["id"] not in backend.registry

    controller, backend, _ = releasing_setup(tmp_path / "promoted")
    candidate = proposed(controller)
    assert controller.trial(candidate["id"])["state"] == "promoted"
    assert candidate["id"] in backend.registry  # the deployed default stays loaded
    result = controller.rollback(candidate["id"], "later preservation regression")
    assert result["state"] == "rolled_back" and result["candidate_released"] is True
    assert backend.current["default_adapter"] is None
    assert candidate["id"] not in backend.registry

    controller, backend, _ = releasing_setup(tmp_path / "journal")
    candidate = proposed(controller)
    original = controller._save

    def fail_promoted(attempt):
        if attempt["state"] == "promoted":
            raise OSError("disk full")
        original(attempt)

    monkeypatch.setattr(controller, "_save", fail_promoted)
    with pytest.raises(OSError, match="disk full"):
        controller.trial(candidate["id"])
    assert controller.get_attempt(candidate["id"])["state"] == "rolled_back"
    assert candidate["id"] not in backend.registry


def test_serving_backend_register_unloads_a_candidate_it_cannot_bind():
    from types import SimpleNamespace

    from mlx2.experimental.weight_space.self_repair import ServingRepairBackend

    calls = []

    def bind(name, token, fingerprint):
        raise ValueError("repair candidate reservation changed during load")

    engine = SimpleNamespace(
        reserve_private_weight_program=lambda name: calls.append(("reserve", name)) or "token",
        load_lora_adapter=lambda name, path: calls.append(("load", name)) or {"fingerprint": "fp"},
        bind_private_weight_program=bind,
        unload_lora_adapter=lambda name: calls.append(("unload", name)),
        release_private_weight_program=lambda name, **kw: calls.append(("release", name, kw)),
    )
    backend = ServingRepairBackend(engine, observe_job=lambda job: job)
    with pytest.raises(ValueError, match="changed during load"):
        backend.register("repair-1", "/tmp/x", SimpleNamespace(program_id="p"))
    assert calls == [("reserve", "repair-1"), ("load", "repair-1"), ("unload", "repair-1"),
                     ("release", "repair-1", {"token": "token"})]
    # If the unload fails too, the reservation stays (the candidate is hidden).
    calls.clear()

    def unload(name):
        raise ValueError("pinned")

    engine.unload_lora_adapter = unload
    with pytest.raises(ValueError, match="changed during load"):
        backend.register("repair-1", "/tmp/x", SimpleNamespace(program_id="p"))
    assert ("release", "repair-1", {"token": "token"}) not in calls
