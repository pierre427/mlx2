"""Sage opt-in contracts: CPU synthetic controls, no model/quality claims."""
import json
import threading
from types import SimpleNamespace

import pytest

from mlx2.adapters.flash_next import chat_template
from mlx2.agentic_chat import (
    AGENTIC_DEFAULTS,
    VERIFICATION_TRIGGER,
    parse_subagent_report,
    template_options,
    validate_chat_options,
    validate_spawn_tasks,
)
from mlx2.desktop_agents import Agents
from mlx2.server import validate_request
from mlx2.serving import render_prompt_tokens

SYSTEM = [{"role": "system", "content": "Check evidence."}, {"role": "user", "content": "Question"}]


def test_verification_alias_and_thinking_resolution():
    body = validate_request({"messages": SYSTEM, "chat_template_kwargs": {"verification": True}})
    assert body["verification"] is True and body["enable_thinking"] is True
    assert "chat_template_kwargs" not in body


@pytest.mark.parametrize("changes", [
    {"chat_template_kwargs": {"verification": "true"}},
    {"chat_template_kwargs": {"verification": 1}},
    {"verification": False, "chat_template_kwargs": {"verification": True}},
    {"enable_thinking": False, "verification": True},
    {"reasoning_effort": "none", "verification": True},
    {"messages": SYSTEM[1:], "verification": True},
    {"messages": [{"role": "system", "content": "  "}, *SYSTEM[1:]], "verification": True},
    {"chat_template_kwargs": {"arbitrary_template_code": True}},
    {"messages": "bad", "verification": True},
    {"messages": ["bad"], "verification": True},
])
def test_invalid_verification_fails_closed(changes):
    with pytest.raises(ValueError):
        validate_request({"messages": SYSTEM, **changes})


def test_template_passes_exact_opt_in_without_changing_off_prompts():
    seen = []
    tokenizer = SimpleNamespace(chat_template="{% if verification %}" + VERIFICATION_TRIGGER,
        apply_chat_template=lambda messages, **kwargs: seen.append((messages, kwargs)) or [1])
    base = {"messages": SYSTEM, "enable_thinking": True}
    chat_template(tokenizer, base, tokenize=True)
    chat_template(tokenizer, {**base, "verification": False}, tokenize=True)
    assert seen[0] == seen[1] and "verification" not in seen[0][1]
    chat_template(tokenizer, {**base, "verification": True}, tokenize=True)
    assert seen[-1][1]["verification"] is True
    assert "verification" in template_options(tokenizer)
    with pytest.raises(ValueError):
        chat_template(SimpleNamespace(chat_template="ordinary"), {**base, "verification": True}, tokenize=True)
    with pytest.raises(ValueError):
        render_prompt_tokens(SimpleNamespace(prompt_tokens=lambda r: [1]), {**base, "verification": True})


@pytest.mark.parametrize("tasks", [[], [{}], [{"task": "x", "context": "c"}],
    [{"task": "x", "context": "c", "expected_output": "e"}] * 6,
    [{"task": "x", "context": "c", "expected_output": "e", "command": "inject"}]])
def test_spawn_bounds_and_schema(tasks):
    with pytest.raises(ValueError): validate_spawn_tasks(tasks)


def test_budgets_reserve_answer_and_input():
    assert validate_chat_options(AGENTIC_DEFAULTS) == AGENTIC_DEFAULTS
    for options in ({"output_tokens": 1024}, {"context_limit": 65536},
                    {"verification": True, "enable_thinking": False}, {"thinking_budget": True}):
        with pytest.raises(ValueError): validate_chat_options(options)


def test_spawn_returns_ordered_structured_reports_without_parent_history(tmp_path, monkeypatch):
    controller = SimpleNamespace(home=tmp_path, preferences={}, dispatch=lambda r: {
        "url": "http://127.0.0.1:9999", "checkpoint": "synthetic@sha256:fixture",
        "source_revision": "synthetic-source", "context_limit": 262144,
        "output_tokens": 65536, "chat_options": AGENTIC_DEFAULTS})
    manager = Agents(controller)
    seen = []
    def complete(job, messages):
        task = json.loads(messages[-1]["content"])
        seen.append(messages)
        return json.dumps({"answer": task["task"], "status": "completed", "evidence": [task["context"]], "unresolved_issues": []})
    monkeypatch.setattr(manager, "_completion", complete)
    tasks = [{"task": str(i), "context": "context " + str(i), "expected_output": "one digit"} for i in range(5)]
    receipt = manager.dispatch({"command": "spawn_subagent", "tasks": tasks,
        "target": {"url": "http://127.0.0.1:9999", "pid": 1}})
    job = manager.jobs[receipt["job_id"]]; job["thread"].join(3)
    assert job["status"] == "completed"
    assert [r["answer"] for r in job["results"]] == list(map(str, range(5)))
    assert job["max_tokens"] == 65536 and job["max_rounds"] == 20
    assert all(len(messages) == 2 for messages in seen)
    assert not job["target"]
    assert "api_key" not in json.dumps(manager.snapshot())


def test_child_payload_carries_reasoning_and_verification():
    target = {"context_limit": 262144, "chat_options": {**AGENTIC_DEFAULTS, "verification": True}}
    body = json.loads(Agents._admit({"target": target, "max_tokens": 65536}, SYSTEM))
    checked = validate_request(body)
    assert checked["verification"] and checked["thinking_budget"] == 32768
    assert checked["context_limit"] == 262144


def test_unknown_report_cannot_claim_success():
    with pytest.raises(ValueError):
        parse_subagent_report(json.dumps({"answer": "guess", "status": "unknown", "evidence": [], "unresolved_issues": []}))


def test_advertised_profile_survives_observed_server_selection(monkeypatch):
    from mlx2 import inference_monitor
    from mlx2.desktop_control import DesktopController
    status = {"model": "synthetic", "artifact": "fixture", "max_context": 262144,
        "capabilities": ["text", "tools"],
        "chat_template_options": {"verification": {"implemented": True}, "recommended_chat": AGENTIC_DEFAULTS}}
    monkeypatch.setattr(inference_monitor, "_get", lambda *a: json.dumps(status).encode())
    holder = SimpleNamespace(engines={}, preferences={"context_reserve_tokens": 4096}, context_limit=lambda path: 32768)
    target = DesktopController.chat_target(holder, {"url": "http://127.0.0.1:9999"})
    assert target["context_limit"] == 262144 and target["output_tokens"] == 65536
    assert target["chat_options"]["thinking_budget"] == 32768


def test_spawn_result_is_compact_even_when_worker_prompts_are_large(tmp_path, monkeypatch):
    controller = SimpleNamespace(home=tmp_path, preferences={}, dispatch=lambda r: {"url": "http://127.0.0.1:9999"})
    manager = Agents(controller)
    manager.jobs["fixture"] = {"id": "fixture", "kind": "spawn", "status": "completed",
        "workers": [{"goal": "private background" * 5000}],
        "results": [{"answer": "UNKNOWN", "status": "unknown", "evidence": [], "unresolved_issues": []}]}
    receipt = manager.dispatch({"command": "agent_result", "job_id": "fixture"})
    assert "workers" not in receipt and len(json.dumps(receipt)) < 1000


def test_unsupported_text_template_endpoint_cannot_ignore_verification():
    from mlx2.serving import ServingEngine
    engine = ServingEngine.__new__(ServingEngine)
    engine.ready = threading.Event(); engine.ready.set()
    engine.error = None
    engine.thread = SimpleNamespace(is_alive=lambda: True)
    engine.prompt_lock = threading.RLock()
    engine.adapter = SimpleNamespace(render_prompt=lambda request: "silently ignored")
    with pytest.raises(ValueError, match="verification is not supported"):
        engine.apply_template({"messages": SYSTEM, "enable_thinking": True, "verification": True})


def test_reference_spawn_schema_accepts_self_contained_tasks_without_background():
    tasks = [{"task": "Summarize the supplied question.", "context": "", "expected_output": ""}]
    assert validate_spawn_tasks(tasks) == tasks
