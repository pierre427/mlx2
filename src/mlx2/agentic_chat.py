"""Template-owned opt-in self-review and bounded agentic chat contracts.

No scheduler or cache model-name switches. A pinned template declares support;
request validation and the adapter refuse unsupported verification explicitly.
"""
from __future__ import annotations

import json
from pathlib import Path

VERIFICATION_TRIGGER = "Verification protocol: enabled"
AGENTIC_DEFAULTS = {"context_limit": 262144, "output_tokens": 65536,
                    "thinking_budget": 32768, "enable_thinking": True,
                    "verification": False}


def template_options(tokenizer):
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str):
        template = getattr(getattr(tokenizer, "_tokenizer", None), "chat_template", "")
    if isinstance(template, str) and VERIFICATION_TRIGGER in template and "verification" in template:
        return {"verification": {"implemented": True, "default": False,
                                  "requires_system_message": True,
                                  "requires_thinking": True},
                "recommended_chat": dict(AGENTIC_DEFAULTS)}
    return {}


def model_chat_defaults(path):
    try:
        config = json.loads((Path(path) / "tokenizer_config.json").read_text())
        holder = type("Template", (), {"chat_template": config.get("chat_template")})()
        return template_options(holder).get("recommended_chat", {})
    except (OSError, ValueError, TypeError):
        return {}


def verification_template_kwargs(tokenizer, request):
    value = request.get("verification", False)
    if type(value) is not bool:
        raise ValueError("verification must be boolean")
    if not value:
        return {}  # preserve every ordinary prompt byte
    if "verification" not in template_options(tokenizer):
        raise ValueError("verification is not supported by this model's chat template")
    messages = request.get("messages") or []
    if (not isinstance(messages, list) or not messages or not isinstance(messages[0], dict)
                or messages[0].get("role") != "system"
            or not isinstance(messages[0].get("content"), str)
            or not messages[0]["content"].strip()):
        raise ValueError("verification requires a non-empty initial system message")
    if request.get("enable_thinking") is not True:
        raise ValueError("verification requires thinking enabled")
    return {"verification": True}


SPAWN_PARAMETERS = {"type": "object", "properties": {"tasks": {
    "type": "array", "minItems": 1, "maxItems": 5, "items": {
        "type": "object", "properties": {key: {"type": "string", **({"minLength": 1} if key == "task" else {})}
            for key in ("task", "context", "expected_output")},
        "required": ["task", "context", "expected_output"], "additionalProperties": False}}},
    "required": ["tasks"], "additionalProperties": False}
SPAWN_DESCRIPTION = (
    "Dispatch up to five independent sub-tasks with isolated conversations, at most four "
    "running concurrently. Children cannot see your conversation. Supply everything in "
    "task, context and expected_output. Children receive only the explicitly admitted "
    "memory/skill tools; no implicit web, shell or recursive delegation. Results return "
    "in input order as {answer, status, evidence, unresolved_issues}. Check status before "
    "using an answer. Unknown findings must be UNKNOWN, never invented.")


def validate_spawn_tasks(tasks):
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 5:
        raise ValueError("spawn_subagent requires 1–5 tasks; oversized batches are rejected")
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {"task", "context", "expected_output"}:
            raise ValueError("Each task requires exactly task, context and expected_output")
        if any(not isinstance(v, str) or len(v.encode()) > 16384 for v in task.values()):
            raise ValueError("Task fields must be strings of at most 16 KiB")
        if not task["task"].strip():
            raise ValueError("A subagent task must be nonempty")
    return tasks


def parse_subagent_report(text):
    if len(text.encode()) > 4096:
        raise ValueError("Subagent report exceeds the 4 KiB result budget")
    report = json.loads(text)
    if (not isinstance(report, dict) or set(report) != {"answer", "status", "evidence", "unresolved_issues"}
            or not isinstance(report["answer"], str)
            or report["status"] not in {"completed", "unknown", "insufficient_evidence"}
            or not isinstance(report["evidence"], list)
            or not isinstance(report["unresolved_issues"], list)):
        raise ValueError("Invalid structured subagent report")
    if report["status"] != "completed" and report["answer"] != "UNKNOWN":
        raise ValueError("Unresolved subagents must answer UNKNOWN")
    return report


def validate_chat_options(options):
    if not isinstance(options, dict) or set(options) - set(AGENTIC_DEFAULTS):
        raise ValueError("Unknown per-model chat option")
    for key in ("verification", "enable_thinking"):
        if key in options and type(options[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key, low, high in (("context_limit", 1024, 262144),
                           ("output_tokens", 128, 65536), ("thinking_budget", 0, 64512)):
        if key in options and (type(options[key]) is not int or not low <= options[key] <= high):
            raise ValueError(f"Invalid {key}")
    resolved = {**AGENTIC_DEFAULTS, **options}
    if resolved["output_tokens"] >= resolved["context_limit"]:
        raise ValueError("Output must leave room for input context")
    if resolved["thinking_budget"] > resolved["output_tokens"] - 1024:
        raise ValueError("Reasoning must leave at least 1024 output tokens for the answer")
    if resolved["verification"] and not resolved["enable_thinking"]:
        raise ValueError("Verification requires thinking")
    return resolved
