#!/usr/bin/env python3
# ruff: noqa: PLC3002
"""Real-server feature smoke with bounded checks and raw-response evidence."""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
import zlib
from dataclasses import dataclass
from pathlib import Path

TOOLS = [
    {"type": "function", "function": {"name": "weather", "description": "Look up weather in a city.", "strict": True,
      "parameters": {"$defs": {"city": {"type": "string"}}, "type": "object",
                     "properties": {"city": {"$ref": "#/$defs/city"}}, "required": ["city"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "clock", "description": "Look up a time zone.",
      "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}, "required": ["zone"], "additionalProperties": False}}},
]
CONSTRAINED_WEATHER_TOOL = [{
    "type": "function",
    "function": {
        "name": "weather",
        "description": "Look up weather in a city.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "const": "Toronto"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}]
SCHEMA_REF = {
    "type": "json_schema",
    "json_schema": {"name": "answer", "strict": True, "schema": {
        "$defs": {"answer": {"type": "string", "minLength": 1, "maxLength": 24}},
        "type": "object", "properties": {"answer": {"$ref": "#/$defs/answer"}},
        "required": ["answer"], "additionalProperties": False,
    }},
}
FIXED_PROMPTS = (
    "Reply with exactly CAMPAIGN_READY.",
    "What is 37 plus 58? Reply with the number only.",
    "Name the capital of Canada in one word.",
    "Translate 'water' to French. One word.",
)


def fixed_prompt_correct(prompt: str, text: str) -> bool:
    """Small deterministic oracle for ordinary/speculative greedy comparison."""
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    expected = {
        FIXED_PROMPTS[0]: "campaign ready",
        FIXED_PROMPTS[1]: "95",
        FIXED_PROMPTS[2]: "ottawa",
        FIXED_PROMPTS[3]: "eau",
    }
    return normalized == expected[prompt]


def supports_budgeted_thinking(capabilities) -> bool:
    """A thinking budget also needs an adapter-declared close marker."""
    capabilities = set(capabilities)
    return "reasoning" in capabilities and "thinking-deferral" in capabilities


CORE_CHECK_NAMES = (
    "chat", "completions", "streaming", "stop_strings", "logprobs_including_bytes",
    "n_2", "seeded_determinism", "json_object", "strict_json_schema_ref", "regex_grammar",
    "tools_auto", "tools_required", "tools_named", "tools_parallel_false",
    "thinking_default", "thinking_budget_state_aware", "thinking_budget_history",
    "reasoning_effort", "structured_after_thinking",
    "responses_text", "responses_stream", "responses_function_roundtrip",
    "responses_store_retrieve_input_items_delete", "responses_previous_response_id",
    "responses_include_reasoning_and_logprobs",
    "messages_text", "messages_stream", "messages_system", "messages_tools_any",
    "messages_tools_named", "messages_tool_result_roundtrip",
    "messages_thinking_signature_roundtrip", "messages_count_tokens",
    "messages_max_tokens_32000", "messages_stop_sequences",
    "apc_repeated_prompt_hit", "apc_skip_writing_prefix_cache",
    "apc_sessions_park_resume_delete", "apc_admin_quiesce_suspend_resume",
    "multimodal_image", "multimodal_audio", "multimodal_plain_text",
    "speculative_greedy_equality",
)
CHECKS_BY_GROUP = {
    "core": CORE_CHECK_NAMES,
    "opt-in": ("constrained_tool_grammar", "tolerant_tool_markers", "apc_interior_checkpoints", "fly_verification"),
    "persist-seed": ("apc_persistent_seed",),
    "persist-rescan": ("apc_persistent_restart_rescan_hit",),
}
MODEL_ID = "campaign"


def _png_data_url() -> str:
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * 8 for _ in range(8))
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _wav_b64() -> str:
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)
    return base64.b64encode(out.getvalue()).decode()


class HTTP:
    def __init__(self, base: str, timeout: float, admin_token: str | None = None):
        self.base, self.timeout, self.admin_token = base.rstrip("/"), timeout, admin_token

    def request(self, method: str, path: str, body=None, *, stream=False):
        request_evidence = {"method": method, "path": path, "body": body}
        headers = {"Content-Type": "application/json"}
        if self.admin_token and path.startswith("/v1/admin"):
            headers["Authorization"] = "Bearer " + self.admin_token
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.base + path, method=method, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if stream:
                    events = []
                    for raw in response:
                        line = raw.decode(errors="replace").strip()
                        if line.startswith("data:"):
                            payload = line[5:].strip()
                            events.append(payload if payload == "[DONE]" else json.loads(payload))
                    return {
                        "status": response.status,
                        "events": events,
                        "request": request_evidence,
                    }
                raw = response.read().decode(errors="replace")
                return {
                    "status": response.status,
                    "body": json.loads(raw) if raw else None,
                    "request": request_evidence,
                }
        except urllib.error.HTTPError as error:
            raw = error.read().decode(errors="replace")
            try:
                response_body = json.loads(raw)
            except json.JSONDecodeError:
                response_body = raw
            return {
                "status": error.code,
                "body": response_body,
                "request": request_evidence,
            }

    def get(self, path):
        return self.request("GET", path)

    def post(self, path, body, *, stream=False):
        return self.request("POST", path, body, stream=stream)

    def health_error(self):
        request = urllib.request.Request(self.base + "/health", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=1.0) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            try:
                payload = json.loads(error.read())
            except (ValueError, OSError):
                return None
        except OSError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload.get("error")


@dataclass
class Outcome:
    passed: bool
    detail: str
    raw: object


class Matrix:
    def __init__(self, args, http: HTTP):
        self.args, self.http = args, http
        self.rows = []
        self.server_error = None
        self.raw_dir = args.raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def check(self, name, function=None, *, applies=True, reason="not applicable"):
        if not applies:
            self.rows.append({"name": name, "status": "SKIP", "reason": reason, "seconds": 0.0})
            print(f"SKIP {name}: {reason}", flush=True)
            return
        if self.server_error is None and self.http is not None:
            self.server_error = self.http.health_error()
        if self.server_error is not None:
            row = {
                "name": name,
                "status": "FAIL",
                "reason": f"server health error: {self.server_error}",
                "seconds": 0.0,
            }
            self.rows.append(row)
            print(f"FAIL {name}: {row['reason']}", flush=True)
            return
        result, failure = {}, []

        def target():
            try:
                result["value"] = function()
            except Exception as error:  # noqa: BLE001 - each check must become a result
                failure.append(f"{type(error).__name__}: {error}")

        started = time.monotonic()
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        deadline = started + self.args.check_timeout
        while thread.is_alive() and time.monotonic() < deadline:
            thread.join(min(0.5, max(0.0, deadline - time.monotonic())))
            if thread.is_alive() and self.http is not None:
                self.server_error = self.http.health_error()
                if self.server_error is not None:
                    break
        elapsed = round(time.monotonic() - started, 3)
        raw_path = self.raw_dir / f"{len(self.rows):03d}-{re.sub('[^a-z0-9]+', '-', name.lower()).strip('-')}.json"
        if self.server_error is not None:
            row = {
                "name": name,
                "status": "FAIL",
                "reason": f"server health error: {self.server_error}",
                "seconds": elapsed,
            }
        elif thread.is_alive():
            row = {"name": name, "status": "FAIL", "reason": f"timeout after {self.args.check_timeout}s", "seconds": elapsed}
        elif failure:
            row = {"name": name, "status": "FAIL", "reason": failure[0], "seconds": elapsed}
        else:
            value = result.get("value")
            if not isinstance(value, Outcome):
                value = Outcome(bool(value), "", value)
            raw_path.write_text(json.dumps(value.raw, indent=2, ensure_ascii=False, default=str) + "\n")
            row = {"name": name, "status": "PASS" if value.passed else "FAIL", "reason": value.detail,
                   "seconds": elapsed, "raw": str(raw_path)}
        self.rows.append(row)
        print(f"{row['status']} {name}{': ' + row['reason'] if row.get('reason') else ''}", flush=True)

    def skip_unselected(self, names, group):
        if self.args.group == group:
            return
        for name in names:
            self.check(name, applies=False, reason=f"runs in {group} server cycle")


def _chat_body(prompt, **extra):
    return {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 96, "enable_thinking": False, **extra}


def _default_chat(http, prompt):
    return http.post(
        "/v1/chat/completions",
        {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 96,
        },
    )


def _chat(http, prompt, **extra):
    return http.post("/v1/chat/completions", _chat_body(prompt, **extra), stream=bool(extra.get("stream")))


def _message(reply):
    return ((reply.get("body") or {}).get("choices") or [{}])[0].get("message", {})


def _text(reply):
    return _message(reply).get("content") or ""


def _finish(reply):
    return (((reply.get("body") or {}).get("choices") or [{}])[0]).get("finish_reason")


def _error_message(reply):
    error = (reply.get("body") or {}).get("error") or {}
    return error.get("message", "") if isinstance(error, dict) else ""


def _structured_json(reply, *, require_answer=False, detail="valid JSON object"):
    """Keep the complete failing HTTP reply, including qualification receipts."""
    parsed = None
    if reply.get("status") == 200:
        try:
            parsed = json.loads(_text(reply))
        except (TypeError, json.JSONDecodeError):
            pass
    passed = isinstance(parsed, dict) and (
        not require_answer or isinstance(parsed.get("answer"), str)
    )
    reason = detail if passed else (
        f"http={reply.get('status')} error={_error_message(reply)}"
    )
    return Outcome(passed, reason, reply)


def _cached_tokens(reply):
    body = reply.get("body") or {}
    cached = (body.get("usage") or {}).get("prompt_tokens_details", {}).get(
        "cached_tokens"
    )
    if cached is None:
        cached = (body.get("mlx2") or {}).get("cached_tokens", 0)
    return int(cached or 0)


def _single_tool_enforced(reply):
    calls = _message(reply).get("tool_calls") or []
    error_message = _error_message(reply)
    rejected_parallel = (
        reply.get("status") == 502
        and (
            "parallel_tool_calls:false permits at most one tool call" in error_message
            or "model emitted parallel calls while parallel_tool_calls was false"
            in error_message
        )
    )
    return Outcome(
        (reply.get("status") == 200 and len(calls) == 1) or rejected_parallel,
        (
            "one tool call returned"
            if reply.get("status") == 200 and len(calls) == 1
            else "parallel model output rejected fail-closed"
            if rejected_parallel
            else f"http={reply.get('status')} error={error_message}"
        ),
        reply,
    )


def _admin_suspend_resume(http, *, timeout):
    prompt = "admin warm filler " * 96 + " Reply ADMIN_OK."
    sid = "quality-campaign-admin-suspend-190919"
    cold = _chat(http, prompt, session_id=sid)
    quiesce = http.post("/v1/admin/quiesce", {"suspend": True})
    states = []
    deadline = time.monotonic() + min(30, timeout / 2)
    while time.monotonic() < deadline:
        state = http.get("/v1/admin/state")
        states.append(state)
        if (state.get("body") or {}).get("state") == "suspended":
            break
        time.sleep(0.05)
    else:
        return Outcome(False, "admin service did not reach suspended", {"cold": cold, "quiesce": quiesce, "states": states})
    resume = http.post("/v1/admin/resume", {"prefetch_sessions": [{"tenant": "default", "session_id": sid}]})
    session_states = []
    while time.monotonic() < deadline:
        session = http.get(f"/v1/apc/sessions/{sid}")
        session_states.append(session)
        if (session.get("body") or {}).get("state") == "resident":
            break
        time.sleep(0.05)
    else:
        return Outcome(False, "suspended session did not prefetch", {"cold": cold, "quiesce": quiesce, "states": states, "resume": resume, "session_states": session_states})
    warm_a = _chat(http, prompt, session_id=sid)
    warm_b = _chat(http, prompt, session_id=sid)
    passed = (
        cold["status"] == warm_a["status"] == warm_b["status"] == 200
        and quiesce["status"] in {200, 202}
        and resume["status"] in {200, 202}
        and _cached_tokens(warm_a) > 0
        and _cached_tokens(warm_b) > 0
        and _text(warm_a) == _text(warm_b)
    )
    return Outcome(passed, "suspended state restored; warm outputs agree", {"cold": cold, "quiesce": quiesce, "states": states, "resume": resume, "session_states": session_states, "warm_a": warm_a, "warm_b": warm_b})


def _fly_verification(http):
    before = http.get("/v1/status")
    reply = _chat(
        http,
        "Explain two benefits of caching.",
        max_tokens=128,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )
    after = http.get("/v1/status")
    receipt = (reply.get("body") or {}).get("mlx2") or {}
    mechanism = receipt.get("mtp") or receipt.get("speculation") or {}
    passed = (
        reply["status"] == 200
        and mechanism.get("verification") == "fly"
        and not mechanism.get("fly_disabled", False)
    )
    return Outcome(passed, f"verification={mechanism.get('verification')} disabled={mechanism.get('fly_disabled')}", {"before": before, "reply": reply, "after": after})


def _ok_text(reply):
    return Outcome(reply["status"] == 200 and bool(_text(reply).strip()), f"http={reply['status']} finish={_finish(reply)}", reply)


def _stream_text(reply):
    pieces = []
    for event in reply.get("events", []):
        if not isinstance(event, dict):
            continue
        for choice in event.get("choices", []):
            pieces.append((choice.get("delta") or {}).get("content") or "")
    return "".join(pieces)


def core_checks(matrix: Matrix):
    h, caps = matrix.http, matrix.args.capabilities
    text, tools, grammar, thinking = "text" in caps, "tools" in caps, "grammar" in caps, "reasoning" in caps
    budgeted_thinking = supports_budgeted_thinking(caps)
    matrix.check("chat", lambda: _ok_text(_chat(h, "Reply with exactly CHAT_OK.")), applies=text)
    matrix.check("completions", lambda: (lambda r: Outcome(r["status"] == 200 and bool(((r.get("body") or {}).get("choices") or [{}])[0].get("text", "").strip()), f"http={r['status']}", r))(h.post("/v1/completions", {"model": MODEL_ID, "prompt": "Reply with exactly COMPLETION_OK.", "temperature": 0, "max_tokens": 32})), applies=text)
    matrix.check("streaming", lambda: (lambda r: Outcome(r["status"] == 200 and bool(_stream_text(r).strip()) and "[DONE]" in r["events"], "typed SSE plus DONE", r))(_chat(h, "Reply with exactly STREAM_OK.", stream=True, stream_options={"include_usage": True})), applies=text)
    matrix.check("stop_strings", lambda: (lambda r: Outcome(r["status"] == 200 and "CAMPAIGN_STOP" not in _text(r) and _finish(r) == "stop", f"finish={_finish(r)}", r))(_chat(h, "Repeat this line exactly and output nothing else: A CAMPAIGN_STOP B", stop=["CAMPAIGN_STOP"])), applies=text)
    matrix.check("logprobs_including_bytes", lambda: (lambda r: Outcome(r["status"] == 200 and any(item.get("bytes") is not None for item in (((r.get("body") or {}).get("choices") or [{}])[0].get("logprobs") or {}).get("content", [])), "logprob bytes present", r))(_chat(h, "Say apple.", max_tokens=8, logprobs=True, top_logprobs=2)), applies=text)
    matrix.check("n_2", lambda: (lambda r: Outcome(r["status"] == 200 and len((r.get("body") or {}).get("choices", [])) == 2, "two choices", r))(_chat(h, "Name a boat.", n=2, temperature=0.7, seed=19, max_tokens=16)), applies=text)

    def seeded():
        request = {
            "temperature": 0.8,
            "seed": 1909,
            "max_tokens": 24,
            "enable_thinking": False,
        }
        a = _chat(h, "Give one unusual color name.", **request)
        b = _chat(h, "Give one unusual color name.", **request)
        return Outcome(a["status"] == b["status"] == 200 and _text(a) == _text(b) and bool(_text(a)), "same seed produced identical output", {"a": a, "b": b})
    matrix.check("seeded_determinism", seeded, applies=text)
    matrix.check("json_object", lambda: _structured_json(_chat(h, "Return JSON with key answer and value yes.", enable_thinking=False, response_format={"type": "json_object"})), applies=grammar, reason="adapter does not declare grammar")
    matrix.check("strict_json_schema_ref", lambda: _structured_json(_chat(h, "Answer yes as JSON.", enable_thinking=False, response_format=SCHEMA_REF), require_answer=True, detail="local $ref resolved"), applies=grammar, reason="adapter does not declare grammar")
    matrix.check("regex_grammar", lambda: (lambda r: Outcome(r["status"] == 200 and re.fullmatch(r"(?:YES|NO)", _text(r).strip()) is not None, "regex matched", r))(_chat(h, "Answer YES or NO: is water wet?", enable_thinking=False, grammar=r"(?:YES|NO)")), applies=grammar, reason="adapter does not declare grammar")

    matrix.check("tools_auto", lambda: (lambda r: Outcome(r["status"] == 200 and bool(_message(r).get("tool_calls") or _text(r).strip()), "valid auto response", r))(_chat(h, "Use the weather tool for Toronto.", tools=TOOLS, tool_choice="auto", max_tokens=128)), applies=tools, reason="adapter does not declare tools")
    for name, choice in (("tools_required", "required"), ("tools_named", {"type": "function", "function": {"name": "weather"}})):
        matrix.check(name, lambda choice=choice: (lambda r: Outcome(r["status"] == 200 and bool(_message(r).get("tool_calls")), "required tool call emitted", r))(_chat(h, "Use the weather tool for Toronto.", tools=TOOLS, tool_choice=choice, max_tokens=128)), applies=tools, reason="adapter does not declare tools")
    matrix.check("tools_parallel_false", lambda: _single_tool_enforced(_chat(h, "Use a tool for Toronto and UTC.", tools=TOOLS, tool_choice="required", parallel_tool_calls=False, max_tokens=128)), applies=tools, reason="adapter does not declare tools")

    matrix.check("thinking_default", lambda: (lambda r: Outcome(r["status"] == 200 and (bool(_message(r).get("reasoning_content")) or bool(_text(r))), "default request completed", r))(_default_chat(h, "What is 19+23?")), applies=thinking, reason="adapter does not declare reasoning")
    for name, mode in (("thinking_budget_state_aware", "state_aware"), ("thinking_budget_history", "history")):
        matrix.check(name, lambda mode=mode: _ok_text(_chat(h, "What is 41+1?", enable_thinking=True, thinking_budget=128, thinking_budget_mode=mode, max_tokens=256)), applies=budgeted_thinking, reason="adapter does not declare a thinking-close marker")
    matrix.check("reasoning_effort", lambda: (lambda r: Outcome(r["status"] == 200 and bool(_message(r).get("reasoning_content") or _text(r)), "reasoning or answer emitted", r))(_chat(h, "What is 6 times 7?", enable_thinking=True, reasoning_effort="low", max_tokens=256)), applies=thinking, reason="adapter does not declare reasoning")
    matrix.check("structured_after_thinking", lambda: _structured_json(_chat(h, "Think, then answer yes as JSON.", enable_thinking=True, thinking_budget=128, response_format=SCHEMA_REF, max_tokens=256), require_answer=True, detail="thinking deferral produced schema output"), applies=thinking and grammar and "thinking-deferral" in caps, reason="adapter does not declare reasoning-to-grammar deferral")
    responses_checks(matrix, applies=text)
    messages_checks(matrix, applies=text)
    apc_checks(matrix, applies="apc" in caps)
    multimodal_checks(matrix)
    speculative_equality(matrix)


def responses_checks(matrix: Matrix, *, applies: bool):
    h = matrix.http
    def response_text(body):
        return "".join(part.get("text", "") for item in body.get("output", []) if item.get("type") == "message" for part in item.get("content", []) if part.get("type") == "output_text")
    matrix.check("responses_text", lambda: (lambda r: Outcome(r["status"] == 200 and bool(response_text(r["body"]).strip()), "output_text present", r))(h.post("/v1/responses", {"model": MODEL_ID, "temperature": 0, "input": "Reply with RESPONSES_OK.", "max_output_tokens": 64, "reasoning": {"effort": "none"}})), applies=applies)
    matrix.check("responses_stream", lambda: (lambda r: Outcome(r["status"] == 200 and any(isinstance(e, dict) and e.get("type") == "response.completed" for e in r["events"]), "typed response.completed", r))(h.post("/v1/responses", {"model": MODEL_ID, "temperature": 0, "input": "Reply with STREAM_OK.", "stream": True, "reasoning": {"effort": "none"}}, stream=True)), applies=applies)

    def tool_roundtrip():
        first = h.post("/v1/responses", {"model": MODEL_ID, "temperature": 0, "input": "Use weather for Toronto.", "tools": [{"type": "function", "name": "weather", "description": "Weather", "parameters": TOOLS[0]["function"]["parameters"], "strict": True}], "tool_choice": {"type": "function", "name": "weather"}, "max_output_tokens": 256})
        calls = [item for item in (first.get("body") or {}).get("output", []) if item.get("type") == "function_call"]
        if not calls:
            return Outcome(False, "no function_call", first)
        call = calls[0]
        second = h.post("/v1/responses", {"model": MODEL_ID, "temperature": 0, "input": [{"type": "message", "role": "user", "content": "Use weather for Toronto."}, {"type": "function_call", "call_id": call["call_id"], "name": call["name"], "arguments": call["arguments"]}, {"type": "function_call_output", "call_id": call["call_id"], "output": "21 C"}], "max_output_tokens": 256, "reasoning": {"effort": "none"}})
        return Outcome(second["status"] == 200 and bool(response_text(second["body"]).strip()), "function output accepted", {"first": first, "second": second})
    matrix.check("responses_function_roundtrip", tool_roundtrip, applies=applies and "tools" in matrix.args.capabilities, reason="adapter does not declare tools")

    state = {}
    def store_lifecycle():
        first = h.post("/v1/responses", {"model": MODEL_ID, "temperature": 0, "input": "Remember CAMPAIGN_19.", "store": True, "max_output_tokens": 64, "reasoning": {"effort": "none"}})
        rid = (first.get("body") or {}).get("id"); state["id"] = rid
        get = h.get(f"/v1/responses/{rid}") if rid else {"status": -1}
        items = h.get(f"/v1/responses/{rid}/input_items") if rid else {"status": -1}
        delete = h.request("DELETE", f"/v1/responses/{rid}") if rid else {"status": -1}
        return Outcome(first["status"] == get["status"] == items["status"] == delete["status"] == 200, "store/retrieve/input_items/delete", {"create": first, "retrieve": get, "items": items, "delete": delete})
    matrix.check("responses_store_retrieve_input_items_delete", store_lifecycle, applies=applies)
    def previous():
        first = h.post("/v1/responses", {"model": MODEL_ID, "temperature": 0, "input": "Remember BLUE.", "store": True, "max_output_tokens": 64, "reasoning": {"effort": "none"}})
        rid = (first.get("body") or {}).get("id")
        second = h.post("/v1/responses", {"model": MODEL_ID, "temperature": 0, "input": "What color?", "previous_response_id": rid, "store": True, "max_output_tokens": 64, "reasoning": {"effort": "none"}})
        return Outcome(second["status"] == 200 and (second.get("body") or {}).get("previous_response_id") == rid, "previous response linked", {"first": first, "second": second})
    matrix.check("responses_previous_response_id", previous, applies=applies)
    def include():
        r = h.post("/v1/responses", {"model": MODEL_ID, "temperature": 0, "input": "Say yes.", "include": ["reasoning.encrypted_content", "message.output_text.logprobs"], "top_logprobs": 1, "reasoning": {"effort": "low"}, "max_output_tokens": 256})
        output = (r.get("body") or {}).get("output", [])
        encrypted = any(item.get("type") == "reasoning" and item.get("encrypted_content") for item in output)
        logs = any(part.get("logprobs") for item in output for part in item.get("content", []))
        return Outcome(r["status"] == 200 and encrypted and logs, "encrypted reasoning and output logprobs included", r)
    matrix.check("responses_include_reasoning_and_logprobs", include, applies=applies and "reasoning" in matrix.args.capabilities, reason="adapter does not declare reasoning")


def messages_checks(matrix: Matrix, *, applies: bool):
    h = matrix.http
    def post(body, stream=False):
        return h.post("/v1/messages", {"model": MODEL_ID, "max_tokens": 256, "temperature": 0, "thinking": {"type": "disabled"}, **body}, stream=stream)
    def blocks(reply, kind):
        return [item for item in (reply.get("body") or {}).get("content", []) if item.get("type") == kind]
    matrix.check("messages_text", lambda: (lambda r: Outcome(r["status"] == 200 and bool(blocks(r, "text")), "text block", r))(post({"messages": [{"role": "user", "content": "Reply MESSAGE_OK."}]})), applies=applies)
    matrix.check("messages_stream", lambda: (lambda r: Outcome(r["status"] == 200 and any(isinstance(e, dict) and e.get("type") == "message_stop" for e in r["events"]), "typed message_stop", r))(post({"messages": [{"role": "user", "content": "Reply STREAM_OK."}], "stream": True}, stream=True)), applies=applies)
    matrix.check("messages_system", lambda: (lambda r: Outcome(r["status"] == 200 and bool(blocks(r, "text")), "system accepted", r))(post({"system": "Be concise.", "messages": [{"role": "user", "content": "Say OK."}]})), applies=applies)
    anthropic_tools = [{"name": t["function"]["name"], "description": t["function"]["description"], "input_schema": t["function"]["parameters"]} for t in TOOLS]
    for name, choice in (("messages_tools_any", {"type": "any"}), ("messages_tools_named", {"type": "tool", "name": "weather"})):
        matrix.check(name, lambda choice=choice: (lambda r: Outcome(r["status"] == 200 and bool(blocks(r, "tool_use")), "tool_use block", r))(post({"messages": [{"role": "user", "content": "Use weather for Toronto."}], "tools": anthropic_tools, "tool_choice": choice})), applies=applies and "tools" in matrix.args.capabilities, reason="adapter does not declare tools")
    def tool_result():
        first = post({"messages": [{"role": "user", "content": "Use weather for Toronto."}], "tools": anthropic_tools, "tool_choice": {"type": "tool", "name": "weather"}})
        calls = blocks(first, "tool_use")
        if not calls: return Outcome(False, "no tool_use", first)
        second = post({"messages": [{"role": "user", "content": "Use weather for Toronto."}, {"role": "assistant", "content": (first.get("body") or {}).get("content", [])}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": calls[0]["id"], "content": "21 C"}]}]})
        return Outcome(second["status"] == 200 and bool(blocks(second, "text")), "tool_result accepted", {"first": first, "second": second})
    matrix.check("messages_tool_result_roundtrip", tool_result, applies=applies and "tools" in matrix.args.capabilities, reason="adapter does not declare tools")
    def thinking_signature():
        first = post({"messages": [{"role": "user", "content": "Think about 6*7."}], "thinking": {"type": "enabled", "budget_tokens": 1024}, "max_tokens": 2048})
        thoughts = blocks(first, "thinking")
        if not thoughts or not thoughts[0].get("signature"): return Outcome(False, "missing signed thinking block", first)
        second = post({"messages": [{"role": "user", "content": "Think about 6*7."}, {"role": "assistant", "content": (first.get("body") or {}).get("content", [])}, {"role": "user", "content": "Continue."}], "thinking": {"type": "enabled", "budget_tokens": 1024}, "max_tokens": 2048})
        return Outcome(second["status"] == 200, "signature roundtrip accepted", {"first": first, "second": second})
    matrix.check("messages_thinking_signature_roundtrip", thinking_signature, applies=applies and "thinking-deferral" in matrix.args.capabilities, reason="adapter does not declare budgeted thinking")
    matrix.check("messages_count_tokens", lambda: (lambda r: Outcome(r["status"] == 200 and (r.get("body") or {}).get("input_tokens", 0) > 0, "positive token count", r))(h.post("/v1/messages/count_tokens", {"model": MODEL_ID, "messages": [{"role": "user", "content": "Count this."}]})), applies=applies)
    matrix.check("messages_max_tokens_32000", lambda: (lambda r: Outcome(r["status"] == 200, "32K accepted by 2M ceiling", r))(post({"messages": [{"role": "user", "content": "Reply briefly."}], "max_tokens": 32000})), applies=applies)
    matrix.check("messages_stop_sequences", lambda: (lambda r: Outcome(r["status"] == 200 and (r.get("body") or {}).get("stop_reason") == "stop_sequence", "stop_sequence reported", r))(post({"messages": [{"role": "user", "content": "Repeat this line exactly and output nothing else: A END B"}], "stop_sequences": ["END"], "thinking": {"type": "disabled"}})), applies=applies)


def apc_checks(matrix: Matrix, *, applies: bool):
    h = matrix.http
    def repeat():
        prompt = "Remember marker APC_1909. " + "cache filler " * 96 + "Reply APC_1909."
        cold, warm = _chat(h, prompt), _chat(h, prompt)
        cached = ((warm.get("body") or {}).get("usage") or {}).get("prompt_tokens_details", {}).get("cached_tokens")
        cached = cached if cached is not None else ((warm.get("body") or {}).get("mlx2") or {}).get("cached_tokens", 0)
        return Outcome(cold["status"] == warm["status"] == 200 and cached > 0, f"cached_tokens={cached}", {"cold": cold, "warm": warm})
    matrix.check("apc_repeated_prompt_hit", repeat, applies=applies)
    def suppress():
        prompt = "Unique no-store marker 190919. " + "suppress filler " * 96
        a = _chat(h, prompt, skip_writing_prefix_cache=True)
        b = _chat(h, prompt, skip_writing_prefix_cache=True)
        cached_a, cached_b = _cached_tokens(a), _cached_tokens(b)
        return Outcome(a["status"] == b["status"] == 200 and cached_b == cached_a, f"cached_tokens={cached_a}->{cached_b}", {"a": a, "b": b})
    matrix.check("apc_skip_writing_prefix_cache", suppress, applies=applies)
    def sessions():
        sid = "quality-campaign-190919"
        seeded = _chat(h, "session filler " * 96 + " Reply SESSION_OK.", session_id=sid)
        parked = h.post(f"/v1/apc/sessions/{sid}/park", {"ttl_seconds": 120})
        deadline = time.monotonic() + 20
        state = h.get(f"/v1/apc/sessions/{sid}")
        while (state.get("body") or {}).get("state") != "disk" and time.monotonic() < deadline:
            time.sleep(.1); state = h.get(f"/v1/apc/sessions/{sid}")
        resumed = h.post(f"/v1/apc/sessions/{sid}/resume", {})
        restored = _chat(h, "session filler " * 96 + " Reply SESSION_OK.", session_id=sid)
        deleted = h.request("DELETE", f"/v1/apc/sessions/{sid}")
        return Outcome(all(row["status"] in {200, 202} for row in (seeded, parked, state, resumed, restored, deleted)), "park/resume/delete completed", {"seeded": seeded, "parked": parked, "state": state, "resumed": resumed, "restored": restored, "deleted": deleted})
    matrix.check("apc_sessions_park_resume_delete", sessions, applies=applies)
    matrix.check("apc_admin_quiesce_suspend_resume", lambda: _admin_suspend_resume(h, timeout=matrix.args.check_timeout), applies=applies and bool(matrix.args.admin_token), reason="admin token not configured")


def opt_in_checks(matrix: Matrix):
    h, caps = matrix.http, matrix.args.capabilities
    tools = "tools" in caps
    def constrained_tool():
        request = {
            "tools": CONSTRAINED_WEATHER_TOOL,
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "max_tokens": 128,
        }
        reply = _chat(h, "Use weather for Toronto.", **request)
        calls = _message(reply).get("tool_calls") or []
        passed = (
            reply["status"] == 200
            and len(calls) == 1
            and calls[0]["function"]["name"] == "weather"
            and json.loads(calls[0]["function"]["arguments"])
            == {"city": "Toronto"}
        )
        return Outcome(
            passed,
            "constrained tool call emitted",
            {"request": request, "response": reply},
        )

    matrix.check(
        "constrained_tool_grammar",
        constrained_tool,
        applies=tools,
        reason="adapter does not declare tools",
    )
    matrix.check("tolerant_tool_markers", lambda: (lambda r: Outcome(r["status"] == 200, "tolerant parser request completed", r))(_chat(h, "Use weather for Toronto.", tools=TOOLS, tool_choice="auto", max_tokens=128)), applies=tools, reason="adapter does not declare tools")
    def interior():
        before = h.get("/v1/status")
        reply = _chat(h, "interior checkpoint filler " * 256 + " Reply INTERIOR_OK.", max_tokens=32)
        after = h.get("/v1/status")
        bc = ((before.get("body") or {}).get("counts") or {})
        ac = ((after.get("body") or {}).get("counts") or {})
        delta = ac.get("apc_interior_checkpoints_captured", 0) - bc.get("apc_interior_checkpoints_captured", 0)
        return Outcome(reply["status"] == 200 and delta > 0, f"captured_delta={delta}", {"before": before, "reply": reply, "after": after})
    matrix.check(
        "apc_interior_checkpoints",
        interior,
        applies="apc-interior" in caps,
        reason="selected route does not support interior APCv2 checkpoints",
    )
    matrix.check("fly_verification", lambda: _fly_verification(h), applies=matrix.args.fly, reason="FLy applies only to native-MTP/external-draft routes")


def persistence_check(matrix: Matrix, *, rescan: bool):
    h = matrix.http
    prompt = "PERSIST_190919 " + "persistent cache filler " * 256 + " Reply PERSIST_190919."
    reply = _chat(h, prompt, session_id="persist-190919", max_tokens=32)
    cached = ((reply.get("body") or {}).get("mlx2") or {}).get("cached_tokens", 0)
    if rescan:
        matrix.check("apc_persistent_restart_rescan_hit", lambda: Outcome(reply["status"] == 200 and cached > 0, f"cached_tokens={cached}", reply))
    else:
        matrix.check("apc_persistent_seed", lambda: Outcome(reply["status"] == 200, "persistent entry seeded for restart", reply))


def multimodal_checks(matrix: Matrix):
    if "vision" not in matrix.args.capabilities:
        matrix.check("multimodal_image", applies=False, reason="adapter is text-only")
    else:
        body = _chat_body("unused", max_tokens=16)
        body["messages"] = [{"role": "user", "content": [{"type": "input_image", "image_url": _png_data_url()}, {"type": "text", "text": "What solid color is this?"}]}]
        matrix.check("multimodal_image", lambda: _ok_text(matrix.http.post("/v1/chat/completions", body)))
    if "audio" not in matrix.args.capabilities:
        matrix.check("multimodal_audio", applies=False, reason="adapter does not declare audio input")
    else:
        body = _chat_body("unused", max_tokens=16)
        body["messages"] = [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": _wav_b64(), "format": "wav"}}, {"type": "text", "text": "Describe this audio briefly."}]}]
        matrix.check("multimodal_audio", lambda: _ok_text(matrix.http.post("/v1/chat/completions", body)))
    matrix.check("multimodal_plain_text", lambda: _ok_text(_chat(matrix.http, "Reply with TEXT_OK.")), applies="vision" in matrix.args.capabilities, reason="adapter is text-only")


def speculative_equality(matrix: Matrix):
    if not matrix.args.speculative:
        matrix.check("speculative_greedy_equality", applies=False, reason="ordinary reference route")
        return
    outputs = []
    for prompt in FIXED_PROMPTS:
        reply = _chat(matrix.http, prompt, enable_thinking=False, max_tokens=64)
        outputs.append({"prompt": prompt, "status": reply["status"], "text": _text(reply), "reply": reply})
    current = {row["prompt"]: row["text"] for row in outputs if row["status"] == 200 and row["text"]}
    baseline_path = matrix.args.ordinary_baseline
    if not baseline_path or not baseline_path.is_file():
        matrix.check("speculative_greedy_equality", applies=False, reason="ordinary baseline not yet available")
        return
    baseline = json.loads(baseline_path.read_text())
    same = sum(current.get(prompt) == text for prompt, text in baseline.items())
    correctness = sum(
        fixed_prompt_correct(prompt, baseline[prompt])
        and fixed_prompt_correct(prompt, current.get(prompt, ""))
        for prompt in baseline
    )
    total = len(baseline)
    matrix.check(
        "speculative_greedy_equality",
        lambda: Outcome(
            total == len(FIXED_PROMPTS) and correctness == total,
            f"exact={same}/{total}; semantically_correct={correctness}/{total}",
            {"baseline": baseline, "speculative": current},
        ),
    )


def write_ordinary_baseline(matrix: Matrix):
    if matrix.server_error or matrix.args.speculative or not matrix.args.ordinary_baseline:
        return
    values = {}
    for prompt in FIXED_PROMPTS:
        reply = _chat(matrix.http, prompt, enable_thinking=False, max_tokens=64)
        if reply["status"] == 200 and _text(reply): values[prompt] = _text(reply)
    matrix.args.ordinary_baseline.parent.mkdir(parents=True, exist_ok=True)
    matrix.args.ordinary_baseline.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n")


def main(argv=None):
    global MODEL_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--group", choices=("core", "opt-in", "persist-seed", "persist-rescan"), default="core")
    parser.add_argument("--capability", dest="capabilities", action="append", default=[])
    parser.add_argument("--speculative", action="store_true")
    parser.add_argument("--fly", action="store_true")
    parser.add_argument("--ordinary-baseline", type=Path)
    parser.add_argument("--admin-token-file", type=Path)
    parser.add_argument("--check-timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    MODEL_ID = args.model_id
    if args.check_timeout <= 0: parser.error("--check-timeout must be positive")
    args.capabilities = set(args.capabilities)
    admin_token = args.admin_token_file.read_text().strip() if args.admin_token_file else None
    args.admin_token = admin_token
    matrix = Matrix(args, HTTP(args.url, args.check_timeout, admin_token))
    if args.group == "core":
        core_checks(matrix); write_ordinary_baseline(matrix)
    elif args.group == "opt-in":
        opt_in_checks(matrix)
    elif args.group == "persist-seed":
        persistence_check(matrix, rescan=False)
    else:
        persistence_check(matrix, rescan=True)
    report = {
        "schema": "mlx2.quality-feature-smoke.v1", "model": args.model, "route": args.route,
        "group": args.group, "checks": matrix.rows,
        "summary": {status: sum(row["status"] == status for row in matrix.rows) for status in ("PASS", "FAIL", "SKIP")},
        "passed": not any(row["status"] == "FAIL" for row in matrix.rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"SUMMARY PASS={report['summary']['PASS']} FAIL={report['summary']['FAIL']} SKIP={report['summary']['SKIP']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
