"""Agent-client conformance: real Codex CLI / Claude Code request shapes.

Fixtures under ``tests/fixtures/agent_clients`` are scrubbed captures of the
requests codex-cli 0.145.0 (``wire_api = "responses"``, ``store=false``) and
Claude Code 2.1.269 sent to a scripted mlx2 server during a two-turn tool
session (see ``scripts/agent_client_conformance.py``).  Text, ids and tool
schemas are replaced by placeholders; item/block/tool types, roles, phases,
include/thinking/context-management controls and the Codex ``apply_patch``
Lark grammar are retained verbatim.  Every flag-on case asserts that its
agent-compat mechanism counter moved.
"""

import copy
import json
import threading
import urllib.error
import urllib.request
from collections import Counter
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from mlx2.agent_compat import AgentCompat, fold_system_messages
from mlx2.lark_regex import LarkGrammarError, lark_to_regex
from mlx2.reasoning_signatures import ReasoningSigner
from mlx2.server import handler_for
from mlx2.serving import Job
from mlx2.structured_output import compile_constraint

FIXTURES = Path(__file__).parent / "fixtures" / "agent_clients"
MODEL = "agent-fixture"
PATCH = "*** Begin Patch\n*** Add File: hello.txt\n+hello from mlx2\n*** End Patch\n"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


CODEX = _load("codex_0.145.0_session.json")
CLAUDE = _load("claude_code_2.1.269_session.json")
APPLY_PATCH_LARK = next(
    tool["format"]["definition"]
    for tool in CODEX[0]["tools"]
    if tool.get("type") == "custom"
)


class ScriptedEngine:
    model_path = MODEL
    max_context = 262144

    def __init__(self):
        self.counts = Counter()
        self.reasoning_signer = ReasoningSigner(b"agent-conformance")
        self.requests = []
        self.script = None

    def status(self):
        return {
            "healthy": True,
            "error": None,
            "model": MODEL,
            "structured_output": {"thinking_deferral": True},
            "settings": {"constrained_tool_grammar": False},
        }

    def batching_status(self):
        return {"schema": "mlx2.batch-runtime.v1", "gauges": {"queue_depth": 0}}

    def count_tokens(self, request):
        return 7

    def submit(self, request, *, tenant_id="default"):
        self.requests.append(copy.deepcopy(request))
        job = Job(request)
        job.tenant_id = tenant_id
        job.prompt_tokens = 6
        job.cached_tokens = 1
        if request.get("enable_thinking"):
            job.events.put({"delta": {"reasoning_content": "careful thought"}})
        call = self.script(request) if self.script else None
        if call is not None:
            name, arguments = call
            job.events.put({"delta": {"content": "Let me do that."}})
            job.events.put(
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": name, "arguments": arguments},
                            }
                        ]
                    }
                }
            )
            job.completion_tokens = 5
            job.events.put({"finish_reason": "tool_calls", "receipt": {"cache": "apcv2"}})
        else:
            job.events.put({"delta": {"content": "done"}})
            job.completion_tokens = 1
            job.events.put({"finish_reason": "stop", "receipt": {"cache": "apcv2"}})
        return job


def _first_turn_call(name, arguments):
    def script(request):
        if any(message.get("role") == "tool" for message in request["messages"]):
            return None
        names = {tool["function"]["name"] for tool in request.get("tools", ())}
        return (name, arguments) if name in names else None

    return script


@pytest.fixture
def serve():
    servers = []

    def start(*, enabled=True, grammar="off", policy=None):
        engine = ScriptedEngine()
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            handler_for(
                engine,
                agent_compat=policy
                or AgentCompat(enabled=enabled, custom_tool_grammar=grammar),
            ),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return engine, f"http://127.0.0.1:{server.server_port}"

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join()


def _post(url, body, *, path="/v1/responses", headers=None):
    request = urllib.request.Request(
        url + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def _events(raw):
    events = []
    for block in raw.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
    return events


def _codex(turn, engine, **overrides):
    body = copy.deepcopy(CODEX[turn])
    body["model"] = MODEL
    signer = engine.reasoning_signer
    for item in body["input"]:
        if item.get("type") == "reasoning":
            summary = "".join(part["text"] for part in item["summary"])
            item["encrypted_content"] = signer.sign_responses(
                model=MODEL, tenant="default", text=summary
            )
    body.update(overrides)
    return body


def _claude(turn, engine, **overrides):
    body = copy.deepcopy(CLAUDE[turn])
    body["model"] = MODEL
    for message in body["messages"]:
        if isinstance(message["content"], list):
            for block in message["content"]:
                if block.get("type") == "thinking":
                    block["signature"] = engine.reasoning_signer.sign_anthropic_carrying(
                        model=MODEL, tenant="default", text="careful thought"
                    )
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# Regression: flag off keeps main's fail-closed behavior.


def test_flag_off_codex_turn_one_fails_closed_like_main(serve):
    engine, url = serve(enabled=False)
    status, raw = _post(url, _codex(0, engine))
    assert status in {400, 501}
    assert "tool" in json.loads(raw)["error"]["message"]
    assert not engine.requests
    assert not any(key.startswith("agent_compat") for key in engine.counts)


def test_flag_off_claude_turn_one_fails_closed_like_main(serve):
    engine, url = serve(enabled=False)
    status, raw = _post(url, _claude(0, engine), path="/v1/messages")
    assert status == 400
    assert "context_management" in json.loads(raw)["error"]["message"]


def test_client_metadata_is_validated_and_ignored_without_the_flag(serve):
    engine, url = serve(enabled=False)
    body = {"model": MODEL, "input": "hi", "client_metadata": {"turn_id": "t"}}
    status, raw = _post(url, body)
    assert status == 200, raw
    status, raw = _post(url, {**body, "client_metadata": {"turn_id": 1}})
    assert status == 400


# ---------------------------------------------------------------------------
# Codex CLI over Responses.


def test_codex_turn_one_streams_a_custom_apply_patch_call(serve):
    engine, url = serve(grammar="validate")
    engine.script = _first_turn_call("apply_patch", json.dumps({"input": PATCH}))
    status, raw = _post(url, _codex(0, engine))
    assert status == 200, raw
    events = _events(raw)
    kinds = [event["type"] for event in events]
    assert kinds[0] == "response.created" and kinds[-1] == "response.completed"
    start = kinds.index("response.custom_tool_call_input.delta")
    assert kinds[start - 1] == "response.output_item.added"
    assert kinds[start + 1] == "response.custom_tool_call_input.done"
    assert kinds[start + 2] == "response.output_item.done"
    assert "response.function_call_arguments.delta" not in kinds
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    output = events[-1]["response"]["output"]
    by_type = {item["type"]: item for item in output}
    assert by_type["custom_tool_call"]["input"] == PATCH
    assert by_type["custom_tool_call"]["id"].startswith("ctc_")
    assert by_type["message"]["phase"] == "commentary"
    assert by_type["reasoning"]["encrypted_content"].startswith("mlx2.reasoning.v1.")
    receipt = events[-1]["response"]["mlx2"]["agent_compat"]
    assert receipt["custom_tools"] == ["apply_patch"]
    assert set(receipt["dropped_tools"]) == {"tool_search", "web_search"}
    assert receipt["grammar"] == "validate"
    assert (receipt["enabled"], receipt["source"]) == (True, "server")
    # The shim reached the chat contract as a one-string function.
    (request,) = engine.requests
    shim = next(t for t in request["tools"] if t["function"]["name"] == "apply_patch")
    assert shim["function"]["parameters"]["required"] == ["input"]
    assert [m["role"] for m in request["messages"]][:2] == ["system", "user"]
    assert sum(m["role"] == "system" for m in request["messages"]) == 1
    for key in (
        "agent_compat_requests",
        "agent_compat_custom_tools_declared",
        "agent_compat_custom_tool_calls",
        "agent_compat_custom_tool_grammar_validated",
        "agent_compat_hosted_tools_dropped",
        "agent_compat_phase_outputs",
        "agent_compat_system_folded",
    ):
        assert engine.counts[key] > 0, key


def test_codex_turn_two_replays_reasoning_custom_call_and_phase(serve):
    engine, url = serve(grammar="validate")
    status, raw = _post(url, _codex(1, engine, stream=False))
    assert status == 200, raw
    payload = json.loads(raw)
    (message,) = [item for item in payload["output"] if item["type"] == "message"]
    assert message["phase"] == "final_answer"
    (request,) = engine.requests
    assistant = [m for m in request["messages"] if m["role"] == "assistant"]
    assert len(assistant) == 1, "reasoning, commentary and call are one turn"
    turn = assistant[0]
    assert turn["reasoning_content"] == "<text>"
    assert turn["content"] == "<text>"
    (call,) = turn["tool_calls"]
    assert json.loads(call["function"]["arguments"]) == {"input": PATCH}
    assert request["messages"][-1]["role"] == "tool"
    assert engine.counts["reasoning_signature_rejections"] == 0
    assert engine.counts["agent_compat_custom_tool_replays"] == 1
    assert engine.counts["agent_compat_phase_inputs"] == 1


def test_codex_turn_prefix_is_stable_for_apcv2_reuse(serve):
    engine, url = serve()
    assert _post(url, _codex(0, engine, stream=False))[0] == 200
    assert _post(url, _codex(1, engine, stream=False))[0] == 200
    assert _post(url, _codex(1, engine, stream=False))[0] == 200
    first, second, again = engine.requests
    assert second["messages"] == again["messages"]
    assert second["tools"] == first["tools"]
    assert second["messages"][: len(first["messages"])] == first["messages"]


def test_custom_tool_grammar_violation_fails_closed(serve):
    engine, url = serve(grammar="validate")
    engine.script = _first_turn_call("apply_patch", json.dumps({"input": "rm -rf /"}))
    status, raw = _post(url, _codex(0, engine, stream=False))
    assert status == 502
    assert "grammar" in json.loads(raw)["error"]["message"]
    assert engine.counts["agent_compat_custom_tool_grammar_rejected"] == 1
    status, raw = _post(url, _codex(0, engine))
    assert status == 200
    assert _events(raw)[-1]["type"] == "response.failed"


def test_grammar_rejection_logs_the_offending_input(serve, caplog):
    engine, url = serve(grammar="validate")
    engine.script = _first_turn_call("apply_patch", json.dumps({"input": "rm -rf /"}))
    with caplog.at_level("WARNING", logger="mlx2.agent_compat"):
        assert _post(url, _codex(0, engine, stream=False))[0] == 502
    messages = [record.getMessage() for record in caplog.records]
    assert any("apply_patch" in m and "rm -rf /" in m for m in messages), messages


def test_custom_tool_grammar_off_describes_but_does_not_validate(serve):
    engine, url = serve(grammar="off")
    engine.script = _first_turn_call("apply_patch", json.dumps({"input": "free text"}))
    status, raw = _post(url, _codex(0, engine, stream=False))
    assert status == 200
    (call,) = [i for i in json.loads(raw)["output"] if i["type"] == "custom_tool_call"]
    assert call["input"] == "free text"
    shim = next(
        t for t in engine.requests[0]["tools"] if t["function"]["name"] == "apply_patch"
    )
    assert "*** Begin Patch" in shim["function"]["description"]
    assert engine.counts["agent_compat_custom_tool_grammar_validated"] == 0


def test_unenforceable_grammar_is_rejected_in_validate_mode(serve):
    engine, url = serve(grammar="validate")
    body = {
        "model": MODEL,
        "input": "hi",
        "tools": [
            {
                "type": "custom",
                "name": "expr",
                "format": {
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": 'start: "(" start ")" | "x"',
                },
            }
        ],
    }
    status, raw = _post(url, body)
    assert status == 400
    assert "recursive" in json.loads(raw)["error"]["message"]


def test_custom_shim_must_carry_one_string_input(serve):
    engine, url = serve()
    engine.script = _first_turn_call("apply_patch", json.dumps({"patch": PATCH}))
    status, _ = _post(url, _codex(0, engine, stream=False))
    assert status == 502


def test_namespace_tools_flatten_and_split(serve):
    engine, url = serve()
    body = {
        "model": MODEL,
        "input": "spawn",
        "tools": [
            {
                "type": "namespace",
                "name": "multi_agent_v1",
                "tools": [
                    {
                        "type": "function",
                        "name": "spawn_agent",
                        "strict": False,
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            }
        ],
    }
    engine.script = _first_turn_call("multi_agent_v1.spawn_agent", "{}")
    status, raw = _post(url, body)
    assert status == 200, raw
    (call,) = [i for i in json.loads(raw)["output"] if i["type"] == "function_call"]
    assert (call["name"], call["namespace"]) == ("spawn_agent", "multi_agent_v1")
    assert engine.counts["agent_compat_namespace_calls"] == 1
    replay = {
        **body,
        "input": [
            {"type": "message", "role": "user", "content": "spawn"},
            {**call, "type": "function_call"},
            {"type": "function_call_output", "call_id": call["call_id"], "output": [
                {"type": "input_text", "text": "spawned"}
            ]},
        ],
    }
    status, raw = _post(url, replay)
    assert status == 200, raw
    replayed = engine.requests[-1]["messages"][1]["tool_calls"][0]["function"]
    assert replayed["name"] == "multi_agent_v1.spawn_agent"
    assert engine.requests[-1]["messages"][-1] == {
        "role": "tool", "tool_call_id": call["call_id"], "content": "spawned"
    }


def test_unknown_hosted_tool_still_fails_closed(serve):
    engine, url = serve()
    body = {"model": MODEL, "input": "x", "tools": [{"type": "computer_use_preview"}]}
    status, _ = _post(url, body)
    assert status in {400, 501}


def test_all_tools_dropped_runs_plain_generation(serve):
    engine, url = serve()
    body = {
        "model": MODEL,
        "input": "x",
        "tools": [{"type": "web_search"}],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
    }
    status, raw = _post(url, body)
    assert status == 200, raw
    assert "tools" not in engine.requests[0]
    assert json.loads(raw)["mlx2"]["agent_compat"]["dropped_tools"] == ["web_search"]


# ---------------------------------------------------------------------------
# Claude Code over Messages.


def test_claude_turn_one_adaptive_omitted_thinking_and_system_fold(serve):
    engine, url = serve()
    engine.script = _first_turn_call("Read", json.dumps({"file_path": "notes.txt"}))
    status, raw = _post(url, _claude(0, engine), path="/v1/messages")
    assert status == 200, raw
    events = _events(raw)
    kinds = [event["type"] for event in events]
    assert kinds[0] == "message_start" and kinds[-1] == "message_stop"
    thinking_start = next(
        e for e in events
        if e["type"] == "content_block_start" and e["content_block"]["type"] == "thinking"
    )
    index = thinking_start["index"]
    deltas = [e["delta"] for e in events if e["type"] == "content_block_delta" and e["index"] == index]
    assert [d["type"] for d in deltas] == ["signature_delta"], "display omitted"
    assert deltas[0]["signature"].startswith("mlx2.thinkingc.v1.")
    assert any(
        e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use"
        for e in events
    )
    (request,) = engine.requests
    assert request["enable_thinking"] is True
    assert request["reasoning_effort"] == "high"
    assert "thinking_budget" not in request
    roles = [m["role"] for m in request["messages"]]
    assert roles == ["system", "user"], roles
    for key in (
        "agent_compat_adaptive_thinking",
        "agent_compat_thinking_omitted",
        "agent_compat_output_effort",
        "agent_compat_system_folded",
    ):
        assert engine.counts[key] > 0, key


def test_claude_turn_two_restores_omitted_thinking_from_signature(serve):
    engine, url = serve()
    status, raw = _post(url, _claude(1, engine, stream=False), path="/v1/messages")
    assert status == 200, raw
    (request,) = engine.requests
    (assistant,) = [m for m in request["messages"] if m["role"] == "assistant"]
    assert assistant["reasoning_content"] == "careful thought"
    assert engine.counts["agent_compat_thinking_restored"] == 1
    assert engine.counts["reasoning_signature_rejections"] == 0
    content = json.loads(raw)["content"]
    assert content[0] == {"type": "thinking", "thinking": "", "signature": content[0]["signature"]}
    assert request["messages"][-1]["role"] == "user"


def test_forged_or_cross_tenant_carrying_signature_is_dropped(serve):
    engine, url = serve()
    body = _claude(1, engine, stream=False)
    other = ReasoningSigner(b"someone-else").sign_anthropic_carrying(
        model=MODEL, tenant="default", text="injected"
    )
    for message in body["messages"]:
        for block in message["content"] if isinstance(message["content"], list) else ():
            if block.get("type") == "thinking":
                block["signature"] = other
    status, raw = _post(url, body, path="/v1/messages")
    assert status == 200, raw
    (assistant,) = [m for m in engine.requests[0]["messages"] if m["role"] == "assistant"]
    assert "reasoning_content" not in assistant
    assert engine.counts["reasoning_signature_rejections"] == 1
    assert engine.counts["agent_compat_thinking_restored"] == 0


def test_context_management_clear_thinking_keeps_last_turns(serve):
    engine, url = serve()
    signer = engine.reasoning_signer

    def turn(text):
        return {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": text,
                    "signature": signer.sign_anthropic(model=MODEL, tenant="default", text=text),
                },
                {"type": "text", "text": "ok"},
            ],
        }

    body = {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "a"}, turn("first"),
            {"role": "user", "content": "b"}, turn("second"),
            {"role": "user", "content": "c"},
        ],
        "context_management": {
            "edits": [{"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 1}}]
        },
    }
    status, raw = _post(url, body, path="/v1/messages")
    assert status == 200, raw
    assistants = [m for m in engine.requests[0]["messages"] if m["role"] == "assistant"]
    assert "reasoning_content" not in assistants[0]
    assert assistants[1]["reasoning_content"] == "second"
    assert engine.counts["agent_compat_thinking_cleared"] == 1
    body["context_management"] = {"edits": [{"type": "clear_tool_uses_20250919"}]}
    status, raw = _post(url, body, path="/v1/messages")
    assert status == 400
    assert "clear_tool_uses_20250919" in json.loads(raw)["error"]["message"]


def test_count_tokens_accepts_claude_code_shapes(serve):
    engine, url = serve()
    body = _claude(0, engine)
    for key in ("stream", "max_tokens"):
        body.pop(key)
    status, raw = _post(url, body, path="/v1/messages/count_tokens")
    assert status == 200, raw


# ---------------------------------------------------------------------------
# Pure translation units.


def test_fold_system_messages_is_deterministic_and_template_safe():
    messages = [
        {"role": "system", "content": "a"},
        {"role": "system", "content": "b"},
        {"role": "user", "content": "u"},
        {"role": "system", "content": "late"},
        {"role": "assistant", "content": "x"},
        {"role": "system", "content": "tail"},
    ]
    counts = Counter()
    folded = fold_system_messages(messages, counts)
    assert folded == [
        {"role": "system", "content": "a\n\nb"},
        {"role": "user", "content": "u\n\nlate"},
        {"role": "assistant", "content": "x"},
        {"role": "user", "content": "tail"},
    ]
    assert fold_system_messages(messages) == folded
    assert counts["agent_compat_system_folded"] == 3


@pytest.mark.parametrize(
    "patch, valid",
    [
        (PATCH, True),
        (PATCH.rstrip("\n"), True),
        (
            "*** Begin Patch\n*** Update File: a.py\n*** Move to: b.py\n@@ def f():\n"
            "-    return 1\n+    return 2\n context\n*** End of File\n*** End Patch",
            True,
        ),
        ("*** Begin Patch\n*** Delete File: gone.txt\n*** End Patch\n", True),
        ("*** Begin Patch\n*** End Patch\n", False),
        ("*** Begin Patch\n*** Add File: x\n*** End Patch\n", False),
        ('{"input": "*** Begin Patch"}', False),
        ("*** Begin Patch\n*** Add File: x\n+a\n", False),
    ],
)
def test_codex_apply_patch_grammar_lowers_to_the_regex_automaton(patch, valid):
    constraint = compile_constraint(grammar=lark_to_regex(APPLY_PATCH_LARK))
    assert (constraint.fullmatch(patch) is not None) is valid
    # The same automaton supports prefix checks, the decode-time seam.
    assert constraint.fullmatch("*** Begin Patch\n*** Add", partial=True) is not None


def test_lark_subset_features_and_rejections():
    grammar = """
    // comment
    start: greeting WS_INLINE? NAME ("," NAME)~0..2 [PUNCT]
    greeting: "hello"i
        | "hi"
    NAME: /[a-z]+/
    PUNCT.2: "!" | "?"
    %import common.WS_INLINE
    """
    pattern = compile_constraint(grammar=lark_to_regex(grammar))
    assert pattern.fullmatch("HELLO bob,al!")
    assert pattern.fullmatch("hibob")
    assert not pattern.fullmatch("hi a,b,c,d")
    for bad, message in [
        ('start: a\na: "x" a | "y"', "recursive"),
        ('start: "x"\n%ignore " "', "directive"),
        ('start: "x"\n%import common.ESCAPED_STRING', "common import"),
        ("start: missing", "undefined"),
        ('start: ("x"', "unbalanced"),
        ("a: \"x\"", "start rule"),
    ]:
        with pytest.raises(LarkGrammarError, match=message):
            lark_to_regex(bad)


def test_every_agent_compat_counter_is_exported_to_prometheus():
    from mlx2.agent_compat import COUNTERS
    from mlx2.prometheus import _ENGINE_EVENTS

    for key, event in COUNTERS.items():
        assert _ENGINE_EVENTS[key] == ("agent_compat", event)


def test_status_reports_agent_compat_settings(serve):
    engine, url = serve(grammar="validate")
    with urllib.request.urlopen(url + "/v1/status", timeout=10) as response:
        status = json.load(response)
    assert status["agent_compat"] == {
        "mode": "on",
        "enabled": True,
        "custom_tool_grammar": "validate",
        "tenant_policies": 0,
        "detection": False,
    }
