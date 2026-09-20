"""Per-request agent-compat selection: header > tenant policy > detection.

Detection fixtures are the identity headers actually observed on the wire from
codex-cli 0.145.0, Claude Code 2.1.269 and the official OpenAI/Anthropic
Python SDKs (``tests/fixtures/agent_clients/headers.json``).
"""

import copy
import json
from collections import Counter

import pytest

from mlx2.agent_compat import (
    AgentCompatError,
    AgentCompatPolicy,
    detect_client,
    load_tenant_policy,
)
from mlx2.server import build_parser

from test_agent_conformance import (  # noqa: F401 - shared fixtures/helpers
    MODEL,
    PATCH,
    _claude,
    _codex,
    _events,
    _first_turn_call,
    _post,
    serve,
)
from pathlib import Path

HEADERS = json.loads(
    (Path(__file__).parent / "fixtures" / "agent_clients" / "headers.json").read_text()
)["requests"]


def _observed(client):
    return next(entry for entry in HEADERS if entry["expected_client"] == client)


def _wire(entry):
    """Headers a client would send, minus hop-by-hop fields urllib manages."""
    return {
        key: value
        for key, value in entry["headers"].items()
        if key not in {"accept-encoding", "content-type"}
    }


CODEX_HEADERS = _wire(_observed("codex"))
CLAUDE_HEADERS = _wire(_observed("claude_code"))
SDK_HEADERS = {
    entry["path"]: _wire(entry) for entry in HEADERS if entry["expected_client"] is None
}


# ---------------------------------------------------------------------------
# Detection


@pytest.mark.parametrize("entry", HEADERS, ids=[entry["name"] for entry in HEADERS])
def test_detection_on_real_captured_headers(entry):
    assert detect_client(entry["headers"], entry["path"]) == entry["expected_client"]


@pytest.mark.parametrize(
    "headers, path, expected",
    [
        # Version- and entry-point-tolerant.
        ({"originator": "codex_cli_rs", "user-agent": "codex_cli_rs/0.201.3 (Linux)"}, "/v1/responses", "codex"),
        ({"Originator": "codex_vscode", "User-Agent": "codex_vscode/1.0.0-beta.2"}, "/v1/responses", "codex"),
        ({"user-agent": "claude-cli/3.0.0 (external, cli)", "x-app": "cli"}, "/v1/messages", "claude_code"),
        ({"user-agent": "claude-cli/2.2.0", "anthropic-beta": "claude-code-20250219"}, "/v1/messages/count_tokens", "claude_code"),
        # Conservative: both signals must agree.
        ({"originator": "codex_exec"}, "/v1/responses", None),
        ({"user-agent": "codex_exec/0.145.0"}, "/v1/responses", None),
        ({"originator": "codex_exec", "user-agent": "codex_cli_rs/0.145.0"}, "/v1/responses", None),
        ({"originator": "codex exec", "user-agent": "codex exec/0.1"}, "/v1/responses", None),
        ({"user-agent": "claude-cli/2.1.269"}, "/v1/messages", None),
        ({"user-agent": "my-claude-cli/2.1.269", "x-app": "cli"}, "/v1/messages", None),
        # Each client only on its own API family.
        (_observed("codex")["headers"], "/v1/messages", None),
        (_observed("codex")["headers"], "/v1/chat/completions", None),
        (_observed("claude_code")["headers"], "/v1/responses", None),
        ({}, "/v1/responses", None),
    ],
)
def test_detection_rules_are_conservative_and_version_tolerant(headers, path, expected):
    assert detect_client(headers, path) == expected


# ---------------------------------------------------------------------------
# Precedence matrix


def _expected(mode, header, tenant, detected):
    if mode == "off":
        return False, "server"
    if header is not None:
        return header == "on", "header"
    if mode == "on":
        return True, "server"
    if tenant is not None:
        return tenant == "on", "tenant"
    if detected and mode == "auto":
        return True, "detected"
    return False, "server"


@pytest.mark.parametrize("mode", ["off", "on", "opt-in", "auto"])
@pytest.mark.parametrize("header", [None, "on", "off"])
@pytest.mark.parametrize("tenant", [None, "on", "off"])
@pytest.mark.parametrize("detected", [False, True])
def test_precedence_matrix(mode, header, tenant, detected):
    tenants = {"acme": {"agent_compat": tenant, "custom_tool_grammar": None}} if tenant else {}
    policy = AgentCompatPolicy(mode, "off", tenants)
    headers = dict(CODEX_HEADERS) if detected else {}
    if header is not None:
        headers["X-MLX2-Agent-Compat"] = header
    counts = Counter()
    resolved = policy.resolve(headers, "acme", "/v1/responses", counts)
    assert (resolved.enabled, resolved.source) == _expected(mode, header, tenant, detected)
    assert resolved.client == ("codex" if detected and mode == "auto" else None)
    if resolved.notable:
        assert counts[f"agent_compat_source_{resolved.source}"] == 1
    assert counts["agent_compat_detected_codex"] == int(detected and mode == "auto")


def test_header_off_beats_tenant_and_detection():
    policy = AgentCompatPolicy("auto", tenants={"t": {"agent_compat": "on", "custom_tool_grammar": None}})
    resolved = policy.resolve(
        {**CODEX_HEADERS, "X-MLX2-Agent-Compat": "OFF"}, "t", "/v1/responses"
    )
    assert (resolved.enabled, resolved.source, resolved.client) == (False, "header", "codex")


def test_grammar_precedence_header_tenant_server():
    tenants = {"t": {"agent_compat": "on", "custom_tool_grammar": "validate"}}
    policy = AgentCompatPolicy("auto", "off", tenants)
    assert policy.resolve({}, "t", "/v1/responses").custom_tool_grammar == "validate"
    assert policy.resolve({}, "other", "/v1/responses").custom_tool_grammar == "off"
    assert (
        policy.resolve({"X-MLX2-Custom-Tool-Grammar": "off"}, "t", "/v1/responses")
        .custom_tool_grammar
        == "off"
    )
    with pytest.raises(AgentCompatError):
        policy.resolve({"X-MLX2-Custom-Tool-Grammar": "strict"}, "t", "/v1/responses")
    with pytest.raises(AgentCompatError):
        policy.resolve({"X-MLX2-Agent-Compat": "yes"}, "t", "/v1/responses")


def test_tenant_policy_file_validation(tmp_path):
    good = tmp_path / "tenants.json"
    good.write_text(json.dumps({
        "acme": {"agent_compat": "on", "custom_tool_grammar": "validate"},
        "legacy": {"agent_compat": "off"},
    }))
    policy = load_tenant_policy(good)
    assert policy["legacy"] == {"agent_compat": "off", "custom_tool_grammar": None}
    for bad in (
        [],
        {"acme": {"agent_compat": True}},
        {"acme": {"custom_tool_grammar": "validate"}},
        {"acme": {"agent_compat": "on", "keys": []}},
        {"acme": {"agent_compat": "on", "custom_tool_grammar": "strict"}},
    ):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_tenant_policy(path)


def test_cli_mode_defaults_to_auto_and_bare_flag_means_on():
    base = ["--model", "m"]
    assert build_parser().parse_args(base).agent_compat == "auto"
    assert build_parser().parse_args([*base, "--agent-compat"]).agent_compat == "on"
    assert build_parser().parse_args([*base, "--agent-compat", "opt-in"]).agent_compat == "opt-in"
    with pytest.raises(SystemExit):
        build_parser().parse_args([*base, "--agent-compat", "maybe"])


# ---------------------------------------------------------------------------
# End to end over HTTP with the default (auto) policy


def test_auto_detects_real_codex_headers_end_to_end(serve):
    engine, url = serve(policy=AgentCompatPolicy("auto", "validate"))
    engine.script = _first_turn_call("apply_patch", json.dumps({"input": PATCH}))
    status, raw = _post(url, _codex(0, engine), headers=CODEX_HEADERS)
    assert status == 200, raw
    response = _events(raw)[-1]["response"]
    receipt = response["mlx2"]["agent_compat"]
    assert (receipt["enabled"], receipt["source"], receipt["client"], receipt["grammar"]) == (
        True, "detected", "codex", "validate"
    )
    assert any(item["type"] == "custom_tool_call" for item in response["output"])
    assert engine.counts["agent_compat_source_detected"] == 1
    assert engine.counts["agent_compat_detected_codex"] == 1
    assert engine.counts["agent_compat_custom_tool_grammar_validated"] == 1
    # Same body without identity headers: exactly main's fail-closed answer.
    status, _ = _post(url, _codex(0, engine))
    assert status in {400, 501}


def test_auto_detects_real_claude_code_headers_end_to_end(serve):
    engine, url = serve(policy=AgentCompatPolicy("auto"))
    status, raw = _post(url, _claude(0, engine), path="/v1/messages", headers=CLAUDE_HEADERS)
    assert status == 200, raw
    delta = next(e for e in _events(raw) if e["type"] == "message_delta")
    assert delta["mlx2"]["agent_compat"]["client"] == "claude_code"
    assert engine.counts["agent_compat_detected_claude_code"] == 1
    status, raw = _post(url, _claude(0, engine), path="/v1/messages")
    assert status == 400 and "context_management" in raw


def test_header_off_disables_detected_client_over_http(serve):
    engine, url = serve(policy=AgentCompatPolicy("auto"))
    status, raw = _post(
        url, _codex(0, engine), headers={**CODEX_HEADERS, "X-MLX2-Agent-Compat": "off"}
    )
    assert status in {400, 501}
    assert not engine.requests


def test_tenant_policy_enables_opt_in_over_http(serve):
    policy = AgentCompatPolicy(
        "opt-in", tenants={"acme": {"agent_compat": "on", "custom_tool_grammar": None}}
    )
    engine, url = serve(policy=policy)
    status, raw = _post(url, _codex(0, engine, stream=False), headers={"X-Tenant-ID": "acme"})
    assert status == 200, raw
    assert json.loads(raw)["mlx2"]["agent_compat"]["source"] == "tenant"
    # opt-in never detects: the real Codex identity alone is not enough.
    status, _ = _post(url, _codex(0, engine, stream=False), headers=CODEX_HEADERS)
    assert status in {400, 501}


def test_invalid_selection_header_is_a_400(serve):
    engine, url = serve(policy=AgentCompatPolicy("auto"))
    status, raw = _post(url, {"model": MODEL, "input": "hi"}, headers={"X-MLX2-Agent-Compat": "maybe"})
    assert status == 400 and "X-MLX2-Agent-Compat" in raw


# ---------------------------------------------------------------------------
# Plain SDK traffic is exactly main


def _normalized(raw):
    payload = json.loads(raw)
    for key in ("id", "created", "created_at"):
        payload.pop(key, None)
    for item in payload.get("output", ()):
        item.pop("id", None)
    return payload


@pytest.mark.parametrize(
    "path, body",
    [
        ("/v1/responses", {"model": MODEL, "input": [
            {"type": "message", "role": "developer", "content": "be brief"},
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "message", "role": "system", "content": "late"},
        ]}),
        ("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/messages", {"model": MODEL, "max_tokens": 8, "system": "s",
                          "messages": [{"role": "user", "content": "hi"}]}),
    ],
)
def test_sdk_requests_are_identical_to_translation_off(serve, path, body):
    auto_engine, auto_url = serve(policy=AgentCompatPolicy("auto", "validate"))
    off_engine, off_url = serve(policy=AgentCompatPolicy("off"))
    headers = SDK_HEADERS[path]
    auto = _post(auto_url, body, path=path, headers=headers)
    off = _post(off_url, body, path=path, headers=headers)
    assert auto[0] == off[0] == 200, (auto, off)
    assert _normalized(auto[1]) == _normalized(off[1])
    assert "agent_compat" not in (json.loads(auto[1]).get("mlx2") or {})
    assert auto_engine.requests == off_engine.requests
    assert not any(key.startswith("agent_compat") for key in auto_engine.counts)


# ---------------------------------------------------------------------------
# Spoofing only enables translation


def test_spoofed_identity_cannot_bypass_signatures_or_grammar(serve):
    engine, url = serve(policy=AgentCompatPolicy("auto", "validate"))
    body = _codex(1, engine, stream=False)
    for item in body["input"]:
        if item.get("type") == "reasoning":
            item["encrypted_content"] = "mlx2.reasoning.v1.forged.Zm9v.YmFy"
    status, raw = _post(url, body, headers=CODEX_HEADERS)
    assert status == 200, raw
    (assistant,) = [m for m in engine.requests[0]["messages"] if m["role"] == "assistant"]
    assert "reasoning_content" not in assistant
    assert engine.counts["reasoning_signature_rejections"] == 1
    engine.script = _first_turn_call("apply_patch", json.dumps({"input": "not a patch"}))
    status, raw = _post(url, _codex(0, engine, stream=False), headers=CODEX_HEADERS)
    assert status == 502 and "grammar" in raw
    assert engine.counts["agent_compat_custom_tool_grammar_rejected"] == 1


def test_spoofed_claude_identity_cannot_forge_omitted_thinking(serve):
    engine, url = serve(policy=AgentCompatPolicy("auto"))
    body = _claude(1, engine, stream=False)
    for message in body["messages"]:
        for block in message["content"] if isinstance(message["content"], list) else ():
            if block.get("type") == "thinking":
                block["signature"] = "mlx2.thinkingc.v1.x.aW5qZWN0ZWQ.Zm9yZ2Vk"
    status, raw = _post(url, body, path="/v1/messages", headers=CLAUDE_HEADERS)
    assert status == 200, raw
    (assistant,) = [m for m in engine.requests[0]["messages"] if m["role"] == "assistant"]
    assert "reasoning_content" not in assistant
    assert engine.counts["reasoning_signature_rejections"] == 1


# ---------------------------------------------------------------------------
# Conversation consistency across modes


def test_previous_response_chain_must_keep_its_mode(serve):
    engine, url = serve(policy=AgentCompatPolicy("auto"))
    on = {"X-MLX2-Agent-Compat": "on"}
    status, raw = _post(url, {"model": MODEL, "input": "hi", "store": True}, headers=on)
    assert status == 200, raw
    first = json.loads(raw)
    assert first["mlx2"]["agent_compat"]["enabled"] is True
    follow = {"model": MODEL, "input": "more", "previous_response_id": first["id"]}
    status, raw = _post(url, follow, headers=on)
    assert status == 200, raw
    status, raw = _post(url, follow)
    assert status == 400
    message = json.loads(raw)["error"]["message"]
    assert "previous_response_id" in message and "agent-compat on" in message
    # And the reverse: an off-mode response cannot continue under on.
    status, raw = _post(url, {"model": MODEL, "input": "hi", "store": True})
    plain = json.loads(raw)
    assert "agent_compat" not in plain["mlx2"]
    status, raw = _post(
        url, {**follow, "previous_response_id": plain["id"]}, headers=on
    )
    assert status == 400
    assert engine.counts["agent_compat_mode_conflicts"] == 2


@pytest.mark.parametrize(
    "item",
    [
        {"type": "custom_tool_call", "call_id": "c", "name": "apply_patch", "input": PATCH},
        {"type": "custom_tool_call_output", "call_id": "c", "output": "ok"},
        {"type": "function_call", "call_id": "c", "name": "spawn", "arguments": "{}",
         "namespace": "multi_agent_v1"},
        {"type": "message", "role": "assistant", "phase": "commentary",
         "content": [{"type": "output_text", "text": "x"}]},
    ],
)
def test_agent_history_replayed_without_compat_is_a_clear_400(serve, item):
    engine, url = serve(policy=AgentCompatPolicy("auto"))
    body = {"model": MODEL, "input": [{"type": "message", "role": "user", "content": "u"}, item]}
    status, raw = _post(url, body)
    assert status == 400
    message = json.loads(raw)["error"]["message"]
    assert "agent-compat on" in message and "source: server" in message


def test_omitted_thinking_replayed_without_compat_is_a_clear_400(serve):
    engine, url = serve(policy=AgentCompatPolicy("auto"))
    body = _claude(1, engine, stream=False)
    for key in ("context_management", "output_config", "thinking"):
        body.pop(key)
    body["messages"] = [m for m in body["messages"] if m["role"] != "system"]
    status, raw = _post(
        url, body, path="/v1/messages",
        headers={**CLAUDE_HEADERS, "X-MLX2-Agent-Compat": "off"},
    )
    assert status == 400
    message = json.loads(raw)["error"]["message"]
    assert "omitted-display thinking" in message and "source: header" in message
    assert not engine.requests
