import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_sdk_smoke():
    script = Path(__file__).parents[1] / "scripts" / "sdk_smoke.py"
    spec = importlib.util.spec_from_file_location("mlx2_sdk_smoke", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_sdk_client_subprocess_has_an_overall_timeout(monkeypatch):
    smoke = _load_sdk_smoke()
    observed = {}

    def run(command, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(smoke.subprocess, "run", run)
    assert smoke.run_server(sys.executable) == 0
    assert observed["timeout"] == 45


def test_official_sdk_smoke_when_interpreter_is_configured():
    sdk_python = os.environ.get("MLX2_SDK_PYTHON")
    if not sdk_python:
        pytest.skip("set MLX2_SDK_PYTHON to run the official client SDK smoke")
    script = Path(__file__).parents[1] / "scripts" / "sdk_smoke.py"
    result = subprocess.run(
        [sys.executable, str(script), "--sdk-python", sdk_python],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_server_url_mode_reexecutes_sdk_python(monkeypatch):
    smoke = _load_sdk_smoke()
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(smoke.subprocess, "run", run)
    assert smoke.run_url("/sdk/python", "http://127.0.0.1:8297/") == 0
    assert observed["command"][0] == "/sdk/python"
    assert observed["command"][-2:] == ["--client-base-url", "http://127.0.0.1:8297"]
    assert observed["timeout"] == 7200


def test_responses_stream_lifecycle_allows_multiple_text_deltas():
    smoke = _load_sdk_smoke()
    smoke._assert_response_stream_lifecycle([
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ])


def test_responses_stream_lifecycle_rejects_out_of_order_events():
    smoke = _load_sdk_smoke()
    with pytest.raises(AssertionError):
        smoke._assert_response_stream_lifecycle([
            "response.created",
            "response.output_text.delta",
            "response.output_item.added",
            "response.completed",
        ])


def test_structured_chat_uses_a_bounded_thinking_window_and_preserves_raw_failure():
    smoke = _load_sdk_smoke()

    assert smoke.STRUCTURED_THINKING_BUDGET == 128
    # Both Chat Completions and Responses call this shared request-body helper;
    # a missing Responses allowance previously let Qwen3.6 spend all 256
    # output tokens in its private reasoning channel.
    assert smoke._structured_thinking_extra_body() == {"thinking_budget": 128}
    with pytest.raises(AssertionError, match='raw choice:.*"finish_reason": "length"'):
        smoke._parse_structured_content(
            "",
            evidence={
                "finish_reason": "length",
                "message": {"content": "", "reasoning_content": "still thinking"},
            },
        )


def test_signed_thinking_followup_may_answer_without_another_thinking_block():
    smoke = _load_sdk_smoke()
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="The answer is 42.")]
    )

    smoke._assert_anthropic_followup_content(response)


def test_real_server_feature_gates_require_declared_capabilities():
    smoke = _load_sdk_smoke()
    assert smoke._server_features({
        "capabilities": ["text"],
        "structured_output": {"thinking_deferral": True},
    }) == {"thinking_deferral": False, "tools": False}
    assert smoke._server_features({
        "capabilities": ["text", "reasoning", "tools"],
        "structured_output": {"thinking_deferral": True},
    }) == {"thinking_deferral": True, "tools": True}


@pytest.mark.parametrize(
    ("parallel_names", "passes"),
    [(["weather", "clock"], True), (["weather"], False), (["clock", "clock"], False)],
)
def test_parallel_tool_case_requires_both_requested_tools(
    monkeypatch, capsys, parallel_names, passes
):
    # Regression: the case accepted any nonempty subset of {weather, clock},
    # so a server that dropped every call after the first still passed.
    import types

    def call(name):
        return SimpleNamespace(function=SimpleNamespace(name=name, arguments="{}"),
                               id="c", type="function")

    class Completions:
        def create(self, **kwargs):
            names = parallel_names if kwargs.get("parallel_tool_calls") else ["weather"]
            if not kwargs.get("tools"):
                raise RuntimeError("not exercised")
            message = SimpleNamespace(tool_calls=[call(name) for name in names])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    # Minimal stand-ins: only the chat tool case reaches a real assertion.
    openai = types.ModuleType("openai")
    openai.OpenAI = lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()), responses=SimpleNamespace()
    )
    openai.NotFoundError = type("NotFoundError", (Exception,), {})
    anthropic = types.ModuleType("anthropic")
    anthropic.Anthropic = lambda **kwargs: SimpleNamespace(messages=SimpleNamespace())
    anthropic.NotFoundError = type("NotFoundError", (Exception,), {})
    anthropic.BadRequestError = type("BadRequestError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setitem(sys.modules, "anthropic", anthropic)

    smoke = _load_sdk_smoke()
    smoke.run_client("http://127.0.0.1:9", scripted=True)
    output = capsys.readouterr().out
    verdict = "PASS" if passes else "FAIL"
    assert f"{verdict} openai.chat.tools_required_named_parallel" in output
