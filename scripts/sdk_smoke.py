#!/usr/bin/env python3
"""Smoke mlx2's compatibility APIs through the official Python SDKs.

By default the parent process runs a CPU-only scripted mlx2 server.  ``--url``
instead targets an already-running real server while still re-executing the
client under ``--sdk-python`` so the SDKs need not be installed in mlx2's
environment.  Real-server mode retains wire-shape assertions and relaxes only
the scripted engine's canned-text expectations.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import urllib.request
from collections import Counter
from pathlib import Path

MODEL = "fixture"
STRUCTURED_THINKING_BUDGET = 128
WEATHER_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Look up weather.",
            "parameters": WEATHER_SCHEMA,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clock",
            "description": "Look up a clock.",
            "parameters": {
                "type": "object",
                "properties": {"zone": {"type": "string"}},
                "required": ["zone"],
                "additionalProperties": False,
            },
        },
    },
]
RESPONSES_TOOLS = [
    {
        "type": "function",
        "name": "weather",
        "description": "Look up weather.",
        "parameters": WEATHER_SCHEMA,
        "strict": True,
    }
]
ANTHROPIC_TOOLS = [
    {
        "name": "weather",
        "description": "Look up weather.",
        "input_schema": WEATHER_SCHEMA,
    },
    {
        "name": "clock",
        "description": "Look up a clock.",
        "input_schema": {
            "type": "object",
            "properties": {"zone": {"type": "string"}},
            "required": ["zone"],
            "additionalProperties": False,
        },
    },
]


def _case(name, function, failures):
    try:
        function()
    except Exception as error:  # noqa: BLE001 - every SDK/parser failure belongs in the matrix
        failures.append(name)
        print(f"FAIL {name}: {type(error).__name__}: {error}", flush=True)
    else:
        print(f"PASS {name}", flush=True)


def _dump_blocks(blocks):
    return [block.model_dump(mode="json") for block in blocks]


def _assert_anthropic_followup_content(message):
    """A signed-thinking continuation may answer directly without thinking again."""
    assert any(
        (
            getattr(block, "thinking", None)
            if getattr(block, "type", None) == "thinking"
            else getattr(block, "text", None)
            if getattr(block, "type", None) == "text"
            else None
        )
        for block in message.content
    )


def _parse_structured_content(content, *, evidence):
    """Parse a structured reply while retaining the raw choice on failure."""
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError) as error:
        raw = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
        raise AssertionError(f"non-JSON structured reply; raw choice: {raw}") from error


def _structured_thinking_extra_body():
    """Use one bounded private-reasoning allowance for both OpenAI APIs."""
    return {"thinking_budget": STRUCTURED_THINKING_BUDGET}


def _assert_response_stream_lifecycle(event_types):
    assert event_types[0] == "response.created"
    assert event_types[-1] == "response.completed"
    required = [
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
    ]
    cursor = 0
    for event_type in event_types[1:-1]:
        if cursor < len(required) and event_type == required[cursor]:
            cursor += 1
        elif event_type != "response.output_text.delta":
            raise AssertionError(f"unexpected Responses stream event {event_type!r}")
    assert cursor == len(required), event_types


def _server_features(status):
    capabilities = set(status.get("capabilities", ()))
    return {
        "thinking_deferral": "reasoning" in capabilities
        and bool(status.get("structured_output", {}).get("thinking_deferral")),
        "tools": "tools" in capabilities,
    }


def run_client(base_url, *, scripted=True):
    global MODEL
    import anthropic
    import openai

    supports_thinking_deferral = True
    supports_tools = True
    if not scripted:
        with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=30) as response:
            catalog = json.load(response)
        models = catalog.get("data", [])
        if not models or not isinstance(models[0].get("id"), str):
            raise RuntimeError("real server returned no usable /v1/models identity")
        MODEL = models[0]["id"]
        with urllib.request.urlopen(base_url.rstrip("/") + "/v1/status", timeout=30) as response:
            status = json.load(response)
        features = _server_features(status)
        supports_thinking_deferral = features["thinking_deferral"]
        supports_tools = features["tools"]

    failures = []
    skipped = []
    openai_client = openai.OpenAI(
        api_key="sdk-smoke",
        base_url=base_url + "/v1",
        max_retries=0,
        timeout=10 if scripted else 1800,
    )
    anthropic_client = anthropic.Anthropic(
        api_key="sdk-smoke",
        base_url=base_url,
        max_retries=0,
        timeout=10 if scripted else 1800,
    )

    def chat_nonstream():
        response = openai_client.chat.completions.create(
            temperature=0,
            model=MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "__json__" if scripted else
                    "Return a JSON object whose answer field is the string ok."
                ),
            }],
            max_completion_tokens=8 if scripted else 256,
            reasoning_effort="high",
            # A schema constrains the final channel only. Bound the private
            # reasoning channel so a run-on model cannot consume the entire
            # completion before grammar enforcement begins.
            extra_body=_structured_thinking_extra_body(),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        choice = response.choices[0]
        parsed = _parse_structured_content(
            choice.message.content,
            evidence=choice.model_dump(mode="json"),
        )
        assert parsed == {"answer": "ok"} if scripted else isinstance(parsed.get("answer"), str)
        assert response.choices[0].finish_reason == "stop"

    def chat_stream():
        stream = openai_client.chat.completions.create(
            temperature=0,
            model=MODEL,
            messages=[{"role": "user", "content": "stream text"}],
            stream=True,
            stream_options={"include_usage": True},
            logprobs=True,
            top_logprobs=1,
            max_completion_tokens=128,
            extra_body={"enable_thinking": False},
        )
        chunks = list(stream)
        text = "".join(
            choice.delta.content or ""
            for chunk in chunks
            for choice in chunk.choices
        )
        assert text == "hello from mlx2" if scripted else bool(text.strip())
        logprob_count = sum(
            len(choice.logprobs.content or [])
            for chunk in chunks
            for choice in chunk.choices
            if choice.logprobs is not None
        )
        assert logprob_count == 1 if scripted else logprob_count > 0
        assert any(chunk.usage and (chunk.usage.total_tokens == 8 if scripted else chunk.usage.total_tokens > 0) for chunk in chunks)

    def chat_tools():
        parallel = openai_client.chat.completions.create(
            temperature=0,
            model=MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "__parallel_tools__" if scripted else
                    "Call weather for Toronto and clock for UTC. Use both tools."
                ),
            }],
            tools=OPENAI_TOOLS,
            tool_choice="required",
            parallel_tool_calls=True,
            max_completion_tokens=256,
        )
        calls = parallel.choices[0].message.tool_calls
        assert calls and all(
            call.function.name in {"weather", "clock"} for call in calls
        )
        named = openai_client.chat.completions.create(
            temperature=0,
            model=MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "named tool" if scripted else
                    "Call the weather tool for Toronto."
                ),
            }],
            tools=OPENAI_TOOLS,
            tool_choice={"type": "function", "function": {"name": "weather"}},
            max_completion_tokens=256,
        )
        calls = named.choices[0].message.tool_calls
        assert len(calls) == 1 and calls[0].function.name == "weather"

    if supports_thinking_deferral:
        _case("openai.chat.nonstream", chat_nonstream, failures)
    else:
        skipped.append("openai.chat.nonstream")
        print(
            "SKIP openai.chat.nonstream: adapter does not declare thinking deferral",
            flush=True,
        )
    _case("openai.chat.stream_usage_logprobs", chat_stream, failures)
    if supports_tools:
        _case("openai.chat.tools_required_named_parallel", chat_tools, failures)
    else:
        skipped.append("openai.chat.tools_required_named_parallel")
        print("SKIP openai.chat.tools_required_named_parallel: adapter does not declare tools", flush=True)

    response_state = {}

    def responses_nonstream():
        response = openai_client.responses.create(
            temperature=0,
            model=MODEL,
            input=(
                "__json__" if scripted else
                "Return a JSON object whose answer field is the string ok."
            ),
            max_output_tokens=8 if scripted else 256,
            include=[
                "reasoning.encrypted_content",
                "message.output_text.logprobs",
            ],
            reasoning={"effort": "high", "summary": "auto"},
            # Responses maps this top-level extension onto the same Chat
            # request control. Keep its final-channel schema from losing the
            # entire output budget to private reasoning.
            extra_body=_structured_thinking_extra_body(),
            top_logprobs=1,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        parsed = _parse_structured_content(
            response.output_text,
            evidence=response.model_dump(mode="json"),
        )
        assert parsed == {"answer": "ok"} if scripted else isinstance(parsed.get("answer"), str)
        assert [item.type for item in response.output] == ["reasoning", "message"]
        assert response.output[0].encrypted_content
        logprobs = response.output[1].content[0].logprobs
        assert logprobs and (logprobs[0].token == "json" if scripted else logprobs[0].bytes is not None)

    def responses_stream():
        with openai_client.responses.stream(
            model=MODEL,
            input="typed response stream",
            max_output_tokens=128,
            reasoning={"effort": "none"},
            stream_options={"include_usage": True},
        ) as stream:
            event_types = [event.type for event in stream]
            final = stream.get_final_response()
        _assert_response_stream_lifecycle(event_types)
        assert final.output_text == "hello from mlx2" if scripted else bool(final.output_text.strip())

    def responses_tools():
        first = openai_client.responses.create(
            temperature=0,
            model=MODEL,
            input=(
                "call a tool" if scripted else
                "Call the weather tool for Toronto."
            ),
            tools=RESPONSES_TOOLS,
            tool_choice={"type": "function", "name": "weather"},
            max_output_tokens=256,
        )
        call = next(item for item in first.output if item.type == "function_call")
        second = openai_client.responses.create(
            temperature=0,
            model=MODEL,
            input=[
                {
                    "role": "user",
                    "content": "Call the weather tool for Toronto.",
                },
                {
                    "type": "function_call",
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                },
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": '{"temperature":21}',
                },
            ],
            max_output_tokens=256,
            reasoning={"effort": "none"},
        )
        assert second.output_text == "tool result accepted" if scripted else bool(second.output_text.strip())

    def responses_store():
        first = openai_client.responses.create(
            temperature=0,
            model=MODEL,
            input="stored first turn",
            store=True,
            max_output_tokens=64,
            reasoning={"effort": "none"},
        )
        retrieved = openai_client.responses.retrieve(first.id)
        assert retrieved.id == first.id
        items = openai_client.responses.input_items.list(first.id)
        assert len(items.data) == 1 and items.data[0].type == "message"
        continued = openai_client.responses.create(
            temperature=0,
            model=MODEL,
            input="continued turn",
            previous_response_id=first.id,
            store=True,
            max_output_tokens=64,
            reasoning={"effort": "none"},
        )
        assert continued.previous_response_id == first.id
        openai_client.responses.delete(first.id)
        try:
            openai_client.responses.retrieve(first.id)
        except openai.NotFoundError:
            pass
        else:
            raise AssertionError("deleted response remained retrievable")
        response_state["continued"] = continued.id

    if supports_thinking_deferral:
        _case("openai.responses.nonstream_reasoning_schema_logprobs", responses_nonstream, failures)
    else:
        skipped.append("openai.responses.nonstream_reasoning_schema_logprobs")
        print("SKIP openai.responses.nonstream_reasoning_schema_logprobs: adapter does not declare thinking deferral", flush=True)
    _case("openai.responses.typed_stream", responses_stream, failures)
    if supports_tools:
        _case("openai.responses.function_roundtrip", responses_tools, failures)
    else:
        skipped.append("openai.responses.function_roundtrip")
        print("SKIP openai.responses.function_roundtrip: adapter does not declare tools", flush=True)
    _case("openai.responses.store_retrieve_input_items_delete", responses_store, failures)

    def anthropic_nonstream_tools():
        for choice in (
            {"type": "auto"},
            {"type": "any"},
            {"type": "tool", "name": "weather"},
        ):
            message = anthropic_client.messages.create(
                temperature=0,
                model=MODEL,
                max_tokens=64,
                system=[
                    {
                        "type": "text",
                        "text": "Be concise.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": "use a tool" if scripted else "You must use the weather tool for Toronto."}],
                tools=ANTHROPIC_TOOLS,
                tool_choice=choice,
            )
            uses = [block for block in message.content if block.type == "tool_use"]
            if choice["type"] == "auto":
                assert message.content
            else:
                assert uses and uses[0].name == "weather"

    def anthropic_tool_roundtrip():
        first = anthropic_client.messages.create(
            temperature=0,
            model=MODEL,
            max_tokens=64,
            messages=[{
                "role": "user",
                "content": (
                    "use a tool" if scripted else
                    "Call the weather tool for Toronto."
                ),
            }],
            tools=ANTHROPIC_TOOLS,
            tool_choice={"type": "tool", "name": "weather"},
        )
        call = next(block for block in first.content if block.type == "tool_use")
        second = anthropic_client.messages.create(
            temperature=0,
            model=MODEL,
            max_tokens=64,
            messages=[
                {"role": "user", "content": "Call the weather tool for Toronto."},
                {"role": "assistant", "content": _dump_blocks(first.content)},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "content": "21 C",
                        }
                    ],
                },
            ],
        )
        assert second.content[-1].text == "tool result accepted" if scripted else bool(second.content[-1].text.strip())

    def anthropic_thinking_roundtrip():
        first = anthropic_client.messages.create(
            temperature=0,
            model=MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": "think"}],
            thinking={"type": "enabled", "budget_tokens": 1024},
        )
        thinking = next(block for block in first.content if block.type == "thinking")
        assert thinking.signature
        second = anthropic_client.messages.create(
            temperature=0,
            model=MODEL,
            max_tokens=2048,
            messages=[
                {"role": "user", "content": "think"},
                {"role": "assistant", "content": _dump_blocks(first.content)},
                {"role": "user", "content": "continue"},
            ],
            thinking={"type": "enabled", "budget_tokens": 1024},
        )
        _assert_anthropic_followup_content(second)

    def anthropic_stops():
        stopped = anthropic_client.messages.create(
            temperature=0,
            model=MODEL,
            max_tokens=64,
            stop_sequences=["END"],
            messages=[{"role": "user", "content": "__stop_sequence__" if scripted else "Repeat this line exactly and output nothing else: A END B"}],
            thinking={"type": "disabled"},
        )
        assert stopped.stop_reason == "stop_sequence" and stopped.stop_sequence == "END"
        limited = anthropic_client.messages.create(
            temperature=0,
            model=MODEL,
            max_tokens=1,
            messages=[{"role": "user", "content": "__length__"}],
            thinking={"type": "disabled"},
        )
        assert limited.stop_reason == "max_tokens"

    def anthropic_stream():
        with anthropic_client.messages.stream(
            model=MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "stream"}],
            thinking={"type": "disabled"},
        ) as stream:
            text = "".join(stream.text_stream)
            final = stream.get_final_message()
        assert text == "hello from mlx2" if scripted else bool(text.strip())
        assert final.content[-1].text == text

    def anthropic_count():
        count = anthropic_client.messages.count_tokens(
            model=MODEL,
            system=[
                {
                    "type": "text",
                    "text": "cached",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": "hello"}],
        )
        assert count.input_tokens == 7 if scripted else count.input_tokens > 0

    def anthropic_errors():
        try:
            anthropic_client.messages.create(
                temperature=0,
                model="not-loaded",
                max_tokens=8,
                messages=[{"role": "user", "content": "hello"}],
            )
        except anthropic.NotFoundError:
            pass
        else:
            raise AssertionError("unknown model did not raise NotFoundError")
        try:
            anthropic_client.messages.create(
                temperature=0,
                model=MODEL,
                max_tokens=8,
                messages=[],
            )
        except anthropic.BadRequestError:
            pass
        else:
            raise AssertionError("invalid request did not raise BadRequestError")

    if supports_tools:
        _case("anthropic.messages.system_tool_choices", anthropic_nonstream_tools, failures)
        _case("anthropic.messages.tool_result_roundtrip", anthropic_tool_roundtrip, failures)
    else:
        for name in (
            "anthropic.messages.system_tool_choices",
            "anthropic.messages.tool_result_roundtrip",
        ):
            skipped.append(name)
            print(f"SKIP {name}: adapter does not declare tools", flush=True)
    if supports_thinking_deferral:
        _case("anthropic.messages.thinking_signature_roundtrip", anthropic_thinking_roundtrip, failures)
    else:
        skipped.append("anthropic.messages.thinking_signature_roundtrip")
        print("SKIP anthropic.messages.thinking_signature_roundtrip: adapter does not declare thinking deferral", flush=True)
    _case("anthropic.messages.stop_and_length", anthropic_stops, failures)
    _case("anthropic.messages.stream_helper", anthropic_stream, failures)
    _case("anthropic.messages.count_tokens", anthropic_count, failures)
    _case("anthropic.messages.typed_errors", anthropic_errors, failures)

    print(
        f"SUMMARY {14 - len(failures) - len(skipped)}/14 passed"
        + (f"; skipped: {', '.join(skipped)}" if skipped else "")
        + (f"; failed: {', '.join(failures)}" if failures else ""),
        flush=True,
    )
    return int(bool(failures))


def _request_text(request):
    pieces = []
    for message in request.get("messages", []):
        content = message.get("content", "")
        if isinstance(content, str):
            pieces.append(content)
    return "\n".join(pieces)


def _tool_arguments(name):
    return (
        '{"city":"Toronto"}' if name == "weather" else '{"zone":"UTC"}'
    )


def run_server(sdk_python):
    import mlx.core as mx

    # This must precede every mlx2 import: the smoke owns no GPU resources.
    mx.set_default_device(mx.cpu)

    from http.server import ThreadingHTTPServer

    from mlx2.reasoning_signatures import ReasoningSigner
    from mlx2.server import handler_for
    from mlx2.serving import Job

    class ScriptedEngine:
        model_path = MODEL
        max_context = 16384

        def __init__(self):
            self.counts = Counter()
            self.lock = threading.Lock()
            self.reasoning_signer = ReasoningSigner(b"official-sdk-smoke")

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
            job = Job(request)
            job.tenant_id = tenant_id
            job.prompt_tokens = 6
            job.cached_tokens = 1
            text = _request_text(request)
            tool_history = any(
                message.get("role") == "tool" for message in request["messages"]
            )
            if request.get("enable_thinking"):
                job.events.put({"delta": {"reasoning_content": "careful thought"}})
            if request.get("tools") and request.get("tool_choice") != "none" and not tool_history:
                definitions = request["tools"]
                selected = (
                    definitions
                    if "__parallel_tools__" in text
                    else definitions[:1]
                )
                calls = [
                    {
                        "index": index,
                        "id": f"call_{index}_{tool['function']['name']}",
                        "type": "function",
                        "function": {
                            "name": tool["function"]["name"],
                            "arguments": _tool_arguments(tool["function"]["name"]),
                        },
                    }
                    for index, tool in enumerate(selected)
                ]
                job.events.put({"delta": {"tool_calls": calls}})
                finish = "tool_calls"
                completion_tokens = len(calls)
                receipt = {"cache": "apcv2"}
            else:
                output = (
                    "tool result accepted"
                    if tool_history
                    else '{"answer":"ok"}'
                    if "__json__" in text
                    else "hello from mlx2"
                )
                if request.get("logprobs"):
                    token = "json" if "__json__" in text else "hello"
                    encoded = list(token.encode())
                    job.events.put(
                        {
                            "logprob": {
                                "id": 7,
                                "token": token,
                                "logprob": -0.1,
                                "bytes": encoded,
                                "top_logprobs": [
                                    {
                                        "id": 7,
                                        "token": token,
                                        "logprob": -0.1,
                                        "bytes": encoded,
                                    }
                                ],
                            }
                        }
                    )
                job.events.put({"delta": {"content": output}})
                finish = "length" if request.get("max_tokens") == 1 else "stop"
                completion_tokens = 1 if finish == "length" else 2
                receipt = {"cache": "apcv2"}
                if "__stop_sequence__" in text and request.get("stop"):
                    receipt["stop_sequence"] = request["stop"][0]
            job.completion_tokens = completion_tokens
            job.events.put({"finish_reason": finish, "receipt": receipt})
            return job

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(ScriptedEngine()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        command = [
            str(sdk_python),
            str(Path(__file__).resolve()),
            "--client-base-url",
            f"http://127.0.0.1:{server.server_port}",
            "--scripted-client",
        ]
        try:
            return subprocess.run(command, check=False, timeout=45).returncode
        except subprocess.TimeoutExpired:
            print("FAIL official SDK client timed out after 45 seconds", file=sys.stderr)
            return 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run_url(sdk_python, url):
    command = [
        str(sdk_python), str(Path(__file__).resolve()),
        "--client-base-url", url.rstrip("/"),
    ]
    try:
        return subprocess.run(command, check=False, timeout=7200).returncode
    except subprocess.TimeoutExpired:
        print("FAIL official SDK real-server client timed out after 7200 seconds", file=sys.stderr)
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sdk-python",
        help="Python interpreter containing the official openai and anthropic SDKs",
    )
    parser.add_argument("--client-base-url", help=argparse.SUPPRESS)
    parser.add_argument("--scripted-client", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--url", help="target an already-running real mlx2 server")
    args = parser.parse_args()
    if args.client_base_url:
        return run_client(args.client_base_url, scripted=args.scripted_client)
    if not args.sdk_python:
        print("SKIP official SDK smoke: pass --sdk-python to select the SDK environment")
        return 0
    sdk_python = Path(args.sdk_python)
    if not sdk_python.is_file() or not os.access(sdk_python, os.X_OK):
        print(f"FAIL sdk interpreter is not executable: {sdk_python}", file=sys.stderr)
        return 2
    return run_url(sdk_python, args.url) if args.url else run_server(sdk_python)


if __name__ == "__main__":
    raise SystemExit(main())
