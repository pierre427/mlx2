"""A buffered Chat tool stream carries the logprobs it was asked for.

Strict, required, named and single-call Chat streams are held until the
terminal tool contract passes.  The held stream sent one delta chunk and a
finish chunk, neither with ``logprobs``, so a stream that asked for them got
none while the same non-streaming request returned them.
"""

import json
import threading
from http.server import ThreadingHTTPServer

import pytest
from test_serving_contract import TOOLS, FakeEngine, post

from mlx2.server import handler_for
from mlx2.serving import Job

LOOSE_TOOLS = [
    {"type": "function", "function": {**TOOLS[0]["function"], "strict": False}}
]
CALL = {
    "index": 0,
    "id": "call_weather",
    "type": "function",
    "function": {"name": "weather", "arguments": '{"city":"T"}'},
}


def _logprob(token):
    return {"logprob": {
        "id": token, "token": str(token), "logprob": -0.5,
        "top_logprobs": [{"id": token, "token": str(token), "logprob": -0.5}],
    }}


class ToolCallEngine(FakeEngine):
    def submit(self, request, *, tenant_id="default"):
        self.job = Job(request)
        self.job.tenant_id = tenant_id
        self.job.prompt_tokens, self.job.completion_tokens = 5, 2
        # The call token's logprob precedes its delta; the stop token's
        # logprob trails it.
        script = [_logprob(5), {"delta": {"tool_calls": [CALL]}}, _logprob(6)]
        for event in script:
            if "logprob" not in event or request.get("logprobs"):
                self.job.events.put(event)
        self.job.events.put({"finish_reason": "tool_calls", "receipt": {"cache": "apcv2"}})
        return self.job


@pytest.fixture
def tool_engine():
    engine = ToolCallEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield engine, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def _chunks(response):
    return [
        json.loads(line[len("data: "):])
        for line in response.read().decode().splitlines()
        if line.startswith("data: {")
    ]


@pytest.mark.parametrize(
    "controls",
    [
        # Unbuffered control: already streams its logprobs.
        {"tools": LOOSE_TOOLS, "tool_choice": "auto"},
        # Buffered through the terminal contract.
        {"tools": TOOLS},
        {"tools": LOOSE_TOOLS, "tool_choice": "required"},
        {"tools": LOOSE_TOOLS,
         "tool_choice": {"type": "function", "function": {"name": "weather"}}},
        {"tools": LOOSE_TOOLS, "parallel_tool_calls": False},
    ],
    ids=["auto", "strict", "required", "named", "single_call"],
)
def test_buffered_tool_stream_carries_requested_logprobs(tool_engine, controls):
    _, base = tool_engine
    with post(base, logprobs=True, top_logprobs=1, **controls) as response:
        reported = [
            value["token"]
            for value in json.load(response)["choices"][0]["logprobs"]["content"]
        ]
    with post(base, stream=True, logprobs=True, top_logprobs=1, **controls) as response:
        chunks = _chunks(response)
    streamed = [
        value["token"]
        for chunk in chunks
        for choice in chunk.get("choices", ())
        for value in (choice.get("logprobs") or {}).get("content", ())
    ]
    calls = [
        call
        for chunk in chunks
        for choice in chunk.get("choices", ())
        for call in choice.get("delta", {}).get("tool_calls", ())
    ]
    assert [call["function"]["name"] for call in calls] == ["weather"]
    assert reported == ["5", "6"]
    assert streamed == reported


def test_buffered_tool_stream_without_logprobs_sends_none(tool_engine):
    _, base = tool_engine
    with post(base, stream=True, tools=TOOLS) as response:
        chunks = _chunks(response)
    assert all(
        "logprobs" not in choice
        for chunk in chunks
        for choice in chunk.get("choices", ())
    )
