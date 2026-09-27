"""Chat Completions usage reports ``completion_tokens_details.reasoning_tokens``.

A scripted engine drives the real OutputParser the way serving.py's decode loop
does and counts reasoning-channel tokens into ``job.reasoning_tokens``; these
tests check the HTTP layer reports that count on every usage shape.
"""

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import pytest

from mlx2.output import OutputParser
from mlx2.runtime.tool_parsers.qwen3_coder import parse_tool_call
from mlx2.server import handler_for
from mlx2.serving import Job
from test_serving_contract import FakeEngine


class ScriptedEngine(FakeEngine):
    script = []
    finish = "length"

    def submit(self, request, *, tenant_id="default"):
        job = self.job = Job(request)
        job.tenant_id = tenant_id
        parser = OutputParser(
            chat=True, thinking=True, tools=request.get("tools"), parse_tool=parse_tool_call
        )
        job.prompt_tokens, job.completion_tokens, job.reasoning_tokens = 5, 0, 0
        for index, text in enumerate(self.script):
            job.completion_tokens += 1
            if parser.channel == "reasoning_content":
                job.reasoning_tokens += 1
            for delta in parser.push(text, final=index == len(self.script) - 1):
                job.events.put({"delta": delta})
        job.events.put({"finish_reason": self.finish, "receipt": {"cache": "apcv2"}})
        return job


@pytest.fixture
def served():
    engine = ScriptedEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield engine, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def call(base, **body):
    body.setdefault("model", "fixture")
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    response = urlopen(
        Request(
            base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
    )
    raw = response.read().decode()
    if body.get("stream"):
        return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
    return json.loads(raw)


# Thinking opened by the prompt; ``max_tokens`` lands inside the think block.
TRUNCATED = ["Let", " me", " think", " </", "th"]
CLOSED = ["a", "b", "</think>", "x"]


@pytest.mark.parametrize(
    "script,finish,expected", [(CLOSED, "stop", 3), (TRUNCATED, "length", 5)]
)
def test_nonstream_chat_usage_reports_reasoning_tokens(served, script, finish, expected):
    engine, base = served
    engine.script, engine.finish = script, finish
    result = call(base, max_tokens=len(script))
    assert result["choices"][0]["finish_reason"] == finish
    assert engine.job.reasoning_tokens == expected
    assert result["usage"]["completion_tokens_details"] == {"reasoning_tokens": expected}
    assert result["usage"]["completion_tokens"] == len(script)


@pytest.mark.parametrize(
    "script,finish,expected", [(CLOSED, "stop", 3), (TRUNCATED, "length", 5)]
)
def test_stream_chat_usage_reports_reasoning_tokens(served, script, finish, expected):
    engine, base = served
    engine.script, engine.finish = script, finish
    chunks = call(
        base, max_tokens=len(script), stream=True, stream_options={"include_usage": True}
    )
    usages = [chunk["usage"] for chunk in chunks if chunk.get("usage")]
    # Both the finish chunk and the choice-less include_usage chunk carry it.
    assert len(usages) == 2
    for usage in usages:
        assert usage["completion_tokens_details"] == {"reasoning_tokens": expected}
    assert chunks[-1]["choices"] == []


def test_parallel_sampling_usage_sums_reasoning_tokens(served):
    engine, base = served
    engine.script, engine.finish = CLOSED, "stop"
    jobs = []

    def submit_many(samples, **kwargs):
        for sample in samples:
            jobs.append(engine.submit(sample, tenant_id=kwargs.get("tenant_id", "default")))
        return list(jobs)

    engine.submit_many = submit_many
    engine.admit_parallel_samples = lambda count: {"samples": count}
    result = call(base, n=2, max_tokens=4)
    assert len(result["choices"]) == 2
    assert result["usage"]["completion_tokens_details"] == {"reasoning_tokens": 6}
