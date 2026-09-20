"""CPU-only North tool routing through the real ServingEngine loop."""

import json
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace as NS
from urllib.request import Request, urlopen

import pytest

from mlx2.adapters.north_mini_code import NORTH_MINI_CODE, NorthMiniCodeAdapter
from mlx2.server import collect_nonstream_job, handler_for


TOOL = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "Return weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}
ACTION = (
    '[{"tool_name":"weather","parameters":{"city":"Toronto"}}]'
    "<|END_ACTION|>"
)


def _engine(
    monkeypatch,
    script,
    *,
    use_processors=True,
    route_selection_source="engine_argument",
):
    import mlx.core as mx

    from mlx2 import memory, serving
    from mlx2.runtime import apc_v2, generate, os_memory

    class Detokenizer:
        pieces = {
            5: "think",
            6: "<|END_THINKING|>",
            10: "<|START_ACTION|>",
            11: ACTION,
            12: '[{"tool_name":"weather","parameters":{"city":"Toro',
        }

        def reset(self):
            self.text = ""
            self.offset = 0

        def add_token(self, token):
            self.text += self.pieces[token]

        def finalize(self):
            pass

        @property
        def last_segment(self):
            value = self.text[self.offset :]
            self.offset = len(self.text)
            return value

    class Tokenizer:
        vocab_size = 64
        eos_token_ids = ()

        def apply_chat_template(self, messages, **kwargs):
            return [1, 2]

        def encode(self, text, **kwargs):
            return {
                "<|START_ACTION|>": [10],
                "<|END_THINKING|>": [6],
            }[text]

        @property
        def detokenizer(self):
            return Detokenizer()

    class Adapter:
        descriptor = NORTH_MINI_CODE
        max_context = 1_000
        identity = {"fingerprint": "fake-north"}
        environment = {}
        layout = "fake-north"
        model = None

        def __init__(self, path):
            self.tokenizer = Tokenizer()

        def profile_name(self, mtp):
            return "fake-north"

        def execution_config(self, **kwargs):
            return {"num_draft": 0}

        prompt_tokens = NorthMiniCodeAdapter.prompt_tokens
        output_parser = NorthMiniCodeAdapter.output_parser

        def request_logits_processors(self, request, *, prompt_length):
            if not use_processors:
                return ()
            return NorthMiniCodeAdapter.request_logits_processors(
                self, request, prompt_length=prompt_length
            )

        def diagnostics(self):
            return {}

        def close(self):
            pass

    class APC:
        def __init__(self, **kwargs):
            self.apc_stats = {}

        def key(self, *args, **kwargs):
            return "key"

        def lookup(self, key, tokens, **kwargs):
            return NS(
                cache=None,
                cached_tokens=0,
                remaining_tokens=list(tokens),
                miss_reason=None,
                sidecar=None,
            )

        def store(self, *args, **kwargs):
            pass

        def spill_idle_entries(self):
            pass

        def evict_oldest_unleased(self):
            return False

        def clear(self):
            pass

    class Batch:
        scheduler_stats = {}
        observed_processor_counts = []

        def __init__(self, *args, **kwargs):
            self.index = 0

        def insert(self, prompts, *, all_tokens, logits_processors, **kwargs):
            self.history = list(all_tokens[0]) + list(prompts[0])
            self.processors = list(logits_processors[0])
            type(self).observed_processor_counts.append(len(self.processors))
            return [0]

        def next(self):
            token = script[self.index]
            logits = mx.zeros((1, 64), dtype=mx.float32)
            for processor in self.processors:
                logits = processor(mx.array(self.history), logits)
            assert bool(mx.isfinite(logits[0, token]).item())
            self.history.append(token)
            self.index += 1
            return [], [
                NS(
                    uid=0,
                    execution_width=1,
                    finish_reason=("length" if self.index == len(script) else None),
                    token=token,
                    mtp_state=None,
                    all_tokens=list(self.history),
                    prompt_cache=[],
                    mtp_receipt=None,
                )
            ]

        def remove(self, uids):
            self.index = len(script)

        def close(self):
            pass

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(apc_v2, "APCv2", APC)
    monkeypatch.setattr(generate, "BatchGenerator", Batch)
    engine = serving.ServingEngine(
        "fake",
        adapter_factory=Adapter,
        qualification_mode=True,
        mtp=False,
        max_lanes=1,
        route_selection_source=route_selection_source,
    )
    assert engine.ready.wait(5)
    return engine, Batch


@pytest.mark.parametrize(
    ("tool_choice", "thinking", "script"),
    [
        ("required", False, [10, 11]),
        ("required", True, [5, 6, 10, 11]),
        ({"type": "function", "function": {"name": "weather"}}, False, [10, 11]),
        ({"type": "function", "function": {"name": "weather"}}, True, [5, 6, 10, 11]),
    ],
)
def test_north_required_and_named_force_action_after_optional_thinking(
    monkeypatch, tool_choice, thinking, script
):
    engine, batch = _engine(monkeypatch, script)
    request = {
        "messages": [{"role": "user", "content": "Weather in Toronto?"}],
        "tools": [TOOL],
        "tool_choice": tool_choice,
        "enable_thinking": thinking,
        "temperature": 0,
        "max_tokens": len(script),
    }
    try:
        choice, _, _ = collect_nonstream_job(engine.submit(request), request, chat=True)
        assert choice["message"]["tool_calls"][0]["function"] == {
            "name": "weather",
            "arguments": '{"city": "Toronto"}',
        }
        assert choice["finish_reason"] == "tool_calls"
        assert batch.observed_processor_counts[-1] == 1
    finally:
        engine.close()


def test_anthropic_required_partial_action_at_budget_is_max_tokens(monkeypatch):
    engine, _ = _engine(monkeypatch, [10, 12], use_processors=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    body = {
        "model": "fake",
        "messages": [{"role": "user", "content": "Weather in Toronto?"}],
        "max_tokens": 2,
        "thinking": {"type": "disabled"},
        "tools": [
            {
                "name": "weather",
                "description": "Return weather for a city.",
                "input_schema": TOOL["function"]["parameters"],
            }
        ],
        "tool_choice": {"type": "any"},
    }
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/messages",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            payload = json.load(response)
        assert payload["stop_reason"] == "max_tokens"
        assert payload["content"] == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        engine.close()


def test_status_and_terminal_receipt_report_resolved_route_source(monkeypatch):
    engine, _ = _engine(
        monkeypatch,
        [10, 11],
        route_selection_source="adapter_default",
    )
    request = {
        "messages": [{"role": "user", "content": "Weather in Toronto?"}],
        "tools": [TOOL],
        "tool_choice": "required",
        "enable_thinking": False,
        "temperature": 0,
        "max_tokens": 2,
    }
    try:
        assert engine.snapshot["settings"]["route"] == "ordinary"
        assert (
            engine.snapshot["settings"]["route_selection_source"]
            == "adapter_default"
        )
        _, _, receipt = collect_nonstream_job(
            engine.submit(request), request, chat=True
        )
        assert receipt["route"] == "ordinary"
        assert receipt["route_selection_source"] == "adapter_default"
    finally:
        engine.close()
