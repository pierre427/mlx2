"""CPU-only Muse tool-call integration through the real ServingEngine loop."""

from types import SimpleNamespace as NS

import pytest

from mlx2.adapters.muse_glimmer import MUSE_GLIMMER, MuseGlimmerAdapter
from mlx2.server import collect_nonstream_job


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
ATEM = (
    '<atem:function_calls><atem:invoke name="weather">'
    '<atem:parameter name="city">Toronto</atem:parameter>'
    "</atem:invoke></atem:function_calls>"
)


@pytest.mark.parametrize(
    ("tool_choice", "script"),
    [
        ("required", [10, 11, 12, 20]),
        (
            {"type": "function", "function": {"name": "weather"}},
            [20],
        ),
    ],
)
def test_muse_required_and_named_tool_streams_complete_through_serving_engine(
    monkeypatch, tool_choice, script
):
    import mlx.core as mx

    from mlx2 import memory, serving
    from mlx2.runtime import apc_v2, generate, os_memory

    class Detokenizer:
        pieces = {10: " to=", 11: "weather", 12: "<|message|>", 20: ATEM}

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

        def __init__(self):
            self.rendered = None

        def apply_chat_template(self, messages, **kwargs):
            return "<|start|>assistant"

        def encode(self, text, **kwargs):
            headers = {
                " to=user<|message|>": [10, 13, 12],
                " to=weather<|message|>": [10, 11, 12],
            }
            if text in headers:
                return headers[text]
            self.rendered = text
            return [1, 2]

        @property
        def detokenizer(self):
            return Detokenizer()

    class Adapter:
        descriptor = MUSE_GLIMMER
        max_context = 1_000
        identity = {"fingerprint": "fake-muse"}
        environment = {}
        layout = "fake-muse"
        model = None
        last_instance = None

        def __init__(self, path):
            self.tokenizer = Tokenizer()
            type(self).last_instance = self

        def profile_name(self, mtp):
            return "fake-muse"

        def execution_config(self, **kwargs):
            return {"num_draft": 0}

        prompt_tokens = MuseGlimmerAdapter.prompt_tokens
        request_logits_processors = MuseGlimmerAdapter.request_logits_processors
        output_parser = MuseGlimmerAdapter.output_parser

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

        def insert(
            self,
            prompts,
            *,
            all_tokens,
            logits_processors,
            **kwargs,
        ):
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
    )
    request = {
        "messages": [{"role": "user", "content": "Weather in Toronto?"}],
        "tools": [TOOL],
        "tool_choice": tool_choice,
        "enable_thinking": False,
        "temperature": 0,
        "max_tokens": len(script),
    }
    try:
        assert engine.ready.wait(5)
        choice, _, _ = collect_nonstream_job(engine.submit(request), request, chat=True)
        call = choice["message"]["tool_calls"][0]
        assert call["function"] == {
            "name": "weather",
            "arguments": '{"city": "Toronto"}',
        }
        assert choice["finish_reason"] == "tool_calls"
        assert not engine.error
        if tool_choice == "required":
            assert Batch.observed_processor_counts[-1] == 1
            assert Adapter.last_instance.tokenizer.rendered == "<|start|>assistant"
        else:
            assert Batch.observed_processor_counts[-1] == 0
            assert Adapter.last_instance.tokenizer.rendered.endswith(
                " to=weather<|message|>"
            )
    finally:
        engine.close()
