"""return_progress prompt_progress events, POST /tokenize, POST /apply-template."""

import json
import queue
import threading
from collections import Counter
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace as NS
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from mlx2.anthropic_compat import anthropic_request_to_chat
from mlx2.openai_compat import responses_to_chat_request
from mlx2.server import handler_for, validate_request
from mlx2.serving import HostPromptCache, Job, ServingEngine, take_prompt_progress

MESSAGES = [{"role": "user", "content": "hello"}]


# -- request validation and translation ------------------------------------


def test_return_progress_validation_requires_stream_and_boolean():
    assert validate_request(
        {"messages": MESSAGES, "stream": True, "return_progress": True}
    )["return_progress"] is True
    assert "return_progress" in validate_request(
        {"messages": MESSAGES, "return_progress": False}
    )
    with pytest.raises(ValueError, match="return_progress requires stream"):
        validate_request({"messages": MESSAGES, "return_progress": True})
    with pytest.raises(ValueError, match="return_progress must be boolean"):
        validate_request({"messages": MESSAGES, "stream": True, "return_progress": 1})
    # Responses and Messages forward the flag to the shared chat request.
    request, _ = responses_to_chat_request(
        {"input": "hi", "stream": True, "return_progress": True}
    )
    assert request["return_progress"] is True
    translated = anthropic_request_to_chat(
        {"messages": MESSAGES, "max_tokens": 4, "stream": True, "return_progress": True}
    )
    assert translated["return_progress"] is True
    # A generation-only control: it must not split the host prompt cache.
    assert HostPromptCache.key({"messages": MESSAGES}) == HostPromptCache.key(
        {"messages": MESSAGES, "stream": True, "return_progress": True}
    )


# -- coalescing slot ----------------------------------------------------------


def _progress_engine():
    engine = ServingEngine.__new__(ServingEngine)
    return engine


def test_progress_is_whole_prompt_monotonic_and_coalesced():
    engine = _progress_engine()
    job = Job({"return_progress": True, "stream": True})
    job.prompt_tokens, job.cached_tokens = 10, 4
    # Suffix-relative (ordinary generator): 6 uncached tokens.
    engine._emit_prompt_progress(job, (2, 6))
    engine._emit_prompt_progress(job, (4, 6))  # coalesces into the queued event
    assert job.events.qsize() == 1
    event = job.events.get_nowait()
    first = take_prompt_progress(job, event)
    assert first["processed"] == 8 and first["total"] == 10
    assert first["cached"] == 4 and first["replay"] is False
    assert job.prompt_progress_updates == 2
    # Whole-prompt coordinates (native MTP / PLD / external) agree.
    engine._emit_prompt_progress(job, (8, 10))  # not an advance: dropped
    assert job.events.empty()
    engine._emit_prompt_progress(job, (10, 10))
    assert take_prompt_progress(job, job.events.get_nowait())["processed"] == 10
    engine._emit_prompt_progress(job, (6, 6))
    assert job.events.empty()
    # Replay restarts the floor and says so.
    engine._emit_prompt_progress(job, (0, 6), replay=True)
    replayed = take_prompt_progress(job, job.events.get_nowait())
    assert replayed == {**replayed, "processed": 4, "replay": True}
    # Malformed progress from a generator that reports none is ignored.
    engine._emit_prompt_progress(job, None)
    engine._emit_prompt_progress(job, 0)
    assert job.events.empty()


def test_progress_never_trips_queue_full_or_uses_more_than_one_slot():
    engine = _progress_engine()
    job = Job({"return_progress": True, "stream": True})
    job.events = queue.Queue(maxsize=2)
    job.prompt_tokens = 100
    job.events.put_nowait({"text": "a"})
    job.events.put_nowait({"text": "b"})
    engine._emit_prompt_progress(job, (10, 100))
    assert not job.cancelled.is_set()
    assert job.prompt_progress_dropped == 1
    assert job.prompt_progress_event is None
    job.events.get_nowait()
    engine._emit_prompt_progress(job, (20, 100))
    engine._emit_prompt_progress(job, (30, 100))
    assert job.events.qsize() == 2
    assert take_prompt_progress(job, job.events.queue[-1])["processed"] == 30


def test_progress_producer_and_consumer_threads_never_regress():
    engine = _progress_engine()
    job = Job({"return_progress": True, "stream": True})
    job.prompt_tokens = 5000
    seen = []
    done = threading.Event()

    def consume():
        while not done.is_set() or not job.events.empty():
            try:
                event = job.events.get(timeout=0.01)
            except queue.Empty:
                continue
            seen.append(take_prompt_progress(job, event)["processed"])

    thread = threading.Thread(target=consume)
    thread.start()
    for done_tokens in range(1, 5001):
        engine._emit_prompt_progress(job, (done_tokens, 5000))
        assert job.events.qsize() <= 1
    done.set()
    thread.join(5)
    assert seen == sorted(set(seen)) and seen[-1] == 5000


# -- serving loop ---------------------------------------------------------------


PIECES = ["<eos>", "a", "b", "c"] + [f"<t{i}>" for i in range(60)]
EOS = 0


@pytest.fixture
def progress_engine(monkeypatch):
    """A real ServingEngine loop over a generator with chunked prefill."""
    import mlx.core as mx

    from mlx2 import memory, serving
    from mlx2.runtime import apc_v2, generate, os_memory, pld

    state = {"prompt": list(range(1, 11)), "cached": 4, "chunk": 2}

    class APC:
        def __init__(self, **kw): self.apc_stats = {}
        def key(self, *a, **kw): return "key"
        def lookup(self, key, tokens, **kw):
            cached = state["cached"]
            return NS(cache=[NS(nbytes=0)], cached_tokens=cached,
                      remaining_tokens=list(tokens[cached:]), sidecar=None,
                      miss_reason=None)
        def store(self, *a, **kw): pass
        def spill_idle_entries(self): pass
        def evict_oldest_unleased(self): return False
        def clear(self): pass
        def __len__(self): return 0

    class Batch:
        scheduler_stats = {}

        @staticmethod
        def validate_policy(policy):
            return dict(policy)

        def __init__(self, *a, **kw):
            self.lanes = {}
            self.uid = 0

        def insert(self, prompts, max_tokens=None, **kw):
            uid = self.uid
            self.uid += 1
            self.lanes[uid] = {"done": 0, "span": len(prompts[0]),
                               "generated": [], "max": max_tokens[0]}
            return [uid]

        def pop_prompt_boundary(self, uid):
            return None

        def next(self):
            prompts, responses = [], []
            for uid, lane in list(self.lanes.items()):
                if lane["done"] < lane["span"]:
                    # Suffix-relative progress, like PromptProcessingBatch.
                    lane["done"] = min(lane["span"], lane["done"] + state["chunk"])
                    end = lane["done"] == lane["span"]
                    prompts.append(NS(uid=uid, progress=(lane["done"], lane["span"]),
                                      end_of_segment=end, end_of_prompt=end))
                    if not end:
                        continue
                token = 1 if len(lane["generated"]) < 2 else EOS
                lane["generated"].append(token)
                finish = "stop" if token == EOS else None
                if finish:
                    del self.lanes[uid]
                responses.append(NS(uid=uid, execution_width=1, finish_reason=finish,
                                    token=token, mtp_state=None,
                                    all_tokens=None, prompt_cache=[],
                                    mtp_receipt=None, logprobs=None))
            return prompts, responses

        def remove(self, uids):
            for uid in uids:
                self.lanes.pop(uid, None)

        def close(self): pass

    class Detokenizer:
        def __init__(self): self.segment = ""
        def reset(self): self.segment = ""
        def add_token(self, token): self.segment += PIECES[token]
        def finalize(self): pass

        @property
        def last_segment(self):
            segment, self.segment = self.segment, ""
            return segment

    class Adapter:
        max_context = 2000
        identity = {"fingerprint": "fake"}
        environment = {}
        layout = "fake"
        model = None
        tool_constraint = None

        def __init__(self, path):
            self.tokenizer = NS(
                vocab_size=len(PIECES), eos_token_ids=[EOS],
                decode=lambda ids, **_kw: "".join(PIECES[i] for i in ids),
                encode=lambda text, **_kw: [PIECES.index(c) for c in text],
                detokenizer=Detokenizer(),
            )
            self.descriptor = NS(capabilities=frozenset())

        def profile_name(self, mtp): return "fake"
        def execution_config(self, **kw): return {"num_draft": 0}
        def prompt_tokens(self, request): return list(state["prompt"])

        def output_parser(self, request):
            from mlx2.output import OutputParser

            return OutputParser(chat="messages" in request, thinking=False)

        def diagnostics(self): return {}
        def close(self): pass

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(apc_v2, "APCv2", APC)
    monkeypatch.setattr(generate, "BatchGenerator", Batch)
    monkeypatch.setattr(pld, "PromptLookupBatchGenerator", Batch)
    monkeypatch.setattr(mx, "synchronize", lambda: None)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)

    engine = serving.ServingEngine(
        "fake", adapter_factory=Adapter, qualification_mode=True, mtp=False
    )
    assert engine.ready.wait(5)
    yield engine, state
    engine.close()
    assert not engine.error


def _drain(job):
    events = []
    while True:
        event = job.events.get(timeout=10)
        if "prompt_progress" in event:
            event = {"prompt_progress": take_prompt_progress(job, event)}
        events.append(event)
        if "finish_reason" in event or "error" in event:
            return events


def test_serving_emits_progress_before_first_token(progress_engine):
    engine, state = progress_engine
    request = {"messages": MESSAGES, "stream": True, "return_progress": True,
               "max_tokens": 8, "temperature": 0}
    events = _drain(engine.submit(request))
    kinds = ["progress" if "prompt_progress" in e else "text" if "delta" in e or "text" in e
             else "finish" if "finish_reason" in e else "other" for e in events]
    first_text = kinds.index("text")
    progress = [e["prompt_progress"] for e in events if "prompt_progress" in e]
    assert progress and all(kind == "progress" for kind in kinds[:len(progress)])
    assert kinds.index("text") > max(i for i, k in enumerate(kinds) if k == "progress")
    processed = [p["processed"] for p in progress]
    assert processed == sorted(set(processed))
    assert processed[-1] == 10 and processed[0] > 4
    assert all(p["total"] == 10 and p["cached"] == 4 and not p["replay"] for p in progress)
    assert first_text > 0
    receipt = events[-1]["receipt"]
    assert receipt["prompt_progress"]["updates"] == 3  # 6 uncached tokens / chunk 2
    assert receipt["prompt_progress"]["dropped"] == 0

    # Default requests are unchanged: no progress events, no receipt key.
    plain = _drain(engine.submit({"messages": MESSAGES, "stream": True,
                                  "max_tokens": 8, "temperature": 0}))
    assert not any("prompt_progress" in e for e in plain)
    assert "prompt_progress" not in plain[-1]["receipt"]


def test_render_prompt_is_the_admission_prompt(progress_engine):
    engine, state = progress_engine
    assert engine.render_prompt({"messages": MESSAGES}) == state["prompt"]
    assert engine.count_tokens({"messages": MESSAGES}) == 10
    # The fake adapter has no text renderer.
    assert engine.apply_template({"messages": MESSAGES}) is None


# -- HTTP ---------------------------------------------------------------------


class HTTPEngine:
    model_path = "fixture"
    max_context = 4096

    def __init__(self):
        self.counts = Counter()
        self.render_template = True
        self.rendered = []
        self.progress = [(3, 8), (8, 8)]

    def status(self):
        return {"healthy": True, "error": None, "model": "fixture"}

    def batching_status(self):
        return {}

    def render_prompt(self, request):
        self.rendered.append(request)
        return [7, 8, 9]

    def count_tokens(self, request):
        return len(self.render_prompt(request))

    def apply_template(self, request):
        self.rendered.append(request)
        return "<user>hello</user>" if self.render_template else None

    def submit(self, request, *, tenant_id="default"):
        job = Job(request)
        job.prompt_tokens, job.cached_tokens, job.completion_tokens = 8, 2, 1
        if request.get("return_progress"):
            # One coalesced update through the real producer, then a second
            # already-queued one as a later loop iteration would add it.
            ServingEngine.__new__(ServingEngine)._emit_prompt_progress(job, (3, 6))
            job.events.put_nowait({"prompt_progress": {
                "processed": 8, "total": 8, "cached": 2, "replay": False, "time_ms": 1}})
        if request.get("tools"):
            job.events.put({"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                                      "function": {"name": "lookup", "arguments": "{}"}}]}})
            finish = "tool_calls"
        else:
            job.events.put({"delta": {"content": "answer"}})
            finish = "stop"
        job.events.put({"finish_reason": finish, "receipt": {"cache": "apcv2"}})
        return job


@pytest.fixture
def endpoint():
    engine = HTTPEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield engine, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def _post(base, path, body, tenant="tenant-a"):
    return urlopen(Request(base + path, method="POST", data=json.dumps(body).encode(),
                           headers={"Content-Type": "application/json",
                                    "X-Tenant-ID": tenant}))


def _sse(wire):
    frames = []
    for block in wire.strip().split("\n\n"):
        data = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if data and data[0] != "[DONE]":
            frames.append(json.loads(data[0]))
    return frames


def test_chat_and_completions_stream_progress_chunks_before_content(endpoint):
    _engine, base = endpoint
    with _post(base, "/v1/chat/completions",
               {"messages": MESSAGES, "stream": True, "return_progress": True}) as response:
        frames = _sse(response.read().decode())
    progress = [f for f in frames if "prompt_progress" in f]
    assert [f["prompt_progress"]["processed"] for f in progress] == [5, 8]
    assert all(f["choices"][0]["delta"] == {} for f in progress)
    assert frames.index(progress[-1]) < next(
        i for i, f in enumerate(frames) if f["choices"][0]["delta"].get("content"))

    with _post(base, "/v1/completions",
               {"prompt": "hi", "stream": True, "return_progress": True}) as response:
        frames = _sse(response.read().decode())
    assert frames[0]["choices"][0]["text"] == "" and "prompt_progress" in frames[0]

    with pytest.raises(HTTPError) as error:
        _post(base, "/v1/chat/completions", {"messages": MESSAGES, "return_progress": True})
    assert error.value.code == 400


def test_responses_and_messages_progress_events(endpoint):
    _engine, base = endpoint
    with _post(base, "/v1/responses",
               {"input": "hi", "stream": True, "return_progress": True}) as response:
        wire = response.read().decode()
    types = [f["type"] for f in _sse(wire)]
    assert types[:3] == ["response.created", "response.in_progress", "response.in_progress"]
    assert "prompt_progress" in _sse(wire)[1]

    with _post(base, "/v1/messages",
               {"messages": MESSAGES, "max_tokens": 8, "stream": True,
                "return_progress": True}) as response:
        frames = _sse(response.read().decode())
    assert [f["type"] for f in frames[:3]] == ["message_start", "ping", "ping"]
    assert frames[2]["prompt_progress"]["processed"] == 8


def test_buffered_tool_stream_drops_progress(endpoint):
    _engine, base = endpoint
    tools = [{"type": "function", "function": {"name": "lookup", "strict": True,
                                               "parameters": {"type": "object"}}}]
    with _post(base, "/v1/chat/completions",
               {"messages": MESSAGES, "stream": True, "return_progress": True,
                "tools": tools, "tool_choice": "required"}) as response:
        frames = _sse(response.read().decode())
    assert not any("prompt_progress" in f for f in frames)
    assert frames[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "lookup"


def test_tokenize_and_apply_template(endpoint):
    engine, base = endpoint
    with _post(base, "/tokenize", {"messages": MESSAGES}, tenant="tenant-b") as response:
        assert json.load(response) == {"tokens": [7, 8, 9], "count": 3, "max_model_len": 4096}
    # The body is validated as the generation request it describes.
    assert engine.rendered[-1]["messages"] == MESSAGES
    with _post(base, "/tokenize", {"prompt": "hi", "model": "fixture"}) as response:
        assert json.load(response)["count"] == 3
    with _post(base, "/apply-template", {"messages": MESSAGES}) as response:
        assert json.load(response) == {"prompt": "<user>hello</user>"}

    for body, code in (
        ({"messages": MESSAGES, "model": "other"}, 404),
        ({"messages": []}, 400),
        ({"messages": MESSAGES, "bogus": 1}, 400),
        ({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:,"}}]}]}, 400),
    ):
        with pytest.raises(HTTPError) as error:
            _post(base, "/tokenize", body)
        assert error.value.code == code, body

    engine.render_template = False
    with pytest.raises(HTTPError) as error:
        _post(base, "/apply-template", {"messages": MESSAGES})
    assert error.value.code == 501


def test_route_labels():
    handler = handler_for(HTTPEngine())
    for path, label in (("/tokenize", "tokenize"), ("/apply-template", "apply_template")):
        instance = handler.__new__(handler)
        instance.path = path + "?x=1"
        assert instance._metric_route() == label


# -- adapter renderers ----------------------------------------------------------


MODELS = Path.home() / "mlx-models"
CASES = [
    {"messages": MESSAGES},
    {"messages": [{"role": "system", "content": "Be brief."},
                  {"role": "user", "content": "Sum 1 and 2."},
                  {"role": "assistant", "content": "",
                   "tool_calls": [{"id": "c1", "type": "function",
                                   "function": {"name": "sum", "arguments": "{\"a\": 1, \"b\": 2}"}}]},
                  {"role": "tool", "tool_call_id": "c1", "content": "3"},
                  {"role": "user", "content": "Thanks — and now?"}],
     "tools": [{"type": "function", "function": {
         "name": "sum", "description": "add", "parameters": {
             "type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}}}]},
    {"messages": MESSAGES, "enable_thinking": True},
    {"prompt": "raw <|im_start|> text"},
]


def _real_tokenizer(name):
    path = MODELS / name
    if not (path / "tokenizer_config.json").exists():
        pytest.skip(f"{name} tokenizer not available")
    from mlx2.runtime.tokenizer_utils import TokenizerWrapper
    from transformers import AutoTokenizer

    return TokenizerWrapper(
        AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    )


@pytest.mark.parametrize(
    "module,cls,model",
    [
        ("mlx2.adapters.flash_next", "FlashNextAdapter", "Qwen3.8-Flash-Next-MLX-4bit-MTP"),
        ("mlx2.adapters.qwen38_27b", "Qwen3827BAdapter", "Qwen3.8-27B-MLX-4bit"),
        ("mlx2.adapters.north_mini_code", "NorthMiniCodeAdapter", "North-Mini-Code-1.0-mlx-4bit"),
        ("mlx2.adapters.muse_glimmer", "MuseGlimmerAdapter", "Muse-Glimmer-30B-mlx-4bit"),
    ],
)
def test_encode_render_prompt_equals_prompt_tokens(module, cls, model):
    import importlib

    adapter_class = getattr(importlib.import_module(module), cls)
    adapter = adapter_class.__new__(adapter_class)
    adapter.tokenizer = _real_tokenizer(model)
    for case in CASES:
        text = adapter.render_prompt(case)
        assert isinstance(text, str)
        assert list(adapter.tokenizer.encode(text, add_special_tokens=False)) == list(
            adapter.prompt_tokens(case)
        ), (model, case)


def test_mlx_vlm_renderer_matches_and_rejects_prepared_media():
    from mlx2.adapters.mlx_vlm import _MLXVLMAdapter

    adapter = _MLXVLMAdapter.__new__(_MLXVLMAdapter)
    adapter.processor = NS(
        chat_template="x",
        apply_chat_template=lambda messages, **_kw: "<u>" + messages[-1]["content"],
    )
    adapter.tokenizer = NS(encode=lambda text, **_kw: [ord(c) for c in text])
    assert adapter.render_prompt({"messages": MESSAGES}) == "<u>hello"
    assert adapter.prompt_tokens({"messages": MESSAGES}) == [ord(c) for c in "<u>hello"]
    assert adapter.render_prompt({"prompt": "p"}) == "p"
    with pytest.raises(ValueError, match="no text rendering"):
        adapter.render_prompt({"_mlx2_prompt_tokens": [1]})
    assert adapter.prompt_tokens({"_mlx2_prompt_tokens": [1]}) == [1]


def test_external_and_pld_prefill_report_progress():
    from collections import deque

    from mlx2.runtime.pld import PromptLookupBatchGenerator

    generator = PromptLookupBatchGenerator.__new__(PromptLookupBatchGenerator)
    generator.prefill_step = 3
    generator.scheduler_stats = Counter()
    generator.boundaries = {}
    generator.model = lambda *a, **kw: None
    lane = NS(uid=1, remaining=deque([5, 6, 7, 8, 9]), history=[1, 2],
              lookup_history=[1, 2, 5, 6, 7, 8, 9], cache=[])
    generator._capture_lane_recovery = lambda lane: None
    generator._arm_lane_speculation = lambda lane: None
    import mlx2.runtime.pld as pld_module

    first = pld_module.PromptLookupBatchGenerator._prefill(generator, lane)
    assert first.progress == (5, 7) and not first.end_of_prompt
    second = pld_module.PromptLookupBatchGenerator._prefill(generator, lane)
    assert second.progress == (7, 7) and second.end_of_prompt

    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    external = ExternalDraftBatchGenerator.__new__(ExternalDraftBatchGenerator)
    empty = NS(shape=(1, 0))
    external.prefill_step = 3
    external.layers = None
    external.scheduler_stats = Counter()
    external.boundaries = {}
    external.mx = NS(array=lambda value: value, eval=lambda *a: None)
    external.model = NS(prefill_body=lambda *a: empty)
    external._sidecar = lambda lane: None
    lane = NS(uid=2, remaining=deque([5, 6, 7, 8, 9]), history=[1, 2], tail=empty,
              cache=[], draft_cache=[])
    first = ExternalDraftBatchGenerator._prefill(external, lane)
    assert first.progress == (5, 7) and not first.end_of_prompt
    second = ExternalDraftBatchGenerator._prefill(external, lane)
    assert second.progress == (7, 7) and second.end_of_prompt
