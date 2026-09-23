"""A request-private prefill row waits for an isolated B=1 boundary.

A row carrying ``prefill_inputs`` (encoder media, or a persistent concept
payload) must run its first prefill alone.  Admission used to isolate it only
from other rows picked in the same round, so a media request arriving while a
text request was still mid-prefill joined that prompt batch and
``PromptProcessingBatch.prompt`` raised inside ``batch.next()``, killing the
serving worker.  These tests drive the real ``BatchGenerator`` and the real
``ServingEngine`` loop on CPU with a tiny attention model.
"""

import time

import mlx.core as mx
import mlx.nn as nn
import pytest

mx.set_default_device(mx.cpu)

from mlx2 import memory, serving
from mlx2.runtime import os_memory
from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.models.base import create_attention_mask
from mlx2.runtime.models.cache import KVCache
from mlx2.serving import ServingEngine

V = 32
D = 16


class TinyMediaModel(nn.Module):
    """Two attention layers; ``pixel_values`` shifts the prompt embedding."""

    def __init__(self, delay=0.0):
        super().__init__()
        mx.random.seed(0)
        self.embed = nn.Embedding(V, D)
        self.qs = [nn.Linear(D, D, bias=False) for _ in range(2)]
        self.ks = [nn.Linear(D, D, bias=False) for _ in range(2)]
        self.vs = [nn.Linear(D, D, bias=False) for _ in range(2)]
        self.out = nn.Linear(D, V, bias=False)
        self.delay = delay
        self.media_calls = []
        mx.eval(self.parameters())

    @property
    def layers(self):
        return [0, 1]

    def make_cache(self):
        return [KVCache() for _ in range(2)]

    def __call__(self, inputs, cache=None, pixel_values=None, **_kwargs):
        if self.delay:
            time.sleep(self.delay)
        h = self.embed(inputs)
        if pixel_values is not None:
            self.media_calls.append(tuple(inputs.shape))
            h = h + 3.0 * pixel_values.sum()
        B, L, _ = h.shape
        for i in range(2):
            q = self.qs[i](h).reshape(B, L, 1, D).transpose(0, 2, 1, 3)
            k = self.ks[i](h).reshape(B, L, 1, D).transpose(0, 2, 1, 3)
            v = self.vs[i](h).reshape(B, L, 1, D).transpose(0, 2, 1, 3)
            mask = create_attention_mask(h, cache[i], return_array=True)
            k, v = cache[i].update_and_fetch(k, v)
            o = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=D**-0.5, mask=mask
            )
            h = h + o.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.out(h)


def _generator(model):
    # Configured as serving builds the ordinary route with max_lanes=4.
    return BatchGenerator(
        model,
        completion_batch_size=4,
        prefill_batch_size=2,
        prefill_step_size=8,
        prefill_batch_window=1,
        adaptive_prefill=True,
    )


def _drive(gen, uids, limit=500):
    out = {uid: [] for uid in uids}
    done = set()
    for _ in range(limit):
        _, responses = gen.next()
        for response in responses:
            out.setdefault(response.uid, []).append(int(response.token))
            if response.finish_reason:
                done.add(response.uid)
        if done >= set(uids):
            return out
    raise AssertionError(f"lanes did not finish: {sorted(set(uids) - done)}")


TEXT = [(3 * i + 1) % (V - 1) + 1 for i in range(29)]  # several 8-token chunks
MEDIA = [3, 4, 5, 6]


def _media_inputs():
    # Evaluate here: serving prepares media on another thread, and a lazy
    # array would be bound to that thread's stream.
    pixels = mx.ones((1, 3))
    mx.eval(pixels)
    return {"pixel_values": pixels}


def test_media_row_waits_for_a_mid_prefill_text_lane_then_runs_alone():
    alone = {}
    for name, prompt, inputs in (
        ("text", TEXT, None),
        ("media", MEDIA, _media_inputs()),
    ):
        gen = _generator(TinyMediaModel())
        extra = {"prefill_inputs": [inputs]} if inputs is not None else {}
        uid = gen.insert([prompt], max_tokens=[4], **extra)[0]
        alone[name] = _drive(gen, [uid])[uid]

    model = TinyMediaModel()
    gen = _generator(model)
    text_uid = gen.insert([TEXT], max_tokens=[4])[0]
    gen.next()
    assert gen._prompt_batch.uids == [text_uid]
    media_uid = gen.insert(
        [MEDIA], max_tokens=[4], prefill_inputs=[_media_inputs()]
    )[0]
    tokens = _drive(gen, [text_uid, media_uid])

    assert tokens == {text_uid: alone["text"], media_uid: alone["media"]}
    # The encoder payload reached the model once, at a B=1 boundary.
    assert model.media_calls == [(1, len(MEDIA) - 1)]


class _Detok:
    def __init__(self):
        self.last_segment = ""

    def reset(self):
        self.last_segment = ""

    def add_token(self, token):
        self.last_segment = f"{int(token)} "

    def finalize(self):
        pass


class _Parser:
    stopped = False
    tool_count = 0

    def push(self, text, final=False):
        return [{"content": text}] if text else []


class _Tokenizer:
    vocab_size = V
    eos_token_ids = []

    @property
    def detokenizer(self):
        return _Detok()


def _adapter(model):
    class Adapter:
        max_context = 8192
        identity = {"fingerprint": "tiny-media"}
        environment = {}
        layout = "tiny-media-layout"
        tokenizer = _Tokenizer()

        def __init__(self, _path):
            self.model = model

        def profile_name(self, _mtp):
            return "tiny-ordinary"

        def execution_config(self, *, max_lanes, prefill_step):
            return {
                "persistent": True,
                "num_draft": 0,
                "backend": "ordinary",
                "rate_gate": False,
                "prefill_step_size": prefill_step,
            }

        def prompt_tokens(self, request):
            if "_mlx2_prompt_tokens" in request:
                return list(request["_mlx2_prompt_tokens"])
            return list(request["tokens"])

        def prepare_multimodal_request(self, request, *, file_loader=None):
            return {
                **request,
                "_mlx2_prompt_tokens": list(MEDIA) + [7],
                "_mlx2_prefill_inputs": _media_inputs(),
                "_mlx2_media_token_end": len(MEDIA),
                "_mlx2_media_fingerprint": "img-1",
            }

        def output_parser(self, _request):
            return _Parser()

        def diagnostics(self):
            return {}

        def close(self):
            pass

    return Adapter


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)


LONG_TEXT = [(i % 30) + 1 for i in range(1200)]
IMAGE_REQUEST = {
    "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    "max_tokens": 4,
    "temperature": 0,
}
TEXT_REQUEST = {"tokens": LONG_TEXT, "max_tokens": 4, "temperature": 0}


def _final_tokens(job):
    tokens = []
    while True:
        event = job.events.get(timeout=60)
        assert "error" not in event, event
        if "delta" in event:
            tokens.extend(int(t) for t in event["delta"]["content"].split())
        if "finish_reason" in event:
            return tokens


def _engine(model):
    engine = ServingEngine(
        "tiny",
        adapter_factory=_adapter(model),
        qualification_mode=True,
        mtp=False,
        max_lanes=4,
        prefill_step=8,
    )
    assert engine.ready.wait(60), engine.error
    return engine


def _alone(request):
    engine = _engine(TinyMediaModel())
    try:
        return _final_tokens(engine.submit(dict(request)))
    finally:
        engine.close()


def test_serving_media_request_during_text_prefill_keeps_the_worker(host):
    text_alone = _alone(TEXT_REQUEST)
    image_alone = _alone(IMAGE_REQUEST)

    # Each chunk sleeps so the long text prompt is still mid-prefill when
    # the image request reaches the worker.
    engine = _engine(TinyMediaModel(delay=0.002))
    try:
        text_job = engine.submit({**TEXT_REQUEST, "return_progress": True})
        first = text_job.events.get(timeout=60)
        assert "prompt_progress" in first, first
        image_job = engine.submit(dict(IMAGE_REQUEST))
        image_tokens = _final_tokens(image_job)
        text_tokens = _final_tokens(text_job)
        assert engine.error is None
        assert engine.thread.is_alive()
    finally:
        engine.close()

    assert image_tokens == image_alone
    assert text_tokens == text_alone
