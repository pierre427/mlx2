"""Prompt rendering happens on the submitting thread, not the generation worker.

Admission used to render every fresh prompt on ``mlx2-generation`` under
``prompt_lock``, stalling every decoding lane for the render (~0.7 ms per 1k
prompt tokens).  Grammar compilation already moved off the worker for the same
reason; rendering follows.
"""

import threading

import mlx.core as mx
import pytest

from route_harness import collect, make_engine, patch_host, tiny_qwen38_mtp


class RecordingRender:
    max_context = 4096
    threads = []

    def prompt_tokens(self, request):
        RecordingRender.threads.append(threading.current_thread().name)
        return list(request["tokens"])


@pytest.fixture
def engine(monkeypatch):
    patch_host(monkeypatch)
    mx.set_default_device(mx.cpu)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=False, max_lanes=2, adapter_mixin=RecordingRender)
    engine.host_prompt_cache.max_entries = 0  # every request renders
    try:
        yield engine
    finally:
        engine.close()


def test_fresh_prompt_renders_off_the_worker(engine):
    RecordingRender.threads.clear()
    prompt = [(i * 7) % 100 + 1 for i in range(24)]
    job = engine.submit({"tokens": prompt, "max_tokens": 3, "temperature": 0})
    result = collect(job)
    assert "error" not in result, result
    assert job.prompt_tokens == len(prompt)
    assert RecordingRender.threads, "prompt_tokens never ran"
    assert engine.thread.name not in RecordingRender.threads


def test_render_failure_is_still_reported_by_admission(engine):
    class Boom(Exception):
        pass

    def broken(request):
        raise ValueError("template rejected the request")

    engine.adapter.prompt_tokens = broken
    job = engine.submit({"tokens": [1, 2, 3], "max_tokens": 2, "temperature": 0})
    result = collect(job)
    assert "error" in result
    assert "template rejected" in str(result["error"])
