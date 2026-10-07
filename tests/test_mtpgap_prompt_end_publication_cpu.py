"""Prompt-end APCv2 publication must not hold back a neighbour's tokens.

GPU loop trace (mtpgap 2026-10-07, Qwen3.8-27B, APCv2 at its resident cap):
the round that finished a 16K prompt returned the decoding neighbour's
tokens together with the prompt-end response, and the worker stored the
prompt boundary (plus, for self-MTP, its interior checkpoints) before it
delivered those tokens.  Each store spilled older entries to disk
synchronously: 591 ms (ordinary) and 869 ms (native MTP) were added to the
neighbour gap that already carried the final prefill slice.

Here the store is made slow and the event order is recorded: the
neighbour's tokens decoded in the round that ended the second prompt must
be delivered before that prompt's boundary is stored.
"""

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_approximate_kv_serving import engine_for, tiny_model

from mlx2 import memory, serving
from mlx2.runtime import generate as G
from mlx2.runtime import os_memory


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)


def test_neighbour_tokens_are_delivered_before_prompt_end_publication(
    monkeypatch, host
):
    log = []
    lock = threading.Lock()
    real_next = G.BatchGenerator.next

    def next_(self):
        prompts, responses = real_next(self)
        with lock:
            log.append((
                "next",
                [r.uid for r in responses],
                [r.uid for r in prompts if r.end_of_prompt],
            ))
        return prompts, responses

    monkeypatch.setattr(G.BatchGenerator, "next", next_)
    engine = engine_for(tiny_model(), operations=None)
    assert engine.ready.wait(30), engine.error
    real_emit = engine._emit
    real_publish = engine._publish_checkpoint

    def emit(job, event):
        if "delta" in event:
            with lock:
                log.append(("deliver", job.uid))
        return real_emit(job, event)

    def publish(apc, key, tokens, prompt_cache, **kwargs):
        with lock:
            log.append(("store", len(tokens), kwargs.get("retention_role")))
        if kwargs.get("retention_role") == "committed_prompt_boundary":
            time.sleep(0.05)
        return real_publish(apc, key, tokens, prompt_cache, **kwargs)

    engine._emit = emit
    engine._publish_checkpoint = publish
    try:
        neighbour = engine.submit(
            {"tokens": [3, 4, 5, 6, 7], "max_tokens": 120, "temperature": 0,
             "ignore_eos": True}
        )
        for _ in range(3):
            assert "delta" in neighbour.events.get(timeout=60)
        second = engine.submit(
            {"tokens": list(range(10, 50)), "max_tokens": 3, "temperature": 0}
        )
        while True:
            event = second.events.get(timeout=60)
            assert "error" not in event, event
            if "finish_reason" in event:
                break
        with lock:
            events = list(log)
    finally:
        engine.close()
    uid_a, uid_b = neighbour.uid, second.uid
    # The round whose prompt responses ended the second prompt.
    end = next(
        i for i, e in enumerate(events) if e[0] == "next" and uid_b in e[2]
    )
    decoded = events[end][1].count(uid_a)
    assert decoded >= 1, "the neighbour did not decode in the prompt-end round"
    store = next(
        i for i, e in enumerate(events)
        if i > end and e[0] == "store" and e[2] == "committed_prompt_boundary"
    )
    delivered_first = [
        e for e in events[end + 1:store] if e == ("deliver", uid_a)
    ]
    assert len(delivered_first) >= 1, events[end:store + 3]


def test_a_lone_request_publishes_before_its_first_token(monkeypatch, host):
    """Guard (Flash-Next confirm-20261007 triage): with no neighbour the
    reorder changes nothing.  The prompt-end round returns no token for the
    request itself, so its boundary is stored before its first token, inside
    its TTFT, as before -- the 16-token decode-rate drop measured there for
    the lone long request is not this change."""
    log = []
    lock = threading.Lock()
    engine = engine_for(tiny_model(), operations=None)
    assert engine.ready.wait(30), engine.error
    real_emit = engine._emit
    real_publish = engine._publish_checkpoint

    def emit(job, event):
        if "delta" in event:
            with lock:
                log.append(("deliver", job.uid))
        return real_emit(job, event)

    def publish(apc, key, tokens, prompt_cache, **kwargs):
        with lock:
            log.append(("store", kwargs.get("retention_role")))
        return real_publish(apc, key, tokens, prompt_cache, **kwargs)

    engine._emit = emit
    engine._publish_checkpoint = publish
    try:
        job = engine.submit(
            {"tokens": list(range(10, 50)), "max_tokens": 4, "temperature": 0}
        )
        while "finish_reason" not in (event := job.events.get(timeout=60)):
            assert "error" not in event, event
        with lock:
            events = list(log)
    finally:
        engine.close()
    store = events.index(("store", "committed_prompt_boundary"))
    first = events.index(("deliver", job.uid))
    assert store < first, events
