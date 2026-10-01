"""The scheduler's mixed prefill-and-decode round (MLX2_MIXED_PREFILL_DECODE).

A prompt mid-prefill beside decoding lanes runs its slice inside the decode
step's forward.  Greedy outputs must equal the ordinary rounds' outputs on
CPU, the rounds must actually be mixed, and the switch is off by default.
"""

import threading

import mlx.core as mx
import pytest

import route_harness as rh
from mlx2.runtime import generate


def _run(monkeypatch, *, mixed, model_override=None):
    if mixed:
        monkeypatch.setenv("MLX2_MIXED_PREFILL_DECODE", "1")
    else:
        monkeypatch.delenv("MLX2_MIXED_PREFILL_DECODE", raising=False)
    rh.patch_host(monkeypatch)
    live = {}
    original = generate.BatchGenerator._next_mixed

    def spy(self):
        live["stats"] = self.scheduler_stats  # the live dict, not a snapshot
        return original(self)

    monkeypatch.setattr(generate.BatchGenerator, "_next_mixed", spy)
    chunks = live.setdefault("chunks", [])
    original_prompt = generate.PromptProcessingBatch.prompt

    def prompt_spy(self, tokens, *, forward_fn=None):
        if any(len(t) for t in tokens):  # empty calls slice nothing
            chunks.append(("mixed" if forward_fn else "plain", bool(getattr(self, "decode_active", False))))
        return original_prompt(self, tokens, forward_fn=forward_fn)

    monkeypatch.setattr(generate.PromptProcessingBatch, "prompt", prompt_spy)
    model, vocab = rh.tiny_qwen38_mtp()
    if model_override is not None:
        model = model_override(model)
    engine = rh.make_engine(model, vocab, mtp=False, max_lanes=4)
    engine.host_prompt_cache.max_entries = 0
    try:
        a_prompt = [(i * 7) % (vocab - 1) + 1 for i in range(24)]
        b_prompt = [(i * 13 + 5) % (vocab - 1) + 1 for i in range(150)]
        results = {}
        job_a = engine.submit({"tokens": a_prompt, "max_tokens": 60, "temperature": 0})
        # Let A reach decode before B arrives, so B's slices contend.
        first = job_a.events.get(timeout=120)
        while "delta" not in first and "finish_reason" not in first:
            first = job_a.events.get(timeout=120)
        rest = threading.Thread(target=lambda: results.update(a=rh.collect(job_a)))
        rest.start()
        results["b"] = rh.collect(
            engine.submit({"tokens": b_prompt, "max_tokens": 12, "temperature": 0})
        )
        rest.join()
        return results, dict(live.get("stats") or {}, _chunks=list(chunks))
    finally:
        engine.close()


def _tokens(result):
    assert "error" not in result, result
    return result["tokens"]


def test_mixed_rounds_engage_and_match_ordinary_greedy_outputs(monkeypatch):
    ordinary, ordinary_stats = _run(monkeypatch, mixed=False)
    mixed, mixed_stats = _run(monkeypatch, mixed=True)
    assert ordinary_stats.get("mixed_rounds", 0) == 0
    assert mixed_stats.get("mixed_rounds", 0) > 0, mixed_stats
    assert mixed_stats.get("mixed_declined_rounds", 0) == 0
    assert mixed_stats.get("mixed_prompt_tokens", 0) > 0
    # Beside a decoding lane every slice rides in a mixed forward: the round
    # that admits the prompt defers its slice instead of running a plain one
    # back to back with the first mixed forward.
    contended_plain = [c for c in mixed_stats["_chunks"] if c == ("plain", True)]
    assert contended_plain == [], mixed_stats["_chunks"]
    assert mixed_stats.get("mixed_deferred_slices", 0) >= 1
    assert _tokens(mixed["b"]) == _tokens(ordinary["b"])
    # A's first token was taken above; the remainder must match too.
    assert _tokens(mixed["a"]) == _tokens(ordinary["a"])


def test_models_without_a_mixed_forward_keep_ordinary_rounds(monkeypatch):
    class NoMixed:
        def __init__(self, model):
            self._model = model

        def __getattr__(self, name):
            if name == "mixed_forward":
                raise AttributeError(name)
            return getattr(self._model, name)

        def __call__(self, *args, **kwargs):
            return self._model(*args, **kwargs)

    _, stats = _run(monkeypatch, mixed=True, model_override=NoMixed)
    assert stats.get("mixed_rounds", 0) == 0
