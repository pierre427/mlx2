"""Self-MTP emission validity in one host read per round (mlx-lm #1950 class).

``MTPGenerationBatch`` validated every emitted token with its own
``mx.isfinite(...).item()`` sync.  The round now reads every emitted log
probability in one stacked check and falls back to the unchanged per-token
loop only when that check fails, so tokens, history and error messages are
identical.
"""

import random
import traceback
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime import generate as G

import route_harness as H


def _out(token, values, dtype=mx.float32):
    return SimpleNamespace(token=token, logprobs=mx.array(values, dtype=dtype))


def test_batched_predicate_matches_per_token_check():
    rng = random.Random(0)
    specials = [float("nan"), float("inf"), -float("inf")]
    for _ in range(300):
        outputs = []
        for _ in range(rng.randint(0, 6)):
            vocab = rng.randint(1, 5)
            values = [
                rng.choice(specials) if rng.random() < 0.08 else rng.uniform(-9, 0)
                for _ in range(vocab)
            ]
            token = rng.choice([*range(vocab), vocab, -1, 0xFFFFFFFF])
            dtype = rng.choice([mx.float32, mx.float16, mx.bfloat16])
            outputs.append(_out(token, values, dtype))
        expected = all(
            G._invalid_output_reason(int(o.token), o.logprobs) is None for o in outputs
        )
        assert G._outputs_all_valid(iter(outputs)) is expected


@pytest.fixture
def mtp_engine(monkeypatch):
    H.patch_host(monkeypatch)
    model, vocab = H.tiny_qwen38_mtp()
    engine = H.make_engine(model, vocab, mtp=True, num_draft=2)
    yield engine
    engine.close()


REQUEST = {"tokens": [1, 2, 3, 4, 5], "max_tokens": 12, "temperature": 0}


def test_valid_rounds_skip_per_token_reads_and_fallback_is_identical(
    mtp_engine, monkeypatch
):
    calls = []
    original = G._invalid_output_reason

    def counted(token, logprobs):
        caller = traceback.extract_stack()[-2].name
        if caller not in ("_emit_initial", "_sample_from_logprobs"):
            calls.append(token)  # the per-round emission check
        return original(token, logprobs)

    monkeypatch.setattr(G, "_invalid_output_reason", counted)
    batched = H.run(mtp_engine, REQUEST)
    assert "error" not in batched, batched
    assert len(batched["tokens"]) == 12
    # The initial token and the sampling seam keep their own (existing)
    # reads; the round's emitted tokens no longer pay one each.
    assert calls == []
    # Forcing the per-token fallback gives the same tokens.
    monkeypatch.setattr(G, "_outputs_all_valid", lambda outputs: False)
    calls.clear()
    fallback = H.run(mtp_engine, REQUEST)
    assert fallback["tokens"] == batched["tokens"]
    assert len(calls) >= 10
