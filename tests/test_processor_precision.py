"""Penalties and logit_bias must be applied in float32, not the model's bf16.

A bf16 logit at |x| in [16, 32) has a 0.125 rounding step, so a 0.1 penalty
applied in bf16 rounds to 0 or 0.125 and a repetition factor below 1.0039
rounds to 1.0 (a no-op).  HF and vLLM upcast before processors; the OpenAI
penalty parameters are defined on exact logit arithmetic.  Every route that
builds a target law from processed logits must do the same, so the routes
stay law-identical with each other and with the requested values.
"""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.external_speculative import Lane, TreeDraftRow
from mlx2.runtime.generate import GenerationBatch, StopSequenceMatcher
from mlx2.runtime.hybrid_speculative import (
    _apply_logits_processors,
    _probe_logits_processors,
)
from mlx2.runtime.paged_native_continuation import NativeQwen3Continuation
from mlx2.runtime.pld import PromptLookupBatchGenerator
from mlx2.runtime.sample_utils import make_logits_processors
from mlx2.runtime.speculative_sampling import RequestRNG

from test_dflash_pair_select import generator, tiny

# Token 1 leads by one bf16 step; a penalty smaller than that step must keep
# it in front (fp32: 30.15 > 30.125), not round it into a tie that argmax
# resolves to token 0.
_VOCAB = 64
_RAW = [30.125, 30.25] + [10.0] * (_VOCAB - 2)
_HISTORY = [5, 1]

_CASES = {
    "logit_bias": dict(logit_bias={1: -0.1}),
    "presence": dict(presence_penalty=0.1, presence_context_size=0),
    "frequency": dict(frequency_penalty=0.1, frequency_context_size=0),
    "repetition": dict(repetition_penalty=1.004, repetition_context_size=0),
}


def _bf16_logits():
    return mx.array([_RAW], dtype=mx.bfloat16)


def _reference_logprobs(kwargs):
    values = _bf16_logits().astype(mx.float32)
    for processor in make_logits_processors(**kwargs):
        values = processor(mx.array(_HISTORY), values)
    return values - mx.logsumexp(values, axis=-1, keepdims=True)


def _assert_matches_reference(logprobs, kwargs):
    reference = _reference_logprobs(kwargs)
    mx.eval(logprobs, reference)
    assert int(mx.argmax(logprobs, axis=-1).reshape(-1)[0]) == 1
    margin = float(logprobs.reshape(-1)[1] - logprobs.reshape(-1)[0])
    expected = float(reference[0, 1] - reference[0, 0])
    assert margin == pytest.approx(expected, abs=1e-4)


def _logprobs(values):
    values = values.astype(mx.float32)
    return values - mx.logsumexp(values, axis=-1, keepdims=True)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_ordinary_decode_applies_processors_in_float32(name):
    kwargs = _CASES[name]

    class Model:
        def __call__(self, inputs, cache):
            del inputs, cache
            return _bf16_logits()[:, None, :]

    batch = GenerationBatch(
        Model(),
        uids=[1],
        inputs=mx.array([_HISTORY[-1]], dtype=mx.uint32),
        prompt_cache=[],
        tokens=[_HISTORY[:-1]],
        samplers=[],
        fallback_sampler=lambda values: mx.argmax(values, axis=-1),
        logits_processors=[make_logits_processors(**kwargs)],
        stop_matchers=[StopSequenceMatcher()],
        max_tokens=[1],
    )

    _assert_matches_reference(batch._next_logprobs[0][None], kwargs)
    assert int(batch._next_tokens[0]) == 1


@pytest.mark.parametrize("name", sorted(_CASES))
def test_prompt_lookup_applies_processors_in_float32(name):
    kwargs = _CASES[name]
    generator_ = PromptLookupBatchGenerator.__new__(PromptLookupBatchGenerator)
    lane = SimpleNamespace(processors=make_logits_processors(**kwargs))
    row = generator_._processed_row(lane, _bf16_logits()[0], mx.array(_HISTORY))

    _assert_matches_reference(row[None], kwargs)


@pytest.mark.parametrize("apply", [_apply_logits_processors, _probe_logits_processors])
@pytest.mark.parametrize("name", sorted(_CASES))
def test_self_mtp_verify_applies_processors_in_float32(name, apply):
    kwargs = _CASES[name]
    values = apply(
        make_logits_processors(**kwargs), mx.array(_HISTORY), _bf16_logits()[0]
    )

    _assert_matches_reference(_logprobs(values)[None], kwargs)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_native_continuation_applies_processors_in_float32(name):
    kwargs = _CASES[name]
    lane = SimpleNamespace(
        processors=make_logits_processors(**kwargs),
        tokens=list(_HISTORY),
        sampler=lambda values: mx.argmax(values, axis=-1),
    )
    sampled, logprobs = NativeQwen3Continuation._stage_sample(lane, _bf16_logits()[0])

    _assert_matches_reference(logprobs, kwargs)
    assert int(sampled[0]) == 1


@pytest.mark.parametrize("name", sorted(_CASES))
def test_external_target_law_applies_processors_in_float32(name):
    kwargs = _CASES[name]
    m, d = tiny(vocab=_VOCAB)
    b = generator(m, d)
    lane = SimpleNamespace(
        processors=make_logits_processors(**kwargs),
        sampling={"sampling_temp": 0.0},
    )
    rows = []
    token = b._target_law(
        lane, _bf16_logits()[0], list(_HISTORY), response_rows=rows,
        greedy_token=True,
    )

    assert token == 1
    _assert_matches_reference(rows[0][None], kwargs)


@pytest.mark.parametrize("batched", [False, True])
def test_external_tree_laws_apply_presence_in_float32(monkeypatch, batched):
    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    m, d = tiny(vocab=_VOCAB, top_k=16, block_size=8)
    b = generator(m, d)
    block = TreeDraftRow([1] * 15, list(range(-1, 14)))
    logits = mx.array([[_RAW] * 16], dtype=mx.bfloat16)
    lane = Lane(
        uid=0, history=list(_HISTORY), remaining=[], cache=[], draft_cache=[],
        tail=None, rng=RequestRNG(3), maximum=64,
        processors=make_logits_processors(presence_penalty=0.1, presence_context_size=0),
        sampling={"sampling_temp": 0.0, "emit_logprobs": True},
        anchor=_HISTORY[-1],
    )
    if batched:
        fence, state = b._launch_tree_laws(lane, block, logits)
        mx.eval(*fence)
        decision = b._verify_tree_batched(lane, block, state)
    else:
        decision = b._verify_tree(lane, block, logits)

    # Every row penalizes the already-present token 1 by 0.1 < one bf16 step,
    # so it stays the greedy pick and the whole chain of 1s is accepted.
    assert decision.emitted == [1] * 16
    first = np.asarray(decision.response_logprobs[0].astype(mx.float32))
    assert float(first[1] - first[0]) == pytest.approx(0.025, abs=1e-4)
