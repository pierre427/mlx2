from types import SimpleNamespace

import mlx.core as mx

from mlx2.runtime.generate import GenerationBatch, StopSequenceMatcher
from mlx2.runtime.pld import PromptLookupBatchGenerator
from mlx2.runtime.sample_utils import make_transformed_logprobs


def _bf16_tie_witness():
    """Two distinct bf16 logits whose bf16-normalized values collapse."""
    return mx.array([[-1.0, -0.99609375] + [-1.0] * 32], dtype=mx.bfloat16)


def _assert_float32_preserves_true_max(logprobs):
    mx.eval(logprobs)
    assert logprobs.dtype == mx.float32
    assert float(logprobs[0, 1]) > float(logprobs[0, 0])
    assert int(mx.argmax(logprobs, axis=-1)[0]) == 1


def test_bf16_witness_collapses_only_when_normalized_in_bf16():
    logits = _bf16_tie_witness()
    native = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(native)
    assert float(native[0, 0]) == float(native[0, 1])
    assert int(mx.argmax(native, axis=-1)[0]) == 0


def test_ordinary_generation_normalizes_bf16_logits_in_float32():
    class Model:
        def __call__(self, inputs, cache):
            del inputs, cache
            return _bf16_tie_witness()[:, None, :]

    batch = GenerationBatch(
        Model(),
        uids=[1],
        inputs=mx.array([7], dtype=mx.uint32),
        prompt_cache=[],
        tokens=[[]],
        samplers=[],
        fallback_sampler=lambda values: mx.argmax(values, axis=-1),
        logits_processors=[],
        stop_matchers=[StopSequenceMatcher()],
        max_tokens=[1],
    )

    _assert_float32_preserves_true_max(batch._next_logprobs[0][None])
    assert int(batch._next_tokens[0]) == 1


def test_prompt_lookup_normalizes_bf16_logits_in_float32():
    generator = PromptLookupBatchGenerator.__new__(PromptLookupBatchGenerator)
    lane = SimpleNamespace(processors=[], lookup_history=[])
    row = generator._processed_row(lane, _bf16_tie_witness()[0], tentative=[])

    _assert_float32_preserves_true_max(row[None])


def test_sampling_transform_normalizes_bf16_logits_in_float32():
    logprobs = make_transformed_logprobs(1.0)(_bf16_tie_witness())

    _assert_float32_preserves_true_max(logprobs)
