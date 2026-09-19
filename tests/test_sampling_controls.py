import mlx.core as mx
import numpy as np
from mlx2.runtime.sample_utils import make_logits_processors, make_transformed_logprobs


def test_penalties_cover_complete_history_and_count_frequency():
    tokens = mx.array([1, 1, 2], mx.uint32)
    logits = mx.array([[0.0, 4.0, -2.0, 3.0]])
    processors = make_logits_processors(
        logit_bias={3: -1}, repetition_penalty=2, repetition_context_size=0,
        presence_penalty=0.5, presence_context_size=0,
        frequency_penalty=0.25, frequency_context_size=0,
    )
    for processor in processors:
        logits = processor(tokens, logits)
    mx.eval(logits)
    np.testing.assert_array_equal(np.array(logits), [[0.0, 1.0, -4.75, 2.0]])


def test_min_p_removes_small_relative_probability_and_normalizes():
    logits = mx.array([[3.0, 2.0, -9.0]])
    result = make_transformed_logprobs(0.7, min_p=0.5)(logits)
    mx.eval(result)
    assert np.isneginf(np.array(result)[0, 2])
    np.testing.assert_allclose(np.exp(np.array(result)).sum(), 1.0, rtol=1e-6)


def test_sub_float32_temperature_is_rejected_and_small_positive_kept():
    import pytest

    from mlx2.server import MIN_POSITIVE_TEMPERATURE, validate_request

    with pytest.raises(ValueError, match="temperature must be 0 or at least"):
        validate_request({"prompt": "hi", "temperature": 1e-40}, chat=False)
    validate_request({"prompt": "hi", "temperature": 0}, chat=False)
    validate_request({"prompt": "hi", "temperature": MIN_POSITIVE_TEMPERATURE}, chat=False)
    # Small but float32-finite reciprocals stay valid.
    validate_request({"prompt": "hi", "temperature": 1e-5}, chat=False)
    validate_request({"prompt": "hi", "temperature": 1e-38}, chat=False)
    assert np.isfinite(np.float32(1.0 / MIN_POSITIVE_TEMPERATURE))
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.float32(1.0 / 1e-40))


def test_top_p_nucleus_is_dtype_independent_on_wide_vocabularies():
    """bf16 cumulative sums over a 248K vocabulary collapsed the nucleus on the
    CPU backend; the mass arithmetic now runs in float32 on every device."""
    from mlx2.runtime.sample_utils import apply_top_p

    vocab = 248320
    rng = np.random.default_rng(0)
    ranks = np.arange(1, vocab + 1)
    for kind in ("floor", "flat"):
        if kind == "floor":
            row = -0.9 * np.log(ranks) + rng.gumbel(0, 0.3, vocab)
        else:
            row = rng.normal(0, 0.5, vocab)
        row = rng.permutation(row).astype(np.float32)
        lp32 = mx.array(row)
        lp32 = lp32 - mx.logsumexp(lp32)
        lp16 = lp32.astype(mx.bfloat16)
        kept32 = np.array(mx.isfinite(apply_top_p(lp32, 0.8)))
        kept16 = np.array(mx.isfinite(apply_top_p(lp16, 0.8)))
        assert kept16.sum() > 0
        # bf16 inputs quantize the logprobs themselves, so allow a small
        # boundary difference, but the nucleus must have the same size class.
        assert abs(int(kept16.sum()) - int(kept32.sum())) < 0.02 * kept32.sum(), (kind, kept16.sum(), kept32.sum())
        assert apply_top_p(lp16, 0.8).dtype == mx.bfloat16
