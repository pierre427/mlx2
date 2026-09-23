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


def test_sub_float32_temperature_is_rejected_and_small_positive_accepted():
    import pytest

    from mlx2.sampling_defaults import SAMPLING_EPS, resolve_sampling
    from mlx2.server import MIN_POSITIVE_TEMPERATURE, validate_request

    with pytest.raises(ValueError, match="temperature must be 0 or at least"):
        validate_request({"prompt": "hi", "temperature": 1e-40}, chat=False)
    validate_request({"prompt": "hi", "temperature": 0}, chat=False)
    validate_request({"prompt": "hi", "temperature": MIN_POSITIVE_TEMPERATURE}, chat=False)
    # Small positive temperatures stay valid requests.
    validate_request({"prompt": "hi", "temperature": 1e-5}, chat=False)
    validate_request({"prompt": "hi", "temperature": 1e-38}, chat=False)
    assert np.isfinite(np.float32(1.0 / MIN_POSITIVE_TEMPERATURE))
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.float32(1.0 / 1e-40))
    # A finite 1/temperature still overflows once it scales a logprob, so
    # every temperature below the sampling epsilon resolves to greedy.
    for tiny in (MIN_POSITIVE_TEMPERATURE, 1e-38, 1e-6):
        effective, record = resolve_sampling({"temperature": tiny}, None, thinking=None)
        assert effective["temperature"] == 0
        assert record["greedy_temperature"] == {"requested": tiny, "epsilon": SAMPLING_EPS}
    effective, record = resolve_sampling({"temperature": SAMPLING_EPS}, None, thinking=None)
    assert effective["temperature"] == SAMPLING_EPS and "greedy_temperature" not in record


def _tiny_temperature_routes():
    from route_harness import (
        make_engine, make_external_engine, tiny_muse_dflash, tiny_qwen38_mtp,
    )

    qwen, qwen_vocab = tiny_qwen38_mtp()
    muse, draft, muse_vocab = tiny_muse_dflash()
    segmented = {"segment_aware_live_tip": True, "segment_aware_cohort_size": 1}
    return {
        "ordinary": lambda: make_engine(qwen, qwen_vocab, mtp=False),
        "native_mtp": lambda: make_engine(qwen, qwen_vocab, mtp=True, extra=segmented),
        "prompt_lookup": lambda: make_engine(qwen, qwen_vocab, mtp=False, prompt_lookup=True),
        "external_draft": lambda: make_external_engine(muse, draft, muse_vocab),
    }


def test_tiny_positive_temperature_is_greedy_on_every_route(monkeypatch):
    # 1/temperature was finite, but scaling a logprob by it overflowed: every
    # route sampled from an all-NaN law.  Ordinary and prompt lookup emitted
    # a constant token, native MTP emitted id 0, and the external route
    # raised "Invalid probability distribution" and killed the worker.
    from route_harness import patch_host, run

    from mlx2.server import MIN_POSITIVE_TEMPERATURE

    patch_host(monkeypatch)
    prompt = [(5 * i + 2) % 100 + 1 for i in range(20)]
    for name, factory in _tiny_temperature_routes().items():
        engine = factory()
        try:
            greedy = run(engine, {"tokens": prompt, "max_tokens": 6, "temperature": 0})
            outputs = [
                run(engine, {"tokens": prompt, "max_tokens": 6, "temperature": temperature, "seed": 1})
                for temperature in (3e-39, MIN_POSITIVE_TEMPERATURE, 1e-6)
            ]
            alive, error = engine.thread.is_alive(), engine.error
        finally:
            engine.close()
        assert alive and error is None, name
        assert len(greedy["tokens"]) == 6, name
        for output in outputs:
            assert output.get("tokens") == greedy["tokens"], (name, output.get("error"))


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


def test_tiny_top_p_keeps_the_most_likely_token_in_every_dtype():
    """mlx-lm#1912: once ``1 - top_p`` rounds to the accumulated mass, the
    absolute threshold masked every token and sampling drew uniform noise.
    bf16 logprobs failed from top_p <= 1e-4, inside the range clients send."""
    from mlx2.runtime.sample_utils import apply_top_p, make_sampler

    rng = np.random.default_rng(0)
    row = mx.array(rng.normal(0, 3, 4096).astype(np.float32))
    row = row - mx.logsumexp(row)
    best = int(mx.argmax(row))
    for dtype in (mx.float32, mx.float16, mx.bfloat16):
        for top_p in (1e-8, 1e-6, 1e-4, 1e-3):
            kept = np.array(mx.isfinite(apply_top_p(row.astype(dtype), top_p)))
            assert kept.sum() >= 1, (dtype, top_p)
            assert kept[best], (dtype, top_p)
    sampler = make_sampler(temp=0.8, top_p=1e-8)
    assert {int(sampler(row[None])[0]) for _ in range(8)} == {best}
    batch = mx.stack([row, row[::-1]]).astype(mx.bfloat16)
    kept = np.array(mx.isfinite(apply_top_p(batch, 1e-6)))
    assert kept.sum(axis=-1).tolist() == [1, 1]
    assert kept[0, best] and kept[1, 4095 - best]
