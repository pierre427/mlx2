"""Self-MTP prefill follows MLX2_PREFILL_CLEAR_CACHE like the ordinary path.

Both self-MTP prefill entry points used to call ``mx.clear_cache()`` after
every slice unconditionally, so ``idle``/``never`` had no effect on MTP routes
and each slice returned live decode lanes' buffers to Metal.
"""

from unittest.mock import patch

import mlx.core as mx

from mlx2.runtime import hybrid_speculative as hs
from test_batched_mtp import _tiny_qwen4_model

PROMPT = [1, 7, 3, 9, 2, 8, 4, 6, 5]
COMMON = {
    "uid": 11,
    "max_tokens": 8,
    "lane_rng": None,
    "num_draft": 2,
    "sampling_temp": 0.0,
    "sampling_top_p": 1.0,
    "sampling_top_k": 0,
    "sampling_min_p": 0.0,
    "accept_rule": "residual",
    "logits_processors": [],
    "prefill_step_size": 4,
    "share_qsa_indices": False,
}


def _advance(model, clear_cache):
    with patch.object(hs.mx, "clear_cache") as clear:
        kwargs = {} if clear_cache is None else {"clear_cache": clear_cache}
        remaining, _, _, processed = hs.advance_self_mtp_prefill(
            mx.array(PROMPT, mx.uint32),
            model,
            prompt_cache=None,
            mtp_state=None,
            max_tokens=4,
            **kwargs,
        )
    return clear.call_count, remaining.tolist(), processed


def _prepare(model, clear_cache):
    with patch.object(hs.mx, "clear_cache") as clear:
        kwargs = {} if clear_cache is None else {"clear_cache": clear_cache}
        _, first = hs.prepare_self_mtp_lane(
            mx.array(PROMPT, mx.uint32),
            model,
            prompt_cache=None,
            mtp_state=None,
            **COMMON,
            **kwargs,
        )
    return clear.call_count, first


def test_advance_slice_clears_by_default_and_skips_when_told():
    mx.random.seed(71)
    model = _tiny_qwen4_model()
    default_calls, default_rest, n = _advance(model, None)
    kept_calls, kept_rest, m = _advance(model, False)
    assert default_calls == 1
    assert kept_calls == 0
    assert (default_rest, n) == (kept_rest, m)


def test_prepare_lane_clears_by_default_and_skips_when_told():
    mx.random.seed(71)
    model = _tiny_qwen4_model()
    default_calls, default_first = _prepare(model, None)
    kept_calls, kept_first = _prepare(model, False)
    assert default_calls >= 1
    assert kept_calls == 0
    assert default_first.token == kept_first.token
    assert mx.array_equal(default_first.logprobs, kept_first.logprobs).item()


def test_batch_generator_passes_the_prefill_clear_cache_decision():
    import inspect

    from mlx2.runtime import generate

    source = inspect.getsource(generate.BatchGenerator)
    assert source.count("clear_cache=prefill_clear_cache(self._has_active_decode())") == 2
