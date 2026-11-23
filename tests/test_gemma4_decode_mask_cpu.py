"""Gemma 4 full-attention decode masks under MLX2_DECODE_MASK (recon 2026-11-22).

Stock MLX sends head_dim-512 decode to its fused vector kernels only without
an array mask.  Gemma 4 used to force an array mask on every BatchKVCache
step; now N == 1 lets the cache decide, so ``MLX2_DECODE_MASK=none`` drops
the all-true mask on an unpadded cache.  Default (``array``) is unchanged.
"""

import collections

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm")

import test_gemma4_native_cpu as T  # noqa: E402
from mlx2.adapters.gemma4 import _Gemma4LogitsModel  # noqa: E402
from mlx2.runtime.generate import BatchGenerator  # noqa: E402
from mlx2.runtime.models import cache as C  # noqa: E402
from mlx2.runtime.models import gemma4 as native  # noqa: E402

FULL = T.TEXT["global_head_dim"]


@pytest.fixture(scope="module")
def model():
    return _Gemma4LogitsModel(T._pair({})[1])


def _run(model, prompts, monkeypatch, steps=6):
    seen = collections.Counter()
    orig = native.scaled_dot_product_attention

    def spy(q, k, v, cache, scale, mask, sinks=None):
        if q.shape[-1] == FULL and q.shape[2] == 1:
            seen["none" if mask is None else type(mask).__name__] += 1
        return orig(q, k, v, cache, scale, mask, sinks)

    monkeypatch.setattr(native, "scaled_dot_product_attention", spy)
    gen = BatchGenerator(model, completion_batch_size=len(prompts),
                         prefill_batch_size=len(prompts), prefill_step_size=64)
    uids = gen.insert(prompts, max_tokens=[steps] * len(prompts))
    out = {uid: [] for uid in uids}
    done = set()
    for _ in range(200):
        _, responses = gen.next()
        for r in responses:
            out[r.uid].append(r.token)
            if r.finish_reason:
                done.add(r.uid)
        if done == set(uids):
            break
    gen.close()
    assert done == set(uids)
    return [out[u] for u in uids], seen


def test_default_mode_keeps_the_array_mask(model, monkeypatch):
    monkeypatch.delenv("MLX2_DECODE_MASK", raising=False)
    _, seen = _run(model, [T.PROMPT], monkeypatch)
    assert seen["array"] > 0 and seen["none"] == 0


def test_none_mode_drops_the_unpadded_decode_mask_with_same_tokens(model, monkeypatch):
    monkeypatch.delenv("MLX2_DECODE_MASK", raising=False)
    reference, _ = _run(model, [T.PROMPT], monkeypatch)
    monkeypatch.setenv("MLX2_DECODE_MASK", "none")
    before = C.DECODE_MASK_STATS["maskless_decode_masks"]
    tokens, seen = _run(model, [T.PROMPT], monkeypatch)
    # The one array call left is the prefill tail (the last prefill row's
    # one-row query keeps its slice of the prefill mask), not a decode step.
    assert seen["none"] > 0 and seen["array"] <= 1
    assert C.DECODE_MASK_STATS["maskless_decode_masks"] > before
    assert tokens == reference


def test_none_mode_keeps_the_mask_on_a_padded_batch(model, monkeypatch):
    monkeypatch.setenv("MLX2_DECODE_MASK", "none")
    _, seen = _run(model, [T.PROMPT, T.PROMPT[:7]], monkeypatch)
    assert seen["array"] > 0 and seen["none"] == 0
