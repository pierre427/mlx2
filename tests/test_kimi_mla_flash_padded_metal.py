"""Metal oracle: Kimi latent flash over left-padded batch-cache lanes.

Opt in with MLX2_RUN_METAL_TESTS=1 on an Apple M5 (the NAX kernels' gate).
"""
import os

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import kimi_linear_mla_flash as fl

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="explicit Metal oracle (MLX2_RUN_METAL_TESTS=1)",
)


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    if not fl.available("absorbed"):
        mx.set_default_device(previous)
        pytest.skip("latent flash unavailable on this device")
    yield
    mx.set_default_device(previous)


def _operands(b, rows, width, seed, cap=None):
    rng = np.random.default_rng(seed)
    cap = width if cap is None else cap

    def arr(shape, std):
        return mx.array(rng.normal(0, std, shape).astype(np.float32)).astype(mx.bfloat16)
    return (arr((b, fl.HEADS, rows, fl.LATENT), .05), arr((b, fl.HEADS, rows, fl.SHARED), .5),
            arr((b, 1, cap, fl.LATENT), 1), arr((b, 1, cap, fl.SHARED), 1))


def _np(x):
    return np.array(x.astype(mx.float32))


def _rel(ref, got):
    ref, got = _np(ref), _np(got)
    return float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-20))


@pytest.mark.parametrize("rows", [1, 4, 8])
@pytest.mark.parametrize("width,pads,split", [(300, [0, 37, 299], None), (5000, [0, 2049, 4100, 4999], None),
                                               (5000, [0, 2049, 4100, 4999], 2048), (64, [0, 64], None)])
def test_left_padded_lanes_match_the_reference(rows, width, pads, split):
    q, qs, lat, sh = _operands(len(pads), rows, width, rows + width)
    starts = mx.array([min(p, width) for p in pads], mx.int32)
    scale = (fl.NOPE + fl.SHARED) ** -0.5
    got = fl.absorbed(q, qs, lat, sh, "causal", scale=scale, starts=starts, split=split)
    ref = fl.absorbed_reference(q, qs, lat, sh, "causal", scale=scale, starts=starts)
    mx.eval(got, ref)
    assert np.isfinite(_np(got)).all()
    assert _rel(ref, got) < 2e-2
    # Rows with no visible key are exactly zero, as in the latent law.
    hidden = _np(ref) == 0
    assert not np.any(_np(got)[hidden.all(-1)])


def test_zero_starts_are_bit_identical_to_no_starts():
    q, qs, lat, sh = _operands(3, 4, 3000, 7)
    scale = (fl.NOPE + fl.SHARED) ** -0.5
    a = fl.absorbed(q, qs, lat, sh, "causal", scale=scale)
    b = fl.absorbed(q, qs, lat, sh, "causal", scale=scale, starts=mx.zeros((3,), mx.int32))
    assert np.array_equal(_np(a), _np(b))


def test_padded_lanes_in_place_buffers_match_sliced_operands():
    width, cap = 1500, 2048
    q, qs, lat, sh = _operands(2, 1, width, 11, cap=cap)
    starts = mx.array([0, 700], mx.int32)
    scale = (fl.NOPE + fl.SHARED) ** -0.5
    sliced = fl.absorbed(q, qs, lat[:, :, :width], sh[:, :, :width], "causal", scale=scale, starts=starts)
    inplace = fl.absorbed(q, qs, lat, sh, "causal", scale=scale, starts=starts, width=width)
    assert np.array_equal(_np(sliced), _np(inplace))


def test_padded_batch_decode_runs_latent_flash_and_matches_the_latent_law():
    import json
    from pathlib import Path

    from mlx2.runtime.models.cache import BatchKVCache
    from mlx2.runtime.models.kimi_linear import (
        LeftPaddedCausal,
        ModelArgs,
        MultiLatentAttention,
    )

    config = Path.home() / "Models/Kimi-Linear-48B-A3B-Instruct-8bit/config.json"
    if not config.is_file():
        pytest.skip("48B config not on disk")
    args = ModelArgs.from_dict(json.loads(config.read_text()))
    mx.random.seed(1103)
    attn = MultiLatentAttention(args)
    attn.eval()
    attn.set_dtype(mx.bfloat16)
    attn.policy, attn.flash = "auto", "all"
    pads = [0, 300, 650]
    prompt = (mx.random.normal((3, 700, args.hidden_size)) * .5).astype(mx.bfloat16)
    step = (mx.random.normal((3, 1, args.hidden_size)) * .5).astype(mx.bfloat16)
    outs = {}
    for name in ("flash", "latent_single"):
        cache = BatchKVCache(list(pads))
        attn.policy = "latent_single" if name == "latent_single" else "auto"
        mask = cache.make_mask(700)
        attn(prompt, mask=mask, cache=cache)
        mask = cache.make_mask(1)
        before = dict(attn.calls)
        outs[name] = attn(step, mask=LeftPaddedCausal(mask, cache.left_padding), cache=cache)
        mx.eval(outs[name])
        delta = {k: v - before.get(k, 0) for k, v in attn.calls.items() if v != before.get(k, 0)}
        if name == "flash":
            assert delta.get("latent_flash_padded") == 1 and delta.get("latent_flash") == 1
            assert not delta.get("flash_declined")
        else:
            assert not delta.get("latent_flash")
    assert _rel(outs["latent_single"], outs["flash"]) < 2e-2
