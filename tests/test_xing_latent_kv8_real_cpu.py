"""Real MLX CPU checks for the experimental Xing latent-only cache format."""

import copy

import mlx.core as mx

from mlx2.runtime.models.cache import load_prompt_cache, save_prompt_cache
from mlx2.runtime.models.xing_latent_kv8 import (
    SegmentedBatchXingLatentKV8Cache,
    XingLatentKV8Cache,
)
from mlx2.runtime.segmented_batch_cache import build_segmented_batch_cache_group


def test_xing_latent_kv8_real_cpu_roundtrip_and_rollback(tmp_path):
    assert mx.default_device() == mx.cpu
    mx.random.seed(74)
    latent = mx.random.normal((1, 1, 5, 512)).astype(mx.bfloat16)
    rope = mx.random.normal((1, 1, 5, 64)).astype(mx.bfloat16)
    cache = XingLatentKV8Cache()
    decoded, saved_rope = cache.update_and_fetch(latent, rope)
    mx.eval(decoded, saved_rope)
    assert cache.offset == 5
    assert saved_rope.dtype == mx.bfloat16
    assert mx.max(mx.abs(decoded.astype(mx.float32) - latent.astype(mx.float32))).item() < 0.1
    assert mx.array_equal(saved_rope, rope).item()

    clone = copy.deepcopy(cache)
    save_prompt_cache(str(tmp_path / "candidate.safetensors"), [cache])
    (restored,) = load_prompt_cache(str(tmp_path / "candidate.safetensors"))
    for candidate in (clone, restored):
        assert type(candidate) is XingLatentKV8Cache
        assert candidate.offset == 5
        assert candidate.trim(2) == 2
        extra = mx.zeros((1, 1, 2, 512), dtype=mx.bfloat16)
        extra_rope = mx.zeros((1, 1, 2, 64), dtype=mx.bfloat16)
        out, keys = candidate.update_and_fetch(extra, extra_rope)
        mx.eval(out, keys)
        assert candidate.offset == 5
        assert mx.array_equal(keys[..., :3, :], rope[..., :3, :]).item()


def test_xing_latent_kv8_segmented_rows_are_private():
    assert mx.default_device() == mx.cpu
    rows = [XingLatentKV8Cache(), XingLatentKV8Cache()]
    for row, length in zip(rows, (3, 4)):
        row.update_and_fetch(
            mx.zeros((1, 1, length, 512), dtype=mx.bfloat16),
            mx.zeros((1, 1, length, 64), dtype=mx.bfloat16),
        )
    (view,) = build_segmented_batch_cache_group([[rows[0]], [rows[1]]])
    assert type(view) is SegmentedBatchXingLatentKV8Cache
    view.prepare(lengths=[2, 1], right_padding=[0, 1])
    view.update_and_fetch(
        mx.ones((2, 1, 2, 512), dtype=mx.bfloat16),
        mx.ones((2, 1, 2, 64), dtype=mx.bfloat16),
    )
    assert [row.offset for row in rows] == [5, 5]
    viewed = list(view.row_views(None))
    assert [int(item[1]) for item in viewed] == [2, 1]
    mx.eval(*[item[2] for item in viewed], *[item[3] for item in viewed])
    view.trim_ragged([2, 1])
    assert [row.offset for row in rows] == [3, 4]


def test_xing_latent_kv8_mask_uses_cache_offset_after_append():
    assert mx.default_device() == mx.cpu
    cache = XingLatentKV8Cache()
    cache.update_and_fetch(
        mx.zeros((1, 1, 5, 512), dtype=mx.bfloat16),
        mx.zeros((1, 1, 5, 64), dtype=mx.bfloat16),
    )
    assert cache.make_mask(1, return_array=False, window_size=None) is None
    mask = cache.make_mask(2, return_array=True, window_size=None)
    mx.eval(mask)
    assert mask.shape == (2, 7)
    assert bool(mask[0, 5].item())
    assert not bool(mask[0, 6].item())
