"""Released draft metadata and trained visibility; no model/GPU qualification."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.glm53_drafters import Glm53DFlashConfig
from mlx2.runtime.drafters.dflash2_config import DFlash2Config
from mlx2.runtime.drafters.dflash_base import DFlashAttention

ROOT = Path("~/mlx-models")


def test_both_released_heads_keep_their_trained_causal_law():
    for name, causal, layers in [
        ("GLM-5.3-Flash-DFlash", True, [23, 27, 31, 35, 39, 43]),
        ("GLM-5.3-Flash-DFlash2", False, [5, 14, 24, 33, 42]),
    ]:
        path = ROOT / name / "config.json"
        if not path.exists():
            pytest.skip("downloaded metadata unavailable")
        config = Glm53DFlashConfig.from_dict(json.loads(path.read_text()))
        assert config.is_causal is causal and config.target_layer_ids == layers
        if causal:
            with pytest.raises(ValueError, match="causal"):
                DFlash2Config.from_dict(json.loads(path.read_text()))


def test_causal_block_mask_has_prefix_and_sliding_visibility():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        attention = DFlashAttention.__new__(DFlashAttention)
        from mlx import nn

        nn.Module.__init__(attention)
        # nn.Module stores fields in its own dictionary mapping.
        attention.is_causal = True
        attention.is_sliding = True
        attention.sliding_window = 4
        mask = np.array(attention._proposal_mask(3, 4))[0, 0]
        q = 3 + np.arange(4)[:, None]
        k = np.arange(7)[None, :]
        np.testing.assert_array_equal(mask, (k <= q) & (k > q - 4))
        attention.is_causal = False
        assert attention._proposal_mask(3, 4) is None
    finally:
        mx.set_default_device(previous)


def test_causal_attention_cannot_read_later_proposal_tokens():
    from types import SimpleNamespace

    from mlx import nn

    from mlx2.runtime.models.cache import KVCache

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        config = SimpleNamespace(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            layer_types=["full_attention"],
            num_hidden_layers=1,
            rms_norm_eps=1e-5,
            is_causal=True,
        )
        attention = DFlashAttention(config, 0)
        rope = nn.RoPE(4, base=10000)
        rng = np.random.default_rng(212)
        x = mx.array(rng.normal(size=(1, 4, 8)).astype(np.float32))
        ctx = mx.array(rng.normal(size=(1, 2, 8)).astype(np.float32))
        changed = mx.concatenate((x[:, :3], x[:, 3:] + 13), axis=1)
        a = np.array(attention(x, ctx, rope, KVCache()))
        b = np.array(attention(changed, ctx, rope, KVCache()))
        np.testing.assert_array_equal(a[:, :3], b[:, :3])
        assert np.max(np.abs(a[:, 3:] - b[:, 3:])) > 1e-3
        attention.is_causal = False
        a = np.array(attention(x, ctx, rope, KVCache()))
        b = np.array(attention(changed, ctx, rope, KVCache()))
        assert np.max(np.abs(a[:, :1] - b[:, :1])) > 1e-3
    finally:
        mx.set_default_device(previous)


def test_bf16_draft_boundary_avoids_target_fp32_promotion_and_covers_append():
    from types import SimpleNamespace

    from mlx import nn

    from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        config = Glm53DFlashConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=64,
            num_target_layers=4,
            target_layer_ids=[0, 3],
            conv_kernel_size=2,
            conv_group_size=2,
            selector_rank=4,
            selector_top_k=4,
            block_size=4,
            mask_token_id=63,
            max_position_embeddings=256,
            layer_types=["full_attention"],
        )
        draft = DFlash2DraftModel(config)
        draft.apply(lambda value: value.astype(mx.bfloat16))
        target = SimpleNamespace(
            args=SimpleNamespace(hidden_size=16, vocab_size=64, num_hidden_layers=4),
            embed_tokens=nn.Embedding(64, 16),
            lm_head=nn.Linear(16, 64, bias=False),
        )
        draft.bind(target)
        features = mx.ones((1, 2, 32), mx.float32)
        tokens = mx.array([[1, 63, 63, 63]])
        ordinary = draft.make_cache()
        baseline = draft._hidden(tokens, features, ordinary)
        assert baseline.dtype == mx.float32 and ordinary[0].keys.dtype == mx.float32
        object.__setattr__(draft, "target_activation_dtype", mx.bfloat16)
        cache = draft.make_cache()
        actual = draft._hidden(tokens, features, cache)
        mx.eval(actual, cache[0].state)
        assert actual.dtype == mx.bfloat16 and cache[0].keys.dtype == mx.bfloat16
        appended = draft.make_cache()
        draft.append_context(features, appended)
        mx.eval(appended[0].state)
        assert appended[0].keys.dtype == appended[0].values.dtype == mx.bfloat16
        assert (
            features.dtype
            == target.embed_tokens.weight.dtype
            == target.lm_head.weight.dtype
            == mx.float32
        )
        assert draft.fc.weight.dtype == mx.bfloat16
        assert bool(mx.all(mx.isfinite(draft._logits(actual))).item())
    finally:
        mx.set_default_device(previous)
