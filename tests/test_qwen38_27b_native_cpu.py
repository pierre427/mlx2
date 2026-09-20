"""Tiny native Qwen3.8 27B state checks on the MLX CPU backend.

The production artifact is never opened.  A small topology exercises the
ported dense trunk and embedded MTP head without allocating Metal buffers.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]


def test_tiny_native_trunk_and_mtp_split_replay_on_cpu():
    source = r'''
import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx2.runtime.models.qwen3_5 import TextModelArgs
from mlx2.runtime.models.qwen38_27b import TextModel

args = TextModelArgs(
    model_type="qwen3_5",
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    vocab_size=128,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=3,
    full_attention_interval=4,
    mtp_num_hidden_layers=1,
    partial_rotary_factor=0.5,
    rope_parameters=None,
    max_position_embeddings=128,
)
mx.random.seed(7)
model = TextModel(args)
model.eval()

tokens = mx.array([[1, 2, 3, 4]])
ordinary_full = model(tokens, cache=model.make_cache())
ordinary_cache = model.make_cache()
ordinary_left = model(tokens[:, :2], cache=ordinary_cache)
ordinary_right = model(tokens[:, 2:], cache=ordinary_cache)
ordinary_split = mx.concatenate([ordinary_left, ordinary_right], axis=1)

hidden = model.model(mx.array([[1, 2, 3]]), cache=model.make_cache())
following = mx.array([[2, 3, 4]])
mtp_full_cache = model.make_mtp_cache()
mtp_full_logits, mtp_full_hidden = model.mtp_step(
    hidden, following, mtp_full_cache
)
mtp_split_cache = model.make_mtp_cache()
mtp_left_logits, mtp_left_hidden = model.mtp_step(
    hidden[:, :1], following[:, :1], mtp_split_cache
)
mtp_right_logits, mtp_right_hidden = model.mtp_step(
    hidden[:, 1:], following[:, 1:], mtp_split_cache
)
mtp_split_logits = mx.concatenate([mtp_left_logits, mtp_right_logits], axis=1)
mtp_split_hidden = mx.concatenate([mtp_left_hidden, mtp_right_hidden], axis=1)

mx.eval(
    ordinary_full,
    ordinary_split,
    mtp_full_logits,
    mtp_split_logits,
    mtp_full_hidden,
    mtp_split_hidden,
)
assert ordinary_full.shape == (1, 4, 128)
assert mtp_full_logits.shape == (1, 3, 128)
assert mtp_full_hidden.shape == (1, 3, 32)
assert mtp_full_cache[0].offset == 3
assert mtp_split_cache[0].offset == 3
assert mx.allclose(ordinary_full, ordinary_split, rtol=1e-5, atol=1e-5).item()
assert mx.allclose(mtp_full_logits, mtp_split_logits, rtol=1e-5, atol=1e-5).item()
assert mx.allclose(mtp_full_hidden, mtp_split_hidden, rtol=1e-5, atol=1e-5).item()
'''
    environment = dict(os.environ)
    environment.update(
        {
            "MLX_GDN_PACKED": "0",
            "MLX_GDN_CORE": "0",
            "PYTHONPATH": str(ROOT / "src"),
        }
    )
    subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=environment,
        check=True,
        timeout=30,
    )
