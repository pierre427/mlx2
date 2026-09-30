"""YaRN honours ``truncate`` like transformers and OpenAI's gpt-oss reference.

gpt-oss configs set ``truncate: false``: the correction range stays
unrounded.  mlx2 always rounded it, so 7 of gpt-oss's 32 frequency bands were
off by up to 76% (0.9 rad of rotation at position 4096).
"""

import numpy as np
import pytest

import mlx.core as mx

from mlx2.runtime.models.rope_utils import initialize_rope

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

GPT_OSS_ROPE = {
    "rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0,
    "original_max_position_embeddings": 4096, "truncate": False,
}


def _hf_inv_freq(rope):
    from transformers import GptOssConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    config = GptOssConfig(head_dim=64, rope_theta=150000, max_position_embeddings=131072,
                          rope_scaling=dict(rope))
    inv, attention = ROPE_INIT_FUNCTIONS["yarn"](config, "cpu")
    return inv.numpy(), attention


@pytest.mark.parametrize("truncate", [False, True])
def test_yarn_frequencies_match_transformers(truncate):
    mx.set_default_device(mx.cpu)
    rope = {**GPT_OSS_ROPE, "truncate": truncate}
    ours = initialize_rope(64, 150000, False, dict(rope), 131072)
    hf_inv, hf_attention = _hf_inv_freq(rope)
    np.testing.assert_allclose(1.0 / np.array(ours._freqs), hf_inv, rtol=1e-5)
    assert abs(ours.mscale - hf_attention) < 1e-6


def test_absent_truncate_keeps_the_rounded_range():
    mx.set_default_device(mx.cpu)
    rope = {k: v for k, v in GPT_OSS_ROPE.items() if k != "truncate"}
    explicit = initialize_rope(64, 150000, False, {**rope, "truncate": True}, 131072)
    default = initialize_rope(64, 150000, False, dict(rope), 131072)
    assert np.array_equal(np.array(default._freqs), np.array(explicit._freqs))
