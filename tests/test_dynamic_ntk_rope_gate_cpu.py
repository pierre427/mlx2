"""Dynamic NTK RoPE cannot run on the batched serving cache; refuse it at preflight.

``DynamicNTKScalingRoPE`` computes one scalar base per call and fails closed on
a per-row offset array.  Every served request prefills and decodes through the
merged ``BatchKVCache`` (array offsets), even at one lane, so an artifact that
declares ``rope_scaling.type == "dynamic"`` would pass inspection, report
ready, and then kill the generation worker on its first request.
"""

import json
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2.adapters.standard_decoder import inspect_artifact

_BASE = {
    "hidden_size": 128,
    "num_hidden_layers": 2,
    "intermediate_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 256,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
}
_FAMILY = {
    "llama": {},
    "qwen2": {},
    "qwen3": {"head_dim": 32},
    "qwen3_moe": {
        "head_dim": 32,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "norm_topk_prob": True,
    },
}


def _artifact(tmp_path: Path, family: str, rope_scaling) -> Path:
    config = {**_BASE, **_FAMILY[family], "model_type": family}
    if rope_scaling is not None:
        config["rope_scaling"] = rope_scaling
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"test-shard")
    return tmp_path


@pytest.mark.parametrize("family", sorted(_FAMILY))
@pytest.mark.parametrize("key", ["type", "rope_type"])
def test_dynamic_ntk_rope_is_refused_before_load(tmp_path, family, key):
    path = _artifact(tmp_path, family, {key: "dynamic", "factor": 2.0})
    with pytest.raises(ValueError, match=r"(?i)rope"):
        inspect_artifact(path, expected=family)


def test_batched_cache_still_rejects_dynamic_ntk():
    """Pin the reason for the gate: relax it only once this decodes."""
    from mlx2.runtime.generate import _merge_caches
    from mlx2.runtime.models.standard_decoder import Model, ModelArgs

    model = Model(
        ModelArgs(
            model_type="llama", hidden_size=32, num_hidden_layers=1,
            intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-5,
            vocab_size=64, num_key_value_heads=2, max_position_embeddings=64,
            rope_theta=10000.0, tie_word_embeddings=True,
            rope_scaling={"type": "dynamic", "factor": 2.0},
        )
    )
    batch = _merge_caches([model.make_cache()])
    with pytest.raises(ValueError, match="scalar integer offset"):
        mx.eval(model(mx.array([[1, 2, 3]]), batch))


def test_supported_scaling_still_admitted(tmp_path):
    path = _artifact(tmp_path, "llama", {"rope_type": "linear", "factor": 2.0})
    assert inspect_artifact(path, expected="llama")["config"]["rope_scaling"]["rope_type"] == "linear"
