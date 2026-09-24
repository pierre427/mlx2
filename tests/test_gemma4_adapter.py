"""GPU-free dispatch and policy gates for the two Gemma 4 layouts."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.adapters.gemma4 import (
    GEMMA4_31B,
    GEMMA4_A4B,
    Gemma4A4BAdapter,
    Gemma431BAdapter,
    _Gemma4LogitsModel,
    inspect_gemma4_artifact,
)
from mlx2.adapters.mlx_vlm import _media_token_end
from mlx2.adapters.registry import inspect_model, resolve_adapter
from mlx2.contracts import Capability


def _artifact(path, *, sparse):
    layers = 30 if sparse else 60
    config = {
        "model_type": "gemma4",
        "video_token_id": 258884,
        "audio_config": None,
        "vision_config": {"model_type": "gemma4_vision"},
        "text_config": {
            "num_hidden_layers": layers,
            "hidden_size": 2816 if sparse else 5376,
            "enable_moe_block": sparse,
            "num_experts": 128 if sparse else None,
            "sliding_window": 1024,
            "max_position_embeddings": 262144,
            "layer_types": [
                "full_attention" if index % 6 == 5 else "sliding_attention"
                for index in range(layers)
            ],
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors").write_bytes(b"metadata-only fixture")
    return config


@pytest.mark.parametrize("sparse,expected,adapter_name", [
    (True, GEMMA4_A4B, "Gemma4A4BAdapter"),
    (False, GEMMA4_31B, "Gemma431BAdapter"),
])
def test_exact_topologies_resolve_without_loading_weights(tmp_path, sparse, expected, adapter_name):
    _artifact(tmp_path, sparse=sparse)
    result = inspect_model(tmp_path)
    assert result.adapter_type.__name__ == adapter_name
    assert result.descriptor == expected
    assert result.artifact["max_context"] == 262144
    assert result.artifact["full_attention_layers"] == (5 if sparse else 10)
    assert result.default_route == "ordinary"
    assert Capability.VISION in expected.capabilities
    assert Capability.VIDEO in expected.capabilities
    assert Capability.AUDIO not in expected.capabilities
    with pytest.raises(ValueError, match="native MTP"):
        resolve_adapter(tmp_path, mtp=True)


def test_mismatched_model_or_quantization_fails_before_load(tmp_path):
    config = _artifact(tmp_path, sparse=True)
    config["text_config"]["num_experts"] = 64
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="topology"):
        inspect_gemma4_artifact(tmp_path)
    config["text_config"]["num_experts"] = 128
    config["audio_config"] = {"hidden_size": 1024}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="without an audio tower"):
        inspect_gemma4_artifact(tmp_path)
    config["audio_config"] = None
    config["quantization"] = {"bits": 4, "group_size": 64, "mode": "affine"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="affine 8-bit"):
        inspect_gemma4_artifact(tmp_path)


def test_conversion_provenance_must_match_resolved_variant(tmp_path):
    config = _artifact(tmp_path, sparse=False)
    config["quantization"] = {"bits": 8, "group_size": 64, "mode": "affine"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "source-and-quantization.json").write_text(json.dumps({
        "source_repo": "google/gemma-4-26B-A4B",
        "source_revision": "24548b62aa021d562695c04aaf7758a1ea47990b",
    }))
    with pytest.raises(ValueError, match="conversion source"):
        inspect_gemma4_artifact(tmp_path)


def test_mixed_attention_uses_mlx2_cache_types_and_media_boundary():
    from mlx2.runtime.models.cache import KVCache, RotatingKVCache

    text = SimpleNamespace(
        sliding_window=1024,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
    )
    model = SimpleNamespace(
        config=SimpleNamespace(text_config=text),
        language_model=SimpleNamespace(model=SimpleNamespace(first_kv_shared_layer_idx=3)),
    )
    cache = _Gemma4LogitsModel(model).make_cache()
    assert [type(item) for item in cache] == [RotatingKVCache, KVCache, RotatingKVCache]
    assert cache[0].max_size == cache[2].max_size == 1024
    assert _media_token_end({"mm_token_type_ids": np.array([[0, 1, 2, 0]])}) == 3


def test_both_variants_use_the_pinned_vendor_sampling_defaults():
    for adapter in (Gemma4A4BAdapter, Gemma431BAdapter):
        defaults = adapter.sampling_defaults.profiles["general"]
        assert defaults.values() == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
        assert defaults.source == "generation_config.json"
