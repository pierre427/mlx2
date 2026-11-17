"""Laguna XS 2.1: poolside's per-expert bf16 packaging (topology + CPU parity)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from mlx2.adapters.laguna_xs21 import _vendor_required, inspect_artifact
from mlx2.adapters.registry import inspect_model

from test_laguna_xs21_port import laguna_config

ROOT = Path(__file__).resolve().parents[1]
ROPE = {
    "full_attention": {
        "rope_theta": 500000.0, "rope_type": "yarn", "factor": 32.0,
        "original_max_position_embeddings": 8192, "beta_slow": 1.0, "beta_fast": 64.0,
        "attention_factor": 1.3465735902799727, "partial_rotary_factor": 0.5,
    },
    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
}


def vendor_artifact(path, *, drop=(), extra=(), config_update=None):
    config = {**laguna_config(), "rope_parameters": ROPE, **(config_update or {})}
    (path / "config.json").write_text(json.dumps(config))
    shards = [f"model-{i:05d}-of-00002.safetensors" for i in (1, 2)]
    for name in shards:
        (path / name).write_bytes(b"static inspection does not load payload")
    keys = (set(_vendor_required()) - set(drop)) | set(extra)
    weight_map = {key: shards[hash(key) % 2] for key in sorted(keys)}
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return path


def test_vendor_closure_matches_the_poolside_release_inventory():
    keys = _vendor_required()
    # poolside/Laguna-XS-2.1 bf16 index: 30513 tensors.
    assert len(keys) == 30513
    assert sum(".mlp.experts." in k and k.split(".")[5].isdigit() for k in keys) == 39 * 256 * 3


def test_vendor_layout_is_accepted_and_named(tmp_path):
    record = inspect_artifact(vendor_artifact(tmp_path))
    assert record["expert_layout"] == "per-expert-vendor"
    assert record["qualification"] == "pending"
    assert inspect_model(tmp_path).artifact["expert_layout"] == "per-expert-vendor"


@pytest.mark.parametrize("missing", [
    "model.layers.39.mlp.experts.255.down_proj.weight",
    "model.layers.17.mlp.experts.e_score_correction_bias",
    "model.layers.5.mlp.gate.weight",
    "model.layers.2.mlp.shared_expert.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.12.self_attn.g_proj.weight",
    "lm_head.weight",
])
def test_one_missing_vendor_tensor_fails_closed(tmp_path, missing):
    vendor_artifact(tmp_path, drop={missing})
    with pytest.raises(ValueError, match="topology is incomplete"):
        inspect_artifact(tmp_path)


def test_mixed_stacked_and_vendor_layouts_fail_closed(tmp_path):
    vendor_artifact(tmp_path, extra={"model.layers.1.mlp.switch_mlp.gate_proj.weight"})
    with pytest.raises(ValueError, match="mixes stacked and per-expert"):
        inspect_artifact(tmp_path)


def test_quantized_vendor_layout_fails_closed(tmp_path):
    vendor_artifact(tmp_path, config_update={"quantization": {"group_size": 64, "bits": 4}})
    with pytest.raises(ValueError, match="quantized per-expert"):
        inspect_artifact(tmp_path)


def test_stacked_layout_still_named(tmp_path):
    from test_laguna_xs21_port import artifact

    assert inspect_artifact(artifact(tmp_path))["expert_layout"] == "stacked-switch-mlp"


def _parity_module():
    spec = importlib.util.spec_from_file_location(
        "laguna_vendor_slice_parity", ROOT / "scripts" / "laguna_vendor_slice_parity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vendor_layout_forward_matches_transformers_reference():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "LagunaForCausalLM"):
        pytest.skip("transformers has no in-tree Laguna reference")
    parity = _parity_module()
    config = parity.tiny_config({**laguna_config(), "rope_parameters": ROPE,
                                 "rms_norm_eps": 1e-6, "eos_token_id": [2, 24]})
    weights = parity.tiny_weights(config, seed=3)
    rng = np.random.default_rng(4)
    sequences = [[int(t) for t in rng.integers(0, config["vocab_size"], n)] for n in (3, 20)]
    ref = parity.reference_logits(config, weights, sequences)
    got = parity.mlx2_logits(config, weights, sequences)
    for a, b in zip(ref, got):
        assert np.max(np.abs(a - b)) < 1e-4
        assert (a.argmax(-1) == b.argmax(-1)).all()
    # Negative control: a perturbed routed expert must move the mlx2 logits,
    # so the agreement above is not blind to the per-expert stack.
    broken = dict(weights)
    for x in range(config["num_experts"]):
        key = f"model.layers.1.mlp.experts.{x}.down_proj.weight"
        broken[key] = weights[key] * 4.0
    moved = parity.mlx2_logits(config, broken, sequences)
    assert max(float(np.max(np.abs(a - b))) for a, b in zip(ref, moved)) > 1e-2
