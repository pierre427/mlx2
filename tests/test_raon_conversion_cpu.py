"""Synthetic CPU fixtures for the two Raon conversion recipes."""
import json

import mlx.core as mx
import pytest

from mlx2.conversion.raon1_mini import convert, convert_affine


def _source(path):
    path.mkdir()
    config = {"model_type": "raon1_mini", "num_experts": 1, "num_hidden_layers": 2,
              "first_k_dense_replace": 1, "moe_layer_freq": 1, "quantization_config": {
        "quant_method": "mxfp8", "weight_block_size": [1, 32]}}
    (path / "config.json").write_text(json.dumps(config))
    tensors = {"model.norm.weight": mx.ones((64,), dtype=mx.bfloat16)}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        prefix = f"model.layers.1.moe.experts.0.{projection}"
        tensors[prefix + ".weight"] = mx.zeros((64, 64), dtype=mx.uint8)
        tensors[prefix + ".weight_scale_inv"] = mx.full((64, 2), 127, dtype=mx.uint8)
    shard = "model.safetensors"
    mx.save_safetensors(str(path / shard), tensors)
    (path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {key: shard for key in tensors}}))


@pytest.mark.parametrize("recipe", [convert, convert_affine])
def test_incomplete_indexed_source_is_refused(tmp_path, recipe):
    source, dest = tmp_path / "source", tmp_path / "dest"
    _source(source)
    index = source / "model.safetensors.index.json"
    doc = json.loads(index.read_text())
    doc["weight_map"]["lm_head.weight"] = "model.safetensors"
    index.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="lm_head.weight"):
        recipe(source, dest)
    assert not dest.exists()


@pytest.mark.parametrize("recipe", [convert, convert_affine])
def test_interrupted_conversion_can_retry_same_destination(tmp_path, recipe, monkeypatch):
    source, dest = tmp_path / "source", tmp_path / "dest"
    _source(source)
    dest.mkdir()
    with monkeypatch.context() as patch:
        def fail_copy(*args):
            raise OSError("copy interrupted")

        (source / "tokenizer.json").write_text("{}")
        patch.setattr("mlx2.conversion.raon1_mini.shutil.copy2", fail_copy)
        with pytest.raises(OSError, match="interrupted"):
            recipe(source, dest)
    assert list(dest.iterdir()) == []
    receipt = recipe(source, dest)
    assert receipt["tensors"] == (7 if recipe is convert else 10)
    assert (dest / "mlx2-conversion-receipt.json").is_file()

@pytest.mark.parametrize("corruption", ["empty", "absent_layer", "missing_plane", "extra_expert", "wrong_grid"])
def test_expert_inventory_must_match_source_topology(tmp_path, corruption):
    source, dest = tmp_path / "source", tmp_path / "dest"
    _source(source)
    path = source / "model.safetensors"
    tensors = mx.load(str(path))
    first = "model.layers.1.moe.experts.0.gate_proj."
    if corruption == "empty":
        tensors = {}
    elif corruption == "absent_layer":
        tensors = {"model.norm.weight": tensors["model.norm.weight"]}
    elif corruption == "missing_plane":
        del tensors[first + "weight_scale_inv"]
    elif corruption == "extra_expert":
        tensors[first.replace("experts.0.", "experts.1.") + "weight"] = tensors[first + "weight"]
    else:
        tensors[first + "weight_scale_inv"] = mx.zeros((1, 2), dtype=mx.uint8)
    if tensors:
        mx.save_safetensors(str(source / "modified.safetensors"), tensors)
    (source / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {key: "modified.safetensors" for key in tensors}}))
    with pytest.raises(ValueError, match="expert|empty"):
        convert(source, dest)
    assert not dest.exists()
