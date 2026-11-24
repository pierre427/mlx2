import json

import mlx.core as mx
import numpy as np
import pytest

from mlx2.conversion import ling32_air as conv


def _fp8_block(weight, block=128):
    """Encode ``weight`` as (fp8 codes, scale_inv) with per-block absmax scaling."""
    rows, cols = weight.shape
    grid = np.zeros((-(-rows // block), -(-cols // block)), np.float32)
    codes = np.zeros(weight.shape, np.float32)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            tile = weight[i * block:(i + 1) * block, j * block:(j + 1) * block]
            scale = max(float(np.abs(tile).max()), 1e-6) / 448.0
            grid[i, j] = scale
            codes[i * block:(i + 1) * block, j * block:(j + 1) * block] = tile / scale
    return mx.to_fp8(mx.array(codes)), mx.array(grid)


def test_block_dequant_matches_reference_with_ragged_edges():
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((200, 300)).astype(np.float32)
    codes, scale = _fp8_block(weight)
    restored = np.array(conv.dequantize_block_fp8(codes, scale).astype(mx.float32))
    assert restored.shape == weight.shape
    # e4m3 keeps ~3 mantissa bits: relative error per element <= 2^-4 of block absmax.
    assert np.max(np.abs(restored - weight)) < 0.1 * np.abs(weight).max()
    with pytest.raises(ValueError):
        conv.dequantize_block_fp8(codes, scale[:1])


def _write_source(path, *, experts=4, hidden=128, inter=128, layers=2):
    rng = np.random.default_rng(1)
    tensors = {}

    def fp8(name, shape):
        codes, scale = _fp8_block(rng.standard_normal(shape).astype(np.float32) * 0.05)
        tensors[name] = codes
        tensors[name + "_scale_inv"] = scale

    for index in range(layers + 1):
        p = f"model.layers.{index}."
        tensors[p + "input_layernorm.weight"] = mx.ones((hidden,), mx.bfloat16)
        fp8(p + "attention.q_proj.weight", (hidden, hidden))
        tensors[p + "attention.b_proj.weight"] = mx.zeros((4, hidden), mx.bfloat16)
        fp8(p + "attention.kv_b_proj.weight", (256, 128))
        tensors[p + "mlp.gate.weight"] = mx.zeros((experts, hidden), mx.bfloat16)
        tensors[p + "mlp.gate.expert_bias"] = mx.zeros((experts,), mx.float32)
        for e in range(experts):
            for proj, shape in (("gate", (inter, hidden)), ("up", (inter, hidden)),
                                ("down", (hidden, inter))):
                fp8(f"{p}mlp.experts.{e}.{proj}_proj.weight", shape)
    tensors["model.word_embeddings.weight"] = mx.zeros((256, hidden), mx.bfloat16)
    tensors["lm_head.weight"] = mx.zeros((256, hidden), mx.bfloat16)
    tensors["model.norm.weight"] = mx.ones((hidden,), mx.bfloat16)
    tensors["model.visual.patch.weight"] = mx.zeros((8, 8), mx.bfloat16)
    shard = path / "model-00001.safetensors"
    mx.save_safetensors(str(shard), tensors)
    # MLX writes fp8 codes as U8; the vendor checkpoint declares F8_E4M3.
    raw = shard.read_bytes()
    size = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + size])
    for name, entry in header.items():
        if name != "__metadata__" and entry["dtype"] == "U8":
            entry["dtype"] = "F8_E4M3"
    encoded = json.dumps(header).encode()
    shard.write_bytes(len(encoded).to_bytes(8, "little") + encoded + raw[8 + size:])
    (path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": dict.fromkeys(tensors, "model-00001.safetensors")}))
    (path / "config.json").write_text(json.dumps({
        "model_type": "bailing_moe_v3_vl",
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
        "text_config": {"num_hidden_layers": layers, "num_nextn_predict_layers": 1,
                        "num_experts": experts, "hidden_size": hidden},
        "vision_config": {"depth": 1},
    }))
    (path / "tokenizer.json").write_text("{}")
    (path / ".download-complete").write_text(conv.SOURCE_REVISION + " done\n")


def test_convert_streams_layers_into_loader_layout(tmp_path):
    source, dest = tmp_path / "src", tmp_path / "out"
    source.mkdir()
    _write_source(source)
    receipt = conv.convert(source, dest, log=lambda *_: None)
    config = json.loads((dest / "config.json").read_text())
    weight_map = json.loads((dest / "model.safetensors.index.json").read_text())["weight_map"]
    assert config["model_type"] == "bailing_hybrid"
    assert config["num_nextn_predict_layers"] == 1
    # Routed experts: pre-stacked 4-bit with a per-module override.
    stacked = "model.layers.1.mlp.switch_mlp.gate_proj"
    assert config["quantization"][stacked] == {"group_size": 64, "bits": 4}
    assert {stacked + s for s in (".weight", ".scales", ".biases")} <= weight_map.keys()
    assert not any(".mlp.experts." in name for name in weight_map)
    # Other FP8 linears are 8-bit; vendor full-precision modules and kv_b keep dtype.
    assert "model.layers.0.attention.q_proj.scales" in weight_map
    assert "model.layers.0.attention.b_proj.scales" not in weight_map
    assert "model.layers.0.attention.kv_b_proj.scales" not in weight_map
    assert "model.layers.0.mlp.gate.scales" not in weight_map
    assert "lm_head.scales" in weight_map and "model.word_embeddings.scales" in weight_map
    # MTP layer renamed to the runtime's mtp module; vision kept out of the text map.
    assert "mtp.layers.0.attention.q_proj.weight" in weight_map
    assert not any(name.startswith("model.layers.2.") for name in weight_map)
    assert config["quantization"]["mtp.layers.0.mlp.switch_mlp.up_proj"] == {"group_size": 64, "bits": 4}
    assert not any(name.startswith("model.visual.") for name in weight_map)
    assert (dest / "vision.safetensors").is_file()
    loaded = mx.load(str(dest / weight_map[stacked + ".weight"]))
    assert loaded[stacked + ".weight"].shape[0] == 4
    assert loaded["model.layers.1.mlp.gate.weight"].dtype == mx.float32
    assert set(receipt["shards"]) == set(weight_map.values())
    with pytest.raises(FileExistsError):
        conv.convert(source, dest)


def test_convert_refuses_incomplete_download(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    _write_source(source)
    (source / ".download-complete").unlink()
    with pytest.raises(ValueError, match="not complete"):
        conv.convert(source, tmp_path / "out")
