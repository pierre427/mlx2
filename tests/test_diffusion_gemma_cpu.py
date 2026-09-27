"""CPU artifact and direct lifecycle contract without importing real MLX."""

import importlib.abc
import json
import sys
import tempfile
import unittest
from pathlib import Path


class BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError(f"real MLX import forbidden: {fullname}")
        return None


sys.meta_path.insert(0, BlockMLX())
from mlx2.adapters.diffusion_gemma import (  # noqa: E402
    DiffusionGemmaAdapter, inspect_diffusion_gemma,
)


def fixture(root):
    layers = ["full_attention" if i % 6 == 5 else "sliding_attention" for i in range(30)]
    config = {
        "model_type": "diffusion_gemma", "architectures": ["DiffusionGemmaForBlockDiffusion"],
        "canvas_length": 256,
        "text_config": {"num_hidden_layers": 30, "hidden_size": 2816,
                        "num_experts": 128, "top_k_experts": 8, "sliding_window": 1024,
                        "vocab_size": 262144, "layer_types": layers},
        "vision_config": {"num_hidden_layers": 27, "hidden_size": 1152,
                          "num_attention_heads": 16, "patch_size": 16},
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "generation_config.json").write_text(json.dumps({"max_denoising_steps": 48}))
    for name in ("tokenizer.json", "tokenizer_config.json", "processor_config.json"):
        (root / name).write_text("{}")
    shapes = {
        "model.decoder.embed_tokens.weight": [262144, 704],
        "model.encoder.embed_vision.embedding_projection.weight": [2816, 144],
        "model.encoder.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight": [1152, 1152],
    }
    for i in range(30):
        for name in ("experts.gate_up_proj", "experts.down_proj", "router.proj"):
            shapes[f"model.decoder.layers.{i}.{name}.weight"] = [1]
    weight_map = {}
    for i in range(4):
        shard = f"model-{i+1:05d}-of-00004.safetensors"
        header = {}
        for n, (name, shape) in enumerate(shapes.items()):
            if n % 4 == i:
                header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]}
                weight_map[name] = shard
        raw = json.dumps(header).encode()
        (root / shard).write_bytes(len(raw).to_bytes(8, "little") + raw)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return config


class DiffusionGemmaCPUTest(unittest.TestCase):
    def test_artifact_and_fail_closed_direct_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            artifact = inspect_diffusion_gemma(root)
            self.assertEqual(artifact["tensor_count"], 93)
            self.assertFalse(artifact["qualified"])
            self.assertFalse(artifact["selected"])
            adapter = DiffusionGemmaAdapter(root)
            with self.assertRaisesRegex(ValueError, "nonempty"):
                adapter.generate_text(" ")
            with self.assertRaisesRegex(ValueError, "denoising"):
                adapter.generate_text("hello", max_denoising_steps=49)
            self.assertNotIn("mlx", sys.modules)

    def test_rejects_changed_topology_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            config["text_config"]["layer_types"][5] = "sliding_attention"
            (root / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "attention pattern"):
                inspect_diffusion_gemma(root)
            config["text_config"]["layer_types"][5] = "full_attention"
            (root / "config.json").write_text(json.dumps(config))
            index = json.loads((root / "model.safetensors.index.json").read_text())
            index["weight_map"].pop("model.decoder.layers.29.router.proj.weight")
            (root / "model.safetensors.index.json").write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "index disagrees"):
                inspect_diffusion_gemma(root)


if __name__ == "__main__":
    unittest.main()
