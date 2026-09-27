"""CPU-only ordinary-decoder artifact and capability checks."""

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
from mlx2.adapters.standard_decoder import (  # noqa: E402
    StandardDecoderAdapter, descriptor_for, inspect_artifact,
)
from mlx2.contracts import Capability  # noqa: E402


BASE = {
    "hidden_size": 1024, "num_hidden_layers": 2, "intermediate_size": 2048,
    "num_attention_heads": 8, "num_key_value_heads": 2,
    "vocab_size": 4096, "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
    "tie_word_embeddings": True,
}


class StandardDecoderCPUTest(unittest.TestCase):
    def make_artifact(self, family):
        temp = tempfile.TemporaryDirectory()
        path = Path(temp.name)
        config = dict(BASE, model_type=family)
        if family in {"qwen3", "qwen3_moe"}:
            config["head_dim"] = 128
        if family == "qwen3_moe":
            config.update(
                num_experts=4, num_experts_per_tok=2, moe_intermediate_size=512,
                decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True,
            )
        (path / "config.json").write_text(json.dumps(config))
        weight_map = {
            "model.embed_tokens.weight": "model.safetensors",
            "model.norm.weight": "model.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
        }
        if family == "qwen3_moe":
            weight_map["model.layers.0.mlp.gate.weight"] = "model.safetensors"
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        (path / "model.safetensors").write_bytes(b"test-shard")
        return temp, path, config

    def test_all_four_are_ordinary_and_unqualified(self):
        for family in ("qwen3", "qwen3_moe", "qwen2", "llama"):
            with self.subTest(family=family):
                temp, path, _ = self.make_artifact(family)
                with temp:
                    artifact = inspect_artifact(path, expected=family)
                    descriptor = descriptor_for(family)
                    self.assertEqual(artifact["qualification"], "pending")
                    self.assertEqual(artifact["has_mtp"], False)
                    self.assertEqual(descriptor.model_type, family)
                    self.assertIn(Capability.APC_V2, descriptor.capabilities)
                    self.assertNotIn(Capability.MTP, descriptor.capabilities)
                    self.assertEqual(StandardDecoderAdapter.default_route, "ordinary")
                    self.assertNotIn("mlx", sys.modules)

    def test_rejects_mismatched_family_and_missing_shard(self):
        temp, path, _ = self.make_artifact("qwen3")
        with temp:
            with self.assertRaises(ValueError):
                inspect_artifact(path, expected="llama")
            (path / "model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "missing or empty"):
                inspect_artifact(path)

    def test_rejects_unsupported_topology_and_speculative_head(self):
        temp, path, config = self.make_artifact("llama")
        with temp:
            config["sliding_window"] = 64
            (path / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "sliding Llama"):
                inspect_artifact(path)
            config.pop("sliding_window")
            (path / "config.json").write_text(json.dumps(config))
            index_path = path / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            index["weight_map"]["mtp.head.weight"] = "model.safetensors"
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "speculative head"):
                inspect_artifact(path)


if __name__ == "__main__":
    unittest.main()
