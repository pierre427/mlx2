"""CPU-only validation for 122B ordinary and embedded candidate contracts."""

import json
import sys
import tempfile
import unittest
from mlx_blocker import install_for_test_case
from pathlib import Path



from mlx2.adapters.qwen35_122b import (  # noqa: E402
    EXPECTED, QWEN35_122B, Qwen35122BA10BAdapter, inspect_artifact,
)
from mlx2.contracts import Capability  # noqa: E402
from mlx2.adapters.qwen35_122b_vision import Qwen35122BVisionCandidate  # noqa: E402


class Qwen35122BCPUTest(unittest.TestCase):
    def setUp(self):
        install_for_test_case(self)

    def make_artifact(self):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name)
        text = dict(EXPECTED)
        text.update(
            layer_types=["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(48)],
            rope_parameters={"type": "default", "rope_theta": 10000000,
                             "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
                             "mrope_interleaved": True},
            mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=False,
        )
        config = {
            "model_type": "qwen3_5_moe", "text_config": text,
            "vision_config": {"depth": 27, "hidden_size": 1152,
                              "num_heads": 16, "out_hidden_size": 3072,
                              "patch_size": 16, "spatial_merge_size": 2,
                              "temporal_patch_size": 2},
            "quantization": {"bits": 4, "group_size": 64, "mode": "affine",
                             "language_model.model.layers.0.linear_attn.out_proj":
                             {"bits": 5, "group_size": 128, "mode": "affine"}},
        }
        (path / "config.json").write_text(json.dumps(config))
        keys = {
            "language_model.model.embed_tokens.weight",
            "language_model.model.norm.weight", "language_model.lm_head.weight",
            "vision_tower.blocks.0.attn.proj.weight",
            "vision_tower.blocks.0.attn.qkv.weight",
            "vision_tower.blocks.26.attn.qkv.weight",
        }
        keys.update({
            "language_model.mtp.fc.weight", "language_model.mtp.norm.weight",
            "language_model.mtp.pre_fc_norm_embedding.weight",
            "language_model.mtp.pre_fc_norm_hidden.weight",
            "language_model.mtp.layers.0.self_attn.q_proj.weight",
            "language_model.mtp.layers.0.self_attn.k_proj.weight",
            "language_model.mtp.layers.0.self_attn.v_proj.weight",
            "language_model.mtp.layers.0.self_attn.o_proj.weight",
            "language_model.mtp.layers.0.mlp.gate.weight",
            "language_model.mtp.layers.0.mlp.switch_mlp.down_proj.weight",
        })
        for i in range(48):
            stem = f"language_model.model.layers.{i}."
            keys.update({
                stem + "input_layernorm.weight", stem + "post_attention_layernorm.weight",
                stem + "mlp.gate.weight", stem + "mlp.switch_mlp.down_proj.weight",
                stem + "mlp.shared_expert.gate_proj.weight",
                stem + ("self_attn.q_proj.weight" if (i + 1) % 4 == 0 else "linear_attn.in_proj_qkv.weight"),
            })
        (path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {key: "model.safetensors" for key in keys}
        }))
        (path / "model.safetensors").write_bytes(b"test-shard")
        return tmp, path, config

    def test_ordinary_only_and_sidecars_have_candidate_loaders(self):
        tmp, path, _ = self.make_artifact()
        with tmp:
            artifact = inspect_artifact(path)
            self.assertFalse(artifact["has_mtp"])
            self.assertEqual(artifact["embedded_mtp_tensor_count"], 10)
            self.assertEqual(artifact["embedded_vision_tensor_count"], 3)
            self.assertIn(Capability.APC_V2, QWEN35_122B.capabilities)
            self.assertNotIn(Capability.MTP, QWEN35_122B.capabilities)
            self.assertEqual(Qwen35122BA10BAdapter.default_route, "ordinary")
            vision = Qwen35122BVisionCandidate(path)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                vision.generate_text("describe")
            with self.assertRaisesRegex(ValueError, "media file is missing"):
                vision.generate_text("describe", image=path / "missing.png")
            self.assertNotIn("mlx", sys.modules)

    def test_rejects_shape_and_missing_shard(self):
        tmp, path, config = self.make_artifact()
        with tmp:
            config["text_config"]["hidden_size"] = 2048
            (path / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "topology"):
                inspect_artifact(path)
            config["text_config"]["hidden_size"] = 3072
            (path / "config.json").write_text(json.dumps(config))
            (path / "model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "missing weight shard"):
                inspect_artifact(path)


if __name__ == "__main__":
    unittest.main()
