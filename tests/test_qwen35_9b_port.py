"""CPU-only Qwen3.5 9B adapter checks."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from mlx2.adapters.qwen35_9b import Qwen359BAdapter, descriptor_for, inspect_artifact
from mlx2.adapters.registry import inspect_model
from mlx2.contracts import Capability


CONFIG = {
    "model_type": "qwen3_5",
    "text_config": {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "full_attention_interval": 4,
        "vocab_size": 248320,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "mtp_num_hidden_layers": 1,
        "max_position_embeddings": 262144,
    },
}


class Qwen359BPortTests(unittest.TestCase):
    def artifact(self, root: Path):
        config = copy.deepcopy(CONFIG)
        config["text_config"]["layer_types"] = [
            "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
            for index in range(32)
        ]
        (root / "config.json").write_text(json.dumps(config))
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "language_model.model.embed_tokens.weight": "model.safetensors",
                        "vision_tower.blocks.0.weight": "model.safetensors",
                    }
                }
            )
        )
        (root / "model.safetensors").write_bytes(b"metadata-only")
        return root

    def test_exact_topology_dispatches_to_ordinary_text_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            resolution = inspect_model(self.artifact(Path(directory)))
        self.assertIs(resolution.adapter_type, Qwen359BAdapter)
        self.assertEqual(resolution.default_route, "ordinary")
        self.assertFalse(resolution.artifact["has_mtp"])
        self.assertNotIn(Capability.MTP, resolution.descriptor.capabilities)
        self.assertNotIn(Capability.VISION, resolution.descriptor.capabilities)

    def test_config_mtp_claim_does_not_confer_capability(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = inspect_artifact(self.artifact(Path(directory)))
        self.assertEqual(artifact["advertised_mtp_layers"], 1)
        self.assertEqual(artifact["mtp_tensor_count"], 0)
        with self.assertRaisesRegex(ValueError, "not implemented"):
            descriptor_for(has_mtp=True)

    def test_wrong_topology_and_path_escape_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.artifact(Path(directory))
            config = json.loads((root / "config.json").read_text())
            config["text_config"]["hidden_size"] = 5120
            (root / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "topology"):
                inspect_artifact(root)
        with tempfile.TemporaryDirectory() as directory:
            root = self.artifact(Path(directory))
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "../escape.safetensors"}})
            )
            with self.assertRaisesRegex(ValueError, "within the artifact"):
                inspect_artifact(root)

    def test_profile_is_ordinary_only(self):
        adapter = object.__new__(Qwen359BAdapter)
        self.assertEqual(adapter.profile_name(False), "qwen35-9b-apcv2-ordinary")
        with self.assertRaisesRegex(ValueError, "not implemented"):
            adapter.profile_name(True)

    def test_classifier_labels_must_be_distinct_single_tokens(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                self.add_special_tokens = add_special_tokens
                return {" store": [10], " defer": [11], " reject": [12]}.get(
                    text, [1, 2]
                )

        adapter = object.__new__(Qwen359BAdapter)
        adapter.tokenizer = Tokenizer()
        self.assertEqual(
            adapter.classifier_token_ids(("store", "defer", "reject")),
            {"store": 10, "defer": 11, "reject": 12},
        )
        self.assertFalse(adapter.tokenizer.add_special_tokens)
        with self.assertRaisesRegex(ValueError, "not one token"):
            adapter.classifier_token_ids(("multi token",))


if __name__ == "__main__":
    unittest.main()
