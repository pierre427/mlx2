"""Qwen2 ordinary artifact topology and generation defaults, without Metal."""

import json
import tempfile
import unittest
from pathlib import Path

from mlx_blocker import install_for_test_case

from mlx2.adapters.standard_decoder import inspect_artifact

BASE = {
    "model_type": "qwen2",
    "hidden_size": 896,
    "num_hidden_layers": 2,
    "intermediate_size": 4864,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
    "hidden_act": "silu",
    "sliding_window": 32768,
    "use_sliding_window": False,
}


class Qwen2ArtifactGateCPU(unittest.TestCase):
    def setUp(self):
        install_for_test_case(self)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        (self.path / "model.safetensors").write_bytes(b"test-shard")

    def inspect(self, changes=None, generation=None):
        (self.path / "config.json").write_text(json.dumps(BASE | (changes or {})))
        if generation is not None:
            (self.path / "generation_config.json").write_text(json.dumps(generation))
        return inspect_artifact(self.path, expected="qwen2")

    def test_dormant_sliding_window_metadata_is_accepted(self):
        artifact = self.inspect()
        self.assertEqual(artifact["config"]["head_dim"], 64)
        self.assertIsNone(artifact["sampling_defaults"])
        self.assertFalse(artifact["has_mtp"])

    def test_unsupported_topology_fails_before_model_load(self):
        cases = (
            ({"use_sliding_window": True}, "full attention"),
            ({"layer_types": ["sliding_attention"] * 2}, "full attention"),
            ({"attention_bias": True}, "projection biases"),
            ({"mlp_bias": True}, "projection biases"),
            ({"hidden_act": "gelu"}, "SwiGLU"),
            ({"head_dim": 32}, "head_dim"),
            ({"num_experts": 2}, "expert layers"),
        )
        for changes, message in cases:
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(ValueError, message),
            ):
                self.inspect(changes)

    def test_generation_defaults_bind_and_bad_values_fail_closed(self):
        artifact = self.inspect(
            generation={"do_sample": True, "temperature": 0.7, "top_p": 0.8}
        )
        self.assertEqual(
            artifact["sampling_defaults"], {"temperature": 0.7, "top_p": 0.8}
        )
        for generation, message in (
            ({"do_sample": "false"}, "boolean"),
            ({"do_sample": False, "temperature": 0.7}, "conflicts"),
            ({"do_sample": True, "top_p": 2}, "top_p"),
        ):
            with (
                self.subTest(generation=generation),
                self.assertRaisesRegex(ValueError, message),
            ):
                self.inspect(generation=generation)
        self.assertEqual(
            self.inspect(generation={"do_sample": False})["sampling_defaults"],
            {"temperature": 0},
        )


if __name__ == "__main__":
    unittest.main()
