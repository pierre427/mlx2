"""CPU-only checks. Never import MLX or instantiate the tensor model."""

import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mlx2.adapters.qwen38_27b import (
    CACHE_LAYOUT,
    Qwen3827BAdapter,
    configure_environment,
    descriptor_for,
    inspect_artifact,
)
from mlx2.contracts import Capability, StatePlane

CONFIG = {
    "model_type": "qwen3_5",
    "text_config": {
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "full_attention_interval": 4,
        "vocab_size": 248320,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "mtp_num_hidden_layers": 1,
        "max_position_embeddings": 262144,
    },
}
MTP_KEYS = [
    "fc",
    "norm",
    "pre_fc_norm_embedding",
    "pre_fc_norm_hidden",
    "layers.0.self_attn.q_proj",
    "layers.0.self_attn.k_proj",
    "layers.0.self_attn.v_proj",
    "layers.0.self_attn.o_proj",
    "layers.0.mlp.gate_proj",
    "layers.0.mlp.up_proj",
    "layers.0.mlp.down_proj",
]


class PortTests(unittest.TestCase):
    def artifact(self, root, *, mtp=True):
        (root / "config.json").write_text(json.dumps(CONFIG))
        index = {"language_model.model.embed_tokens.weight": "model.safetensors"}
        if mtp:
            index.update(
                {
                    f"language_model.mtp.{name}.weight": "model.safetensors"
                    for name in MTP_KEYS
                }
            )
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": index})
        )
        (root / "model.safetensors").write_bytes(b"metadata-test-only")
        return root

    def test_import_does_not_import_tensor_runtime(self):
        code = """import sys
import mlx2.adapters.qwen38_27b
assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules)
assert 'mlx2.runtime.models.qwen38_27b' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_config_claim_does_not_confer_mtp(self):
        with tempfile.TemporaryDirectory() as d:
            artifact = inspect_artifact(self.artifact(Path(d), mtp=False))
            self.assertFalse(artifact["has_mtp"])
            self.assertNotIn(Capability.MTP, descriptor_for(has_mtp=False).capabilities)
            with self.assertRaisesRegex(ValueError, "embedded head"):
                Qwen3827BAdapter(d, require_mtp=True)

    def test_complete_embedded_head_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.artifact(Path(d))
            before = inspect_artifact(root)
            self.assertTrue(before["has_mtp"])
            self.assertEqual(before["mtp_tensor_count"], len(MTP_KEYS))
            (root / "tokenizer_config.json").write_text('{"chat_template": "changed"}')
            after = inspect_artifact(root)
            self.assertNotEqual(
                before["identity"]["fingerprint"], after["identity"]["fingerprint"]
            )

    def test_partial_head_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.artifact(Path(d))
            index = json.loads((root / "model.safetensors.index.json").read_text())
            del index["weight_map"]["language_model.mtp.fc.weight"]
            (root / "model.safetensors.index.json").write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                inspect_artifact(root)

    def test_wrong_topology_rejected_before_gpu_import(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.artifact(Path(d))
            cfg = copy.deepcopy(CONFIG)
            cfg["text_config"]["hidden_size"] = 4096
            (root / "config.json").write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, "topology"):
                inspect_artifact(root)

    def test_missing_shard_and_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.artifact(Path(d))
            (root / "model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "missing weight"):
                inspect_artifact(root)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "../outside"}})
            )
            with self.assertRaisesRegex(ValueError, "within the artifact"):
                inspect_artifact(root)

    def test_modern_cache_declarations_and_no_qsa(self):
        descriptor = descriptor_for(has_mtp=True)
        self.assertEqual(descriptor.cache_layout, CACHE_LAYOUT)
        self.assertTrue(
            {
                Capability.APC_V2,
                Capability.LAYERED_CACHE,
                Capability.CONTINUOUS_BATCH,
                Capability.SEGMENTED_MTP,
            }
            <= descriptor.capabilities
        )
        self.assertNotIn(StatePlane.SPARSE_INDEX, descriptor.state_planes)
        self.assertNotIn(Capability.VISION, descriptor.capabilities)
        self.assertEqual(descriptor.metadata["qualification"], "pending")

    def test_experimental_environment_does_not_leak(self):
        with patch.dict(os.environ, {"MLX_QWEN4_MEGAKERNEL": "1", "MLX_GDN_CORE": "1"}):
            profile = configure_environment()
            self.assertNotIn("MLX_QWEN4_MEGAKERNEL", os.environ)
            self.assertEqual(os.environ["MLX_GDN_CORE"], "0")
            self.assertEqual(profile["MLX_LM_SEGMENTED_SELF_MTP"], "1")

    def test_ordinary_profile_rejects_mtp(self):
        adapter = object.__new__(Qwen3827BAdapter)
        adapter.descriptor = descriptor_for(has_mtp=False)
        self.assertEqual(adapter.profile_name(False), "qwen38-27b-apcv2-ordinary")
        with self.assertRaises(ValueError):
            adapter.profile_name(True)

    def test_mined_sanitizer_does_not_double_shift_converted_norms(self):
        # Execute only the pure sanitizer method, not the tensor module.
        path = Path(__file__).parents[1] / "src/mlx2/runtime/models/qwen38_27b.py"
        model = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "TextModel"
        )
        method = next(
            n
            for n in model.body
            if isinstance(n, ast.FunctionDef) and n.name == "sanitize"
        )
        namespace = {}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            namespace,
        )

        class Array:
            def __init__(self, shape, value=0):
                self.shape, self.ndim, self.value = shape, len(shape), value

            def moveaxis(self, _a, _b):
                return Array((self.shape[0], self.shape[2], self.shape[1]), self.value)

            def __add__(self, value):
                return Array(self.shape, self.value + value)

        from types import SimpleNamespace

        model = SimpleNamespace(
            mtp=object(), args=SimpleNamespace(tie_word_embeddings=False)
        )
        weights = {
            "model.layers.0.linear_attn.conv1d.weight": Array((4, 3, 1)),
            "model.norm.weight": Array((4,), 1),
            "mtp.norm.weight": Array((4,), 1),
        }
        actual = namespace["sanitize"](model, weights)
        self.assertEqual(actual["model.norm.weight"].value, 1)
        self.assertEqual(actual["mtp.norm.weight"].value, 1)
        weights["model.layers.0.linear_attn.conv1d.weight"] = Array((4, 1, 3))
        actual = namespace["sanitize"](model, weights)
        self.assertEqual(actual["model.norm.weight"].value, 2)
        self.assertEqual(
            actual["model.layers.0.linear_attn.conv1d.weight"].shape, (4, 3, 1)
        )

    def test_execution_policy_rejects_qsa_and_invalid_draft_before_artifact_read(self):
        for policy in (
            {"shared_qsa_suffix": "on"},
            {"num_draft": 0},
            {"num_draft": True},
            [],
        ):
            with self.assertRaises(ValueError):
                Qwen3827BAdapter("/nonexistent/artifact", execution_policy=policy)
        adapter = object.__new__(Qwen3827BAdapter)
        adapter.descriptor = descriptor_for(has_mtp=False)
        config = adapter.execution_config(max_lanes=4, prefill_step=2048)
        self.assertEqual(config["num_draft"], 0)
        self.assertEqual(config["segment_aware_cohort_size"], 4)
        self.assertTrue(config["segment_aware_live_tip"])

    def test_cache_budget_is_dense_family_geometry(self):
        from mlx2.adapters.qwen38_memory import Qwen38CacheBudget

        adapter = object.__new__(Qwen3827BAdapter)
        adapter.model = SimpleNamespace(args=SimpleNamespace(text_config={
            "num_hidden_layers": 64,
            "full_attention_interval": 4,
            "mtp_num_hidden_layers": 1,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "linear_num_value_heads": 48,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
        }))
        ordinary = adapter.cache_budget(mtp=False)
        mtp = adapter.cache_budget(mtp=True)
        self.assertIsInstance(ordinary, Qwen38CacheBudget)
        self.assertEqual(ordinary.attention_layers, 16)
        self.assertEqual(ordinary.mtp_layers, 0)
        self.assertEqual(mtp.mtp_layers, 1)
        self.assertGreater(mtp.project(262144), ordinary.project(262144))

    def test_tensor_module_syntax_and_compile_claim(self):
        path = Path(__file__).parents[1] / "src/mlx2/runtime/models/qwen38_27b.py"
        source = path.read_text()
        compile(source, str(path), "exec")
        tree = ast.parse(source)
        names = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
        self.assertTrue(
            {"Model", "TextModel", "MTPModule", "Qwen3NextAttention"} <= names
        )
        self.assertNotIn("supports_compiled_decode_replay", source)
        self.assertNotIn("mlx_lm", source)


if __name__ == "__main__":
    unittest.main()
