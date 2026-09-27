"""Import-guarded local Flash-Next VLM candidate preflight."""

import importlib.abc
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError(f"real MLX import forbidden: {fullname}")
        return None


sys.meta_path.insert(0, BlockMLX())
from mlx2.adapters.flash_next_vlm import (  # noqa: E402
    FlashNextVisionCandidate, inspect_flash_next_vlm, split_candidate_mtp,
)

ARTIFACT = Path("~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP-VLM")


@unittest.skipUnless(ARTIFACT.is_dir(), "local Flash-Next VLM artifact unavailable")
class FlashNextVLMCPUTest(unittest.TestCase):
    def test_local_complete_embedded_components_are_candidates(self):
        record = inspect_flash_next_vlm(ARTIFACT)
        self.assertEqual((record["tensor_count"], record["shards"]), (3845, 22))
        self.assertEqual((record["vision_tensors"], record["mtp_tensors"],
                          record["ple_index_tensors"]), (333, 78, 384))
        self.assertFalse(record["ple_rows_bin"])
        self.assertFalse(record["qualified"])
        self.assertFalse(record["selected"])
        candidate = FlashNextVisionCandidate(ARTIFACT)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            candidate.generate_text("describe")
        with self.assertRaisesRegex(ValueError, "media file is missing"):
            candidate.generate_text("describe", image=ARTIFACT / "missing.png")
        with self.assertRaises(FileExistsError):
            split_candidate_mtp(ARTIFACT, ARTIFACT / "config.json")
        self.assertNotIn("mlx", sys.modules)

    def test_rejects_tampered_component_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("config.json", "model.safetensors.index.json", "tokenizer.json",
                         "tokenizer_config.json", "preprocessor_config.json"):
                (root / name).symlink_to(ARTIFACT / name)
            index = json.loads((ARTIFACT / "model.safetensors.index.json").read_text())
            key = next(k for k in index["weight_map"] if k.startswith("mtp."))
            index["weight_map"].pop(key)
            (root / "model.safetensors.index.json").unlink()
            (root / "model.safetensors.index.json").write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "tensor counts differ"):
                inspect_flash_next_vlm(root)

    def test_direct_candidate_binds_media_and_extracted_mtp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            image.write_bytes(b"fixture")
            draft = root / "draft"
            draft.mkdir()
            calls = {}
            model = types.SimpleNamespace(config={"model_type": "qwen4_exp"})
            def template(*args, **kwargs):
                calls["template"] = kwargs
                return "formatted"
            def generate(*args, **kwargs):
                calls["generate"] = (args, kwargs)
                return types.SimpleNamespace(text="ok", finish_reason="stop")
            def validate(*args):
                calls["validated"] = True
            modules = {
                "mlx_vlm": types.ModuleType("mlx_vlm"),
                "mlx_vlm.generate": types.SimpleNamespace(generate=generate),
                "mlx_vlm.prompt_utils": types.SimpleNamespace(apply_chat_template=template),
                "mlx_vlm.speculative": types.ModuleType("mlx_vlm.speculative"),
                "mlx_vlm.speculative.drafters": types.SimpleNamespace(
                    load_drafter=lambda *args, **kwargs: (object(), "mtp"),
                    validate_drafter_compatibility=validate),
            }
            with patch.dict(sys.modules, modules):
                candidate = FlashNextVisionCandidate(
                    ARTIFACT, backend_factory=lambda path: (model, object()))
                result = candidate.generate_text("describe", image=image, draft_model=draft)
            self.assertTrue(result.mtp_requested)
            self.assertTrue(calls["validated"])
            self.assertEqual(calls["template"]["num_images"], 1)
            self.assertEqual(calls["generate"][0][2], "formatted")
            self.assertEqual(calls["generate"][1]["draft_kind"], "mtp")
            self.assertNotIn("mlx", sys.modules)


if __name__ == "__main__":
    unittest.main()
