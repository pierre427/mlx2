"""CPU-only checks for the separate Qwen3.5 full vision candidate."""

import importlib.abc
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError(f"real MLX import forbidden: {fullname}")
        return None


sys.meta_path.insert(0, BlockMLX())

from mlx2.adapters.qwen35_vlm_candidate import Qwen35VLMAdapter, inspect_artifact  # noqa: E402


ROOT = Path("~/mlx-models")
CACHE = Path("~/.cache/huggingface/hub")


def test_local_vision_artifacts_and_mtp_are_distinct():
    small = inspect_artifact(CACHE / "models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3")
    optiq = inspect_artifact(CACHE / "models--mlx-community--Qwen3.5-4B-OptiQ-4bit/snapshots/6cb5bdfd0bf15f484881fb9f1ab6d7c840fddde9")
    medium = inspect_artifact(CACHE / "models--mlx-community--Qwen3.5-9B-4bit/snapshots/8b2b98c00a6b4d291155e4890773ca8f769aee53")
    dense = inspect_artifact(ROOT / "Qwen3.8-27B-MLX-4bit")
    moe = inspect_artifact(ROOT / "Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp")
    assert (dense["family"], dense["vision_tensor_count"], dense["mtp_candidate"]) == ("qwen3_5", 333, False)
    assert (small["vision_tensor_count"], medium["vision_tensor_count"]) == (297, 333)
    assert optiq["vision_tensor_count"] == 297
    assert (moe["family"], moe["vision_tensor_count"], moe["mtp_tensor_count"]) == ("qwen3_5_moe", 333, 42)
    assert not dense["selected"] and not moe["qualified"]
    assert "mlx" not in sys.modules


def test_direct_media_request_uses_injected_backend_without_mlx(tmp_path, monkeypatch):
    image = tmp_path / "frame.png"
    image.write_bytes(b"example")
    seen = []
    fake = ModuleType("mlx_vlm.generate")

    def generate(model, processor, prompt, **kwargs):
        seen.append((model, processor, prompt, kwargs))
        return SimpleNamespace(text="done", finish_reason="stop")

    fake.generate = generate
    prompt_utils = ModuleType("mlx_vlm.prompt_utils")

    def apply_chat_template(processor, config, prompt, **kwargs):
        seen.append(("template", processor, config, prompt, kwargs))
        return "<|vision_start|><|image_pad|><|vision_end|> " + prompt

    prompt_utils.apply_chat_template = apply_chat_template
    package = ModuleType("mlx_vlm")
    package.generate = fake
    package.prompt_utils = prompt_utils
    monkeypatch.setitem(sys.modules, "mlx_vlm", package)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", fake)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)
    adapter = Qwen35VLMAdapter(
        ROOT / "Qwen3.8-27B-MLX-4bit",
        backend_factory=lambda _: (SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5")), object()),
    )
    result = adapter.generate_response("Describe this", images=[image], max_tokens=32)
    assert result["text"] == "done"
    assert seen[0][0] == "template" and seen[0][4]["num_images"] == 1
    assert seen[1][2].startswith("<|vision_start|><|image_pad|>")
    assert seen[1][3]["image"] == [str(image)]
    assert seen[1][3]["video"] is None
    with pytest.raises(ValueError, match="no embedded MTP"):
        adapter.generate_response("Describe this", draft_model=tmp_path)
    with pytest.raises(ValueError, match="media file is missing"):
        adapter.generate_response("Describe this", images=[tmp_path / "missing.png"])
    assert "mlx" not in sys.modules
