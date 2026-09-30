"""CPU-only artifact and request-contract checks for Muse image generation."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from mlx_blocker import block_mlx_imports



from mlx2.adapters.muse_glimmer_vision_candidate import (  # noqa: E402
    MuseGlimmerVisionCandidate,
    inspect_artifact,
)


@pytest.fixture(autouse=True)
def block_mlx(monkeypatch):
    block_mlx_imports(monkeypatch, __name__)


ROOT = Path("~/mlx-models")
TARGETS = (
    "Muse-Glimmer-30B",
    "Muse-Glimmer-30B-mlx-bf16",
    "Muse-Glimmer-30B-mlx-4bit",
    "Muse-Glimmer-30B-mlx-8bit",
    "Muse-Glimmer-30B-cyber-clone-bf16",
    "Muse-Glimmer-30B-cyber-clone-4bit",
)


@pytest.mark.parametrize("name", TARGETS)
def test_local_muse_artifacts_have_full_vision_stack(name):
    target = ROOT / name
    if not target.is_dir():
        pytest.skip("local optional Muse artifact absent")
    record = inspect_artifact(target)
    assert record["vision_tensor_count"] >= 790
    assert record["vision_candidate"]
    assert not record["qualified"] and not record["selected"]
    assert all((target / item[0]).is_file() for item in record["files"])
    assert "mlx" not in sys.modules


def test_incomplete_vision_stack_fails_closed(tmp_path):
    source = ROOT / "Muse-Glimmer-30B-mlx-4bit"
    if not source.is_dir():
        pytest.skip("local optional Muse artifact absent")
    config = json.loads((source / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "chat_template.jinja").write_text("<|patch|>")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"vision_tower.patch_embedder.patch_embedding.weight": "one.safetensors"}
    }))
    with pytest.raises(ValueError, match="incomplete"):
        inspect_artifact(tmp_path)


def test_direct_image_request_uses_processor_and_generation_path(tmp_path, monkeypatch):
    target = ROOT / "Muse-Glimmer-30B-mlx-4bit"
    if not target.is_dir():
        pytest.skip("local optional Muse artifact absent")
    images = [tmp_path / "one.png", tmp_path / "two.png"]
    for image in images:
        image.write_bytes(b"image-placeholder")
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(("template", messages, kwargs))
            count = sum(part["type"] == "image" for part in messages[0]["content"])
            return "<|patch|>" * count + " Describe.<|start|>assistant"

    generate_module = ModuleType("mlx_vlm.generate")

    def generate(model, processor, prompt, **kwargs):
        calls.append(("generate", model, processor, prompt, kwargs))
        return SimpleNamespace(text="two objects", finish_reason="stop")

    generate_module.generate = generate
    package = ModuleType("mlx_vlm")
    package.generate = generate_module
    monkeypatch.setitem(sys.modules, "mlx_vlm", package)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", generate_module)
    processor = SimpleNamespace(tokenizer=Tokenizer())
    model = object()
    adapter = MuseGlimmerVisionCandidate(target, backend_factory=lambda _: (model, processor))
    result = adapter.generate_response("Describe both.", images=images, max_tokens=32)
    assert result["text"] == "two objects"
    assert calls[0][1][0]["content"][:2] == [{"type": "image"}, {"type": "image"}]
    assert calls[1][4]["image"] == [str(image) for image in images]
    assert calls[1][4]["max_tokens"] == 32
    assert calls[0][2]["reasoning_strength"] == "low"
    assert calls[1][3].endswith(" to=user<|message|>")
    assert result["route"] == "muse-glimmer-vision-direct-candidate"
    assert not result["qualified"]
    with pytest.raises(ValueError, match="missing local image"):
        adapter.generate_response("Describe.", images=[tmp_path / "missing.png"])
    with pytest.raises(ValueError, match="inserted by the chat template"):
        adapter.generate_response("<|patch|>", images=images)
    assert "mlx" not in sys.modules


def test_direct_image_output_requires_client_visible_channel(tmp_path, monkeypatch):
    target = ROOT / "Muse-Glimmer-30B-mlx-4bit"
    if not target.is_dir():
        pytest.skip("local optional Muse artifact absent")
    image = tmp_path / "one.png"
    image.write_bytes(b"image-placeholder")

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "<|patch|><|start|>assistant"

    responses = iter((
        " to=self<|message|>The image shows a teapot.",
        " to=user<|message|>A red teapot.<|eot|>",
    ))
    generate_module = ModuleType("mlx_vlm.generate")
    generate_module.generate = lambda *args, **kwargs: SimpleNamespace(
        text=next(responses), finish_reason="length"
    )
    package = ModuleType("mlx_vlm")
    package.generate = generate_module
    monkeypatch.setitem(sys.modules, "mlx_vlm", package)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", generate_module)
    adapter = MuseGlimmerVisionCandidate(
        target, backend_factory=lambda _: (object(), SimpleNamespace(tokenizer=Tokenizer()))
    )
    with pytest.raises(ValueError, match="no client-visible answer"):
        adapter.generate_response("What is shown?", images=[image], max_tokens=32)
    result = adapter.generate_response("What is shown?", images=[image], max_tokens=32)
    assert result["text"] == "A red teapot."
    assert "mlx" not in sys.modules
