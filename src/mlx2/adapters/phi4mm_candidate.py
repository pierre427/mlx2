"""Complete Phi-4 multimodal artifact preflight; serving requires a modality route."""

from __future__ import annotations

import json
from pathlib import Path

from ._direct_mlx_vlm import load_backend, validate_media_paths


SOURCE_ROOT = str(Path.home() / "Desktop/mlx-uag/mlx-vlm-qwen4-exp")
SOURCE_REVISION = "653f1f13e238abb313fd45071bbd04b3de414635"
SOURCE_PATHS = (
    "mlx_vlm/models/phi4mm", "mlx_vlm/generate/dispatch.py",
    "mlx_vlm/prompt_utils.py", "mlx_vlm/utils.py",
)


def inspect_artifact(model_path: str | Path) -> dict:
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    expected = {"model_type": "phi4mm", "num_hidden_layers": 32,
                "hidden_size": 3072, "num_attention_heads": 24,
                "num_key_value_heads": 8, "vocab_size": 200064}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("unsupported Phi-4 multimodal topology")
    if config.get("audio_processor", {}).get("config", {}).get("num_blocks") != 24:
        raise ValueError("missing Phi-4 Conformer configuration")
    if config.get("vision_lora", {}).get("r") != 256 or config.get("speech_lora", {}).get("r") != 320:
        raise ValueError("missing Phi-4 modality LoRA configuration")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("missing Phi-4 tensor index")
    components = {
        "language": "model.embed_tokens.weight",
        "vision_tower": "image_embed.img_processor",
        "vision_projector": "image_embed.img_projection",
        "audio_encoder": "audio_embed.encoder",
        "audio_projector": "audio_embed.audio_projection",
        "vision_lora_a": ".lora_A.vision.",
        "vision_lora_b": ".lora_B.vision.",
        "speech_lora_a": ".lora_A.speech.",
        "speech_lora_b": ".lora_B.speech.",
    }
    counts = {name: sum(marker in key for key in weights) for name, marker in components.items()}
    if any(not value for value in counts.values()):
        raise ValueError(f"Phi-4 multimodal component missing: {counts}")
    shards = sorted(set(weights.values()))
    for name in shards:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or Path(name).suffix != ".safetensors":
            raise ValueError("unsafe Phi-4 shard path")
        item = path / name
        # Hugging Face snapshots intentionally symlink shards into the same
        # repository's blobs directory.
        if not item.is_file() or not item.resolve().is_relative_to(path.parent.parent):
            raise ValueError(f"missing Phi-4 shard {name}")
    sidecars = {}
    for modality in ("vision", "speech"):
        directory = path / f"{modality}-lora"
        adapter = directory / "adapter_model.safetensors"
        configuration = directory / "adapter_config.json"
        if not adapter.is_file() or not configuration.is_file():
            raise ValueError(f"missing Phi-4 {modality} LoRA sidecar")
        sidecars[modality] = {"weights_bytes": adapter.stat().st_size,
                              "configuration": json.loads(configuration.read_text())}
    for name in ("preprocessor_config.json", "processor_config.json", "tokenizer.json"):
        if not (path / name).is_file():
            raise ValueError(f"missing Phi-4 processor file {name}")
    return {"model_type": "phi4mm", "path": str(path), "shards": len(shards),
            "indexed_tensors": len(weights), "components": counts,
            "sidecars": sidecars, "serving_route": None,
            "implementation": "direct-mlx-vlm-candidate",
            "blocker": "mlx2 media serving lifecycle and source-bound qualification are pending"}


class Phi4MMCandidate:
    """Direct standalone tri-modal candidate; no mlx2 serving registration."""

    def __init__(self, model_path: str | Path, *, backend=None):
        self.artifact = inspect_artifact(model_path)
        self.backend = backend or load_backend(SOURCE_ROOT, SOURCE_REVISION, SOURCE_PATHS)
        self.model, self.processor = self.backend.load(self.artifact["path"], strict=True)
        if getattr(self.model, "model_type", None) != "phi4mm":
            raise ValueError("MLX-VLM loaded a different model family")

    def generate(self, prompt: str, *, images: list[str | Path] | None = None,
                 audios: list[str | Path] | None = None, max_tokens: int = 128,
                 temperature: float = 0.0) -> dict:
        if not isinstance(prompt, str) or not 0 < max_tokens <= 4096 or temperature < 0:
            raise ValueError("invalid Phi-4 generation parameters")
        images = validate_media_paths(images, kind="image")
        audios = validate_media_paths(audios, kind="audio")
        self.model.set_modality(has_image=bool(images), has_audio=bool(audios))
        formatted = self.backend.apply_chat_template(
            self.processor, self.model.config, prompt,
            num_images=len(images), num_audios=len(audios),
        )
        result = self.backend.generate(
            model=self.model, processor=self.processor, prompt=formatted,
            image=images or None, audio=audios or None,
            max_tokens=max_tokens, temperature=temperature, verbose=False,
        )
        return {"text": result.text, "route_receipt": {
            "model_type": "phi4mm", "execution": "direct-mlx-vlm-candidate",
            "source_revision": SOURCE_REVISION, "qualification": "pending",
            "modalities": ["text", *(["image"] if images else []), *(["audio"] if audios else [])],
            "lora_mode": "both" if images and audios else "vision" if images else "speech" if audios else "base",
        }}
