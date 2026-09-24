"""Artifact-bound image and video adapters; execution remains unqualified.

These adapters are deliberately separate from the causal-token resolver. They
inspect weights without importing MLX, then delegate model-specific math to
version-pinned MLX image/video runtimes only when explicitly invoked.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

QWEN_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
GGUF_REVISION = "40319fb15542f0ad22921e0124a191a8a935a60a"
LTX_REVISION = "5e6e71018ee1756ed329b697a7b4aedc934dfce9"
LTX_RUNTIME_REVISION = "fbc4b0524dd1e01da2d07a44e14dd9dfe0a74d5e"
QWEN_BACKEND_REVISION = "cc8b86f110278505296d461f612ee41c21d5fd65"


@dataclass(frozen=True, slots=True)
class MediaArtifact:
    kind: str
    path: Path
    source_revision: str
    fingerprint: str
    execution_qualified: bool = False


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    data: bytes
    mime_type: str
    width: int
    height: int
    artifact_fingerprint: str


@dataclass(frozen=True, slots=True)
class GeneratedVideo:
    path: Path
    mime_type: str
    artifact_fingerprint: str


def _json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise TypeError(f"expected JSON object at {path}")
    return result


def _fingerprint(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _complete_snapshot(path: Path, *, repo: str, revision: str) -> dict[str, Any]:
    receipt = _json(path / ".hf-download-complete.json")
    manifest = _json(path / ".hf-download-manifest.json")
    if receipt.get("repo") != repo or receipt.get("revision") != revision:
        raise ValueError(f"unverified {repo} revision at {path}")
    if manifest.get("repo") != repo or manifest.get("revision") != revision:
        raise ValueError(f"manifest revision differs at {path}")
    if receipt.get("files") != len(manifest["files"]) or receipt.get("total_bytes") != sum(
        entry["size"] for entry in manifest["files"]
    ):
        raise ValueError(f"completion marker differs from manifest at {path}")
    for entry in manifest["files"]:
        rel = Path(entry["path"])
        target = (path / rel).resolve()
        if rel.is_absolute() or not target.is_relative_to(path.resolve()):
            raise ValueError("manifest path escapes model artifact")
        if not target.is_file() or target.stat().st_size != entry["size"]:
            raise ValueError(f"missing or incomplete model file: {rel}")
    return manifest


def inspect_qwen_image21(path: str | Path) -> MediaArtifact:
    root = Path(path).expanduser().resolve()
    model = _json(root / "model_index.json")
    transformer = _json(root / "transformer" / "config.json")
    if model.get("_class_name") != "QwenImage21Pipeline":
        raise ValueError("expected Qwen-Image-2.1 pipeline")
    if transformer.get("_class_name") != "QwenImage21Transformer2DModel":
        raise ValueError("expected Qwen-Image-2.1 transformer")
    if (transformer.get("num_layers"), transformer.get("num_attention_heads")) != (32, 32):
        raise ValueError("unexpected Qwen-Image-2.1 topology")
    conversion = root / "mlx2-conversion.json"
    if conversion.exists():
        proof = _json(conversion)
        if proof.get("source_revision") != GGUF_REVISION or proof.get("base_revision") != QWEN_REVISION:
            raise ValueError("converted GGUF lacks its pinned base components")
        if proof.get("tensor_count") != 297 or len(proof.get("output_files", {})) != 33:
            raise ValueError("converted GGUF transformer is incomplete")
        for name, record in proof["output_files"].items():
            target = (root / "transformer" / name).resolve()
            if not target.is_relative_to(root) or target.stat().st_size != record["size"]:
                raise ValueError(f"converted transformer shard missing: {name}")
        for rel in ("processor/tokenizer.json", "text_encoder/config.json", "vae/config.json"):
            if not (root / rel).is_file():
                raise ValueError(f"converted pipeline lacks {rel}")
        return MediaArtifact("qwen-image-2.1-gguf-mlx", root, GGUF_REVISION, _fingerprint(proof))
    manifest = _complete_snapshot(root, repo="Qwen/Qwen-Image-2.1", revision=QWEN_REVISION)
    return MediaArtifact("qwen-image-2.1", root, QWEN_REVISION, _fingerprint(manifest))


def inspect_ltx25_source(path: str | Path) -> MediaArtifact:
    root = Path(path).expanduser().resolve()
    manifest = _complete_snapshot(root, repo="Lightricks/LTX-2.5", revision=LTX_REVISION)
    names = {entry["path"] for entry in manifest["files"]}
    required = {
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
    }
    if not required.issubset(names):
        raise ValueError("LTX-2.5 distilled pipeline is incomplete")
    return MediaArtifact("ltx-2.5-distilled-source", root, LTX_REVISION, _fingerprint(manifest))


def _verify_qwen_backend_revision() -> None:
    spec = importlib.util.find_spec("mlx_vlm")
    if spec is None or spec.origin is None:
        raise RuntimeError("mlx-vlm Qwen image backend is not installed")
    root = Path(spec.origin).resolve().parents[1]
    if not (root / ".git").exists():
        raise RuntimeError("mlx-vlm backend must be installed from the pinned local checkout")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, timeout=5,
    ).stdout.strip()
    if revision != QWEN_BACKEND_REVISION:
        raise RuntimeError("mlx-vlm Qwen image backend revision differs from the inspected source")


class QwenImage21Adapter:
    """Direct generation/edit adapter using the pinned mlx-vlm Qwen backend."""

    def __init__(
        self,
        path: str | Path,
        *,
        backend_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.artifact = inspect_qwen_image21(path)
        self._backend_factory = backend_factory
        self._generator: Any = None
        self._editor: Any = None

    def _model(self, *, edit: bool) -> Any:
        if self._backend_factory is not None:
            return self._backend_factory(self.artifact.path, edit=edit)
        _verify_qwen_backend_revision()
        from mlx_vlm.models.qwen_image.model import (
            QwenImageEditModel,
            QwenImageGenerationModel,
        )

        cls = QwenImageEditModel if edit else QwenImageGenerationModel
        return cls.from_model_id(str(self.artifact.path), download=False)

    def generate_image(
        self, prompt: str, *, width: int = 1024, height: int = 1024,
        steps: int = 30, seed: int = 0,
    ) -> GeneratedImage:
        if not prompt.strip() or width < 256 or height < 256 or width % 16 or height % 16:
            raise ValueError("prompt and 16-aligned dimensions of at least 256 are required")
        if not 1 <= steps <= 100:
            raise ValueError("steps must be in 1..100")
        from mlx_vlm.generate.image import ImageGenerationRequest

        if self._generator is None:
            self._generator = self._model(edit=False)
        result = self._generator.generate(
            ImageGenerationRequest(prompt=prompt, width=width, height=height, steps=steps, seed=seed)
        )
        return self._encode(result.array)

    def edit_image(
        self, prompt: str, image_paths: list[str | Path], *,
        width: int = 1024, height: int = 1024, steps: int = 40, seed: int = 0,
    ) -> GeneratedImage:
        if not prompt.strip() or not image_paths:
            raise ValueError("prompt and at least one reference image are required")
        from mlx_vlm.generate.edit_image import ImageEditRequest

        if self._editor is None:
            self._editor = self._model(edit=True)
        result = self._editor.edit(ImageEditRequest(
            prompt=prompt, image_paths=[str(Path(p).expanduser().resolve()) for p in image_paths],
            width=width, height=height, steps=steps, seed=seed,
        ))
        return self._encode(result.array)

    def _encode(self, array: Any) -> GeneratedImage:
        import numpy as np
        from PIL import Image

        pixels = np.array(array)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
            raise ValueError("Qwen image backend returned invalid pixels")
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        return GeneratedImage(buffer.getvalue(), "image/png", pixels.shape[1], pixels.shape[0], self.artifact.fingerprint)


class LTX25Adapter:
    """Distilled LTX video bridge to a pinned local ltx-2-mlx runtime."""

    def __init__(self, *, source: str | Path, mlx_model: str | Path, runtime_root: str | Path) -> None:
        self.artifact = inspect_ltx25_source(source)
        self.mlx_model = Path(mlx_model).expanduser().resolve()
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        config = _json(self.mlx_model / "config.json")
        if not str(config.get("model_version", "")).startswith("2.5"):
            raise ValueError("LTX MLX conversion is not version 2.5")
        if not (self.mlx_model / "transformer-distilled.safetensors").is_file():
            raise ValueError("LTX MLX distilled transformer is missing")
        conversion = _json(self.mlx_model / ".mlx2-cpu-conversion.json")
        if (
            conversion.get("source_fingerprint") != self.artifact.fingerprint
            or conversion.get("runtime_revision") != LTX_RUNTIME_REVISION
            or set(conversion.get("steps", {})) != {
                "config", "transformer-distilled", "connector", "text-encoder",
                "vae", "audio-vae", "duration-head", "upscalers",
            }
        ):
            raise ValueError("LTX conversion receipt is incomplete or source mismatched")
        revision = subprocess.run(
            ["git", "-C", str(self.runtime_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if revision != LTX_RUNTIME_REVISION:
            raise ValueError("LTX runtime revision differs from the inspected implementation")
        self.executable = self.runtime_root / ".venv" / "bin" / "python"
        if not self.executable.is_file():
            raise ValueError("LTX runtime executable is missing")

    def generate_video(
        self, prompt: str, *, output: str | Path, width: int = 704,
        height: int = 480, frames: int = 97, frame_rate: int = 24,
        seed: int = 0, timeout_seconds: int = 7200,
    ) -> GeneratedVideo:
        if not prompt.strip() or width % 32 or height % 32 or width < 256 or height < 256:
            raise ValueError("prompt and 32-aligned dimensions of at least 256 are required")
        if frames < 9 or (frames - 1) % 8 or frame_rate <= 0:
            raise ValueError("LTX frames must be 8n+1 and frame rate positive")
        target = Path(output).expanduser().resolve()
        if target.suffix.lower() != ".mp4" or target.exists():
            raise ValueError("output must be a new .mp4 path")
        target.parent.mkdir(parents=True, exist_ok=True)
        runner = (
            "import sys; from ltx_pipelines_mlx.cli import main; "
            "sys.argv=['ltx-2-mlx', *sys.argv[1:], '--prompt', sys.stdin.read()]; main()"
        )
        command = [
            str(self.executable), "-c", runner, "generate", "--distilled", "--model", str(self.mlx_model),
            "--gemma", str(self.mlx_model / "text_encoder"), "--output", str(target),
            "--width", str(width), "--height", str(height), "--frames", str(frames),
            "--frame-rate", str(frame_rate), "--seed", str(seed), "--quiet",
        ]
        environment = os.environ.copy()
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        completed = subprocess.run(
            command, input=prompt, env=environment, capture_output=True,
            text=True, timeout=timeout_seconds, check=False,
        )
        if completed.returncode or not target.is_file() or target.stat().st_size == 0:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"LTX generation failed: {completed.stderr[-2000:]}")
        return GeneratedVideo(target, "video/mp4", self.artifact.fingerprint)
