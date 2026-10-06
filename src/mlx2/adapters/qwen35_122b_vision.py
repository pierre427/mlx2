"""Direct Qwen3.5 122B image/video candidate on the pinned mlx-vlm source.

This is separate from the ordinary text serving route. The optional source
owns vision projection and media processing; no vision route is qualified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .qwen35_122b import inspect_artifact

SOURCE_REVISION = "8a5e704e0fe43cd8654c144c4ecbd4c8aececeb5"
SOURCE_ROOT = Path.home() / "Desktop/mlx-uag/worktrees/agnes-vlm-support"
SOURCE_PATHS = ("mlx_vlm/models/qwen3_5", "mlx_vlm/models/qwen3_5_moe",
                "mlx_vlm/generate/dispatch.py", "mlx_vlm/prompt_utils.py")


@dataclass(frozen=True, slots=True)
class VisionText:
    text: str
    artifact_fingerprint: str
    finish_reason: str | None


class Qwen35122BVisionCandidate:
    """Explicit offline candidate for the checkpoint's embedded vision tower."""

    def __init__(self, path: str | Path, *, backend_factory: Callable | None = None):
        artifact = inspect_artifact(path)
        self.artifact = artifact
        self._backend_factory = backend_factory
        self._backend: tuple[Any, Any] | None = None

    def _load(self):
        if self._backend is None:
            if self._backend_factory is not None:
                self._backend = self._backend_factory(Path(self.artifact["identity"]["path"]))
            else:
                from ._direct_mlx_vlm import load_backend
                backend = load_backend(SOURCE_ROOT, SOURCE_REVISION, SOURCE_PATHS)
                self._backend = backend.load(self.artifact["identity"]["path"],
                                             lazy=False, strict=True,
                                             trust_remote_code=False)
        return self._backend

    def generate_text(self, prompt: str, *, image: str | Path | None = None,
                      video: str | Path | None = None,
                      max_tokens: int = 256) -> VisionText:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Qwen3.5 vision prompt must be nonempty")
        if (image is None) == (video is None):
            raise ValueError("supply exactly one image or video")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 2048:
            raise ValueError("Qwen3.5 vision max_tokens must be 1..2048")
        media = Path(image if image is not None else video).expanduser().resolve()
        if not media.is_file():
            raise ValueError("Qwen3.5 vision media file is missing")
        model, processor = self._load()
        from ..runtime.chat_templates import secure_model_chat_templates
        secure_model_chat_templates(processor)
        from mlx_vlm.generate import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        formatted = apply_chat_template(processor, model.config, prompt,
                                        num_images=int(image is not None),
                                        video=[str(media)] if video is not None else None,
                                        fps=1.0)
        result = generate(model, processor, formatted,
                          image=str(media) if image is not None else None,
                          video=str(media) if video is not None else None,
                          verbose=False, max_tokens=max_tokens)
        return VisionText(result.text, self.artifact["identity"]["fingerprint"],
                          result.finish_reason)

    def close(self):
        self._backend = None
