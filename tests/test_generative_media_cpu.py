"""CPU-only artifact and conversion boundary checks for generative media."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.adapters.generative_media import (
    GGUF_REVISION,
    LTX_REVISION,
    QWEN_REVISION,
    QwenImage21Adapter,
    inspect_ltx25_source,
    inspect_qwen_image21,
)


def _write(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _snapshot(root: Path, repo: str, revision: str, files: list[str]) -> None:
    for name in files:
        if not (root / name).exists():
            _write(root / name)
    (root / ".hf-download-manifest.json").write_text(json.dumps({
        "repo": repo, "revision": revision,
        "files": [{"path": name, "size": (root / name).stat().st_size} for name in files],
    }))
    (root / ".hf-download-complete.json").write_text(json.dumps({
        "repo": repo, "revision": revision, "files": len(files),
        "total_bytes": sum((root / name).stat().st_size for name in files),
    }))


def test_qwen_native_requires_complete_snapshot(tmp_path: Path) -> None:
    _write(tmp_path / "model_index.json", b'{"_class_name":"QwenImage21Pipeline"}')
    _write(tmp_path / "transformer/config.json", b'{"_class_name":"QwenImage21Transformer2DModel","num_layers":32,"num_attention_heads":32}')
    with pytest.raises(FileNotFoundError):
        inspect_qwen_image21(tmp_path)
    _snapshot(tmp_path, "Qwen/Qwen-Image-2.1", QWEN_REVISION, ["model_index.json", "transformer/config.json"])
    artifact = inspect_qwen_image21(tmp_path)
    assert artifact.kind == "qwen-image-2.1"
    assert artifact.execution_qualified is False


def test_converted_gguf_requires_complete_base_components(tmp_path: Path) -> None:
    _write(tmp_path / "model_index.json", b'{"_class_name":"QwenImage21Pipeline"}')
    _write(tmp_path / "transformer/config.json", b'{"_class_name":"QwenImage21Transformer2DModel","num_layers":32,"num_attention_heads":32}')
    shards = {}
    for index in range(33):
        name = f"block-{index:02d}.safetensors"
        _write(tmp_path / "transformer" / name)
        shards[name] = {"size": 1}
    (tmp_path / "mlx2-conversion.json").write_text(json.dumps({
        "source_revision": GGUF_REVISION, "base_revision": QWEN_REVISION,
        "tensor_count": 297, "output_files": shards,
    }))
    with pytest.raises(ValueError, match="lacks"):
        inspect_qwen_image21(tmp_path)
    for name in ("processor/tokenizer.json", "text_encoder/config.json", "vae/config.json"):
        _write(tmp_path / name)
    assert inspect_qwen_image21(tmp_path).kind == "qwen-image-2.1-gguf-mlx"


def test_ltx_distilled_source_requires_all_pipeline_components(tmp_path: Path) -> None:
    names = [
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
    ]
    _snapshot(tmp_path, "Lightricks/LTX-2.5", LTX_REVISION, names)
    assert inspect_ltx25_source(tmp_path).kind == "ltx-2.5-distilled-source"
    (tmp_path / names[0]).unlink()
    with pytest.raises(ValueError, match="missing or incomplete"):
        inspect_ltx25_source(tmp_path)


def test_qwen_adapter_encodes_backend_pixels_without_model_load(tmp_path: Path) -> None:
    _write(tmp_path / "model_index.json", b'{"_class_name":"QwenImage21Pipeline"}')
    _write(tmp_path / "transformer/config.json", b'{"_class_name":"QwenImage21Transformer2DModel","num_layers":32,"num_attention_heads":32}')
    _snapshot(tmp_path, "Qwen/Qwen-Image-2.1", QWEN_REVISION, ["model_index.json", "transformer/config.json"])
    seen = []

    class Backend:
        def generate(self, request):
            seen.append(request)
            return SimpleNamespace(array=np.zeros((256, 256, 3), dtype=np.uint8))

    adapter = QwenImage21Adapter(tmp_path, backend_factory=lambda path, edit: Backend())
    with pytest.raises(ValueError, match="16-aligned"):
        adapter.generate_image("test", width=257, height=256)
    result = adapter.generate_image("test", width=256, height=256)
    assert result.data.startswith(b"\x89PNG\r\n\x1a\n")
    assert result.width == result.height == 256
    assert seen[0].prompt == "test"


def test_artifact_adapter_import_does_not_import_mlx() -> None:
    script = (
        "import builtins; original=builtins.__import__; "
        "builtins.__import__=lambda name,*args,**kwargs: "
        "(_ for _ in ()).throw(RuntimeError('MLX imported')) "
        "if name == 'mlx' or name.startswith('mlx.') else original(name,*args,**kwargs); "
        "import mlx2.adapters.generative_media"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run([sys.executable, "-c", script], env=environment, check=True)
