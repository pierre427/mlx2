"""CPU-only artifact and conversion boundary checks for generative media."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.adapters.generative_media import (
    GGUF_REVISION,
    LTX_CONVERTER_SHA256,
    LTX_REVISION,
    LTX_RUNTIME_REVISION,
    QWEN_BACKEND_REVISION,
    QWEN_REVISION,
    LTX25Adapter,
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
        shards[name] = {"size": 1, "sha256": hashlib.sha256(b"x").hexdigest()}
    (tmp_path / "mlx2-conversion.json").write_text(json.dumps({
        "source_revision": GGUF_REVISION, "base_revision": QWEN_REVISION,
        "tensor_count": 297, "output_dtype": "bfloat16", "output_files": shards,
    }))
    with pytest.raises(ValueError, match="lacks"):
        inspect_qwen_image21(tmp_path)
    for name in ("processor/tokenizer.json", "text_encoder/config.json", "vae/config.json"):
        _write(tmp_path / name)
    proof_path = tmp_path / "mlx2-conversion.json"
    proof = json.loads(proof_path.read_text())
    proof["base_files"] = {
        name: {"size": (tmp_path / name).stat().st_size, "sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}
        for name in ("model_index.json", "processor/tokenizer.json", "text_encoder/config.json", "vae/config.json")
    }
    proof_path.write_text(json.dumps(proof))
    assert inspect_qwen_image21(tmp_path).kind == "qwen-image-2.1-gguf-mlx-bf16"
    _write(tmp_path / "processor/tokenizer.json", b"y")
    with pytest.raises(ValueError, match="base component changed"):
        inspect_qwen_image21(tmp_path)


def test_ltx_distilled_source_requires_all_pipeline_components(tmp_path: Path) -> None:
    names = [
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
    ]
    _snapshot(tmp_path, "Lightricks/LTX-2.5", LTX_REVISION, names)
    manifest_path = tmp_path / ".hf-download-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["sha256"] = hashlib.sha256(b"x").hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    assert inspect_ltx25_source(tmp_path).kind == "ltx-2.5-distilled-source"
    _write(tmp_path / names[0], b"y")
    with pytest.raises(ValueError, match="hash changed"):
        inspect_ltx25_source(tmp_path)
    _write(tmp_path / names[0])
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
    with pytest.raises(FileNotFoundError, match="reference image"):
        adapter.edit_image("change the image", [tmp_path / "missing.png"], width=256, height=256)


def test_official_qwen_8bit_requires_pinned_quantization_and_hashes(tmp_path: Path) -> None:
    quant = {"group_size": 64, "bits": 8, "mode": "affine"}
    _write(tmp_path / "model_index.json", b'{"_class_name":"QwenImage21Pipeline"}')
    _write(tmp_path / "transformer/config.json", json.dumps({
        "_class_name": "QwenImage21Transformer2DModel", "num_layers": 32,
        "num_attention_heads": 32, "mlx_format": True, "quantization": quant,
    }).encode())
    _write(tmp_path / "text_encoder/config.json", json.dumps({"mlx_format": True, "quantization": quant}).encode())
    _write(tmp_path / "transformer/model.safetensors")
    files = {
        name: {"size": (tmp_path / name).stat().st_size, "sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}
        for name in ("model_index.json", "transformer/config.json", "transformer/model.safetensors", "text_encoder/config.json")
    }
    (tmp_path / "mlx2-official-conversion.json").write_text(json.dumps({
        "source_repo": "Qwen/Qwen-Image-2.1", "source_revision": QWEN_REVISION,
        "backend_revision": QWEN_BACKEND_REVISION, "quantization": quant,
        "output_files": files,
    }))
    assert inspect_qwen_image21(tmp_path).kind == "qwen-image-2.1-official-mlx-8bit"
    _write(tmp_path / "transformer/model.safetensors", b"y")
    with pytest.raises(ValueError, match="output changed"):
        inspect_qwen_image21(tmp_path)


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


def test_ltx_conversion_resume_rechecks_output_hash(tmp_path: Path) -> None:
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "prepare_ltx25_mlx.py"))
    path = tmp_path / "transformer-distilled.safetensors"
    _write(path, b"abc")
    records = {path.name: {"size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}}
    assert module["_matches"](tmp_path, records, (path.name,))
    _write(path, b"xyz")
    assert not module["_matches"](tmp_path, records, (path.name,))


def test_ltx_partial_conversion_checks_pinned_source_hash(tmp_path: Path) -> None:
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "prepare_ltx25_mlx.py"))
    names = ["diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"] + [f"other-{i}" for i in range(7)]
    _snapshot(tmp_path, "Lightricks/LTX-2.5", LTX_REVISION, names)
    manifest_path = tmp_path / ".hf-download-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["sha256"] = hashlib.sha256(b"x").hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    path = tmp_path / names[0]
    assert module["_partial_fingerprint"](tmp_path, (path,))
    _write(path, b"y")
    with pytest.raises(ValueError, match="hash changed"):
        module["_partial_fingerprint"](tmp_path, (path,))


def test_ltx_adapter_rejects_changed_conversion_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, output, runtime = (tmp_path / part for part in ("source", "output", "runtime"))
    names = [
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
    ]
    _snapshot(source, "Lightricks/LTX-2.5", LTX_REVISION, names)
    fingerprint = inspect_ltx25_source(source).fingerprint
    _write(output / "config.json", b'{"model_version":"2.5.0"}')
    _write(output / "transformer-distilled.safetensors")
    _write(output / "probe.safetensors")
    record = {"probe.safetensors": {"size": 1, "sha256": hashlib.sha256(b"x").hexdigest()}}
    steps = {name: record for name in (
        "config", "transformer-distilled", "connector", "text-encoder",
        "vae", "audio-vae", "duration-head", "upscalers",
    )}
    (output / ".mlx2-cpu-conversion.json").write_text(json.dumps({
        "source_fingerprint": fingerprint, "runtime_revision": LTX_RUNTIME_REVISION,
        "converter_sha256": LTX_CONVERTER_SHA256,
        "steps": steps,
    }))
    _write(runtime / ".venv/bin/python")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=LTX_RUNTIME_REVISION + "\n"))
    assert LTX25Adapter(source=source, mlx_model=output, runtime_root=runtime).artifact.fingerprint == fingerprint
    receipt_path = output / ".mlx2-cpu-conversion.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["converter_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt is incomplete"):
        LTX25Adapter(source=source, mlx_model=output, runtime_root=runtime)
    receipt["converter_sha256"] = LTX_CONVERTER_SHA256
    receipt_path.write_text(json.dumps(receipt))
    _write(output / "probe.safetensors", b"y")
    with pytest.raises(ValueError, match="output changed"):
        LTX25Adapter(source=source, mlx_model=output, runtime_root=runtime)
