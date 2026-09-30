"""Source-bound MiniMax-Music3 MLX acoustic LoRA and bounded generation bridge.

External port identity and license evidence: provenance/media-lora.json.
No automatic serving route is registered by this direct adapter.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

from .generative_media import (
    LTX25Adapter,
    MediaArtifact,
    _file_sha256,
    _fingerprint,
    _json,
)
from .media_lora_control import MediaLoRAControl, serialized
from .music3_pin import SOURCE_SHA256, UNIFIED_SHA256, UNIFIED_TREE_SHA256

MUSIC3_SOURCE_REVISION = "fbdf52fbaaca799592917417eb05f1899f1255ec"
MUSIC3_RUNTIME_REVISION = "36cddae1146af463cace351af4a1404042ce3268"


def inspect_music3(path):
    root = Path(path).expanduser().resolve()
    proof = _json(root / "mlx2-music3-manifest.json")
    if (
        proof.get("source_repo") != "MiniMaxAI/MiniMax-Music3"
        or proof.get("source_revision") != MUSIC3_SOURCE_REVISION
    ):
        raise ValueError("Music3 source revision is not pinned")
    files = proof.get("files")
    required = {"flowmatching_vae.pth", "dav.pth"}
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError("Music3 manifest is incomplete")
    if not any(
        k.startswith("qwen_7B/qwen_7B/") and k.endswith(".safetensors") for k in files
    ):
        raise ValueError("Music3 backbone/depth weights missing from manifest")
    if (
        any(root.rglob("*.aria2"))
        or any(root.rglob("*.partial"))
        or any(root.rglob("*.incomplete"))
    ):
        raise ValueError("Music3 snapshot contains partial downloads")
    index_path = root / "qwen_7B/qwen_7B/model.safetensors.index.json"
    index = _json(index_path)
    shards = set(index.get("weight_map", {}).values())
    if not shards or any("qwen_7B/qwen_7B/" + name not in files for name in shards):
        raise ValueError("Music3 backbone/depth shard coverage is incomplete")
    if (
        "qwen_7B/qwen_7B/config.json" not in files
        or str(index_path.relative_to(root)) not in files
    ):
        raise ValueError("Music3 backbone config/index is unbound")
    tokenizer = "qwen_7B/qwen3-8B-tokenizer-music/tokenizer.json"
    if tokenizer not in files:
        raise ValueError("Music3 tokenizer missing from manifest")
    # Every file consumed by the backbone/depth loader and tokenizer must be bound.
    consumed = {
        str(p.relative_to(root))
        for folder in ("qwen_7B/qwen_7B", "qwen_7B/qwen3-8B-tokenizer-music")
        for p in (root / folder).rglob("*")
        if p.is_file()
    }
    if not consumed.issubset(files):
        raise ValueError("Music3 unbound loader input files")
    for name, record in files.items():
        target = (root / name).resolve()
        if (
            not target.is_relative_to(root)
            or not isinstance(record, dict)
            or not target.is_file()
            or target.stat().st_size != record.get("size")
            or _file_sha256(target) != record.get("sha256")
        ):
            raise ValueError(f"Music3 artifact file changed: {name}")
    return MediaArtifact(
        "minimax-music3", root, MUSIC3_SOURCE_REVISION, _fingerprint(proof)
    )


@dataclass(frozen=True)
class GeneratedMusic:
    data: bytes
    mime_type: str
    sample_rate: int
    frames: int
    artifact_fingerprint: str
    lora_fingerprint: str | None = None
    state_epoch: int = 0


def _runtime_modules(runtime_root):
    root = Path(runtime_root).expanduser().resolve()
    for name, expected in SOURCE_SHA256.items():
        if _file_sha256(root / name) != expected:
            raise ValueError(f"Music3 runtime source changed: {name}")
    unified = (
        Path(os.environ.get("MM3_MLX_LM_UNIFIED") or root.parent / "mlx-lm-unified")
        .expanduser()
        .resolve()
    )
    tree = hashlib.sha256()
    for path in sorted((unified / "mlx_lm").rglob("*.py")):
        name = str(path.relative_to(unified))
        tree.update(name.encode() + b"\0" + bytes.fromhex(_file_sha256(path)))
    if tree.hexdigest() != UNIFIED_TREE_SHA256:
        raise ValueError("Music3 unified Python package differs from pinned revision")
    for name, expected in UNIFIED_SHA256.items():
        if _file_sha256(unified / name) != expected:
            raise ValueError(f"Music3 unified dependency changed: {name}")
    package = root / "minimax_music3_mlx"
    for name, module in list(sys.modules.items()):
        if name == "minimax_music3_mlx" or name.startswith("minimax_music3_mlx."):
            file = getattr(module, "__file__", None)
            if file is None or not Path(file).resolve().is_relative_to(package):
                raise RuntimeError("another Music3 runtime is already imported")
    for name, module in list(sys.modules.items()):
        if name == "mlx_lm" or name.startswith("mlx_lm."):
            file = getattr(module, "__file__", None)
            if file is None or not Path(file).resolve().is_relative_to(
                unified / "mlx_lm"
            ):
                raise RuntimeError("another unified dependency is already imported")
    sys.path.insert(0, str(root))
    try:
        modules = {
            name: importlib.import_module("minimax_music3_mlx." + name)
            for name in (
                "backbone",
                "depth_decoder",
                "condition_encoder",
                "dit",
                "vocoder",
                "pipeline",
                "prompt",
            )
        }
        for name, module in list(sys.modules.items()):
            if name == "mlx_lm" or name.startswith("mlx_lm."):
                file = getattr(module, "__file__", None)
                if file is None or not Path(file).resolve().is_relative_to(
                    unified / "mlx_lm"
                ):
                    raise RuntimeError(
                        "Music3 unified dependency import identity differs"
                    )
        return modules
    finally:
        sys.path.remove(str(root))


class Music3Adapter(MediaLoRAControl):
    def __init__(self, path, *, runtime_root, backend_factory=None):
        self.artifact = inspect_music3(path)
        proof = _json(self.artifact.path / "mlx2-music3-manifest.json")
        self._input_identity = {
            name: LTX25Adapter._file_identity(self.artifact.path / name)
            for name in (*proof["files"], "mlx2-music3-manifest.json")
        }
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self._backend_factory = backend_factory
        if inspect_music3(path).fingerprint != self.artifact.fingerprint:
            raise ValueError("Music3 snapshot changed during inspection")
        self._backend = None
        self._init_lora("minimax-music3", MUSIC3_RUNTIME_REVISION)

    def _verify_input_identity(self):
        for name, expected in self._input_identity.items():
            if LTX25Adapter._file_identity(self.artifact.path / name) != expected:
                raise ValueError(
                    f"Music3 loader input changed since inspection: {name}"
                )
        root = self.artifact.path
        for folder in ("qwen_7B/qwen_7B", "qwen_7B/qwen3-8B-tokenizer-music"):
            if any(
                str(p.relative_to(root)) not in self._input_identity
                for p in (root / folder).rglob("*")
                if p.is_file()
            ):
                raise ValueError("Music3 new unbound loader input")

    def _ensure_backend(self):
        if self._backend is None:
            self._verify_input_identity()
            if self._backend_factory is not None:
                candidate = self._backend_factory(self.artifact.path)
            else:
                from types import SimpleNamespace

                import mlx.core as mx
                from transformers import AutoTokenizer

                runtime = _runtime_modules(self.runtime_root)
                root = self.artifact.path
                tokenizer = AutoTokenizer.from_pretrained(
                    str(root / "qwen_7B/qwen3-8B-tokenizer-music"),
                    local_files_only=True,
                )
                runtime["prompt"].validate_tokenizer_ids(tokenizer)
                backbone, _ = runtime["backbone"].load_backbone(
                    root / "qwen_7B/qwen_7B", dtype=mx.bfloat16
                )
                depth = runtime["depth_decoder"].load_depth_decoder(
                    root / "qwen_7B/qwen_7B", dtype=mx.bfloat16
                )
                encoder = runtime["condition_encoder"].load_condition_encoder(
                    root / "flowmatching_vae.pth", dtype=mx.float32
                )
                dit = runtime["dit"].load_dit(
                    root / "flowmatching_vae.pth", dtype=mx.float32
                )
                vocoder = runtime["vocoder"].load_vocoder(root / "dav.pth")

                def generate(caption, lyrics, *, seed, max_frames, steps):
                    ids = tokenizer.encode(
                        runtime["prompt"].build_prompt(caption, lyrics),
                        add_special_tokens=False,
                    )
                    if not ids or len(ids) > 5000:
                        raise ValueError("Music3 prompt must have 1..5000 tokens")
                    return runtime["pipeline"].generate_music(
                        backbone,
                        depth,
                        encoder,
                        dit,
                        vocoder,
                        ids,
                        seed=seed,
                        max_frames=max_frames,
                        num_steps=steps,
                        multiwindow=False,
                    )

                candidate = SimpleNamespace(dit=dit, generate=generate)
            self._verify_input_identity()
            self._attach_lora(candidate.dit)
            self._backend = candidate
        return self._backend

    def _lora_models(self):
        return [self._ensure_backend().dit]

    @serialized
    def generate_music(self, caption, lyrics="", *, seed=0, max_frames=200, steps=30):
        import numpy as np

        if (
            not isinstance(caption, str)
            or not caption.strip()
            or not isinstance(lyrics, str)
        ):
            raise ValueError("Music3 requires a caption and string lyrics")
        if (
            isinstance(max_frames, bool)
            or not isinstance(max_frames, int)
            or not 1 <= max_frames <= 200
            or isinstance(steps, bool)
            or not isinstance(steps, int)
            or not 1 <= steps <= 100
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed < 2**63
        ):
            raise ValueError(
                "Music3 requires bounded frames (1..200), steps (1..100), and unsigned seed"
            )
        result, frames = self._ensure_backend().generate(
            caption, lyrics, seed=seed, max_frames=max_frames, steps=steps
        )
        samples = np.asarray(result, dtype=np.float32)
        if (
            samples.ndim != 2
            or samples.shape[0] != 2
            or samples.shape[1] <= 0
            or not np.isfinite(samples).all()
        ):
            raise ValueError("Music3 backend returned invalid stereo waveform")
        if not isinstance(frames, int) or not 1 <= frames <= max_frames:
            raise ValueError("Music3 backend violated frame bound")
        output = io.BytesIO()
        with wave.open(output, "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(32000)
            handle.writeframes(
                (np.clip(samples, -1, 1).T * 32767).astype("<i2").tobytes()
            )
        self._mark_lora_used()
        return GeneratedMusic(
            output.getvalue(),
            "audio/wav",
            32000,
            frames,
            self.artifact.fingerprint,
            self._lora.fingerprint if self._lora else None,
            self._lora_epoch,
        )


def prepare_music3_manifest(path):
    """Bind an existing completed, SHA-verified pinned HF snapshot; no download."""
    from mlx2.runtime.media_lora import _digest

    from .generative_media import _complete_snapshot

    root = Path(path).expanduser().resolve()
    manifest = _complete_snapshot(
        root, repo="MiniMaxAI/MiniMax-Music3", revision=MUSIC3_SOURCE_REVISION
    )
    files = {}
    for entry in manifest["files"]:
        name = entry["path"]
        if name in ("flowmatching_vae.pth", "dav.pth") or name.startswith(
            ("qwen_7B/qwen_7B/", "qwen_7B/qwen3-8B-tokenizer-music/")
        ):
            if not entry.get("sha256"):
                raise ValueError(
                    "Music3 source manifest requires SHA-256 for every loader input"
                )
            files[name] = {"size": entry["size"], "sha256": entry["sha256"]}
    proof = {
        "source_repo": "MiniMaxAI/MiniMax-Music3",
        "source_revision": MUSIC3_SOURCE_REVISION,
        "upstream_manifest_sha256": _digest(root / ".hf-download-manifest.json"),
        "files": files,
    }
    target = root / "mlx2-music3-manifest.json"
    # Exclusive creation preserves previously recorded artifact identities.
    with target.open("x") as handle:
        json.dump(proof, handle, indent=2)
        handle.write("\n")
    try:
        return inspect_music3(root)
    except BaseException:
        target.unlink()
        raise
