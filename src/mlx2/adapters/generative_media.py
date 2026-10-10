"""Artifact-bound image and video adapters; serving routes remain unqualified.

These adapters are deliberately separate from the causal-token resolver. They
inspect weights without importing MLX, then delegate model-specific math to
version-pinned MLX image/video runtimes only when explicitly invoked.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .media_lora_control import MediaLoRAControl, serialized
from .mlx_vlm_pin import MLX_VLM_REVISION, require_pinned_mlx_vlm
from .pinned_imports import PinnedSourceFinder, PinnedSourceLoader

QWEN_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
GGUF_REVISION = "40319fb15542f0ad22921e0124a191a8a935a60a"
LTX_REVISION = "5e6e71018ee1756ed329b697a7b4aedc934dfce9"
LTX_RUNTIME_REVISION = "fbc4b0524dd1e01da2d07a44e14dd9dfe0a74d5e"
LTX_CONVERTER_SHA256 = (
    "2c880778d7772275b96f4eef4ee32376a29de0c94de52f4ca2b0aaccad9af98e"
)
# The Qwen image backend runs on the one mlx-vlm pin shared by every mlx-vlm
# route (``mlx_vlm_pin.MLX_VLM_REVISION``).
QWEN_BACKEND_REVISION = MLX_VLM_REVISION
# mlx-vlm revisions whose Qwen-Image conversion code (``mlx_vlm/models/qwen_image``
# convert/weights/config) is identical, so an official 8-bit conversion
# recorded under either is accepted.  cc8b86f1 is the local fork commit the
# existing 8-bit artifact was converted and first qualified on (upstream
# 884bbd93 plus a cache.py-only change); its qwen_image tree differs from
# 67599f2e only by the num_images batching of #2355.
QWEN_CONVERSION_BACKEND_REVISIONS = frozenset(
    {
        "cc8b86f110278505296d461f612ee41c21d5fd65",
        MLX_VLM_REVISION,
    }
)


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
    lora_fingerprint: str | None = None
    state_epoch: int = 0
    # Request parameters, so the receipt reproduces the image.
    seed: int | None = None
    steps: int | None = None
    reference_sha256: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GeneratedVideo:
    path: Path
    mime_type: str
    artifact_fingerprint: str
    lora_fingerprint: str | None = None
    state_epoch: int = 0
    # Request parameters, so the receipt reproduces the clip.
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    frame_rate: int | None = None


# Direct-route size bounds (memory, not quality): Qwen-Image's default is
# 1024x1024; LTX's 704x480x97.  Raise with a measured run.
MAX_IMAGE_SIDE = 2048
MAX_VIDEO_SIDE = 1920
MAX_VIDEO_FRAMES = 257


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_seed(seed) -> None:
    if not _is_int(seed) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an unsigned integer")


def _json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise TypeError(f"expected JSON object at {path}")
    return result


def _fingerprint(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _complete_snapshot(path: Path, *, repo: str, revision: str) -> dict[str, Any]:
    receipt = _json(path / ".hf-download-complete.json")
    manifest = _json(path / ".hf-download-manifest.json")
    if receipt.get("repo") != repo or receipt.get("revision") != revision:
        raise ValueError(f"unverified {repo} revision at {path}")
    if manifest.get("repo") != repo or manifest.get("revision") != revision:
        raise ValueError(f"manifest revision differs at {path}")
    if receipt.get("files") != len(manifest["files"]) or receipt.get(
        "total_bytes"
    ) != sum(entry["size"] for entry in manifest["files"]):
        raise ValueError(f"completion marker differs from manifest at {path}")
    for entry in manifest["files"]:
        rel = Path(entry["path"])
        target = (path / rel).resolve()
        if rel.is_absolute() or not target.is_relative_to(path.resolve()):
            raise ValueError("manifest path escapes model artifact")
        if not target.is_file() or target.stat().st_size != entry["size"]:
            raise ValueError(f"missing or incomplete model file: {rel}")
        if entry.get("sha256") and _file_sha256(target) != entry["sha256"]:
            raise ValueError(f"model file hash changed: {rel}")
    return manifest


def inspect_qwen_image21(path: str | Path) -> MediaArtifact:
    root = Path(path).expanduser().resolve()
    model = _json(root / "model_index.json")
    transformer = _json(root / "transformer" / "config.json")
    if model.get("_class_name") != "QwenImage21Pipeline":
        raise ValueError("expected Qwen-Image-2.1 pipeline")
    if transformer.get("_class_name") != "QwenImage21Transformer2DModel":
        raise ValueError("expected Qwen-Image-2.1 transformer")
    if (transformer.get("num_layers"), transformer.get("num_attention_heads")) != (
        32,
        32,
    ):
        raise ValueError("unexpected Qwen-Image-2.1 topology")
    official_conversion = root / "mlx2-official-conversion.json"
    if official_conversion.exists():
        proof = _json(official_conversion)
        quantization = {"group_size": 64, "bits": 8, "mode": "affine"}
        if (
            proof.get("source_repo") != "Qwen/Qwen-Image-2.1"
            or proof.get("source_revision") != QWEN_REVISION
            or proof.get("backend_revision") not in QWEN_CONVERSION_BACKEND_REVISIONS
            or proof.get("quantization") != quantization
            or transformer.get("quantization") != quantization
            or transformer.get("mlx_format") is not True
        ):
            raise ValueError("official Qwen 8-bit conversion identity differs")
        encoder = _json(root / "text_encoder" / "config.json")
        if (
            encoder.get("quantization") != quantization
            or encoder.get("mlx_format") is not True
        ):
            raise ValueError("Qwen text encoder is not the matching MLX 8-bit format")
        files = proof.get("output_files")
        if not isinstance(files, dict) or not files:
            raise ValueError("official Qwen 8-bit output checksums are missing")
        for name, record in files.items():
            target = (root / name).resolve()
            if (
                not target.is_relative_to(root)
                or not isinstance(record, dict)
                or not target.is_file()
                or target.stat().st_size != record.get("size")
                or _file_sha256(target) != record.get("sha256")
            ):
                raise ValueError(f"official Qwen 8-bit output changed: {name}")
        return MediaArtifact(
            "qwen-image-2.1-official-mlx-8bit", root, QWEN_REVISION, _fingerprint(proof)
        )
    conversion = root / "mlx2-conversion.json"
    if conversion.exists():
        proof = _json(conversion)
        if (
            proof.get("source_revision") != GGUF_REVISION
            or proof.get("base_revision") != QWEN_REVISION
        ):
            raise ValueError("converted GGUF lacks its pinned base components")
        if proof.get("tensor_count") != 297 or len(proof.get("output_files", {})) != 33:
            raise ValueError("converted GGUF transformer is incomplete")
        for name, record in proof["output_files"].items():
            target = (root / "transformer" / name).resolve()
            if (
                not target.is_relative_to(root)
                or target.stat().st_size != record["size"]
                or _file_sha256(target) != record["sha256"]
            ):
                raise ValueError(f"converted transformer shard missing: {name}")
        base_files = proof.get("base_files")
        if not isinstance(base_files, dict) or not base_files:
            raise ValueError("converted pipeline lacks verified base components")
        for name, record in base_files.items():
            target = (root / name).resolve()
            if (
                not target.is_relative_to(root)
                or not isinstance(record, dict)
                or not target.is_file()
                or target.stat().st_size != record.get("size")
                or _file_sha256(target) != record.get("sha256")
            ):
                raise ValueError(f"converted base component changed: {name}")
        for rel in (
            "processor/tokenizer.json",
            "text_encoder/config.json",
            "vae/config.json",
        ):
            if not (root / rel).is_file():
                raise ValueError(f"converted pipeline lacks {rel}")
        quantization = {"group_size": 64, "bits": 4, "mode": "affine"}
        if (
            proof.get("output_dtype") == "bfloat16"
            and proof.get("quantization") is None
        ):
            kind = "qwen-image-2.1-gguf-mlx-bf16"
        elif (
            proof.get("output_dtype") == "mlx-affine-4bit"
            and proof.get("quantization") == quantization
        ):
            if transformer.get("quantization") != quantization:
                raise ValueError("converted transformer quantization config differs")
            kind = "qwen-image-2.1-gguf-mlx-4bit"
        else:
            raise ValueError("converted transformer format is unsupported")
        config_path = root / "transformer/config.json"
        config_hash = _file_sha256(config_path)
        record = proof.get("transformer_config")
        if record is not None and (
            not isinstance(record, dict)
            or record.get("size") != config_path.stat().st_size
            or record.get("sha256") != config_hash
        ):
            raise ValueError("converted transformer configuration changed")
        # Legacy receipts omitted this small file. Bind its observed bytes in
        # the identity without rewriting weights or trusting an absent checksum.
        fingerprint = _fingerprint(
            {"conversion": proof, "transformer_config_sha256": config_hash}
        )
        return MediaArtifact(kind, root, GGUF_REVISION, fingerprint)
    manifest = _complete_snapshot(
        root, repo="Qwen/Qwen-Image-2.1", revision=QWEN_REVISION
    )
    observed = {
        entry["path"]: _file_sha256(root / entry["path"])
        for entry in manifest["files"]
        if not entry.get("sha256")
    }
    fingerprint = (
        _fingerprint({"manifest": manifest, "observed_sha256": observed})
        if observed
        else _fingerprint(manifest)
    )
    return MediaArtifact("qwen-image-2.1", root, QWEN_REVISION, fingerprint)


def inspect_ltx25_source(path: str | Path) -> MediaArtifact:
    root = Path(path).expanduser().resolve()
    manifest = _complete_snapshot(
        root, repo="Lightricks/LTX-2.5", revision=LTX_REVISION
    )
    names = {entry["path"] for entry in manifest["files"]}
    required = {
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
    }
    if not required.issubset(names):
        raise ValueError("LTX-2.5 distilled pipeline is incomplete")
    return MediaArtifact(
        "ltx-2.5-distilled-source", root, LTX_REVISION, _fingerprint(manifest)
    )


def _verify_qwen_backend_revision() -> dict:
    """Fail closed unless the installed mlx-vlm is the shared pinned revision."""
    try:
        return require_pinned_mlx_vlm()
    except RuntimeError as exc:
        raise RuntimeError(f"mlx-vlm Qwen image backend: {exc}") from None


def _qwen_bound_inputs(root):
    """Capture identities before inspection hashes can race a large shard load."""
    root = Path(root).expanduser().resolve()
    official, converted = (
        root / "mlx2-official-conversion.json",
        root / "mlx2-conversion.json",
    )
    proof_path = (
        official
        if official.exists()
        else converted
        if converted.exists()
        else root / ".hf-download-manifest.json"
    )
    identities = {
        str(proof_path.relative_to(root)): LTX25Adapter._file_identity(proof_path)
    }
    proof = _json(proof_path)
    if proof_path == official:
        records = proof["output_files"]
    elif proof_path == converted:
        records = {
            "transformer/" + name: record
            for name, record in proof["output_files"].items()
        }
        records.update(proof["base_files"])
        records["transformer/config.json"] = proof.get("transformer_config", {})
    else:
        records = {entry["path"]: entry for entry in proof["files"]}
        identities[".hf-download-complete.json"] = LTX25Adapter._file_identity(
            root / ".hf-download-complete.json"
        )
    for name in records:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Qwen loader input escapes artifact")
        identities[name] = LTX25Adapter._file_identity(path)
    if not {"model_index.json", "transformer/config.json"}.issubset(identities):
        raise ValueError("Qwen pipeline configuration is unbound")
    return identities


class QwenImage21Adapter(MediaLoRAControl):
    """Direct generation/edit adapter using a content-bound mlx-vlm Qwen backend."""

    def __init__(
        self,
        path: str | Path,
        *,
        backend_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._input_identity = _qwen_bound_inputs(path)
        self.artifact = inspect_qwen_image21(path)
        self._verify_input_identity()
        self._backend_factory = backend_factory
        self._generator: Any = None
        self._editor: Any = None
        self._init_lora("qwen-image-2.1", QWEN_BACKEND_REVISION)

    def _lora_models(self):
        self._verify_input_identity()
        if self._generator is None and self._editor is None:
            self._generator = self._model(edit=False)
        models = [m for m in (self._generator, self._editor) if m is not None]
        # Some factories share a pipeline: install once per transformer identity.
        return list(
            {
                id(m.pipeline.transformer): m.pipeline.transformer for m in models
            }.values()
        )

    def _verify_input_identity(self):
        root = self.artifact.path
        for name, expected in self._input_identity.items():
            path = (root / name).resolve()
            if (
                not path.is_relative_to(root)
                or LTX25Adapter._file_identity(path) != expected
            ):
                raise ValueError(f"Qwen loader input changed since inspection: {name}")
        for folder in ("transformer", "vae", "text_encoder", "processor"):
            if any(
                str(p.relative_to(root)) not in self._input_identity
                for p in (root / folder).rglob("*")
                if p.is_file() and p.name != ".DS_Store"
            ):
                raise ValueError("Qwen new unbound loader input")

    def _model(self, *, edit: bool) -> Any:
        self._verify_input_identity()
        if self._backend_factory is not None:
            candidate = self._backend_factory(self.artifact.path, edit=edit)
        else:
            from .vlm_runtime import bind_backend, verify_contract

            backend = bind_backend("qwen_image")
            self.mlx_vlm_runtime = backend.identity
            self.mlx_vlm_build = backend.provenance
            from mlx_vlm.models.qwen_image.model import (
                QwenImageEditModel,
                QwenImageGenerationModel,
            )

            cls = QwenImageEditModel if edit else QwenImageGenerationModel
            candidate = cls.from_model_id(str(self.artifact.path), download=False)
            verify_contract("qwen_image", backend.package_root)
        self._verify_input_identity()
        return candidate

    @serialized
    def generate_image(
        self,
        prompt: str,
        *,
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        seed: int = 0,
    ) -> GeneratedImage:
        self._validate_request(prompt, width, height, steps)
        _validate_seed(seed)
        self._verify_input_identity()
        from mlx_vlm.generate.image import ImageGenerationRequest

        if self._generator is None:
            candidate = self._model(edit=False)
            if self._lora is not None and not any(
                session.model is candidate.pipeline.transformer
                for session in self._lora_sessions
            ):
                self._attach_lora(candidate.pipeline.transformer)
            self._generator = candidate
        result = self._generator.generate(
            ImageGenerationRequest(
                prompt=prompt, width=width, height=height, steps=steps, seed=seed
            )
        )
        return self._encode(result.array, seed=seed, steps=steps)

    @serialized
    def edit_image(
        self,
        prompt: str,
        image_paths: list[str | Path],
        *,
        width: int = 1024,
        height: int = 1024,
        steps: int = 40,
        seed: int = 0,
    ) -> GeneratedImage:
        self._validate_request(prompt, width, height, steps)
        _validate_seed(seed)
        self._verify_input_identity()
        if not image_paths:
            raise ValueError("at least one reference image is required")
        references = [Path(path).expanduser().resolve() for path in image_paths]
        if any(not path.is_file() for path in references):
            raise FileNotFoundError("Qwen reference image is missing")
        from mlx_vlm.generate.edit_image import ImageEditRequest

        if self._editor is None:
            candidate = self._model(edit=True)
            if self._lora is not None and not any(
                session.model is candidate.pipeline.transformer
                for session in self._lora_sessions
            ):
                self._attach_lora(candidate.pipeline.transformer)
            self._editor = candidate
        result = self._editor.edit(
            ImageEditRequest(
                prompt=prompt,
                image_paths=[str(path) for path in references],
                width=width,
                height=height,
                steps=steps,
                seed=seed,
            )
        )
        references_sha256 = tuple(
            hashlib.sha256(path.read_bytes()).hexdigest() for path in references
        )
        return self._encode(
            result.array, seed=seed, steps=steps, references=references_sha256
        )

    @staticmethod
    def _validate_request(prompt: str, width: int, height: int, steps: int) -> None:
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or not _is_int(width)
            or not _is_int(height)
            or not 256 <= width <= MAX_IMAGE_SIDE
            or not 256 <= height <= MAX_IMAGE_SIDE
            or width % 16
            or height % 16
        ):
            raise ValueError(
                "prompt and 16-aligned dimensions in "
                f"256..{MAX_IMAGE_SIDE} are required"
            )
        if not _is_int(steps) or not 1 <= steps <= 100:
            raise ValueError("steps must be in 1..100")

    def _encode(self, array: Any, *, seed=None, steps=None, references=()) -> GeneratedImage:
        import numpy as np
        from PIL import Image

        pixels = np.array(array)
        if (
            pixels.dtype != np.uint8
            or pixels.ndim != 3
            or pixels.shape[2] not in (3, 4)
        ):
            raise ValueError("Qwen image backend returned invalid pixels")
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        self._verify_input_identity()
        self._mark_lora_used()
        return GeneratedImage(
            buffer.getvalue(),
            "image/png",
            pixels.shape[1],
            pixels.shape[0],
            self.artifact.fingerprint,
            self._lora.fingerprint if self._lora else None,
            self._lora_epoch,
            seed=seed,
            steps=steps,
            reference_sha256=tuple(references),
        )


def _ltx_source_hashes(root, expected_revision=LTX_RUNTIME_REVISION):
    root = Path(root).resolve()
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != expected_revision:
        raise ValueError("LTX source revision differs during import")
    tracked = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "ls-tree",
            "-r",
            "-z",
            "--name-only",
            head,
            "--",
            "packages",
        ]
    )
    return {
        root / p.decode(): hashlib.sha256(
            subprocess.check_output(
                ["git", "-C", str(root), "show", head + ":" + p.decode()]
            )
        ).hexdigest()
        for p in tracked.split(b"\0")
        if p.endswith(b".py")
    }


def _ltx_origin_probe(expected_revision=LTX_RUNTIME_REVISION):
    # Keep this self-contained: the external LTX venv need not install mlx2.
    return (
        "import sys, tempfile, subprocess, hashlib, importlib.abc, importlib.machinery\n"
        "from pathlib import Path\n"
        "_cache=tempfile.TemporaryDirectory(prefix='mlx2-ltx-code-')\n"
        "sys.pycache_prefix=_cache.name\n"
        "root=Path(sys.argv[1]).resolve()\n"
        + inspect.getsource(PinnedSourceLoader)
        + "\n"
        + inspect.getsource(PinnedSourceFinder)
        + "\n"
        + inspect.getsource(_ltx_source_hashes).replace(
            "expected_revision=LTX_RUNTIME_REVISION",
            "expected_revision=" + repr(expected_revision),
        )
        + "\n_finder=PinnedSourceFinder(('ltx_core_mlx','ltx_pipelines_mlx'), _ltx_source_hashes(root))\n"
        "_finder.__enter__()\n"
        "import ltx_core_mlx, ltx_pipelines_mlx\n"
        "_finder.validate_loaded()\n"
    )


class LTX25Adapter(MediaLoRAControl):
    """Distilled LTX video bridge to a pinned local ltx-2-mlx runtime."""

    def __init__(
        self, *, source: str | Path, mlx_model: str | Path, runtime_root: str | Path
    ) -> None:
        self.artifact = inspect_ltx25_source(source)
        self._init_lora("ltx-2.5", LTX_RUNTIME_REVISION)
        self.mlx_model = Path(mlx_model).expanduser().resolve()
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        config = _json(self.mlx_model / "config.json")
        if not str(config.get("model_version", "")).startswith("2.5"):
            raise ValueError("LTX MLX conversion is not version 2.5")
        if not (self.mlx_model / "transformer-distilled.safetensors").is_file():
            raise ValueError("LTX MLX distilled transformer is missing")
        self._verify_checkpoint_selection()
        conversion = _json(self.mlx_model / ".mlx2-cpu-conversion.json")
        if (
            conversion.get("source_fingerprint") != self.artifact.fingerprint
            or conversion.get("runtime_revision") != LTX_RUNTIME_REVISION
            or conversion.get("converter_sha256") != LTX_CONVERTER_SHA256
            or set(conversion.get("steps", {}))
            != {
                "config",
                "transformer-distilled",
                "connector",
                "text-encoder",
                "vae",
                "audio-vae",
                "duration-head",
                "upscalers",
            }
        ):
            raise ValueError(
                "LTX conversion receipt is incomplete or source mismatched"
            )
        self._conversion_identity = {}
        for records in conversion["steps"].values():
            if not isinstance(records, dict) or not records:
                raise ValueError("LTX conversion output checksums are missing")
            for name, record in records.items():
                path = (self.mlx_model / name).resolve()
                expected_identity = self._file_identity(path)
                if (
                    not path.is_relative_to(self.mlx_model)
                    or not isinstance(record, dict)
                    or not path.is_file()
                    or path.stat().st_size != record.get("size")
                    or _file_sha256(path) != record.get("sha256")
                ):
                    raise ValueError(f"LTX conversion output changed: {name}")
                if self._file_identity(path) != expected_identity:
                    raise ValueError(
                        f"LTX conversion output changed during hashing: {name}"
                    )
                self._conversion_identity[name] = expected_identity
        self._bound_inputs = {
            name for records in conversion["steps"].values() for name in records
        }
        self._verify_input_listing()
        revision = subprocess.run(
            ["git", "-C", str(self.runtime_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if revision != LTX_RUNTIME_REVISION:
            raise ValueError(
                "LTX runtime revision differs from the inspected implementation"
            )
        self._lora_base_identity = _fingerprint(
            {"source_fingerprint": self.artifact.fingerprint, "conversion": conversion}
        )
        self.executable = self.runtime_root / ".venv" / "bin" / "python"
        if not self.executable.is_file():
            raise ValueError("LTX runtime executable is missing")
        if "transformer-distilled.safetensors" not in self._conversion_identity:
            raise ValueError(
                "LTX canonical transformer is absent from conversion receipt"
            )
        self._verify_execution_identity()

    @staticmethod
    def _file_identity(path):
        st = path.stat()
        return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

    def _verify_input_listing(self):
        # Every model/config/tokenizer input must be receipt-bound, including
        # files which native fallback resolvers might prefer over canonical ones.
        for path in self.mlx_model.rglob("*"):
            if not path.is_file():
                continue
            name = str(path.relative_to(self.mlx_model))
            if name == ".mlx2-cpu-conversion.json" or path.name == ".DS_Store":
                continue
            if name not in self._bound_inputs:
                raise ValueError(f"LTX unbound loader input: {name}")

    def _verify_checkpoint_selection(self):
        # The native resolver prefers these files over our receipt-bound name.
        if (self.mlx_model / "transformer.safetensors").exists() or any(
            self.mlx_model.glob("transformer-distilled-*.safetensors")
        ):
            raise ValueError("LTX shadow transformer checkpoint is unbound")

    def _verify_execution_identity(self):
        self._verify_checkpoint_selection()
        self._verify_input_listing()
        # Checksums were verified at construction; reject changed files rather
        # than silently use different weights under the old base identity.
        for name, expected in self._conversion_identity.items():
            if self._file_identity(self.mlx_model / name) != expected:
                raise ValueError(
                    f"LTX converted artifact changed since inspection: {name}"
                )
        revision = subprocess.run(
            ["git", "-C", str(self.runtime_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if revision != LTX_RUNTIME_REVISION:
            raise ValueError("LTX runtime revision changed since inspection")
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(self.runtime_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                "packages",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if dirty:
            raise ValueError("LTX runtime packages differ from pinned source")
        subprocess.run(
            [str(self.executable), "-c", _ltx_origin_probe(), str(self.runtime_root)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def _lora_models(self):
        # Native CLI owns model loading. Validate every canonical target against
        # the converted checkpoint header before forwarding anything to the GPU.
        return []

    @serialized
    def load_lora(self, path):
        from mlx2.runtime.media_lora import inspect_media_lora

        artifact = inspect_media_lora(
            path,
            family="ltx-2.5",
            base_fingerprint=self.lora_base_fingerprint,
            backend_revision=LTX_RUNTIME_REVISION,
        )
        self._validate_ltx_lora(artifact)
        return super().load_lora(path)

    def _validate_ltx_lora(self, artifact):
        self._verify_execution_identity()
        from safetensors import safe_open

        tensors = artifact.tensors()
        with safe_open(
            str(self.mlx_model / "transformer-distilled.safetensors"), framework="numpy"
        ) as f:
            names = set(f.keys())
            for key in artifact.keys:
                weight = f"transformer.{key}.weight"
                if weight not in names:
                    raise ValueError(f"LTX LoRA target missing: {key}")
                if f"transformer.{key}.scales" in names:
                    # Native fusion guesses quantization geometry and re-quantizes;
                    # preserve exact ordinary weights by accepting float targets only.
                    raise ValueError(
                        "LTX LoRA fusion of quantized targets is unsupported"
                    )
                shape = f.get_slice(weight).get_shape()
                a, b = tensors[key + ".lora_a"], tensors[key + ".lora_b"]
                if len(shape) != 2 or (b.shape[1], a.shape[0]) != tuple(shape):
                    raise ValueError(f"LTX LoRA target shape mismatch: {key}")

    @serialized
    def generate_video(
        self,
        prompt: str,
        *,
        output: str | Path,
        width: int = 704,
        height: int = 480,
        frames: int = 97,
        frame_rate: int = 24,
        seed: int = 0,
        timeout_seconds: int = 7200,
    ) -> GeneratedVideo:
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or not _is_int(width)
            or not _is_int(height)
            or width % 32
            or height % 32
            or not 256 <= width <= MAX_VIDEO_SIDE
            or not 256 <= height <= MAX_VIDEO_SIDE
        ):
            raise ValueError(
                "prompt and 32-aligned dimensions in "
                f"256..{MAX_VIDEO_SIDE} are required"
            )
        if (
            not _is_int(frames)
            or not _is_int(frame_rate)
            or not 9 <= frames <= MAX_VIDEO_FRAMES
            or (frames - 1) % 8
            or frame_rate <= 0
        ):
            raise ValueError(
                f"LTX frames must be 8n+1 in 9..{MAX_VIDEO_FRAMES} and frame rate positive"
            )
        _validate_seed(seed)
        self._verify_execution_identity()
        target = Path(output).expanduser().resolve()
        if target.suffix.lower() != ".mp4" or target.exists():
            raise ValueError("output must be a new .mp4 path")
        target.parent.mkdir(parents=True, exist_ok=True)
        runner = _ltx_origin_probe().replace(
            "sys.argv[1]", repr(str(self.runtime_root))
        ) + (
            "import mlx.core as mx; mx.set_default_device(mx.gpu); "
            "from ltx_pipelines_mlx.cli import main; "
            # One "--prompt=" token: a prompt starting with "-" would
            # otherwise parse as an option.
            "sys.argv=['ltx-2-mlx', *sys.argv[1:], '--prompt=' + sys.stdin.read()]; main()"
        )
        environment = os.environ.copy()
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        # Keep partial renders private, including on timeout. A same-filesystem
        # exclusive link publishes the completed file without replacing a
        # competing writer that arrived after the initial existence check.
        with tempfile.TemporaryDirectory(
            prefix=".mlx2-ltx-", dir=target.parent
        ) as staging:
            rendered = Path(staging) / target.name
            command = [
                str(self.executable),
                "-c",
                runner,
                "generate",
                "--distilled",
                "--model",
                str(self.mlx_model),
                "--gemma",
                str(self.mlx_model / "text_encoder"),
                "--output",
                str(rendered),
                "--width",
                str(width),
                "--height",
                str(height),
                "--frames",
                str(frames),
                "--frame-rate",
                str(frame_rate),
                "--seed",
                str(seed),
                "--quiet",
            ]
            if self._lora is not None:
                from mlx2.runtime.media_lora import export_ltx_native

                self._validate_ltx_lora(self._lora)
                native = Path(staging) / "lora.safetensors"
                export_ltx_native(self._lora, native)
                command.extend(["--lora", str(native), "1.0"])
            completed = subprocess.run(
                command,
                input=prompt,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            if (
                completed.returncode
                or not rendered.is_file()
                or rendered.stat().st_size == 0
            ):
                raise RuntimeError(f"LTX generation failed: {completed.stderr[-2000:]}")
            os.link(rendered, target)
        self._mark_lora_used()
        return GeneratedVideo(
            target,
            "video/mp4",
            self.artifact.fingerprint,
            self._lora.fingerprint if self._lora else None,
            self._lora_epoch,
            seed=seed,
            width=width,
            height=height,
            frames=frames,
            frame_rate=frame_rate,
        )
