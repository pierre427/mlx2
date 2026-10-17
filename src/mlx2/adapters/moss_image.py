"""Artifact-bound direct MLX adapter for MOSS-Image-SFT (unqualified)."""

from __future__ import annotations

import hashlib
import io
import json
import math
import threading
from pathlib import Path

from .generative_media import GeneratedImage, MediaArtifact, _file_sha256, _fingerprint

MOSS_IMAGE_REPO = "OpenMOSS-Team/MOSS-Image-SFT"
MOSS_IMAGE_REVISION = "daf706d2e1c7a6369c17b6d1658671aee6db1ebe"
Q8 = {"group_size": 64, "bits": 8, "mode": "affine"}


def snapshot_manifest():
    return json.loads(
        (Path(__file__).with_name("assets") / "moss-image-sft.json").read_text()
    )


def _identity(root, names):
    result = {}
    for name in names:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("MOSS-Image artifact input escapes its root")
        stat = path.stat()
        result[name] = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return result


def inspect_moss_image(path):
    root = Path(path).expanduser().resolve()
    conversion = root / "mlx2-conversion.json"
    if conversion.exists():
        proof = json.loads(conversion.read_text())
        if (
            proof.get("source_repo") != MOSS_IMAGE_REPO
            or proof.get("source_revision") != MOSS_IMAGE_REVISION
            or proof.get("source_fingerprint") != _fingerprint(snapshot_manifest())
            or proof.get("quantization") != Q8
        ):
            raise ValueError("MOSS-Image Q8 conversion identity differs")
        records = proof.get("output_files")
        if not isinstance(records, dict) or not records:
            raise ValueError("MOSS-Image Q8 conversion has no output manifest")
        before = _identity(root, records)
        for name, record in records.items():
            target = (root / name).resolve()
            if (
                not target.is_relative_to(root)
                or target.stat().st_size != record.get("size")
                or _file_sha256(target) != record.get("sha256")
            ):
                raise ValueError(f"MOSS-Image Q8 output changed: {name}")
        if _identity(root, records) != before:
            raise ValueError("MOSS-Image Q8 artifact changed during inspection")
        return MediaArtifact(
            "moss-image-sft-mlx-q8", root, MOSS_IMAGE_REVISION, _fingerprint(proof)
        )

    expected = snapshot_manifest()
    manifest = json.loads((root / ".hf-download-manifest.json").read_text())
    receipt = json.loads((root / ".hf-download-complete.json").read_text())
    if manifest != expected:
        raise ValueError(
            "MOSS-Image download manifest differs from the pinned artifact"
        )
    if receipt != {
        "repo": MOSS_IMAGE_REPO,
        "revision": MOSS_IMAGE_REVISION,
        "files": len(expected["files"]),
        "total_bytes": sum(entry["size"] for entry in expected["files"]),
    }:
        raise ValueError("MOSS-Image snapshot is incomplete or has another revision")
    names = [entry["path"] for entry in expected["files"]]
    before = _identity(
        root, names + [".hf-download-manifest.json", ".hf-download-complete.json"]
    )
    for entry in expected["files"]:
        target = root / entry["path"]
        if target.stat().st_size != entry["size"]:
            raise ValueError(f"incomplete MOSS-Image file: {entry['path']}")
        if entry.get("sha256"):
            observed = _file_sha256(target)
            wanted = entry["sha256"]
        else:
            observed = hashlib.sha1(
                f"blob {entry['size']}\0".encode() + target.read_bytes()
            ).hexdigest()
            wanted = entry["git_blob"]
        if observed != wanted:
            raise ValueError(f"MOSS-Image file hash differs: {entry['path']}")
    if _identity(root, before) != before:
        raise ValueError("MOSS-Image artifact changed during inspection")
    return MediaArtifact(
        "moss-image-sft-bf16", root, MOSS_IMAGE_REVISION, _fingerprint(expected)
    )


class MOSSImageSFTAdapter:
    def __init__(self, path, *, backend_factory=None):
        self.artifact = inspect_moss_image(path)
        conversion = self.artifact.path / "mlx2-conversion.json"
        if conversion.exists():
            self._bound_names = tuple(
                json.loads(conversion.read_text())["output_files"]
            ) + ("mlx2-conversion.json",)
        else:
            self._bound_names = tuple(
                entry["path"] for entry in snapshot_manifest()["files"]
            ) + (".hf-download-manifest.json", ".hf-download-complete.json")
        self._input_identity = _identity(self.artifact.path, self._bound_names)
        self._backend_factory = backend_factory
        self._backend = None
        self._lock = threading.RLock()
        self._observed_uses = 0
        self.last_route_receipt = self.status()

    def _verify_inputs(self):
        if _identity(self.artifact.path, self._bound_names) != self._input_identity:
            raise ValueError("MOSS-Image artifact changed since inspection")

    def status(self):
        q8 = self.artifact.kind.endswith("q8")
        return {
            "object": "mlx2.image.route_receipt",
            "model": MOSS_IMAGE_REPO,
            "model_revision": self.artifact.source_revision,
            "artifact_kind": self.artifact.kind,
            "artifact_fingerprint": self.artifact.fingerprint,
            "execution_kind": "image_diffusion",
            "backend": "native_mlx",
            "implemented": True,
            "qualified": False,
            "selected": False,
            "observed_used": self._observed_uses > 0,
            "observed_uses": self._observed_uses,
            "kv_cache": "not_applicable",
            "apcv2": "not_applicable",
            "streaming": "not_applicable",
            "batch_widths_gt_one": "not_applicable",
            "capabilities": {
                "text_to_image": True,
                "image_edit": False,
                "layout_prompt": True,
                "negative_branch_layer_skip": True,
                "quantized_weights": q8,
            },
        }

    def generate_image(
        self,
        prompt,
        *,
        width=1024,
        height=1024,
        steps=50,
        seed=0,
        guidance_scale=3.0,
        negative_prompt="",
        uncond_skip_layers=(22,),
        **unsupported,
    ):
        with self._lock:
            receipt = {
                **self.status(),
                "operation": "generate",
                "observed_used": False,
                "parameters": {
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "seed": seed,
                    "guidance_scale": guidance_scale,
                    "negative_prompt": negative_prompt,
                    "uncond_skip_layers": (
                        list(uncond_skip_layers)
                        if isinstance(uncond_skip_layers, (list, tuple))
                        else repr(uncond_skip_layers)
                    ),
                },
            }
            self.last_route_receipt = receipt
            try:
                if unsupported:
                    raise ValueError(
                        "unsupported MOSS-Image capabilities: "
                        + ", ".join(sorted(unsupported))
                    )
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("prompt must be a nonempty string")
                if any(
                    type(value) is not int or not 256 <= value <= 2048 or value % 16
                    for value in (width, height)
                ):
                    raise ValueError(
                        "image sides must be 16-aligned integers in 256..2048"
                    )
                if type(steps) is not int or not 1 <= steps <= 100:
                    raise ValueError("steps must be an integer in 1..100")
                if type(seed) is not int or not 0 <= seed < 2**63:
                    raise ValueError("seed must be an integer in [0, 2**63)")
                if (
                    isinstance(guidance_scale, bool)
                    or not isinstance(guidance_scale, (int, float))
                    or not math.isfinite(guidance_scale)
                    or not 0 <= guidance_scale <= 20
                ):
                    raise ValueError("guidance_scale must be finite in 0..20")
                if not isinstance(negative_prompt, str):
                    raise TypeError("negative_prompt must be a string")
                # The publisher documents None as "disable negative-branch skipping".
                layers = () if uncond_skip_layers is None else tuple(uncond_skip_layers)
                if any(
                    type(index) is not int or not 0 <= index < 42 for index in layers
                ):
                    raise ValueError(
                        "uncond_skip_layers must contain block indices in 0..41"
                    )
                receipt["parameters"]["uncond_skip_layers"] = list(layers)
                self._verify_inputs()
                if self._backend is None:
                    if self._backend_factory is None:
                        from .moss_image_pipeline import MOSSImagePipeline

                        self._backend = MOSSImagePipeline(self.artifact.path)
                    else:
                        self._backend = self._backend_factory(self.artifact.path)
                    self._verify_inputs()
                image = self._backend.generate(
                    prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    seed=seed,
                    guidance_scale=float(guidance_scale),
                    negative_prompt=negative_prompt,
                    uncond_skip_layers=layers,
                )
                if image.size != (width, height):
                    raise ValueError(
                        "MOSS-Image backend returned an unexpected image size"
                    )
                self._verify_inputs()
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                data = buffer.getvalue()
                self._observed_uses += 1
                receipt.update(
                    observed_used=True,
                    observed_uses=self._observed_uses,
                    success=True,
                    output_sha256=hashlib.sha256(data).hexdigest(),
                )
                return GeneratedImage(
                    data,
                    "image/png",
                    width,
                    height,
                    self.artifact.fingerprint,
                    seed=seed,
                    steps=steps,
                )
            except Exception as exc:
                receipt.update(success=False, error=f"{type(exc).__name__}: {exc}")
                raise
