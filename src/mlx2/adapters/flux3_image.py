"""Artifact-bound FLUX 3 Image Klein direct image adapter (unqualified).

Inspection is CPU-only and does not import MLX. The native backend is loaded
only by an explicit generation/edit call. See provenance/flux3-image-klein.json.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
from pathlib import Path

from .generative_media import GeneratedImage, MediaArtifact, _file_sha256, _fingerprint

FLUX3_REPO = "black-forest-labs/flux-3-image-klein"
FLUX3_REVISION = "8454017f9ebaada30c998fccd9f67a80c7580f68"


def snapshot_manifest():
    return json.loads(
        (Path(__file__).with_name("assets") / "flux3-image-klein.json").read_text()
    )


def _identity(root):
    identities = {}
    expected = {e["path"] for e in snapshot_manifest()["files"]}
    for path in (root / "text_encoder").rglob("*"):
        if path.is_file() and str(path.relative_to(root)) not in expected:
            raise ValueError("unbound FLUX 3 text encoder input")
    for name in sorted(expected) + [
        ".hf-download-manifest.json",
        ".hf-download-complete.json",
    ]:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"artifact input escapes model root: {name}")
        stat = path.stat()
        identities[name] = (
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
    return identities


def inspect_flux3_image(path):
    root = Path(path).expanduser().resolve()
    expected = snapshot_manifest()
    manifest = json.loads((root / ".hf-download-manifest.json").read_text())
    receipt = json.loads((root / ".hf-download-complete.json").read_text())
    if manifest != expected:
        raise ValueError("FLUX 3 download manifest differs from the pinned artifact")
    if receipt != {
        "repo": FLUX3_REPO,
        "revision": FLUX3_REVISION,
        "files": len(expected["files"]),
        "total_bytes": sum(e["size"] for e in expected["files"]),
    }:
        raise ValueError("FLUX 3 snapshot is incomplete or has another revision")
    before = _identity(root)
    for entry in expected["files"]:
        target = root / entry["path"]
        if target.stat().st_size != entry["size"]:
            raise ValueError(f"incomplete FLUX 3 file: {entry['path']}")
        if entry["sha256"]:
            digest = _file_sha256(target)
            wanted = entry["sha256"]
        else:
            digest = hashlib.sha1(
                f"blob {entry['size']}\0".encode() + target.read_bytes()
            ).hexdigest()
            wanted = entry["git_blob"]
        if digest != wanted:
            raise ValueError(f"FLUX 3 file hash differs: {entry['path']}")
    if _identity(root) != before:
        raise ValueError("FLUX 3 artifact changed during inspection")
    return MediaArtifact(
        "flux-3-image-klein-bf16", root, FLUX3_REVISION, _fingerprint(expected)
    )


class Flux3ImageKleinAdapter:
    def __init__(self, path, *, backend_factory=None):
        root = Path(path).expanduser().resolve()
        before = _identity(root)
        self.artifact = inspect_flux3_image(root)
        if _identity(root) != before:
            raise ValueError("FLUX 3 artifact changed during inspection")
        self._identity = before
        self._backend_factory = backend_factory
        self._backend = None
        self._lock = threading.RLock()
        self._observed_uses = 0
        self.last_route_receipt = self.status()

    def status(self):
        return {
            "object": "mlx2.image.route_receipt",
            "model": FLUX3_REPO,
            "model_revision": self.artifact.source_revision,
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
                "image_edit": True,
                "max_references": 10,
                "flow_map": True,
                "prompt_upsampling": False,
                "lora": False,
                "quantized_variants": False,
            },
        }

    def _verify_inputs(self):
        if _identity(self.artifact.path) != self._identity:
            raise ValueError("FLUX 3 artifact changed since inspection")

    def generate_image(
        self, prompt, *, width=1024, height=1024, steps=16, seed=0, **unsupported
    ):
        return self._run(
            prompt,
            [],
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            unsupported=unsupported,
        )

    def edit_image(
        self,
        prompt,
        image_paths,
        *,
        width=1024,
        height=1024,
        steps=16,
        seed=0,
        **unsupported,
    ):
        return self._run(
            prompt,
            image_paths,
            edit=True,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            unsupported=unsupported,
        )

    def _run(
        self,
        prompt,
        image_paths,
        *,
        edit=False,
        width,
        height,
        steps,
        seed,
        unsupported,
    ):
        with self._lock:
            receipt = {
                **self.status(),
                "operation": "edit" if edit else "generate",
                "observed_used": False,
                "parameters": {
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "seed": seed,
                },
            }
            self.last_route_receipt = receipt
            try:
                if unsupported:
                    raise ValueError(
                        "unsupported FLUX 3 capabilities: "
                        + ", ".join(sorted(unsupported))
                    )
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("prompt must be a nonempty string")
                if any(
                    type(v) is not int or not 64 <= v <= 4096 or v % 16
                    for v in (width, height)
                ):
                    raise ValueError(
                        "image sides must be multiples of 16 between 64 and 4096"
                    )
                if width * height > 4 * 1024 * 1024:
                    raise ValueError("direct image lane is bounded to 4 megapixels")
                if type(steps) is not int or not 1 <= steps <= 100:
                    raise ValueError("steps must be an integer between 1 and 100")
                if type(seed) is not int or not 0 <= seed < 2**63:
                    raise ValueError("seed must be an integer in [0, 2**63)")
                if not isinstance(image_paths, (list, tuple)) or (
                    edit and not 1 <= len(image_paths) <= 10
                ):
                    raise ValueError("editing requires one to ten reference images")
                self._verify_inputs()
                from PIL import Image

                references, hashes = [], []
                for name in image_paths:
                    # Load exactly the bytes that the receipt hashes.
                    raw = Path(name).expanduser().read_bytes()
                    hashes.append(hashlib.sha256(raw).hexdigest())
                    with Image.open(io.BytesIO(raw)) as image:
                        image.load()
                        references.append(image.copy())
                if self._backend is None:
                    if self._backend_factory is None:
                        from .flux3_pipeline import Flux3Pipeline

                        candidate = Flux3Pipeline(self.artifact.path)
                    else:
                        candidate = self._backend_factory(self.artifact.path)
                    self._verify_inputs()
                    self._backend = candidate
                image = self._backend.generate(
                    prompt,
                    references,
                    width=width,
                    height=height,
                    steps=steps,
                    seed=seed,
                )
                self._verify_inputs()
                if image.size != (width, height):
                    raise ValueError("FLUX 3 backend returned an unexpected image size")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                data = buffer.getvalue()
                self._observed_uses += 1
                receipt.update(
                    observed_used=True,
                    observed_uses=self._observed_uses,
                    success=True,
                    reference_sha256=hashes,
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
                    reference_sha256=tuple(hashes),
                )
            except Exception as exc:
                receipt.update(success=False, error=f"{type(exc).__name__}: {exc}")
                raise
