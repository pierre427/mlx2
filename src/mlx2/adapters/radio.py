"""Direct RADIO image encoder; separate from the causal serving registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .radio_config import RadioConfig


def inspect_artifact(model_path):
    path = Path(model_path).expanduser().resolve()
    config = RadioConfig.from_dict(json.loads((path / "config.json").read_text()))
    weights = path / "model.safetensors"
    if not weights.is_file():
        raise ValueError("RADIO requires model.safetensors")
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors"):
        digest.update(name.encode())
        with (path / name).open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
    return path, config, digest.hexdigest()


def map_weights(weights):
    """Accept the published remote-code tree; reject partial or unrelated trees."""
    prefix = "radio_model."
    if not weights or any(not key.startswith(prefix) for key in weights):
        raise ValueError("expected published radio_model.* checkpoint keys")
    return {key[len(prefix) :]: value for key, value in weights.items()}


class RadioImageAdapter:
    """Load all encoder tensors strictly and return summary + spatial features.

    This is an encoder API, not a text-generating Nemotron VL route.
    """

    def __init__(self, model_path):
        import mlx.core as mx

        from ..runtime.models.radio import RadioModel

        path, config, fingerprint = inspect_artifact(model_path)
        weights = map_weights(mx.load(str(path / "model.safetensors")))
        indices = weights.pop("summary_idxs", None)
        if indices is None or tuple(indices.tolist()) != config.summary_idxs:
            raise ValueError("checkpoint summary indices disagree with RADIO recipe")
        # Keep the recipe-owned integer buffer out of parameter matching.
        self.model = RadioModel(config)
        weights["summary_idxs"] = mx.array(config.summary_idxs, dtype=mx.int32)
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        self.config = config
        self.receipt = {
            "schema": "mlx2.radio-image-encoder.v1",
            "adapter": type(self).__name__,
            "artifact_fingerprint": fingerprint,
            "source_sha256": {
                str(
                    source.relative_to(Path(__file__).resolve().parents[1])
                ): hashlib.sha256(source.read_bytes()).hexdigest()
                for source in (
                    Path(__file__).resolve(),
                    Path(__file__).with_name("radio_config.py").resolve(),
                    Path(__file__).resolve().parents[1] / "runtime/models/radio.py",
                )
            },
            "capabilities": ["image_encoder"],
            "implemented": True,
            "qualified": False,
            "selected": False,
            "observed_used": False,
            "loaded_tensors": len(weights),
            "summary_idxs": list(config.summary_idxs),
            "num_cls_tokens": config.num_cls_tokens,
            "num_registers": config.num_registers,
        }

    def encode(self, pixels):
        import mlx.core as mx

        output = self.model(pixels)
        mx.eval(output.summary, output.features)
        self.receipt["observed_used"] = True
        self.receipt["summary_shape"] = list(output.summary.shape)
        self.receipt["features_shape"] = list(output.features.shape)
        return output

    def preprocess(self, image, *, size=None):
        """Resize RGB image to an explicit patch-aligned H,W; return raw NCHW."""
        import mlx.core as mx
        import numpy as np
        from PIL import Image

        height, width = size or self.config.preferred_resolution
        if any(
            d < self.config.patch_size
            or d > self.config.max_resolution
            or d % self.config.patch_size
            for d in (height, width)
        ):
            raise ValueError("invalid RADIO preprocessing resolution")
        image = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        return mx.array(
            np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255
        )
