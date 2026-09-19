#!/usr/bin/env python3
"""Host-only qualification of real Gemma 3n and MiniCPM-o processors.

The harness deliberately replaces ``mlx_vlm.models.base`` with the three
processor helpers it needs.  Importing mlx-vlm's package root imports MLX and
selects a device; processor qualification must not do that or load weights.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
import wave
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from mlx2.adapters.multimodal import (
    Gemma3nVideoPolicy,
    MiniCPMOExecutionPolicy,
    NativeVideoInput,
)
from mlx2.multimodal import MediaValue, decode_wav_audio, media_fingerprint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_processor_only_namespace(source: Path) -> None:
    def package(name: str, package_path: Path) -> None:
        module = types.ModuleType(name)
        module.__path__ = [str(package_path)]
        sys.modules[name] = module

    package("mlx_vlm", source / "mlx_vlm")
    package("mlx_vlm.models", source / "mlx_vlm" / "models")
    for family in ("gemma3n", "minicpmo"):
        package(
            f"mlx_vlm.models.{family}",
            source / "mlx_vlm" / "models" / family,
        )

    base = types.ModuleType("mlx_vlm.models.base")

    def load_chat_template(tokenizer, model_path):
        root = Path(model_path)
        if (root / "chat_template.json").exists():
            tokenizer.chat_template = json.loads(
                (root / "chat_template.json").read_text()
            )["chat_template"]
        elif (root / "chat_template.jinja").exists():
            tokenizer.chat_template = (root / "chat_template.jinja").read_text()
        return tokenizer

    base.load_chat_template = load_chat_template
    base.to_mlx = lambda data: data
    base.install_auto_processor_patch = lambda *_args, **_kwargs: None
    sys.modules[base.__name__] = base


def _load_processor_module(source: Path, family: str):
    name = f"mlx_vlm.models.{family}.processing_{family}"
    source_path = source / "mlx_vlm" / "models" / family / f"processing_{family}.py"
    spec = importlib.util.spec_from_file_location(name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load processor module {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _media_end(processed: dict) -> int:
    types_value = processed.get("token_type_ids")
    if types_value is not None:
        positions = np.where(np.asarray(types_value).reshape(-1) != 0)[0]
        if positions.size:
            return int(positions[-1]) + 1
    end = 0
    for name in ("image_bound", "audio_bounds"):
        for bounds in processed.get(name) or ():
            values = np.asarray(bounds).reshape(-1, 2)
            if values.size:
                end = max(end, int(values[:, 1].max()))
    return end


def _gemma_check(module, artifact: Path) -> dict:
    # Transformers 5 routes AutoImageProcessor through the Torchvision-backed
    # fast class even when the configured SigLIP slow processor is sufficient.
    # Instantiate the same configured components directly so this remains a
    # NumPy/Pillow processor check and cannot pull in a tensor backend.
    from transformers import AutoTokenizer
    from transformers.models.gemma3n.feature_extraction_gemma3n import (
        Gemma3nAudioFeatureExtractor,
    )
    from transformers.models.siglip.image_processing_siglip import (
        SiglipImageProcessor,
    )

    processor_config = json.loads((artifact / "processor_config.json").read_text())
    processor = module.Gemma3nProcessor(
        feature_extractor=Gemma3nAudioFeatureExtractor.from_pretrained(
            artifact, local_files_only=True
        ),
        image_processor=SiglipImageProcessor.from_pretrained(
            artifact, local_files_only=True
        ),
        tokenizer=AutoTokenizer.from_pretrained(
            artifact, local_files_only=True, trust_remote_code=False
        ),
        audio_seq_length=int(processor_config.get("audio_seq_length", 188)),
        image_seq_length=int(processor_config.get("image_seq_length", 256)),
    )
    frames = (
        np.zeros((64, 96, 3), dtype=np.uint8),
        np.full((64, 96, 3), 127, dtype=np.uint8),
        np.full((64, 96, 3), 255, dtype=np.uint8),
    )
    media = MediaValue(
        "video",
        "video/mp4",
        hashlib.sha256(b"gemma-processor-fixture").hexdigest(),
        1,
        frames,
        {
            "source_fps": 30.0,
            "sampled_indices": (0, 30, 60),
            "timestamps_seconds": (0.0, 1.0, 2.0),
        },
    )
    policy = Gemma3nVideoPolicy(max_frames=8, frame_batch_size=2)
    native = NativeVideoInput.from_media(media, policy)
    prepared = native.processor_inputs("Describe the changes.", processor.tokenizer.image_token)
    processed = dict(processor(text=prepared["text"], images=prepared["images"]))
    pixel_values = np.asarray(processed["pixel_values"])
    token_types = np.asarray(processed["token_type_ids"])
    image_tokens = int(np.count_nonzero(token_types == 1))
    expected_tokens = len(frames) * int(processor.image_seq_length)
    assert pixel_values.shape[0] == len(frames)
    assert image_tokens == expected_tokens
    assert _media_end(processed) > 0
    return {
        "passed": True,
        "artifact": str(artifact),
        "config_sha256": _sha256(artifact / "config.json"),
        "processor_class": type(processor).__name__,
        "frames": len(frames),
        "timestamps_seconds": list(native.timestamps_seconds),
        "pixel_values_shape": list(pixel_values.shape),
        "image_soft_tokens": image_tokens,
        "media_token_end": _media_end(processed),
        "frame_batch_size": policy.frame_batch_size,
        "fingerprint": native.fingerprint,
    }


def _wav_media(seconds: float, sample_rate: int = 16_000) -> MediaValue:
    payload = BytesIO()
    with wave.open(payload, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(b"\0\0" * round(seconds * sample_rate))
    return decode_wav_audio(payload.getvalue(), "audio/wav")


def _minicpmo_check(module, artifact: Path) -> dict:
    config = json.loads((artifact / "config.json").read_text())
    policy = MiniCPMOExecutionPolicy.from_config(config)
    processor = module.MiniCPMOProcessor.from_pretrained(
        artifact, local_files_only=True, trust_remote_code=False
    )
    image = Image.new("RGB", (896, 448), color=(12, 34, 56))
    slices = policy.slice_image(image)
    audio = _wav_media(2.25, policy.audio_sample_rate)
    chunks = policy.audio_chunks(audio)
    text = "".join("<image>" for _ in slices) + "".join(
        "<audio>" for _ in chunks
    ) + "Describe the scene and sound."
    processed = dict(
        processor(text=text, images=list(slices), audios=list(chunks))
    )
    pixels = processed["pixel_values"]
    image_bounds = np.asarray(processed["image_bound"][0]).reshape(-1, 2)
    audio_bounds = np.asarray(processed["audio_bounds"][0]).reshape(-1, 2)
    feature_lens = list(processed["audio_feature_lens"][0])
    assert len(pixels) == 1 and len(pixels[0]) == len(slices)
    assert len(image_bounds) == len(slices)
    assert len(audio_bounds) == len(chunks)
    assert len(feature_lens) == len(chunks)
    assert sum(len(chunk) for chunk in chunks) == int(audio.metadata["frames"])
    fingerprint = media_fingerprint(
        [audio], policy={"family": "minicpmo", **policy.receipt()}
    )
    return {
        "passed": True,
        "artifact": str(artifact),
        "config_sha256": _sha256(artifact / "config.json"),
        "processor_class": type(processor).__name__,
        "vision_slices": len(slices),
        "vision_batches": len(policy.vision_batches([image])),
        "pixel_values_shapes": [list(np.asarray(value).shape) for value in pixels[0]],
        "image_bounds": image_bounds.tolist(),
        "audio_chunks": len(chunks),
        "audio_chunk_samples": [len(chunk) for chunk in chunks],
        "audio_feature_lengths": feature_lens,
        "audio_bounds": audio_bounds.tolist(),
        "media_token_end": _media_end(processed),
        "policy": policy.receipt(),
        "fingerprint": fingerprint,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx-vlm-source", type=Path, required=True)
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--minicpmo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.mlx_vlm_source.resolve()
    import torch

    torch_default_device = str(torch.get_default_device())
    if torch_default_device != "cpu":
        raise RuntimeError(
            f"processor qualification requires torch default device cpu, got {torch_default_device}"
        )
    _install_processor_only_namespace(source)
    gemma_module = _load_processor_module(source, "gemma3n")
    minicpmo_module = _load_processor_module(source, "minicpmo")
    source_revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    receipt = {
        "schema": "mlx2.multimodal-processor-qualification.v1",
        "passed": True,
        "scope": "processor_only_no_model_weights",
        "gpu_used": torch_default_device != "cpu",
        "torch_default_device": torch_default_device,
        "metal_imported": "mlx" in sys.modules or "mlx.core" in sys.modules,
        "source": {
            "repository": "Blaizzy/mlx-vlm",
            "revision": source_revision,
            "path": str(source),
        },
        "checks": {
            "gemma3n_video_processor": _gemma_check(gemma_module, args.gemma.resolve()),
            "minicpmo_media_processor": _minicpmo_check(
                minicpmo_module, args.minicpmo.resolve()
            ),
        },
    }
    receipt["metal_imported"] = "mlx" in sys.modules or "mlx.core" in sys.modules
    if receipt["metal_imported"]:
        raise RuntimeError("processor qualification imported MLX/Metal")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
