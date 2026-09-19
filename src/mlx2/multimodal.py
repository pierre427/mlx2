"""Bounded media normalization for adapter-owned multimodal execution.

The image/audio loaders and uniform video sampling are adapted from mlx-vlm
revision 653f1f13e238abb313fd45071bbd04b3de414635 and Rapid-MLX revision
ee101f1bcfb20f343916edd877969aa2a2ac1b98.  mlx2 adds strict byte/frame
bounds, content hashing, no implicit network fetch, and fail-closed decoding.
"""

from __future__ import annotations

import base64
import hashlib
import math
import tempfile
import wave
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable


@dataclass(frozen=True)
class MediaValue:
    kind: str
    mime_type: str
    sha256: str
    byte_count: int
    value: object
    metadata: dict


def decode_data_url(value, *, max_bytes):
    if not isinstance(value, str) or not value.startswith("data:"):
        raise ValueError("media must use a data URL or a local Files API id")
    header, separator, encoded = value.partition(",")
    if not separator or ";base64" not in header:
        raise ValueError("media data URL must use base64")
    mime_type = header[5:].split(";", 1)[0].lower()
    if len(encoded) > ((max_bytes + 2) // 3) * 4 + 8:
        raise ValueError("encoded media exceeds the byte bound")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("media data URL is not valid base64") from error
    if not payload or len(payload) > max_bytes:
        raise ValueError("media must be nonempty and within the byte bound")
    return mime_type, payload


def decode_image(payload, mime_type, *, max_pixels=16_000_000):
    from PIL import Image, ImageOps

    try:
        image = Image.open(BytesIO(payload))
        image.load()
    except (OSError, ValueError) as error:
        raise ValueError(f"failed to decode image: {error}") from error
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if min(width, height) < 3:
        raise ValueError("image minimum dimension is 3 pixels")
    if width * height > max_pixels:
        raise ValueError("image exceeds the pixel bound")
    return MediaValue(
        "image",
        mime_type,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        image,
        {"width": width, "height": height},
    )


def decode_wav_audio(payload, mime_type, *, max_seconds=600):
    try:
        with wave.open(BytesIO(payload), "rb") as stream:
            frames = stream.getnframes()
            rate = stream.getframerate()
            channels = stream.getnchannels()
            width = stream.getsampwidth()
            pcm = stream.readframes(frames)
    except (wave.Error, EOFError) as error:
        raise ValueError(f"failed to decode WAV audio: {error}") from error
    duration = frames / rate if rate else math.inf
    if not rate or channels not in {1, 2} or width not in {1, 2, 3, 4}:
        raise ValueError("unsupported WAV channel or sample format")
    if duration <= 0 or duration > max_seconds:
        raise ValueError("audio duration exceeds the configured bound")
    return MediaValue(
        "audio",
        mime_type,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        pcm,
        {
            "sample_rate": rate,
            "channels": channels,
            "sample_width": width,
            "frames": frames,
            "duration_seconds": duration,
        },
    )


def pcm_to_float32(media: MediaValue):
    """Convert a decoded PCM WAV value to mono float32 without resampling.

    Model processors are required to validate the sample rate they consume.
    Silently resampling here would make request identity and audio timing
    ambiguous, so this helper only normalizes the integer sample encoding.
    """
    if media.kind != "audio":
        raise ValueError("PCM conversion requires an audio media value")
    import numpy as np

    width = int(media.metadata["sample_width"])
    channels = int(media.metadata["channels"])
    if width == 1:
        samples = np.frombuffer(media.value, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(media.value, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(media.value, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raw = np.frombuffer(media.value, dtype=np.uint8).reshape(-1, 3)
        values = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        samples = values.astype(np.float32) / 8388608.0
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(samples, dtype=np.float32)


def chunk_audio(samples, sample_rate, *, chunk_seconds=1.0) -> tuple:
    """Return bounded, lossless consecutive audio chunks for an encoder."""
    import numpy as np

    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if not math.isfinite(float(chunk_seconds)) or chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be finite and positive")
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 1 or not values.size:
        raise ValueError("audio samples must be a nonempty mono vector")
    width = max(1, round(sample_rate * float(chunk_seconds)))
    return tuple(np.ascontiguousarray(values[start : start + width]) for start in range(0, len(values), width))


def uniform_frame_indices(total_frames, source_fps, *, fps=2.0, max_frames=64):
    if total_frames < 1 or source_fps <= 0 or fps <= 0:
        raise ValueError("video metadata and sampling fps must be positive")
    count = min(max_frames, total_frames, max(1, round(total_frames / source_fps * fps)))
    if count == 1:
        return [0]
    return [
        round(index * (total_frames - 1) / (count - 1))
        for index in range(count)
    ]


def decode_video(payload, mime_type, *, fps=2.0, max_frames=64):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("video decoding requires the optional opencv dependency") from error
    suffix = ".mp4" if mime_type == "video/mp4" else ".video"
    with tempfile.NamedTemporaryFile(suffix=suffix) as stream:
        stream.write(payload)
        stream.flush()
        capture = cv2.VideoCapture(stream.name)
        try:
            if not capture.isOpened():
                raise ValueError("failed to open video")
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            indices = uniform_frame_indices(
                total, source_fps, fps=fps, max_frames=max_frames
            )
            frames = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"failed to decode video frame {index}")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
    return MediaValue(
        "video",
        mime_type,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        frames,
        {
            "source_frames": total,
            "source_fps": source_fps,
            "sampled_frames": len(frames),
            "sample_fps": fps,
            "sampled_indices": tuple(indices),
            "timestamps_seconds": tuple(index / source_fps for index in indices),
        },
    )


def media_fingerprint(values: Iterable[MediaValue], *, policy=None) -> str:
    """Stable APCv2/feature-cache identity for ordered media and policy."""
    digest = hashlib.sha256(b"mlx2-media-v1\0")
    for value in values:
        digest.update(value.kind.encode())
        digest.update(b"\0")
        digest.update(value.mime_type.encode())
        digest.update(b"\0")
        digest.update(value.sha256.encode())
        digest.update(b"\0")
    if policy is not None:
        import json

        digest.update(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def resolve_media(source, *, kind, file_loader=None, max_bytes=32 << 20, **options):
    """Resolve a data URL or Files id; network URLs are intentionally refused."""
    if isinstance(source, dict):
        source = source.get("url")
    if isinstance(source, str) and source.startswith("file-"):
        if file_loader is None:
            raise ValueError("media file_id requires a Files API resolver")
        payload, mime_type, _ = file_loader(source)
        if len(payload) > max_bytes:
            raise ValueError("media exceeds the byte bound")
    else:
        mime_type, payload = decode_data_url(source, max_bytes=max_bytes)
    prefixes = {"image": "image/", "audio": "audio/", "video": "video/"}
    if not mime_type.startswith(prefixes[kind]):
        raise ValueError(f"declared {kind} source has incompatible MIME type")
    if kind == "image":
        return decode_image(payload, mime_type, **options)
    if kind == "audio":
        if mime_type not in {"audio/wav", "audio/x-wav", "audio/wave"}:
            raise ValueError("the dependency-free audio path currently accepts WAV")
        return decode_wav_audio(payload, mime_type, **options)
    return decode_video(payload, mime_type, **options)
