"""Adapter-owned media policies for Gemma 3n and MiniCPM-o.

The frame sampling and processor boundary follow mlx-vlm revision
653f1f13e238abb313fd45071bbd04b3de414635.  MiniCPM-o controls are derived
from its ``ModelConfig`` and processor at that revision.  mlx2 adds explicit
timing, bounded tile/chunk planning, batching, media cache identity and strict
configuration validation.  No network I/O occurs in this module.
"""

from __future__ import annotations

import math
import threading
import types
from collections import Counter, OrderedDict
from dataclasses import dataclass

from ..multimodal import MediaValue, chunk_audio, media_fingerprint, pcm_to_float32


def _tree_nbytes(value) -> int:
    if isinstance(value, dict):
        return sum(_tree_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tree_nbytes(item) for item in value)
    return int(getattr(value, "nbytes", 0))


class MediaFeatureCache:
    """Bounded LRU for evaluated, projected encoder features."""

    def __init__(self, *, max_entries=20, max_bytes=2 << 30):
        if max_entries < 0 or max_bytes < 0:
            raise ValueError("media feature-cache bounds must be nonnegative")
        self.max_entries, self.max_bytes = int(max_entries), int(max_bytes)
        self.entries, self.bytes = OrderedDict(), 0
        self.counts, self.lock = Counter(), threading.Lock()

    def get(self, key):
        with self.lock:
            value = self.entries.pop(key, None)
            if value is None:
                self.counts["misses"] += 1
                return None
            self.entries[key] = value
            self.counts["hits"] += 1
            return value[0]

    def put(self, key, value):
        size = _tree_nbytes(value)
        if not self.max_entries or not self.max_bytes or size > self.max_bytes:
            with self.lock:
                self.counts["oversize_skips"] += 1
            return False
        with self.lock:
            old = self.entries.pop(key, None)
            if old is not None:
                self.bytes -= old[1]
            self.entries[key] = (value, size)
            self.bytes += size
            while len(self.entries) > self.max_entries or self.bytes > self.max_bytes:
                _, (_, removed) = self.entries.popitem(last=False)
                self.bytes -= removed
                self.counts["evictions"] += 1
            self.counts["stores"] += 1
        return True

    def clear(self):
        with self.lock:
            removed = len(self.entries)
            self.entries.clear(); self.bytes = 0
            self.counts["clears"] += 1
            self.counts["cleared_entries"] += removed

    def snapshot(self):
        with self.lock:
            return {"entries": len(self.entries), "bytes": self.bytes, **self.counts}


def install_media_feature_cache(model, cache: MediaFeatureCache, *, family: str):
    """Cache evaluated vision projections at the model embedding boundary."""
    original = model.get_input_embeddings

    def cached_embeddings(bound, input_ids=None, pixel_values=None, **kwargs):
        key = kwargs.pop("_mlx2_vision_cache_key", None)
        if key is not None and pixel_values is not None:
            features = cache.get(key)
            if features is None:
                if family == "gemma3n":
                    features = bound.get_image_features(
                        pixel_values, bound.vision_tower, bound.config, bound.embed_vision
                    )
                elif family == "minicpmo":
                    features = bound.get_vision_embedding(
                        pixel_values, kwargs.get("tgt_sizes")
                    )
                else:
                    raise ValueError("unsupported media feature-cache family")
                import mlx.core as mx

                mx.eval(features)
                cache.put(key, features)
            kwargs["cached_image_features"] = features
        return original(input_ids=input_ids, pixel_values=pixel_values, **kwargs)

    model.get_input_embeddings = types.MethodType(cached_embeddings, model)
    return model


def install_gemma3n_vision_batching(model, policy: Gemma3nVideoPolicy):
    """Bound Gemma 3n vision-tower forwards while preserving frame order."""
    original = model.get_image_features

    def batched_features(bound, pixel_values, vision_tower, config, embed_vision):
        import mlx.core as mx

        if pixel_values is None or int(pixel_values.shape[0]) <= policy.frame_batch_size:
            return original(pixel_values, vision_tower, config, embed_vision)
        features = [
            original(
                pixel_values[start : start + policy.frame_batch_size],
                vision_tower,
                config,
                embed_vision,
            )
            for start in range(0, int(pixel_values.shape[0]), policy.frame_batch_size)
        ]
        return mx.concatenate(features, axis=0)

    model.get_image_features = types.MethodType(batched_features, model)
    return model


@dataclass(frozen=True, slots=True)
class Gemma3nVideoPolicy:
    fps: float = 2.0
    max_frames: int = 64
    frame_batch_size: int = 16

    def __post_init__(self):
        if not math.isfinite(float(self.fps)) or self.fps <= 0:
            raise ValueError("Gemma 3n video fps must be finite and positive")
        if self.max_frames < 1 or self.frame_batch_size < 1:
            raise ValueError("Gemma 3n video frame bounds must be positive")


@dataclass(frozen=True, slots=True)
class NativeVideoInput:
    frames: tuple
    timestamps_seconds: tuple[float, ...]
    source_fps: float
    fingerprint: str

    @classmethod
    def from_media(cls, media: MediaValue, policy: Gemma3nVideoPolicy):
        if media.kind != "video":
            raise ValueError("Gemma 3n native video requires a video media value")
        frames = tuple(media.value)
        times = tuple(float(v) for v in media.metadata.get("timestamps_seconds", ()))
        if not frames or len(frames) != len(times):
            raise ValueError("video frames and timestamps must be nonempty and aligned")
        if len(frames) > policy.max_frames:
            raise ValueError("decoded video exceeds the Gemma 3n frame policy")
        return cls(
            frames,
            times,
            float(media.metadata["source_fps"]),
            media_fingerprint([media], policy={"family": "gemma3n", "fps": policy.fps, "max_frames": policy.max_frames}),
        )

    def batches(self, size: int):
        if size < 1:
            raise ValueError("video frame batch size must be positive")
        return tuple(self.frames[start : start + size] for start in range(0, len(self.frames), size))

    def processor_inputs(self, text: str, image_token: str) -> dict:
        """Bridge native video to Gemma 3n's image/audio processor contract.

        Gemma 3n has no separate video tower: ordered sampled frames use the
        vision tower.  Timestamp labels keep temporal position observable and
        one video fingerprint keeps caching distinct from unrelated images.
        """
        if not isinstance(text, str):
            raise TypeError("video prompt text must be a string")
        labels = "\n".join(
            f"[video frame {index + 1}/{len(self.frames)} at {timestamp:.3f}s] {image_token}"
            for index, timestamp in enumerate(self.timestamps_seconds)
        )
        return {
            "text": f"{labels}\n{text}" if text else labels,
            "images": list(self.frames),
            "video_metadata": {
                "timestamps_seconds": self.timestamps_seconds,
                "source_fps": self.source_fps,
                "fingerprint": self.fingerprint,
            },
        }


@dataclass(frozen=True, slots=True)
class MiniCPMOExecutionPolicy:
    batch_vision_input: bool = True
    vision_batch_size: int = 16
    slice_mode: bool = True
    max_slice_nums: int = 9
    scale_resolution: int = 448
    audio_chunk_length: float = 1.0
    chunk_input: bool = True
    audio_sample_rate: int = 16_000

    @classmethod
    def from_config(cls, config: dict):
        slice_config = dict(config.get("slice_config") or {})
        return cls(
            batch_vision_input=bool(config.get("batch_vision_input", True)),
            vision_batch_size=int(config.get("vision_batch_size", 16)),
            slice_mode=bool(config.get("slice_mode", True)),
            max_slice_nums=int(slice_config.get("max_slice_nums", 9)),
            scale_resolution=int(slice_config.get("scale_resolution", config.get("image_size", 448))),
            audio_chunk_length=float(config.get("audio_chunk_length", 1.0)),
            chunk_input=bool(config.get("chunk_input", config.get("stream_input", True))),
            audio_sample_rate=int((config.get("audio_config") or {}).get("sampling_rate", 16_000)),
        )

    def __post_init__(self):
        if self.vision_batch_size < 1 or self.max_slice_nums < 1 or self.scale_resolution < 14:
            raise ValueError("MiniCPM-o vision controls must be positive")
        if not math.isfinite(self.audio_chunk_length) or self.audio_chunk_length <= 0:
            raise ValueError("MiniCPM-o audio_chunk_length must be finite and positive")
        if self.audio_sample_rate < 1:
            raise ValueError("MiniCPM-o audio sample rate must be positive")

    def slice_image(self, image) -> tuple:
        """Create a global view plus a bounded aspect-preserving crop grid."""
        from PIL import Image

        image = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
        width, height = image.size
        if not self.slice_mode:
            return (image,)
        area_ratio = width * height / float(self.scale_resolution**2)
        crop_count = min(self.max_slice_nums, max(1, math.ceil(area_ratio)))
        if crop_count == 1:
            return (image,)
        aspect = width / max(height, 1)
        candidates = []
        for rows in range(1, crop_count + 1):
            cols = math.ceil(crop_count / rows)
            if rows * cols <= self.max_slice_nums:
                candidates.append((abs(cols / rows - aspect), rows, cols))
        _, rows, cols = min(candidates)
        tiles = [image.copy()]
        for row in range(rows):
            top, bottom = round(row * height / rows), round((row + 1) * height / rows)
            for col in range(cols):
                left, right = round(col * width / cols), round((col + 1) * width / cols)
                tiles.append(image.crop((left, top, right, bottom)))
        return tuple(tiles[: self.max_slice_nums + 1])

    def vision_batches(self, images) -> tuple:
        expanded = tuple(tile for image in images for tile in self.slice_image(image))
        size = self.vision_batch_size if self.batch_vision_input else 1
        return tuple(expanded[start : start + size] for start in range(0, len(expanded), size))

    def audio_chunks(self, media: MediaValue) -> tuple:
        if int(media.metadata["sample_rate"]) != self.audio_sample_rate:
            raise ValueError(
                f"MiniCPM-o requires {self.audio_sample_rate} Hz WAV audio; explicit resampling is required"
            )
        samples = pcm_to_float32(media)
        if not self.chunk_input:
            return (samples,)
        return chunk_audio(samples, self.audio_sample_rate, chunk_seconds=self.audio_chunk_length)

    def receipt(self) -> dict:
        return {
            "batch_vision_input": self.batch_vision_input,
            "vision_batch_size": self.vision_batch_size,
            "slice_mode": self.slice_mode,
            "max_slice_nums": self.max_slice_nums,
            "audio_chunk_length": self.audio_chunk_length,
            "chunk_input": self.chunk_input,
            "audio_sample_rate": self.audio_sample_rate,
        }


def install_minicpmo_vision_batching(model, policy: MiniCPMOExecutionPolicy):
    """Bind shape-coherent vision batching to a loaded mlx-vlm MiniCPM-o.

    The upstream MLX implementation iterates images one by one even when the
    artifact declares ``batch_vision_input``.  This replacement groups equal
    spatial shapes, bounds each forward by ``vision_batch_size``, then restores
    sample/image order before language-model fusion.
    """
    if not policy.batch_vision_input:
        return model

    def get_vision_embedding(bound, pixel_values, tgt_sizes):
        import mlx.core as mx
        import numpy as np

        if pixel_values is None:
            return []
        dtype = bound.vision_tower.embeddings.patch_embedding.weight.dtype
        flat = []
        outputs = [[] for _ in pixel_values]
        for sample_index, sample in enumerate(pixel_values):
            sample_tgt = tgt_sizes[sample_index] if tgt_sizes is not None else []
            sample_tgt = np.asarray(sample_tgt, dtype=np.int32).reshape(-1, 2)
            for image_index, pixels in enumerate(sample):
                value = pixels if isinstance(pixels, mx.array) else mx.array(pixels)
                value = value.astype(dtype)
                if value.ndim != 3:
                    raise ValueError("MiniCPM-o vision input must be a CHW/HWC image")
                if value.shape[0] == 3:
                    value = value.transpose(1, 2, 0)
                target = (
                    sample_tgt[image_index]
                    if image_index < len(sample_tgt)
                    else np.array([1, max(int(value.shape[1] // bound.config.patch_size), 1)])
                )
                flat.append((sample_index, image_index, value, target))

        groups = {}
        for record in flat:
            key = tuple(int(v) for v in record[2].shape)
            groups.setdefault(key, []).append(record)
        for records in groups.values():
            for start in range(0, len(records), policy.vision_batch_size):
                chunk = records[start : start + policy.vision_batch_size]
                pixels = mx.stack([record[2] for record in chunk], axis=0)
                targets = mx.array(np.stack([record[3] for record in chunk]), dtype=mx.int32)
                patch_lengths = targets[:, 0] * targets[:, 1]
                max_patches = int(mx.max(patch_lengths).item())
                patch_mask = mx.arange(max_patches)[None, :] < patch_lengths[:, None]
                hidden = bound.vision_tower(
                    pixels,
                    patch_attention_mask=patch_mask[:, None, :],
                    tgt_sizes=targets,
                )
                embeddings = bound.resampler(hidden, targets)
                for row, (sample_index, image_index, _, _) in enumerate(chunk):
                    outputs[sample_index].append((image_index, embeddings[row]))
        return [
            mx.stack([value for _, value in sorted(sample)], axis=0) if sample else []
            for sample in outputs
        ]

    model.get_vision_embedding = types.MethodType(get_vision_embedding, model)
    return model
