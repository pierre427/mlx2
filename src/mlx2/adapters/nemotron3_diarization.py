"""MLX execution adapter for NVIDIA's arrival-order speaker diarizer.

The model is an audio frame classifier, not a token generator. It deliberately
does not implement mlx2's text-generation ExecutionAdapter protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..contracts import Capability, ModelDescriptor
from ..process_env import apply_process_numerics

MODEL_REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
MODEL_SHA256 = "c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6"

DIARIZATION_DESCRIPTOR = ModelDescriptor(
    model_type="nemotron3_diarization",
    family="nemotron3_diarization",
    variant="100m-f32",
    state_planes=frozenset(),
    capabilities=frozenset({Capability.AUDIO, Capability.STREAMING}),
    metadata={"task": "speaker_diarization", "sample_rate": 16000, "speakers": 8},
)

STREAMING_PROFILES = {
    "offline": (340, 40, 40, 300),
    "low": (9, 4, 264, 222),
    "very_low": (6, 2, 264, 222),
    "ultra_low": (3, 1, 264, 222),
}


def inspect_artifact(model_path: str | Path, *, verify_hash: bool = False) -> dict:
    """Fail closed on the released topology and optionally verify the full weight hash."""
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    processor = json.loads((path / "processor_config.json").read_text())
    weight = path / "model.safetensors"
    if config.get("model_type") != "nemotron3_diarization":
        raise ValueError("expected Nemotron 3 Diarization artifact")
    audio, head = config["audio_config"], config["head_config"]
    if (
        audio.get("num_hidden_layers"), audio.get("hidden_size"),
        audio.get("num_attention_heads"), audio.get("num_mel_bins"),
        audio.get("subsampling_factor"), head.get("hidden_size"),
        head.get("num_speakers"),
    ) != (31, 512, 8, 128, 8, 192, 8):
        raise ValueError("unsupported Nemotron 3 Diarization topology")
    feature = processor["feature_extractor"]
    if (feature.get("sampling_rate"), feature.get("hop_length"),
        feature.get("n_fft"), feature.get("win_length"),
        feature.get("preemphasis")) != (16000, 160, 512, 400, 0.97):
        raise ValueError("unsupported Nemotron 3 Diarization frontend")
    if not weight.is_file() or weight.stat().st_size != 396954592:
        raise ValueError("missing or incomplete Nemotron 3 Diarization weights")
    if verify_hash:
        with weight.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != MODEL_SHA256:
            raise ValueError("Nemotron 3 Diarization weight hash mismatch")
    return {"path": str(path), "model_type": config["model_type"],
            "revision": MODEL_REVISION, "weight_sha256": MODEL_SHA256,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "qualification": "pending", "config": config}


def _mel_filter_bank() -> np.ndarray:
    """Librosa/Slaney mel filters for the pinned 16 kHz, 512 FFT frontend."""
    def hz_to_mel(hz):
        hz = np.asarray(hz)
        return np.where(hz < 1000, hz / (200 / 3),
                        15 + np.log(np.maximum(hz, 1) / 1000) / (np.log(6.4) / 27))

    def mel_to_hz(mel):
        return np.where(mel < 15, mel * (200 / 3),
                        1000 * np.exp((mel - 15) * (np.log(6.4) / 27)))

    mel = np.linspace(hz_to_mel(0), hz_to_mel(8000), 130)
    hz = mel_to_hz(mel)
    freqs = np.linspace(0, 8000, 257)
    lower = hz[1:-1, None] - hz[:-2, None]
    upper = hz[2:, None] - hz[1:-1, None]
    rising = (freqs[None, :] - hz[:-2, None]) / lower
    falling = (hz[2:, None] - freqs[None, :]) / upper
    weights = np.maximum(0, np.minimum(rising, falling))
    weights *= (2 / (hz[2:] - hz[:-2]))[:, None]
    return weights.astype(np.float32)


_MEL_FILTERS = _mel_filter_bank()
_WINDOW = np.pad(np.hanning(400).astype(np.float32), (56, 56))


def extract_features(samples: np.ndarray, *, sample_rate: int = 16000) -> np.ndarray:
    """Vectorized, reference-compatible log-mel frontend; returns valid 10 ms frames."""
    if sample_rate != 16000:
        raise ValueError("Nemotron 3 Diarization requires 16 kHz audio")
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or len(samples) < 512 or not np.isfinite(samples).all():
        raise ValueError("audio must be a finite mono waveform of at least 512 samples")
    emphasized = samples.copy()
    emphasized[1:] -= 0.97 * samples[:-1]
    padded = np.pad(emphasized, (256, 256))
    frame_count = len(samples) // 160  # reference mask excludes the trailing frame
    strides = (160 * padded.strides[0], padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(padded, shape=(frame_count, 512), strides=strides)
    # Bound temporary FFT arrays for long recordings while retaining vectorization.
    output = np.empty((frame_count, 128), dtype=np.float32)
    for start in range(0, frame_count, 4096):
        stop = min(start + 4096, frame_count)
        spectrum = np.fft.rfft(frames[start:stop] * _WINDOW, axis=-1)
        power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
        output[start:stop] = np.log(power @ _MEL_FILTERS.T + 2**-24)
    return output


def segments_from_logits(logits: np.ndarray, *, threshold: float = 0.5,
                         min_frames: int = 0) -> list[dict]:
    if not 0 < threshold < 1 or min_frames < 0:
        raise ValueError("threshold must lie in (0, 1) and min_frames be nonnegative")
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[1] != 8:
        raise ValueError("expected [frames, 8] speaker logits")
    # Comparing logits avoids an unnecessary full probability tensor.
    active = logits > math.log(threshold / (1 - threshold))
    changes = np.diff(np.pad(active.astype(np.int8), ((1, 1), (0, 0))), axis=0)
    segments = []
    for speaker in range(8):
        starts = np.flatnonzero(changes[:, speaker] == 1)
        ends = np.flatnonzero(changes[:, speaker] == -1)
        segments.extend({"Start": round(int(a) * 0.01, 2),
                         "End": round(int(b) * 0.01, 2), "Speaker": speaker}
                        for a, b in zip(starts, ends) if b - a >= min_frames)
    return sorted(segments, key=lambda s: (s["Start"], s["Speaker"]))


@dataclass
class SpeakerCache:
    """Request-private AOSC and FIFO state; no global cache or cross-request reuse."""

    fifo_length: int
    update_period: int
    identity: tuple[str, str, str, str] | None = None
    profile: str | None = None
    capacity: int = 264
    embeds: object | None = None
    probs: object | None = None
    fifo: object | None = None
    compressed: bool = False
    compression_count: int = 0

    def context(self, mx):
        parts = [x for x in (self.embeds, self.fifo) if x is not None]
        return mx.concatenate(parts, axis=1) if parts else None

    def update(self, mx, input_embeds, logits, chunk_frames: int, silence_embeds):
        factor = 8
        probs = mx.mean(mx.sigmoid(logits).reshape(logits.shape[0], -1, factor, 8), axis=2)
        old_cache = 0 if self.embeds is None else self.embeds.shape[1]
        old_fifo = 0 if self.fifo is None else self.fifo.shape[1]
        chunk = input_embeds[:, old_cache + old_fifo:old_cache + old_fifo + chunk_frames]
        fifo = mx.concatenate([self.fifo, chunk], axis=1) if self.fifo is not None else chunk
        if fifo.shape[1] <= self.fifo_length:
            self.fifo = fifo
            return
        popped = min(fifo.shape[1], max(self.update_period, fifo.shape[1] - self.fifo_length))
        old_probs = self.probs if self.compressed else probs[:, :old_cache]
        cache_embeds = mx.concatenate([self.embeds, fifo[:, :popped]], axis=1) if self.embeds is not None else fifo[:, :popped]
        cache_probs = mx.concatenate([old_probs, probs[:, old_cache:old_cache + popped]], axis=1) if old_cache else probs[:, :popped]
        self.fifo = fifo[:, popped:]
        if cache_embeds.shape[1] > self.capacity:
            cache_embeds, cache_probs = self._compress(mx, cache_embeds, cache_probs, silence_embeds)
            self.compressed = True
            self.compression_count += 1
        self.embeds, self.probs = cache_embeds, cache_probs

    def _compress(self, mx, embeds, probs, silence_embeds):
        # Compression happens only after a FIFO spill. Select indices on the CPU;
        # retain embeddings and all encoder work on the GPU.
        mx.eval(probs)
        # Match the reference cache's F32 score arithmetic. Promoting these
        # nearly equal probabilities to F64 changes top-k frame selection and
        # compounds into speaker drift on longer recordings.
        p = np.asarray(probs, dtype=np.float32)
        batch, frames, speakers = p.shape
        budget = self.capacity // speakers - 1
        selected = []
        for row in range(batch):
            q = p[row]
            log_on = np.log(np.maximum(q, 0.25))
            log_off = np.log(np.maximum(1 - q, 0.25))
            scores = log_on - log_off + log_off.sum(axis=1, keepdims=True) - math.log(0.5)
            speech = q > 0.5
            scores[~speech] = -np.inf
            enough = (scores > 0).sum(axis=0) >= math.floor(budget * 0.5)
            scores[(scores <= 0) & speech & enough[None, :]] = -np.inf
            scores[self.capacity:] += 0.05
            for count, boost in ((math.floor(budget * 0.75), -2 * math.log(0.5)),
                                 (math.floor(budget * 1.5), -math.log(0.5))):
                if count:
                    ix = np.argpartition(scores, -count, axis=0)[-count:]
                    scores[ix, np.arange(speakers)] += boost
            padded = np.concatenate([scores, np.full((1, speakers), np.inf, dtype=np.float32)])
            flat = padded.T.reshape(-1)
            top = np.argpartition(flat, -self.capacity)[-self.capacity:]
            top = np.sort(np.where(flat[top] == -np.inf, (frames + 1) * speakers, top))
            selected.append(np.where(top == (frames + 1) * speakers, frames,
                                     np.minimum(top % (frames + 1), frames)))
        index = mx.array(np.stack(selected), dtype=mx.int32)
        silence = mx.broadcast_to(silence_embeds[None, None, :], (batch, 1, embeds.shape[-1]))
        embeds = mx.concatenate([embeds, silence], axis=1)
        probs = mx.concatenate([probs, mx.zeros((batch, 1, speakers), probs.dtype)], axis=1)
        gather = index[:, :, None]
        return (mx.take_along_axis(embeds, mx.broadcast_to(gather, (batch, self.capacity, embeds.shape[-1])), axis=1),
                mx.take_along_axis(probs, mx.broadcast_to(gather, (batch, self.capacity, speakers)), axis=1))


def _model_classes():
    """Import MLX lazily so artifact inspection remains CPU-only."""
    import mlx.core as mx
    from mlx import nn

    class EncoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_norm1 = nn.LayerNorm(512)
            self.layer_norm2 = nn.LayerNorm(512)
            self.qkv = nn.Linear(512, 1536, bias=False)
            self.o_proj = nn.Linear(512, 512)
            self.fc1 = nn.Linear(512, 2048)
            self.fc2 = nn.Linear(2048, 512)

        def __call__(self, x, cos, sin, *, eager=False):
            residual = x
            hidden = self.layer_norm1(x)
            batch, length, _ = hidden.shape
            qkv = self.qkv(hidden).reshape(batch, length, 3, 8, 64)
            q = qkv[:, :, 0].transpose(0, 2, 1, 3)
            k = qkv[:, :, 1].transpose(0, 2, 1, 3)
            v = qkv[:, :, 2].transpose(0, 2, 1, 3)
            def rotate(t):
                return mx.concatenate([-t[..., 32:], t[..., :32]], axis=-1)
            q = q * cos + rotate(q) * sin
            k = k * cos + rotate(k) * sin
            if eager:
                scores = (q @ k.transpose(0, 1, 3, 2)) * (64**-0.5)
                attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
                result = attn @ v
            else:
                result = mx.fast.scaled_dot_product_attention(q, k, v, scale=64**-0.5)
            x = residual + self.o_proj(result.transpose(0, 2, 1, 3).reshape(batch, length, 512))
            return x + self.fc2(nn.gelu(self.fc1(self.layer_norm2(x))))

    class AudioTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedder = nn.Linear(1024, 512, bias=False)
            self.input_layer_norm = nn.LayerNorm(512)
            self.layers = [EncoderLayer() for _ in range(31)]
            self.layer_norm = nn.LayerNorm(512)

        def __call__(self, x, *, eager=False):
            x = self.input_layer_norm(x)
            length = x.shape[1]
            position = mx.arange(length, dtype=mx.float32)
            inverse = 1 / (10000 ** (mx.arange(0, 64, 2, dtype=mx.float32) / 64))
            angles = position[:, None] * inverse[None, :]
            angles = mx.concatenate([angles, angles], axis=-1)
            cos, sin = mx.cos(angles)[None, None], mx.sin(angles)[None, None]
            for layer in self.layers:
                x = layer(x, cos, sin, eager=eager)
            return self.layer_norm(x)

    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_tower = AudioTower()
            self.proj = nn.Linear(512, 192)
            self.upsampler = nn.Conv1d(192, 1536, 3, padding=1)

        def __call__(self, embeds, *, eager=False):
            x = self.audio_tower(embeds, eager=eager)
            x = self.upsampler(self.proj(x))
            return x.reshape(x.shape[0], -1, 192)

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(192, 192)
            self.out_proj = nn.Linear(192, 8)

        def __call__(self, x):
            return self.out_proj(nn.relu(self.dense(nn.relu(x))))

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Base()
            self.classifier = Classifier()
            self.silence_embeds = mx.zeros((512,))

        def embed(self, features):
            batch, frames, _ = features.shape
            pad = -frames % 8
            if pad:
                features = mx.pad(features, ((0, 0), (0, pad), (0, 0)))
            return self.model.audio_tower.embedder(features.reshape(batch, -1, 1024))

        def classify(self, embeds, *, eager=False):
            return self.classifier(self.model(embeds, eager=eager))

    return Network


class Nemotron3DiarizationAdapter:
    """Standalone MLX diarization adapter with offline and stream-chunk execution."""

    descriptor = DIARIZATION_DESCRIPTOR

    def __init__(self, model_path: str | Path, *, dtype: str = "float32",
                 attention: str = "flash", verify_hash: bool = True):
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported dtype")
        if attention not in {"flash", "eager"}:
            raise ValueError("attention must be flash or eager")
        # Keep the F32 reference path precise unless the caller explicitly
        # selected TF32 before importing MLX. Reduced precision is opt-in.
        if dtype == "float32" and "mlx.core" not in sys.modules:
            apply_process_numerics()  # one owner for TF32: mlx2.process_env
        import mlx.core as mx

        artifact = inspect_artifact(model_path, verify_hash=verify_hash)
        weights = mx.load(str(Path(artifact["path"]) / "model.safetensors"))
        expected = 417
        if len(weights) != expected:
            raise ValueError(f"expected {expected} weight tensors, found {len(weights)}")
        weights = dict(weights)
        for i in range(31):
            prefix = f"model.audio_tower.layers.{i}."
            parts = [weights.pop(prefix + f"self_attn.{name}_proj.weight") for name in "qkv"]
            weights[prefix + "qkv.weight"] = mx.concatenate(parts, axis=0)
            weights[prefix + "o_proj.weight"] = weights.pop(prefix + "self_attn.o_proj.weight")
            weights[prefix + "o_proj.bias"] = weights.pop(prefix + "self_attn.o_proj.bias")
            for name in ("fc1", "fc2"):
                for kind in ("weight", "bias"):
                    weights[prefix + f"{name}.{kind}"] = weights.pop(prefix + f"mlp.{name}.{kind}")
        weights["model.upsampler.weight"] = weights.pop("model.upsampler.conv.weight").transpose(0, 2, 1)
        weights["model.upsampler.bias"] = weights.pop("model.upsampler.conv.bias")
        weights["model.audio_tower.embedder.weight"] = weights.pop("model.audio_tower.embedder.projection.weight")
        # The released checkpoint is F32. Reduced precision is opt-in and must
        # be independently qualified for speaker-boundary fidelity.
        target_dtype = {"float32": mx.float32, "float16": mx.float16,
                        "bfloat16": mx.bfloat16}[dtype]
        if target_dtype != mx.float32:
            weights = {key: value.astype(target_dtype) for key, value in weights.items()}
        Network = _model_classes()
        self.network = Network()
        self.network.load_weights(list(weights.items()), strict=True)
        mx.eval(self.network.parameters())
        self.artifact = artifact
        self.dtype = target_dtype
        self.attention = attention

    def infer_features(self, features: np.ndarray, *, profile: str = "offline",
                       cache: SpeakerCache | None = None,
                       num_lookahead_frames: int = 0):
        """Infer a complete utterance or one streaming feature chunk."""
        import mlx.core as mx

        if profile not in STREAMING_PROFILES:
            raise ValueError(f"unknown diarization profile {profile!r}")
        features = np.asarray(features, dtype=np.float32)
        unbatched = features.ndim == 2
        if unbatched:
            features = features[None]
        if (features.ndim != 3 or features.shape[2] != 128 or
                features.shape[0] < 1 or features.shape[1] < 1 or
                not np.isfinite(features).all()):
            raise ValueError("expected finite [batch, frames, 128] log-mel features")
        chunk_length, right, fifo_length, update_period = STREAMING_PROFILES[profile]
        if profile == "offline" and (cache is not None or num_lookahead_frames):
            raise ValueError("offline inference cannot accept streaming state")
        identity = (self.artifact["weight_sha256"], self.artifact["source_sha256"],
                    str(self.dtype), self.attention)
        if cache is None:
            cache = SpeakerCache(fifo_length, update_period,
                                 identity=identity, profile=profile)
        elif (cache.identity != identity or cache.profile != profile or
              cache.fifo_length != fifo_length or cache.update_period != update_period):
            raise ValueError("speaker cache belongs to a different model revision or route")
        stored = cache.embeds if cache.embeds is not None else cache.fifo
        if stored is not None and stored.shape[0] != features.shape[0]:
            raise ValueError("speaker cache batch width differs from input")
        x = mx.array(features, dtype=self.dtype)
        embeds = self.network.embed(x)
        if not 0 <= num_lookahead_frames < embeds.shape[1]:
            raise ValueError("invalid lookahead frame count")
        scored_frames = embeds.shape[1] - num_lookahead_frames
        if profile != "offline":
            chunk_length, right = scored_frames, num_lookahead_frames
        results = []
        for start in range(0, scored_frames, chunk_length):
            end = min(start + chunk_length, scored_frames)
            current = embeds[:, start:min(end + right, embeds.shape[1])]
            previous = cache.context(mx)
            context_length = 0 if previous is None else previous.shape[1]
            combined = mx.concatenate([previous, current], axis=1) if previous is not None else current
            logits = self.network.classify(combined, eager=self.attention == "eager")
            cache.update(mx, combined, logits, end - start, self.network.silence_embeds)
            results.append(logits[:, context_length * 8:(context_length + end - start) * 8])
        result = mx.concatenate(results, axis=1)[:, :features.shape[1] - num_lookahead_frames * 8]
        mx.eval(result)
        output = np.asarray(result)
        return output[0] if unbatched else output, cache if profile != "offline" else None

    def diarize(self, samples: np.ndarray, *, sample_rate: int = 16000,
                threshold: float = 0.5, min_frames: int = 0,
                profile: str = "offline"):
        features = extract_features(samples, sample_rate=sample_rate)
        if profile == "offline":
            logits, _ = self.infer_features(features)
        else:
            if profile not in STREAMING_PROFILES:
                raise ValueError(f"unknown diarization profile {profile!r}")
            chunk, right, _, _ = STREAMING_PROFILES[profile]
            step = chunk * 8
            lookahead = right * 8
            cache = None
            parts = []
            for start in range(0, len(features), step):
                end = min(start + step + lookahead, len(features))
                future = right if end < len(features) else 0
                part, cache = self.infer_features(
                    features[start:end], profile=profile, cache=cache,
                    num_lookahead_frames=future,
                )
                parts.append(part)
            logits = np.concatenate(parts, axis=0)[:len(features)]
        return segments_from_logits(logits, threshold=threshold, min_frames=min_frames)
