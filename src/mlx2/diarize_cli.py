"""Dedicated speaker-diarization CLI; does not enter the token-serving router."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

from .adapters.nemotron3_diarization import STREAMING_PROFILES
from .adapters.registry import inspect_audio_model
from .diarization_qualification import load_qualified_diarization_cli


def read_wav(path: str | Path) -> np.ndarray:
    """Read 16 kHz mono PCM WAV and reject implicit resampling or downmixing."""
    with wave.open(str(path), "rb") as reader:
        if reader.getframerate() != 16000 or reader.getnchannels() != 1 or reader.getcomptype() != "NONE":
            raise ValueError("expected uncompressed 16 kHz mono PCM WAV")
        width = reader.getsampwidth()
        frames = reader.readframes(reader.getnframes())
    if width == 2:
        return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768
    if width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        signed = (raw[:, 0].astype(np.int32) |
                  raw[:, 1].astype(np.int32) << 8 |
                  raw[:, 2].astype(np.int32) << 16)
        return ((signed ^ 0x800000) - 0x800000).astype(np.float32) / 8388608
    if width == 4:
        return np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648
    raise ValueError("expected 16, 24, or 32-bit PCM WAV")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mlx2-diarize")
    parser.add_argument("audio", help="16 kHz mono PCM WAV file")
    parser.add_argument("--model", required=True, help="pinned Nemotron-3-Diarization artifact directory")
    parser.add_argument("--profile", choices=STREAMING_PROFILES, default="offline")
    parser.add_argument("--attention", choices=("flash", "eager"), default="flash")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-frames", type=int, default=0)
    parser.add_argument("--output", type=Path, help="write speaker segments as JSON")
    parser.add_argument("--qualification", type=Path,
                        help="source-bound qualified CLI receipt for this exact route")
    args = parser.parse_args(argv)
    try:
        resolution = inspect_audio_model(args.model)
        route_receipt = {"adapter": "Nemotron3DiarizationAdapter",
                          "artifact_sha256": resolution.artifact["weight_sha256"],
                          "source_sha256": resolution.artifact["source_sha256"],
                          "attention": args.attention, "dtype": "float32",
                          "profile": args.profile, "qualification": "candidate"}
        if args.qualification:
            route_receipt = {"adapter": "Nemotron3DiarizationAdapter", **
                             load_qualified_diarization_cli(
                                 args.qualification, artifact=resolution.artifact,
                                 profile=args.profile, attention=args.attention,
                                 dtype="float32", threshold=args.threshold,
                                 min_frames=args.min_frames)}
        adapter = resolution.adapter_type(args.model, attention=args.attention)
        samples = read_wav(args.audio)
        segments = adapter.diarize(samples, profile=args.profile,
                                   threshold=args.threshold, min_frames=args.min_frames)
        rendered = json.dumps({"model": "nvidia/Nemotron-3-Diarization",
                               "model_revision": adapter.artifact["revision"],
                               "route_receipt": route_receipt,
                               "segments": segments}, indent=2)
        if args.output:
            args.output.write_text(rendered + "\n")
        else:
            print(rendered)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
