"""Hash and validate an audio/RTTM corpus for diarization qualification.

Inputs must already be licensed and prepared as 16 kHz mono PCM WAV. The
resulting JSONL is the immutable list of files evaluated by the harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path

from mlx2.diarization_scoring import read_rttm


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(dataset: str, audio_dir: Path, rttm_dir: Path) -> list[dict]:
    audio = {path.stem: path.resolve() for path in audio_dir.rglob("*.wav")
             if not path.name.startswith("._")}
    labels = {path.stem: path.resolve() for path in rttm_dir.rglob("*.rttm")
              if not path.name.startswith("._")}
    if not audio or audio.keys() != labels.keys():
        missing_audio = sorted(labels.keys() - audio.keys())
        missing_labels = sorted(audio.keys() - labels.keys())
        raise ValueError(f"corpus mismatch or empty: missing_audio={missing_audio[:10]}, "
                         f"missing_rttm={missing_labels[:10]}")
    result = []
    for recording_id in sorted(audio):
        path = audio[recording_id]
        with wave.open(str(path), "rb") as reader:
            if (reader.getnchannels(), reader.getframerate(), reader.getcomptype()) != (1, 16000, "NONE"):
                raise ValueError(f"audio must be 16 kHz mono PCM: {path}")
            if reader.getsampwidth() not in (2, 3, 4):
                raise ValueError(f"unsupported sample width: {path}")
            duration = reader.getnframes() / reader.getframerate()
        segments = read_rttm(labels[recording_id], recording_id=recording_id)
        # VoxConverse's 10 Hz labels can extend up to one annotation frame
        # past the exact PCM endpoint. Keep the original RTTM and score only
        # within the WAV duration; reject larger source mismatches.
        if not segments or max(end for _, end, _ in segments) > duration + 0.1:
            raise ValueError(f"RTTM is empty or exceeds audio duration: {recording_id}")
        result.append({"dataset": dataset, "recording_id": recording_id,
                       "audio_filepath": str(path), "audio_sha256": sha256(path),
                       "rttm_filepath": str(labels[recording_id]),
                       "rttm_sha256": sha256(labels[recording_id]),
                       "duration": duration})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--rttm-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        records = collect(args.dataset, args.audio_dir, args.rttm_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    print(json.dumps({"manifest": str(args.output.resolve()),
                      "sha256": sha256(args.output), "recordings": len(records),
                      "audio_hours": sum(row["duration"] for row in records) / 3600}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
