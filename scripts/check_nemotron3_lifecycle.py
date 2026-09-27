"""Exercise the CLI adapter's request isolation and actual equal-length batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from mlx2.adapters.nemotron3_diarization import (
    Nemotron3DiarizationAdapter,
    SpeakerCache,
    extract_features,
)
from mlx2.diarize_cli import read_wav


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--audio-a", required=True, type=Path)
    parser.add_argument("--audio-b", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    samples_a = read_wav(args.audio_a)
    samples_b = read_wav(args.audio_b)
    count = min(len(samples_a), len(samples_b), 5 * 16000)
    if count < 16000 or np.array_equal(samples_a[:count], samples_b[:count]):
        raise ValueError("use two different recordings of at least one second")
    a = extract_features(samples_a[:count])
    b = extract_features(samples_b[:count])
    start = time.perf_counter()
    adapter = Nemotron3DiarizationAdapter(args.model, dtype="float32", attention="flash")
    load_seconds = time.perf_counter() - start
    single_a, _ = adapter.infer_features(a)
    single_b, _ = adapter.infer_features(b)
    batched, _ = adapter.infer_features(np.stack((a, b)))
    batch_error = float(max(np.max(np.abs(batched[0] - single_a)),
                            np.max(np.abs(batched[1] - single_b))))
    batch_flips = int(np.count_nonzero((batched > 0) !=
                                       (np.stack((single_a, single_b)) > 0)))
    foreign = SpeakerCache(264, 222, identity=("foreign", "foreign", "float32", "flash"),
                           profile="low")
    rejects_foreign_cache = False
    try:
        adapter.infer_features(a, profile="low", cache=foreign)
    except ValueError as error:
        rejects_foreign_cache = "different model revision" in str(error)
    rejects_wrong_rate = False
    try:
        extract_features(samples_a[:count], sample_rate=8000)
    except ValueError as error:
        rejects_wrong_rate = "16 kHz" in str(error)
    # A rejected request must not alter the next ordinary inference.
    single_again, _ = adapter.infer_features(a)
    request_error = float(np.max(np.abs(single_again - single_a)))
    first, cache = adapter.infer_features(a[:104], profile="low",
                                          num_lookahead_frames=4)
    last, _ = adapter.infer_features(a[72:120], profile="low", cache=cache)
    partial_final_chunk = bool(first.shape == (72, 8) and last.shape == (48, 8)
                               and np.isfinite(first).all() and np.isfinite(last).all())
    checks = {"batch_width_two_distinct_inputs": batch_error < 0.001 and batch_flips == 0,
              "request_isolation_and_recovery": request_error == 0,
              "partial_final_chunk": partial_final_chunk,
              "foreign_cache_rejected": rejects_foreign_cache,
              "wrong_sample_rate_rejected": rejects_wrong_rate}
    source = Path(__file__).resolve().parents[1] / "src/mlx2/adapters/nemotron3_diarization.py"
    receipt = {"schema": "mlx2.diarization-lifecycle.v1", "checks": checks,
               "passed": all(checks.values()), "load_seconds": load_seconds,
               "qualification_harness": {"name": "scripts/check_nemotron3_lifecycle.py",
                                         "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
               "batch_width": 2, "max_batch_abs_error": batch_error,
               "batch_activity_flips": batch_flips,
               "max_repeat_abs_error": request_error,
               "artifact_sha256": adapter.artifact["weight_sha256"],
               "adapter_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "audio_a": {"path": str(args.audio_a.resolve()),
                           "sha256": hashlib.sha256(args.audio_a.read_bytes()).hexdigest()},
               "audio_b": {"path": str(args.audio_b.resolve()),
                           "sha256": hashlib.sha256(args.audio_b.read_bytes()).hexdigest()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"passed": receipt["passed"], "checks": checks}, indent=2))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
