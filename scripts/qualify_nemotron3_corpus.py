"""Reproducible, per-profile diarization corpus evaluation.

Produces source-bound candidate evidence and RTTMs for the pinned NeMo scorer.
Admission remains closed until the independent NeMo scoring and lifecycle gates
in the qualification plan have been completed and reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from mlx2.adapters.nemotron3_diarization import (
    MODEL_REVISION,
    MODEL_SHA256,
    STREAMING_PROFILES,
    Nemotron3DiarizationAdapter,
    extract_features,
    inspect_artifact,
    segments_from_logits,
)
from mlx2.diarization_scoring import read_rttm, read_uem, score_segments, write_rttm
from mlx2.diarize_cli import read_wav


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _speaker_bin(count: int) -> str:
    if count <= 2:
        return "1-2"
    if count <= 4:
        return "3-4"
    if count <= 8:
        return "5-8"
    return "9+"


def _has_overlap(segments: list[tuple[float, float, str]]) -> bool:
    events = sorted((time, delta, speaker) for start, end, speaker in segments
                    for time, delta in ((start, 1), (end, -1)))
    active = defaultdict(int)
    for _, delta, speaker in events:
        active[speaker] += delta
        if sum(count > 0 for count in active.values()) > 1:
            return True
    return False


def load_inputs(plan_path: Path, manifest_path: Path,
                model_path: Path) -> tuple[dict, list[dict], dict]:
    plan = json.loads(plan_path.read_text())
    if (plan.get("schema") != "mlx2.diarization-plan.v2" or
            plan.get("state_operation") != "approximate_speaker_cache_v1" or
            plan.get("reference_device") != "mps" or
            plan.get("model_revision") != MODEL_REVISION or
            plan.get("model_safetensors_sha256") != MODEL_SHA256 or
            plan.get("dtype") != "float32" or plan.get("attention") != "flash" or
            plan.get("collar_seconds") != 0 or plan.get("ignore_overlap") is not False):
        raise ValueError("qualification plan disagrees with the pinned F32/flash contract")
    if digest(manifest_path) != plan.get("corpus_manifest_sha256"):
        raise ValueError("corpus manifest differs from the preregistered input")
    artifact = inspect_artifact(model_path, verify_hash=True)
    if artifact["revision"] != plan["model_revision"]:
        raise ValueError("artifact revision mismatch")
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty corpus manifest")
    ids = set()
    recording_ids = set()
    coverage = defaultdict(lambda: {"recordings": 0, "audio_hours": 0.0,
                                    "overlap_recordings": 0, "speaker_bins": defaultdict(int)})
    for row in rows:
        name = row["dataset"]
        key = (name, row["recording_id"])
        if key in ids or row["recording_id"] in recording_ids or name not in plan["corpora"]:
            raise ValueError(f"duplicate or unexpected corpus item: {key}")
        ids.add(key)
        recording_ids.add(row["recording_id"])
        for field in ("audio", "rttm"):
            path = Path(row[f"{field}_filepath"])
            if not path.is_file() or digest(path) != row[f"{field}_sha256"]:
                raise ValueError(f"missing or changed {field}: {path}")
        reference = read_rttm(row["rttm_filepath"], recording_id=row["recording_id"])
        if not reference:
            raise ValueError(f"empty reference: {key}")
        coverage[name]["recordings"] += 1
        coverage[name]["audio_hours"] += row["duration"] / 3600
        coverage[name]["speaker_bins"][_speaker_bin(len({speaker for _, _, speaker in reference}))] += 1
        coverage[name]["overlap_recordings"] += _has_overlap(reference)
    for name, minimum in plan["corpora"].items():
        actual = coverage[name]
        if (actual["recordings"] < minimum["minimum_recordings"] or
                actual["audio_hours"] < minimum["minimum_audio_hours"] or
                actual["overlap_recordings"] < minimum.get("minimum_overlap_recordings", 0) or
                any(actual["speaker_bins"][key] < value for key, value in
                    minimum.get("minimum_speaker_bins", {}).items())):
            raise ValueError(f"corpus coverage below preregistered minimum: {name}: {actual}")
    return plan, rows, coverage


def infer_reference(model, features: np.ndarray, profile: str, device: str):
    import torch

    with torch.inference_mode():
        if profile == "offline":
            return model(torch.from_numpy(features)[None].to(device)).logits[0].cpu().numpy()
        chunk, right, _, _ = STREAMING_PROFILES[profile]
        step, lookahead_frames = chunk * 8, right * 8
        cache, parts = None, []
        for start in range(0, len(features), step):
            stop = min(start + step + lookahead_frames, len(features))
            lookahead = right if stop < len(features) else 0
            output = model(torch.from_numpy(features[start:stop])[None].to(device),
                           speaker_cache=cache, num_lookahead_frames=lookahead)
            parts.append(output.logits[0])
            cache = output.speaker_cache
        return torch.cat(parts)[:len(features)].cpu().numpy()


def infer_mlx(adapter, features: np.ndarray, profile: str):
    if profile == "offline":
        logits, _ = adapter.infer_features(features)
        return logits, 0
    chunk, right, _, _ = STREAMING_PROFILES[profile]
    step, lookahead_frames = chunk * 8, right * 8
    cache, parts = None, []
    for start in range(0, len(features), step):
        stop = min(start + step + lookahead_frames, len(features))
        lookahead = right if stop < len(features) else 0
        output, cache = adapter.infer_features(
            features[start:stop], profile=profile, cache=cache,
            num_lookahead_frames=lookahead,
        )
        parts.append(output)
    return np.concatenate(parts)[:len(features)], cache.compression_count


def aggregate(rows: list[dict], arm: str) -> dict:
    score = {key: sum(row[arm][key] for row in rows)
             for key in ("reference_speaker_seconds", "miss_seconds",
                         "false_alarm_seconds", "confusion_seconds", "error_seconds",
                         "speaker_count_correct", "speaker_count_abs_error")}
    score["der"] = score["error_seconds"] / score["reference_speaker_seconds"]
    score["speaker_count_accuracy"] = score["speaker_count_correct"] / len(rows)
    score["speaker_count_mae"] = score["speaker_count_abs_error"] / len(rows)
    return score


def p95_realtime_factor(rows: list[dict]) -> float:
    factors = sorted(row["mlx_realtime_factor"] for row in rows)
    return factors[math.ceil(0.95 * len(factors)) - 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=STREAMING_PROFILES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-checkout", type=Path)
    parser.add_argument("--preflight", action="store_true", help="hash and validate inputs without inference")
    args = parser.parse_args()
    plan, inputs, coverage = load_inputs(args.plan, args.manifest, args.model)
    if args.profile not in plan["profiles"]:
        parser.error("profile is absent from the preregistered plan")
    identity = {"plan_sha256": digest(args.plan), "manifest_sha256": digest(args.manifest),
                "artifact_revision": MODEL_REVISION, "artifact_sha256": MODEL_SHA256,
                "adapter_source_sha256": inspect_artifact(args.model)["source_sha256"],
                "profile": args.profile, "dtype": plan["dtype"], "attention": plan["attention"],
                "threshold": plan["threshold"], "min_frames": plan["min_frames"],
                "collar_seconds": plan["collar_seconds"], "ignore_overlap": False,
                "recordings": len(inputs)}
    if args.preflight:
        print(json.dumps({"status": "ready_for_inference", **identity,
                          "coverage": coverage}, indent=2))
        return 0
    if args.reference_checkout is None:
        parser.error("--reference-checkout is required for inference")
    checkout = args.reference_checkout.resolve()
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                   text=True, stderr=subprocess.DEVNULL).strip()
    if head != plan["reference_transformers_revision"]:
        raise ValueError("Transformers checkout differs from the pinned reference revision")
    if subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain",
                                "--untracked-files=no"], text=True,
                               stderr=subprocess.DEVNULL).strip():
        raise ValueError("Transformers checkout has tracked changes")
    import torch
    import transformers
    from transformers import AutoModelForAudioFrameClassification

    if not Path(transformers.__file__).resolve().is_relative_to(checkout):
        raise ValueError("imported Transformers does not come from the pinned checkout")
    if not torch.backends.mps.is_available():
        raise ValueError("the preregistered MPS reference device is unavailable")
    torch.set_num_threads(4)
    model = AutoModelForAudioFrameClassification.from_pretrained(
        str(args.model), attn_implementation="eager").eval().to(plan["reference_device"])
    adapter = Nemotron3DiarizationAdapter(args.model, dtype="float32", attention="flash")
    args.output.mkdir(parents=True, exist_ok=False)
    for arm in ("reference", "mlx"):
        (args.output / arm).mkdir()
    records = []
    for item in inputs:
        samples = read_wav(item["audio_filepath"])
        if abs(len(samples) / 16000 - item["duration"]) > 1 / 16000:
            raise ValueError(f"manifest duration changed: {item['recording_id']}")
        start = time.perf_counter()
        features = extract_features(samples)
        frontend_seconds = time.perf_counter() - start
        start = time.perf_counter()
        reference_logits = infer_reference(model, features, args.profile,
                                           plan["reference_device"])
        reference_seconds = time.perf_counter() - start + frontend_seconds
        start = time.perf_counter()
        mlx_logits, compressions = infer_mlx(adapter, features, args.profile)
        mlx_seconds = time.perf_counter() - start + frontend_seconds
        if reference_logits.shape != mlx_logits.shape:
            raise ValueError(f"logit shape mismatch: {item['recording_id']}")
        record_id = item["recording_id"]
        truth = read_rttm(item["rttm_filepath"], recording_id=record_id)
        uem = read_uem(item["uem_filepath"], recording_id=record_id) if item.get("uem_filepath") else None
        result = {"dataset": item["dataset"], "recording_id": record_id,
                  "frames": len(reference_logits), "duration": item["duration"],
                  "max_abs_logit_error": float(np.max(np.abs(reference_logits - mlx_logits))),
                  "activity_flips": int(np.count_nonzero((reference_logits > 0) != (mlx_logits > 0))),
                  "activity_decisions": int(reference_logits.size),
                  "cache_compressions": compressions,
                  "reference_seconds": reference_seconds, "mlx_seconds": mlx_seconds,
                  "mlx_realtime_factor": mlx_seconds / item["duration"]}
        for arm, logits in (("reference", reference_logits), ("mlx", mlx_logits)):
            segments = segments_from_logits(logits, threshold=plan["threshold"],
                                            min_frames=plan["min_frames"])
            hypothesis_path = args.output / arm / f"{record_id}.rttm"
            write_rttm(hypothesis_path,
                       recording_id=record_id, segments=segments)
            # NeMo scores the RTTM on disk, including its timestamp precision.
            predicted = read_rttm(hypothesis_path, recording_id=record_id)
            result[arm] = score_segments(truth, predicted, duration=item["duration"],
                                         collar=plan["collar_seconds"], uem=uem)
        records.append(result)
        with (args.output / "per-recording.jsonl").open("a") as stream:
            stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    gates = plan["gates"]
    corpus = {}
    for name in plan["corpora"]:
        group = [row for row in records if row["dataset"] == name]
        recording_regressions = sorted(row["mlx"]["der"] - row["reference"]["der"]
                                       for row in group)
        corpus[name] = {"reference": aggregate(group, "reference"),
                        "mlx": aggregate(group, "mlx"),
                        "mlx_p95_realtime_factor": p95_realtime_factor(group),
                        "p95_recording_der_regression": recording_regressions[
                            math.ceil(0.95 * len(group)) - 1],
                        "max_recording_der_regression": recording_regressions[-1],
                        "activity_flip_fraction": sum(row["activity_flips"] for row in group) /
                        sum(row["activity_decisions"] for row in group)}
    p95 = p95_realtime_factor(records)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak_rss *= 1024
    prediction_digests = {
        arm: {row["recording_id"]: digest(args.output / arm / f"{row['recording_id']}.rttm")
              for row in records}
        for arm in ("reference", "mlx")
    }
    (args.output / "prediction-digests.json").write_text(
        json.dumps(prediction_digests, sort_keys=True, indent=2) + "\n")
    checks = {
        "bounded_activity_divergence": all(item["activity_flip_fraction"] <=
                                           gates["maximum_activity_flip_fraction"]
                                           for item in corpus.values()),
        "der_per_dataset": all(item["mlx"]["der"] <= item["reference"]["der"] + gates["maximum_der_regression_absolute"] for item in corpus.values()),
        "p95_recording_der_regression": all(item["p95_recording_der_regression"] <=
                                            gates["maximum_p95_recording_der_regression_absolute"]
                                            for item in corpus.values()),
        "max_recording_der_regression_by_capacity": all(
            row["mlx"]["der"] - row["reference"]["der"] <= gates[
                "maximum_recording_der_regression_up_to_eight_speakers_absolute"
                if row["reference"]["reference_speakers"] <= 8
                else "maximum_recording_der_regression_nine_plus_speakers_absolute"
            ] for row in records
        ),
        "speaker_count_per_dataset": all(item["mlx"]["speaker_count_accuracy"] + gates["maximum_speaker_count_accuracy_regression_absolute"] >= item["reference"]["speaker_count_accuracy"] for item in corpus.values()),
        "speaker_count_mae_per_dataset": all(item["mlx"]["speaker_count_mae"] <=
                                             item["reference"]["speaker_count_mae"] +
                                             gates["maximum_speaker_count_mae_regression_absolute"]
                                             for item in corpus.values()),
        "realtime_factor": all(item["mlx_p95_realtime_factor"] <=
                               gates["maximum_p95_realtime_factor"]
                               for item in corpus.values()),
        "peak_memory": peak_rss <= gates["maximum_peak_rss_bytes"],
        "cache_compression": args.profile == "offline" or max(row["cache_compressions"] for row in records) >= gates["minimum_cache_compressions_per_streaming_profile"],
    }
    receipt = {"schema": "mlx2.diarization-corpus-candidate.v2",
               "status": "candidate_pending_independent_nemo_score_and_lifecycle",
               "state_operation": plan["state_operation"],
               "qualification_harness": {"name": "scripts/qualify_nemotron3_corpus.py",
                                         "sha256": digest(Path(__file__))},
               "checks": checks, "candidate_checks_passed": all(checks.values()),
               **identity, "corpora": corpus, "p95_realtime_factor": p95,
               "coverage": coverage,
               "peak_rss_bytes": peak_rss,
               "activity_flips": sum(row["activity_flips"] for row in records),
               "activity_decisions": sum(row["activity_decisions"] for row in records),
               "max_abs_logit_error": max(row["max_abs_logit_error"] for row in records),
               "reference": {"transformers_revision": head,
                             "device": plan["reference_device"],
                             "transformers_version": importlib.metadata.version("transformers"),
                             "torch_version": torch.__version__},
               "runtime": {"python": platform.python_version(),
                           "mlx": importlib.metadata.version("mlx"),
                           "machine": platform.machine(), "platform": platform.platform()},
               "per_recording_sha256": digest(args.output / "per-recording.jsonl"),
               "prediction_digests_sha256": digest(args.output / "prediction-digests.json")}
    (args.output / "candidate-receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"candidate_checks_passed": all(checks.values()), "checks": checks,
                      "output": str(args.output.resolve())}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
