"""Admit a pinned CLI diarization profile only after independent NeMo scoring.

This is deliberately separate from mlx2's text-generation route gate. It
cannot qualify a live-audio or HTTP endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from mlx2.adapters.nemotron3_diarization import inspect_artifact

SCORE_PATTERN = re.compile(
    r"\| FA: ([\d.]+) \| MISS: ([\d.]+) \| CER: ([\d.]+) "
    r"\| DER: ([\d.]+) \| Spk\. Count Acc\. ([\d.]+) "
    r"\| Spk\. Count MAE: ([\d.]+)"
)
APPROVED_CANDIDATE_HARNESS_SHA256 = "97cf87cfaa99c5b9ea32cae68c18587561364144d04712087c2c4a5a1691024b"
APPROVED_LIFECYCLE_HARNESS_SHA256 = "2d29b7fdee38ba794d1180299760591761e510a22eb1a8b58f2848d9a38a4901"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def nemo_score(root: Path, python: Path, manifest: Path,
               hypotheses: Path, log_path: Path) -> dict:
    script = root / "scripts/speaker_tasks/score_diarization.py"
    command = [str(python), str(script), "-r", str(manifest),
               "-h", str(hypotheses), "-c", "0"]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log_path.write_text(result.stdout)
    matches = SCORE_PATTERN.findall(result.stdout)
    if result.returncode or len(matches) != 1:
        raise ValueError(f"NeMo scoring failed or report format changed: {log_path}")
    values = [float(item) for item in matches[0]]
    return dict(zip(("false_alarm", "miss", "confusion", "der",
                     "speaker_count_accuracy", "speaker_count_mae"), values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path,
                        help="directory containing candidate runs for the selected profiles")
    parser.add_argument("--profiles", nargs="+", choices=("offline", "low", "very_low", "ultra_low"),
                        help="profiles to admit; default is every profile in the plan")
    parser.add_argument("--lifecycle", required=True, type=Path)
    parser.add_argument("--nemo-root", required=True, type=Path)
    parser.add_argument("--nemo-python", type=Path,
                        help="isolated Python with pinned NeMo scoring dependencies")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    selected_profiles = args.profiles or plan["profiles"]
    if len(set(selected_profiles)) != len(selected_profiles) or not set(selected_profiles).issubset(plan["profiles"]):
        raise ValueError("selected profiles are duplicated or absent from the plan")
    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not manifest:
        raise ValueError("empty corpus")
    if {row["dataset"] for row in manifest} != set(plan["corpora"]):
        raise ValueError("corpus does not match the preregistered datasets")
    if sha256(args.manifest) != plan["corpus_manifest_sha256"]:
        raise ValueError("corpus manifest differs from the preregistered input")
    for row in manifest:
        for kind in ("audio", "rttm"):
            path = Path(row[f"{kind}_filepath"])
            if not path.is_file() or sha256(path) != row[f"{kind}_sha256"]:
                raise ValueError(f"corpus input changed since preflight: {path}")
    artifact = inspect_artifact(args.model, verify_hash=True)
    lifecycle = json.loads(args.lifecycle.read_text())
    if (lifecycle.get("schema") != "mlx2.diarization-lifecycle.v1" or
            lifecycle.get("passed") is not True or
            lifecycle.get("qualification_harness") != {
                "name": "scripts/check_nemotron3_lifecycle.py",
                "sha256": APPROVED_LIFECYCLE_HARNESS_SHA256} or
            not all(lifecycle["checks"].values()) or
            lifecycle.get("artifact_sha256") != artifact["weight_sha256"] or
            lifecycle.get("adapter_source_sha256") != artifact["source_sha256"]):
        raise ValueError("lifecycle evidence is absent, failed, or stale")
    nemo_root = args.nemo_root.resolve()
    nemo_revision = subprocess.check_output(["git", "-C", str(nemo_root),
                                              "rev-parse", "HEAD"], text=True,
                                             stderr=subprocess.DEVNULL).strip()
    if nemo_revision != plan["nemo_speech_revision"]:
        raise ValueError("NeMo scorer checkout differs from the preregistered revision")
    if subprocess.check_output(["git", "-C", str(nemo_root), "status", "--porcelain",
                                "--untracked-files=no"], text=True,
                               stderr=subprocess.DEVNULL).strip():
        raise ValueError("NeMo scorer checkout has tracked changes")
    # Preserve the venv executable symlink: resolving it selects the base Python
    # and silently discards the isolated NeMo environment.
    nemo_python = args.nemo_python.expanduser().absolute() if args.nemo_python else Path(sys.executable)
    environment_probe = """\
import importlib.metadata as metadata
import json
import nemo
print(json.dumps({'nemo_file': nemo.__file__,
                  'packages': sorted((item.metadata['Name'], item.version)
                                     for item in metadata.distributions())}))
"""
    environment_text = subprocess.check_output(
        [str(nemo_python), "-c", environment_probe], text=True)
    environment = json.loads(environment_text)
    if not Path(environment["nemo_file"]).resolve().is_relative_to(nemo_root):
        raise ValueError("installed NeMo scorer does not come from the pinned checkout")
    if args.output.exists():
        raise ValueError("refusing to overwrite an existing qualification output")
    args.output.mkdir(parents=True)
    (args.output / "nemo-scorer-environment.json").write_text(environment_text)
    qualified = {}
    for profile in selected_profiles:
        candidate_dir = args.candidates / profile
        candidate = json.loads((candidate_dir / "candidate-receipt.json").read_text())
        if (candidate.get("schema") != "mlx2.diarization-corpus-candidate.v2" or
                candidate.get("status") != "candidate_pending_independent_nemo_score_and_lifecycle" or
                candidate.get("state_operation") != plan["state_operation"] or
                candidate.get("qualification_harness") != {
                    "name": "scripts/qualify_nemotron3_corpus.py",
                    "sha256": APPROVED_CANDIDATE_HARNESS_SHA256} or
                candidate.get("candidate_checks_passed") is not True or
                not all(candidate["checks"].values()) or
                candidate.get("plan_sha256") != sha256(args.plan) or
                candidate.get("manifest_sha256") != sha256(args.manifest) or
                candidate.get("artifact_sha256") != artifact["weight_sha256"] or
                candidate.get("adapter_source_sha256") != artifact["source_sha256"] or
                candidate.get("profile") != profile or
                candidate.get("dtype") != plan["dtype"] or
                candidate.get("attention") != plan["attention"] or
                candidate.get("threshold") != plan["threshold"] or
                candidate.get("min_frames") != plan["min_frames"] or
                candidate.get("collar_seconds") != plan["collar_seconds"] or
                candidate.get("ignore_overlap") != plan["ignore_overlap"] or
                candidate.get("recordings") != len(manifest) or
                candidate.get("per_recording_sha256") != sha256(candidate_dir / "per-recording.jsonl") or
                candidate.get("prediction_digests_sha256") != sha256(candidate_dir / "prediction-digests.json") or
                candidate["reference"]["transformers_revision"] != plan["reference_transformers_revision"] or
                candidate["reference"]["device"] != plan["reference_device"] or
                candidate["runtime"]["mlx"] != importlib.metadata.version("mlx") or
                candidate["runtime"]["machine"] != platform.machine()):
            raise ValueError(f"candidate evidence is absent, failed, or stale: {profile}")
        prediction_digests = json.loads((candidate_dir / "prediction-digests.json").read_text())
        scored = {}
        for dataset in plan["corpora"]:
            subset = [row for row in manifest if row["dataset"] == dataset]
            if len(subset) < plan["corpora"][dataset]["minimum_recordings"]:
                raise ValueError(f"missing dataset in manifest: {dataset}")
            stage = args.output / profile / dataset
            stage.mkdir(parents=True)
            subset_manifest = stage / "manifest.jsonl"
            subset_manifest.write_text("".join(json.dumps(row) + "\n" for row in subset))
            scored[dataset] = {}
            for arm in ("reference", "mlx"):
                predictions = stage / arm
                predictions.mkdir()
                hypothesis_manifest = stage / f"{arm}.manifest.jsonl"
                hypothesis_rows = []
                for row in subset:
                    source = candidate_dir / arm / f"{row['recording_id']}.rttm"
                    if (not source.is_file() or
                            sha256(source) != prediction_digests[arm][row["recording_id"]]):
                        raise ValueError(f"missing or changed hypothesis: {source}")
                    destination = predictions / source.name
                    shutil.copyfile(source, destination)
                    hypothesis_rows.append(row | {"rttm_filepath": str(destination)})
                hypothesis_manifest.write_text("".join(json.dumps(row) + "\n" for row in hypothesis_rows))
                official = nemo_score(nemo_root, nemo_python, subset_manifest, hypothesis_manifest,
                                      stage / f"{arm}.nemo.log")
                local = candidate["corpora"][dataset][arm]
                for key in ("der", "speaker_count_accuracy", "speaker_count_mae"):
                    if abs(official[key] - local[key]) > plan["gates"]["maximum_nemo_score_disagreement_absolute"]:
                        raise ValueError(f"local/NeMo scoring disagreement: {profile}/{dataset}/{arm}/{key}")
                for official_key, local_key in (("false_alarm", "false_alarm_seconds"),
                                                ("miss", "miss_seconds"),
                                                ("confusion", "confusion_seconds")):
                    local_rate = local[local_key] / local["reference_speaker_seconds"]
                    if abs(official[official_key] - local_rate) > plan["gates"]["maximum_nemo_score_disagreement_absolute"]:
                        raise ValueError(f"local/NeMo scoring disagreement: {profile}/{dataset}/{arm}/{official_key}")
                scored[dataset][arm] = official
            reference, mlx = scored[dataset]["reference"], scored[dataset]["mlx"]
            if (mlx["der"] > reference["der"] + plan["gates"]["maximum_der_regression_absolute"] or
                    mlx["speaker_count_accuracy"] + plan["gates"]["maximum_speaker_count_accuracy_regression_absolute"] < reference["speaker_count_accuracy"] or
                    mlx["speaker_count_mae"] > reference["speaker_count_mae"] +
                    plan["gates"]["maximum_speaker_count_mae_regression_absolute"]):
                raise ValueError(f"official NeMo quality gate failed: {profile}/{dataset}")
        qualified[profile] = {"candidate_receipt_sha256": sha256(candidate_dir / "candidate-receipt.json"),
                              "official_nemo_scores": scored}
    finalizer = Path(__file__).resolve()
    receipt = {"schema": "mlx2.diarization-cli-qualification.v2",
               "status": "qualified_cli", "qualification_harness": {
                   "name": "scripts/finalize_nemotron3_qualification.py",
                   "sha256": sha256(finalizer)},
               "model_revision": artifact["revision"],
               "artifact_sha256": artifact["weight_sha256"],
               "adapter_source_sha256": artifact["source_sha256"],
               "plan_sha256": sha256(args.plan), "manifest_sha256": sha256(args.manifest),
               "nemo_speech_revision": nemo_revision,
               "state_operation": plan["state_operation"],
               "reference_device": plan["reference_device"],
               "nemo_scorer_environment_sha256": sha256(args.output / "nemo-scorer-environment.json"),
               "lifecycle_sha256": sha256(args.lifecycle),
               "settings": {"dtype": plan["dtype"], "attention": plan["attention"],
                            "threshold": plan["threshold"], "min_frames": plan["min_frames"]},
               "runtime": {"mlx": importlib.metadata.version("mlx"),
                           "machine": platform.machine()},
               "profiles": qualified}
    (args.output / "qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"qualified_cli_profiles": list(qualified),
                      "receipt": str((args.output / "qualification.json").resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
