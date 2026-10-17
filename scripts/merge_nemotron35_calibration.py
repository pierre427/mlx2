#!/usr/bin/env python3
"""Merge disjoint completed units; publish full-corpus moments with explicit fallback coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_qualification_matrix import atomic_json

from mlx2.adapters.nemotron35_super import inspect_artifact
from mlx2.conversion.calibration_corpus import verify_corpus
from mlx2.conversion.importance import ActivationMoments, statistics_specs
from mlx2.conversion.nemotron35_plan import make_plan
from mlx2.conversion.nvfp4 import sha256


def text_record_ids(unit):
    """Return one unit's exact MTP source records, rejecting local overlap."""

    image_ids = {sample["sample_id"] for sample in unit["image_samples"]}
    result = []
    for sample in unit["text_samples"]:
        record_ids = sample.get("source_record_ids")
        if (
            not isinstance(record_ids, list)
            or not record_ids
            or any(not isinstance(record_id, str) for record_id in record_ids)
            or len(record_ids) != len(set(record_ids))
            or not set(record_ids) <= image_ids
        ):
            raise ValueError(
                "Calibration MTP packet lacks disjoint unit-bound source records"
            )
        result.extend(record_ids)
    if len(result) != len(set(result)):
        raise ValueError("Calibration unit double-counts MTP source records")
    return tuple(result)


def claim_text_record_ids(seen, unit):
    """Claim one unit's MTP records and reject campaign-level overlap."""

    record_ids = set(text_record_ids(unit))
    overlap = seen & record_ids
    if overlap:
        raise ValueError(
            "Cannot double-count MTP source records across calibration units: "
            + ", ".join(sorted(overlap))
        )
    seen.update(record_ids)


def load_unit(path, *, specs, artifact, corpus):
    path = Path(path).resolve()
    campaign = json.loads((path / "campaign.json").read_text())
    unit = json.loads((path / "calibration/unit.json").read_text())
    if (campaign.get("status") != "completed" or campaign.get("source_unchanged") is not True
            or campaign.get("dependencies_unchanged") is not True
            or campaign.get("owner_after_error") or campaign.get("cleanup_errors")
            or campaign.get("route") != "ordinary" or campaign.get("active_sessions") != 1
            or campaign.get("functional_passed") is not True
            or campaign.get("calibration_unit") != unit
            or campaign.get("target", {}).get("fingerprint") != artifact
            or unit.get("eligible_for_merge") is not True or unit.get("source_unchanged") is not True
            or unit.get("state") not in {"completed", "yielded"}
            or unit.get("binding", {}).get("proxy_artifact") != artifact
            or unit.get("binding", {}).get("corpus_sha256") != corpus):
        raise ValueError("Calibration unit is incomplete or bound to another source/artifact/corpus")
    moments = ActivationMoments.load(path / "calibration/moments", specs=specs)
    if (moments.binding != unit["binding"] or moments.coverage() != unit["coverage"]
            or not moments.binding.get("dependencies")
            or moments.binding["dependencies"] != campaign["dependencies"]
            or moments.binding["collector_sha256"] != campaign["source"]["sha256"]):
        raise ValueError("Calibration unit statistics differ from their execution receipt")
    expected = {sample["sample_id"]: {"modality": sample["modality"], "input_sha256": sample["input_sha256"]}
                for sample in unit["image_samples"] + unit["text_samples"]}
    if (moments.samples != expected or len(expected) != len(unit["image_samples"]) + len(unit["text_samples"])
            or any(sample.get("contamination") for sample in unit["image_samples"] + unit["text_samples"])):
        raise ValueError("Calibration sample receipt closure or host guards differ")
    text_record_ids(unit)
    return moments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("units", type=Path, nargs="+")
    args = parser.parse_args()
    if args.report.exists() or args.destination.exists():
        parser.error("Refusing to replace calibration output/report evidence")
    artifact = inspect_artifact(args.model)
    corpus, records = verify_corpus(args.corpus)
    specs = statistics_specs(make_plan(artifact["config"], expert_bits=artifact["conversion"]["expert_bits"]))
    merged, sources, merged_text_records = None, [], set()
    for path in args.units:
        moments = load_unit(path, specs=specs, artifact=artifact["identity"]["fingerprint"], corpus=corpus["corpus_sha256"])
        unit = json.loads((path / "calibration/unit.json").read_text())
        claim_text_record_ids(merged_text_records, unit)
        if merged is None:
            merged = moments
        else:
            merged.merge(moments)
        sources.append({"path": str(path.resolve()), "campaign_sha256": sha256(path / "campaign.json"),
                        "unit_sha256": sha256(path / "calibration/unit.json"),
                        "moments_manifest_sha256": sha256(path / "calibration/moments/manifest.json")})
    actual_images = {key for key, sample in merged.samples.items() if sample["modality"] == "image_text"}
    expected_images = {record["sample_id"] for record in records}
    if not actual_images <= expected_images:
        raise ValueError("Calibration used image sample IDs outside the verified corpus")
    coverage = merged.coverage()
    corpus_complete = actual_images == expected_images and len(actual_images) >= 1024
    ready = corpus_complete and coverage["complete"]
    report = {"schema": "mlx2.nemotron35-calibration-merge.v1", "binding": merged.binding,
              "units": sources, "coverage": coverage, "missing_image_samples": sorted(expected_images - actual_images),
              "corpus_complete": corpus_complete,
              "ready_for_hybrid_conversion": corpus_complete,
              "ready_for_weighted_conversion": ready,
              "fallback_policy": {"minimum_observations": coverage["minimum_required"],
                                  "undercovered_experts": "stored-BF16 weight-MSE Q4"},
              "qualified": False, "performance_evidence": False}
    if corpus_complete:
        merged.require_image_samples()
        merged.save(args.destination)
        report["published"] = str(args.destination.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, report)
    print(json.dumps({"ready_for_hybrid_conversion": corpus_complete,
                      "ready_for_weighted_conversion": ready, "image_samples": len(actual_images),
                      "expected_image_samples": len(expected_images), "expert_coverage_complete": coverage["complete"]}))


if __name__ == "__main__":
    main()
