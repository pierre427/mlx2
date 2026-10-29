"""Summarize local bounded diagnostics without promoting qualification.

Raw request/response traces stay outside the repository. This emits compact
metrics and immutable hashes pointing to the original ownership/source receipts.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_receipt(value):
    if value is None:
        return None
    keys = ("provider", "source_identity", "source_revision_token", "qualified",
            "dispatches", "physical_target_forwards", "scheduled_attention_calls",
            "packed_target_rounds", "packed_target_lanes", "cohort_rounds", "cohort_max_width",
            "lane_width_histogram", "target_width_rounds", "attention_layout", "kv_layout",
            "last_memory_estimate", "tree_law", "full_history_dequantization",
            "historical_kv_concat", "historical_kv_padding", "single_batched_attention",
            "approximate_operation_revision", "approximate", "apcv2_publication")
    return {key: value[key] for key in keys if key in value}


def summarize(base):
    units = []
    for folder in sorted(base.iterdir()):
        if not folder.is_dir() or not (folder / "unit.json").exists():
            continue
        completion = folder / "complete.json"
        record = {"name": folder.name, "raw_directory": str(folder),
                  "completed": completion.exists(), "qualification": False,
                  "unit": json.loads((folder / "unit.json").read_text()),
                  "evidence_sha256": {p.name: digest(p) for p in sorted(folder.glob("*.json"))},
                  "cases": {}, "cohorts": []}
        if completion.exists():
            record["completion"] = json.loads(completion.read_text())
        status_path = folder / "status-end.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            record["draft_vocab_guard"] = status.get("execution", {}).get("draft_vocab_guard")
        cases = {}
        for path in sorted(folder.glob("*.json")):
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                continue
            if "response" in value and "case" in value:
                response = value["response"]
                info = response["mlx2"]
                tokens = response["usage"]["completion_tokens"]
                elapsed, ttft = info["elapsed_seconds"], info.get("ttft_seconds")
                rate = ((tokens - 1) / (elapsed - ttft)
                        if ttft is not None and tokens > 1 and elapsed > ttft else None)
                cases.setdefault(value["case"], []).append({
                    "file": path.name, "completion_tokens": tokens,
                    "prompt_tokens": response["usage"]["prompt_tokens"],
                    "elapsed_seconds": elapsed, "ttft_seconds": ttft,
                    "decode_tokens_per_second": rate,
                    "thermal_before": value.get("thermal_before"),
                    "thermal_after": value.get("thermal_after"),
                    "token_ids": info.get("token_trace", {}).get("token_ids"),
                    "approximate_kv": info.get("approximate_kv"),
                    "speculation": {**{k: (info.get("speculation") or {}).get(k)
                                    for k in ("accepted", "proposed", "ordinary_fallback")},
                                    "target_tree": tree_receipt((info.get("speculation") or {}).get("target_tree"))},
                })
            if path.name.startswith("cohort") and "records" in value:
                total = sum(r["response"]["usage"]["completion_tokens"] for r in value["records"])
                trees = [tree_receipt((r["response"]["mlx2"].get("speculation") or {}).get("target_tree"))
                         for r in value["records"]]
                record["cohorts"].append({"file": path.name, "width": value["width"],
                    "repeat": value["repeat"], "wall_seconds": value["wall_seconds"],
                    "aggregate_completion_tokens_per_wall_second": total / value["wall_seconds"],
                    "same_shape_replay_exact": value["same_shape_replay_exact"],
                    "thermal_before": value["thermal_before"], "thermal_after": value["thermal_after"],
                    "target_trees": trees})
        for name, rows in cases.items():
            ids = [r.pop("token_ids") for r in rows]
            rates = [r["decode_tokens_per_second"] for r in rows if r["decode_tokens_per_second"] is not None]
            record["cases"][name] = {"median_decode_tokens_per_second": statistics.median(rates) if rates else None,
                                    "repetitions": len(rows),
                                    "replay_comparisons": len(ids) - 1 if len(ids) > 1 and all(x is not None for x in ids) else 0,
                                    "replay_exact": (all(x == ids[0] for x in ids)
                                                     if len(ids) > 1 and all(x is not None for x in ids) else None),
                                    "rows": rows}
        units.append(record)
    return {"schema": "mlx2.nanbeige43-round2-summary.v1", "qualification": False,
            "scope": "Bounded diagnostics; historical failures retained; no answer-quality score.",
            "decode_rate_definition": "(completion_tokens - 1) / (elapsed_seconds - ttft_seconds)",
            "cohort_rate_definition": "sum(completion_tokens) / cohort wall_seconds, including admission/prefill",
            "units": units}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.base.resolve()), indent=2) + "\n")
