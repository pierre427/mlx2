#!/usr/bin/env python3
"""Run the frozen 20x20 paired transcript-rebuild suite through mlx2.

This is a client-side evaluation.  It rebuilds complete HTTP chat transcripts;
it never mutates live KV, recurrent, APCv2, MTP, or external-draft state.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import threading
import time
from typing import Any

from run_qualification_matrix import (
    Client,
    activate,
    applicable_requirements,
    atomic_json,
    digest,
    host_identity,
    identity,
    observed_width,
    request_evidence,
    stable_host_identity,
    validate_counter_deltas,
    validate_manifest,
    validate_requirements,
    validate_bound_qualification_receipt,
)


SCHEMA = "mlx2.spomin-20x20-exact-rebuild.v1"
CORPUS_SCHEMA = "mlx-uag.spomin-20x20-corpus.v1"
FROZEN_CORPUS_SHA256 = "5d7233f805b310e354cb628a0d0b1d12299cc90dbfeebb890eaf35dac966ad12"
NEEDLE_TARGETS = {"early": 0.08, "middle": 0.45, "late": 0.72}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed_cell_ids(report: dict[str, Any]) -> set[str]:
    return {key for key, row in report.get("cells", {}).items() if row.get("passed")}


def rows_for_completed_cells(report: dict[str, Any]) -> list[dict[str, Any]]:
    completed = completed_cell_ids(report)
    return [row for row in report.get("rows", [])
            if f"{row['domain']}:{row['transcript_arm']}" in completed]


def needle_values(case_id: str) -> dict[str, str]:
    values = {}
    for label in NEEDLE_TARGETS:
        value = hashlib.sha256(f"{case_id}:needle:{label}".encode()).hexdigest().upper()
        values[label] = f"NDL-{label[0].upper()}-{value[:5]}-{value[5:10]}"
    return values


def sentinel(case_id: str) -> str:
    value = hashlib.sha256(f"{case_id}:sentinel".encode()).hexdigest().upper()
    return f"SPM20-{value[:5]}-{value[5:10]}"


def filler_text(case: dict[str, Any], index: int, role: str) -> str:
    nonce = hashlib.sha256(f"{case['case_id']}:{index}:{role}".encode()).hexdigest()[:14]
    speaker = "operator" if role == "user" else "archive assistant"
    return (f'<archived_turn index="{index:03d}" nonce="{nonce}">\n'
            f"{speaker}: Routine {case['domain']} status record {index}. This old record "
            "is background only and does not answer the final request. It lists stable "
            "queue counters, ordinary retry bookkeeping, generic maintenance notes, "
            "non-authoritative observations, and repeated audit language. Preserve its "
            "chronological location but do not infer any requested audit token from it.\n"
            "</archived_turn>")


def needle_text(case_id: str, label: str, value: str) -> str:
    return (f'<authoritative_audit_note position="{label}" case="{case_id}">\n'
            f"The {label.upper()} audit token is {value}. Retain this exact value for the "
            "final audit response. This is authoritative even though later background "
            "messages follow it.\n</authoritative_audit_note>")


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def prepare_case(case: dict[str, Any], tokenizer: Any, capacity: int) -> dict[str, Any]:
    fill_target = int(capacity * 0.78)
    compact_target = int(capacity * 0.65)
    needles = needle_values(case["case_id"])
    code = sentinel(case["case_id"])
    query = ("<final_request>\nWithout quoting this request, report the three audit tokens "
             "found in the authoritative notes using exactly `EARLY=... MIDDLE=... LATE=...`. "
             f"Then answer this domain question concisely: {case['question']}\n"
             f"End with the exact line CODE: {code}\n</final_request>")
    query_width = len(token_ids(tokenizer, query))
    segments: list[dict[str, Any]] = []
    cursor = filler_index = 0

    def append(segment_id: str, role: str, content: str, protected: bool = False) -> None:
        nonlocal cursor
        width = len(token_ids(tokenizer, content))
        segments.append({"id": segment_id, "role": role, "content": content,
                         "token_start": cursor, "token_stop": cursor + width,
                         "tokens": width, "protected": protected})
        cursor += width

    def next_role() -> str:
        return "user" if len(segments) % 2 == 0 else "assistant"

    for label, ratio in NEEDLE_TARGETS.items():
        while cursor < int(fill_target * ratio):
            filler_index += 1
            role = next_role()
            append(f"filler:{filler_index:03d}", role, filler_text(case, filler_index, role))
        append(f"needle:{label}", next_role(), needle_text(case["case_id"], label, needles[label]), True)
    while cursor + query_width < fill_target:
        filler_index += 1
        role = next_role()
        append(f"filler:{filler_index:03d}", role, filler_text(case, filler_index, role))
    if next_role() != "user":
        append("query_bridge", "assistant", "The archived context is loaded. I will answer the next final request.")
    append("query", "user", query, True)

    visible = list(segments)
    removed = []
    # The frozen source's synthetic hotspot scorer gives every filler the same
    # low score.  Stable chronological removal is therefore the exact ranking.
    for segment in segments:
        if sum(item["tokens"] for item in visible) <= compact_target:
            break
        if not segment["protected"] and segment["id"] != "query_bridge":
            visible.remove(segment)
            removed.append(segment)
    projected = sum(item["tokens"] for item in visible)
    if projected > compact_target:
        raise RuntimeError(f"{case['case_id']}: compaction shortfall {projected - compact_target}")
    full_messages = [{"role": row["role"], "content": row["content"]} for row in segments]
    compacted_messages = [{"role": row["role"], "content": row["content"]} for row in visible]
    return {"case": case, "sentinel": code, "needles": needles,
            "full_messages": full_messages, "compacted_messages": compacted_messages,
            "receipt": {"trigger": "pressure", "strategy": "lowest_importance",
                        "capacity_tokens": capacity, "full_ledger_tokens": cursor,
                        "target_limit_tokens": compact_target, "projected_target_tokens": projected,
                        "full_message_count": len(full_messages),
                        "compacted_message_count": len(compacted_messages),
                        "selected_segment_ids": [row["id"] for row in removed],
                        "protected_segment_ids": [row["id"] for row in segments if row["protected"]],
                        "needle_selected": {name: f"needle:{name}" in {row["id"] for row in removed}
                                            for name in NEEDLE_TARGETS},
                        "query_selected": "query" in {row["id"] for row in removed},
                        "reclaimed_tokens": cursor - projected, "shortfall_tokens": 0,
                        "source_transcript_digest": digest(full_messages),
                        "result_transcript_digest": digest(compacted_messages),
                        "physical_kv_surgery": False,
                        "state_operation": "exact HTTP transcript rebuild"}}


def concept_hits(text: str, groups: list[list[str]]) -> tuple[int, int]:
    lowered = text.lower()
    return sum(any(term.lower() in lowered for term in group) for group in groups), len(groups)


def lexical_jaccard(left: str, right: str, code: str) -> float:
    pattern = re.compile(r"[a-z0-9]+")
    excluded = set(pattern.findall(code.lower()))
    a, b = set(pattern.findall(left.lower())) - excluded, set(pattern.findall(right.lower())) - excluded
    return len(a & b) / len(a | b) if a | b else 1.0


def make_body(system: str, prepared: dict[str, Any], transcript_arm: str, max_tokens: int) -> dict[str, Any]:
    return {"messages": [{"role": "system", "content": system}]
                        + prepared[f"{transcript_arm}_messages"],
            "temperature": 0, "top_p": 1, "top_k": 0,
            "max_tokens": max_tokens, "enable_thinking": False}


def run_domain(client: Client, group: list[dict[str, Any]], system: str, transcript_arm: str,
               max_tokens: int, serving_arm: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    before = client.get("/v1/status")
    validate_requirements(before, serving_arm.get("status_requirements", []), "status")
    if int(before.get("max_lanes", 0)) < 20:
        raise AssertionError(f"server max_lanes={before.get('max_lanes')} cannot run B20")
    bodies = [make_body(system, item, transcript_arm, max_tokens) for item in group]
    warmup_result = client.post(bodies[0])
    validate_requirements(warmup_result["mlx2"], serving_arm["receipt_requirements"], "warmup receipt")
    warmup = request_evidence(warmup_result)
    rows = []
    barrier = threading.Barrier(21)
    starts: dict[str, float] = {}
    lock = threading.Lock()
    def synchronized_post(body, item):
        barrier.wait(timeout=30)
        started = time.monotonic()
        with lock:
            starts[item["case"]["case_id"]] = started
        return client.post(body)
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(synchronized_post, body, item): item
                   for body, item in zip(bodies, group)}
        barrier.wait(timeout=30)
        for future in as_completed(futures):
            item, result = futures[future], future.result()
            validate_requirements(result["mlx2"], serving_arm["receipt_requirements"], "receipt")
            evidence = request_evidence(result)
            text = result["choices"][0]["message"].get("content", "")
            evidence.update({"case_id": item["case"]["case_id"], "domain": item["case"]["domain"],
                             "transcript_arm": transcript_arm, "text": text,
                             "sentinel_present": item["sentinel"].lower() in text.lower(),
                             "needles_present": {name: value.lower() in text.lower()
                                                 for name, value in item["needles"].items()}})
            rows.append(evidence)
    first_start = min(starts.values())
    start_spread = max(starts.values()) - first_start
    if start_spread > 0.25:
        raise AssertionError(f"Spomin B20 client start spread {start_spread:.6f}s exceeded bound")
    for row in rows:
        row["request_start_offset_seconds"] = starts[row["case_id"]] - first_start
        row["batch_start_spread_seconds"] = start_spread
    after = client.get("/v1/status")
    if identity(before) != identity(after):
        raise AssertionError("server identity changed during Spomin B20 domain batch")
    widths = [observed_width(row["receipt"]) for row in rows]
    if max(widths, default=0) < 2:
        raise AssertionError(f"Spomin B20 did not compose into a multi-request compute batch: {sorted(set(widths))}")
    context_tokens = max(item["receipt"]["full_ledger_tokens"] for item in group)
    counter_requirements = applicable_requirements(
        serving_arm.get("counter_requirements", []),
        {"suite": "spomin20x20", "context_tokens": context_tokens},
    )
    validate_counter_deltas(before, after, counter_requirements)
    validate_requirements(after, serving_arm.get("after_status_requirements", []), "final status")
    return sorted(rows, key=lambda row: row["case_id"]), before, after, warmup


def acceptance(prepared: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["case_id"], row["transcript_arm"]): row for row in rows}
    comparisons = []
    for item in prepared:
        full, compacted = by_key[(item["case"]["case_id"], "full")], by_key[(item["case"]["case_id"], "compacted")]
        full_hits, total = concept_hits(full["text"], item["case"]["expected_concepts"])
        compact_hits, _ = concept_hits(compacted["text"], item["case"]["expected_concepts"])
        comparisons.append({"case_id": item["case"]["case_id"], "domain": item["case"]["domain"],
                            "sentinel_full": full["sentinel_present"],
                            "sentinel_compacted": compacted["sentinel_present"],
                            "needles_full": full["needles_present"],
                            "needles_compacted": compacted["needles_present"],
                            "concept_hits_full": full_hits, "concept_hits_compacted": compact_hits,
                            "concept_groups": total, "concept_delta": compact_hits - full_hits,
                            "lexical_jaccard": lexical_jaccard(full["text"], compacted["text"], item["sentinel"])})
    count = len(comparisons)
    total_full = sum(row["concept_hits_full"] for row in comparisons)
    total_compact = sum(row["concept_hits_compacted"] for row in comparisons)
    checks = {
        "all_paired_responses": len(rows) == count * 2,
        "all_sentinels_full": all(row["sentinel_full"] for row in comparisons),
        "all_sentinels_compacted": all(row["sentinel_compacted"] for row in comparisons),
        "all_three_needles_full": all(all(row["needles_full"].values()) for row in comparisons),
        "all_three_needles_compacted": all(all(row["needles_compacted"].values()) for row in comparisons),
        "no_compaction_specific_needle_loss": all(sum(row["needles_compacted"].values()) >= sum(row["needles_full"].values()) for row in comparisons),
        "concept_total_not_worse_by_more_than_eight": total_compact >= total_full - 8,
        "at_least_95_percent_without_concept_regression": sum(row["concept_delta"] >= 0 for row in comparisons) >= math.ceil(0.95 * count),
        "median_lexical_jaccard_at_least_0_50": statistics.median(row["lexical_jaccard"] for row in comparisons) >= 0.5,
    }
    return {"checks": checks, "accepted": all(checks.values()), "comparisons": comparisons,
            "concept_hits_full": total_full, "concept_hits_compacted": total_compact}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    model = next(row for row in manifest["models"] if row["name"] == args.model)
    config = model["spomin20x20"]
    corpus_path = (args.manifest.parent / config["corpus_path"]).resolve()
    corpus_hash = file_sha256(corpus_path)
    expected_hash = config.get("corpus_sha256", FROZEN_CORPUS_SHA256)
    if corpus_hash != expected_hash or expected_hash != FROZEN_CORPUS_SHA256:
        raise ValueError(f"frozen corpus hash mismatch: {corpus_hash}")
    corpus = json.loads(corpus_path.read_text())
    if corpus.get("schema") != CORPUS_SCHEMA or len(corpus.get("domain_order", [])) != 20 or len(corpus.get("cases", [])) != 400:
        raise ValueError("frozen corpus must contain exactly 20 domains and 400 cases")
    from transformers import AutoTokenizer
    tokenizer_path = (args.manifest.parent / config["tokenizer_path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False, local_files_only=True)
    prepared = [prepare_case(case, tokenizer, int(config.get("capacity_tokens", 8192))) for case in corpus["cases"]]
    if not all(row["receipt"]["shortfall_tokens"] == 0 for row in prepared):
        raise RuntimeError("transcript preparation failed")
    source_tokenizer_path = (args.manifest.parent / config["source_reconciliation_tokenizer_path"]).resolve()
    if source_tokenizer_path == tokenizer_path:
        source_prepared = prepared
    else:
        source_tokenizer = AutoTokenizer.from_pretrained(
            source_tokenizer_path, trust_remote_code=False, local_files_only=True
        )
        source_prepared = [
            prepare_case(case, source_tokenizer, int(config.get("capacity_tokens", 8192)))
            for case in corpus["cases"]
        ]
    source_receipts_path = Path(config["source_receipts_path"]).resolve()
    source_receipts_hash = file_sha256(source_receipts_path)
    if source_receipts_hash != config["source_receipts_sha256"]:
        raise ValueError("frozen source receipt artifact hash mismatch")
    source_receipts = json.loads(source_receipts_path.read_text())["receipts"]
    comparison_fields = ("full_ledger_tokens", "target_limit_tokens", "projected_target_tokens",
                         "full_message_count", "compacted_message_count", "reclaimed_tokens",
                         "shortfall_tokens", "selected_segment_ids", "needle_selected", "query_selected")
    mismatches = []
    for row in source_prepared:
        case_id, actual = row["case"]["case_id"], row["receipt"]
        expected = source_receipts[case_id]
        different = {name: {"actual": actual[name], "expected": expected[name]}
                     for name in comparison_fields if actual[name] != expected[name]}
        if different:
            mismatches.append({"case_id": case_id, "fields": different})
    if mismatches:
        raise RuntimeError(f"ported preparation differs from frozen source: {mismatches[:3]}")
    serving_arm = next(arm for arm in model["arms"] if arm["name"] == config["serving_arm"])
    report = {"schema": SCHEMA, "manifest_sha256": digest(manifest), "corpus_sha256": corpus_hash,
              "model": model["name"], "serving_arm": serving_arm["name"], "host": host_identity(),
              "arm_order_claim": model.get("arm_order_claim", "alternating"),
              "scope": "client-side exact transcript rebuild; no physical cache or recurrent-state surgery",
              "thermal_control": {"enabled": False, "reason": "user-requested no-thermal batch suite"},
              "batch_size": 20, "domains": 20, "cases": 400,
              "source_preparation_reconciliation": {"source_receipts_sha256": source_receipts_hash,
                                                     "source_tokenizer_path": str(source_tokenizer_path),
                                                     "cases_compared": 400,
                                                     "fields": list(comparison_fields),
                                                     "mismatches": 0},
              "preparation_receipts": {row["case"]["case_id"]: row["receipt"] for row in prepared},
              "rows": [], "cells": {}, "passed": False, "started_at": time.time()}
    if args.resume and args.output.exists():
        prior = json.loads(args.output.read_text())
        for key in ("schema", "manifest_sha256", "corpus_sha256", "model", "serving_arm"):
            if prior.get(key) != report.get(key):
                raise ValueError(f"resume {key} mismatch")
        if stable_host_identity(prior["host"]) != stable_host_identity(host_identity()):
            raise ValueError("resume host/OS/runtime/source identity mismatch")
        report = prior
    atomic_json(args.output, report)
    if args.prepare_only:
        return
    client = activate(serving_arm, float(config.get("timeout_seconds", 1800)))
    current_identity = identity(client.get("/v1/status"))
    active_status = client.get("/v1/status")
    report["bound_qualification_receipt"] = validate_bound_qualification_receipt(
        args.manifest, serving_arm, active_status
    )
    prior_identities = {json.dumps(row["identity"], sort_keys=True) for row in report["cells"].values()
                        if row.get("passed") and row.get("identity")}
    if prior_identities and prior_identities != {json.dumps(current_identity, sort_keys=True)}:
        raise ValueError("resume server identity mismatch")
    completed = completed_cell_ids(report)
    if completed:
        report["rows"] = rows_for_completed_cells(report)
    nonce = str(report["started_at"])
    system = corpus["system"] + " Reproduce all three audit tokens exactly and do not omit the final code line. Campaign nonce: " + nonce
    for index, domain in enumerate(corpus["domain_order"]):
        group = [row for row in prepared if row["case"]["domain"] == domain]
        for transcript_arm in (("full", "compacted") if index % 2 == 0 else ("compacted", "full")):
            cell_id = f"{domain}:{transcript_arm}"
            if cell_id in completed:
                continue
            try:
                rows, before, after, warmup = run_domain(client, group, system, transcript_arm,
                                                         int(config.get("max_tokens", 192)), serving_arm)
                report["rows"].extend(rows)
                report["cells"][cell_id] = {"passed": True, "rows": len(rows),
                                              "identity": identity(before),
                                              "warmup": warmup,
                                              "observed_compute_widths": sorted({observed_width(row["receipt"]) for row in rows}),
                                              "status_before": before, "status_after": after}
            except BaseException as exc:
                report["cells"][cell_id] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
                atomic_json(args.output, report)
                raise
            report["last_progress_at"] = time.time()
            atomic_json(args.output, report)
    result = acceptance(prepared, report["rows"])
    report["acceptance"] = result
    report["passed"] = result["accepted"] and len(report["cells"]) == 40
    report["completed_at"] = time.time()
    atomic_json(args.output, report)
    if not report["passed"]:
        raise SystemExit("Spomin 20x20 acceptance failed")


if __name__ == "__main__":
    main()
