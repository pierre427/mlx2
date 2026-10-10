#!/usr/bin/env python3
"""CPU auditor for correctness evidence from scripts/dloop_ab.py.
It checks mechanism behavior only; it does not qualify a route or use speed fields.
"""

import argparse
import hashlib
import json
import math
import platform
import re
from pathlib import Path

WIDTHS = (1, 2, 4, 8)
DEPTHS = tuple(range(1, 9))
ROUTES = {"segmented_self_mtp", "continuous_batched_self_mtp"}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dloop_producer_sha256():
    repo_root = Path(__file__).resolve().parents[3]
    return sha(repo_root / "scripts" / "dloop_ab.py")


def spans(row):
    raw = row.get("verify_span_hist")
    if not isinstance(raw, dict):
        return {}
    try:
        return {
            int(k): v
            for k, v in raw.items()
            if isinstance(v, int) and not isinstance(v, bool) and v > 0
        }
    except (TypeError, ValueError):
        return {}


def _valid_state_continuation(state, expected_tokens):
    if (
        not isinstance(state, dict)
        or state.get("schema") != "mlx2.dloop-state-continuation.v1"
    ):
        return False
    prefix = state.get("prefix_token_ids")
    if (
        not isinstance(prefix, list)
        or len(prefix) < 2
        or any(
            not isinstance(token, int) or isinstance(token, bool) for token in prefix
        )
    ):
        return False
    digest = hashlib.sha256(
        json.dumps(prefix, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        state.get("prefix_tokens") != len(prefix)
        or state.get("prefix_sha256") != digest
    ):
        return False
    for key in ("target_cache_offsets", "mtp_cache_offsets"):
        offsets = state.get(key)
        if not isinstance(offsets, list) or not offsets:
            return False
        for index, item in enumerate(offsets):
            if (
                not isinstance(item, dict)
                or item.get("index") != index
                or not isinstance(item.get("type"), str)
                or not item["type"]
            ):
                return False
            offset = item.get("offset")
            if offset is not None and (
                not isinstance(offset, int) or isinstance(offset, bool) or offset < 0
            ):
                return False
    target_offsets = state["target_cache_offsets"]
    if not any(item["offset"] is not None for item in target_offsets):
        return False
    streams = [
        state.get(key)
        for key in (
            "saved_ordinary_tokens",
            "saved_self_mtp_tokens",
            "cold_ordinary_tokens",
        )
    ]
    if any(
        not isinstance(tokens, list)
        or len(tokens) != expected_tokens
        or any(
            not isinstance(token, int) or isinstance(token, bool) for token in tokens
        )
        for tokens in streams
    ):
        return False
    return streams[0] == streams[1] == streams[2]


def audit_case(width, report, pairs):
    failures = []
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width not in WIDTHS
        or not isinstance(report, dict)
        or report.get("schema") != "mlx2.dloop_ab.v1"
        or not isinstance(report.get("width"), int)
        or isinstance(report.get("width"), bool)
        or report.get("width") != width
        or not isinstance(report.get("model"), str)
        or not report.get("model")
        or not isinstance(report.get("source_revision"), str)
        or not HEX40.fullmatch(report.get("source_revision", ""))
        or not isinstance(report.get("artifact_identity"), str)
        or not HEX64.fullmatch(report.get("artifact_identity", ""))
        or not isinstance(report.get("producer_sha256"), str)
        or not HEX64.fullmatch(report.get("producer_sha256", ""))
        or not isinstance(report.get("runtime_identity"), dict)
    ):
        return {
            "width": width,
            "passed": False,
            "failures": ["bad width or dloop_ab schema"],
        }
    runtime = report["runtime_identity"]
    for key in ("source_sha256", "mlx_native_sha256"):
        if not isinstance(runtime.get(key), str) or not HEX64.fullmatch(runtime[key]):
            failures.append(f"actual runtime identity {key} must be SHA-256")
    for key in ("python", "macos", "mlx", "transformers"):
        if not isinstance(runtime.get(key), str) or not runtime[key]:
            failures.append(f"actual runtime identity {key} is required")
    if report.get("runtime_source_sha256") != runtime.get("source_sha256"):
        failures.append("runtime source digest does not match actual runtime identity")
    if report.get("runtime_native_sha256") != runtime.get("mlx_native_sha256"):
        failures.append("runtime native digest does not match actual runtime identity")
    if not isinstance(report.get("host"), str) or not report["host"]:
        failures.append("actual host identity required")
    state_tokens = report.get("state_oracle_tokens")
    if (
        not isinstance(state_tokens, int)
        or isinstance(state_tokens, bool)
        or state_tokens < 1
    ):
        failures.append("positive state_oracle_tokens evidence is required")
    if (
        not isinstance(pairs, int)
        or isinstance(pairs, bool)
        or pairs < 1
        or not isinstance(report.get("pairs"), int)
        or isinstance(report.get("pairs"), bool)
        or report["pairs"] != pairs
    ):
        failures.append(f"expected {pairs} paired repetitions")
    arms = report.get("arms")
    wanted = {*(f"fixed{d}" for d in DEPTHS), "loop8"}
    if not isinstance(arms, dict) or set(arms) != wanted:
        return {
            "width": width,
            "passed": False,
            "failures": ["arms must cover fixed depths 1..8 and loop8"],
        }
    for depth in DEPTHS:
        cfg = arms.get(f"fixed{depth}")
        if (
            not isinstance(cfg, dict)
            or not isinstance(cfg.get("num_draft"), int)
            or isinstance(cfg.get("num_draft"), bool)
            or cfg.get("num_draft") != depth
            or cfg.get("draft_loop") is not None
        ):
            failures.append(f"fixed{depth} is not a fixed-depth {depth} control")
    gate = arms.get("loop8")
    policy = gate.get("draft_loop") if isinstance(gate, dict) else None
    threshold = policy.get("threshold") if isinstance(policy, dict) else None
    if (
        not isinstance(gate, dict)
        or gate.get("num_draft") != 8
        or isinstance(gate.get("num_draft"), bool)
        or not isinstance(policy, dict)
    ):
        failures.append("loop8 does not configure DLoop over max depth 8")
    elif (
        policy.get("boundaries") != list(DEPTHS)
        or policy.get("cohort") != "any"
        or not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(threshold)
        or threshold != -1e9
    ):
        failures.append(
            "loop8 must use finite threshold, stage boundaries 1..8 and cohort=any"
        )
    results = report.get("results")
    if not isinstance(results, list):
        return {
            "width": width,
            "passed": False,
            "failures": failures + ["results missing"],
        }
    indexed = {}
    for item in results:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("arm"), str)
            or item.get("arm") not in wanted
            or not isinstance(item.get("pair"), int)
            or isinstance(item.get("pair"), bool)
        ):
            failures.append("malformed pair/arm result")
            continue
        key = (item["pair"], item["arm"])
        if key in indexed:
            failures.append(f"duplicate result {key}")
        indexed[key] = item
    if set(indexed) != {(p, arm) for p in range(pairs) for arm in wanted}:
        failures.append("pair/arm coverage incomplete")
    max_span, engaged, extension_requests = 0, 0, 0
    for pair in range(pairs):
        base = indexed.get((pair, "fixed1"), {}).get("rows")
        if not isinstance(base, list) or not base:
            failures.append(f"pair {pair}: fixed1 baseline missing")
            continue
        base_by_prompt = {}
        for row in base:
            if not isinstance(row, dict):
                failures.append(f"pair {pair}: malformed baseline row")
                continue
            prompt, tokens = row.get("prompt"), row.get("tokens")
            if (
                not isinstance(prompt, int)
                or isinstance(prompt, bool)
                or not isinstance(tokens, list)
                or any(
                    not isinstance(token, int) or isinstance(token, bool)
                    for token in tokens
                )
            ):
                failures.append(f"pair {pair}: malformed baseline prompt/token IDs")
                continue
            if prompt in base_by_prompt:
                failures.append(f"pair {pair}: duplicate baseline prompt {prompt}")
            base_by_prompt[prompt] = tokens
        if not base_by_prompt or any(
            r.get("draft_loop") is not None for r in base if isinstance(r, dict)
        ):
            failures.append(
                f"pair {pair}: baseline malformed or unexpectedly has DLoop"
            )
            continue
        baseline_states = {}
        for row in base:
            if not isinstance(row, dict):
                continue
            prompt = row.get("prompt")
            state = row.get("state_continuation")
            if not _valid_state_continuation(state, state_tokens):
                failures.append(
                    f"pair {pair} fixed1 prompt {prompt}: invalid state continuation evidence"
                )
                continue
            baseline_states[prompt] = state
        for arm in wanted - {"fixed1"}:
            item = indexed.get((pair, arm), {})
            rows = item.get("rows")
            if not isinstance(rows, list):
                failures.append(f"pair {pair} {arm}: rows missing")
                continue
            by_prompt = {}
            for row in rows:
                if not isinstance(row, dict):
                    failures.append(f"pair {pair} {arm}: malformed row")
                    continue
                prompt, tokens = row.get("prompt"), row.get("tokens")
                if (
                    not isinstance(prompt, int)
                    or isinstance(prompt, bool)
                    or not isinstance(tokens, list)
                    or any(
                        not isinstance(token, int) or isinstance(token, bool)
                        for token in tokens
                    )
                ):
                    failures.append(f"pair {pair} {arm}: malformed prompt/token IDs")
                    continue
                if prompt in by_prompt:
                    failures.append(f"pair {pair} {arm}: duplicate prompt {prompt}")
                by_prompt[prompt] = row
            if set(by_prompt) != set(base_by_prompt):
                failures.append(
                    f"pair {pair} {arm}: prompt coverage differs from baseline"
                )
                continue
            for prompt, row in by_prompt.items():
                if row["tokens"] != base_by_prompt[prompt]:
                    failures.append(
                        f"pair {pair} {arm} prompt {prompt}: token IDs differ"
                    )
                state = row.get("state_continuation")
                baseline_state = baseline_states.get(prompt)
                if not _valid_state_continuation(state, state_tokens):
                    failures.append(
                        f"pair {pair} {arm} prompt {prompt}: invalid state continuation evidence"
                    )
                elif baseline_state is None:
                    failures.append(
                        f"pair {pair} {arm} prompt {prompt}: fixed1 state evidence missing"
                    )
                elif (
                    state["prefix_sha256"] != baseline_state["prefix_sha256"]
                    or state["prefix_token_ids"] != baseline_state["prefix_token_ids"]
                    or state["target_cache_offsets"]
                    != baseline_state["target_cache_offsets"]
                ):
                    failures.append(
                        f"pair {pair} {arm} prompt {prompt}: saved state differs from fixed1 prefix/cache"
                    )
                hist = spans(row)

                if arm == "fixed8":
                    max_span = max(max_span, max(hist, default=0))
                if arm == "loop8":
                    receipt = row.get("draft_loop")
                    if (
                        not isinstance(receipt, dict)
                        or receipt.get("observed_used") is not True
                    ):
                        failures.append(
                            f"pair {pair} prompt {prompt}: DLoop not observed"
                        )
                        continue
                    decisions, extensions = (
                        receipt.get("decisions"),
                        receipt.get("extensions"),
                    )
                    if (
                        receipt.get("qualified") is not False
                        or receipt.get("boundaries") != list(DEPTHS)
                        or not isinstance(decisions, int)
                        or isinstance(decisions, bool)
                        or decisions < 1
                        or not isinstance(extensions, int)
                        or isinstance(extensions, bool)
                        or extensions < 0
                    ):
                        failures.append(
                            f"pair {pair} prompt {prompt}: DLoop receipt/counters invalid"
                        )
                    if (
                        not isinstance(row.get("route"), str)
                        or row.get("route") not in ROUTES
                    ):
                        failures.append(
                            f"pair {pair} prompt {prompt}: unexpected self-MTP route"
                        )
                    if width > 1:
                        counters = row.get("true_batched")
                        count = (
                            counters.get("true_batched_engaged")
                            if isinstance(counters, dict)
                            else None
                        )
                        if (
                            not isinstance(count, int)
                            or isinstance(count, bool)
                            or count < 1
                        ):
                            failures.append(
                                f"pair {pair} prompt {prompt}: true-batched counter missing"
                            )
                    if hist.get(9, 0) > 0:
                        engaged += 1
                    if (
                        isinstance(extensions, int)
                        and not isinstance(extensions, bool)
                        and extensions > 0
                    ):
                        extension_requests += 1
                        if hist.get(9, 0) < 1:
                            failures.append(
                                f"pair {pair} prompt {prompt}: extended request lacks verify span 9"
                            )
    if max_span != 9:
        failures.append(f"fixed8 did not observe verify span 9; saw {max_span}")
    if extension_requests < 1:
        failures.append("loop8 did not observe any permitted extension at max depth 8")
    return {
        "width": width,
        "passed": not failures,
        "pairs": pairs,
        "depths": list(DEPTHS),
        "max_depth": 8,
        "fixed8_max_verify_span": max_span,
        "loop8_engaged_requests": engaged,
        "loop8_extension_requests": extension_requests,
        "failures": failures,
    }


def audit_manifest(path):
    raw = Path(path).read_bytes()
    try:
        manifest = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {
            "schema": "mlx2.dloop-behavior-evidence.v1",
            "evidence_passed": False,
            "performance_fields_used": False,
            "failures": [f"malformed manifest: {exc}"],
        }
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "mlx2.dloop-behavior-campaign.v1"
    ):
        return {
            "schema": "mlx2.dloop-behavior-evidence.v1",
            "evidence_passed": False,
            "performance_fields_used": False,
            "failures": ["unsupported campaign manifest"],
        }
    pairs = manifest.get("pairs")
    if not isinstance(pairs, int) or isinstance(pairs, bool) or pairs < 1:
        return {
            "schema": "mlx2.dloop-behavior-evidence.v1",
            "evidence_passed": False,
            "performance_fields_used": False,
            "failures": ["pairs must be positive integer"],
        }
    for key in (
        "host",
        "model",
        "source_identity",
        "artifact_identity",
        "runtime_source_sha256",
        "runtime_native_sha256",
        "dloop_producer_sha256",
    ):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            return {
                "schema": "mlx2.dloop-behavior-evidence.v1",
                "evidence_passed": False,
                "performance_fields_used": False,
                "failures": [f"{key} identity required"],
            }
    if (
        not HEX40.fullmatch(manifest["source_identity"])
        or not HEX64.fullmatch(manifest["artifact_identity"])
        or not HEX64.fullmatch(manifest["runtime_source_sha256"])
        or not HEX64.fullmatch(manifest["runtime_native_sha256"])
        or not HEX64.fullmatch(manifest["dloop_producer_sha256"])
    ):
        return {
            "schema": "mlx2.dloop-behavior-evidence.v1",
            "evidence_passed": False,
            "performance_fields_used": False,
            "failures": ["manifest identity format mismatch"],
        }
    settings = manifest.get("settings")
    expected_settings = {
        "fixed_depths": list(DEPTHS),
        "gate_boundaries": list(DEPTHS),
        "gate_threshold": -1e9,
        "cohort": "any",
        "non_dloop_control": "fixed1",
        "explicit_dloop_arm": "loop8",
    }
    if settings != expected_settings:
        return {
            "schema": "mlx2.dloop-behavior-evidence.v1",
            "evidence_passed": False,
            "performance_fields_used": False,
            "failures": ["settings identity mismatch"],
        }
    entries = manifest.get("cases")
    if not isinstance(entries, list):
        return {
            "schema": "mlx2.dloop-behavior-evidence.v1",
            "evidence_passed": False,
            "performance_fields_used": False,
            "failures": ["cases must be a list"],
        }
    cases, failures = {}, []
    for item in entries:
        if not isinstance(item, dict):
            failures.append("malformed case")
            continue
        width, report_ref = item.get("width"), item.get("report")
        if not isinstance(width, int) or isinstance(width, bool):
            failures.append("malformed width")
            continue
        if width in cases:
            failures.append(f"duplicate width {width}")
            continue
        if not isinstance(report_ref, str) or not report_ref:
            failures.append(f"width {width}: malformed report path")
            continue
        report = Path(report_ref)
        try:
            if not report.is_file() or sha(report) != item.get("sha256"):
                failures.append(f"width {width}: report missing or hash mismatch")
                continue
            parsed = json.loads(report.read_text())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
            failures.append(f"width {width}: malformed report: {exc}")
            continue
        if (
            not isinstance(parsed, dict)
            or parsed.get("model") != manifest["model"]
            or parsed.get("source_revision") != manifest["source_identity"]
            or parsed.get("artifact_identity") != manifest["artifact_identity"]
            or parsed.get("host") != manifest["host"]
            or parsed.get("runtime_source_sha256") != manifest["runtime_source_sha256"]
            or parsed.get("runtime_native_sha256") != manifest["runtime_native_sha256"]
            or parsed.get("producer_sha256") != manifest["dloop_producer_sha256"]
            or not isinstance(parsed.get("runtime_identity"), dict)
            or parsed["runtime_identity"].get("source_sha256")
            != manifest["runtime_source_sha256"]
            or parsed["runtime_identity"].get("mlx_native_sha256")
            != manifest["runtime_native_sha256"]
        ):
            failures.append(f"width {width}: report model/source identity mismatch")
            continue
        cases[width] = audit_case(width, parsed, pairs)
    if set(cases) != set(WIDTHS):
        failures.append(
            f"width coverage must be exactly {WIDTHS}; got {tuple(sorted(cases))}"
        )
    failures.extend(
        f"width {w}: {p}" for w, case in cases.items() for p in case["failures"]
    )
    return {
        "schema": "mlx2.dloop-behavior-evidence.v1",
        "scope": "mechanism behavior evidence only; route qualification needs bound serving receipt and preflight",
        "host": manifest.get("host"),
        "model": manifest.get("model"),
        "settings": manifest.get("settings"),
        "performance_fields_used": False,
        "serving_default_auto_selection": "not assessed by direct BatchGenerator A/B",
        "evidence_passed": not failures and all(c["passed"] for c in cases.values()),
        "widths": [cases[w] for w in sorted(cases)],
        "failures": failures,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "producer_sha256": sha(__file__),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--create-manifest", type=Path)
    parser.add_argument(
        "--case", action="append", default=[], metavar="WIDTH=REPORT.json"
    )
    parser.add_argument("--model")
    parser.add_argument("--source-revision")
    parser.add_argument("--artifact-identity")
    parser.add_argument("--runtime-source-sha256")
    parser.add_argument("--runtime-native-sha256")
    parser.add_argument("--dloop-producer-sha256")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.create_manifest:
        if (
            args.manifest
            or not args.model
            or not args.source_revision
            or not args.artifact_identity
            or not args.runtime_source_sha256
            or not args.runtime_native_sha256
            or not args.dloop_producer_sha256
            or not args.case
        ):
            parser.error(
                "--create-manifest requires --model, --source-revision, --artifact-identity, both runtime SHA-256 values, and --case; omit positional manifest"
            )
        cases = []
        for spec in args.case:
            width_text, sep, path_text = spec.partition("=")
            if not sep:
                parser.error(f"bad case {spec!r}; expected WIDTH=REPORT.json")
            path = Path(path_text).resolve()
            if not path.is_file():
                parser.error(f"missing dloop_ab report: {path}")
            cases.append(
                {"width": int(width_text), "report": str(path), "sha256": sha(path)}
            )
        manifest = {
            "schema": "mlx2.dloop-behavior-campaign.v1",
            "host": platform.node(),
            "model": args.model,
            "source_identity": args.source_revision,
            "artifact_identity": args.artifact_identity,
            "runtime_source_sha256": args.runtime_source_sha256,
            "runtime_native_sha256": args.runtime_native_sha256,
            "dloop_producer_sha256": args.dloop_producer_sha256,
            "pairs": args.pairs,
            "settings": {
                "fixed_depths": list(DEPTHS),
                "gate_boundaries": list(DEPTHS),
                "gate_threshold": -1e9,
                "cohort": "any",
                "non_dloop_control": "fixed1",
                "explicit_dloop_arm": "loop8",
            },
            "cases": cases,
        }
        args.create_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.create_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        args.manifest = args.create_manifest
    if args.manifest is None:
        parser.error("pass a campaign manifest or --create-manifest")
    result = audit_manifest(args.manifest)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["evidence_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
