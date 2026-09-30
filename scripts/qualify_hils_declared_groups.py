#!/usr/bin/env python3
"""Real-artifact A/B qualification for HiLS adapter-declared projections."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx

from mlx2.adapters.olmo_hils import OlmoHiLSAdapter, inspect_artifact
from mlx2.contracts import Capability
from mlx2.runtime import lane
from mlx2.runtime.lane import installer
from mlx2.runtime.lane.policy import detect, resolve


GROUP = "hils-attn-qkv-lmkq"
LOCK_RECEIPTS = (
    Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
    Path("/tmp/gpu.lock/owner.json"),
)
ROWS = (2, 4, 8, 16, 32)
ATOL = 0.125
RTOL = 0.02


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, check=False, capture_output=True, text=True
        ).stdout.strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def system_snapshot() -> dict:
    vm = command_output(["vm_stat"])
    swapouts = None
    for line in vm.splitlines():
        if line.startswith("Swapouts:"):
            swapouts = int(line.split()[1].rstrip("."))
            break
    rss = command_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
    return {
        "time": time.time(),
        "pmset_therm": command_output(["pmset", "-g", "therm"]),
        "swapusage": command_output(["sysctl", "-n", "vm.swapusage"]),
        "swapouts_pages": swapouts,
        "process_rss_bytes": int(rss.strip()) * 1024 if rss.strip().isdigit() else None,
        "mlx_active_bytes": int(mx.get_active_memory()),
        "mlx_peak_bytes": int(mx.get_peak_memory()),
    }


def require_lock_receipts() -> dict:
    owners = []
    for path in LOCK_RECEIPTS:
        if not path.is_file():
            raise RuntimeError(f"missing GPU owner receipt: {path}")
        owners.append(json.loads(path.read_text()))
    lease_ids = {owner.get("lease_id") for owner in owners}
    pids = {owner.get("pid") for owner in owners}
    if len(lease_ids) != 1 or None in lease_ids or len(pids) != 1:
        raise RuntimeError("GPU owner receipts do not match")
    return {"paths": [str(path) for path in LOCK_RECEIPTS], "owners": owners}


def digest_tokens(tokens: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()


def make_tokens(adapter, count: int) -> list[int]:
    seed = adapter.tokenizer.encode(
        "The landmark retrieval model reads local context and selected earlier chunks. ",
        add_special_tokens=False,
    )
    if not seed:
        raise RuntimeError("tokenizer produced an empty qualification seed")
    return (list(seed) * math.ceil(count / len(seed)))[:count]


def array_tokens(values: list[int]):
    return mx.array([values], dtype=mx.int32)


def policy_for(adapter, declared: bool) -> dict:
    policy = resolve(
        detect(adapter.model, adapter.config),
        family=adapter.descriptor.family,
        overrides={"declared_groups": declared},
        mode="crossover",
    )
    if policy["mode"] != "crossover" or policy["min_rows"]["q6"] != 8:
        raise RuntimeError(f"unexpected HiLS q6 crossover policy: {policy}")
    return policy


def install_arm(adapter, declared: bool) -> dict:
    policy = policy_for(adapter, declared)
    offered = adapter.lane_projection_groups()
    receipt = lane.apply_policy(adapter.model, policy, declared=offered)
    if not receipt or not receipt.get("covered"):
        raise RuntimeError(f"lane policy did not cover the HiLS artifact: {receipt}")
    return receipt


def declared_counts() -> dict:
    stats = lane.stats()
    return {
        "launches": int(stats.get(f"declared_launches:{GROUP}", 0)),
        "reuses": int(stats.get(f"declared_reuses:{GROUP}", 0)),
        "partial": int(stats.get(f"declared_partial:{GROUP}", 0)),
        "all": stats,
    }


def expected_counts(eligible_forwards: int) -> dict:
    return {
        "launches": 8 * eligible_forwards,
        "reuses": 24 * eligible_forwards,
        "partial": 0,
    }


def inserted_rows(real_segments: list[int], chunk_size: int = 64) -> list[int]:
    """Return the actual Metal row count after HiLS landmark insertion."""
    offset = 0
    rows = []
    for real_rows in real_segments:
        real_offset = offset - offset // chunk_size
        last_real = real_offset + real_rows - 1
        end = (
            last_real
            + last_real // (chunk_size - 1)
            + (1 if (last_real + 1) % (chunk_size - 1) == 0 else 0)
        )
        actual = end - offset + 1
        rows.append(actual)
        offset += actual
    return rows


def eligible_forwards(real_segments: list[int]) -> int:
    return sum(8 <= rows <= 32 for rows in inserted_rows(real_segments))


def counter_passed(counts: dict, expected: dict, declared: bool) -> bool:
    observed = {key: counts[key] for key in ("launches", "reuses", "partial")}
    return observed == (expected if declared else expected_counts(0))


def run_segments(
    model, token_ids: list[int], segment_sizes: list[int]
) -> tuple[object, dict]:
    if sum(segment_sizes) != len(token_ids):
        raise ValueError("segment sizes do not cover token ids")
    cache = model.make_cache()
    offset = 0
    logits = None
    per_segment = []
    actual_rows = inserted_rows(segment_sizes)
    for size, metal_rows in zip(segment_sizes, actual_rows, strict=True):
        started = time.perf_counter_ns()
        logits = model(array_tokens(token_ids[offset : offset + size]), cache=cache)
        mx.eval(logits)
        per_segment.append(
            {
                "real_rows": size,
                "metal_rows": metal_rows,
                "milliseconds": (time.perf_counter_ns() - started) / 1e6,
            }
        )
        offset += size
    assert logits is not None
    return logits, {
        "segments": per_segment,
        "milliseconds": sum(item["milliseconds"] for item in per_segment),
    }


def parity(actual, expected, *, require_bitwise: bool = False) -> dict:
    delta = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
    maximum = float(mx.max(delta).item())
    mean = float(mx.mean(delta).item())
    close = bool(mx.all(delta <= ATOL + RTOL * mx.abs(expected)).item())
    finite = bool(mx.all(mx.isfinite(actual)).item())
    bitwise = bool(mx.array_equal(actual, expected).item())
    argmax_equal = bool(
        mx.array_equal(mx.argmax(actual, axis=-1), mx.argmax(expected, axis=-1)).item()
    )
    return {
        "shape": list(actual.shape),
        "finite": finite,
        "bitwise_equal": bitwise,
        "allclose": close,
        "argmax_equal": argmax_equal,
        "max_abs": maximum,
        "mean_abs": mean,
        "atol": ATOL,
        "rtol": RTOL,
        "require_bitwise": require_bitwise,
        "passed": finite
        and close
        and argmax_equal
        and (bitwise if require_bitwise else True),
    }


def workload_cases(adapter, declared: bool) -> dict:
    records = {}
    all_tokens = make_tokens(adapter, 192)

    installer.STATS.clear()
    prefix = all_tokens[:65]
    cache = adapter.model.make_cache()
    pre = adapter.model(array_tokens(prefix), cache=cache)
    mx.eval(pre)
    started = time.perf_counter_ns()
    logits = adapter.model(array_tokens(all_tokens[65:66]), cache=cache)
    mx.eval(logits)
    timing = (time.perf_counter_ns() - started) / 1e6
    counts = declared_counts()
    expected = expected_counts(0)
    records["one_token_decode"] = {
        "logits": logits,
        "milliseconds": timing,
        "counters": counts,
        "expected_counters": expected,
        "counters_passed": counter_passed(counts, expected, declared),
    }

    for width in ROWS:
        installer.STATS.clear()
        cache = adapter.model.make_cache()
        prefix_logits = adapter.model(array_tokens(all_tokens[:32]), cache=cache)
        mx.eval(prefix_logits)
        started = time.perf_counter_ns()
        logits = adapter.model(array_tokens(all_tokens[32 : 32 + width]), cache=cache)
        mx.eval(logits)
        timing = (time.perf_counter_ns() - started) / 1e6
        eligible = eligible_forwards([32, width])
        counts = declared_counts()
        expected = expected_counts(eligible)
        records[f"verify_width_{width}"] = {
            "logits": logits,
            "milliseconds": timing,
            "counters": counts,
            "expected_counters": expected,
            "counters_passed": counter_passed(counts, expected, declared),
        }

    installer.STATS.clear()
    logits, timing = run_segments(adapter.model, all_tokens[:97], [32, 32, 32, 1])
    counts = declared_counts()
    expected = expected_counts(eligible_forwards([32, 32, 32, 1]))
    records["prefill_tail"] = {
        "logits": logits,
        "timing": timing,
        "counters": counts,
        "expected_counters": expected,
        "counters_passed": counter_passed(counts, expected, declared),
    }
    return records


def serving_run(
    adapter, declared: bool, prompt_tokens: list[int], max_tokens: int
) -> dict:
    installer.STATS.clear()
    mx.reset_peak_memory()
    cache = adapter.model.make_cache()
    prefill_started = time.perf_counter_ns()
    logits = None
    for start in range(0, len(prompt_tokens), 32):
        logits = adapter.model(
            array_tokens(prompt_tokens[start : start + 32]), cache=cache
        )
        mx.eval(logits)
    prefill_ms = (time.perf_counter_ns() - prefill_started) / 1e6
    assert logits is not None
    generated = []
    decode_ms = 0.0
    next_token = int(mx.argmax(logits[0, -1]).item())
    for _ in range(max_tokens):
        generated.append(next_token)
        started = time.perf_counter_ns()
        logits = adapter.model(array_tokens([next_token]), cache=cache)
        mx.eval(logits)
        decode_ms += (time.perf_counter_ns() - started) / 1e6
        next_token = int(mx.argmax(logits[0, -1]).item())
    counts = declared_counts()
    segments = [
        min(32, len(prompt_tokens) - start)
        for start in range(0, len(prompt_tokens), 32)
    ]
    eligible = eligible_forwards(segments)
    expected = expected_counts(eligible)
    total_ms = prefill_ms + decode_ms
    return {
        "arm": "declared" if declared else "default",
        "prompt_tokens": len(prompt_tokens),
        "generated_tokens": generated,
        "prefill_real_segments": segments,
        "prefill_metal_rows": inserted_rows(segments),
        "generated_digest": digest_tokens(generated),
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "prefill_tokens_per_second": len(prompt_tokens) / (prefill_ms / 1000),
        "decode_tokens_per_second": max_tokens / (decode_ms / 1000),
        "goodput_tokens_per_second": (len(prompt_tokens) + max_tokens)
        / (total_ms / 1000),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "active_memory_bytes": int(mx.get_active_memory()),
        "counters": counts,
        "expected_counters": expected,
        "counters_passed": counter_passed(counts, expected, declared),
    }


def public_case(record: dict) -> dict:
    return {key: value for key, value in record.items() if key != "logits"}


def route_receipt(adapter, install_receipt: dict, declared: bool) -> dict:
    capabilities = sorted(item.value for item in adapter.descriptor.capabilities)
    return {
        "adapter": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        "family": adapter.descriptor.family,
        "route": adapter.default_route,
        "qualification": adapter.descriptor.metadata.get("qualification"),
        "capabilities": capabilities,
        "lane": {
            "mode": install_receipt["policy"]["mode"],
            "declared_groups": declared,
            "law_id": install_receipt["law_id"],
            "covered": install_receipt["covered"],
            "declared_group": install_receipt.get("declared_groups", {}).get(GROUP),
        },
        "apcv2": {
            "capability_declared": Capability.APC_V2.value in capabilities,
            "cache_layout": adapter.descriptor.cache_layout,
            "qualified": False,
            "identity": None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--generation-tokens", type=int, default=8)
    args = parser.parse_args(argv)
    if args.repeats < 4 or args.generation_tokens < 1:
        parser.error("repeats must be at least 4 and generation-tokens positive")

    receipt = {
        "schema": "mlx2.hils-declared-groups-real-artifact.v1",
        "status": "running",
        "started_at": time.time(),
        "source_revision": git("rev-parse", "HEAD"),
        "source_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "command": [sys.executable, *sys.argv],
        "platform": platform.platform(),
        "python": sys.version,
        "model_path": str(args.model.resolve()),
        "artifact_manifest": str(args.artifact_manifest.resolve()),
        "lane_mode": "crossover",
        "q6_crossover_rows": 8,
        "parity_law": {"atol": ATOL, "rtol": RTOL, "argmax_equal": True},
        "state": {
            "implemented": True,
            "qualified": False,
            "selected": False,
            "observed_used": False,
        },
        "failures": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        receipt["updated_at"] = time.time()
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    adapter = None
    try:
        receipt["locks"] = require_lock_receipts()
        manifest = json.loads(args.artifact_manifest.read_text())
        if not manifest.get("complete"):
            raise RuntimeError("artifact manifest is not complete")
        receipt["artifact"] = {
            "repo_id": manifest["upstream"]["repo_id"],
            "revision": manifest["upstream"]["revision"],
            "q6_file_sha256": {
                item["name"]: item["sha256"]
                for item in manifest["q6_artifact"]["files"]
            },
            "inspection": inspect_artifact(args.model),
        }
        mx.set_default_device(mx.gpu)
        mx.reset_peak_memory()
        receipt["system_before_load"] = system_snapshot()
        loaded = time.perf_counter_ns()
        adapter = OlmoHiLSAdapter(str(args.model))
        receipt["load_seconds"] = (time.perf_counter_ns() - loaded) / 1e9
        receipt["ready_memory"] = system_snapshot()
        receipt["mlx_version"] = mx.__version__
        save()

        # Compile both arithmetic laws before collecting latency evidence.
        warm_tokens = make_tokens(adapter, 32)
        for declared in (False, True):
            install_arm(adapter, declared)
            warm, _ = run_segments(adapter.model, warm_tokens, [32])
            mx.eval(warm)

        arm_cases = {}
        install_receipts = {}
        route_receipts = {}
        for declared in (False, True):
            name = "declared" if declared else "default"
            install_receipts[name] = install_arm(adapter, declared)
            route_receipts[name] = route_receipt(
                adapter, install_receipts[name], declared
            )
            mx.reset_peak_memory()
            arm_cases[name] = workload_cases(adapter, declared)
            receipt.setdefault("arm_memory", {})[name] = system_snapshot()

        comparisons = {}
        for name in arm_cases["default"]:
            comparisons[name] = parity(
                arm_cases["declared"][name]["logits"],
                arm_cases["default"][name]["logits"],
                require_bitwise=name
                in {"one_token_decode", "verify_width_2", "verify_width_4"},
            )
        receipt["cases"] = {
            arm: {name: public_case(record) for name, record in records.items()}
            for arm, records in arm_cases.items()
        }
        receipt["comparisons"] = comparisons
        receipt["install_receipts"] = install_receipts
        receipt["route_receipts"] = route_receipts

        schedule = [(False, True), (True, False)] * math.ceil(args.repeats / 2)
        profiles = {
            "interactive_256": (make_tokens(adapter, 256), args.generation_tokens),
            "prefill_heavy_1024": (make_tokens(adapter, 1024), args.generation_tokens),
        }
        serving_profiles = {}
        summary_profiles = {}
        for profile, (prompt, output_tokens) in profiles.items():
            serving = []
            for pair in schedule[: args.repeats]:
                for declared in pair:
                    install_arm(adapter, declared)
                    serving.append(
                        serving_run(adapter, declared, prompt, output_tokens)
                    )
            serving_profiles[profile] = serving
            baseline_digests = {
                item["generated_digest"] for item in serving if item["arm"] == "default"
            }
            declared_digests = {
                item["generated_digest"]
                for item in serving
                if item["arm"] == "declared"
            }
            output_parity = (
                len(baseline_digests) == 1 and baseline_digests == declared_digests
            )
            medians = {}
            for metric in (
                "prefill_ms",
                "decode_ms",
                "total_ms",
                "goodput_tokens_per_second",
            ):
                medians[metric] = {
                    arm: statistics.median(
                        item[metric] for item in serving if item["arm"] == arm
                    )
                    for arm in ("default", "declared")
                }
            medians["speedup"] = {
                "prefill": medians["prefill_ms"]["default"]
                / medians["prefill_ms"]["declared"],
                "decode": medians["decode_ms"]["default"]
                / medians["decode_ms"]["declared"],
                "end_to_end": medians["total_ms"]["default"]
                / medians["total_ms"]["declared"],
                "goodput": medians["goodput_tokens_per_second"]["declared"]
                / medians["goodput_tokens_per_second"]["default"],
            }
            summary_profiles[profile] = {
                "output_parity": output_parity,
                "medians": medians,
            }
        receipt["serving_ab"] = serving_profiles
        receipt["serving_summary"] = {"profiles": summary_profiles}

        group = install_receipts["declared"].get("declared_groups", {}).get(GROUP, {})
        formed = group.get("formed") == {"affine-q6-g64": 8}
        all_serving = [
            item for profile in serving_profiles.values() for item in profile
        ]
        output_parity = all(
            profile["output_parity"] for profile in summary_profiles.values()
        )
        counters_passed = all(
            record["counters_passed"]
            for arm in receipt["cases"].values()
            for record in arm.values()
        ) and all(item["counters_passed"] for item in all_serving)
        parity_passed = (
            all(item["passed"] for item in comparisons.values()) and output_parity
        )
        law_passed = (
            "+declared[" not in install_receipts["default"]["law_id"]
            and "+declared[" in install_receipts["declared"]["law_id"]
            and install_receipts["default"]["law_id"]
            != install_receipts["declared"]["law_id"]
        )
        apcv2_passed = all(
            not item["apcv2"]["capability_declared"]
            and item["apcv2"]["cache_layout"] is None
            and item["apcv2"]["identity"] is None
            for item in route_receipts.values()
        )
        benefit = all(
            profile["medians"]["speedup"]["end_to_end"] >= 1.03
            for profile in summary_profiles.values()
        )
        receipt["gates"] = {
            "artifact_complete": True,
            "formed_exactly_eight_hils_groups": formed,
            "counters_and_zero_partial": counters_passed,
            "parity": parity_passed,
            "law_identity": law_passed,
            "apcv2_absence_truthful": apcv2_passed,
            "real_serving_benefit_at_least_1.03x": benefit,
        }
        qualified = all(receipt["gates"].values())
        receipt["state"] = {
            "implemented": True,
            "qualified": qualified,
            "selected": False,
            "observed_used": True,
            "default_off": True,
            "decision": (
                "qualified candidate; selection remains a separate policy decision"
                if qualified
                else "not qualified; default-off because one or more real-artifact gates failed"
            ),
        }
        receipt["status"] = "passed" if qualified else "not_qualified"
    except Exception as exc:  # noqa: BLE001 - preserve a durable failure receipt
        receipt["status"] = "error"
        receipt["failures"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=12),
            }
        )
    finally:
        if adapter is not None:
            try:
                lane.uninstall(adapter.model)
                adapter.close()
                mx.clear_cache()
            except Exception as exc:  # noqa: BLE001 - cleanup failure belongs in receipt
                receipt["failures"].append({"cleanup": f"{type(exc).__name__}: {exc}"})
        receipt["system_after"] = system_snapshot()
        before = receipt.get("system_before_load", {}).get("swapouts_pages")
        after = receipt["system_after"].get("swapouts_pages")
        receipt["swapout_delta_pages"] = (
            after - before if before is not None and after is not None else None
        )
        receipt["finished_at"] = time.time()
        receipt["elapsed_seconds"] = receipt["finished_at"] - receipt["started_at"]
        save()
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "state": receipt["state"],
                "gates": receipt.get("gates"),
                "failures": receipt["failures"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
