#!/usr/bin/env python3
# ruff: noqa: EXE001
"""Target-only owned-M3 diagnosis of adaptive BF16 reference prefill geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "adaptive-receipt", "out"):
        result.add_argument("--" + key, type=Path, required=True)
    result.add_argument("--max-contexts", type=int, choices=(2, 3, 4), default=4)
    result.add_argument("--deadline-seconds", type=int, default=900)
    result.add_argument("--i-own-the-gpu", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def preflight(args):
    if not 1 <= args.deadline_seconds <= 900:
        raise ValueError("deadline-seconds must be in [1,900]")
    if not args.dry_run and not args.i_own_the_gpu:
        raise ValueError("execution requires --i-own-the-gpu under operator ownership")
    if (
        args.out.resolve().is_relative_to(args.model.resolve())
        or args.out.resolve() == args.adaptive_receipt.resolve()
    ):
        raise ValueError("output must not overwrite model or input receipt")
    return {
        "schema": "mlx2.adaptive-bf16-reference-diagnostic.v1",
        "will_execute": not args.dry_run,
        "model": str(args.model.resolve()),
        "adaptive_receipt": str(args.adaptive_receipt.resolve()),
        "max_contexts": args.max_contexts,
        "deadline_seconds": args.deadline_seconds,
        "qualified": False,
        "performance_claim": False,
        "production_math_changed": False,
        "passed": False,
    }


def select_contexts(receipt, maximum):
    """Validate reached contexts; prefer one mismatch per distinct request."""
    if type(maximum) is not int or not 2 <= maximum <= 4:
        raise ValueError("select between two and four mismatch contexts")
    candidates = []
    for check in receipt.get("generation_checks", ()):
        if check.get("temperature") != 0.8:
            continue
        evidence = {row["uid"]: row for row in check.get("token_evidence", ())}
        for mismatch in check.get("sampled_law_mismatches", ()):
            row = evidence.get(mismatch.get("uid"))
            if row is None or row.get("request_index") != mismatch.get("request_index"):
                raise ValueError("mismatch has no matching request token evidence")
            prompt, actual, position = (
                row["prompt_tokens"],
                row["actual_tokens"],
                mismatch["position"],
            )
            if (
                not prompt
                or not isinstance(prompt, list)
                or not isinstance(actual, list)
                or type(position) is not int
                or not 0 <= position < len(actual)
                or len(prompt) + len(actual) > 4096
                or any(type(t) is not int or t < 0 for t in prompt + actual)
            ):
                raise ValueError("invalid or unbounded mismatch tokens")
            if mismatch["context_tokens"] != prompt + actual[:position]:
                raise ValueError("mismatch context differs from delivered token prefix")
            candidates.append(
                {
                    **mismatch,
                    "prompt_tokens": list(prompt),
                    "actual_tokens": list(actual),
                }
            )
    if not candidates:
        raise ValueError("receipt has no sampled .8 mismatch contexts")
    selected, seen = [], set()
    for row in candidates:
        if row["request_index"] not in seen:
            selected.append(row)
            seen.add(row["request_index"])
            if len(selected) == maximum:
                return selected
    for row in candidates:
        if row not in selected:
            selected.append(row)
            if len(selected) == maximum:
                break
    return selected


def source_identity():
    paths = [
        Path(__file__),
        ROOT / "scripts/validate_xpress_metal_matrix.py",
        ROOT / "scripts/validate_adaptive_metal.py",
    ]
    paths += sorted(
        p
        for p in (ROOT / "src").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
    files = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    return {
        "files": files,
        "sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
    }


def delta(a, b):
    import numpy as np

    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("comparison shapes differ")
    return {
        "bitwise_equal": bool(np.array_equal(a, b)),
        "max_absolute_error": float(np.max(np.abs(a - b), initial=0)),
        "l1": float(np.sum(np.abs(a - b))),
    }


def probability(logits):
    import numpy as np

    values = np.asarray(logits, dtype=np.float64) / 0.8
    values = np.exp(values - values.max())
    return values / values.sum()


def reference(model, mx, prompt, actual, position, step, *, whole):
    from mlx2.runtime.cow_cache import snapshot_recovery_descriptors

    caches = model.make_cache()
    prefix = prompt if whole else prompt[:-1]
    logits = None
    for start in range(0, len(prefix), step):
        logits = model(mx.array([prefix[start : start + step]]), cache=caches)
        mx.eval(logits)
    remaining = actual[:position] if whole else [prompt[-1], *actual[:position]]
    frozen = None
    for index, token in enumerate(remaining):
        if index == len(remaining) - 1:
            frozen = snapshot_recovery_descriptors(caches)
        logits = model(mx.array([[token]]), cache=caches)
        mx.eval(logits)
    return logits, caches, frozen


def diagnose_context(model, mx, context, step):
    import numpy as np

    from mlx2.runtime.cow_cache import restore_recovery_descriptors
    from mlx2.runtime.segmented_rotating_kv import SegmentedKVRows

    def host(value):
        return np.asarray(value.astype(mx.float32))

    def cache_hashes(caches):
        result = []
        for cache in caches:
            arrays = () if not cache.offset else cache.keys_and_values()
            mx.eval(*arrays)
            result.append(
                {
                    "offset": cache.offset,
                    "values_sha256": [
                        hashlib.sha256(host(a).tobytes()).hexdigest() for a in arrays
                    ],
                }
            )
        return result

    prompt, actual, position = (
        context["prompt_tokens"],
        context["actual_tokens"],
        context["position"],
    )
    serving, serving_cache, frozen = reference(
        model, mx, prompt, actual, position, step, whole=False
    )
    old, old_cache, _ = reference(model, mx, prompt, actual, position, step, whole=True)
    serving_row, old_row = host(serving)[0, -1], host(old)[0, -1]
    correct_law, old_law = probability(serving_row), probability(old_row)
    token = context["largest_error_token_id"]
    if not 0 <= token < len(correct_law):
        raise ValueError("recorded mismatch token outside target vocabulary")
    result = {
        "request_index": context["request_index"],
        "position": position,
        "context_tokens": context["context_tokens"],
        "serving_vs_whole_prompt_logits": delta(serving_row, old_row),
        "serving_vs_whole_prompt_probabilities": delta(correct_law, old_law),
        "serving_cache_hashes": cache_hashes(serving_cache),
        "whole_prompt_cache_hashes": cache_hashes(old_cache),
        "recorded_probability_token": token,
        "recorded_actual_probability": context["actual_probability"],
        "recorded_expected_probability": context["expected_probability"],
        "serving_probability": float(correct_law[token]),
        "whole_prompt_probability": float(old_law[token]),
        "original_serving_logits": serving_row.tolist(),
        "original_whole_prompt_logits": old_row.tolist(),
    }
    result["recorded_actual_matches_serving"] = bool(
        np.isclose(
            correct_law[token], context["actual_probability"], atol=1e-6, rtol=1e-5
        )
    )
    result["recorded_expected_matches_whole_prompt"] = bool(
        np.isclose(
            old_law[token], context["expected_probability"], atol=1e-6, rtol=1e-5
        )
    )
    branches = restore_recovery_descriptors(*frozen)[0]
    owner = SegmentedKVRows([branches])
    inputs = [context["context_tokens"][-1], *actual[position : position + 3]]
    tx = owner.begin([len(inputs)])
    previous = model._target_verify_row_exact
    try:
        model.configure_target_verify_row_exact(True)
        tapped, _ = model.forward_with_taps(
            mx.array([inputs]), tx.caches, [0, len(model.layers) - 1]
        )
        mx.eval(tapped)
    finally:
        model.configure_target_verify_row_exact(previous)
        tx.abort()
    sequential = restore_recovery_descriptors(*frozen)[0]
    expected = []
    for input_token in inputs:
        output = model(mx.array([[input_token]]), cache=sequential)
        mx.eval(output)
        expected.append(host(output)[0, -1])
    result["same_frozen_prefix_row_exact_vs_original_s1"] = delta(
        host(tapped)[0], np.stack(expected)
    )
    result["same_frozen_prefix_reached_row_vs_serving"] = delta(
        host(tapped)[0, 0], serving_row
    )
    result["target_flag_restored"] = model._target_verify_row_exact == previous
    result["row_exact_input_tokens"] = inputs
    result["passed"] = bool(
        result["recorded_actual_matches_serving"]
        and result["recorded_expected_matches_whole_prompt"]
        and result["same_frozen_prefix_row_exact_vs_original_s1"]["bitwise_equal"]
        and result["same_frozen_prefix_reached_row_vs_serving"]["bitwise_equal"]
        and result["target_flag_restored"]
    )
    return result


def run(args, report):
    import mlx.core as mx
    import psutil
    from validate_xpress_metal_matrix import (
        float32_artifact_forecast,
        guard_float32_footprint,
    )

    from mlx2.adapters.standard_decoder import StandardDecoderAdapter, inspect_artifact

    receipt = json.loads(args.adaptive_receipt.read_text())
    contexts = select_contexts(receipt, args.max_contexts)
    step = receipt.get("context_tokens")
    if type(step) is not int or not 1 <= step <= 4096:
        raise ValueError("receipt must bind a bounded prefill step")
    artifact = inspect_artifact(args.model)
    expected = (
        receipt.get("artifacts", {}).get("identity", {}).get("artifact_fingerprint")
    )
    if expected is None or artifact["identity"]["fingerprint"] != expected:
        raise ValueError("target artifact differs from adaptive receipt")
    if not mx.metal.is_available():
        raise RuntimeError("owned M3 Metal device required")
    mx.set_default_device(mx.gpu)
    device = mx.device_info()
    if "m3" not in json.dumps(device).lower():
        raise RuntimeError("owned M3 Metal device required")
    resident, largest, dtypes = float32_artifact_forecast([args.model])
    if set(dtypes) != {"BF16"}:
        raise ValueError("original homogeneous BF16 target required")
    report["memory_guard"] = guard_float32_footprint(
        resident // 2,
        largest,
        int(psutil.virtual_memory().available),
        int(device.get("max_recommended_working_set_size", 0)),
    )
    report["memory_guard"]["precision"] = "original BF16; no conversion or draft loaded"
    report.update(
        device=device,
        source_identity=source_identity(),
        input_receipt_sha256=hashlib.sha256(
            args.adaptive_receipt.read_bytes()
        ).hexdigest(),
        artifact_identity=artifact["identity"],
        prefill_step=step,
        contexts=[],
    )
    report["reference_conventions"] = {
        "serving": "prompt[:-1] in bounded prompt chunks, S1 anchor, S1 delivered tokens",
        "old_harness": "whole prompt in bounded chunks, S1 delivered tokens",
        "law_temperature": 0.8,
        "cache_hashes": "logical K/V prefix values converted to F32 preserving BF16 values; no unused tail",
        "ownership": "operator acquires and releases locks externally; script acquires none",
    }
    adapter = StandardDecoderAdapter(str(args.model))
    for context in contexts:
        try:
            result = diagnose_context(adapter.model, mx, context, step)
        except Exception as error:  # noqa: BLE001 - keep independent context evidence
            result = {
                "request_index": context["request_index"],
                "position": context["position"],
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        report["contexts"].append(result)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    report["source_unchanged"] = (
        source_identity()["sha256"] == report["source_identity"]["sha256"]
    )
    report["passed"] = (
        all(c["passed"] for c in report["contexts"]) and report["source_unchanged"]
    )
    adapter.close()


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        report = preflight(args)
    except ValueError as error:
        arguments.error(str(error))
    if args.dry_run:
        print(json.dumps(report))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def timeout(*_):
        raise TimeoutError("adaptive reference diagnostic deadline exceeded")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.deadline_seconds)
    try:
        run(args, report)
    except Exception as error:  # noqa: BLE001 - preserve bounded failure receipt
        report.update(
            passed=False,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    finally:
        signal.alarm(0)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "out": str(args.out),
                "error": report.get("error"),
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
