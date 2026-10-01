#!/usr/bin/env python3
"""Bounded real-XPress Metal correctness and adaptive-cost diagnostic.

One target/head load is shared by all cells. Costs are synchronized, uncontrolled
forward diagnostics, never a quoted benchmark or qualification. Run under the
GPU ownership wrapper; this script does not acquire locks or stop services.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import signal
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "mlx2.adaptive-metal-diagnostic.v1"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True, type=Path)
    p.add_argument("--draft", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--num-draft", type=int, default=15)
    p.add_argument("--context-tokens", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--requests", type=int, default=8)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--min-observations", type=int, default=8)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--i-own-the-gpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--target-verify-row-exact", action="store_true")
    p.add_argument("--whole-prompt-reference-diagnostic", action="store_true")
    p.add_argument(
        "--compute-precision",
        choices=("artifact", "float32-diagnostic"),
        default="artifact",
    )
    return p


def plan(args):
    if type(args.target_verify_row_exact) is not bool:
        raise ValueError("target_verify_row_exact must be boolean")
    for name, limit in (
        ("num_draft", 15),
        ("context_tokens", 128),
        ("max_tokens", 48),
        ("requests", 8),
        ("repetitions", 3),
        ("timeout", 900),
    ):
        if not 1 <= getattr(args, name) <= limit:
            raise ValueError(f"{name} must be between 1 and {limit}")
    if args.max_tokens <= args.num_draft or args.context_tokens < 2:
        raise ValueError(
            "max_tokens must exceed num_draft and context needs two tokens"
        )
    if args.requests < 4 or args.min_observations < 1:
        raise ValueError(
            "at least four requests and positive min_observations required"
        )
    return {
        "schema": SCHEMA,
        "target": str(args.target.expanduser().resolve()),
        "draft": str(args.draft.expanduser().resolve()),
        "cohorts": [1, 2, 4],
        "depths": list(range(args.num_draft + 1)),
        "context_tokens": args.context_tokens,
        "requests": args.requests,
        "max_tokens": args.max_tokens,
        "repetitions": args.repetitions,
        "timeout_seconds": args.timeout,
        "source_root": str(ROOT),
        "performance_qualified": False,
        "qualified": False,
        "cost_semantics": "uncontrolled synchronized target forward including bonus; host cache preparation excluded",
        "dry_run": args.dry_run,
        "model_loaded": False,
        "compute_precision": args.compute_precision,
        "target_verify_row_exact": args.target_verify_row_exact,
        "ordinary_reference_convention": "B1 chunked prompt[:-1], original S1 anchor, then original S1 emitted tokens",
        "whole_prompt_reference_diagnostic_requested": args.whole_prompt_reference_diagnostic,
    }


def source_identity():
    digest = hashlib.sha256()
    files = [
        *sorted((ROOT / "src").rglob("*.py")),
        Path(__file__).resolve(),
        ROOT / "scripts/validate_xpress_metal_matrix.py",
    ]
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    revision = (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        if (ROOT / ".git").exists()
        else None
    )
    return {
        "revision": revision,
        "source_bytes_sha256": digest.hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "host": platform.node(),
    }


def token_evidence(active, prompts, outputs, reference=None):
    evidence = []
    for uid, index in active:
        actual = list(outputs[uid])
        entry = {
            "uid": uid,
            "request_index": index,
            "seed": 7100 + index,
            "prompt_tokens": prompts[index],
            "actual_tokens": actual,
        }
        if reference is not None:
            expected = reference[index]
            entry["expected_tokens"] = expected
            differences = [
                i
                for i in range(max(len(actual), len(expected)))
                if i >= len(actual) or i >= len(expected) or actual[i] != expected[i]
            ]
            if differences:
                at = differences[0]
                entry["first_mismatch"] = {
                    "position": at,
                    "actual": actual[at] if at < len(actual) else None,
                    "expected": expected[at] if at < len(expected) else None,
                    "ordinary_context": prompts[index] + expected[:at],
                    "actual_context": prompts[index] + actual[:at],
                }
            else:
                entry["first_mismatch"] = None
        evidence.append(entry)
    return evidence


def artifact_identity(directory):
    """Bind loaded local files by bytes, independent of timestamps or hub paths."""
    directory = directory.expanduser().resolve()
    result = {}
    names = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
    )
    files = sorted(
        {
            *directory.glob("*.safetensors"),
            *(directory / name for name in names if (directory / name).is_file()),
        }
    )
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result[path.name] = {"size": path.stat().st_size, "sha256": digest.hexdigest()}
    if not result:
        raise ValueError("artifact has no files to bind")
    return {"path": str(directory), "files": result}


def audit_trace(frames, stats, census, observed_counts):
    """Independent accounting: actual input geometry and reached labels."""
    executed = sum(b * w for f in frames if f["adaptive"] for b, w in f["shapes"])
    saved = sum(f["saved"] for f in frames if f["adaptive"])
    if executed != stats["external_adaptive_target_rows"]:
        raise AssertionError(
            "actual target geometry differs from adaptive target-row receipt"
        )
    if saved != stats["external_adaptive_trimmed_target_rows"]:
        raise AssertionError(
            "adaptive trim receipt differs from per-round admission budget"
        )
    groups = sum(len(f["shapes"]) for f in frames if f.get("per_request", False))
    if groups != stats.get("external_adaptive_verification_groups", 0):
        raise AssertionError("physical group count differs from grouping receipt")
    if list(census) != list(observed_counts):
        raise AssertionError(
            "estimator labels differ from accepted prefix and first reached rejection"
        )
    return {
        "actual_target_rows": executed,
        "actual_trimmed_admission_rows": saved,
        "observed_used": saved > 0,
        "mixed_depth_rounds": sum(len({w for _, w in f["shapes"]}) > 1 for f in frames),
        "physical_shapes": [f["shapes"] for f in frames],
        "censored_label_counts": list(census),
    }


def ordinary_prefix_cache(model, prompt, *, prefill_step):
    """Match serving's reserved final anchor without selecting alternate math."""
    if not prompt or type(prefill_step) is not int or prefill_step <= 0:
        raise ValueError(
            "ordinary reference needs nonempty tokens and positive prefill step"
        )
    import mlx.core as mx

    cache = model.make_cache()
    for start in range(0, len(prompt) - 1, prefill_step):
        stop = min(start + prefill_step, len(prompt) - 1)
        mx.eval(model(mx.array([prompt[start:stop]]), cache=cache))
    return cache


def ordinary_reference_prefill(model, prompt, *, prefill_step):
    import mlx.core as mx

    cache = ordinary_prefix_cache(model, prompt, prefill_step=prefill_step)
    logits = model(mx.array([[prompt[-1]]]), cache=cache)
    mx.eval(logits, [entry.state for entry in cache])
    return cache, logits


def run(args, receipt):
    # Deferred imports keep --help, --dry-run and rejected CLI calls Metal-free.
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx
    import numpy as np

    from mlx2.adapters.standard_decoder import StandardDecoderAdapter
    from mlx2.runtime.acceptance_estimator import AdaptiveVerificationPolicy
    from mlx2.runtime.cow_cache import (
        restore_recovery_descriptors,
        snapshot_recovery_descriptors,
    )
    from mlx2.runtime.sample_utils import LaneRNG
    from mlx2.runtime.speculative_sampling import softmax

    if not mx.metal.is_available():
        raise RuntimeError("Metal is required")
    mx.set_default_device(mx.gpu)
    receipt["source"] = source_identity()
    receipt["artifact_bytes"] = {
        "target": artifact_identity(args.target),
        "draft": artifact_identity(args.draft),
    }
    receipt["device"] = mx.metal.device_info()
    if args.compute_precision == "float32-diagnostic":
        import psutil
        from validate_xpress_metal_matrix import (
            float32_artifact_forecast,
            guard_float32_footprint,
        )

        resident, largest, dtypes = float32_artifact_forecast([args.target, args.draft])
        receipt["float32_diagnostic"] = {
            "original_dtype_counts": dtypes,
            "memory_guard": guard_float32_footprint(
                resident,
                largest,
                int(psutil.virtual_memory().available),
                int(mx.device_info().get("max_recommended_working_set_size", 0)),
            ),
            "production_precision_changed": False,
        }
    adapter = StandardDecoderAdapter(
        str(args.target),
        execution_policy={
            "draft_model": str(args.draft),
            "num_draft": args.num_draft,
            "target_verify_row_exact": args.target_verify_row_exact,
        },
    )
    receipt["model_loaded"] = True
    receipt["artifacts"] = {
        "identity": adapter.identity,
        "draft_settings": adapter.draft_model.receipt_settings,
        "target_execution": getattr(adapter.model, "external_execution_receipt", None),
    }
    model = adapter.model
    if args.compute_precision == "float32-diagnostic":
        from validate_xpress_metal_matrix import cast_float32_diagnostic

        cast_float32_diagnostic(model, adapter.draft_model)
        if args.target_verify_row_exact:
            model.configure_target_verify_row_exact(True)
    texts = [
        "Write a Python palindrome function and explain its complexity. ",
        "Explain mutexes, races, and thread safety with a short example. ",
        "Write a SQL query counting orders per customer with missing orders. ",
        "Implement FizzBuzz in JavaScript and cover the boundary cases. ",
        "Compare binary search and linear search with precise assumptions. ",
        "Explain why the sky is blue in language a child understands. ",
        "Derive the sum of the first n integers step by step. ",
        "Write a brief story about a telescope finding an unexpected signal. ",
    ]
    prompts = []
    for text in texts[: args.requests]:
        ids = adapter.tokenizer.encode(text * 32, add_special_tokens=False)
        if len(ids) < args.context_tokens:
            raise RuntimeError("tokenized context unexpectedly short")
        prompts.append(ids[: args.context_tokens])
    receipt["prompt_sha256"] = [
        hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts
    ]

    def engine(width, policy=None):
        return adapter.create_external_batch(
            completion_batch_size=width,
            prefill_step_size=args.context_tokens,
            ready_drain="all",
            adaptive_verification=policy,
        )

    def settle(e):
        for lane in e.lanes.values():
            while lane.remaining:
                e._prefill(lane)
        mx.synchronize()

    # Prefill each cohort once. Host cache descriptors reconstruct prefixes for
    # every repetition, avoiding repeated GPU prefill and preserving ownership.
    costs = {}
    draft_samples = {}
    for width in (1, 2, 4):
        e = engine(width)
        e.insert(prompts[:width], max_tokens=[args.max_tokens] * width)
        settle(e)
        lanes = list(e.lanes.values())
        if width == 1:
            reference_cache = ordinary_prefix_cache(
                model, prompts[0], prefill_step=args.context_tokens
            )
            cache_equal = all(
                actual.offset == expected.offset
                and all(
                    np.array_equal(
                        np.asarray(a.astype(mx.float32)),
                        np.asarray(b.astype(mx.float32)),
                    )
                    for a, b in zip(
                        actual.keys_and_values(),
                        expected.keys_and_values(),
                        strict=True,
                    )
                )
                for actual, expected in zip(
                    lanes[0].cache, reference_cache, strict=True
                )
            )
            anchor = mx.array([[lanes[0].anchor]])
            expected_logits = model(anchor, cache=reference_cache)
            transaction = e._target_owner([lanes[0].cache]).begin(lengths=[1])
            try:
                actual_logits, features = model.forward_with_taps(
                    anchor, transaction.caches, e.layers
                )
                mx.eval(actual_logits, features, expected_logits)
                actual_host = np.asarray(actual_logits[0, -1].astype(mx.float32))
                expected_host = np.asarray(expected_logits[0, -1].astype(mx.float32))
                law_error = float(
                    np.max(
                        np.abs(softmax(actual_host, 0.8) - softmax(expected_host, 0.8))
                    )
                )
                receipt["reference_alignment_probe"] = {
                    "context_tokens": len(prompts[0]),
                    "prefix_cache_bitwise_equal": cache_equal,
                    "anchor_logits_bitwise_equal": bool(
                        np.array_equal(actual_host, expected_host)
                    ),
                    "anchor_law_max_absolute_error_at_temperature_0_8": law_error,
                    "passed": cache_equal and law_error <= 1e-4,
                    "ordinary_prefill_chunk_size": args.context_tokens,
                }
            finally:
                transaction.abort()
            if not receipt["reference_alignment_probe"]["passed"]:
                write(args.out, receipt)
                e.close()
                raise AssertionError(
                    "ordinary serving prefix/anchor reference alignment failed"
                )
        snapshots = [snapshot_recovery_descriptors(l.cache) for l in lanes]
        table = []
        samples = []
        for depth in range(args.num_draft + 1):
            timing = []
            tokens = mx.array(
                [[l.anchor] + prompts[i][:depth] for i, l in enumerate(lanes)]
            )
            mx.eval(tokens)
            for repetition in range(args.repetitions + 1):
                restored = [restore_recovery_descriptors(*s)[0] for s in snapshots]
                tx = e._target_owner(restored).begin(lengths=[depth + 1] * width)
                try:
                    mx.synchronize()
                    start = time.perf_counter()
                    logits, taps = model.forward_with_taps(tokens, tx.caches, e.layers)
                    mx.eval(logits, taps)
                    mx.synchronize()
                    elapsed = time.perf_counter() - start
                    if repetition:
                        timing.append(elapsed)
                finally:
                    tx.abort()
            table.append(statistics.median(timing))
            samples.append(timing)
        saved = e._snapshot_round(lanes)
        timing = []
        for repetition in range(args.repetitions + 1):
            e._restore_round(lanes, saved)
            mx.synchronize()
            start = time.perf_counter()
            e._propose(lanes, adaptive_depth=args.num_draft)
            mx.synchronize()
            elapsed = time.perf_counter() - start
            if repetition:
                timing.append(elapsed)
        costs[str(width)] = {"median_seconds": table, "samples_seconds": samples}
        draft_samples[str(width)] = timing
        e.close()
        receipt["cost_measurements"] = costs
        receipt["draft_cost_samples_seconds"] = draft_samples
        write(args.out, receipt)
    policy = {
        "verification_costs": costs["1"]["median_seconds"],
        "verification_costs_by_cohort": {
            w: costs[w]["median_seconds"] for w in ("2", "4")
        },
        "draft_cost": max(t for row in draft_samples.values() for t in row),
        "mode": "per_request",
        "min_observations": args.min_observations,
        "refit_interval": 8,
    }
    receipt["policy"] = policy
    receipt["cost_binding"] = {
        "artifact_identity": adapter.identity,
        "target_verify_row_exact": args.target_verify_row_exact,
        "device": receipt["device"],
        "context_tokens": args.context_tokens,
        "cohorts": [1, 2, 4],
        "application_context_range": [
            args.context_tokens,
            args.context_tokens + args.max_tokens,
        ],
        "context_estimate_note": "fixed-context measured costs are diagnostic estimates when generated histories grow",
        "draft_cost_semantics": "maximum observed synchronized full-trained-block proposal time across measured cohorts; diagnostic estimate, not guaranteed bound",
        "performance_qualified": False,
    }
    AdaptiveVerificationPolicy.from_value(policy, args.num_draft)

    def ordinary(prompt, count):
        cache, logits = ordinary_reference_prefill(
            model, prompt, prefill_step=args.context_tokens
        )
        result = []
        for _ in range(count):
            token = int(mx.argmax(logits[0, -1]).item())
            result.append(token)
            logits = model(mx.array([[token]]), cache=cache)
            mx.eval(logits)
        return result

    ordinary_output = [ordinary(p, args.max_tokens) for p in prompts]
    if args.whole_prompt_reference_diagnostic:
        receipt["whole_prompt_reference_diagnostic"] = []
        for prompt in prompts:
            _, reference = ordinary_reference_prefill(
                model, prompt, prefill_step=args.context_tokens
            )
            whole = model(mx.array([prompt]), cache=model.make_cache())
            mx.eval(reference, whole)
            a, b = (
                np.asarray(value[0, -1].astype(mx.float32))
                for value in (reference, whole)
            )
            receipt["whole_prompt_reference_diagnostic"].append(
                {
                    "scope": "first decode law only; alternate prefill shape, excluded from verifier parity",
                    "logits_max_absolute_delta": float(np.max(np.abs(a - b))),
                    "unit_temperature_law_max_absolute_delta": float(
                        np.max(np.abs(softmax(a, 1.0) - softmax(b, 1.0)))
                    ),
                    "greedy_equal": bool(np.argmax(a) == np.argmax(b)),
                }
            )

    calibrated = [None]

    def generate(width, active_policy, temperature):
        e = engine(width, active_policy)
        out = {}
        laws = {}
        frames = []
        census = np.zeros(args.num_draft, dtype=np.int64)
        failures = []
        check = {"cohort": width, "temperature": temperature, "failures": failures}
        current = [None]
        original_forward = model.forward_with_taps
        original_round = e._round

        def forward(tokens, *a, **k):
            if current[0] is not None:
                current[0]["shapes"].append([int(v) for v in tokens.shape])
            return original_forward(tokens, *a, **k)

        def one_round(cohort):
            before = dict(e.scheduler_stats)
            ready_before = {l.uid: len(l.ready) for l in cohort}
            maxima = [
                min(args.num_draft, l.maximum - l.generated - 1)
                if not l.ordinary
                else 0
                for l in cohort
            ]
            baseline = min(maxima) * len(cohort)
            frame = {
                "shapes": [],
                "adaptive": bool(active_policy) and not all(l.ordinary for l in cohort),
            }
            current[0] = frame
            try:
                return original_round(cohort)
            finally:
                current[0] = None
                frame["saved"] = max(
                    0, baseline - sum(b * (w - 1) for b, w in frame["shapes"])
                )
                frame["per_request"] = e.scheduler_stats.get(
                    "external_adaptive_per_request_rounds", 0
                ) > before.get("external_adaptive_per_request_rounds", 0)
                reported_saved = e.scheduler_stats.get(
                    "external_adaptive_trimmed_target_rows", 0
                ) - before.get("external_adaptive_trimmed_target_rows", 0)
                if (
                    active_policy
                    and frame["adaptive"]
                    and frame["saved"] != reported_saved
                ):
                    failures.append(
                        "physical proposal rows disagree with round trimming receipt"
                    )
                if active_policy:
                    for lane in cohort:
                        for response in list(lane.ready)[ready_before[lane.uid] :]:
                            reported = response.speculative_receipt[
                                "adaptive_verification"
                            ]["round_verify_width"]
                            if reported not in {w for _, w in frame["shapes"]}:
                                failures.append(
                                    "response verify width has no matching physical forward"
                                )
                frames.append(frame)

        original_observe = e._adaptive_observe

        def observe(blocks, decisions, cohort=None):
            for block, decision in zip(blocks, decisions):
                if block is None:
                    continue
                length = int(block.lengths[0])
                accepted = min(decision.accepted, len(decision.emitted))
                rejected = decision.accepted < length and decision.accepted < len(
                    decision.emitted
                )
                census[:accepted] += 1
                if rejected:
                    census[accepted] += 1
            return original_observe(blocks, decisions, cohort)

        e._round = one_round
        e._adaptive_observe = observe
        model.forward_with_taps = forward
        pending = 0
        active = []
        try:
            while pending < len(prompts) or e.lanes:
                while pending < len(prompts) and len(e.lanes) < width:
                    uid = e.insert(
                        [prompts[pending]],
                        max_tokens=[args.max_tokens],
                        lane_rngs=[LaneRNG(7100 + pending)],
                        sampling_configs=[{"sampling_temp": temperature}],
                    )[0]
                    active.append((uid, pending))
                    out[uid] = []
                    laws[uid] = []
                    pending += 1
                _, responses = e.next()
                for response in responses:
                    out[response.uid].append(response.token)
                    if temperature:
                        laws[response.uid].append(
                            np.asarray(response.logprobs.astype(mx.float32))
                        )
            outputs = [out[uid] for uid, _ in active]
            check["output_lengths"] = [len(row) for row in outputs]
            if len(outputs) != len(prompts) or any(
                len(row) != args.max_tokens for row in outputs
            ):
                failures.append("generation did not deliver the requested token budget")
            check["stats"] = dict(e.scheduler_stats)
            if temperature:
                maximum_error = 0.0
                for uid, index in active:
                    cache, logits = ordinary_reference_prefill(
                        model, prompts[index], prefill_step=args.context_tokens
                    )
                    for position, (token, actual_logp) in enumerate(
                        zip(out[uid], laws[uid])
                    ):
                        expected = softmax(
                            np.asarray(logits[0, -1].astype(mx.float32)), temperature
                        )
                        error = float(np.max(np.abs(np.exp(actual_logp) - expected)))
                        maximum_error = max(maximum_error, error)
                        if not np.allclose(
                            np.exp(actual_logp), expected, atol=1e-4, rtol=1e-3
                        ):
                            at = int(np.argmax(np.abs(np.exp(actual_logp) - expected)))
                            check.setdefault("sampled_law_mismatches", []).append(
                                {
                                    "uid": uid,
                                    "request_index": index,
                                    "position": position,
                                    "context_tokens": prompts[index]
                                    + out[uid][:position],
                                    "largest_error_token_id": at,
                                    "expected_probability": float(expected[at]),
                                    "actual_probability": float(
                                        np.exp(actual_logp)[at]
                                    ),
                                    "maximum_absolute_error": error,
                                }
                            )
                        logits = model(mx.array([[token]]), cache=cache)
                        mx.eval(logits)
                check["sampled_target_law_max_absolute_error"] = maximum_error
                if check.get("sampled_law_mismatches"):
                    failures.append("sampled target law differs from ordinary target")
            else:
                check["ordinary_greedy_exact"] = outputs == ordinary_output
                if not check["ordinary_greedy_exact"]:
                    failures.append("greedy output differs from ordinary target")
            if active_policy:
                check["physical_frames"] = frames
                check["censored_label_counts"] = census.tolist()
                check["observed_label_counts"] = (
                    e.acceptance_estimator.observed_counts.tolist()
                )
                try:
                    check["trace"] = audit_trace(
                        frames,
                        e.scheduler_stats,
                        census.tolist(),
                        e.acceptance_estimator.observed_counts.tolist(),
                    )
                except AssertionError as error:
                    failures.append(str(error))
                calibrated[0] = copy.deepcopy(e.acceptance_estimator)
                check["estimator"] = {
                    "slope": e.acceptance_estimator.slope,
                    "intercepts": e.acceptance_estimator.intercepts.tolist(),
                    "rounds": e.acceptance_estimator.rounds,
                    "refits": e.acceptance_estimator.refits,
                }
            return check
        except TimeoutError:
            raise
        except Exception as error:  # noqa: BLE001 - preserve failed cell, continue independent cells
            failures.append(f"{type(error).__name__}: {error}")
            check["error_traceback"] = traceback.format_exc()
            return check
        finally:
            check["stats"] = dict(e.scheduler_stats)
            check["adaptive_selected"] = active_policy is not None
            check["physical_frames"] = frames
            check["censored_label_counts"] = census.tolist()
            if active_policy:
                check["observed_label_counts"] = (
                    e.acceptance_estimator.observed_counts.tolist()
                )
            check["token_evidence"] = token_evidence(
                active, prompts, out, ordinary_output if not temperature else None
            )
            check["passed"] = not failures
            if active_policy:
                check["natural_adaptive_trimmed_rows"] = e.scheduler_stats[
                    "external_adaptive_trimmed_target_rows"
                ]
                check["natural_adaptive_observed_used"] = (
                    check["natural_adaptive_trimmed_rows"] > 0
                )
            model.forward_with_taps = original_forward
            e.close()

    receipt["generation_checks"] = []
    for width in (1, 2, 4):
        for selected in (None, policy):
            check = generate(width, selected, 0.0)
            check["adaptive_selected"] = selected is not None
            receipt["generation_checks"].append(check)
            write(args.out, receipt)
    receipt["generation_checks"].append(generate(4, policy, 0.8))
    write(args.out, receipt)

    # These Metal cells exercise real proposal/tensor paths while the estimator
    # remains cold, at a forced exploration boundary, or with unknown B3 costs.
    receipt["backstop_checks"] = []
    for kind, width in (("cold", 2), ("exploration", 2), ("unknown_cohort", 3)):
        p = {**policy, "min_observations": 10**9} if kind == "cold" else policy
        e = engine(width, p)
        check = {"kind": kind, "cohort": width, "passed": False, "failures": []}
        try:
            e.insert(prompts[:width], max_tokens=[args.max_tokens] * width)
            settle(e)
            if kind != "cold" and calibrated[0] is not None:
                e.acceptance_estimator = copy.deepcopy(calibrated[0])
            if kind == "exploration":
                e.acceptance_estimator.rounds = e.adaptive_policy.full_depth_interval
            e._round(list(e.lanes.values()))
            check["depth"] = e.scheduler_stats["external_adaptive_round_depth"]
            check["trimmed_target_rows"] = e.scheduler_stats[
                "external_adaptive_trimmed_target_rows"
            ]
            if check["trimmed_target_rows"]:
                check["failures"].append(
                    f"{kind} backstop trimmed a cold/exploration/unknown cohort"
                )
            if check["depth"] != args.num_draft:
                check["failures"].append(f"{kind} did not preserve full proposal depth")
            check["passed"] = not check["failures"]
        except TimeoutError:
            raise
        except Exception as error:  # noqa: BLE001 - independent failure evidence
            check["failures"].append(f"{type(error).__name__}: {error}")
            check["error_traceback"] = traceback.format_exc()
        finally:
            check["stats"] = dict(e.scheduler_stats)
            e.close()
            receipt["backstop_checks"].append(check)
            write(args.out, receipt)
    receipt["passed"] = all(
        c["passed"]
        for c in [*receipt["generation_checks"], *receipt["backstop_checks"]]
    )
    receipt["adaptive_observed_used"] = any(
        c.get("natural_adaptive_observed_used", False)
        for c in receipt["generation_checks"]
    )
    receipt["mixed_depth_observed"] = any(
        c.get("trace", {}).get("mixed_depth_rounds", 0) > 0
        for c in receipt["generation_checks"]
    )
    receipt["note"] = (
        "Natural measured-cost decisions may preserve fixed depth. Correctness does not establish speedup or production qualification."
    )
    receipt["target_execution"] = getattr(model, "external_execution_receipt", None)


def write(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        receipt = plan(args)
    except ValueError as error:
        parser().error(str(error))
    if args.dry_run:
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if not args.i_own_the_gpu:
        parser().error("--i-own-the-gpu required; run under GPU ownership wrapper")
    receipt["started_at"] = time.time()
    receipt["passed"] = False

    def timeout(*_):
        raise TimeoutError("diagnostic timeout reached")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.timeout)
    try:
        run(args, receipt)
    except BaseException as error:  # noqa: BLE001 - always write failed/timeout receipt
        receipt["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        return_code = 1
    else:
        return_code = 0 if receipt["passed"] else 1
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        receipt["elapsed_seconds"] = time.time() - receipt["started_at"]
        write(args.out, receipt)
    print(
        json.dumps(
            {
                "receipt": str(args.out),
                "passed": receipt["passed"],
                "adaptive_observed_used": receipt.get("adaptive_observed_used", False),
            }
        )
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
