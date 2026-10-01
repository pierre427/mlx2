#!/usr/bin/env python3
"""Real-checkpoint top-15 complete-path Metal checks and matched cost probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import statistics
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = (
    "Write a Python palindrome function and explain its complexity. ",
    "Explain mutexes, races, and thread safety with a short example. ",
    "Write a SQL query counting orders per customer. ",
    "Explain why the sky is blue to a child. ",
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "draft", "out"):
        result.add_argument("--" + key, type=Path, required=True)
    result.add_argument(
        "--compute-precision",
        choices=("artifact", "float32-diagnostic"),
        default="artifact",
    )
    result.add_argument("--context-tokens", type=int, default=128)
    result.add_argument("--max-tokens", type=int, default=32)
    result.add_argument("--repetitions", type=int, default=2)
    result.add_argument("--deadline-seconds", type=int, default=900)
    result.add_argument("--i-own-the-gpu", action="store_true")
    result.add_argument("--target-verify-row-exact", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def preflight(args):
    for name, low, high in (
        ("context_tokens", 16, 128),
        ("max_tokens", 17, 48),
        ("repetitions", 1, 3),
        ("deadline_seconds", 1, 900),
    ):
        if not low <= getattr(args, name) <= high:
            raise ValueError(f"{name} must be in [{low},{high}]")
    if not args.dry_run and not args.i_own_the_gpu:
        raise ValueError("Metal execution requires --i-own-the-gpu under both locks")
    return {
        "schema": "mlx2.complete-continuation-metal.v1",
        "qualified": False,
        "performance_claim": False,
        "will_execute": not args.dry_run,
        "model": str(args.model.resolve()),
        "draft": str(args.draft.resolve()),
        "compute_precision": args.compute_precision,
        "target_verify_row_exact": args.target_verify_row_exact,
        "max_sequences": 15,
        "max_depth": 15,
        "cost_path_widths": [1, 5, 15],
        "cost_depths": list(range(16)),
        "context_tokens": args.context_tokens,
        "draft_cost_authority": "resident B1 fixed-context proposal probe; no published feedback",
        "checks": [],
        "passed": False,
    }


def sources():
    paths = [
        Path(__file__),
        ROOT / "scripts/validate_xpress_metal_matrix.py",
        *sorted((ROOT / "src").rglob("*.py")),
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def bind_compute_precision(draft, precision):
    """Bind harness computation policy before any proposal/critic observation."""
    from dataclasses import replace

    from mlx2.runtime.proposal_pool import ProposalPool, ProposalRankingRegistry

    previous = draft.proposal_pool
    if previous._pending or previous.feedback_revision:
        raise ValueError("precision binding must precede proposal generation")
    revision = hashlib.sha256(
        json.dumps(
            [
                draft.session.session_revision,
                "metal_harness_compute_precision",
                precision,
            ]
        ).encode()
    ).hexdigest()
    draft.session = replace(draft.session, session_revision=revision)
    draft.source_records = {
        key: replace(source, session_revision=revision)
        for key, source in draft.source_records.items()
    }
    registry = previous.ranking_registry
    isolated = ProposalRankingRegistry(
        prior_acceptance=registry.prior_acceptance,
        prior_strength=registry.prior_strength,
        model_shrinkage=registry.model_shrinkage,
        session_shrinkage=registry.session_shrinkage,
        confidence_bin_width=registry.confidence_bin_width,
        confidence_shrinkage=registry.confidence_shrinkage,
        max_models=registry.max_models,
        max_sessions=registry.max_sessions,
        max_sources=registry.max_sources,
        max_depth=registry.max_depth,
    )
    draft.proposal_pool = ProposalPool(
        draft.session,
        draft.source_records.values(),
        ranking_registry=isolated,
        max_depth=previous.max_depth,
        score_mode=previous.score_mode,
        cost_units=previous.cost_units,
    )
    return revision


def measure_proposal_cost(
    probe, lane, synchronize, repetitions, clock=time.perf_counter
):
    """Resident B1 full-provider cost at one fixed committed context.

    Snapshot/restore and ticket cleanup are outside the timed interval. Probes
    never verify target outputs or publish labels/costs to the shared critic.
    """
    pool = probe.draft.proposal_pool
    if pool._pending:
        raise ValueError("cost probe requires no pending selections")
    local_revision, global_revision = (
        pool.feedback_revision,
        pool.ranking_registry.revision,
    )
    snapshot = probe._snapshot_round([lane])
    samples, path_counts = [], []
    for iteration in range(repetitions + 1):
        probe._continuation_open_selections = []
        try:
            synchronize()
            started = clock()
            blocks = probe._propose([lane])
            synchronize()
            elapsed = clock() - started
            if len(blocks) != 1 or blocks[0] is None:
                raise AssertionError("resident full proposal probe produced no block")
            count = len(blocks[0].continuation_selection.paths)
            if count != 15:
                raise AssertionError(
                    "resident probe did not produce 15 complete sequences"
                )
            path_counts.append(count)
            if iteration:
                samples.append(elapsed)
        finally:
            selections = [
                *probe._continuation_open_selections,
                *getattr(probe.draft, "last_continuation_selections", ()),
            ]
            seen = set()
            for selection in selections:
                if selection is not None and id(selection) not in seen:
                    seen.add(id(selection))
                    pool.discard(selection)
            probe._continuation_open_selections = []
            probe._restore_round([lane], snapshot)
    if pool._pending or (pool.feedback_revision, pool.ranking_registry.revision) != (
        local_revision,
        global_revision,
    ):
        raise AssertionError(
            "diagnostic proposal probes changed committed ranking evidence"
        )
    return {
        "seconds": statistics.median(samples),
        "samples_seconds": samples,
        "selected_sequence_counts": path_counts,
        "requests": 1,
        "context_scope": "one fixed committed B1 context",
        "includes": "resident backbone plus all admitted provider enumeration/ranking",
        "excludes": "snapshot/restore and ticket cleanup",
        "critic_observations_published": False,
        "performance_qualified": False,
    }


def run(args, report):
    import mlx.core as mx
    import numpy as np

    from mlx2.adapters.standard_decoder import StandardDecoderAdapter
    from mlx2.runtime.cow_cache import (
        restore_recovery_descriptors,
        snapshot_recovery_descriptors,
    )
    from mlx2.runtime.sample_utils import LaneRNG

    mx.set_default_device(mx.gpu)
    device = mx.device_info()
    if not mx.metal.is_available() or "m3" not in json.dumps(device).lower():
        raise RuntimeError("this validation requires the owned M3 Metal device")
    report["device"] = device
    report["source_sha256"] = sources()
    if args.compute_precision == "float32-diagnostic":
        import psutil
        from validate_xpress_metal_matrix import (
            float32_artifact_forecast,
            guard_float32_footprint,
        )

        resident, largest, _ = float32_artifact_forecast([args.model, args.draft])
        report["memory_guard"] = guard_float32_footprint(
            resident,
            largest,
            int(psutil.virtual_memory().available),
            int(device.get("max_recommended_working_set_size", 0)),
        )
    adapter = StandardDecoderAdapter(
        str(args.model),
        execution_policy={
            "draft_model": str(args.draft),
            "num_draft": 15,
            "target_verify_row_exact": args.target_verify_row_exact,
            "continuation_pool": {"limit": 15, "ngram_min": 1, "ngram_max": 3},
        },
    )
    if args.compute_precision == "float32-diagnostic":
        from validate_xpress_metal_matrix import cast_float32_diagnostic

        cast_float32_diagnostic(adapter.model, adapter.draft_model)
    model, draft = adapter.model, adapter.draft_model
    report["critic_session_revision"] = bind_compute_precision(
        draft, args.compute_precision
    )
    report["artifact_identity"] = adapter.identity
    report["draft_settings"] = draft.receipt_settings
    prompts = [
        adapter.tokenizer.encode(text * 32, add_special_tokens=False)[
            : args.context_tokens
        ]
        for text in PROMPTS
    ]

    def fresh(ids):
        cache = model.make_cache()
        for start in range(0, len(ids) - 1, 128):
            mx.eval(
                model(
                    mx.array([ids[start : min(start + 128, len(ids) - 1)]]), cache=cache
                )
            )
        logits = model(mx.array([[ids[-1]]]), cache=cache)[0, -1]
        mx.eval(logits)
        return cache, logits

    def ordinary(ids):
        cache, logits = fresh(ids)
        result = []
        for index in range(args.max_tokens):
            token = int(mx.argmax(logits).item())
            result.append(token)
            if token in probe.stops:
                break
            if index + 1 < args.max_tokens:
                logits = model(mx.array([[token]]), cache=cache)[0, -1]
        return result

    def engine(width, policy=None):
        return adapter.create_external_batch(
            completion_batch_size=width,
            prefill_step_size=args.context_tokens,
            ready_drain="all",
            adaptive_verification=policy,
        )

    probe = engine(1)
    uid = probe.insert([prompts[0]], max_tokens=[args.max_tokens])[0]
    lane = probe.lanes[uid]
    while lane.remaining:
        probe._prefill(lane)
    frozen = snapshot_recovery_descriptors(lane.cache)
    proposal_cost = measure_proposal_cost(probe, lane, mx.synchronize, args.repetitions)
    report["proposal_cost"] = proposal_cost
    costs, observations = {}, {}
    for width in (1, 5, 15):
        medians, details = [], []
        for depth in range(16):
            samples = []
            for iteration in range(args.repetitions + 1):
                caches = [
                    restore_recovery_descriptors(*frozen)[0] for _ in range(width)
                ]
                transaction = probe._target_owner(caches).begin(
                    lengths=[depth + 1] * width
                )
                inputs = mx.array(
                    [
                        [lane.anchor]
                        + [int((index + position) % 100) for position in range(depth)]
                        for index in range(width)
                    ]
                )
                try:
                    mx.synchronize()
                    started = time.perf_counter()
                    logits, hidden = model.forward_with_taps(
                        inputs, transaction.caches, probe.layers
                    )
                    mx.eval(logits, hidden)
                    mx.synchronize()
                    elapsed = time.perf_counter() - started
                    if iteration:
                        samples.append(elapsed)
                finally:
                    transaction.abort()
            medians.append(float(np.median(samples)))
            details.append(samples)
        costs[str(width)] = medians
        observations[str(width)] = details
        print(json.dumps({"event": "cost_width_measured", "width": width}), flush=True)
    report["continuation_costs"] = costs
    report["cost_samples_seconds"] = observations
    report["cost_binding"] = {
        "source_sha256": report["source_sha256"],
        "artifact_identity": adapter.identity,
        "compute_precision": args.compute_precision,
        "context_tokens": args.context_tokens,
        "requests_per_forward": 1,
        "path_width_semantics": "physical independent continuation rows, including bonus",
        "performance_qualified": False,
        "critic_session_revision": report["critic_session_revision"],
        "proposal_cost_seconds": proposal_cost["seconds"],
        "scope": "fixed-context resident probes; timing informational only",
    }

    original = model.forward_with_taps
    frames = []

    def tracked(inputs, *positional, **keywords):
        frames.append(list(inputs.shape))
        return original(inputs, *positional, **keywords)

    model.forward_with_taps = tracked
    try:
        for adaptive, mixed_sampling in ((False, False), (True, False), (True, True)):
            policy = (
                {
                    "continuation_costs": costs,
                    "min_observations": 1,
                    "full_depth_interval": 8,
                    "draft_cost": proposal_cost["seconds"],
                }
                if adaptive
                else None
            )
            batch = engine(2, policy)
            temperatures = [0.0, 0.8, 0.0, 0.8] if mixed_sampling else [0.0] * 4
            ids = batch.insert(
                prompts,
                max_tokens=[args.max_tokens] * 4,
                lane_rngs=[LaneRNG(320 + index) for index in range(4)],
                sampling_configs=[{"sampling_temp": temp} for temp in temperatures],
            )
            output, ends = {uid: [] for uid in ids}, {}
            law_checks = {}
            if mixed_sampling:
                from mlx2.runtime.sample_utils import make_transformed_logprobs
                from mlx2.runtime.speculative_sampling import probability

                references = {}
                transform = make_transformed_logprobs(0.8)
                for uid, prompt, temp in zip(ids, prompts, temperatures, strict=True):
                    if temp:
                        _cache, first_logits = fresh(prompt)
                        references[uid] = probability(
                            np.asarray(mx.exp(transform(first_logits[None])[0]))
                        )
                original_law = batch._target_law

                def checked_law(
                    lane,
                    logits,
                    history,
                    *positional,
                    _law=original_law,
                    _references=references,
                    _checks=law_checks,
                    **keywords,
                ):
                    law = _law(lane, logits, history, *positional, **keywords)
                    if lane.uid in _references and lane.uid not in _checks:
                        _checks[lane.uid] = {
                            "first_processed_law_l1_vs_ordinary": float(
                                np.abs(law - _references[lane.uid]).sum()
                            ),
                            "history_tokens": list(history),
                            "temperature": 0.8,
                            "law_authority": "ordinary S1 versus canonical ranked target branch before RNG draw",
                            "empirical_distribution_qualification": False,
                        }
                    return law

                batch._target_law = checked_law
            start = len(frames)
            failures = []
            for _ in range(512):
                _, responses = batch.next()
                failures.extend(str(value) for value in batch.take_lane_failures())
                for response in responses:
                    output[response.uid].append(response.token)
                    if response.finish_reason:
                        ends[response.uid] = response
                if not batch.lanes:
                    break
            actual_frames = frames[start:]
            tokens_equal = [
                output[uid] == ordinary(prompt) if temp == 0 else None
                for uid, prompt, temp in zip(ids, prompts, temperatures, strict=True)
            ]
            if any(value is False for value in tokens_equal):
                failures.append("greedy continuation-pool output differs from ordinary")
            if not any(shape[0] == 15 for shape in actual_frames):
                failures.append("no physical 15-path target forward observed")
            if set(ends) != set(ids):
                failures.append("generation did not terminate all four requests")
            if mixed_sampling and (
                len(law_checks) != 2
                or any(
                    value["first_processed_law_l1_vs_ordinary"] > 0.01
                    for value in law_checks.values()
                )
            ):
                failures.append(
                    "sampled canonical target branch law differs from ordinary S1 or was not exercised"
                )
            check = {
                "adaptive": adaptive,
                "mixed_sampling": mixed_sampling,
                "passed": not failures,
                "failures": failures,
                "tokens_equal": tokens_equal,
                "tokens": output,
                "sampling_law_checks": law_checks,
                "frames": actual_frames,
                "stats": dict(batch.scheduler_stats),
                "receipts": {uid: end.speculative_receipt for uid, end in ends.items()},
            }
            report["checks"].append(check)
            print(
                json.dumps(
                    {
                        "event": "generation_checked",
                        "adaptive": adaptive,
                        "passed": check["passed"],
                    }
                ),
                flush=True,
            )
    finally:
        model.forward_with_taps = original
        probe.close()
        adapter.close()
    report["source_unchanged"] = sources() == report["source_sha256"]
    report["passed"] = (
        all(check["passed"] for check in report["checks"])
        and report["source_unchanged"]
    )


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

    def deadline(*_):
        raise TimeoutError("complete-path Metal validation deadline exceeded")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.deadline_seconds)
    try:
        run(args, report)
    except Exception as error:  # noqa: BLE001 - preserve all failed validation evidence
        report.update(
            passed=False,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    finally:
        signal.alarm(0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "out": str(args.out),
                "error": report.get("error"),
            }
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
