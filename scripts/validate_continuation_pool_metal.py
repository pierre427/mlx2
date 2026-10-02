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
from contextlib import contextmanager
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
    result.add_argument("--prefill-step", type=int, default=128)
    result.add_argument("--batch-size", type=int, choices=(1, 2, 4), default=2)
    result.add_argument("--max-tokens", type=int, default=32)
    result.add_argument("--repetitions", type=int, default=2)
    result.add_argument("--deadline-seconds", type=int, default=900)
    result.add_argument("--i-own-the-gpu", action="store_true")
    result.add_argument("--target-verify-row-exact", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def preflight(args):
    for name, low, high in (
        ("context_tokens", 16, 4096),
        ("prefill_step", 16, 512),
        ("max_tokens", 17, 48),
        ("repetitions", 1, 3),
        ("deadline_seconds", 1, 900),
    ):
        if not low <= getattr(args, name) <= high:
            raise ValueError(f"{name} must be in [{low},{high}]")
    if not args.dry_run and not args.i_own_the_gpu:
        raise ValueError("Metal execution requires --i-own-the-gpu under both locks")
    if args.batch_size not in (1, 2, 4):
        raise ValueError("batch_size must be 1, 2 or 4")
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
        "prefill_step": args.prefill_step,
        "batch_size": args.batch_size,
        "ordinary_reference_convention": "B1 prompt[:-1] in matching prefill chunks, then original S1 anchor and emitted tokens",
        "memory_admission": {"status": "not_measured", "model_constructed": False},
        "draft_cost_authority": "resident B1 fixed-context proposal probe; no published feedback",
        "checks": [],
        "continuation_costs_complete": False,
        "generation_completed": False,
        "law_capture_limit_rows": 4 * args.max_tokens,
        "passed": False,
    }


def exact_prompt_tokens(tokenizer, text, length):
    """Keep the legacy 32-repeat prefix and extend the bounded corpus as needed."""
    for repeats in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        tokens = list(tokenizer.encode(text * repeats, add_special_tokens=False))
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("prompt tokenizer returned invalid token IDs")
        if len(tokens) >= length:
            return tokens[:length]
        if not tokens:
            raise ValueError("prompt tokenizer returned no token IDs")
    raise ValueError("bounded prompt corpus is shorter than the requested context")


def context_memory_forecast(
    target,
    draft,
    parameter_bytes,
    args,
    *,
    target_element_bytes=None,
    draft_element_bytes=None,
):
    """Conservative phase reservations, not a measured MLX peak or donation claim.

    The executor currently verifies each request separately. Reserve all 15
    branches for every admitted cohort request anyway, plus authoritative and
    rollback versions, so safety does not depend on lazy graph release timing.
    """

    def positive(config, key):
        value = config.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"memory forecast requires positive {key}")
        return value

    if type(parameter_bytes) is not int or parameter_bytes <= 0:
        raise ValueError("memory forecast requires positive parameter bytes")
    default_bytes = 4 if args.compute_precision == "float32-diagnostic" else 2
    target_element_bytes = (
        default_bytes if target_element_bytes is None else target_element_bytes
    )
    draft_element_bytes = (
        default_bytes if draft_element_bytes is None else draft_element_bytes
    )
    if any(
        type(value) is not int or value not in (2, 4)
        for value in (target_element_bytes, draft_element_bytes)
    ):
        raise ValueError("memory forecast requires known floating element widths")
    capacity = ((args.context_tokens + args.max_tokens + 16 + 255) // 256) * 256

    def kv_bytes(config, element_bytes):
        return (
            2
            * positive(config, "num_hidden_layers")
            * positive(config, "num_key_value_heads")
            * positive(config, "head_dim")
            * element_bytes
            * capacity
        )

    target_row = kv_bytes(target, target_element_bytes)
    draft_row = kv_bytes(draft, draft_element_bytes)
    vocab = positive(target, "vocab_size")
    hidden = positive(target, "hidden_size")
    taps = draft.get("dflash_config", {}).get("target_layer_ids")
    if not isinstance(taps, list) or not taps:
        raise ValueError("memory forecast requires target tap metadata")
    # Each sampled law may retain an F32 raw row and F64 probability row;
    # reserve that pair for all four requests, including response diagnostics.
    captured_laws = 4 * args.max_tokens * (12 * vocab + 8 * capacity)
    forward_outputs = (
        2
        * args.batch_size
        * 15
        * 16
        * target_element_bytes
        * (vocab + hidden * len(taps))
    )
    phases = {
        "cost_probe": {
            "target_rows": 16,
            "draft_rows": 1,
            "capture_bytes": 0,
            "forward_output_bytes": forward_outputs,
        },
        "generation": {
            "target_rows": 4 + 16 * args.batch_size,
            "draft_rows": 4,
            "capture_bytes": captured_laws,
            "forward_output_bytes": forward_outputs,
        },
        "ordinary_law_audit": {
            "target_rows": 5,
            "draft_rows": 4,
            "capture_bytes": captured_laws,
            "forward_output_bytes": 0,
        },
    }
    for phase in phases.values():
        phase["required_bytes"] = (
            parameter_bytes
            + phase["target_rows"] * target_row
            + phase["draft_rows"] * draft_row
            + phase["capture_bytes"]
            + phase["forward_output_bytes"]
            + (4 << 30)
        )
    return {
        "authority": "header/config-based conservative reservation; not observed peak memory",
        "parameter_bytes": parameter_bytes,
        "kv_capacity_tokens": capacity,
        "target_element_bytes": target_element_bytes,
        "draft_element_bytes": draft_element_bytes,
        "target_row_bytes": target_row,
        "draft_row_bytes": draft_row,
        "maximum_admitted_requests": args.batch_size,
        "resident_request_count": 4,
        "full_paths_per_request": 15,
        "maximal_cohort_branch_reservation": 15 * args.batch_size,
        "transient_allowance_bytes": 4 << 30,
        "phases": phases,
        "required_bytes": max(phase["required_bytes"] for phase in phases.values()),
        "status": "forecast_only",
    }


def artifact_memory_metadata(args):
    """Read only configs and safetensors headers, before a model constructor."""
    from validate_xpress_metal_matrix import float32_artifact_forecast

    parameters, elements = 0, []
    for path in (args.model, args.draft):
        expanded, _largest, dtypes = float32_artifact_forecast([path])
        widths = {"BF16": 2, "F16": 2, "F32": 4}
        if not dtypes or not set(dtypes).issubset(widths):
            raise ValueError("memory forecast requires floating source tensors")
        if args.compute_precision == "float32-diagnostic":
            parameters += expanded
            elements.append(4)
        else:
            parameters += sum(widths[key] * count for key, count in dtypes.items())
            elements.append(max(widths[key] for key in dtypes))
    target = json.loads((args.model / "config.json").read_text())
    draft = json.loads((args.draft / "config.json").read_text())
    return target, draft, parameters, *elements


def construct_admitted_adapter(args, report, device, factory, available_bytes):
    target, draft, parameters, target_bytes, draft_bytes = artifact_memory_metadata(
        args
    )
    forecast = context_memory_forecast(
        target,
        draft,
        parameters,
        args,
        target_element_bytes=target_bytes,
        draft_element_bytes=draft_bytes,
    )
    recommended = int(device.get("max_recommended_working_set_size", 0))
    forecast.update(available_bytes=available_bytes, recommended_bytes=recommended)
    report["memory_admission"] = forecast
    if (
        type(available_bytes) is not int
        or available_bytes <= 0
        or recommended <= 0
        or forecast["required_bytes"] > min(available_bytes, recommended)
    ):
        forecast["status"] = "refused"
        report["skipped_geometries"] = [
            {
                "reason": "memory_admission",
                "context_tokens": args.context_tokens,
                "batch_size": args.batch_size,
                "paths_per_request": 15,
                "cost_policy_exported": False,
                "executed": False,
            }
        ]
        raise ValueError(
            "full 15-path context ladder exceeds the available/recommended memory reservation"
        )
    forecast["status"] = "admitted"
    adapter = factory(
        str(args.model),
        execution_policy={
            "draft_model": str(args.draft),
            "num_draft": 15,
            "target_verify_row_exact": args.target_verify_row_exact,
            "continuation_pool": {"limit": 15, "ngram_min": 1, "ngram_max": 3},
        },
    )
    forecast["model_constructed"] = True
    return adapter


@contextmanager
def managed_batch(adapter, args, *, width=None, policy=None):
    batch = adapter.create_external_batch(
        completion_batch_size=args.batch_size if width is None else width,
        prefill_step_size=args.prefill_step,
        ready_drain="all",
        adaptive_verification=policy,
    )
    original_law = batch._target_law
    try:
        yield batch
    finally:
        batch._target_law = original_law
        batch.close()


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


def measure_context_costs(adapter, args, prompt, mx, np, report):
    """No probe lane, snapshot or transient tensor escapes this phase."""
    with managed_batch(adapter, args, width=1) as probe:
        uid = probe.insert([prompt], max_tokens=[args.max_tokens])[0]
        lane = probe.lanes[uid]
        while lane.remaining:
            probe._prefill(lane)
        frozen = snapshot_probe_cache(lane.cache)
        offsets = [int(layer.offset) for layer in lane.cache]
        if not offsets or any(offset != len(prompt) - 1 for offset in offsets):
            raise AssertionError(
                "cost probe does not match the reserved prompt-anchor boundary"
            )
        proposal = measure_proposal_cost(probe, lane, mx.synchronize, args.repetitions)
        costs, observations = {}, {}
        report["continuation_costs"] = costs
        report["cost_samples_seconds"] = observations
        for width in (1, 5, 15):
            medians, details = [], []
            for depth in range(16):
                samples = []
                for iteration in range(args.repetitions + 1):
                    caches = [restore_probe_cache(frozen) for _ in range(width)]
                    transaction = probe._target_owner(caches).begin(
                        lengths=[depth + 1] * width
                    )
                    inputs = mx.array(
                        [
                            [lane.anchor]
                            + [
                                int((index + position) % 100)
                                for position in range(depth)
                            ]
                            for index in range(width)
                        ]
                    )
                    try:
                        mx.synchronize()
                        started = time.perf_counter()
                        logits, hidden = adapter.model.forward_with_taps(
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
            print(
                json.dumps({"event": "cost_width_measured", "width": width}), flush=True
            )
        report["continuation_costs_complete"] = True
        return costs, observations, proposal, frozenset(probe.stops), offsets


def snapshot_probe_cache(cache):
    from mlx2.runtime.cow_cache import snapshot_recovery_descriptors

    return snapshot_recovery_descriptors(cache)


def restore_probe_cache(snapshot):
    from mlx2.runtime.cow_cache import restore_recovery_descriptors

    return restore_recovery_descriptors(*snapshot)[0]


def capture_law_observation(
    observations, uid, logits, law, history, reachable, mx, np, limit
):
    """Bound retained diagnostics; overflow fails the cell before another copy.

    No callback row is silently discarded. This cap also covers unused future
    callbacks if a final-budget chain round runs beside a longer request.
    """
    if sum(len(rows) for rows in observations.values()) >= limit:
        raise ValueError("target-law diagnostic capture exceeds reserved memory rows")
    observations[uid].append(
        {
            "raw_logits": np.asarray(logits.astype(mx.float32)).copy(),
            "law": law.copy(),
            "history_tokens": list(history),
            "reachable": reachable,
        }
    )


def ordinary_reached_rows(model, mx, prompt, emitted, prefill_step):
    """Original serving convention: prompt prefix chunks, then only S1 steps.

    Generated history must never be re-prefilled as a block: that would change
    the reduction geometry whose ordinary law this diagnostic is auditing.
    """
    if not prompt or prefill_step < 1:
        raise ValueError("ordinary reference requires a prompt and positive chunk size")
    cache = model.make_cache()
    prefix = prompt[:-1]
    for start in range(0, len(prefix), prefill_step):
        mx.eval(model(mx.array([prefix[start : start + prefill_step]]), cache=cache))
    history = list(prompt)
    anchor = prompt[-1]
    for token in emitted:
        logits = model(mx.array([[anchor]]), cache=cache)[0, -1]
        mx.eval(logits)
        yield (
            history.copy(),
            logits.astype(mx.float32),
            [int(layer.offset) for layer in cache],
        )
        history.append(int(token))
        anchor = int(token)


def audit_reached_laws(
    observations, prompt, emitted, references, transform, *, temperature
):
    """Audit only callbacks on committed emitted prefixes, including bonuses.

    Chain fallback may process unused future rows. Matching the complete history
    against the emitted walk excludes those rows; duplicate attempt histories
    are reported and the latest callback supplies the committed-prefix check.
    """
    import numpy as np

    expected = [(*prompt, *emitted[:index]) for index in range(len(emitted))]
    expected_set = set(expected)
    candidates = {}
    ignored = 0
    for observation in observations:
        history = tuple(observation["history_tokens"])
        if not observation["reachable"] or history not in expected_set:
            ignored += 1
        else:
            candidates.setdefault(history, []).append(observation)
    diagnostics, missing = [], []
    references = iter(references)
    for position, history in enumerate(expected):
        reference_history, raw, offsets = next(references)
        if tuple(reference_history) != history:
            raise ValueError("ordinary reference history does not match emitted prefix")
        matches = candidates.get(history, ())
        if not matches:
            missing.append(position)
            continue
        observation = matches[-1]
        reference_raw = np.asarray(raw, dtype=np.float32)
        actual_raw = np.asarray(observation["raw_logits"], dtype=np.float32)
        reference_law = np.asarray(transform(raw), dtype=np.float64)
        actual_law = np.asarray(observation["law"], dtype=np.float64)
        if (
            actual_raw.shape != reference_raw.shape
            or actual_law.shape != reference_law.shape
        ):
            raise ValueError("target/reference vocabulary geometry differs")
        raw_delta = actual_raw.astype(np.float64) - reference_raw
        law_delta = actual_law - reference_law
        diagnostics.append(
            {
                "emitted_position": position,
                "emitted_token": int(emitted[position]),
                "history_tokens": list(history),
                "ordinary_cache_offsets": offsets,
                "callback_attempts_at_prefix": len(matches),
                "processed_law_l1_vs_ordinary": float(np.abs(law_delta).sum()),
                "processed_law_max_abs_vs_ordinary": float(np.abs(law_delta).max()),
                "raw_logits_max_abs_vs_ordinary": float(np.abs(raw_delta).max()),
                "raw_logits_rmse_vs_ordinary": float(np.sqrt(np.mean(raw_delta**2))),
                "raw_logits_exact_value_equal": bool(
                    np.array_equal(actual_raw, reference_raw)
                ),
                "target_raw_argmax": int(np.argmax(actual_raw)),
                "ordinary_raw_argmax": int(np.argmax(reference_raw)),
                "finite": bool(
                    np.isfinite(raw_delta).all() and np.isfinite(law_delta).all()
                ),
            }
        )
    if next(references, None) is not None:
        raise ValueError("ordinary reference has extra unconsumed prefixes")
    first = (
        diagnostics[0]
        if diagnostics and diagnostics[0]["emitted_position"] == 0
        else None
    )
    return {
        "first_processed_law_l1_vs_ordinary": None
        if first is None
        else first["processed_law_l1_vs_ordinary"],
        "history_tokens": list(prompt),
        "temperature": temperature,
        "law_authority": "original prompt-prefix chunks plus S1 anchor/each emitted token versus canonical ranked target branch before RNG draw",
        "empirical_distribution_qualification": False,
        "candidate_callback_count": len(observations),
        "reached_prefix_count": len(expected),
        "compared_reached_prefix_count": len(diagnostics),
        "unreached_callback_count": ignored,
        "duplicate_reached_callback_count": sum(
            max(0, len(values) - 1) for values in candidates.values()
        ),
        "missing_reached_prefixes": missing,
        "maximum_processed_law_l1_vs_ordinary": max(
            (value["processed_law_l1_vs_ordinary"] for value in diagnostics),
            default=None,
        ),
        "maximum_processed_law_max_abs_vs_ordinary": max(
            (value["processed_law_max_abs_vs_ordinary"] for value in diagnostics),
            default=None,
        ),
        "processed_law_tolerances": {"l1": 0.01, "max_absolute": 1e-4},
        "raw_equality_authority": "exact numeric values after independent float32 host conversion; not an artifact-byte or bitwise assertion",
        "every_reached_raw_logit_values_equal": bool(expected)
        and not missing
        and all(value["raw_logits_exact_value_equal"] for value in diagnostics),
        "every_reached_law_matches_ordinary": bool(expected)
        and not missing
        and all(
            value["finite"]
            and value["processed_law_l1_vs_ordinary"] <= 0.01
            and value["processed_law_max_abs_vs_ordinary"] <= 1e-4
            for value in diagnostics
        ),
        "prefix_diagnostics": diagnostics,
    }


def check_continuation_generation(
    adapter,
    args,
    prompts,
    policy,
    adaptive,
    mixed_sampling,
    mx,
    np,
    frames,
    ordinary,
):
    from mlx2.runtime.sample_utils import LaneRNG

    with managed_batch(adapter, args, policy=policy) as batch:
        temperatures = [0.0, 0.8, 0.0, 0.8] if mixed_sampling else [0.0] * 4
        ids = batch.insert(
            prompts,
            max_tokens=[args.max_tokens] * 4,
            lane_rngs=[LaneRNG(320 + index) for index in range(4)],
            sampling_configs=[{"sampling_temp": temp} for temp in temperatures],
        )
        output, ends = {uid: [] for uid in ids}, {}
        law_checks = {}
        law_observations = {}
        if mixed_sampling:
            from mlx2.runtime.sample_utils import make_transformed_logprobs
            from mlx2.runtime.speculative_sampling import probability

            transform = make_transformed_logprobs(0.8)
            for uid, prompt, temp in zip(ids, prompts, temperatures, strict=True):
                if temp:
                    law_observations[uid] = []
            original_law = batch._target_law

            def checked_law(
                lane,
                logits,
                history,
                *positional,
                _law=original_law,
                _observations=law_observations,
                **keywords,
            ):
                law = _law(lane, logits, history, *positional, **keywords)
                if lane.uid in _observations:
                    capture_law_observation(
                        _observations,
                        lane.uid,
                        logits,
                        law,
                        history,
                        bool(
                            keywords.get(
                                "reachable", positional[0] if positional else True
                            )
                        ),
                        mx,
                        np,
                        4 * args.max_tokens,
                    )
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
        if mixed_sampling:

            def transformed_reference(raw, _transform=transform):
                return probability(np.asarray(mx.exp(_transform(raw[None])[0])))

            for uid, prompt, temp in zip(ids, prompts, temperatures, strict=True):
                if temp:
                    law_checks[uid] = audit_reached_laws(
                        law_observations[uid],
                        prompt,
                        output[uid],
                        ordinary_reached_rows(
                            adapter.model, mx, prompt, output[uid], args.prefill_step
                        ),
                        transformed_reference,
                        temperature=temp,
                    )
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
                not value["every_reached_law_matches_ordinary"]
                for value in law_checks.values()
            )
        ):
            failures.append(
                "a reached sampled canonical target branch law differs from ordinary prefix-plus-S1 reference or was not exercised"
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
        return check


def run(args, report):
    import mlx.core as mx
    import numpy as np

    from mlx2.adapters.standard_decoder import StandardDecoderAdapter

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
    import psutil

    adapter = construct_admitted_adapter(
        args,
        report,
        device,
        StandardDecoderAdapter,
        int(psutil.virtual_memory().available),
    )
    try:
        if args.compute_precision == "float32-diagnostic":
            from validate_xpress_metal_matrix import cast_float32_diagnostic

            cast_float32_diagnostic(adapter.model, adapter.draft_model)
            if args.target_verify_row_exact:
                adapter.model.configure_target_verify_row_exact(True)
        model, draft = adapter.model, adapter.draft_model
        report["critic_session_revision"] = bind_compute_precision(
            draft, args.compute_precision
        )
        report["artifact_identity"] = adapter.identity
        report["draft_settings"] = draft.receipt_settings
        prompts = [
            exact_prompt_tokens(adapter.tokenizer, text, args.context_tokens)
            for text in PROMPTS
        ]
        report["prompt_lengths"] = [len(prompt) for prompt in prompts]
        report["prompt_ids_sha256"] = [
            hashlib.sha256(
                json.dumps(prompt, separators=(",", ":")).encode()
            ).hexdigest()
            for prompt in prompts
        ]

        def fresh(ids):
            cache = model.make_cache()
            for start in range(0, len(ids) - 1, args.prefill_step):
                mx.eval(
                    model(
                        mx.array(
                            [ids[start : min(start + args.prefill_step, len(ids) - 1)]]
                        ),
                        cache=cache,
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
                if token in stops:
                    break
                if index + 1 < args.max_tokens:
                    logits = model(mx.array([[token]]), cache=cache)[0, -1]
            return result

        costs, _observations, proposal_cost, stops, probe_offsets = (
            measure_context_costs(adapter, args, prompts[0], mx, np, report)
        )
        report["proposal_cost"] = proposal_cost
        report["cost_binding"] = {
            "source_sha256": report["source_sha256"],
            "artifact_identity": adapter.identity,
            "compute_precision": args.compute_precision,
            "context_tokens": args.context_tokens,
            "actual_context_tokens": len(prompts[0]),
            "prefill_step": args.prefill_step,
            "target_probe_offsets": probe_offsets,
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
            for adaptive, mixed_sampling in (
                (False, False),
                (True, False),
                (True, True),
            ):
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
                check = check_continuation_generation(
                    adapter,
                    args,
                    prompts,
                    policy,
                    adaptive,
                    mixed_sampling,
                    mx,
                    np,
                    frames,
                    ordinary,
                )
                report["checks"].append(check)
        finally:
            model.forward_with_taps = original
        report["source_unchanged"] = sources() == report["source_sha256"]
        report["generation_completed"] = True
        report["passed"] = (
            all(check["passed"] for check in report["checks"])
            and report["source_unchanged"]
        )
    finally:
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
