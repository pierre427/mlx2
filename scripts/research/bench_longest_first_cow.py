#!/usr/bin/env python3
"""Bounded real-model probe for longest-first continuation verification.

This is a research gate, not a serving route.  It compares the existing
parallel complete-path layout with a serial longest-first attempt and with a
request-private COW checkpoint at a shared proposal prefix.  No checkpoint is
published to APCv2 and ordinary decoding remains the reference path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path.home() / "mlx-models" / "Qwen3.8-27B-MLX-4bit"
CAPTURE_LAYERS = (5, 19, 33, 47, 61)
LENGTHS = (7, 7, 6, 5, 5, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_tree_sha256() -> str:
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    digest = hashlib.sha256()
    for raw in sorted(filter(None, files)):
        path = ROOT / os.fsdecode(raw)
        if path.is_file():
            digest.update(raw + b"\0" + bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--context-tokens", type=int, default=128)
    result.add_argument("--repetitions", type=int, default=3)
    result.add_argument("--source-commit", required=True)
    result.add_argument("--i-own-the-gpu", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def plan(args) -> dict:
    if not 32 <= args.context_tokens <= 1024:
        raise ValueError("context-tokens must be in [32,1024]")
    if not 1 <= args.repetitions <= 5:
        raise ValueError("repetitions must be in [1,5]")
    return {
        "schema": "mlx2.longest-first-cow-probe.v1",
        "status": "planned",
        "gpu_executed": False,
        "implemented": False,
        "research_harness_implemented": True,
        "qualified": False,
        "selected": False,
        "observed_used": False,
        "research_probe_observed_used": False,
        "performance_claim": False,
        "model": str(args.model.resolve()),
        "context_tokens": args.context_tokens,
        "repetitions": args.repetitions,
        "candidate_lengths": list(LENGTHS),
        "shared_prefix_tokens": 3,
        "capture_layers": list(CAPTURE_LAYERS),
        "arms": {
            "ordinary_s1": {"forwards": 8, "dense_rows": 8},
            "parallel_15": {"forwards": 1, "dense_rows": 15 * 8},
            "longest_first_accept": {"forwards": 1, "dense_rows": 8},
            "parallel_pair_fallback": {"forwards": 1, "dense_rows": 2 * 8},
            "naive_replay_fallback": {"forwards": 2, "dense_rows": 2 * 8},
            "cow_suffix_fallback": {"forwards": 2, "dense_rows": 8 + 4},
        },
        "boundaries": {
            "request_private_only": True,
            "apcv2_publication": False,
            "serving_integration": False,
            "ordinary_reference_preserved": True,
        },
    }


def exact_prompt_tokens(tokenizer, length: int) -> list[int]:
    text = "Explain exact cache rollback and speculative decoding clearly. "
    for repeats in (16, 32, 64, 128, 256, 512):
        tokens = list(tokenizer.encode(text * repeats, add_special_tokens=False))
        if len(tokens) >= length:
            return tokens[:length]
    raise ValueError("bounded prompt corpus is too short")


def mutate(token: int, vocab: int, salt: int = 1) -> int:
    result = (token + salt) % vocab
    return (result + 1) % vocab if result == token else result


def candidates(good: tuple[int, ...], vocab: int) -> tuple[tuple[int, ...], ...]:
    rows = [good]
    for index, length in enumerate(LENGTHS[1:], 1):
        keep = min(3, max(0, length - 1))
        row = list(good[:length])
        row[keep] = mutate(row[keep], vocab, index)
        for position in range(keep + 1, length):
            row[position] = mutate(row[position], vocab, index + position)
        rows.append(tuple(row))
    if len(rows) != 15 or len(set(rows)) != 15:
        raise RuntimeError("controlled proposal set is not fifteen distinct paths")
    return tuple(rows)


def clone_cache(cache):
    from mlx2.runtime.cow_cache import (
        restore_recovery_descriptors,
        snapshot_recovery_descriptors,
    )

    return restore_recovery_descriptors(*snapshot_recovery_descriptors(cache))[0]


def greedy_tokens(model, mx, cache, anchor: int, count: int) -> tuple[int, ...]:
    result = []
    token = anchor
    for _ in range(count):
        logits = model(mx.array([[token]]), cache=cache)[0, -1]
        mx.eval(logits)
        token = int(mx.argmax(logits).item())
        result.append(token)
    return tuple(result)


def argmax_rows(mx, logits, row: int, count: int) -> tuple[int, ...]:
    return tuple(int(value) for value in mx.argmax(logits[row, :count], axis=-1).tolist())


def execute(args, report):
    if not args.i_own_the_gpu:
        raise ValueError("Metal execution requires --i-own-the-gpu under both locks")
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/research")]
    from varlen_pack_price_bench import _gpuq_owner

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if head != args.source_commit or dirty:
        raise RuntimeError("probe requires the exact clean source commit")
    report.update(
        source_commit=head,
        source_tree_sha256=source_tree_sha256(),
        script_sha256=sha256(Path(__file__)),
        gpuq_owner=_gpuq_owner(),
        status="running",
        gpu_executed=True,
    )

    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter, configure_environment

    configure_environment()
    import mlx.core as mx

    from mlx2.runtime.continuation_verification import prepare_continuations
    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows

    adapter = None
    try:
        began = time.perf_counter()
        adapter = Qwen3827BAdapter(str(args.model), require_mtp=False)
        report["model_load_seconds"] = time.perf_counter() - began
        report["artifact_identity"] = adapter.identity
        model = getattr(adapter.model, "language_model", adapter.model)
        prompt = exact_prompt_tokens(adapter.tokenizer, args.context_tokens)
        base = model.make_cache()
        for start in range(0, len(prompt) - 1, 128):
            end = min(start + 128, len(prompt) - 1)
            mx.eval(model(mx.array([prompt[start:end]]), cache=base))
        anchor = prompt[-1]
        target = greedy_tokens(model, mx, clone_cache(base), anchor, 8)
        good = target[:7]
        vocab = int(model.args.vocab_size)
        paths = candidates(good, vocab)
        bad = paths[1]
        if bad[:3] != good[:3] or bad[3] == good[3]:
            raise RuntimeError("fallback pair does not share exactly the intended prefix")
        report.update(
            prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
            target_tokens=list(target),
            proposal_sha256=hashlib.sha256(json.dumps(paths).encode()).hexdigest(),
        )

        def prepare(cache, selected_paths):
            started = time.perf_counter_ns()
            _paths, logits, _features, tx = prepare_continuations(
                model,
                mx,
                cache,
                anchor,
                selected_paths,
                CAPTURE_LAYERS,
                HybridVerifyRows,
                max_sequences=15,
                max_depth=7,
            )
            return (time.perf_counter_ns() - started) / 1e9, logits, tx

        def ordinary_arm():
            cache = clone_cache(base)
            started = time.perf_counter_ns()
            observed = greedy_tokens(model, mx, cache, anchor, 8)
            return (time.perf_counter_ns() - started) / 1e9, observed

        def parallel_15_arm():
            seconds, logits, tx = prepare(base, paths)
            try:
                observed = argmax_rows(mx, logits, 0, 8)
            finally:
                tx.abort()
            return seconds, observed

        def longest_first_arm():
            seconds, logits, tx = prepare(base, (good,))
            try:
                observed = argmax_rows(mx, logits, 0, 8)
            finally:
                tx.abort()
            return seconds, observed

        def parallel_pair_arm():
            seconds, logits, tx = prepare(base, (bad, good))
            try:
                observed = argmax_rows(mx, logits, 1, 8)
            finally:
                tx.abort()
            return seconds, observed

        def naive_replay_arm():
            first, _logits, tx = prepare(base, (bad,))
            tx.abort()
            second, logits, tx = prepare(base, (good,))
            try:
                observed = argmax_rows(mx, logits, 0, 8)
            finally:
                tx.abort()
            return first + second, observed

        def cow_suffix_arm():
            first, logits, tx = prepare(base, (bad,))
            prefix = argmax_rows(mx, logits, 0, 4)
            checkpoint = tx.commit([4])[0]
            if prefix != target[:4]:
                raise RuntimeError("first proposal did not reach the shared-prefix correction")
            started = time.perf_counter_ns()
            _paths, suffix_logits, _features, suffix_tx = prepare_continuations(
                model,
                mx,
                checkpoint,
                good[3],
                (good[4:],),
                CAPTURE_LAYERS,
                HybridVerifyRows,
                max_sequences=15,
                max_depth=7,
            )
            second = (time.perf_counter_ns() - started) / 1e9
            try:
                observed = prefix + argmax_rows(mx, suffix_logits, 0, 4)
            finally:
                suffix_tx.abort()
            return first + second, observed

        arms = {
            "ordinary_s1": ordinary_arm,
            "parallel_15": parallel_15_arm,
            "longest_first_accept": longest_first_arm,
            "parallel_pair_fallback": parallel_pair_arm,
            "naive_replay_fallback": naive_replay_arm,
            "cow_suffix_fallback": cow_suffix_arm,
        }
        observations = {name: [] for name in arms}
        orders = []
        names = tuple(arms)
        for repetition in range(args.repetitions + 1):
            shift = repetition % len(names)
            order = names[shift:] + names[:shift]
            if repetition:
                orders.append(list(order))
            for name in order:
                seconds, observed = arms[name]()
                if observed != target:
                    raise RuntimeError(f"{name} target-token parity failed")
                if repetition:
                    observations[name].append(seconds)
        medians = {name: statistics.median(values) for name, values in observations.items()}
        pair_break_even = (
            (medians["cow_suffix_fallback"] - medians["parallel_pair_fallback"])
            / (medians["cow_suffix_fallback"] - medians["longest_first_accept"])
        )
        report.update(
            status="passed_probe",
            research_probe_observed_used=True,
            implementation_scope="research_harness_only",
            measured_orders=orders,
            timings_seconds={
                name: {"samples": values, "median": medians[name]}
                for name, values in observations.items()
            },
            ratios={
                "parallel15_over_longest_accept": medians["parallel_15"] / medians["longest_first_accept"],
                "naive_replay_over_cow_suffix": medians["naive_replay_fallback"] / medians["cow_suffix_fallback"],
                "parallel_pair_over_cow_suffix": medians["parallel_pair_fallback"] / medians["cow_suffix_fallback"],
                "ordinary_s1_over_longest_accept": medians["ordinary_s1"] / medians["longest_first_accept"],
            },
            policy_analysis={
                "two_candidate_full_acceptance_break_even_probability": pair_break_even,
                "formula": "p*longest_first_accept + (1-p)*cow_suffix_fallback < parallel_pair_fallback",
                "scope": "controlled two-candidate medians only; not a serving threshold",
            },
            correctness={
                "all_arms_exact_greedy_tokens": True,
                "shared_prefix_tokens_reused": 3,
                "cow_first_forward_input_rows": 8,
                "cow_second_forward_input_rows": 4,
                "full_second_proposal_recomputed": False,
                "apcv2_state_published": False,
            },
        )
    finally:
        if adapter is not None:
            adapter.close()


def main(argv=None) -> int:
    arguments = parser()
    args = arguments.parse_args(argv)
    report = plan(args)
    code = 0
    if not args.dry_run:
        try:
            execute(args, report)
        except BaseException as error:  # noqa: BLE001 - always preserve a receipt
            report.update(
                status="failed",
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
            code = 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "out": str(args.out), "error": report.get("error")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
