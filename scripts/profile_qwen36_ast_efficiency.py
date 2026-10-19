#!/usr/bin/env python3
"""Matched, synchronized ordinary-versus-AST efficiency diagnostic for Qwen3.6.

This is an opt-in diagnostic, not qualification or publishable performance
evidence.  It runs the same greedy prompt and fixed output budget through the
ordinary BatchGenerator and the adapter-owned external AST generator.  Decode
starts only after each arm has completed prefill.  Every measured scheduler
poll is device-synchronized; AST additionally enables synchronized phase
boundaries so queued device work is charged to the phase that submitted it.

Run only under gpuq.sh/CPG and the paired GPU-lock owner receipts.  The harness
refuses an unpinned source, target, draft, policy, dirty ``src/`` tree, or an
existing output path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_OWNERS = (
    Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
    Path("/tmp/gpu.lock/owner.json"),
)
PROMPTS = {
    "short_explain": (
        "Explain to a new engineer why a write-ahead log lets a database recover "
        "after a crash. Cover what is written, in what order, and what happens "
        "during recovery. Keep it to a few short paragraphs."
    ),
    "code_edit": '''Rename the parameter `items` to `records` everywhere in this function, add a
one-line docstring, and return the complete edited function.

```python
def summarize(items, key, *, limit=None, skip_empty=True):
    totals = {}
    for item in items:
        value = item.get(key)
        if value is None and skip_empty:
            continue
        totals[item.get("group", "default")] = value or 0
    return totals
```
''',
}


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def require_clean_pinned_source(expected: str) -> str:
    actual = git_head()
    if actual != expected:
        raise RuntimeError(f"source mismatch: expected {expected}, found {actual}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("refusing a dirty src/ tree")
    return actual


def require_gpu_owners(environ=None, paths=GPU_OWNERS) -> dict:
    environment = os.environ if environ is None else environ
    session, lease = environment.get("GPUQ_SESSION"), environment.get("GPUQ_LEASE")
    if not session or not lease:
        raise RuntimeError("GPUQ_SESSION and GPUQ_LEASE are required")
    owners = []
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"missing paired GPU owner receipt: {path}")
        owner = json.loads(path.read_text())
        if not isinstance(owner, dict) or not owner.get("lease_id"):
            raise RuntimeError(f"invalid GPU owner receipt: {path}")
        owners.append(owner)
    if owners[0] != owners[1]:
        raise RuntimeError("paired GPU owner receipts disagree")
    owner = owners[0]
    if owner.get("session") != session or owner.get("lease_id") != lease:
        raise RuntimeError("paired GPU owner receipt does not match gpuq lease")
    return owner


def numeric_snapshot(stats: dict) -> dict:
    return {
        key: int(value)
        for key, value in stats.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


NON_MONOTONIC_GAUGES = frozenset(
    {"reservation_bytes", "external_adaptive_round_depth", "external_adaptive_verify_width"}
)


def numeric_delta(before: dict, after: dict) -> dict:
    left, right = numeric_snapshot(before), numeric_snapshot(after)
    changed = {
        key: right.get(key, 0) - left.get(key, 0)
        for key in sorted(set(left) | set(right))
        if right.get(key, 0) != left.get(key, 0)
    }
    rollback = {
        key: value
        for key, value in changed.items()
        if value < 0 and key not in NON_MONOTONIC_GAUGES
    }
    if rollback:
        raise RuntimeError(f"scheduler counter rollback: {rollback}")
    return changed


def phase_nanoseconds(delta: dict) -> dict:
    return {
        key.removeprefix("external_phase_").removesuffix("_ns"): int(value)
        for key, value in delta.items()
        if key.startswith("external_phase_")
        and key.endswith("_ns")
        and int(value) != 0
    }


def selected_counters(delta: dict) -> dict:
    names = {
        "external_rounds",
        "accepted_proposals",
        "proposed_tokens",
        "external_progressive_verify_rounds",
        "external_progressive_verify_launches",
        "external_progressive_verify_physical_launches",
        "external_progressive_verify_target_rows",
        "external_progressive_verify_full_tiles",
        "external_progressive_verify_rejections",
        "external_progressive_verify_padding_rows",
        "external_progressive_verify_ragged_launches",
        "external_progressive_target_executor_row_exact_launches",
        "external_progressive_target_executor_nonexact_launches",
        "external_phase_synchronizations",
        "external_ordinary_pipeline_starts",
        "external_ordinary_pipelined_rounds",
        "external_ordinary_pipeline_drains",
        "external_ordinary_pipeline_aborts",
        "external_ordinary_fast_path_rounds",
        "external_ordinary_fast_path_lanes",
        "external_draft_context_skipped",
        "external_taps_skipped",
        "external_transactions_skipped",
        "ordinary_rounds",
        "segmented_transactions",
        "external_cost_handoff_events",
        "external_cost_handoff_lanes",
        "external_cost_ordinary_probe_rounds",
        "external_cost_speculative_rounds",
    }
    return {key: int(delta[key]) for key in sorted(names & delta.keys())}


def summarize(ordinary: dict, ast: dict) -> dict:
    ordinary_tokens = int(ordinary["output_tokens"])
    ast_tokens = int(ast["output_tokens"])
    counters = ast["scheduler_delta"]
    launches = int(
        counters.get("external_progressive_verify_physical_launches")
        or counters.get("external_progressive_verify_launches", 0)
    )
    rows = int(counters.get("external_progressive_verify_target_rows", 0))
    proposed = int(counters.get("proposed_tokens", 0))
    accepted = int(counters.get("accepted_proposals", 0))
    ast_ns, ordinary_ns = int(ast["decode_nanoseconds"]), int(ordinary["decode_nanoseconds"])
    phases = ast.get("phase_nanoseconds", {})
    return {
        "token_exact": ordinary["token_ids"] == ast["token_ids"],
        "ordinary_output_tokens": ordinary_tokens,
        "ast_output_tokens": ast_tokens,
        "ordinary_target_forwards": len(ordinary.get("target_forwards", [])),
        "ast_progressive_target_calls": len(ast.get("progressive_target_calls", [])),
        "ast_physical_target_launches": launches,
        "ast_target_rows": rows,
        "ast_tokens_per_physical_target_launch": ast_tokens / launches if launches else None,
        "ast_target_rows_per_physical_launch": rows / launches if launches else None,
        "ast_acceptance_rate": accepted / proposed if proposed else None,
        "decode_wall_ratio_ast_over_ordinary": ast_ns / ordinary_ns if ordinary_ns else None,
        "decode_wall_ratio_candidate_over_ordinary": (
            ast_ns / ordinary_ns if ordinary_ns else None
        ),
        "decode_seconds": {"ordinary": ordinary_ns / 1e9, "ast": ast_ns / 1e9},
        "ast_synchronized_phase_share": {
            key: value / ast_ns for key, value in sorted(phases.items())
        } if ast_ns else {},
        "diagnostic_only": True,
        "performance_claim": False,
    }


class CallRecorder:
    def __init__(self, mx):
        self.mx = mx
        self.enabled = False
        self.kind = None
        self.calls = []

    def record(self, kind, inputs, call):
        if not self.enabled:
            return call()
        self.mx.synchronize()
        started = time.perf_counter_ns()
        result = call()
        self.mx.synchronize()
        self.calls.append(
            {
                "kind": kind,
                "nanoseconds": time.perf_counter_ns() - started,
                "input_shape": [int(v) for v in getattr(inputs, "shape", ())],
            }
        )
        return result


def install_target_call_recorder(model, recorder: CallRecorder):
    cls = type(model)
    original = cls.__call__
    local = "__call__" in cls.__dict__

    def wrapped(instance, inputs, *args, **kwargs):
        if instance is not model:
            return original(instance, inputs, *args, **kwargs)
        return recorder.record(
            "target_model_call", inputs,
            lambda: original(instance, inputs, *args, **kwargs),
        )

    cls.__call__ = wrapped

    def restore():
        if local:
            cls.__call__ = original
        else:
            delattr(cls, "__call__")

    return restore


def prepare_ordinary(batch, uid):
    for _ in range(10000):
        prompts, responses = batch.next()
        if responses:
            raise RuntimeError("ordinary decode began before the measured boundary")
        if any(item.uid == uid and item.end_of_prompt for item in prompts):
            batch.pop_prompt_boundary(uid)
            return
    raise RuntimeError("ordinary prefill did not finish")


def run_ordinary(
    adapter, prompt, max_tokens, seed, recorder, *, synchronized: bool = True
) -> dict:
    import mlx.core as mx
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG

    batch = BatchGenerator(
        adapter.model, completion_batch_size=1, prefill_batch_size=1,
        prefill_step_size=2048,
    )
    try:
        uid = batch.insert(
            [list(prompt)], max_tokens=[max_tokens], lane_rngs=[LaneRNG(seed)],
            logits_processors=[[]],
        )[0]
        prepare_ordinary(batch, uid)
        mx.synchronize()
        recorder.calls.clear()
        recorder.enabled = synchronized
        steps, tokens = [], []
        started = time.perf_counter_ns()
        while len(tokens) < max_tokens:
            call_at = len(recorder.calls)
            if synchronized:
                mx.synchronize()
            step_start = time.perf_counter_ns()
            _prompts, responses = batch.next()
            if synchronized:
                mx.synchronize()
            emitted = [int(row.token) for row in responses if row.uid == uid]
            tokens.extend(emitted)
            steps.append({
                "poll": len(steps),
                "nanoseconds": time.perf_counter_ns() - step_start,
                "emitted_tokens": emitted,
                "target_forwards": recorder.calls[call_at:],
            })
            if any(row.finish_reason for row in responses if row.uid == uid):
                break
        mx.synchronize()
        elapsed = time.perf_counter_ns() - started
        recorder.enabled = False
        return {
            "arm": "ordinary", "output_tokens": len(tokens), "token_ids": tokens,
            "decode_nanoseconds": elapsed, "scheduler_polls": len(steps),
            "target_forwards": list(recorder.calls), "steps": steps,
        }
    finally:
        recorder.enabled = False
        batch.close()


def prepare_ast(batch, uid):
    lane = batch.lanes[uid]
    for _ in range(10000):
        if lane.anchor is not None:
            return
        batch._prefill(lane)
    raise RuntimeError("AST prefill did not finish")


def run_ast(
    adapter,
    prompt,
    max_tokens,
    seed,
    recorder,
    *,
    synchronized: bool = True,
    force_ordinary: bool = False,
) -> dict:
    import mlx.core as mx
    from mlx2.runtime.sample_utils import LaneRNG
    from mlx2.runtime import external_speculative as es

    batch = adapter.create_external_batch(
        completion_batch_size=1, prefill_step_size=2048, stop_tokens=[],
    )
    original_progressive = es.execute_progressive_target_rows
    progressive_calls = []

    def progressive(model, inputs, caches, layers, **kwargs):
        before = len(recorder.calls)
        result = recorder.record(
            "progressive_target_rows", inputs,
            lambda: original_progressive(model, inputs, caches, layers, **kwargs),
        )
        if recorder.enabled:
            progressive_calls.extend(
                row
                for row in recorder.calls[before:]
                if row["kind"] == "progressive_target_rows"
            )
        return result

    if synchronized:
        es.execute_progressive_target_rows = progressive
    try:
        uid = batch.insert(
            [list(prompt)], max_tokens=[max_tokens],
            lane_rngs=[LaneRNG(seed)],
            sampling_configs=[{"sampling_temp": 0.0, "emit_logprobs": False}],
            logits_processors=[[]],
        )[0]
        prepare_ast(batch, uid)
        if force_ordinary:
            batch.disable_speculation(uid)
        mx.synchronize()
        before = dict(batch.scheduler_stats)
        recorder.calls.clear()
        recorder.enabled = synchronized
        tokens, steps = [], []
        started = time.perf_counter_ns()
        while batch.lanes and len(tokens) < max_tokens:
            stats_before = dict(batch.scheduler_stats)
            call_at = len(progressive_calls)
            if synchronized:
                mx.synchronize()
            step_start = time.perf_counter_ns()
            _prompts, responses = batch.next()
            if synchronized:
                mx.synchronize()
            emitted = [int(row.token) for row in responses if row.uid == uid]
            tokens.extend(emitted)
            step_delta = numeric_delta(stats_before, batch.scheduler_stats)
            steps.append({
                "poll": len(steps),
                "nanoseconds": time.perf_counter_ns() - step_start,
                "emitted_tokens": emitted,
                "scheduler_delta": selected_counters(step_delta),
                "phase_nanoseconds": phase_nanoseconds(step_delta),
                "progressive_target_calls": progressive_calls[call_at:],
                "round_receipts": [
                    row.speculative_receipt for row in responses
                    if row.uid == uid and row.speculative_receipt
                ],
            })
        mx.synchronize()
        elapsed = time.perf_counter_ns() - started
        recorder.enabled = False
        delta = numeric_delta(before, batch.scheduler_stats)
        return {
            "arm": "external_ordinary" if force_ordinary else "ast",
            "output_tokens": len(tokens), "token_ids": tokens,
            "decode_nanoseconds": elapsed, "scheduler_polls": len(steps),
            "scheduler_delta": selected_counters(delta),
            "phase_nanoseconds": phase_nanoseconds(delta),
            "progressive_target_calls": progressive_calls,
            "all_target_calls": list(recorder.calls), "steps": steps,
        }
    finally:
        recorder.enabled = False
        if synchronized:
            es.execute_progressive_target_rows = original_progressive
        batch.close()


def validate_policy(policy: dict, *, target_revision: str, draft_revision: str):
    required = {
        "draft_model", "draft_revision", "target_revision", "num_draft",
        "target_verify_row_exact", "progressive_verification_tile",
    }
    missing = sorted(required - policy.keys())
    if missing:
        raise ValueError(f"policy missing pinned fields: {missing}")
    if policy["target_revision"] != target_revision:
        raise ValueError("policy target_revision mismatch")
    if policy["draft_revision"] != draft_revision:
        raise ValueError("policy draft_revision mismatch")
    if policy["target_verify_row_exact"] is not True:
        raise ValueError("policy must select exact target verification")
    if policy["progressive_verification_tile"] != "auto":
        raise ValueError("policy must select automatic progressive packing")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--policy", required=True, type=Path)
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="short_explain")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--repeat-pairs", type=int, default=1)
    ap.add_argument(
        "--measurement-mode",
        choices=("synchronized_ast", "wall_ast", "wall_external_ordinary"),
        default="synchronized_ast",
        help=(
            "synchronized_ast attributes AST phases; wall_ast measures the "
            "selected AST policy without per-poll synchronization; "
            "wall_external_ordinary forces the post-prefill external lane "
            "onto its permanent ordinary pipeline"
        ),
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--expected-source", required=True)
    ap.add_argument("--expected-target-fingerprint", required=True)
    ap.add_argument("--expected-target-revision", required=True)
    ap.add_argument("--expected-draft-revision", required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    args = ap.parse_args(argv)
    if args.out.exists():
        ap.error(f"refusing to overwrite {args.out}")
    if not 1 <= args.max_tokens <= 512:
        ap.error("--max-tokens must be in 1..512")
    if not 1 <= args.repeat_pairs <= 3:
        ap.error("--repeat-pairs must be in 1..3")
    if args.measurement_mode == "wall_external_ordinary" and args.max_tokens < 3:
        ap.error("wall_external_ordinary requires --max-tokens >= 3")
    policy = json.loads(args.policy.read_text())
    validate_policy(
        policy, target_revision=args.expected_target_revision,
        draft_revision=args.expected_draft_revision,
    )
    plan = {
        "schema": "mlx2.qwen36-ast-efficiency-profile.v1",
        "diagnostic_only": True,
        "performance_claim": False,
        "model": str(args.model.resolve()), "policy": str(args.policy.resolve()),
        "expected_source": args.expected_source,
        "expected_target_fingerprint": args.expected_target_fingerprint,
        "expected_target_revision": args.expected_target_revision,
        "expected_draft_revision": args.expected_draft_revision,
        "prompt": args.prompt, "max_tokens": args.max_tokens, "seed": args.seed,
        "repeat_pairs": args.repeat_pairs,
        "measurement_mode": args.measurement_mode,
        "arms": [
            "ordinary",
            (
                "external_ordinary"
                if args.measurement_mode == "wall_external_ordinary"
                else "ast"
            ),
        ],
        "prefill_excluded": True,
        "fixed_length_no_stop_tokens": True,
        "ast_device_synchronized_phase_timing": (
            args.measurement_mode == "synchronized_ast"
        ),
        "call_recorder": args.measurement_mode == "synchronized_ast",
        "per_poll_device_sync": args.measurement_mode == "synchronized_ast",
        "external_round_timing": args.measurement_mode == "synchronized_ast",
        "forced_after_prefill": args.measurement_mode == "wall_external_ordinary",
        "cost_handoff_policy_exercised": (
            args.measurement_mode != "wall_external_ordinary"
        ),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.i_own_the_gpu:
        ap.error("--execute requires --i-own-the-gpu")
    owner = require_gpu_owners()
    source = require_clean_pinned_source(args.expected_source)
    os.environ["MLX_ENABLE_TF32"] = "0"
    synchronized = args.measurement_mode == "synchronized_ast"
    if synchronized:
        os.environ["MLX2_EXTERNAL_ROUND_TIMING"] = "1"
        os.environ["MLX2_EXTERNAL_ROUND_TIMING_SYNC"] = "1"
    else:
        os.environ.pop("MLX2_EXTERNAL_ROUND_TIMING", None)
        os.environ.pop("MLX2_EXTERNAL_ROUND_TIMING_SYNC", None)
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx
    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter, inspect_artifact

    mx.set_default_device(mx.gpu)
    if not args.model.is_dir():
        raise RuntimeError(f"target artifact is not a directory: {args.model}")
    if not Path(policy["draft_model"]).is_dir():
        raise RuntimeError(
            f"draft artifact is not a directory: {policy['draft_model']}"
        )
    target_checkpoint_identity = inspect_artifact(args.model)["identity"]
    if target_checkpoint_identity.get("fingerprint") != args.expected_target_fingerprint:
        raise RuntimeError(
            "target checkpoint fingerprint does not match the pin: "
            f"{target_checkpoint_identity.get('fingerprint')}"
        )
    adapter = Qwen3635BA3BAdapter(str(args.model), execution_policy=policy)
    if adapter.identity.get("target_revision") != args.expected_target_revision:
        raise RuntimeError("loaded target source binding does not match the pin")
    if adapter.identity.get("draft_revision") != args.expected_draft_revision:
        raise RuntimeError("loaded draft revision does not match the pin")
    prompt = adapter.prompt_tokens(
        {"messages": [{"role": "user", "content": PROMPTS[args.prompt]}]}
    )
    recorder = CallRecorder(mx)
    restore_target = (
        install_target_call_recorder(adapter.model, recorder)
        if synchronized
        else lambda: None
    )
    try:
        pairs = []
        for repeat in range(args.repeat_pairs):
            ordinary = run_ordinary(
                adapter,
                prompt,
                args.max_tokens,
                args.seed,
                recorder,
                synchronized=synchronized,
            )
            mx.clear_cache()
            ast = run_ast(
                adapter,
                prompt,
                args.max_tokens,
                args.seed,
                recorder,
                synchronized=synchronized,
                force_ordinary=(
                    args.measurement_mode == "wall_external_ordinary"
                ),
            )
            pairs.append({
                "repeat": repeat,
                "ordinary": ordinary,
                "ast": ast,
                "summary": summarize(ordinary, ast),
            })
            mx.clear_cache()
    finally:
        restore_target()
    result = {
        **plan, "source": source, "gpu_owner": owner,
        "target_checkpoint_identity": target_checkpoint_identity,
        "route_identity": adapter.identity, "pairs": pairs,
        "ordinary": ordinary, "ast": ast, "summary": pairs[-1]["summary"],
    }
    if not all(pair["summary"]["token_exact"] for pair in pairs):
        raise RuntimeError("AST output is not token-exact with ordinary")
    forced_ordinary = args.measurement_mode == "wall_external_ordinary"
    if forced_ordinary and not all(
        int(pair["ast"]["scheduler_delta"].get(
            "external_ordinary_pipelined_rounds", 0
        )) > 0
        for pair in pairs
    ):
        raise RuntimeError("external ordinary pipeline did not execute")
    if forced_ordinary:
        expected_equal = {
            "external_ordinary_pipelined_rounds": args.max_tokens,
            "external_ordinary_fast_path_rounds": args.max_tokens,
            "external_ordinary_fast_path_lanes": args.max_tokens,
            "external_draft_context_skipped": args.max_tokens,
            "external_taps_skipped": args.max_tokens,
            "external_transactions_skipped": args.max_tokens,
            "ordinary_rounds": args.max_tokens,
        }
        expected_zero = {
            "external_rounds",
            "proposed_tokens",
            "accepted_proposals",
            "segmented_transactions",
            "external_progressive_verify_physical_launches",
        }
        for pair in pairs:
            delta = pair["ast"]["scheduler_delta"]
            if int(delta.get("external_ordinary_pipeline_starts", 0)) < 1:
                raise RuntimeError("external ordinary pipeline never started")
            wrong = {
                key: int(delta.get(key, 0))
                for key, expected in expected_equal.items()
                if int(delta.get(key, 0)) != expected
            }
            nonzero = {
                key: int(delta.get(key, 0))
                for key in expected_zero
                if int(delta.get(key, 0)) != 0
            }
            if wrong or nonzero:
                raise RuntimeError(
                    f"external ordinary route proof failed: wrong={wrong}, "
                    f"unexpected={nonzero}"
                )
            receipts = [
                receipt
                for step in pair["ast"]["steps"]
                for receipt in step["round_receipts"]
            ]
            if len(receipts) != args.max_tokens or any(
                receipt.get("current_execution") != "ordinary_target"
                or receipt.get("ordinary_fallback") is not True
                or receipt.get("cache_sidecar") is not None
                for receipt in receipts
            ):
                raise RuntimeError("external ordinary route receipts are incomplete")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, default=str))
    temporary.replace(args.out)
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
