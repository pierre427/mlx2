#!/usr/bin/env python3
"""Bounded exact-token and saved-state probe for DLoop first divergences.

This is a correctness diagnostic, not a benchmark. It compares ordinary decode
with fixed self-MTP depths 1 and 8 plus the unchanged depth-8 DLoop policy on a
small prompt subset, records top-two response log-probs, then checks saved and
cold continuations at the first divergent token boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dloop_ab import (
    _cold_history,
    _metadata_artifact_fingerprint,
    _state_evidence,
    _terminal_state,
    _validate_identity_args,
    _verify_frozen_git,
)
from mtp_confidence_gpu import PROMPTS, _install_lane

ARMS = {
    "ordinary": {"reference": "ordinary_decode"},
    "fixed1": {"num_draft": 1},
    "fixed8": {"num_draft": 8},
    "loop8": {
        "num_draft": 8,
        "draft_loop": {
            "boundaries": list(range(1, 9)),
            "threshold": -1e9,
            "cohort": "any",
        },
    },
}


def _top_two_logprobs(logprobs, mx):
    """Return top-two token IDs/log-probs; preserve their exact ordering/margin."""
    if len(logprobs.shape) != 1 or int(logprobs.shape[0]) < 2:
        raise RuntimeError("response logprobs must be a one-dimensional vocabulary row")
    indices = mx.argpartition(logprobs, kth=-2)[-2:]
    indices = indices[mx.argsort(logprobs[indices])[::-1]]
    scores = logprobs[indices]
    top_ids, top_scores = indices.tolist(), scores.tolist()
    if len(top_ids) != 2 or len(top_scores) != 2:
        raise RuntimeError("top-two logprob selection returned malformed values")
    top_scores = [float(score) for score in top_scores]
    if not all(math.isfinite(score) for score in top_scores):
        raise RuntimeError("top-two response logprobs must be finite")
    return {
        "entries": [
            {"token_id": int(top_ids[index]), "logprob": top_scores[index]}
            for index in range(2)
        ],
        "margin_nats": top_scores[0] - top_scores[1],
    }


def _run_arm(
    model, prompt_tokens, prompt_index, config, max_tokens, capture_state=False
):
    import mlx.core as mx
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG

    if config is not None and config.get("reference") == "ordinary_decode":
        config = None
    generator = BatchGenerator(
        model,
        completion_batch_size=1,
        prefill_batch_size=1,
        prefill_step_size=2048,
        self_mtp=(
            {
                "persistent": True,
                "segment_aware_live_tip": True,
                "segment_aware_cohort_size": 1,
                **config,
            }
            if config is not None
            else None
        ),
    )
    uid = generator.insert(
        [prompt_tokens],
        max_tokens=[max_tokens],
        lane_rngs=[LaneRNG(prompt_index)],
        self_mtp_configs=[{"sampling_temp": 0.0}] if config is not None else None,
    )[0]
    tokens, trace, receipt, terminal_state = [], [], None, None
    done = False
    try:
        while not done:
            _, responses = generator.next()
            for response in responses:
                if response.uid != uid:
                    raise RuntimeError("probe received an unexpected request UID")
                tokens.append(int(response.token))
                trace.append(_top_two_logprobs(response.logprobs, mx))
                if getattr(response, "mtp_receipt", None):
                    receipt = response.mtp_receipt
                if response.finish_reason is not None:
                    if capture_state:
                        terminal_state = _terminal_state(response)
                    done = True
        mx.synchronize()
    finally:
        generator.close()
    stats = (receipt or {}).get("stats", {})
    return {
        "tokens": tokens,
        "top2_logprobs": trace,
        "route": (receipt or {}).get("route"),
        "verify_span_hist": stats.get("verify_span_hist"),
        "draft_loop": (receipt or {}).get("draft_loop"),
        "terminal_state": terminal_state,
    }


def _first_difference(reference, candidate):
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return index
    if len(reference) != len(candidate):
        return min(len(reference), len(candidate))
    return None


def _trace_window(trace, index, prompt_tokens, generated_tokens):
    if index is None:
        return {}
    window = {}
    for step in range(max(index - 1, 0), min(index + 2, len(trace))):
        context = [*prompt_tokens, *generated_tokens[:step]]
        window[str(step)] = {
            "context_tokens": len(context),
            "context_sha256": hashlib.sha256(
                json.dumps(context, separators=(",", ":")).encode()
            ).hexdigest(),
            "generated_prefix_tokens": generated_tokens[:step],
            "top2_logprobs": trace[step],
        }
    return window


def _boundary_evidence(
    model, prompt_tokens, prompt_index, arm, config, boundary, n, expected_prefix
):
    import hashlib

    result = _run_arm(
        model, prompt_tokens, prompt_index, config, boundary, capture_state=True
    )
    if result["tokens"] != expected_prefix:
        raise RuntimeError(f"{arm}: boundary rerun differs from scanned common prefix")
    state = result["terminal_state"]
    expected_history = [*prompt_tokens, *result["tokens"][:-1]]
    if (
        state is None
        or [int(token) for token in state["all_tokens"]] != expected_history
    ):
        raise RuntimeError(
            f"{arm}: extracted history does not end before terminal token"
        )
    if result["tokens"][-1] != state["terminal_response_token"]:
        raise RuntimeError(
            f"{arm}: terminal response.token does not match emitted token"
        )
    evidence = _state_evidence(model, state, config, n)
    saved_prompt = [state["terminal_response_token"]]
    cold_tokens = _cold_history(state)
    return {
        "arm": arm,
        "boundary_tokens": boundary,
        "history_tokens": expected_history,
        "history_sha256": hashlib.sha256(
            json.dumps(expected_history, separators=(",", ":")).encode()
        ).hexdigest(),
        "terminal_response_token": state["terminal_response_token"],
        "saved_cache_insert": {
            "prompt_tokens": saved_prompt,
            "all_tokens": expected_history,
            "prompt_token_count": len(saved_prompt),
            "reused_history_token_count": len(expected_history),
        },
        "cold_ordinary_prompt_tokens": cold_tokens,
        "cold_ordinary_full_prefill_token_count": len(cold_tokens),
        "original_full_prefill_token_count": len(prompt_tokens),
        "state_continuation": evidence,
    }


def _validate_args(args, parser):
    if args.scan_tokens < 2:
        parser.error("--scan-tokens must be at least 2")
    if args.continuation_tokens < 1:
        parser.error("--continuation-tokens must be positive")
    if not args.prompt_index or len(args.prompt_index) != len(set(args.prompt_index)):
        parser.error("--prompt-index values must be nonempty and unique")
    if any(index < 0 or index >= len(PROMPTS) for index in args.prompt_index):
        parser.error(f"--prompt-index values must be in 0..{len(PROMPTS) - 1}")
    try:
        _validate_identity_args(
            args.source_revision, args.runtime_source_sha256, args.artifact_identity
        )
        _validate_identity_args(
            args.source_revision, args.runtime_native_sha256, args.artifact_identity
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    return _verify_frozen_git(args.source_revision)


def _scan_limits(prompt_indices, max_scan_tokens, overrides):
    limits = {index: max_scan_tokens for index in prompt_indices}
    if not overrides:
        if 2 in limits:
            limits[2] = min(max_scan_tokens, 8)
        if 7 in limits:
            limits[7] = min(max_scan_tokens, 80)
    seen = set()
    for spec in overrides:
        index_text, sep, count_text = spec.partition("=")
        if not sep:
            raise ValueError(f"bad prompt scan limit {spec!r}; expected INDEX=TOKENS")
        try:
            index, count = int(index_text), int(count_text)
        except ValueError as exc:
            raise ValueError(f"bad prompt scan limit {spec!r}") from exc
        if index not in limits or index in seen:
            raise ValueError(f"prompt scan limit index {index} is missing or repeated")
        if count < 2 or count > max_scan_tokens:
            raise ValueError("prompt scan limits must be in 2..--scan-tokens")
        limits[index] = count
        seen.add(index)
    return limits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--runtime-source-sha256", required=True)
    parser.add_argument("--runtime-native-sha256", required=True)
    parser.add_argument("--artifact-identity", required=True)
    parser.add_argument("--prompt-index", type=int, action="append", default=[])
    parser.add_argument("--scan-tokens", type=int, default=80)
    parser.add_argument("--prompt-scan-limit", action="append", default=[])
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument(
        "--lane-matmul", choices=("off", "auto", "crossover", "exact"), default="off"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.prompt_index:
        args.prompt_index = [2, 7]
    git_revision = _validate_args(args, parser)
    try:
        scan_limits = _scan_limits(
            args.prompt_index, args.scan_tokens, args.prompt_scan_limit
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not args.i_own_the_gpu:
        raise SystemExit("refusing to run on Metal without --i-own-the-gpu")

    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.adapters.registry import resolve_adapter
    from mlx2.serving import runtime_identity

    adapter_cls = resolve_adapter(args.model, mtp=True)
    inspector = getattr(adapter_cls, "artifact_inspector", None)
    if not callable(inspector):
        raise SystemExit("REFUSED: adapter lacks a metadata-only artifact inspector")
    if _metadata_artifact_fingerprint(inspector(args.model)) != args.artifact_identity:
        raise SystemExit("REFUSED: metadata artifact fingerprint mismatch")

    import mlx.core as mx

    actual_runtime = runtime_identity()
    if (
        actual_runtime.get("source_sha256") != args.runtime_source_sha256
        or actual_runtime.get("mlx_native_sha256") != args.runtime_native_sha256
    ):
        raise SystemExit("REFUSED: runtime source/native identity mismatch")
    mx.set_cache_limit(4 << 30)
    adapter = adapter_cls(args.model)
    actual_artifact = adapter.identity.get("fingerprint")
    if actual_artifact != args.artifact_identity:
        raise SystemExit("REFUSED: loaded artifact fingerprint mismatch")
    lane = _install_lane(args, adapter)
    tokenizer = adapter.tokenizer
    prompts = {
        index: list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": PROMPTS[index]}],
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
            )
        )
        for index in args.prompt_index
    }
    result_prompts = []
    for index, input_ids in prompts.items():
        scan = {
            arm: _run_arm(adapter.model, input_ids, index, config, scan_limits[index])
            for arm, config in ARMS.items()
        }
        loop_receipt = scan["loop8"]["draft_loop"]
        if (
            not isinstance(loop_receipt, dict)
            or loop_receipt.get("observed_used") is not True
        ):
            raise RuntimeError(
                f"prompt {index}: loop8 did not report observed DLoop use"
            )
        reference = scan["ordinary"]["tokens"]
        comparisons = {}
        divergence_indices = []
        for arm in (name for name in ARMS if name != "ordinary"):
            generated = scan[arm]["tokens"]
            divergence = _first_difference(reference, generated)
            if divergence is not None:
                divergence_indices.append(divergence)
                if (
                    divergence < min(len(reference), len(generated))
                    and reference[:divergence] != generated[:divergence]
                ):
                    raise RuntimeError(
                        f"prompt {index} {arm}: divergence prefix is inconsistent"
                    )
            comparisons[arm] = {
                "first_divergent_token_index": divergence,
                "common_prefix_token_count": (
                    divergence
                    if divergence is not None
                    else min(len(reference), len(generated))
                ),
                "ordinary_token": reference[divergence]
                if divergence is not None and divergence < len(reference)
                else None,
                "candidate_token": generated[divergence]
                if divergence is not None and divergence < len(generated)
                else None,
                "top2_logprobs_near_divergence": _trace_window(
                    scan[arm]["top2_logprobs"],
                    divergence,
                    input_ids,
                    generated,
                ),
                "ordinary_top2_logprobs_near_divergence": _trace_window(
                    scan["ordinary"]["top2_logprobs"],
                    divergence,
                    input_ids,
                    reference,
                ),
            }
        earliest = min(divergence_indices) if divergence_indices else None
        state_evidence = []
        if earliest is not None and earliest > 0:
            for arm, config in ARMS.items():
                state_evidence.append(
                    _boundary_evidence(
                        adapter.model,
                        input_ids,
                        index,
                        arm,
                        None if arm == "ordinary" else config,
                        earliest,
                        args.continuation_tokens,
                        reference[:earliest],
                    )
                )
        result_prompts.append(
            {
                "prompt_index": index,
                "prompt_sha256": hashlib.sha256(
                    json.dumps(input_ids, separators=(",", ":")).encode()
                ).hexdigest(),
                "scan_tokens": scan_limits[index],
                "arms": {
                    name: {
                        "config": ARMS[name],
                        "full_prefill_token_count": len(input_ids),
                        "tokens": item["tokens"],
                        "route": item["route"],
                        "verify_span_hist": item["verify_span_hist"],
                        "draft_loop": item["draft_loop"],
                    }
                    for name, item in scan.items()
                },
                "comparisons_to_ordinary": comparisons,
                "earliest_divergence_token_index": earliest,
                "state_evidence_at_first_divergence": state_evidence,
            }
        )
        mx.clear_cache()
    args.out.write_text(
        json.dumps(
            {
                "schema": "mlx2.dloop-first-divergence.v1",
                "host": platform.node(),
                "model": args.model,
                "source_revision": git_revision,
                "runtime_identity": actual_runtime,
                "artifact_identity": actual_artifact,
                "producer_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "width": 1,
                "sampling": "greedy_argmax",
                "lane_matmul": lane,
                "arms": ARMS,
                "continuation_tokens": args.continuation_tokens,
                "prompts": result_prompts,
            },
            indent=1,
        )
    )
    print(json.dumps({"output": str(args.out), "prompts": result_prompts}, indent=1))


if __name__ == "__main__":
    main()
