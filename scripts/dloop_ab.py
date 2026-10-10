#!/usr/bin/env python3
"""In-process interleaved A/B: fixed self-MTP depth against the draft-loop gate.

One model load, one process.  Each arm is a ``BatchGenerator`` self-MTP
configuration; arms run in the order A B C ... per pair, over the same
chat-templated prompt set (thinking off, greedy), one request at a time.
Per request we time decode only (first emitted token to completion), so
prefill does not dilute the comparison, and keep the exact token ids so
greedy arms can be compared token for token.

Every gated request must carry a ``draft_loop`` receipt with
``observed_used``; every fixed request must carry none.  A missing
mechanism refuses the run rather than reporting a no-op as a result.

Refuses to touch Metal without ``--i-own-the-gpu``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mtp_confidence_gpu import PROMPTS, _install_lane

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _validate_identity_args(source_revision, runtime_source_sha256, artifact_identity):
    if not isinstance(source_revision, str) or not _HEX40.fullmatch(source_revision):
        raise ValueError(
            "--source-revision must be a 40-character lowercase Git commit"
        )
    for name, value in (
        ("--runtime-source-sha256", runtime_source_sha256),
        ("--artifact-identity", artifact_identity),
    ):
        if not isinstance(value, str) or not _HEX64.fullmatch(value):
            raise ValueError(f"{name} must be a 64-character lowercase SHA-256")


def _metadata_artifact_fingerprint(inspected):
    if not isinstance(inspected, dict):
        return None
    identity = inspected.get("identity")
    return identity.get("fingerprint") if isinstance(identity, dict) else None


def _validate_arms(arms):
    if not arms or len(arms) != len({name for name, _ in arms}):
        raise ValueError("arms must have unique names")
    for name, config in arms:
        if not name or not isinstance(config, dict):
            raise ValueError("each arm needs a name and object config")
        depth = config.get("num_draft")
        if not isinstance(depth, int) or isinstance(depth, bool) or not 1 <= depth <= 8:
            raise ValueError(f"arm {name!r} num_draft must be in 1..8")
        loop = config.get("draft_loop")
        if loop is not None:
            if not isinstance(loop, dict):
                raise ValueError(f"arm {name!r} draft_loop must be an object")
            boundaries = loop.get("boundaries")
            if boundaries is not None and (
                not isinstance(boundaries, list)
                or not boundaries
                or any(
                    not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= depth
                    for v in boundaries
                )
                or boundaries != sorted(set(boundaries))
            ):
                raise ValueError(f"arm {name!r} has invalid draft-loop boundaries")


def _validate_numeric_args(pairs, max_tokens, state_oracle_tokens, width):
    if not isinstance(pairs, int) or isinstance(pairs, bool) or pairs < 1:
        raise ValueError("--pairs must be a positive integer")
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens < 2
    ):
        raise ValueError("--max-tokens must be at least 2")
    if (
        not isinstance(state_oracle_tokens, int)
        or isinstance(state_oracle_tokens, bool)
        or state_oracle_tokens < 0
    ):
        raise ValueError("--state-oracle-tokens must be a nonnegative integer")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width not in (1, 2, 4, 8)
    ):
        raise ValueError("--width must be one of 1, 2, 4, or 8")


def _decode_token_total(rows, context):
    """Require every measured request to contribute at least one decode token."""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError(f"{context}: no request rows to summarize")
    counts = []
    for index, row in enumerate(rows):
        count = row.get("decode_tokens") if isinstance(row, dict) else None
        if type(count) is not int or count <= 0:
            raise ValueError(
                f"{context} row {index}: insufficient decode-token count; "
                "each request must emit at least one decode token"
            )
        counts.append(count)
    return sum(counts)


def _verify_frozen_git(expected_revision):
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "status",
                "--porcelain",
                "--untracked-files=normal",
                "--",
                "src",
                "scripts",
                "qualification/runs/qualify-1010-correctness",
            ],
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"unable to verify frozen Git source: {exc}") from exc
    if actual != expected_revision:
        raise ValueError(
            f"Git HEAD mismatch: expected {expected_revision}, found {actual}"
        )
    if dirty:
        raise ValueError(
            "qualification source scope is dirty; freeze/commit it before running"
        )
    return actual


def _producer_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def parse_arm(text):
    """``name=num_draft``, ``name=num_draft:stage:threshold`` or
    ``name=num_draft:b1/b2/...:threshold`` (explicit stage boundaries)."""
    name, _, spec = text.partition("=")
    if spec.startswith("{"):
        # Full self-MTP config as JSON, e.g. {"num_draft": 3, "draft_loop": {...}}.
        return name, json.loads(spec)
    parts = spec.split(":")
    if not name or len(parts) not in (1, 3):
        raise argparse.ArgumentTypeError(f"bad arm {text!r}")
    config = {"num_draft": int(parts[0])}
    if len(parts) == 3:
        if "/" in parts[1]:
            stages = {"boundaries": [int(v) for v in parts[1].split("/")]}
        else:
            stages = {"stage": int(parts[1])}
        config["draft_loop"] = {**stages, "threshold": float(parts[2])}
    return name, config


def _state_offsets(caches):
    """Per-layer state type and public offset when the cache exposes one."""
    offsets = []
    for index, cache in enumerate(caches):
        offset = getattr(cache, "offset", None)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            offset = None
        offsets.append({"index": index, "type": type(cache).__name__, "offset": offset})
    return offsets


def _state_continuation(model, state, config, max_tokens):
    """Resume an emitted request from its extracted state and return token IDs."""
    import copy

    import mlx.core as mx

    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG

    history = [int(token) for token in state["all_tokens"]]
    if len(history) < 2:
        raise RuntimeError("state continuation needs a prompt prefix and final token")
    mtp = config is not None
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
            if mtp
            else None
        ),
    )
    kwargs = {
        "max_tokens": [max_tokens],
        "caches": [copy.deepcopy(state["target_cache"])],
        "all_tokens": [history[:-1]],
        "lane_rngs": [LaneRNG(0)],
    }
    if mtp:
        kwargs["mtp_states"] = [copy.deepcopy(state["mtp_state"])]
        kwargs["self_mtp_configs"] = [{"sampling_temp": 0.0}]
    uid = generator.insert([[history[-1]]], **kwargs)[0]
    output, done = [], False
    while not done:
        _, responses = generator.next()
        for response in responses:
            if response.uid != uid:
                raise RuntimeError("unexpected UID in state continuation")
            output.append(int(response.token))
            if response.finish_reason is not None:
                done = True
    mx.synchronize()
    generator.close()
    return output


def _cold_continuation(model, history, max_tokens):
    """Recompute the full prefix with ordinary decode, then emit a continuation."""
    import mlx.core as mx

    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG

    generator = BatchGenerator(
        model,
        completion_batch_size=1,
        prefill_batch_size=1,
        prefill_step_size=2048,
    )
    uid = generator.insert(
        [list(history)], max_tokens=[max_tokens], lane_rngs=[LaneRNG(0)]
    )[0]
    output, done = [], False
    while not done:
        _, responses = generator.next()
        for response in responses:
            if response.uid != uid:
                raise RuntimeError("unexpected UID in cold state continuation")
            output.append(int(response.token))
            if response.finish_reason is not None:
                done = True
    mx.synchronize()
    generator.close()
    return output


def _state_evidence(model, state, config, max_tokens):
    import hashlib
    import json

    history = [int(token) for token in state["all_tokens"]]
    if len(history) < 2 or state["target_cache"] is None or state["mtp_state"] is None:
        raise RuntimeError("extracted self-MTP state is not continuable")
    ordinary = _state_continuation(model, state, None, max_tokens)
    self_mtp = _state_continuation(model, state, config, max_tokens)
    cold = _cold_continuation(model, history, max_tokens)
    return {
        "schema": "mlx2.dloop-state-continuation.v1",
        "prefix_tokens": len(history),
        "prefix_token_ids": history,
        "prefix_sha256": hashlib.sha256(
            json.dumps(history, separators=(",", ":")).encode()
        ).hexdigest(),
        "target_cache_offsets": _state_offsets(state["target_cache"]),
        "mtp_cache_offsets": _state_offsets(state["mtp_state"][0]),
        "saved_ordinary_tokens": ordinary,
        "saved_self_mtp_tokens": self_mtp,
        "cold_ordinary_tokens": cold,
    }


def run_group(model, prompts, indices, config, max_tokens, state_oracle_tokens=0):
    """Decode ``indices`` together in one generator; return rows and engagement."""
    import mlx.core as mx

    from mlx2.runtime import segmented_self_mtp as seg
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG

    width = len(indices)
    before = dict(seg._STATS)
    gen = BatchGenerator(
        model,
        completion_batch_size=width,
        prefill_batch_size=width,
        prefill_step_size=2048,
        self_mtp={
            "persistent": True,
            "segment_aware_live_tip": True,
            "segment_aware_cohort_size": width,
            **config,
        },
    )
    uids = gen.insert(
        [prompts[i] for i in indices],
        max_tokens=[max_tokens] * width,
        lane_rngs=[LaneRNG(i) for i in indices],
        self_mtp_configs=[{"sampling_temp": 0.0}] * width,
    )
    tokens = {uid: [] for uid in uids}
    receipts, terminal_states, first, done = {}, {}, None, set()
    while len(done) < width:
        _, responses = gen.next()
        for response in responses:
            if first is None:
                mx.synchronize()
                first = time.perf_counter()
            tokens[response.uid].append(int(response.token))
            if getattr(response, "mtp_receipt", None):
                receipts[response.uid] = response.mtp_receipt
            if response.finish_reason is not None:
                done.add(response.uid)
                if state_oracle_tokens:
                    terminal_states[response.uid] = {
                        "target_cache": response.prompt_cache,
                        "all_tokens": response.all_tokens,
                        "mtp_state": response.mtp_state,
                    }
    mx.synchronize()
    wall = time.perf_counter() - first
    gen.close()
    engaged = {
        k: seg._STATS[k] - before.get(k, 0)
        for k in seg._STATS
        if k.startswith("true_batched") and seg._STATS[k] != before.get(k, 0)
    }
    rows = []
    for uid, index in zip(uids, indices):
        receipt = receipts.get(uid) or {}
        stats = receipt.get("stats", {})
        rows.append(
            {
                "prompt": index,
                "tokens": tokens[uid],
                "decode_tokens": len(tokens[uid]) - 1,
                # The group's wall time, shared evenly: the sum over a group's
                # rows is its wall time, so totals give aggregate ms/token.
                "decode_seconds": wall / width,
                "draft_loop": receipt.get("draft_loop"),
                "route": receipt.get("route"),
                "verify_span_hist": stats.get("verify_span_hist"),
                "true_batched": engaged,
            }
        )
        if state_oracle_tokens:
            terminal_state = terminal_states.pop(uid)
            rows[-1]["state_continuation"] = _state_evidence(
                model, terminal_state, config, state_oracle_tokens
            )
            del terminal_state
    mx.clear_cache()
    return rows


def run_arm(
    model, tokenizer, prompts, config, max_tokens, width=1, state_oracle_tokens=0
):
    import mlx.core as mx

    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG

    if width > 1:
        rows = []
        for start in range(0, len(prompts), width):
            indices = list(range(start, min(start + width, len(prompts))))
            rows.extend(
                run_group(
                    model, prompts, indices, config, max_tokens, state_oracle_tokens
                )
            )
        return rows
    rows = []
    for index, prompt in enumerate(prompts):
        gen = BatchGenerator(
            model,
            completion_batch_size=1,
            prefill_batch_size=1,
            prefill_step_size=2048,
            self_mtp={
                "persistent": True,
                "segment_aware_live_tip": True,
                "segment_aware_cohort_size": 1,
                **config,
            },
        )
        gen.insert(
            [prompt],
            max_tokens=[max_tokens],
            lane_rngs=[LaneRNG(index)],
            self_mtp_configs=[{"sampling_temp": 0.0}],
        )
        tokens, receipt, terminal_state, first, done = [], None, None, None, False
        while not done:
            _, responses = gen.next()
            for response in responses:
                if first is None:
                    mx.synchronize()
                    first = time.perf_counter()
                tokens.append(int(response.token))
                if getattr(response, "mtp_receipt", None):
                    receipt = response.mtp_receipt
                if response.finish_reason is not None:
                    done = True
                    if state_oracle_tokens:
                        terminal_state = {
                            "target_cache": response.prompt_cache,
                            "all_tokens": response.all_tokens,
                            "mtp_state": response.mtp_state,
                        }
        mx.synchronize()
        seconds = time.perf_counter() - first
        gen.close()
        stats = (receipt or {}).get("stats", {})
        rows.append(
            {
                "prompt": index,
                "tokens": tokens,
                "decode_tokens": len(tokens) - 1,
                "decode_seconds": seconds,
                "draft_loop": (receipt or {}).get("draft_loop"),
                "route": (receipt or {}).get("route"),
                "verify_span_hist": stats.get("verify_span_hist"),
            }
        )
        if state_oracle_tokens:
            rows[-1]["state_continuation"] = _state_evidence(
                model, terminal_state, config, state_oracle_tokens
            )
        mx.clear_cache()
    return rows


def summarize(results, arms, baseline):
    out = {}
    for name in arms:
        runs = [r for r in results if r["arm"] == name]
        if not runs:
            raise ValueError(f"arm {name!r}: no completed pairs to summarize")
        per_pair = []
        for run in runs:
            context = f"arm {name!r} pair {run.get('pair', '?')}"
            decode_tokens = _decode_token_total(run.get("rows"), context)
            per_pair.append(
                sum(row["decode_seconds"] for row in run["rows"]) / decode_tokens
            )
        loops = [x["draft_loop"] for r in runs for x in r["rows"] if x["draft_loop"]]
        out[name] = {
            "ms_per_token_pairs": [1e3 * v for v in per_pair],
            "ms_per_token_median": 1e3 * statistics.median(per_pair),
            "gate_decisions": sum(l["decisions"] for l in loops),
            "gate_extensions": sum(l["extensions"] for l in loops),
        }
    base = out[baseline]["ms_per_token_median"]
    reference = {
        x["prompt"]: x["tokens"]
        for r in results
        if r["arm"] == baseline
        for x in r["rows"]
    }
    for name in arms:
        out[name]["speedup_vs_baseline"] = base / out[name]["ms_per_token_median"]
        mismatched = set()
        first_divergence = []
        for r in results:
            if r["arm"] != name:
                continue
            for x in r["rows"]:
                ref = reference[x["prompt"]]
                if x["tokens"] != ref:
                    mismatched.add(x["prompt"])
                    at = next(
                        (i for i, (a, b) in enumerate(zip(x["tokens"], ref)) if a != b),
                        min(len(x["tokens"]), len(ref)),
                    )
                    first_divergence.append((x["prompt"], at))
        out[name]["prompts_differing_from_baseline"] = sorted(mismatched)
        out[name]["first_divergence"] = sorted(set(first_divergence))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--runtime-source-sha256", required=True)
    parser.add_argument("--runtime-native-sha256", required=True)
    parser.add_argument("--artifact-identity", required=True)
    parser.add_argument(
        "--arm",
        action="append",
        type=parse_arm,
        required=True,
        help="name=num_draft[:stage:threshold]; the first arm is the baseline",
    )
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument(
        "--state-oracle-tokens",
        type=int,
        default=0,
        help="run saved-ordinary, saved-self-MTP, and cold-ordinary state continuations after each arm",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1,
        help="prompts decoded together per generator (cohort width)",
    )
    parser.add_argument(
        "--lane-matmul", choices=("off", "auto", "crossover", "exact"), default="off"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        _validate_numeric_args(
            args.pairs, args.max_tokens, args.state_oracle_tokens, args.width
        )
        _validate_identity_args(
            args.source_revision, args.runtime_source_sha256, args.artifact_identity
        )
        _validate_identity_args(
            args.source_revision, args.runtime_native_sha256, args.artifact_identity
        )
        _validate_arms(args.arm)
        if args.out.exists():
            raise ValueError(f"output already exists: {args.out}")
        git_revision = _verify_frozen_git(args.source_revision)
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
        raise SystemExit(
            "REFUSED: adapter does not expose a metadata-only artifact inspector"
        )
    inspected = inspector(args.model)
    if _metadata_artifact_fingerprint(inspected) != args.artifact_identity:
        raise SystemExit(
            "REFUSED: metadata artifact fingerprint does not match expected identity"
        )

    import mlx.core as mx

    actual_runtime_identity = runtime_identity()
    if (
        actual_runtime_identity.get("source_sha256") != args.runtime_source_sha256
        or actual_runtime_identity.get("mlx_native_sha256")
        != args.runtime_native_sha256
    ):
        raise SystemExit("REFUSED: runtime source/native identity mismatch")
    mx.set_cache_limit(4 << 30)
    adapter = adapter_cls(args.model)
    actual_artifact_identity = adapter.identity.get("fingerprint")
    if actual_artifact_identity != args.artifact_identity:
        raise SystemExit(
            "REFUSED: actual adapter artifact fingerprint does not match expected identity"
        )
    lane = _install_lane(args, adapter)
    tokenizer = adapter.tokenizer
    prompts = [
        list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
            )
        )
        for text in PROMPTS
    ]
    arms = dict(args.arm)
    names = list(arms)
    results = []
    # Warm-up: one short pass per arm, discarded.
    for name in names:
        run_arm(
            adapter.model,
            tokenizer,
            prompts[: args.width],
            arms[name],
            32,
            width=args.width,
        )
    for pair in range(args.pairs):
        for name in names:
            rows = run_arm(
                adapter.model,
                tokenizer,
                prompts,
                arms[name],
                args.max_tokens,
                width=args.width,
                state_oracle_tokens=args.state_oracle_tokens,
            )
            gated = "draft_loop" in arms[name]
            if gated and not all(
                r["draft_loop"] and r["draft_loop"]["observed_used"] for r in rows
            ):
                raise SystemExit(
                    f"REFUSED arm {name}: draft_loop not observed on every request"
                )
            if not gated and any(r["draft_loop"] for r in rows):
                raise SystemExit(f"REFUSED arm {name}: unexpected draft_loop receipt")
            results.append(
                {"pair": pair, "arm": name, "config": arms[name], "rows": rows}
            )
            total = _decode_token_total(rows, f"arm {name!r} pair {pair}")
            secs = sum(r["decode_seconds"] for r in rows)
            print(
                json.dumps(
                    {
                        "pair": pair,
                        "arm": name,
                        "tokens": total,
                        "ms_per_token": round(1e3 * secs / total, 3),
                    }
                ),
                flush=True,
            )
    summary = summarize(results, names, names[0])
    args.out.write_text(
        json.dumps(
            {
                "schema": "mlx2.dloop_ab.v1",
                "host": platform.node(),
                "model": args.model,
                "source_revision": git_revision,
                "runtime_identity": actual_runtime_identity,
                "runtime_source_sha256": args.runtime_source_sha256,
                "runtime_native_sha256": args.runtime_native_sha256,
                "artifact_identity": actual_artifact_identity,
                "producer_sha256": _producer_sha256(),
                "width": args.width,
                "lane_matmul": lane,
                "arms": arms,
                "pairs": args.pairs,
                "max_tokens": args.max_tokens,
                "state_oracle_tokens": args.state_oracle_tokens,
                "summary": summary,
                "results": results,
            },
            indent=1,
        )
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
