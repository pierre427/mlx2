#!/usr/bin/env python3
# ruff: noqa: EXE001
"""Bounded real Qwen3/XPress Metal smoke; no performance claim.

The operator owns GPU locks/lease. --i-own-the-gpu acknowledges that external
ownership; this script does not acquire or infer ownership. --dry-run imports
no tensor libraries. One shared target is loaded, then ordinary greedy token
IDs, B1/B2 draft activity, exact point-mass rejection laws, and paired APCv2
resume are checked. Timings are diagnostic only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "mlx2.parallel-draft-metal-smoke.v1"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-draft", type=int, default=15)
    parser.add_argument("--xpress-num-passes", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def preflight(args):
    if not 2 <= args.max_tokens <= 16:
        raise ValueError("max-tokens must be between 2 and 16")
    if not 16 <= args.prompt_tokens <= 128:
        raise ValueError("prompt-tokens must be between 16 and 128")
    if not 1 <= args.num_draft <= 15 or not 1 <= args.xpress_num_passes <= 16:
        raise ValueError("draft depth/passes exceed bounded smoke limits")
    if not 1 <= args.timeout_seconds <= 300:
        raise ValueError("timeout-seconds must be between 1 and 300")
    if not args.dry_run and not args.i_own_the_gpu:
        raise ValueError(
            "Metal execution requires --i-own-the-gpu under operator GPU ownership"
        )
    return {
        "schema": SCHEMA,
        "model": str(args.model.expanduser().resolve()),
        "draft": str(args.draft.expanduser().resolve()),
        "max_tokens": args.max_tokens,
        "prompt_tokens": args.prompt_tokens,
        "timeout_seconds": args.timeout_seconds,
        "policy": {
            "draft_model": str(args.draft.expanduser().resolve()),
            "num_draft": args.num_draft,
            "xpress_num_passes": args.xpress_num_passes,
        },
        "performance_claim": False,
        "qualified": False,
        "cases": ["b1_greedy", "b2_mixed_sampling", "apcv2_paired_resume"],
        "will_execute": not args.dry_run,
    }


def source_identity():
    files = [
        ROOT / "scripts/smoke_parallel_draft_metal.py",
        *sorted((ROOT / "src").rglob("*.py")),
    ]
    per_file = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    revision = hashlib.sha256(json.dumps(per_file, sort_keys=True).encode()).hexdigest()
    return {"root": str(ROOT), "source_sha256": revision, "files_sha256": per_file}


def host_float32(tensor):
    """NumPy cannot represent MLX BF16 buffers directly."""
    import mlx.core as mx
    import numpy as np

    return np.asarray(tensor.astype(mx.float32))


def run(args, report):
    import mlx.core as mx
    import numpy as np

    if not mx.metal.is_available():
        raise RuntimeError("Metal backend is unavailable")
    mx.set_default_device(mx.gpu)
    if mx.default_device() != mx.gpu:
        raise RuntimeError("default device is not Metal GPU")
    report["device"] = "Metal GPU"
    import mlx2.runtime.external_speculative as external
    from mlx2.adapters.standard_decoder import StandardDecoderAdapter
    from mlx2.runtime.apc_v2 import APCKey, APCv2
    from mlx2.runtime.sample_utils import LaneRNG
    from mlx2.runtime.speculative_sampling import softmax

    if not Path(external.__file__).resolve().is_relative_to((ROOT / "src").resolve()):
        raise RuntimeError("imported executor is outside the hashed source archive")

    started = time.perf_counter()
    adapter = StandardDecoderAdapter(str(args.model), execution_policy=report["policy"])
    if adapter.descriptor.model_type != "qwen3":
        raise RuntimeError("smoke requires a Qwen3 target")
    report["load_seconds_diagnostic"] = time.perf_counter() - started
    report["artifact_identity"] = adapter.identity
    report["draft_settings"] = adapter.draft_model.receipt_settings
    report["qualification"] = "unqualified_candidate_smoke"
    model = adapter.model
    prompts = []
    for text in (
        "Explain how to write a Python function that computes the sum of a list. Include a clear example. ",
        "Describe a mutex and explain why two threads can race when updating a shared counter. ",
    ):
        ids = adapter.tokenizer.encode(text * 20, add_special_tokens=False)
        prompts.append(list(ids[: args.prompt_tokens]))
    report["prompt_ids"] = prompts

    def engine():
        result = adapter.create_external_batch(
            completion_batch_size=2,
            prefill_step_size=128,
            stop_tokens=(),
            ready_drain="all",
        )
        if not isinstance(result, external.ExternalDraftBatchGenerator):
            raise TypeError("unexpected external generator")
        return result

    def ordinary(prompt, count):
        cache = model.make_cache()
        tokens = []
        inputs = list(prompt)
        for _ in range(count):
            logits = model(mx.array([inputs]), cache=cache)
            token = int(mx.argmax(logits[0, -1]).item())
            tokens.append(token)
            inputs = [token]
        return tokens

    def drain(batch):
        # Admit both prompts before decoding so B2 is actually exercised.
        for lane in list(batch.lanes.values()):
            while lane.anchor is None:
                batch._prefill(lane)
        outputs, finishes = {}, {}
        for _ in range(128):
            _, responses = batch.next()
            failures = batch.take_lane_failures()
            if failures:
                raise RuntimeError(f"lane failures: {failures}")
            for response in responses:
                outputs.setdefault(response.uid, []).append(response.token)
                if response.finish_reason:
                    finishes[response.uid] = response
            if not batch.lanes:
                break
        else:
            raise RuntimeError("bounded scheduler polls exhausted")
        stats = dict(batch.scheduler_stats)
        if (
            stats["external_rounds"] <= 0
            or stats["proposed_tokens"] <= 0
            or stats["draft_fallbacks"]
        ):
            raise AssertionError(f"missing draft engagement or fallback: {stats}")
        for final in finishes.values():
            receipt = final.speculative_receipt
            if receipt["kind"] != "external_xpress" or receipt["ordinary_fallback"]:
                raise AssertionError("wrong route receipt or ordinary fallback")
            if (
                receipt["draft_settings"]["proposal_distribution"]
                != "deterministic_point_mass"
            ):
                raise AssertionError("proposal distribution receipt is incorrect")
        return outputs, finishes, stats

    started = time.perf_counter()
    batch = engine()
    uid = batch.insert([prompts[0]], max_tokens=[args.max_tokens])[0]
    outputs, final, stats = drain(batch)
    expected = ordinary(prompts[0], args.max_tokens)
    if outputs[uid] != expected:
        raise AssertionError(f"B1 greedy mismatch: {outputs[uid]} != {expected}")
    report["b1_greedy"] = {
        "token_ids": outputs[uid],
        "ordinary_ids": expected,
        "stats": stats,
        "receipt": final[uid].speculative_receipt,
        "seconds_diagnostic": time.perf_counter() - started,
    }

    # Inspect the actual transformed sampled p/q law; stochastic output IDs
    # need not match a differently scheduled ordinary random stream.
    sampled_reference = softmax(
        host_float32(model(mx.array([prompts[1]]), cache=model.make_cache())[0, -1]),
        0.8,
    )
    law_checks = {
        "point_mass_rows": 0,
        "sampled_rows": 0,
        "residual_identity_max_error": 0.0,
        "first_sampled_l1": None,
    }
    original_verify = external.verify_proposals

    def checked_verify(tokens, proposals, targets, rng, **kwargs):
        for token, q, p in zip(tokens, proposals, targets):
            q = np.asarray(q)
            p = np.asarray(p, dtype=np.float64)
            if q[token] != 1 or np.count_nonzero(q) != 1:
                raise AssertionError("actual XPress q is not a point mass")
            if not np.all(np.isfinite(p)) or np.min(p) < 0 or abs(p.sum() - 1) > 1e-6:
                raise AssertionError("invalid transformed target probability law")
            law_checks["point_mass_rows"] += 1
            if np.count_nonzero(p) > 1:
                law_checks["sampled_rows"] += 1
                if law_checks["first_sampled_l1"] is None:
                    error = float(np.abs(p - sampled_reference).sum())
                    law_checks["first_sampled_l1"] = error
                    if error > 0.002:
                        raise AssertionError(
                            f"sampled target law differs from ordinary prompt: L1={error}"
                        )
                acceptance = float(p[token])
                residual = p.copy()
                residual[token] = 0
                mixed = residual + acceptance * q
                error = float(np.max(np.abs(mixed - p)))
                law_checks["residual_identity_max_error"] = max(
                    law_checks["residual_identity_max_error"], error
                )
                if error > 1e-12:
                    raise AssertionError(
                        "point-mass acceptance/residual identity failed"
                    )
        return original_verify(tokens, proposals, targets, rng, **kwargs)

    started = time.perf_counter()
    batch2 = engine()
    external.verify_proposals = checked_verify
    try:
        ids = batch2.insert(
            prompts,
            max_tokens=[args.max_tokens] * 2,
            lane_rngs=[LaneRNG(302), LaneRNG(303)],
            sampling_configs=[{"sampling_temp": 0.0}, {"sampling_temp": 0.8}],
        )
        outputs2, final2, stats2 = drain(batch2)
    finally:
        external.verify_proposals = original_verify
    if outputs2[ids[0]] != expected:
        raise AssertionError("B2 mixed cohort greedy row differs from ordinary")
    if stats2["target_max_width"] < 2 or not law_checks["sampled_rows"]:
        raise AssertionError("B2 cohort or sampled law was not exercised")
    report["b2_mixed_sampling"] = {
        "token_ids": outputs2,
        "stats": stats2,
        "receipts": {uid: end.speculative_receipt for uid, end in final2.items()},
        "law_checks": law_checks,
        "empirical_distribution_qualification": False,
        "seconds_diagnostic": time.perf_counter() - started,
    }

    end = final[uid]
    apc = APCv2(max_size=1, layout_name=adapter.layout)
    key = APCKey(
        "qwen3+xpress",
        revision=adapter.identity["fingerprint"],
        cache_layout_fingerprint=adapter.layout,
    )
    try:
        apc.store(key, end.all_tokens, end.prompt_cache, sidecar=end.cache_sidecar)
        hit = apc.lookup(key, end.all_tokens + [end.token])
        if not hit.hit or hit.hit_kind != "external_draft_sidecar":
            raise AssertionError("paired APCv2 hit missing")
        resumed = engine()
        resumed.insert(
            [[end.token]],
            max_tokens=[args.max_tokens],
            caches=[hit.cache],
            all_tokens=[end.all_tokens],
            cache_states=[hit.sidecar],
        )
        actual, finishes, reuse_stats = drain(resumed)
        reused_expected = ordinary(end.all_tokens + [end.token], args.max_tokens)
        if actual[0] != reused_expected or reuse_stats["paired_cache_resumes"] != 1:
            raise AssertionError("paired resume token parity or mechanism failed")
        report["apcv2_paired_resume"] = {
            "token_ids": actual[0],
            "ordinary_ids": reused_expected,
            "stats": reuse_stats,
            "receipt": finishes[0].speculative_receipt,
        }
        hit.cache.close()
    finally:
        apc.clear(release_memory=False)
    report["draft_counters"] = dict(adapter.draft_model.stats)
    adapter.close()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = preflight(args)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    report["started_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def timeout(_signal, _frame):
        raise TimeoutError(f"bounded smoke exceeded {args.timeout_seconds} seconds")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.timeout_seconds)
    try:
        report["source_identity"] = source_identity()
        run(args, report)
        report["passed"] = True
    except Exception as error:  # noqa: BLE001 - persist diagnostic failure receipt
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        signal.alarm(0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "out": str(args.out),
                "error": report.get("error"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
