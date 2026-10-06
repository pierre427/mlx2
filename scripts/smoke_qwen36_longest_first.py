#!/usr/bin/env python3
"""Bounded owned-GPU smoke for the Qwen3.6 longest-first DFlash2 route.

This is mechanism-engagement evidence only. It does not qualify the route or
measure performance. Invoke it inside a CPG lease and both GPU lock owners.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path


SHARED_LOCK = Path("/Users/Shared/mlxuag/gpu.lock")
TMP_LOCK = Path("/tmp/gpu.lock")


def source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src" / "mlx2").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def lock_receipts() -> list[dict]:
    if not SHARED_LOCK.is_dir():
        raise RuntimeError(f"shared GPU lock directory is absent: {SHARED_LOCK}")
    owner_json = SHARED_LOCK / "owner.json"
    owner_text = SHARED_LOCK / "owner.txt"
    if owner_json.is_file():
        shared_owner = json.loads(owner_json.read_text())
    elif owner_text.is_file():
        shared_owner = owner_text.read_text().strip()
    else:
        raise RuntimeError("shared GPU lock has no owner receipt")
    if TMP_LOCK.is_dir():
        tmp_owner = TMP_LOCK / "owner.json"
        if not tmp_owner.is_file():
            raise RuntimeError("/tmp GPU lock directory has no owner receipt")
        tmp_receipt = {
            "path": str(TMP_LOCK),
            "kind": "directory",
            "owner": json.loads(tmp_owner.read_text()),
        }
    elif TMP_LOCK.is_file():
        descriptor = os.open(TMP_LOCK, os.O_RDWR)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                tmp_held = True
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                tmp_held = False
        finally:
            os.close(descriptor)
        if not tmp_held:
            raise RuntimeError("legacy /tmp GPU flock is not held")
        tmp_receipt = {"path": str(TMP_LOCK), "kind": "flock", "held": True}
    else:
        raise RuntimeError(f"legacy GPU lock is absent: {TMP_LOCK}")
    return [
        {"path": str(SHARED_LOCK), "kind": "directory", "owner": shared_owner},
        tmp_receipt,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--cpg-lease", required=True)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        parser.error("Metal execution requires --i-own-the-gpu")
    if args.max_tokens < 4:
        parser.error("--max-tokens must be at least four")

    root = Path(__file__).resolve().parents[1]
    report = {
        "schema": "mlx2.qwen36-longest-first-smoke.v1",
        "implemented": True,
        "qualified": False,
        "selected": True,
        "observed_used": False,
        "performance_claim": False,
        "passed": False,
        "target": str(args.target.expanduser().resolve()),
        "draft": str(args.draft.expanduser().resolve()),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_hash": source_hash(root),
        "platform": platform.platform(),
        "started_at": time.time(),
        "cpg": {
            "session": os.environ.get("GPUQ_SESSION"),
            "lease": os.environ.get("GPUQ_LEASE"),
            "declared_lease": args.cpg_lease,
            "outer_wrapper": "gpuq.sh -> cpg_job.py --lease --require-radio",
        },
    }
    adapter = None
    batch = None
    try:
        report["locks"] = lock_receipts()
        import mlx.core as mx

        from mlx2.adapters.dflash2 import (
            content_revision as draft_content_revision,
            inspect_drafter,
        )
        from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
        from mlx2.adapters.qwen38_27b import content_revision as target_revision

        draft_record = inspect_drafter(args.draft, args.target)
        policy = {
            "draft_model": str(args.draft.expanduser().resolve()),
            "draft_revision": draft_content_revision(draft_record),
            "target_revision": target_revision(args.target),
            "num_draft": 3,
            "continuation_pool": {"sources": ["external"], "limit": 15},
            "continuation_strategy": "longest_first_exact_prefix",
        }
        report["execution_policy"] = policy
        load_started = time.perf_counter()
        adapter = Qwen3635BA3BAdapter(args.target, execution_policy=policy)
        report["load_seconds"] = time.perf_counter() - load_started
        prompt = "Continue this exact sequence: alpha beta gamma alpha beta gamma"
        tokens = list(adapter.tokenizer.encode(prompt, add_special_tokens=False))
        if len(tokens) < 4:
            raise RuntimeError("smoke prompt encoded to fewer than four tokens")
        stops = [[int(token)] for token in adapter.tokenizer.eos_token_ids]
        batch = adapter.create_external_batch(
            completion_batch_size=1,
            stop_tokens=stops,
        )
        uid = batch.insert(
            [tokens],
            max_tokens=[args.max_tokens],
            sampling_configs=[{"sampling_temp": 0}],
        )[0]
        generated = []
        receipts = []
        while batch.lanes:
            _, responses = batch.next()
            for response in responses:
                if response.uid != uid:
                    raise RuntimeError("smoke received a response for another request")
                generated.append(int(response.token))
                receipts.append(response.speculative_receipt)
        mx.synchronize()
        if not receipts:
            raise RuntimeError("smoke emitted no route receipts")
        final = receipts[-1]
        pool = final.get("continuation_pool", {})
        strategy = pool.get("strategy", {})
        stats = dict(batch.scheduler_stats)
        report.update(
            {
                "prompt_tokens": len(tokens),
                "generated_tokens": generated,
                "final_receipt": final,
                "scheduler_stats": stats,
                "peak_memory_bytes": int(mx.get_peak_memory()),
                "observed_used": bool(strategy.get("observed_used")),
            }
        )
        report["passed"] = bool(
            generated
            and strategy.get("implemented") is True
            and strategy.get("qualified") is False
            and strategy.get("selected") is True
            and strategy.get("observed_used") is True
            and strategy.get("proposal_state_authoritative") is False
            and strategy.get("apcv2_publication") is False
            and int(strategy.get("attempts", 0)) > 0
            and int(strategy.get("target_rows", 0)) > 0
            and int(strategy.get("routed_expert_rows", 0)) > 0
            and int(stats.get("external_continuation_longest_first_attempts", 0))
            > 0
        )
        if not report["passed"]:
            raise RuntimeError("longest-first route receipt failed the smoke gate")
        return 0
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if batch is not None:
            batch.close()
        if adapter is not None:
            adapter.close()
        try:
            import mlx.core as mx

            mx.synchronize()
            mx.clear_cache()
        except BaseException as exc:
            report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        report["finished_at"] = time.time()
        report["elapsed_seconds"] = report["finished_at"] - report["started_at"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
