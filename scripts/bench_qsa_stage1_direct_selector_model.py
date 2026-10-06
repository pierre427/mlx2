#!/usr/bin/env python3
"""Loaded-model Qwen3.8 Flash-Next A/B for a QSA direct selector.

The model is loaded once.  Each arm prefills the same deterministic real-token
prompt into a fresh cache, evaluates one greedy decode step, and records QSA
stage-one dispatch counters.  Run only through the shared GPU queue.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _command(*args: str) -> str:
    result = subprocess.run(
        args, cwd=ROOT, check=False, capture_output=True, text=True, timeout=30
    )
    return (result.stdout + result.stderr).strip()


def _lock_owner(path: str) -> dict:
    owner_path = Path(path) / "owner.json"
    if not owner_path.is_file():
        raise SystemExit(f"missing GPU lock owner: {owner_path}")
    return json.loads(owner_path.read_text())


def _prove_gpu_ownership() -> dict:
    expected_lease = os.environ.get("GPUQ_LEASE")
    expected_session = os.environ.get("GPUQ_SESSION")
    if not expected_lease:
        raise SystemExit("GPUQ_LEASE does not own both locks")
    if not expected_session:
        raise SystemExit("GPUQ_SESSION does not own both locks")
    shared = _lock_owner("/Users/Shared/mlxuag/gpu.lock")
    temporary = _lock_owner("/tmp/gpu.lock")
    if any(
        owner.get("lease_id") != expected_lease for owner in (shared, temporary)
    ):
        raise SystemExit("GPUQ_LEASE does not own both locks")
    if any(
        owner.get("session") != expected_session for owner in (shared, temporary)
    ):
        raise SystemExit("GPUQ_SESSION does not own both locks")
    return {"shared": shared, "temporary": temporary}


def _snapshot() -> dict:
    return {
        "swapusage": _command("sysctl", "-n", "vm.swapusage"),
        "vm_stat": _command("vm_stat"),
        "thermal": _command("pmset", "-g", "therm"),
        "active_memory_bytes": int(mx.get_active_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }


def _swapouts() -> int:
    for line in _command("vm_stat").splitlines():
        if line.startswith("Swapouts:"):
            return int(line.split(":", 1)[1].strip().rstrip("."))
    raise RuntimeError("vm_stat did not report Swapouts")


def _arrays_from_cache(cache) -> list[mx.array]:
    from mlx.utils import tree_flatten

    return [
        value
        for _, value in tree_flatten([getattr(layer, "state", None) for layer in cache])
        if isinstance(value, mx.array)
    ]


def _repeat_tokens(tokenizer, context: int) -> list[int]:
    text = (
        "A deterministic systems benchmark compares exact sparse attention "
        "routes while preserving every model output and cache transition. "
    )
    seed = list(tokenizer.encode(text, add_special_tokens=False))
    if not seed:
        raise RuntimeError("tokenizer produced an empty seed")
    return (seed * ((context + len(seed) - 1) // len(seed)))[:context]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--prefill-step", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--candidate", choices=("direct4", "direct8"), default="direct4"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    lock_evidence = _prove_gpu_ownership()
    if os.environ.get("MLX_QWEN4_QSA_STAGE1_DIRECT_SELECTOR", "off") != "off":
        raise SystemExit("benchmark requires the imported runtime default to be off")

    report = {
        "schema": "mlx2.qwen4-qsa-stage1-direct-selector-model-ab.v1",
        "status": "loading",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "commit": _command("git", "rev-parse", "HEAD"),
            "branch": _command("git", "branch", "--show-current"),
            "origin_main": _command("git", "rev-parse", "origin/main"),
            "worktree_dirty": bool(_command("git", "status", "--porcelain")),
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mlx_version": version("mlx"),
            "device": mx.device_info(),
            "lock_evidence": lock_evidence,
        },
        "controls": {
            "model": str(args.model.expanduser().resolve()),
            "context": args.context,
            "prefill_step": args.prefill_step,
            "repeats": args.repeats,
            "arms": ["off", args.candidate],
            "runtime_default": "off",
            "runtime_default_changed": False,
            "cache": "fresh for every arm",
            "decode_steps": 1,
        },
        "host_before": _snapshot(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    adapter = None
    stage1 = None
    try:
        from mlx2.adapters.flash_next import FlashNextAdapter

        adapter = FlashNextAdapter(str(args.model))
        from mlx2.runtime.models.cache import make_prompt_cache

        model = adapter.model
        mx.eval(model.parameters())
        mx.set_cache_limit(8 << 30)

        from mlx2.runtime.models import qwen4_exp
        from mlx2.runtime.models import qwen4_qsa_stage1 as stage1_module

        stage1 = stage1_module

        tokens = mx.array(
            _repeat_tokens(adapter.tokenizer, args.context), dtype=mx.uint32
        )[None]
        mx.eval(tokens)
        report["model"] = {
            "identity": adapter.identity,
            "policy": adapter.policy.as_dict(),
            "tokens": int(tokens.shape[1]),
            "active_memory_after_load_bytes": int(mx.get_active_memory()),
        }

        def counts() -> dict:
            return stage1.qsa_stage1_candidate_status()["runtime_counts"]

        def run(arm: str, run_tokens: mx.array) -> dict:
            stage1._DIRECT_SELECTOR = arm
            stage1._ONEPASS_TOPK = False
            qwen4_exp.qsa_stage1_status(reset=True)
            before = counts()
            swapouts_before = _swapouts()
            cache = make_prompt_cache(model)
            started = time.perf_counter()
            logits = None
            for offset in range(0, int(run_tokens.shape[1]), args.prefill_step):
                chunk = run_tokens[:, offset : offset + args.prefill_step]
                logits = model(chunk, cache=cache)
                mx.eval(logits, *_arrays_from_cache(cache))
            if logits is None:
                raise RuntimeError("empty prefill")
            prefill_seconds = time.perf_counter() - started
            prefill_last = logits[0, -1].astype(mx.float32)
            next_token = mx.argmax(prefill_last).astype(mx.uint32).reshape(1, 1)
            decode_started = time.perf_counter()
            decode_logits = model(next_token, cache=cache)
            mx.eval(decode_logits, *_arrays_from_cache(cache))
            decode_seconds = time.perf_counter() - decode_started
            decode_last = decode_logits[0, -1].astype(mx.float32)
            mx.eval(prefill_last, decode_last, next_token)
            after = counts()
            route_status = qwen4_exp.qsa_stage1_status()
            swapouts_after = _swapouts()
            count_delta = {
                key: value - before.get(key, 0)
                for key, value in after.items()
                if value != before.get(key, 0)
            }
            result = {
                "arm": arm,
                "prefill_seconds": prefill_seconds,
                "prefill_tokens_per_second": int(run_tokens.shape[1]) / prefill_seconds,
                "decode_seconds": decode_seconds,
                "next_token": int(next_token.item()),
                "counter_delta": count_delta,
                "qsa_stage1": route_status,
                "swapouts_delta": swapouts_after - swapouts_before,
                "prefill_last": np.asarray(prefill_last),
                "decode_last": np.asarray(decode_last),
            }
            del cache, logits, decode_logits, prefill_last, decode_last, next_token
            gc.collect()
            mx.clear_cache()
            return result

        warmup_tokens = tokens[:, : min(4096, args.context)]
        for arm in ("off", args.candidate):
            warmup = run(arm, warmup_tokens)
            report.setdefault("warmup", {})[arm] = {
                key: value
                for key, value in warmup.items()
                if key not in {"prefill_last", "decode_last"}
            }

        # Compile both selector widths outside the timed model cells.  The
        # selector kernels are shape-generic; the model's 65K admission gate
        # need not be crossed merely to warm their JIT variants.
        warm_scores = mx.random.uniform(shape=(64, 4096), dtype=mx.float32)
        warm_positions = mx.full((64,), 4096 * 4 - 1, dtype=mx.int32)
        for arm in ("off", args.candidate):
            stage1._DIRECT_SELECTOR = arm
            for topk in (512, 544):
                selected = stage1._select_scores(
                    warm_scores,
                    warm_positions,
                    topk=topk,
                    compress_ratio=4,
                )
                mx.eval(selected)
        del warm_scores, warm_positions, selected
        mx.clear_cache()

        samples = {"off": [], args.candidate: []}
        outputs = {"off": [], args.candidate: []}
        orders = []
        for repeat in range(args.repeats):
            order = (
                ["off", args.candidate] if repeat % 2 == 0 else [args.candidate, "off"]
            )
            orders.append(order)
            for arm in order:
                result = run(arm, tokens)
                samples[arm].append(
                    {
                        key: value
                        for key, value in result.items()
                        if key not in {"prefill_last", "decode_last"}
                    }
                )
                outputs[arm].append(
                    (result.pop("prefill_last"), result.pop("decode_last"))
                )

        reference_prefill, reference_decode = outputs["off"][0]
        parity = []
        for arm in ("off", args.candidate):
            for repeat, (prefill, decode) in enumerate(outputs[arm]):
                parity.append(
                    {
                        "arm": arm,
                        "repeat": repeat,
                        "prefill_bit_identical": bool(
                            np.array_equal(prefill, reference_prefill)
                        ),
                        "decode_bit_identical": bool(
                            np.array_equal(decode, reference_decode)
                        ),
                        "prefill_max_abs": float(
                            np.max(np.abs(prefill - reference_prefill))
                        ),
                        "decode_max_abs": float(
                            np.max(np.abs(decode - reference_decode))
                        ),
                    }
                )
        medians = {
            arm: statistics.median(row["prefill_seconds"] for row in rows)
            for arm, rows in samples.items()
        }
        gain = 1.0 - medians[args.candidate] / medians["off"]
        dispatches = sum(
            row["counter_delta"].get(f"{args.candidate}_topk_dispatches", 0)
            for row in samples[args.candidate]
        )
        exact = all(
            row["prefill_bit_identical"] and row["decode_bit_identical"]
            for row in parity
        )
        report.update(
            {
                "status": "passed" if exact and dispatches > 0 else "failed",
                "orders": orders,
                "samples": samples,
                "parity": parity,
                "summary": {
                    "candidate": args.candidate,
                    "median_prefill_seconds": medians,
                    "candidate_prefill_gain": gain,
                    "candidate_dispatches": dispatches,
                    "exact": exact,
                    "loaded": True,
                    "decoded": True,
                    "selected": False,
                    "observed_used_in_production": False,
                },
                "peak_memory_bytes": int(mx.get_peak_memory()),
            }
        )
        digest = hashlib.sha256()
        digest.update(reference_prefill.tobytes())
        digest.update(reference_decode.tobytes())
        report["reference_output_sha256"] = digest.hexdigest()
    except BaseException as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if stage1 is not None:
            stage1._DIRECT_SELECTOR = "off"
        if adapter is not None:
            adapter.close()
        gc.collect()
        mx.clear_cache()
        report["host_after"] = _snapshot()
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)

    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
