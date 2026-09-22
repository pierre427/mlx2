#!/usr/bin/env python3
"""Current-source Qwen4 eager-dispatch GPU qualification.

The harness loads the production Flash-Next adapter once, compares ordinary
forwards with eager dispatch disabled/enabled on independent caches, exercises
the configured row boundary, and runs an interleaved decode stride sweep.  It
requires an explicit caller-bound exclusive lease with finite expiry, checks
both GPU-lock receipts before each forward, and records evidence only; it
does not change the serving policy or qualification registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LOCK_RECEIPTS = (
    Path("/tmp/gpu.lock/owner.json"),
    Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
)
SCHEMA = "mlx2.eager-dispatch-qualification.v1"


def _load_lock_receipts(*, expected: dict, session_id: str, owner: str,
                        now: float | None = None) -> list[dict]:
    """Check the caller's explicit lease receipt against both current locks.

    The external lease owner supplies a receipt with a Unix ``expires_at``;
    older receipts without an expiry fail closed. This validates bounded lease
    evidence, not acquisition or renewal: the external coordinator still owns
    those operations. Never infer caller ownership from the global lock files.
    """
    now = time.time() if now is None else now
    if not isinstance(expected, dict) or not session_id or not owner:
        raise RuntimeError("an explicit caller session, owner and GPU lease receipt are required")
    identity_keys = ("campaign_id", "session_id", "lease_id", "generation", "owner")
    if any(not isinstance(expected.get(key), str) or not expected[key]
           for key in identity_keys if key != "generation") or "generation" not in expected:
        raise RuntimeError("caller GPU lease receipt has an incomplete identity")
    if expected["session_id"] != session_id or expected["owner"] != owner:
        raise RuntimeError("GPU lease receipt belongs to another caller session or owner")
    if type(expected["generation"]) is not int or expected["generation"] < 1:
        raise RuntimeError("GPU lease generation must be a positive integer")
    missing = [str(path) for path in LOCK_RECEIPTS if not path.is_file()]
    if missing:
        raise RuntimeError(f"GPU qualification requires both lock receipts; missing {missing}")
    receipts = [json.loads(path.read_text()) for path in LOCK_RECEIPTS]
    for receipt in [expected, *receipts]:
        if not isinstance(receipt, dict):
            raise RuntimeError("GPU lease receipt must be an object")
        if any(receipt.get(key) != expected[key] for key in identity_keys):
            raise RuntimeError("GPU lock receipts do not match the caller's lease identity")
        if receipt.get("resource_key") != "gpu" or receipt.get("mode") != "exclusive":
            raise RuntimeError("GPU qualification requires an exclusive GPU lease")
        expires = receipt.get("expires_at")
        if (isinstance(expires, bool) or not isinstance(expires, (int, float))
                or not math.isfinite(expires) or expires <= now):
            raise RuntimeError("GPU lease receipt is expired or lacks a finite expires_at")
        if receipt.get("status", "active") != "active":
            raise RuntimeError("GPU lease is not active")
    return receipts


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src" / "mlx2").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _compare(mx, left, right, *, lease_check) -> dict:
    lease_check()
    delta = mx.abs(left.astype(mx.float32) - right.astype(mx.float32))
    maximum = mx.max(delta)
    exact = mx.all(left == right)
    finite = mx.all(mx.isfinite(left)) & mx.all(mx.isfinite(right))
    mx.eval(maximum, exact, finite)
    return {
        "exact": bool(exact.item()),
        "finite": bool(finite.item()),
        "max_abs": float(maximum.item()),
    }


def _set_mode(qwen4_exp, round_levers, *, enabled: bool, max_rows: int, stride: int):
    qwen4_exp._EAGER_DISPATCH = bool(enabled)
    qwen4_exp._EAGER_DISPATCH_MAX_ROWS = int(max_rows)
    qwen4_exp._EAGER_DISPATCH_STRIDE = int(stride)
    round_levers.reset_counters()


def _forward(mx, model, tokens, cache, *, lease_check) -> tuple[object, float]:
    lease_check()
    started = time.perf_counter_ns()
    logits = model(mx.array(tokens, mx.uint32)[None], cache=cache)
    mx.eval(logits)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return logits, elapsed_ms


def _summary(samples: list[float]) -> dict:
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _tokens_for_width(prompt: list[int], width: int) -> list[int]:
    return (prompt * ((width + len(prompt) - 1) // len(prompt)))[:width]


def _row_boundary_checks(mx, model, prompt, qwen4_exp, round_levers, *, max_rows, stride, lease_check):
    layer_count = len(model.layers)
    expected_evals = (layer_count + stride - 1) // stride
    checks = []
    for width in (min(8, max_rows), max_rows, max_rows + 1):
        tokens = _tokens_for_width(prompt, width)
        arms = {}
        for enabled in (False, True):
            _set_mode(
                qwen4_exp,
                round_levers,
                enabled=enabled,
                max_rows=max_rows,
                stride=stride,
            )
            logits, elapsed_ms = _forward(mx, model, tokens, model.make_cache(), lease_check=lease_check)
            arms["on" if enabled else "off"] = {
                "logits": logits,
                "elapsed_ms": elapsed_ms,
                "status": qwen4_exp.qwen4_eager_dispatch_status(),
            }
        parity = _compare(mx, arms["off"]["logits"], arms["on"]["logits"], lease_check=lease_check)
        on_status = arms["on"]["status"]
        should_engage = width <= max_rows
        engagement_ok = (
            on_status["forwards"] == int(should_engage)
            and on_status["row_declines"] == int(not should_engage)
            and on_status["async_evals"] == (expected_evals if should_engage else 0)
        )
        off_status = arms["off"]["status"]
        off_clean = all(
            off_status[name] == 0
            for name in ("forwards", "row_declines", "async_evals")
        )
        checks.append(
            {
                "width": width,
                "should_engage": should_engage,
                "parity": parity,
                "engagement_ok": engagement_ok,
                "off_clean": off_clean,
                "off_ms": arms["off"]["elapsed_ms"],
                "on_ms": arms["on"]["elapsed_ms"],
                "on_status": on_status,
            }
        )
    return checks


def _decode_sweep(
    mx,
    model,
    prompt,
    qwen4_exp,
    round_levers,
    *,
    max_rows,
    strides,
    warmups,
    rounds,
    lease_check,
):
    arms = {"off": {"enabled": False, "stride": 1}}
    arms.update(
        {
            f"stride_{stride}": {"enabled": True, "stride": stride}
            for stride in strides
        }
    )
    for arm in arms.values():
        arm["cache"] = model.make_cache()
        arm["samples_ms"] = []
        arm["async_evals"] = []
        _set_mode(
            qwen4_exp,
            round_levers,
            enabled=arm["enabled"],
            max_rows=max_rows,
            stride=arm["stride"],
        )
        logits, _ = _forward(mx, model, prompt, arm["cache"], lease_check=lease_check)
        arm["last_logits"] = logits

    prefill_parity = {
        name: _compare(mx, arms["off"]["last_logits"], arm["last_logits"], lease_check=lease_check)
        for name, arm in arms.items()
        if name != "off"
    }
    names = tuple(arms)
    parity = []
    total_steps = warmups + rounds
    fixed_token = [int(prompt[-1])]
    for step in range(total_steps):
        order = names[step % len(names) :] + names[: step % len(names)]
        step_outputs = {}
        for name in order:
            arm = arms[name]
            _set_mode(
                qwen4_exp,
                round_levers,
                enabled=arm["enabled"],
                max_rows=max_rows,
                stride=arm["stride"],
            )
            logits, elapsed_ms = _forward(mx, model, fixed_token, arm["cache"], lease_check=lease_check)
            status = qwen4_exp.qwen4_eager_dispatch_status()
            step_outputs[name] = logits
            if step >= warmups:
                arm["samples_ms"].append(elapsed_ms)
                arm["async_evals"].append(status["async_evals"])
        for name in names[1:]:
            check = _compare(mx, step_outputs["off"], step_outputs[name], lease_check=lease_check)
            parity.append({"step": step, "arm": name, **check})

    layer_count = len(model.layers)
    results = {}
    for name, arm in arms.items():
        timing = _summary(arm["samples_ms"])
        expected_evals = (
            (layer_count + arm["stride"] - 1) // arm["stride"]
            if arm["enabled"]
            else 0
        )
        results[name] = {
            "enabled": arm["enabled"],
            "stride": arm["stride"],
            "timing": timing,
            "expected_async_evals_per_forward": expected_evals,
            "observed_async_evals_per_forward": arm["async_evals"],
            "engagement_ok": bool(arm["async_evals"])
            and all(value == expected_evals for value in arm["async_evals"]),
        }
    off_median = results["off"]["timing"]["median_ms"]
    for name in names[1:]:
        results[name]["speedup_vs_off"] = (
            off_median / results[name]["timing"]["median_ms"]
        )
    return {
        "prefill_parity": prefill_parity,
        "decode_parity": parity,
        "arms": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-lease-receipt", type=Path, required=True,
                        help="independently supplied current exclusive lease receipt")
    parser.add_argument("--session-id", required=True, help="executing CPG session")
    parser.add_argument("--owner", required=True, help="executing lease owner")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--strides", default="2,1,4")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument(
        "--prompt",
        default="Explain why exact cache state matters during speculative decoding.",
    )
    args = parser.parse_args()
    if args.max_rows < 1 or args.warmups < 0 or args.rounds < 1:
        raise SystemExit("max-rows and rounds must be positive; warmups must be nonnegative")
    strides = tuple(int(value) for value in args.strides.split(","))
    if not strides or any(value < 1 for value in strides):
        raise SystemExit("strides must be a comma-separated list of positive integers")

    expected = json.loads(args.gpu_lease_receipt.read_text())

    def lease_check():
        return _load_lock_receipts(expected=expected, session_id=args.session_id, owner=args.owner)

    locks = lease_check()
    import mlx.core as mx

    from mlx2.adapters.flash_next import FlashNextAdapter
    from mlx2.runtime import round_levers

    if not mx.metal.is_available():
        raise RuntimeError("Metal GPU is unavailable")
    mx.set_default_device(mx.gpu)
    started_at = time.time()
    lease_check()
    adapter = FlashNextAdapter(str(args.model))
    original = None
    try:
        from mlx2.runtime.models import qwen4_exp

        original = (
            qwen4_exp._EAGER_DISPATCH,
            qwen4_exp._EAGER_DISPATCH_MAX_ROWS,
            qwen4_exp._EAGER_DISPATCH_STRIDE,
        )
        prompt = adapter.tokenizer.encode(args.prompt, add_special_tokens=False)
        if not prompt:
            raise RuntimeError("qualification prompt encoded to zero tokens")
        row_checks = _row_boundary_checks(
            mx,
            adapter.model,
            prompt,
            qwen4_exp,
            round_levers,
            max_rows=args.max_rows,
            stride=strides[0],
            lease_check=lease_check,
        )
        decode = _decode_sweep(
            mx,
            adapter.model,
            _tokens_for_width(prompt, min(16, args.max_rows)),
            qwen4_exp,
            round_levers,
            max_rows=args.max_rows,
            strides=strides,
            warmups=args.warmups,
            rounds=args.rounds,
            lease_check=lease_check,
        )
        row_passed = all(
            check["parity"]["exact"]
            and check["parity"]["finite"]
            and check["engagement_ok"]
            and check["off_clean"]
            for check in row_checks
        )
        decode_parity_passed = all(
            check["exact"] and check["finite"] for check in decode["decode_parity"]
        ) and all(
            check["exact"] and check["finite"]
            for check in decode["prefill_parity"].values()
        )
        engagement_passed = all(
            arm["engagement_ok"] for arm in decode["arms"].values()
        )
        passed = row_passed and decode_parity_passed and engagement_passed
        result = {
            "schema": SCHEMA,
            "status": "passed" if passed else "failed",
            "qualification": (
                "qualified_exact_observed_used" if passed else "failed"
            ),
            "selection": {
                "state": "selected",
                "owner": "FlashNextPolicy",
                "enabled": bool(adapter.policy.eager_dispatch),
                "max_rows": int(adapter.policy.eager_dispatch_max_rows),
                "stride": int(adapter.policy.eager_dispatch_stride),
                "registry_mutated_by_harness": False,
            },
            "started_at": started_at,
            "finished_at": time.time(),
            "git_head": _git_head(),
            "source_sha256": _source_hash(),
            "artifact": adapter.identity,
            "runtime": {
                "python": platform.python_version(),
                "macos": platform.mac_ver()[0],
                "mlx": metadata.version("mlx"),
                "device": str(mx.default_device()),
            },
            "locks": locks,
            "policy": {
                "max_rows": args.max_rows,
                "strides": list(strides),
                "default": adapter.policy.as_dict(),
            },
            "row_boundary_checks": row_checks,
            "decode_sweep": decode,
            "gates": {
                "row_boundary_exact_and_engaged": row_passed,
                "decode_exact": decode_parity_passed,
                "all_arms_observed": engagement_passed,
            },
            "note": (
                "Timing is an interleaved single-run diagnostic, not a serving "
                "selection decision. The harness never mutates qualification registry state."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "status": result["status"],
            "output": str(args.output),
            "gates": result["gates"],
            "timing": {
                name: arm["timing"]["median_ms"]
                for name, arm in decode["arms"].items()
            },
        }, indent=2, sort_keys=True))
        return 0 if passed else 1
    finally:
        if original is not None:
            (
                qwen4_exp._EAGER_DISPATCH,
                qwen4_exp._EAGER_DISPATCH_MAX_ROWS,
                qwen4_exp._EAGER_DISPATCH_STRIDE,
            ) = original
        round_levers.reset_counters()
        adapter.close()
        mx.clear_cache()


if __name__ == "__main__":
    raise SystemExit(main())
