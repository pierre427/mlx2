#!/usr/bin/env python3
"""Bounded repair diagnostics, without qualification or performance claims.

Run one profile per process under the shared GPU queue and CPG lease. The
profile file, runtime bytes, token traces and run-local mechanisms are retained.
"""

import argparse
import hashlib
import json
import signal
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    from scripts.qualify_native_vlm_media import _clear_apc_for_isolated_baseline, _drain
    from scripts.qualify_serving import feature_observations
    from mlx2.serving import ServingEngine, runtime_identity

    def timeout(_signal, _frame):
        raise TimeoutError("diagnostic exceeded its 300-second device budget")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    raw = args.profiles.read_bytes()
    policy = json.loads(raw)[args.profile]
    report = {
        "schema": "mlx2.flash-next-regression-diagnostic.v1",
        "qualification_claim": False,
        "performance_claim": False,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "runtime_before": runtime_identity(),
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "profiles_sha256": hashlib.sha256(raw).hexdigest(),
        "profile": args.profile,
        "requested_policy": policy,
    }
    engine = None
    try:
        engine = ServingEngine(
            args.model, mtp=True, max_lanes=args.width, max_inflight=8,
            max_context=4096, cache_bytes=2 << 30, lane_matmul="auto",
            qualification_mode=True, execution_policy=policy or None,
        )
        if not engine.ready.wait(120) or engine.error:
            raise RuntimeError(f"engine failed to load: {engine.error}")
        initial = engine.status()
        report["initial"] = initial
        prompts = (
            "Write a detailed explanation of how a compiler works in numbered sections.",
            "Implement a Python LRU cache and explain its design.",
            "Prove that the square root of two is irrational, showing every step.",
            "Write a JSON array of 20 fictional users with id, name and email.",
        )[:args.width]
        context = "Reference notes: " + "An inference runtime schedules bounded work and preserves cache ownership. " * 100
        def request(prompt):
            return {"messages": [{"role": "user", "content": context + prompt}],
                    "temperature": 0, "max_tokens": args.max_tokens, "thinking": False,
                    "return_token_trace": True}

        serial = []
        report["serial"] = serial
        for prompt in prompts:
            _clear_apc_for_isolated_baseline(engine)
            serial.append(_drain(engine.submit(request(prompt))))
        _clear_apc_for_isolated_baseline(engine)
        cohort = {"id": "regression-flash-next", "size": args.width}
        jobs = [engine.submit({**request(prompt), "batch_cohort": cohort}) for prompt in prompts]
        batched = [_drain(job) for job in jobs]
        _clear_apc_for_isolated_baseline(engine)
        cold = _drain(engine.submit(request(prompts[0])))
        warm = _drain(engine.submit(request(prompts[0])))
        report.update(serial=serial, batched=batched, cold=cold, warm=warm)
        final = engine.status()
        report["final"] = final
        report["feature_observations"] = feature_observations(final, initial)
        report["completed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if engine is not None:
            report.setdefault("final", engine.status())
            engine.close()
        signal.alarm(0)
        report["runtime_after"] = runtime_identity()
        report["source_stable"] = report["runtime_after"] == report["runtime_before"]
        with args.out.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
