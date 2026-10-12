#!/usr/bin/env python3
"""Repeat the historical 1024-token replay fixture with a bounded margin window."""

import argparse
import hashlib
import json
import signal
import subprocess
import tempfile
from pathlib import Path


def drain(job):
    logs = []
    while True:
        event = job.events.get(timeout=120)
        if "error" in event:
            raise RuntimeError(str(event))
        if "logprob" in event:
            logs.append(event["logprob"])
        if "finish_reason" in event:
            return {"receipt": event.get("receipt"), "margin_window": logs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from mlx2.serving import ServingEngine, runtime_identity
    from scripts.qualify_native_vlm_media import _clear_apc_for_isolated_baseline

    def timeout(_signal, _frame):
        raise TimeoutError("replay diagnostic exceeded its 300-second device budget")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    prompt = (
        "Session be28e3113112 lane 0. Read the archive and answer the question at the end.\n"
        + " archival evidence" * 471
        + "\nThe unique fact is: the launch code is NEEDLE-BE28E3-0.\n"
        "What is the launch code? Reply with NEEDLE-BE28E3-0 first."
    )
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    if digest != "f815a03f0a6cc73face0d0fcc821192a04b9162f7a1456fb450ce0c92009a449":
        raise ValueError("historical prompt bytes differ")
    report = {
        "schema": "mlx2.flash-next-replay-regression-diagnostic.v1",
        "qualification_claim": False, "performance_claim": False,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime_before": runtime_identity(), "prompt_sha256": digest,
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pairs": [],
    }
    engine = None
    cache_dir = None
    try:
        cache_dir = tempfile.TemporaryDirectory(prefix="mlx2-regression-replay-")
        engine = ServingEngine(
            args.model, mtp=True, max_lanes=1, max_inflight=2,
            max_context=65536, cache_bytes=16 << 30, cache_dir=cache_dir.name,
            qualification_mode=True,
            execution_policy={"num_draft": 2, "adaptive_mtp_depth": False,
                              "mtp_ordinary_handoff": {"enabled": True, "max_mtp_width": 3}},
        )
        if not engine.ready.wait(120) or engine.error:
            raise RuntimeError(f"engine failed to load: {engine.error}")
        report["initial"] = engine.status()
        body = {"messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": 128, "min_tokens": 128,
                "enable_thinking": False, "return_token_trace": True,
                "logprobs": True, "top_logprobs": 2,
                "logprobs_start": 85, "logprobs_end": 90}
        for _ in range(3):
            _clear_apc_for_isolated_baseline(engine)
            pair = {"cold": drain(engine.submit(body)), "warm": drain(engine.submit(body))}
            for arm in pair.values():
                if arm["receipt"]["prompt_tokens"] != 1024:
                    raise ValueError("historical token geometry differs")
            report["pairs"].append(pair)
        report["completed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if engine is not None:
            report["final"] = engine.status()
            engine.close()
        if cache_dir is not None:
            cache_dir.cleanup()
        signal.alarm(0)
        report["runtime_after"] = runtime_identity()
        report["source_stable"] = report["runtime_before"] == report["runtime_after"]
        with args.out.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
