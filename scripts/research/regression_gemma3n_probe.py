#!/usr/bin/env python3
"""Bounded serving regression diagnostic; no qualification/performance claim.

Run only under the shared GPU queue and CPG lease. Retains serial/batched
token traces and raw/chat fixture behavior without changing the serving model.
"""

import argparse
import hashlib
import json
import signal
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lane-matmul", choices=("off", "crossover", "exact"), default="crossover")
    args = parser.parse_args()
    from scripts.qualify_native_vlm_media import _clear_apc_for_isolated_baseline, _drain
    from mlx2.serving import ServingEngine, runtime_identity

    def timeout(_signal, _frame):
        raise TimeoutError("diagnostic exceeded its 300-second device budget")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    root = Path(__file__).resolve().parents[2]
    report = {
        "schema": "mlx2.gemma3n-regression-diagnostic.v1",
        "qualification_claim": False,
        "performance_claim": False,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "runtime_before": runtime_identity(),
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started": time.time(),
        "cases": {},
    }
    engine = None
    try:
        engine = ServingEngine(args.model, mtp=False, max_lanes=2, max_inflight=4,
                               max_context=8192, prefill_step=256, cache_bytes=2 << 30,
                               lane_matmul=args.lane_matmul)
        if not engine.ready.wait(120) or engine.error:
            raise RuntimeError(f"engine failed to load: {engine.error}")
        report["artifact"] = engine.snapshot.get("artifact")
        report["settings"] = engine.snapshot.get("settings")
        prompts = ("Two plus two is", "Three plus three is")
        for fixture in ("raw", "chat"):
            def request(prompt):
                body = {"temperature": 0, "max_tokens": 8, "return_token_trace": True}
                body.update({"prompt": prompt} if fixture == "raw" else
                            {"messages": [{"role": "user", "content": prompt}]})
                return body

            serial = []
            for prompt in prompts:
                _clear_apc_for_isolated_baseline(engine)
                serial.append(_drain(engine.submit(request(prompt))))
            _clear_apc_for_isolated_baseline(engine)
            cohort = {"id": f"regression-gemma3n-{fixture}", "size": 2}
            jobs = [engine.submit({**request(prompt), "batch_cohort": cohort})
                    for prompt in prompts]
            batched = [_drain(job) for job in jobs]
            _clear_apc_for_isolated_baseline(engine)
            cold = _drain(engine.submit(request(prompts[0])))
            warm = _drain(engine.submit(request(prompts[0])))
            report["cases"][fixture] = {"serial": serial, "batched": batched,
                                          "cold": cold, "warm": warm}
            print(f"completed {fixture}", flush=True)
        report["completed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if engine is not None:
            engine.close()
        signal.alarm(0)
        report["runtime_after"] = runtime_identity()
        report["source_stable"] = report["runtime_after"] == report["runtime_before"]
        report["finished"] = time.time()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
