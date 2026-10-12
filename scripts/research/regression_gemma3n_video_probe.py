#!/usr/bin/env python3
"""Bounded source/video geometry diagnostic; no qualification/performance claim."""

import argparse
import gc
import hashlib
import json
import signal
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from scripts.qualify_native_vlm_media import (
        _batching_observation, _clear_apc_for_isolated_baseline, _drain,
        _enable_batching_observer, _fixture, _parity, _request,
    )
    from mlx2.serving import ServingEngine, runtime_identity

    def timeout(_signal, _frame):
        raise TimeoutError("video diagnostic exceeded its 300-second device budget")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    report = {
        "schema": "mlx2.gemma3n-video-regression-diagnostic.v1",
        "qualification_claim": False, "performance_claim": False,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime_before": runtime_identity(),
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    engine = None
    try:
        # Load the direct-reference adapter on this thread. Serving owns its
        # worker's Metal stream; its tensors cannot be directly evaluated on
        # the producer thread's stream registry.
        import mlx.core as mx
        from mlx2.adapters.mlx_vlm import Gemma3nAdapter

        fixture = _fixture("video")
        body = {**_request("video", fixture), "return_token_trace": True}
        adapter = Gemma3nAdapter(args.model)
        prepared = adapter.prepare_multimodal_request(body)
        report["direct_artifact"] = adapter.identity["fingerprint"]
        report["fixture_sha256"] = fixture["sha256"]
        report["prompt_tokens"] = len(prepared["_mlx2_prompt_tokens"])
        _enable_batching_observer(adapter, "gemma3n")
        report["direct"] = _parity(adapter, prepared, "gemma3n")
        report["encoder"] = _batching_observation(adapter, "gemma3n")
        del prepared, adapter
        gc.collect()
        mx.clear_cache()
        engine = ServingEngine(
            args.model, mtp=False, max_lanes=2, max_inflight=4, max_context=8192,
            prefill_step=256, cache_bytes=2 << 30, lane_matmul="crossover",
        )
        if not engine.ready.wait(120) or engine.error:
            raise RuntimeError(f"engine failed to load: {engine.error}")
        report["initial"] = engine.status()
        _clear_apc_for_isolated_baseline(engine)
        report["cold"] = _drain(engine.submit(body))
        report["warm"] = _drain(engine.submit(body))
        report["final"] = engine.status()
        report["completed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if engine is not None:
            engine.close()
        signal.alarm(0)
        report["runtime_after"] = runtime_identity()
        report["source_stable"] = report["runtime_before"] == report["runtime_after"]
        with args.out.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
