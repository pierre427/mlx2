#!/usr/bin/env python3
"""Bounded one-expert source I/O/QAT oracle, never a serving qualification."""
import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(Path(__file__).parent))

from mlx2.desktop_campaign_worker import atomic, check_ownership
from mlx2.desktop_runner import digest, source_manifest


def unit(out):
    spec = json.loads((out / "spec.json").read_text())
    report = {**spec, "passed": False, "qualification": False,
              "performance_claim": False, "scope": "one routed expert operator only; synthetic activations",
              "cells": []}
    mx = None
    try:
        report["ownership"] = check_ownership(spec)
        if digest(source_manifest(ROOT)) != spec["source_snapshot_sha256"]:
            raise RuntimeError("source changed before operator oracle")
        from probe_deepseek_v41_gpu import environment
        report["before"] = environment()
        import mlx.core as mx
        import numpy as np
        from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact, read_tensors
        from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights, expert_names, PROJECTIONS
        from test_deepseek_v41_source_activation_reuse_cpu import _original

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(4 << 30)
        mx.set_cache_limit(128 << 20)
        artifact = inspect_artifact(spec["model"])
        if artifact["identity"]["fingerprint"] != spec["fingerprint"]:
            raise RuntimeError("source checkpoint identity changed")
        names = expert_names(0, 0)
        grouped = read_tensors(artifact, names)
        tensor_rows = []
        for name in names:
            separate = read_tensors(artifact, [name])[name]
            if grouped[name].tobytes() != separate.tobytes():
                raise RuntimeError("grouped read changed official source bytes")
            tensor_rows.append({"name": name, "bytes": grouped[name].nbytes,
                                "sha256": hashlib.sha256(grouped[name]).hexdigest()})
        report["source_tensors"] = tensor_rows
        data = {}
        for kind, part in PROJECTIONS.items():
            prefix = f"layers.0.ffn.experts.0.{part}."
            data[kind + ".qweight"] = mx.array(grouped[prefix + "weight"].view("<u4"))
            data[kind + ".scales"] = mx.array(grouped[prefix + "scale"])
        mx.eval(data)
        del grouped, separate
        weights = object.__new__(SourceV41Weights)
        weights._ready = lambda: None
        hidden = artifact["args"].hidden_size
        # Width one isolates one expert; this is not the checkpoint's six-way
        # routing or a model forward, and cannot establish model quality.
        weights.artifact = {"args": SimpleNamespace(hidden_size=hidden, num_experts_per_tok=1)}
        weights.expert_stream = SimpleNamespace(_active=True, acquire=lambda prefix, expert: data,
                                               stats={"lookups": 0})
        weights.scheduled_expert_calls = {}
        rng = np.random.default_rng(41025)
        for rows in (1, 7, 32, 64):
            x = mx.array(rng.normal(0, .25, (rows, hidden)).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)
            ids = np.zeros((rows, 1), np.uint32)
            scores = mx.array(rng.uniform(.1, 1.0, (rows, 1)).astype(np.float32))
            expected = _original({0: data}, x, ids, scores)
            mx.eval(expected)
            actual = weights.moe(x, ids, scores, "blk.0")
            mx.eval(actual)
            a, b = np.array(expected), np.array(actual)
            equal = a.tobytes() == b.tobytes()
            report["cells"].append({"rows": rows, "hidden": hidden, "exact": equal,
                                    "max_absolute_error": float(np.max(np.abs(a-b))),
                                    "output_sha256": hashlib.sha256(b).hexdigest()})
            atomic(out / "results.json", report)
            if not equal:
                raise RuntimeError("reused activation changed official expert GPU output")
            del expected, actual, x, ids, scores, a, b
        report["peak_mlx_bytes"] = mx.get_peak_memory()
        if digest(source_manifest(ROOT)) != spec["source_snapshot_sha256"]:
            raise RuntimeError("source changed during operator oracle")
        report["after"] = environment()
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        if mx is not None:
            mx.synchronize()
            gc.collect()
            mx.clear_cache()
        atomic(out / "results.json", report)
    print(json.dumps({"passed": report["passed"], "cells": report["cells"], "error": report.get("error")}))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "unit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("~/mlx-models/DeepSeek-V4.1-Flash-1017-Source"))
    args = parser.parse_args()
    out = args.output.resolve()
    if args.mode == "unit":
        return unit(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    from mlx2.desktop_campaign_process import spawn
    artifact = inspect_artifact(args.model)
    spec = dict(out=str(out), model=str(args.model.resolve()),
                checkpoint=args.model.name + "@" + artifact["identity"]["fingerprint"],
                fingerprint=artifact["identity"]["fingerprint"],
                source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                source_snapshot_sha256=digest(source_manifest(ROOT)), job_id=out.name,
                mcp_url="http://127.0.0.1:8766/mcp", thermal_controlled=False,
                thermal_policy="recorded; user waived thermal gate", memory_limit_bytes=4 << 30)
    atomic(out / "spec.json", spec)
    atomic(out / "source.json", source_manifest(ROOT))
    with DesktopCPGLease(out, spec["job_id"], checkpoint=spec["checkpoint"], source_revision=spec["source_revision"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", spec["job_id"],
                   "--desktop-cpg-lease", str(lease.path), "--receipt", str(out / "paired-locks.json"),
                   "--", sys.executable, str(Path(__file__).resolve()), "unit", "--output", str(out)]
        child = spawn(command, start_new_session=True)
        try:
            deadline = time.monotonic() + 180
            while child.poll() is None:
                lease.check()
                if time.monotonic() >= deadline:
                    raise TimeoutError("one-expert GPU oracle ownership deadline")
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            child.stop(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
