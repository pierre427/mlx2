#!/usr/bin/env python3
"""One-artifact owned gate-policy diagnostic; no full preflight/qualification.

The direct worker exits before the one loopback server starts, so Flash-Next
weights are never resident twice. All source snapshots and failures are kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from bench_uzu_candidates import host_load, paired_summary
from probe_uzu_kernels import atomic, swapouts
from run_single_model_smoke_ladder import stop_server, thermal_sample, wait_health

ROOT = Path(__file__).resolve().parents[1]


def policy_for(model, candidate=True):
    config = json.loads((model / "config.json").read_text())
    if config["model_type"] == "qwen4_exp":
        return {"row_exact_verify": True, "q4_gate_three_rows": candidate, "num_draft": 2}
    if config["model_type"] != "qwen3_5":
        raise ValueError("this campaign is scoped to Flash-Next and dense Qwen27B")
    return {"draft_model": str(model.parent / "Qwen3.8-27B-DFlash2"), "num_draft": 7,
            "minimum_draft_proposals": 2, "progressive_verification_tile": 3,
            "target_verify_row_exact": True, "cost_per_commit_handoff": False,
            "q4_gate_three_rows": candidate}


def direct(args):
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from smoke_local_model import _ordinary_generate

    from mlx2.adapters.registry import inspect_model

    record = {"status": "running", "cells": [], "performance": []}
    adapter = None

    def save():
        atomic(args.out / "direct.json", record)

    try:
        mx.set_memory_limit(80 << 30)
        mx.set_cache_limit(512 << 20)
        resolution = inspect_model(args.model)
        identity = resolution.artifact.get("identity", resolution.artifact)
        record["checkpoint"] = args.model.name + "@" + identity["fingerprint"]
        save()
        adapter = resolution.adapter_type(str(args.model), execution_policy=policy_for(args.model, args.candidate == "on"))
        record["device"] = mx.device_info().get("device_name")
        record["mlx_version"] = mx.__version__
        record["ordinary_reference"] = _ordinary_generate(adapter, "Say hello in one sentence.", 16)
        if args.reference_only:
            record["status"] = "passed"
            return
        from mlx2.runtime.cow_cache import snapshot_prompt_cache_descriptors
        from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
        from mlx2.runtime.models import row_exact_qmv as qmv
        from mlx2.runtime.models.cache import make_prompt_cache

        handle = adapter.q4_gate_three_rows
        model = adapter.model
        record["policy"] = handle.identity()
        record["modules"] = [p for p, _ in handle.modules]
        record["projection_geometries"] = sorted({(m.weight.shape[0], m.weight.shape[1] * 8) for _, m in handle.modules})
        record["auxiliary_draft"] = getattr(adapter, "external_policy", {}).get("draft_model")
        mx.random.seed(1034)

        def arrays(cache):
            return [x for _, x in tree_flatten([c.state for c in cache]) if isinstance(x, mx.array)]

        def same(a, b):
            mx.eval(a, b)
            return a.shape == b.shape and a.dtype == b.dtype and bool(
                mx.array_equal(a.view(mx.uint8), b.view(mx.uint8)).item())

        def digest(items):
            h = hashlib.sha256()
            for name, item in tree_flatten(items):
                h.update(name.encode())
                if isinstance(item, mx.array):
                    mx.eval(item)
                    h.update(str((item.shape, item.dtype)).encode())
                    h.update(np.asarray(item.view(mx.uint8)).tobytes())
                else:
                    h.update(repr(item).encode())
            return h.hexdigest()

        def cache_digest(cache):
            return digest([(c.state, getattr(c, "meta_state", None)) for c in cache])

        if args.phase == "functional":
            # Every target gate; Q5 and routed 3D tensors remain outside scope.
            for path, module in handle.modules:
                k = module.weight.shape[1] * 8
                for rows in (1, 2, 3, 4, 8):
                    for scale in (0.05, 1.0, 8.0):
                        x = (mx.random.normal((1, rows, k)) * scale).astype(mx.bfloat16)
                        handle.set_enabled(False)
                        baseline, _ = qmv.quantized_linear(module, x)
                        mx.eval(baseline)
                        handle.set_enabled(True)
                        candidate, route = qmv.quantized_linear(module, x)
                        reference = qmv.per_row(module, x)
                        exact = same(candidate, baseline) and same(candidate, reference)
                        record["cells"].append({"kind": "gate", "path": path, "rows": rows,
                                                "scale": scale, "exact": exact, "route": route})
                        if not exact:
                            raise AssertionError("real gate scheduling parity failed")
                save()
            print(f"all {len(handle.modules)} gate weights passed", flush=True)
            # Exercise the actual grouped consumer used by shared experts.
            modules = dict(model.language_model.named_modules())
            for path, module in handle.modules:
                peer = modules.get(path.rsplit(".", 1)[0] + ".up_proj")
                if peer is None:
                    continue
                x = mx.random.normal((1, 3, module.weight.shape[1] * 8)).astype(mx.bfloat16)
                handle.set_enabled(False)
                baseline, _ = qmv.quantized_linears((module, peer), x)
                mx.eval(baseline)
                handle.set_enabled(True)
                candidate, route = qmv.quantized_linears((module, peer), x)
                exact = all(same(a, b) for a, b in zip(baseline, candidate))
                record["cells"].append({"kind": "grouped_gate_up", "path": path, "exact": exact, "route": route})
                if not exact:
                    raise AssertionError("grouped gate/up scheduling parity failed")
            save()

        flash = resolution.descriptor.model_type == "qwen4_exp"
        layers = len(model.language_model.model.layers)
        captures = (layers - 1,) if flash else tuple(adapter.draft_model.config.target_layer_ids)
        if flash:
            model.select_external_feature_source("mtp_seed")
            model.configure_progressive_external_row_exact(True)
        text = "The engineering team records exact evidence and checks all state transitions. " * 200
        ids = list(adapter.prompt_tokens({"prompt": text}))
        swap0 = swapouts()
        for context in (128, 512):
            cache = make_prompt_cache(model)
            logits = model(mx.array(ids[:context])[None], cache=cache)
            mx.eval(logits[:, -1:], arrays(cache))
            del logits
            prefix_hash = cache_digest(cache)

            def verify(enabled, width, accepted, *, hashes=True, _cache=cache, _context=context):
                handle.set_enabled(enabled)
                branch = snapshot_prompt_cache_descriptors(_cache)[0]
                tx = HybridVerifyRows([branch]).begin([width])
                tokens = mx.array(ids[_context:_context + width])[None]
                mx.eval(tokens, arrays(branch))
                before = dict(handle.counts)
                mx.synchronize()
                start = time.perf_counter()
                try:
                    output = model.progressive_external_target_rows(tokens, tx.caches, captures)
                    mx.eval(output.logits, output.features, arrays(branch))
                    tx.commit([accepted])
                    mx.eval(arrays(branch))
                    mx.synchronize()
                    elapsed = time.perf_counter() - start
                    result = {"seconds": elapsed, "row_exact": output.row_exact, "detail": output.detail,
                              "dispatch_delta": {key: handle.counts[key] - before.get(key, 0) for key in handle.counts}}
                    if hashes:
                        result.update(logits=digest(output.logits), features=digest(output.features), state=cache_digest(branch))
                        continuation = []
                        token = int(mx.argmax(output.logits[0, accepted - 1]).item())
                        for _ in range(4):
                            continuation.append(token)
                            row = model(mx.array([[token]]), cache=branch)[0, -1]
                            mx.eval(row, arrays(branch))
                            token = int(mx.argmax(row).item())
                        result["continuation"] = continuation
                        result["continued_state"] = cache_digest(branch)
                    return result
                finally:
                    if not tx.closed:
                        tx.abort()

            if args.phase == "functional":
                shapes = [(width, width) for width in (1, 2, 3, 4, 8)] + [(3, 1), (3, 2)]
                for width, accepted in shapes:
                    arms = [verify(on, width, accepted) for on in (False, True)]
                    exact = all(arms[0][key] == arms[1][key] for key in
                                ("logits", "features", "state", "continuation", "continued_state"))
                    record["cells"].append({"kind": "target", "context": context, "width": width,
                                            "accepted": accepted, "exact": exact, "arms": arms})
                    save()
                    if not exact or not all(a["row_exact"] for a in arms):
                        raise AssertionError("whole-target policy parity or exact route failed")
                    if cache_digest(cache) != prefix_hash:
                        raise AssertionError("private verify branch mutated committed prefix")
                    print(f"target context={context} width={width} commit={accepted} passed", flush=True)
            else:
                samples = []
                for rep in range(args.reps):
                    time.sleep(4)
                    before = thermal_sample()
                    if before["thermal_state"] != 0 or swapouts() != swap0:
                        raise RuntimeError("performance thermal/swap admission refused")
                    order = (False, True) if rep % 2 == 0 else (True, False)
                    arms = {}
                    for enabled in order:
                        verify(enabled, 3, 3, hashes=False)
                        arms[str(enabled)] = verify(enabled, 3, 3, hashes=False)
                    after = thermal_sample()
                    if after["thermal_state"] != 0 or swapouts() != swap0:
                        raise RuntimeError("performance thermal/swap sample refused")
                    samples.append({"rep": rep, "order": list(order), "arms": arms,
                                    "thermal_before": before, "thermal_after": after, "foreign_cpu": host_load()})
                record["performance"].append({"context": context, "width": 3, "samples": samples,
                    "summary": paired_summary([s["arms"]["False"]["seconds"] for s in samples],
                                              [s["arms"]["True"]["seconds"] for s in samples])})
                save()
        handle.set_enabled(True)
        record["final_policy"] = handle.status()
        record["status"] = "passed"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if adapter is not None:
            adapter.close()
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--phase", choices=("functional", "performance"), default="functional")
    parser.add_argument("--reps", type=int, default=6)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--reference-only", action="store_true", help="ordinary reference plus HTTP follow-up for an already completed direct gate")
    parser.add_argument("--candidate", choices=("on", "off"), default="on")
    args = parser.parse_args()
    if args.worker:
        direct(args)
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError("refusing a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)
    from qualification_gpu_ownership import require_qualification_lease
    from run_with_gpu_locks import foreign_model_processes

    owner = json.loads((args.cpg_owner_lock / "owner.json").read_text())
    ownership = require_qualification_lease(task_id=owner["cpg_task"], generation=owner["cpg_generation"],
                                          cpg_owner_lock=args.cpg_owner_lock)
    if foreign_model_processes():
        raise RuntimeError("foreign model process contamination")
    start_thermal = thermal_sample()
    if start_thermal["thermal_state"] not in (0, 1) or (args.phase == "performance" and start_thermal["thermal_state"] != 0):
        raise RuntimeError("thermal admission refused")
    files = ["scripts/probe_q4_gate_rows.py", "src/mlx2/runtime/row_exact/gate_rows.py",
             "src/mlx2/runtime/row_exact/ops.py", "src/mlx2/runtime/models/row_exact_qmv.py",
             "src/mlx2/runtime/models/qwen4_row_exact.py", "src/mlx2/runtime/models/qwen_hybrid_row_exact.py",
             "src/mlx2/adapters/flash_next.py", "src/mlx2/adapters/flash_next_policy.py", "src/mlx2/adapters/qwen38_27b.py"]
    result = {"schema": "mlx2.q4-gate-rows-campaign.v1", "phase": args.phase, "qualified": False,
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_files": {}, "ownership": ownership, "thermal_start": start_thermal, "status": "running"}
    for path in files:
        data = (ROOT / path).read_bytes()
        result["source_files"][path] = hashlib.sha256(data).hexdigest()
        target = args.out / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    policy = policy_for(args.model, args.candidate == "on")
    result["candidate"] = args.candidate
    atomic(args.out / "policy.json", policy)
    atomic(args.out / "result.json", result)
    worker = server = None
    swap0 = swapouts()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(str(signum))

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
        with (args.out / "direct.log").open("w") as log:
            worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if worker.wait() != 0:
                raise RuntimeError("direct worker failed; see direct.json/direct.log")
        result["direct"] = json.loads((args.out / "direct.json").read_text())
        result["checkpoint"] = result["direct"]["checkpoint"]
        atomic(args.out / "result.json", result)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        command = [sys.executable, "-m", "mlx2.server", "--model", str(args.model), "--execution-policy",
                   str(args.out / "policy.json"), "--host", "127.0.0.1", "--port", str(port),
                   "--qualification-mode", "--max-lanes", "1", "--max-inflight", "1", "--max-context", "4096",
                   "--cache-bytes", str(1 << 30)]
        command.append("--external-draft" if "draft_model" in policy else "--native-mtp")
        with (args.out / "server.log").open("w") as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            result["server_startup_status"] = wait_health(base, server, 300)
            with urllib.request.urlopen(base + "/v1/models", timeout=20) as response:
                served = json.load(response)["data"][0]["id"]
            responses = []
            for arm in ("cold", "warm"):
                request = {"model": served, "messages": [{"role": "user", "content":
                           "The test identifier is GATE-7492. " * 24 + "Reply only with the test identifier."}],
                           "enable_thinking": False, "temperature": 0, "max_tokens": 32, "return_token_trace": True}
                req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(request).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=240) as response:
                    body = json.load(response)
                atomic(args.out / f"http-{arm}.json", body)
                responses.append(body)
            from mlx2.generation_trace import read_token_trace

            traces = [read_token_trace(body["mlx2"]) for body in responses]
            if traces[0]["token_ids"] != traces[1]["token_ids"]:
                raise AssertionError("HTTP B1 cold/warm token mismatch")
            if not all(b["choices"][0]["message"]["content"].strip() == "GATE-7492" for b in responses):
                raise AssertionError("HTTP identifier oracle failed")
            if not responses[1]["mlx2"]["cached_tokens"]:
                raise AssertionError("HTTP APCv2 reuse not observed")
            with urllib.request.urlopen(base + "/v1/status", timeout=20) as response:
                result["server_final_status"] = json.load(response)
            observed = result["server_final_status"].get("execution", {}).get("q4_gate_three_rows", {})
            if args.candidate == "on" and not observed.get("observed_dispatched"):
                raise AssertionError("HTTP route selected policy without three-row dispatch")
            result["http"] = {"exact_tokens": True, "functional_oracle": True,
                              "warm_reused_tokens": responses[1]["mlx2"]["cached_tokens"],
                              "policy_observed": observed}
        result["status"] = "passed"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop_server(worker)
        stop_server(server)
        result["thermal_end"] = thermal_sample()
        result["swapouts_delta"] = swapouts() - swap0
        result["process_groups_stopped"] = all(p is None or p.poll() is not None for p in (worker, server))
        if (args.out / "direct.json").exists():
            result["direct"] = json.loads((args.out / "direct.json").read_text())
            result["checkpoint"] = result["direct"].get("checkpoint")
        atomic(args.out / "result.json", result)


if __name__ == "__main__":
    main()
