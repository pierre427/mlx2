#!/usr/bin/env python3
"""One-artifact functional GPU probes for the opt-in tree-spec intake.

This is repair evidence, not qualification or controlled performance testing.
One CPG window owns the component probes, ordinary reference and HTTP smoke.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def environment():
    return {name: subprocess.run(command, capture_output=True, text=True).stdout.strip()
        for name, command in {"swap": ["sysctl", "vm.swapusage"],
            "thermals": ["pmset", "-g", "therm"],
            "processes": ["ps", "-axo", "pid,ppid,command"]}.items()}


def source():
    paths = sorted((ROOT / "src/mlx2").rglob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return {"source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": digest.hexdigest(), "root": str(ROOT),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def unit(spec):
    from mlx2.desktop_campaign_worker import check_ownership, http_json
    ownership = check_ownership(spec)
    import mlx.core as mx
    import mlx.nn as nn
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(1 << 30)
    from mlx2.adapters.registry import inspect_model

    out = Path(spec["out"])
    report = {**source(), "checkpoint": spec["checkpoint"], "qualified": False,
        "scope": "bounded functional probes", "ownership": ownership,
        "before": environment(), "passed": False, "device": mx.device_info()}
    server = None
    try:
        resolution = inspect_model(spec["model"])
        if resolution.artifact_fingerprint != spec["fingerprint"]:
            raise RuntimeError("artifact changed after inventory")
        # Ordinary reference precedes every candidate operation.
        for name in ("MLX2_MTP_SHARED_HISTORY", "MLX2_TILED_PREFILL", "MLX2_GDN_LIVE_SLOTS"):
            os.environ[name] = "0"
        adapter = resolution.adapter_type(spec["model"])
        model = adapter.model
        from mlx2.runtime.models import qwen38_shared_draft as shared
        from mlx2.runtime.lane import tiled_prefill as packed
        from mlx2.runtime.models import qwen38_tree_gdn as gdn
        from mlx2.runtime.models.cache import KVCache
        from mlx2.runtime.tensorfold_qwen38 import lane_qmm
        prompt = list(adapter.tokenizer.encode("Write a short Python function that adds two numbers.\n"))
        cache, tokens = model.make_cache(), []
        logits = model(mx.array([prompt]), cache=cache)
        for _ in range(12):
            token = int(mx.argmax(logits[:, -1]).item())
            tokens.append(token)
            logits = model(mx.array([[token]]), cache=cache)
        mx.eval(logits)
        report["ordinary_reference"] = {"tokens": tokens, "text": adapter.tokenizer.decode(tokens),
            "finite": bool(mx.all(mx.isfinite(logits)).item())}
        assert report["ordinary_reference"]["finite"]
        atomic(out / "progress.json", report)

        # Actual MTP weights: seed once, compare shared-prefix branches with
        # independent repeated-prefix branches for 4 steps. Keep logits at the
        # first differing argmax only; later margins are downstream evidence.
        text = model.language_model
        ids = list(adapter.tokenizer.encode("A cache revision binds model state. " * 1400))[:8193]
        hidden = text.model(mx.array([ids]), cache=model.make_cache())
        mtp = model.make_mtp_cache()
        _, seed = model.mtp_step(hidden[:, :-1], mx.array([ids[1:]]), mtp)
        mx.eval(seed, [c.state for c in mtp])
        before = [copy.deepcopy(c) for c in mtp]
        branches = 3
        candidate = [shared.SharedDraftKV(c, branches) for c in mtp]
        repeated = []
        for c in mtp:
            row = KVCache()
            row.keys = mx.repeat(c.keys[..., :c.offset, :], branches, axis=0)
            row.values = mx.repeat(c.values[..., :c.offset, :], branches, axis=0)
            row.offset = c.offset
            repeated.append(row)
        h = mx.repeat(seed[:, -1:], branches, axis=0)
        initial = mx.array([[ids[-1] + i] for i in range(branches)], mx.uint32)
        candidate_h, ordinary_h = h, h
        candidate_ids, ordinary_ids = initial, initial
        cells, first = [], None
        for step in range(4):
            a, candidate_h = model.mtp_step(candidate_h, candidate_ids, candidate)
            b, ordinary_h = model.mtp_step(ordinary_h, ordinary_ids, repeated)
            ai, bi = mx.argmax(a[:, -1], axis=-1), mx.argmax(b[:, -1], axis=-1)
            mx.eval(ai, bi)
            cell = {"step": step, "candidate": ai.tolist(), "reference": bi.tolist(),
                "max_abs_logit": float(mx.max(mx.abs(a.astype(mx.float32)-b.astype(mx.float32))).item())}
            if first is None and cell["candidate"] != cell["reference"]:
                first = {"step": step, "arms": []}
                for value in (a[:, -1], b[:, -1]):
                    ix = mx.argsort(value, axis=-1)[:, -8:][:, ::-1]
                    lp = value.astype(mx.float32) - mx.logsumexp(value.astype(mx.float32), axis=-1, keepdims=True)
                    first["arms"].append({"ids": ix.tolist(), "logprobs": mx.take_along_axis(lp, ix, -1).tolist()})
            cells.append(cell)
            candidate_ids, ordinary_ids = ai[:, None], bi[:, None]
        report["real_mtp"] = {"context_tokens": len(ids), "cells": cells, "first_divergence": first,
            "committed_offset_unchanged": all(a.offset == b.offset for a, b in zip(mtp, before)),
            "committed_values_unchanged": all(bool(mx.array_equal(x, y).item())
                for a, b in zip(mtp, before) for x, y in zip(a.state, b.state))}
        assert report["real_mtp"]["committed_values_unchanged"]
        del adapter, model, text, logits, hidden, seed, a, b, h, mtp, before, candidate, repeated, cache
        gc.collect()
        mx.clear_cache()

        attention_cells = []
        for length in (3, 257, 8193):
            for branches in (1, 3):
                q = (mx.random.normal((branches, 6, 1, 256)) * .1).astype(mx.bfloat16)
                c = KVCache()
                k = (mx.random.normal((1, 2, length, 256)) * .1).astype(mx.bfloat16)
                v = mx.random.normal(k.shape).astype(mx.bfloat16)
                c.update_and_fetch(k, v)
                private = shared.SharedDraftKV(c, branches)
                kt, vt = (mx.random.normal((branches, 2, 1, 256)).astype(mx.bfloat16) * .1 for _ in range(2))
                kt, vt = kt.astype(mx.bfloat16), vt.astype(mx.bfloat16)
                actual = private.append_and_attend(q, kt, vt, 1/16)
                keys = mx.concatenate((mx.repeat(k, branches, 0), kt), 2)
                values = mx.concatenate((mx.repeat(v, branches, 0), vt), 2)
                expected = mx.fast.scaled_dot_product_attention(q, keys, values, scale=1/16)
                err = float(mx.max(mx.abs(actual.astype(mx.float32)-expected.astype(mx.float32))).item())
                attention_cells.append({"context": length, "branches": branches, "max_abs": err})
                assert err < .01
        report["attention_components"] = attention_cells
        atomic(out / "progress.json", report)

        os.environ[packed.ENV] = "1"
        prefill_cells = []
        for k, n in ((512, 128), (1024, 192)):
            weights = (mx.random.normal((n, k)) * .02).astype(mx.bfloat16)
            w, scales, biases = mx.quantize(weights, group_size=64, bits=4)
            for nt in (0, 32, 64):
                module = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=4)
                module.weight, module.scales, module.biases = w if nt == 0 else lane_qmm.tile_weight(w, nt, bits=4), scales, biases
                for rows in (129, 200, 1024):
                    x = (mx.random.normal((1, rows, k)) * .5).astype(mx.bfloat16)
                    y = packed.project(x, module, nt)
                    reference = mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=64, bits=4)
                    exact = x.astype(mx.float32) @ mx.dequantize(w, scales, biases, group_size=64, bits=4).astype(mx.float32).T
                    error = lambda t: float(mx.max(mx.abs(t.astype(mx.float32) - exact)).item())
                    cell = {"k": k, "n": n, "rows": rows, "nt": nt,
                        "candidate_error": error(y), "reference_error": error(reference),
                        "bit_exact": bool(mx.array_equal(y, reference).item())}
                    prefill_cells.append(cell)
                    assert cell["candidate_error"] <= max(.002, cell["reference_error"] * 1.1)
        report["prefill_components"] = prefill_cells
        os.environ["MLX2_GDN_LIVE_SLOTS"] = "0"
        gdn_cells = []
        for parents in (list(range(-1, 15)), [-1] + [0] * 15, [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14]):
            rows, nk, nv, d = len(parents), 2, 4, 128
            q, k = (mx.random.normal((1, rows, nk, d)).astype(mx.bfloat16)*.02 for _ in range(2))
            v = mx.random.normal((1, rows, nv, d)).astype(mx.bfloat16)*.02
            gates, beta = mx.full((1, rows, nv), .97), mx.full((1, rows, nv), .3)
            states = [mx.random.normal((1, nv, d, d))*.01]
            plan = gdn.TreePlan([parents])
            for order in ("served", "fma"):
                os.environ["MLX2_GDN_LIVE_SLOTS"] = "0"
                expected = gdn.tree(q, k, v, gates, beta, states, plan, order=order)
                os.environ["MLX2_GDN_LIVE_SLOTS"] = "1"
                actual = gdn.tree(q, k, v, gates, beta, states, plan, order=order)
                equal = bool(mx.array_equal(actual, expected).item())
                gdn_cells.append({"parents": parents, "order": order, "slots": plan.compact_slots,
                    "previous_slots": plan.slots, "bit_exact": equal})
                assert equal
        report["gdn_components"] = gdn_cells
        report["component_counters"] = {"shared": shared.status(), "prefill": packed.status(), "gdn": gdn.stats()}
        atomic(out / "progress.json", report)
        gc.collect()
        mx.clear_cache()

        # One loopback server, the real packed native-MTP tree route, then
        # cold/warm replay with actual route counters and cache receipts.
        for name in (shared.ENV, packed.ENV, "MLX2_GDN_LIVE_SLOTS"):
            os.environ[name] = "1"
        atomic(out / "policy.json", {"draft_model": spec["model"], "num_draft": 5,
            "ast_topology": "3x5", "ast_tiles": 1})
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command = [sys.executable, "-m", "mlx2.server", "--model", spec["model"],
            "--host", "127.0.0.1", "--port", str(port), "--fabric", "off", "--max-lanes", "2",
            "--max-inflight", "2", "--max-context", "4096", "--prefill-step", "256",
            "--cache-bytes", str(512 << 20), "--cache-dir", str(out / "apcv2"),
            "--execution-policy", str(out / "policy.json"), "--external-draft", "--qualification-mode"]
        with (out / "server.log").open("w") as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError("server exited; see server.log")
                try:
                    status = http_json(base, "/v1/status")
                    if status.get("healthy"):
                        break
                except Exception:
                    pass
                time.sleep(.5)
            else:
                raise TimeoutError("server startup")
            report["status_before"] = status
            model_id = http_json(base, "/v1/models")["data"][0]["id"]
            request = {"model": model_id, "prompt": "A cache binds tokens and model revisions. " * 70 + "\nExplain why briefly:",
                "max_tokens": 32, "temperature": 0}
            report["cold"] = http_json(base, "/v1/completions", request)
            report["warm"] = http_json(base, "/v1/completions", request)
            from concurrent.futures import ThreadPoolExecutor
            import threading
            barrier = threading.Barrier(2)
            def concurrent_request(i):
                barrier.wait(timeout=10)
                return http_json(base, "/v1/completions", {**request,
                    "prompt": f"Case {i}. " + request["prompt"], "max_tokens": 48})
            with ThreadPoolExecutor(max_workers=2) as pool:
                report["concurrent"] = list(pool.map(concurrent_request, (1, 2)))
            report["status_after"] = http_json(base, "/v1/status")
        report["cold_warm_equal"] = report["cold"]["choices"][0]["text"] == report["warm"]["choices"][0]["text"]
        report["cached_tokens"] = report["warm"].get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
        engagement = report["status_after"].get("execution", {}).get("tree_spec_intake", {})
        report["serving_engagement"] = {
            "shared_history": engagement.get("shared_draft", {}).get("metal_calls", 0),
            "direct_prefill": engagement.get("tiled_prefill", {}).get("calls", 0),
            "compact_gdn": engagement.get("tree_gdn", {}).get("compact_tree_calls", 0),
        }
        report["passed"] = (report["cold_warm_equal"] and report["cached_tokens"] > 0
            and all(report["serving_engagement"].values()))
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
    finally:
        if server is not None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        report["after"] = environment()
        report["source_after"] = source()
        atomic(out / "result.json", report)
    print(json.dumps({"passed": report["passed"], "error": report.get("error"), "out": str(out)}), flush=True)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.mode == "unit":
        return unit(spec)
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    out = Path(spec["out"])
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    with DesktopCPGLease(out, spec["job_id"], checkpoint=spec["checkpoint"], source_revision=spec["source_revision"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"), "--session", lease.owner["cpg_session"],
            "--label", spec["job_id"], "--desktop-cpg-lease", str(lease.path), "--receipt", str(out / "paired-locks.json"),
            "--", sys.executable, str(Path(__file__)), "unit", str(args.spec.resolve())]
        child = subprocess.Popen(command, start_new_session=True)
        deadline = time.monotonic() + 1800
        try:
            while child.poll() is None:
                lease.check()
                if time.monotonic() > deadline:
                    raise TimeoutError("bounded artifact ownership window expired")
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
