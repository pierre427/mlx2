#!/usr/bin/env python3
"""One-checkpoint Uzu diagnostic under an existing live CPG/paired-lock window.

Ordinary reference, checkpoint-weight QMV probes, fused/unfused INT8 parity,
then one loopback candidate server. This is neither preflight nor qualification.
"""

from __future__ import annotations

import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[3] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import argparse
import gc
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from probe_uzu_kernels import atomic, swapouts

POLICY = {"enabled": True, "scope": "mlp", "row_threshold": 32,
          "q45_inplace": True, "inplace_only": True, "act_scale": "group64"}


def server_main():
    from mlx2 import server

    original = server.serving_engine_kwargs

    def kwargs(*args, **kw):
        result = original(*args, **kw)
        result["int8_prefill"] = {**POLICY, "fused_mlp_stage_a": True}
        return result

    server.serving_engine_kwargs = kwargs
    sys.argv.remove("--research-server")
    server.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError("refusing to overwrite nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)
    from qualification_gpu_ownership import require_qualification_lease
    from run_single_model_smoke_ladder import stop_server, thermal_sample, wait_health
    from run_with_gpu_locks import foreign_model_processes

    owner = json.loads((args.cpg_owner_lock / "owner.json").read_text())
    ownership = require_qualification_lease(task_id=owner["cpg_task"],
                                          generation=owner["cpg_generation"],
                                          cpg_owner_lock=args.cpg_owner_lock)
    if foreign_model_processes():
        raise RuntimeError("foreign model process contamination")
    thermal = thermal_sample()
    if thermal["thermal_state"] not in (0, 1):
        raise RuntimeError("thermal admission refused")
    root = Path(__file__).resolve().parents[1]
    files = ["src/mlx2/runtime/fused_mlp_stage_a.py", "src/mlx2/runtime/int8_prefill.py",
             "src/mlx2/runtime/models/row_exact_qmv.py", "src/mlx2/runtime/models/qwen3_next.py",
             "scripts/probe_uzu_checkpoint.py"]
    result = {"schema": "mlx2.uzu-checkpoint-probe.v1", "model_path": str(args.model),
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "source_dirty": True, "source_files": {}, "qualification": False,
              "ownership": ownership, "thermal_start": thermal, "cells": [], "status": "running",
              "profile": "B1; ordinary decode; INT8 Q4/Q5 group64; threshold32; APCv2 1GiB; contexts512/1024"}
    for name in files:
        source = (root / name).read_bytes()
        result["source_files"][name] = hashlib.sha256(source).hexdigest()
        target = args.out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
    atomic(args.out / "result.json", result)
    swap0 = swapouts()
    server_process = None
    adapter = None
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from smoke_local_model import _ordinary_generate

        from mlx2.adapters.registry import inspect_model

        resolution = inspect_model(args.model)
        identity = resolution.artifact.get("identity", resolution.artifact)
        result["checkpoint"] = args.model.name + "@" + identity["fingerprint"]
        result["family"] = resolution.descriptor.family
        result["mlx_version"] = mx.__version__
        result["device"] = mx.device_info().get("device_name")
        atomic(args.out / "result.json", result)
        adapter = resolution.adapter_type(str(args.model))
        # Adapter construction pins import-time numerics before model modules.
        from mlx2.runtime import int8_prefill as ip
        from mlx2.runtime.models import row_exact_qmv as qmv
        from mlx2.runtime.models.activations import swiglu
        from mlx2.runtime.models.cache import make_prompt_cache
        from mlx2.runtime.models.qwen3_next import Qwen3NextMLP

        result["ordinary_reference"] = _ordinary_generate(adapter, "Reply with hello in one sentence.", 16)
        atomic(args.out / "result.json", result)
        print("ordinary reference passed", flush=True)

        def equal(a, b):
            if a.shape != b.shape or a.dtype != b.dtype:
                return False
            if a.dtype in (mx.bfloat16, mx.float16):
                a, b = a.view(mx.uint16), b.view(mx.uint16)
            elif a.dtype == mx.float32:
                a, b = a.view(mx.uint32), b.view(mx.uint32)
            return bool(mx.array_equal(a, b).item())

        def states(cache):
            return [v for _, v in tree_flatten([c.state for c in cache]) if isinstance(v, mx.array)]

        def record(cell):
            result["cells"].append(cell)
            atomic(args.out / "result.json", result)

        prompt = "A careful engineer checks exact arithmetic and records the evidence. " * 180
        ids = list(adapter.prompt_tokens({"prompt": prompt}))
        if len(ids) < 1024:
            raise RuntimeError("diagnostic prompt unexpectedly short")
        # Native MTP owns an auxiliary MLP that ordinary decode never visits.
        modules = [(p, m) for p, m in adapter.model.named_modules()
                   if type(m) is Qwen3NextMLP and ".mtp." not in p]
        chosen = [modules[i] for i in (0, len(modules) // 2, len(modules) - 1)]
        captured = {}
        originals = []
        try:
            for path, mlp in chosen:
                base = type(mlp)

                def capture(self, x, *a, _path=path, _base=base, **kw):
                    captured[_path] = x
                    return _base.__call__(self, x, *a, **kw)

                originals.append((mlp, base))
                mlp.__class__ = type("UzuDiagnosticCapture", (base,), {"__call__": capture})
            cache = make_prompt_cache(adapter.model)
            hidden = adapter.model.model(mx.array(ids[:512])[None], cache=cache)
            mx.eval(hidden, states(cache), list(captured.values()))
        finally:
            for mlp, base in originals:
                mlp.__class__ = base
        del cache, hidden
        for path, mlp in chosen:
            x = captured[path][0, :3]
            down_input = swiglu(mlp.gate_proj(x), mlp.up_proj(x))
            for name, inputs in (("gate_proj", x), ("up_proj", x), ("down_proj", down_input)):
                linear = getattr(mlp, name)
                reason = qmv.decline_reason(linear, inputs)
                if reason is not None:
                    record({"kind": "qmv_checkpoint", "path": path + "." + name, "declined": reason})
                    continue
                reference = qmv.per_row(linear, inputs)
                fast = qmv.qmv_fast_layout(inputs.shape[-1], linear.weight.shape[0], linear.bits)
                for sharing in range(1, min(3, qmv.row_group_limit(linear.bits, fast)) + 1):
                    for rps in (1, 2, 4, 8):
                        got, route = qmv.quantized_linear(linear, inputs, geometry=qmv.TileGeometry(sharing, rps))
                        exact = equal(got, reference)
                        record({"kind": "qmv_checkpoint", "path": path + "." + name,
                                "shape": [3, linear.weight.shape[0], inputs.shape[-1]],
                                "bits": linear.bits, "group_size": linear.group_size,
                                "rows_per_group": sharing, "results_per_simdgroup": rps,
                                "route": route, "exact": exact})
                        if not exact:
                            raise AssertionError("checkpoint QMV changed ordinary output bits")
        captured.clear()
        del down_input, x, reference, got
        print("checkpoint QMV probes passed", flush=True)

        def forward_arm(length, fused):
            handle = ip.apply_for_adapter(adapter, {**POLICY, "fused_mlp_stage_a": fused})
            try:
                cache = make_prompt_cache(adapter.model)
                hidden = adapter.model.model(mx.array(ids[:length])[None], cache=cache)
                logits = adapter.model.logits(hidden[:, -1:])[0, -1]
                mx.eval(logits, states(cache))
                initial_logits = logits
                tokens = []
                for _ in range(8):
                    token = int(mx.argmax(logits).item())
                    tokens.append(token)
                    logits = adapter.model(mx.array([[token]]), cache=cache)[0, -1]
                    mx.eval(logits, states(cache))
                return initial_logits, tokens, cache, handle.receipt()
            finally:
                ip.remove(handle)

        for length in (512, 1024):
            before = thermal_sample()
            if before["thermal_state"] not in (0, 1):
                raise RuntimeError("thermal cell admission refused")
            a, ta, ca, ra = forward_arm(length, False)
            b, tb, cb, rb = forward_arm(length, True)
            sa, sb = states(ca), states(cb)
            exact_states = len(sa) == len(sb) and all(equal(x, y) for x, y in zip(sa, sb))
            cell = {"kind": "whole_model_fusion", "prompt_tokens": length, "decode_tokens": 8,
                    "exact_initial_logits": equal(a, b), "exact_tokens": ta == tb,
                    "exact_final_state_arrays": exact_states, "state_arrays": len(sa),
                    "unfused_token_ids": ta, "fused_token_ids": tb,
                    "unfused_receipt": ra, "fused_receipt": rb, "thermal_before": before,
                    "thermal_after": thermal_sample()}
            record(cell)
            if not all(cell[key] for key in ("exact_initial_logits", "exact_tokens", "exact_final_state_arrays")):
                raise AssertionError("full-model fusion/unfused parity failed")
            if not rb["fused_mlp_stage_a"]["calls"]:
                raise AssertionError("full-model fusion was not observed")
            del a, b, ca, cb, sa, sb
            print(f"whole-model {length}-token parity passed", flush=True)
        adapter.close()
        adapter = None
        del mlp, linear, chosen, modules, originals
        gc.collect()
        mx.clear_cache()

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        command = [sys.executable, str(Path(__file__).resolve()), "--research-server", "--model", str(args.model),
                   "--host", "127.0.0.1", "--port", str(port), "--ordinary", "--qualification-mode",
                   "--max-lanes", "1", "--max-inflight", "1", "--max-context", "4096",
                   "--cache-bytes", str(1 << 30)]
        result["server_command"] = command
        result["server_policy"] = {**POLICY, "fused_mlp_stage_a": True}
        with (args.out / "server.log").open("w") as log:
            server_process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=True, env=os.environ.copy())
            result["server_startup_status"] = wait_health(base, server_process, 240)
            with urllib.request.urlopen(base + "/v1/models", timeout=20) as response:
                served_models = json.load(response)["data"]
            if len(served_models) != 1:
                raise RuntimeError("expected exactly one served target")
            served_model = served_models[0]["id"]
            result["served_model_id"] = served_model
            atomic(args.out / "result.json", result)
            for repeat in (24, 64):
                before = thermal_sample()
                if before["thermal_state"] not in (0, 1):
                    raise RuntimeError("HTTP thermal admission refused")
                request = {"model": served_model, "messages": [{"role": "user", "content":
                           ("The project code is UZU-7492. " * repeat) +
                           "Reply with only the project code."}], "enable_thinking": False,
                           "temperature": 0, "max_tokens": 32, "return_token_trace": True}
                responses = []
                for arm in ("cold", "warm"):
                    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(request).encode(),
                                                 headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=180) as response:
                        body = json.load(response)
                    atomic(args.out / f"http-{repeat}-{arm}.json", body)
                    responses.append(body)
                # Traces are authoritative sampled IDs, never re-tokenized rendered text.
                from mlx2.generation_trace import read_token_trace

                def route(body):
                    for value in (body.get("mlx2"), body.get("mlx2_route"), body.get("route")):
                        if isinstance(value, dict) and "token_trace" in value:
                            return value
                    raise AssertionError("HTTP response lacks authoritative token trace")

                routes = [route(body) for body in responses]
                traces = [read_token_trace(r) for r in routes]
                texts = [body["choices"][0]["message"]["content"] for body in responses]
                cell = {"kind": "http_cold_warm", "prompt_repetitions": repeat,
                        "exact_tokens": traces[0]["token_ids"] == traces[1]["token_ids"],
                        "functional_oracle": [text.strip() == "UZU-7492" for text in texts],
                        "warm_reused_tokens": routes[1].get("cached_tokens", 0),
                        "fused_calls": [r.get("int8_prefill", {}).get("fused_mlp_stage_a", {}).get("calls", 0)
                                        for r in routes],
                        "usage": [body["usage"] for body in responses], "routes": routes,
                        "thermal_before": before, "thermal_after": thermal_sample()}
                record(cell)
                if not cell["exact_tokens"] or not all(cell["functional_oracle"]):
                    raise AssertionError("HTTP B1 exact replay or functional oracle failed")
                if not cell["warm_reused_tokens"] or not all(cell["fused_calls"]):
                    raise AssertionError("APCv2 reuse or served fusion was not observed")
                print(f"HTTP cold/warm {repeat} passed", flush=True)
            with urllib.request.urlopen(base + "/v1/status", timeout=20) as response:
                result["server_final_status"] = json.load(response)
        result["status"] = "passed"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop_server(server_process)
        if adapter is not None:
            adapter.close()
        result["server_stopped"] = server_process is None or server_process.poll() is not None
        result["thermal_end"] = thermal_sample()
        result["swapouts_delta"] = swapouts() - swap0
        result["finished_at"] = time.time()
        atomic(args.out / "result.json", result)


if __name__ == "__main__":
    if "--research-server" in sys.argv:
        server_main()
    else:
        main()
