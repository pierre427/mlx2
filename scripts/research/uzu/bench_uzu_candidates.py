#!/usr/bin/env python3
"""Source-bound, single-checkpoint A/B performance experiment; no qualification.

Run under cpg_job and run_with_gpu_locks. Fresh-cache full-model prefill isolates
fusion from the existing INT8 route; QMV screening and held-out comparisons use
rotating real checkpoint weights. One loopback candidate sanity check closes
the artifact ownership window. Every timed result is wall time to GPU completion.
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
import random
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from probe_uzu_checkpoint import POLICY
from probe_uzu_kernels import atomic, swapouts
from run_single_model_smoke_ladder import stop_server, thermal_sample, wait_health


def host_load():
    hits = []
    for line in subprocess.check_output(["ps", "-Ao", "pcpu=,pid=,command="], text=True).splitlines():
        fields = line.split(None, 2)
        if len(fields) != 3 or int(fields[1]) == os.getpid():
            continue
        cpu, command = float(fields[0]), fields[2]
        if cpu >= 60 or ("pytest" in command and cpu >= 5):
            hits.append({"pid": int(fields[1]), "cpu": cpu, "command": command[:140]})
    return hits


def paired_summary(baseline, candidate):
    """Unadjusted paired bootstrap interval; retained raw pairs are authoritative."""
    ratios = [a / b for a, b in zip(baseline, candidate)]
    rng = random.Random(1027)
    draws = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(5000))
    return {"baseline_median_seconds": statistics.median(baseline),
            "candidate_median_seconds": statistics.median(candidate),
            "paired_median_speedup": statistics.median(ratios),
            "paired_speedup_bootstrap95": [draws[125], draws[4874]],
            "pairs": len(ratios), "interval_scope": "per-cell unadjusted paired median bootstrap"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=9)
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--step", type=int, default=2048)
    parser.add_argument("--background-cpu-policy", choices=("reject", "record"), default="reject")
    parser.add_argument("--skip-prefill", action="store_true", help="QMV-only bounded window after a separate prefill experiment")
    parser.add_argument("--fixed-qmv", type=Path, help="independent confirmation mapping role:qBITS:mROWS to a previously selected geometry")
    parser.add_argument("--qmv-baseline", default="default", help="named row/RPS control to isolate output tiling after row grouping")
    args = parser.parse_args()
    if args.reps < 6:
        parser.error("at least six balanced repetitions required")
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError("refusing to overwrite a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)
    from qualification_gpu_ownership import require_qualification_lease
    from run_with_gpu_locks import foreign_model_processes

    owner = json.loads((args.cpg_owner_lock / "owner.json").read_text())
    ownership = require_qualification_lease(task_id=owner["cpg_task"], generation=owner["cpg_generation"],
                                          cpg_owner_lock=args.cpg_owner_lock)
    if foreign_model_processes():
        raise RuntimeError("foreign model process contamination")
    initial_thermal = thermal_sample()
    initial_load = host_load()
    if initial_thermal["thermal_state"] != 0 or (initial_load and args.background_cpu_policy == "reject"):
        raise RuntimeError("performance admission requires nominal thermals and a quiet host")
    root = Path(__file__).resolve().parents[1]
    paths = ["src/mlx2/runtime/fused_mlp_stage_a.py", "src/mlx2/runtime/int8_prefill.py",
             "src/mlx2/runtime/models/row_exact_qmv.py", "src/mlx2/runtime/models/qwen3_next.py",
             "scripts/bench_uzu_candidates.py", "scripts/probe_uzu_checkpoint.py"]
    result = {"schema": "mlx2.uzu-performance.v1", "source_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), "source_dirty": True,
              "source_files": {}, "ownership": ownership, "thermal_start": initial_thermal,
              "status": "running", "qualification": False, "prefill": [], "qmv": [],
              "model_path": str(args.model), "reps": args.reps, "step": args.step,
              "background_cpu_policy": args.background_cpu_policy, "initial_foreign_cpu": initial_load,
              "prefill_skipped": args.skip_prefill,
              "fixed_qmv": None if args.fixed_qmv is None else json.loads(args.fixed_qmv.read_text()),
              "qmv_baseline": args.qmv_baseline,
              "measurement": "wall time through evaluated outputs/state plus mx.synchronize; excludes load, policy install and compilation",
              "qmv_scope": "explicit component API; not a serving dispatch or model throughput claim",
              "prefill_scope": "direct one-stream fresh-cache model; last-position vocab projection; no APC hits; not HTTP TTFT"}
    result["prefill_clock_control"] = "one untimed same-arm prefill immediately before each timed prefill; steady-state GPU, not idle wake-up latency"
    for path in paths:
        data = (root / path).read_bytes()
        result["source_files"][path] = hashlib.sha256(data).hexdigest()
        target = args.out / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    atomic(args.out / "result.json", result)
    swap0 = swapouts()
    adapter = None
    server = None
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from smoke_local_model import _ordinary_generate

        from mlx2.adapters.registry import inspect_model

        resolution = inspect_model(args.model)
        identity = resolution.artifact.get("identity", resolution.artifact)
        result["checkpoint"] = args.model.name + "@" + identity["fingerprint"]
        result["family"] = resolution.descriptor.family
        adapter = resolution.adapter_type(str(args.model))
        from mlx2.runtime import int8_prefill as ip
        from mlx2.runtime.models import row_exact_qmv as qmv
        from mlx2.runtime.models.cache import make_prompt_cache
        from mlx2.runtime.models.qwen3_next import Qwen3NextMLP

        model = adapter.model
        mx.eval(model.parameters())
        mx.set_cache_limit(6 << 30)
        mx.random.seed(1027)
        result["seed"] = 1027
        result["mlx_version"] = mx.__version__
        result["device"] = mx.device_info().get("device_name")
        result["ordinary_reference"] = _ordinary_generate(adapter, "Say hello in one sentence.", 16)
        policies = {"ordinary": None, "unfused_int8": POLICY, "fused_int8": {**POLICY, "fused_mlp_stage_a": True}}
        ids = list(adapter.prompt_tokens({"prompt":
            "A careful engineer measures the system, records exact results, and checks the evidence. " * 3000}))
        if len(ids) < max(args.contexts):
            raise RuntimeError("insufficient prompt tokens")
        result["prompt_tokens_sha256"] = hashlib.sha256(json.dumps(ids[:max(args.contexts)]).encode()).hexdigest()

        def save():
            atomic(args.out / "result.json", result)

        def arrays(cache):
            return [v for _, v in tree_flatten([c.state for c in cache]) if isinstance(v, mx.array)]

        def environment():
            thermal = thermal_sample()
            load = host_load()
            swap = swapouts() - swap0
            return {"thermal": thermal, "foreign_cpu": load, "swapouts_delta": swap}

        def clean(env):
            return (env["thermal"]["thermal_state"] == 0 and not env["swapouts_delta"]
                    and (args.background_cpu_policy == "record" or not env["foreign_cpu"]))

        def check_environment():
            env = environment()
            if not clean(env):
                raise RuntimeError(f"performance contamination: {env}")
            return env

        def install(name):
            handle = None if policies[name] is None else ip.apply_for_adapter(adapter, policies[name])
            if handle is not None:
                handle.warmup()
            return handle

        def prefill(length):
            cache = make_prompt_cache(model)
            tokens = mx.array(ids[:length])[None]
            mx.eval(tokens)
            mx.synchronize()
            start = time.perf_counter()
            for offset in range(0, length, args.step):
                hidden = model.model(tokens[:, offset:offset + args.step], cache=cache)
                if offset + args.step >= length:
                    output = model.logits(hidden[:, -1:])
                else:
                    output = hidden[:, -1:]
                mx.eval(output, arrays(cache))
            mx.synchronize()
            elapsed = time.perf_counter() - start
            finite = bool(mx.all(mx.isfinite(output)).item())
            top = int(mx.argmax(output[0, -1]).item())
            del cache, hidden, output
            if not finite:
                raise AssertionError("nonfinite timed prefill output")
            return elapsed, top

        # Warm all kernels/metadata before measurements. Each length is a
        # separate profile; arm order rotates in balanced three-way blocks.
        for length in ([] if args.skip_prefill else args.contexts):
            print(f"prefill warmup context={length}", flush=True)
            for name in policies:
                handle = install(name)
                try:
                    prefill(length)
                finally:
                    ip.remove(handle)
            cell = {"context": length, "step": args.step, "samples": [], "discarded": [], "summary": {}}
            result["prefill"].append(cell)
            times = {name: [] for name in policies}
            attempts = 0
            consecutive_rejects = 0
            while len(cell["samples"]) < args.reps:
                rep = len(cell["samples"])
                attempts += 1
                if attempts > args.reps * 3 or consecutive_rejects >= 6:
                    raise RuntimeError("bounded retry budget exhausted by performance contamination")
                time.sleep(2)
                before = environment()
                if not clean(before):
                    cell["discarded"].append({"attempt": attempts, "before": before})
                    consecutive_rejects += 1
                    save()
                    continue
                names = list(policies)
                names = names[rep % 3:] + names[:rep % 3]
                if (rep // 3) % 2:
                    names.reverse()
                sample = {"rep": rep, "order": names, "before": before, "arms": {}}
                for name in names:
                    handle = install(name)
                    try:
                        warmup_elapsed, _ = prefill(length)
                        elapsed, top = prefill(length)
                        sample["arms"][name] = {"seconds": elapsed, "tokens_per_second": length / elapsed,
                                                "untimed_warmup_seconds": warmup_elapsed,
                                                "last_top_token": top,
                                                "receipt": None if handle is None else handle.receipt()}
                        if handle is not None and not handle.counts["engaged_calls"]:
                            raise AssertionError("timed INT8 arm did not engage")
                        if name == "fused_int8" and not handle.counts["fused_mlp_stage_a_calls"]:
                            raise AssertionError("timed fusion did not engage")
                    finally:
                        ip.remove(handle)
                sample["after"] = environment()
                sample["quiet_cpu_boundaries"] = not before["foreign_cpu"] and not sample["after"]["foreign_cpu"]
                if not clean(sample["after"]):
                    sample["attempt"] = attempts
                    cell["discarded"].append(sample)
                    consecutive_rejects += 1
                    save()
                    continue
                if sample["arms"]["unfused_int8"]["last_top_token"] != sample["arms"]["fused_int8"]["last_top_token"]:
                    raise AssertionError("timed fusion changed final argmax")
                cell["samples"].append(sample)
                for name in policies:
                    times[name].append(sample["arms"][name]["seconds"])
                consecutive_rejects = 0
                save()
                print(f"prefill context={length} rep={rep + 1}/{args.reps}", flush=True)
            cell["summary"] = {"fusion_vs_unfused": paired_summary(times["unfused_int8"], times["fused_int8"]),
                               "fusion_vs_ordinary": paired_summary(times["ordinary"], times["fused_int8"]),
                               "unfused_vs_ordinary": paired_summary(times["ordinary"], times["unfused_int8"])}
            quiet = [s for s in cell["samples"] if s["quiet_cpu_boundaries"]]
            cell["quiet_cpu_pairs"] = len(quiet)
            cell["quiet_cpu_fusion_vs_unfused"] = paired_summary(
                [s["arms"]["unfused_int8"]["seconds"] for s in quiet],
                [s["arms"]["fused_int8"]["seconds"] for s in quiet]) if len(quiet) >= 3 else None
            save()

        # Screen on cache-resistant pools of actual target weights, grouped by
        # role/precision. Different allocations and input tensors prevent CSE.
        groups = {}
        for path, mlp in model.named_modules():
            if type(mlp) is not Qwen3NextMLP or ".mtp." in path:
                continue
            for role in ("gate_proj", "up_proj", "down_proj"):
                linear = getattr(mlp, role)
                if qmv.layout_decline_reason(linear) is None:
                    groups.setdefault((role, linear.bits), []).append((path + "." + role, linear))
        for (role, bits), members in sorted(groups.items()):
            pool = []
            packed_bytes = 0
            for path, linear in members:
                pool.append((path, linear))
                packed_bytes += linear.weight.nbytes
                if packed_bytes >= 256 << 20:
                    break
            for rows in (2, 3, 8):
                cell_key = f"{role}:q{bits}:m{rows}"
                if result["fixed_qmv"] is not None and cell_key not in result["fixed_qmv"]:
                    continue
                time.sleep(1)
                before = check_environment()
                inputs = []
                for _, linear in pool:
                    k = linear.weight.shape[-1] * 32 // bits
                    inputs.append(mx.random.normal((rows, k)).astype(linear.scales.dtype))
                mx.eval(inputs)
                first = pool[0][1]
                k, n = inputs[0].shape[-1], first.weight.shape[0]
                fast = qmv.qmv_fast_layout(k, n, bits)
                candidates = {"default": None}
                for sharing in range(1, min(rows, qmv.row_group_limit(bits, fast)) + 1):
                    for rps in (1, 2, 4, 8):
                        candidates[f"rows{sharing}_rps{rps}"] = qmv.TileGeometry(sharing, rps)

                def run(name, _candidates=candidates, _pool=pool, _inputs=inputs):
                    outputs = [qmv.quantized_linear(linear, x, geometry=_candidates[name])[0]
                               for (_, linear), x in zip(_pool, _inputs)]
                    mx.eval(outputs)
                    mx.synchronize()
                    return outputs

                def measure(name, _run=run):
                    mx.synchronize()
                    start = time.perf_counter()
                    outputs = _run(name)
                    elapsed = time.perf_counter() - start
                    del outputs
                    return elapsed

                reference = run("default")
                for name in candidates:
                    output = run(name)
                    if not all(bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item())
                               for a, b in zip(output, reference)):
                        raise AssertionError("timed QMV geometry changed output bits")
                del output, reference
                screen = {name: [] for name in candidates}
                for rep in range(3):
                    names = list(candidates)
                    random.Random(1027 + rep).shuffle(names)
                    for name in names:
                        screen[name].append(measure(name))
                alternative = min((name for name in candidates if name != "default"),
                                  key=lambda name: statistics.median(screen[name]))
                if result["fixed_qmv"] is not None:
                    alternative = result["fixed_qmv"][cell_key]
                baseline = args.qmv_baseline
                if baseline not in candidates or baseline == alternative:
                    raise ValueError("QMV control must exist and differ from the candidate")
                cell = {"role": role, "bits": bits, "rows": rows, "n": n, "k": k,
                        "group_size": first.group_size, "dtype": str(inputs[0].dtype),
                        "pool": [path for path, _ in pool], "packed_weight_bytes": packed_bytes,
                        "cache_scope": "rotating distinct checkpoint weights; not proof of DRAM coldness",
                        "screen_seconds": screen, "selected_for_holdout": alternative,
                        "selection": "fixed from independent window" if result["fixed_qmv"] is not None else "fastest alternate in this screen",
                        "baseline_geometry": baseline,
                        "before": before, "holdout": []}
                base_times, alt_times = [], []
                for rep in range(args.reps):
                    names = [baseline, alternative] if rep % 2 == 0 else [alternative, baseline]
                    timed = {name: measure(name) for name in names}
                    base_times.append(timed[baseline])
                    alt_times.append(timed[alternative])
                    cell["holdout"].append({"rep": rep, "order": names, "seconds": timed})
                cell["after"] = check_environment()
                cell["quiet_cpu_boundaries"] = not before["foreign_cpu"] and not cell["after"]["foreign_cpu"]
                cell["summary"] = paired_summary(base_times, alt_times)
                result["qmv"].append(cell)
                save()
                print(f"QMV {role} b{bits} M{rows}: {alternative}, speedup={cell['summary']['paired_median_speedup']:.3f}", flush=True)
        groups.clear()
        pool.clear()
        inputs.clear()
        model = first = linear = mlp = members = None
        adapter.close()
        adapter = None
        gc.collect()
        mx.clear_cache()

        # Functional serving witness, deliberately outside performance timing.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        command = [sys.executable, str(root / "scripts/probe_uzu_checkpoint.py"), "--research-server",
                   "--model", str(args.model), "--host", "127.0.0.1", "--port", str(port),
                   "--ordinary", "--qualification-mode", "--max-lanes", "1", "--max-inflight", "1",
                   "--max-context", "4096", "--cache-bytes", str(1 << 30)]
        with (args.out / "server.log").open("w") as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            result["server_startup_status"] = wait_health(base, server, 240)
            with urllib.request.urlopen(base + "/v1/models", timeout=20) as response:
                served = json.load(response)["data"][0]["id"]
            bodies = []
            for arm in ("cold", "warm"):
                request = {"model": served, "messages": [{"role": "user", "content":
                           "The experiment code is UZU-7492. " * 24 + "Reply with only the experiment code."}],
                           "enable_thinking": False, "temperature": 0, "max_tokens": 32, "return_token_trace": True}
                req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(request).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as response:
                    body = json.load(response)
                atomic(args.out / f"http-{arm}.json", body)
                bodies.append(body)
            if bodies[0]["mlx2"]["token_trace"]["token_ids"] != bodies[1]["mlx2"]["token_trace"]["token_ids"]:
                raise AssertionError("post-benchmark HTTP exact replay failed")
            if not all(body["choices"][0]["message"]["content"].strip() == "UZU-7492" for body in bodies):
                raise AssertionError("post-benchmark functional oracle failed")
            if not bodies[1]["mlx2"]["cached_tokens"]:
                raise AssertionError("post-benchmark APCv2 reuse not observed")
            result["server_witness"] = {"exact_tokens": True, "functional_oracle": True,
                                        "warm_reused_tokens": bodies[1]["mlx2"]["cached_tokens"]}
        result["status"] = "passed"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop_server(server)
        if adapter is not None:
            adapter.close()
        result["server_stopped"] = server is None or server.poll() is not None
        result["thermal_end"] = thermal_sample()
        result["swapouts_delta"] = swapouts() - swap0
        result["finished_at"] = time.time()
        atomic(args.out / "result.json", result)


if __name__ == "__main__":
    main()
