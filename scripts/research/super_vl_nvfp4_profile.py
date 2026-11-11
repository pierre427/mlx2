#!/usr/bin/env python3
"""Nemotron 3.5 Super VL NVFP4 export: decode / prefill attribution profile.

``owned OUT`` takes a desktop CPG lease and the paired GPU locks and runs
``unit OUT`` under them.  The unit, in one process:

1. ``kernels``: synthetic weights at this model's real projection shapes.
   Dense Mamba in_proj [18560, 4096] / out_proj [4096, 8192] as BF16 matmul,
   ``mxfp8`` (E4M3, unit block scales, the vendor FP8 law), affine8 g64 and
   NVFP4 quantized matmul; routed experts [512, 2688, 1024] (fc1) and
   [512, 1024, 2688] (fc2) as NVFP4, affine4 g64 and mxfp4 sorted
   ``gather_qmm`` plus decoded-BF16 ``gather_mm``; rows M = 1, 8, 128, 2048.
2. ``model``: loads the served adapter (``--ordinary`` composition), times
   uninstrumented prefill (2048-row chunks) at the requested contexts and
   B1 decode, then an exclusive-time split per component, with ``mx.eval``
   at module boundaries (absolute times include the sync floor it reports).

Attribution data without thermal holds: not publishable statistics, never
qualification.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
TARGET = Path.home() / "mlx-models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def timed(fn, reps, warmup=2):
    import mlx.core as mx

    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    for _ in range(reps):
        mx.synchronize()
        started = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def kernels(reps):
    import mlx.core as mx

    out = {"dense": [], "routed": []}
    for name, (n, k) in {"mamba_in_proj": (18560, 4096), "mamba_out_proj": (4096, 8192)}.items():
        weight = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
        forms = {"bf16": None}
        for mode, gs, bits in (("mxfp8", 32, 8), ("affine8", 64, 8), ("nvfp4", 16, 4)):
            forms[mode] = mx.quantize(weight, group_size=gs, bits=bits,
                                      mode="affine" if mode.startswith("affine") else mode)
        mx.eval(weight, forms)
        for rows in (1, 8, 128, 2048):
            x = mx.random.normal((rows, k)).astype(mx.bfloat16)
            row = {"projection": name, "rows": rows}
            for mode, q in forms.items():
                if q is None:
                    fn = lambda: x @ weight.T  # noqa: E731
                    nbytes = weight.nbytes
                else:
                    gs, bits, m = {"mxfp8": (32, 8, "mxfp8"), "affine8": (64, 8, "affine"),
                                   "nvfp4": (16, 4, "nvfp4")}[mode]
                    fn = (lambda q=q, gs=gs, bits=bits, m=m: mx.quantized_matmul(
                        x, q[0], q[1], q[2] if len(q) > 2 else None, transpose=True,
                        group_size=gs, bits=bits, mode=m))
                    nbytes = sum(a.nbytes for a in q)
                seconds = timed(fn, reps)
                row[mode] = {"ms": round(seconds * 1e3, 4),
                             "weight_gbs": round(nbytes / seconds / 1e9, 1)}
            out["dense"].append(row)
            print(json.dumps(row), flush=True)
        del weight, forms
        mx.clear_cache()
    for name, (n, k) in {"routed_fc1": (2688, 1024), "routed_fc2": (1024, 2688)}.items():
        weight = (mx.random.normal((512, n, k)) * 0.02).astype(mx.bfloat16)
        forms = {}
        for mode, gs, bits in (("nvfp4", 16, 4), ("affine4", 64, 4), ("mxfp4", 32, 4)):
            forms[mode] = mx.quantize(weight, group_size=gs, bits=bits,
                                      mode="affine" if mode.startswith("affine") else mode)
        mx.eval(forms)
        for tokens in (1, 8, 128, 2048):
            rows = tokens * 22
            indices = mx.sort(mx.random.randint(0, 512, (rows,)).astype(mx.uint32))
            x = mx.random.normal((rows, 1, k)).astype(mx.bfloat16)
            row = {"projection": name, "tokens": tokens, "rows": rows}
            for mode, q in forms.items():
                gs, bits, m = {"nvfp4": (16, 4, "nvfp4"), "affine4": (64, 4, "affine"),
                               "mxfp4": (32, 4, "mxfp4")}[mode]
                fn = (lambda q=q, gs=gs, bits=bits, m=m: mx.gather_qmm(
                    x, q[0], q[1], q[2] if len(q) > 2 else None, rhs_indices=indices,
                    transpose=True, group_size=gs, bits=bits, mode=m, sorted_indices=True))
                row[mode] = {"ms": round(timed(fn, reps) * 1e3, 4)}
            fn = lambda: mx.gather_mm(x, weight.swapaxes(-1, -2), rhs_indices=indices,  # noqa: E731
                                      sorted_indices=True)
            row["bf16_gather_mm"] = {"ms": round(timed(fn, reps) * 1e3, 4)}
            out["routed"].append(row)
            print(json.dumps(row), flush=True)
        del weight, forms
        mx.clear_cache()
    return out


def rowexact_check(reps):
    """fp_row_exact_qmv vs stock one-row calls at the real projection shapes."""
    import mlx.core as mx

    from mlx2.runtime.models import fp_row_exact_qmv as fq

    shapes = {"fp8_mamba_in": (18560, 4096, "mxfp8", 8, 32),
              "fp8_mamba_out": (4096, 8192, "mxfp8", 8, 32),
              "fp8_attn_q": (4096, 4096, "mxfp8", 8, 32),
              "fp8_lm_head": (131072, 4096, "mxfp8", 8, 32),
              "nvfp4_shared_up": (5376, 4096, "nvfp4", 4, 16),
              "nvfp4_shared_down": (4096, 5376, "nvfp4", 4, 16)}
    out = []
    for name, (n, k, mode, bits, gs) in shapes.items():
        w, s, *_ = mx.quantize((mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16),
                               group_size=gs, bits=bits, mode=mode)
        mx.eval(w, s)
        for rows in (2, 3, 4, 8):
            x = (mx.random.normal((1, rows, k)) * 2.0).astype(mx.bfloat16)
            one = lambda: mx.concatenate([mx.quantized_matmul(  # noqa: E731
                x[:, r:r + 1], w, scales=s, biases=None, transpose=True, group_size=gs,
                bits=bits, mode=mode) for r in range(rows)], axis=1)
            fast = lambda: fq.rows_qmv(x, w, s, bits=bits, group_size=gs, mode="loop")  # noqa: E731
            batched = lambda: mx.quantized_matmul(x, w, scales=s, biases=None,  # noqa: E731
                                                  transpose=True, group_size=gs, bits=bits, mode=mode)
            fused = lambda: fq.rows_qmv(x, w, s, bits=bits, group_size=gs, mode="fused")  # noqa: E731
            ref, got, bat, fus = one(), fast(), batched(), fused()
            exact = bool(mx.all(ref.view(mx.uint16) == got.view(mx.uint16)).item())
            fused_exact = bool(mx.all(ref.view(mx.uint16) == fus.view(mx.uint16)).item())
            stock_batched_exact = bool(mx.all(ref.view(mx.uint16) == bat.view(mx.uint16)).item())
            row = {"shape": name, "rows": rows, "exact": exact,
                   "stock_batched_exact": stock_batched_exact,
                   "per_row_ms": round(timed(one, reps) * 1e3, 4),
                   "kernel_ms": round(timed(lambda: fq.rows_qmv(x, w, s, bits=bits, group_size=gs, mode="loop"), reps) * 1e3, 4),
                   "fused_exact": fused_exact,
                   "fused_ms": round(timed(fused, reps) * 1e3, 4),
                   "stock_batched_ms": round(timed(batched, reps) * 1e3, 4)}
            out.append(row)
            print(json.dumps(row), flush=True)
        del w, s
        mx.clear_cache()
    return out


_VIEWS = {"Reshape", "Broadcast", "ExpandDims", "Squeeze", "Flatten", "Unflatten", "Slice",
          "AsStrided", "Transpose", "StopGradient", "Depends", "Split", "Full"}


def decode_census(out, steps=48):
    """One B1 decode step: kernel-launching primitives, host build vs eval."""
    import collections
    import re

    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter

    adapter = resolve_adapter(TARGET)(str(TARGET))
    model = adapter.model
    ids = adapter.tokenizer.encode(" ".join(
        f"Entry {i}: invoice {i * 53 % 1009} for client {i % 37} totals {i * 11 % 211}."
        for i in range(400)))[:2048]
    cache = model.make_cache()
    logits = model._checkpoint_logits(mx.array([ids]), cache)
    token = mx.argmax(logits[:, -1], axis=-1)[:, None]
    mx.eval(token)
    for _ in range(8):
        token = mx.argmax(model(token, cache=cache)[:, -1], axis=-1)[:, None]
        mx.eval(token)
    build, run = [], []
    for _ in range(steps):
        mx.synchronize()
        t0 = time.perf_counter()
        nxt = mx.argmax(model(token, cache=cache)[:, -1], axis=-1)[:, None]
        t1 = time.perf_counter()
        mx.eval(nxt)
        t2 = time.perf_counter()
        build.append(t1 - t0)
        run.append(t2 - t1)
        token = nxt
    # The graph of one more step, without evaluating it first.
    logits = model(token, cache=cache)
    dot = out / "decode_step.dot"
    mx.export_to_dot(str(dot), logits)
    counts = collections.Counter(re.findall(r'label ="([^"]+)"', dot.read_text()))
    dispatch = {k: v for k, v in counts.items() if k not in _VIEWS}
    token = mx.argmax(logits[:, -1], axis=-1)[:, None]
    mx.eval(token)
    # Pipelined like a server: build step n+1 before reading step n.
    mx.synchronize()
    t0 = time.perf_counter()
    nxt = mx.argmax(model(token, cache=cache)[:, -1], axis=-1)[:, None]
    mx.async_eval(nxt)
    for _ in range(steps):
        cur = nxt
        nxt = mx.argmax(model(cur, cache=cache)[:, -1], axis=-1)[:, None]
        mx.async_eval(nxt)
        cur.item()
    nxt.item()
    pipelined = (steps + 1) / (time.perf_counter() - t0)
    report = {
        "host_build_ms_median": round(statistics.median(build) * 1e3, 3),
        "eval_ms_median": round(statistics.median(run) * 1e3, 3),
        "sync_tok_s": round(1 / statistics.median([b + r for b, r in zip(build, run)]), 2),
        "pipelined_tok_s": round(pipelined, 2),
        "dispatching_primitives": sum(dispatch.values()),
        "view_primitives": sum(v for k, v in counts.items() if k in _VIEWS),
        "by_primitive": dict(sorted(dispatch.items(), key=lambda kv: -kv[1])),
        "layout": adapter.layout,
    }
    report["greedy_after_census"] = int(token.item())
    print(json.dumps({k: v for k, v in report.items() if k != "by_primitive"}), flush=True)
    print(json.dumps(dict(list(report["by_primitive"].items())[:25])), flush=True)
    adapter.close()
    return report


def routed_parity(steps=64):
    """Greedy decode with MLX2_NEMOTRON_ROUTED_DECODE off then on: logits bits."""
    import hashlib

    import mlx.core as mx
    import numpy as np

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.models import nemotron_routed_decode as rd

    adapter = resolve_adapter(TARGET)(str(TARGET))
    model = adapter.model
    ids = adapter.tokenizer.encode(" ".join(
        f"Note {i}: the courier left hub {i * 29 % 997} with {i % 17} parcels for route {i * 7 % 61}."
        for i in range(300)))[:2048]
    arms = {}
    for mode in ("off", "on"):
        os.environ["MLX2_NEMOTRON_ROUTED_DECODE"] = mode
        rd.COUNTS.update(engaged=0, declined=0)
        cache = model.make_cache()
        logits = model._checkpoint_logits(mx.array([ids]), cache)
        token = mx.argmax(logits[:, -1], axis=-1)[:, None]
        mx.eval(token)
        tokens, digests = [], []
        for _ in range(steps):
            logits = model(token, cache=cache)[:, -1]
            token = mx.argmax(logits, axis=-1)[:, None]
            mx.eval(logits, token)
            digests.append(hashlib.sha256(np.asarray(logits.astype(mx.float32)).tobytes()).hexdigest())
            tokens.append(int(token.item()))
        arms[mode] = {"tokens": tokens, "digests": digests, "counts": dict(rd.COUNTS)}
    os.environ["MLX2_NEMOTRON_ROUTED_DECODE"] = "off"
    first = next((i for i, (a, b) in enumerate(zip(arms["off"]["digests"], arms["on"]["digests"]))
                  if a != b), None)
    report = {"steps": steps, "tokens_identical": arms["off"]["tokens"] == arms["on"]["tokens"],
              "logits_bitwise_identical_steps": sum(a == b for a, b in zip(
                  arms["off"]["digests"], arms["on"]["digests"])),
              "first_logits_difference": first, "counts_on": arms["on"]["counts"],
              "counts_off": arms["off"]["counts"]}
    print(json.dumps(report), flush=True)
    adapter.close()
    return report


def model_profile(contexts, decode_steps, reps, policy=None, split=True):
    import mlx.core as mx
    from mlx import nn

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime import nvfp4_linear
    from mlx2.runtime.models import nemotron_h

    adapter = resolve_adapter(TARGET)(str(TARGET), execution_policy=policy or None)
    model = adapter.model
    names = {id(module): name for name, module in model.named_modules()}
    report = {"layout": adapter.layout, "profile": adapter.profile_name(False),
              "policy": policy or {}, "env": {k: v for k, v in os.environ.items()
                                             if k.startswith("MLX2_NVFP4") or k.startswith("MLX2_MOE")}}

    def category(module):
        name = names.get(id(module), "")
        leaf = name.rsplit(".", 1)[-1]
        if isinstance(module, nemotron_h.NemotronHMamba2Mixer):
            return "mamba_core(conv,ssm,norm)"
        if isinstance(module, nemotron_h.NemotronHAttention):
            return "attention_core"
        if isinstance(module, nemotron_h.NemotronHMoE):
            return "moe_glue(route,combine)"
        if isinstance(module, nemotron_h.MoEGate):
            return "moe_router"
        if isinstance(module, nemotron_h.NemotronHMLP):
            return "shared_expert_glue"
        if isinstance(module, nvfp4_linear.ScaledNVFP4SwitchLinear):
            return "routed_" + leaf
        if isinstance(module, nvfp4_linear.ScaledNVFP4Linear):
            return "shared_" + leaf
        if leaf in {"in_proj", "out_proj"}:
            return "mamba_" + leaf
        if leaf in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            return "attention_proj"
        if "latent" in leaf:
            return "moe_latent_proj"
        if leaf == "lm_head":
            return "lm_head"
        return "other_linear:" + leaf

    classes = [nemotron_h.NemotronHMamba2Mixer, nemotron_h.NemotronHAttention,
               nemotron_h.NemotronHMoE, nemotron_h.MoEGate, nemotron_h.NemotronHMLP,
               nvfp4_linear.ScaledNVFP4SwitchLinear, nvfp4_linear.ScaledNVFP4Linear, nn.Linear]
    originals = {cls: cls.__call__ for cls in classes}
    state = {"on": False, "stack": [], "totals": defaultdict(float), "calls": defaultdict(int)}

    def wrap(cls):
        original = originals[cls]

        def call(self, *a, **kw):
            if not state["on"]:
                return original(self, *a, **kw)
            mx.eval([v for v in a if isinstance(v, mx.array)])
            state["stack"].append(0.0)
            started = time.perf_counter()
            result = original(self, *a, **kw)
            mx.eval(result)
            elapsed = time.perf_counter() - started
            children = state["stack"].pop()
            if state["stack"]:
                state["stack"][-1] += elapsed
            key = category(self)
            state["totals"][key] += elapsed - children
            state["calls"][key] += 1
            return result

        cls.__call__ = call

    for cls in classes:
        wrap(cls)
    prompt_ids = adapter.tokenizer.encode(
        " ".join(f"Item {i}: the quarterly ledger for account {i * 37 % 1009} lists"
                 f" {i * 13 % 97} entries, reviewed by team {i % 23}." for i in range(4000)))

    def prefill(tokens, cache):
        logits = None
        for start in range(0, tokens, 2048):
            chunk = mx.array([prompt_ids[start:min(tokens, start + 2048)]])
            logits = model._checkpoint_logits(chunk, cache)
            mx.eval(logits, [c.state for c in cache if hasattr(c, "state")])
        return logits

    sync = timed(lambda: mx.array(1.0) + 1, 20)
    report["sync_floor_ms"] = round(sync * 1e3, 4)
    report["prefill"] = []
    for tokens in contexts:
        if tokens > len(prompt_ids):
            raise ValueError("prompt corpus shorter than context")
        samples = []
        for _ in range(reps):
            cache = model.make_cache()
            mx.synchronize()
            started = time.perf_counter()
            prefill(tokens, cache)
            mx.synchronize()
            samples.append(time.perf_counter() - started)
        seconds = statistics.median(samples)
        row = {"tokens": tokens, "seconds": round(seconds, 3), "tok_s": round(tokens / seconds, 1)}
        report["prefill"].append(row)
        print(json.dumps({"prefill": row}), flush=True)
    # Decode after a 2048-token context.
    cache = model.make_cache()
    logits = prefill(2048, cache)
    token = mx.argmax(logits[:, -1], axis=-1)[:, None]
    mx.eval(token)
    started = None
    decoded = []
    for step in range(decode_steps + 4):
        if step == 4:
            mx.synchronize()
            started = time.perf_counter()
        logits = model(token, cache=cache)
        token = mx.argmax(logits[:, -1], axis=-1)[:, None]
        mx.eval(token)
        decoded.append(int(token.item()))
    mx.synchronize()
    report["decode"] = {"context": 2048, "steps": decode_steps,
                        "tok_s": round(decode_steps / (time.perf_counter() - started), 2),
                        "greedy_token_ids": decoded}
    from mlx2.runtime.models import moe_nax_gather as nax

    report["nax"] = {"mode": nax.MODE, "status": nax.status() if hasattr(nax, "status") else None}
    print(json.dumps({"decode": report["decode"]}), flush=True)
    # Exclusive split: one instrumented 2048-row prefill chunk, then 16 decode steps.
    for phase, run in () if not split else (
        ("prefill_2048_chunk", lambda: prefill(2048, model.make_cache())),
        ("decode_x16", None),
    ):
        state["totals"].clear()
        state["calls"].clear()
        if phase == "decode_x16":
            cache = model.make_cache()
            logits = prefill(2048, cache)
            token = mx.argmax(logits[:, -1], axis=-1)[:, None]
            mx.eval(token)
            state["on"] = True
            wall = time.perf_counter()
            for _ in range(16):
                logits = model(token, cache=cache)
                token = mx.argmax(logits[:, -1], axis=-1)[:, None]
                mx.eval(token)
        else:
            state["on"] = True
            wall = time.perf_counter()
            run()
        wall = time.perf_counter() - wall
        state["on"] = False
        total = sum(state["totals"].values())
        split = {k: {"ms": round(v * 1e3, 2), "share": round(v / total, 3),
                     "calls": state["calls"][k]}
                 for k, v in sorted(state["totals"].items(), key=lambda kv: -kv[1])}
        report[phase] = {"instrumented_wall_s": round(wall, 3), "split": split}
        print(json.dumps({phase: report[phase]}), flush=True)
    for cls, original in originals.items():
        cls.__call__ = original
    adapter.close()
    return report


def served_prefill(flags, tokens, port, env_extra):
    """TTFT of distinct ``tokens``-long prompts through ``mlx2.server``."""
    sys.path.insert(0, str(ROOT / "scripts/model_matrix"))
    import run_matrix as rm
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TARGET, local_files_only=True)
    command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(TARGET), "--host",
               "127.0.0.1", "--port", str(port), "--fabric", "off", "--max-context", "32768",
               "--cache-bytes", str(2 << 30), *flags]
    server = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                              cwd=ROOT, start_new_session=True,
                              env={**os.environ, "PYTHONPATH": str(ROOT / "src"), **env_extra})
    base = f"http://127.0.0.1:{port}"
    rows = []
    try:
        rm.wait_health(base, server, 1800, lambda: None)
        for rep in range(3):
            words = " ".join(f"Record {rep}-{i}: shipment {i * 31 % 997} left depot {i % 41}"
                             f" carrying {i * 7 % 113} crates." for i in range(tokens // 14))
            ids = tok.encode(words)[:tokens]
            text = tok.decode(ids)
            body = {"model": TARGET.name, "messages": [{"role": "user", "content": text}],
                    "max_tokens": 1, "temperature": 0, "stream": False, "enable_thinking": False}
            code, response, wall = rm.http_json(base + "/v1/chat/completions", body, timeout=900)
            block = response.get("mlx2", {}) if isinstance(response, dict) else {}
            prompt = (response.get("usage") or {}).get("prompt_tokens") if code == 200 else None
            ttft = block.get("ttft_seconds")
            rows.append({"rep": rep, "status": code, "prompt_tokens": prompt, "ttft_seconds": ttft,
                         "route": block.get("route"),
                         "prefill_tok_s": round(prompt / ttft, 1) if prompt and ttft else None})
            print(json.dumps({"served": flags, **rows[-1]}), flush=True)
    finally:
        rm.stop_server(server)
    return rows


def source():
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", "src"]).returncode != 0
    return {"commit": commit, "src_dirty": dirty}


def checkpoint():
    from mlx2.adapters.registry import inspect_model

    return f"{TARGET.name}@sha256:{inspect_model(TARGET).artifact_fingerprint}"


def unit(out, args):
    import mlx.core as mx

    report = {"schema": "mlx2.super-vl-nvfp4-profile.v1", "checkpoint": checkpoint(),
              "source": source(), "host": "m5", "mlx": mx.__version__, "qualified": False,
              "evidence_kind": "attribution profile, no thermal hold"}
    if "kernels" in args.parts:
        report["kernels"] = kernels(args.reps)
        atomic(out / "report.json", report)
    if "model" in args.parts:
        report["model"] = model_profile(args.contexts, args.decode_steps, args.reps,
                                        policy=json.loads(args.policy), split=not args.no_split)
        atomic(out / "report.json", report)
    if "routed" in args.parts:
        report["routed"] = routed_parity()
        atomic(out / "report.json", report)
    if "census" in args.parts:
        report["census"] = decode_census(out)
        atomic(out / "report.json", report)
    if "rowexact" in args.parts:
        report["rowexact"] = rowexact_check(max(args.reps, 5))
        atomic(out / "report.json", report)
    if "served" in args.parts:
        report["served"] = {
            name: served_prefill(flags, args.served_tokens, args.port, {})
            for name, flags in (("ordinary", ["--ordinary"]), ("default_route", []))
        }
        atomic(out / "report.json", report)
    return 0


def owned(out, args):
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    job_id = "super-vl-nvfp4-profile-" + out.name
    with DesktopCPGLease(out, job_id, checkpoint=checkpoint(),
                         source_revision=source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "arms" if args.arms_file else "unit",
                   str(out), "--parts", *args.parts, "--contexts", *map(str, args.contexts),
                   "--decode-steps", str(args.decode_steps), "--reps", str(args.reps),
                   "--policy", args.policy, "--served-tokens", str(args.served_tokens),
                   "--port", str(args.port), *(["--no-split"] if args.no_split else []),
                   *(["--arms-file", str(args.arms_file)] if args.arms_file else [])]
        child = subprocess.Popen(command, cwd=ROOT, start_new_session=True,
                                 env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        try:
            while child.poll() is None:
                lease.check()
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=120)


def arms(out, args):
    """Each arm in its own process (environment switches are read at load)."""
    results = {}
    for arm in json.loads(Path(args.arms_file).read_text()):
        target = out / arm["name"]
        target.mkdir()
        command = [sys.executable, "-u", str(Path(__file__)), "unit", str(target),
                   "--parts", *arm.get("parts", args.parts),
                   "--contexts", *map(str, args.contexts), "--decode-steps", str(args.decode_steps),
                   "--reps", str(args.reps), "--policy", json.dumps(arm.get("policy", {})),
                   "--served-tokens", str(args.served_tokens), "--port", str(args.port),
                   *(["--no-split"] if arm.get("no_split", args.no_split) else [])]
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), **arm.get("env", {})}
        with (target / "unit.log").open("w") as log:
            code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        results[arm["name"]] = code
        print(json.dumps({"arm": arm["name"], "returncode": code}), flush=True)
        atomic(out / "arms.json", results)
    return 0 if all(code == 0 for code in results.values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("owned", "unit", "arms"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--parts", nargs="+", default=["kernels", "model"],
                        choices=["kernels", "model", "served", "rowexact", "census", "routed"])
    parser.add_argument("--policy", default="{}")
    parser.add_argument("--no-split", action="store_true")
    parser.add_argument("--arms-file", type=Path)
    parser.add_argument("--served-tokens", type=int, default=8192)
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--contexts", type=int, nargs="+", default=[2048, 8192, 16384])
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    return {"owned": owned, "unit": unit, "arms": arms}[args.mode](args.out, args)


if __name__ == "__main__":
    sys.exit(main())
