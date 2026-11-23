#!/usr/bin/env python3
"""Maxwell-V1-Flash GPU window: functional checks and lever A/B for the full adapter.

  maxwell_overnight.py window --out DIR [--model DIR] [--stop-at 06:45] [--gsm8k 50]
      Run every unit below, one model per process, under the caller's GPU lock
      (wrap the call in /Users/Shared/mlxuag/gpuq-reference.sh).  Refuses a
      nonempty --out unless --resume, which skips units whose receipt exists.
  maxwell_overnight.py direct --out DIR --arm NAME [--model DIR] [--cells short,long]
      One process: load the adapter with the arm's environment levers, run
      the cells, write DIR/receipt.json.

Units, in order (each gates the next where noted):
  1. smoke      scripts/smoke_maxwell_moe.py window (truncated torch parity + greedy).
  2. baseline   direct arm, default levers: short greedy prompts plus needle
                retrieval at 3K (dense regime), 6K, 16K and 32K (sparse), with
                Block Sparse GQA counters.  Gate: every needle answered.
  3. levers     direct arms eager / mhc_kernel / head_bf16 / fused_qkv /
                all_levers / masked_prefill, each compared with the baseline:
                greedy token identity, the first divergence with both arms'
                top-2 log-probabilities, and speed.
  4. serving    loopback server per arm (ordinary, native MTP num_draft 1/2/3):
                APCv2 cold/warm replay of the 6K needle prompt (cached tokens and
                text), then scripts/spec_acceptance_gsm8k.py in base-url mode.

Sanity and attribution numbers, no thermal hold: not publishable statistics.
Every receipt names the checkpoint (<name>@<fingerprint>) and the mlx2 commit.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path(os.environ.get("MLX2_MAXWELL_CHECKPOINT", str(Path.home() / "Models/Maxwell-V1-Flash")))
PORT = 18311

ARMS = {
    "baseline": {},
    "eager": {"MLX2_MAXWELL_COMPILE": "0"},
    "mhc_kernel": {"MLX2_MAXWELL_MHC_KERNEL": "1"},
    "head_bf16": {"MLX2_MAXWELL_HEAD": "bf16"},
    "fused_qkv": {"MLX2_MAXWELL_FUSED_QKV": "1"},
    "all_levers": {"MLX2_MAXWELL_MHC_KERNEL": "1", "MLX2_MAXWELL_HEAD": "bf16", "MLX2_MAXWELL_FUSED_QKV": "1"},
    "masked_prefill": {"MLX2_MAXWELL_SPARSE_PREFILL": "masked"},
    "index_kernel": {"MLX2_MAXWELL_INDEX_KERNEL": "1"},
    "head_fp32": {"MLX2_MAXWELL_HEAD": "fp32"},
}
ARM_CELLS = {"masked_prefill": ("long",)}
SHORT_PROMPTS = [
    "What is the capital of Australia, and why was it chosen instead of Sydney or Melbourne?",
    "Write a short Python function that returns the n-th Fibonacci number iteratively.",
    "In two sentences, explain why the daytime sky looks blue.",
]
NEEDLE_CONTEXTS = (3000, 6000, 16000, 32000)
SERVER_ARMS = [
    ("ordinary", ["--ordinary"], None),
    ("mtp1", ["--native-mtp"], {"num_draft": 1}),
    ("mtp2", ["--native-mtp"], {"num_draft": 2}),
    ("mtp3", ["--native-mtp"], {"num_draft": 3}),
]

_CITIES = ["Lyon", "Osaka", "Quito", "Tromso", "Perth", "Accra", "Lima", "Graz", "Hanoi", "Leeds",
           "Porto", "Tunis", "Cusco", "Malmo", "Dakar", "Split", "Bergen", "Cali", "Nantes", "Kobe"]
_GOODS = ["cedar planks", "copper wire", "wool bales", "glass jars", "tin cans", "salt sacks",
          "rope coils", "clay bricks", "oak barrels", "linen rolls", "steel bolts", "paper reams"]
_VERBS = ["shipped", "received", "stored", "inspected", "returned", "weighed", "sorted", "repacked"]


def memory():
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    mp = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    free = re.search(r"free percentage: (\d+)%", mp)
    return {"swapouts": int(re.search(r"Swapouts:\s+(\d+)", vm).group(1)),
            "swap_used_mb": float(re.search(r"used = ([\d.]+)M", swap).group(1)),
            "free_pct": int(free.group(1)) if free else None, "time": time.time()}


def revision():
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "tests"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    return {"mlx2_commit": rev, "dirty": bool(dirty)}


def write_atomic(path: Path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=str) + "\n")
    tmp.replace(path)


def foreign_models():
    out = subprocess.run(["pgrep", "-fl", "mlx2.server|mlx_lm|mlx_vlm|smoke_maxwell|maxwell_overnight.py direct"],
                         capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if str(os.getpid()) not in line.split()[:1]]


# ---------------------------------------------------------------- prompts


def needle_prompt(tokenizer, target_tokens, seed):
    """Varied ledger sentences (no repeated strings) with one code at ~35% depth."""
    rng = random.Random(seed)
    code = f"{rng.choice('BCDFGHJKLMNPQRSTVWXZ')}{rng.randint(2, 9)}-{rng.choice('KQXZ')}{rng.randint(10, 99)}-" \
           f"{''.join(rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ') for _ in range(3))}"
    needle = f"Note for the archive: the access code for the north vault is {code}."
    question = "What is the access code for the north vault? Reply with the code only."
    lines, n = [], 0
    while True:
        lines.append(f"Ledger entry {1000 + n}: the {rng.choice(_CITIES)} depot {rng.choice(_VERBS)} "
                     f"{rng.randint(3, 997)} {rng.choice(_GOODS)} on day {rng.randint(1, 365)} "
                     f"for client {rng.randint(10000, 99999)}.")
        n += 1
        if n % 50 == 0 and len(tokenizer.encode(" ".join(lines), add_special_tokens=False)) >= target_tokens:
            break
    lines.insert(int(len(lines) * 0.35), needle)
    return {"messages": [{"role": "user", "content": " ".join(lines) + "\n\n" + question}],
            "enable_thinking": False}, code


def normalize_code(text):
    return re.sub(r"[‐-―−]", "-", text)


# ---------------------------------------------------------------- direct arm


def greedy(adapter, request, max_tokens, probe_steps=0):
    import mlx.core as mx

    model = adapter.model
    toks = adapter.prompt_tokens(request)
    cache = model.make_cache()
    eos = set(adapter.tokenizer.eos_token_ids)
    t0 = time.perf_counter()
    # Chunked prefill like the engine (prefill_step_default 512): one forward
    # over a 32K prompt holds GiB-scale mHC/MoE temporaries.
    step = adapter.prefill_step_default()
    for start in range(0, len(toks) - 1, step):
        end = min(start + step, len(toks) - 1)
        model(mx.array([toks[start:end]]), cache)
        mx.eval([c.state for c in cache if c.keys is not None])
        mx.clear_cache()
    logits = model(mx.array([toks[-1:]]), cache)[:, -1]
    first = None
    gen, top2 = [], []
    for step in range(max_tokens):
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        idx = mx.argpartition(-lp, kth=1, axis=-1)[..., :2]
        vals = mx.take_along_axis(lp, idx, axis=-1)
        mx.eval(idx, vals)
        if first is None:
            first = time.perf_counter()
        pair = sorted(zip(idx[0].tolist(), vals[0].tolist()), key=lambda p: -p[1])
        top2.append([[int(t), round(float(v), 5)] for t, v in pair])
        nxt = pair[0][0]
        if nxt in eos:
            break
        gen.append(nxt)
        logits = model(mx.array([[nxt]]), cache)[:, -1]
    end = time.perf_counter()
    mx.eval(logits)
    probe = None
    if probe_steps:
        # Decode-rate probe: a needle answer is a handful of tokens, too few to time.
        # Feed argmax for a fixed number of further steps (content not scored).
        nxt = mx.argmax(logits, axis=-1)
        mx.eval(nxt)
        p0 = time.perf_counter()
        for _ in range(probe_steps):
            logits = model(nxt[:, None], cache)[:, -1]
            nxt = mx.argmax(logits, axis=-1)
            mx.eval(nxt)
        probe = round(probe_steps / (time.perf_counter() - p0), 2)
    del cache
    mx.clear_cache()
    decode_steps = max(len(top2) - 1, 1)
    return {"prompt_tokens": len(toks), "tokens": gen, "top2": top2,
            "text": adapter.tokenizer.decode(gen), "ttft_s": round(first - t0, 3),
            "prefill_tok_s": round(len(toks) / (first - t0), 1),
            "decode_tok_s": round(decode_steps / max(end - first, 1e-9), 2), "decode_probe_tok_s": probe}


def direct(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arm = args.arm
    env = ARMS[arm]
    for key, value in env.items():
        if os.environ.get(key) != value:
            raise SystemExit(f"direct arm {arm} must be started with {key}={value}")
    import mlx.core as mx

    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.adapters.registry import inspect_model
    from mlx2.runtime.models import maxwell_moe as M

    mx.set_cache_limit(4 << 30)
    t0 = time.time()
    res = inspect_model(Path(args.model))
    adapter = res.adapter_type(str(args.model))
    receipt = {"arm": arm, "env": env, **revision(),
               "checkpoint": f"{Path(args.model).name}@{adapter.identity['fingerprint']}",
               "load_s": round(time.time() - t0, 1), "active_gb": round(mx.get_active_memory() / 2**30, 2),
               "levers": M.maxwell_stats()["levers"], "qualification": "unqualified", "cells": {}}
    cells = set(args.cells.split(","))
    m0 = memory()
    if "short" in cells:
        rows = []
        for prompt in SHORT_PROMPTS:
            rows.append({"prompt": prompt, **greedy(adapter, {"messages": [{"role": "user", "content": prompt}]},
                                                    args.short_tokens)})
            print(json.dumps({k: rows[-1][k] for k in ("prompt_tokens", "ttft_s", "decode_tok_s")}), flush=True)
        receipt["cells"]["short"] = rows
    if "long" in cells:
        rows = []
        contexts = tuple(int(c) for c in args.contexts.split(",")) if args.contexts else NEEDLE_CONTEXTS
        for n, target in enumerate(contexts):
            request, code = needle_prompt(adapter.tokenizer, target, seed=20261122 + n)
            M.maxwell_stats(reset=True)
            row = greedy(adapter, request, 24, probe_steps=32)
            row.update(target=target, code=code, found=code in normalize_code(row["text"]),
                       attention=M.maxwell_stats())
            row.pop("levers", None)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("target", "prompt_tokens", "found", "ttft_s", "decode_tok_s",
                                                  "decode_probe_tok_s")}),
                  flush=True)
        receipt["cells"]["long"] = rows
    receipt["memory"] = {"before": m0, "after": memory(), "peak_gb": round(mx.get_peak_memory() / 2**30, 2)}
    receipt["diagnostics"] = adapter.diagnostics()
    adapter.close()
    write_atomic(out / "receipt.json", receipt)


# ---------------------------------------------------------------- comparison


def first_divergence(a, b):
    """Index of the first differing generated token and both arms' top-2 rows there."""
    for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])):
        if x != y:
            return i
    return None if len(a["tokens"]) == len(b["tokens"]) else min(len(a["tokens"]), len(b["tokens"]))


def classify(base, other, oracle_ok=True):
    """exact / near_tie_equivalent (<= 0.5 nat both arms, same top-2 set) / unclassified."""
    i = first_divergence(base, other)
    if i is None:
        return {"verdict": "exact"}
    if i >= len(base["top2"]) or i >= len(other["top2"]):
        return {"verdict": "unclassified", "index": i, "reason": "length"}
    (ba, bav), (bb, bbv) = base["top2"][i]
    (oa, oav), (ob, obv) = other["top2"][i]
    margins = (bav - bbv, oav - obv)
    same_pair = {ba, bb} == {oa, ob}
    ok = same_pair and max(margins) <= 0.5 and oracle_ok
    return {"verdict": "near_tie_equivalent" if ok else "unclassified", "index": i,
            "base_top2": base["top2"][i], "other_top2": other["top2"][i], "margins": margins}


def compare(baseline, other):
    out = {"arm": other["arm"], "env": other["env"]}
    for cell in ("short", "long"):
        if cell not in baseline["cells"] or cell not in other["cells"]:
            continue
        rows = []
        for b, o in zip(baseline["cells"][cell], other["cells"][cell]):
            oracle = o.get("found", True) and b.get("found", True)
            rows.append({"prompt_tokens": b["prompt_tokens"], **classify(b, o, oracle),
                         "found": o.get("found"), "ttft_s": (b["ttft_s"], o["ttft_s"]),
                         "decode_tok_s": (b["decode_tok_s"], o["decode_tok_s"])})
        out[cell] = rows
    return out


# ---------------------------------------------------------------- serving


class Server:
    def __init__(self, model, out, args, policy):
        self.out, self.args = out, args
        out.mkdir(parents=True, exist_ok=True)
        self.token = os.urandom(24).hex()
        (out / "admin-token").write_text(self.token)
        (out / "admin-token").chmod(0o600)
        command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(model), "--host", "127.0.0.1",
                   "--port", str(PORT), "--max-inflight", "1", "--max-lanes", "1", "--max-context", "40000",
                   "--cache-bytes", str(4 << 30), "--cache-dir", str(out / "cache"),
                   "--admin-token-file", str(out / "admin-token"), *args]
        if policy is not None:
            (out / "policy.json").write_text(json.dumps(policy))
            command += ["--execution-policy", str(out / "policy.json")]
        self.command = command
        self.log = (out / "server.log").open("w")
        self.proc = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True,
                                     cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})

    def call(self, path, payload=None, timeout=1800):
        from urllib.request import Request, urlopen

        data = None if payload is None else json.dumps(payload).encode()
        req = Request(f"http://127.0.0.1:{PORT}{path}", data=data,
                      headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.token})
        with urlopen(req, timeout=timeout) as r:
            body = r.read()
        return json.loads(body) if body.strip().startswith(b"{") else body.decode()

    def wait_ready(self, limit=1200):
        end = time.monotonic() + limit
        while time.monotonic() < end:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited before readiness")
            try:
                status = self.call("/v1/status", timeout=10)
                if status.get("healthy"):
                    return status
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("server readiness timed out")

    def stop(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=10)
        self.log.close()


def apc_replay(server, tokenizer, model_name):
    request, code = needle_prompt(tokenizer, 6000, seed=20261122 + 1)
    rows = []
    for attempt in ("cold", "warm"):
        t0 = time.perf_counter()
        r = server.call("/v1/chat/completions", {"model": model_name, "messages": request["messages"],
                                                  "logprobs": True,
                                                  "max_tokens": 24, "temperature": 0,
                                                  "chat_template_kwargs": {"enable_thinking": False}})
        usage = r.get("usage", {})
        text = r["choices"][0]["message"].get("content") or ""
        logprobs = r["choices"][0].get("logprobs") or {}
        token_ids = [item["id"] for item in logprobs.get("content") or []]
        rows.append({"attempt": attempt, "token_ids": token_ids, "seconds": round(time.perf_counter() - t0, 2), "text": text,
                     "found": code in normalize_code(text), "prompt_tokens": usage.get("prompt_tokens"),
                     "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                     "receipt": r.get("mlx2") or r.get("receipt")})
    cold, warm = rows
    text_identical = cold["text"] == warm["text"]
    tokens_identical = bool(cold["token_ids"]) and cold["token_ids"] == warm["token_ids"]
    warm_hit = (warm["cached_tokens"] or 0) > 0
    passed = (text_identical and tokens_identical and cold["found"] and warm["found"]
              and cold["cached_tokens"] == 0 and warm_hit)
    return {"rows": rows, "passed": passed, "text_identical": text_identical,
            "tokens_identical": tokens_identical, "warm_hit": warm_hit}


# ---------------------------------------------------------------- window


def run_unit(out, name, command, env=None, resume=False):
    unit = out / name
    if resume and (unit / "done.json").is_file():
        print(f"[skip] {name} (done)", flush=True)
        return json.loads((unit / "done.json").read_text())
    if unit.exists() and any(unit.iterdir()):
        raise SystemExit(f"refusing nonempty unit directory {unit} (use a new --out, or --resume)")
    unit.mkdir(parents=True, exist_ok=True)
    before = memory()
    foreign = foreign_models()
    if foreign:
        raise SystemExit(f"abort before {name}: foreign model processes {foreign}")
    t0 = time.time()
    with (unit / "stdout.log").open("w") as log:
        code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONPATH": str(ROOT / "src"), **(env or {})}).returncode
    after = memory()
    done = {"unit": name, "returncode": code, "seconds": round(time.time() - t0, 1),
            "swapouts_delta": after["swapouts"] - before["swapouts"], "memory": [before, after]}
    write_atomic(unit / "done.json", done)
    print(json.dumps({k: done[k] for k in ("unit", "returncode", "seconds", "swapouts_delta")}), flush=True)
    if done["swapouts_delta"] > 20000:
        raise SystemExit(f"abort: swap growth during {name}")
    return done


def window(args):
    out = Path(args.out).resolve()
    args.model = args.model.resolve()
    if args.reference_model:
        args.reference_model = args.reference_model.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise SystemExit(f"refusing nonempty evidence directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from spec_acceptance_gsm8k import stop_time

    deadline = stop_time(args.stop_at)
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.adapters.registry import inspect_model

    reference = args.reference_model or args.model
    fingerprint = inspect_model(args.model).artifact_fingerprint
    ref_fingerprint = fingerprint if reference == args.model else inspect_model(reference).artifact_fingerprint
    binding = {**revision(), "target_path": str(args.model), "target_fingerprint": fingerprint,
               "reference_path": str(reference), "reference_fingerprint": ref_fingerprint, "gsm8k": args.gsm8k}
    summary_path = out / "summary.json"
    if args.resume and any(out.iterdir()):
        if not summary_path.is_file():
            raise SystemExit("cannot resume units without a source and checkpoint binding")
        summary = json.loads(summary_path.read_text())
        if binding["dirty"] or summary.get("binding") != binding:
            raise SystemExit("cannot resume: source, checkpoint pair, or GSM8K settings are unverified or changed")
    else:
        summary = {"schema": "mlx2.maxwell-window.v1", "mlx2_commit": binding["mlx2_commit"],
                   "dirty": binding["dirty"], "model": str(args.model), "binding": binding, "units": {}}
        write_atomic(summary_path, summary)
    py = sys.executable

    def late(name):
        if args.resume and (out / name / "done.json").is_file():
            return False
        if deadline is not None and time.time() >= deadline:
            summary["units"][name] = {"skipped": f"--stop-at {args.stop_at}"}
            write_atomic(summary_path, summary)
            return True
        return False

    if not late("smoke"):
        smoke = [py, "scripts/smoke_maxwell_moe.py", "window", "--model", str(args.model),
                 "--out", str(out / "smoke" / "run")]
        if args.reference_model:
            smoke += ["--reference-model", str(args.reference_model)]
        summary["units"]["smoke"] = run_unit(out, "smoke", smoke,
                                             resume=args.resume)
        write_atomic(summary_path, summary)
        if summary["units"]["smoke"]["returncode"]:
            return 1
    me = [py, str(Path(__file__).resolve()), "direct", "--model", str(args.model)]
    receipts = {}
    for arm in ARMS:
        if late(arm):
            continue
        cells = ",".join(ARM_CELLS.get(arm, ("short", "long")))
        done = run_unit(out, arm, me + ["--arm", arm, "--out", str(out / arm), "--cells", cells],
                        env=ARMS[arm], resume=args.resume)
        path = out / arm / "receipt.json"
        if done["returncode"] == 0 and path.is_file():
            receipts[arm] = json.loads(path.read_text())
        summary["units"][arm] = {**done, "checkpoint": receipts.get(arm, {}).get("checkpoint")}
        if arm == "baseline":
            long_rows = receipts.get("baseline", {}).get("cells", {}).get("long", [])
            gate = bool(long_rows) and all(r["found"] for r in long_rows)
            summary["baseline_gate"] = {"passed": gate,
                                        "needles": [(r["prompt_tokens"], r["found"]) for r in long_rows]}
            write_atomic(summary_path, summary)
            if not gate:
                print("baseline needle gate failed; stopping before levers and serving", flush=True)
                return 1
        elif "baseline" in receipts and arm in receipts:
            summary.setdefault("comparisons", {})[arm] = compare(receipts["baseline"], receipts[arm])
        write_atomic(summary_path, summary)
    # serving arms
    from transformers import AutoTokenizer

    tokenizer = None
    for name, flags, policy in SERVER_ARMS:
        if late(f"serve_{name}"):
            continue
        unit = out / f"serve_{name}"
        if args.resume and (unit / "done.json").is_file():
            previous = json.loads((unit / "done.json").read_text())
            summary["units"][f"serve_{name}"] = {
                **previous, "passed": bool(previous.get("passed") and (previous.get("apc") or {}).get("passed"))
            }
            continue
        if unit.exists() and any(unit.iterdir()):
            raise SystemExit(f"refusing incomplete nonempty serving unit {unit}")
        if foreign_models():
            raise SystemExit(f"abort before serve_{name}: foreign model processes")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        before = memory()
        server = Server(args.model, unit, flags, policy)
        row = {"flags": flags, "policy": policy, "command": server.command}
        try:
            row["startup_status"] = server.wait_ready()
            row["apc"] = apc_replay(server, tokenizer, row["startup_status"].get("model") or args.model.name)
            gsm = unit / "gsm8k"
            cmd = [py, "scripts/spec_acceptance_gsm8k.py", "--output", str(gsm), "--base-url",
                   f"http://127.0.0.1:{PORT}", "--checkpoint", f"{args.model.name}@sha256:{fingerprint}",
                   "--num-questions", str(args.gsm8k), "--max-tokens", "512",
                   "--request", json.dumps({"enable_thinking": False})]
            with (unit / "gsm8k.log").open("w") as log:
                row["gsm8k_returncode"] = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                         env={**os.environ, "PYTHONPATH": str(ROOT / "src")}
                                                         ).returncode
            row["final_status"] = server.call("/v1/status")
            row["passed"] = row["gsm8k_returncode"] == 0 and row["apc"]["passed"]
        except Exception as error:
            row["passed"] = False
            row["error"] = repr(error)
        finally:
            server.stop()
        after = memory()
        row["swapouts_delta"] = after["swapouts"] - before["swapouts"]
        write_atomic(unit / "done.json", row)
        summary["units"][f"serve_{name}"] = {k: row.get(k) for k in ("passed", "error", "apc", "swapouts_delta")}
        write_atomic(summary_path, summary)
        if row["swapouts_delta"] > 20000:
            raise SystemExit("abort: swap growth during serving")
    failed = [name for name, row in summary["units"].items()
              if row.get("returncode", 0) != 0 or row.get("passed") is False]
    incomplete = [name for name, row in summary["units"].items() if row.get("skipped")]
    summary["incomplete_units"] = incomplete
    summary["failed_units"] = failed
    write_atomic(summary_path, summary)
    print(json.dumps({"done": not incomplete, "failed_units": failed, "incomplete_units": incomplete, "summary": str(summary_path)}), flush=True)
    return int(bool(failed or incomplete))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("window")
    w.add_argument("--out", required=True)
    w.add_argument("--model", type=Path, default=MODEL)
    w.add_argument("--reference-model", type=Path, help="BF16 source for the smoke's torch reference")
    w.add_argument("--stop-at", default="06:45")
    w.add_argument("--gsm8k", type=int, default=50)
    w.add_argument("--resume", action="store_true")
    d = sub.add_parser("direct")
    d.add_argument("--out", required=True)
    d.add_argument("--arm", choices=sorted(ARMS), required=True)
    d.add_argument("--model", type=Path, default=MODEL)
    d.add_argument("--cells", default="short,long")
    d.add_argument("--short-tokens", type=int, default=96)
    d.add_argument("--contexts", help="comma-separated needle prompt lengths (default 3000,6000,16000,32000)")
    args = parser.parse_args()
    if args.cmd == "window":
        return window(args)
    return direct(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
