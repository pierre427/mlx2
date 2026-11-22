#!/usr/bin/env python3
"""Ordinary-route smoke of Inkling 2 Mini through the mlx2 adapter.

Original mlx2 script.  One bounded GPU unit, run under the paired-lock /
CPG ownership window (``scripts/run_with_gpu_locks.py``); never touches
port 8282.  Refuses a nonempty run directory and writes atomic receipts that
name ``<artifact>@<fingerprint>`` and the mlx2 commit.

Phase ``direct`` (adapter in process, B1 greedy):
  * functional oracles on chat prompts (content channel parsed);
  * decode-versus-prefill consistency: one prefill vs prefill + 8 decode
    steps, and 512- vs 128-row chunked prefill over a prompt longer than the
    1024-token local window; first divergence top-2 margins recorded;
  * sanity prefill/decode rates (not performance evidence).

Phase ``server`` (loopback server on ``--port``, 2 lanes, APCv2):
  * cold/warm replay of the same request after an APC clear (exact text +
    cached-token receipt);
  * two different requests concurrently versus each alone (B2 vs B1).

Usage::

    smoke_inkling2.py MODEL RUN_DIR [--phase direct|server|both] [--port 8391]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROMPTS = (
    ("capital", "What is the capital of France? Answer in one word.", r"\bParis\b"),
    ("arithmetic", "What is 17 + 25? Reply with the number only.", r"\b42\b"),
    ("prose", "Write one sentence about a lighthouse keeper.", r"[A-Za-z]{3,}.*[A-Za-z]{3,}"),
    ("code", "Write a Python function named is_even that returns True for even integers. "
             "Reply with only the code.", r"def is_even\("),
)
# Long, varied prose (no repeated strings: reasoning models loop on them).
PASSAGE = (
    "The harbour town woke slowly. Fishmongers stacked crates of mackerel beside the quay "
    "while gulls argued over scraps. A ferry from the northern islands sounded its horn twice, "
    "and the schoolteacher, late as always, cycled past the bakery with a satchel of corrected "
    "essays. Down by the slipway an old boatbuilder planed a strip of larch, measuring twice "
    "and cutting once, as his father had taught him. Rain clouds gathered over the headland; "
    "the lighthouse keeper logged the barometer at nine and noted a falling trend. "
)


def swap_used_mb() -> float:
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    match = re.search(r"used = ([\d.]+)M", out)
    return float(match.group(1)) if match else -1.0


def git_commit() -> dict:
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                                 "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    return {"commit": head, "dirty": dirty}


def write_atomic(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def long_prompt_tokens(adapter, target):
    words, i = [], 0
    while True:
        words.append(f"Day {i + 1}. " + PASSAGE)
        i += 1
        tokens = adapter.tokenizer.encode(" ".join(words), add_special_tokens=False)
        if len(tokens) >= target:
            return tokens[:target]


def margin(row):
    import mlx.core as mx

    top = mx.topk(row.astype(mx.float32), 2).tolist()
    return round(abs(top[0] - top[1]), 4)


def compare_rows(a, b):
    """Max |diff|, argmax agreement and the first disagreeing row's margins."""
    import mlx.core as mx

    a, b = a.astype(mx.float32), b.astype(mx.float32)
    agree = (a.argmax(-1) == b.argmax(-1)).tolist()
    first = next((i for i, ok in enumerate(agree) if not ok), None)
    result = {"max_abs_logit_diff": round(mx.abs(a - b).max().item(), 5),
              "argmax_agreement": f"{sum(agree)}/{len(agree)}", "first_disagreement": first}
    if first is not None:
        result["margins_at_first"] = [margin(a[first]), margin(b[first])]
    return result


def direct(adapter, max_tokens):
    import mlx.core as mx

    model = adapter.model
    eos = int(adapter.config["eos_token_id"])
    out = {}

    def greedy(tokens, n):
        cache = model.make_cache()
        t0 = time.monotonic()
        logits = model(mx.array([tokens]), cache)
        mx.eval(logits)
        prefill = time.monotonic() - t0
        generated, first = [], logits[0, -1]
        t1 = time.monotonic()
        for _ in range(n):
            token = int(mx.argmax(logits[0, -1]).item())
            generated.append(token)
            if token == eos:
                break
            logits = model(mx.array([[token]]), cache)
        decode = time.monotonic() - t1
        if not bool(mx.all(mx.isfinite(first)).item()):
            raise RuntimeError("nonfinite logits on the first decode step")
        return generated, prefill, decode, margin(first)

    cases = []
    for name, prompt, oracle in PROMPTS:
        request = {"messages": [{"role": "user", "content": prompt}], "reasoning_effort": "none"}
        tokens = adapter.prompt_tokens(request)
        generated, prefill, decode, first_margin = greedy(tokens, max_tokens)
        text = adapter.tokenizer.decode(generated, skip_special_tokens=False)
        parser = adapter.output_parser(request)
        events = parser.push(text) + parser.finish("", "stop")
        content = "".join(e.get("content", "") for e in events)
        reasoning = "".join(e.get("reasoning_content", "") for e in events)
        case = {"name": name, "prompt_tokens": len(tokens), "output_tokens": len(generated),
                "output_token_ids": generated,
                "finished_eos": bool(generated and generated[-1] == eos), "content": content,
                "reasoning_chars": len(reasoning), "oracle": oracle,
                "oracle_pass": bool(re.search(oracle, content)), "first_step_margin": first_margin,
                "decode_tok_s": round(len(generated) / decode, 2) if decode else None}
        cases.append(case)
        print(f"[{name}] {'PASS' if case['oracle_pass'] else 'FAIL'} {content[:120]!r}", flush=True)
    out["cases"] = cases

    # Decode vs prefill on real weights (local ring wraps past 1024).
    tokens = long_prompt_tokens(adapter, 1400)
    full = model(mx.array([tokens]), model.make_cache())[0, -8:]
    cache = model.make_cache()
    rows = []
    model(mx.array([tokens[:-8]]), cache)
    for t in tokens[-8:]:
        rows.append(model(mx.array([[t]]), cache)[0, -1:])
    out["decode_vs_prefill_1400"] = compare_rows(full, mx.concatenate(rows, axis=0))

    def chunked(step):
        cache = model.make_cache()
        last = None
        for a in range(0, len(tokens), step):
            last = model(mx.array([tokens[a:a + step]]), cache)
        return last[0, -1:]

    out["chunk512_vs_chunk128_last_row"] = compare_rows(chunked(512), chunked(128))
    print("consistency", out["decode_vs_prefill_1400"], out["chunk512_vs_chunk128_last_row"], flush=True)

    # Sanity rates (B1, 512-row chunks), not performance evidence.
    rates = {}
    for n in (4096, 16384):
        long = long_prompt_tokens(adapter, n)
        cache = model.make_cache()
        mx.synchronize()
        t0 = time.monotonic()
        for a in range(0, n, 512):
            # Body only, head on the final row: what serving's checkpoint rows do.
            hidden = model.model(mx.array([long[a:a + 512]]), cache)
            mx.eval(hidden)
        logits = model._project_logits(model.model.norm(hidden[:, -1:]))
        mx.eval(logits)
        prefill = time.monotonic() - t0
        token = int(mx.argmax(logits[0, -1]).item())
        t1 = time.monotonic()
        steps = 64
        for _ in range(steps):
            logits = model(mx.array([[token]]), cache)
            token = int(mx.argmax(logits[0, -1]).item())
        decode = time.monotonic() - t1
        rates[str(n)] = {"prefill_tok_s": round(n / prefill, 1), "decode_tok_s": round(steps / decode, 2)}
        print("rates", n, rates[str(n)], flush=True)
    out["sanity_rates_b1"] = rates
    out["all_oracles_pass"] = all(c["oracle_pass"] for c in cases)
    return out


def fused_projection_ab(adapter, max_tokens, baseline):
    """Opt-in fused projections: token-exactness vs the baseline cases and an
    A-B-A decode/prefill sanity comparison (not performance evidence)."""
    import mlx.core as mx

    model = adapter.model
    eos = int(adapter.config["eos_token_id"])

    def greedy(tokens, n):
        cache = model.make_cache()
        logits = model(mx.array([tokens]), cache)
        generated = []
        for _ in range(n):
            token = int(mx.argmax(logits[0, -1]).item())
            generated.append(token)
            if token == eos:
                break
            logits = model(mx.array([[token]]), cache)
        return generated, logits

    def rates(n):
        long = long_prompt_tokens(adapter, n)
        cache = model.make_cache()
        mx.synchronize()
        t0 = time.monotonic()
        for a in range(0, n, 512):
            hidden = model.model(mx.array([long[a:a + 512]]), cache)
            mx.eval(hidden)
        logits = model._project_logits(model.model.norm(hidden[:, -1:]))
        token = int(mx.argmax(logits[0, -1]).item())
        prefill = time.monotonic() - t0
        t1 = time.monotonic()
        for _ in range(64):
            logits = model(mx.array([[token]]), cache)
            token = int(mx.argmax(logits[0, -1]).item())
        return {"prefill_tok_s": round(n / prefill, 1), "decode_tok_s": round(64 / (time.monotonic() - t1), 2)}

    out = {}
    for label, policy in (("fused", dict(fused_projections=True)),
                          ("head", dict(head_kernel=True)),
                          ("fused_head", dict(fused_projections=True, head_kernel=True)),
                          ("sink", dict(sink_sdpa=True)),
                          ("all", dict(fused_projections=True, head_kernel=True, sink_sdpa=True))):
        model.configure_kernels(**policy)
        cases = []
        for (name, prompt, _), base in zip(PROMPTS, baseline["cases"]):
            tokens = adapter.prompt_tokens({"messages": [{"role": "user", "content": prompt}],
                                            "reasoning_effort": "none"})
            generated, _ = greedy(tokens, max_tokens)
            reference = base.get("output_token_ids")
            first = next((i for i, (x, y) in enumerate(zip(generated, reference)) if x != y), None)
            if first is None and len(generated) != len(reference):
                first = min(len(generated), len(reference))
            cases.append({"name": name, "exact_tokens": generated == reference,
                          "first_divergence": first, "tokens": len(generated)})
        out[label] = {"cases": cases, "receipt": model.kernel_receipt()}
        print("exact", label, [c["exact_tokens"] for c in cases], flush=True)
    arms = []
    for label, policy in (("A_ordinary", {}), ("B_fused", dict(fused_projections=True)),
                          ("C_head", dict(head_kernel=True)),
                          ("D_fused_head", dict(fused_projections=True, head_kernel=True)),
                          ("E_sink", dict(sink_sdpa=True)),
                          ("F_all", dict(fused_projections=True, head_kernel=True, sink_sdpa=True)),
                          ("A2_ordinary", {})):
        model.configure_kernels(**policy)
        arms.append({"arm": label, **{str(n): rates(n) for n in (4096, 16384)}})
        print("ab", arms[-1], flush=True)
    model.configure_kernels()
    out["aba_rates_b1"] = arms
    return out


def _post(port, path, body, token=None, timeout=900):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def server(model_path, run_dir, port):
    token = secrets.token_hex(16)
    token_file = run_dir / "admin.token"
    token_file.write_text(token)
    os.chmod(token_file, 0o600)
    log = open(run_dir / "server.log", "w")
    cmd = [sys.executable, "-m", "mlx2.server", "--model", str(model_path), "--host", "127.0.0.1",
           "--port", str(port), "--max-lanes", "2", "--cache-bytes", str(8 << 30),
           "--admin-token-file", str(token_file)]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    result = {"command": cmd}
    try:
        deadline = time.monotonic() + 900
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("server did not become healthy; see server.log")
            time.sleep(2)

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as r:
            model_id = json.loads(r.read())["data"][0]["id"]
        result["model_id"] = model_id

        def chat(content, max_tokens=160):
            body = {"model": model_id, "messages": [{"role": "user", "content": content}],
                    "temperature": 0, "max_tokens": max_tokens, "reasoning_effort": "none"}
            payload = _post(port, "/v1/chat/completions", body)
            message = payload["choices"][0]["message"]
            usage = payload.get("usage", {})
            return {"content": message.get("content"),
                    "reasoning_chars": len(message.get("reasoning_content") or ""),
                    "finish_reason": payload["choices"][0].get("finish_reason"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                    "receipt": payload.get("mlx2")}

        # Varied text, not a repeated passage (repeated inputs derail
        # reasoning models): this repo's Inkling 2 doc.
        story = ("Summarize the following document in two sentences.\n\n"
                 + (ROOT / "docs/INKLING2-MINI.md").read_text()[:6000])
        _post(port, "/v1/admin/apc/clear", {"reason": "inkling2-smoke-cold"}, token)
        cold = chat(story)
        warm = chat(story)
        result["apc_cold_warm"] = {
            "cold": cold, "warm": warm, "exact_text": cold["content"] == warm["content"],
            "warm_cached_tokens": warm["cached_tokens"], "prompt_tokens": warm["prompt_tokens"]}
        print("apc", result["apc_cold_warm"]["exact_text"], warm["cached_tokens"], "/",
              warm["prompt_tokens"], flush=True)

        prompts = ["Explain in three sentences why the sky looks blue.",
                   "List four uses for a paperclip, one per line."]
        _post(port, "/v1/admin/apc/clear", {"reason": "inkling2-smoke-solo"}, token)
        solo = [chat(p) for p in prompts]
        _post(port, "/v1/admin/apc/clear", {"reason": "inkling2-smoke-b2"}, token)
        pair = [None, None]

        def run(i):
            pair[i] = chat(prompts[i])

        threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        result["b2_vs_b1"] = [{"solo": s, "pair": p, "exact_text": s["content"] == p["content"]}
                              for s, p in zip(solo, pair)]
        print("b2", [r["exact_text"] for r in result["b2_vs_b1"]], flush=True)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/admin/state", timeout=10) as r:
                result["admin_state"] = json.loads(r.read())
        except OSError as error:
            result["admin_state_error"] = str(error)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        log.close()
        token_file.unlink(missing_ok=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--phase", choices=("direct", "server", "both"), default="both")
    parser.add_argument("--port", type=int, default=8391)
    parser.add_argument("--max-tokens", type=int, default=384)
    a = parser.parse_args(argv)
    if a.port == 8282:
        raise SystemExit("port 8282 is reserved for release trees")
    a.run_dir.mkdir(parents=True, exist_ok=True)
    if any(a.run_dir.iterdir()):
        raise SystemExit(f"{a.run_dir} is not empty; use a fresh run directory")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    from mlx2.adapters.inkling2 import REFERENCE_POLICY, Inkling2Adapter

    receipt = {"script": "scripts/smoke_inkling2.py", "mlx2": git_commit(), "mlx_version": mx.__version__,
               "swap_used_mb_before": swap_used_mb(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "qualification": "unqualified (smoke only; not a qualification run)"}
    if a.phase in ("direct", "both"):
        metal = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             str(ROOT / "tests/test_inkling2_metal.py")],
            capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
        receipt["metal_tests"] = {"returncode": metal.returncode, "tail": metal.stdout[-1500:]}
        print("metal tests rc", metal.returncode, flush=True)
        t0 = time.monotonic()
        adapter = Inkling2Adapter(str(a.model), execution_policy=dict(REFERENCE_POLICY))
        try:
            receipt["load_seconds"] = round(time.monotonic() - t0, 1)
            receipt["checkpoint"] = adapter.checkpoint
            receipt["active_memory_gb_after_load"] = round(mx.get_active_memory() / 2**30, 2)
            receipt["direct"] = direct(adapter, a.max_tokens)
            write_atomic(a.run_dir / "direct-partial.json", receipt)
            receipt["kernel_ab"] = fused_projection_ab(adapter, a.max_tokens, receipt["direct"])
            receipt["route_receipt"] = adapter.diagnostics()
            receipt["numerics_contract"] = adapter.execution_numerics_contract()
            receipt["peak_memory_gb"] = round(mx.get_peak_memory() / 2**30, 2)
        finally:
            adapter.close()
            del adapter
            mx.clear_cache()
        write_atomic(a.run_dir / "direct.json", receipt)
    if a.phase in ("server", "both"):
        receipt["server"] = server(a.model.expanduser().resolve(), a.run_dir, a.port)
    receipt["swap_used_mb_after"] = swap_used_mb()
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_atomic(a.run_dir / "smoke.json", receipt)
    ok = receipt.get("direct", {}).get("all_oracles_pass", True)
    print(json.dumps({"checkpoint": receipt.get("checkpoint"), "mlx2": receipt["mlx2"],
                      "all_oracles_pass": ok}, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
