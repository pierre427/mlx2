#!/usr/bin/env python3
"""Bounded upstream-EXL3 Qwen3.6 serving probe: ordinary reference, APCv2, B4.

One artifact per ownership window (run under the GPU lock chain); no thermal
holds or performance claim. Arms:

reference  greedy decode straight through the adapter, in its own process
cold/warm  the same long-context identifier request twice through a loopback
           server; warm must reuse cached prompt tokens and match cold
           exactly (APCv2 width-one replay gate) and the ordinary reference
b4         four distinct requests concurrently, each against its solo run;
           mismatches record the first divergent position (width > 1 may
           need near-tie classification, not done here)

Same lifecycle as scripts/probe_ripple_serving.py.
Usage: probe_exl3_serving.py --i-own-the-gpu --model DIR --execution rows --output NEW_DIR
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
# Distinct sentences: reasoning models degrade on repeated filler.
FACTS = [f"Ledger entry {n}: crate {(n * 37) % 1000:03d} moved from bay {n % 13} to bay {(n * 7) % 17}."
         for n in range(1, 61)]
LONG = {"role": "user", "content": "Warehouse log:\n" + "\n".join(FACTS)
        + "\nThe audit reference for this log is QK-5521. Reply with the audit reference only."}
SOLO = [
    "Name the largest planet in the solar system in one word.",
    "Write one sentence describing a rainbow.",
    "What is 17 multiplied by 23? Give the number only.",
    "Translate 'good morning' into French.",
]
MAX_TOKENS = 96


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def oracle(text):
    text = re.sub(r"[‐-―]", "-", text)
    return re.search(r"(?<!\w)QK-5521(?!\w)", text) is not None


def first_divergence(a, b):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def reference(args):
    """Ordinary greedy decode through the adapter (own process)."""
    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter

    adapter = resolve_adapter(args.model)(args.model, execution_policy={"exl3_execution": args.execution})
    # Model modules import only after the adapter pinned its profile.
    from mlx2.runtime.models.cache import make_prompt_cache

    tokenizer = adapter.tokenizer
    eos = set(tokenizer.eos_token_ids)
    out = {}
    for name, messages in [("long", [LONG])] + [(f"solo{i}", [{"role": "user", "content": p}])
                                                for i, p in enumerate(SOLO)]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        cache = make_prompt_cache(adapter.model)
        logits = adapter.model(mx.array([ids], mx.uint32), cache=cache)[:, -1]
        generated = []
        for _ in range(MAX_TOKENS):
            token = int(mx.argmax(logits, axis=-1).item())
            if token in eos:
                break
            generated.append(token)
            logits = adapter.model(mx.array([[token]], mx.uint32), cache=cache)[:, -1]
        out[name] = {"prompt_tokens": len(ids), "token_ids": generated, "text": tokenizer.decode(generated)}
    atomic_json(args.output / "ordinary-reference.json", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution", default="rows",
                        choices=("reference", "rows", "fused", "fused_nax", "fused_w8a8", "nax", "nax_fused"))
    parser.add_argument("--port", type=int, default=18361)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        parser.error("this runs Metal and a model server; pass --i-own-the-gpu under the GPU lock")
    if args.reference_only:
        return reference(args)

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    from mlx2.adapters.registry import inspect_model

    artifact = inspect_model(args.model).artifact
    if artifact.get("exl3_upstream") is None:
        raise SystemExit("not an upstream EXL3 checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    name = args.model.parent.parent.name.removeprefix("models--").replace("--", "/", 1) \
        if args.model.parent.name == "snapshots" else args.model.name
    receipt = {"schema": "mlx2.exl3-serving-probe.v1",
               "checkpoint": f"{name}@{artifact['identity']['fingerprint']}",
               "execution": args.execution, "qualified": False,
               "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "source_dirty": bool(subprocess.check_output(
                   ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)),
               "kind": "bounded correctness smoke", "rows": [], "b4": []}
    receipt["thermal_before"] = subprocess.check_output(["pmset", "-g", "therm"], text=True)
    receipt["swap_before"] = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    subprocess.run([sys.executable, __file__, "--i-own-the-gpu", "--model", str(args.model),
                    "--output", str(args.output), "--execution", args.execution, "--reference-only"],
                   check=True, timeout=1800)
    ordinary = json.loads((args.output / "ordinary-reference.json").read_text())
    receipt["ordinary_reference"] = ordinary
    root = f"http://127.0.0.1:{args.port}"

    def get(path):
        with urlopen(root + path, timeout=20) as response:
            return json.load(response)

    served = {}
    eos_ids = set()
    for name_ in ("config.json", "generation_config.json"):
        item = args.model / name_
        if item.is_file():
            data = json.loads(item.read_text())
            for source in (data, data.get("text_config") or {}):
                value = source.get("eos_token_id")
                eos_ids.update(value if isinstance(value, list) else [value] if value is not None else [])

    def chat(messages):
        payload = {"model": served["model"], "messages": messages, "temperature": 0, "max_tokens": MAX_TOKENS,
                   "return_token_trace": True, "stream": False,
                   "chat_template_kwargs": {"enable_thinking": False}}
        request = Request(root + "/v1/chat/completions", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=1800) as response:
            result = json.load(response)
        trace = list(result["mlx2"]["token_trace"]["token_ids"])
        if trace and trace[-1] in eos_ids:
            trace.pop()  # the reference stops before EOS
        return result, trace

    policy = args.output / "policy.json"
    atomic_json(policy, {"exl3_execution": args.execution})
    command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(args.model), "--host", "127.0.0.1",
               "--port", str(args.port), "--max-lanes", "4", "--max-inflight", "4", "--max-context", "8192",
               "--cache-bytes", str(2 << 30), "--cache-dir", str(args.output / "apcv2"),
               "--execution-policy", str(policy)]
    log = (args.output / "server.log").open("w")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                              env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    receipt["server_command"] = command
    try:
        deadline = time.monotonic() + 900
        while True:
            if server.poll() is not None:
                raise RuntimeError("EXL3 server exited before readiness")
            try:
                status = get("/v1/status")
                if status.get("healthy"):
                    break
            except (URLError, TimeoutError, ConnectionError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("EXL3 server readiness timeout")
            time.sleep(1)
        receipt["startup_status"] = status
        served["model"] = status["model"]  # the server's own model id
        for arm in ("cold", "warm"):
            result, trace = chat([LONG])
            content = result["choices"][0]["message"].get("content") or ""
            receipt["rows"].append({
                "arm": arm, "oracle_passed": oracle(content), "content": content, "token_ids": trace,
                "reference_first_divergence": first_divergence(trace, ordinary["long"]["token_ids"]),
                "usage": result.get("usage"), "route": result["mlx2"].get("route"),
                "mlx2": {k: v for k, v in result["mlx2"].items() if k != "token_trace"},
            })
        solo = {}
        for i, prompt in enumerate(SOLO):
            solo[i] = chat([{"role": "user", "content": prompt}])[1]
        concurrent = {}

        def worker(i, prompt):
            concurrent[i] = chat([{"role": "user", "content": prompt}])[1]

        threads = [threading.Thread(target=worker, args=(i, p)) for i, p in enumerate(SOLO)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for i in range(len(SOLO)):
            receipt["b4"].append({
                "prompt": SOLO[i], "solo_vs_concurrent_first_divergence": first_divergence(solo[i], concurrent[i]),
                "solo_vs_reference_first_divergence": first_divergence(solo[i], ordinary[f"solo{i}"]["token_ids"]),
                "solo_text": ordinary[f"solo{i}"]["text"],
            })
        receipt["final_status"] = get("/v1/status")
        exl3 = (receipt["final_status"].get("execution") or {}).get("exl3") or {}
        linears = exl3.get("observed_linears") or {}
        blocks = exl3.get("observed_blocks") or []
        calls = "reference_calls" if args.execution == "reference" else "rows_calls"
        receipt["route_observed"] = {
            "execution": exl3.get("execution"),
            "linears_called": linears.get("called"), "linears": linears.get("modules"),
            "laws_called": linears.get("laws_called"),
            "switches_called": sum(1 for b in blocks if b.get(calls) or b.get("nax_calls") or b.get("chain_calls")),
            "switches": len(blocks),
            "switches_on_nax": sum(1 for b in blocks if b.get("nax_calls")),
        }
        observed = receipt["route_observed"]
        route_ok = (observed["execution"] == args.execution and observed["linears"]
                    and observed["linears_called"] == observed["linears"]
                    and observed["laws_called"] == [{"nax": "fused_nax", "nax_fused": "fused_nax"}.get(args.execution, args.execution)]
                    and observed["switches"] and observed["switches_called"] == observed["switches"])
        cold, warm = receipt["rows"]
        cached = ((warm.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        receipt["warm_cached_tokens"] = cached
        receipt["cold_warm_exact"] = cold["token_ids"] == warm["token_ids"]
        receipt["passed"] = (all(r["oracle_passed"] and r["reference_first_divergence"] is None for r in receipt["rows"])
                             and receipt["cold_warm_exact"] and cached > 0 and bool(route_ok))
        receipt["b4_exact"] = all(r["solo_vs_concurrent_first_divergence"] is None for r in receipt["b4"])
    except BaseException as exc:
        receipt.update(passed=False, error=repr(exc))
        raise
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=15)
        log.close()
        receipt["server_stopped"] = server.poll() is not None
        receipt["thermal_after"] = subprocess.check_output(["pmset", "-g", "therm"], text=True)
        receipt["swap_after"] = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
        atomic_json(args.output / "receipt.json", receipt)
    print(json.dumps({"passed": receipt["passed"], "b4_exact": receipt["b4_exact"],
                      "checkpoint": receipt["checkpoint"]}), flush=True)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
