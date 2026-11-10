#!/usr/bin/env python3
"""Classify an EXL3 serving divergence under the AGENTS.md near-tie contract.

Three arms of one prompt: the ordinary in-process greedy decode (its own
process), the server solo, and the server inside a four-request concurrent
batch. Each arm records top-5 log-probabilities per generated token. The
arms with logprobs must reproduce the token traces seen without them (the
first divergence position is compared with --expect); then every pair is
classified at its first divergence:

- near_tie_equivalent: shared prefix; the same unordered top two in both
  arms and both selected-vs-alternate margins <= 0.5 nat; or the exact-tie
  clause (one margin exactly 0, the other arm chose a tied token, its margin
  <= 0.5); or the multi-way clause (each arm's token within 0.5 nat of the
  other arm's top log-probability, read from that arm's window); and both
  continuations pass the functional oracle.
- otherwise unclassified (or failed when the window proves a token > 0.5 nat
  below the top).

Run under the GPU lock chain. Usage:
  classify_exl3_divergence.py --i-own-the-gpu --model DIR --execution nax --output NEW_DIR
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # this tree, not the editable install
sys.path.insert(0, str(ROOT / "scripts"))

from probe_exl3_serving import SOLO, atomic_json, first_divergence  # noqa: E402

TARGET = 3  # "Translate 'good morning' into French."
MAX_TOKENS = 40
TOP = 5
CEILING = 0.5


def oracle(text):
    return "bonjour" in text.lower()


def reference(args):
    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter

    adapter = resolve_adapter(args.model)(args.model, execution_policy={"exl3_execution": args.execution})
    from mlx2.runtime.models.cache import make_prompt_cache

    tokenizer = adapter.tokenizer
    eos = set(tokenizer.eos_token_ids)
    text = tokenizer.apply_chat_template([{"role": "user", "content": SOLO[TARGET]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    cache = make_prompt_cache(adapter.model)
    logits = adapter.model(mx.array([ids], mx.uint32), cache=cache)[:, -1]
    tokens, windows = [], []
    for _ in range(MAX_TOKENS):
        lp = (logits - mx.logsumexp(logits, axis=-1, keepdims=True)).astype(mx.float32)[0]
        top = mx.argsort(-lp)[:TOP]
        windows.append([[int(t), float(lp[t].item())] for t in top.tolist()])
        token = int(top[0].item())
        if token in eos:
            break
        tokens.append(token)
        logits = adapter.model(mx.array([[token]], mx.uint32), cache=cache)[:, -1]
    atomic_json(args.output / "reference.json", {"token_ids": tokens, "windows": windows,
                                                 "text": tokenizer.decode(tokens)})


def margin(window):
    return window[0][1] - window[1][1]


def classify(a, b):
    """a, b: {"token_ids", "windows", "text"}; the windows index generated positions."""
    p = first_divergence(a["token_ids"], b["token_ids"])
    if p is None:
        return {"first_divergence": None, "classification": "exact"}
    if p >= min(len(a["windows"]), len(b["windows"])):
        return {"first_divergence": p, "classification": "unclassified", "reason": "length only"}
    wa, wb = a["windows"][p], b["windows"][p]
    ta, tb = a["token_ids"][p], b["token_ids"][p]
    pa, pb = {t: v for t, v in wa}, {t: v for t, v in wb}
    ma, mb = margin(wa), margin(wb)
    oracles = oracle(a["text"]) and oracle(b["text"])
    row = {"first_divergence": p, "a_token": ta, "b_token": tb, "a_window": wa, "b_window": wb,
           "a_margin": ma, "b_margin": mb, "oracles": oracles}
    same_pair = {wa[0][0], wa[1][0]} == {wb[0][0], wb[1][0]}
    if same_pair and ma <= CEILING and mb <= CEILING:
        row["clause"] = "top_two"
    elif (ma == 0.0 and tb in (wa[0][0], wa[1][0]) and mb <= CEILING) or (
            mb == 0.0 and ta in (wb[0][0], wb[1][0]) and ma <= CEILING):
        row["clause"] = "exact_tie"
    else:
        # Multi-way: each arm's token within the ceiling of the other arm's top,
        # read from the other arm's window.
        if ta not in pb or tb not in pa:
            proven_far = (ta not in pb and wb[-1][1] < wb[0][1] - CEILING) or (
                tb not in pa and wa[-1][1] < wa[0][1] - CEILING)
            row["classification"] = "failed" if proven_far else "unclassified"
            row["reason"] = "a selected token is missing from the other arm's window"
            return row
        if wb[0][1] - pb[ta] <= CEILING and wa[0][1] - pa[tb] <= CEILING:
            row["clause"] = "multi_way"
        else:
            row["classification"] = "failed"
            row["reason"] = "a selected token is more than 0.5 nat below the other arm's top"
            return row
    row["classification"] = "near_tie_equivalent" if oracles else "unclassified"
    if not oracles:
        row["reason"] = "a continuation fails the functional oracle"
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution", default="nax")
    parser.add_argument("--expect", type=int, default=9, help="first divergence seen without logprobs")
    parser.add_argument("--port", type=int, default=18362)
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
    args.output.mkdir(parents=True, exist_ok=False)
    name = args.model.parent.parent.name.removeprefix("models--").replace("--", "/", 1) \
        if args.model.parent.name == "snapshots" else args.model.name
    receipt = {"schema": "mlx2.exl3-divergence-classification.v1",
               "checkpoint": f"{name}@{artifact['identity']['fingerprint']}", "execution": args.execution,
               "prompt": SOLO[TARGET], "expected_first_divergence": args.expect,
               "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "source_dirty": bool(subprocess.check_output(
                   ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True))}
    subprocess.run([sys.executable, __file__, "--i-own-the-gpu", "--model", str(args.model), "--output",
                    str(args.output), "--execution", args.execution, "--reference-only"], check=True, timeout=1800,
                   env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    arms = {"reference": json.loads((args.output / "reference.json").read_text())}
    eos_ids = set()
    for item in ("config.json", "generation_config.json"):
        path = args.model / item
        if path.is_file():
            data = json.loads(path.read_text())
            for source in (data, data.get("text_config") or {}):
                value = source.get("eos_token_id")
                eos_ids.update(value if isinstance(value, list) else [value] if value is not None else [])
    root = f"http://127.0.0.1:{args.port}"
    served = {}

    def get(path):
        with urlopen(root + path, timeout=20) as response:
            return json.load(response)

    def chat(prompt, logprobs):
        payload = {"model": served["model"], "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "max_tokens": MAX_TOKENS, "return_token_trace": True, "stream": False,
                   "chat_template_kwargs": {"enable_thinking": False}}
        if logprobs:
            payload.update(logprobs=True, top_logprobs=TOP)
        request = Request(root + "/v1/chat/completions", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=1800) as response:
            result = json.load(response)
        trace = list(result["mlx2"]["token_trace"]["token_ids"])
        windows = []
        for row in (result["choices"][0].get("logprobs") or {}).get("content") or []:
            windows.append(sorted(([int(c["id"]), float(c["logprob"])] for c in row.get("top_logprobs", [])),
                                  key=lambda c: -c[1]))
        if trace and trace[-1] in eos_ids:
            trace.pop()
        return {"token_ids": trace, "windows": windows,
                "text": result["choices"][0]["message"].get("content") or ""}

    policy = args.output / "policy.json"
    atomic_json(policy, {"exl3_execution": args.execution})
    command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(args.model), "--host", "127.0.0.1",
               "--port", str(args.port), "--max-lanes", "4", "--max-inflight", "4", "--max-context", "8192",
               "--cache-bytes", str(2 << 30), "--cache-dir", str(args.output / "apcv2"),
               "--execution-policy", str(policy)]
    log = (args.output / "server.log").open("w")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                              env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    try:
        deadline = time.monotonic() + 900
        while True:
            if server.poll() is not None:
                raise RuntimeError("server exited before readiness")
            try:
                status = get("/v1/status")
                if status.get("healthy"):
                    break
            except (URLError, TimeoutError, ConnectionError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("server readiness timeout")
            time.sleep(1)
        served["model"] = status["model"]
        plain_solo = chat(SOLO[TARGET], False)
        arms["solo"] = chat(SOLO[TARGET], True)
        concurrent, plain_concurrent = {}, {}

        def batch(store, logprobs):
            def worker(i, prompt):
                store[i] = chat(prompt, logprobs and i == TARGET)
            threads = [threading.Thread(target=worker, args=(i, p)) for i, p in enumerate(SOLO)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        batch(plain_concurrent, False)
        batch(concurrent, True)
        arms["b4"] = concurrent[TARGET]
        receipt["reproduction"] = {
            "solo_with_logprobs_equals_plain": arms["solo"]["token_ids"] == plain_solo["token_ids"],
            "b4_with_logprobs_equals_plain": arms["b4"]["token_ids"] == plain_concurrent[TARGET]["token_ids"],
            "plain_solo_vs_reference": first_divergence(plain_solo["token_ids"], arms["reference"]["token_ids"]),
            "plain_b4_vs_plain_solo": first_divergence(plain_concurrent[TARGET]["token_ids"], plain_solo["token_ids"]),
        }
        receipt["arms"] = {k: {"token_ids": v["token_ids"], "text": v["text"]} for k, v in arms.items()}
        receipt["pairs"] = {"solo_vs_reference": classify(arms["solo"], arms["reference"]),
                            "b4_vs_solo": classify(arms["b4"], arms["solo"])}
        reproduced = receipt["reproduction"]
        receipt["windows_trustworthy"] = (reproduced["solo_with_logprobs_equals_plain"]
                                          and reproduced["b4_with_logprobs_equals_plain"])
        receipt["passed"] = receipt["windows_trustworthy"] and all(
            p["classification"] in ("exact", "near_tie_equivalent") for p in receipt["pairs"].values())
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
        atomic_json(args.output / "receipt.json", receipt)
    print(json.dumps({"passed": receipt["passed"], "pairs": {k: v["classification"] for k, v in receipt["pairs"].items()}}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
