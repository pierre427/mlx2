#!/usr/bin/env python3
"""Kimi multi-lane (left-padded) decode over one no-flag loopback server.

Four prompts of very different lengths (distinct documentation text, each
with its own planted fact and question) run one at a time (the B1 reference)
and then concurrently, so the batch cache left-pads the shorter lanes. Each
concurrent output is classified against its B1 reference with the AGENTS.md
first-divergence near-tie contract; MLA flash counters (latent_flash,
latent_flash_padded, flash_declined) and decode rates are recorded.
Server flags follow the production Kimi service (4 lanes). Caller owns the
CPG lease and locks.
"""
import argparse
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))
from kimi_linear_speculative import (
    classify,
    find,
    metric_values,
    swapouts_mb,
    token_rows,
)

FACTS = (("release codename", "ORCHID-7431"), ("backup site", "Kestrel Ridge"),
         ("on-call rotation length", "eleven days"), ("migration ticket", "MIG-20584"))
SOURCES = ("docs/SERVING.md", "docs/METRICS.md", "docs/QUALIFICATION.md", "docs/PROVENANCE.md",
           "README.md", "docs/KERNEL-RELOAD.md")
TARGET_CHARS = (4_000, 16_000, 48_000, 110_000)  # ~1K / 4K / 12K / 28K tokens
LONG_CHARS = (8_000, 64_000, 192_000, 400_000)  # ~2K / 16K / 48K / 100K tokens


def prompts(long=False):
    text = "\n\n".join((ROOT / path).read_text() for path in SOURCES if (ROOT / path).is_file())
    cases, start = [], 0
    for (topic, value), size in zip(FACTS, LONG_CHARS if long else TARGET_CHARS):
        if start + size > len(text):
            start = 0
        body = text[start:start + size]
        start += size
        cut = len(body) // 3
        body = body[:cut] + f"\n\nNote for this deployment: the {topic} is {value}.\n\n" + body[cut:]
        tail = (" Then summarize what this part of the documentation covers in about 200 words."
                if long else "")
        content = (f"Here is part of the project documentation.\n\n{body}\n\n"
                   f"Question: according to the note above, what is the {topic}? "
                   "Answer in one short sentence." + tail)
        cases.append((f"lane_{size // 4000}k", content, value))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpg-lock", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18296)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--top-logprobs", type=int, default=4)
    parser.add_argument("--long", action="store_true",
                        help="~2K-100K-token lanes, ~200-word answers, streamed overlap-window decode rate")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("refusing nonempty evidence directory")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from qualification_gpu_ownership import require_qualification_lease
    owner = json.loads((args.cpg_lock / "owner.json").read_text())
    lease = require_qualification_lease(task_id=owner["cpg_task"], generation=owner["cpg_generation"],
                                        cpg_owner_lock=args.cpg_lock)
    from mlx2.adapters.kimi_linear import checkpoint_name, inspect_artifact
    target = inspect_artifact(args.model)
    receipt = {
        "checkpoint": f"{checkpoint_name(args.model)}:{target['identity']['fingerprint']}",
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                                     text=True, cwd=ROOT).strip()),
        "ownership": lease, "qualified": False, "width": 4, "max_tokens": args.max_tokens,
        "kind": "multi-lane left-padded decode vs B1 reference, no thermal hold", "cases": []}
    token_file = args.output / "admin-token"
    token_file.write_text(secrets.token_hex(24))
    token_file.chmod(0o600)
    token = token_file.read_text()
    root = f"http://127.0.0.1:{args.port}"

    def call(path, payload=None, *, admin=False, timeout=1800):
        headers = {"Content-Type": "application/json"}
        if admin:
            headers["Authorization"] = "Bearer " + token
        data = None if payload is None else json.dumps(payload).encode()
        with urlopen(Request(root + path, data=data, headers=headers), timeout=timeout) as response:
            body = response.read()
        return json.loads(body) if body.strip().startswith(b"{") else body.decode()

    def chat(content, *, logprobs):
        payload = {"model": args.model.name, "temperature": 0, "stream": False,
                   "max_tokens": args.max_tokens, "messages": [{"role": "user", "content": content}]}
        if logprobs:
            payload.update(logprobs=True, top_logprobs=args.top_logprobs)
        start = time.monotonic()
        result = call("/v1/chat/completions", payload)
        elapsed = time.monotonic() - start
        usage, ttft = result.get("usage", {}), find(result, "ttft_seconds")
        completion = usage.get("completion_tokens", 0)
        return result, {"content": result["choices"][0]["message"].get("content", ""),
                        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": completion,
                        "start": start, "elapsed_seconds": elapsed, "ttft_seconds": ttft,
                        "decode_tps": (completion - 1) / (elapsed - ttft)
                        if ttft and completion > 1 and elapsed > ttft else None}

    def stream(content):
        payload = {"model": args.model.name, "temperature": 0, "stream": True,
                   "max_tokens": args.max_tokens, "messages": [{"role": "user", "content": content}]}
        request = Request(root + "/v1/chat/completions", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        stamps, text = [], []
        with urlopen(request, timeout=1800) as response:
            for line in response:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                delta = (json.loads(line[5:])["choices"] or [{}])[0].get("delta", {})
                if delta.get("content"):
                    stamps.append(time.monotonic())
                    text.append(delta["content"])
        return stamps, "".join(text)

    def mla_counters():
        values = metric_values(call("/metrics"))
        return {k.split('event="')[1].split('"')[0]: v for k, v in values.items()
                if k.startswith("mlx2_mla_events_total{")}

    def concurrent(cases, *, logprobs):
        results = [None] * len(cases)

        def run(i, text):
            results[i] = chat(text, logprobs=logprobs)
        threads = [threading.Thread(target=run, args=(i, text)) for i, (_, text, _) in enumerate(cases)]
        wall = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return results, time.monotonic() - wall

    command = [sys.executable, "-m", "mlx2.server", "--model", str(args.model), "--host", "127.0.0.1",
               "--port", str(args.port), "--max-context", "262144", "--default-max-tokens", "8192",
               "--max-lanes", "4", "--max-inflight", "8", "--cache-bytes", str(4 << 30),
               "--cache-dir", str(args.output / "cache"), "--agent-compat", "on",
               "--admin-token-file", str(token_file)]
    (args.output / "command.json").write_text(json.dumps(command))
    log = (args.output / "server.log").open("w")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "MLX2_CACHE_LIMIT_GIB": "8"}
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                              env=env, cwd=ROOT)
    receipt["server_pid"] = server.pid
    try:
        swap0 = swapouts_mb()  # inside try: a failed sample still stops the server
        deadline = time.monotonic() + 900
        while True:
            if server.poll() is not None:
                raise RuntimeError("server exited before readiness (see server.log)")
            try:
                status = call("/v1/status", timeout=10)
                if status.get("ready") or status.get("state") == "ready" or status.get("status") == "ready":
                    break
            except (URLError, TimeoutError, ConnectionError, OSError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("server readiness timed out")
            time.sleep(1)
        receipt["route"] = {k: status.get(k) for k in ("profile", "qualification", "route_receipt")}
        cases = prompts(args.long)
        chat("Say hello in one word.", logprobs=False)
        reference = {}
        for name, text, fact in cases:
            call("/v1/admin/apc/clear", {"reason": "padded-lanes-reference"}, admin=True)
            raw, evidence = chat(text, logprobs=True)
            (args.output / f"{name}.b1.logprobs.json").write_text(json.dumps(raw, indent=1, default=str))
            reference[name] = (token_rows(raw), fact in evidence["content"], evidence)
        call("/v1/admin/apc/clear", {"reason": "padded-lanes-concurrent"}, admin=True)
        before = mla_counters()
        results, _ = concurrent(cases, logprobs=True)
        time.sleep(1.5)  # one worker publication interval
        receipt["mla_counters_concurrent_logprob_pass"] = {
            k: v - before.get(k, 0) for k, v in mla_counters().items() if v != before.get(k, 0)}
        call("/v1/admin/apc/clear", {"reason": "padded-lanes-timed"}, admin=True)
        before = mla_counters()
        timed, wall = concurrent(cases, logprobs=False)
        time.sleep(1.5)
        receipt["mla_counters_concurrent_timed_pass"] = {
            k: v - before.get(k, 0) for k, v in mla_counters().items() if v != before.get(k, 0)}
        receipt["timed_wall_seconds"] = wall
        if args.long:
            # Streamed pass: aggregate decode rate only while all lanes decode
            # (after the last lane's first token, before the first lane ends).
            call("/v1/admin/apc/clear", {"reason": "padded-lanes-stream"}, admin=True)
            streamed = [None] * len(cases)

            def run_stream(i, text):
                streamed[i] = stream(text)
            threads = [threading.Thread(target=run_stream, args=(i, text))
                       for i, (_, text, _) in enumerate(cases)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            lo = max(stamps[0] for stamps, _ in streamed if stamps)
            hi = min(stamps[-1] for stamps, _ in streamed if stamps)
            inside = sum(sum(lo <= t <= hi for t in stamps) for stamps, _ in streamed)
            receipt["overlap_window"] = {"seconds": hi - lo, "chunks": inside,
                                         "aggregate_chunks_per_second": inside / (hi - lo) if hi > lo else None,
                                         "per_lane_chunks": [len(stamps) for stamps, _ in streamed],
                                         "oracles": [fact in text for (_, _, fact), (_, text)
                                                     in zip(cases, streamed)]}
        receipt["timed_completion_tokens"] = sum(t[1]["completion_tokens"] for t in timed)
        for (name, text, fact), (raw, evidence), (_, timing) in zip(cases, results, timed):
            (args.output / f"{name}.b4.logprobs.json").write_text(json.dumps(raw, indent=1, default=str))
            ref_tokens, ref_ok, ref_evidence = reference[name]
            ok = fact in evidence["content"]
            receipt["cases"].append({
                "case": name, "prompt_tokens": evidence["prompt_tokens"], "oracle": ok,
                "reference_oracle": ref_ok, "timed_oracle": fact in timing["content"],
                "b1_decode_tps": ref_evidence["decode_tps"], "b4_decode_tps": timing["decode_tps"],
                "b4_ttft_seconds": timing["ttft_seconds"],
                "vs_b1": classify(ref_tokens, token_rows(raw), ref_ok, ok),
                "content": evidence["content"]})
            print(json.dumps({k: receipt["cases"][-1][k] for k in ("case", "prompt_tokens", "oracle",
                                                                   "b4_decode_tps")}
                             | {"verdict": receipt["cases"][-1]["vs_b1"]["verdict"]}), flush=True)
        receipt["swapouts_mb"] = swapouts_mb() - swap0
        (args.output / "status-final.json").write_text(json.dumps(call("/v1/status"), indent=2, default=str))
        (args.output / "metrics.prom").write_text(call("/metrics"))
        # The timed and streamed passes are measured outputs too, and the
        # feature under test must have engaged in both concurrent passes.
        padded = [receipt[k].get("latent_flash_padded", 0) > 0 for k in (
            "mla_counters_concurrent_logprob_pass", "mla_counters_concurrent_timed_pass")]
        receipt["padded_flash_engaged"] = padded
        receipt["passed"] = all(c["oracle"] and c["reference_oracle"] and c["timed_oracle"]
                                for c in receipt["cases"]) and all(
            c["vs_b1"]["verdict"] in ("exact", "near_tie_equivalent") for c in receipt["cases"]) and all(
            receipt.get("overlap_window", {}).get("oracles", [True])) and all(padded)
    except BaseException as error:
        receipt["passed"] = False
        receipt["error"] = repr(error)
        raise
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
        log.close()
        token_file.unlink(missing_ok=True)
        tmp = args.output / "receipt.tmp"
        tmp.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
        os.replace(tmp, args.output / "receipt.json")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
