#!/usr/bin/env python3
"""Kimi Linear execution-policy A/B over loopback servers (visible to mlx2.top).

One server per arm, one at a time, same prompts, greedy decoding. Each arm
records cold/warm APCv2 replay, a long varied-text prefill with a retrieval
oracle, a decode-rate request, the paired answer-quality tasks, and the
server's own /metrics. No thermal holds: sanity/attribution numbers, not
publishable statistics. Requires the caller's CPG lease and paired locks;
every server process group is stopped in ``finally``.
"""
import argparse
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]


def long_prompt(chars, offset=0, code="QV-4471"):
    # Realistic, varied filler (never one repeated sentence): project docs.
    text, needle = [], f"The archive release code is {code}."
    for path in sorted((ROOT/"docs").rglob("*.md"))[offset:]:
        # Prose only: a slice ending inside a code fence gets continued as
        # code instead of answered (ab-r3/r4, identically in every arm).
        prose, fenced = [], False
        for line in path.read_text(errors="ignore").splitlines():
            if line.lstrip().startswith("```"):
                fenced = not fenced
            elif not fenced:
                prose.append(line)
        text.append("\n".join(prose))
        if sum(map(len, text)) > chars:
            break
    body = "\n".join(text)[:chars]
    middle = len(body)//2
    return (body[:middle]+"\n"+needle+"\n"+body[middle:]
            +"\n\nWhat is the archive release code? Reply with the code only."), code


def find(document, key):
    if isinstance(document, dict):
        if key in document:
            return document[key]
        for value in document.values():
            found = find(value, key)
            if found is not None:
                return found
    elif isinstance(document, list):
        for value in document:
            found = find(value, key)
            if found is not None:
                return found
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpg-lock", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True, help="JSON list of {name, policy}")
    parser.add_argument("--port", type=int, default=18292)
    parser.add_argument("--long-chars", type=int, default=32000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--deep-chars", type=int, default=128000)
    parser.add_argument("--deeper-chars", type=int, default=0, help="optional second deep cell")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("refusing nonempty A/B evidence directory")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kimi_linear_norm_quality import answer_oracle, oracle, tasks
    from qualification_gpu_ownership import require_qualification_lease
    owner = json.loads((args.cpg_lock/"owner.json").read_text())
    lease = require_qualification_lease(task_id=owner["cpg_task"], generation=owner["cpg_generation"],
                                        cpg_owner_lock=args.cpg_lock)
    from mlx2.adapters.kimi_linear import checkpoint_name, inspect_artifact
    arms = json.loads(args.arms.read_text())
    receipt = {"checkpoint": checkpoint_name(args.model)+":"+inspect_artifact(args.model)["identity"]["fingerprint"][:16],
               "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
               "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                                            text=True, cwd=ROOT).strip()),
               "ownership": lease, "qualified": False, "kind": "loopback policy A/B, no thermal hold",
               "arms": [], "thermal_before": subprocess.check_output(["pmset", "-g", "therm"], text=True).strip(),
               "swap_before": subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()}
    token_file = args.output/"admin-token"
    token_file.write_text(secrets.token_hex(24))
    token_file.chmod(0o600)
    token = token_file.read_text()
    root = f"http://127.0.0.1:{args.port}"

    def call(path, payload=None, *, admin=False, timeout=900):
        headers = {"Content-Type": "application/json"}
        if admin:
            headers["Authorization"] = "Bearer "+token
        data = None if payload is None else json.dumps(payload).encode()
        with urlopen(Request(root+path, data=data, headers=headers), timeout=timeout) as response:
            body = response.read()
        return json.loads(body) if body.strip().startswith(b"{") else body.decode()

    def chat(content, max_tokens):
        start = time.monotonic()
        result = call("/v1/chat/completions", {"model": args.model.name, "temperature": 0, "stream": False,
                      "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]})
        elapsed = time.monotonic()-start
        usage = result.get("usage", {})
        ttft = find(result, "ttft_seconds")
        completion = usage.get("completion_tokens", 0)
        return {"content": result["choices"][0]["message"].get("content", ""),
                "finish_reason": result["choices"][0].get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": completion,
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "elapsed_seconds": elapsed, "ttft_seconds": ttft,
                "prefill_tps": usage.get("prompt_tokens", 0)/ttft if ttft else None,
                "decode_tps": (completion-1)/(elapsed-ttft) if ttft and completion > 1 and elapsed > ttft else None}

    longs = [long_prompt(args.long_chars, offset, code)
             for offset, code in ((0, "QV-4471"), (40, "HB-2093"), (80, "TW-5518"))]
    deep, deep_code = long_prompt(args.deep_chars, 120, "KM-7730")
    deeper = long_prompt(args.deeper_chars, 200, "PX-6614") if args.deeper_chars else None
    decodes = ["Write a detailed, multi-paragraph history of the printing press.",
               "Explain, step by step and at length, how a refrigerator moves heat.",
               "Describe in detail the life cycle of a star like the Sun."]
    quality = tasks()
    for arm in arms:
        out = args.output/arm["name"]
        out.mkdir()
        (out/"policy.json").write_text(json.dumps(arm["policy"]))
        command = [sys.executable, "-m", "mlx2.server", "--model", str(args.model), "--host", "127.0.0.1",
                   "--port", str(args.port), "--max-lanes", "2", "--max-inflight", "2",
                   "--cache-bytes", str(4 << 30), "--cache-dir", str(out/"cache"),
                   "--admin-token-file", str(token_file), "--execution-policy", str(out/"policy.json")]
        log = (out/"server.log").open("w")
        env = {**os.environ, "PYTHONPATH": str(ROOT/"src")}
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env, cwd=ROOT)
        row = {"name": arm["name"], "policy": arm["policy"], "server_pid": server.pid}
        try:
            deadline = time.monotonic()+300
            while True:
                if server.poll() is not None:
                    raise RuntimeError("server exited before readiness")
                try:
                    status = call("/v1/status", timeout=10)
                    if status.get("ready") or status.get("state") == "ready" or status.get("status") == "ready":
                        break
                except (URLError, TimeoutError, ConnectionError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("server readiness timed out")
                time.sleep(.5)
            row["route"] = {k: status.get(k) for k in ("profile", "qualification", "route_receipt", "max_context")}
            chat("Say hello in one word.", 4)
            call("/v1/admin/apc/clear", {"reason": "ab-cell"}, admin=True)
            replay = "Shipment log for the night audit.\nThe shipment identifier is ZX-3187.\nReply with the shipment identifier only."
            row["cold"], row["warm"] = chat(replay, 24), chat(replay, 24)
            row["cold_warm_identical"] = row["cold"]["content"] == row["warm"]["content"]
            call("/v1/admin/apc/clear", {"reason": "ab-cell"}, admin=True)
            row["long_runs"] = []
            for text, code in longs:
                call("/v1/admin/apc/clear", {"reason": "ab-cell"}, admin=True)
                got = chat(text, 64)  # room past an opening code fence
                got["oracle"] = code in got["content"].replace("–", "-")
                row["long_runs"].append(got)
            row["decode_runs"] = []
            for text in decodes:
                call("/v1/admin/apc/clear", {"reason": "ab-cell"}, admin=True)
                row["decode_runs"].append(chat(text, args.max_tokens))
            call("/v1/admin/apc/clear", {"reason": "ab-cell"}, admin=True)
            row["deep"] = chat(deep+"\nThen, after the code, write a long summary of the text.", 160)
            row["deep"]["oracle"] = deep_code in row["deep"]["content"].replace("–", "-")
            if deeper:
                call("/v1/admin/apc/clear", {"reason": "ab-cell"}, admin=True)
                row["deeper"] = chat(deeper[0]+"\nThen, after the code, write a long summary of the text.", 96)
                row["deeper"]["oracle"] = deeper[1] in row["deeper"]["content"].replace("–", "-")
            median = lambda xs: sorted(xs)[len(xs)//2]
            row["long"] = {"prefill_tps": median([r["prefill_tps"] for r in row["long_runs"]]),
                           "oracle": all(r["oracle"] for r in row["long_runs"])}
            row["decode"] = {"decode_tps": median([r["decode_tps"] for r in row["decode_runs"]])}
            row["quality"] = []
            for name, task, expected in quality:
                call("/v1/admin/apc/clear", {"reason": "ab-cell"}, admin=True)
                got = chat(task, args.max_tokens)
                row["quality"].append({"task": name, "content": got["content"], "finish_reason": got["finish_reason"],
                                       "strict": oracle(got["content"], expected),
                                       "answer": answer_oracle(got["content"], expected)})
            # Bounded first-divergence window: only the first answer token of
            # the known near-tie prompt carries a top-5 log-probability row.
            call("/v1/admin/apc/clear", {"reason": "ab-cell"}, admin=True)
            probe = call("/v1/chat/completions", {"model": args.model.name, "temperature": 0, "stream": False,
                         "max_tokens": 1, "logprobs": True, "top_logprobs": 5,
                         "messages": [{"role": "user", "content": dict((n, t) for n, t, _ in quality)["capital"]}]})
            row["capital_first_token"] = probe["choices"][0].get("logprobs")
            row["quality_answer"] = sum(q["answer"] for q in row["quality"])
            row["quality_strict"] = sum(q["strict"] for q in row["quality"])
            time.sleep(1.5)  # one worker publication interval
            row["execution"] = call("/v1/status").get("execution", {})
            (out/"metrics.prom").write_text(call("/metrics"))
            row["passed"] = bool(row["cold"]["content"] and row["long"]["oracle"])
        except BaseException as error:
            row["passed"] = False
            row["error"] = repr(error)
            raise
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=5)
            log.close()
            row["server_stopped"] = server.poll() is not None
            receipt["arms"].append(row)
            receipt["swap_after"] = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()
            receipt["thermal_after"] = subprocess.check_output(["pmset", "-g", "therm"], text=True).strip()
            tmp = args.output/"receipt.tmp"
            tmp.write_text(json.dumps(receipt, indent=2, default=str)+"\n")
            os.replace(tmp, args.output/"receipt.json")
        print(json.dumps({k: row.get(k) for k in ("name", "cold_warm_identical", "quality_answer", "quality_strict")}
                         | {"prefill_tps": row["long"]["prefill_tps"], "decode_tps": row["decode"]["decode_tps"],
                            "deep_prompt": row["deep"]["prompt_tokens"], "deep_prefill_tps": row["deep"]["prefill_tps"],
                            "deep_decode_tps": row["deep"]["decode_tps"], "deep_oracle": row["deep"]["oracle"]}
                         | ({"deeper_prompt": row["deeper"]["prompt_tokens"], "deeper_prefill_tps": row["deeper"]["prefill_tps"],
                             "deeper_decode_tps": row["deeper"]["decode_tps"], "deeper_oracle": row["deeper"]["oracle"]}
                            if deeper else {})),
              flush=True)
    token_file.unlink()
    return 0 if all(arm["passed"] for arm in receipt["arms"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
