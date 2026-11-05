#!/usr/bin/env python3
"""Ripple loopback A/B across source trees and server flags (one artifact).

One server per arm, one at a time, greedy decoding, the same prompts in every
arm. Each arm records cold/warm APCv2 replay, long retrieval prompts that
cross the 1,024-token window (TTFT, prefill rate, oracle), prose decodes and
copy-heavy code rewrites with full token traces (cross-arm exactness and
prompt-lookup benefit), a four-request concurrency cell, /v1/status and
/metrics. No thermal holds: sanity and attribution numbers, not publishable
statistics. Requires the caller's CPG lease and paired locks; every server
process group is stopped in ``finally``.

Arms JSON: [{"name": str, "tree": "/path/to/mlx2/checkout", "args": [...]}].
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

DECODES = [
    "Write a detailed, multi-paragraph history of the printing press.",
    "Explain, step by step and at length, how a refrigerator moves heat.",
    "Describe in detail the life cycle of a star like the Sun.",
]
CONCURRENT = [
    "Explain how a bicycle's gears change the effort needed to pedal uphill.",
    "Describe how bread dough rises and what yeast does during proofing.",
    "Summarize how tides are caused by the Moon and the Sun.",
    "Explain why the sky appears blue during the day and red at sunset.",
]
COPY_SOURCES = ["scripts/download_ripple.py", "src/mlx2/adapters/ripple_memory.py"]
REPLAY = ("Shipment log for the night audit.\nThe shipment identifier is ZX-3187.\n"
          "Reply with the shipment identifier only.")


def long_prompt(chars, offset, code):
    # Varied real text (project docs), never one repeated sentence.
    text, needle = [], f"The archive release code is {code}."
    for path in sorted((ROOT / "docs").glob("*.md"))[offset:]:
        text.append(path.read_text(errors="ignore"))
        if sum(map(len, text)) > chars:
            break
    body = "\n".join(text)[:chars]
    middle = len(body) // 2
    return (body[:middle] + "\n" + needle + "\n" + body[middle:]
            + "\n\nWhat is the archive release code? Reply with the code only."), code


def copy_prompt(path):
    source = (ROOT / path).read_text()
    return ("Reproduce the following Python file exactly, changing only one thing: add a one-line "
            "comment `# reviewed` directly above every `def` line. Output only the file.\n\n```python\n"
            + source + "```")


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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpg-lock", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=18296)
    parser.add_argument("--long-chars", type=int, default=24000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--copy-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("refusing nonempty A/B evidence directory")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from qualification_gpu_ownership import require_qualification_lease

    owner = json.loads((args.cpg_lock / "owner.json").read_text())
    lease = require_qualification_lease(task_id=owner["cpg_task"], generation=owner["cpg_generation"],
                                        cpg_owner_lock=args.cpg_lock)
    from mlx2.adapters.ripple import inspect_artifact

    arms = json.loads(args.arms.read_text())
    artifact = inspect_artifact(args.model)

    def revision(tree):
        return {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=tree).strip(),
                "dirty_tracked": bool(subprocess.check_output(
                    ["git", "status", "--porcelain", "--untracked-files=no"], text=True, cwd=tree).strip())}

    receipt = {"schema": "mlx2.ripple-ab.v1",
               "checkpoint": args.model.name + "@sha256:" + artifact["identity"]["fingerprint"],
               "storage_format": artifact["storage_format"], "driver_revision": revision(ROOT),
               "ownership": lease, "qualified": False,
               "kind": "loopback A/B, greedy, no thermal hold; sanity and attribution only",
               "arms": [], "thermal_before": subprocess.check_output(["pmset", "-g", "therm"], text=True).strip(),
               "swap_before": subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()}
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

    def chat(content, max_tokens):
        start = time.monotonic()
        result = call("/v1/chat/completions", {
            "model": args.model.name, "temperature": 0, "stream": False, "max_tokens": max_tokens,
            "return_token_trace": True, "messages": [{"role": "user", "content": content}]})
        elapsed = time.monotonic() - start
        usage = result.get("usage", {})
        ttft = find(result, "ttft_seconds")
        completion = usage.get("completion_tokens", 0)
        trace = (result.get("mlx2") or {}).get("token_trace") or {}
        return {"content": result["choices"][0]["message"].get("content", ""),
                "finish_reason": result["choices"][0].get("finish_reason"),
                "token_ids": trace.get("token_ids"),
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": completion,
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "elapsed_seconds": elapsed, "ttft_seconds": ttft,
                "prefill_tps": usage.get("prompt_tokens", 0) / ttft if ttft else None,
                "decode_tps": ((completion - 1) / (elapsed - ttft)
                               if ttft and completion > 1 and elapsed > ttft else None),
                "route": (result.get("mlx2") or {}).get("route"),
                "ordinary_compute_widths": (result.get("mlx2") or {}).get("ordinary_compute_widths"),
                "speculation": (result.get("mlx2") or {}).get("speculation")}

    def clear():
        call("/v1/admin/apc/clear", {"reason": "ripple-ab-cell"}, admin=True)

    longs = [long_prompt(args.long_chars, offset, code)
             for offset, code in ((0, "QV-4471"), (12, "HB-2093"))]
    copies = [copy_prompt(path) for path in COPY_SOURCES]
    for arm in arms:
        out = args.output / arm["name"]
        out.mkdir()
        tree = Path(arm["tree"]).resolve()
        command = [args.python, "-u", "-m", "mlx2.server", "--model", str(args.model),
                   "--host", "127.0.0.1", "--port", str(args.port), "--max-inflight", "4",
                   "--max-context", "16384", "--cache-bytes", str(4 << 30),
                   "--cache-dir", str(out / "cache"), "--admin-token-file", str(token_file), *arm["args"]]
        if "--max-lanes" not in arm["args"]:
            command += ["--max-lanes", "1"]
        log = (out / "server.log").open("w")
        env = {**os.environ, "PYTHONPATH": str(tree / "src")}
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True, env=env, cwd=tree)
        row = {"name": arm["name"], "tree": str(tree), "revision": revision(tree), "args": arm["args"],
               "command": command, "server_pid": server.pid}
        try:
            started = time.monotonic()
            deadline = started + 900
            while True:
                if server.poll() is not None:
                    raise RuntimeError("server exited before readiness")
                try:
                    status = call("/v1/status", timeout=10)
                    if status.get("healthy"):
                        break
                except (URLError, TimeoutError, ConnectionError, OSError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("server readiness timed out")
                time.sleep(.5)
            row["load_seconds"] = time.monotonic() - started
            row["startup_status"] = status
            chat("Say hello in one word.", 4)
            clear()
            row["cold"], row["warm"] = chat(REPLAY, 24), chat(REPLAY, 24)
            row["cold_warm_exact"] = row["cold"]["token_ids"] == row["warm"]["token_ids"]
            row["long_runs"] = []
            for text, code in longs:
                clear()
                got = chat(text, 48)
                got["oracle"] = code in got["content"].replace("–", "-").replace("‑", "-")
                row["long_runs"].append(got)
            row["decode_runs"] = []
            for text in DECODES:
                clear()
                row["decode_runs"].append(chat(text, args.max_tokens))
            row["copy_runs"] = []
            for text in copies:
                clear()
                row["copy_runs"].append(chat(text, args.copy_tokens))
            clear()
            results, errors = [None] * len(CONCURRENT), []

            def worker(index, text):
                try:
                    results[index] = chat(text, 128)
                except BaseException as error:  # recorded, re-raised below
                    errors.append(repr(error))

            threads = [threading.Thread(target=worker, args=(i, t)) for i, t in enumerate(CONCURRENT)]
            wall = time.monotonic()
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            wall = time.monotonic() - wall
            if errors:
                raise RuntimeError(f"concurrency cell failed: {errors}")
            row["concurrent"] = {"runs": results, "wall_seconds": wall,
                                 "aggregate_tps": sum(r["completion_tokens"] for r in results) / wall}
            time.sleep(1.5)  # one worker publication interval
            row["final_status"] = call("/v1/status")
            (out / "metrics.prom").write_text(call("/metrics"))
            median = lambda xs: sorted(xs)[len(xs) // 2]
            row["summary"] = {
                "load_seconds": row["load_seconds"],
                "long_ttft_s": [r["ttft_seconds"] for r in row["long_runs"]],
                "long_prefill_tps_median": median([r["prefill_tps"] for r in row["long_runs"]]),
                "long_oracle": all(r["oracle"] for r in row["long_runs"]),
                "decode_tps": [r["decode_tps"] for r in row["decode_runs"]],
                "decode_tps_median": median([r["decode_tps"] for r in row["decode_runs"]]),
                "copy_tps": [r["decode_tps"] for r in row["copy_runs"]],
                "concurrent_aggregate_tps": row["concurrent"]["aggregate_tps"],
                "cold_warm_exact": row["cold_warm_exact"],
                "warm_cached_tokens": row["warm"]["cached_tokens"],
            }
            row["passed"] = bool(row["cold_warm_exact"] and row["summary"]["long_oracle"])
        except BaseException as error:
            row["passed"] = False
            row["error"] = repr(error)
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
            row["server_stopped"] = server.poll() is not None
            receipt["arms"].append(row)
            receipt["swap_after"] = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()
            receipt["thermal_after"] = subprocess.check_output(["pmset", "-g", "therm"], text=True).strip()
            tmp = args.output / "receipt.tmp"
            tmp.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
            os.replace(tmp, args.output / "receipt.json")
        print(json.dumps({"arm": row["name"], **row.get("summary", {})}), flush=True)
    token_file.unlink()
    return 0 if all(arm["passed"] for arm in receipt["arms"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
