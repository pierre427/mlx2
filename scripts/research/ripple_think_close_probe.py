#!/usr/bin/env python3
"""Ripple live probe: which token ids the model emits to close ``<think>``.

``</think>`` is not one token in Ripple's tokenizer (``</`` ``think`` then
``>``, ``>\\n`` or ``>\\n\\n``), so a thinking guard or budget needs to know
the sequence the model actually generates. Sends varied chat prompts (greedy
and sampled) to one loopback server on the no-flag route, reads the generated
ids from the token trace, and records every ``</think`` occurrence with its
surrounding ids. Requires the caller's CPG lease and paired locks; the server
process group is stopped in ``finally``. Sanity evidence, not qualification.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = [
    "What is 17 * 23? Answer with the number.",
    "Name the capital of Australia in one word.",
    "Write a haiku about autumn rain.",
    "A train leaves at 3:40 pm and the trip takes 2 h 35 min. When does it arrive?",
    "Explain in two sentences why the sky is blue.",
    "List three prime numbers greater than 50.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpg-lock", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=18299)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--server-arg", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("refusing nonempty evidence directory")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from qualification_gpu_ownership import require_qualification_lease
    from mlx2.adapters.registry import inspect_model
    from tokenizers import Tokenizer

    owner = json.loads((args.cpg_lock / "owner.json").read_text())
    lease = require_qualification_lease(task_id=owner["cpg_task"], generation=owner["cpg_generation"],
                                        cpg_owner_lock=args.cpg_lock)
    resolution = inspect_model(args.model)
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    close_prefix = tuple(tokenizer.encode("</think", add_special_tokens=False).ids)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                         text=True, cwd=ROOT).strip())
    receipt = {"schema": "mlx2.ripple-think-close-probe.v1", "qualified": False,
               "checkpoint": f"{args.model.name}@sha256:{resolution.artifact_fingerprint}",
               "source_revision": {"head": head, "dirty_tracked": dirty}, "ownership": lease,
               "server_args": args.server_arg, "close_prefix_ids": close_prefix, "cases": []}
    root = f"http://127.0.0.1:{args.port}"

    command = [args.python, "-u", "-m", "mlx2.server", "--model", str(args.model), "--host", "127.0.0.1",
               "--port", str(args.port), "--max-inflight", "1", "--max-lanes", "1", "--max-context", "8192",
               "--cache-bytes", str(1 << 30), "--cache-dir", str(args.output / "cache"), *args.server_arg]
    log = (args.output / "server.log").open("w")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                              env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, cwd=ROOT)
    try:
        deadline = time.monotonic() + 900
        while True:
            if server.poll() is not None:
                raise RuntimeError("server exited before readiness")
            try:
                with urlopen(root + "/v1/status", timeout=10) as response:
                    if json.loads(response.read()).get("healthy"):
                        break
            except (URLError, OSError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("server not ready")
            time.sleep(2)
        for prompt in PROMPTS:
            for temperature in (0, 0.7):
                payload = {"model": args.model.name, "max_tokens": args.max_tokens, "seed": 7,
                           "temperature": temperature, "top_p": 0.95, "top_k": 20,
                           "messages": [{"role": "user", "content": prompt}], "return_token_trace": True}
                request = Request(root + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"})
                with urlopen(request, timeout=1800) as response:
                    result = json.loads(response.read())
                trace = (result.get("mlx2") or {}).get("token_trace") or {}
                ids = [int(t) for t in trace.get("token_ids", [])]
                width = len(close_prefix)
                hits = [i for i in range(len(ids) - width + 1) if tuple(ids[i:i + width]) == close_prefix]
                message = result["choices"][0]["message"]
                receipt["cases"].append({
                    "prompt": prompt, "temperature": temperature,
                    "finish_reason": result["choices"][0].get("finish_reason"),
                    "completion_tokens": len(ids), "truncated": trace.get("truncated"),
                    "first_ids": ids[:4], "first_pieces": [tokenizer.id_to_token(t) for t in ids[:4]],
                    "close_hits": [{"at": i, "next_ids": ids[i + width:i + width + 2],
                                    "next_pieces": [tokenizer.id_to_token(t) for t in ids[i + width:i + width + 2]],
                                    "ids": ids[max(0, i - 2):i + width + 3],
                                    "pieces": [tokenizer.id_to_token(t) for t in ids[max(0, i - 2):i + width + 3]]}
                                   for i in hits],
                    "reasoning_chars": len(message.get("reasoning_content") or ""),
                    "content": (message.get("content") or "")[:200],
                })
        receipt["close_next_ids"] = sorted({tuple(hit["next_ids"]) for case in receipt["cases"]
                                            for hit in case["close_hits"]})
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        except ProcessLookupError:
            pass
        log.close()
        tmp = args.output / "receipt.json.tmp"
        tmp.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
        os.replace(tmp, args.output / "receipt.json")


if __name__ == "__main__":
    main()
