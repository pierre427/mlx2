#!/usr/bin/env python3
"""Cached-prompt replay receipt through the real HTTP server (default route).

Design input: Inco Splash (Apache-2.0, f786bed) reports TTFT for a replayed
25,812-token prompt (275 ms on an M3 Max; 282 ms at 32K on an M5 Pro).  This
script measures the same shape on mlx2's default route; no Splash code is used.

For each target size (e.g. 25K and 32K tokens) and round, with a fresh nonce at
the head of the system prompt so sizes and rounds never share a prefix:

  cold      system(document) + user question            -> cold TTFT, prefill tok/s
  replay    the identical request again                  -> warm TTFT
  followup  cold messages + the cold reply + a new user turn
  prefix    same system(document), a different question  -> shared-prefix TTFT

Per request it records client TTFT, prompt/cached tokens, the ``mlx2`` receipt
(cache_checkpoint_role, route), and deltas of the APCv2 / interior-checkpoint
counters from ``/v1/status``.  The server is started bare (``--model`` plus
port only): whatever the adapter declares as its default route and execution
policy is what gets measured.

A swap watchdog samples ``vm_stat`` Swapouts: it allows ``--load-swap-mib``
through model load, then aborts the run on a rise of ``--swap-abort-mib``.

Refuses to run without ``--i-own-the-gpu``.
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
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualify_interior_checkpoints import _apcv2, _delta, _mechanism, server_env  # noqa: E402

PAGE = 16384


def swapouts_bytes() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    match = re.search(r"Swapouts:\s+(\d+)", out)
    return int(match.group(1)) * PAGE if match else 0


def pageins() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    match = re.search(r"Pageins:\s+(\d+)", out)
    return int(match.group(1)) if match else 0


class SwapWatch(threading.Thread):
    def __init__(self, abort_bytes: int, on_abort):
        super().__init__(daemon=True)
        self.baseline = swapouts_bytes()
        self.abort_bytes = abort_bytes
        self.on_abort = on_abort
        self.peak = 0
        self.stop = threading.Event()

    def rebase(self):
        self.baseline = swapouts_bytes()

    def run(self):
        while not self.stop.wait(2.0):
            rise = swapouts_bytes() - self.baseline
            self.peak = max(self.peak, rise)
            if rise >= self.abort_bytes:
                self.on_abort(rise)
                return


def corpus(root: Path) -> str:
    """Natural-ish technical text: this checkout's docs and sources."""
    parts = []
    for pattern in ("docs/*.md", "src/mlx2/*.py", "src/mlx2/runtime/*.py"):
        for path in sorted(root.glob(pattern)):
            try:
                parts.append(path.read_text(errors="ignore"))
            except OSError:
                pass
    return "\n\n".join(parts)


def build_document(text: str, tokenizer, target: int, offset: int) -> str:
    ids = tokenizer.encode(text[offset:offset + target * 8]).ids
    if len(ids) < target:
        raise ValueError("corpus too small for target")
    return tokenizer.decode(ids[:target])


def stream_chat(base: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    ttft = None
    text, usage, receipt = [], None, None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("mlx2"):
                receipt = chunk["mlx2"]
            for choice in chunk.get("choices") or ():
                delta = choice.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    text.append(piece)
    return {
        "ttft_s": ttft,
        "total_s": time.perf_counter() - start,
        "text": "".join(text),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
        "receipt": {
            key: (receipt or {}).get(key)
            for key in ("cached_tokens", "cache_checkpoint_role", "route", "route_selection_source")
        } if receipt else None,
    }


def status(base: str) -> dict:
    with urllib.request.urlopen(base + "/v1/status", timeout=60) as r:
        return json.loads(r.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--sizes", default="25812,32768")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--port", type=int, default=8391)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--load-swap-mib", type=int, default=1536)
    parser.add_argument("--swap-abort-mib", type=int, default=256)
    parser.add_argument("--extra-server-args", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        print("refusing: pass --i-own-the-gpu under the GPU lock wrapper", file=sys.stderr)
        return 2

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    root = Path(__file__).resolve().parents[1]
    text = corpus(root)
    sizes = [int(s) for s in args.sizes.split(",") if s]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".server.log")
    command = [
        sys.executable, "-u", "-m", "mlx2.server", "--model", args.model,
        "--host", "127.0.0.1", "--port", str(args.port),
        *args.extra_server_args.split(),
    ]
    aborted = {}
    with open(log_path, "w") as log:
        server = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=server_env(),
            start_new_session=True,
        )

    def abort(rise):
        aborted["swap_rise_bytes"] = rise
        print(f"ABORT: swapouts rose {rise >> 20} MiB", flush=True)
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except OSError:
            pass

    watch = SwapWatch(args.load_swap_mib << 20, abort)
    watch.start()
    base = f"http://127.0.0.1:{args.port}"
    record = {"command": command, "sizes": sizes, "rows": []}
    try:
        deadline = time.monotonic() + args.startup_timeout
        while True:
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                pass
            if server.poll() is not None or time.monotonic() > deadline or aborted:
                raise RuntimeError(f"server failed to start; see {log_path}")
            time.sleep(2)
        record["load_swap_rise_mib"] = (swapouts_bytes() - watch.baseline) >> 20
        watch.rebase()
        watch.abort_bytes = args.swap_abort_mib << 20
        initial = status(base)
        model_id = initial.get("model")
        record["settings"] = initial.get("settings")
        record["qualification"] = initial.get("qualification")

        def run(kind, size, round_index, messages):
            body = {
                "model": model_id, "messages": messages, "max_tokens": args.max_tokens,
                "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}, "enable_thinking": False,
            }
            before = _mechanism(status(base))
            pins = pageins()
            try:
                result = stream_chat(base, body, args.timeout)
            except urllib.error.HTTPError as error:
                # Record the refusal (e.g. memory admission) and keep going.
                detail = error.read().decode(errors="replace")[:2000]
                row = {"kind": kind, "size": size, "round": round_index,
                       "http_error": error.code, "error_body": detail,
                       "status_counts": status(base).get("counts")}
                record["rows"].append(row)
                out.write_text(json.dumps(record, indent=1))
                print(f"[{size} r{round_index}] {kind:8s} HTTP {error.code}: {detail[:300]}", flush=True)
                return None
            after = _mechanism(status(base))
            row = {
                "kind": kind, "size": size, "round": round_index,
                **{k: v for k, v in result.items() if k != "text"},
                "reply_chars": len(result["text"]),
                "pageins_delta": pageins() - pins,
                "mechanism": _delta(after, before),
            }
            uncached = (row["prompt_tokens"] or 0) - (row["cached_tokens"] or 0)
            if row["ttft_s"] and uncached > 0:
                row["uncached_tokens"] = uncached
                row["prefill_tok_s_incl_first_step"] = uncached / row["ttft_s"]
            record["rows"].append(row)
            out.write_text(json.dumps(record, indent=1))
            print(
                f"[{size} r{round_index}] {kind:8s} P={row['prompt_tokens']} "
                f"cached={row['cached_tokens']} ttft={row['ttft_s']:.3f}s "
                f"role={(row['receipt'] or {}).get('cache_checkpoint_role')} "
                f"int_hits+={row['mechanism']['interior_hits']} "
                f"captured+={row['mechanism']['captured']}",
                flush=True,
            )
            return result

        offset = 0
        for round_index in range(args.rounds):
            for size in sizes:
                if aborted:
                    raise RuntimeError("aborted on swap")
                nonce = uuid.uuid4().hex
                doc = build_document(text, tokenizer, size - 120, offset)
                offset = (offset + 20011) % max(1, len(text) - size * 8)
                system = {
                    "role": "system",
                    "content": f"Session {nonce}. Answer questions about this project "
                    f"source and documentation.\n\n{doc}",
                }
                question = {"role": "user", "content": "In two sentences, what does this code do?"}
                cold = run("cold", size, round_index, [system, question])
                run("replay", size, round_index, [system, question])
                if cold is not None:
                    run(
                        "followup", size, round_index,
                        [system, question, {"role": "assistant", "content": cold["text"]},
                         {"role": "user", "content": "Name one function it defines and why."}],
                    )
                run(
                    "prefix", size, round_index,
                    [system, {"role": "user", "content": "List three file names mentioned above."}],
                )
        final = status(base)
        record["final_status_counts"] = final.get("counts")
        record["final_apcv2"] = {k: v for k, v in _apcv2(final).items() if k != "entries"}
        record["swap_peak_rise_mib_after_load"] = watch.peak >> 20
    finally:
        watch.stop.set()
        record["aborted"] = aborted or None
        out.write_text(json.dumps(record, indent=1))
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=60)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except OSError:
                pass
    return 1 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
