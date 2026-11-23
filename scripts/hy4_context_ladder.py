#!/usr/bin/env python3
"""Single-rep needle-retrieval context ladder against one loopback mlx2 server.

Original mlx2 script.  A sanity and correctness ladder, not publishable
statistics: one repetition per cell, no thermal gating.  The caller owns the
GPU (``run_with_gpu_locks.py`` under the CPG lease, and the :8282 service
paused if needed).  It starts one ``mlx2.server`` on a loopback port other
than 8282 with an admin token (for ``POST /v1/admin/apc/clear``), and per cell:

* builds a prompt of about ``cell`` tokens from varied repository text
  (``realistic_haystack``, never repeated strings) with one access code
  buried at a per-cell depth, asking for that code only;
* clears APCv2, sends the prompt cold and then warm (same body), streaming,
  with the vendor's recommended sampling (``--sampling vendor``, default: no
  temperature/top_p in the request, so the server applies the adapter's
  generation_config defaults, Hy4-flash T 0.9 / top_p 1.0) and a fixed
  seed, ``reasoning_effort`` ``no_think`` as in the vendor's examples;
  ``--sampling greedy`` sends temperature 0;
* records prompt tokens, client TTFT, prefill tok/s (prompt tokens / TTFT
  on the cold request), decode tok/s (SSE arrivals after the first), the
  functional oracle (code present; Unicode dashes normalized, otherwise
  exact) with the raw response, the terminal mlx2 receipt (route, cached
  tokens), server ``metal_peak_bytes`` / physical footprint, and swap.

Stops (and records why) when a cell fails to run, or swap grows by more than
``--swap-stop-mb`` over the start.  The server process group is stopped in
``finally``.  Refuses a nonempty output directory.  Writes ``ladder.json``
atomically after every cell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from realistic_haystack import bury, haystack  # noqa: E402

DEPTHS = (0.10, 0.25, 0.50, 0.75, 0.90, 0.33, 0.60, 0.15, 0.80, 0.45)
DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")


def swap_used_mb() -> float:
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    match = re.search(r"used = ([\d.]+)M", out)
    return float(match.group(1)) if match else -1.0


def git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def get_json(base: str, path: str, timeout=10) -> dict:
    with urllib.request.urlopen(base + path, timeout=timeout) as response:
        return json.load(response)


def post_json(base: str, path: str, body: dict, token: str) -> dict:
    request = urllib.request.Request(
        base + path, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def find_key(tree, key):
    """First value under ``key`` anywhere in a JSON tree."""
    if isinstance(tree, dict):
        if key in tree:
            return tree[key]
        for value in tree.values():
            got = find_key(value, key)
            if got is not None:
                return got
    elif isinstance(tree, list):
        for value in tree:
            got = find_key(value, key)
            if got is not None:
                return got
    return None


def stream_one(base: str, body: dict, *, timeout: float) -> dict:
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    arrivals, ids, parts, receipt, finish, done = [], [], [], None, None, False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"generation SSE error: {event['error']}")
            if event.get("mlx2") is not None:
                receipt = event["mlx2"]
            for choice in event.get("choices", ()):
                for item in (choice.get("logprobs") or {}).get("content", ()) or ():
                    ids.append(int(item["id"]))
                    arrivals.append(time.perf_counter() - started)
                parts.append((choice.get("delta") or {}).get("content") or "")
                finish = choice.get("finish_reason") or finish
    if not done or receipt is None:
        raise ValueError("incomplete generation stream or absent terminal receipt")
    first = arrivals[0] if arrivals else None
    decode = ((len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
              if len(arrivals) > 1 and arrivals[-1] > arrivals[0] else None)
    return {"text": "".join(parts), "token_ids": ids, "ttft_s": first,
            "elapsed_s": time.perf_counter() - started, "decode_tok_s": decode,
            "finish_reason": finish, "receipt": receipt}


def make_code(rng_seed: int) -> str:
    digest = hashlib.sha256(f"hy4-ladder-{rng_seed}".encode()).hexdigest().upper()
    letters = "".join(c for c in digest if c.isalpha())[:3] or "QZX"
    return f"{letters}-{int(digest[:8], 16) % 90000 + 10000}-{digest[8:10]}"


def build_prompt(tokenizer, cell: int, depth: float, code: str, offset: int):
    fact = f"The access code for the north archive vault is {code}."
    question = ("\n\nQuestion: What is the access code for the north archive vault? "
                "Reply with the code only.")
    intro = "Read the reference material below, then answer the question at the end.\n\n"
    chars = int(cell * 4.0)
    for _ in range(4):  # calibrate the length to the tokenizer
        text = intro + bury(haystack(chars, offset=offset), fact, depth) + question
        tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if abs(tokens - cell) <= max(32, cell // 100):
            break
        chars = max(256, int(chars * cell / tokens))
    return text, tokens


def checkpoint_label(name, status):
    """``<artifact name>@<fingerprint>`` from ``/v1/status``.

    The adapter's ``execution.checkpoint`` names the artifact; the top-level
    ``artifact`` field is a route digest once a drafter is bound, so it is
    only the fallback.
    """
    label = (status.get("execution") or {}).get("checkpoint")
    if isinstance(label, str) and "@" in label:
        return f"{name}@{label.split('@', 1)[1]}"
    artifact = status.get("artifact")
    fingerprint = artifact if isinstance(artifact, str) else find_key(artifact or {}, "fingerprint")
    return f"{name}@{fingerprint}" if fingerprint else None


def ladder_verdict(cells):
    """Correctness failures of the finished cells.

    An oracle failure, or a warm arm that restored nothing from APCv2, fails
    the ladder.  A cold/warm token mismatch is only counted: it can be a
    passing near tie and needs classification, not an automatic failure.
    """
    ran = [row for row in cells if "cold" in row and "warm" in row]
    return {
        "oracle_failures": [f"{row['cell']}:{arm}" for row in ran for arm in ("cold", "warm")
                            if not row[arm].get("oracle_pass")],
        "warm_misses": [row["cell"] for row in ran if not row["warm"].get("cached_tokens")],
        "cold_warm_mismatches": [row["cell"] for row in ran if row.get("cold_warm_tokens_equal") is False],
    }


def stop_group(process) -> str:
    """Stop the server's process group; ``"stopped"`` or ``"still_running"``."""
    if process is None or process.poll() is not None:
        return "stopped"
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(60)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(30)
        except subprocess.TimeoutExpired:
            return "still_running"  # the caller records it and fails the ladder
    return "stopped" if process.poll() is not None else "still_running"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--execution-policy", type=Path, required=True)
    parser.add_argument("--route", choices=("ordinary", "ast"), default="ordinary",
                        help="server route flag (ast: the artifact's embedded DFly2 drafter)")
    parser.add_argument("--cells", default="1024,2048,4096,8192,16384,32768,65536,131072")
    parser.add_argument("--port", type=int, default=8285)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--sampling", choices=("vendor", "greedy"), default="vendor")
    parser.add_argument("--seed", type=int, default=20261120)
    # The server's generic vocabulary; the Hy4-flash adapter renders "none" as
    # the template's no_think (the vendor's examples use no_think).
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--swap-stop-mb", type=float, default=2048.0)
    parser.add_argument("--load-timeout", type=float, default=900.0)
    parser.add_argument("--cell-timeout", type=float, default=3600.0)
    a = parser.parse_args(argv)
    if a.port == 8282:
        parser.error("port 8282 is reserved for release trees")
    out = a.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error(f"{out} is not empty")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", a.port)) == 0:
            parser.error(f"port {a.port} is occupied")
    cells = [int(c) for c in a.cells.split(",")]
    model = a.model.expanduser().resolve(strict=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    token_file = out / "admin-token"
    fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    token = secrets.token_hex(24)
    os.write(fd, (token + "\n").encode())
    os.close(fd)
    base = f"http://127.0.0.1:{a.port}"
    swap0 = swap_used_mb()
    report = {
        "schema": "mlx2.hy4-context-ladder.v1",
        "kind": "single-rep sanity/correctness ladder; no thermal gating; not publishable statistics",
        "source": {"commit": git("rev-parse", "HEAD"),
                   "tracked_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                   "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "model_path": str(model), "execution_policy": json.loads(a.execution_policy.read_text()),
        "cells_requested": cells, "max_tokens": a.max_tokens, "swap_used_mb_start": swap0,
        "sampling": a.sampling, "seed": a.seed, "reasoning_effort": a.reasoning_effort,
        "started_at": datetime.now(UTC).isoformat(), "cells": [], "status": "running",
    }

    def save():
        tmp = out / "ladder.json.partial"
        tmp.write_text(json.dumps(report, indent=2, default=str) + "\n")
        os.replace(tmp, out / "ladder.json")

    server = None
    try:
        command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(model),
                   "--host", "127.0.0.1", "--port", str(a.port), f"--{a.route}",
                   "--max-lanes", "1", "--max-inflight", "1",
                   "--max-context", str(max(cells) + 4096),
                   "--execution-policy", str(a.execution_policy.expanduser().resolve()),
                   "--admin-token-file", str(token_file), "--no-cache-dir"]
        report["server_command"] = command
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1"}
        log = open(out / "server.log", "w")
        t0 = time.monotonic()
        server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"server exited {server.returncode} during load; see server.log")
            try:
                if get_json(base, "/health").get("status") == "ok":
                    break
            except (urllib.error.URLError, ConnectionError, OSError, ValueError):
                pass
            if time.monotonic() - t0 > a.load_timeout:
                raise TimeoutError("server not healthy within the load timeout")
            time.sleep(1)
        report["load_seconds"] = round(time.monotonic() - t0, 1)
        status = get_json(base, "/v1/status")
        report["server_status_start"] = status
        report["checkpoint"] = checkpoint_label(model.name, status)
        save()
        offset = 0
        for i, cell in enumerate(cells):
            depth = DEPTHS[i % len(DEPTHS)]
            code = make_code(cell)
            prompt, est = build_prompt(tokenizer, cell, depth, code, offset)
            offset += 50_000  # a different stretch of corpus per cell
            row = {"cell": cell, "needle_depth": depth, "code": code, "prompt_tokens_estimate": est,
                   "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
            report["cells"].append(row)
            try:
                row["apc_clear"] = post_json(base, "/v1/admin/apc/clear", {"reason": f"ladder-{cell}"}, token)
                body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens,
                        "seed": a.seed, "reasoning_effort": a.reasoning_effort,
                        "stream": True, "logprobs": True}
                if a.sampling == "greedy":
                    body["temperature"] = 0
                for arm in ("cold", "warm"):
                    result = stream_one(base, body, timeout=a.cell_timeout)
                    receipt = result["receipt"]
                    text = result["text"].translate(DASHES)
                    result["oracle_pass"] = code in text
                    result["prompt_tokens"] = receipt.get("prompt_tokens")
                    result["cached_tokens"] = receipt.get("cached_tokens")
                    result["sampling_receipt"] = {"request": find_key(receipt, "sampling"),
                                                  "applied_defaults": find_key(receipt, "sampling_defaults")}
                    pt = result["prompt_tokens"]
                    # Prefill throughput is a cold-arm figure: a warm TTFT
                    # measures a cache restore, not a prefill.
                    result["prefill_tok_s"] = (round(pt / result["ttft_s"], 1)
                                               if arm == "cold" and pt and result["ttft_s"] else None)
                    row[arm] = result
                    print(f"[{cell} {arm}] tokens={pt} cached={result['cached_tokens']} "
                          f"ttft={result['ttft_s'] and round(result['ttft_s'], 2)}s "
                          f"prefill={result['prefill_tok_s']} decode={result['decode_tok_s'] and round(result['decode_tok_s'], 1)} "
                          f"{'PASS' if result['oracle_pass'] else 'FAIL'} {result['text']!r}", flush=True)
                row["cold_warm_tokens_equal"] = row["cold"]["token_ids"] == row["warm"]["token_ids"]
                status = get_json(base, "/v1/status")
                row["metal_peak_gb"] = round((find_key(status, "metal_peak_bytes") or 0) / 2**30, 2)
                row["physical_footprint_gb"] = round(
                    (find_key(status, "process_physical_footprint_bytes") or 0) / 2**30, 2)
                row["status_after"] = status
            except Exception as error:  # a cell that will not run is recorded, then the ladder stops
                row["error"] = f"{type(error).__name__}: {error}"
                if isinstance(error, urllib.error.HTTPError):
                    try:
                        row["error_body"] = error.read().decode("utf-8", "replace")[:4000]
                    except Exception:  # noqa: BLE001
                        pass
                print(f"[{cell}] ERROR {row['error']}", flush=True)
                report["status"] = f"stopped at {cell}: cell did not run"
                row["swap_used_mb"] = swap_used_mb()
                save()
                break
            row["swap_used_mb"] = swap_used_mb()
            save()
            if row["swap_used_mb"] - swap0 > a.swap_stop_mb:
                report["status"] = f"stopped after {cell}: swap grew {row['swap_used_mb'] - swap0:.0f} MB"
                break
        else:
            report["status"] = "completed"
        report["verdict"] = ladder_verdict(report["cells"])
        # The fp8 latent-KV route declares no APCv2 and never publishes: a warm arm that restored nothing is
        # its expected behaviour, not a miss.
        apc = report["execution_policy"].get("latent_kv") != "fp8"
        report["verdict"]["warm_restore_expected"] = apc
        failed = [k for k in ("oracle_failures", "warm_misses") if report["verdict"][k] and (apc or k != "warm_misses")]
        if failed and report["status"] == "completed":  # an early stop keeps its own reason
            report["status"] = "failed: " + ", ".join(
                f"{len(report['verdict'][k])} {k.replace('_', ' ')}" for k in failed)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        report["server_stop"] = stop_group(server)
        if report["server_stop"] != "stopped":
            report["status"] = f"failed: server process group {server.pid} did not stop"
        report["swap_used_mb_end"] = swap_used_mb()
        report["finished_at"] = datetime.now(UTC).isoformat()
        try:
            token_file.unlink()
        except FileNotFoundError:
            pass
        save()
    print(json.dumps({"status": report["status"], "checkpoint": report.get("checkpoint"),
                      "commit": report["source"]["commit"], "out": str(out)}))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
