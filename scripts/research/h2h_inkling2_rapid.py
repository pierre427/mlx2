#!/usr/bin/env python3
"""Same-host head-to-head: mlx2 vs Rapid-MLX on Inkling 2 Mini.

Original mlx2 script (2026-11-23).  Rapid-MLX published Inkling-2-Mini numbers
from a 36 GB M5 Max Mac Studio (raullenchai/Rapid-MLX@44680c946,
docs/benchmarks/inkling-2-mini-m5-max.md); this replays their client method on
this host so the stacks compare on one machine:

* OpenAI-compatible streaming chat completions, ``max_tokens`` 256,
  temperature 0, one fresh server process per arm, no other model server;
* a unique random-word prompt per request at about 659, 4,996 and 19,845
  prompt tokens, three repetitions per length, medians reported (so no
  prefix-cache hits);
* TTFT = time to the first streamed token; prefill tok/s = prompt tokens /
  TTFT; decode tok/s excludes TTFT ((completion tokens - 1) / (last - first));
* four simultaneous requests: aggregate output tok/s and per-request tok/s.

Arms (``--arms``, run in the order given, one server each):

* ``mlx2``: ``mlx2.server`` default route on the official checkpoint;
* ``rapid-nvfp4``: ``rapid-mlx serve`` on the same official checkpoint
  directory (same weight files);
* ``rapid-affine``: ``rapid-mlx serve`` on the community affine 4-bit
  artifact (mlx2 does not load it: its adapter declares only the official
  NVFP4/MXFP8 variant), for Rapid-MLX's "NVFP4 decodes ~18% slower than
  affine 4-bit" note on this host.

The same frozen prompt file (``prepare``) feeds every arm.  A memory guard
stops an arm (and the run) when swap grows by more than ``--swap-abort-mb``
or free memory falls under ``--free-abort-gib``.  Each arm writes
``<arm>.json`` atomically; ``--out-dir`` must be empty.  Exploratory, no
thermal hold: not publishable statistics.  Run each arm under the GPU lease.

Usage:
  python scripts/research/h2h_inkling2_rapid.py prepare --tokenizer <model dir> --out prompts.json
  python scripts/research/h2h_inkling2_rapid.py run --prompts prompts.json --arm mlx2 \
      --model <official dir> --out-dir <empty dir> [--rapid-bin <venv>/bin/rapid-mlx]
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import random
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "mlx2.h2h-inkling2-rapid.v1"
TARGETS = (659, 4996, 19845)  # Rapid-MLX's published prompt lengths
REPS = 3
CONCURRENT = 4
WORDS = Path("/usr/share/dict/words")
INSTRUCTION = "Write a short story that uses as many of the following words as you can:\n\n"


# -- prompts ------------------------------------------------------------------------


def prepare(a) -> None:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(a.tokenizer) / "tokenizer.json"))
    vocab = [w for w in WORDS.read_text().split() if w.isalpha() and w.islower() and 3 <= len(w) <= 10]
    rng = random.Random(a.seed)
    prompts = []

    def make(target, tag):
        words, text = [], INSTRUCTION
        # grow by chunks, then trim: raw tokenizer count of the user text
        while len(tok.encode(text).ids) < target:
            words += rng.sample(vocab, 64)
            text = INSTRUCTION + " ".join(words)
        while len(tok.encode(text).ids) > target and words:
            words.pop()
            text = INSTRUCTION + " ".join(words)
        return {"tag": tag, "target": target, "user_tokens": len(tok.encode(text).ids),
                "sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text}

    for target in TARGETS:
        for rep in range(REPS):
            prompts.append(make(target, f"t{target}-r{rep}"))
    for i in range(CONCURRENT):
        prompts.append(make(TARGETS[0], f"conc-{i}"))
    Path(a.out).write_text(json.dumps({"schema": SCHEMA + ".prompts", "seed": a.seed,
                                       "tokenizer": str(a.tokenizer), "prompts": prompts}, indent=1))
    print(f"wrote {len(prompts)} prompts to {a.out}")


# -- client -------------------------------------------------------------------------


def stream_chat(port: int, model: str, text: str, max_tokens: int, timeout: float) -> dict:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": text}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}})
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:500]!r}")
    first = last = None
    chunks, usage, finish, content, reasoning = 0, None, None, [], []
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        event = json.loads(data)
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            piece = delta.get("content") or ""
            think = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if piece or think:
                now = time.perf_counter()
                first = first if first is not None else now
                last = now
                chunks += 1
                content.append(piece)
                reasoning.append(think)
            finish = choice.get("finish_reason") or finish
    conn.close()
    completion = (usage or {}).get("completion_tokens") or chunks
    prompt_tokens = (usage or {}).get("prompt_tokens")
    ttft = None if first is None else first - t0
    decode = None
    if first is not None and last is not None and last > first and completion > 1:
        decode = (completion - 1) / (last - first)
    return {"ttft_s": ttft, "decode_tok_s": decode, "prompt_tokens": prompt_tokens,
            "completion_tokens": completion, "content_chunks": chunks,
            "usage_reported": usage is not None, "finish_reason": finish,
            "total_s": time.perf_counter() - t0,
            "prefill_tok_s": (prompt_tokens / ttft) if prompt_tokens and ttft else None,
            "content_sha256": hashlib.sha256("".join(content).encode()).hexdigest(),
            "reasoning_chars": sum(map(len, reasoning)), "content_chars": sum(map(len, content))}


# -- memory guard -------------------------------------------------------------------


def swap_used_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False).stdout
    part = out.split("used = ")[1].split()[0]
    return float(part.rstrip("M"))


def free_gib() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False).stdout
    page = int(out.split("page size of ")[1].split()[0])
    free = int(next(ln for ln in out.splitlines() if ln.startswith("Pages free")).split()[-1].rstrip("."))
    return free * page / 2**30


class Guard:
    """Kill the server group when swap grows or free memory runs out."""

    def __init__(self, pgid: int, swap_abort_mb: float, free_abort_gib: float):
        self.pgid, self.swap_abort_mb, self.free_abort_gib = pgid, swap_abort_mb, free_abort_gib
        self.base_swap = swap_used_mb()
        self.tripped = None
        self.min_free = None
        self.max_swap_growth = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(2.0):
            growth, free = swap_used_mb() - self.base_swap, free_gib()
            self.max_swap_growth = max(self.max_swap_growth, growth)
            self.min_free = free if self.min_free is None else min(self.min_free, free)
            if growth > self.swap_abort_mb or free < self.free_abort_gib:
                self.tripped = {"swap_growth_mb": round(growth, 1), "free_gib": round(free, 2),
                                "at": time.strftime("%FT%T%z")}
                stop_group(self.pgid)
                return

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(5)


# -- servers ------------------------------------------------------------------------


def _group_gone(pgid: int, proc=None) -> bool:
    """True once no live process is left in the group.

    ``proc`` (the leader's Popen, when the caller owns it) is reaped here:
    macOS answers ``killpg`` with EPERM, not ESRCH, while the group holds only
    an unreaped zombie leader, so a probe without reaping never sees "gone"
    (2026-11-25: every arm lost its receipt to that PermissionError).
    """
    if proc is not None:
        proc.poll()
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        # Nothing in the group can be signalled: an exited (zombie) leader
        # with no live members.  With the leader reaped, the group is done.
        return proc is None or proc.returncode is not None
    return False


def stop_group(pgid: int, proc=None) -> None:
    for sig, wait in ((signal.SIGTERM, 20), (signal.SIGKILL, 5)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        except PermissionError:
            if _group_gone(pgid, proc):
                return
        t0 = time.monotonic()
        while time.monotonic() - t0 < wait:
            if _group_gone(pgid, proc):
                return
            time.sleep(0.5)


def port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def server_command(a, arm: str, cache_dir: Path) -> tuple[list[str], dict]:
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "RAPID_MLX_TELEMETRY": "0", "DO_NOT_TRACK": "1"}
    if arm == "mlx2":
        env["PYTHONPATH"] = str(ROOT / "src")
        return ([sys.executable, "-u", "-m", "mlx2.server", "--model", str(a.model),
                 "--host", "127.0.0.1", "--port", str(a.port), "--fabric", "off",
                 "--cache-dir", str(cache_dir)], env)
    model = a.model if arm == "rapid-nvfp4" else a.affine_model
    if not a.rapid_bin or not model:
        raise SystemExit(f"{arm} needs --rapid-bin and its model dir")
    env.pop("PYTHONPATH", None)
    return ([str(a.rapid_bin), "serve", str(model), "--host", "127.0.0.1", "--port", str(a.port),
             "--disable-disk-caches", "--disable-model-downloads"], env)


def _healthy(port: int) -> bool:
    """``/health`` is 200 (or absent).  mlx2 lists its model while the worker is
    still loading and answers chat 503 until ``/health`` turns 200
    (2026-11-25: the mlx2 arms died on a 503 warm-up 1.7 s after start)."""
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        resp.read()
        conn.close()
    except (OSError, http.client.HTTPException):
        return False
    return resp.status in (200, 404)


def wait_ready(port: int, server, timeout: float) -> str:
    t0 = time.monotonic()
    while server.poll() is None and time.monotonic() - t0 < timeout:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            data = json.loads(resp.read() or b"{}")
            conn.close()
            if resp.status == 200 and data.get("data") and _healthy(port):
                return data["data"][0]["id"]
        except (OSError, ValueError, http.client.HTTPException):
            pass
        time.sleep(2)
    raise RuntimeError(f"server not ready (rc {server.poll()})")


def get_json(port: int, path: str):
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        return json.loads(body) if resp.status == 200 else {"http_status": resp.status}
    except (OSError, ValueError, http.client.HTTPException) as exc:
        return {"error": str(exc)}


def git_rev(path: Path) -> str | None:
    r = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return r.stdout.strip() or None


def median(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.median(vals) if vals else None


def run(a) -> int:
    out = Path(a.out_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; refuse")
    out.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(a.prompts).read_text())["prompts"]
    if not port_free(a.port):
        raise SystemExit(f"port {a.port} busy")
    cache_dir = Path(tempfile.mkdtemp(prefix="h2h-ink-cache-", dir=out))
    cmd, env = server_command(a, a.arm, cache_dir)
    rapid_pkg = None
    if a.arm != "mlx2":
        probe = ("import importlib.metadata as m, json, mlx.core as mx;"
                 "d=m.distribution('rapid-mlx');print(json.dumps({'version': d.version,"
                 "'direct_url': d.read_text('direct_url.json'), 'mlx': mx.__version__}))")
        rapid_pkg = subprocess.run([str(Path(a.rapid_bin).with_name("python")), "-c", probe],
                                   capture_output=True, text=True, check=False).stdout.strip()
    receipt = {"schema": SCHEMA, "arm": a.arm, "kind": "exploratory same-host A/B, no thermal hold",
               "command": cmd, "model_dir": str(a.model if a.arm != "rapid-affine" else a.affine_model),
               "mlx2_revision": git_rev(ROOT), "rapid_package": rapid_pkg,
               "started": time.strftime("%FT%T%z"), "swap_before": subprocess.run(
                   ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False).stdout.strip(),
               "free_gib_before": round(free_gib(), 1), "requests": [], "errors": []}
    log = open(out / f"{a.arm}.server.log", "w")  # noqa: SIM115
    server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True)
    try:
        with Guard(server.pid, a.swap_abort_mb, a.free_abort_gib) as guard:
            t0 = time.monotonic()
            model_id = wait_ready(a.port, server, a.load_timeout)
            receipt["load_s"] = round(time.monotonic() - t0, 1)
            receipt["served_model_id"] = model_id
            if a.arm == "mlx2":
                status = get_json(a.port, "/v1/status")
                receipt["mlx2_status"] = {k: status.get(k) for k in (
                    "model", "artifact", "route", "qualification", "default_route") if k in status}
            stream_chat(a.port, model_id, "Say hello.", 16, a.request_timeout)  # warm-up
            for p in prompts:
                if not p["tag"].startswith("t"):
                    continue
                r = stream_chat(a.port, model_id, p["text"], a.max_tokens, a.request_timeout)
                receipt["requests"].append({"tag": p["tag"], "target": p["target"],
                                            "prompt_sha256": p["sha256"], **r})
                if guard.tripped:
                    break
            if not guard.tripped:
                conc = [p for p in prompts if p["tag"].startswith("conc-")]
                results = [None] * len(conc)

                def one(i):
                    try:
                        results[i] = stream_chat(a.port, model_id, conc[i]["text"], a.max_tokens,
                                                 a.request_timeout)
                    except Exception as exc:  # noqa: BLE001
                        results[i] = {"error": str(exc)}

                t0 = time.perf_counter()
                threads = [threading.Thread(target=one, args=(i,)) for i in range(len(conc))]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                wall = time.perf_counter() - t0
                tokens = sum(r.get("completion_tokens") or 0 for r in results)
                receipt["concurrent"] = {"requests": results, "wall_s": wall,
                                         "aggregate_output_tok_s": tokens / wall if wall else None,
                                         "per_request_decode_tok_s": median(results, "decode_tok_s")}
            receipt["guard"] = {"tripped": guard.tripped, "max_swap_growth_mb": round(guard.max_swap_growth, 1),
                                "min_free_gib": None if guard.min_free is None else round(guard.min_free, 1)}
    except Exception as exc:  # noqa: BLE001 - the receipt records it
        receipt["errors"].append(repr(exc))
    finally:
        stop_group(server.pid, server)
        log.close()
    summary = {}
    for target in TARGETS:
        rows = [r for r in receipt["requests"] if r["target"] == target]
        summary[str(target)] = {k: median(rows, k) for k in
                                ("prompt_tokens", "ttft_s", "prefill_tok_s", "decode_tok_s")}
    receipt["summary"] = summary
    receipt["finished"] = time.strftime("%FT%T%z")
    tmp = out / f".{a.arm}.json.tmp"
    tmp.write_text(json.dumps(receipt, indent=1))
    tmp.replace(out / f"{a.arm}.json")
    print(json.dumps({"arm": a.arm, "summary": summary, "concurrent": {
        k: (receipt.get("concurrent") or {}).get(k) for k in ("aggregate_output_tok_s", "per_request_decode_tok_s")},
        "errors": receipt["errors"], "guard": receipt.get("guard")}, indent=1))
    return 0 if not receipt["errors"] and not (receipt.get("guard") or {}).get("tripped") else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20261123)
    r = sub.add_parser("run")
    r.add_argument("--prompts", required=True)
    r.add_argument("--arm", choices=("mlx2", "rapid-nvfp4", "rapid-affine"), required=True)
    r.add_argument("--model", required=True, help="official Inkling-2-Mini directory")
    r.add_argument("--affine-model", help="community affine 4-bit directory")
    r.add_argument("--rapid-bin", help="rapid-mlx executable")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--port", type=int, default=18421)
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--load-timeout", type=float, default=900)
    r.add_argument("--request-timeout", type=float, default=600)
    r.add_argument("--swap-abort-mb", type=float, default=2048)
    r.add_argument("--free-abort-gib", type=float, default=2)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare(a)
        return 0
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
