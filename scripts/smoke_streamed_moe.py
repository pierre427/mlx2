#!/usr/bin/env python3
"""One-artifact smoke of an oversized MoE served with expert disk streaming.

Run inside ``scripts/run_with_gpu_locks.py``.  Starts one loopback
``mlx2.server`` on the artifact with no streaming policy (the engine must
auto-select expert streaming because the resident load does not fit), waits
for ``/health`` with a long load timeout, sends a short code prompt and a
short chat prompt (ordinary decode, temperature 0), and writes one atomic
receipt: auto-selection decision, plan, expert-cache counters, TTFT and decode
rate (labelled streamed, not a benchmark), sampled process memory, swap and
thermal samples.  A steady rise in swap-outs aborts the window.  The server's
whole process group is stopped in ``finally``.  A nonempty evidence directory
is refused so a retry stays distinguishable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {
    "code": "Write a Python function `fib(n)` that returns the n-th Fibonacci "
            "number iteratively. Reply with the code only.",
    "chat": "In two short sentences, explain why the sky looks blue during the day.",
}
GIB = float(1 << 30)


def _atomic_json(path: Path, payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(descriptor, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _vm_pages() -> dict:
    text = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    pages = {}
    for key, label in (("free", "Pages free"), ("wired", "Pages wired down"),
                       ("anonymous", "Anonymous pages"), ("file", "File-backed pages"),
                       ("compressor", "Pages occupied by compressor"),
                       ("swapouts", "Swapouts")):
        match = re.search(label + r":\s+(\d+)", text)
        pages[key] = int(match.group(1)) if match else -1
    return pages


def _swapouts() -> int:
    return _vm_pages()["swapouts"]


def _swap_used_mib() -> float:
    text = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                          text=True, check=True).stdout
    match = re.search(r"used = ([\d.]+)M", text)
    return float(match.group(1)) if match else -1.0


def _rss_bytes(pid: int) -> int:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, check=False,
                         text=True).stdout.strip()
    return int(out) * 1024 if out else 0


def _thermal() -> str:
    out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, check=False).stdout
    return " ".join(out.split())[-300:]


class Watch(threading.Thread):
    """Sample swap-outs and server RSS; flag a steady swap-out rise."""

    def __init__(self, pid: int, *, interval=5.0, rising_samples=6, rising_pages=4096,
                 total_pages=131072):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.rising_samples = rising_samples
        self.rising_pages = rising_pages
        # A burst is as dangerous as a trend with swap nearly full: also abort
        # once the window's cumulative swap-outs pass 2 GiB (16 KiB pages).
        self.total_pages = total_pages
        self.baseline_swapouts = _swapouts()
        self.baseline_swap_mib = _swap_used_mib()
        self.samples = []
        self.peak_rss = 0
        self.abort_reason = None
        self._stop = threading.Event()

    def run(self):
        streak, streak_start = 0, self.baseline_swapouts
        previous = self.baseline_swapouts
        while not self._stop.wait(self.interval):
            pages = _vm_pages()
            swapouts = pages["swapouts"]
            rss = _rss_bytes(self.pid)
            self.peak_rss = max(self.peak_rss, rss)
            self.samples.append({"t": time.time(), "swapouts": swapouts,
                                 "swap_used_mib": _swap_used_mib(), "rss": rss,
                                 "pages_16k": pages})
            if swapouts > previous:
                if streak == 0:
                    streak_start = previous
                streak += 1
            else:
                streak = 0
            previous = swapouts
            if swapouts - self.baseline_swapouts >= self.total_pages:
                self.abort_reason = (
                    f"swap-outs +{swapouts - self.baseline_swapouts} pages this window"
                )
                return
            if streak >= self.rising_samples and swapouts - streak_start >= self.rising_pages:
                self.abort_reason = (
                    f"swap-outs rose for {streak} consecutive samples "
                    f"(+{swapouts - streak_start} pages)"
                )
                return

    def stop(self):
        self._stop.set()

    def summary(self):
        last = self.samples[-1] if self.samples else {}
        return {
            "baseline_swapouts": self.baseline_swapouts,
            "final_swapouts": last.get("swapouts"),
            "swapouts_delta": (last.get("swapouts", self.baseline_swapouts)
                               - self.baseline_swapouts),
            "baseline_swap_used_mib": self.baseline_swap_mib,
            "final_swap_used_mib": last.get("swap_used_mib"),
            "peak_server_rss_gib": round(self.peak_rss / GIB, 3),
            "samples": len(self.samples),
            "abort_reason": self.abort_reason,
        }


def _get(base, path, timeout=30):
    with urllib.request.urlopen(base + path, timeout=timeout) as response:
        return response.status, json.loads(response.read() or b"{}")


def _stream_chat(base, prompt, *, max_tokens, thinking_off, watch, timeout):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if thinking_off:
        body["enable_thinking"] = False
    request = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.time()
    first = last = None
    text, reasoning, usage, chunks, finish = [], [], None, 0, None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            if watch.abort_reason:
                raise RuntimeError(f"aborted: {watch.abort_reason}")
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", ()):
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if piece or thought:
                    now = time.time()
                    first = first or now
                    last = now
                    chunks += 1
                    text.append(piece)
                    reasoning.append(thought)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    completion = (usage or {}).get("completion_tokens") or chunks
    decode_s = (last - first) if first and last and last > first else None
    return {
        "prompt": prompt,
        "request": {k: v for k, v in body.items() if k != "messages"},
        "text": "".join(text),
        "reasoning": "".join(reasoning),
        "finish_reason": finish,
        "usage": usage,
        "ttft_s": round(first - started, 3) if first else None,
        "wall_s": round(time.time() - started, 3),
        "decode_tok_s": (
            round((completion - 1) / decode_s, 4) if decode_s and completion > 1 else None
        ),
        "throughput_label": "streamed, not a benchmark",
    }


def _coherence(name, result):
    text = (result["text"] or "").strip()
    words = text.split()
    distinct = len(set(words)) / max(1, len(words))
    checks = {
        "nonempty": bool(text),
        "not_degenerate": len(words) < 8 or distinct > 0.3,
        "printable": all(ch.isprintable() or ch in "\n\t" for ch in text),
    }
    if name == "code":
        checks["has_code"] = "def " in text or "fib" in text
    else:
        checks["mentions_scattering"] = bool(
            re.search(r"scatter|rayleigh|wavelength|blue light", text, re.IGNORECASE)
        )
    return {"checks": checks, "pass": all(checks.values()), "distinct_word_ratio": round(distinct, 3)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True, help="<artifact name>@<fingerprint>")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--load-timeout", type=float, default=3600.0)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--server-arg", action="append", default=[])
    args = parser.parse_args()

    evidence = args.evidence
    if evidence.exists() and any(evidence.iterdir()):
        print(f"refusing nonempty evidence directory {evidence}", file=sys.stderr)
        return 2
    evidence.mkdir(parents=True, exist_ok=True)
    source = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"],
                           capture_output=True, text=True, check=True).stdout.strip()
    receipt = {
        "schema": "mlx2.streamed-moe-smoke.v1",
        "checkpoint": args.checkpoint,
        "model_path": args.model,
        "source_revision": source,
        "source_dirty": bool(dirty),
        "host": os.uname().nodename,
        "started_at": time.time(),
        "thermal_before": _thermal(),
        "status": "running",
        "label": "smoke only; streamed throughput is not a benchmark",
    }
    base = f"http://127.0.0.1:{args.port}"
    command = [
        sys.executable, "-u", "-m", "mlx2.server", "--model", args.model,
        "--host", "127.0.0.1", "--port", str(args.port), "--fabric", "off",
        "--max-context", "8192", "--ordinary", *args.server_arg,
    ]
    receipt["server_command"] = command
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    log = open(evidence / "server.log", "wb")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                              env=env, start_new_session=True)
    watch = Watch(server.pid)
    watch.start()
    try:
        started = time.time()
        while True:
            if watch.abort_reason:
                raise RuntimeError(f"aborted during load: {watch.abort_reason}")
            if server.poll() is not None:
                raise RuntimeError(f"server exited during load ({server.returncode})")
            try:
                code, _ = _get(base, "/health", timeout=10)
                if code == 200:
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            if time.time() - started > args.load_timeout:
                raise RuntimeError("load timeout")
            time.sleep(5)
        receipt["load_s"] = round(time.time() - started, 1)
        _, status = _get(base, "/v1/status", timeout=60)
        receipt["status_after_load"] = {
            key: status.get(key)
            for key in ("model", "route", "profile", "weight_streaming", "qualification",
                        "descriptor", "settings")
            if key in status
        }
        receipt["counts_after_load"] = {
            k: v for k, v in (status.get("counts") or {}).items() if k.startswith("stream_")
        }
        receipt["rss_after_load_gib"] = round(_rss_bytes(server.pid) / GIB, 3)
        for key in ("metal_active_bytes", "metal_peak_bytes", "process_physical_footprint_bytes"):
            if key in status:
                receipt[f"{key}_after_load"] = status[key]
        results = {}
        for name, prompt in PROMPTS.items():
            try:
                result = _stream_chat(base, prompt, max_tokens=args.max_tokens,
                                      thinking_off=True, watch=watch,
                                      timeout=args.request_timeout)
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")[:500]
                if error.code != 400 or "thinking" not in detail:
                    raise RuntimeError(f"{name}: HTTP {error.code}: {detail}") from error
                result = _stream_chat(base, prompt, max_tokens=args.max_tokens,
                                      thinking_off=False, watch=watch,
                                      timeout=args.request_timeout)
            _, status = _get(base, "/v1/status", timeout=60)
            counts = {k: v for k, v in (status.get("counts") or {}).items()
                      if k.startswith("stream_")}
            hits = counts.get("stream_expert_hits_total", 0)
            misses = counts.get("stream_expert_misses_total", 0)
            result["stream_counts_after"] = counts
            result["expert_hit_rate_cumulative"] = (
                round(hits / (hits + misses), 4) if hits + misses else None
            )
            result["coherence"] = _coherence(name, result)
            results[name] = result
            _atomic_json(evidence / "receipt.json", {**receipt, "results": results})
        receipt["results"] = results
        _, status = _get(base, "/v1/status", timeout=60)
        receipt["status_final"] = {
            key: status.get(key)
            for key in ("weight_streaming", "metal_active_bytes", "metal_peak_bytes",
                        "process_physical_footprint_bytes", "headroom_bytes",
                        "cache_bytes", "profile", "route")
            if key in status
        }
        peak = status.get("metal_peak_bytes")
        receipt["metal_peak_gib"] = round(peak / GIB, 3) if peak else None
        receipt["status"] = "pass" if all(r["coherence"]["pass"] for r in results.values()) \
            else "pass_with_warnings"
    except BaseException as error:  # noqa: BLE001 - record and stop the server
        receipt["status"] = "error"
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        watch.stop()
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=120)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=60)
        log.close()
        receipt["memory_and_swap"] = watch.summary()
        _atomic_json(evidence / "memory-samples.json", {"samples": watch.samples})
        receipt["thermal_after"] = _thermal()
        receipt["finished_at"] = time.time()
        _atomic_json(evidence / "receipt.json", receipt)
    print(json.dumps({k: receipt.get(k) for k in ("checkpoint", "status", "error", "load_s")}))
    return 0 if receipt["status"] in ("pass", "pass_with_warnings") else 1


if __name__ == "__main__":
    raise SystemExit(main())
