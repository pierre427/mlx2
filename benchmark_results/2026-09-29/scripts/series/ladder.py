#!/usr/bin/env python3
"""Context-length cold/warm quality and performance ladder for one real server."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REQUESTED_LENGTHS = (1024, 4096, 16384, 32768, 65536, 131072, 262144)
WIDTHS = (1, 4)


def percentile(values, fraction):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def calculate_metrics(*, prompt_tokens, completion_tokens, wall_seconds, ttft_seconds, token_times=(), receipt=None):
    """Derive bounded prefill/decode metrics from wire and receipt timings."""
    receipt = receipt or {}
    prefill_seconds = receipt.get("prefill_seconds") or receipt.get("prompt_seconds") or ttft_seconds
    prefill_tps = prompt_tokens / prefill_seconds if prompt_tokens and prefill_seconds and prefill_seconds > 0 else None
    intervals = [b - a for a, b in itertools.pairwise(token_times) if b > a]
    steady_decode_tps = 1 / statistics.median(intervals) if intervals else None
    if steady_decode_tps is None and completion_tokens > 1 and wall_seconds > ttft_seconds:
        steady_decode_tps = (completion_tokens - 1) / (wall_seconds - ttft_seconds)
    return {
        "ttft_seconds": ttft_seconds,
        "prefill_tokens_per_second": prefill_tps,
        "decode_tokens_per_second": steady_decode_tps,
        "wall_seconds": wall_seconds,
    }


def extract_peak_memory(status):
    """Return the largest finite peak/memory byte gauge visible in status."""
    found = []
    def visit(value, path=""):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else key)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            low = path.lower()
            if "byte" in low and ("peak" in low or "alloc" in low or "memory" in low):
                found.append((path, int(value)))
    visit(status)
    return max(found, key=lambda item: item[1]) if found else (None, None)


class Client:
    def __init__(self, base, timeout, model="campaign"):
        self.base, self.timeout, self.model = base.rstrip("/"), timeout, model

    def json(self, path, body=None):
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        def perform():
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        return self._guarded(perform)

    def health_error(self):
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=1.0) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            try:
                payload = json.loads(error.read())
            except (ValueError, OSError):
                return None
        except OSError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload.get("error")

    def _guarded(self, function):
        error = self.health_error()
        if error:
            raise RuntimeError(f"server health error: {error}")
        result, failure = {}, []

        def target():
            try:
                result["value"] = function()
            except BaseException as raised:  # noqa: BLE001 - propagate in caller
                failure.append(raised)

        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        deadline = time.monotonic() + self.timeout
        while worker.is_alive() and time.monotonic() < deadline:
            worker.join(min(0.5, max(0.0, deadline - time.monotonic())))
            if worker.is_alive():
                error = self.health_error()
                if error:
                    raise RuntimeError(f"server health error: {error}")
        if worker.is_alive():
            raise TimeoutError(f"server request exceeded {self.timeout}s")
        if failure:
            raise failure[0]
        return result["value"]

    def status(self):
        return self.json("/v1/status")

    def count(self, text):
        result = self.json("/v1/messages/count_tokens", {
            "model": self.model, "messages": [{"role": "user", "content": text}],
        })
        return int(result["input_tokens"])

    def _stream(self, text, *, max_tokens=64):
        body = {
            "model": self.model, "messages": [{"role": "user", "content": text}],
            "temperature": 0, "enable_thinking": False, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True},
        }
        request = urllib.request.Request(
            self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        started, first = time.monotonic(), None
        token_times, content, usage, receipt, done = [], [], {}, {}, False
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True; break
                event = json.loads(payload)
                usage = event.get("usage") or usage
                receipt = event.get("mlx2") or receipt
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                    if piece:
                        now = time.monotonic()
                        if first is None: first = now
                        token_times.append(now - started)
                        content.append(delta.get("content") or "")
        wall = time.monotonic() - started
        if first is None:
            first = started + wall
        return {
            "done": done, "content": "".join(content), "usage": usage, "receipt": receipt,
            "token_times": token_times,
            "metrics": calculate_metrics(
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                wall_seconds=wall, ttft_seconds=first - started,
                token_times=token_times, receipt=receipt,
            ),
        }

    def stream(self, text, *, max_tokens=64):
        return self._guarded(lambda: self._stream(text, max_tokens=max_tokens))


def calibrate_prompt(client: Client, target_tokens: int, needle: str, nonce: str):
    """Server-tokenizer calibration; heterogeneous adapters need no local renderer."""
    tail = (
        f"\nThe unique fact is: the launch code is {needle}.\n"
        f"Campaign nonce: {nonce}. What is the launch code? Reply with {needle} only."
    )
    unit = " archival evidence"
    if client.count(tail) > target_tokens:
        raise ValueError(f"target {target_tokens} is smaller than request framing")
    low, high = 0, target_tokens
    while low < high:
        middle = (low + high + 1) // 2
        if client.count(unit * middle + tail) <= target_tokens:
            low = middle
        else:
            high = middle - 1
    text = unit * low + tail
    actual = client.count(text)
    # Tokenizers can have no one-token suffix that reaches an exact target.
    # Preserve the largest prompt at or below the requested effective length.
    return text, actual


def run_cell(client, *, requested, effective, width, warm, prompts, needles):
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=width) as pool:
        rows = list(pool.map(client.stream, prompts[:width]))
    elapsed = time.monotonic() - started
    for row, needle in zip(rows, needles[:width]):
        row["needle"] = needle
        row["needle_correct"] = needle in row["content"]
        row["cached_tokens"] = int((row["receipt"] or {}).get("cached_tokens") or 0)
        row["output_sha256"] = hashlib.sha256(row["content"].encode()).hexdigest()
    return {
        "requested_tokens": requested, "effective_target_tokens": effective,
        "width": width, "temperature": "warm" if warm else "cold",
        "elapsed_seconds": elapsed, "requests": rows,
        "needle_correct": sum(row["needle_correct"] for row in rows),
        "apc_hits": sum(row["cached_tokens"] > 0 for row in rows),
        "ttft_p50_seconds": percentile([row["metrics"]["ttft_seconds"] for row in rows], .5),
        "prefill_p50_tokens_per_second": percentile([row["metrics"]["prefill_tokens_per_second"] for row in rows if row["metrics"]["prefill_tokens_per_second"] is not None], .5),
        "decode_p50_tokens_per_second": percentile([row["metrics"]["decode_tokens_per_second"] for row in rows if row["metrics"]["decode_tokens_per_second"] is not None], .5),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args(argv)
    if args.max_context < 1024 or args.timeout <= 0 or args.max_tokens < 2:
        parser.error("invalid positive ladder bounds")
    client = Client(args.url, args.timeout, args.model_id)
    initial = client.status()
    rows, prompts_record = [], {}
    for requested in REQUESTED_LENGTHS:
        effective = min(requested, args.max_context - max(args.max_tokens + 128, 256))
        if effective < 768 or requested > args.max_context:
            continue
        for width in WIDTHS:
            prompts, needles = [], []
            for lane in range(width):
                needle = f"NEEDLE_{requested}_{width}_{lane}_190919"
                text, actual = calibrate_prompt(client, effective, needle, f"{args.model}-{args.route}-{lane}")
                prompts.append(text); needles.append(needle)
                prompts_record[f"{requested}-w{width}-l{lane}"] = {
                    "requested": effective, "actual": actual,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(), "needle": needle,
                }
            cold = run_cell(client, requested=requested, effective=effective, width=width, warm=False, prompts=prompts, needles=needles)
            warm = run_cell(client, requested=requested, effective=effective, width=width, warm=True, prompts=prompts, needles=needles)
            rows.extend((cold, warm))
            print(f"{requested:6d} w={width} cold_ttft={cold['ttft_p50_seconds']:.3f}s warm_ttft={warm['ttft_p50_seconds']:.3f}s warm_hits={warm['apc_hits']}/{width} needle={cold['needle_correct'] + warm['needle_correct']}/{2 * width}", flush=True)
    final = client.status()
    peak_path, peak_bytes = extract_peak_memory(final)
    passed = all(
        row["needle_correct"] == row["width"]
        and all(request["done"] for request in row["requests"])
        and (row["temperature"] != "warm" or row["apc_hits"] == row["width"])
        for row in rows
    )
    report = {
        "schema": "mlx2.quality-context-ladder.v1", "model": args.model, "route": args.route,
        "max_context": args.max_context, "requested_lengths": list(REQUESTED_LENGTHS),
        "widths": list(WIDTHS), "initial": initial, "final": final,
        "peak_memory": {"path": peak_path, "bytes": peak_bytes},
        "prompts": prompts_record, "rows": rows, "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"SUMMARY cells={len(rows)} passed={passed} peak_memory_bytes={peak_bytes}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
