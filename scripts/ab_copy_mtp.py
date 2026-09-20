#!/usr/bin/env python3
"""Interleaved served A/B: self-MTP vs self-MTP + copy drafts (rm01).

GPU-only (starts real model servers).  Refuses to run without
``--i-own-the-gpu``; ``--dry-run`` prints the plan and exits.

Arms (same model, same flags, only the execution policy differs):

* ``off``: ``{"num_draft": N}``
* ``on``:  ``{"num_draft": N, "self_mtp_copy_draft": {"enabled": true, ...}}``
* ``ord`` (``--ordinary-arm``): the ``--ordinary`` route, no execution policy.
  Informational baseline (main made ordinary the Qwen3.6-35B default in
  16059c7); the pre-registered go/no-go stays on vs off.

Self-MTP arms pass ``--native-mtp`` when the server under test has that flag
(main after 16059c7), so the script keeps selecting self-MTP after folding.

Arms alternate per rep (off,on / on,off / ...), each in a fresh server, so
drift hits both sides (back-to-back A/Bs drift ~10%).  Every request carries
a nonce prefix so neither arm reuses the other's prefix cache.

Hard gates (the harness refuses rather than reporting a null):

* every receipt is the native self-MTP route;
* the on-arm's ``self_mtp_copy_rounds`` must increase on the code corpus at
  B=1 (cohorts do not copy under the default ``batched_max_span=0``, so the
  gate is asserted where the mechanism is allowed to run, and the batched
  cells report their counters);
* the off-arm must not expose any ``self_mtp_copy_*`` counter.

Go / no-go (plan wiki/docs/plans/mlx2-rm01-copy-mtp.md), per model:
prose on/off >= 0.98 (worst on-rep >= 0.96 x best off-rep) at every width,
code on/off >= 1.10 at B=1 and >= 1.00 at B=4, greedy outputs identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]

PROSE = [
    "Explain how a compiler works, in numbered sections.",
    "Explain how a database transaction works, in numbered sections.",
    "Explain, step by step and in plain language, how a Metal compute kernel "
    "is dispatched on Apple silicon, from command buffer creation to completion.",
    "Write a short essay on why cities build public libraries.",
]

_MODULE = '''class AdaptiveLookback:
    """Miss-driven lookback with rejection backoff inside a scheduler cap."""

    def __init__(self, ladder=(256, 1024, 4096, 16384), *, misses=4, rejects=2):
        self.ladder = tuple(int(value) for value in ladder)
        self.index = 0
        self.cap = self.ladder[-1]
        self.misses_to_widen = int(misses)
        self.rejects_to_narrow = int(rejects)
        self._misses = self._rejects = 0
        self.widen_events = self.narrow_events = 0

    @property
    def current(self):
        return min(self.ladder[self.index], self.cap)

    def observe(self, proposed, accepted):
        if not proposed:
            self._rejects = 0
            self._misses += 1
            if self._misses >= self.misses_to_widen:
                if self.index < len(self.ladder) - 1:
                    self.index += 1
                    self.widen_events += 1
                self._misses = 0
        elif not accepted:
            self._misses = 0
            self._rejects += 1
            if self._rejects >= self.rejects_to_narrow:
                if self.index:
                    self.index -= 1
                    self.narrow_events += 1
                self._rejects = 0
        else:
            self._misses = self._rejects = 0
'''
_BUGGY = '''def merge_intervals(intervals):
    """Merge overlapping closed intervals [start, end]."""
    ordered = sorted(intervals)
    merged = []
    for start, end in ordered:
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]
'''
CODE = [
    f"Here is a Python class:\n\n```python\n{_MODULE}```\n\nReturn the whole class "
    "unchanged except rename `_misses` to `_miss_count` everywhere. Output only the code.",
    f"Here is Python code:\n\n```python\n{_BUGGY}```\n\nBoth functions treat touching "
    "closed intervals as disjoint. Fix them and return the complete corrected code.",
    f"Here is a Python class:\n\n```python\n{_MODULE}```\n\nReturn the whole class with "
    "type hints added to every method signature. Output only the code.",
    f"Here is Python code:\n\n```python\n{_BUGGY}```\n\nReturn the same code with a "
    "docstring added to `overlaps`. Output only the code.",
]
CORPORA = {"prose": PROSE, "code": CODE}


def _post(url, body, timeout):
    request = Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _status(url):
    with urlopen(url + "/v1/status", timeout=30) as response:
        return json.load(response)


def _wait_ready(url, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode}")
        try:
            status = _status(url)
            if status.get("healthy") and status.get("ready", True):
                return status
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError("server did not become ready")


def _request(url, text, *, max_tokens, temperature, timeout, nonce, expect_mtp=True):
    body = {
        "messages": [{"role": "user", "content": f"[{nonce}] {text}"}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }
    if temperature > 0:
        body["seed"] = 1234
    result = _post(url, body, timeout)
    receipt = result["mlx2"]
    if expect_mtp and not receipt.get("mtp"):
        raise RuntimeError("refusing arm: receipt is not the native self-MTP route")
    if not expect_mtp and receipt.get("mtp"):
        raise RuntimeError("refusing arm: ordinary arm served a self-MTP receipt")
    tokens = int(receipt["completion_tokens"])
    decode_s = float(receipt["elapsed_seconds"]) - float(receipt["ttft_seconds"])
    content = result["choices"][0]["message"].get("content") or ""
    return {
        "completion_tokens": tokens,
        "decode_tok_s": (tokens - 1) / decode_s if decode_s > 0 and tokens > 1 else None,
        "output_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "output_chars": len(content),
        "output_head": content[:200],
        "copy_draft": (receipt["mtp"] or {}).get("copy_draft"),
    }


def _copy_counters(status):
    scheduler = status.get("scheduler") or {}
    return {k: v for k, v in scheduler.items() if k.startswith("self_mtp_copy_")}


def server_has_native_mtp_flag():
    source = ROOT / "src" / "mlx2" / "server.py"
    return '"--native-mtp"' in source.read_text()


def server_route_args(arm, policy_path, native_mtp_flag):
    if arm == "ord":
        return ["--ordinary"]
    route = ["--native-mtp"] if native_mtp_flag else []
    return route + ["--execution-policy", str(policy_path)]


def run_arm(args, arm, rep, policy_path):
    port = args.port
    url = f"http://127.0.0.1:{port}"
    command = [
        sys.executable, "-m", "mlx2.server", "--model", args.model,
        "--host", "127.0.0.1", "--port", str(port),
        "--max-context", str(args.max_context), "--max-lanes", str(max(args.widths)),
        "--max-inflight", str(2 * max(args.widths)),
        "--cache-bytes", str(args.cache_gib << 30), "--qualification-mode",
        *server_route_args(arm, policy_path, args.native_mtp_flag),
    ]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    log = open(Path(args.out).with_suffix(f".{arm}.{rep}.server.log"), "w")
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    cells = []
    try:
        _wait_ready(url, process, args.startup_timeout)
        # Warm-up, identical work in every arm, so no first-use Metal shape
        # compilation lands inside a measured cell.  The code requests matter:
        # a copy row verifies at a width the head never proposes and its
        # kernels compile on first use, which is a per-process cost, not a
        # steady-state one.  Warming on prose alone charged it to the on-arm's
        # first measured prose cell.
        expect_mtp = arm != "ord"
        _request(url, PROSE[0], max_tokens=16, temperature=0.0,
                 timeout=args.timeout, nonce=uuid.uuid4().hex, expect_mtp=expect_mtp)
        for text in CODE[:2]:
            _request(url, text, max_tokens=args.warmup_tokens, temperature=0.0,
                     timeout=args.timeout, nonce=uuid.uuid4().hex, expect_mtp=expect_mtp)
        for temperature in args.temperatures:
            for corpus, prompts in CORPORA.items():
                for width in args.widths:
                    before = _copy_counters(_status(url))
                    nonce = uuid.uuid4().hex
                    batch = prompts[:width] if width > 1 else prompts
                    if width > 1:
                        with ThreadPoolExecutor(max_workers=width) as pool:
                            rows = list(pool.map(
                                lambda text: _request(
                                    url, text, max_tokens=args.max_tokens,
                                    temperature=temperature, timeout=args.timeout,
                                    nonce=nonce, expect_mtp=expect_mtp),
                                batch))
                    else:
                        rows = [
                            _request(url, text, max_tokens=args.max_tokens,
                                     temperature=temperature, timeout=args.timeout,
                                     nonce=nonce, expect_mtp=expect_mtp)
                            for text in batch
                        ]
                    status_after = _status(url)
                    after = _copy_counters(status_after)
                    delta = {k: after.get(k, 0) - before.get(k, 0) for k in after}
                    if arm != "on" and after:
                        raise RuntimeError(f"refusing arm: {arm}-arm exposes copy counters")
                    if (arm == "on" and corpus == "code" and width == 1
                            and delta.get("self_mtp_copy_rounds", 0) <= 0):
                        raise RuntimeError(
                            "refusing arm: copy drafts enabled but self_mtp_copy_rounds "
                            f"did not move on code at B=1 (temp {temperature})"
                        )
                    rates = [row["decode_tok_s"] for row in rows if row["decode_tok_s"]]
                    cells.append({
                        "arm": arm, "rep": rep, "corpus": corpus, "width": width,
                        "temperature": temperature,
                        "decode_tok_s_mean": statistics.mean(rates) if rates else None,
                        "rows": rows, "copy_counter_delta": delta,
                        "metal_peak_bytes": status_after.get("metal_peak_bytes"),
                    })
                    print(json.dumps({k: v for k, v in cells[-1].items() if k != "rows"}), flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()
    return cells


def verdict(cells, widths, temperatures):
    out = {"cells": {}, "go": True, "reasons": []}
    for temperature in temperatures:
        for corpus in CORPORA:
            for width in widths:
                sel = [c for c in cells if c["corpus"] == corpus and c["width"] == width
                       and c["temperature"] == temperature and c["decode_tok_s_mean"]]
                on = [c["decode_tok_s_mean"] for c in sel if c["arm"] == "on"]
                off = [c["decode_tok_s_mean"] for c in sel if c["arm"] == "off"]
                if not on or not off:
                    continue
                ratio = statistics.mean(on) / statistics.mean(off)
                worst = min(on) / max(off)
                key = f"{corpus}/B{width}/t{temperature}"
                out["cells"][key] = {"on_mean": statistics.mean(on), "off_mean": statistics.mean(off),
                                     "ratio": ratio, "worst_on_vs_best_off": worst}
                ordinary = [c["decode_tok_s_mean"] for c in sel if c["arm"] == "ord"]
                if ordinary:
                    # Informational: main's Qwen3.6-35B baseline is ordinary.
                    ord_mean = statistics.mean(ordinary)
                    out["cells"][key].update({
                        "ord_mean": ord_mean,
                        "on_vs_ord": statistics.mean(on) / ord_mean,
                        "off_vs_ord": statistics.mean(off) / ord_mean,
                    })
                if corpus == "prose" and (ratio < 0.98 or worst < 0.96):
                    out["go"] = False
                    out["reasons"].append(f"prose regression {key}: {ratio:.3f} (worst {worst:.3f})")
                if corpus == "code" and temperature == 0.0:
                    need = 1.10 if width == 1 else 1.00
                    if ratio < need:
                        out["go"] = False
                        out["reasons"].append(f"code gain short {key}: {ratio:.3f} < {need}")
    # Greedy exactness across arms (documented q_len numerics may differ;
    # report, and require the plan's quality review if any differ).
    mismatches = 0
    greedy = [c for c in cells if c["temperature"] == 0.0]
    by = {}
    for c in greedy:
        for index, row in enumerate(c["rows"]):
            by.setdefault((c["corpus"], c["width"], index), {}).setdefault(c["arm"], set()).add(row["output_sha256"])
    ord_mismatches = 0
    for arms in by.values():
        if "on" in arms and "off" in arms and arms["on"] != arms["off"]:
            mismatches += 1
        if "on" in arms and "ord" in arms and arms["on"] != arms["ord"]:
            ord_mismatches += 1
    out["greedy_output_mismatches"] = mismatches
    out["greedy_output_mismatches_vs_ordinary"] = ord_mismatches
    # Within-arm greedy nondeterminism: a cohort's composition varies between
    # reps, so an arm can disagree with itself.  Without this, a cross-arm
    # mismatch cannot be attributed to the lever.
    out["greedy_within_arm_nondeterminism"] = {
        arm: sum(1 for arms in by.values() if len(arms.get(arm, ())) > 1)
        for arm in ("off", "on", "ord")
    }
    # Peak memory: process-wide monotonic peak per arm (max over cells covers
    # B=1 and B=4); plan budget is +0.5 GiB for the on-arm.
    peaks = {}
    for c in cells:
        if c.get("metal_peak_bytes") is not None:
            peaks.setdefault(c["arm"], []).append(int(c["metal_peak_bytes"]))
    if "on" in peaks and "off" in peaks:
        on_peak = max(peaks["on"])
        off_peak = max(peaks["off"])
        out["peak_memory"] = {"on_bytes": on_peak, "off_bytes": off_peak,
                              "delta_gib": (on_peak - off_peak) / (1 << 30)}
        if on_peak - off_peak > (1 << 29):
            out["go"] = False
            out["reasons"].append(f"peak memory +{(on_peak - off_peak) / (1 << 30):.2f} GiB > 0.5")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.7])
    parser.add_argument("--num-draft", type=int, default=2)
    parser.add_argument("--copy-policy", default='{"enabled": true}',
                        help="JSON for self_mtp_copy_draft in the on-arm")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--warmup-tokens", type=int, default=192,
                        help="tokens per warm-up code request (compiles copy verify widths)")
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--cache-gib", type=int, default=8)
    parser.add_argument("--port", type=int, default=8311)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--ordinary-arm", action="store_true",
                        help="also run an --ordinary arm (informational baseline)")
    parser.add_argument("--native-mtp", choices=("auto", "always", "never"), default="auto",
                        help="pass --native-mtp to self-MTP arms (auto: when the server has it)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    copy_policy = json.loads(args.copy_policy)
    policies = {
        "off": {"num_draft": args.num_draft},
        "on": {"num_draft": args.num_draft, "self_mtp_copy_draft": copy_policy},
    }
    if args.ordinary_arm:
        base = ("off", "on", "ord")
        order = [base[rep % 3:] + base[:rep % 3] for rep in range(args.reps)]
    else:
        order = [("off", "on") if rep % 2 == 0 else ("on", "off") for rep in range(args.reps)]
    args.native_mtp_flag = (
        args.native_mtp == "always"
        or (args.native_mtp == "auto" and server_has_native_mtp_flag())
    )
    plan = {"model": args.model, "policies": policies, "order": order,
            "widths": args.widths, "temperatures": args.temperatures,
            "corpora": {k: len(v) for k, v in CORPORA.items()}, "max_tokens": args.max_tokens,
            "native_mtp_flag": args.native_mtp_flag, "ordinary_arm": args.ordinary_arm}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.i_own_the_gpu:
        parser.error("this starts real model servers on Metal; pass --i-own-the-gpu under the GPU lock")
    tmp = Path(tempfile.mkdtemp(prefix="rm01-ab-"))
    paths = {}
    for arm, policy in policies.items():
        paths[arm] = tmp / f"{arm}.json"
        paths[arm].write_text(json.dumps(policy))
    cells = []
    for rep, arms in enumerate(order):
        for arm in arms:
            cells.extend(run_arm(args, arm, rep, paths.get(arm)))
    result = {"schema": "mlx2.copy-mtp-ab.v1", "plan": plan, "cells": cells,
              "verdict": verdict(cells, args.widths, args.temperatures)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["verdict"], indent=2))
    return 0 if result["verdict"]["go"] else 2


if __name__ == "__main__":
    sys.exit(main())
