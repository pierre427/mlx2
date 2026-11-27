#!/usr/bin/env python3
"""Phase 0 round-cost harness: one route, one lane, quiet vs contended host.

Design: docs/research/2026-11-25-gpu-resident-spec-rounds-design.md (Phase 0,
§4). Measurement only; it changes no route and no default.

``run`` drives one already-running loopback server started with
``MLX2_ROUND_PROBE=1`` (``src/mlx2/runtime/round_probe.py``). For each context
size it builds a fixed prompt from distinct GSM8K test questions (no repeated
strings), clears APCv2, sends one cold warm-up request, then measures warm
requests (decode only; the prefix is cached) until at least ``--rounds``
decode rounds are booked. Every quiet cell runs before any contended cell, so
the load average of a quiet cell is never inflated by the generator's tail.

Per cell it records rounds/s, tokens/s, tokens per round (accepted + bonus),
host ms per round split into encode/submit, wait and host Python, syncs per
round (explicit and implicit, with call sites), the 1-minute load average and
the instantaneous runnable-process count, sampled every 5 s. Probe numbers
come from the per-request receipt's cumulative ``round_probe`` block (fresh at
the request's last token), differenced between consecutive requests.

``--xctrace`` adds one traced request per quiet cell (never under load) under a
bounded Metal System Trace attached to the server pid; GPU busy time is the
union of that pid's ``metal-gpu-intervals`` divided by the probe rounds of that
request. The traced request is excluded from the speed numbers (xctrace costs
CPU). A trace that matched no server-pid interval, or that hit its time limit,
yields no GPU time. Traces are large (6-14 GB at 64K on 2026-11-25), so only
cells up to ``--trace-max-size`` (16K) are traced, with a short request
(``--trace-max-tokens``), only when ``--trace-min-free-gib`` is free, and the
bundle is deleted after parsing unless ``--keep-traces``.

Contended cells start ``--load-workers`` busy Python processes (pytest-like:
bytecode, dict/list churn, allocation and GC; default priority, as the real
pytest load was; ``--load-nice`` lowers it) ``--load-ramp-s`` before measuring, and stop them after the
cell. Each contended cell holds the load at most ``--load-ramp-s`` +
``--cell-max-s`` and is followed by ``--cooldown-s`` without load, so lease
heartbeats get through (2026-11-25: the CPG radio timed out at load ~48, so
loaded stretches are at most 10 + 50 s with 60 s of cooldown, and 48 workers).
The 1-minute load average lags the ramp, so the
instantaneous runnable-process count is the headline contention figure.

Long contexts (2026-11-26 reruns).  A contended request that outlasts the load
window is dropped, and at 64K one 256-token request is longer than the window.
``--size-max-tokens 65536=32`` shortens the requests of that context size in
*both* conditions: the report compares tokens per round exactly, so the quiet
and contended cells of a size must answer the same greedy prefix.
``--contended-rounds`` lowers only the contended cell's round target; tokens
per round still has to match the quiet cell exactly, so a route whose requests
differ (a mid-cell route change) leaves the pair incomplete rather than
comparing.  Both default to the Phase 0 behaviour, and ``run`` refuses a
combination that cannot book its round target (a round commits at least one
token) before it writes any evidence.

Route transitions.  Each request row records the response receipt's ``mtp``
transition (``ordinary_after_mtp_fallback`` / ``_handoff`` with its reason,
widths and committed tokens), and the full ``mlx2`` receipt of every request
goes to ``receipts-<size>-<condition>.jsonl`` (an overrun request is marked
``past_deadline``; the traced request is ``"trace"``).  Cells also list
``probe_route_changes``: requests whose probed round routes differ from the
first request's, which catches a target-only APCv2 hit that decodes ordinary
with no ``mtp`` receipt.  ``--scheduler-deltas`` also reads ``/v1/status``
after each request, waiting (at most 2.5 s) for two newer worker snapshots,
and stores the changed counts, scheduler, APCv2 and segmented-MTP counters (a native-MTP route that drops to the
ordinary batcher mid-cell, as Maxwell 16K did on 2026-11-25, is then
explained by its own counters).

``load`` runs only the generator (calibration). ``report`` joins cell files
into the go/no-go table of §4.4: Phase 1 is a go for a route when contended
round time rises by >= 20 % at unchanged tokens per round, or when the quiet host share of round
time is >= 15 %. The round period is decode time / rounds as the client sees
it, so it includes the serving loop between rounds. Host share is (period -
GPU busy) / period when a usable trace exists, else (encode/submit + host
Python + between rounds) / period; the in-round share (encode/submit + host
Python) / in-round wall is reported beside it.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SCHEMA = "mlx2.spec-round-phase0.v1"
GO_PERIOD_INCREASE = 0.20
GO_HOST_SHARE = 0.15
INSTRUCTION = ("\n\nSummarise the kinds of problems listed above in about 150 words, "
               "as plain prose.")


# --------------------------------------------------------------- host load --

def _burn(deadline, parent, nice=0):
    """One pytest-like CPU worker: Python bytecode, containers, allocation, GC."""
    import gc
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if nice:
        os.nice(nice)
    table, spin = {}, 0
    while time.monotonic() < deadline:
        for i in range(2000):
            key = f"k{(spin * 2654435761 + i) % 50021}"
            table[key] = [i, key, (i, spin)]
            if len(table) > 20000:
                table.clear()
        spin += 1
        if spin % 64 == 0:
            gc.collect(0)
            if os.getppid() != parent:  # orphaned: never outlive the harness
                return


def runnable_processes():
    """Instantaneous count of runnable processes (``ps`` state R)."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "state="], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return sum(1 for line in out.split() if line.startswith("R"))


def load_sample():
    one, five, fifteen = os.getloadavg()
    return {"t": time.time(), "load1": round(one, 2), "load5": round(five, 2),
            "load15": round(fifteen, 2), "runnable": runnable_processes()}


class LoadGenerator:
    """N busy worker processes with a hard deadline; idempotent stop."""

    def __init__(self, workers, max_seconds, nice=0):
        self.workers = int(workers)
        self.nice = int(nice)
        self.max_seconds = float(max_seconds)
        self.procs = []

    def start(self):
        ctx = multiprocessing.get_context("spawn")
        self.deadline = time.monotonic() + self.max_seconds
        try:
            for _ in range(self.workers):
                proc = ctx.Process(target=_burn, args=(self.deadline, os.getpid(), self.nice), daemon=True)
                proc.start()
                self.procs.append(proc)
        except BaseException:
            self.stop()
            raise
        return self

    def stop(self):
        for proc in self.procs:
            if proc.is_alive():
                proc.terminate()
        for proc in self.procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
        self.procs = []


class Sampler(threading.Thread):
    def __init__(self, every=5.0):
        super().__init__(daemon=True)
        self.every, self.samples, self._halt = every, [], threading.Event()

    def run(self):
        while not self._halt.is_set():
            self.samples.append(load_sample())
            self._halt.wait(self.every)

    def finish(self):
        self._halt.set()
        self.join(timeout=15)
        self.samples.append(load_sample())
        return self.samples


def load_summary(samples):
    if not samples:
        return {}
    ones = [s["load1"] for s in samples]
    runnable = [s["runnable"] for s in samples if s["runnable"] is not None]
    return {"load1_start": ones[0], "load1_end": ones[-1], "load1_max": max(ones),
            "load1_mean": round(statistics.fmean(ones), 2),
            "runnable_median": statistics.median(runnable) if runnable else None,
            "samples": samples}


# ------------------------------------------------------------- probe math --

def probe_routes(block):
    return (block or {}).get("routes") or {}


def probe_delta(after, before):
    """Per-route difference of two cumulative round_probe blocks."""
    out = {}
    before_routes = probe_routes(before)
    for route, row in probe_routes(after).items():
        base = before_routes.get(route) or {}
        delta = {key: int(row.get(key) or 0) - int(base.get(key) or 0)
                 for key in ("rounds", "tokens", "wall_ns", "encode_submit_ns",
                             "wait_ns", "host_python_ns", "syncs", "blocking_syncs")}
        # Replay feasibility P0 (MLX2_ROUND_PROBE_GPU): GPU timeline counters.
        for key in GPU_KEYS:
            if key in row:
                delta[key] = int(row.get(key) or 0) - int(base.get(key) or 0)
        for group in ("explicit_syncs", "implicit_syncs"):
            kinds = {}
            for kind, value in (row.get(group) or {}).items():
                prior = (base.get(group) or {}).get(kind) or {}
                kinds[kind] = {k: int(value.get(k) or 0) - int(prior.get(k) or 0)
                               for k in ("calls", "blocking", "encode_submit_ns", "wait_ns")}
            delta[group] = {k: v for k, v in kinds.items() if v["calls"]}
        if delta["rounds"]:
            out[route] = delta
    return out


def add_deltas(total, delta):
    for route, row in delta.items():
        into = total.setdefault(route, {"explicit_syncs": {}, "implicit_syncs": {}})
        for key, value in row.items():
            if isinstance(value, dict):
                for kind, counts in value.items():
                    slot = into[key].setdefault(kind, {"calls": 0, "blocking": 0, "encode_submit_ns": 0, "wait_ns": 0})
                    for k, v in counts.items():
                        slot[k] = slot.get(k, 0) + v
            else:
                into[key] = into.get(key, 0) + value
    return total


GPU_KEYS = (
    "gpu_rounds", "gpu_wall_ns", "gpu_busy_ns", "gpu_eval_ns", "gpu_eval_busy_ns",
    "gpu_eval_tail_ns", "gpu_command_buffers", "recorded_rounds",
)


def gpu_per_round(row):
    """P0 shares from the native GPU timeline (round_probe MLX2_ROUND_PROBE_GPU).

    ``idle_share``: GPU idle over attributed round wall time. ``eval_idle_less_tail_share``:
    idle inside the round's sync windows minus completion latency, the part replaying the
    command stream could remove (conservative P0 bound). ``outside_eval_idle_share``: idle
    during host Python, removable only if graph construction moves into the recorded program.
    """
    rounds, wall = row.get("gpu_rounds") or 0, row.get("gpu_wall_ns") or 0
    if not rounds or not wall:
        return None
    idle = wall - row.get("gpu_busy_ns", 0)
    eval_idle = row.get("gpu_eval_ns", 0) - row.get("gpu_eval_busy_ns", 0)
    removable = max(0, eval_idle - row.get("gpu_eval_tail_ns", 0))
    return {
        "rounds": rounds, "recorded_rounds": row.get("recorded_rounds", 0),
        "wall_ms": round(wall / rounds / 1e6, 4),
        "busy_ms": round(row.get("gpu_busy_ns", 0) / rounds / 1e6, 4),
        "eval_ms": round(row.get("gpu_eval_ns", 0) / rounds / 1e6, 4),
        "eval_idle_ms": round(eval_idle / rounds / 1e6, 4),
        "eval_tail_ms": round(row.get("gpu_eval_tail_ns", 0) / rounds / 1e6, 4),
        "command_buffers": round(row.get("gpu_command_buffers", 0) / rounds, 3),
        "idle_share": round(idle / wall, 4),
        "eval_idle_less_tail_share": round(removable / wall, 4),
        "outside_eval_idle_share": round(max(0, idle - eval_idle) / wall, 4),
    }


def per_round(row):
    rounds = row.get("rounds") or 0
    if not rounds:
        return {}
    ms = lambda key: round(row.get(key, 0) / rounds / 1e6, 4)  # noqa: E731
    calls = lambda group: round(sum(v["calls"] for k, v in row.get(group, {}).items()  # noqa: E731
                                    if k != "async_eval") / rounds, 3)
    out = {"wall_ms": ms("wall_ns"), "encode_submit_ms": ms("encode_submit_ns"),
           "wait_ms": ms("wait_ns"), "host_python_ms": ms("host_python_ns"),
           "tokens": round(row.get("tokens", 0) / rounds, 4),
           "syncs": round(row.get("syncs", 0) / rounds, 3),
           "blocking_syncs": round(row.get("blocking_syncs", 0) / rounds, 3),
           "explicit_syncs": calls("explicit_syncs"), "implicit_syncs": calls("implicit_syncs"),
           "by_kind": {kind: round(v["calls"] / rounds, 3)
                       for group in ("explicit_syncs", "implicit_syncs")
                       for kind, v in sorted(row.get(group, {}).items())}}
    gpu = gpu_per_round(row)
    if gpu is not None:
        out["gpu_timeline"] = gpu
    return out


# ------------------------------------------------------------- GPU trace --

def recording_started(text):
    return "Recording started" in text or "Ctrl-C to stop the recording" in text


class MetalTrace:
    """Bounded ``xctrace`` Metal System Trace attached to one pid."""

    def __init__(self, pid, path, time_limit_s):
        self.pid, self.path, self.limit = int(pid), Path(path), int(time_limit_s)
        self.proc, self.log, self.ready = None, None, False

    def start(self):
        if self.path.exists():
            raise FileExistsError(f"refuse to overwrite {self.path}")
        logpath = self.path.with_suffix(".xctrace.log")
        self.log = logpath.open("xb")
        self.proc = subprocess.Popen(
            ["xcrun", "xctrace", "record", "--template", "Metal System Trace",
             "--attach", str(self.pid), "--time-limit", f"{self.limit}s", "--no-prompt",
             "--output", str(self.path)], stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline and self.proc.poll() is None:
            if recording_started(logpath.read_text(errors="replace")):
                self.ready = True
                break
            time.sleep(0.25)
        if not self.ready:
            self.stop()
        return self.ready

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        if self.log is not None:
            self.log.close()
            self.log = None
        return self.proc.returncode if self.proc is not None else None


def export_gpu_intervals(trace, out_xml):
    subprocess.run(
        ["xcrun", "xctrace", "export", "--input", str(trace), "--xpath",
         '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]',
         "--output", str(out_xml)], check=True, capture_output=True, timeout=600)
    return parse_gpu_intervals(out_xml.read_text())


def parse_gpu_intervals(xml_text):
    """Rows of ``metal-gpu-intervals`` as dicts keyed by column mnemonic.

    xctrace writes each distinct value once with ``id`` and later as ``ref``;
    a ``<sentinel/>`` is an empty cell. Values are (text, fmt).
    """
    root = ET.fromstring(xml_text)
    schema = root.find(".//schema")
    if schema is None:
        return []
    columns = [col.findtext("mnemonic") for col in schema.findall("col")]
    ids, rows = {}, []
    for row in root.iter("row"):
        values = {}
        for column, cell in zip(columns, list(row)):
            if cell.tag == "sentinel":
                continue
            for node in cell.iter():
                if node.get("id") is not None:
                    ids[node.get("id")] = node
            node = ids.get(cell.get("ref"), cell) if cell.get("ref") else cell
            values[column] = ((node.text or "").strip(), node.get("fmt") or "")
        rows.append(values)
    return rows


def gpu_busy(rows, pid=None):
    """Union of GPU interval time (ns) for ``pid`` (all rows when none match)."""
    spans, matched = [], 0
    for row in rows:
        try:
            start = int(row["start"][0])
            duration = int(row["duration"][0])
        except (KeyError, ValueError):
            continue
        process = row.get("process", ("", ""))[1]
        match = re.search(r"\((\d+)\)", process)
        owner = int(match.group(1)) if match else None
        spans.append((owner, start, start + duration))
        matched += pid is not None and owner == pid
    use = [(s, e) for owner, s, e in spans if pid is None or not matched or owner == pid]
    use.sort()
    busy, last_end, first, last = 0, None, None, None
    for start, end in use:
        first = start if first is None else first
        last = end if last is None else max(last, end)
        if last_end is None or start >= last_end:
            busy += end - start
            last_end = end
        elif end > last_end:
            busy += end - last_end
            last_end = end
    return {"intervals": len(use), "pid_matched_intervals": matched,
            "busy_ns": busy, "span_ns": (last - first) if use else 0,
            "filter": "pid" if pid is not None and matched else "all_rows"}


# ------------------------------------------------------------------ run --

def build_prompts(sizes):
    import spec_acceptance_gsm8k as gsm
    from ling32_air_prefill_ab import prompt_text
    _, files = gsm.dataset_files()
    test = gsm.load_split(files["test"])
    return {size: prompt_text(test, size).rsplit("\n\n", 1)[0] + INSTRUCTION for size in sizes}


def parse_size_max_tokens(value):
    """``"65536=32,16384=128"`` -> {65536: 32, 16384: 128}."""
    out = {}
    for item in (value or "").split(","):
        if not item.strip():
            continue
        size, sep, tokens = item.partition("=")
        if not sep or int(size) <= 0 or int(tokens) <= 0:
            raise ValueError(f"--size-max-tokens wants SIZE=TOKENS, got {item!r}")
        out[int(size)] = int(tokens)
    return out


def validate_rerun_options(args, sizes, conditions):
    """Refuse options that cannot produce a complete cell, before any evidence is written."""
    unknown = sorted(set(args.size_max_tokens) - set(sizes))
    if unknown:
        raise SystemExit(f"--size-max-tokens names sizes not in --sizes: {unknown}")
    for size in sizes:
        for condition in conditions:
            # A round commits at least one token, so a request books at most
            # max_tokens rounds.
            reachable = args.max_requests * max_tokens_for(args, size)
            if reachable < rounds_target(args, condition):
                raise SystemExit(
                    f"{condition} {size}: --max-requests {args.max_requests} x max_tokens "
                    f"{max_tokens_for(args, size)} cannot book {rounds_target(args, condition)} rounds")


def max_tokens_for(args, size):
    return (getattr(args, "size_max_tokens", None) or {}).get(size, getattr(args, "max_tokens", None))


def rounds_target(args, condition):
    contended = getattr(args, "contended_rounds", None)
    return contended if condition == "contended" and contended else getattr(args, "rounds", None)


def mtp_transition(receipt):
    """Compact summary of a response receipt's ``mtp`` block (None if absent)."""
    mtp = (receipt or {}).get("mtp")
    if not isinstance(mtp, dict):
        return None
    detail = mtp.get("mtp_plain_fallback") or mtp.get("mtp_ordinary_handoff") or {}
    out = {"route": mtp.get("route"), "num_draft": mtp.get("num_draft"),
           "observed_compute_widths": mtp.get("observed_compute_widths"),
           "ordinary_compute_widths": mtp.get("ordinary_compute_widths")}
    if detail:
        out.update({
            "transition": detail.get("transition") or "handoff",
            "reason": detail.get("reason"),
            "committed_tokens_before": detail.get("committed_tokens_before_fallback",
                                                  detail.get("committed_tokens_before_handoff")),
            "projected_width": detail.get("projected_width"),
            "decision": detail.get("decision"),
        })
    return out


COUNTER_GROUPS = ("counts", "scheduler", "apcv2", "segmented_self_mtp")


def counter_snapshot(status):
    """Flat numeric leaves of the status counter groups."""
    flat = {}

    def walk(prefix, value):
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            flat[prefix] = value
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(f"{prefix}.{key}", item)

    for group in COUNTER_GROUPS:
        walk(group, (status or {}).get(group))
    return flat


def snapshot_collections(status):
    return int(((status or {}).get("host_snapshot_timing") or {}).get("collections") or 0)


def fresh_status(client, wait_s=2.5, poll_s=0.25):
    """``/v1/status`` after two further worker snapshots (bounded).

    scheduler/APCv2/segmented-MTP counters come from a worker snapshot that
    is republished about once a second; a read right after a response can
    miss what the request's end published.  Returns (status, info); a failed
    read yields ({}, {"error": ...}) and never ends the run.
    """
    started = time.monotonic()
    try:
        status = client.call("/v1/status", timeout=30)
        if not isinstance(status, dict):
            raise TypeError(f"status is {type(status).__name__}")
        first = snapshot_collections(status)
        while (snapshot_collections(status) < first + 2
               and time.monotonic() - started < wait_s):
            time.sleep(poll_s)
            status = client.call("/v1/status", timeout=30)
            if not isinstance(status, dict):
                raise TypeError(f"status is {type(status).__name__}")
    except Exception as error:  # noqa: BLE001 - evidence, never fatal
        return {}, {"error": f"{type(error).__name__}: {error}"}
    return status, {"collections": snapshot_collections(status),
                    "fresh": snapshot_collections(status) >= first + 2,
                    "waited_s": round(time.monotonic() - started, 3)}


def counter_delta(after, before):
    return {key: round(value - before.get(key, 0), 6) for key, value in sorted(after.items())
            if value != before.get(key, 0)}
# ``--request-json``: extra request fields for every request of the run (for
# example ``{"presence_penalty": 0}``); recorded in the run receipt.
REQUEST_EXTRA: dict = {}


def request(client, model, text, max_tokens):
    import hashlib
    from ling32_air_prefill_ab import find
    started = time.monotonic()
    result = client.call("/v1/chat/completions", {
        "model": model, "temperature": 0, "stream": False, "max_tokens": max_tokens,
        "enable_thinking": False, "messages": [{"role": "user", "content": text}],
        **REQUEST_EXTRA})
    elapsed = time.monotonic() - started
    usage = result.get("usage") or {}
    completion = int(usage.get("completion_tokens") or 0)
    ttft = find(result, "ttft_seconds")
    decode_s = (elapsed - ttft) if ttft is not None and elapsed > ttft else None
    return {
        "elapsed_s": round(elapsed, 4), "ttft_s": ttft, "decode_s": decode_s,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "completion_tokens": completion,
        "decode_tok_s": ((completion - 1) / decode_s if decode_s and completion > 1 else None),
        "finish_reason": (result.get("choices") or [{}])[0].get("finish_reason"),
        "route": (result.get("mlx2") or {}).get("route"),
        "mtp": mtp_transition(result.get("mlx2")),
        "mlx2_receipt": {k: v for k, v in (result.get("mlx2") or {}).items() if k != "round_probe"},
        # Greedy arms of one snapshot compare their outputs by this digest.
        "text_sha256": hashlib.sha256(
            str(((result.get("choices") or [{}])[0].get("message") or {}).get("content")).encode()
        ).hexdigest(),
    }, find(result, "round_probe")


def measure_cell(client, model, text, args, condition, out_dir, size, server_pid, baseline, load_deadline=None):
    """Warm requests until ``--rounds`` decode rounds; returns (cell, last probe block)."""
    rows, totals, last = [], {}, baseline
    max_tokens, target = max_tokens_for(args, size), rounds_target(args, condition)
    receipts = out_dir / f"receipts-{size}-{condition}.jsonl"
    deltas = getattr(args, "scheduler_deltas", False)
    counters = counter_snapshot(fresh_status(client)[0]) if deltas else None
    sampler = Sampler()
    sampler.start()
    overran = None
    try:
        for index in range(args.max_requests):
            if load_deadline is not None and time.monotonic() >= load_deadline:
                break
            row, block = request(client, model, text, max_tokens)
            receipt = row.pop("mlx2_receipt", None)
            past = load_deadline is not None and time.monotonic() > load_deadline
            if receipt is not None:
                with receipts.open("a") as handle:
                    handle.write(json.dumps({"request": index, "past_deadline": past,
                                             "receipt": receipt}) + "\n")
            if deltas:
                status, info = fresh_status(client)
                row["counter_read"] = info
                if status:
                    now = counter_snapshot(status)
                    row["counter_deltas"] = counter_delta(now, counters or {})
                    counters = now
            if not (block or {}).get("enabled"):
                raise SystemExit("server receipt has no enabled round_probe; start it with MLX2_ROUND_PROBE=1")
            delta = probe_delta(block, last)
            last = block
            if past:
                overran = row
                break
            row["probe"] = delta
            rows.append(row)
            add_deltas(totals, delta)
            print(json.dumps({"size": size, "condition": condition, "request": index,
                              "decode_tok_s": row["decode_tok_s"],
                              "rounds": sum(r["rounds"] for r in delta.values()),
                              "routes": sorted(delta),
                              "mtp_route": (row.get("mtp") or {}).get("route"),
                              "mtp_reason": (row.get("mtp") or {}).get("reason")}), flush=True)
            booked = sum(r.get("rounds", 0) for r in totals.values())
            if booked >= target:
                break
    finally:
        samples = sampler.finish()
    trace = None
    # Trace quiet cells only: the host-share criterion reads the quiet GPU time,
    # and a trace under load would keep the generator running for minutes.
    # Metal traces are large (6-14 GB at 64K on 2026-11-25; they filled the
    # disk): trace only up to --trace-max-size, a short request, with free
    # space checked first, and delete the bundle once parsed.
    free_gib = shutil.disk_usage(out_dir).free / 2**30
    if args.xctrace and server_pid and condition == "quiet" and (
            size > args.trace_max_size or free_gib < args.trace_min_free_gib):
        trace = {"status": "skipped", "free_gib": round(free_gib, 1),
                 "reason": (f"context {size} > --trace-max-size {args.trace_max_size}"
                            if size > args.trace_max_size
                            else f"free disk {free_gib:.0f} GiB < {args.trace_min_free_gib}")}
    elif args.xctrace and server_pid and condition == "quiet":
        trace_path = out_dir / f"metal-{size}-{condition}.trace"
        tracer = MetalTrace(server_pid, trace_path, args.trace_limit_s)
        trace = {"path": str(trace_path), "status": "unavailable"}
        try:
            try:
                if tracer.start():
                    row, block = request(client, model, text, args.trace_max_tokens)
                    receipt = row.pop("mlx2_receipt", None)
                    if receipt is not None:
                        with receipts.open("a") as handle:
                            handle.write(json.dumps({"request": "trace", "receipt": receipt}) + "\n")
                    delta = probe_delta(block, last)
                    last = block
                    trace.update(status="recorded", request=row, probe=delta)
            finally:
                trace["recorder_returncode"] = tracer.stop()
            if trace["status"] == "recorded" and trace_path.exists():
                try:
                    rows_xml = export_gpu_intervals(trace_path, out_dir / f"metal-{size}-{condition}.xml")
                    busy = gpu_busy(rows_xml, server_pid)
                    rounds = sum(r["rounds"] for r in trace["probe"].values())
                    wall = sum(r["wall_ns"] for r in trace["probe"].values())
                    traced = trace["request"]
                    truncated = traced["elapsed_s"] >= args.trace_limit_s - 5
                    usable = busy["filter"] == "pid" and not truncated and rounds
                    trace.update(status="parsed" if usable else "unusable", gpu=busy,
                                 truncated=truncated,
                                 unusable_reason=None if usable else (
                                     "no interval matched the server pid" if busy["filter"] != "pid"
                                     else "trace time limit reached" if truncated else "no rounds"),
                                 gpu_ms_per_round=(busy["busy_ns"] / rounds / 1e6) if usable else None,
                                 probe_wall_ms_per_round=(wall / rounds / 1e6) if rounds else None,
                                 note="GPU busy includes the request's short warm prefill")
                except (subprocess.SubprocessError, OSError, ET.ParseError) as error:
                    trace.update(status="export_failed", error=f"{type(error).__name__}: {error}")
        finally:
            # start() reserves the log exclusively before launching xctrace.
            # Only a launched recorder can own a partial bundle here.
            if tracer.proc is not None and trace_path.exists():
                trace["trace_bytes"] = sum(f.stat().st_size for f in trace_path.rglob("*") if f.is_file()) \
                    if trace_path.is_dir() else trace_path.stat().st_size
                if not args.keep_traces:
                    if trace_path.is_dir():
                        shutil.rmtree(trace_path)
                    else:
                        trace_path.unlink()
                    trace["trace_deleted"] = True
    # Terminal ordinary steps and speculative rounds can share a request.
    # Both consumed the measured decode interval.
    combined = {}
    for counters in totals.values():
        add_deltas(combined, {"all": counters})
    top = combined.get("all", {})
    name = ", ".join(sorted(totals)) or None
    decode = [r["decode_tok_s"] for r in rows if r["decode_tok_s"]]
    decode_s = sum(r["decode_s"] or 0 for r in rows)
    rounds = top.get("rounds", 0)
    pr = per_round(top)
    if pr and decode_s and rounds:
        # Round period as the client sees it: in-scope round time plus the
        # serving loop between rounds (publication, detokenization, APCv2).
        pr["period_ms"] = round(decode_s * 1000 / rounds, 4)
        pr["outside_ms"] = round(max(0.0, pr["period_ms"] - pr["wall_ms"]), 4)
    cell = {
        "size": size, "condition": condition, "requests": rows,
        "route_label": name, "routes": totals, "request_past_load_deadline": overran,
        "rounds": rounds, "rounds_target": target, "max_tokens": max_tokens,
        "route_transitions": [
            {"request": i, **r["mtp"]} for i, r in enumerate(rows)
            if (r.get("mtp") or {}).get("transition")],
        # A warm request that re-prefilled (no APCv2 hit) measured decode
        # only, but its prefill ate the contended load window (Ling 64K,
        # 2026-11-25: cached 0 of 70,905, TTFT 85-110 s).
        "cold_requests": sum(1 for r in rows if r.get("prompt_tokens") and r.get("cached_tokens") == 0),
        "partial_hit_requests": sum(1 for r in rows if r.get("prompt_tokens") and r.get("cached_tokens")
                                    and r["cached_tokens"] < r["prompt_tokens"] - 1),
        # A route the receipt does not report as a transition can still
        # change under one request (Maxwell 16K, 2026-11-25: a target-only
        # APCv2 hit decodes on the ordinary batcher with no mtp receipt).
        "probe_route_changes": [
            {"request": i, "routes": sorted(r.get("probe") or {}), "first_routes": sorted(rows[0].get("probe") or {})}
            for i, r in enumerate(rows) if i and sorted(r.get("probe") or {}) != sorted(rows[0].get("probe") or {})],
        "rounds_per_s": (rounds / decode_s) if decode_s and rounds else None,
        "decode_tok_s_median": statistics.median(decode) if decode else None,
        "per_round": pr,
        "prompt_tokens": statistics.median([r["prompt_tokens"] for r in rows if r["prompt_tokens"]] or [0]) or None,
        "load": load_summary(samples),
        "metal_trace": trace,
    }
    return cell, last


def run(args):
    import spec_acceptance_gsm8k as gsm
    # A TERM from the unit's trap must unwind through ``finally`` so the load
    # workers stop with the harness (they also exit when orphaned).
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    sizes = [int(v) for v in args.sizes.split(",")]
    conditions = [c for c in args.conditions.split(",") if c]
    try:
        args.size_max_tokens = parse_size_max_tokens(args.size_max_tokens)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    validate_rerun_options(args, sizes, conditions)
    if args.request_json:
        extra = json.loads(args.request_json)
        fixed = {"model", "messages", "max_tokens", "stream", "temperature", "enable_thinking"}
        if not isinstance(extra, dict) or set(extra) & fixed:
            raise SystemExit("--request-json must be an object that does not set " + "/".join(sorted(fixed)))
        REQUEST_EXTRA.clear()
        REQUEST_EXTRA.update(extra)
    out_dir = Path(args.output)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refuse nonempty evidence directory {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    client = gsm.Client(args.base_url, Path(args.admin_token).read_text().strip())
    status = client.call("/v1/status", timeout=30)
    model = gsm.served_model_name(status, args.checkpoint)
    prompts = build_prompts(sizes)
    meta = {
        "schema": SCHEMA, "checkpoint": args.checkpoint, "arm": args.arm,
        "route": args.route, "source_revision": args.source_revision,
        "served_artifact": status.get("artifact"), "runtime": status.get("runtime"),
        "route_receipt": status.get("route_receipt"),
        "settings": status.get("settings"),
        "command_buffer_env": ((status.get("round_probe") or {}).get("command_buffer_env")
                               or {k: os.environ.get(k) for k in ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER")}),
        "load_workers": args.load_workers, "load_nice": args.load_nice,
        "load_ramp_s": args.load_ramp_s, "cell_max_s": args.cell_max_s, "cooldown_s": args.cooldown_s,
        "rounds_target": args.rounds, "request_extra": dict(REQUEST_EXTRA),
        "rounds_target_contended": rounds_target(args, "contended"),
        "max_tokens": args.max_tokens,
        "size_max_tokens": {str(k): v for k, v in sorted(args.size_max_tokens.items())},
        "scheduler_deltas": bool(args.scheduler_deltas), "server_pid": args.server_pid,
        "host": {"cpus": os.cpu_count()}, "started_at": time.time(),
        "thermal_admission": False, "performance_claim": False,
        "purpose": "Phase 0 attribution (quiet vs contended); not publishable statistics",
    }
    cells, last = [], None
    for condition in conditions:  # every quiet cell before any contended cell
        for size in sizes:
            client.call("/v1/admin/apc/clear", {"reason": f"phase0-{size}"}, admin=True)
            warm, last = request(client, model, prompts[size], 8)  # cold prefill, unmeasured
            warm.pop("mlx2_receipt", None)
            generator = None
            if not (last or {}).get("enabled"):
                raise SystemExit("warmup receipt has no enabled round_probe")
            try:
                if condition == "contended":
                    generator = LoadGenerator(args.load_workers, args.load_ramp_s + args.cell_max_s, nice=args.load_nice)
                    generator.start()
                    time.sleep(args.load_ramp_s)
                cell, last = measure_cell(client, model, prompts[size], args, condition,
                                          out_dir, size, args.server_pid, last,
                                          generator.deadline if generator is not None else None)
            finally:
                if generator is not None:
                    generator.stop()
            cell["warmup"] = warm
            cells.append(cell)
            gsm.write_atomic(out_dir / f"cell-{size}-{condition}.json", cell)
            if generator is not None:
                time.sleep(args.cooldown_s)
    status_after = client.call("/v1/status", timeout=30)
    document = {**meta, "finished_at": time.time(), "cells": cells,
                "round_probe_final": status_after.get("round_probe"),
                "scheduler_final": status_after.get("scheduler")}
    gsm.write_atomic(out_dir / "phase0.json", document)
    print(json.dumps(summarize(document), indent=1))


# --------------------------------------------------------------- report --

def summarize(document):
    """Go/no-go per (checkpoint, arm, size) against §4.4."""
    by_size = {}
    for cell in document.get("cells", []):
        by_size.setdefault(cell["size"], {})[cell["condition"]] = cell
    rows = []
    for size, pair in sorted(by_size.items()):
        quiet, contended = pair.get("quiet"), pair.get("contended")
        row = {"checkpoint": document.get("checkpoint"), "arm": document.get("arm"),
               "route": document.get("route"), "size": size,
               "source_revision": document.get("source_revision"),
               "command_buffer_env": document.get("command_buffer_env")}
        for name, cell in (("quiet", quiet), ("contended", contended)):
            if cell is None:
                continue
            pr = cell.get("per_round") or {}
            trace = cell.get("metal_trace") or {}
            row[name] = {
                "rounds": cell.get("rounds"),
                "rounds_target": cell.get("rounds_target") or (
                    (document.get("rounds_target_contended") or document.get("rounds_target"))
                    if name == "contended" else document.get("rounds_target")),
                "max_tokens": cell.get("max_tokens"),
                # Requests, not events: a fallback row shows in both lists.
                "route_transitions": (None if "route_transitions" not in cell else len(
                    {t["request"] for t in cell["route_transitions"]}
                    | {t["request"] for t in cell.get("probe_route_changes") or []})),
                "cold_requests": cell.get("cold_requests"),
                "partial_hit_requests": cell.get("partial_hit_requests"),
                "route_label": cell.get("route_label"),
                "decode_tok_s": cell.get("decode_tok_s_median"),
                "rounds_per_s": cell.get("rounds_per_s"),
                "tokens_per_round": pr.get("tokens"),
                "prompt_tokens": cell.get("prompt_tokens"),
                "period_ms": pr.get("period_ms"), "outside_ms": pr.get("outside_ms"),
                "wall_ms": pr.get("wall_ms"), "encode_submit_ms": pr.get("encode_submit_ms"),
                "wait_ms": pr.get("wait_ms"), "host_python_ms": pr.get("host_python_ms"),
                "gpu_ms": trace.get("gpu_ms_per_round"),
                "syncs": pr.get("syncs"), "blocking_syncs": pr.get("blocking_syncs"),
                "explicit_syncs": pr.get("explicit_syncs"),
                "implicit_syncs": pr.get("implicit_syncs"),
                "load1_mean": (cell.get("load") or {}).get("load1_mean"),
                "runnable_median": (cell.get("load") or {}).get("runnable_median"),
            }
        q, c = row.get("quiet"), row.get("contended")
        slowdown = None
        if q and c and q["decode_tok_s"] and c["decode_tok_s"]:
            slowdown = 1 - c["decode_tok_s"] / q["decode_tok_s"]
        host_share, basis, in_scope = None, None, None
        if q and q.get("wall_ms"):
            period = q.get("period_ms") or q["wall_ms"]
            outside = q.get("outside_ms") or 0.0
            if q.get("gpu_ms") is not None:
                host_share, basis = max(0.0, 1 - q["gpu_ms"] / period), "1 - gpu_ms/period_ms"
            else:
                host_share = (q["encode_submit_ms"] + q["host_python_ms"] + outside) / period
                basis = "(encode_submit_ms + host_python_ms + outside_ms)/period_ms"
            in_scope = (q["encode_submit_ms"] + q["host_python_ms"]) / q["wall_ms"]
        mal_drift = None
        if q and c and q["tokens_per_round"] and c["tokens_per_round"]:
            mal_drift = c["tokens_per_round"] / q["tokens_per_round"] - 1
        row["contended_slowdown"] = slowdown
        row["quiet_host_share"] = host_share
        row["host_share_basis"] = basis
        row["quiet_in_round_host_share"] = in_scope
        row["tokens_per_round_drift"] = mal_drift
        target = document.get("rounds_target", 0)
        quiet_complete = q is not None and (q["rounds"] or 0) >= (q.get("rounds_target") or target)
        pair_complete = quiet_complete and c is not None and (c["rounds"] or 0) >= (c.get("rounds_target") or target)
        # Different request lengths answer different greedy prefixes.
        if q and c and q.get("max_tokens") and c.get("max_tokens") and q["max_tokens"] != c["max_tokens"]:
            pair_complete = False
        period_increase = None
        if q and c and q["rounds_per_s"] and c["rounds_per_s"]:
            period_increase = q["rounds_per_s"] / c["rounds_per_s"] - 1
        row["contended_period_increase"] = period_increase
        # Comparing different numbers of tokens per round confounds work per
        # round with host contention on this fixed, greedy prompt.
        # A cell where some request left the probed route (an MTP lane that
        # decoded as a target-only plain fallback) averages two routes'
        # rounds: neither criterion describes either route.  Older documents
        # without transition data (None) keep their verdicts.
        mixed = [name for name, cell in (("quiet", q), ("contended", c))
                 if cell and cell.get("route_transitions")]
        row["mixed_route_cells"] = mixed
        comparable = (pair_complete and mal_drift == 0 and period_increase is not None
                      and not mixed)
        valid_host_share = quiet_complete and host_share is not None and "quiet" not in mixed
        reasons = []
        if comparable and period_increase >= GO_PERIOD_INCREASE - 1e-12:
            reasons.append(f"contended round period {period_increase:.1%} longer at unchanged tokens/round")
        if valid_host_share and host_share >= GO_HOST_SHARE:
            reasons.append(f"quiet host share {host_share:.1%} >= 15%")
        row["go_reasons"] = reasons
        row["phase1_go"] = True if reasons else None
        if not reasons and comparable and valid_host_share:
            row["phase1_go"] = False
        rows.append(row)
    return {"schema": SCHEMA + ".summary", "rows": rows}


def compare_gsm8k_arms(runs, names, reference, pairs):
    """Compare complete base-URL GSM8K receipts for the selected Phase 0 arms."""
    from technology_parity_compare import indexed_rows

    output_keys = ("text", "completion_tokens", "predicted", "finish_reason")
    keys = (*output_keys, "accept_hist", "depth_hist", "token_ids")
    report = {"reference": reference, "pairs": pairs, "keys": list(keys),
              "arms": {}, "incomplete": {}}
    loaded = {}
    for name in names:
        path = runs / name / "gsm8k"
        if not (path / "receipt.json").is_file():
            report["incomplete"][name] = "missing receipt"
            continue
        receipt = json.loads((path / "receipt.json").read_text())
        dataset = receipt.get("dataset") or {}
        count = dataset.get("questions")
        arms = receipt.get("arms") or []
        # These launchers always request the contiguous prefix of the test split.
        if (len(arms) != 1 or type(count) is not int or count < 1
                or type(dataset.get("offset")) is not int or dataset["offset"] != 0
                or "indices" in dataset):
            report["incomplete"][name] = "unexpected arm or question selection"
            continue
        identity = (receipt.get("checkpoint"), (receipt.get("driver_revision") or {}).get("head"),
                    dataset.get("revision"), count)
        if not all(isinstance(item, str) and item.strip() for item in identity[:3]):
            report["incomplete"][name] = "missing checkpoint or revision"
            continue
        arm = arms[0]
        rows = indexed_rows(arm, range(count), path)
        if rows is None or (arm.get("summary") or {}).get("questions") != count:
            report["incomplete"][name] = f"expected {count} complete question rows"
            continue
        if not all(isinstance(row.get("token_ids"), list)
                   and len(row["token_ids"]) == row["completion_tokens"]
                   and all(type(t) is int and t >= 0 for t in row["token_ids"])
                   and type(row["test_index"]) is int and "predicted" in row for row in rows.values()):
            report["incomplete"][name] = "missing or incomplete token evidence"
            continue
        loaded[name] = rows, arm["summary"], identity

    if reference in loaded:
        report["reference_summary"] = loaded[reference][1]
    for name in names:
        against = pairs.get(name, reference)
        if name == against or against not in names:
            # ARMS can intentionally select one treatment without its reference.
            continue
        entry = {"against": against, "compared": 0, "token_exact": None, "exact": None}
        report["arms"][name] = entry
        if name not in loaded or against not in loaded:
            report["incomplete"].setdefault(name, f"cannot compare with {against}")
            continue
        got, summary, identity = loaded[name]
        ref, _, ref_identity = loaded[against]
        if identity != ref_identity:
            report["incomplete"][name] = "checkpoint, revision or question selection differs"
            continue
        token_diffs = [i for i in ref if got[i]["token_ids"] != ref[i]["token_ids"]]
        diffs = [i for i in ref if any(got[i][key] != ref[i][key] for key in keys)]
        entry.update(compared=len(ref), token_mismatched=token_diffs, token_exact=not token_diffs,
                     acceptance_or_token_mismatched=diffs, mismatched=diffs, exact=not diffs,
                     checkpoint=identity[0], source_revision=identity[1], summary=summary)
    return report


def fmt(value, spec=".1f"):
    return "-" if value is None else format(value, spec)


def report(args):
    rows = []
    for path in args.inputs:
        rows.extend(summarize(json.loads(Path(path).read_text()))["rows"])
    lines = ["| checkpoint | mlx2 revision | arm | prompt tok | quiet tok/s | contended tok/s | round time increase | tok/round q/c | "
             "period ms q | in-round ms q | encode ms q | wait ms q | python ms q | between ms q | GPU ms q | "
             "blocking syncs/round q | runnable c (load1) | host share q | route transitions q/c | Phase 1 |",
             "|" + "---|" * 20]
    for r in rows:
        q, c = r.get("quiet") or {}, r.get("contended") or {}
        lines.append("| " + " | ".join([
            str(r["checkpoint"]), str(r["source_revision"] or "-"), str(r["arm"]),
            fmt(q.get("prompt_tokens") or r["size"], ".0f"),
            fmt(q.get("decode_tok_s")), fmt(c.get("decode_tok_s")),
            fmt(r["contended_period_increase"], ".1%"),
            f"{fmt(q.get('tokens_per_round'), '.2f')}/{fmt(c.get('tokens_per_round'), '.2f')}",
            fmt(q.get("period_ms"), ".2f"), fmt(q.get("wall_ms"), ".2f"),
            fmt(q.get("encode_submit_ms"), ".2f"), fmt(q.get("wait_ms"), ".2f"),
            fmt(q.get("host_python_ms"), ".2f"), fmt(q.get("outside_ms"), ".2f"),
            fmt(q.get("gpu_ms"), ".2f"), fmt(q.get("blocking_syncs"), ".1f"),
            f"{fmt(c.get('runnable_median'), '.0f')} ({fmt(c.get('load1_mean'), '.0f')})",
            fmt(r["quiet_host_share"], ".1%"),
            f"{fmt(q.get('route_transitions'), '.0f')}/{fmt(c.get('route_transitions'), '.0f')}",
            {True: "go", False: "no-go", None: "mixed route" if r.get("mixed_route_cells")
             else "incomplete"}[r["phase1_go"]]]) + " |")
    text = "\n".join(lines) + "\n"
    document = {"schema": SCHEMA + ".report", "rows": rows,
                "criteria": {"contended_period_increase": GO_PERIOD_INCREASE, "quiet_host_share": GO_HOST_SHARE}}
    if args.output:
        Path(args.output).write_text(json.dumps(document, indent=1) + "\n")
        Path(args.output).with_suffix(".md").write_text(text)
    print(text)


def load(args):
    generator = LoadGenerator(args.workers, args.seconds + 30).start()
    try:
        end = time.time() + args.seconds
        while time.time() < end:
            time.sleep(min(5, max(0, end - time.time())))
            print(json.dumps(load_sample()), flush=True)
    finally:
        generator.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base-url", required=True)
    r.add_argument("--admin-token", required=True)
    r.add_argument("--checkpoint", required=True, help="<artifact name>@<fingerprint>")
    r.add_argument("--arm", required=True)
    r.add_argument("--route", default="", help="human route description for the receipt")
    r.add_argument("--source-revision", required=True, help="mlx2 commit the server runs")
    r.add_argument("--sizes", default="1024,16384")
    r.add_argument("--conditions", default="quiet,contended")
    r.add_argument("--rounds", type=int, default=200, help="minimum decode rounds per cell")
    r.add_argument("--max-requests", type=int, default=10)
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--size-max-tokens", default="",
                   help="SIZE=TOKENS[,...]: max_tokens for that context size, quiet and contended "
                        "alike (a 256-token contended request at 64K outlasts the load window)")
    r.add_argument("--contended-rounds", type=int, default=None,
                   help="round target of contended cells (default --rounds)")
    r.add_argument("--scheduler-deltas", action="store_true",
                   help="read /v1/status around each request; store changed scheduler/APCv2 counters")
    r.add_argument("--load-workers", type=int, default=48)
    r.add_argument("--load-nice", type=int, choices=range(20), default=0,
                   help="nice increment for the load workers (0 = default priority, as the real pytest load)")
    r.add_argument("--load-ramp-s", type=float, default=10)
    r.add_argument("--cell-max-s", type=float, default=50,
                   help="contended: stop measuring after this long under load")
    r.add_argument("--cooldown-s", type=float, default=60)
    r.add_argument("--server-pid", type=int)
    r.add_argument("--request-json", default=None,
                   help='extra request fields for every request, e.g. \'{"presence_penalty": 0}\'')
    r.add_argument("--xctrace", action="store_true")
    r.add_argument("--trace-limit-s", type=int, default=60)
    r.add_argument("--trace-max-size", type=int, default=16384, help="trace only cells up to this context")
    r.add_argument("--trace-max-tokens", type=int, default=64, help="max_tokens of the traced request")
    r.add_argument("--trace-min-free-gib", type=float, default=60)
    r.add_argument("--keep-traces", action="store_true", help="keep .trace bundles after parsing")
    r.add_argument("--output", required=True)
    p = sub.add_parser("report")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--output")
    g = sub.add_parser("load")
    g.add_argument("--workers", type=int, default=64)
    g.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args(argv)
    {"run": run, "report": report, "load": load}[args.command](args)


if __name__ == "__main__":
    main()
