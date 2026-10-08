#!/usr/bin/env python3
"""Instrumented HTTP A/B profiler for mlx2 serving features.

Each arm is one server configuration (CLI args, environment, execution
policy).  Every rep launches each arm in its own fresh process (rotated arm
order, odd reps reversed) with a fresh APCv2 directory, warms it with one
discarded request, then runs the selected workloads through the real HTTP
path.  Per workload it records:

* client timing: TTFT, inter-token gaps (p50/p95/max), decode tok/s
  (tokens after the first divided by first-to-last span), wall time;
* every request's ``mlx2`` receipt (route, speculation, prefill chunks);
* the delta of every numeric leaf in ``/v1/status`` and ``/metrics`` across
  the workload, i.e. which counters the workload actually moved (the
  engagement evidence for a feature);
* host samples: vm_stat swapouts and ``pmset -g therm`` before/after.

Token identity: the hash of each request's generated text is compared with
the first arm of the same rep, so a speed delta can be read together with
whether the outputs changed.

Run it under the shared GPU lease, e.g.::

  scripts/run_with_gpu_locks.py --session S --label L --receipt r.json -- \\
    .venv/bin/python scripts/profile_features.py --spec spec.json --out DIR

The spec is JSON::

  {"model": "/path", "common_args": ["--max-lanes", "8"],
   "arms": {"base": {"args": [], "env": {}, "policy": {...}},
            "cand": {"args": ["--lane-matmul", "off"]}},
   "workloads": ["short", "long:8000", "batch:4", "contention:16000", ...],
   "reps": 2, "max_tokens": 256}

Optional spec keys:

* ``"sampling": {"temperature": 0.7, "top_p": ..., "top_k": ..., "seed": 1}``
  -- request sampling (default greedy).  With ``seed`` every request carries
  ``seed + crc32(messages)``, the same in every arm.
* ``"vary_prompts_by_rep": false`` -- every rep sends the same prompts, so a
  seeded arm's reps must reproduce each other (``identical_across_reps``).
* ``"status_capture": ["status.execution.qsa_stage1"]`` -- keep the raw
  ``/v1/status`` values (and their per-workload deltas) under these prefixes;
  ``select_timing`` histograms found there are reduced to P50/P99.
* ``"status_sample_s": 1.0`` -- poll interval of the ``agents`` workload's
  resident-bytes sampler.

Workload ``agents:N[:P[:R]]``: N agent sessions, each with its own P-token
system prefix, R requests in all (default 6:6000:36), interleaved by a
seeded Zipf schedule; each request extends its session's history with the
previous answers.  It reports prefill ms saved by APCv2 hits (cached tokens
x the cold per-token prefill time measured in the same run) per resident
APCv2 GiB-second (``apcv2.idle_disk.resident_bytes`` integrated over the
workload): the B3 value-retention metric, both as cached tokens x the cold
rate and as measured (cold estimate minus observed TTFT, which charges disk
restores).  Workload ``serial:N``: N short prompts in sequence at B1.

Speculative receipts are reduced per workload to accepted draft tokens and
emitted tokens per verify round (``apv`` / ``tpv`` in the summary).

``--dry-run`` validates every arm without a server or model weights: the
server argparse, route selection and adapter policy defaults, the
server-owned policy keys, the adapter policy where a static validator
exists (``--deep`` adds the payload-hashing revision checks), the model and
drafter paths, and the env var names.  ``--rep-start``/``--reps`` run a slice
of the reps with the same arm rotation, to keep each GPU job short.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PYTHON = os.environ.get("MLX2_PROFILE_PYTHON", sys.executable)

SHORT_PROMPTS = [
    "Explain how a hash map handles collisions, in one paragraph.",
    "Write a short Python function that merges two sorted lists, then explain it briefly.",
    "Summarize the causes of the French Revolution in five sentences.",
    "Describe how a CPU cache hierarchy works, in one paragraph.",
    "Give three tips for writing clear technical documentation, with one example each.",
    "Explain the difference between TCP and UDP in plain language.",
    "Write a haiku sequence (three haiku) about autumn rain.",
    "Explain what a Bloom filter is and when to use one.",
]


# --------------------------------------------------------------------------- host


def swapouts() -> int | None:
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    m = re.search(r"Swapouts:\s+(\d+)", out)
    return int(m.group(1)) if m else None


def thermal() -> str | None:
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    lines = [ln.strip() for ln in out.splitlines() if "CPU_Speed_Limit" in ln or "warning" in ln.lower()]
    return "; ".join(lines) or out.strip()[-200:]


# --------------------------------------------------------------------------- corpus


def corpus_text() -> str:
    parts = []
    for p in sorted((ROOT / "docs").glob("*.md")):
        try:
            parts.append(p.read_text())
        except Exception:
            pass
    return "\n\n".join(parts)


CORPUS = None


def document(tokens: int, offset: int) -> str:
    """About ``tokens`` tokens of repository prose (4 chars ~ 1 token)."""
    global CORPUS
    if CORPUS is None:
        CORPUS = corpus_text()
    chars = tokens * 4
    start = (offset * 7919) % max(1, len(CORPUS) - chars - 1)
    body = CORPUS[start:start + chars]
    while len(body) < chars:
        body += "\n\n" + CORPUS[: chars - len(body)]
    return body


# --------------------------------------------------------------------------- http


def flatten(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(value, list):
        # Per-table diagnostics (e.g. execution.ple_tables) are lists of dicts.
        for i, v in enumerate(value):
            out.update(flatten(v, f"{prefix}{i}."))
    elif isinstance(value, bool):
        pass
    elif isinstance(value, (int, float)):
        out[prefix[:-1]] = value
    return out


def parse_metrics(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            name, val = line.rsplit(" ", 1)
            out[name] = float(val)
        except ValueError:
            continue
    return out


class Client:
    def __init__(self, port: int, sampling=None):
        self.base = f"http://127.0.0.1:{port}"
        self.model = None
        # Spec-level request sampling; ``seed`` is a base, see request_body.
        self.sampling = dict(sampling or {})

    def get(self, path, timeout=30):
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            body = r.read().decode()
        return json.loads(body) if path != "/metrics" else body

    def snapshot(self):
        snap = {}
        try:
            snap.update({"status." + k: v for k, v in flatten(self.get("/v1/status")).items()})
        except Exception as exc:  # pragma: no cover - diagnostic only
            snap["status_error"] = str(exc)
        try:
            snap.update({"batching." + k: v for k, v in flatten(self.get("/v1/status/batching")).items()})
        except Exception:
            pass
        try:
            snap.update({"metrics." + k: v for k, v in parse_metrics(self.get("/metrics")).items()})
        except Exception:
            pass
        return snap

    def request_body(self, messages, *, max_tokens, temperature=None, extra=None):
        """The chat body: spec sampling, then an explicit temperature, then extra.

        A seeded spec gives every request ``seed + crc32(messages)``: the same
        request draws the same seed in every arm and rep.
        """
        sampling = dict(self.sampling)
        seed = sampling.pop("seed", None)
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": 0.0, "stream": True,
                "stream_options": {"include_usage": True}}
        body.update(sampling)
        if temperature is not None:
            body["temperature"] = temperature
        if seed is not None:
            digest = zlib.crc32(json.dumps(messages, sort_keys=True).encode())
            body["seed"] = (int(seed) + digest) % (1 << 31)
        if extra:
            body.update(extra)
        return body

    def chat(self, messages, *, max_tokens, temperature=None, extra=None, timeout=1800):
        """Stream one chat request; return timing + text + receipt."""
        body = self.request_body(messages, max_tokens=max_tokens,
                                 temperature=temperature, extra=extra)
        req = urllib.request.Request(self.base + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        t0 = time.perf_counter()
        arrivals, pieces, usage, receipt, error = [], [], None, None, None
        content = []
        tool_calls = {}
        finish = None
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                for raw in r:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    if chunk.get("mlx2"):
                        receipt = chunk["mlx2"]
                    for ch in chunk.get("choices", []):
                        delta = ch.get("delta", {})
                        content.append(delta.get("content") or "")
                        for tc in delta.get("tool_calls") or []:
                            slot = tool_calls.setdefault(tc.get("index", len(tool_calls)), {
                                "id": None, "type": "function",
                                "function": {"name": "", "arguments": ""}})
                            slot["id"] = tc.get("id") or slot["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name") and not slot["function"]["name"]:
                                slot["function"]["name"] = fn["name"]
                            slot["function"]["arguments"] += fn.get("arguments") or ""
                        txt = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                        if delta.get("tool_calls"):
                            txt += json.dumps(delta["tool_calls"])
                        if txt:
                            arrivals.append(time.perf_counter())
                            pieces.append(txt)
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
        except urllib.error.HTTPError as exc:
            error = f"HTTP {exc.code}: {exc.read()[:300]!r}"
        except Exception as exc:
            error = repr(exc)[:300]
        t_end = time.perf_counter()
        text = "".join(pieces)
        gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
        toks = (usage or {}).get("completion_tokens") or len(arrivals)
        span = (arrivals[-1] - arrivals[0]) if len(arrivals) > 1 else None
        return {
            "t0": t0, "t_first": arrivals[0] if arrivals else None, "t_end": t_end,
            "ttft_s": (arrivals[0] - t0) if arrivals else None,
            "wall_s": t_end - t0,
            "completion_tokens": toks,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
            "decode_tps": ((toks - 1) / span) if span and toks > 1 else None,
            "gap_p50_ms": 1e3 * statistics.median(gaps) if gaps else None,
            "gap_p95_ms": 1e3 * sorted(gaps)[int(0.95 * (len(gaps) - 1))] if gaps else None,
            "gap_max_ms": 1e3 * max(gaps) if gaps else None,
            "arrivals": arrivals,
            "sha": hashlib.sha256(text.encode()).hexdigest()[:16],
            "text_head": text[:160],
            # The answer alone (no reasoning), for multi-turn history; dropped
            # from the saved results by strip().
            "content": "".join(content),
            "tool_calls": [tool_calls[k] for k in sorted(tool_calls)],
            "finish": finish,
            "error": error,
            "receipt": receipt,
        }


# --------------------------------------------------------------------------- server


def arm_policy(spec, arm):
    """The arm's execution policy: common_policy overlaid by the arm's."""
    policy = dict(spec.get("common_policy", {}))
    policy.update(arm.get("policy", {}))
    return policy


def arm_env(spec, arm):
    """Only the spec-set variables (common_env overlaid by the arm's env)."""
    env = {k: str(v) for k, v in spec.get("common_env", {}).items()}
    env.update({k: str(v) for k, v in arm.get("env", {}).items()})
    return env


def server_argv(spec, arm, port, tmp: Path):
    """``mlx2.server`` arguments for one arm (after ``-m mlx2.server``).

    Writes the policy file under ``tmp`` when the arm has a policy.
    """
    args = ["--model", spec["model"], "--host", "127.0.0.1", "--port", str(port),
            "--cache-dir", str(tmp / "apc")]
    args += list(spec.get("common_args", []))
    policy = arm_policy(spec, arm)
    if policy:
        pf = tmp / "policy.json"
        pf.write_text(json.dumps(policy))
        args += ["--execution-policy", str(pf)]
    args += list(arm.get("args", []))
    return args


class Server:
    def __init__(self, spec, arm_name, arm, port, logdir, trace_path=None):
        self.port = port
        self.tmp = Path(tempfile.mkdtemp(prefix=f"mlx2prof-{arm_name}-"))
        args = [PYTHON, "-u", "-m", "mlx2.server"] + server_argv(spec, arm, port, self.tmp)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        env.update(arm_env(spec, arm))
        if spec.get("loop_trace") and trace_path is not None:
            # Opt-in serving-loop timeline (src/mlx2/runtime/loop_trace.py).
            env["MLX2_LOOP_TRACE"] = str(trace_path)
        self.log = open(logdir / f"server-{arm_name}-{port}.log", "w")
        self.args = args
        self.t_launch = time.perf_counter()
        self.proc = subprocess.Popen(args, env=env, cwd=str(ROOT), stdout=self.log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        self.client = Client(port, spec.get("sampling"))

    def wait_ready(self, timeout=1200):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited rc={self.proc.returncode}")
            try:
                st = self.client.get("/v1/status", timeout=5)
                if st.get("healthy"):
                    self.client.model = self.client.get("/v1/models")["data"][0]["id"]
                    self.load_s = time.perf_counter() - self.t_launch
                    return st
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError("server not ready")

    def stop(self):
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=60)
        except Exception:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except Exception:
                pass
        self.log.close()
        shutil.rmtree(self.tmp, ignore_errors=True)


# --------------------------------------------------------------------------- workloads


def user(text):
    return [{"role": "user", "content": text}]


def concurrent(client, jobs):
    results = [None] * len(jobs)

    def run(i, job):
        results[i] = client.chat(**job)

    threads = [threading.Thread(target=run, args=(i, j)) for i, j in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def aggregate(results):
    firsts = [r["t_first"] for r in results if r["t_first"]]
    ends = [r["t_end"] for r in results]
    toks = sum(r["completion_tokens"] or 0 for r in results)
    all_started = max(firsts) if len(firsts) == len(results) else None
    span = (max(ends) - min(firsts)) if firsts else None
    steady = None
    if all_started:
        # tokens delivered after every lane has started, over that span
        n = sum(sum(1 for a in r["arrivals"] if a > all_started) for r in results)
        last = max(max(r["arrivals"]) for r in results if r["arrivals"])
        if last > all_started:
            steady = n / (last - all_started)
    return {"lanes": len(results), "tokens": toks,
            "aggregate_tps": toks / span if span else None,
            "steady_aggregate_pieces_per_s": steady,
            "ttft_max_s": max((r["ttft_s"] or 0) for r in results),
            "ttft_p50_s": statistics.median([r["ttft_s"] or 0 for r in results]),
            "errors": [r["error"] for r in results if r["error"]]}


AGENT_QUESTIONS = [
    "Using the reference, list the three most important rules it states.",
    "Which of those rules would be hardest to enforce in code? Explain briefly.",
    "Draft a short checklist an engineer could follow based on the reference.",
    "What does the reference say about failure handling? Quote one phrase.",
    "Summarize what changed in your understanding since your last answer.",
    "Name one ambiguity in the reference and propose a clarification.",
    "Write two test ideas that would catch a violation of the reference.",
    "Give a one-paragraph status update for this task so far.",
]


def agent_schedule(sessions: int, requests: int, seed: int) -> list:
    """Seeded Zipf session order (session s has weight 1/(s+1)).

    Every session appears at least once (its cold first turn), the rest are
    drawn by weight, so a few sessions are hot and the tail is revisited
    rarely: the regime where retention order matters.
    """
    import random

    rng = random.Random(seed)
    weights = [1.0 / (s + 1) for s in range(sessions)]
    order = list(range(sessions))
    order += rng.choices(range(sessions), weights=weights, k=max(0, requests - sessions))
    head, tail = order[:sessions], order[sessions:]
    rng.shuffle(head)
    return head + tail


class StatusSampler:
    """Poll ``/v1/status`` for one numeric leaf; integrate it over time."""

    def __init__(self, client, suffix, interval):
        self.client, self.suffix, self.interval = client, suffix, float(interval)
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def read(self):
        try:
            flat = flatten(self.client.get("/v1/status", timeout=10))
        except Exception:
            return None
        for key, value in flat.items():
            if key.endswith(self.suffix):
                return value
        return None

    def _run(self):
        while not self._stop.is_set():
            value = self.read()
            if value is not None:
                self.samples.append((time.perf_counter(), float(value)))
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=30)
        value = self.read()
        if value is not None:
            self.samples.append((time.perf_counter(), float(value)))

    def integral(self):
        """Trapezoid integral (value x seconds) and the sampled span."""
        pts = self.samples
        area = sum((t1 - t0) * (v0 + v1) / 2 for (t0, v0), (t1, v1) in zip(pts, pts[1:]))
        span = (pts[-1][0] - pts[0][0]) if len(pts) > 1 else 0.0
        return area, span


def agent_value(requests, resident_byte_seconds, span_s):
    """B3 metric: APCv2-saved prefill ms per resident GiB-second.

    The per-token cold prefill cost comes from this run's own cold requests
    (cached share under 10%), so both arms are charged at their measured
    rate; saved ms is cached tokens x that rate (linear: an underestimate
    for deep prefixes, the same rule in both arms).
    """
    cold = [r["ttft_s"] / max(1, r["prompt_tokens"] - (r["cached_tokens"] or 0))
            for r in requests
            if r.get("ttft_s") and r.get("prompt_tokens")
            and (r["cached_tokens"] or 0) < 0.1 * r["prompt_tokens"]]
    rate_ms = 1e3 * statistics.median(cold) if cold else None
    cached = sum(r["cached_tokens"] or 0 for r in requests)
    prompt = sum(r["prompt_tokens"] or 0 for r in requests)
    saved_ms = cached * rate_ms if rate_ms is not None else None
    # Measured: the cold estimate of each prompt minus its observed TTFT.
    # Unlike the cached-token estimate this charges disk restores and
    # checkpoint replays to the arm that caused them.
    saved_ttft_ms = (sum(rate_ms * (r["prompt_tokens"] or 0) - 1e3 * (r["ttft_s"] or 0)
                         for r in requests if r.get("ttft_s"))
                     if rate_ms is not None else None)
    gib_s = resident_byte_seconds / (1 << 30)
    return {
        "requests": len(requests),
        "cold_requests": len(cold),
        "cold_prefill_ms_per_token": rate_ms,
        "cached_tokens": cached,
        "prompt_tokens": prompt,
        "cached_fraction": cached / prompt if prompt else None,
        "saved_prefill_ms": saved_ms,
        "resident_gib_s": gib_s,
        "resident_mean_gib": gib_s / span_s if span_s else None,
        "saved_ms_per_resident_gib_s": (saved_ms / gib_s) if saved_ms is not None and gib_s else None,
        "saved_ttft_ms": saved_ttft_ms,
        "saved_ttft_ms_per_resident_gib_s": (
            saved_ttft_ms / gib_s if saved_ttft_ms is not None and gib_s else None),
        "sum_ttft_s": sum(r["ttft_s"] or 0 for r in requests),
        "errors": [r["error"] for r in requests if r.get("error")],
    }


def run_agents(client, arg, rep, max_tokens, off, sample_s):
    parts = [int(x) for x in arg.split(":") if x] if arg else []
    sessions, prefix, total = (parts + [6, 6000, 36][len(parts):])[:3]
    histories = {s: [] for s in range(sessions)}
    systems = {s: (f"You are agent {s} working on a long engineering task. "
                   "Answer briefly and only from the reference.\n\nReference:\n"
                   + document(prefix, off + 1009 * (s + 1)))
               for s in range(sessions)}
    turns = {s: 0 for s in range(sessions)}
    out = []
    with StatusSampler(client, "apcv2.idle_disk.resident_bytes", sample_s) as sampler:
        for s in agent_schedule(sessions, total, 7919 + rep):
            question = AGENT_QUESTIONS[turns[s] % len(AGENT_QUESTIONS)]
            messages = ([{"role": "system", "content": systems[s]}] + histories[s]
                        + [{"role": "user", "content": question}])
            r = client.chat(messages, max_tokens=min(max_tokens, 96),
                            extra={"chat_template_kwargs": {"enable_thinking": False}})
            r["session"], r["turn"] = s, turns[s]
            histories[s] += [{"role": "user", "content": question},
                             {"role": "assistant", "content": r["content"] or "(no answer)"}]
            turns[s] += 1
            out.append(r)
    area, span = sampler.integral()
    return {"requests": out, "agents": agent_value(out, area, span),
            "resident_samples": len(sampler.samples)}


TOOLLOOP_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from the repository",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run the test suite and return the failures",
        "parameters": {"type": "object", "properties": {}}}},
]


def run_toolloop(client, arg, rep, off):
    """One agent conversation of R tool rounds (mlx-serve #759's trace shape).

    Each round's prompt is the previous prompt plus the previous reply plus a
    tool result.  ``reply_reused`` is the fraction of the previous reply the
    APC hit covered: 0 means the reply was re-prefilled.  The second argument
    turns thinking on (1) or off (0); history carries content and tool calls
    only, as an OpenAI-style client sends it.
    """
    parts = [int(float(x)) for x in arg.split(":") if x] if arg else []
    rounds, think = (parts + [6, 0][len(parts):])[:2]
    messages = [
        {"role": "system", "content": "You are a coding agent. Use the tools; keep each "
                                      "message short.\n\nProject notes:\n" + document(3000, off + 611)},
        {"role": "user", "content": "The test suite fails. Find the cause with the tools and "
                                    "propose a one-line fix."}]
    extra = {"tools": TOOLLOOP_TOOLS,
             "chat_template_kwargs": {"enable_thinking": bool(think)}}
    out, rows, prev = [], [], None
    for i in range(rounds):
        r = client.chat(messages, max_tokens=512 if think else 256, temperature=0.0, extra=extra)
        r["round"] = i
        out.append(r)
        if prev is not None and r["prompt_tokens"] and r["cached_tokens"] is not None:
            reply = prev["completion_tokens"] or 0
            rows.append({"round": i, "prompt_tokens": r["prompt_tokens"],
                         "cached_tokens": r["cached_tokens"],
                         "prev_prompt_tokens": prev["prompt_tokens"],
                         "prev_completion_tokens": reply,
                         "cached_beyond_prev_prompt": r["cached_tokens"] - prev["prompt_tokens"],
                         # Clamped: template tokens around the reply (end
                         # markers, tool-call tags) are not completion tokens.
                         "reply_reused": (min(1.0, max(0, r["cached_tokens"] - prev["prompt_tokens"])
                                              / reply) if reply else None),
                         "reprefill_tokens": r["prompt_tokens"] - r["cached_tokens"],
                         "ttft_s": r["ttft_s"]})
        prev = r
        if r["error"]:
            break
        calls = [c for c in r.get("tool_calls") or [] if c["function"]["name"]]
        if calls:
            for k, c in enumerate(calls):
                c["id"] = c["id"] or f"call_{i}_{k}"
            messages.append({"role": "assistant", "content": r["content"] or None,
                             "tool_calls": calls})
            for c in calls:
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": document(300, off + 97 * i + 7)})
        else:
            messages.append({"role": "assistant", "content": r["content"] or "(no answer)"})
            messages.append({"role": "user", "content": "Continue: call a tool for the next step."})
    reused = [x["reply_reused"] for x in rows if x["reply_reused"] is not None]
    return {"requests": out, "toolloop": {
        "rounds": rows,
        "tool_rounds": sum(1 for r in out if r.get("tool_calls")),
        "reply_reused_mean": statistics.mean(reused) if reused else None,
        "reprefill_tokens_total": sum(x["reprefill_tokens"] for x in rows)}}


def _cpu_traffic_worker(stop_at):
    import numpy as np
    a = np.ones(64 << 20, dtype=np.float32)  # 256 MiB: far past the SLC
    b = np.empty_like(a)
    while time.time() < stop_at:
        np.copyto(b, a)
        np.copyto(a, b)


def run_cpuload(client, arg, rep, off):
    """Decode and prefill under W host processes streaming memory (PhaseGate,
    arXiv 2610.04537).  W=0 is the in-run control."""
    import multiprocessing as mp
    workers = int(float(arg or 0))
    stop_at = time.time() + 600
    procs = [mp.get_context("spawn").Process(target=_cpu_traffic_worker, args=(stop_at,), daemon=True)
             for _ in range(workers)]
    for p in procs:
        p.start()
    try:
        if procs:
            time.sleep(3)  # let the workers reach steady traffic
        # A different document per W: a shared one turns every loaded run
        # into an APC hit and hides the prefill comparison.
        r = client.chat(user("Here is a document:\n\n" + document(4000, off + 211 + 37 * workers)
                             + "\n\nSummarize it in about 250 words."),
                        max_tokens=384, temperature=0.0)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=10)
    return {"requests": [r], "cpuload": {"workers": workers, "ttft_s": r["ttft_s"],
                                         "decode_tps": r["decode_tps"]}}


def run_workload(client, name, rep, max_tokens, *, vary_prompts=True, sample_s=1.0):
    kind, _, arg = name.partition(":")
    off = rep * 13 if vary_prompts else 0
    if kind == "agents":
        return run_agents(client, arg, rep if vary_prompts else 0, max_tokens, off, sample_s)
    if kind == "toolloop":
        return run_toolloop(client, arg, rep, off)
    if kind == "cpuload":
        return run_cpuload(client, arg, rep, off)
    if kind == "idleshort":
        # Idle gap, then a fresh short prompt: TTFT is dominated by any
        # wake-up or re-residency stall rather than by prefill (mlx #4633).
        parts = [x for x in arg.split(":") if x]
        gap = float(parts[0]) if parts else 6.0
        n = int(float(parts[1])) if len(parts) > 1 else 5
        out = []
        for i in range(n):
            time.sleep(gap)
            out.append(client.chat(user(f"[{rep}.{i} after {gap}s] "
                                        + SHORT_PROMPTS[(rep + i) % len(SHORT_PROMPTS)]),
                                   max_tokens=1, temperature=0.0))
        ttfts = sorted(r["ttft_s"] for r in out if r["ttft_s"] is not None)
        return {"requests": out, "idle": {"gap_s": gap, "ttft_s": ttfts,
                                          "ttft_median_s": statistics.median(ttfts) if ttfts else None}}
    if kind == "short":
        r = client.chat(user(SHORT_PROMPTS[rep % len(SHORT_PROMPTS)]), max_tokens=max_tokens)
        return {"requests": [r]}
    if kind == "serial":
        # N different short prompts one after another (B1): more B1 samples
        # per server launch than ``short``.
        n = int(arg or 8)
        start = rep if vary_prompts else 0
        return {"requests": [client.chat(user(SHORT_PROMPTS[(start + i) % len(SHORT_PROMPTS)]),
                                         max_tokens=max_tokens) for i in range(n)]}
    if kind == "long":
        n = int(arg or 8000)
        doc = document(n, off + 1)
        r = client.chat(user("Here is a document:\n\n" + doc + "\n\nIn three bullet points, what does it describe?"),
                        max_tokens=min(max_tokens, 128))
        return {"requests": [r]}
    if kind == "warm":
        # Same document as long:N with a different question -> APCv2 hit.
        n = int(arg or 8000)
        doc = document(n, off + 1)
        cold = client.chat(user("Here is a document:\n\n" + doc + "\n\nList two key terms it uses."),
                           max_tokens=48)
        warm = client.chat(user("Here is a document:\n\n" + doc + "\n\nName one design rule it states."),
                           max_tokens=48)
        return {"requests": [cold, warm]}
    if kind == "batch":
        n = int(arg or 4)
        jobs = [dict(messages=user("Here is a note:\n\n" + document(700, off + 31 * i)
                                   + "\n\nWrite a careful 150-word summary of it."),
                     max_tokens=max_tokens) for i in range(n)]
        res = concurrent(client, jobs)
        return {"requests": res, "aggregate": aggregate(res)}
    if kind == "contention":
        n = int(arg or 16000)
        out = {}
        bg = {}

        def lane_a():
            bg["a"] = client.chat(user(SHORT_PROMPTS[(rep + 3) % len(SHORT_PROMPTS)]
                                       + " Be thorough; write at least 400 words."),
                                  max_tokens=max(600, max_tokens))
        t = threading.Thread(target=lane_a)
        t.start()
        time.sleep(1.5)
        b = client.chat(user("Here is a document:\n\n" + document(n, off + 77)
                             + "\n\nWhat is its title?"), max_tokens=16)
        t.join()
        a = bg["a"]
        # neighbour gaps while B was prefilling
        win = [(x, y) for x, y in zip(a["arrivals"], a["arrivals"][1:])
               if b["t0"] <= y and x <= (b["t_first"] or b["t_end"])]
        gaps = [y - x for x, y in win]
        out["requests"] = [a, b]
        out["neighbour"] = {"window_gaps": len(gaps),
                            "gap_max_ms": 1e3 * max(gaps) if gaps else None,
                            "gap_p95_ms": 1e3 * sorted(gaps)[int(0.95 * (len(gaps) - 1))] if gaps else None,
                            "long_ttft_s": b["ttft_s"],
                            # Where the worst gaps sit: absolute client clock
                            # (perf_counter, host-wide mach clock on macOS, so
                            # a server loop trace can be aligned with it) and
                            # offsets from the long request's start / first token.
                            "top_gaps": [
                                {"start": x, "ms": 1e3 * (y - x),
                                 "after_b_t0_s": x - b["t0"],
                                 "before_b_first_s": (b["t_first"] or b["t_end"]) - x}
                                for x, y in sorted(win, key=lambda w: w[0] - w[1])[:8]
                            ]}
        return out
    if kind == "idle":
        # Idle gap, then a fresh ~4K-token prompt (no prefix hit): TTFT after
        # the GPU has had ``arg`` seconds to clock down.
        gap = float(arg or 6)
        out = []
        for i in range(3):
            time.sleep(gap)
            out.append(client.chat(user(f"[idle {gap} {rep} {i}] " + document(4000, off + 400 + 9 * i)
                                        + "\n\nTitle?"), max_tokens=1))
        return {"requests": out}
    if kind == "copy":
        excerpt = document(450, off + 5)
        r = client.chat(user("Reproduce the following text exactly, with no commentary:\n\n" + excerpt),
                        max_tokens=max(600, max_tokens))
        return {"requests": [r]}
    if kind == "json":
        schema = {"type": "object", "properties": {
            "title": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}},
            "priority": {"type": "integer"}}, "required": ["title", "tags", "priority"]}
        r = client.chat(user("Create a task record for: fix the login timeout bug on mobile."),
                        max_tokens=200,
                        extra={"response_format": {"type": "json_schema",
                                                   "json_schema": {"name": "task", "schema": schema}}})
        return {"requests": [r]}
    if kind == "tool":
        tools = [{"type": "function", "function": {
            "name": "get_weather", "description": "Get current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}}]
        r = client.chat(user("What's the weather in Montreal right now?"), max_tokens=200,
                        extra={"tools": tools})
        return {"requests": [r]}
    raise ValueError(f"unknown workload {name}")


def diff(after, before):
    out = {}
    for k, v in after.items():
        b = before.get(k)
        if isinstance(v, (int, float)) and isinstance(b, (int, float)) and v != b:
            out[k] = v - b
        elif b is None and isinstance(v, (int, float)) and v:
            out[k] = v
    return out


NOISY = re.compile(r"(_bytes|bytes\.|latency_ms|\.p50|\.p95|\.p99|seconds|_sum$|uptime|\.since|\.at$|"
                   r"timestamp|_ms$|bucket|headroom|footprint|rates\.|tenant_token_rate|recent_)")


def engagement(d):
    """Counter-like deltas, dropping timing/memory noise."""
    return {k: v for k, v in d.items() if not NOISY.search(k)}


# --------------------------------------------------------------------------- main


def speculation_counts(receipt):
    """(verify rounds, accepted draft tokens, emitted tokens) of one request.

    External draft: ``speculation.external_rounds``/``accepted`` (each round
    also emits its correction/bonus token).  Native self-MTP:
    ``mtp.stats.cycles``, draft + retrieval acceptances and
    ``total_emitted``.  ``None`` when the route did not speculate.
    """
    receipt = receipt or {}
    spec = receipt.get("speculation") or {}
    if spec.get("external_rounds"):
        rounds, accepted = int(spec["external_rounds"]), int(spec.get("accepted") or 0)
        return rounds, accepted, accepted + rounds
    stats = (receipt.get("mtp") or {}).get("stats") or {}
    if stats.get("cycles"):
        accepted = int(stats.get("draft_accepted") or 0) + int(stats.get("retrieval_accepted") or 0)
        return int(stats["cycles"]), accepted, int(stats.get("total_emitted") or 0)
    return None


def speculation_summary(requests):
    rounds = accepted = emitted = 0
    for r in requests:
        counts = speculation_counts(r.get("receipt"))
        if counts:
            rounds += counts[0]
            accepted += counts[1]
            emitted += counts[2]
    if not rounds:
        return None
    return {"verify_rounds": rounds, "accepted": accepted, "emitted": emitted,
            "accepted_per_verify": accepted / rounds, "tokens_per_verify": emitted / rounds}


def histogram_percentiles(values: dict, prefix: str, quantiles=(0.5, 0.99)):
    """Reduce ``<prefix>.<producer>.buckets.le_<us>`` counts to percentiles.

    ``values`` is a flat mapping (a per-workload delta); each quantile is the
    upper edge (ms) of the bucket where the cumulative count reaches it.
    """
    out = {}
    pattern = re.compile(re.escape(prefix) + r"\.([^.]+)\.buckets\.le_(\d+)$")
    per = {}
    for key, value in values.items():
        m = pattern.search(key)
        if m and value:
            per.setdefault(m.group(1), []).append((int(m.group(2)), value))
    for producer, buckets in per.items():
        buckets.sort()
        total = sum(c for _, c in buckets)
        entry = {"count": total}
        for q in quantiles:
            cum = 0
            for edge, count in buckets:
                cum += count
                if cum >= q * total:
                    entry[f"p{round(q * 100):d}_ms"] = edge / 1e3
                    break
        out[producer] = entry
    return out


def capture(after, d, prefixes):
    """Raw after-values and deltas of the ``status_capture`` prefixes."""
    if not prefixes:
        return None
    def keep(key):
        return any(key.startswith(p) for p in prefixes)

    out = {"after": {k: v for k, v in after.items() if keep(k)},
           "delta": {k: v for k, v in d.items() if keep(k)}}
    timing = histogram_percentiles(
        out["delta"], "status.execution.qsa_stage1.candidates.select_timing.producers")
    if timing:
        out["select_timing"] = timing
    return out


def strip(r):
    r = dict(r)
    r.pop("arrivals", None)
    r.pop("tool_calls", None)
    r.pop("content", None)
    rec = r.get("receipt") or {}
    keep = {k: rec.get(k) for k in ("route", "profile", "qualification", "cached_tokens",
                                    "mtp", "speculation", "prefill_chunk", "ingress_cohort",
                                    "ordinary_compute_width", "ttft_seconds", "elapsed_seconds",
                                    "parallel_prefill", "cache_checkpoint_role") if k in rec}
    r["receipt"] = keep
    return r


# --------------------------------------------------------------------------- dry run

WORKLOAD_KINDS = {"short", "serial", "long", "warm", "batch", "contention", "idle", "copy",
                  "json", "tool", "agents", "toolloop", "cpuload", "idleshort"}
SAMPLING_KEYS = {"temperature", "top_p", "top_k", "min_p", "seed"}


class _StopBeforeHash(Exception):
    """Raised in place of the first payload hash of a shallow dry run."""


def _check_workload(name, max_context):
    kind, _, arg = name.partition(":")
    if kind not in WORKLOAD_KINDS:
        return f"unknown workload kind {kind!r}"
    parts = [x for x in arg.split(":") if x]
    try:
        values = [float(x) for x in parts]
    except ValueError:
        return f"workload {name!r}: non-numeric argument"
    if kind == "agents" and len(values) > 3:
        return "agents takes at most N:P:R"
    if kind in ("toolloop", "idleshort") and len(values) > 2:
        return f"{kind} takes at most two arguments"
    if kind not in ("agents", "toolloop", "idleshort") and len(values) > 1:
        return f"workload {name!r} takes one argument"
    # Qwen3.8's tokenizer gives 0.92-1.08 tokens per nominal corpus token
    # (long:128000 is 137,675 tokens): leave 10% slack.
    if kind in ("long", "warm", "contention") and max_context and values \
            and values[0] * 1.1 > max_context:
        return (f"workload {name!r} may exceed --max-context {max_context} "
                "(up to 1.1 tokens per nominal token)")
    return None


def _validate_adapter_policy(resolution, route, policy, model, deep):
    """Static adapter validation where one exists; returns a note."""
    module = resolution.adapter_type.__module__.rsplit(".", 1)[-1]
    if route == "external_draft":
        draft = policy.get("draft_model")
        if not draft or not Path(draft).expanduser().exists():
            raise ValueError(f"draft_model {draft!r} does not exist")
        if module in ("qwen38_27b", "qwen36_35b"):
            from mlx2.adapters import qwen38_27b

            if module == "qwen36_35b":
                from mlx2.adapters.qwen36_35b import EXTERNAL_POLICY_KEYS

                unknown = set(policy) - EXTERNAL_POLICY_KEYS
                if unknown:
                    raise ValueError(f"Qwen3.6 external draft policy has unknown keys: {sorted(unknown)}")
            allow = module == "qwen36_35b"
            if module == "qwen38_27b":
                # The adapter consumes its target-wide keys before it splits
                # the external policy off (adapters/qwen38_27b.py).
                policy = {
                    key: value for key, value in policy.items()
                    if key not in qwen38_27b.TARGET_POLICY_KEYS
                }
            if deep:
                qwen38_27b.inspect_external_policy(policy, model, allow_continuation_strategy=allow)
                return "qwen38 inspect_external_policy (deep: revision pins hashed)"

            def stop(_path):
                raise _StopBeforeHash

            original = qwen38_27b._legacy_content_revision
            qwen38_27b._legacy_content_revision = stop
            try:
                qwen38_27b.inspect_external_policy(policy, model, allow_continuation_strategy=allow)
            except _StopBeforeHash:
                pass
            finally:
                qwen38_27b._legacy_content_revision = original
            return "qwen38 inspect_external_policy up to the revision hashes (--deep checks them)"
        if module == "muse_glimmer":
            from mlx2.adapters.muse_glimmer import normalize_external_policy

            normalize_external_policy(policy)
            if deep:
                from mlx2.adapters.dflash2 import inspect_drafter

                inspect_drafter(draft, model)
                return "muse normalize_external_policy + DFlash2 drafter/target inspection"
            return "muse normalize_external_policy"
        parse = getattr(resolution.adapter_type, "_parse_external_policy", None)
        if parse is not None:
            from types import SimpleNamespace

            parse(SimpleNamespace(), policy, family=resolution.adapter_type.__name__)
            if deep and module == "north_mini_code":
                from mlx2.adapters.cohere_eagle import inspect_drafter

                config = json.loads((Path(model) / "config.json").read_text())
                inspect_drafter(draft, target_config=config,
                                block_size=resolution.adapter_type.EXTERNAL_MAX_BLOCK)
                return "ExternalDraftAdapterMixin key check + EAGLE drafter/target inspection"
            return "ExternalDraftAdapterMixin key check"
        return "external policy not statically validated"
    if not policy:
        return "no adapter policy"
    if module == "flash_next":
        from mlx2.adapters.flash_next_policy import FlashNextPolicy

        FlashNextPolicy.from_mapping(policy)
        return "FlashNextPolicy.from_mapping"
    return f"adapter policy keys {sorted(policy)} checked only at load"


# Prefixes every model adapter's configure_environment() clears from the
# inherited environment (adapters/*.py clear_inherited_profile): an arm that
# sets one runs the default profile, so its A/B measures nothing.
ADAPTER_CLEARED_PREFIXES = ("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_", "MLX_AGNES_")


def _check_env(env, source_text):
    problems = []
    for name, value in env.items():
        if name.startswith(ADAPTER_CLEARED_PREFIXES):
            problems.append(
                f"env {name} is cleared by the model adapter at load "
                "(clear_inherited_profile); select it through the adapter's "
                "execution policy instead"
            )
        if name.startswith(("MLX_", "MLX2_")) and f'"{name}"' not in source_text:
            problems.append(f"env {name} is not read anywhere under src/ (typo?)")
        if name == "MLX_QWEN4_QSA_STAGE1_DIRECT_SELECTOR":
            from mlx2.runtime.models.qwen4_qsa_stage1 import _direct_selector_mode

            try:
                _direct_selector_mode(value)
            except ValueError as exc:
                problems.append(str(exc))
    return problems


def validate_arm(spec, name, arm, *, deep=False, source_text=""):
    """Validate one arm as ``mlx2.server`` would before loading weights."""
    import contextlib
    import io

    from mlx2.adapters.registry import inspect_model
    from mlx2.runtime.adaptive_policy import (
        AdaptiveMTPDepthPolicy,
        MTPOrdinaryHandoffPolicy,
    )
    from mlx2.runtime.apc_retention import apc_retention_policy
    from mlx2.server import (
        build_parser,
        resolve_execution_policy_defaults,
        resolve_route_selection,
    )
    from mlx2.serving import SERVER_OWNED_EXECUTION_POLICY_KEYS

    out = {"errors": [], "warnings": []}
    with tempfile.TemporaryDirectory(prefix=f"mlx2prof-dry-{name}-") as tmp:
        argv = server_argv(spec, arm, 0, Path(tmp))
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                ns = build_parser().parse_args(argv)
        except SystemExit:
            out["errors"].append("server argparse: " + err.getvalue().strip().splitlines()[-1])
            return out
    policy = arm_policy(spec, arm) or None
    try:
        resolution = inspect_model(spec["model"])
        route = resolve_route_selection(ns, policy, resolution)
        skipped = {}
        resolved = resolve_execution_policy_defaults(
            policy, route, resolution,
            approximate_kv=getattr(ns, "approximate_kv", None) is not None,
            max_lanes=ns.max_lanes, skipped=skipped)
    except (OSError, ValueError) as exc:
        out["errors"].append(f"route/policy resolution: {exc}")
        return out
    resolved = dict(resolved or {})
    out["route"] = route.route
    out["adapter"] = resolution.adapter_type.__name__
    out["resolved_policy"] = resolved
    if skipped:
        out["skipped_route_defaults"] = skipped
    native = route.route == "native_mtp"
    checks = [
        ("apc_retention_policy", lambda v: apc_retention_policy(v)),
        ("mtp_ordinary_handoff", lambda v: MTPOrdinaryHandoffPolicy.from_value(v)),
        ("adaptive_mtp_depth", lambda v: AdaptiveMTPDepthPolicy.from_value(v)),
    ]
    for key, check in checks:
        if key in resolved:
            try:
                parsed = check(resolved[key])
            except ValueError as exc:
                out["errors"].append(f"{key}: {exc}")
                continue
            if key != "apc_retention_policy" and getattr(parsed, "enabled", False) and not native:
                out["errors"].append(f"{key} requires the native self-MTP route")
    adapter_policy = {k: v for k, v in resolved.items()
                      if k not in SERVER_OWNED_EXECUTION_POLICY_KEYS}
    try:
        note = _validate_adapter_policy(
            resolution, route.route, adapter_policy, spec["model"], deep)
        if "only at load" in note or "not statically" in note:
            out["warnings"].append(note)
        else:
            out["adapter_check"] = note
    except (OSError, ValueError, RuntimeError) as exc:
        out["errors"].append(f"adapter policy: {exc}")
    out["errors"] += _check_env(arm_env(spec, arm), source_text)
    return out


def dry_run(spec, *, deep=False):
    """Validate a spec's arms and workloads without servers or weights."""
    report = {"ok": True, "spec_errors": [], "arms": {}}
    model = Path(spec.get("model", "")).expanduser()
    if not (model / "config.json").exists():
        report["spec_errors"].append(f"model {model} has no config.json")
    unknown = set(spec.get("sampling", {})) - SAMPLING_KEYS
    if unknown:
        report["spec_errors"].append(f"unknown sampling keys {sorted(unknown)}")
    if not spec.get("arms"):
        report["spec_errors"].append("spec has no arms")
    args = list(spec.get("common_args", []))
    max_context = None
    if "--max-context" in args:
        max_context = int(args[args.index("--max-context") + 1])
    for wl in spec.get("workloads", []):
        problem = _check_workload(wl, max_context)
        if problem:
            report["spec_errors"].append(problem)
    source_text = "\n".join(p.read_text() for p in (ROOT / "src" / "mlx2").rglob("*.py"))
    if not report["spec_errors"]:
        for name, arm in spec["arms"].items():
            report["arms"][name] = validate_arm(spec, name, arm, deep=deep,
                                                source_text=source_text)
    report["ok"] = not report["spec_errors"] and all(
        not a["errors"] for a in report["arms"].values())
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--port", type=int, default=8941)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate every arm without servers or weights, then exit")
    ap.add_argument("--deep", action="store_true",
                    help="with --dry-run: also run payload-hashing revision checks")
    ap.add_argument("--rep-start", type=int, default=0,
                    help="first rep to run (arm rotation follows the absolute rep)")
    ap.add_argument("--reps", type=int, default=None,
                    help="number of reps to run from --rep-start (default: the spec's)")
    args = ap.parse_args(argv)
    spec = json.loads(args.spec.read_text())
    if args.dry_run:
        report = dry_run(spec, deep=args.deep)
        print(json.dumps(report, indent=1, default=str))
        return 0 if report["ok"] else 1
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "spec.json").write_text(json.dumps(spec, indent=2))
    arms = list(spec["arms"])
    reps = int(spec.get("reps", 2))
    rep_end = reps if args.reps is None else args.rep_start + args.reps
    max_tokens = int(spec.get("max_tokens", 256))
    vary = bool(spec.get("vary_prompts_by_rep", True))
    sample_s = float(spec.get("status_sample_s", 1.0))
    prefixes = list(spec.get("status_capture", []))
    results = []
    previous = args.out / "results.json"
    if args.rep_start and previous.exists():
        # A later slice of the same run: keep the earlier reps' rows.
        results = [r for r in json.loads(previous.read_text())
                   if not args.rep_start <= r["rep"] < rep_end]
    swap0 = swapouts()
    try:
        from mlx2.apple_telemetry import EnergySampler, TemperatureSampler

        energy, temps = EnergySampler(), TemperatureSampler()
    except Exception:  # pragma: no cover - telemetry is optional
        energy = temps = None
    for rep in range(args.rep_start, rep_end):
        shift = (rep // 2) % len(arms)
        order = arms[shift:] + arms[:shift]
        if rep % 2:
            order = list(reversed(order))
        for arm_name in order:
            arm = spec["arms"][arm_name]
            srv = Server(spec, arm_name, arm, args.port, args.out,
                         trace_path=args.out / f"loop-trace-{arm_name}-r{rep}.jsonl")
            row = {"arm": arm_name, "rep": rep, "workloads": {}, "argv": srv.args[3:]}
            try:
                st = srv.wait_ready()
                row["load_s"] = srv.load_s
                row["route_receipt"] = st.get("route_receipt")
                row["selected_capabilities"] = st.get("selected_capabilities")
                row["settings"] = st.get("settings")
                srv.client.chat(user("Say hello in five words."), max_tokens=16)  # warm-up
                for wl in spec["workloads"]:
                    before = srv.client.snapshot()
                    sw, th = swapouts(), thermal()
                    die0 = temps.die_summary() if temps else None
                    if energy:
                        energy.read()
                    t0 = time.time()
                    res = run_workload(srv.client, wl, rep, max_tokens,
                                       vary_prompts=vary, sample_s=sample_s)
                    if energy:
                        e = energy.read()
                        toks = sum(r["completion_tokens"] or 0 for r in res["requests"])
                        joules = ((e.gpu_watts or 0) + (e.dram_watts or 0)) * e.seconds
                        res["energy"] = {**e.as_dict(), "gpu_dram_joules": joules,
                                         "joules_per_output_token": joules / toks if toks else None,
                                         "die_before": die0, "die_after": temps.die_summary()}
                    after = srv.client.snapshot()
                    d = diff(after, before)
                    res["requests"] = [strip(r) for r in res["requests"]]
                    res["speculation"] = speculation_summary(res["requests"])
                    captured = capture(after, d, prefixes)
                    if captured is not None:
                        res["captured"] = captured
                    res["engaged"] = engagement(d)
                    res["peak_metal_bytes"] = after.get("status.metal_peak_bytes")
                    res["swapouts_delta"] = (swapouts() or 0) - (sw or 0)
                    res["thermal_before"] = th
                    res["elapsed_s"] = time.time() - t0
                    row["workloads"][wl] = res
                    print(f"[{arm_name} r{rep}] {wl}: " + json.dumps(summary_line(res)), flush=True)
            except Exception as exc:
                row["error"] = repr(exc)
                print(f"[{arm_name} r{rep}] ERROR {exc!r}", flush=True)
            finally:
                srv.stop()
            results.append(row)
            (args.out / "results.json").write_text(json.dumps(results, indent=1, default=str))
            time.sleep(float(spec.get("cooldown_s", 20)))
    summary = summarize(results, arms, spec["workloads"])
    summary["swapouts_total_delta"] = (swapouts() or 0) - (swap0 or 0)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str)[:6000])


def summary_line(res):
    rs = res["requests"]
    out = {"ttft": [round(r["ttft_s"], 3) if r["ttft_s"] else None for r in rs],
           "tps": [round(r["decode_tps"], 1) if r["decode_tps"] else None for r in rs],
           "err": [r["error"] for r in rs if r["error"]]}
    if "aggregate" in res:
        out["agg_tps"] = res["aggregate"]["aggregate_tps"]
    if "neighbour" in res:
        out["neighbour"] = res["neighbour"]
    if res.get("speculation"):
        out["apv"] = round(res["speculation"]["accepted_per_verify"], 3)
    if "agents" in res:
        out["agents"] = {k: res["agents"][k] for k in (
            "saved_prefill_ms", "saved_ttft_ms", "resident_gib_s",
            "saved_ms_per_resident_gib_s", "cached_fraction", "sum_ttft_s")}
        out["ttft"] = out["tps"] = None  # per-request lists are too long here
    if (res.get("captured") or {}).get("select_timing"):
        out["select_timing"] = res["captured"]["select_timing"]
    if "energy" in res:
        en = res["energy"]
        out["gpu_w"] = round(en["gpu_w"] or 0, 1)
        out["J_per_tok"] = round(en["joules_per_output_token"] or 0, 3)
        out["pstate"] = round(en["gpu_mean_pstate"] or 0, 2)
        out["die_max"] = round((en["die_after"] or {}).get("die_max_c") or 0, 1)
    return out


def metric_of(res, key):
    rs = res["requests"]
    if key == "ttft":
        return rs[-1]["ttft_s"]
    if key == "tps":
        return rs[-1]["decode_tps"]
    if key == "agg_tps":
        return (res.get("aggregate") or {}).get("aggregate_tps")
    if key == "gap_max":
        return (res.get("neighbour") or {}).get("gap_max_ms")
    if key in ("apv", "tpv"):
        spec = res.get("speculation") or {}
        return spec.get("accepted_per_verify" if key == "apv" else "tokens_per_verify")
    if key == "agents_value":
        return (res.get("agents") or {}).get("saved_ms_per_resident_gib_s")
    if key == "agents_saved_ms":
        return (res.get("agents") or {}).get("saved_prefill_ms")
    if key == "agents_value_ttft":
        return (res.get("agents") or {}).get("saved_ttft_ms_per_resident_gib_s")
    if key == "agents_sum_ttft":
        return (res.get("agents") or {}).get("sum_ttft_s")
    if key == "wall":
        return res.get("elapsed_s")
    if key == "tps_all":
        # Median decode tok/s over every request of the workload.
        vals = [r["decode_tps"] for r in rs if r.get("decode_tps")]
        return statistics.median(vals) if vals else None
    if key.startswith("select_p"):
        # select_p99 -> the slowest producer's P99 selection time (ms).
        timing = (res.get("captured") or {}).get("select_timing") or {}
        vals = [t.get(key[len("select_"):] + "_ms") for t in timing.values()]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None
    return None


METRICS = ("ttft", "tps", "tps_all", "agg_tps", "gap_max", "apv", "tpv",
           "agents_value", "agents_value_ttft", "agents_saved_ms", "agents_sum_ttft",
           "select_p50", "select_p99", "wall")


def summarize(results, arms, workloads):
    base = arms[0]
    out = {"baseline": base, "workloads": {}}
    for wl in workloads:
        table = {}
        for arm in arms:
            rows = [r for r in results if r["arm"] == arm and wl in r.get("workloads", {})]
            entry = {}
            for key in METRICS:
                vals = [metric_of(r["workloads"][wl], key) for r in rows]
                vals = [v for v in vals if v is not None]
                if vals:
                    entry[key + "_median"] = statistics.median(vals)
                    entry[key + "_all"] = [round(v, 4) for v in vals]
            # identity vs baseline arm, same rep
            same = total = 0
            for r in rows:
                b = next((x for x in results if x["arm"] == base and x["rep"] == r["rep"]
                          and wl in x.get("workloads", {})), None)
                if b is None:
                    continue
                for q, p in zip(r["workloads"][wl]["requests"], b["workloads"][wl]["requests"]):
                    total += 1
                    same += q["sha"] == p["sha"]
            entry["identical_vs_base"] = f"{same}/{total}"
            # Same arm, rep 0 vs later reps: the repeatability of a seeded,
            # fixed-prompt spec (only meaningful with vary_prompts_by_rep off).
            first = min(rows, key=lambda r: r["rep"]) if rows else None
            same = total = 0
            for r in rows:
                if r is first:
                    continue
                for q, p in zip(r["workloads"][wl]["requests"], first["workloads"][wl]["requests"]):
                    total += 1
                    same += q["sha"] == p["sha"]
            entry["identical_across_reps"] = f"{same}/{total}"
            keys = {}
            for r in rows:
                for k, v in r["workloads"][wl]["engaged"].items():
                    keys.setdefault(k, []).append(v)
            entry["engaged"] = {k: statistics.median(v) for k, v in sorted(keys.items())}
            table[arm] = entry
        for arm in arms[1:]:
            for key in METRICS:
                a, b = table[arm].get(key + "_median"), table[base].get(key + "_median")
                if a is not None and b:
                    table[arm][key + "_vs_base_pct"] = round(100 * (a / b - 1), 2)
        out["workloads"][wl] = table
    return out


if __name__ == "__main__":
    sys.exit(main())
