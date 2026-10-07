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
    def __init__(self, port: int):
        self.base = f"http://127.0.0.1:{port}"
        self.model = None

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

    def chat(self, messages, *, max_tokens, temperature=0.0, extra=None, timeout=1800):
        """Stream one chat request; return timing + text + receipt."""
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "stream": True,
                "stream_options": {"include_usage": True}}
        if extra:
            body.update(extra)
        req = urllib.request.Request(self.base + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        t0 = time.perf_counter()
        arrivals, pieces, usage, receipt, error = [], [], None, None, None
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
            "finish": finish,
            "error": error,
            "receipt": receipt,
        }


# --------------------------------------------------------------------------- server


class Server:
    def __init__(self, spec, arm_name, arm, port, logdir):
        self.port = port
        self.tmp = Path(tempfile.mkdtemp(prefix=f"mlx2prof-{arm_name}-"))
        args = [PYTHON, "-u", "-m", "mlx2.server", "--model", spec["model"],
                "--host", "127.0.0.1", "--port", str(port),
                "--cache-dir", str(self.tmp / "apc")]
        args += list(spec.get("common_args", []))
        policy = dict(spec.get("common_policy", {}))
        policy.update(arm.get("policy", {}))
        if policy:
            pf = self.tmp / "policy.json"
            pf.write_text(json.dumps(policy))
            args += ["--execution-policy", str(pf)]
        args += list(arm.get("args", []))
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        env.update({k: str(v) for k, v in spec.get("common_env", {}).items()})
        env.update({k: str(v) for k, v in arm.get("env", {}).items()})
        self.log = open(logdir / f"server-{arm_name}-{port}.log", "w")
        self.args = args
        self.t_launch = time.perf_counter()
        self.proc = subprocess.Popen(args, env=env, cwd=str(ROOT), stdout=self.log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        self.client = Client(port)

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


def run_workload(client, name, rep, max_tokens):
    kind, _, arg = name.partition(":")
    off = rep * 13
    if kind == "short":
        r = client.chat(user(SHORT_PROMPTS[rep % len(SHORT_PROMPTS)]), max_tokens=max_tokens)
        return {"requests": [r]}
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
                            "long_ttft_s": b["ttft_s"]}
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


def strip(r):
    r = dict(r)
    r.pop("arrivals", None)
    rec = r.get("receipt") or {}
    keep = {k: rec.get(k) for k in ("route", "profile", "qualification", "cached_tokens",
                                    "mtp", "speculation", "prefill_chunk", "ingress_cohort",
                                    "ordinary_compute_width", "ttft_seconds", "elapsed_seconds",
                                    "parallel_prefill", "cache_checkpoint_role") if k in rec}
    r["receipt"] = keep
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--port", type=int, default=8941)
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "spec.json").write_text(json.dumps(spec, indent=2))
    arms = list(spec["arms"])
    reps = int(spec.get("reps", 2))
    max_tokens = int(spec.get("max_tokens", 256))
    results = []
    swap0 = swapouts()
    try:
        from mlx2.apple_telemetry import EnergySampler, TemperatureSampler

        energy, temps = EnergySampler(), TemperatureSampler()
    except Exception:  # pragma: no cover - telemetry is optional
        energy = temps = None
    for rep in range(reps):
        shift = (rep // 2) % len(arms)
        order = arms[shift:] + arms[:shift]
        if rep % 2:
            order = list(reversed(order))
        for arm_name in order:
            arm = spec["arms"][arm_name]
            srv = Server(spec, arm_name, arm, args.port, args.out)
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
                    res = run_workload(srv.client, wl, rep, max_tokens)
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
    return None


def summarize(results, arms, workloads):
    base = arms[0]
    out = {"baseline": base, "workloads": {}}
    for wl in workloads:
        table = {}
        for arm in arms:
            rows = [r for r in results if r["arm"] == arm and wl in r.get("workloads", {})]
            entry = {}
            for key in ("ttft", "tps", "agg_tps", "gap_max"):
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
            keys = {}
            for r in rows:
                for k, v in r["workloads"][wl]["engaged"].items():
                    keys.setdefault(k, []).append(v)
            entry["engaged"] = {k: statistics.median(v) for k, v in sorted(keys.items())}
            table[arm] = entry
        for arm in arms[1:]:
            for key in ("ttft", "tps", "agg_tps", "gap_max"):
                a, b = table[arm].get(key + "_median"), table[base].get(key + "_median")
                if a is not None and b:
                    table[arm][key + "_vs_base_pct"] = round(100 * (a / b - 1), 2)
        out["workloads"][wl] = table
    return out


if __name__ == "__main__":
    main()
