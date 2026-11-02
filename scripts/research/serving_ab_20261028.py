#!/usr/bin/env python3
"""Loopback-server A/B driver for the 2026-10-28 GPU campaign (one artifact).

One bounded ownership unit, run under the GPU locks + CPG lease (the caller
wraps it in ``run_with_gpu_locks.py``): ordinary-reference smoke first
(``scripts/smoke_local_model.py``), then one loopback server per *session*,
sessions in the plan's counterbalanced arm order (e.g. A B B A), every
server's whole process group stopped in ``finally``, a thermal gate
(``thermal_state == 0``, polled every 15 s, every sample recorded) before
every arm and every cell instead of fixed cooldowns, ``POST
/v1/admin/apc/clear`` before every cell, ``vm_stat`` Swapouts checked after
every session (abort on a rise).  Refuses a nonempty output directory.

Plan (JSON):

    {"name": str, "model": path, "scenario": "ttft"|"decode"|"slice"|
     "interior"|"concurrent", "arms": {"A": {"server_args": [...],
     "execution_policy": {...}|null}, ...}, "order": ["A", "B", ...],
     "cooldown_s": 0, "params": {...}}

Scenarios (all timing is client-side wall clock plus the server's own
``mlx2.ttft_seconds`` receipt; prompts are real repository prose,
``corpus_text()``, at fixed token windows that are identical across arms):

  ttft        lengths x reps, non-streaming, ``max_tokens`` 1: TTFT and
              prompt tok/s per cell.
  decode      fixed real chat prompts, greedy, ``return_token_trace``: decode
              tok/s and token ids for cross-arm agreement.
  slice       APCv2 slice-schedule validation: idle 2K (planned, resend
              hits), contended 2K beside a long decode lane (off plan,
              refusal counted, resend does not hit), staggered multi-turn
              prompt for interior checkpoints at or before the deviation.
  interior    shared-prefix / divergent-suffix TTFT (interior checkpoints).
  concurrent  B-way concurrent 2K prompts: TTFT, aggregate tok/s, admission.

Exploratory unless the plan has >= 3 sessions per arm; never qualification.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from run_single_model_smoke_ladder import (  # noqa: E402
    atomic_json, stop_server, thermal_sample, wait_health)

SCHEMA = "mlx2.serving-ab-20261028.v1"
COUNTER_KEYS = ("apc", "slice", "interior", "prefill", "admission", "memory", "fairness",
                "decode_first", "handoff", "spec", "lane")

DECODE_PROMPTS = [
    "Explain how a database transaction works, in numbered sections.",
    "Write a Python function that parses an ISO 8601 duration string such as "
    "P3DT4H12M into total seconds, with a docstring and three doctests.",
    "Write a short story (about 300 words) about a lighthouse keeper who finds a "
    "message in a bottle written in her own handwriting.",
    "Compare TCP and UDP for a multiplayer game server. Give a table and a recommendation.",
    "What were the main causes of the French Revolution? Answer in five bullet points "
    "with one sentence each.",
    "Write a SQL query that finds, for every customer, their three most recent orders, "
    "and explain how the window function works.",
]
LONG_LANE_PROMPT = ("Write a detailed, multi-section technical essay on how a modern "
                    "relational database executes a query: parsing, planning, cost "
                    "estimation, join algorithms, buffer management, and logging. "
                    "Be thorough.")


def http(url, body=None, *, token=None, timeout=1800):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if data is not None else {}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw)
        except ValueError:
            return error.code, {"raw": raw.decode(errors="replace")}


def swapouts():
    out = subprocess.check_output(["vm_stat"], text=True)
    line = next(item for item in out.splitlines() if item.startswith("Swapouts"))
    return int(line.split(":")[1].strip().rstrip("."))


def pmset_therm():
    return subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout


def numeric_counters(status):
    out = {}
    for section in ("counts", "apcv2", "admission", "scheduler"):
        value = status.get(section)
        if not isinstance(value, dict):
            continue
        for key, item in value.items():
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                out[f"{section}.{key}"] = item
    for key in ("metal_peak_bytes", "metal_active_bytes", "process_physical_footprint_bytes"):
        if isinstance(status.get(key), (int, float)):
            out[key] = status[key]
    return out


def delta(before, after):
    return {k: after[k] - before.get(k, 0) for k in after
            if after[k] != before.get(k, 0) and any(s in k for s in COUNTER_KEYS)}


def spread(values):
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return None
    return {"n": len(values), "median": statistics.median(values), "min": min(values),
            "max": max(values), "mean": statistics.fmean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}


class Corpus:
    """Real repository prose at fixed, arm-independent token windows."""

    def __init__(self, model_path):
        from transformers import AutoTokenizer
        from lane_model_sweep import corpus_text

        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.ids = self.tok.encode(corpus_text(), add_special_tokens=False)
        self.cursor = 0

    def text(self, offset, size):
        offset %= max(1, len(self.ids) - size - 1)
        if offset + size > len(self.ids):
            raise RuntimeError("corpus exhausted")
        return self.tok.decode(self.ids[offset:offset + size])

    def prompt(self, slot, size, question="Reply with one word: ready."):
        # slot -> deterministic offset; the same slot is the same text in every arm
        offset = (slot * 104729) % max(1, len(self.ids) - size - 1)
        return (f"Read these project notes, then answer the question at the end.\n\n"
                f"{self.text(offset, size)}\n\nQuestion: {question}")


class Unit:
    def __init__(self, plan, out, port, token, base_env):
        self.plan, self.out, self.port, self.token = plan, out, port, token
        self.base = f"http://127.0.0.1:{port}"
        self.env = base_env
        self.model_name = Path(plan["model"]).name

    # -- requests ---------------------------------------------------------------
    def clear(self, reason):
        code, body = http(self.base + "/v1/admin/apc/clear", {"reason": reason}, token=self.token)
        if code != 200:
            raise RuntimeError(f"APC clear failed: {code} {body}")
        return body

    def status(self):
        return http(self.base + "/v1/status")[1]

    def chat(self, content, *, max_tokens=1, system=None, messages=None, extra=None):
        msgs = messages or ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": content}]
        body = {"model": self.model_name, "messages": msgs, "max_tokens": max_tokens,
                "temperature": 0, **(extra or {})}
        started = time.perf_counter()
        code, response = http(self.base + "/v1/chat/completions", body)
        elapsed = time.perf_counter() - started
        if code != 200:
            raise RuntimeError(f"chat failed {code}: {str(response)[:500]}")
        receipt = response.get("mlx2") or {}
        usage = response.get("usage") or {}
        choice = (response.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        return {"client_elapsed_s": elapsed, "ttft_s": receipt.get("ttft_seconds"),
                "server_elapsed_s": receipt.get("elapsed_seconds"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "route": receipt.get("route"), "finish_reason": choice.get("finish_reason"),
                "content": (msg.get("content") or "")[:2000],
                "reasoning_chars": len(msg.get("reasoning_content") or ""),
                "prefill_slice_schedule": receipt.get("prefill_slice_schedule"),
                "token_trace": receipt.get("token_trace"),
                "speculation_summary": {k: (receipt.get("speculation") or {}).get(k)
                                        for k in ("accepted", "proposed", "rounds")},
                "receipt_keys": sorted(receipt)}

    def stream_lane(self, content, max_tokens):
        """Background streaming decode lane; returns (thread, state)."""
        state = {"recv": [], "done": False, "error": None}
        body = {"model": self.model_name, "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens, "temperature": 0, "stream": True}

        def pump():
            req = urllib.request.Request(self.base + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=3600) as response:
                    for raw in response:
                        line = raw.decode(errors="replace").strip()
                        if line.startswith("data:") and "[DONE]" not in line:
                            state["recv"].append(time.perf_counter())
            except Exception as error:  # noqa: BLE001
                state["error"] = repr(error)
            state["done"] = True

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        return thread, state

    def wait_tokens(self, state, count, timeout=600):
        deadline = time.monotonic() + timeout
        while len(state["recv"]) < count:
            if state["done"] or time.monotonic() > deadline:
                raise RuntimeError(f"decode lane never reached {count} events: {state}")
            time.sleep(0.01)

    # -- gates ------------------------------------------------------------------
    def thermal_gate(self, limit_s=900, poll_s=15):
        """Wait while thermal_state > 0 (nominal = 0), polling; keep every sample."""
        samples, started = [], time.monotonic()
        while True:
            sample = thermal_sample()
            samples.append({"t": round(time.monotonic() - started, 1), **sample})
            waited = time.monotonic() - started
            if sample["thermal_state"] == 0 or waited >= limit_s:
                return {"state": sample["thermal_state"], "waited_s": round(waited, 1),
                        "passed": sample["thermal_state"] == 0, "samples": samples}
            time.sleep(poll_s)

    # -- scenarios --------------------------------------------------------------
    def cell(self, label, fn):
        gate = self.thermal_gate()
        clear = self.clear(label)
        before = numeric_counters(self.status())
        result = fn()
        after = numeric_counters(self.status())
        return {"label": label, "thermal": gate, "apc_cleared": bool(clear is not None),
                "counters_delta": delta(before, after), **result}

    def scenario_ttft(self, corpus, params, session_index):
        lengths, reps = params["lengths"], params.get("reps", 1)
        cells = []
        for rep in range(reps):
            ordered = lengths if (rep + session_index) % 2 == 0 else lengths[::-1]
            for length in ordered:
                slot = params.get("slot_base", 0) + rep * 31 + lengths.index(length)
                prompt = corpus.prompt(slot, length)
                cells.append(self.cell(f"ttft-{length}-r{rep}", lambda: {
                    "length": length, "rep": rep, "slot": slot,
                    **self.chat(prompt, max_tokens=params.get("max_tokens", 1))}))
                c = cells[-1]
                c["prompt_tok_s"] = (c["prompt_tokens"] / c["ttft_s"]) if c.get("ttft_s") else None
                print(json.dumps({k: c.get(k) for k in ("label", "prompt_tokens", "ttft_s",
                                                         "prompt_tok_s", "cached_tokens")}), flush=True)
        return cells

    def scenario_decode(self, corpus, params, session_index):
        reps, max_tokens = params.get("reps", 1), params.get("max_tokens", 256)
        prompts = DECODE_PROMPTS[:params.get("prompts", len(DECODE_PROMPTS))]
        extra = {"return_token_trace": True, **(params.get("extra") or {})}
        cells = []
        for rep in range(reps):
            for index, prompt in enumerate(prompts):
                r = self.cell(f"decode-p{index}-r{rep}", lambda: {
                    "prompt_index": index, "rep": rep,
                    **self.chat(prompt, max_tokens=max_tokens, extra=extra)})
                n, el, tt = r.get("completion_tokens"), r.get("server_elapsed_s"), r.get("ttft_s")
                r["decode_tok_s"] = ((n - 1) / (el - tt)) if n and el and tt and n > 1 and el > tt else None
                cells.append(r)
                print(json.dumps({k: r.get(k) for k in ("label", "completion_tokens", "decode_tok_s",
                                                         "ttft_s", "route")}), flush=True)
        return cells

    def scenario_slice(self, corpus, params, session_index):
        size = params.get("prompt_tokens", 2048)
        lane_tokens = params.get("lane_tokens", 1200)
        cells = []
        # (1) idle: planned, resend hits
        p1 = corpus.prompt(params.get("slot_base", 0) + 1, size)

        def idle():
            first = self.chat(p1, max_tokens=4)
            resend = self.chat(p1, max_tokens=4)
            return {"first": first, "resend": resend}
        cells.append(self.cell("s1-idle", idle))
        # (2) contended: a long decode lane, then a different 2K prompt
        p2 = corpus.prompt(params.get("slot_base", 0) + 2, size)

        def contended():
            thread, state = self.stream_lane(LONG_LANE_PROMPT, lane_tokens)
            self.wait_tokens(state, 24)
            t0 = len(state["recv"])
            first = self.chat(p2, max_tokens=4)
            lane_during = len(state["recv"]) - t0
            thread.join(timeout=3600)
            resend = self.chat(p2, max_tokens=4)
            return {"first": first, "resend": resend, "lane_events": len(state["recv"]),
                    "lane_events_during_prefill": lane_during, "lane_error": state["error"]}
        cells.append(self.cell("s2-contended", contended))
        # (4) staggered multi-turn prompt: decode lane starts mid-prefill
        turns = []
        for i in range(params.get("turns", 6)):
            turns.append({"role": "user", "content": corpus.prompt(
                params.get("slot_base", 0) + 10 + i, params.get("turn_tokens", 1400),
                question="Note the key point.")})
            turns.append({"role": "assistant", "content": f"Noted point {i + 1}."})
        final_a = turns + [{"role": "user", "content": "Summarise all the notes in one sentence."}]
        final_b = turns + [{"role": "user", "content": "List three risks the notes mention."}]

        def staggered():
            result = {}
            box = {}

            def run_a():
                box["a"] = self.chat(None, max_tokens=4, messages=final_a)
            ta = threading.Thread(target=run_a, daemon=True)
            ta.start()
            time.sleep(params.get("stagger_s", 1.5))
            thread, state = self.stream_lane(LONG_LANE_PROMPT, lane_tokens)
            ta.join(timeout=3600)
            thread.join(timeout=3600)
            result["first"] = box.get("a")
            result["lane_events"] = len(state["recv"])
            result["lane_error"] = state["error"]
            result["divergent_suffix"] = self.chat(None, max_tokens=4, messages=final_b)
            return result
        cells.append(self.cell("s4-staggered-interior", staggered))
        for c in cells:
            print(json.dumps({"label": c["label"], "counters_delta": c["counters_delta"]}), flush=True)
        return cells

    def scenario_interior(self, corpus, params, session_index):
        reps, size = params.get("reps", 3), params.get("prefix_tokens", 8192)
        cells = []
        for rep in range(reps):
            system = corpus.prompt(params.get("slot_base", 0) + 100 + rep, size,
                                   question="Use these notes for the conversation.")

            def run():
                prime = self.chat("Summarise the notes in one sentence.", system=system, max_tokens=1)
                probe = self.chat("Which section mentions qualification first?", system=system,
                                  max_tokens=1)
                return {"rep": rep, "prime": prime, "probe": probe}
            c = self.cell(f"interior-r{rep}", run)
            cells.append(c)
            print(json.dumps({"label": c["label"], "probe_ttft": c["probe"]["ttft_s"],
                              "prime_ttft": c["prime"]["ttft_s"],
                              "probe_cached": c["probe"]["cached_tokens"]}), flush=True)
        return cells

    def scenario_interior_parity(self, corpus, params, session_index):
        """Warm continuation from an interior checkpoint against a cold prefill.

        Per rep: prime a shared system prompt (its interior checkpoints
        publish), then a different question with greedy ``max_tokens``
        (warm); clear APCv2 and send that request cold twice (cold, plus a
        cold-versus-cold control).  When warm and cold token ids differ, the
        same session replays both arms with one bounded logprob window at the
        first divergent position only (AGENTS.md cold/warm drift rule).
        """
        reps, size = params.get("reps", 3), params.get("prefix_tokens", 8192)
        max_tokens, top = params.get("max_tokens", 128), params.get("top_logprobs", 20)
        trace = {"return_token_trace": True}
        cells = []

        def ids(result):
            return list(((result.get("token_trace") or {}).get("token_ids")) or [])

        def first_divergence(a, b):
            for index, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    return index
            return None if len(a) == len(b) else min(len(a), len(b))

        for rep in range(reps):
            system = corpus.prompt(params.get("slot_base", 0) + 300 + rep, size,
                                   question="Use these notes for the conversation.")
            question = params.get("question", "Explain in detail what the notes say about "
                                  "qualification and route receipts.")

            def run():
                prime = self.chat("Summarise the notes in one sentence.", system=system,
                                  max_tokens=1)
                warm = self.chat(question, system=system, max_tokens=max_tokens, extra=trace)
                self.clear(f"parity-r{rep}-cold")
                cold = self.chat(question, system=system, max_tokens=max_tokens, extra=trace)
                self.clear(f"parity-r{rep}-cold2")
                cold2 = self.chat(question, system=system, max_tokens=max_tokens, extra=trace)
                result = {"rep": rep, "prime": prime, "warm": warm, "cold": cold,
                          "cold2": cold2,
                          "warm_vs_cold_first_divergence": first_divergence(ids(warm), ids(cold)),
                          "cold_vs_cold_first_divergence": first_divergence(ids(cold), ids(cold2))}
                index = result["warm_vs_cold_first_divergence"]
                if index is not None and index < min(len(ids(warm)), len(ids(cold))):
                    position = index + 1
                    window = {"logprobs": True, "top_logprobs": top,
                              "logprobs_start": position, "logprobs_end": position, **trace}

                    def windowed():
                        body = {"model": self.model_name, "temperature": 0,
                                "max_tokens": position, "messages": [
                                    {"role": "system", "content": system},
                                    {"role": "user", "content": question}], **window}
                        code, response = http(self.base + "/v1/chat/completions", body)
                        choice = (response.get("choices") or [{}])[0] if code == 200 else {}
                        receipt = (response.get("mlx2") or {}) if code == 200 else {}
                        return {"http_status": code, "logprobs": choice.get("logprobs"),
                                "token_trace": receipt.get("token_trace"),
                                "cached_tokens": ((response.get("usage") or {}).get(
                                    "prompt_tokens_details") or {}).get("cached_tokens")
                                if code == 200 else None}
                    self.clear(f"parity-r{rep}-window-warm")
                    self.chat("Summarise the notes in one sentence.", system=system, max_tokens=1)
                    warm_window = windowed()
                    self.clear(f"parity-r{rep}-window-cold")
                    cold_window = windowed()
                    result["divergence_window"] = {"position": position,
                                                   "warm": warm_window, "cold": cold_window}
                return result
            c = self.cell(f"parity-r{rep}", run)
            cells.append(c)
            print(json.dumps({"label": c["label"], "warm_cached": c["warm"]["cached_tokens"],
                              "warm_tokens": len(ids(c["warm"])),
                              "warm_vs_cold": c["warm_vs_cold_first_divergence"],
                              "cold_vs_cold": c["cold_vs_cold_first_divergence"]}), flush=True)
        return cells

    def scenario_logprob_window(self, corpus, params, session_index):
        """Localize a known first divergence: replay the greedy prompt and read
        one bounded logprob window at that already-selected position only."""
        cells = []
        for window in params["windows"]:
            index, position = window["prompt_index"], window["position"]
            extra = {"return_token_trace": True, "logprobs": True,
                     "top_logprobs": params.get("top_logprobs", 8),
                     "logprobs_start": position, "logprobs_end": position}

            def run():
                body = {"model": self.model_name, "temperature": 0, "max_tokens": position,
                        "messages": [{"role": "user", "content": DECODE_PROMPTS[index]}], **extra}
                code, response = http(self.base + "/v1/chat/completions", body)
                if code != 200:
                    return {"http_status": code, "error": str(response)[:800]}
                choice = (response.get("choices") or [{}])[0]
                receipt = response.get("mlx2") or {}
                return {"prompt_index": index, "position": position,
                        "logprobs": choice.get("logprobs"),
                        "token_trace": receipt.get("token_trace"), "route": receipt.get("route")}
            c = self.cell(f"window-p{index}-t{position}", run)
            cells.append(c)
            print(json.dumps({"label": c["label"], "route": c.get("route"),
                              "error": c.get("error")}), flush=True)
        return cells

    def scenario_multi(self, corpus, params, session_index):
        cells = []
        for step in params["steps"]:
            fn = getattr(self, "scenario_" + step["scenario"])
            for c in fn(corpus, step.get("params") or {}, session_index):
                c["scenario"] = step["scenario"]
                cells.append(c)
        return cells

    def scenario_concurrent(self, corpus, params, session_index):
        reps, width = params.get("reps", 3), params.get("width", 4)
        size, max_tokens = params.get("prompt_tokens", 2048), params.get("max_tokens", 256)
        cells = []
        for rep in range(reps):
            prompts = [corpus.prompt(params.get("slot_base", 0) + 200 + rep * width + i, size,
                                     question=DECODE_PROMPTS[i % len(DECODE_PROMPTS)])
                       for i in range(width)]

            def run():
                box = [None] * width
                started = time.perf_counter()

                def one(i):
                    box[i] = self.chat(prompts[i], max_tokens=max_tokens)
                threads = [threading.Thread(target=one, args=(i,)) for i in range(width)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                wall = time.perf_counter() - started
                tokens = sum((b or {}).get("completion_tokens") or 0 for b in box)
                return {"rep": rep, "requests": box, "wall_s": wall, "out_tok_s": tokens / wall,
                        "ttft_s": [b["ttft_s"] for b in box]}
            c = self.cell(f"concurrent-b{width}-r{rep}", run)
            cells.append(c)
            print(json.dumps({"label": c["label"], "out_tok_s": c["out_tok_s"], "ttft": c["ttft_s"]}),
                  flush=True)
        return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8471)
    ap.add_argument("--token-dir", required=True)
    ap.add_argument("--lock-dir", default="/tmp/mlx2-gpu-c.lock")
    ap.add_argument("--max-swapout-pages", type=int, default=2000)
    ap.add_argument("--reference-max-tokens", type=int, default=16)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    plan = json.loads(Path(a.plan).read_text())
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing nonempty evidence dir {out}")
    out.mkdir(parents=True, exist_ok=True)
    from mlx2.adapters.registry import inspect_model

    resolution = inspect_model(plan["model"])
    checkpoint = f"{Path(plan['model']).name}@sha256:{resolution.artifact_fingerprint}"
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain",
                                     "--untracked-files=no"], text=True).strip()
    owner = json.loads((Path(a.lock_dir) / "owner.json").read_text())
    lease = f"{owner.get('cpg_task')}:{owner.get('cpg_generation')}"
    sessions_per_arm = {k: plan["order"].count(k) for k in plan["arms"]}
    report = {"schema": SCHEMA, "plan": plan, "checkpoint": checkpoint, "mlx2_commit": commit,
              "tree_dirty": bool(dirty), "lock_owner": owner, "started_at": time.time(),
              "controlled": min(sessions_per_arm.values()) >= 3 or bool(plan.get("controlled")),
              "sessions_per_arm": sessions_per_arm, "sessions": [], "status": "running",
              "kind": "performance A/B / feature validation; not qualification"}
    save = lambda: atomic_json(out / "unit.json", report)  # noqa: E731
    report["swapouts_start"] = swapouts()
    save()
    token_dir = Path(a.token_dir)
    token_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1"}
    server = None
    try:
        ref = subprocess.run(
            [sys.executable, str(ROOT / "scripts/smoke_local_model.py"), "--model", plan["model"],
             "--out", str(out / "ordinary-reference.json"), "--max-tokens", str(a.reference_max_tokens),
             "--i-own-gpu", "--cpg-lease", lease],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
        (out / "ordinary-reference.log").write_text(ref.stdout + ref.stderr)
        report["ordinary_reference_rc"] = ref.returncode
        save()
        if ref.returncode != 0:
            raise RuntimeError("ordinary-reference smoke failed")
        corpus = Corpus(plan["model"])
        for index, arm_name in enumerate(plan["order"]):
            arm = plan["arms"][arm_name]
            sdir = out / f"s{index:02d}-{arm_name}"
            sdir.mkdir()
            token_file = token_dir / f"admin-token-{os.getpid()}-{index}"
            token = secrets.token_hex(24)
            token_file.write_text(token)
            os.chmod(token_file, 0o600)
            command = [sys.executable, "-m", "mlx2.server", "--model", plan["model"], "--host",
                       "127.0.0.1", "--port", str(a.port), "--fabric", "off", "--cache-dir",
                       str(sdir / "cache"), "--admin-token-file", str(token_file),
                       *plan.get("common_server_args", []), *arm.get("server_args", [])]
            if arm.get("execution_policy") is not None:
                policy_path = sdir / "execution-policy.json"
                policy_path.write_text(json.dumps(arm["execution_policy"], indent=1))
                command += ["--execution-policy", str(policy_path)]
            # Thermal control instead of fixed cooldowns (Pierre, 2026-10-28):
            # start each arm as soon as the host is nominal, wait only while
            # it is above that, and record every sample.
            pre_gate = Unit(plan, out, a.port, None, env).thermal_gate()
            session = {"index": index, "arm": arm_name, "server_command": command,
                       "swapouts_before": swapouts(), "thermal_gate_before_arm": pre_gate,
                       "thermal_before": thermal_sample(), "pmset_therm_before": pmset_therm()}
            report["sessions"].append(session)
            save()
            unit = Unit(plan, out, a.port, token, env)
            try:
                load_t0 = time.monotonic()
                with (sdir / "server.log").open("w") as log:
                    server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True)
                status = wait_health(unit.base, server, 1800)
                session["load_s"] = time.monotonic() - load_t0
                atomic_json(sdir / "startup-status.json", status)
                binding = (status.get("execution") or {}).get("artifact_binding") or {}
                target = binding.get("target_fingerprint") or binding.get("fingerprint")
                if target != resolution.artifact_fingerprint:
                    raise RuntimeError(f"target artifact mismatch {target}")
                session["startup"] = {
                    "route": (status.get("settings") or {}).get("route"),
                    "metal_peak_bytes_after_load": status.get("metal_peak_bytes"),
                    "footprint_after_load": status.get("process_physical_footprint_bytes"),
                    "qualification": status.get("qualification") if not isinstance(
                        status.get("qualification"), dict) else (status["qualification"].get("state")),
                    "route_receipt": status.get("route_receipt") if not isinstance(
                        status.get("route_receipt"), dict) else status["route_receipt"].get("state"),
                }
                # warm-up, discarded
                unit.clear("warmup")
                session["warmup"] = unit.chat(corpus.prompt(999, 512), max_tokens=8)
                fn = getattr(unit, "scenario_" + plan["scenario"])
                session["cells"] = fn(corpus, plan.get("params") or {}, index)
                atomic_json(sdir / "status-after.json", unit.status())
            finally:
                stop_server(server)
                session["server_stopped"] = server is None or server.poll() is not None
                server = None
                try:
                    token_file.unlink()
                except OSError:
                    pass
            session["swapouts_after"] = swapouts()
            session["thermal_after"] = thermal_sample()
            session["pmset_therm_after"] = pmset_therm()
            save()
            grew = session["swapouts_after"] - session["swapouts_before"]
            if grew > a.max_swapout_pages:
                raise RuntimeError(f"aborting: Swapouts grew by {grew} pages in session {index}")
            if index + 1 < len(plan["order"]) and plan.get("cooldown_s", 0):
                time.sleep(plan["cooldown_s"])  # optional; default 0 (thermal gate instead)
        report["status"] = "finished"
    except BaseException as error:  # noqa: BLE001 - one receipt, always
        report["status"] = "error"
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    finally:
        stop_server(server)
        report["finished_at"] = time.time()
        report["swapouts_end"] = swapouts()
        save()
    raise SystemExit(0 if report["status"] == "finished" else 1)


if __name__ == "__main__":
    main()
