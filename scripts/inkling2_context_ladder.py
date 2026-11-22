#!/usr/bin/env python3
"""Inkling 2 Mini single-rep context ladder: ordinary, DSpark spec decode and opt-in knob A/B arms.

Original mlx2 script.  A sanity and correctness ladder, not publishable statistics: one
repetition per cell, no thermal gating (AGENTS.md).  The caller owns the GPU
(``cpg_job.py`` lease -> ``run_with_gpu_locks.py``; see ``scripts/inkling2_ladder_window.sh``).

What "AST spec decode" means here.  Inkling 2 declares no AST tree tiles and no native MTP
head; ``--ast`` lowers to ``--native-mtp`` (the artifact carries ``mtp.safetensors``) and the
server refuses it before any weight loads.  The ladder records that refusal as its first step
(``ast_probe``: CPU-only argument resolution, no model load) and runs the closest supported
speculative route instead, labelled ``dspark``: the bundled DSpark drafter as an external draft,
``--external-draft --execution-policy {"draft_model": "embedded_dspark", "num_draft": 7,
"proposal_composition": false}`` (implemented, opt-in, unqualified).

Phases, each one loopback ``mlx2.server`` (never :8282) under the memory watchdog
(``scripts/research/memwatched_python``), stopped as a whole process group in ``finally``,
in priority order (a budget cut drops the tail):

  ordinary_b1          --ordinary, --max-lanes 1, full ladder        (reference)
  dspark_b1            DSpark external draft, --max-lanes 1, full ladder  vs ordinary_b1
  ordinary_b4          --ordinary, --max-lanes 4, --b4-cells (1K, 4K)  lane 0 vs ordinary_b1
  dspark_b4_refusal    DSpark at --max-lanes 4: the server must refuse at startup (B1 only;
                       HybridVerifyRows rotating-KV rollback).  No request is sent.
  hard_no_thinking_b1  --ordinary + {"inkling2_hard_no_thinking": true}, --ab-cells
                       (prompt differs by the text-channel prefill: no token comparison)
  sink_sdpa_b1         --ordinary + {"inkling2_sink_sdpa": true}, --ab-cells  vs ordinary_b1
  mask512_b1           --ordinary + {"inkling2_prefill_mask_mb": 512}, --ab-cells  vs ordinary_b1

Per cell (context length) and phase:

* APCv2 is cleared (``POST /v1/admin/apc/clear``, admin token);
* ``width`` prompts of about ``cell`` tokens (varied repository text from
  ``realistic_haystack``, never repeated strings), each with its own access code buried at its
  own depth; lane 0's prompt is the B1 prompt of that cell, so B1 and B4 lane 0 compare;
* cold: all lanes sent concurrently, streaming, greedy (temperature 0), fixed seed,
  ``reasoning_effort`` none, ``logprobs`` on; then warm: the same bodies again.  At width 1 the
  warm request also carries ``top_logprobs`` 5 (a non-prompt field: same APCv2 key, same route),
  so every width-1 cell has a full evidence window bound to its ids without an extra request;
  its decode rate is therefore marked evidence-bearing, and cold is the timing arm;
* evidence: after the timed arms, the first divergences that need a window are listed (cold vs
  warm per lane; arm vs reference per lane).  Missing windows are captured by untimed
  ``top_logprobs`` 5 requests capped at ``max(k) + 1`` tokens, all lanes concurrently so the
  batch shape matches: a warm-shaped attempt (APC hit) and, when a cold-arm window is still
  missing and the cell is at most ``--cold-evidence-max``, a cold-shaped attempt (APC cleared
  first, so the prefix is computed by cold prefill as in the timed cold arm).  A window counts
  only when the evidence ids equal that timed arm's ids through the divergence;
* recorded: TTFT per lane, cold prefill tok/s, decode tok/s per lane and aggregate, the
  functional oracle scored on ``reasoning_content`` + ``content`` (code present, Unicode dashes
  normalized, otherwise exact; raw text of both channels kept) with the channel the code
  appeared in, ``request_controls.thinking_control.mode``, the terminal receipt (route, cached
  tokens, speculation), mean acceptance length (``spec_acceptance_gsm8k.acceptance_summary``),
  the adapter's ``kernel_policy`` and ``kernel_counters`` (knob engagement), server
  ``metal_peak_bytes`` and physical footprint, swap, and cell wall time.

Classification (``mlx2.near_tie``: shared top-two / exact-tie / multi-way, 0.5-nat ceiling, plus
the functional oracle on both continuations): ``near_tie_equivalent`` passes; a divergence whose
windows prove it beyond the contract is ``high_margin_divergence`` (a failure); a missing or
unbound window leaves it ``unclassified``.  Cold vs warm at width 1 must be exact (APCv2
width-one replay gate) and is classified only as diagnosis; at width 4 a classified near tie is
a passing row.

Budget.  ``--budget-min`` bounds the GPU wall time: before each cell the ladder predicts its cost
(the first run's measured ordinary B1/B4 TTFT and decode rates at ``--max-tokens``, corrected by
the actual/predicted ratio of cells already run) and skips a cell that would end past the
budget.  ``--cell-cap-s`` skips further evidence attempts in a cell past the cap and stops a
phase after a cell that exceeded it.

Refuses a nonempty output directory and an occupied port.  Writes ``ladder.json`` atomically
after every cell.  ``--dry-run`` builds every prompt with the tokenizer, runs the ``--ast``
probe, and writes the plan, the server commands and the cost estimate without starting a server
(CPU only).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import hy4_context_ladder as H  #  (swap, git, get/post_json, find_key, stop_group, checkpoint_label)
from realistic_haystack import bury, haystack

SCHEMA = "mlx2.inkling2-context-ladder.v2"
DSPARK_POLICY = {"draft_model": "embedded_dspark", "num_draft": 7, "proposal_composition": False}
DEPTHS = (0.10, 0.25, 0.50, 0.75, 0.90, 0.33, 0.60, 0.15, 0.80, 0.45)
LANE_DEPTH_STEP = 0.17
# policy key -> the adapter's kernel_policy name (src/mlx2/adapters/inkling2.py validate_policy)
KNOB_NAMES = {"inkling2_hard_no_thinking": "hard_no_thinking", "inkling2_sink_sdpa": "sink_sdpa",
              "inkling2_prefill_mask_mb": "prefill_mask_mb", "inkling2_sink_slot": "sink_slot",
              "inkling2_fuse_controllers": "fuse_controllers", "inkling2_fuse_residual": "fuse_residual",
              "inkling2_fuse_sconv": "fuse_sconv", "inkling2_fuse_attention": "fuse_attention",
              "inkling2_verify_splitk": "verify_splitk"}
# Everything exact or near-tie classified so far, composed (2026-11-22, main
# 82c964317): MoE fast paths (default) + sink SDPA (one-token decode) + sink
# slot (multi-row calls) + the four exact launch-count fusions.
COMPOSED = {"inkling2_sink_sdpa": True, "inkling2_sink_slot": True,
            "inkling2_fuse_controllers": True, "inkling2_fuse_residual": True,
            "inkling2_fuse_sconv": True, "inkling2_fuse_attention": True}
PHASES = (
    {"name": "ordinary_b1", "route": "ordinary", "width": 1, "reference": None, "policy": {},
     "cells": "full"},
    {"name": "dspark_b1", "route": "dspark", "width": 1, "reference": "ordinary_b1",
     "policy": DSPARK_POLICY, "cells": "full"},
    {"name": "ordinary_b4", "route": "ordinary", "width": 4, "reference": "ordinary_b1", "policy": {},
     "cells": "b4"},
    {"name": "dspark_b4_refusal", "route": "dspark", "width": 4, "reference": None,
     "policy": DSPARK_POLICY, "cells": None, "refusal": "B1 only"},
    {"name": "hard_no_thinking_b1", "route": "ordinary", "width": 1, "reference": None,
     "policy": {"inkling2_hard_no_thinking": True}, "cells": "ab", "thinking_mode": "hard_text_prefill"},
    {"name": "sink_sdpa_b1", "route": "ordinary", "width": 1, "reference": "ordinary_b1",
     "policy": {"inkling2_sink_sdpa": True}, "cells": "ab", "counter": "sink_sdpa_calls"},
    {"name": "mask512_b1", "route": "ordinary", "width": 1, "reference": "ordinary_b1",
     "policy": {"inkling2_prefill_mask_mb": 512}, "cells": "ab"},
    # 2026-11-21 (main 63c2cad98, MoE fast paths default on in every phase):
    # DSpark with the per-chunk prefill allocator reclaim (the 32K memwatch fix;
    # Pierre: default if it passes), and the all-optimizations ordinary arm
    # (MoE defaults + sink SDPA decode) over the full ladder.
    {"name": "dspark_reclaim_b1", "route": "dspark", "width": 1, "reference": "ordinary_b1",
     "policy": {**DSPARK_POLICY, "inkling2_dspark_prefill_reclaim": True}, "cells": "full",
     "status_flag": "prefill_reclaim"},
    {"name": "allopt_b1", "route": "ordinary", "width": 1, "reference": "ordinary_b1",
     "policy": {"inkling2_sink_sdpa": True}, "cells": "full", "counter": "sink_sdpa_calls"},
    # Everything plus the drafter (Pierre 2026-11-21): DSpark + prefill reclaim
    # + MoE defaults + sink SDPA (single-row rounds) + sink slot (the 8-row
    # verify windows, which sink SDPA does not cover).
    {"name": "dspark_allopt_b1", "route": "dspark", "width": 1, "reference": "ordinary_b1",
     "policy": {**DSPARK_POLICY, "inkling2_dspark_prefill_reclaim": True,
                "inkling2_sink_sdpa": True, "inkling2_sink_slot": True},
     "cells": "full", "counter": "sink_slot_calls", "status_flag": "prefill_reclaim"},
    {"name": "composed_b1", "route": "ordinary", "width": 1, "reference": "ordinary_b1",
     "policy": dict(COMPOSED), "cells": "full", "counter": "sink_sdpa_calls"},
    {"name": "dspark_composed_b1", "route": "dspark", "width": 1, "reference": "ordinary_b1",
     "policy": {**DSPARK_POLICY, **COMPOSED}, "cells": "full", "counter": "sink_slot_calls",
     "status_flag": "prefill_reclaim"},
    # The composed set plus the split-K short-block attention kernel
    # (inkling2_verify_splitk, omlx #5511 port): it takes the 1-31 row blocks
    # (decode, DSpark verify) ahead of sink SDPA and the sink slot.
    {"name": "composed_splitk_b1", "route": "ordinary", "width": 1, "reference": "ordinary_b1",
     "policy": {**COMPOSED, "inkling2_verify_splitk": True}, "cells": "full",
     "counter": "verify_splitk_calls"},
    {"name": "dspark_composed_splitk_b1", "route": "dspark", "width": 1, "reference": "ordinary_b1",
     "policy": {**DSPARK_POLICY, **COMPOSED, "inkling2_verify_splitk": True}, "cells": "full",
     "counter": "verify_splitk_calls", "status_flag": "prefill_reclaim"},
)
QUESTION = ("\n\nQuestion: What is the access code for the north archive vault? Give the code "
            "on the first line, then summarize the reference material above in about 120 words.")
INTRO = "Read the reference material below, then answer the question at the end.\n\n"
EVIDENCE_TOP = 5
CEILING = 0.5
# First run (main 01d3b52a5, Inkling-2-Mini@440956c8, ordinary): B1 cold TTFT s and decode tok/s;
# B4 TTFT multiplier and slowest-lane decode fraction of B1.  Only for the budget estimate.
R1_B1 = {1024: (1.0, 28.3), 4096: (3.2, 27.0), 16384: (14.8, 27.1), 32768: (40.3, 24.5),
         65536: (144.9, 21.4), 131072: (548.4, 15.0)}
R1_B4_TTFT = 5.5
R1_B4_DECODE = {1024: 0.51, 4096: 0.37, 16384: 0.23, 32768: 0.23, 65536: 0.23, 131072: 0.23}
LOAD_S = 15.0
REFUSAL_S = 30.0


def make_code(cell: int, lane: int) -> str:
    digest = hashlib.sha256(f"inkling2-ladder-{cell}-{lane}".encode()).hexdigest().upper()
    letters = "".join(c for c in digest if c.isalpha())[:3] or "QZX"
    return f"{letters}-{int(digest[:8], 16) % 90000 + 10000}-{digest[8:10]}"


def build_prompt(tokenizer, cell: int, depth: float, code: str, offset: int):
    fact = f"The access code for the north archive vault is {code}."
    chars = int(cell * 4.0)
    for _ in range(5):  # calibrate the length to the tokenizer
        text = INTRO + bury(haystack(chars, offset=offset), fact, depth) + QUESTION
        tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if abs(tokens - cell) <= max(32, cell // 100):
            break
        chars = max(256, int(chars * cell / tokens))
    return text, tokens


def lane_prompts(tokenizer, cell_index: int, cell: int, width: int):
    """``width`` lane prompts; lane 0 is the same prompt at every width."""
    out = []
    for lane in range(width):
        depth = round((DEPTHS[cell_index % len(DEPTHS)] + lane * LANE_DEPTH_STEP) % 0.9 + 0.05, 3)
        code = make_code(cell, lane)
        offset = cell_index * 200_000 + lane * 50_000
        text, tokens = build_prompt(tokenizer, cell, depth, code, offset)
        out.append({"lane": lane, "depth": depth, "code": code, "offset": offset, "text": text,
                    "prompt_tokens_estimate": tokens,
                    "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()})
    return out


def parse_stream(lines, started: float, clock=time.perf_counter) -> dict:
    """One SSE chat stream: both text channels, token ids (and top windows), timings, receipt."""
    arrivals, ids, windows, parts, reasoning = [], [], [], [], []
    receipt, finish, done = None, None, False
    for raw in lines:
        line = raw.decode("utf-8").strip() if isinstance(raw, bytes) else raw.strip()
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
                top = {int(t["id"]): float(t["logprob"]) for t in item.get("top_logprobs") or () if "id" in t}
                if top and "logprob" in item:  # the selected token's own value, from the same row
                    top.setdefault(int(item["id"]), float(item["logprob"]))
                windows.append(top or None)
                arrivals.append(clock() - started)
            delta = choice.get("delta") or {}
            parts.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
            finish = choice.get("finish_reason") or finish
    if not done or receipt is None:
        raise ValueError("incomplete generation stream or absent terminal receipt")
    decode = ((len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
              if len(arrivals) > 1 and arrivals[-1] > arrivals[0] else None)
    return {"text": "".join(parts), "reasoning_text": "".join(reasoning), "token_ids": ids,
            "windows": windows, "ttft_s": arrivals[0] if arrivals else None,
            "first_s": arrivals[0] if arrivals else None, "last_s": arrivals[-1] if arrivals else None,
            "elapsed_s": clock() - started, "decode_tok_s": decode, "finish_reason": finish,
            "receipt": receipt}


def stream_one(base: str, body: dict, *, timeout: float, t0: float | None = None) -> dict:
    import urllib.request

    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter() if t0 is None else t0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return parse_stream(response, started)


def send_lanes(send, bodies):
    """All lanes concurrently from one shared start (``send(body, t0)``)."""
    t0 = time.perf_counter()
    if len(bodies) == 1:
        return [send(bodies[0], t0)]
    with ThreadPoolExecutor(max_workers=len(bodies)) as pool:
        return list(pool.map(lambda body: send(body, t0), bodies))


def oracle(text: str, code: str) -> bool:
    return code in (text or "").translate(H.DASHES)


def oracle_channels(result: dict, code: str) -> dict:
    """The functional oracle over ``reasoning_content`` + ``content`` and where the code was."""
    content = oracle(result.get("text"), code)
    reasoning = oracle(result.get("reasoning_text"), code)
    channel = "both" if content and reasoning else "content" if content else "reasoning" if reasoning else None
    return {"pass": content or reasoning, "channel": channel}


def scored(result: dict, code: str) -> bool:
    return oracle_channels(result, code)["pass"]


def aggregate(results, prompt_tokens):
    """Width-level timing: aggregate decode over the union of lanes' decode spans."""
    firsts = [r["first_s"] for r in results if r.get("first_s") is not None]
    lasts = [r["last_s"] for r in results if r.get("last_s") is not None]
    generated = sum(max(len(r["token_ids"]) - 1, 0) for r in results)
    span = (max(lasts) - min(firsts)) if firsts and lasts else None
    pt = sum(t for t in prompt_tokens if t)
    ttft_max = max(firsts) if firsts else None
    return {"ttft_max_s": ttft_max, "ttft_min_s": min(firsts) if firsts else None,
            "prefill_tok_s_aggregate": round(pt / ttft_max, 1) if pt and ttft_max else None,
            "decode_tok_s_aggregate": round(generated / span, 2) if span and span > 0 else None,
            "generated_tokens": sum(len(r["token_ids"]) for r in results)}


def first_divergence(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _evidence_list(evidence):
    if evidence is None:
        return []
    return [evidence] if isinstance(evidence, dict) else [e for e in evidence if e]


def bound_window(timed_ids, k, evidence):
    """``(record, window)`` of the first evidence whose ids equal ``timed_ids`` through ``k``."""
    for record in _evidence_list(evidence):
        ids, windows = record.get("token_ids") or [], record.get("windows") or []
        if len(ids) > k and len(windows) > k and ids[:k + 1] == list(timed_ids[:k + 1]) and windows[k]:
            return record, windows[k]
    return None, None


def _proves_far(window, token):
    """The window proves ``token`` is more than the ceiling below its top (absent or present)."""
    top = max(window.values())
    if token in window:
        return top - window[token] > CEILING
    return len(window) >= EVIDENCE_TOP and top - min(window.values()) > CEILING


def classify(ref, arm, ref_evidence, arm_evidence, code):
    """First-divergence row for one lane: ``ref``/``arm`` timed results, evidence lists bound to them."""
    from mlx2.near_tie import near_tie_equivalent

    a, b = ref["token_ids"], arm["token_ids"]
    if not a and not b:
        return {"verdict": "no_evidence", "why": "no token ids in either response"}
    k = first_divergence(a, b)
    if k is None:
        return {"verdict": "identical", "tokens": len(a)}
    row = {"first_divergence": k, "ref_token": a[k] if k < len(a) else None,
           "arm_token": b[k] if k < len(b) else None,
           "ref_text": ref.get("text"), "arm_text": arm.get("text"),
           "ref_reasoning_text": ref.get("reasoning_text"), "arm_reasoning_text": arm.get("reasoning_text")}
    if k >= min(len(a), len(b)):
        return {**row, "verdict": "unclassified", "why": "length-only difference"}
    func = scored(ref, code) and scored(arm, code)
    row["functional_oracle_both"] = func
    windows = {}
    for name, timed, ev in (("ref", a, ref_evidence), ("arm", b, arm_evidence)):
        record, window = bound_window(timed, k, ev)
        windows[name] = window
        row[f"{name}_window"] = {str(t): v for t, v in window.items()} if window else None
        row[f"{name}_window_source"] = record.get("shape") if record else None
    if not func:
        return {**row, "verdict": "failed", "why": "functional oracle failed on a continuation"}
    if not (windows["ref"] and windows["arm"]):
        return {**row, "verdict": "unclassified",
                "why": "evidence window missing or its ids differ from the timed ids through the divergence"}
    if near_tie_equivalent(a[k], windows["ref"], b[k], windows["arm"], ceiling=CEILING):
        return {**row, "verdict": "near_tie_equivalent", "why": None}
    if _proves_far(windows["ref"], b[k]) or _proves_far(windows["arm"], a[k]):
        return {**row, "verdict": "high_margin_divergence",
                "why": "a selected token is more than 0.5 nat below the other arm's top"}
    return {**row, "verdict": "unclassified",
            "why": "outside the near-tie contract, but a token is missing from a window that does not bound it"}


def compare_phases(reference, arm):
    """Arm vs its reference, per cell and lane present in both (cold ids, evidence bound to them)."""
    rows, counts = [], {}
    ref_cells = {c["cell"]: c for c in reference.get("cells", ()) if c.get("cold")}
    for cell in arm.get("cells", ()):
        ref = ref_cells.get(cell["cell"])
        if not cell.get("cold") or ref is None:
            continue
        for lane, (r, s) in enumerate(zip(ref["cold"], cell["cold"])):
            row = classify(r, s, (ref.get("evidence") or [[]] * 8)[lane],
                           (cell.get("evidence") or [[]] * 8)[lane], cell["lanes"][lane]["code"])
            row.update(cell=cell["cell"], lane=lane)
            counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
            rows.append(row)
    return {"reference": reference["name"], "arm": arm["name"], "counts": counts, "rows": rows}


def evidence_needs(row, width, ref_cell=None):
    """``[(lane, "cold"|"warm", k)]``: first divergences that need a window on that timed arm."""
    needs = []
    for lane in range(width):
        cold, warm = row["cold"][lane]["token_ids"], row["warm"][lane]["token_ids"]
        k = first_divergence(cold, warm)
        if k is not None and k < min(len(cold), len(warm)):
            needs += [(lane, "cold", k), (lane, "warm", k)]
        if ref_cell and lane < len(ref_cell.get("cold") or ()):
            ref = ref_cell["cold"][lane]["token_ids"]
            k = first_divergence(ref, cold)
            if k is not None and k < min(len(ref), len(cold)):
                needs.append((lane, "cold", k))
    return sorted(set(needs))


def unmet_needs(row, needs):
    return [(lane, arm, k) for lane, arm, k in needs
            if bound_window(row[arm][lane]["token_ids"], k, row["evidence"][lane])[0] is None]


def cold_warm(cell, width):
    out = []
    for lane, (c, w) in enumerate(zip(cell["cold"], cell["warm"])):
        k = first_divergence(c["token_ids"], w["token_ids"])
        entry = {"lane": lane, "equal": k is None, "first_divergence": k,
                 "warm_cached_tokens": w.get("cached_tokens")}
        if k is not None:
            ev = (cell.get("evidence") or [[]] * 8)[lane]
            entry["classification"] = {key: v for key, v in classify(c, w, ev, ev, cell["lanes"][lane]["code"]).items()
                                       if not key.endswith("_text")}
        out.append(entry)
    return {"width": width, "lanes": out, "exact_required": width == 1,
            "all_equal": all(r["equal"] for r in out)}


def phase_verdict(phase):
    ran = [c for c in phase.get("cells", ()) if c.get("cold") and c.get("warm")]
    oracle_fail = [f"{c['cell']}:{arm}:{i}" for c in ran for arm in ("cold", "warm")
                   for i, r in enumerate(c[arm]) if not r.get("oracle_pass")]
    channels = {}
    for c in ran:
        for arm in ("cold", "warm"):
            for r in c[arm]:
                channels[str(r.get("oracle_channel"))] = channels.get(str(r.get("oracle_channel")), 0) + 1
    empty_content = [f"{c['cell']}:{arm}:{i}" for c in ran for arm in ("cold", "warm")
                     for i, r in enumerate(c[arm]) if not (r.get("text") or "").strip()]
    warm_miss = [f"{c['cell']}:{i}" for c in ran for i, r in enumerate(c["warm"]) if not r.get("cached_tokens")]
    exact_fail = [c["cell"] for c in ran if c.get("cold_warm") and c["cold_warm"]["exact_required"]
                  and not c["cold_warm"]["all_equal"]]
    batched = [(c["cell"], lane) for c in ran for lane in (c.get("cold_warm") or {}).get("lanes", ())
               if not c["cold_warm"]["exact_required"] and not lane["equal"]]
    batched_verdicts = {}
    for _cell, lane in batched:
        v = (lane.get("classification") or {}).get("verdict", "unclassified")
        batched_verdicts[v] = batched_verdicts.get(v, 0) + 1
    spec_missing = ([c["cell"] for c in ran if not all((r.get("acceptance") or {}).get("verify_rounds")
                                                       for r in c["cold"])]
                    if phase["route"] == "dspark" else [])
    route_wrong = [c["cell"] for c in ran if any(r.get("route") != phase["expected_route"] for r in c["cold"])]
    mode_wrong = [f"{c['cell']}:{arm}:{i}" for c in ran for arm in ("cold", "warm")
                  for i, r in enumerate(c[arm]) if r.get("thinking_mode") != phase["expected_thinking_mode"]]
    knob_unengaged = [c["cell"] for c in ran if c.get("knob_engaged") is False]
    return {"cells_ran": len(ran), "cells_skipped": [c["cell"] for c in phase.get("cells", ()) if c.get("skipped")],
            "oracle_failures": oracle_fail, "oracle_channels": channels, "empty_content": empty_content,
            "warm_misses": warm_miss,
            "width_one_cold_warm_exact_failures": exact_fail,
            "batched_cold_warm_divergences": batched_verdicts,
            "batched_cold_warm_failures": [f"{cell}:{lane['lane']}" for cell, lane in batched
                                           if (lane.get("classification") or {}).get("verdict") != "near_tie_equivalent"],
            "spec_cells_without_verify_rounds": spec_missing, "route_mismatches": route_wrong,
            "thinking_mode_mismatches": mode_wrong, "knob_not_engaged": knob_unengaged}


def _policy(phase):
    return dict(phase.get("policy") or {})


def server_command(a, phase, model, token_file, policy_file, max_context):
    cmd = [str(a.server_python), "-u", "-m", "mlx2.server", "--model", str(model),
           "--host", "127.0.0.1", "--port", str(a.port),
           "--max-lanes", str(phase["width"]), "--max-inflight", str(phase["width"]),
           "--max-context", str(max_context), "--admin-token-file", str(token_file), "--no-cache-dir"]
    if phase["route"] == "ordinary":
        cmd.append("--ordinary")
        if _policy(phase):
            cmd += ["--execution-policy", str(policy_file)]
    else:
        cmd += ["--external-draft", "--execution-policy", str(policy_file)]
    return cmd


def ast_probe(a, model, out):
    """``--ast`` on Inkling 2: argument resolution only (exits before weights load)."""
    cmd = [str(a.probe_python), "-m", "mlx2.server", "--model", str(model), "--host", "127.0.0.1",
           "--port", str(a.port), "--ast"]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1"}
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
                              check=False)
    except subprocess.TimeoutExpired:
        return {"command": cmd, "verdict": "BUG: --ast did not fail closed within 120 s"}
    tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
    refused = proc.returncode != 0 and time.monotonic() - t0 < 60
    (out / "ast-probe.log").write_text(proc.stdout + proc.stderr)
    return {"command": cmd, "returncode": proc.returncode, "seconds": round(time.monotonic() - t0, 2),
            "message": tail[0], "verdict": "fails_closed" if refused else "BUG: --ast did not fail closed"}


# ----- cost estimate (budget guard and dry-run report) -----

def _interp(cell, table):
    keys = sorted(table)
    if cell <= keys[0]:
        return table[keys[0]]
    if cell >= keys[-1]:
        return table[keys[-1]]
    for lo, hi in itertools.pairwise(keys):
        if lo <= cell <= hi:
            t = (math.log2(cell) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
            a, b = table[lo], table[hi]
            if isinstance(a, tuple):
                return tuple(math.exp(math.log(x) + t * (math.log(y) - math.log(x))) for x, y in zip(a, b))
            return a + t * (b - a)
    raise AssertionError


def estimate_cell_s(cell, width, max_tokens, evidence_tokens=160):
    """Predicted wall seconds of one cell if every request decodes ``max_tokens`` (an upper bound)."""
    ttft, decode = _interp(cell, R1_B1)
    if width > 1:
        ttft, decode = ttft * R1_B4_TTFT, decode * _interp(cell, R1_B4_DECODE)
        # cold + warm (some lanes miss APCv2 at B4: ~0.6 of a cold prefill) + warm- and cold-shaped evidence
        return 3 + ttft + 0.6 * ttft + 2 * max_tokens / decode + (0.6 * ttft + ttft + 2 * evidence_tokens / decode)
    return 3 + ttft + 2 * max_tokens / decode  # width 1: warm carries the evidence windows


def phase_cells(phase, a):
    return {"full": a.cells, "b4": a.b4_cells, "ab": a.ab_cells}.get(phase.get("cells"), [])


def estimate_plan(phases, a):
    out, total = {}, 0.0
    for phase in phases:
        if phase.get("refusal"):
            cells, s = {}, REFUSAL_S
        else:
            cells = {c: round(estimate_cell_s(c, phase["width"], a.max_tokens)) for c in phase_cells(phase, a)}
            s = LOAD_S + sum(cells.values())
        out[phase["name"]] = {"cells_s": cells, "phase_s": round(s)}
        total += s
    return {"basis": "first-run (01d3b52a5) ordinary B1/B4 TTFT and decode rates; every request at "
                     "max_tokens (upper bound: an early EOS shortens it); DSpark assumed no faster",
            "phases": out, "total_s": round(total), "total_min": round(total / 60, 1)}


# ----- phases -----

def _token_file(pdir):
    token_file = pdir / "admin-token"
    fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    token = secrets.token_hex(24)
    os.write(fd, (token + "\n").encode())
    os.close(fd)
    return token_file, token


def _env(a, pdir):
    return {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "MEMWATCH_LIMIT_GB": str(a.memwatch_limit_gb),
            "MEMWATCH_SWAP_GROWTH_GB": str(a.memwatch_swap_growth_gb),
            "MEMWATCH_RECEIPT": str(pdir / "memwatch-stop.json")}


def run_refusal(a, phase, model, out, report, save):
    """DSpark at width > 1 must be refused at startup; no request is sent."""
    pdir = out / phase["name"]
    pdir.mkdir()
    token_file, _ = _token_file(pdir)
    policy_file = pdir / "policy.json"
    policy_file.write_text(json.dumps(_policy(phase)) + "\n")
    command = server_command(a, phase, model, token_file, policy_file, 4096 + 4096 + a.max_tokens)
    rec = {"name": phase["name"], "route": phase["route"], "width": phase["width"], "kind": "fail_closed_check",
           "execution_policy": _policy(phase), "server_command": command, "status": "running", "cells": []}
    report["phases"].append(rec)
    base = f"http://127.0.0.1:{a.port}"
    server = log = None
    try:
        log = open(pdir / "server.log", "w")  # noqa: SIM115 - closed in finally
        t0 = time.monotonic()
        server = subprocess.Popen(command, cwd=ROOT, env=_env(a, pdir), stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        healthy = False
        while server.poll() is None and time.monotonic() - t0 < a.load_timeout:
            try:
                healthy = H.get_json(base, "/health").get("status") == "ok"
            except (urllib.error.URLError, ConnectionError, OSError, ValueError):
                pass
            if healthy:
                break
            time.sleep(1)
        rec["seconds"] = round(time.monotonic() - t0, 1)
        log.flush()
        text = (pdir / "server.log").read_text(errors="replace")
        rec["returncode"] = server.poll()
        rec["message"] = next((line for line in reversed(text.splitlines()) if phase["refusal"] in line), None)
        if healthy:
            rec["verdict"] = "BUG: server became healthy with DSpark at width > 1"
        elif rec["returncode"] not in (None, 0) and rec["message"]:
            rec["verdict"] = "fails_closed"
        elif rec["returncode"] not in (None, 0):
            rec["verdict"] = "refused_without_the_expected_message"
        else:
            rec["verdict"] = "BUG: no refusal within the load timeout"
        rec["status"] = "completed"
    except Exception as error:  # noqa: BLE001
        rec["status"] = "failed"
        rec["error"] = f"{type(error).__name__}: {error}"
    finally:
        rec["server_stop"] = H.stop_group(server)
        if rec["server_stop"] != "stopped":
            rec["status"] = f"failed: server process group {server.pid} did not stop"
        if log is not None:
            log.close()
        token_file.unlink(missing_ok=True)
        save()
    print(f"[{phase['name']}] {rec.get('verdict')}: {rec.get('message')}", flush=True)
    return rec


def _finish_result(result, lane, arm, receipt):
    oc = oracle_channels(result, lane["code"])
    result["oracle_pass"], result["oracle_channel"] = oc["pass"], oc["channel"]
    result["prompt_tokens"] = receipt.get("prompt_tokens")
    result["cached_tokens"] = receipt.get("cached_tokens")
    result["route"] = receipt.get("route")
    controls = receipt.get("request_controls") or {}
    result["thinking_mode"] = (controls.get("thinking_control") or {}).get("mode")
    spec = G_speculation(receipt)
    result["speculation"] = spec
    result["acceptance"] = G_acceptance([spec], DSPARK_POLICY["num_draft"]) if spec else None
    result["receipt"] = receipt
    result["prefill_tok_s"] = (round(result["prompt_tokens"] / result["ttft_s"], 1)
                               if arm == "cold" and result["prompt_tokens"] and result["ttft_s"] else None)


def _evidence_record(result, shape, attempt, max_tokens):
    return {"shape": shape, "attempt": attempt, "max_tokens": max_tokens, "token_ids": result["token_ids"],
            "windows": result["windows"], "text": result["text"], "reasoning_text": result["reasoning_text"],
            "cached_tokens": result["receipt"].get("cached_tokens")}


def knob_engagement(phase, status):
    """Whether the adapter reports the phase's opt-in knobs (and counters) as engaged."""
    knobs = {k: v for k, v in _policy(phase).items() if k in KNOB_NAMES}
    if not knobs and not phase.get("counter") and not phase.get("status_flag"):
        return None, None
    policy = H.find_key(status, "kernel_policy") or {}
    counters = H.find_key(status, "kernel_counters") or {}
    engaged = all(policy.get(KNOB_NAMES[k]) == v for k, v in knobs.items())
    if phase.get("counter"):
        engaged = engaged and bool(counters.get(phase["counter"]))
    flag = None
    if phase.get("status_flag"):
        flag = H.find_key(status, phase["status_flag"])
        engaged = engaged and flag is True
    return engaged, {"kernel_policy": policy, "kernel_counters": counters,
                     **({phase["status_flag"]: flag} if phase.get("status_flag") else {})}


def run_cell(a, phase, base, token, lanes, row, ref_cell, cell_t0):
    width = phase["width"]
    row["apc_clear"] = H.post_json(base, "/v1/admin/apc/clear", {"reason": f"{phase['name']}-{row['cell']}"}, token)
    bodies = [{"messages": [{"role": "user", "content": lane["text"]}],
               "max_tokens": a.max_tokens, "temperature": 0, "seed": a.seed,
               "reasoning_effort": a.reasoning_effort, "stream": True, "logprobs": True}
              for lane in lanes]
    send = lambda body, t0: stream_one(base, body, timeout=a.cell_timeout, t0=t0)
    row["evidence"] = [[] for _ in lanes]
    for arm in ("cold", "warm"):
        carries = arm == "warm" and width == 1
        results = send_lanes(send, [{**b, "top_logprobs": EVIDENCE_TOP} for b in bodies] if carries else bodies)
        for i, (lane, result) in enumerate(zip(lanes, results)):
            receipt = result.pop("receipt")
            result["receipt"] = receipt
            if carries:
                row["evidence"][i].append(_evidence_record(result, "warm_timed", 0, a.max_tokens))
            result.pop("windows")
            _finish_result(result, lane, arm, receipt)
            result["evidence_bearing"] = carries
        row[arm] = results
        row[f"{arm}_aggregate"] = aggregate(results, [r["prompt_tokens"] for r in results])
        if phase["route"] == "dspark":
            specs = [r["speculation"] for r in results if r["speculation"]]
            row[f"{arm}_acceptance"] = G_acceptance(specs, DSPARK_POLICY["num_draft"]) if specs else None
        agg = row[f"{arm}_aggregate"]
        print(f"[{phase['name']} {row['cell']} {arm}] ttft_max={agg['ttft_max_s'] and round(agg['ttft_max_s'], 2)}s "
              f"decode_agg={agg['decode_tok_s_aggregate']} n={[len(r['token_ids']) for r in results]} "
              f"cached={[r['cached_tokens'] for r in results]} "
              f"oracle={[r['oracle_channel'] for r in results]} mode={results[0]['thinking_mode']} "
              f"MAL={(row.get(f'{arm}_acceptance') or {}).get('mean_acceptance_length')}", flush=True)
    needs = evidence_needs(row, width, ref_cell)
    row["evidence_needs"] = [list(n) for n in needs]
    attempts = []
    shapes = ["cold", "warm"] if width == 1 else ["warm", "cold", "warm"]
    for shape in shapes[:a.evidence_attempts]:
        unmet = unmet_needs(row, needs)
        if not unmet:
            break
        if time.monotonic() - cell_t0 > a.cell_cap_s:
            attempts.append({"shape": shape, "skipped": "cell cap"})
            break
        if shape == "cold" and not any(arm == "cold" for _, arm, _ in unmet):
            continue
        if shape == "cold" and row["cell"] > a.cold_evidence_max:
            attempts.append({"shape": shape, "skipped": f"cell above --cold-evidence-max {a.cold_evidence_max}"})
            continue
        if shape == "warm" and width == 1 and not any(arm == "warm" for _, arm, _ in unmet):
            continue  # a width-one warm replay reproduces the warm ids; it cannot bind a cold window
        cap = min(a.max_tokens, max(k for _, _, k in unmet) + 1)
        if shape == "cold":
            H.post_json(base, "/v1/admin/apc/clear", {"reason": f"{phase['name']}-{row['cell']}-evidence"}, token)
        t0 = time.monotonic()
        evidence = send_lanes(send, [{**b, "max_tokens": cap, "top_logprobs": EVIDENCE_TOP} for b in bodies])
        n = len(attempts) + 1
        for i, result in enumerate(evidence):
            row["evidence"][i].append(_evidence_record(result, shape, n, cap))
        attempts.append({"shape": shape, "max_tokens": cap, "seconds": round(time.monotonic() - t0, 1),
                         "unmet_before": [list(u) for u in unmet]})
    row["evidence_attempts"] = attempts
    row["evidence_unmet"] = [list(u) for u in unmet_needs(row, needs)]
    row["cold_warm"] = cold_warm(row, width)
    status = H.get_json(base, "/v1/status")
    row["knob_engaged"], row["knob_receipt"] = knob_engagement(phase, status)
    row["metal_peak_gb"] = round((H.find_key(status, "metal_peak_bytes") or 0) / 2**30, 2)
    row["physical_footprint_gb"] = round((H.find_key(status, "process_physical_footprint_bytes") or 0) / 2**30, 2)
    row["status_after"] = status


def run_phase(a, phase, model, cell_plan, out, report, save, references, clock):
    pdir = out / phase["name"]
    pdir.mkdir()
    token_file, token = _token_file(pdir)
    policy_file = pdir / "policy.json"
    policy_file.write_text(json.dumps(_policy(phase)) + "\n")
    wanted = set(phase_cells(phase, a))
    cells = [c for c in cell_plan if c["cell"] in wanted]
    max_context = max(c["cell"] for c in cells) + 4096 + a.max_tokens
    base = f"http://127.0.0.1:{a.port}"
    rec = {"name": phase["name"], "route": phase["route"], "width": phase["width"],
           "reference": phase["reference"],
           "expected_route": "ordinary" if phase["route"] == "ordinary" else "external_draft",
           "expected_thinking_mode": phase.get("thinking_mode", "template_effort"),
           "execution_policy": _policy(phase) or None,
           "label": ("ordinary decode" if phase["route"] == "ordinary" else
                     "DSpark external-draft chain (closest supported to AST; implemented, opt-in, unqualified)")
           + ("; opt-in knob arm (unqualified)" if set(_policy(phase)) & set(KNOB_NAMES) else ""),
           "cells": [], "status": "running", "swap_used_mb_start": H.swap_used_mb()}
    report["phases"].append(rec)
    command = server_command(a, phase, model, token_file, policy_file, max_context)
    rec["server_command"] = command
    smallest = LOAD_S + estimate_cell_s(cells[0]["cell"], phase["width"], a.max_tokens) * clock.factor()
    if clock.elapsed() + smallest > a.budget_min * 60:
        rec["status"] = f"budget_skipped: not started ({clock.elapsed() / 60:.1f} min used)"
        rec["cells"] = [{"cell": c["cell"], "skipped": "budget: phase not started"} for c in cells]
        token_file.unlink(missing_ok=True)
        rec["verdict"] = phase_verdict(rec)
        print(f"[{phase['name']}] SKIP {rec['status']}", flush=True)
        save()
        return rec
    server = log = None
    try:
        log = open(pdir / "server.log", "w")  # noqa: SIM115 - closed in finally
        t0 = time.monotonic()
        server = subprocess.Popen(command, cwd=ROOT, env=_env(a, pdir), stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"server exited {server.returncode} during load; see server.log")
            try:
                if H.get_json(base, "/health").get("status") == "ok":
                    break
            except (urllib.error.URLError, ConnectionError, OSError, ValueError):
                pass
            if time.monotonic() - t0 > a.load_timeout:
                raise TimeoutError("server not healthy within the load timeout")
            time.sleep(1)
        rec["load_seconds"] = round(time.monotonic() - t0, 1)
        status = H.get_json(base, "/v1/status")
        rec["server_status_start"] = status
        rec["checkpoint"] = H.checkpoint_label(model.name, status)
        save()
        ref_cells = {c["cell"]: c for c in (references.get(phase["reference"]) or {}).get("cells", ()) if c.get("cold")}
        for plan in cells:
            lanes = plan["lanes"][:phase["width"]]
            row = {"cell": plan["cell"], "lanes": [{k: v for k, v in lane.items() if k != "text"} for lane in lanes]}
            rec["cells"].append(row)
            predicted = estimate_cell_s(plan["cell"], phase["width"], a.max_tokens)
            corrected = predicted * clock.factor()
            row["predicted_s"], row["predicted_corrected_s"] = round(predicted), round(corrected)
            if clock.elapsed() + corrected > a.budget_min * 60:
                row["skipped"] = (f"budget: {clock.elapsed() / 60:.1f} min used + {corrected / 60:.1f} min "
                                  f"predicted > {a.budget_min} min")
                print(f"[{phase['name']} {plan['cell']}] SKIP {row['skipped']}", flush=True)
                save()
                continue
            cell_t0 = time.monotonic()
            try:
                run_cell(a, phase, base, token, lanes, row, ref_cells.get(plan["cell"]), cell_t0)
            except Exception as error:  # noqa: BLE001 - a cell that will not run stops this phase
                row["error"] = f"{type(error).__name__}: {error}"
                if isinstance(error, urllib.error.HTTPError):
                    try:
                        row["error_body"] = error.read().decode("utf-8", "replace")[:4000]
                    except Exception:  # noqa: BLE001
                        pass
                print(f"[{phase['name']} {plan['cell']}] ERROR {row['error']}", flush=True)
                rec["status"] = f"stopped at {plan['cell']}: cell did not run"
                row["wall_s"] = round(time.monotonic() - cell_t0, 1)
                row["swap_used_mb"] = H.swap_used_mb()
                save()
                break
            row["wall_s"] = round(time.monotonic() - cell_t0, 1)
            clock.record(predicted, row["wall_s"])
            row["swap_used_mb"] = H.swap_used_mb()
            save()
            if row["swap_used_mb"] - rec["swap_used_mb_start"] > a.swap_stop_mb:
                rec["status"] = f"stopped after {plan['cell']}: swap grew"
                break
            if row["wall_s"] > a.cell_cap_s:
                rec["status"] = f"stopped after {plan['cell']}: cell exceeded --cell-cap-s {a.cell_cap_s}"
                break
        else:
            skipped = [c["cell"] for c in rec["cells"] if c.get("skipped")]
            rec["status"] = "completed" if not skipped else f"budget_truncated: skipped {skipped}"
    except Exception as error:  # noqa: BLE001
        rec["status"] = "failed"
        rec["error"] = f"{type(error).__name__}: {error}"
    finally:
        rec["server_stop"] = H.stop_group(server)
        if rec["server_stop"] != "stopped":
            rec["status"] = f"failed: server process group {server.pid} did not stop"
        if log is not None:
            log.close()
        memwatch = pdir / "memwatch-stop.json"
        if memwatch.exists():
            rec["memwatch_stop"] = json.loads(memwatch.read_text())
        rec["swap_used_mb_end"] = H.swap_used_mb()
        token_file.unlink(missing_ok=True)
        rec["verdict"] = phase_verdict(rec)
        save()
    return rec


class Clock:
    """Wall time since the ladder started and the actual/predicted cost ratio so far."""

    def __init__(self):
        self.t0, self.predicted, self.actual = time.monotonic(), 0.0, 0.0

    def elapsed(self):
        return time.monotonic() - self.t0

    def record(self, predicted, actual):
        self.predicted += predicted
        self.actual += actual

    def factor(self):
        return min(3.0, max(0.25, self.actual / self.predicted)) if self.predicted > 0 else 1.0


def G_speculation(receipt):
    import spec_acceptance_gsm8k as G

    return G.speculation_receipt(receipt)


def G_acceptance(specs, depth):
    import spec_acceptance_gsm8k as G

    try:
        return G.acceptance_summary(specs, draft_depth=depth)
    except ValueError as error:
        return {"error": str(error)}


def build_plan(tokenizer, cells, widths):
    """``widths``: cell -> lanes needed; lane prompts depend only on (cell index, cell, lane)."""
    return [{"cell": cell, "lanes": lane_prompts(tokenizer, i, cell, widths[cell])} for i, cell in enumerate(cells)]


def _cells(text):
    return sorted({int(c) for c in text.split(",") if c})


def _int_window_keys(obj):
    """JSON turns logprob-window token ids into strings; live phases use int
    ids. A reloaded reference must match, or every window lookup misses and a
    near tie reads as a high-margin divergence."""
    if isinstance(obj, dict):
        if obj and all(isinstance(k, str) and k.lstrip("-").isdigit() for k in obj) and \
                all(isinstance(v, (int, float)) for v in obj.values()):
            return {int(k): v for k, v in obj.items()}
        return {k: _int_window_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_int_window_keys(v) for v in obj]
    return obj


def _terminate(signum, frame):
    # SIGTERM unwinds like an exception so every phase's ``finally`` stops its
    # server process group (a bare SIGTERM left a loopback server running).
    raise SystemExit(128 + signum)


def main(argv=None) -> int:
    import signal

    signal.signal(signal.SIGTERM, _terminate)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cells", default="1024,4096,16384,32768,65536,131072",
                        help="full ladder (ordinary_b1, dspark_b1)")
    parser.add_argument("--b4-cells", default="1024,4096",
                        help="cells for ordinary_b4 (16384 adds ~11 min at 1024 tokens: past the 75-min budget)")
    parser.add_argument("--ab-cells", default="1024,16384,65536", help="cells for the opt-in knob arms")
    parser.add_argument("--phases", default=",".join(p["name"] for p in PHASES))
    parser.add_argument("--port", type=int, default=8396)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--evidence-attempts", type=int, default=3,
                        help="untimed top-logprob attempts per cell when a divergence lacks a window")
    parser.add_argument("--cold-evidence-max", type=int, default=16384,
                        help="largest cell for a cold-shaped (APC cleared, full prefill) evidence attempt")
    parser.add_argument("--budget-min", type=float, default=75.0, help="skip cells predicted to end past this")
    parser.add_argument("--cell-cap-s", type=float, default=1200.0)
    parser.add_argument("--server-python", type=Path, default=ROOT / "scripts/research/memwatched_python",
                        help="interpreter for the server (default: the memory-watchdog wrapper)")
    parser.add_argument("--probe-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--memwatch-limit-gb", type=float, default=80.0)
    parser.add_argument("--memwatch-swap-growth-gb", type=float, default=2.0)
    parser.add_argument("--swap-stop-mb", type=float, default=2048.0)
    parser.add_argument("--load-timeout", type=float, default=900.0)
    parser.add_argument("--cell-timeout", type=float, default=900.0, help="per-request socket timeout")
    parser.add_argument("--dry-run", action="store_true", help="plan, estimate and --ast probe only; no server")
    parser.add_argument("--reference-ladder", type=Path,
                        help="reuse reference phases (e.g. ordinary_b1) from an earlier ladder.json instead of "
                             "re-running them; every planned lane's prompt_sha256 must match (pin the corpus "
                             "with MLX2_CAMPAIGN_ROOT=<that run's snapshot>)")
    a = parser.parse_args(argv)
    if a.port == 8282:
        parser.error("port 8282 is reserved for release trees")
    names = [n for n in a.phases.split(",") if n]
    unknown = set(names) - {p["name"] for p in PHASES}
    if unknown:
        parser.error(f"unknown phases: {sorted(unknown)}")
    phases = [p for p in PHASES if p["name"] in names]
    a.cells, a.b4_cells, a.ab_cells = _cells(a.cells), _cells(a.b4_cells), _cells(a.ab_cells)
    out = a.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error(f"{out} is not empty")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", a.port)) == 0:
            parser.error(f"port {a.port} is occupied")
    widths = {}
    for phase in phases:
        for cell in phase_cells(phase, a):
            widths[cell] = max(widths.get(cell, 1), phase["width"])
    cells = sorted(widths)
    model = a.model.expanduser().resolve(strict=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    plan = build_plan(tokenizer, cells, widths)
    estimate = estimate_plan(phases, a)
    report = {
        "schema": SCHEMA,
        "kind": "single-rep sanity/correctness ladder; no thermal gating; not publishable statistics",
        "source": {"commit": H.git("rev-parse", "HEAD"),
                   "tracked_dirty": bool(H.git("status", "--porcelain", "--untracked-files=no")),
                   "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "model_path": str(model),
        "spec_route_mapping": {
            "requested": "AST spec decode",
            "ran": "dspark (external draft, bundled embedded_dspark, num_draft 7, proposal_composition false)",
            "why": "Inkling 2 declares no AST tree tiles and no native MTP; --ast is refused (ast_probe)",
        },
        "phases_requested": [p["name"] for p in phases],
        "cells": {"full": a.cells, "b4": a.b4_cells, "ab": a.ab_cells},
        "max_tokens": a.max_tokens, "sampling": "greedy (temperature 0)", "seed": a.seed,
        "reasoning_effort": a.reasoning_effort, "evidence_top_logprobs": EVIDENCE_TOP,
        "evidence": {"width_one": "warm timed request carries top_logprobs (evidence-bearing)",
                     "attempts": a.evidence_attempts, "cold_evidence_max": a.cold_evidence_max},
        "oracle": "code in reasoning_content + content (Unicode dashes normalized); channel recorded",
        "budget": {"budget_min": a.budget_min, "cell_cap_s": a.cell_cap_s, "estimate": estimate},
        "plan": [{"cell": c["cell"], "lanes": [{k: v for k, v in lane.items() if k != "text"}
                                                for lane in c["lanes"]]} for c in plan],
        "started_at": datetime.now(UTC).isoformat(), "phases": [], "status": "running",
        "swap_used_mb_start": H.swap_used_mb(),
    }

    def save():
        tmp = out / "ladder.json.partial"
        tmp.write_text(json.dumps(report, indent=2, default=str) + "\n")
        os.replace(tmp, out / "ladder.json")

    report["ast_probe"] = ast_probe(a, model, out)
    print(f"[ast_probe] {report['ast_probe']['verdict']}: {report['ast_probe'].get('message')}", flush=True)
    if a.dry_run:
        report["dry_run"] = {}
        for p in phases:
            biggest = max(phase_cells(p, a), default=4096)
            report["dry_run"][p["name"]] = server_command(a, p, model, out / p["name"] / "admin-token",
                                                          out / p["name"] / "policy.json",
                                                          biggest + 4096 + a.max_tokens)
        report["status"] = "dry_run"
        save()
        print(json.dumps({"status": "dry_run", "out": str(out),
                          "cells": [(c["cell"], [l["prompt_tokens_estimate"] for l in c["lanes"]]) for c in plan],
                          "estimate_min": estimate["total_min"],
                          "phase_s": {k: v["phase_s"] for k, v in estimate["phases"].items()}}))
        return 0
    save()
    by_name, clock = {}, Clock()
    if a.reference_ladder:
        prior = json.loads(Path(a.reference_ladder).expanduser().read_text())
        wanted = {p["reference"] for p in phases if p.get("reference")} - {p["name"] for p in phases}
        for name in sorted(wanted):
            ref = next((p for p in prior.get("phases", ()) if p["name"] == name), None)
            if ref is None or ref.get("status") != "completed":
                raise SystemExit(f"--reference-ladder has no completed {name} phase")
            ref_cells = {c["cell"]: c for c in ref.get("cells", ())}
            for row in plan:
                old = ref_cells.get(row["cell"])
                if old is None or [l["prompt_sha256"] for l in old["lanes"]] != \
                        [l["prompt_sha256"] for l in row["lanes"]][:len(old["lanes"])]:
                    raise SystemExit(f"--reference-ladder {name} cell {row['cell']}: prompts differ "
                                     "(pin MLX2_CAMPAIGN_ROOT to that run's snapshot)")
            by_name[name] = _int_window_keys(ref)
        report["reference_ladder"] = {"path": str(a.reference_ladder), "phases": sorted(wanted),
                                      "source": prior.get("source"), "checkpoints": prior.get("checkpoints"),
                                      "campaign_root": os.environ.get("MLX2_CAMPAIGN_ROOT")}
        save()
    try:
        for phase in phases:
            if phase.get("refusal"):
                by_name[phase["name"]] = run_refusal(a, phase, model, out, report, save)
            else:
                by_name[phase["name"]] = run_phase(a, phase, model, plan, out, report, save, by_name, clock)
        report["comparisons"] = [compare_phases(by_name[p["reference"]], by_name[p["name"]])
                                 for p in phases if p.get("reference") in by_name and p["name"] in by_name]
        checkpoints = {p["name"]: p.get("checkpoint") for p in report["phases"] if p.get("checkpoint")}
        report["checkpoints"] = checkpoints
        report["checkpoint_note"] = ("a DSpark phase's adapter checkpoint is the route-bound (target + drafter) "
                                     "identity; the target checkpoint is the ordinary phases' label")
        bad = [p["name"] for p in report["phases"]
               if p["status"] != "completed" or (p.get("kind") == "fail_closed_check" and p.get("verdict") != "fails_closed")]
        report["status"] = "completed" if not bad else "partial: " + ", ".join(bad)
    except Exception as error:  # noqa: BLE001
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        report["swap_used_mb_end"] = H.swap_used_mb()
        report["finished_at"] = datetime.now(UTC).isoformat()
        report["gpu_wall_min"] = round(clock.elapsed() / 60, 1)
        save()
    print(json.dumps({"status": report["status"],
                      "checkpoints": sorted({p.get("checkpoint") or "?" for p in report["phases"]}),
                      "commit": report["source"]["commit"], "out": str(out),
                      "gpu_wall_min": report["gpu_wall_min"]}))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
