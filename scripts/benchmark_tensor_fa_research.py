#!/usr/bin/env python3
"""Controlled fair paired timing for the approximate tensor-API attention research candidate.

Scope: attention-entry-region timing (dispatch + completion, NOT pure GPU
kernel duration) of ``mlx2.adapters.tensor_fa_research`` against
ordinary ``mx.fast.scaled_dot_product_attention`` on a predeclared synthetic
catalogue (B1, float16 Q, materialized float16 KV, native GQA Hq12/Hkv2,
D128/256, L32/65, N1024/8192, causal final window), plus a separately
reported KV conversion/materialization experiment. It is NOT model, cache,
continuation, acceptance, serving, sparse/QSA or production qualification and
never a whole-model gain; ``qualified``/``selected``/``model_gain`` are always
false. The qualifier's ``--timing`` refusal is unchanged; this is a separate
script.

Gate: timing runs only after the FULL mandatory numeric qualifier
(``q.run_gate`` on ``q.NativeBackend`` as in ``q._run_native``, but under the
combined qualifier+timing guard, then ``q.native_verdict``) has passed live in
THIS process on the same admitted source and MLX build. No external JSON, report or pass dict
is accepted as a gate; if the live gate fails, nothing is timed.

Evidence vs execution: ``evaluate_timing()`` judges a report and can never
establish native timing. Only the live in-process run object built by
``_run_native_timing`` (real ``NativeTimingOps`` built from the live passing
gate run) can stamp ``native_timing_execution`` in ``timing_verdict``; a JSON
round trip drops it. A trust boundary against forged reports and fake
backends, not a security claim against code that monkeypatches this module.

Fairness (predeclared, see ``FAIRNESS``):
  * both arms get the SAME pre-evaluated device q/k/v (float16, head-major,
    KV un-tiled) and the same visible keys: candidate ``causal=True,
    q_start=N-L``; baseline ``mask="causal"`` (MLX lower-right alignment =
    final window). Every output is checked against the float64 visible-key
    reference, so a mask mismatch fails closed;
  * identical timed boundary: clock read, fresh arm invocation (new lazy
    graph and output every repeat), ``mx.eval`` of its outputs,
    ``mx.synchronize()``, clock read. Inputs, references, guards, counter
    reads, engagement readback, host copies, hashing, fidelity checks and
    logging are outside. The candidate is the INSTRUMENTED kernel: its
    per-threadgroup atomic engagement counter and two tiny uniform arrays are
    inside its region; host plan admission is inside it too;
  * kernel compilation happened in the numeric gate; each cell also runs one
    untimed witness per arm and ``WARMUP_PAIRS`` checked warm-up pairs that
    are not reported as samples. Cold compilation is excluded;
  * ``PAIRS`` pairs per cell, alternating order (balanced), raw samples and
    per-pair candidate/baseline ratios reported; medians are per cell only.
    Thermal state is not controlled; order-split medians expose drift.

Every timed (and warm-up/witness) output is read back after its timer and
bound: dtype/shape/finite, full sha256, fidelity to the host float64
reference (qualifier host tolerance), to the cell witness and, for the
candidate, to the same pair's actual baseline output (qualifier native
tolerance), plus confirmed engagement with the planned threadgroup count and
issued/confirmed counter deltas of exactly 1. Any failure stops the run.

Admission (before any MLX import) reuses ``q.native_admission`` and also
requires this script, its test and provenance committed at HEAD with
identical working bytes. Before EVERY dispatch and after every repeat, cell
and the run: admitted HEAD, clean src/scripts/tests/provenance, bound
qualifier+timing file hashes, MLX build files and module paths. Cleanup runs
in ``finally``. Nothing imports MLX at module import, ``--help``,
``--catalogue`` or admission. ``--i-own-the-gpu`` is an acknowledgement, not
ownership: the parent wrapper owns the CPG lease and both flocks.

  # future only, after the user GPU hold, with these files committed:
  MLX2_INTAKE_SOURCE_ROOT=<this checkout> MLX2_INTAKE_SOURCE_COMMIT=<full HEAD sha> \\
  MLX2_INTAKE_CPG_WORKFLOW=<fresh live workflow> MLX2_INTAKE_CPG_TASK=<fresh native task> \\
  python /tmp/mlx2-intake/stage3_gpu.py tensor-fa-timing \\
      ~/Desktop/mlx2/.venv/bin/python scripts/benchmark_tensor_fa_research.py \\
      --run-native --i-own-the-gpu --source-root <this checkout> --source-commit <full HEAD sha> \\
      --out <new receipt>.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import qualify_tensor_fa_research as q  # no MLX import

SCHEMA = "mlx2.native.tensor-fa-fair-timing.v1"
SCOPE = ("controlled paired attention-entry-region timing (instrumented; not pure GPU kernel duration) of the approximate tensor-API attention research candidate vs "
         "ordinary native-GQA float16 SDPA on a synthetic B1 catalogue, plus a separately reported synthetic KV "
         "conversion/materialization cost; not model, cache, continuation, acceptance, serving, sparse/QSA or "
         "production qualification and never an end-to-end or model gain")
THIS_SCRIPT = "scripts/benchmark_tensor_fa_research.py"
QUALIFIER_SCRIPT = "scripts/qualify_tensor_fa_research.py"
TIMING_FILES = (THIS_SCRIPT, "tests/test_tensor_fa_research_timing.py", "provenance/tensor-fa-timing.json")
CAND, BASE = "candidate", "baseline"
CONVERT, MATERIALIZE = "convert_f32_to_f16", "materialize_strided_f16"
WARMUP_PAIRS, PAIRS = 3, 24                 # PAIRS even: alternating order is exactly balanced
CONV_WARMUP, CONV_REPEATS = 2, 16
ARM_DTYPE = {CAND: "float32", BASE: "float16"}
HQ, HKV = 12, 2


def _baseline_meta(cell):
    return {**q.BASELINE, "kv_heads": cell["hkv"], "q_heads": cell["hq"], "q_dtype": "float16",
            "mask": "causal (MLX lower-right alignment = final window)"}


def _cell(dim, length, kv_len):
    cid = f"tf-d{dim}-g6-l{length}-n{kv_len}-final"
    cell = dict(q._case(cid, dim, "float16", HQ, HKV, length, kv_len, True, kv_len - length, "random"))
    del cell["expected_memory_bytes"]        # the qualifier's numeric-cell estimate; timing has its own below
    kv = 2 * HKV * kv_len * dim
    cell["conversion"] = length == 32
    cell["device_bytes"] = {                 # resident at once on the device (one cell, one output at a time)
        "kernel_phase": HQ * length * dim * 2 + kv * 2 + HQ * length * dim * 4 + 4,
        "conversion_phase": (kv * 4 + kv * 2 + kv * 2 + kv * 2) if cell["conversion"] else 0}
    return cell


CELLS = tuple(_cell(d, length, n) for d in (128, 256) for n in (1024, 8192) for length in (32, 65))
FAIRNESS = {
    "inputs": "same pre-evaluated device q (float16) and k/v (float16, head-major contiguous, KV un-tiled) for both "
              "arms, uploaded and evaluated outside the timer",
    "visible_keys": "candidate causal=True q_start=N-L; baseline mask='causal' (lower-right); verified per output "
                    "against the float64 visible-key reference",
    "timed_region": "clock; fresh arm invocation; mx.eval(outputs); mx.synchronize(); clock (identical both arms). "
                    "Measures attention-entry invocation + completion (host op construction, dispatch, GPU execution, "
                    "synchronization), NOT pure GPU kernel duration",
    "candidate_scope": "instrumented candidate: per-threadgroup atomic engagement counter, two tiny uniform arrays "
                       "and host plan admission are inside its region; uninstrumented kernel time is not measured",
    "excluded": "input upload/eval, host references, source/build guards, counter reads, engagement readback, host "
                "copies, hashing, fidelity checks, memory probes, logging, compilation (done in the numeric gate), "
                "witness and warm-up dispatches",
    "schedule": f"{WARMUP_PAIRS} checked warm-up pairs then {PAIRS} pairs; pair i runs candidate first iff i is even",
    "not_used": "candidate.reference_attention / host_mirror (fp32-expanded, KV-tiled) and any float32-query arm",
    "thermal": "not controlled; order-split medians are reported to expose drift",
}
CONVERSION_SCOPE = {
    CONVERT: "synthetic float32 KV holding the cell's float16-representable values -> astype(float16); head-major "
             "contiguous [1,Hkv,N,D] in and out; output must equal the cell's float16 KV bit-exactly",
    MATERIALIZE: "float16 KV stored token-major [1,N,Hkv,D] -> transpose(0,2,1,3) view -> mx.contiguous (explicit "
                 "copy) -> head-major [1,Hkv,N,D]; values must equal the cell's float16 KV bit-exactly (values, not "
                 "layout, are verified: MLX exposes no public strides)",
    "applies_to": "both arms equally: candidate and baseline consume the same materialized float16 head-major KV, so "
                  "this is a shared cost that cannot change the candidate/baseline kernel ratio",
    "not": ["quantized KV cache conversion or dequantization", "the served route or any real cache layout",
            "sparse/QSA selection", "end-to-end or model gain",
            "whatever copy MLX may perform inside either arm for a strided input (both timed arms get contiguous KV)"],
    "combination": "reported independently; never summed with kernel medians into an end-to-end claim; no pipeline "
                   "arm is implemented",
    "memory": "allocator peak across the untimed conversion warm-ups (resident sources included)",
}


class Refused(q.Refused):
    """Fail closed: never a pass."""


def pair_order(i):
    return (CAND, BASE) if i % 2 == 0 else (BASE, CAND)


def _sha(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()


# ================================================================ checks (pure)

def _tol_problem(metrics, key, tol):
    m = metrics.get(key) if isinstance(metrics, dict) else None
    if not isinstance(m, dict) or not all(isinstance(m.get(k), float) for k in tol):
        return f"{key} metrics missing"
    return None if q._within(m, tol) else f"{key} outside tolerance {m}"


def output_problems(cell, rec, *, witness_required=True):
    """Per-output binding (also re-run by the evaluator on every record)."""
    if not isinstance(rec, dict) or rec.get("arm") not in ARM_DTYPE:
        return ["record or arm missing"]
    arm = rec["arm"]
    shape = [1, cell["hq"], cell["length"], cell["dim"]]
    out = rec.get("output") or {}
    problems = []
    if rec.get("array_problem") is not None:
        problems.append(f"output {rec['array_problem']}")
    if out.get("dtype") != ARM_DTYPE[arm] or out.get("shape") != shape or not q.SHA.fullmatch(str(out.get("sha256"))):
        problems.append("output dtype/shape/hash binding missing or wrong")
    metrics = rec.get("metrics")
    problems.append(_tol_problem(metrics, "vs_host_f64", q.TOLERANCE["host_f64"]))
    if witness_required:
        problems.append(_tol_problem(metrics, "vs_witness", q.TOLERANCE["native_f16"]))
    if arm == CAND:
        plan, eng, want = rec.get("plan") or {}, rec.get("engagement") or {}, cell["expected_threadgroups"]
        if plan.get("threadgroups") != want or plan.get("kv_end") != list(cell["kv_end"]):
            problems.append("plan threadgroups/kv_end differ from the catalogue law")
        if not (want > 0 and eng.get("confirmed") is True and eng.get("observed_threadgroups") == want
                and eng.get("expected_threadgroups") == want
                and eng.get("issued_delta") == 1 and eng.get("confirmed_delta") == 1):
            problems.append("engagement not confirmed with the expected threadgroups and counter deltas")
    return [p for p in problems if p]


def pair_problems(rec):
    p = _tol_problem(rec.get("metrics"), "vs_pair_baseline", q.TOLERANCE["native_f16"])
    return [p] if p else []


def _cell_problems(cell, rec):
    problems = []
    if rec.get("cell") != cell:
        problems.append("cell parameters differ from the predeclared catalogue")
    inputs = q.host_inputs(cell)
    if rec.get("input_hashes") != q.input_hashes(inputs):
        problems.append("input identity hashes differ from the catalogue law")
    if rec.get("baseline_meta") != _baseline_meta(cell):
        problems.append("baseline is not ordinary native-GQA float16 causal sdpa on the same heads")
    want = {"q": ["float16", [1, cell["hq"], cell["length"], cell["dim"]]],
            "k": ["float16", [1, cell["hkv"], cell["kv_len"], cell["dim"]]],
            "v": ["float16", [1, cell["hkv"], cell["kv_len"], cell["dim"]]]}
    if rec.get("device_inputs") != want:
        problems.append("device inputs are not float16 q and un-tiled float16 KV of the cell shape")
    witness = rec.get("witness") or {}
    for arm in (BASE, CAND):
        problems += [f"witness {arm}: {p}" for p in output_problems(cell, witness.get(arm), witness_required=False)]
    w_cand = witness.get(CAND) or {}
    p = _tol_problem(w_cand.get("metrics"), "vs_baseline_witness", q.TOLERANCE["native_f16"])
    if p:
        problems.append(f"witness candidate: {p}")
    for phase, count in (("warmup", WARMUP_PAIRS), ("samples", PAIRS)):
        problems += [f"{phase}: {p}" for p in _schedule_problems(cell, rec.get(phase), count)]
    if cell["conversion"]:
        problems += [f"conversion: {p}" for p in _conversion_problems(cell, inputs, rec.get("conversion"))]
    elif "conversion" in rec:
        problems.append("conversion record on a cell without a declared conversion experiment")
    return problems


def _schedule_problems(cell, samples, count):
    if not isinstance(samples, list):
        return ["samples missing"]
    expected = [(i, pos, arm) for i in range(count) for pos, arm in enumerate(pair_order(i))]
    if [(s.get("pair"), s.get("position"), s.get("arm")) if isinstance(s, dict) else None
            for s in samples] != expected:
        return ["samples missing, duplicated or out of the predeclared alternating order"]
    problems = []
    serials = {CAND: [], BASE: []}
    for s in samples:
        if type(s.get("ns")) is not int or s["ns"] <= 0:
            problems.append(f"pair {s['pair']} {s['arm']}: no positive integer sample")
        serials[s["arm"]].append(s.get("serial"))
        problems += [f"pair {s['pair']} {s['arm']}: {p}" for p in output_problems(cell, s)]
        if s["arm"] == CAND:
            problems += [f"pair {s['pair']} candidate: {p}" for p in pair_problems(s)]
    every = serials[CAND] + serials[BASE]
    if not all(type(x) is int for x in every) or len(set(every)) != len(every) or \
            any(a >= b for arm in serials for a, b in zip(serials[arm], serials[arm][1:])):
        problems.append("invocation serials not fresh (missing, reused or not increasing)")
    return problems


def _conversion_problems(cell, inputs, rec):
    if not isinstance(rec, dict):
        return ["conversion record missing"]
    problems = []
    want = {"k": _sha(inputs["k"]), "v": _sha(inputs["v"])}
    shape = [1, cell["hkv"], cell["kv_len"], cell["dim"]]
    if rec.get("scope") != CONVERSION_SCOPE:
        problems.append("conversion scope labels differ from the predeclared scope")
    exps = rec.get("experiments") or {}
    for name in (CONVERT, MATERIALIZE):
        exp = exps.get(name)
        if name == MATERIALIZE and isinstance(exp, dict) and exp.get("status") == "deferred":
            if not exp.get("reason") or "samples" in exp:
                problems.append(f"{name} deferred without a reason or with samples")
            continue
        samples = exp.get("samples") if isinstance(exp, dict) else None
        if not isinstance(samples, list) or [s.get("repeat") for s in samples] != list(range(CONV_REPEATS)):
            problems.append(f"{name}: samples missing, duplicated or out of order")
            continue
        for s in samples:
            if type(s.get("ns")) is not int or s["ns"] <= 0:
                problems.append(f"{name} repeat {s['repeat']}: no positive integer sample")
            if s.get("hashes") != want or s.get("dtype") != "float16" or s.get("shape") != shape:
                problems.append(f"{name} repeat {s['repeat']}: output is not the cell's float16 KV")
    return problems


def _summary(cell, rec):
    pairs = {}
    for s in rec["samples"]:
        pairs.setdefault(s["pair"], {})[s["arm"]] = s["ns"]
    cand = [pairs[i][CAND] for i in sorted(pairs)]
    base = [pairs[i][BASE] for i in sorted(pairs)]
    ratio = [c / b for c, b in zip(cand, base)]
    out = {"candidate_ns": cand, "baseline_ns": base, "pair_ratio": ratio,
           "median_pair_ratio": statistics.median(ratio),
           "median_candidate_ns": statistics.median(cand), "median_baseline_ns": statistics.median(base),
           "median_ratio_candidate_first": statistics.median(ratio[0::2]),
           "median_ratio_baseline_first": statistics.median(ratio[1::2]),
           "ratio_definition": "candidate_ns / baseline_ns of the same pair (<1: candidate region faster); "
                               "synthetic kernel region only, no model or end-to-end meaning"}
    conv = rec.get("conversion")
    if conv:
        out["conversion"] = {name: ({"ns": [s["ns"] for s in exp["samples"]],
                                     "median_ns": statistics.median(s["ns"] for s in exp["samples"])}
                                    if "samples" in exp else {"status": "deferred", "reason": exp["reason"]})
                             for name, exp in conv["experiments"].items()}
        out["conversion"]["bytes"] = conv.get("bytes")
    return out


def evaluate_timing(report):
    """Verdict over a timing report: evidence only, NEVER native execution."""
    refusals, summary = [], {}
    records = report.get("cells") if isinstance(report, dict) else None
    if not isinstance(records, list):
        records, refusals = [], ["no cell evidence"]
    ids = [r.get("id") if isinstance(r, dict) else None for r in records]
    if ids != [c["id"] for c in CELLS]:
        refusals.append("cells missing, duplicated, unexpected or out of order (the full catalogue is mandatory)")
    if not isinstance(report, dict) or report.get("fairness") != FAIRNESS:
        refusals.append("fairness declaration differs from the predeclared one")
    table = {c["id"]: c for c in CELLS}
    for rec in records:
        cell = table.get(rec.get("id")) if isinstance(rec, dict) else None
        if cell is None:
            continue
        problems = _cell_problems(cell, rec)
        refusals += [f"{cell['id']}: {p}" for p in problems]
        if not problems:
            summary[cell["id"]] = _summary(cell, rec)
    if refusals:
        summary = {}
    return {"verdict": "refused" if refusals else "timing_evidence_complete", "refusals": refusals,
            "summary": summary, "native_timing_execution": False, "qualified": False, "selected": False,
            "model_gain": False, "scope": SCOPE,
            "evidence_label": "evaluation of supplied timing evidence; does not establish native timing execution"}


# ================================================================ orchestration (backend-agnostic)

def _record(cell, arm, host, ref, witness, engagement, plan):
    import numpy as np

    bad = q._array_ok(host, ARM_DTYPE[arm], (1, cell["hq"], cell["length"], cell["dim"]))
    rec = {"arm": arm, "array_problem": bad, "metrics": {},
           "output": ({"dtype": str(host.dtype), "shape": list(host.shape), "sha256": _sha(host)}
                      if isinstance(host, np.ndarray) else {})}
    if bad is None:
        rec["metrics"]["vs_host_f64"] = q._metrics(host, ref)
        if witness is not None:
            rec["metrics"]["vs_witness"] = q._metrics(host, witness)
            rec["bitwise_witness"] = rec["output"]["sha256"] == _sha(witness)
    if arm == CAND:
        rec["engagement"] = engagement
        rec["plan"] = {"threadgroups": getattr(plan, "threadgroups", None),
                       "kv_end": list(getattr(plan, "kv_end", ()) or ())}
    return rec


def _one(ops, prep, arm, clock, guard):
    """One fresh invocation. Only dispatch + eval + synchronize sit between the two clock reads."""
    guard()                                          # before EVERY dispatch
    before = ops.counters() if arm == CAND else None
    t0 = clock()
    d = ops.dispatch(arm, prep)
    ops.complete(d)
    t1 = clock()
    engagement = None
    if arm == CAND:
        observed, confirmed = ops.engagement(d)      # GPU readback, after the timer
        after = ops.counters()
        engagement = {"expected_threadgroups": getattr(d.plan, "threadgroups", None),
                      "observed_threadgroups": observed, "confirmed": confirmed,
                      "issued_delta": after["issued"] - before["issued"],
                      "confirmed_delta": after["confirmed"] - before["confirmed"]}
    host = ops.to_host(d)[0]                         # drops the device outputs
    return d, t1 - t0, host, engagement


def _check(cell, where, problems):
    if problems:
        raise Refused(f"{cell['id']} {where}: {'; '.join(problems)}")


def _witness(ops, cell, prep, ref, guard):
    recs, hosts, memory = {}, {}, {}
    for arm in (BASE, CAND):                         # baseline first: the candidate witness is checked against it
        memory[arm] = dict(ops.memory_reset())
        d, _ns, host, eng = _one(ops, prep, arm, lambda: 0, guard)
        memory[arm].update(ops.memory_peak())
        recs[arm] = _record(cell, arm, host, ref, None, eng, d.plan)
        if arm == CAND and recs[arm]["array_problem"] is None:
            recs[arm]["metrics"]["vs_baseline_witness"] = q._metrics(host, hosts[BASE])
        _check(cell, f"witness {arm}", output_problems(cell, recs[arm], witness_required=False))
        hosts[arm] = host
        guard()
    _check(cell, "witness candidate", [p for p in [_tol_problem(recs[CAND]["metrics"], "vs_baseline_witness",
                                                                q.TOLERANCE["native_f16"])] if p])
    return hosts, recs, memory


def _pairs(ops, cell, prep, ref, witness, guard, clock, count, serials):
    samples = []
    for i in range(count):
        recs, hosts = {}, {}
        for pos, arm in enumerate(pair_order(i)):
            d, ns, host, eng = _one(ops, prep, arm, clock, guard)
            if type(d.serial) is not int or (serials and d.serial <= serials[-1]):
                raise Refused(f"{cell['id']} pair {i} {arm}: invocation serial not fresh")
            serials.append(d.serial)
            rec = _record(cell, arm, host, ref, witness[arm], eng, d.plan)
            rec.update(pair=i, position=pos, ns=ns, serial=d.serial)
            _check(cell, f"pair {i} {arm}", output_problems(cell, rec))
            recs[arm], hosts[arm] = rec, host
        recs[CAND]["metrics"]["vs_pair_baseline"] = q._metrics(hosts[CAND], hosts[BASE])
        _check(cell, f"pair {i} candidate", pair_problems(recs[CAND]))
        guard()                                      # after every repeat
        samples += [recs[arm] for arm in pair_order(i)]
    return samples


def _conversion(ops, cell, inputs, guard, clock):
    want = {"k": _sha(inputs["k"]), "v": _sha(inputs["v"])}
    shape = [1, cell["hkv"], cell["kv_len"], cell["dim"]]
    kv = 2 * cell["hkv"] * cell["kv_len"] * cell["dim"]
    src = None
    try:
        guard()
        src = ops.prepare_conversion(cell, inputs)
        deferred = ops.materialize_deferred()
        names = [CONVERT] if deferred else [CONVERT, MATERIALIZE]
        samples = {n: [] for n in names}
        memory = dict(ops.memory_reset())
        for i in range(CONV_WARMUP + CONV_REPEATS):
            if i == CONV_WARMUP:
                memory.update(ops.memory_peak())
            for name in (names if i % 2 == 0 else names[::-1]):
                guard()
                t0 = clock()
                d = ops.dispatch(name, src)
                ops.complete(d)
                t1 = clock()
                outs = ops.to_host(d)
                got = {"k": _sha(outs[0]), "v": _sha(outs[1])}
                ok = (got == want and all(str(o.dtype) == "float16" and list(o.shape) == shape for o in outs))
                _check(cell, f"{name} repeat {i}", [] if ok else ["conversion output is not the cell's float16 KV"])
                if i >= CONV_WARMUP:
                    samples[name].append({"repeat": i - CONV_WARMUP, "ns": t1 - t0, "serial": d.serial,
                                          "hashes": got, "dtype": "float16", "shape": shape})
            guard()
    finally:
        if src is not None:
            ops.release_cell(src)
    experiments = {n: {"samples": s} for n, s in samples.items()}
    if deferred:
        experiments[MATERIALIZE] = {"status": "deferred", "reason": deferred}
    return {"scope": CONVERSION_SCOPE, "experiments": experiments, "memory": memory,
            "bytes": {CONVERT: {"read_float32": kv * 4, "write_float16": kv * 2},
                      MATERIALIZE: {"read_float16": kv * 2, "write_float16": kv * 2}}}


def _run_cell(ops, cell, guard, clock):
    inputs = q.host_inputs(cell)
    reason = q.precheck(cell, inputs)
    if reason is not None:
        raise Refused(f"{cell['id']}: {reason}")
    ref = q.host_reference(cell, inputs)
    rec = {"id": cell["id"], "cell": cell, "input_hashes": q.input_hashes(inputs),
           "baseline_meta": _baseline_meta(cell)}
    prep = None
    try:
        guard()
        prep = ops.prepare(cell, inputs)
        rec["device_inputs"] = ops.describe(prep)
        witness, rec["witness"], rec["memory"] = _witness(ops, cell, prep, ref, guard)
        serials = []
        rec["warmup"] = _pairs(ops, cell, prep, ref, witness, guard, clock, WARMUP_PAIRS, serials)
        rec["samples"] = _pairs(ops, cell, prep, ref, witness, guard, clock, PAIRS, serials)
    finally:
        if prep is not None:
            ops.release_cell(prep)
    guard()
    if cell["conversion"]:
        rec["conversion"] = _conversion(ops, cell, inputs, guard, clock)
    return rec


def run_timing(ops, cells=CELLS, *, guard, clock):
    """One cell at a time; fail-stop on any binding failure; cleanup in finally.

    The returned report is EVIDENCE: it carries no execution status.
    """
    records = []
    try:
        for cell in cells:
            records.append(_run_cell(ops, cell, guard, clock))
            guard()
    finally:
        ops.release()
    guard()
    return {"schema": SCHEMA, "producer": type(ops).__name__, "fairness": FAIRNESS, "cells": records,
            "identity": ops.identity()}


# ================================================================ admission and guards (no MLX)

def timing_hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in TIMING_FILES if (ROOT / p).exists()}


def _module_paths_refusal(paths=None):
    if Path(q.__file__).resolve() != ROOT / QUALIFIER_SCRIPT or Path(__file__).resolve() != ROOT / THIS_SCRIPT:
        return "qualifier or timing script was not loaded from this checkout"
    if paths is not None:
        if Path(paths["candidate"]).resolve() != ROOT / q.CANDIDATE_MODULE:
            return "candidate module path is not this checkout's"
        if str(Path(paths["mx"]).resolve()) not in paths["mlx_files_before"]:
            return "mlx.core path is not the admitted extension"
    return None


def timing_admission(args, environ):
    """q.native_admission (HEAD, clean tree, qualifier files, MLX build, --out) plus the timing files at HEAD."""
    admission = q.native_admission(args, environ)
    expected = {}
    for path in TIMING_FILES:
        blob = q._git("show", f"HEAD:{path}")
        if blob is None:
            raise Refused(f"{path} is not committed at HEAD")
        expected[path] = hashlib.sha256(blob).hexdigest()
    if timing_hashes() != expected:
        raise Refused("working timing files differ from HEAD blobs")
    refusal = _module_paths_refusal()
    if refusal:
        raise Refused(refusal)
    return {**admission, "timing_expected": expected}


def timing_guard(admission, paths=None):
    """Admitted HEAD, clean tree, qualifier+timing hashes, MLX build and module paths (raises to stop).

    ``paths`` (candidate and mlx.core files) is None only before the gate backend has imported MLX.
    """
    q.source_guard(admission)
    if not q.same_snapshot(timing_hashes(), admission.get("timing_expected")):
        raise Refused("bound timing files changed since admission")
    refusal = _module_paths_refusal(None if paths is None else {**paths,
                                                                "mlx_files_before": admission["mlx_files_before"]})
    if refusal:
        raise Refused(refusal)


# ================================================================ native (never constructed in CPU tests)

@dataclass
class Dispatch:
    arm: str
    outputs: tuple[Any, ...] | None
    engaged: Any
    plan: Any
    serial: int


class NativeTimingOps:
    """Real Metal ops, built ONLY from a live in-process numeric gate run that passed."""

    def __init__(self, gate_run):
        if type(gate_run) is not q._LiveNativeRun or getattr(gate_run, "_witness", None) is not q._WITNESS:
            raise Refused("timing ops need the live in-process numeric gate run")
        if not q.native_verdict(gate_run)["native_synthetic_gate"]:
            raise Refused("live numeric gate did not pass")
        self.gate_run = gate_run
        self.mx, self.cand = gate_run.backend.mx, gate_run.backend.cand
        self._serial = itertools.count(1)

    def identity(self):
        return {**self.gate_run.backend.identity(), "timing_script": str(Path(__file__).resolve())}

    def module_paths(self):
        return {"candidate": self.cand.__file__, "mx": self.mx.__file__}

    def prepare(self, cell, inputs):
        arrays = {n: self.mx.array(inputs[n]) for n in ("q", "k", "v")}
        self.mx.eval(list(arrays.values()))
        return {"cell": cell, **arrays}

    def describe(self, prep):
        return {n: [str(prep[n].dtype).replace("mlx.core.", ""), list(prep[n].shape)] for n in ("q", "k", "v")}

    def prepare_conversion(self, cell, inputs):
        import numpy as np

        mx = self.mx
        src = {"k32": mx.array(inputs["k"].astype(np.float32)), "v32": mx.array(inputs["v"].astype(np.float32)),
               "k_tok": mx.array(np.ascontiguousarray(inputs["k"].transpose(0, 2, 1, 3))),
               "v_tok": mx.array(np.ascontiguousarray(inputs["v"].transpose(0, 2, 1, 3)))}
        mx.eval(list(src.values()))
        return src

    def materialize_deferred(self):
        return None if hasattr(self.mx, "contiguous") else "mx.contiguous unavailable in this MLX build"

    def dispatch(self, arm, prep):
        mx, serial = self.mx, next(self._serial)
        if arm == CAND:
            cell = prep["cell"]
            out, engaged, plan = self.cand.tensor_fa_attention(
                prep["q"], prep["k"], prep["v"], scale=cell["scale"], causal=True, q_start=cell["q_start"],
                allow_research_metal=True)
            return Dispatch(arm, (out,), engaged, plan, serial)
        if arm == BASE:
            out = mx.fast.scaled_dot_product_attention(prep["q"], prep["k"], prep["v"], scale=prep["cell"]["scale"],
                                                       mask="causal")
            return Dispatch(arm, (out,), None, None, serial)
        if arm == CONVERT:
            return Dispatch(arm, (prep["k32"].astype(mx.float16), prep["v32"].astype(mx.float16)), None, None, serial)
        if arm == MATERIALIZE:
            return Dispatch(arm, (mx.contiguous(prep["k_tok"].transpose(0, 2, 1, 3)),
                                  mx.contiguous(prep["v_tok"].transpose(0, 2, 1, 3))), None, None, serial)
        raise Refused(f"unknown arm {arm}")

    def complete(self, d):
        self.mx.eval(*d.outputs, *([d.engaged] if d.engaged is not None else []))
        self.mx.synchronize()

    def engagement(self, d):
        observed = int(d.engaged.item())
        return observed, self.cand.confirm_engagement(d.engaged)

    def counters(self):
        s = self.cand.status()
        return {"issued": s["native_launches_issued"], "confirmed": s["native_engagement_confirmed"]}

    def to_host(self, d):
        import numpy as np

        hosts = tuple(np.array(x) for x in d.outputs)
        d.outputs = d.engaged = None
        return hosts

    def memory_reset(self):
        mx = self.mx
        mx.synchronize()
        if not hasattr(mx, "reset_peak_memory"):
            return {"scope": "omitted: reset_peak_memory unavailable"}
        mx.reset_peak_memory()
        return {"active_bytes_before": int(mx.get_active_memory())}

    def memory_peak(self):
        mx = self.mx
        mx.synchronize()
        if not hasattr(mx, "get_peak_memory"):
            return {"scope": "omitted: get_peak_memory unavailable"}
        return {"peak_bytes": int(mx.get_peak_memory()),
                "scope": "synchronized allocator peak (resident inputs included), outside timed regions"}

    def release_cell(self, prep):
        prep.clear()
        self.mx.synchronize()
        self.mx.clear_cache()

    def release(self):
        self.mx.synchronize()
        self.mx.clear_cache()


_TIMING_WITNESS = object()   # module-private; held only by live runs built in _run_native_timing


class _LiveTimingRun:
    """In-process record of an actual native timing orchestration (never serialized)."""

    __slots__ = ("_witness", "admission", "gate_run", "ops", "post_run", "report")

    def __init__(self, witness, ops, admission, gate_run, report, post_run):
        self._witness, self.ops, self.admission, self.gate_run = witness, ops, admission, gate_run
        self.report, self.post_run = report, list(post_run)


def _run_numeric_gate(admission, cases, full_requested, backend_factory=None):
    """The qualifier's numeric gate (q.run_gate, q.post_run_refusals) under the COMBINED qualifier+timing guard.

    Mirrors q._run_native, whose own guard covers only the qualifier files. The run object is the
    qualifier's live type, so q.native_verdict still refuses any backend that is not q.NativeBackend.
    """
    timing_guard(admission)                                  # before MLX is imported by the backend
    backend = (backend_factory or q.NativeBackend)(admission)
    paths = {"candidate": backend.cand.__file__, "mx": backend.mx.__file__}
    report = q.run_gate(backend, cases, guard=lambda: timing_guard(admission, paths))
    post = q.post_run_refusals(admission, backend.cand.__file__, backend.mx.__file__)
    try:
        timing_guard(admission, paths)
    except q.Refused as error:
        post.append(str(error))
    return q._LiveNativeRun(q._WITNESS, backend, admission, report, post, cases, full_requested)


def _run_native_timing(admission):
    cases, full = q.select_cases(None)                       # the full mandatory numeric catalogue
    gate_run = _run_numeric_gate(admission, cases, full)
    gate = q.native_verdict(gate_run)
    if not gate["native_synthetic_gate"]:
        raise Refused("live numeric gate did not pass; no timed dispatch: "
                      + "; ".join((gate["reasons"] + gate["evaluation"].get("refusals", []))[:8]))
    ops = NativeTimingOps(gate_run)
    paths = ops.module_paths()
    report = run_timing(ops, CELLS, guard=lambda: timing_guard(admission, paths), clock=time.perf_counter_ns)
    post = []
    try:
        timing_guard(admission, ops.module_paths())
    except q.Refused as error:
        post.append(str(error))
    return _LiveTimingRun(_TIMING_WITNESS, ops, admission, gate_run, report, post)


def timing_verdict(live):
    """The ONLY place native timing execution can be stamped: a live run object, never a dict/JSON.

    The numeric gate is re-derived here from the live gate run and the timing evaluation from the run's
    own report; no caller-supplied pass, gate or evaluation is consulted.
    """
    reasons = []
    gate = q.native_verdict(getattr(live, "gate_run", None))
    report = getattr(live, "report", None)
    evaluation = evaluate_timing(report) if isinstance(report, dict) else {
        "verdict": "refused", "refusals": ["no live report"], "summary": {}}
    if type(live) is not _LiveTimingRun or getattr(live, "_witness", None) is not _TIMING_WITNESS:
        reasons.append("no live native timing orchestration (evidence alone never establishes native timing)")
    elif type(live.ops) is not NativeTimingOps or live.ops.gate_run is not live.gate_run:
        reasons.append("timing ops are not the native ops bound to this run's live gate")
    else:
        reasons += live.post_run
    if not gate["native_synthetic_gate"]:
        reasons.append("live numeric gate did not pass")
    if evaluation.get("verdict") != "timing_evidence_complete":
        reasons.append("timing evidence incomplete or failed")
    return {"native_timing_execution": not reasons, "reasons": reasons, "evaluation": evaluation,
            "gate": {"native_synthetic_gate": gate["native_synthetic_gate"], "reasons": gate["reasons"],
                     "verdict": gate["evaluation"].get("verdict")},
            "qualified": False, "selected": False, "model_gain": False}


# ================================================================ CLI

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalogue", action="store_true", help="print the timing catalogue (no MLX import)")
    ap.add_argument("--run-native", action="store_true")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--source-root")
    ap.add_argument("--source-commit")
    ap.add_argument("--out")
    return ap


def main(argv=None, environ=None):
    args = build_parser().parse_args(argv)
    environ = os.environ if environ is None else environ
    if args.catalogue:
        print(json.dumps({"cells": CELLS, "fairness": FAIRNESS, "conversion": CONVERSION_SCOPE,
                          "schedule": {"warmup_pairs": WARMUP_PAIRS, "pairs": PAIRS, "conversion_warmup": CONV_WARMUP,
                                       "conversion_repeats": CONV_REPEATS},
                          "tolerance": q.TOLERANCE, "scope": SCOPE}, indent=1))
        return 0
    try:
        admission = timing_admission(args, environ)              # before any MLX import
        live = _run_native_timing(admission)                      # numeric gate first, then timing
        verdict = timing_verdict(live)
    except q.Refused as error:
        print(json.dumps({"verdict": "refused", "refusals": [str(error)], "native_timing_execution": False,
                          "qualified": False, "selected": False, "model_gain": False}))
        return 1
    receipt = {"schema": SCHEMA, "scope": SCOPE, "fairness": FAIRNESS, "tolerance": q.TOLERANCE,
               "verdict": verdict, "report": live.report, "qualified": False, "selected": False, "model_gain": False,
               "note": "native_timing_execution was stamped in-process; re-evaluating this JSON can never "
                       "re-establish it; timing never qualifies a model, cache, continuation or production route"}
    with open(args.out, "x") as handle:                           # exclusive creation, completed runs only
        json.dump(receipt, handle, indent=1)
    print(json.dumps({k: verdict[k] for k in ("native_timing_execution", "reasons", "gate")}))
    return 0 if verdict["native_timing_execution"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
