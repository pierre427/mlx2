#!/usr/bin/env python3
"""Where a Hy4-flash DFly2 round's time goes at B=1, 2 and 4 (one process, one load).

Original mlx2 script.  A profiler to rank B>1 tuning opportunities, not
qualification and not publishable statistics: no thermal gating, one pass
per arm.  The model is loaded once through ``HYV4FlashAdapter`` (default
policy ``{"draft_model": "embedded_dfly", "vision": false}``: DFly2, the
packed chain executor) and the weights are wired as serving wires them.  For
each width in ``--widths`` the first B GSM8K test questions
(``benchmarks/gsm8k/test.jsonl``, chat template, the adapter's default
reasoning effort) decode greedily on the external-draft engine, B lanes in
one batch, at most ``--rounds`` scheduler calls after every lane has its
first token:

``plain``         no wrappers: wall time per ``batch.next()`` call, committed
                  tokens per lane per call (accept lengths), tok/s, engine
                  counter deltas.
``instrumented``  the same prompts again with class-level wrappers around
                  every target verify dispatch (``Hy4ChainTreeExecutor._run``),
                  every drafter call and any other target forward.  While a
                  wrapped call builds its graph ``mx.async_eval`` is a no-op,
                  so building ends before the GPU starts; ``mx.eval`` of its
                  outputs is then the GPU wait (encode + execute + one sync).
                  Per call: ``gpu_ms`` = sum of those waits (GPU busy, an
                  upper bound), ``build_ms`` = graph build (Python + Engram
                  host hashing), ``host_ms`` = wall - build - gpu (engine
                  scheduling, commit, sampling reads), ``idle_ms`` = wall -
                  gpu.  This pass serializes the round, so its tok/s is below
                  the plain pass.
                  Per verify dispatch: lanes, rows per lane, waves, padded
                  rows (rows a short lane spends waiting on the longest one
                  in the joint forward), re-fed root-path rows
                  (``verify_phase_align``), and the routed experts of every
                  trunk MoE layer (``Router`` outputs).  Logical sets use
                  the new rows only: unique experts per lane, their union,
                  the sum of per-lane sets, pairwise Jaccard.  Kernel counts
                  use every routed row, padding and re-fed rows included
                  (``MoE.__call__`` routes them all) and report those two
                  apart: assignments, union, assignments per unique expert.
                  At verify sizes (B=4 ~384 assignments, far below MLX's
                  ~4 x 512 streaming-gather floor, switch_layers ``_RHS_*``)
                  each row's gather_qmv reads its own experts, so kernel
                  assignments, not the union, are the routed-weight reads.
                  Per call: whether every active lane verified in one joint
                  dispatch, or why not (engine counter deltas: decode-group
                  split by depth / parked / ordinary, memory-fit split,
                  draining lanes, lanes not ready, parked plain-anchor
                  rounds).
``replay``        for the first ``--replay-rounds`` multi-lane verify
                  dispatches of the instrumented pass, the dispatch's inputs
                  and pre-round caches (aliased descriptor clones) are kept
                  and, after that scheduler call returns, replayed outside
                  the round: the cohort jointly, then each lane alone, each
                  ``--replay-reps`` times after one warm-up (median).  Gives
                  verify time per lane joint vs alone, the round-boundary
                  stall per lane (joint - alone: time a lane waits on the
                  cohort), and the joint saving (sum alone - joint).
                  Each replay gives the lanes private copies of their lent
                  buffers, evaluated before the clock, so the replayed root
                  writes in place as the live round does.  Capturing a
                  live round's caches still makes that round's lent buffers
                  copy on write once; those rounds are flagged
                  ``replay_captured`` and left out of the medians.

Budget: load (~1-2 min for the 4.6-bit pack) + per width 2 x ``--rounds``
calls + replays.  The defaults (widths 1 2 4, 40 rounds, 6 replays, 3 reps)
fit in about 10 GPU minutes on an M5 Max; memory is the model plus at most
four ~0.5-1K-token lanes, the replay clones (copy-on-write rows of those
lanes) and host-side expert sets: well under 2 GB beyond the weights.

GPU (the caller owns the GPU; one model per process):

    PYTHONPATH="$PWD/src" .venv/bin/python scripts/profile_hy4_batched_verify.py --i-own-the-gpu \\
        --model ~/mlx-models/Hy4-flash-MLX-4.6bit-mlx2 --out <empty dir> [--widths 1 2 4] [--rounds 40]

``--cpu`` runs on the CPU device (smoke test on the tiny fixtures).  Writes
``receipt.json`` (schema ``mlx2.profile_hy4_batched_verify.v1``, naming
``checkpoint`` = ``<artifact>@<fingerprint>`` and the mlx2 commit) after
every width; refuses a nonempty output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SCHEMA = "mlx2.profile_hy4_batched_verify.v1"
DEFAULT_POLICY = {"draft_model": "embedded_dfly", "vision": False}
INSTRUCTION = "\nThink step by step, then give the final answer after '####'."
# Engine counters whose per-call deltas explain a split round.
SPLIT_COUNTERS = {
    "external_decode_group_split_depth_phases": "group_split_depth",
    "external_decode_parked_split_phases": "group_split_parked",
    "external_decode_group_split_ordinary_phases": "group_split_ordinary",
    "external_decode_fit_split_groups": "memory_fit_split",
    "external_decode_ast_turn_unpackable_lanes": "ast_turn_unpackable_lane",
    "external_decode_ast_turn_kind_splits": "ast_turn_kind_split",
    "external_decode_ast_turn_limit_splits": "ast_turn_lane_limit",
    "external_decode_ast_turn_tile_policy_limit_splits": "ast_turn_tile_policy_limit_one",
    "external_decode_draining_lanes_skipped": "draining_lanes",
    "external_cost_parked_rounds": "parked_plain_anchor_round",
    "external_parked_pipelined_rounds": "parked_pipelined_round",
    "external_cost_plain_probe_rounds": "ordinary_probe_round",
}
pc = time.perf_counter
mx = None
P = None  # scripts/profile_hy4_paths (shared helpers)


def log(msg):
    print(f"[profile_hy4_b4] {msg}", flush=True)


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def ms(seconds):
    return round(seconds * 1e3, 4)


def gsm8k_prompts(adapter, count):
    rows = [json.loads(line) for line in (ROOT / "benchmarks/gsm8k/test.jsonl").read_text().splitlines()[:count]]
    vocab = adapter.model.args.vocab_size
    out = []
    for index, row in enumerate(rows):
        try:
            ids = [int(i) for i in adapter.prompt_tokens(
                {"messages": [{"role": "user", "content": row["question"] + INSTRUCTION}]})]
        except Exception:  # noqa: BLE001 - a tiny fixture has no chat template
            ids = []
        if not ids or any(not 0 <= i < vocab for i in ids) or len(set(ids)) < 8:
            # A tiny fixture tokenizer maps text to specials: synthetic ids.
            rng = np.random.default_rng(100 + index)
            ids, source = [int(x) for x in rng.integers(4, vocab, size=24 + 5 * index)], "synthetic_random_ids"
        else:
            source = f"gsm8k test[{index}], chat template"
        out.append((ids, source))
    return out


# ------------------------------------------------------------ recorder --

class Recorder:
    """Class-level wrappers for one instrumented scheduler call at a time."""

    def __init__(self, model, replay_limit):
        self.model = model
        self.call = None
        self.depth = 0
        self.capture = None  # list of (layer, indices) while a verify forward builds
        self.replay_limit = replay_limit
        self.replays = []
        self.router_layer = {}
        from mlx2.runtime.models import hy_v4_flash as H

        for i, layer in enumerate(model.model.layers):
            gate = getattr(getattr(layer, "mlp", None), "gate", None)
            if isinstance(gate, H.Router):
                self.router_layer[id(gate)] = i

    def begin(self):
        self.call = {"dispatches": [], "draft": Counter(), "target_other": Counter()}

    def end(self):
        call, self.call = self.call, None
        return call

    def _timed(self, orig, args, kwargs, result):
        with P.no_async_eval():
            t = pc()
            out = orig(*args, **kwargs)
            built = pc() - t
        t = pc()
        P.eval_tree(result(out))
        return out, built, pc() - t

    @contextmanager
    def active(self):
        from mlx2.adapters import hy_v4_flash_ast as X
        from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor
        from mlx2.runtime.drafters.hy4_dfly import Hy4DFlyDraftModel
        from mlx2.runtime.models import hy_v4_flash as H

        rec = self
        run = Hy4ChainTreeExecutor.__dict__["_run"]
        wave = X._wave_forward
        router = H.Router.__dict__["__call__"]
        hidden = H.Model.__dict__["hidden"]

        def run_wrapper(executor, model, inputs, parents, caches, layers, *, plain=False):
            if rec.call is None or rec.depth:
                return run(executor, model, inputs, parents, caches, layers, plain=plain)
            row = {"kind": "plain" if plain else "verify", "lanes": len(inputs),
                   "rows_per_lane": [len(r) for r in inputs], "waves": 0, "forwards": [],
                   "padded_rows": 0}
            capture = None
            if not plain and len(inputs) > 1 and len(rec.replays) < rec.replay_limit:
                capture = {"inputs": [list(map(int, r)) for r in inputs],
                           "parents": [list(map(int, p)) for p in parents],
                           "caches": [X.clone_caches(c) for c in caches], "layers": layers}
                row["replay_captured"] = True
            rec.call["current"] = row
            rec.depth += 1
            try:
                out, built, waited = rec._timed(
                    lambda: run(executor, model, inputs, parents, caches, layers, plain=plain), (), {},
                    lambda o: (o[0], o[1]))
            finally:
                rec.depth -= 1
                rec.call.pop("current", None)
            row["build_ms"], row["gpu_ms"] = ms(built), ms(waited)
            row["waves"] = len(row["forwards"])
            for forward in row["forwards"]:
                # Router outputs were evaluated with the dispatch's outputs:
                # reading them now adds no sync inside the timed region.
                forward["experts"] = expert_stats(forward.pop("_captured"), forward["rows"],
                                                  forward.pop("_lanes"), forward["packed"],
                                                  forward.pop("_refed"))
            rec.call["dispatches"].append(row)
            if capture is not None:
                rec.replays.append(capture)
                row["replay_index"] = len(rec.replays) - 1
            return out

        def wave_wrapper(model, jobs, layers, joint, lanes=None, prefix=None):
            row = rec.call.get("current") if rec.call is not None else None
            if row is None:
                return wave(model, jobs, layers, joint, lanes, prefix)
            lengths = [len(tokens) for tokens, _ in jobs]
            lane_ids = [0] * len(jobs) if lanes is None else list(lanes)
            rec.capture = []
            try:
                out = wave(model, jobs, layers, joint, lanes, prefix)
            finally:
                captured, rec.capture = rec.capture, None
            packed = out[1]
            width = max(lengths)
            refed = [0] * len(jobs) if prefix is None else [int(n) for n in prefix]
            forward = {"chains": len(jobs), "packed": bool(packed), "rows": lengths,
                       "refed_rows": refed,
                       "padded_rows": (width * len(jobs) - sum(lengths)) if packed else 0}
            row["padded_rows"] += forward["padded_rows"]
            row["refed_rows"] = row.get("refed_rows", 0) + sum(refed)
            forward["_captured"], forward["_lanes"] = captured, lane_ids
            forward["_refed"] = refed
            row["forwards"].append(forward)
            return out

        def router_wrapper(gate, x):
            out = router(gate, x)
            if rec.capture is not None and id(gate) in rec.router_layer:
                rec.capture.append((rec.router_layer[id(gate)], out[0]))
            return out

        def drafter(name):
            orig = Hy4DFlyDraftModel.__dict__[name]

            def wrapper(*args, **kwargs):
                if rec.call is None or rec.depth:
                    return orig(*args, **kwargs)
                rec.depth += 1
                try:
                    out, built, waited = rec._timed(orig, args, kwargs, lambda o: o)
                finally:
                    rec.depth -= 1
                key = "context" if name == "append_context" else "propose"
                rec.call["draft"][f"{key}_calls"] += 1
                rec.call["draft"][f"{key}_build_s"] += built
                rec.call["draft"][f"{key}_gpu_s"] += waited
                return out
            return wrapper

        def hidden_wrapper(model, *args, **kwargs):
            if rec.call is None or rec.depth:
                return hidden(model, *args, **kwargs)
            rec.depth += 1
            try:
                out, built, waited = rec._timed(hidden, (model, *args), kwargs, lambda o: o)
            finally:
                rec.depth -= 1
            rec.call["target_other"]["calls"] += 1
            rec.call["target_other"]["build_s"] += built
            rec.call["target_other"]["gpu_s"] += waited
            return out

        with ExitStack() as stack:
            stack.enter_context(P.patched(Hy4ChainTreeExecutor, "_run", run_wrapper))
            stack.enter_context(P.patched(H.Router, "__call__", router_wrapper))
            stack.enter_context(P.patched(H.Model, "hidden", hidden_wrapper))
            for name in ("propose_tree", "draft_distributions", "append_context"):
                if name in Hy4DFlyDraftModel.__dict__:
                    stack.enter_context(P.patched(Hy4DFlyDraftModel, name, drafter(name)))
            X._wave_forward = wave_wrapper
            try:
                yield self
            finally:
                X._wave_forward = wave


def expert_stats(captured, lengths, lanes, packed, refed=None):
    """Per MoE layer: routed experts of the verified rows, and what the kernel read.

    ``captured`` is ``[(layer, indices)]`` in call order: one ``[B, W, k]``
    entry per layer for a packed forward, else one ``[1, L_j, k]`` entry per
    layer per chain (job-major).  Each job's rows split into re-fed root-path
    rows (its first ``refed[j]``, ``verify_phase_align``), new rows, and, in
    a packed forward, padding rows up to the width ``W`` -- ``MoE.__call__``
    routes padding rows too, so the kernel reads their experts.

    Logical (new rows only; chains of one lane pool into its set):
    ``per_lane``, ``union``, ``sum``, ``jaccard_mean``, ``assignments``.
    Kernel (every routed row): ``kernel_assignments`` (= new + re-fed +
    padding assignments), ``kernel_union``, ``kernel_assignments_per_unique``
    (below ~4 x n_experts assignments MLX gathers per row, gather_qmv, so
    each assignment reads its expert slice).
    """
    refed = [0] * len(lengths) if refed is None else list(refed)
    by_layer = {}
    for layer, indices in captured:
        by_layer.setdefault(layer, []).append(np.array(indices))
    out = []
    for layer, entries in sorted(by_layer.items()):
        full = ([entries[0][j] for j in range(len(lengths))] if packed
                else [e[0] for e in entries[:len(lengths)]])
        sets, kernel = {}, set()
        new_n = refed_n = pad_n = 0
        for lane, row, n, skip in zip(lanes, full, lengths, refed):
            new, pre, pad = row[skip:n], row[:skip], row[n:]
            sets.setdefault(lane, set()).update(int(e) for e in new.reshape(-1))
            kernel.update(int(e) for e in row.reshape(-1))
            new_n, refed_n, pad_n = new_n + new.size, refed_n + pre.size, pad_n + pad.size
        lane_sets = list(sets.values())
        union = set().union(*lane_sets)
        pairs = [len(a & b) / max(1, len(a | b)) for a, b in combinations(lane_sets, 2)]
        total = new_n + refed_n + pad_n
        out.append({"layer": layer, "per_lane": [len(s) for s in lane_sets], "union": len(union),
                    "sum": sum(len(s) for s in lane_sets),
                    "jaccard_mean": round(float(np.mean(pairs)), 4) if pairs else None,
                    "assignments": int(new_n),
                    "refed_assignments": int(refed_n), "padding_assignments": int(pad_n),
                    "kernel_assignments": int(total), "kernel_union": len(kernel),
                    "kernel_assignments_per_unique": round(total / max(1, len(kernel)), 3)})
    return out


# ---------------------------------------------------------------- drive --

def drive(adapter, a, width, prompts, recorder=None):
    batch = adapter.create_external_batch(completion_batch_size=width, prefill_step_size=a.prefill_step)
    stats0 = dict(batch.scheduler_stats)
    uids = batch.insert([p for p, _ in prompts], max_tokens=[a.max_tokens] * width,
                        sampling_configs=[{"sampling_temp": 0}] * width)
    got = {u: [] for u in uids}
    done = set()
    calls = []
    started = None
    for _ in range(8 * (a.rounds + a.max_tokens) + 64):
        before = dict(batch.scheduler_stats)
        active = [u for u in uids if u not in done]
        if recorder is not None:
            recorder.begin()
        mx.synchronize()
        t = pc()
        result = batch.next()
        mx.synchronize()
        wall = pc() - t
        detail = recorder.end() if recorder is not None else None
        responses = result[1] if isinstance(result, tuple) else result
        tokens = Counter()
        for r in responses:
            if r.uid in got:
                got[r.uid].append(int(r.token))
                tokens[r.uid] += 1
                if getattr(r, "finish_reason", None) or len(got[r.uid]) >= a.max_tokens:
                    done.add(r.uid)
        if started is None:
            if all(got[u] for u in uids):
                started = len(calls)  # rounds counted after every lane's first token
            calls.append(None)
            continue
        after = dict(batch.scheduler_stats)
        delta = {k: after[k] - before.get(k, 0) for k in after
                 if isinstance(after[k], (int, float)) and after[k] != before.get(k, 0)}
        row = {"wall_ms": ms(wall), "active_lanes": len(active),
               "committed": [tokens.get(u, 0) for u in uids], "stats_delta": delta}
        if detail is not None:
            row.update(classify(detail, row, delta))
        calls.append(row)
        if len(done) == len(uids) or len(calls) - started - 1 >= a.rounds:
            break
    stats1 = dict(batch.scheduler_stats)
    close = getattr(batch, "close", None)
    if callable(close):
        close()
    del batch
    P.free()
    rounds = [c for c in calls if c is not None]
    seconds = sum(c["wall_ms"] for c in rounds) / 1e3
    committed = sum(sum(c["committed"]) for c in rounds)
    lane_accepts = [n for c in rounds for n in c["committed"] if n]
    return {
        "rounds": len(rounds), "finished_lanes": len(done),
        "tok_s": committed / seconds if seconds else None,
        "committed_per_call_mean": committed / len(rounds) if rounds else None,
        "accept_length_per_lane_round": {"mean": float(np.mean(lane_accepts)) if lane_accepts else None,
                                         "hist": dict(Counter(lane_accepts))},
        "wall_ms_median": med([c["wall_ms"] for c in rounds]),
        "scheduler_stats_delta": {k: stats1[k] - stats0.get(k, 0) for k in stats1
                                  if isinstance(stats1[k], (int, float)) and stats1[k] != stats0.get(k, 0)},
        "per_call": rounds,
    }


def classify(detail, row, delta):
    """Split a call's time and say whether every active lane verified jointly."""
    disp = detail["dispatches"]
    verify = [d for d in disp if d["kind"] == "verify"]
    plain = [d for d in disp if d["kind"] == "plain"]
    draft, other = detail["draft"], detail["target_other"]
    gpu = sum(d["gpu_ms"] for d in disp) + ms(draft["propose_gpu_s"] + draft["context_gpu_s"]
                                               + other["gpu_s"])
    build = sum(d["build_ms"] for d in disp) + ms(draft["propose_build_s"] + draft["context_build_s"]
                                                   + other["build_s"])
    lanes_verified = sum(d["lanes"] for d in verify)
    reasons = sorted({name for key, name in SPLIT_COUNTERS.items() if delta.get(key)})
    if not disp:
        shape = "no_target_dispatch"
    elif row["active_lanes"] == 1:
        shape = "single_lane_verify" if verify else "single_lane_plain"
    elif len(disp) == 1 and disp[0]["lanes"] == row["active_lanes"]:
        # Every active lane in one target dispatch: a joint verify, or a
        # joint anchor-only step (all lanes parked / ordinary probe).
        shape = "joint_all_verify" if verify else "joint_all_plain"
    else:
        shape = "split"
        if lanes_verified + sum(d["lanes"] for d in plain) < row["active_lanes"]:
            reasons.append("lanes_not_in_any_dispatch")
        if plain:
            reasons.append("plain_anchor_dispatch_beside_verify")
        if any(d["waves"] > 1 for d in verify):
            reasons.append("multi_wave_tree")
    return {
        "shape": shape, "split_reasons": reasons,
        "gpu_busy_ms": round(gpu, 4), "build_ms": round(build, 4),
        "host_ms": round(row["wall_ms"] - gpu - build, 4), "idle_ms": round(row["wall_ms"] - gpu, 4),
        "draft": {"propose_calls": draft["propose_calls"], "context_calls": draft["context_calls"],
                  "gpu_ms": ms(draft["propose_gpu_s"] + draft["context_gpu_s"]),
                  "build_ms": ms(draft["propose_build_s"] + draft["context_build_s"])},
        "target_other": {"calls": other["calls"], "gpu_ms": ms(other["gpu_s"]),
                         "build_ms": ms(other["build_s"])},
        "dispatches": disp,
    }


def replay(adapter, captures, reps):
    from mlx2.adapters import hy_v4_flash_ast as X

    model = adapter.model
    token = adapter.tree_executor.source_revision_token if adapter.tree_executor else "replay"
    joint = getattr(adapter.tree_executor, "joint", True)
    phase = getattr(adapter.tree_executor, "phase_align", False)

    def owned(caches):
        """A clone whose lent (append-only) buffers are private copies, made
        and evaluated before the clock: a replayed root chain then writes
        them in place, as the live round does, instead of copying the
        captured history it would otherwise alias.  Sliding rings stay
        aliased (never lent: the live root copies them too)."""
        lives = X.clone_caches(caches)
        fresh = []
        for row in X._row_caches(lives):
            for name, _, _ in row._RECOVERY_APPEND_ONLY_FIELDS:
                data = getattr(row, name, None)
                if data is not None:
                    setattr(row, name, data + 0)
                    fresh.append(getattr(row, name))
        mx.eval(fresh)
        return lives

    def timed(lanes):
        times = []
        for rep in range(reps + 1):
            lives = [owned(captures_c) for captures_c in lanes["caches"]]
            mx.synchronize()
            t = pc()
            logits, features, transactions, _ = X.cohort_forward(
                model, lanes["inputs"], lanes["parents"], lives, lanes["layers"], token,
                joint=joint, phase_align=phase)
            mx.eval(logits, features)
            if rep:
                times.append(pc() - t)
            for tr in transactions:
                tr.abort()
            del lives, logits, features, transactions
        return statistics.median(times)

    out = []
    for c in captures:
        joint_s = timed(c)
        alone = [timed({"inputs": [c["inputs"][j]], "parents": [c["parents"][j]], "caches": [c["caches"][j]],
                        "layers": c["layers"]}) for j in range(len(c["inputs"]))]
        out.append({"lanes": len(c["inputs"]), "rows_per_lane": [len(r) for r in c["inputs"]],
                    "joint_ms": ms(joint_s), "alone_ms": [ms(s) for s in alone],
                    "sum_alone_ms": ms(sum(alone)), "joint_saving_ms": ms(sum(alone) - joint_s),
                    "stall_ms_per_lane": [ms(joint_s - s) for s in alone],
                    "joint_over_slowest_alone": round(joint_s / max(alone), 4)})
        P.free()
    return out


def summarize(inst, rep):
    calls = [c for c in inst["per_call"] if not any(d.get("replay_captured") for d in c.get("dispatches", []))]
    shapes = Counter(c["shape"] for c in calls)
    reasons = Counter(r for c in calls if c["shape"] == "split" for r in c["split_reasons"])
    fwd = [f for c in calls for d in c["dispatches"] if d["kind"] == "verify" for f in d["forwards"]]
    layers = {}
    for f in fwd:
        for e in f["experts"]:
            layers.setdefault(e["layer"], []).append(e)
    per_layer = {str(k): {"union_median": med([e["union"] for e in v]),
                          "sum_of_lanes_median": med([e["sum"] for e in v]),
                          "per_lane_median": med([x for e in v for x in e["per_lane"]]),
                          "jaccard_median": med([e["jaccard_mean"] for e in v]),
                          "kernel_union_median": med([e["kernel_union"] for e in v]),
                          "kernel_assignments_median": med([e["kernel_assignments"] for e in v]),
                          "kernel_assignments_per_unique_median": med(
                              [e["kernel_assignments_per_unique"] for e in v])}
                 for k, v in sorted(layers.items())}
    unions = [e["union"] for f in fwd for e in f["experts"]]
    sums = [e["sum"] for f in fwd for e in f["experts"]]
    return {
        "calls": len(calls), "shapes": dict(shapes), "split_reasons": dict(reasons),
        "wall_ms_median": med([c["wall_ms"] for c in calls]),
        "gpu_busy_ms_median": med([c["gpu_busy_ms"] for c in calls]),
        "idle_ms_median": med([c["idle_ms"] for c in calls]),
        "host_ms_median": med([c["host_ms"] for c in calls]),
        "build_ms_median": med([c["build_ms"] for c in calls]),
        "gpu_busy_fraction": (round(sum(c["gpu_busy_ms"] for c in calls) / sum(c["wall_ms"] for c in calls), 4)
                              if calls else None),
        "verify_gpu_ms_median": med([d["gpu_ms"] for c in calls for d in c["dispatches"] if d["kind"] == "verify"]),
        "draft_gpu_ms_median": med([c["draft"]["gpu_ms"] for c in calls if c["draft"]["propose_calls"]]),
        "draft_propose_calls_per_verify_call_mean": (
            float(np.mean([c["draft"]["propose_calls"] for c in calls
                           if any(d["kind"] == "verify" for d in c["dispatches"])]))
            if any(d["kind"] == "verify" for c in calls for d in c["dispatches"]) else None),
        "verify_call_wall_ms_median": med([c["wall_ms"] for c in calls
                                           if any(d["kind"] == "verify" for d in c["dispatches"])]),
        "verify_call_gpu_busy_ms_median": med([c["gpu_busy_ms"] for c in calls
                                               if any(d["kind"] == "verify" for d in c["dispatches"])]),
        "padded_rows_mean": (float(np.mean([d["padded_rows"] for c in calls for d in c["dispatches"]
                                             if d["kind"] == "verify"])) if fwd else None),
        "refed_rows_mean": (float(np.mean([d.get("refed_rows", 0) for c in calls for d in c["dispatches"]
                                            if d["kind"] == "verify"])) if fwd else None),
        # Logical sets: the verified (new) rows only.
        "experts_union_median": med(unions), "experts_sum_of_lanes_median": med(sums),
        "expert_set_overlap": round(sum(sums) / sum(unions), 4) if unions else None,
        # What the gather kernel read: every routed row, padding and re-fed included.
        "kernel_assignments_median": med([e["kernel_assignments"] for f in fwd for e in f["experts"]]),
        "kernel_union_median": med([e["kernel_union"] for f in fwd for e in f["experts"]]),
        "kernel_refed_assignments_total": sum(e["refed_assignments"] for f in fwd for e in f["experts"]),
        "kernel_padding_assignments_total": sum(e["padding_assignments"] for f in fwd for e in f["experts"]),
        "kernel_assignments_per_unique_median": med(
            [e["kernel_assignments_per_unique"] for f in fwd for e in f["experts"]]),
        "experts_per_layer": per_layer,
        "replay_joint_saving_ms_median": med([r["joint_saving_ms"] for r in rep]),
        "replay_stall_ms_median": med([s for r in rep for s in r["stall_ms_per_lane"]]),
        "replay_joint_over_slowest_alone_median": med([r["joint_over_slowest_alone"] for r in rep]),
    }


# ------------------------------------------------------------------ main --

def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="output directory (must be empty or absent)")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="run on the CPU device (smoke test)")
    ap.add_argument("--no-wire", action="store_true", help="leave the weights pageable (default: wire them)")
    ap.add_argument("--policy-json", default=json.dumps(DEFAULT_POLICY),
                    help="adapter execution policy (e.g. add \"verify_phase_align\": true for an arm)")
    ap.add_argument("--widths", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--rounds", type=int, default=40, help="scheduler calls per pass after the first tokens")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--replay-rounds", type=int, default=6)
    ap.add_argument("--replay-reps", type=int, default=3)
    ap.add_argument("--prefill-step", type=int, default=2048)
    a = ap.parse_args(argv)
    if not (a.cpu or a.i_own_the_gpu):
        ap.error("refusing Metal execution without --i-own-the-gpu (or pass --cpu)")
    if any(not 1 <= w <= 4 for w in a.widths):
        ap.error("--widths must be in 1..4 (the Hy4 AST lane bound)")
    if a.rounds < 1:
        ap.error("--rounds must be positive")
    out = Path(a.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        ap.error(f"{out} exists and is not an empty directory; refusing to overwrite evidence")
    a.policy = json.loads(a.policy_json)
    return a


def write(out, doc):
    tmp = out / "receipt.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1, default=str))
    tmp.replace(out / "receipt.json")


def main(argv=None):
    global mx, P
    a = parse(argv)
    import mlx.core as _mx

    mx = _mx
    if a.cpu:
        mx.set_default_device(mx.cpu)
    import profile_hy4_paths as _P

    P = _P
    P.mx = mx
    from mlx2.adapters.hy_v4_flash import HYV4FlashAdapter
    from mlx2.runtime.models import hy_v4_flash as H
    from mlx2.runtime.models import switch_layers

    P.H = H
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        info = mx.device_info()
    except Exception:  # noqa: BLE001
        info = {}
    doc = {"schema": SCHEMA, "started": now(), "finished": None, "mlx2": P.git_state(), "mlx": mx.__version__,
           "device": str(mx.default_device()), "device_info": dict(info), "policy": a.policy,
           "model_path": str(Path(a.model).expanduser()),
           "args": {k: v for k, v in vars(a).items() if k != "policy"},
           "env": {k: v for k, v in os.environ.items() if k.startswith(("MLX_", "MLX2_"))},
           "method": {k: v.strip() for k, v in (
               ("gpu_busy", "sum of mx.eval waits of wrapped target/drafter calls, built with mx.async_eval "
                            "disabled (encode + execute + one sync each; an upper bound on GPU busy)"),
               ("host", "call wall - graph build - GPU waits: engine scheduling, commit, host reads"),
               ("experts", "trunk Router top-k indices of every verify forward, padding rows dropped"),
               ("replay", "captured pre-round caches (aliased clones) replayed after the call: joint "
                          "cohort vs each lane alone, median of reps after one warm-up"),
               ("not_publishable", "no thermal gating, one pass per arm"))},
           "swap": {}, "memory": {}, "widths": {}}
    write(out, doc)
    t0 = pc()
    swap0 = P.swap_state()
    adapter = HYV4FlashAdapter(str(Path(a.model).expanduser()), execution_policy=a.policy)
    doc["load_s"] = round(pc() - t0, 2)
    doc["swap"]["load"] = {"before": swap0, "after": P.swap_state()}
    doc["checkpoint"] = adapter.diagnostics()["checkpoint"]
    if not a.no_wire:
        from mlx2.runtime.weight_residency import wire_serving_weights

        doc["weight_residency"] = wire_serving_weights()
    log(f"loaded {doc['checkpoint']} in {doc['load_s']} s on {doc['device']}")
    try:
        if adapter.drafter_kind != "dfly":
            raise SystemExit("the policy selects no embedded DFly2 drafter")
        prompts = gsm8k_prompts(adapter, max(a.widths))
        doc["prompts"] = [{"tokens": len(p), "source": s} for p, s in prompts]
        for width in a.widths:
            ts = pc()
            doc["swap"][str(width)] = {"before": P.swap_state()}
            if hasattr(mx, "reset_peak_memory"):
                mx.reset_peak_memory()
            switch_layers.moe_pad_status(reset=True)
            plain = drive(adapter, a, width, prompts[:width])
            log(f"B={width} plain: {plain['tok_s']} tok/s over {plain['rounds']} calls")
            recorder = Recorder(adapter.model, a.replay_rounds if width > 1 else 0)
            with recorder.active():
                inst = drive(adapter, a, width, prompts[:width], recorder)
            rep = replay(adapter, recorder.replays, a.replay_reps)
            recorder.replays.clear()
            summary = summarize(inst, rep)
            log(f"B={width} instrumented: wall {summary['wall_ms_median']} ms, GPU busy "
                f"{summary['gpu_busy_ms_median']} ms, host {summary['host_ms_median']} ms, shapes "
                f"{summary['shapes']}, split reasons {summary['split_reasons']}, expert set overlap "
                f"{summary['expert_set_overlap']}, kernel assignments/expert "
                f"{summary['kernel_assignments_per_unique_median']}")
            doc["widths"][str(width)] = {"plain": plain, "instrumented": inst, "replay": rep,
                                         "summary": summary,
                                         "moe_pad_status": switch_layers.moe_pad_status()}
            P.free()
            doc["swap"][str(width)]["after"] = P.swap_state()
            doc["memory"][str(width)] = {**P.memory_state(), "seconds": round(pc() - ts, 1)}
            write(out, doc)
        d = adapter.diagnostics()
        doc["receipts"] = {k: d.get(k) for k in ("route", "drafter", "checkpoint", "ast", "dfly2")}
    finally:
        adapter.close()
        doc["finished"] = now()
        doc["total_s"] = round(pc() - t0, 1)
        write(out, doc)
    log(f"wrote {out / 'receipt.json'} ({doc['total_s']} s)")
    return doc


if __name__ == "__main__":
    main()
